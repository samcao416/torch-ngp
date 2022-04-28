import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()



def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses

def ngp_matrix_to_nerf(pose, scale = 0.33):
    new_pose = np.array([
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] / scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] / scale],
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] / scale],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    return new_pose

def write_cam_tra_json(old_json, poses, target_path):
    angle_x = old_json["camera_angle_x"]
    angle_y = old_json["camera_angle_y"]
    f = old_json["fl_x"]
    k1 = old_json["k1"]
    k2 = old_json["k2"]
    p1 = old_json["p1"]
    p2 = old_json["p2"]
    b1 = old_json["b1"]
    b2 = old_json["b2"]
    cx = old_json["cx"]
    cy = old_json["cy"]
    width = old_json["w"]
    height = old_json['h']
    AABB_SCALE = old_json["aabb_scale"]

    out = {
        "camera_angle_x" : angle_x,
        "camera_angle_y" : angle_y,
        "fl_x" : f,
        "fl_y" : f,
        "k1" : k1,
        "k2" : k2,
        "p1" : p1,
        "p2" : p2,
        "b1" : b1,
        "b2" : b2,
        "cx" : cx,
        "cy" : cy,
        "w" : width,
        "h" : height,
        "aabb_scale" : AABB_SCALE,
        "frames" : [],
    }

    for i in range(len(poses)):
        name = os.path.join("out_images", '%04d.png' %(i))
        frame = {"file_path" : name, "transform_matrix" : poses[i]}
        out["frames"].append(frame)
    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(f"writing {target_path}")
    with open(target_path, "w") as outfile:
        json.dump(out, outfile, indent=2)
    return 1

class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.mode = opt.mode # colmap, blender, llff, agi
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        # load nerf-compatible format data.
        if self.mode == 'colmap' or 'agi':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])    
        
        # for colmap, manually interpolate a test set.
        '''
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)
        '''
        if (self.mode == 'agi' or 'colmap') and type == 'test':
            # use a camera trajectory to prepare pose and render views
            # if the camera trajectory json file does not exist, then create one
            cam_tra_path = os.path.join(self.root_path, 'cam_tra', 'camera_render.json')
            self.poses = []
            self.images = None
            if os.path.exists(cam_tra_path):
                print("Camera trajectory exists")
                with open(cam_tra_path, 'r') as f:
                    cam_tra = json.load(f)
                
                # Read the focal
                fl_x = self.W / (2 * np.tan(cam_tra['camera_angle_x'] / 2)) if 'camera_angle_x' in cam_tra else None
                fl_y = self.H / (2 * np.tan(cam_tra['camera_angle_y'] / 2)) if 'camera_angle_y' in cam_tra else None
                if fl_x is None: fl_x = fl_y
                if fl_y is None: fl_y = fl_x

                # Read the extrinsic
                self.cam_num = len(cam_tra['frames'])
                for i in range(self.cam_num):
                    pose = nerf_matrix_to_ngp(np.array(cam_tra['frames'][i]['transform_matrix'], dtype = np.float32), scale = self.scale)
                    self.poses.append(pose)
                
            else:
                print("Camera trajectory not exists, will generate one and save it")
                if not os.path.exists(os.path.join(self.root_path, 'cam_tra')):
                    os.makedirs(os.path.join(self.root_path, 'cam_tra'))
                # Initialize number of test frames
                n_test = 120
                
                # Focal part, use the same camera_angles and fl_x and fl_y
                cam_angle_x = transform['camera_angle_x']
                cam_angle_y = transform['camera_angle_y']
                fl_x = transform['fl_x']
                fl_y = transform['fl_y']

                # Camera Pose Part, The Slerp Method
                
                f0, f1 = frames[1], frames[-1] # choose 2 views as fern2v from fern 20 views

                pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
                pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
                rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                slerp = Slerp([0, 1], rots)

                self.poses = []
                poses_in_opengl = []
                

                for i in range(n_test + 1):
                    ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = slerp(ratio).as_matrix()
                    pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                    self.poses.append(pose)

                    # Transform the ngp pose back to opengl form so that we can write back to json file
                    pose_opengl = ngp_matrix_to_nerf(pose, scale = self.scale)
                    poses_in_opengl.append(pose_opengl)

                if (write_cam_tra_json(old_json = transform, poses = poses_in_opengl, target_path = cam_tra_path)):
                    print("New camera trajectory generated and saved.")
                else:
                    print("The process failed, some error occurs, check it again.")

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all': use all frames
            
            self.poses = []
            self.images = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data:'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and f_path[-4:] != '.png':
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if type == 'train' and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                self.images = self.images.to(torch.half if self.fp16 else torch.float).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.H / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.W / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])


    def collate(self, index):

        B = len(index) # always 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map)
        
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        return loader