import os
import json
import math
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    cam_center = F.normalize(cameras.mean(0), p=2, dim=-1) * cameras.mean(0).norm(2)
    eigvecs = torch.linalg.eig(cameras.T @ cameras).eigenvectors
    rot_axis = F.normalize(eigvecs[:,1].real.float(), p=2, dim=-1)
    up = rot_axis
    rot_dir = torch.cross(rot_axis, cam_center)
    max_angle = (F.normalize(cameras, p=2, dim=-1) * F.normalize(cam_center, p=2, dim=-1)).sum(-1).acos().max()

    all_c2w = []
    for theta in torch.linspace(-max_angle, max_angle, n_steps):
        cam_pos = cam_center * math.cos(theta) + rot_dir * math.sin(theta)
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w

class DTUDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        cams = np.load(os.path.join(self.config.root_dir, self.config.cameras_file))

        img_sample = cv2.imread(os.path.join(self.config.root_dir, 'image', '000000.png'))
        H, W = img_sample.shape[0], img_sample.shape[1]

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = int(W / self.config.img_downscale + 0.5), int(H / self.config.img_downscale + 0.5)
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")

        self.w, self.h = w, h
        self.img_wh = (w, h)
        self.factor = w / W

        mask_dir = os.path.join(self.config.root_dir, 'mask')
        self.has_mask = True
        self.apply_mask = self.config.apply_mask
        
        self.directions = []
        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

        n_images = max([int(k.split('_')[-1]) for k in cams.keys()]) + 1

        for i in range(n_images):
            world_mat, scale_mat = cams[f'world_mat_{i}'], cams[f'scale_mat_{i}']
            P = (world_mat @ scale_mat)[:3,:4]
            K, c2w = load_K_Rt_from_P(P)
            fx, fy, cx, cy = K[0,0] * self.factor, K[1,1] * self.factor, K[0,2] * self.factor, K[1,2] * self.factor
            directions = get_ray_directions(w, h, fx, fy, cx, cy)
            self.directions.append(directions)
            
            c2w = torch.from_numpy(c2w).float()

            # blender follows opengl camera coordinates (right up back)
            # NeuS DTU data coordinate system (right down front) is different from blender
            # https://github.com/Totoro97/NeuS/issues/9
            # for c2w, flip the sign of input camera coordinate yz
            c2w_ = c2w.clone()
            c2w_[:3,1:3] *= -1. # flip input sign
            self.all_c2w.append(c2w_[:3,:4])         

            if self.split in ['train', 'val']:
                img_path = os.path.join(self.config.root_dir, 'image', f'{i:06d}.png')
                img = Image.open(img_path)
                img = img.resize(self.img_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]

                mask_path = os.path.join(mask_dir, f'{i:03d}.png')
                mask = Image.open(mask_path).convert('L') # (H, W, 1)
                mask = mask.resize(self.img_wh, Image.BICUBIC)
                mask = TF.to_tensor(mask)[0]

                self.all_fg_masks.append(mask) # (h, w)
                self.all_images.append(img)

        self.all_c2w = torch.stack(self.all_c2w, dim=0)

        if self.split == 'test':
            self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=self.config.n_test_traj_steps)
            self.all_images = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
            self.all_fg_masks = torch.zeros((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
            self.directions = self.directions[0]
        else:
            self.all_images, self.all_fg_masks = torch.stack(self.all_images, dim=0), torch.stack(self.all_fg_masks, dim=0)  
            self.directions = torch.stack(self.directions, dim=0)

        self.directions = self.directions.float()
        self.all_c2w, self.all_images, self.all_fg_masks = \
            self.all_c2w.float(), \
            self.all_images.float(), \
            self.all_fg_masks.float()
        

class DTUDataset(Dataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class DTUIterableDataset(IterableDataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('dtu')
class DTUDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = DTUIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = DTUDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = DTUDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = DTUDataset(self.config, 'train')    

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
