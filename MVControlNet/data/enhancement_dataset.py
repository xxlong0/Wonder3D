from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import cv2
import random

import json
import os, sys
import math

from glob import glob

import PIL.Image
from .normal_utils import trans_normal, normal2img, img2normal
import pdb

from torchvision.utils import make_grid

from rembg import remove
import cv2
import numpy as np
import traceback

def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def scale_and_place_object(image, scale_factor):
    assert np.shape(image)[-1]==4  # RGBA

    # Extract the alpha channel (transparency) and the object (RGB channels)
    alpha_channel = image[:, :, 3]

    # Find the bounding box coordinates of the object
    coords = cv2.findNonZero(alpha_channel)
    x, y, width, height = cv2.boundingRect(coords)

    # Calculate the scale factor for resizing
    original_height, original_width = image.shape[:2]

    if width > height:
        size = width
        original_size = original_width
    else:
        size = height
        original_size = original_height

    scale_factor = min(scale_factor, size / (original_size+0.0))

    new_size = scale_factor * original_size
    scale_factor = new_size / size

    # Calculate the new size based on the scale factor
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    center_x = original_width // 2
    center_y = original_height // 2

    paste_x = center_x - (new_width // 2)
    paste_y = center_y - (new_height // 2)

    # Resize the object (RGB channels) to the new size
    rescaled_object = cv2.resize(image[y:y+height, x:x+width], (new_width, new_height))

    # Create a new RGBA image with the resized image
    new_image = np.zeros((original_height, original_width, 4), dtype=np.uint8)

    new_image[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = rescaled_object

    return new_image

class InferenceImageDataset(Dataset):
    def __init__(self,
        num_views: int = 4,
        img_wh: Tuple[int, int] = [512, 512],
        bg_color: str = 'white',
        crop_size: int = 192,
        cam_types: Optional[list] = None,
        mv_imgs: Optional[list] = None,
        mv_normals: Optional[list] = None,
        renderd_mv_imgs: Optional[list] = None,
        renderd_mv_normals: Optional[list] = None,
        front_img: Optional = None,
        load_cam_type: Optional[bool] = True,
        refine_froms: str = ['continuous_remeshing_ortho']
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.load_cam_type = load_cam_type
        self.cam_types = cam_types
        self.refine_froms = refine_froms

        self.mv_imgs = mv_imgs
        self.mv_normals = mv_normals
        self.rendered_mv_imgs = renderd_mv_imgs
        self.renderd_mv_normals = renderd_mv_normals
        self.front_img = front_img

        if self.num_views == 4:
            self.view_types  = ['front', 'right', 'back', 'left']
        elif self.num_views == 5:
            self.view_types  = ['front', 'front_right', 'right', 'back', 'left']
        elif self.num_views == 6:
            self.view_types  = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        
        self.fix_cam_pose_dir = "./mv_diffusion_30/data/fixed_poses/nine_views"
        
        self.fix_cam_poses = self.load_fixed_poses()  # world2cam matrix

        self.bg_color = self.get_bg_color()

    def __len__(self):
        return len([self.mv_imgs])

    def load_fixed_poses(self):
        poses = {}
        for face in self.view_types:
            RT = np.loadtxt(os.path.join(self.fix_cam_pose_dir,'%03d_%s_RT.txt'%(0, face)))
            poses[face] = RT

        return poses
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T # change to cam2world

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        # d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_theta, d_azimuth

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color
    
    
    def load_image(self, img_path, bg_color, return_type='pt', Imagefile=None):
        # pil always returns uint8
        if Imagefile is None:
            image_input = Image.open(img_path).resize(self.img_wh)
        else:
            image_input = Imagefile.resize(self.img_wh)

        image_size = self.img_wh[0]
        img = np.array(image_input)

        if img.shape[-1] != 4: # not RGBA
            img = remove(image_input)
        img = np.array(img)
        img = img.astype(np.float32) / 255. # [0, 1]

        alpha = img[..., 3:4]
        img_rgb = img[..., :3]
        mask = (alpha < 0.5).squeeze(-1)
        alpha[mask] = 0.
        alpha[~mask] = 1.0
        img_rgb[mask] = [1, 1, 1]

        # temp_img = Image.fromarray((img_rgb * 255).astype(np.uint8))
        # temp_img.save(os.path.join('outputs', 'debug', '000_normal.png'))
        img = img_rgb * alpha + bg_color * (1 - alpha)
        img = np.clip(img, 0, 1)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError
        
        return img, alpha

    def load_image_mv(self, img_path, bg_color, return_type='pt', Imagefile=None):
        # pil always returns uint8
        if Imagefile is None:
            image_input = Image.open(img_path).resize(self.img_wh)
        else:
            image_input = Imagefile.resize(self.img_wh)
        image_size = self.img_wh[0]

        img = np.array(image_input)

        img = np.array(img)
        img = img.astype(np.float32) / 255.  # [0, 1]

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)

        else:
            raise NotImplementedError

        return img

    def load_image_front(self, img_path, bg_color, return_type='np', Imagefile=None):
        # pil always returns uint8
        if Imagefile is None:
            image_input = Image.open(img_path)
        else:
            image_input = Imagefile
        image_size = self.img_wh[0]

        if np.asarray(image_input).shape[-1] == 3:
            image_input = remove(image_input)

        # if self.crop_size != -1:
        #     alpha_np = np.asarray(image_input)[:, :, 3]
        #     coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
        #     min_x, min_y = np.min(coords, 0)
        #     max_x, max_y = np.max(coords, 0)
        #     ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
        #     h, w = ref_img_.height, ref_img_.width
        #     scale = self.crop_size / max(h, w)
        #     h_, w_ = int(scale * h), int(scale * w)
        #     ref_img_ = ref_img_.resize((w_, h_))
        #     image_input = add_margin(ref_img_, size=image_size)
        # else:
        #     image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
        #     image_input = image_input.resize((image_size, image_size))

        # img = scale_and_place_object(img, self.scale_ratio)
        img = np.array(image_input)
        img = img.astype(np.float32) / 255.  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError

        return img, alpha

    def __getitem__(self, index):

        img_tensors_in = []
        img_tensors_in_mv = []
        normal_tensors_in = []
        normal_tensors_in_mv = []
        for i, view in enumerate(self.view_types):

            normal_tensor_mv, alpha = self.load_image(None, self.bg_color, return_type="pt", Imagefile=self.mv_normals[i])
            normal_tensor, alpha = self.load_image(None, self.bg_color, return_type="pt", Imagefile=self.renderd_mv_normals[i])
            img_tensor, alpha = self.load_image(None, self.bg_color, return_type="pt", Imagefile=self.rendered_mv_imgs[i])
            img_tensor_mv = self.load_image_mv(None, self.bg_color, return_type="pt", Imagefile=self.mv_imgs[i])

            img_tensor_mv = img_tensor_mv.permute(2, 0, 1)
            img_tensor = img_tensor.permute(2, 0, 1)
            normal_tensor_mv = normal_tensor_mv.permute(2, 0, 1)
            normal_tensor = normal_tensor.permute(2, 0, 1)

            img_tensors_in_mv.append(img_tensor_mv)
            img_tensors_in.append(img_tensor)
            normal_tensors_in.append(normal_tensor)
            normal_tensors_in_mv.append(normal_tensor_mv)

        alpha_tensors_in = [
            alpha.permute(2, 0, 1)
        ] * self.num_views

        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        normal_tensors_in = torch.stack(normal_tensors_in, dim=0).float()  # (Nv, 3, H, W)
        img_tensors_in_mv = torch.stack(img_tensors_in_mv, dim=0).float() # (Nv, 3, H, W)
        normal_tensors_in_mv = torch.stack(normal_tensors_in_mv, dim=0).float()  # (Nv, 3, H, W)

        # stack images to grid
        img_tensors_in_grid = make_grid(img_tensors_in, nrow=self.num_views // 2, padding=0, value_range=(0, 1))
        normal_tensors_in_grid = make_grid(normal_tensors_in, nrow=self.num_views // 2, padding=0,
                                            value_range=(0, 1))

        img_tensors_in_grid_mv = make_grid(img_tensors_in_mv, nrow=self.num_views // 2, padding=0, value_range=(0, 1))
        normal_tensors_in_grid_mv = make_grid(normal_tensors_in_mv, nrow=self.num_views // 2, padding=0,
                                            value_range=(0, 1))

        task_name_color = '4views, multiview color'
        task_name_normal = '4views, multiview normal map'

        front_in_color, alpha = self.load_image_front(None, self.bg_color, return_type="pt", Imagefile=self.front_img)
        front_in_color = front_in_color.permute(2, 0, 1)
        front_in_normal = front_in_color

        out = {
            'imgs_in': img_tensors_in_grid,
            'normals_in': normal_tensors_in_grid,
            'alphas': alpha_tensors_in,
            'task_name_color': task_name_color,
            'task_name_normal': task_name_normal,
            'front_in_color': front_in_color,
            'front_in_normal': front_in_normal,
            'imgs_mv': img_tensors_in_grid_mv,
            'normals_mv': normal_tensors_in_grid_mv,
        }

        return out





