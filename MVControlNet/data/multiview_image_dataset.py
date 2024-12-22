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
        root_dir_mv: str,
        root_dir_render: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        cam_types: Optional[list] = None,
        cond_type: Optional[str] = None,
        load_cam_type: Optional[bool] = True,
        refine_froms: str =['continuous_remeshing_ortho']
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir_mv = root_dir_mv
        self.root_dir_render = root_dir_render
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type
        self.load_cam_type = load_cam_type
        self.cam_types = cam_types
        self.refine_froms = refine_froms

        if self.num_views == 4:
            self.view_types  = ['front', 'right', 'back', 'left']
        elif self.num_views == 5:
            self.view_types  = ['front', 'front_right', 'right', 'back', 'left']
        elif self.num_views == 6:
            self.view_types  = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        
        self.fix_cam_pose_dir = "./mv_diffusion_30/data/fixed_poses/nine_views"
        
        self.fix_cam_poses = self.load_fixed_poses()  # world2cam matrix

        if filepaths is None:
            # Get a list of all files in the directory
            file_list = os.listdir(self.root_dir_mv)
            if self.refine_froms == 'mv':
                self.cam_types = ['ortho'] * len(file_list) + ['persp']* len(file_list)
                file_list = file_list * 2
            else:
                self.cam_types = []
                for refine_from in self.refine_froms:
                    self.cam_types += [refine_from] * len(file_list)
        else:
            file_list = filepaths
        print(filepaths, root_dir_mv)
        # Filter the files that end with .png or .jpg
        self.file_list = [file for file in file_list]

        self.bg_color = self.get_bg_color()




    def __len__(self):
        return len(self.file_list)

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

        # if self.crop_size!=-1:
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
        if img.shape[-1] != 4: # not RGBA
            img = remove(image_input)
        img = np.array(img)
        img = img.astype(np.float32) / 255. # [0, 1]


        alpha = img[...,3:4]
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

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
    

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        try:
            # image = self.all_images[index%len(self.all_images)]
            # alpha = self.all_alphas[index%len(self.all_images)]
            cam_type = self.cam_types[index%len(self.file_list)]
            if self.file_list is not None:
                filename = self.file_list[index%len(self.file_list)].replace(".png", "")
            else:
                filename = 'null'

            cond_w2c = self.fix_cam_poses['front']

            tgt_w2cs = [self.fix_cam_poses[view] for view in self.view_types]

            elevations = []
            azimuths = []

            img_tensors_in = []
            img_tensors_in_mv = []
            normal_tensors_in = []
            normal_tensors_in_mv = []
            for view in self.view_types:
                # if self.refine_from == 'mv':
                #     img_path = os.path.join(self.root_dir_mv, filename, cam_type, "rgb_000_%s.png" % (view))
                # else:
                #     img_path = os.path.join(self.root_dir_mv, filename, self.refine_from, 'render_out', "rgb_render_%s.png" % (view))
                img_path_mv = os.path.join(self.root_dir_mv, filename, cam_type.split('_')[-1], 'colors', "rgb_000_%s.png" % (view))
                img_path = os.path.join(self.root_dir_render, filename, cam_type, 'render_out',
                                        "rgb_render_%s.png" % (view))
                img_tensor, alpha = self.load_image(img_path, self.bg_color, return_type="pt")
                img_tensor_mv = self.load_image_mv(img_path_mv, self.bg_color, return_type="pt")
                img_tensor_mv = img_tensor_mv.permute(2, 0, 1)

                img_tensor = img_tensor.permute(2, 0, 1)
                img_tensors_in_mv.append(img_tensor_mv)
                img_tensors_in.append(img_tensor)


                # if self.refine_from == 'mv':
                #     img_path = os.path.join(self.root_dir_mv, filename, cam_type, "normals_000_%s.png" % (view))
                # else:
                #     img_path = os.path.join(self.root_dir_mv, filename, self.refine_from, 'render_out', "normal_render_%s.png" % (view))
                img_path = os.path.join(self.root_dir_render, filename, cam_type, 'render_out',
                                        "normal_render_%s.png" % (view))
                normal_path_mv = os.path.join(self.root_dir_mv, filename, cam_type.split('_')[-1], 'normals', "normals_000_%s.png" % (view))
                normal_tensor, alpha = self.load_image(img_path, self.bg_color, return_type="pt")
                normal_tensor = normal_tensor.permute(2, 0, 1)
                normal_tensors_in.append(normal_tensor)

                normal_tensor_mv = self.load_image_mv(normal_path_mv, self.bg_color, return_type="pt")
                normal_tensor_mv = normal_tensor_mv.permute(2, 0, 1)
                normal_tensors_in_mv.append(normal_tensor_mv)

            alpha_tensors_in = [
                alpha.permute(2, 0, 1)
            ] * self.num_views

            for view, tgt_w2c in zip(self.view_types, tgt_w2cs):
                # evelations, azimuths
                elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
                elevations.append(elevation)
                azimuths.append(azimuth)

            img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
            normal_tensors_in = torch.stack(normal_tensors_in, dim=0).float()  # (Nv, 3, H, W)
            img_tensors_in_mv = torch.stack(img_tensors_in_mv, dim=0).float() # (Nv, 3, H, W)
            normal_tensors_in_mv = torch.stack(normal_tensors_in_mv, dim=0).float()  # (Nv, 3, H, W)

            elevations = torch.as_tensor(elevations).float().squeeze(1)
            azimuths = torch.as_tensor(azimuths).float().squeeze(1)
            elevations_cond = torch.as_tensor([0] * self.num_views).float()

            normal_class = torch.tensor([1, 0]).float()
            normal_task_embeddings = torch.stack([normal_class] * self.num_views, dim=0)  # (Nv, 2)
            color_class = torch.tensor([0, 1]).float()
            depth_task_embeddings = torch.stack([color_class] * self.num_views, dim=0)  # (Nv, 2)

            camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1) # (Nv, 3)

            if cam_type == 'ortho':
                cam_type_emb = torch.tensor([0, 1]).expand(self.num_views, -1)
            else:
                cam_type_emb = torch.tensor([1, 0]).expand(self.num_views, -1)

            if self.load_cam_type:
                camera_embeddings = torch.cat((camera_embeddings, cam_type_emb), dim=-1)  # (Nv, 5)

            # stack images to grid
            img_tensors_in_grid = make_grid(img_tensors_in, nrow=self.num_views // 2, padding=0, value_range=(0, 1))
            normal_tensors_in_grid = make_grid(normal_tensors_in, nrow=self.num_views // 2, padding=0,
                                                value_range=(0, 1))

            img_tensors_in_grid_mv = make_grid(img_tensors_in_mv, nrow=self.num_views // 2, padding=0, value_range=(0, 1))
            normal_tensors_in_grid_mv = make_grid(normal_tensors_in_mv, nrow=self.num_views // 2, padding=0,
                                                value_range=(0, 1))

            task_name_color = '4views, multiview color'
            task_name_normal = '4views, multiview normal map'

            if 'png' not in filename and 'jpg' not in filename:
                front_img_path = os.path.join(
                    './example_images', f"{filename}.png")
            else:
                front_img_path = os.path.join(
                    './example_images', f"{filename}")
            front_in_color, alpha = self.load_image_front(front_img_path, self.bg_color, return_type="pt")
            front_in_color = front_in_color.permute(2, 0, 1)
            front_in_normal = front_in_color
            # front_in_color, front_in_normal = img_tensors_in[0], normal_tensors_in[0]

            out =  {
                'imgs_in': img_tensors_in_grid,
                'normals_in': normal_tensors_in_grid,
                'alphas': alpha_tensors_in,
                'camera_embeddings': camera_embeddings,
                'task_name_color': task_name_color,
                'task_name_normal': task_name_normal,
                'filename': filename,
                'cam_type': cam_type,
                'front_in_color': front_in_color,
                'front_in_normal': front_in_normal,
                'imgs_mv': img_tensors_in_grid_mv,
                'normals_mv': normal_tensors_in_grid_mv,
            }
            return out
        except Exception as e:
            # if self.file_list is not None:
            #     filename = self.file_list[index % len(self.file_list)].replace(".png", "")
            # else:
            #     filename = 'null'
            traceback.print_exc()
            print(f'Warning: Failed to load object {filename}. Skipping this entry. Error: {str(e)}')
            return None





