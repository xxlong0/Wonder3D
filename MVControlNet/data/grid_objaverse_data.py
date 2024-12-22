from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import cv2
import random

import json
import os, sys
import math

import PIL.Image
from .normal_utils import trans_normal, normal2img, img2normal
import pdb
from .depth_utils import scale_depth_to_model
import traceback

from torch.nn import functional as F

from MVControlNet.data.basicsr_utils.degradations import circular_lowpass_kernel, random_mixed_kernels
from MVControlNet.data.basicsr_utils.img_process_util import filter2D, USMSharp
from MVControlNet.data.basicsr_utils.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from MVControlNet.data.basicsr_utils.diffjpeg import DiffJPEG


class grid_refine_dataset(Dataset):
    def __init__(self,
                 root_dir_ortho: str,
                 root_dir_persp: str,
                 pred_ortho: bool,
                 pred_persp: bool,
                 num_views: int,
                 bg_color: Any,
                 img_wh: Tuple[int, int],
                 object_list: str,
                 groups_num: int = 1,
                 validation: bool = False,
                 data_view_num: int = 6,
                 num_validation_samples: int = 64,
                 num_samples: Optional[int] = None,
                 invalid_list: Optional[str] = None,
                 trans_norm_system: bool = True,  # if True, transform all normals map into the cam system of front view
                 augment_data: bool = False,
                 read_normal: bool = True,
                 read_color: bool = False,
                 read_depth: bool = False,
                 read_mask: bool = False,
                 pred_type: str = 'joint',
                 suffix: str = 'png',
                 subscene_tag: int = 2,
                 load_cam_type: bool = False,
                 backup_scene: str = "0306b42594fb447ca574f597352d4b56",
                 ortho_crop_size: int = 360,
                 persp_crop_size: int = 440,
                 load_switcher: bool = False,
                 sinc_prob: float = 0.1,
                 gray_noise_prob: float = 0.4,
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.load_cam_type = load_cam_type
        self.root_dir_ortho = Path(root_dir_ortho)
        self.root_dir_persp = Path(root_dir_persp)
        self.pred_ortho = pred_ortho
        self.pred_persp = pred_persp
        self.num_views = num_views
        self.bg_color = bg_color
        self.validation = validation
        self.num_samples = num_samples
        self.trans_norm_system = trans_norm_system
        self.augment_data = augment_data
        self.invalid_list = invalid_list
        self.groups_num = groups_num
        self.img_wh = img_wh
        self.read_normal = read_normal
        self.read_color = read_color
        self.read_depth = read_depth
        self.read_mask = read_mask
        self.pred_type = pred_type  # load type
        self.suffix = suffix
        self.subscene_tag = subscene_tag
        if self.num_views == 6:
            self.view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        elif self.num_views == 4:
            self.view_types = ['front', 'right', 'back', 'left']
        self.fix_cam_pose_dir = "./mv_diffusion_30/data/fixed_poses/nine_views"

        self.fix_cam_poses = self.load_fixed_poses()  # world2cam matrix
        self.ortho_crop_size = ortho_crop_size
        self.persp_crop_size = persp_crop_size
        self.load_switcher = load_switcher
        self.cond_img_size = [256, 256]
        self.out_img_size = [512, 512]

        # for generate low quality img
        self.sinc_prob = sinc_prob
        self.kernel_range = [2 * v + 1 for v in range(3, 11)] # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        self.jpeger = DiffJPEG(differentiable=False).requires_grad_(False) # simulate JPEG compression artifacts
        self.gaussian_noise_prob = 0
        self.gaussian_noise_prob2 = 0
        self.resize_prob = [0.3, 0.4, 0.3]
        self.resize_prob2 = [0.3, 0.4, 0.3]
        self.second_blur_prob = 0.8
        self.resize_range = [0.15, 1.5]
        self.resize_range2 = [0.3, 1.2]
        self.noise_range = [0.15, 1.5]
        self.poisson_scale_range = [0.05, 3]
        self.poisson_scale_range2 = [0.3, 1.2]

        self.jpeg_range = [30, 95]
        self.jpeg_range2 = [30, 95]
        self.scale = 1

        self.noise_range2 = [1, 25]

        self.gray_noise_prob = gray_noise_prob
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']

        self.blur_sigma = [0.2, 3]
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range = [0.5, 4]
        self.betag_range2 = [0.5, 4]
        self.betap_range = [1, 2]
        self.betap_range2 = [1, 2]

        if object_list is not None:
            with open(object_list) as f:
                self.objects = json.load(f)
            self.objects = [os.path.basename(o).replace(".glb", "") for o in self.objects]
        else:
            self.objects = os.listdir(self.root_dir)
            self.objects = sorted(self.objects)

        if self.invalid_list is not None:
            with open(self.invalid_list) as f:
                self.invalid_objects = json.load(f)
            self.invalid_objects = [os.path.basename(o).replace(".glb", "") for o in self.invalid_objects]
        else:
            self.invalid_objects = []

        self.all_objects = set(self.objects) - (set(self.invalid_objects) & set(self.objects))
        self.all_objects = list(self.all_objects)

        if not validation:
            self.all_objects = self.all_objects[:-num_validation_samples]
        else:
            self.all_objects = self.all_objects[-num_validation_samples:]
        if num_samples is not None:
            self.all_objects = self.all_objects[:num_samples]

        print("loading ", len(self.all_objects), " objects in the dataset")

        self.backup_data = self.__getitem_color_normal_grid__(0, backup_scene)

    def __len__(self):
        return len(self.objects) * self.total_view

    def load_fixed_poses(self):
        poses = {}
        for face in self.view_types:
            RT = np.loadtxt(os.path.join(self.fix_cam_pose_dir, '%03d_%s_RT.txt' % (0, face)))
            poses[face] = RT

        return poses

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T  # change to cam2world

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
        elif self.bg_color == 'three_choices':
            white = np.array([1., 1., 1.], dtype=np.float32)
            black = np.array([0., 0., 0.], dtype=np.float32)
            gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            bg_color = random.choice([white, black, gray])
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_mask(self, img_path, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = np.float32(img > 0)

        assert len(np.shape(img)) == 2

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def load_mask_from_rgba(self, img_path, camera_type):
        img = Image.open(img_path)

        if camera_type == 'ortho':
            left = (img.width - self.ortho_crop_size) // 2
            right = (img.width + self.ortho_crop_size) // 2
            top = (img.height - self.ortho_crop_size) // 2
            bottom = (img.height + self.ortho_crop_size) // 2
            img = img.crop((left, top, right, bottom))
        if camera_type == 'persp':
            left = (img.width - self.persp_crop_size) // 2
            right = (img.width + self.persp_crop_size) // 2
            top = (img.height - self.persp_crop_size) // 2
            bottom = (img.height + self.persp_crop_size) // 2
            img = img.crop((left, top, right, bottom))

        img = img.resize(self.out_img_size)
        img = np.array(img).astype(np.float32) / 255.  # [0, 1]
        assert img.shape[-1] == 4  # must RGBA

        alpha = img[:, :, 3:]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        return alpha

    def load_image(self, img_path, bg_color, alpha, return_type='np', camera_type=None, read_depth=False):
        img = Image.open(img_path)

        if camera_type == 'ortho':
            left = (img.width - self.ortho_crop_size) // 2
            right = (img.width + self.ortho_crop_size) // 2
            top = (img.height - self.ortho_crop_size) // 2
            bottom = (img.height + self.ortho_crop_size) // 2
            img = img.crop((left, top, right, bottom))
        if camera_type == 'persp':
            left = (img.width - self.persp_crop_size) // 2
            right = (img.width + self.persp_crop_size) // 2
            top = (img.height - self.persp_crop_size) // 2
            bottom = (img.height + self.persp_crop_size) // 2
            img = img.crop((left, top, right, bottom))

        img = img.resize(self.out_img_size)
        img_downsampled = img.resize(self.cond_img_size)
        img_downsampled = img_downsampled.resize(self.out_img_size)  # Downsample to 256x256 resolution

        img = np.array(img).astype(np.float32) / 255.  # [0, 1]
        img_downsampled = np.array(img_downsampled).astype(np.float32) / 255.  # [0, 1]

        assert img.shape[-1] == 3 or img.shape[-1] == 4  # RGB or RGBA
        assert img_downsampled.shape[-1] == 3 or img_downsampled.shape[-1] == 4  # RGB or RGBA

        if alpha is None and img.shape[-1] == 4:
            alpha = img[:, :, 3:]
            img = img[:, :, :3]

            alpha_downsampled = img_downsampled[:, :, 3:]
            img_downsampled = img_downsampled[:, :, :3]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]
        if alpha_downsampled.shape[-1] != 1:
            alpha_downsampled = alpha_downsampled[:, :, None]

        if read_depth:
            bg_color = np.array([1., 1., 1.], dtype=np.float32)

        img = img[..., :3] * alpha + bg_color * (1 - alpha)
        img_downsampled = img_downsampled[..., :3] * alpha_downsampled + bg_color * (1 - alpha_downsampled)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            img_downsampled = torch.from_numpy(img_downsampled)
        elif return_type == "pil":
            img = Image.fromarray((img * 255).astype(np.uint8))
            img_downsampled = Image.fromarray((img_downsampled * 255).astype(np.uint8))
        else:
            raise NotImplementedError

        return img, img_downsampled

    def load_depth(self, img_path, bg_color, alpha, return_type='np', camera_type=None):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        depth_bg_color = np.array([1., 1., 1.], dtype=np.float32)  # white for depth
        depth_map = Image.open(img_path)

        if camera_type == 'ortho':
            left = (depth_map.width - self.ortho_crop_size) // 2
            right = (depth_map.width + self.ortho_crop_size) // 2
            top = (depth_map.height - self.ortho_crop_size) // 2
            bottom = (depth_map.height + self.ortho_crop_size) // 2
            depth_map = depth_map.crop((left, top, right, bottom))
        if camera_type == 'persp':
            left = (depth_map.width - self.persp_crop_size) // 2
            right = (depth_map.width + self.persp_crop_size) // 2
            top = (depth_map.height - self.persp_crop_size) // 2
            bottom = (depth_map.height + self.persp_crop_size) // 2
            depth_map = depth_map.crop((left, top, right, bottom))

        depth_map = depth_map.resize(self.out_img_size)
        depth_map = np.array(depth_map)

        # scale the depth map:
        depth_map = scale_depth_to_model(depth_map.astype(np.float32))
        # depth_map = depth_map / 65535. # [0, 1]
        # depth_map[depth_map > 0.4] = 0
        # depth_map = depth_map / 0.4

        assert depth_map.ndim == 2  # depth
        img = np.stack([depth_map] * 3, axis=-1)

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        # print(np.max(img[:, :, 0]))
        # print(np.min(img[...,:3]), np.max(img[...,:3]))
        img = img[..., :3] * alpha + depth_bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)

        else:
            raise NotImplementedError

        return img

    def transform_mask_as_input(self, mask, return_type='np'):

        mask = mask * 255

        # mask = mask.resize(self.img_wh)
        mask = np.squeeze(mask, axis=-1)
        assert mask.ndim == 2  #
        mask = np.stack([mask] * 3, axis=-1)
        if return_type == "np":
            pass
        elif return_type == "pt":
            mask = torch.from_numpy(mask)
        else:
            raise NotImplementedError
        return mask

    def load_normal(self, img_path, bg_color, alpha, RT_w2c=None, RT_w2c_cond=None, return_type='np', camera_type=None):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        normal = Image.open(img_path)

        if camera_type == 'ortho':
            left = (normal.width - self.ortho_crop_size) // 2
            right = (normal.width + self.ortho_crop_size) // 2
            top = (normal.height - self.ortho_crop_size) // 2
            bottom = (normal.height + self.ortho_crop_size) // 2
            normal = normal.crop((left, top, right, bottom))
        if camera_type == 'persp':
            left = (normal.width - self.persp_crop_size) // 2
            right = (normal.width + self.persp_crop_size) // 2
            top = (normal.height - self.persp_crop_size) // 2
            bottom = (normal.height + self.persp_crop_size) // 2
            normal = normal.crop((left, top, right, bottom))

        normal_img = normal
        normal = np.array(normal_img.resize(self.out_img_size))

        normal_downsampled = normal_img.resize(self.cond_img_size) # Downsample to 256x256 resolution
        normal_downsampled = np.array(normal_downsampled.resize(self.out_img_size))

        assert normal.shape[-1] == 3 or normal.shape[-1] == 4  # RGB or RGBA
        assert normal_downsampled.shape[-1] == 3 or normal_downsampled.shape[-1] == 4  # RGB or RGBA

        if alpha is None and normal.shape[-1] == 4:
            alpha = normal[:, :, 3:] / 255.
            normal = normal[:, :, :3]

            alpha_downsampled = normal_downsampled[:, :, 3:] / 255.
            normal_downsampled = normal_downsampled[:, :, :3]  # [0, 1]

        normal = trans_normal(img2normal(normal), RT_w2c, RT_w2c_cond)
        normal_downsampled = trans_normal(img2normal(normal_downsampled), RT_w2c, RT_w2c_cond)

        img = (normal * 0.5 + 0.5).astype(np.float32)  # [0, 1]
        img_downsampled = (normal_downsampled * 0.5 + 0.5).astype(np.float32)  # [0, 1]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]
            alpha_downsampled = alpha_downsampled[:, :, None]

        img = img[..., :3] * alpha + bg_color * (1 - alpha)
        img_downsampled = img_downsampled[..., :3] * alpha_downsampled + bg_color * (1 - alpha_downsampled)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            img_downsampled = torch.from_numpy(img_downsampled)
        elif return_type == "pil":
            img = Image.fromarray((img * 255).astype(np.uint8))
            img_downsampled = Image.fromarray((img_downsampled * 255).astype(np.uint8))
        else:
            raise NotImplementedError

        return img, img_downsampled

    def __len__(self):
        return len(self.all_objects)

    def get_lq_twin(self, img_gt1, img_gt2):
        # Ensure both images are in the same format
        img_gt1 = img_gt1.unsqueeze(0).to(torch.float32)
        img_gt2 = img_gt2.unsqueeze(0).to(torch.float32)

        # Store original sizes
        ori_h, ori_w = img_gt1.size()[2:4]

        # ------------------------ Locking random parameters ------------------------ #
        # Choose kernel size and generate the first kernel
        kernel_size_1 = random.choice(self.kernel_range)
        use_sinc_1 = np.random.uniform() < self.sinc_prob
        if use_sinc_1:
            omega_c_1 = np.random.uniform(np.pi / 3, np.pi) if kernel_size_1 < 13 else np.random.uniform(np.pi / 5,
                                                                                                         np.pi)
            kernel_1 = circular_lowpass_kernel(omega_c_1, kernel_size_1, pad_to=False)
        else:
            kernel_1 = random_mixed_kernels(
                self.kernel_list, self.kernel_prob, kernel_size_1, self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi], self.betag_range, self.betap_range, noise_range=None
            )
        pad_size_1 = (21 - kernel_size_1) // 2
        kernel_1 = np.pad(kernel_1, ((pad_size_1, pad_size_1), (pad_size_1, pad_size_1)))

        # Choose kernel size and generate the second kernel
        kernel_size_2 = random.choice(self.kernel_range)
        use_sinc_2 = np.random.uniform() < self.sinc_prob
        if use_sinc_2:
            omega_c_2 = np.random.uniform(np.pi / 3, np.pi) if kernel_size_2 < 13 else np.random.uniform(np.pi / 5,
                                                                                                         np.pi)
            kernel_2 = circular_lowpass_kernel(omega_c_2, kernel_size_2, pad_to=False)
        else:
            kernel_2 = random_mixed_kernels(
                self.kernel_list2, self.kernel_prob2, kernel_size_2, self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi], self.betag_range2, self.betap_range2, noise_range=None
            )
        pad_size_2 = (21 - kernel_size_2) // 2
        kernel_2 = np.pad(kernel_2, ((pad_size_2, pad_size_2), (pad_size_2, pad_size_2)))

        # Generate the final sinc kernel
        use_sinc_final = np.random.uniform() < self.sinc_prob
        if use_sinc_final:
            kernel_size_final = random.choice(self.kernel_range)
            omega_c_final = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c_final, kernel_size_final, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        kernel_1 = torch.FloatTensor(kernel_1)
        kernel_2 = torch.FloatTensor(kernel_2)

        lr_d = {'kernel1': kernel_1, 'kernel2': kernel_2, 'sinc_kernel': sinc_kernel}

        # Random choices for degradation steps
        updown_type_1 = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        scale_1 = (np.random.uniform(1, self.resize_range[1]) if updown_type_1 == 'up' else
                   np.random.uniform(self.resize_range[0], 1) if updown_type_1 == 'down' else 1)
        mode_1 = random.choice(['area', 'bilinear', 'bicubic'])

        add_noise_1 = np.random.uniform() < self.gaussian_noise_prob
        add_poisson_noise_1 = not add_noise_1

        updown_type_2 = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        scale_2 = (np.random.uniform(1, self.resize_range2[1]) if updown_type_2 == 'up' else
                   np.random.uniform(self.resize_range2[0], 1) if updown_type_2 == 'down' else 1)
        mode_2 = random.choice(['area', 'bilinear', 'bicubic'])

        add_noise_2 = np.random.uniform() < self.gaussian_noise_prob2
        add_poisson_noise_2 = not add_noise_2

        jpeg_p_1 = np.random.uniform(*self.jpeg_range)
        jpeg_p_2 = np.random.uniform(*self.jpeg_range2)

        use_second_blur = np.random.uniform() < self.second_blur_prob
        apply_sinc_first = np.random.uniform() < 0.5

        # ------------------------ Apply locked degradation steps ------------------------ #
        def apply_degradation(img_gt, lr_d, scale_1, mode_1, add_noise_1, add_poisson_noise_1, jpeg_p_1,
                              scale_2, mode_2, add_noise_2, add_poisson_noise_2, jpeg_p_2, use_second_blur,
                              apply_sinc_first):
            kernel1 = lr_d['kernel1'].to(torch.float32)
            kernel2 = lr_d['kernel2'].to(torch.float32)
            sinc_kernel = lr_d['sinc_kernel'].to(torch.float32)

            # First degradation process
            out = filter2D(img_gt, kernel1)
            out = F.interpolate(out, scale_factor=scale_1, mode=mode_1)

            # if add_noise_1:
            #     out = random_add_gaussian_noise_pt(out, sigma_range=self.noise_range, clip=True, rounds=False,
            #                                        gray_prob=self.gray_noise_prob)
            # elif add_poisson_noise_1:
            #     out = random_add_poisson_noise_pt(out, scale_range=self.poisson_scale_range,
            #                                       gray_prob=self.gray_noise_prob, clip=True, rounds=False)

            out = torch.clamp(out, 0, 1).to(torch.float32)
            # out = self.jpeger(out, quality=jpeg_p_1).to('cpu') # Assuming jpeger function

            # Second degradation process
            if use_second_blur:
                out = filter2D(out, kernel2)
            out = F.interpolate(out, size=(int(ori_h / self.scale * scale_2), int(ori_w / self.scale * scale_2)),
                                mode=mode_2)

            # if add_noise_2:
            #     out = random_add_gaussian_noise_pt(out, sigma_range=self.noise_range2, clip=True, rounds=False,
            #                                        gray_prob=self.gray_noise_prob)
            # elif add_poisson_noise_2:
            #     out = random_add_poisson_noise_pt(out, scale_range=self.poisson_scale_range2,
            #                                       gray_prob=self.gray_noise_prob, clip=True, rounds=False)

            if apply_sinc_first:
                out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode_2)
                out = filter2D(out, sinc_kernel)
                # out = self.jpeger(out, quality=jpeg_p_2).to('cpu')
            else:
                # out = self.jpeger(out, quality=jpeg_p_2).to('cpu')
                out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode_2)
                out = filter2D(out, sinc_kernel)

            lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            return lq.contiguous()

        # Apply degradation process to both images
        img_lq1 = apply_degradation(img_gt1, lr_d, scale_1, mode_1, add_noise_1, add_poisson_noise_1, jpeg_p_1,
                                    scale_2, mode_2, add_noise_2, add_poisson_noise_2, jpeg_p_2, use_second_blur,
                                    apply_sinc_first).squeeze(0)
        img_lq2 = apply_degradation(img_gt2, lr_d, scale_1, mode_1, add_noise_1, add_poisson_noise_1, jpeg_p_1,
                                    scale_2, mode_2, add_noise_2, add_poisson_noise_2, jpeg_p_2, use_second_blur,
                                    apply_sinc_first).squeeze(0)

        return img_lq1, img_lq2

    def get_lq(self, img_gt):
        img_gt = img_gt.unsqueeze(0).to(torch.float32)
        # img_gt = img_gt.permute(0, 3, 1, 2)
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        lr_d = {'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}

        # ------------------------------------- generate the lq img ------------------------------------- #
        kernel1 = lr_d['kernel1'].to(torch.float32)
        kernel2 = lr_d['kernel2'].to(torch.float32)
        sinc_kernel = lr_d['sinc_kernel'].to(torch.float32)

        ori_h, ori_w = img_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(img_gt, kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.gray_noise_prob
        if np.random.uniform() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression

        out = torch.clamp(out, 0, 1).to(torch.float32)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)

        # out = self.jpeger(out, quality=jpeg_p.to(out.dtype)).to('cpu')

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / self.scale * scale), int(ori_w / self.scale * scale)), mode=mode)
        # add noise
        gray_noise_prob = self.gray_noise_prob
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2).to(out.dtype)
            out = torch.clamp(out, 0, 1)
            # out = self.jpeger(out, quality=jpeg_p.to(out.dtype)).to('cpu')
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2).to(out.dtype)
            out = torch.clamp(out, 0, 1)
            # out = self.jpeger(out, quality=jpeg_p.to(out.dtype)).to('cpu')
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, sinc_kernel)

        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # training pair pool
        # self._dequeue_and_enqueue()
        img_lq = lq.contiguous()

        return img_lq.squeeze(0)

    def __getitem_color_normal_grid__(self, index, debug_object=None):
        if debug_object is not None:
            object_name = debug_object  #
            set_idx = random.sample(range(0, self.groups_num), 1)[0]  # without replacement
        else:
            object_name = self.all_objects[index % len(self.all_objects)]
            set_idx = 0

        if self.augment_data:
            cond_view = random.sample(self.view_types, k=1)[0]
        else:
            cond_view = 'front'

        assert self.pred_ortho or self.pred_persp
        if self.pred_ortho and self.pred_persp:
            if random.random() < 0.5:
                load_dir = self.root_dir_ortho
                load_cam_type = 'ortho'
            else:
                load_dir = self.root_dir_persp
                load_cam_type = 'persp'
        elif self.pred_ortho and not self.pred_persp:
            load_dir = self.root_dir_ortho
            load_cam_type = 'ortho'
        elif self.pred_persp and not self.pred_ortho:
            load_dir = self.root_dir_persp
            load_cam_type = 'persp'

        view_types = self.view_types

        cond_w2c = self.fix_cam_poses[cond_view]

        tgt_w2cs = [self.fix_cam_poses[view] for view in view_types]

        elevations = []
        azimuths = []

        # get the bg color
        bg_color = self.get_bg_color()

        if self.read_mask:
            cond_alpha = self.load_mask(os.path.join(load_dir, object_name[:self.subscene_tag], object_name,
                                                     "mask_%03d_%s.%s" % (set_idx, cond_view, self.suffix)),
                                        return_type='np')
        else:
            cond_alpha = None

        img_tensors_out = []
        img_tensors_cond = []
        normal_tensors_out = []
        normal_tensors_cond = []

        read_color, read_normal = True, True

        for view, tgt_w2c in zip(view_types, tgt_w2cs):
            img_path = os.path.join(load_dir, object_name[:self.subscene_tag], object_name,
                                    "rgb_%03d_%s.%s" % (set_idx, view, self.suffix))
            mask_path = os.path.join(load_dir, object_name[:self.subscene_tag], object_name,
                                     "mask_%03d_%s.%s" % (set_idx, view, self.suffix))

            if self.read_mask:
                alpha = self.load_mask(mask_path, return_type='np')
            else:
                alpha = None

            if read_color:
                img_tensor, img_downsampled_tensor = self.load_image(img_path, bg_color, alpha, return_type="pt", camera_type=load_cam_type,
                                             read_depth=False)
                img_tensor = img_tensor.permute(2, 0, 1) # 3, h_out, w_out
                img_downsampled_tensor = img_downsampled_tensor.permute(2, 0, 1)

                img_tensors_out.append(img_tensor)
                img_tensors_cond.append(img_downsampled_tensor)

            if read_normal:
                normal_path = os.path.join(load_dir, object_name[:self.subscene_tag], object_name,
                                           "normals_%03d_%s.%s" % (set_idx, view, self.suffix))
                normal_tensor, normal_downsampled_tensor = self.load_normal(normal_path, bg_color, alpha, RT_w2c=tgt_w2c, RT_w2c_cond=cond_w2c,
                                                 return_type="pt", camera_type=load_cam_type)
                normal_tensor = normal_tensor.permute(2, 0, 1) # 3, h_out, w_out
                normal_downsampled_tensor = normal_downsampled_tensor.permute(2, 0, 1)

                normal_tensors_out.append(normal_tensor)
                normal_tensors_cond.append(normal_downsampled_tensor)

            # evelations, azimuths
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        if read_color:
            front_img_path = os.path.join(load_dir, object_name[:self.subscene_tag], object_name,
                                    "rgb_%03d_front.%s" % (set_idx,  self.suffix))
            front_in_color, _ = self.load_image(front_img_path, bg_color, alpha=None, camera_type=load_cam_type, return_type="pt",
                                             read_depth=False)
            front_in_color = np.transpose(front_in_color, (2, 0, 1))
            # front_in_color = front_in_color.permute(2, 0, 1)
        if read_normal and read_color:
            front_in_normal = front_in_color

        # stack images to grid
        img_tensors_out_grid = make_grid(img_tensors_out, nrow=self.num_views // 2, padding=0, value_range=(0, 1))
        # img_tensors_cond_grid = self.get_lq(img_tensors_out_grid)
        normal_tensors_out_grid = make_grid(normal_tensors_out, nrow=self.num_views // 2, padding=0, value_range=(0, 1))
        # normal_tensors_cond_grid = self.get_lq(normal_tensors_out_grid)
        img_tensors_cond_grid, normal_tensors_cond_grid = self.get_lq_twin(img_tensors_out_grid, normal_tensors_out_grid)


        elevations = torch.as_tensor(elevations).float().squeeze(1)
        azimuths = torch.as_tensor(azimuths).float().squeeze(1)
        elevations_cond = torch.as_tensor([0] * self.num_views).float()  # fixed only use 4 views to train

        if load_cam_type == 'ortho':
            cam_type_emb = torch.tensor([0, 1]).expand(self.num_views, -1)
        else:
            cam_type_emb = torch.tensor([1, 0]).expand(self.num_views, -1)
        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1)
        # if self.pred_ortho and self.pred_persp:
        if self.load_cam_type:
            camera_embeddings = torch.cat((camera_embeddings, cam_type_emb), dim=-1)  # (Nv, 5)

        task_name_color = '4views, multiview color'
        task_name_normal = '4views, multiview normal map'

        return {
            'elevations_cond': elevations_cond,
            'elevations_cond_deg': torch.rad2deg(elevations_cond),
            'elevations': elevations,
            'azimuths': azimuths,
            'elevations_deg': torch.rad2deg(elevations),
            'azimuths_deg': torch.rad2deg(azimuths),
            'img_tensors_out': img_tensors_out_grid,
            'img_tensors_cond': img_tensors_cond_grid,
            'normal_tensors_out': normal_tensors_out_grid,
            'normal_tensors_cond': normal_tensors_cond_grid,
            'camera_embeddings': camera_embeddings,
            'task_name_color': task_name_color,
            'task_name_normal': task_name_normal,
            'front_in_color': front_in_color,
            'front_in_normal': front_in_normal
        }

    def __getitem__(self, index):
        try:
            data = self.backup_data = self.__getitem_color_normal_grid__(index)
            return data

        except:
            print("load error ", self.all_objects[index % len(self.all_objects)])
            traceback.print_exc()
            return self.backup_data


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights
        self.num_datasets = len(datasets)

    def __getitem__(self, i):
        chosen = random.choices(self.datasets, self.weights, k=1)[0]
        return chosen[i]

    def __len__(self):
        return max(len(d) for d in self.datasets)
