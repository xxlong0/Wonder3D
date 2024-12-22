import os
from typing import Dict, Optional, Tuple, List
from PIL import Image
import cv2
import numpy as np
from dataclasses import dataclass

from collections import defaultdict

import torch

import torch.utils.checkpoint
from mv_diffusion_30.models.unet_mv2d_condition import UNetMV2DConditionModel

from mv_diffusion_30.data.single_image_dataset import SingleImageDataset as MVDiffusionDataset

from mv_diffusion_30.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

from einops import rearrange
import rembg
from torchvision.utils import make_grid, save_image

import torchvision.transforms as transforms

weight_dtype = torch.half

VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

to_pil = transforms.ToPILImage()
@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path: Optional[str]
    revision: Optional[str]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation
    load_task: bool

def save_image(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr


def save_depth_numpy(depth, fp, alpha):
    depth = depth.mul(0.4).mul(65535.).add_(0.5).to("cpu", torch.float32).numpy().mean(0)
    print(depth.min(), depth.max())

    depth[alpha < 128] = 0

    depth = depth.astype(np.uint16)

    kernel = np.ones((3, 3), np.uint8)  # kernel for erode

    # erode
    depth = cv2.erode(depth, kernel, iterations=1)

    im = Image.fromarray(depth)
    im.save(fp)


def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)


def load_wonder3d_pipeline(cfg):
    if cfg.pretrained_unet_path:
        print("load pre-trained unet from ", cfg.pretrained_unet_path)
        unet = UNetMV2DConditionModel.from_pretrained(cfg.pretrained_unet_path, revision=cfg.revision,
                                                               **cfg.unet_from_pretrained_kwargs)

    pipeline = MVDiffusionImagePipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        pred_type=cfg.pred_type,
        safety_checker=None,
        unet=unet
    )

    if torch.cuda.is_available():
        pipeline.to('cuda:0')

    return pipeline


def pred_multiview_joint(image, pipeline, seed=42, crop_size=192, camera_type='ortho', cfg=None, case_name='img', output_path='outputs'):

    validation_dataset = MVDiffusionDataset(
        single_image=image,
        num_views=6,
        bg_color='white',
        img_wh=[256, 256],
        crop_size=crop_size,
        cam_types=[camera_type],
        load_cam_type=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    images_cond, normals_pred, images_pred = [], defaultdict(list), defaultdict(list)

    batch = next(iter(validation_dataloader))

    # repeat  (2B, Nv, 3, H, W)
    imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0)

    filename = batch['filename']

    # (2B, Nv, Nce)
    camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0)

    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)

    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

    # (B*Nv, 3, H, W)
    imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W").to(weight_dtype)
    # (B*Nv, Nce)
    camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce").to(weight_dtype)

    images_cond.append(imgs_in)
    num_views = len(VIEWS)
    with torch.autocast("cuda"):
        # B*Nv images
        for guidance_scale in cfg.validation_guidance_scales:
            out = pipeline(
                imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale,
                output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
            ).images

            bsz = out.shape[0] // 2

            normals_pred = out[:bsz]
            images_pred = out[bsz:]
            color_pred_grid = make_grid(images_pred, nrow=6, padding=0, value_range=(0, 1))
            normal_pred_grid = make_grid(normals_pred, nrow=6, padding=0, value_range=(0, 1))

            rm_normals = []
            colors = []
            for i in range(bsz // num_views):
                scene = os.path.basename(case_name.split('.')[0])
                scene_dir = os.path.join(output_path, scene, 'mv', batch['cam_type'][0])

                normal_dir = os.path.join(scene_dir, "normals")
                color_dir = os.path.join(scene_dir, "colors")
                masked_colors_dir = os.path.join(scene_dir, "masked_colors")
                os.makedirs(normal_dir, exist_ok=True)
                os.makedirs(masked_colors_dir, exist_ok=True)
                os.makedirs(color_dir, exist_ok=True)
                print(scene, batch['cam_type'], scene_dir)
                rembg_session = rembg.new_session()
                for j in range(num_views):
                    view = VIEWS[j]
                    idx = i * num_views + j
                    normal = normals_pred[idx]
                    color = images_pred[idx]

                    normal_filename = f"normals_000_{view}.png"
                    rgb_filename = f"rgb_000_{view}.png"
                    normal = save_image(normal, os.path.join(normal_dir, normal_filename))
                    color = save_image(color, os.path.join(color_dir, rgb_filename))

                    rm_normal = rembg.remove(normal, alpha_matting=True, session=rembg_session)
                    rm_normals.append(Image.fromarray(rm_normal))
                    colors.append(to_pil(color))

                    save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))

            save_image(color_pred_grid, os.path.join(scene_dir, f'color_grid_img.png'))
            save_image(normal_pred_grid, os.path.join(scene_dir, f'normal_grid_img.png'))

            return rm_normals, colors