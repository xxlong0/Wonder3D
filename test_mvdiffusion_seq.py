import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple, List
from omegaconf import OmegaConf
from PIL import Image
import cv2
import numpy as np
from dataclasses import dataclass
from packaging import version
import shutil
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image

import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

from mvdiffusion.data.single_image_dataset import SingleImageDataset as MVDiffusionDataset

from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

from einops import rearrange
from rembg import remove
import pdb

weight_dtype = torch.float16


@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path:str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
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

    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool


def log_validation(dataloader, pipeline, cfg: TestConfig, weight_dtype, name, save_dir):


    pipeline.set_progress_bar_config(disable=True)

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.device).manual_seed(cfg.seed)
    
    images_cond, images_pred = [], defaultdict(list)
    for i, batch in tqdm(enumerate(dataloader)):
        # (B, Nv, 3, H, W)
        imgs_in = batch['imgs_in']
        alphas = batch['alphas']
        # (B, Nv, Nce)
        camera_embeddings = batch['camera_embeddings']
        filename = batch['filename']

        bsz, num_views = imgs_in.shape[0], imgs_in.shape[1]
        # (B*Nv, 3, H, W)
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
        alphas = rearrange(alphas, "B Nv C H W -> (B Nv) C H W")
        # (B*Nv, Nce)
        camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

        images_cond.append(imgs_in)

        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                out = pipeline(
                    imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
                ).images
                images_pred[f"{name}-sample_cfg{guidance_scale:.1f}"].append(out)
                cur_dir = os.path.join(save_dir, f"cropsize-{cfg.validation_dataset.crop_size}-cfg{guidance_scale:.1f}")

                # pdb.set_trace()
                for i in range(bsz):
                    scene = os.path.basename(filename[i])
                    print(scene)
                    scene_dir = os.path.join(cur_dir, scene)
                    outs_dir = os.path.join(scene_dir, "outs")
                    masked_outs_dir = os.path.join(scene_dir, "masked_outs")
                    os.makedirs(outs_dir, exist_ok=True)
                    os.makedirs(masked_outs_dir, exist_ok=True)
                    img_in = imgs_in[i*num_views]
                    alpha = alphas[i*num_views]
                    img_in = torch.cat([img_in, alpha], dim=0)
                    save_image(img_in, os.path.join(scene_dir, scene+".png"))
                    for j in range(num_views):
                        view = VIEWS[j]
                        idx = i*num_views + j
                        pred = out[idx]

                        # pdb.set_trace()
                        out_filename = f"{cfg.pred_type}_000_{view}.png"
                        pred = save_image(pred, os.path.join(outs_dir, out_filename))

                        rm_pred = remove(pred)

                        save_image_numpy(rm_pred, os.path.join(scene_dir, out_filename))
    torch.cuda.empty_cache()



def save_image(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr

def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)

def log_validation_joint(dataloader, pipeline, cfg: TestConfig, weight_dtype, name, save_dir):

    pipeline.set_progress_bar_config(disable=True)

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.device).manual_seed(cfg.seed)
    
    images_cond, normals_pred, images_pred = [], defaultdict(list), defaultdict(list)
    for i, batch in tqdm(enumerate(dataloader)):
        # repeat  (2B, Nv, 3, H, W)
        imgs_in = torch.cat([batch['imgs_in']]*2, dim=0)

        filename = batch['filename']
        
        # (2B, Nv, Nce)
        camera_embeddings = torch.cat([batch['camera_embeddings']]*2, dim=0)

        task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)

        camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

        # (B*Nv, 3, H, W)
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
        # (B*Nv, Nce)
        camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

        images_cond.append(imgs_in)
        num_views = len(VIEWS)
        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                out = pipeline(
                    imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
                ).images

                bsz = out.shape[0] // 2
                normals_pred = out[:bsz]
                images_pred = out[bsz:]

                cur_dir = os.path.join(save_dir, f"cropsize-{cfg.validation_dataset.crop_size}-cfg{guidance_scale:.1f}")

                for i in range(bsz//num_views):
                    scene = filename[i]
                    scene_dir = os.path.join(cur_dir, scene)
                    normal_dir = os.path.join(scene_dir, "normals")
                    masked_colors_dir = os.path.join(scene_dir, "masked_colors")
                    os.makedirs(normal_dir, exist_ok=True)
                    os.makedirs(masked_colors_dir, exist_ok=True)
                    for j in range(num_views):
                        view = VIEWS[j]
                        idx = i*num_views + j
                        normal = normals_pred[idx]
                        color = images_pred[idx]

                        normal_filename = f"normals_000_{view}.png"
                        rgb_filename = f"rgb_000_{view}.png"
                        normal = save_image(normal, os.path.join(normal_dir, normal_filename))
                        color = save_image(color, os.path.join(scene_dir, rgb_filename))

                        rm_normal = remove(normal)
                        rm_color = remove(color)

                        save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))
                        save_image_numpy(rm_color, os.path.join(masked_colors_dir, rgb_filename))

    torch.cuda.empty_cache()


def load_wonder3d_pipeline(cfg):

    pipeline = MVDiffusionImagePipeline.from_pretrained(
    cfg.pretrained_model_name_or_path,
    torch_dtype=weight_dtype
    )

    # pipeline.to('cuda:0')
    pipeline.unet.enable_xformers_memory_efficient_attention()


    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    # sys.main_lock = threading.Lock()
    return pipeline


def main(
    cfg: TestConfig
):

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    pipeline = load_wonder3d_pipeline(cfg)

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipeline.unet.enable_xformers_memory_efficient_attention()
            print("use xformers.")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the  dataset
    validation_dataset = MVDiffusionDataset(
        **cfg.validation_dataset
    )


    # DataLoaders creation:
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )


    os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.pred_type == 'joint':
        log_validation_joint(
                    validation_dataloader,
                    pipeline,
                    cfg,
                    weight_dtype,
                    'validation',
                    cfg.save_dir
                    )
    else:
        log_validation(
                    validation_dataloader,
                    pipeline,
                    cfg,
                    weight_dtype,
                    'validation',
                    cfg.save_dir
                    )
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()

    from utils.misc import load_config    

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    print(cfg)
    schema = OmegaConf.structured(TestConfig)
    # cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)

    if cfg.num_views == 6:
        VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
    elif cfg.num_views == 4:
        VIEWS = ['front', 'right', 'back', 'left']
    main(cfg)
