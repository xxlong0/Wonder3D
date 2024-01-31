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
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid, save_image

import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

from mvdiffusion.data.objaverse_dataset import ObjaverseDataset as MVDiffusionDataset

from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

from einops import rearrange

import time
import pdb

logger = get_logger(__name__, log_level="INFO")


@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path: Optional[str]
    revision: Optional[str]
    train_dataset: Dict
    validation_dataset: Dict
    validation_train_dataset: Dict
    output_dir: str
    seed: Optional[int]
    train_batch_size: int
    validation_batch_size: int
    validation_train_batch_size: int
    max_train_steps: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    scale_lr: bool
    lr_scheduler: str
    lr_warmup_steps: int
    snr_gamma: Optional[float]
    use_8bit_adam: bool
    allow_tf32: bool
    use_ema: bool
    dataloader_num_workers: int
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: Optional[float]
    prediction_type: Optional[str]
    logging_dir: str
    vis_dir: str
    mixed_precision: Optional[str]
    report_to: Optional[str]
    local_rank: int
    checkpointing_steps: int
    checkpoints_total_limit: Optional[int]
    resume_from_checkpoint: Optional[str]
    enable_xformers_memory_efficient_attention: bool
    validation_steps: int
    validation_sanity_check: bool
    tracker_project_name: str

    trainable_modules: Optional[list]
    use_classifier_free_guidance: bool
    condition_drop_rate: float
    scale_input_latents: bool

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str

    drop_type: str

    last_global_step: int


def log_validation(dataloader, vae, feature_extractor, image_encoder, unet, cfg: TrainingConfig, accelerator, weight_dtype, global_step, name, save_dir):
    logger.info(f"Running {name} ... ")

    pipeline = MVDiffusionImagePipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, vae=vae, unet=accelerator.unwrap_model(unet), safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler"),
        **cfg.pipe_kwargs
    )

    pipeline.set_progress_bar_config(disable=True)

    if cfg.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()    

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)
    
    images_cond, images_gt, images_pred = [], [], defaultdict(list)
    for i, batch in enumerate(dataloader):
        # (B, Nv, 3, H, W)
        imgs_in, colors_out, normals_out = batch['imgs_in'], batch['imgs_out'], batch['normals_out']
        
        # repeat  (2B, Nv, 3, H, W)
        imgs_in = torch.cat([imgs_in]*2, dim=0)
        imgs_out = torch.cat([normals_out, colors_out], dim=0)
        
        # (2B, Nv, Nce)
        camera_embeddings = torch.cat([batch['camera_embeddings']]*2, dim=0)

        task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)

        camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

        # (B*Nv, 3, H, W)
        imgs_in, imgs_out = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W"), rearrange(imgs_out, "B Nv C H W -> (B Nv) C H W")
        # (B*Nv, Nce)
        camera_task_embeddings = rearrange(camera_task_embeddings, "B Nv Nce -> (B Nv) Nce")

        images_cond.append(imgs_in)
        images_gt.append(imgs_out)
        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                out = pipeline(
                    imgs_in, camera_task_embeddings, generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
                ).images
                shape = out.shape
                out0, out1 = out[:shape[0]//2], out[shape[0]//2:]
                out = []
                for ii in range(shape[0]//2):
                    out.append(out0[ii])
                    out.append(out1[ii])
                out = torch.stack(out, dim=0)
                images_pred[f"{name}-sample_cfg{guidance_scale:.1f}"].append(out)
    images_cond_all = torch.cat(images_cond, dim=0)
    images_gt_all = torch.cat(images_gt, dim=0)
    images_pred_all = {}
    for k, v in images_pred.items():
        images_pred_all[k] = torch.cat(v, dim=0)
    
    nrow = cfg.validation_grid_nrow
    ncol = images_cond_all.shape[0] // nrow
    images_cond_grid = make_grid(images_cond_all, nrow=nrow, ncol=ncol, padding=0, value_range=(0, 1))
    images_gt_grid = make_grid(images_gt_all, nrow=nrow, ncol=ncol, padding=0, value_range=(0, 1))
    images_pred_grid = {}
    for k, v in images_pred_all.items():
        images_pred_grid[k] = make_grid(v, nrow=nrow, ncol=ncol, padding=0, value_range=(0, 1))
    save_image(images_cond_grid, os.path.join(save_dir, f"{global_step}-{name}-cond.jpg"))
    save_image(images_gt_grid, os.path.join(save_dir, f"{global_step}-{name}-gt.jpg"))
    for k, v in images_pred_grid.items():
        save_image(v, os.path.join(save_dir, f"{global_step}-{k}.jpg"))
    torch.cuda.empty_cache()


def main(
    cfg: TrainingConfig
):
    # override local_rank with envvar
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != cfg.local_rank:
        cfg.local_rank = env_local_rank

    vis_dir = os.path.join(cfg.output_dir, cfg.vis_dir)
    logging_dir = os.path.join(cfg.output_dir, cfg.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_encoder", revision=cfg.revision)
    feature_extractor = CLIPImageProcessor.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="feature_extractor", revision=cfg.revision)
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision)
    if cfg.pretrained_unet_path is None:
        unet = UNetMV2DConditionModel.from_pretrained_2d(cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, **cfg.unet_from_pretrained_kwargs)
    else:
        print("load pre-trained unet from ", cfg.pretrained_unet_path)
        unet = UNetMV2DConditionModel.from_pretrained(cfg.pretrained_unet_path, subfolder="unet", revision=cfg.revision, **cfg.unet_from_pretrained_kwargs)
    if cfg.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNetMV2DConditionModel, model_config=unet.config)

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    if cfg.trainable_modules is None:
        unet.requires_grad_(True)
    else:
        unet.requires_grad_(False)
        for name, module in unet.named_modules():
            if name.endswith(tuple(cfg.trainable_modules)):
                for params in module.parameters():
                    # print("trainable: ", params)
                    params.requires_grad = True                

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            print("use xformers to speed up")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if cfg.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if cfg.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNetMV2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetMV2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True        

    if cfg.scale_lr:
        cfg.learning_rate = (
            cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params, params_class_embedding = [], []
    for name, param in unet.named_parameters():
        if 'class_embedding' in name:
            params_class_embedding.append(param)
        else:
            params.append(param)
    optimizer = optimizer_cls(
        [
            {"params": params, "lr": cfg.learning_rate},
            {"params": params_class_embedding, "lr": cfg.learning_rate * cfg.camera_embedding_lr_mult}
        ],
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
    )

    # Get the training dataset
    train_dataset = MVDiffusionDataset(
        **cfg.train_dataset
    )
    validation_dataset = MVDiffusionDataset(
        **cfg.validation_dataset
    )
    validation_train_dataset = MVDiffusionDataset(
        **cfg.validation_train_dataset
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.dataloader_num_workers,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )
    validation_train_dataloader = torch.utils.data.DataLoader(
        validation_train_dataset, batch_size=cfg.validation_train_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if cfg.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        cfg.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        cfg.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:,None,None].to(accelerator.device, dtype=torch.float32)
    clip_image_std = torch.as_tensor(feature_extractor.image_std)[:,None,None].to(accelerator.device, dtype=torch.float32)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # tracker_config = dict(vars(cfg))
        tracker_config = {}
        accelerator.init_trackers(cfg.tracker_project_name, tracker_config)    

    # Train!
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            if os.path.exists(os.path.join(cfg.output_dir, "checkpoint")):
                path = "checkpoint"
            else:
                dirs = os.listdir(cfg.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            # global_step = int(path.split("-")[1])
            global_step = cfg.last_global_step

            resume_global_step = global_step * cfg.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * cfg.gradient_accumulation_steps)        

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, cfg.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if cfg.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % cfg.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # (B, Nv, 3, H, W)
                imgs_in, colors_out, normals_out = batch['imgs_in'], batch['imgs_out'], batch['normals_out']

                bnm, Nv = imgs_in.shape[:2]
                
                # repeat  (2B, Nv, 3, H, W)
                imgs_in = torch.cat([imgs_in]*2, dim=0)
                imgs_out = torch.cat([normals_out, colors_out], dim=0)
                
                # (2B, Nv, Nce)
                camera_embeddings = torch.cat([batch['camera_embeddings']]*2, dim=0)

                task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)

                camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

                # (B*Nv, 3, H, W)
                imgs_in, imgs_out = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W"), rearrange(imgs_out, "B Nv C H W -> (B Nv) C H W")
                # (B*Nv, Nce)
                camera_task_embeddings = rearrange(camera_task_embeddings, "B Nv Nce -> (B Nv) Nce")
                # (B*Nv, Nce')
                if cfg.camera_embedding_type == 'e_de_da_sincos':
                    camera_task_embeddings = torch.cat([
                        torch.sin(camera_task_embeddings),
                        torch.cos(camera_task_embeddings)
                    ], dim=-1)
                else:
                    raise NotImplementedError

                imgs_in, imgs_out, camera_task_embeddings = imgs_in.to(weight_dtype), imgs_out.to(weight_dtype), camera_task_embeddings.to(weight_dtype)

                # (B*Nv, 4, Hl, Wl)
                # pdb.set_trace()
                cond_vae_embeddings = vae.encode(imgs_in * 2.0 - 1.0).latent_dist.mode()
                if cfg.scale_input_latents:
                    cond_vae_embeddings = cond_vae_embeddings * vae.config.scaling_factor
                latents = vae.encode(imgs_out * 2.0 - 1.0).latent_dist.sample() * vae.config.scaling_factor

                # DO NOT use this! Very slow!                
                # imgs_in_pil = [TF.to_pil_image(img) for img in imgs_in]
                # imgs_in_proc = feature_extractor(images=imgs_in_pil, return_tensors='pt').pixel_values.to(dtype=latents.dtype, device=latents.device)

                imgs_in_proc = TF.resize(imgs_in, (feature_extractor.crop_size['height'], feature_extractor.crop_size['width']), interpolation=InterpolationMode.BICUBIC)
                # do the normalization in float32 to preserve precision
                imgs_in_proc = ((imgs_in_proc.float() - clip_image_mean) / clip_image_std).to(weight_dtype)        

                # (B*Nv, 1, 768)
                image_embeddings = image_encoder(imgs_in_proc).image_embeds.unsqueeze(1)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # same noise for different views of the same object
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz // cfg.num_views,), device=latents.device).repeat_interleave(cfg.num_views)
                timesteps = timesteps.long()                

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if cfg.use_classifier_free_guidance and cfg.condition_drop_rate > 0.:
                    # assert cfg.drop_type == 'drop_as_a_whole'
                    if cfg.drop_type == 'drop_as_a_whole':
                        # drop a group of normals and colors as a whole
                        random_p = torch.rand(bnm, device=latents.device, generator=generator)
                        
                        # Sample masks for the conditioning images.
                        image_mask_dtype = cond_vae_embeddings.dtype
                        image_mask = 1 - (
                            (random_p >= cfg.condition_drop_rate).to(image_mask_dtype)
                            * (random_p < 3 * cfg.condition_drop_rate).to(image_mask_dtype)
                        )
                        image_mask = image_mask.reshape(bnm, 1, 1, 1, 1).repeat(1, Nv, 1, 1, 1)
                        image_mask = rearrange(image_mask, "B Nv C H W -> (B Nv) C H W")
                        image_mask = torch.cat([image_mask]*2, dim=0)
                        # Final image conditioning.
                        cond_vae_embeddings = image_mask * cond_vae_embeddings

                        # Sample masks for the conditioning images.
                        clip_mask_dtype = image_embeddings.dtype
                        clip_mask = 1 - (
                            (random_p < 2 * cfg.condition_drop_rate).to(clip_mask_dtype)
                        )
                        clip_mask = clip_mask.reshape(bnm, 1, 1, 1).repeat(1, Nv, 1, 1)
                        clip_mask = rearrange(clip_mask, "B Nv M C -> (B Nv) M C")
                        clip_mask = torch.cat([clip_mask]*2, dim=0)
                        # Final image conditioning.
                        image_embeddings = clip_mask * image_embeddings
                    elif cfg.drop_type == 'drop_independent':
                        # randomly drop all independently
                        random_p = torch.rand(bsz, device=latents.device, generator=generator)

                        # Sample masks for the conditioning images.
                        image_mask_dtype = cond_vae_embeddings.dtype
                        image_mask = 1 - (
                            (random_p >= cfg.condition_drop_rate).to(image_mask_dtype)
                            * (random_p < 3 * cfg.condition_drop_rate).to(image_mask_dtype)
                        )
                        image_mask = image_mask.reshape(bsz, 1, 1, 1)
                        # Final image conditioning.
                        cond_vae_embeddings = image_mask * cond_vae_embeddings

                        # Sample masks for the conditioning images.
                        clip_mask_dtype = image_embeddings.dtype
                        clip_mask = 1 - (
                            (random_p < 2 * cfg.condition_drop_rate).to(clip_mask_dtype)
                        )
                        clip_mask = clip_mask.reshape(bsz, 1, 1)
                        # Final image conditioning.
                        image_embeddings = clip_mask * image_embeddings
                    elif cfg.drop_type == 'drop_joint':
                        # randomly drop all independently
                        random_p = torch.rand(bsz//2, device=latents.device, generator=generator)

                        # Sample masks for the conditioning images.
                        image_mask_dtype = cond_vae_embeddings.dtype
                        image_mask = 1 - (
                            (random_p >= cfg.condition_drop_rate).to(image_mask_dtype)
                            * (random_p < 3 * cfg.condition_drop_rate).to(image_mask_dtype)
                        )
                        image_mask = torch.cat([image_mask]*2, dim=0)
                        image_mask = image_mask.reshape(bsz, 1, 1, 1)
                        # Final image conditioning.
                        cond_vae_embeddings = image_mask * cond_vae_embeddings

                        # Sample masks for the conditioning images.
                        clip_mask_dtype = image_embeddings.dtype
                        clip_mask = 1 - (
                            (random_p < 2 * cfg.condition_drop_rate).to(clip_mask_dtype)
                        )
                        clip_mask = torch.cat([clip_mask]*2, dim=0)
                        clip_mask = clip_mask.reshape(bsz, 1, 1)
                        # Final image conditioning.
                        image_embeddings = clip_mask * image_embeddings
                
                # (B*Nv, 8, Hl, Wl)
                latent_model_input = torch.cat([noisy_latents, cond_vae_embeddings], dim=1)

                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=image_embeddings,
                    class_labels=camera_task_embeddings
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}") 

                if cfg.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()                    

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients and cfg.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(unet.parameters(), cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfg.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(cfg.output_dir, f"checkpoint")
                        accelerator.save_state(save_path)
                        try:
                            unet.module.save_pretrained(os.path.join(cfg.output_dir, f"unet-{global_step}/unet"))
                        except:
                            unet.save_pretrained(os.path.join(cfg.output_dir, f"unet-{global_step}/unet"))
                        logger.info(f"Saved state to {save_path}")

                if global_step % cfg.validation_steps == 0 or (cfg.validation_sanity_check and global_step == 1):
                    if accelerator.is_main_process:
                        if cfg.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        log_validation(
                            validation_dataloader,
                            vae,
                            feature_extractor,
                            image_encoder,
                            unet,
                            cfg,
                            accelerator,
                            weight_dtype,
                            global_step,
                            'validation',
                            vis_dir
                        )
                        log_validation(
                            validation_train_dataloader,
                            vae,
                            feature_extractor,
                            image_encoder,
                            unet,
                            cfg,
                            accelerator,
                            weight_dtype,
                            global_step,
                            'validation_train',
                            vis_dir
                        )                       
                        if cfg.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())                        

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if cfg.use_ema:
            ema_unet.copy_to(unet.parameters())
        pipeline = MVDiffusionImagePipeline(
            image_encoder=image_encoder, feature_extractor=feature_extractor, vae=vae, unet=unet, safety_checker=None,
            scheduler=DDIMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler"),
            **cfg.pipe_kwargs
        )            
        os.makedirs(os.path.join(cfg.output_dir, "pipeckpts"), exist_ok=True)
        pipeline.save_pretrained(os.path.join(cfg.output_dir, "pipeckpts"))

    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)
