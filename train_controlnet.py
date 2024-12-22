from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
from omegaconf import OmegaConf

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage


from torchvision.utils import make_grid, save_image

import diffusers
from diffusers import (
    AutoencoderKL,
    # ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler
)

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
# from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import _get_model_file
from diffusers.models.modeling_utils import load_state_dict
from safetensors import safe_open
from collections import defaultdict
from transformers import CLIPImageProcessor


from MVControlNet.model.controlnet import ControlNetModel
from MVControlNet.pipeline.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline

from MVControlNet.data.grid_objaverse_data import grid_refine_dataset

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union


from dataclasses import dataclass
from typing import Optional, Dict, List
from transformers import CLIPVisionModelWithProjection
import torch.multiprocessing as mp



to_pil = ToPILImage()

@dataclass
class TrainingConfig:
    # Validation configurations
    train_dataset: Dict
    validation_dataset: Dict
    vis_dir: str
    pred_type: str

    # Model and paths
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_name_or_path: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    output_dir: str = "outputs-controlnet-v0"

    # Dataset configurations
    root_dir_ortho: str = "/mvfs/multiview-data/ortho-13views"
    root_dir_persp: str = "/mvfs/workspace/data/multiview-renders/persp_13views/persp_13views"
    pred_ortho: bool = True
    pred_persp: bool = False
    object_list: str = "./datalist/datalist.json"
    invalid_list: str = None
    num_views: int = 4
    groups_num: int = 1
    bg_color: str = "three_choices"
    img_wh: Tuple[int, int] = (256, 256)
    validation: bool = False
    num_validation_samples: int = 32
    read_normal: bool = False
    read_color: bool = True
    pred_type: str = "joint"
    load_cam_type: bool = True

    # Training parameters
    seed: int = 42
    resolution: int = 1024
    train_batch_size: int = 4
    validation_batch_size: int = 1
    validation_train_batch_size: int = 1
    max_train_steps: Optional[int] = None
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-5
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    snr_gamma: Optional[float] = 5
    use_8bit_adam: bool = False
    allow_tf32: bool = False
    dataloader_num_workers: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    noise_offset: float = 0.1

    # Logging and checkpointing
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = None
    mixed_precision: Optional[str] = None
    enable_xformers_memory_efficient_attention: bool = False
    set_grads_to_none: bool = False

    # Validation
    validation_steps: int = 100
    validation_sanity_check: bool = False
    tracker_project_name: str = "train_controlnet"
    num_validation_images: int = 4



if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.19.3")

logger = get_logger(__name__)

images = []
to_tensor = ToTensor()


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def load_image_encoder():
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    )
    return image_encoder

def log_validation(
    dataloader, vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, global_step, save_dir, cfg, is_final_validation=False
):
    logger.info("Running validation... ")

    images_cond, images_gt, images_pred = [], [], defaultdict(list)
    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=accelerator.unwrap_model(controlnet),
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline.to(device=accelerator.device)

    # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    # load ip_adapter to pipeline
    image_encoder = load_image_encoder().to(accelerator.device)
    pipeline.image_encoder = image_encoder
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors")
    pipeline.set_ip_adapter_scale(1.0)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
    infer_images = []


    for i, batch in enumerate(dataloader):
        print(i, save_dir, len(infer_images))

        input_image, input_normal = batch['img_tensors_cond'].to(dtype=weight_dtype), batch['normal_tensors_cond'].to(
            dtype=weight_dtype)
        if cfg.pred_type == 'rgb_only':
            inputs = input_image
        else:
            inputs = torch.cat([input_image, input_normal], dim=0)

        controlnet_image = inputs.to(device=unet.device)

        text_color = batch['task_name_color']
        text_normal = batch['task_name_normal']

        if cfg.pred_type == 'rgb_only':
            texts = text_color
        else:
            texts = text_color + text_normal
        validation_prompt = texts

        images_cond.append(input_image)

        images_gt.append(batch["img_tensors_out"])


        input_images_front, input_normal_front = (batch['front_in_color'].to(device=unet.device),
                                                  batch['front_in_normal'].to(device=unet.device))

        NEG_PROMPT = "sketch, sculpture, hand drawing, outline, single color, NSFW, lowres, bad anatomy,bad hands, text, error, missing fingers, yellow sleeves, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,(worst quality:1.4),(low quality:1.4)"
        if cfg.pred_type == 'rgb_only':
            validation_image = input_images_front
        else:
            validation_image = torch.cat([input_images_front, input_normal_front], dim=0).to(device=unet.device)
        np_validation_image = np.array(validation_image.to("cpu"))
        print(controlnet_image[0].shape)
        with inference_ctx:
            images = pipeline(
                prompt=validation_prompt,
                neg_prompt=[NEG_PROMPT]*controlnet_image.shape[0],
                image=controlnet_image,
                ip_adapter_image=np_validation_image,
                control_image=controlnet_image,
                num_inference_steps=50,
                strength=0.2,
                generator=generator,
                guidance_scale=1.,
                output_type='pt',
            ).images

            length = len(images)
            for j in range(length):
                # image = torch.from_numpy(np.array(images[j]).astype(np.float32) / 255.).permute(2, 0, 1)
                image = images[j]
                print(torch.min(image), torch.max(image))
                infer_images.append(image)


        # image_logs.append(
        #     {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        # )

    infer_images = torch.stack(infer_images, dim=0)
    images_cond = torch.stack(images_cond, dim=0)
    images_gt = torch.stack(images_gt, dim=0)

    tracker_key = "test" if is_final_validation else "validation"
    N, B, C, H, W = images_cond.shape
    images_cond = images_cond.reshape(N*B, C, H, W)
    N, B, C, H, W = images_gt.shape
    images_gt = images_gt.reshape(N*B, C, H, W)
    print(images_gt.shape, images_cond.shape, infer_images.shape)
    img_infer = make_grid(infer_images, nrow=2, padding=0, value_range=(0, 1))
    images_cond_grid = make_grid(images_cond, nrow=2, padding=0, value_range=(0, 1))
    images_gt_grid = make_grid(images_gt, nrow=2, padding=0, value_range=(0, 1))

    print(images_gt_grid.shape, images_cond_grid.shape, img_infer.shape)
    save_image(images_cond_grid, os.path.join(save_dir, f"{global_step}-validation-cond.jpg"))
    save_image(images_gt_grid, os.path.join(save_dir, f"{global_step}-validation-gt.jpg"))
    save_image(img_infer, os.path.join(save_dir, f"{global_step}-validation-infer.jpg"))
    to_pil(validation_image[0]).save(os.path.join(save_dir, f"{global_step}-front-case.jpg"))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, global_step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    vis_dir = os.path.join(cfg.output_dir, cfg.vis_dir)
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'config.yaml'))

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
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # if args.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        #     ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, low_cpu_mem_usage=False)
    if args.controlnet_model_name_or_path:
        logger.info(f"Loading existing controlnet weights from {args.controlnet_model_name_or_path}")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, low_cpu_mem_usage=False)
        print()
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = grid_refine_dataset(
        **cfg.train_dataset
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
        num_workers=cfg.dataloader_num_workers, prefetch_factor = 4
    )

    validation_dataset = grid_refine_dataset(
        **cfg.validation_dataset
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size,
        shuffle=False, num_workers=cfg.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float16
    # accelerator.mixed_precision = args.mixed_precision
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    # load ip_adapter for pipeline
    image_encoder = load_image_encoder().to(accelerator.device)
    pipeline.image_encoder = image_encoder
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
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

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    if accelerator.is_main_process:
        experiment = Experiment(
            api_key="your_api_key",
            project_name="your_project_name",
            workspace="your_workspace"
        )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode((batch["img_tensors_out"] * 2.0 - 1.0).to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents_normal = vae.encode((batch["normal_tensors_out"] * 2.0 - 1.0).to(dtype=weight_dtype)).latent_dist.sample()
                latents_normal = latents_normal * vae.config.scaling_factor

                if cfg.pred_type == 'rgb_only':
                    latents = latents
                else:
                    latents = torch.cat([latents, latents_normal], dim=0)

                input_image, input_normal = batch['img_tensors_cond'].to(dtype=weight_dtype), batch['normal_tensors_cond'].to(dtype=weight_dtype)

                if cfg.pred_type == 'rgb_only':
                    inputs = input_image
                else:
                    inputs = torch.cat([input_image, input_normal], dim=0)

                input_images_front, input_normal_front = batch['front_in_color'].to(dtype=weight_dtype, device=latents.device), batch['front_in_normal'].to(dtype=weight_dtype, device=latents.device)
                input_fronts = torch.cat([input_images_front, input_normal_front], dim=0)
                input_fronts = np.array(input_fronts.to("cpu"))
                image_embeds = pipeline.prepare_ip_adapter_image_embeds(input_fronts,
                                                                        device=accelerator.device,
                                                                        num_images_per_prompt=1,
                                                                        ip_adapter_image_embeds=None,
                                                                        do_classifier_free_guidance=False)

                # Add image embeds for IP-Adapter
                added_cond_kwargs = (
                    {"image_embeds": image_embeds}
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # if cfg.noise_offset:
                #     # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                #     noise += args.noise_offset * torch.randn(
                #         (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                #     )
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=weight_dtype, device=latents.device)

                # Get the text embedding for conditioning
                text_color = batch['task_name_color']
                text_normal = batch['task_name_normal']
                if cfg.pred_type == 'rgb_only':
                    texts = text_color
                else:
                    texts = text_color + text_normal

                prompt_embeds, _ = pipeline.encode_prompt(
                    texts,
                    accelerator.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    prompt_embeds=None,
                )
                encoder_hidden_states = prompt_embeds
                controlnet_image = inputs.to(dtype=weight_dtype)

                # print(noisy_latents.shape, controlnet_image.shape)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual with ip_adapter
                # print(f"noisy_latents dtype: {noisy_latents.dtype}, encoder_hidden_states dtype: {encoder_hidden_states.dtype}, image_embeds dtype: {image_embeds.dtype}", weight_dtype)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

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

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0 or (cfg.validation_sanity_check and global_step == 6):
                        log_validation(
                            validation_dataloader,
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            vis_dir,
                            cfg
                        )
                        unet.to(dtype=weight_dtype)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            log_validation(
                validation_dataloader,
                vae,
                text_encoder,
                tokenizer,
                unet,
                controlnet,
                args,
                accelerator,
                weight_dtype,
                global_step,
            )

    accelerator.end_training()
    if accelerator.is_main_process:
        hyper_params = {
            "learning_rate": 0.0001,
            "steps": 60000,
            "batch_size": 256
        }
        experiment.log_parameters(hyper_params)
        log_model(experiment, model=unet, model_name="mv_controlnet_sep")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)