import os
from typing import Dict, Optional, Tuple, List
from PIL import Image
import numpy as np
from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler
)

from collections import defaultdict
import rembg
from torchvision.utils import make_grid, save_image

from MVControlNet.model.controlnet import ControlNetModel
from MVControlNet.pipeline.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline

from MVControlNet.data.enhancement_dataset import InferenceImageDataset
import torchvision.transforms as transforms

to_pil = transforms.ToPILImage()

weight_dtype = torch.half

@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    controlnet_model_name_or_path: str
    revision: Optional[str]
    validation_batch_size: int

    pipe_validation_kwargs: Dict
    validation_guidance_scales: List[float]
    camera_embedding_lr_mult: float

    num_views: int


def save_image(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr

def load_image_encoder():
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    )
    return image_encoder

def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)

def unmake_grid(img_grid, nrow):
    img_grid = img_grid[0]
    _, H, W = img_grid.shape
    num_images = nrow * (H // (W // nrow))
    img_height = H // (num_images // nrow)
    img_width = W // nrow

    images = []
    for i in range(0, H, img_height):
        for j in range(0, W, img_width):
            image = img_grid[:, i:i+img_height, j:j+img_width]
            images.append(image)

    return images

def load_controlnet_pipeline(cfg):
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model_name_or_path, low_cpu_mem_usage=False)

    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    # load ip_adapter to pipeline
    image_encoder = load_image_encoder().to('cuda:0')
    pipeline.image_encoder = image_encoder
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors")
    pipeline.set_ip_adapter_scale(1.0)

    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    # sys.main_lock = threading.Lock()
    return pipeline

NEG_PROMPT = "sketch, sculpture, hand drawing, outline, single color, NSFW, lowres, bad anatomy,bad hands, text, error, missing fingers, yellow sleeves, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,(worst quality:1.4),(low quality:1.4)"

def custom_collate(batch):

    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def pred_enhancement_joint(mv_image, mv_normlas, renderd_mv_image, renderd_mv_normal, front_image, pipeline, seed=42, crop_size=192, camera_type='ortho', cfg=None, case_name='img', refine_idx=0, output_path='outputs'):
    VIEWS = ['front', 'right', 'back', 'left']
    pipeline.set_progress_bar_config(disable=True)

    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    # Get the  dataset
    validation_dataset = InferenceImageDataset(
        crop_size=crop_size,
        mv_imgs=mv_image,
        mv_normals=mv_normlas,
        renderd_mv_imgs=renderd_mv_image,
        renderd_mv_normals=renderd_mv_normal,
        front_img=front_image,
    )

    # DataLoaders creation:
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers,
        collate_fn=custom_collate
    )

    images_cond, images_gt, images_pred = [], [], defaultdict(list)
    batch = next(iter(validation_dataloader))

    # repeat  (2B, Nv, 3, H, W)
    input_image, input_normal = batch['imgs_in'].to(dtype=weight_dtype), batch['normals_in'].to(
        dtype=weight_dtype)
    input_image_mv, input_normal_mv = batch['imgs_mv'].to(dtype=weight_dtype), batch['normals_mv'].to(
        dtype=weight_dtype)

    inputs = torch.cat([input_image, input_normal], dim=0)
    inputs_mv = torch.cat([input_image_mv, input_normal_mv], dim=0)
    images_cond.append(inputs)

    controlnet_image = inputs_mv.to(device=pipeline.device)

    text_color = batch['task_name_color']
    text_normal = batch['task_name_normal']

    validation_prompt = text_color + text_normal

    input_images_front, input_normal_front = (batch['front_in_color'].to(device=pipeline.device),
                                              batch['front_in_normal'].to(device=pipeline.device))

    validation_image = torch.cat([input_images_front, input_normal_front], dim=0).to(device=pipeline.device)
    np_validation_image = np.array(validation_image.to("cpu"))

    num_views = len(VIEWS)
    with torch.autocast("cuda"):
        # B*Nv images
        if refine_idx == 0:
            strength = 0.2
            controlnet_condition_scle = 0.7
        else:
            strength = 0.1
            controlnet_condition_scle = 1.
        for guidance_scale in cfg.validation_guidance_scales:
            images = pipeline(
                prompt=validation_prompt,
                neg_prompt=[NEG_PROMPT] * controlnet_image.shape[0],
                image=controlnet_image,
                ip_adapter_image=np_validation_image,
                control_image=controlnet_image,
                num_inference_steps=50,
                strength=strength,
                height=1024,
                width=1024,
                generator=generator,
                guidance_scale=guidance_scale,
                output_type='pt',
                controlnet_conditioning_scale=controlnet_condition_scle
            ).images

            bsz = images.shape[0] // 2
            images_pred  = images[:bsz]
            normals_pred = images[bsz:]

            rm_normals_pil = []
            colors_pil = []
            for i in range(bsz):
                scene = os.path.basename(case_name.split('.')[0])
                scene_dir = os.path.join(output_path, scene, 'mv-enhancement-'+str(refine_idx), camera_type)
                normal_dir = os.path.join(scene_dir, "normals")
                color_dir = os.path.join(scene_dir, "colors")
                masked_colors_dir = os.path.join(scene_dir, "masked_colors")
                os.makedirs(normal_dir, exist_ok=True)
                os.makedirs(masked_colors_dir, exist_ok=True)
                os.makedirs(color_dir, exist_ok=True)
                normals = unmake_grid(normals_pred, 2)
                colors = unmake_grid(images_pred, 2)

                rembg_session = rembg.new_session()
                for j in range(num_views):
                    view = VIEWS[j]
                    idx = i * num_views + j

                    normal = normals[idx]
                    color = colors[idx]

                    normal_filename = f"normals_000_{view}.png"
                    rgb_filename = f"rgb_000_{view}.png"
                    normal = save_image(normal, os.path.join(normal_dir, normal_filename))
                    color = save_image(color, os.path.join(color_dir, rgb_filename))

                    rm_normal = rembg.remove(normal, alpha_matting=True, session=rembg_session)

                    save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))

                    rm_normals_pil.append(Image.fromarray(rm_normal))
                    colors_pil.append(to_pil(color))

            save_image(images_pred[0], os.path.join(scene_dir, f'color_grid_img.png'))
            save_image(normals_pred[0], os.path.join(scene_dir, f'normal_grid_img.png'))
            save_image(input_image[0], os.path.join(scene_dir, f'color_grid_img_cond.png'))
            save_image(input_normal[0], os.path.join(scene_dir, f'normal_grid_img_cond.png'))
            return rm_normals_pil, colors_pil