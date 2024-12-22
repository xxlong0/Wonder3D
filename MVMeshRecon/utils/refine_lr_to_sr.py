import torch
import os

import numpy as np
from hashlib import md5
import cv2
from PIL import Image
from rembg import remove


def sr_wonder3d_images(images):
    print(images.shape)
    images = [Image.fromarray(image) for image in images]
    refined_imgs = [img.resize((512, 512), resample=Image.LANCZOS) for img in images]
    sr_image_list = run_sr_fast(refined_imgs)
    sr_image_list_np = np.array([np.array(img) for img in sr_image_list])

    return sr_image_list_np

def sr_front_img(image, out_size=512):
    print("front img size to sr:", image.shape)
    image = Image.fromarray(image)
    image = run_sr_fast([image])[0]
    sr_image = np.array(image.resize((out_size, out_size), Image.Resampling.LANCZOS))
    print("front img size after sr:", sr_image.size)
    return sr_image


def erode_mask_np(mask: torch.Tensor, kernel_size=8):
    mask_np = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask_np_erode = cv2.erode(mask_np, kernel=kernel)

    return mask_np_erode

def sr_front(image, out_size=2048):
    print("front img size to sr:", image.size)
    mode = image.mode
    if mode != 'RGBA':
        image = remove(image)

    image = run_sr_fast([image])[0]
    # sr_image = image.resize((out_size, out_size), Image.Resampling.LANCZOS)
    sr_image_np = np.array(image)
    if sr_image_np.shape[-1] == 4:
        print('erode_mask_for_sr_img')

        mask = sr_image_np[..., 3]

        mask = erode_mask_np(mask)

        sr_image_np[..., 3] = mask

        sr_image = Image.fromarray(sr_image_np, 'RGBA')
    sr_image_np = np.array(sr_image)

    sr_image = Image.fromarray(sr_image_np, 'RGBA')
    print("front img size after sr:", sr_image.size)
    return sr_image


def hash_img(img):
    return md5(np.array(img).tobytes()).hexdigest()
def hash_any(obj):
    return md5(str(obj).encode()).hexdigest()

def refine_lr_with_sd(pil_image_list, concept_img_list, control_image_list, prompt_list, pipe=None, strength=0.35, neg_prompt_list="", output_size=(512, 512), controlnet_conditioning_scale=1.):
    with torch.no_grad():
        images = pipe(
            image=pil_image_list,
            ip_adapter_image=concept_img_list,
            prompt=prompt_list,
            neg_prompt=neg_prompt_list,
            num_inference_steps=50,
            strength=strength,
            height=output_size[0],
            width=output_size[1],
            control_image=control_image_list,
            guidance_scale=5.0,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=torch.manual_seed(233),
        ).images
    return images

SR_cache = None

def run_sr_fast(source_pils, scale=4):
    from PIL import Image
    from utils.upsampler import RealESRGANer
    import numpy as np
    global SR_cache
    if SR_cache is not None:
        upsampler = SR_cache
    else:
        from utils.rrdbnet import RRDBNet
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device="cuda"
        )
    ret_pils = []
    for idx, img_pils in enumerate(source_pils):
        np_in = isinstance(img_pils, np.ndarray)
        assert isinstance(img_pils, (Image.Image, np.ndarray))
        img = np.array(img_pils)
        output, _ = upsampler.enhance(img, outscale=scale)
        if np_in:
            ret_pils.append(output)
        else:
            ret_pils.append(Image.fromarray(output))
    if SR_cache is None:
        SR_cache = upsampler
    return ret_pils