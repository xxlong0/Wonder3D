import os
import torch
import argparse
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

from run_mv_prediction import load_wonder3d_pipeline, pred_multiview_joint
from run_mv_enhancement import load_controlnet_pipeline, pred_enhancement_joint

from MVMeshRecon.Coarse_recon import coarse_recon
from MVMeshRecon.Iterative_refine import iterative_refine
from MVMeshRecon.utils.refine_lr_to_sr import sr_front
from rembg import remove

# step 1: Load input image and configuration
device = torch.device('cuda:0')

parser = argparse.ArgumentParser()
parser.add_argument('--config_mvdiffusion', type=str, default='configs/mvdiffusion-joint.yaml', help='Path to multi-view diffusion config.')
parser.add_argument('--config_controlnet', type=str, default='configs/controlnet.yaml', help='Path to enhancement controlnet config')
parser.add_argument('--input_path', type=str, default='example_images', help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--crop_size', type=int, default=192, help='Crop size of the input image, this is a relative num that assume the resolution of input image is 256.')
parser.add_argument('--camera_type', type=str, default='ortho', help='ortho or persp')
parser.add_argument('--num_refine',type=int, default=2, help='number of iterative refinement')
args = parser.parse_args()

config_mv = OmegaConf.load(args.config_mvdiffusion)
config_controlnet = OmegaConf.load(args.config_controlnet)

def views_6to4(imgs):
    outs = []
    for i in range(6):
        if i == 1 or i == 5:
            continue
        outs.append(imgs[i])
    return outs

def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def process_image(image_input, image_size=2048, crop_size=2048*0.75):

    if np.asarray(image_input).shape[-1] == 3:
        image_input = remove(image_input)

    if crop_size != -1:
        alpha_np = np.asarray(image_input)[:, :, 3]
        coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
        min_x, min_y = np.min(coords, 0)
        max_x, max_y = np.max(coords, 0)
        ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))

        width, height = ref_img_.size

        # upsamle the input image if the quality of input image is quite low
        if width < 400 or height < 400:
            ref_img_ = sr_front(ref_img_)

        h, w = ref_img_.height, ref_img_.width
        scale = crop_size / max(h, w)
        h_, w_ = int(scale * h), int(scale * w)
        ref_img_ = ref_img_.resize((w_, h_))
        image_input = add_margin(ref_img_, size=image_size)
    else:
        image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
        image_input = image_input.resize((image_size, image_size))

    return image_input

# load input image
if os.path.isdir(args.input_path):
    input_files = [
        os.path.join(args.input_path, file)
        for file in os.listdir(args.input_path)
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
    ]
else:
    input_files = [args.input_path]

print(f'Total number of input images: {len(input_files)}')

print('Loading mv diffusion pipeline ...')
mv_pipeline = load_wonder3d_pipeline(config_mv).to(device)

print('Loading mv enhancement pipeline ...')
enhancement_pipeline = load_controlnet_pipeline(config_controlnet).to(device)

normals_mv_out, imgs_mv_out = [], []

for i, image_file in enumerate(input_files):
    try:
        # step 2: Multiview inference
        # preprocess the input image
        input_image = Image.open(image_file)

        input_image = process_image(image_input=input_image, crop_size=args.crop_size*8, image_size=2048)

        # save front image
        os.makedirs(os.path.join(args.output_path, os.path.basename(image_file).split('.')[0]), exist_ok=True)
        input_image.save(os.path.join(args.output_path, os.path.basename(image_file).split('.')[0], 'front_img.png'))

        normals_pred, images_pred = pred_multiview_joint(input_image,
                                                         mv_pipeline,
                                                         seed=args.seed,
                                                         crop_size=args.crop_size,
                                                         camera_type=args.camera_type,
                                                         cfg=config_mv,
                                                         case_name=image_file,
                                                         output_path=args.output_path)

        normals_mv_out.append((normals_pred))
        imgs_mv_out.append((images_pred))

        # stage 3: Geometric initialize and coarse reconstruction
        vertices_init_list, faces_init_list = [], []
        rendered_imgs, rendered_normals = [], []

        mv_normals, mv_imgs = normals_mv_out[i], imgs_mv_out[i]

        rendered_rgbs, rendered_normal, vertices, faces = coarse_recon(front_image=input_image,
                                                                         rgbs=mv_imgs,
                                                                         normals=mv_normals,
                                                                         camera_type=args.camera_type,
                                                                         scence_name=os.path.basename(image_file).split('.')[0],
                                                                         crop_size=args.crop_size,
                                                                         output_path=args.output_path)
        vertices_init_list.append(vertices)
        faces_init_list.append(faces)
        rendered_imgs.append(rendered_rgbs)
        rendered_normals.append(rendered_normal)

        # stage 4: MV-Enhancement and iterative refinement
        mv_normals, mv_imgs = views_6to4(normals_mv_out[i]), views_6to4(imgs_mv_out[i])

        for refine_idx in range(args.num_refine):

            refined_vertices, refined_faces, rendered_refined_imgs, rendered_refined_normals = [], [], [], []

            rendered_mv_normals, rendered_mv_imgs = rendered_normals.pop(0), rendered_imgs.pop(0)

            normals_pred, images_pred = pred_enhancement_joint(mv_image=mv_imgs,
                                                           mv_normlas=mv_normals,
                                                           renderd_mv_image=rendered_mv_imgs,
                                                           renderd_mv_normal=rendered_mv_normals,
                                                           front_image=input_image,
                                                           pipeline=enhancement_pipeline,
                                                           seed=args.seed,
                                                           crop_size=args.crop_size,
                                                           camera_type=args.camera_type,
                                                           cfg=config_controlnet,
                                                           case_name=image_file,
                                                           refine_idx=refine_idx,
                                                           output_path=args.output_path)

            vertices_init, faces_init = vertices_init_list.pop(0), faces_init_list.pop(0)

            rendered_rgbs, rendered_normal, vertices, faces = iterative_refine(vertex_init=vertices_init,
                                                                                face_init=faces_init,
                                                                                front_image=input_image,
                                                                                rgbs=images_pred,
                                                                                normals=normals_pred,
                                                                                camera_type=args.camera_type,
                                                                                scence_name=os.path.basename(image_file).split('.')[0],
                                                                                crop_size=args.crop_size,
                                                                                output_path=args.output_path,
                                                                                refine_idx=refine_idx,
                                                                                do_sr=(refine_idx==(args.num_refine-1))
                                                                                )
            vertices_init_list.append(vertices)
            faces_init_list.append(faces)
            rendered_imgs.append(rendered_rgbs)
            rendered_normals.append(rendered_normal)
            torch.cuda.empty_cache()

    except AssertionError as e:
        print(f"error for {image_file}")

