import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch

from utils.general_utils import c_n2w_n, w_n2c_n , rotate_point_cloud
from utils.w3d_utils import load_a_from_pils, make_wonder3D_cameras
from Meshrecon import MeshRecon
from CoarseMeshRecon.CoarseRecon import CoarseRecon
from remeshing.util.func import save_images, save_obj

from utils.refine_lr_to_sr import sr_wonder3d_images, sr_front_img

import torch
import torch.nn.functional as F
from PIL import Image


VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

def c2w_to_w2c(c2w:torch.Tensor):
    # y = Rx + t, x = R_inv(y - t)
    w2c = torch.zeros((*c2w.shape[:-2], 4, 4), dtype=c2w.dtype, device=c2w.device)
    w2c[..., :3, :3] = c2w[..., :3, :3].transpose(-1, -2)
    w2c[..., :3, 3:] = -c2w[..., :3, :3].transpose(-1, -2) @ c2w[..., :3, 3:]
    w2c[..., 3, 3] = 1.0
    return w2c

do_sr = False

def transoform_rendered_to_pils(images):

    if len(images) == 6:
        view_names = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
    else:
        view_names = ['front', 'right', 'back', 'left']

    outs = []
    for i, view_name in enumerate(view_names):
        if i == 1 or i == 5:
            continue

        image = Image.fromarray((images.detach()[i, :, :, :] * 255).clamp(max=255).type(torch.uint8).cpu().numpy())
        outs.append(image)
    return outs


def coarse_recon(front_image, rgbs, normals, camera_type, scence_name, crop_size, output_path='outputs'):

        # torch.cuda.empty_cache()
        weights = None
        mv, proj = make_wonder3D_cameras(cam_type=camera_type)
        # change the data loading
        RGBs, normal_masks, normals, normals_world, c2ws, w2cs, color_masks, front_img = load_a_from_pils(front_image=front_image,
                                                                                          rgbs=rgbs,
                                                                                          rm_normal=normals,
                                                                                          imSize=[256,256],
                                                                                          view_types=['front', 'front_right', 'right', 'back', 'left', 'front_left'],
                                                                                          load_color=True,
                                                                                          cam_pose_dir='mv_diffusion_30/data/fixed_poses/nine_views',
                                                                                          normal_system='front',
                                                                                          crop_size=crop_size)

        normal_masks, color_masks = torch.tensor(normal_masks).cuda().float(), torch.tensor(color_masks).cuda().float()

        RGBs[0] = front_img[..., :3]
        color_masks[0] = torch.tensor(front_img[..., 3])

        if do_sr:
            RGBs = sr_wonder3d_images(RGBs)
            normals_world = sr_wonder3d_images(normals_world)
            h, w = RGBs.shape[1:3]
            color_masks = F.interpolate(color_masks.unsqueeze(1), size=(h, w), mode='bilinear',
                                        align_corners=False).squeeze(1)
            normal_masks = F.interpolate(normal_masks.unsqueeze(1), size=(h, w), mode='bilinear',
                                         align_corners=False).squeeze(1)


        RGBs, normals, normals_world, c2ws, w2cs = torch.tensor(RGBs).cuda().float()/255,  torch.tensor(normals).cuda().float()/255, torch.tensor(normals_world).cuda().float()/255, torch.tensor(c2ws).cuda().float(), torch.tensor(w2cs).cuda().float()

        normal_masks, color_masks = torch.tensor(normal_masks).cuda().float()/255, torch.tensor(color_masks).cuda().float()/255

        RGBs = torch.cat([RGBs, color_masks.unsqueeze(-1)], dim=-1)
        normals = torch.cat([normals, normal_masks.unsqueeze(-1)], dim=-1)
        normals_world = torch.cat([normals_world, normal_masks.unsqueeze(-1)], dim=-1)

        if len(proj.shape) == 2:
            proj = proj[None, ...].repeat([len(normals_world),1,1])

        output_path = os.path.join(output_path, scence_name, 'coarse_recon', camera_type)

        if weights is None:
            if len(RGBs) == 8:
                weights = [2.0, 0.05, 0.2, 0.02, 1.0, 0.02, 0.2, 0.05]
            elif len(RGBs) == 6:
                weights = [2.0, 0.05, 0.2, 1.0, 0.2, 0.05]
            elif len(RGBs) == 4:
                # weights = [2.0, 0.2, 1.0, 0.2]
                weights = [1.0, 0.5, 1.0, 0.5]
            elif len(RGBs) == 2:
                weights = [1.0, 1.0]
            else:
                raise ValueError("weights is None, and can not be guessed from image_list")

        normals = w_n2c_n(normals_world, mv)

        if camera_type == 'persp':
            persp = True
        else:
            persp = False

        vertices_init, faces_init = CoarseRecon(front_normal=normals[0], back_normal=normals[3], side_normal=normals[2], output_path=output_path, is_persp=persp)

        if vertices_init is not None:
            save_obj(vertices_init, faces_init, f"{output_path}/3d_model/init_3d_model.obj")
        else:
            print("for sunken case use sphere as init!")

        texture, rgb_rendered, normals_rendered, vertices, faces = MeshRecon(mv_normal=mv, proj_normal=proj, gt_normals=normals_world, mv_RGB=mv, proj_RGB=proj, gt_RGBs=RGBs, vertices_init=vertices_init, faces_init=faces_init,
                            RGB_refine_index=np.array([0,1,2,3,4,5]), RGB_view_weights=np.array(weights), debug_path=output_path, clean_mesh=False, persp=persp, output_path=output_path)

        obj_path = f'{output_path}/3d_model/model.glb'

        texture.export(obj_path)
        rgb_rendered = transoform_rendered_to_pils(rgb_rendered)
        normals_rendered = transoform_rendered_to_pils(normals_rendered)

        return rgb_rendered, normals_rendered, vertices, faces
