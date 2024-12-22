'''
Mesh remapping for FigGen
* vertex color
* UV + texture
'''
import math
import os
from typing import List, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch
import pymeshlab as ml

from .common.mesh.structure import Mesh, Texture
from .common.cameras.mvp_matrix_v2 import project
from .common.renderers.nvdiffrast.renderer import NVDiffRendererBase
from .common.renderers.nvdiffrast.uv_dilation import uv_dilation

from remeshing.util.func import save_images, save_obj

def remapping_uv_texture(
    mesh: Mesh, 
    c2ws: torch.Tensor, 
    intrinsics: torch.Tensor, 
    images: torch.Tensor, 
    map_Kd: Optional[torch.Tensor]=None,
    proj: Optional[torch.Tensor]=None,
    weights: Optional[Union[List[float], np.ndarray]]=None,
    use_alpha=True, 
    use_soft_overlay=True, confidence=0.2,
    use_inpainting=True,
    use_dilation=True,
    render_size=512, 
    texture_size=1024, 
    visualize=False,
    b_perspective=True
):
    batch_size = c2ws.shape[0]
    assert weights is None or len(weights) == batch_size, \
        f'size mismatch: length of weights should be {batch_size} but {len(weights)}'
    if map_Kd is None:
        map_Kd = torch.ones_like((texture_size, texture_size, 4), dtype=c2ws.dtype, device=c2ws.device)
        map_Kd[:, :, [3]] = 1.0
    if weights is None:
        if batch_size == 8:
            weights = [2.0, 0.05, 0.2, 0.02, 1.0, 0.02, 0.2, 0.05]
        elif batch_size == 6:
            weights = [2.0, 0.05, 0.2, 1.0, 0.2, 0.05]
        elif batch_size == 4:
            weights = [2.0, 0.2, 1.0, 0.2]
        elif batch_size == 2:
            weights = [1.0, 1.0]
        else:
            weights = [1.0] * batch_size
    weights = torch.as_tensor(weights, dtype=torch.float32, device=map_Kd.device)
    if visualize:
        os.makedirs('./.cache', exist_ok=True)

    # accumulate
    map_Kd_rgb_acc, map_Kd_weight_acc = project_uv_texture(
        mesh, c2ws, intrinsics, images,
        proj=proj,
        map_Kd=map_Kd,
        weights=weights,
        use_alpha=use_alpha, 
        render_size=render_size, 
        texture_size=texture_size,
        visualize=visualize,
        b_perspective=b_perspective
    )

    if visualize:
        cv2.imwrite(f'./.cache/map_Kd_rgb_acc.png', torch.cat([map_Kd_rgb_acc[..., :3], map_Kd[..., [3]]], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
        cv2.imwrite(f'./.cache/map_Kd_weight_acc.png', torch.cat([(map_Kd_weight_acc / batch_size)[..., [0,0,0]], map_Kd[..., [3]]], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
        cv2.imwrite(f'./.cache/map_Kd_rgb_acc_div_weight_acc.png', torch.cat([torch.div(map_Kd_rgb_acc, map_Kd_weight_acc)[..., :3], map_Kd[..., [3]]], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())

    # overlay
    map_Kd_rgb_acc = torch.where(map_Kd_weight_acc > confidence, torch.div(map_Kd_rgb_acc, map_Kd_weight_acc), map_Kd_rgb_acc)
    if visualize:
        cv2.imwrite(f'./.cache/map_Kd_rgb_acc_overlay_1.png', torch.cat([map_Kd_rgb_acc[..., :3], map_Kd[..., [3]]], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
    if use_soft_overlay:
        map_Kd_rgb_acc = torch.where(map_Kd_weight_acc <= confidence, (map_Kd * (confidence - map_Kd_weight_acc) + map_Kd_rgb_acc) / confidence, map_Kd_rgb_acc)
    else:
        map_Kd_rgb_acc = torch.where(map_Kd_weight_acc <= confidence, map_Kd, map_Kd_rgb_acc)
    if visualize:
        cv2.imwrite(f'./.cache/map_Kd_rgb_acc_overlay_2.png', torch.cat([map_Kd_rgb_acc[..., :3], map_Kd[..., [3]]], dim=-1).mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
    
    # inpainting
    map_Kd_alpha = map_Kd[..., [3]]
    if use_inpainting:
        map_Kd_mask_inpainting = (map_Kd_weight_acc <= confidence) * (map_Kd_alpha > 0.6)
        map_Kd_rgb_acc_np = map_Kd_rgb_acc[..., :3].clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        map_Kd_mask_inpainting_np = map_Kd_mask_inpainting.detach().cpu().numpy().astype(np.uint8)
        map_Kd_rgb_acc_inpainting_np = cv2.inpaint(map_Kd_rgb_acc_np, map_Kd_mask_inpainting_np, -1, cv2.INPAINT_TELEA)
        if visualize:
            cv2.imwrite(
                f'./.cache/map_Kd_rgb_acc_inpainting.png', 
                np.concatenate([
                    map_Kd_rgb_acc_np[..., [2,1,0]], 
                    map_Kd_rgb_acc_inpainting_np[..., [2,1,0]], 
                    map_Kd_mask_inpainting_np[..., [0,0,0]] * 255,
                ], axis=1),
            )
        map_Kd_rgb_acc = torch.as_tensor(map_Kd_rgb_acc_inpainting_np, dtype=torch.float32, device=map_Kd_rgb_acc.device).div(255.0)
    
    if use_dilation:
        map_Kd_mask_dilation = ~(map_Kd_alpha > 0.6)
        map_Kd_rgb_acc_dilation = uv_dilation(map_Kd_rgb_acc[..., :3].permute(2, 0, 1).unsqueeze(0), ~map_Kd_mask_dilation.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        if visualize:
            cv2.imwrite(
                f'./.cache/map_Kd_rgb_acc_dilation.png', 
                torch.cat([map_Kd_rgb_acc[..., [2,1,0]], map_Kd_rgb_acc_dilation[..., [2,1,0]], map_Kd_mask_dilation.repeat_interleave(3, dim=-1)], dim=-2).clamp(0, 1).mul(255).detach().cpu().numpy(),
            )
        map_Kd_rgb_acc = map_Kd_rgb_acc_dilation
    
    map_Kd_rgb_acc = torch.cat([map_Kd_rgb_acc[..., :3], map_Kd[..., [3]]], dim=-1)
    return map_Kd_rgb_acc


def remapping_vertex_color(
    mesh: Mesh, 
    c2ws: torch.Tensor, 
    intrinsics: torch.Tensor, 
    images: torch.Tensor, 
    v_rgb: Optional[torch.Tensor]=None,
    proj: Optional[torch.Tensor]=None,
    weights: Optional[Union[List[float], np.ndarray]]=None,
    use_alpha=True, 
    use_soft_overlay=True, confidence=0.2,
    use_inpainting=True,
    render_size=512, 
    visualize=False,
    b_perspective=True,
):
    batch_size = c2ws.shape[0]
    assert weights is None or len(weights) == batch_size, \
        f'size mismatch: length of weights should be {batch_size} but {len(weights)}'
    if v_rgb is None:
        v_rgb = torch.zeros_like(mesh.v_pos)
    if weights is None:
        if batch_size == 8:
            weights = [2.0, 0.05, 0.2, 0.02, 1.0, 0.02, 0.2, 0.05]
        elif batch_size == 6:
            weights = [2.0, 0.05, 0.2, 1.0, 0.2, 0.05]
        elif batch_size == 4:
            weights = [2.0, 0.2, 1.0, 0.2]
        elif batch_size == 2:
            weights = [1.0, 1.0]
        else:
            weights = [1.0] * batch_size
    weights = torch.as_tensor(weights, dtype=torch.float32, device=v_rgb.device)
    if visualize:
        os.makedirs('./.cache', exist_ok=True)

    # accumulate
    v_rgb_acc, v_weight_acc = project_vertex_color(
        mesh, c2ws, intrinsics, images, proj=proj,
        v_rgb=v_rgb,
        weights=weights,
        use_alpha=use_alpha, 
        render_size=render_size, 
        visualize=visualize,
        b_perspective=b_perspective
    )

    if visualize:
        Texture(mesh, v_rgb=v_rgb_acc).export(f'./.cache/v_rgb_acc.obj')
        Texture(mesh, v_rgb=v_weight_acc.repeat((1,3))/torch.max(v_weight_acc)).export(f'./.cache/v_weight_acc.obj')

    # overlay
    v_rgb_acc = torch.where(v_weight_acc > confidence, torch.div(v_rgb_acc, v_weight_acc), v_rgb_acc)
    if visualize:
        Texture(mesh, v_rgb=v_rgb_acc).export(f'./.cache/v_rgb_acc_overlay_1.obj')
    if use_soft_overlay:
        v_rgb_acc = torch.where(v_weight_acc <= confidence, (v_rgb * (confidence - v_weight_acc) + v_rgb_acc) / confidence, v_rgb_acc)
    else:
        v_rgb_acc = torch.where(v_weight_acc <= confidence, v_rgb, v_rgb_acc)
    if visualize:
        Texture(mesh, v_rgb=v_rgb_acc).export(f'./.cache/v_rgb_acc_overlay_2.obj')

    # inpainting
    if use_inpainting:
        v_idx_inpainting = torch.where(v_weight_acc[:, 0] < confidence)[0]
        v_rgb_acc_inpainting = inpainting_vertex_color(mesh, v_rgb_acc, v_idx_inpainting)
        if visualize:
            Texture(mesh, v_rgb=v_rgb_acc).export(f'./.cache/v_rgb_acc_inpainting_before.obj')
            Texture(mesh, v_rgb=v_rgb_acc_inpainting).export(f'./.cache/v_rgb_acc_inpainting_after.obj')
            Texture(mesh, v_rgb=(v_weight_acc < confidence)[..., [0,0,0]]).export(f'./.cache/v_rgb_acc_inpainting_mask.obj')
        v_rgb_acc = v_rgb_acc_inpainting
        
    return v_rgb_acc


def initial_map_Kd_with_v_rgb(
    mesh: Mesh,
    v_rgb: torch.Tensor,
    texture_size=1024,
    use_pymeshlab=True,
    visualize=False,
):
    if visualize:
        os.makedirs('./.cache', exist_ok=True)
    
    if not use_pymeshlab:
        renderer = NVDiffRendererBase()
        ret = renderer.simple_inverse_rendering(
            mesh, v_rgb, None, None, None, texture_size, 
            render_v_attr=True,
            uv_interpolate_mode='nearest', enable_antialis=False
        )
        map_Kd_rgb = ret['v_attr'].squeeze(0)
    else:
        mesh, v_rgb, _ = mesh.merge_faces(v_rgb)
        mesh_ml: ml.Mesh = ml.Mesh(
            vertex_matrix=mesh.v_pos.detach().cpu().numpy(),
            face_matrix=mesh.t_pos_idx.cpu().numpy(),
            v_color_matrix=torch.cat([v_rgb, torch.ones_like(v_rgb[..., [0]])], dim=-1).detach().cpu().numpy(),
            v_tex_coords_matrix=mesh.v_tex.detach().cpu().numpy(),
        )
        meshset_ml = ml.MeshSet()
        meshset_ml.add_mesh(mesh_ml, mesh_name='model', set_as_current=True)
        meshset_ml.apply_filter('compute_texcoord_transfer_vertex_to_wedge')
        meshset_ml.apply_filter(
            'compute_texmap_from_color', textname='material_0.png', 
            textw=texture_size, texth=texture_size,
            overwrite=False, pullpush=True,
        )
        mesh_ml = meshset_ml.current_mesh()
        map_Kd_rgb: ml.Image = mesh_ml.texture('material_0.png')
        os.makedirs('debug/pymeshlab', exist_ok=True)
        map_Kd_rgb.save('debug/pymeshlab/material_0.png')
        map_Kd_rgb = np.array(Image.open('debug/pymeshlab/material_0.png').convert('RGB'), dtype=np.uint8)
        map_Kd_rgb = torch.as_tensor(map_Kd_rgb, dtype=torch.float32).div(255.0).flip(-3).to(v_rgb)

    map_Kd_alpha = mesh.compute_uv_mask(texture_size).to(v_rgb)
    map_Kd = torch.cat([map_Kd_rgb, map_Kd_alpha], dim=-1)

    if visualize:
        Texture(mesh, v_rgb, map_Kd).export(f'./.cache/map_Kd_from_v_rgb.obj', backend='open3d')
        cv2.imwrite(f'./.cache/map_Kd_from_v_rgb.png', map_Kd.mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
    return map_Kd


def project_uv_texture(
    mesh: Mesh, 
    c2ws: torch.Tensor, 
    intrinsics: torch.Tensor, 
    images: torch.Tensor,
    proj: torch.Tensor,
    map_Kd: torch.Tensor,
    weights: torch.Tensor,
    use_alpha=True, 
    render_size=512, 
    texture_size=1024,
    visualize=False,
    b_perspective=True
):
    '''
    c2ws: [B, 4, 4]
    intrinsics: [B, 3, 3]
    images: [B, 4, H, W], rgba
    map_Kd: [H, W, 4], UV map with alpha channel
    weights: [B,]

    map_Kd_rgb_acc: [H, W, 4]
    map_Kd_weight_acc: [H, W, 1]
    '''
    batch_size = c2ws.shape[0]
    renderer = NVDiffRendererBase()
    renderer.enable_orthogonal()
    if visualize:
        os.makedirs('./.cache', exist_ok=True)


    ret = renderer.simple_rendering(mesh, None, None, c2ws, intrinsics, render_size, render_cos_ray_normal=True, proj=proj, persp=b_perspective)
    images_cos = ret['cos_ray_normal'].clamp(-1.0, 0.0)
    images_arccos = ((torch.arccos(images_cos) - torch.pi / 2) / (torch.pi / 2)).clamp(0.0, 1.0)
    images_arccos_dy, images_arccos_dx = torch.gradient(images_arccos, dim=[1, 2])
    images_arccos_grad = torch.sqrt(images_arccos_dy.square() + images_arccos_dx.square())
    images_arccos_alpha = (images_arccos_grad > math.radians(10) / (math.pi / 2)).float()
    kernel_size = 2 * (render_size // 512) + 1
    if kernel_size > 1:
        images_arccos_alpha_dilate = torch.nn.functional.max_pool2d(images_arccos_alpha.permute(0, 3, 1, 2), kernel_size, 1, kernel_size // 2).permute(0, 2, 3, 1)
    else:
        images_arccos_alpha_dilate = images_arccos_alpha

    ret = renderer.simple_inverse_rendering(
        mesh, None, torch.cat([images.permute(0, 2, 3, 1), images_cos, images_arccos_alpha_dilate], dim=-1), c2ws, intrinsics, texture_size, 
        render_uv=True, render_map_attr=True, render_cos_ray_normal=True,
        uv_interpolate_mode='nearest', enable_antialis=False, proj=proj, persp=b_perspective
    )   # 将第一次render得到的RGB，cos，arccos等信息插值到uv map的空间
    map_Kd_rgb, map_Kd_cos, map_Kd_arccos_alpha = torch.split(ret['map_attr'], [4, 1, 1], dim=-1)
    map_Kd_alpha = ret['uv_alpha'] 
    if not use_alpha:
        map_Kd_alpha = torch.ones_like(map_Kd_alpha)
    # weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * map_Kd_cos.square().square() * map_Kd_alpha * (1 - map_Kd_arccos_alpha)
    weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * map_Kd_cos.square() * map_Kd_alpha * (1 - map_Kd_arccos_alpha)

    if visualize:
        for i in range(batch_size):
            cv2.imwrite(f'./.cache/images_cos_{i:04d}.png', images_cos.square().square()[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_grad_{i:04d}.png', images_arccos_grad[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_alpha_{i:04d}.png', images_arccos_alpha[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/map_Kd_cos_{i:04d}.png', map_Kd_cos.square()[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/map_Kd_rgb_{i:04d}.png', map_Kd_rgb[i].mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
            cv2.imwrite(f'./.cache/map_Kd_alpha_{i:04d}.png', map_Kd_alpha[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/map_Kd_arccos_alpha_{i:04d}.png', map_Kd_arccos_alpha[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_alpha_dilate_{i:04d}.png', images_arccos_alpha_dilate[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/weights_{i:04d}.png', weights[i].mul(255.0).detach().cpu().numpy())
    # n_view 个 batch_size 的加权和   先和第一部得到的coarse color叠加
    map_Kd_rgb_acc = ((map_Kd_alpha * map_Kd_rgb + (1 - map_Kd_alpha) * map_Kd) * weights).sum(0)  
    map_Kd_weight_acc = weights.sum(0)
    return map_Kd_rgb_acc, map_Kd_weight_acc


def project_vertex_color(
    mesh: Mesh, 
    c2ws: torch.Tensor, 
    intrinsics: torch.Tensor, 
    images: torch.Tensor,
    proj: torch.Tensor,
    v_rgb: torch.Tensor,
    weights: torch.Tensor,
    use_alpha=True, 
    render_size=512, 
    visualize=False,
    b_perspective=True,
):
    '''
    c2ws: [B, 4, 4]
    intrinsics: [B, 3, 3]
    images: [B, 4, H, W], rgba
    v_rgb: [V, 3]
    weights: [B,]

    v_rgb_acc: [V, 3]
    v_weight_acc: [V, 1]
    '''
    batch_size = c2ws.shape[0]
    renderer = NVDiffRendererBase()
    if not b_perspective:
        renderer.enable_orthogonal()
    else:
        renderer.enable_perspective()
    if visualize:
        os.makedirs('./.cache', exist_ok=True)

    v_visible_mask = renderer.get_visible_vertices(mesh, c2ws, intrinsics, render_size)
    ret = renderer.simple_rendering(mesh, None, None, c2ws, intrinsics, render_size, render_cos_ray_normal=True)
    images_cos = ret['cos_ray_normal'].clamp(-1.0, 0.0)
    images_arccos = ((torch.arccos(images_cos) - torch.pi / 2) / (torch.pi / 2)).clamp(0.0, 1.0)
    images_arccos_dy, images_arccos_dx = torch.gradient(images_arccos, dim=[1, 2])
    images_arccos_grad = torch.sqrt(images_arccos_dy.square() + images_arccos_dx.square())
    images_arccos_alpha = (images_arccos_grad > math.radians(10) / (math.pi / 2)).float()
    kernel_size = 2 * (render_size // 512) + 1
    if kernel_size > 1:
        images_arccos_alpha_dilate = torch.nn.functional.max_pool2d(images_arccos_alpha.permute(0, 3, 1, 2), kernel_size, 1, kernel_size // 2).permute(0, 2, 3, 1)
    else:
        images_arccos_alpha_dilate = images_arccos_alpha

    v_rgb_acc = torch.zeros_like(v_rgb)
    v_weight_acc = torch.zeros_like(v_rgb[..., [0]])
    for b in range(batch_size): # for each view
        v_visible = mesh.v_pos[v_visible_mask[b], :]
        v_rgb_visible = v_rgb[v_visible_mask[b], :].clone()
        v_weight_visible = torch.zeros_like(v_rgb_visible[..., [0]])

        v_ndc, _ = project(v_visible, c2ws[b], intrinsics[b], perspective=b_perspective, proj=proj)
        v_ndc_mask_valid = torch.logical_and(v_ndc > -1, v_ndc < 1).prod(dim=-1, keepdim=False).to(dtype=torch.bool)
        v_ndc_valid = v_ndc[v_ndc_mask_valid, :]
        v_rgba_valid = torch.nn.functional.grid_sample(
            images[b].unsqueeze(0),  # [B, C, H, W]
            v_ndc_valid.unsqueeze(0).unsqueeze(-2),  # [B, V, 1, 2]
            padding_mode='reflection',
            mode='bilinear',
        )  # [B, C, V, 1]
        v_cos_valid, v_arccos_alpha_valid = torch.split(torch.nn.functional.grid_sample(
            torch.cat([images_cos[[b]], images_arccos_alpha_dilate[[b]]], dim=-1).permute(0, 3, 1, 2),  # [B, C, H, W]
            v_ndc_valid.unsqueeze(0).unsqueeze(-2),  # [B, V, 1, 2]
            padding_mode='zeros',
            mode='bilinear',
        ), [1, 1], dim=1)  # [B, C, V, 1]

        v_rgba_valid = v_rgba_valid.squeeze(-1).permute(0, 2, 1).squeeze(0)
        v_rgb_valid, v_alpha_valid  = torch.split(v_rgba_valid, (3, 1), dim=-1)
        v_alpha_valid = (v_alpha_valid>0.5).float()
        if not use_alpha:  # false
            v_alpha_valid = torch.ones_like(v_alpha_valid)
        v_cos_valid = v_cos_valid.squeeze(-1).permute(0, 2, 1).squeeze(0)
        v_arccos_alpha_valid = v_arccos_alpha_valid.squeeze(-1).permute(0, 2, 1).squeeze(0)
        
        if visualize:
            mesh_tmp = Mesh(v_pos=v_visible[v_ndc_mask_valid, :][(v_alpha_valid[...,0]>0.5), :], t_pos_idx=torch.ones([0,3]).to(v_visible).long())
            tex = Texture(mesh_tmp, v_rgb_valid[(v_alpha_valid[...,0]>0.5), :] )
            tex.export(f"./.cache/visible_points_colored_{b}.obj")
            tex = Texture(mesh_tmp, v_cos_valid[(v_alpha_valid[...,0]>0.5), :].repeat((1,3)).abs() )
            tex.export(f"./.cache/visible_points_cos_{b}.obj")
        
        v_rgb_visible[v_ndc_mask_valid, :] = v_rgb_valid * v_alpha_valid # + v_rgb_visible[v_ndc_mask_valid, :] * (1 - v_alpha_valid)
        v_weight_visible[v_ndc_mask_valid, :] = weights[b].unsqueeze(-1) * v_cos_valid.square() * v_alpha_valid #.square() * (1 - v_arccos_alpha_valid)
        v_rgb_acc[v_visible_mask[b], :] = v_rgb_acc[v_visible_mask[b], :] + v_weight_visible * v_rgb_visible * v_alpha_valid
        v_weight_acc[v_visible_mask[b], :] = v_weight_acc[v_visible_mask[b], :] + v_weight_visible * v_alpha_valid
    
    if visualize:
        for i in range(batch_size):
            # cv2.imwrite(f'./.cache/images_cos_{i:04d}.png', images_cos.square().square()[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_cos_{i:04d}.png', images_cos.square()[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_grad_{i:04d}.png', images_arccos_grad[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_alpha_{i:04d}.png', images_arccos_alpha[i].mul(255.0).detach().cpu().numpy())
            cv2.imwrite(f'./.cache/images_arccos_alpha_dilate_{i:04d}.png', images_arccos_alpha_dilate[i].mul(255.0).detach().cpu().numpy())
    return v_rgb_acc, v_weight_acc
    
    

def inpainting_vertex_color(mesh: Mesh, v_rgb: torch.Tensor, v_idx_inpainting: torch.Tensor, max_iters=1000):
    '''
    v_rgb: [V, 3]
    v_idx_inpainting: [V_inpainting,]
    '''
    device = v_rgb.device
    L = mesh.laplacian()  # [V, V]
    
    v_mask = torch.ones((v_rgb.shape[0], 1), dtype=torch.float32, device=device)
    v_mask[v_idx_inpainting, :] = 0
    v_mask_cnt = v_mask.sum()
    L_invalid = torch.index_select(L, 0, v_idx_inpainting)    # [V_inpainting, V]
    for _ in range(max_iters):
        v_rgb_mean = torch.matmul(L_invalid, v_rgb * v_mask)  # [V_inpainting, 3]
        v_mask_mean = torch.matmul(L_invalid, v_mask)  # [V_inpainting, 1]
        v_rgb[v_idx_inpainting, :] = torch.where(v_mask_mean > 0, v_rgb_mean / v_mask_mean, v_rgb[v_idx_inpainting, :])
        v_mask[v_idx_inpainting, :] = (v_mask_mean > 0).to(dtype=torch.float32)
        v_mask_cnt_cur = v_mask.sum()
        if v_mask_cnt_cur > v_mask_cnt:
            v_mask_cnt = v_mask_cnt_cur
        else:
            break
    return v_rgb

