'''
Mesh refine in coarse stage for MVLRM/AIDoll/FigGen
* vertex color
* UV + texture
'''

from glob import glob
import math
import os
import cv2
import imageio
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import trimesh
from .common.cameras.rotation_conversions import (
    euler_angles_to_matrix, matrix_to_euler_angles,
    quaternion_to_matrix, matrix_to_quaternion,
    rotation_6d_to_matrix, matrix_to_rotation_6d,
    axis_angle_to_matrix, matrix_to_axis_angle,
)
from .common.mesh.structure import Mesh
from .common.renderers.nvdiffrast.uv_dilation import uv_dilation

def rot_to_mat(rot):
    return axis_angle_to_matrix(rot)

def mat_to_rot(mat):
    return matrix_to_axis_angle(mat)

def overshoot_sigmoid(x, eps=0.1):
    return torch.sigmoid(x) / (1 - 2 * eps) + eps

class OvershootSigmoid(nn.Module):
    def __init__(self, eps=0.1) -> None:
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return overshoot_sigmoid(x, eps=self.eps)
        
def trunc_rev_sigmoid(x, eps=1e-2):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

class TruncRevSigmoid(nn.Module):
    def __init__(self, eps=1e-2) -> None:
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return trunc_rev_sigmoid(x, eps=self.eps)

def make_divisible(x, m=8):
    return int(math.ceil(x / m) * m)

# linear correspond between view and channel
def gram(x):
    a, b, c, d = x.shape
    x = x.reshape(a * b, c * d)
    return x.matmul(x.t()).div(c * d)

def compute_v_nrm_weight(v_nrm, c2ws, mode='arbitrary', exponent=4):
    '''
    v_nrm: [V, 3]
    c2ws: [M, 4, 4]
    '''
    std_normal = torch.tensor([0.0, 0.0, 1.0]).to(c2ws)
    v_nrm_cam = torch.matmul(v_nrm, c2ws[:, :3, :3])
    v_nrm_cos = (v_nrm_cam * std_normal).sum(dim=-1, keepdim=True)  # [M, V, 1]

    if mode == 'hard':
        v_weight = v_nrm_cos.clamp(0.0, 1.0)
    elif mode == 'neutral':
        v_weight = 1 - v_nrm_cos.clamp(0.0, 1.0).arccos() / math.pi * 2
    elif mode == 'soft':
        v_weight = (v_nrm_cos.clamp(0.0, 1.0).arccos() * 2).cos() / 2 + 1 / 2
    elif mode == 'arbitrary':
        v_weight = v_nrm_cos.clamp(0.0, 1.0).pow(exponent)
    else:
        raise NotImplementedError
    
    return v_weight

def remesh_pipeline(mesh, mode=0):
    mesh_cpu = trimesh.Trimesh(
        vertices=mesh.v_pos.cpu().numpy(),
        faces=mesh.t_pos_idx.cpu().numpy(),
        process=True, validate=True,
    )

    # smooth, simplify/subdivide, retopology, fix
    if mode == 0:
        mesh_cpu = trimesh.smoothing.filter_laplacian(mesh_cpu)
        mesh_cpu = mesh_cpu.subdivide_to_size(0.01)  # FIXME: triangle soup
        # trimesh.repair.broken_faces(mesh_cpu)
        # trimesh.repair.fix_inversion(mesh_cpu)
        # trimesh.repair.fix_normals(mesh_cpu)
        # trimesh.repair.fill_holes(mesh_cpu)
        # trimesh.repair.fix_winding(mesh_cpu)
    elif mode == 1:
        mesh_cpu = trimesh.smoothing.filter_laplacian(mesh_cpu)
        mesh_cpu = mesh_cpu.subdivide_loop(iterations=2)
    elif mode == 2:
        mesh_cpu = trimesh.smoothing.filter_laplacian(mesh_cpu)
        if mesh_cpu.faces.shape[0] > 100000:
            mesh_cpu = mesh_cpu.simplify_quadric_decimation(50000)
    else:
        raise NotImplementedError(f'mode {mode} is invalid')

    mesh_gpu = Mesh(
        v_pos=torch.as_tensor(mesh_cpu.vertices.copy(), dtype=torch.float32, device=mesh.device),
        t_pos_idx=torch.as_tensor(mesh_cpu.faces.copy(), dtype=torch.int32, device=mesh.device),
    )
    return mesh_gpu


class LipipsLoss(nn.Module):
    def __init__(self, input_H=512, input_W=512):
        super().__init__()
        import lpips
        self.input_H = input_H
        self.input_W = input_W
        self.model = lpips.LPIPS(net='vgg').requires_grad_(False)
    
    def forward(self, x, y):
        '''
        x, y: [N, C, H, W]
        '''
        if x.ndim > 4:
            x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        if y.ndim > 4:
            y = y.reshape(-1, y.shape[-3], y.shape[-2], y.shape[-1])
        if x.shape[-2] != self.input_H or x.shape[-1] != self.input_W:
            x = torch.nn.functional.interpolate(x, (512, 512), mode='nearest')
        if y.shape[-2] != self.input_H or y.shape[-1] != self.input_W:
            y = torch.nn.functional.interpolate(y, (512, 512), mode='nearest')
        return self.model(x, y, normalize=True).mean()


def refine_mesh_vertex(
    mesh, v_rgb, cam2world_matrices, intrinsics, input_rgb, 
    nvdiff_vertex_renderer,
    confidence=1.0, n_iters=50, 
    clean_mask_artifacts=True,
    use_vertex_normal=True, vertex_normal_mode='arbitrary', vertex_normal_exponent=4,
    optimize_extrinsics=True, optimize_intrinsics=True,
    use_sigmoid=False,
    random_resize=False, resolution_min=0.125, resolution_max=4.0,
    use_style_loss=True, use_lipips_loss=False, use_mask_loss=False,
    visualize=False,
):
    '''
    for inference and refine

    mesh: Mesh
    v_rgb: (V, 3), rgb of vertex
    cam2world_matrices: (M, 4, 4)
    intrinsics: (M ,3, 3)
    input_rgb: (M, 3, H, W) or (M, 4, H, W), source rgb images
    nvdiff_vertex_renderer: vertex renderer from nvdiffrast

    tricks:
        * clean_mask_artifacts
        * use_vertex_normal
        * use_sigmoid
        * optimize_extrinsics
        * optimize_intrinsics
        * random_resize
        * use_style_loss
        * use_lipips
        * use_mask_loss
    '''
    assert confidence >= 0.0 and confidence <= 1.0

    device = v_rgb.device
    M = cam2world_matrices.shape[0]
    H, W = input_rgb.shape[2:]
    zero_zero_zero_one = cam2world_matrices[:, 3, :].clone()
    zeros = zero_zero_zero_one[:, 0]
    ones = zero_zero_zero_one[:, 3]
    v_rgb_init = v_rgb.clone()

    if use_lipips_loss:
        lipips_loss = LipipsLoss().to(device=device)
    
    # mask from input
    if input_rgb.shape[1] == 4:
        input_mask_raw = input_rgb[:, [3], :, :]
        input_rgb = input_rgb[:, :3, :, :]
    else:
        input_mask_raw = (input_rgb < 1.0).prod(dim=1, keepdim=True).float()

    # clean island artifacts with dilation/erosion
    if clean_mask_artifacts:
        input_mask = 1 - torch.nn.functional.max_pool2d(1 - input_mask_raw, 7, 1, 3)
    else:
        input_mask = input_mask_raw

    # random resize, max resolution of nvdiffrast rendering is 2048
    if random_resize:
        ssaa = np.clip(2 * np.random.random(n_iters), resolution_min, min(resolution_max, 2048 / H))
    else:
        H_dst = make_divisible(min(H * resolution_max, 2048))
        target_rgb = torch.nn.functional.interpolate(input_rgb, (H_dst, H_dst), mode='bilinear')
        target_mask = torch.nn.functional.interpolate(input_mask, (H_dst, H_dst), mode='bilinear')
        target_mask_raw = torch.nn.functional.interpolate(input_mask_raw, (H_dst, H_dst), mode='bilinear')


    # vertex weight from normal in camera view
    if use_vertex_normal:
        v_weight = compute_v_nrm_weight(
            mesh.v_nrm, cam2world_matrices, 
            mode=vertex_normal_mode, exponent=vertex_normal_exponent
        )

    # map optimization reigon
    if use_sigmoid:
        v_rgb = trunc_rev_sigmoid(v_rgb)

    # initialize params
    v_rgb = torch.nn.Parameter(v_rgb, requires_grad=True)
    optimizer = torch.optim.AdamW([v_rgb], lr=1e-1)
    
    # add extra params
    if optimize_extrinsics:
        c2w_rot_mat = cam2world_matrices[:, :3, :3]
        c2w_rot_so3 = mat_to_rot(c2w_rot_mat)
        c2w_trans = cam2world_matrices[:, :3, 3]
        c2w_rot_so3 = torch.nn.Parameter(c2w_rot_so3, requires_grad=True)
        c2w_trans = torch.nn.Parameter(c2w_trans, requires_grad=True)
        optimizer.add_param_group({'params': [c2w_rot_so3], 'lr': 1e-3})
        optimizer.add_param_group({'params': [c2w_trans], 'lr': 1e-3})
    if optimize_intrinsics:
        fx_fy_cx_cy = torch.stack([
            intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2]
        ], dim=-1)
        fx_fy_cx_cy = torch.nn.Parameter(fx_fy_cx_cy, requires_grad=True)
        optimizer.add_param_group({'params': [fx_fy_cx_cy], 'lr': 1e-3})
    
    # optimize all params, too many if-else effects speed of iteration slightly
    for iter in range(n_iters):
        if iter > n_iters // 2:
            optimize_extrinsics = False
            optimize_intrinsics = False

        if optimize_extrinsics:
            c2w_rot_mat = rot_to_mat(c2w_rot_so3)
            # inplace op will cause error if backward twice
            c2w_mat = torch.cat([
                torch.cat([c2w_rot_mat, c2w_trans.unsqueeze(-1)], dim=2),
                zero_zero_zero_one.unsqueeze(-2),
            ], dim=1)
            cam2world_matrices = c2w_mat.detach()
        else:
            c2w_mat = cam2world_matrices
        
        if optimize_intrinsics:
            intr_mat = torch.stack([
                fx_fy_cx_cy[:, 0], zeros, fx_fy_cx_cy[:, 2],
                zeros, fx_fy_cx_cy[:, 1], fx_fy_cx_cy[:, 3],
                zeros, zeros, ones,
            ], dim=1).reshape(fx_fy_cx_cy.shape[0], 3, 3)
            intrinsics = intr_mat.detach()
        else:
            intr_mat = intrinsics
        
        if random_resize:
            H_dst = make_divisible(H * ssaa[iter])
            target_rgb = torch.nn.functional.interpolate(input_rgb, (H_dst, H_dst), mode='bilinear')
            target_mask = torch.nn.functional.interpolate(input_mask, (H_dst, H_dst), mode='bilinear')
        
        if use_vertex_normal:
            if optimize_extrinsics:  # update v_weight
                v_weight = compute_v_nrm_weight(
                    mesh.v_nrm, cam2world_matrices, 
                    mode=vertex_normal_mode, exponent=vertex_normal_exponent
                )
            v_attr = torch.cat([v_rgb.tile(v_weight.shape[0], 1, 1), v_weight], dim=-1)
        else:
            v_attr = v_rgb
        
        cur_ret_dict = nvdiff_vertex_renderer(mesh, v_attr, c2w_mat, intr_mat, H_dst)
        pred_attr = cur_ret_dict["v_attr"].permute(0, 3, 1, 2)
        
        if use_vertex_normal:
            pred_rgb = pred_attr[:, :3]
            pred_mask = cur_ret_dict["alpha"].permute(0, 3, 1, 2) * pred_attr[:, 3:]
        else:
            pred_rgb = pred_attr
            pred_mask = cur_ret_dict["alpha"].permute(0, 3, 1, 2)
        
        if use_sigmoid:
            pred_rgb = overshoot_sigmoid(pred_rgb)

        loss_dict = {}
        mask = target_mask * pred_mask.detach()
        loss_dict['rgb'] = torch.nn.MSELoss()(pred_rgb * mask, target_rgb * mask)
        if use_style_loss:
            loss_dict['rgb_style'] = torch.nn.MSELoss()(gram(pred_rgb * mask), gram(target_rgb * mask))
        if use_lipips_loss:
            loss_dict['rgb_lipips'] = lipips_loss(pred_rgb * mask, target_rgb * mask)
        if use_mask_loss:
            loss_dict['mask'] = torch.nn.BCELoss()(pred_mask.clamp(0.0, 1.0), target_mask.clamp(0.0, 1.0))
        # print({k: v.item() for k, v in loss_dict.items()})
        loss = sum(v for v in loss_dict.values())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    with torch.no_grad():
        # blend and clip
        if use_sigmoid:
            v_rgb_fin = confidence * overshoot_sigmoid(v_rgb) + (1 - confidence) * v_rgb_init
        else:
            v_rgb_fin = confidence * v_rgb + (1 - confidence) * v_rgb_init
        v_rgb_fin = v_rgb_fin.clamp(0.0, 1.0)
    
    # visualize
    if visualize:
        import cv2
        os.makedirs('./.cache', exist_ok=True)
        for i in range(pred_rgb.shape[0]):
            cv2.imwrite(f'./.cache/pred_rgb_{i}.png', pred_rgb[i].permute(1, 2, 0).detach().flip(-1).cpu().numpy() * 255)
            cv2.imwrite(f'./.cache/mask_{i}.png', input_mask_raw[i].permute(1, 2, 0).detach().flip(-1).cpu().numpy() * 255)
    return mesh, v_rgb_fin


def refine_mesh_uv(
    mesh, map_Kd, cam2world_matrices, intrinsics, input_rgb, 
    nvdiff_simple_renderer, 
    confidence=1.0, n_iters=100,
    clean_mask_artifacts=False, 
    use_vertex_normal=True, vertex_normal_mode='arbitrary', vertex_normal_exponent=4,
    optimize_extrinsics=False, optimize_intrinsics=False,
    use_sigmoid=False, use_conv=False, 
    random_resize=False, resolution_min=0.125, resolution_max=4.0,
    use_style_loss=True, use_lipips_loss=False, use_mask_loss=False, use_tv_loss=False,
    soft_blending=False, poisson_fusion_func=None,
    visualize=False,
):
    '''
    for inference and refine

    mesh: Mesh
    map_Kd: (H', W', 4), rgba of map_Kd
    cam2world_matrices: (M, 4, 4)
    intrinsics: (M ,3, 3)
    input_rgb: (M, 3, H, W) or (M, 4, H, W), source rgb images
    nvdiff_simple_renderer: vertex and uv renderer from nvdiffrast
    '''
    assert confidence >= 0.0 and confidence <= 1.0
    assert (not soft_blending) or poisson_fusion_func is not None
    if visualize:
        os.makedirs('./.cache', exist_ok=True)

    device = map_Kd.device
    M = cam2world_matrices.shape[0]
    H, W = input_rgb.shape[2:]
    zero_zero_zero_one = cam2world_matrices[:, 3, :].clone()
    zeros = zero_zero_zero_one[:, 0]
    ones = zero_zero_zero_one[:, 3]
    map_Kd_mask = map_Kd[..., [-1]].permute(2, 0, 1).unsqueeze(0)
    map_Kd_mask_boundary = (torch.nn.functional.max_pool2d(map_Kd_mask, 3, 1, 1) - \
        (1 - (torch.nn.functional.max_pool2d(1 - map_Kd_mask, 3, 1, 1)))).to(dtype=torch.bool)
    map_Kd_mask = map_Kd_mask.squeeze(0).permute(1, 2, 0)
    map_Kd_mask_origin = map_Kd_mask.clone()
    map_Kd_mask = (map_Kd_mask > 0.6).to(dtype=map_Kd_mask.dtype)
    map_Kd_mask_boundary = map_Kd_mask_boundary.squeeze(0).permute(1, 2, 0)
    map_Kd = map_Kd[..., :3]
    map_Kd_init = map_Kd.clone()

    if use_lipips_loss:
        lipips_loss = LipipsLoss().to(device=device)

    # mask from input
    if input_rgb.shape[1] == 4:
        input_mask_raw = input_rgb[:, [3], :, :]
        input_rgb = input_rgb[:, :3, :, :]
    else:
        input_mask_raw = (input_rgb < 1.0).prod(dim=1, keepdim=True).float()

    # clean island artifacts with dilation/erosion
    if clean_mask_artifacts:
        input_mask = 1 - torch.nn.functional.max_pool2d(1 - input_mask_raw, 7, 1, 3)
    else:
        input_mask = input_mask_raw

    # vertex weight from normal in camera view
    if use_vertex_normal:
        v_weight = compute_v_nrm_weight(
            mesh.v_nrm, cam2world_matrices, 
            mode=vertex_normal_mode, exponent=vertex_normal_exponent
        )
    else:
        v_weight = None

    # random resize, max resolution of nvdiffrast rendering is 2048
    if random_resize:
        ssaa = np.clip(2 * np.random.random(n_iters), resolution_min, min(resolution_max, 2048 / H))
    else:
        H_dst = make_divisible(min(H * resolution_max, 2048))
        target_rgb = torch.nn.functional.interpolate(input_rgb, (H_dst, H_dst), mode='bilinear')
        target_mask = torch.nn.functional.interpolate(input_mask, (H_dst, H_dst), mode='bilinear')

    if use_sigmoid:
        map_Kd = trunc_rev_sigmoid(map_Kd)
    map_Kd = torch.nn.Parameter(map_Kd, requires_grad=True)

    optimizer = torch.optim.AdamW([map_Kd], lr=1e-1)
    if optimize_extrinsics:
        c2w_rot_mat = cam2world_matrices[:, :3, :3]
        c2w_rot_so3 = mat_to_rot(c2w_rot_mat)
        c2w_trans = cam2world_matrices[:, :3, 3]
        c2w_rot_so3 = torch.nn.Parameter(c2w_rot_so3, requires_grad=True)
        c2w_trans = torch.nn.Parameter(c2w_trans, requires_grad=True)
        optimizer.add_param_group({'params': [c2w_rot_so3], 'lr': 1e-3})
        optimizer.add_param_group({'params': [c2w_trans], 'lr': 1e-3})
    if optimize_intrinsics:
        fx_fy_cx_cy = torch.stack([
            intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2]
        ], dim=-1)
        fx_fy_cx_cy = torch.nn.Parameter(fx_fy_cx_cy, requires_grad=True)
        optimizer.add_param_group({'params': [fx_fy_cx_cy], 'lr': 1e-3})
    if use_conv:
        tiny_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1),
        ).to(device=device).train()
        optimizer.add_param_group({'params': [p for p in tiny_cnn.parameters()], 'lr': 1e-2})

    for iter in range(n_iters):
        if iter > n_iters // 2:
            optimize_extrinsics = False
            optimize_intrinsics = False

        if optimize_extrinsics:
            c2w_rot_mat = rot_to_mat(c2w_rot_so3)
            # inplace op will cause error if backward twice
            c2w_mat = torch.cat([
                torch.cat([c2w_rot_mat, c2w_trans.unsqueeze(-1)], dim=2),
                zero_zero_zero_one.unsqueeze(-2),
            ], dim=1)
            cam2world_matrices = c2w_mat.detach()
        else:
            c2w_mat = cam2world_matrices
        
        if optimize_intrinsics:
            intr_mat = torch.stack([
                fx_fy_cx_cy[:, 0], zeros, fx_fy_cx_cy[:, 2],
                zeros, fx_fy_cx_cy[:, 1], fx_fy_cx_cy[:, 3],
                zeros, zeros, ones,
            ], dim=1).reshape(fx_fy_cx_cy.shape[0], 3, 3)
            intrinsics = intr_mat.detach()
        else:
            intr_mat = intrinsics
        
        if random_resize:
            H_dst = make_divisible(H * ssaa[iter])
            target_rgb = torch.nn.functional.interpolate(input_rgb, (H_dst, H_dst), mode='bilinear')
            target_mask = torch.nn.functional.interpolate(input_mask, (H_dst, H_dst), mode='bilinear')
        
        # texture dilation (inplace op, no grad)
        _map_Kd = map_Kd.data.permute(2, 0, 1).unsqueeze(0)
        _map_Kd_mask = map_Kd_mask.permute(2, 0, 1).unsqueeze(0)
        _map_Kd = uv_dilation(_map_Kd * _map_Kd_mask, _map_Kd_mask, max_iters=1)
        _map_Kd = _map_Kd.squeeze(0).permute(1, 2, 0)
        map_Kd.data = _map_Kd

         # update v_weight
        if use_vertex_normal and optimize_extrinsics:
            v_weight = compute_v_nrm_weight(
                mesh.v_nrm, cam2world_matrices, 
                mode=vertex_normal_mode, exponent=vertex_normal_exponent
            )

        # res-cnn is better than raw cnn
        if use_conv:
            map_Kd_input = map_Kd + tiny_cnn(map_Kd.unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0)
        else:
            map_Kd_input = map_Kd

        loss_dict = {}
        # total variation loss with boundary mask
        if use_tv_loss:
            loss_dict['gradient'] = ((torch.diff(map_Kd_input[:-1, 1:-1], dim=0).square() + \
                torch.diff(map_Kd_input[1:-1, :-1], dim=1).square()) * map_Kd_mask_boundary[1:-1, 1:-1]).sum() / (min(map_Kd_mask_boundary[1:-1, 1:-1].sum(), 1))

        cur_ret_dict = nvdiff_simple_renderer(
            mesh, v_weight, map_Kd_input, 
            c2w_mat, intr_mat, H_dst, 
            render_z_depth=False,
            render_distance=False,
            render_world_normal=False,
            render_camera_normal=False, 
            render_world_position=False,
            render_camera_position=False, 
            render_v_attr=use_vertex_normal,
            render_uv=True,
            render_map_attr=True,
            background=None,
            uv_interpolate_mode="nearest",  # NOTE: nearest: 1 pixel to 1 pixel, bilinear: 1 pixel to 4 pixels
            enable_antialis=False,
        )
        if use_vertex_normal:
            pred_mask = (cur_ret_dict["alpha"] * cur_ret_dict["v_attr"]).permute(0, 3, 1, 2)
        else:
            pred_mask = cur_ret_dict["alpha"].permute(0, 3, 1, 2)
        pred_rgb = cur_ret_dict["map_attr"].permute(0, 3, 1, 2)

        if use_sigmoid:
            pred_rgb = overshoot_sigmoid(pred_rgb)

        mask = target_mask * pred_mask.detach()
        loss_dict['rgb'] = torch.nn.MSELoss()(pred_rgb * mask, target_rgb * mask)
        if use_style_loss:
            loss_dict['rgb_style'] = torch.nn.MSELoss()(gram(pred_rgb * mask), gram(target_rgb * mask))
        if use_lipips_loss:
            loss_dict['rgb_lipips'] = lipips_loss(pred_rgb * mask, target_rgb * mask)
        if use_mask_loss:
            loss_dict['mask'] = torch.nn.BCELoss()(pred_mask.clamp(0.0, 1.0), target_mask.clamp(0.0, 1.0))
        print(f'{iter}/{n_iters}', {k: v.item() for k, v in loss_dict.items()})
        loss = sum(v for v in loss_dict.values())
        loss.backward()
        if iter == 0:
            map_Kd_grad_mask = ((map_Kd.grad.data.abs() > 0).sum(-1, keepdim=True) > 0).to(dtype=torch.float32)
            map_Kd_grad_mask_raw = map_Kd_grad_mask.to(dtype=torch.bool)
            map_Kd_grad_mask = map_Kd_grad_mask.permute(2, 0, 1).unsqueeze(0)
            map_Kd_grad_mask = torch.nn.functional.max_pool2d(map_Kd_grad_mask, 3, 1, 1)
            map_Kd_grad_mask = 1-torch.nn.functional.max_pool2d(1-map_Kd_grad_mask, 3, 1, 1)
            map_Kd_grad_mask = map_Kd_grad_mask.squeeze(0).permute(1, 2, 0)
            map_Kd_grad_mask = map_Kd_grad_mask.to(dtype=torch.bool)
        # map_Kd.grad = map_Kd.grad.masked_fill(~map_Kd_grad_mask, 0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if visualize:
            cv2.imwrite(f'./.cache/map_Kd_iter_{iter:04d}.png', map_Kd.detach().flip(-1).cpu().numpy() * 255)

    with torch.no_grad():
        # texture dilation
        map_Kd = map_Kd.permute(2, 0, 1).unsqueeze(0)
        _map_Kd_mask = map_Kd_mask.permute(2, 0, 1).unsqueeze(0)
        map_Kd = uv_dilation(map_Kd * _map_Kd_mask, _map_Kd_mask, max_iters=1000)
        map_Kd = map_Kd.squeeze(0).permute(1, 2, 0)

        # blend and clip
        if use_conv:
            map_Kd_input = map_Kd + tiny_cnn(map_Kd.unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0)
        else:
            map_Kd_input = map_Kd
        if use_sigmoid:
            map_Kd_input = overshoot_sigmoid(map_Kd_input)
        if not soft_blending:
            map_Kd_fin = confidence * map_Kd_input + (1 - confidence) * map_Kd_init
        else:
            map_Kd_fin = poisson_fusion_func(map_Kd_input.unsqueeze(0), map_Kd_init.unsqueeze(0), map_Kd_grad_mask.unsqueeze(0)).squeeze(0)
        map_Kd_fin = map_Kd_fin.clamp(0.0, 1.0)
        map_Kd_fin = torch.cat([map_Kd_fin, map_Kd_mask_origin], dim=-1)
    
    # visualize
    if visualize:
        for i in range(pred_rgb.shape[0]):
            cv2.imwrite(f'./.cache/pred_rgb_{i}.png', pred_rgb[i].permute(1, 2, 0).detach().flip(-1).cpu().numpy() * 255)
            cv2.imwrite(f'./.cache/mask_{i}.png', input_mask_raw[i].permute(1, 2, 0).detach().flip(-1).cpu().numpy() * 255)
        cv2.imwrite(f'./.cache/map_Kd.png', map_Kd_fin.detach()[..., [2, 1, 0, 3]].cpu().numpy() * 255)
        video_out_path = f"./.cache/map_Kd.mp4"
        image_paths = sorted(glob(f"./.cache/map_Kd_*.png"))
        H, W = 1024, 1024
        video = np.zeros((len(image_paths), H, W, 3), dtype=np.uint8)
        for i, image_path in enumerate(tqdm(image_paths, total=len(image_paths))):
            video[i] = cv2.putText(cv2.resize(np.flip(cv2.imread(image_path), -1), (W, H), interpolation=cv2.INTER_LINEAR_EXACT), f'step: {i:04d}', [0, 20], 0, 1, [255, 0, 255], 2)
        imageio.mimsave(video_out_path, video, fps=15)
    return mesh, map_Kd_fin

