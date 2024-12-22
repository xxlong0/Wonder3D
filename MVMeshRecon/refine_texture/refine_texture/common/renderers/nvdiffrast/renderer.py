import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr

from ...cameras.mvp_matrix_v2 import (
    intr_to_proj,
    proj_to_intr,
    c2w_to_w2c,
    get_mvp_mtx,
    discretize,
    undiscretize,
    project,
    unproject,
)
from ...mesh.topology import erode_face

def draw_mask(v_pos_pix:torch.Tensor, H:int, W:int):
    pre_shape = v_pos_pix.shape[:-2]
    image_mask = torch.zeros((*pre_shape, H, W, 1), dtype=torch.bool, device=v_pos_pix.device)
    ui, vi = v_pos_pix.unbind(-1)
    points_mask = torch.logical_and(torch.logical_and(ui >= 0, ui <= W-1), torch.logical_and(vi >= 0, vi <= H-1))
    idx_spatial = vi * W + ui
    idx_spatial = torch.masked_select(idx_spatial, points_mask).reshape(*pre_shape, -1)
    image_mask = torch.scatter(image_mask.reshape(*pre_shape, -1, 1), -2, idx_spatial.unsqueeze(-1), 1).reshape(*pre_shape, H, W, 1)
    return image_mask


class NVDiffRendererBase(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.enable_nvdiffrast_cuda_ctx()
        # self.enable_orthogonal()
        self.erode_neighbor = 0
    
    def enable_orthogonal(self):
        self.intr_to_proj = lambda intr: intr_to_proj(intr, perspective=False)
    
    def enable_perspective(self):
        self.intr_to_proj = lambda intr: intr_to_proj(intr, perspective=True)
    
    def enable_nvdiffrast_cuda_ctx(self):
        self.ctx = dr.RasterizeCudaContext(device=self.device)
    
    def enable_nvdiffrast_opengl_ctx(self):
        self.ctx = dr.RasterizeGLContext(device=self.device)

    def get_visible_faces(self, mesh, c2ws:Tensor, intrinsics:Tensor, render_size:int):
        batch_size = c2ws.shape[0]
        device = c2ws.device
        height, width = render_size, render_size

        v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
        w2cs_mtx = c2w_to_w2c(c2ws)
        proj_mtx = self.intr_to_proj(intrinsics)
        mvp_mtx = get_mvp_mtx(proj_mtx, w2cs_mtx)
        v_pos_clip = torch.matmul(v_pos_homo, mvp_mtx.permute(0, 2, 1))
        t_pos_idx = mesh.t_pos_idx.to(dtype=torch.int32)
        
        rast, _ = dr.rasterize(self.ctx, v_pos_clip, t_pos_idx, (height, width))
        t_idx_visible_dummy = torch.as_tensor(rast[..., [3]], dtype=torch.int64).reshape(batch_size, -1)
        t_mask_visible_dummy = torch.zeros((batch_size, t_pos_idx.shape[0] + 1), dtype=torch.bool, device=device)
        t_mask_visible_dummy = torch.scatter(t_mask_visible_dummy, -1, t_idx_visible_dummy, 1) # 标记每个纹理坐标是否可见的 0-1 mask
        t_mask_visible = t_mask_visible_dummy[:, 1:]  # rast的三角形编号从1开始
        if self.erode_neighbor > 0:
            for b in range(batch_size):
                t_mask_visible[b] = erode_face(mesh.t_pos_idx, t_mask_visible[b], mesh.v_pos.shape[0], self.erode_neighbor)
        return t_mask_visible  #每个顶点是否至少在一个三角形可见，bool
    
    def get_visible_vertices(self, mesh, c2ws:Tensor, intrinsics:Tensor, render_size:int):
        batch_size = c2ws.shape[0]
        device = c2ws.device

        t_pos_idx = mesh.t_pos_idx
        t_mask_visible = self.get_visible_faces(mesh, c2ws, intrinsics, render_size)
        v_pos_mask_visible = torch.zeros((batch_size, mesh.v_pos.shape[0]), dtype=torch.bool, device=device)
        for b in range(batch_size):
            t_pos_idx_visible = torch.masked_select(t_pos_idx, t_mask_visible[b].unsqueeze(-1))
            t_pos_idx_visible_unique = torch.unique(t_pos_idx_visible, return_inverse=False, return_counts=False)
            v_pos_mask_visible[b] = torch.index_fill(v_pos_mask_visible[b], -1, t_pos_idx_visible_unique, 1)
        return v_pos_mask_visible
    
    def simple_rendering(
        self, mesh, v_attr:Tensor, map_attr:Tensor, 
        c2ws:Tensor, intrinsics:Tensor, render_size:int, 
        render_all_point_cloud=False,
        render_visible_point_cloud=False,
        render_z_depth=False,
        render_distance=False,
        render_world_normal=False,
        render_camera_normal=False, 
        render_world_position=False,
        render_camera_position=False,
        render_ray_direction=False,
        render_cos_ray_normal=False,
        render_v_attr=False,
        render_uv=False,
        render_map_attr=False,
        background=None,
        uv_interpolate_mode="bilinear",
        enable_antialis=True,
        proj=None,
        persp=None,
        **kwargs,
    ):
        '''
        v_attr: [V, C] or [M, V, C], vertex attribute of mesh
        map_attr: [H, W, C] or [M, H, W, C], UV map attribute of mesh
        c2ws: [M, 4, 4]
        intrinsics: [M, 3, 3], normalized
        background: [M, H, W, C] or [H, W, C] or [C,] or float in range(0, 1) or None
        '''
        batch_size = c2ws.shape[0]
        height, width = render_size, render_size

        v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
        w2cs_mtx = c2w_to_w2c(c2ws)
        if proj is not None:
            proj_mtx = proj[0]
        else:
            proj_mtx = intr_to_proj(intrinsics, perspective=persp)

        if kwargs.get('mvps') is not None:
            print('using input mvps')
            mvp_mtx = kwargs['mvps']
            mvp_mtx = mvp_mtx.view(*intrinsics.shape[:-2], 4, 4)
        else:
            mvp_mtx = get_mvp_mtx(proj_mtx, w2cs_mtx)

        v_pos_clip = torch.matmul(v_pos_homo, mvp_mtx.permute(0, 2, 1))
        v_pos_clip = v_pos_clip / v_pos_clip[:, :, [3]]
        t_pos_idx = mesh.t_pos_idx.to(dtype=torch.int32)

        rast, _ = dr.rasterize(self.ctx, v_pos_clip, t_pos_idx, (height, width))
        mask = rast[..., [3]] > 0
        if enable_antialis:
            alpha = dr.antialias(mask.float(), rast, v_pos_clip, t_pos_idx)
        else:
            alpha = mask.float()
        out = {"mask": mask, "alpha": alpha}

        if render_all_point_cloud or render_visible_point_cloud:
            v_pos_ndc = v_pos_clip[:, :, :2] / v_pos_clip[:, :, [3]]
            v_pos_pix = discretize(v_pos_ndc, height, width)
            if render_all_point_cloud:
                all_point_cloud = draw_mask(v_pos_pix, height, width)
                out.update({"all_point_cloud": all_point_cloud})
            if render_visible_point_cloud:  # FIXME: batch version
                t_pos_visible = torch.masked_select(rast[0, :, :, 3].to(dtype=torch.int64), mask[0, :, :, 0]).sub(1)
                t_pos_visible_unique = torch.unique(t_pos_visible, return_inverse=False, return_counts=False)
                v_pos_idx_visible = torch.index_select(t_pos_idx, 0, t_pos_visible_unique).reshape(-1)
                v_pos_idx_visible_unique = torch.unique(v_pos_idx_visible, return_inverse=False, return_counts=False)
                v_pos_pix_visible = v_pos_pix[0, v_pos_idx_visible_unique, :]
                visible_point_cloud = draw_mask(v_pos_pix_visible, height, width).unsqueeze(0)
                out.update({"visible_point_cloud": visible_point_cloud})

        if render_z_depth:
            z_depth, _ = dr.interpolate(v_pos_clip[:, :, [3]].contiguous(), rast, t_pos_idx)
            if enable_antialis:
                z_depth = dr.antialias(z_depth, rast, v_pos_clip, t_pos_idx)
            z_depth = torch.lerp(torch.zeros_like(z_depth), z_depth, alpha)
            out.update({"z_depth": z_depth})

        # if render_world_normal:
        #     world_normal, _ = dr.interpolate(mesh.v_nrm, rast, t_pos_idx)
        #     world_normal = F.normalize(world_normal, dim=-1)
        #     if enable_antialis:
        #         world_normal = dr.antialias(world_normal, rast, v_pos_clip, t_pos_idx)
        #     if background is None:
        #         gb_bg = torch.ones_like(world_normal)
        #     elif isinstance(background, float) and background >= 0.0 and background <= 1.0:
        #         gb_bg = torch.full_like(world_normal, fill_value=background)
        #     elif isinstance(background, torch.Tensor):
        #         gb_bg = background.to(world_normal).expand_as(world_normal)
        #     else:
        #         raise NotImplementedError
        #     world_normal = torch.lerp(gb_bg, world_normal, alpha)
        #     out.update({"world_normal": world_normal})
        if render_world_normal:
            world_normal, _ = dr.interpolate(mesh.v_nrm, rast, t_pos_idx)
            world_normal = F.normalize(world_normal, dim=-1)
            if enable_antialis:
                world_normal = dr.antialias(world_normal, rast, v_pos_clip, t_pos_idx)
            if background is None:
                gb_bg = torch.ones_like(world_normal)
            elif isinstance(background, float) and background >= 0.0 and background <= 1.0:
                gb_bg = torch.full_like(world_normal, fill_value=background)
            elif isinstance(background, torch.Tensor):
                gb_bg = background.to(world_normal).expand_as(world_normal)
            else:
                raise NotImplementedError
            world_normal = torch.lerp(gb_bg, world_normal, alpha)
            out.update({"world_normal": world_normal})

        if render_camera_normal:
            v_nrm_cam = torch.matmul(mesh.v_nrm, c2ws[:, :3, :3])
            v_nrm_cam = torch.nn.functional.normalize(v_nrm_cam, dim=-1)
            camera_normal, _ = dr.interpolate(v_nrm_cam, rast, t_pos_idx)
            camera_normal = torch.nn.functional.normalize(camera_normal, dim=-1)
            if enable_antialis:
                camera_normal = dr.antialias(camera_normal, rast, v_pos_clip, t_pos_idx)
            camera_normal = torch.lerp(torch.full_like(camera_normal, fill_value=-1.0), camera_normal, alpha)
            out.update({"camera_normal": camera_normal})

        if render_world_position:
            world_position, _ = dr.interpolate(mesh.v_pos, rast, t_pos_idx)
            if enable_antialis:
                world_position = dr.antialias(world_position, rast, v_pos_clip, t_pos_idx)
            world_position = torch.lerp(torch.full_like(world_position, fill_value=-1.0), world_position, alpha)
            out.update({"world_position": world_position})

        if render_camera_position or render_distance or render_ray_direction:
            v_pos_cam = torch.matmul(v_pos_homo, w2cs_mtx.permute(0, 2, 1))[:, :, :3].contiguous()
            camera_position, _ = dr.interpolate(v_pos_cam, rast, t_pos_idx)
            if render_camera_position:
                if enable_antialis:
                    camera_position = dr.antialias(camera_position, rast, v_pos_clip, t_pos_idx)
                camera_position = torch.lerp(torch.full_like(camera_position, fill_value=0.0), camera_position, alpha)
                out.update({"camera_position": camera_position})
            if render_distance:
                distance = torch.norm(camera_position, p=2, dim=-1, keepdim=True)
                if enable_antialis:
                    distance = dr.antialias(distance, rast, v_pos_clip, t_pos_idx)
                distance = torch.lerp(torch.full_like(distance, fill_value=0.0), distance, alpha)
                out.update({"distance": distance})
            if render_ray_direction:
                ray_direction = torch.nn.functional.normalize(camera_position, dim=-1)
                if enable_antialis:
                    ray_direction = dr.antialias(ray_direction, rast, v_pos_clip, t_pos_idx)
                ray_direction = torch.lerp(torch.full_like(ray_direction, fill_value=-1.0), ray_direction, alpha)
                out.update({"ray_direction": ray_direction})

        if render_cos_ray_normal:
            v_nrm_cam = torch.matmul(mesh.v_nrm, c2ws[:, :3, :3])
            v_nrm_cam = torch.nn.functional.normalize(v_nrm_cam, dim=-1)
            camera_normal, _ = dr.interpolate(v_nrm_cam, rast, t_pos_idx)
            camera_normal = torch.nn.functional.normalize(camera_normal, dim=-1)
            v_pos_cam = torch.matmul(v_pos_homo, w2cs_mtx.permute(0, 2, 1))[:, :, :3].contiguous()
            camera_position, _ = dr.interpolate(v_pos_cam, rast, t_pos_idx)
            ray_direction = torch.nn.functional.normalize(camera_position, dim=-1)
            cos_ray_normal = torch.sum(camera_normal * ray_direction, dim=-1, keepdim=True)
            if enable_antialis:
                    cos_ray_normal = dr.antialias(cos_ray_normal, rast, v_pos_clip, t_pos_idx)
            cos_ray_normal = torch.lerp(torch.full_like(cos_ray_normal, fill_value=0.0), cos_ray_normal, alpha)
            out.update({"cos_ray_normal": cos_ray_normal})

        if render_v_attr:
            gb_v_attr, _ = dr.interpolate(v_attr, rast, t_pos_idx)
            if enable_antialis:
                gb_v_attr = dr.antialias(gb_v_attr, rast, v_pos_clip, t_pos_idx)
            if background is None:
                gb_bg = torch.ones_like(gb_v_attr)
            elif isinstance(background, float) and background >= 0.0 and background <= 1.0:
                gb_bg = torch.full_like(gb_v_attr, fill_value=background)
            elif isinstance(background, torch.Tensor):
                gb_bg = background.to(gb_v_attr).expand_as(gb_v_attr)
            else:
                raise NotImplementedError
            gb_v_attr = torch.lerp(gb_bg, gb_v_attr, alpha)
            out.update({"v_attr": gb_v_attr})

        if render_uv:
            v_tex_ndc = mesh.v_tex * 2.0 - 1.0
            t_tex_idx = mesh.t_tex_idx.to(torch.int32)
            gb_uv, _ = dr.interpolate(v_tex_ndc, rast, t_tex_idx)
            gb_bg = torch.full_like(gb_uv, fill_value=-1.0)
            gb_uv = torch.lerp(gb_bg, gb_uv, mask.float())
            if enable_antialis:
                gb_uv_aa = dr.antialias(gb_uv, rast, v_pos_clip, t_pos_idx)
                gb_uv_aa = torch.lerp(gb_bg, gb_uv_aa, alpha)
                out.update({"uv": gb_uv_aa})
            else:
                out.update({"uv": gb_uv})

            if render_map_attr:
                map_attr = map_attr.expand(batch_size, *map_attr.shape[-3:])
                gb_map_attr = torch.nn.functional.grid_sample(map_attr.permute(0, 3, 1, 2), gb_uv, mode=uv_interpolate_mode).permute(0, 2, 3, 1)
                if enable_antialis:
                    gb_map_attr = dr.antialias(gb_map_attr.contiguous(), rast, v_pos_clip, t_pos_idx)
                if background is None:
                    gb_bg = torch.ones_like(gb_map_attr)
                elif isinstance(background, float) and background >= 0.0 and background <= 1.0:
                    gb_bg = torch.full_like(gb_map_attr, fill_value=background)
                elif isinstance(background, torch.Tensor):
                    gb_bg = background.to(gb_map_attr).expand_as(gb_map_attr)
                else:
                    raise NotImplementedError
                gb_map_attr = torch.lerp(gb_bg, gb_map_attr, alpha)
                out.update({"map_attr": gb_map_attr})
                
        return out
    
    def simple_inverse_rendering(
        self, mesh, v_attr:Tensor, map_attr:Tensor, 
        c2ws:Tensor, intrinsics:Tensor, render_size:int, 
        render_all_point_cloud=False,  # prepare for multi texture map
        render_visible_point_cloud=False,  # prepare for multi texture map
        render_z_depth=False,
        render_distance=False,
        render_world_normal=False,
        render_camera_normal=False, 
        render_world_position=False,
        render_camera_position=False,
        render_ray_direction=False,
        render_cos_ray_normal=False,
        render_v_attr=False,
        render_uv=False,
        render_map_attr=False,
        background=None,
        uv_interpolate_mode="bilinear",
        enable_antialis=True,
        proj=None,
        persp=None,
        **kwargs,
    ):

        height, width = render_size, render_size
        
        v_tex_ndc = mesh.v_tex * 2.0 - 1.0  # (-1,1)
        v_tex_clip = torch.cat([v_tex_ndc, torch.zeros_like(v_tex_ndc[:, [0]]), torch.ones_like(v_tex_ndc[:, [0]])], dim=-1).unsqueeze(0)
        t_tex_idx = mesh.t_tex_idx.to(dtype=torch.int32)

        rast, _ = dr.rasterize(self.ctx, v_tex_clip, t_tex_idx, (height, width)) # 纹理坐标光栅化
        mask = rast[..., [3]] > 0
        if enable_antialis:
            alpha = dr.antialias(mask.float(), rast, v_tex_clip, t_tex_idx)
        else:
            alpha = mask.float()
        out = {"mask": mask, "alpha": alpha}

        if c2ws is not None:
            batch_size = c2ws.shape[0]
            rast_duplicated = rast.tile(batch_size, 1, 1, 1)
            v_tex_clip_duplicated = v_tex_clip.tile(batch_size, 1, 1)
        else:
            batch_size = None
            rast_duplicated = None
            v_tex_clip_duplicated = None

        # if render_world_normal:
        #     world_normal, _ = dr.interpolate(mesh.v_nrm, rast, t_tex_idx)
        #     world_normal = F.normalize(world_normal, dim=-1)
        #     if enable_antialis:
        #         world_normal = dr.antialias(world_normal, rast, v_tex_clip, t_tex_idx)
        #     world_normal = torch.lerp(torch.full_like(world_normal, fill_value=-1.0), world_normal, alpha)
        #     out.update({"world_normal": world_normal})

        if render_world_normal:
            world_normal, _ = dr.interpolate(mesh.v_nrm, rast, t_pos_idx)
            world_normal = F.normalize(world_normal, dim=-1)
            if enable_antialis:
                world_normal = dr.antialias(world_normal, rast, v_pos_clip, t_pos_idx)
            if background is None:
                gb_bg = torch.ones_like(world_normal)
            elif isinstance(background, float) and background >= 0.0 and background <= 1.0:
                gb_bg = torch.full_like(world_normal, fill_value=background)
            elif isinstance(background, torch.Tensor):
                gb_bg = background.to(world_normal).expand_as(world_normal)
            else:
                raise NotImplementedError
            world_normal = torch.lerp(gb_bg, world_normal, alpha)

            # 将法向量映射到 RGB 空间
            world_normal_rgb = (world_normal + 1.0) / 2.0  # 从 [-1, 1] 映射到 [0, 1]

            # 获取深度值
            z_depth, _ = dr.interpolate(v_pos_clip[:, :, [3]].contiguous(), rast, t_pos_idx)
            if enable_antialis:
                z_depth = dr.antialias(z_depth, rast, v_pos_clip, t_pos_idx)
            z_depth = torch.lerp(torch.zeros_like(z_depth), z_depth, alpha)

            # 将 RGB 和深度组合成 RGBD
            rgbd = torch.cat((world_normal_rgb, z_depth), dim=-1)

            out.update({"world_normal_rgbd": rgbd})



        if render_camera_normal:
            v_nrm_cam = torch.matmul(mesh.v_nrm, c2ws[:, :3, :3])
            v_nrm_cam = F.normalize(v_nrm_cam, dim=-1)
            camera_normal, _ = dr.interpolate(v_nrm_cam, rast_duplicated, t_tex_idx)
            camera_normal = F.normalize(camera_normal, dim=-1)
            if enable_antialis:
                camera_normal = dr.antialias(camera_normal, rast_duplicated, v_tex_clip_duplicated, t_tex_idx)
            camera_normal = torch.lerp(torch.full_like(camera_normal, fill_value=-1.0), camera_normal, alpha)
            out.update({"camera_normal": camera_normal})

        if render_world_position:
            world_position, _ = dr.interpolate(mesh.v_pos, rast, t_tex_idx)  # 每个纹理坐标对应的世界坐标
            if enable_antialis:
                world_position = dr.antialias(world_position, rast, v_tex_clip, t_tex_idx)
            world_position = torch.lerp(torch.full_like(world_position, fill_value=-1.0), world_position, alpha)
            out.update({"world_position": world_position})

        if render_camera_position or render_distance or render_z_depth or render_ray_direction:
            batch_size = c2ws.shape
            v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
            w2cs_mtx = c2w_to_w2c(c2ws)
            v_pos_cam = torch.matmul(v_pos_homo, w2cs_mtx.permute(0, 2, 1))[:, :, :3].contiguous()
            camera_position, _ = dr.interpolate(v_pos_cam, rast_duplicated, t_tex_idx)
            if render_camera_position:
                if enable_antialis:
                    camera_position = dr.antialias(camera_position, rast_duplicated, v_tex_clip_duplicated, t_tex_idx)  #纹理坐标到每个相机下的局部坐标系
                camera_position = torch.lerp(torch.full_like(camera_position, fill_value=0.0), camera_position, alpha)
                out.update({"camera_position": camera_position})
            if render_distance:
                distance = torch.norm(camera_position, p=2, dim=-1, keepdim=True)
                if enable_antialis:
                    distance = dr.antialias(distance, rast_duplicated, v_tex_clip_duplicated, t_tex_idx)  # 每个纹理坐标到每个相机的距离
                distance = torch.lerp(torch.full_like(distance, fill_value=0.0), distance, alpha)
                out.update({"distance": distance})
            if render_z_depth:
                z_depth = camera_position[:, :, :, [-1]]
                if enable_antialis:
                    z_depth = dr.antialias(z_depth, rast_duplicated, v_tex_clip_duplicated, t_tex_idx)
                z_depth = torch.lerp(torch.zeros_like(z_depth), z_depth, alpha)
                out.update({"z_depth": z_depth})
            if render_ray_direction:
                ray_direction = torch.nn.functional.normalize(camera_position, dim=-1)
                if enable_antialis:
                    ray_direction = dr.antialias(ray_direction, rast_duplicated, v_tex_clip_duplicated, t_tex_idx)
                ray_direction = torch.lerp(torch.full_like(ray_direction, fill_value=-1.0), ray_direction, alpha)  # 透明区域默认填充-1
                out.update({"ray_direction": ray_direction})

        if render_cos_ray_normal:
            v_nrm_cam = torch.matmul(mesh.v_nrm, c2ws[:, :3, :3])
            v_nrm_cam = torch.nn.functional.normalize(v_nrm_cam, dim=-1)
            camera_normal, _ = dr.interpolate(v_nrm_cam, rast_duplicated, t_tex_idx)  # 相机坐标系下的法向
            camera_normal = torch.nn.functional.normalize(camera_normal, dim=-1)
            v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
            w2cs_mtx = c2w_to_w2c(c2ws)
            v_pos_cam = torch.matmul(v_pos_homo, w2cs_mtx.permute(0, 2, 1))[:, :, :3].contiguous()
            camera_position, _ = dr.interpolate(v_pos_cam, rast_duplicated, t_tex_idx)
            ray_direction = torch.nn.functional.normalize(camera_position, dim=-1)  # 相机坐标系下的位置代表的方向即射线方向
            cos_ray_normal = torch.sum(camera_normal * ray_direction, dim=-1, keepdim=True)
            if enable_antialis:
                    cos_ray_normal = dr.antialias(cos_ray_normal, rast_duplicated, v_tex_clip_duplicated, t_tex_idx)
            cos_ray_normal = torch.lerp(torch.full_like(cos_ray_normal, fill_value=-1.0), cos_ray_normal, alpha)
            out.update({"cos_ray_normal": cos_ray_normal})

        if render_v_attr:
            gb_v_attr, _ = dr.interpolate(v_attr, rast, t_tex_idx)
            if enable_antialis:
                gb_v_attr = dr.antialias(gb_v_attr, rast, v_tex_clip, t_tex_idx)
            if background is None:
                gb_bg = torch.ones_like(gb_v_attr)
            elif isinstance(background, float) and background >= 0.0 and background <= 1.0:
                gb_bg = torch.full_like(gb_v_attr, fill_value=background)
            elif isinstance(background, torch.Tensor):
                gb_bg = background.to(gb_v_attr).expand_as(gb_v_attr)
            else:
                raise NotImplementedError
            gb_v_attr = torch.lerp(gb_bg, gb_v_attr, alpha)
            out.update({"v_attr": gb_v_attr})
        
        if render_uv:
            v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
            w2cs_mtx = c2w_to_w2c(c2ws)
            # proj_mtx = self.intr_to_proj(intrinsics)
            if proj is not None:
                proj_mtx = proj[0]
                # print("proj:", proj_mtx, intr_to_proj(intrinsics, perspective=persp))
            else:
                proj_mtx = intr_to_proj(intrinsics, perspective=persp)
                # print("intr_to_proj:", proj_mtx, proj)
            mvp_mtx = get_mvp_mtx(proj_mtx, w2cs_mtx)
            v_pos_clip = torch.matmul(v_pos_homo, mvp_mtx.permute(0, 2, 1))
            v_pos_ndc = v_pos_clip[:, :, :2] / v_pos_clip[:, :, [3]]
            t_pos_idx = mesh.t_pos_idx.to(dtype=torch.int32)

            t_mask_visible = self.get_visible_faces(mesh, c2ws, intrinsics, render_size)
            v_tex_clip_visible = v_tex_clip.tile(batch_size, 1, 1)
            rast_visible = rast[:, :, :, [3]].clone().to(dtype=torch.int64).sub(1).tile(batch_size, 1, 1, 1)
            for b in range(batch_size):
                t_idx_visible = torch.where(t_mask_visible[b, :])[0]
                rast_visible[b, :, :, :] = torch.where(torch.isin(rast_visible[b, :, :, :], t_idx_visible), rast_visible[b, :, :, :], -1)
            rast_visible = torch.cat([rast[:, :, :, :3].tile(batch_size, 1, 1, 1), rast_visible.add(1).to(dtype=rast.dtype)], dim=-1)
            mask_visible = rast_visible[..., [3]] > 0
            if enable_antialis:
                alpha_visible = dr.antialias(mask_visible.float(), rast_visible, v_tex_clip_visible, t_tex_idx)
            else:
                alpha_visible = mask_visible.float()
            out.update({"uv_alpha": alpha_visible})  # uv map中哪个部分是相机可见的
            
            gb_uv, _ = dr.interpolate(v_pos_ndc, rast_visible, t_pos_idx)  # 所有可见点的设备坐标系的的光栅化，[-1,1]
            gb_bg = torch.full_like(gb_uv, fill_value=-1.0)
            gb_uv = torch.lerp(gb_bg, gb_uv, alpha_visible) # 不可见的位置填充-1
            if enable_antialis:
                gb_uv_aa = dr.antialias(gb_uv, rast_visible, v_tex_clip_visible, t_tex_idx)
                gb_uv_aa = torch.lerp(gb_bg, gb_uv_aa, alpha_visible)
                out.update({"uv": gb_uv_aa})
            else:
                out.update({"uv": gb_uv}) # uv map 坐标对应的ndc坐标。 前两维就是uv坐标？

            if render_map_attr:
                map_attr = map_attr.expand(batch_size, *map_attr.shape[-3:])  # map_attr x batch_size
                # 根据uv map中对应的设备坐标系，从对应的图像中取值
                gb_map_attr = torch.nn.functional.grid_sample(map_attr.permute(0, 3, 1, 2), gb_uv, mode=uv_interpolate_mode).permute(0, 2, 3, 1) 
                if enable_antialis:
                    gb_map_attr = dr.antialias(gb_map_attr.contiguous(), rast_visible, v_tex_clip_visible, t_tex_idx)
                if background is None:
                    gb_bg = torch.ones_like(gb_map_attr)
                elif isinstance(background, float) and background >= 0.0 and background <= 1.0:
                    gb_bg = torch.full_like(gb_map_attr, fill_value=background)
                elif isinstance(background, torch.Tensor):
                    gb_bg = background.to(gb_map_attr).expand_as(gb_map_attr)
                else:
                    raise NotImplementedError
                gb_map_attr = torch.lerp(gb_bg, gb_map_attr, alpha_visible)  # outi​=starti​+weighti​×(endi​−starti​)
                out.update({"map_attr": gb_map_attr})  #might be some part 
        return out

    def geometry_rendering(
        self, mesh, 
        c2ws:Tensor, intrinsics:Tensor, render_size:int,
        render_all_point_cloud=True,
        render_visible_point_cloud=True,
        render_z_depth=True,
        render_distance=True,
        render_world_normal=True,
        render_camera_normal=True,
        render_world_position=True,
        render_camera_position=True,
        render_ray_direction=True,
        render_cos_ray_normal=True,
        render_uv=True,
        **kwargs,
    ):
        return self.simple_rendering(
            mesh, None, None,
            c2ws, intrinsics, render_size,
            render_all_point_cloud=render_all_point_cloud,
            render_visible_point_cloud=render_visible_point_cloud,
            render_z_depth=render_z_depth,
            render_distance=render_distance,
            render_world_normal=render_world_normal,
            render_camera_normal=render_camera_normal,
            render_world_position=render_world_position,
            render_camera_position=render_camera_position,
            render_ray_direction=render_ray_direction,
            render_cos_ray_normal=render_cos_ray_normal,
            render_v_attr=False,
            render_uv=render_uv,
            render_map_attr=False,
            background=None,
            **kwargs,
        )
    
    def vertex_rendering(
        self, mesh, v_rgb:Tensor, 
        c2ws:Tensor, intrinsics:Tensor, render_size:int, 
        render_rgb=True,
        render_all_point_cloud=False,
        render_visible_point_cloud=False,
        render_z_depth=False,
        render_distance=False,
        render_world_normal=False,
        render_camera_normal=False, 
        render_world_position=False,
        render_camera_position=False,
        render_ray_direction=False,
        render_cos_ray_normal=False,
        background=None,
        **kwargs,
    ):
        render_result = self.simple_rendering(
            mesh, v_rgb, None,
            c2ws, intrinsics, render_size,
            render_all_point_cloud=render_all_point_cloud,
            render_visible_point_cloud=render_visible_point_cloud,
            render_z_depth=render_z_depth,
            render_distance=render_distance,
            render_world_normal=render_world_normal,
            render_camera_normal=render_camera_normal,
            render_world_position=render_world_position,
            render_camera_position=render_camera_position,
            render_ray_direction=render_ray_direction,
            render_cos_ray_normal=render_cos_ray_normal,
            render_v_attr=render_rgb,
            render_uv=False,
            render_map_attr=False,
            background=background,
            **kwargs,
        )
        if render_rgb:
            render_result['rgb'] = render_result['v_attr']
        return render_result
    
    def uv_rendering(
        self, mesh, map_Kd:Tensor, 
        c2ws:Tensor, intrinsics:Tensor, render_size:int, 
        render_rgb=True,
        render_all_point_cloud=False,
        render_visible_point_cloud=False,
        render_z_depth=False,
        render_distance=False,
        render_world_normal=False,
        render_camera_normal=False, 
        render_world_position=False,
        render_camera_position=False,
        render_ray_direction=False,
        render_cos_ray_normal=False,
        background=None,
        **kwargs,
    ):
        render_result = self.simple_rendering(
            mesh, None, map_Kd,
            c2ws, intrinsics, render_size,
            render_all_point_cloud=render_all_point_cloud,
            render_visible_point_cloud=render_visible_point_cloud,
            render_z_depth=render_z_depth,
            render_distance=render_distance,
            render_world_normal=render_world_normal,
            render_camera_normal=render_camera_normal,
            render_world_position=render_world_position,
            render_camera_position=render_camera_position,
            render_ray_direction=render_ray_direction,
            render_cos_ray_normal=render_cos_ray_normal,
            render_v_attr=False,
            render_uv=True,
            render_map_attr=render_rgb,
            background=background,
            **kwargs,
        )
        if render_rgb:
            render_result['rgb'] = render_result['map_attr']
        return render_result
    
    def pbr_rendering(
        self, mesh, map_Kd, map_Ks, pbr_model,
        c2ws, intrinsics, render_size:int,
        render_rgb=True,
        render_all_point_cloud=False,
        render_visible_point_cloud=False,
        render_z_depth=False,
        render_distance=False,
        render_world_normal=False,
        render_camera_normal=False, 
        render_world_position=False,
        render_camera_position=False,
        render_ray_direction=False,
        render_cos_ray_normal=False,
        background=None,
        **kwargs,
    ):
        render_result = self.simple_rendering(
            mesh, None, torch.cat([map_Kd, map_Ks], dim=-1),
            c2ws, intrinsics, render_size,
            render_all_point_cloud=render_all_point_cloud,
            render_visible_point_cloud=render_visible_point_cloud,
            render_z_depth=render_z_depth,
            render_distance=render_distance,
            render_world_normal=render_world_normal or render_rgb,
            render_camera_normal=render_camera_normal,
            render_world_position=render_world_position or render_rgb,
            render_camera_position=render_camera_position,
            render_ray_direction=render_ray_direction,
            render_cos_ray_normal=render_cos_ray_normal,
            render_v_attr=False,
            render_uv=render_rgb,
            render_map_attr=render_rgb,
            background=background,
            **kwargs,
        )
        if render_rgb:
            gb_pos = render_result['world_position']
            gb_normal = render_result['world_normal']
            gb_Kd, gb_Ks = render_result['map_attr'].split([3, 3], dim=-1)
            view_pos = c2ws[:, :3, 3].unsqueeze(1).unsqueeze(1)
            diffuse, specular = pbr_model(view_pos, gb_pos, gb_normal, gb_Kd, gb_Ks)
            if background is None:
                gb_bg = torch.ones_like(diffuse)
            elif isinstance(background, float) and background >= 0.0 and background <= 1.0:
                gb_bg = torch.full_like(diffuse, fill_value=background)
            elif isinstance(background, torch.Tensor):
                gb_bg = background.to(diffuse).expand_as(diffuse)
            else:
                raise NotImplementedError
            render_result['rgb'] = torch.lerp(gb_bg, diffuse + specular, render_result['mask'].float())
        return render_result


