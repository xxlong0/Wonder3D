

from functools import partial
import os
import numpy as np
import cv2
import imageio
import torch
import trimesh
import torchvision.transforms as T

from .common.mesh.structure import Texture
from .common.renderers.nvdiffrast.renderer import NVDiffRendererBase
from .common.cameras.generate_cameras import generate_intrinsics, generate_orbit_views_c2ws

class VideoExporter:
    def __init__(self) -> None:
        self.mesh_renderer = NVDiffRendererBase()
    
    def export_arbit_video(self, texture: Texture, video_path):
        texture = texture.to(self.mesh_renderer.device)
        radius = 3 * texture.mesh.radius().item()
        
        c2ws = generate_orbit_views_c2ws(120, radius=radius, height=0.0, theta_0=90.0, degree=True)
        # intrinsics = generate_intrinsics(60, 60, fov=True, degree=True)
        intrinsics = generate_intrinsics(1, 1, fov=False, degree=True )
        video, cover = self.export_video(texture, None, c2ws, intrinsics, 1024)
        
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        cover_path = os.path.splitext(video_path)[0] + '.png'

        # NOTE: check data range and channel order during exporting
        imageio.mimsave(video_path, video.clamp(0.0, 1.0).mul(255).numpy().astype(np.uint8), fps=15)
        cv2.imwrite(cover_path, cover.clamp(0.0, 1.0).mul(255)[..., [2,1,0,3]].numpy().astype(np.uint8))

    def export_video(
        self, texture:Texture, pbr_model, 
        c2ws, intrinsics, render_size: int, chunk_size=2,
        with_alpha=True, normalize=False, key='rgb',
    ):
        '''
        intrinsics: [1, 3, 3] or [3, 3]
        c2ws: [num_views, 4, 4]
        video: [N, H, W, 3]
        cover: [H, W, 4]
        '''
        mesh = texture.mesh
        v_rgb = texture.v_rgb
        map_Kd = texture.map_Kd
        map_Ks = texture.map_Ks
        assert v_rgb is not None or map_Kd is not None
        device = v_rgb.device if v_rgb is not None else map_Kd.device

        if map_Kd is not None:
            if map_Ks is not None and pbr_model is not None:
                renderer = partial(
                    self.mesh_renderer.pbr_rendering,
                    mesh=mesh,
                    map_Kd=map_Kd.contiguous(), 
                    map_Ks=map_Ks.contiguous(), 
                    pbr_model=pbr_model,
                )
            else:
                renderer = partial(
                    self.mesh_renderer.uv_rendering,
                    mesh=mesh,
                    map_Kd=map_Kd.contiguous(), 
                )
        else:
            renderer = partial(
                self.mesh_renderer.vertex_rendering,
                mesh=mesh,
                v_rgb=v_rgb.contiguous(), 
            )
        
        num_views, _, _ = c2ws.shape
        c2ws = c2ws.to(dtype=torch.float32, device=device)
        intrinsics = intrinsics.expand(num_views, 3, 3).to(dtype=torch.float32, device=device)
        
        video = torch.empty((num_views, render_size, render_size, 3), dtype=torch.float32)
        with torch.no_grad():
            for idx in range(0, c2ws.shape[0], chunk_size):
                render_result = renderer(
                    c2ws=c2ws[idx:idx+chunk_size], 
                    intrinsics=intrinsics[idx:idx+chunk_size], 
                    render_size=render_size,
                    background=0.0,
                    **{f'render_{key}': True} if key != 'uv' else {},
                )
                rgb = render_result[key]
                if rgb.shape[-1] == 1:
                    rgb = torch.repeat_interleave(rgb, 3, dim=-1)
                elif rgb.shape[-1] == 2:
                    rgb = torch.cat([
                        rgb[..., [0]], 
                        torch.full_like(rgb[..., [0]], -1), 
                        rgb[..., [1]]
                    ], dim=-1)
                elif rgb.shape[-1] == 3:
                    rgb = rgb
                elif rgb.shape[-1] > 3:
                    rgb = rgb[..., :3]
                if normalize:
                    mask = torch.repeat_interleave(render_result['mask'], 3, dim=-1)
                    rgb_sel = rgb[mask]
                    rgb[mask] = (rgb_sel - rgb_sel.min()) / (rgb_sel.max() - rgb_sel.min())
                video[idx: idx+chunk_size] = rgb.cpu()
                if idx == 0:
                    if with_alpha:
                        alpha = render_result['alpha']
                        cover = torch.cat([rgb[0], alpha[0]], dim=-1).cpu()
                    else:
                        cover = rgb[0].cpu()
        return video, cover

    def export_mvs(
            self, texture: Texture, pbr_model,
            render_size=256, chunk_size=2,
            with_alpha=True, normalize=False, key='rgb', num_views=4, save_path='', cam_type='ortho',
            mvps=None,
    ):
        '''
        intrinsics: [1, 3, 3] or [3, 3]
        video: [N, H, W, 3]
        cover: [H, W, 4]
        '''
        mesh = texture.mesh
        v_rgb = texture.v_rgb
        map_Kd = texture.map_Kd
        map_Ks = texture.map_Ks
        assert v_rgb is not None or map_Kd is not None
        device = v_rgb.device if v_rgb is not None else map_Kd.device

        c2ws = generate_orbit_views_c2ws(num_views, radius=3 * texture.mesh.radius().item(), height=0.0, theta_0=90.0,
                                         degree=True)

        if map_Kd is not None:
            if map_Ks is not None and pbr_model is not None:
                renderer = partial(
                    self.mesh_renderer.pbr_rendering,
                    mesh=mesh,
                    map_Kd=map_Kd.contiguous(),
                    map_Ks=map_Ks.contiguous(),
                    pbr_model=pbr_model,
                )
            else:
                renderer = partial(
                    self.mesh_renderer.uv_rendering,
                    mesh=mesh,
                    map_Kd=map_Kd.contiguous(),
                )
        else:
            renderer = partial(
                self.mesh_renderer.vertex_rendering,
                mesh=mesh,
                v_rgb=v_rgb.contiguous(),
            )

        if cam_type == 'ortho':
            self.mesh_renderer.enable_orthogonal()
        else:
            self.mesh_renderer.enable_perspective()
        print(c2ws.shape)
        num_views, _, _ = c2ws.shape
        c2ws = c2ws.to(dtype=torch.float32, device=device)
        # intrinsics = torch.as_tensor([[1,   0.    , 0.    ],
        #                              [  0.    , 1,     0.  ],
        #                              [  0.    ,   0.    ,   1.    ],], dtype=torch.float32)
        intrinsics = generate_intrinsics(1, 1, fov=False, degree=True)
        intrinsics = intrinsics.expand(num_views, 3, 3).to(dtype=torch.float32, device=device)

        video = torch.empty((num_views, render_size, render_size, 3), dtype=torch.float32)
        with torch.no_grad():
            for idx in range(0, c2ws.shape[0], chunk_size):
                render_result = renderer(
                    c2ws=c2ws[idx:idx + chunk_size],
                    intrinsics=intrinsics[idx:idx + chunk_size],
                    render_size=render_size,
                    background=0.0,
                    **{f'render_{key}': True, f'render_world_normal': True, f'mvps': mvps[idx:idx + chunk_size]} if key != 'uv' else {},
                )
                rgb = render_result[key]
                normal = render_result['world_normal']

                # 处理和保存每个视角的 RGB 图像
                for view in range(chunk_size):
                    view_idx = idx + view
                    if view_idx < num_views:
                        # rgb_image = rgb[view].cpu().permute(2, 0 ,1)
                        # normal_image = normal[view].cpu().permute(2, 0 ,1)
                        #
                        # # 将 RGB 图像和 normal 图像转换为 PIL 格式
                        # rgb_pil = T.ToPILImage()(rgb_image)
                        # mask = (normal_image == 0).all(dim=0)
                        # depth_channel = torch.ones_like(mask)
                        # depth_channel[mask] = 0
                        #
                        # normal_rgb = (normal_image + 1) / 2  # 将 normal 映射到 [0, 1]
                        # normal_rgb = torch.cat([normal_rgb, depth_channel.unsqueeze(0)] , dim=0)
                        # normal_pil = T.ToPILImage()(normal_rgb)

                        # 视角名称（可以根据实际需求调整顺序）
                        view_names = ['front', 'right', 'back', 'left']
                        view_name = view_names[view_idx]

                        # 保存 RGB 和 normal 图像
                        # rgb_pil.save(os.path.join(save_path,f'rgb_render_{view_name}.png'))
                        # normal_pil.save(os.path.join(save_path,f'normal_render_{view_name}.png'))

                        cv2.imwrite(os.path.join(save_path,f'rgb_render_{view_name}.png'),
                                    torch.cat([rgb[view][..., :3], render_result['alpha'][view]], dim=-1).cpu().clamp(0.0, 1.0).mul(255)[..., [2, 1, 0, 3]].numpy().astype(np.uint8))
                        cv2.imwrite(os.path.join(save_path,f'normal_render_{view_name}.png'),
                                    torch.cat([(normal[view][..., :3]+1)/2, render_result['alpha'][view]], dim=-1).cpu().clamp(0.0, 1.0).mul(255)[..., [2, 1, 0, 3]].numpy().astype(np.uint8))

                if rgb.shape[-1] == 1:
                    rgb = torch.repeat_interleave(rgb, 3, dim=-1)
                elif rgb.shape[-1] == 2:
                    rgb = torch.cat([
                        rgb[..., [0]],
                        torch.full_like(rgb[..., [0]], -1),
                        rgb[..., [1]]
                    ], dim=-1)
                elif rgb.shape[-1] == 3:
                    rgb = rgb
                elif rgb.shape[-1] > 3:
                    rgb = rgb[..., :3]

                if normalize:
                    mask = torch.repeat_interleave(render_result['mask'], 3, dim=-1)
                    rgb_sel = rgb[mask]
                    rgb[mask] = (rgb_sel - rgb_sel.min()) / (rgb_sel.max() - rgb_sel.min())

                video[idx: idx + chunk_size] = rgb.cpu()

                if idx == 0:
                    if with_alpha:
                        alpha = render_result['alpha']
                        cover = torch.cat([rgb[0], alpha[0]], dim=-1).cpu()
                    else:
                        cover = rgb[0].cpu()

                cover_path = os.path.splitext(save_path)[0] + '.png'

        return video, cover

