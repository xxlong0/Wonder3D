from matplotlib import image
import nvdiffrast.torch as dr
import torch
import os
from torchvision.utils import save_image
from typing import Tuple

def _warmup(glctx):
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device='cuda', **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])


class Renderer:
    
    # _glctx:dr.RasterizeGLContext = None
    _glctx:dr.RasterizeCudaContext = None
    
    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: tuple[int,int],
            ):
        self._mvp = proj @ mv
        self._mv = mv
        self._proj = proj
        self._image_size = image_size
        self._glctx = dr.RasterizeCudaContext(device='cuda')
        _warmup(self._glctx)

    def render_normal(self,
            vertices: torch.Tensor, #V,3 float
            normals: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        vert_col = (normals+1)/2 #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4

    def render_RGB_vclolor(self,
            vertices: torch.Tensor, #V,3 float
            vert_col: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V ,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        print(vert_hom.shape[0])
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4

    def render_depth(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vert_cam = torch.matmul(vert_hom, self._mv.transpose(-2,-1)).float()
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        # disp = 1 / (vert_cam[..., [2]] + 1e-20)  # for controlnet
        # disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-20) # pre-normalize
        # depth, _ = dr.interpolate(disp, rast_out, faces) # [1, H, W, 1]
        v_depth = vert_cam[..., [2]]
        depth, _ = dr.interpolate(v_depth, rast_out, faces) # [1, H, W, 1]
        #depth = depth.squeeze(0) # [H, W, 1]
        return depth #C,H,W,4
    #
    def render_RGB_texture(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            vt, # [V,2], range(0, 1)
            ft, # [F,3]
            texture, # [B,H,W,3] or [H,W,3]
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        C = self._mvp.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        if len(texture.shape) == 3:
            texture = texture.unsqueeze(0)
        # texture = texture.flip(dims=[-3])
        texture = texture.tile(C,1,1,1)

        rast_out, rast_db = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size) #C,H,W,4
        mask = (rast_out[..., [-1]] > 0)
        alpha = mask.float()
        alpha = torch.clamp(rast_out[..., -1:], max=1)

        # rgb texture
        v_tex_ndc = vt * 2.0 - 1.0
        t_tex_idx = ft.to(torch.int32)
        gb_uv, _ = dr.interpolate(v_tex_ndc, rast_out, t_tex_idx)
        gb_uv = torch.lerp(torch.full_like(gb_uv, fill_value=-1.0), gb_uv, alpha)
        gb_texture = torch.nn.functional.grid_sample(texture.permute(0, 3, 1, 2), gb_uv, mode='nearest').permute(0, 2, 3, 1)
        gb_texture = dr.antialias(gb_texture.contiguous(), rast_out, vertices_clip, faces)
        gb_texture = torch.lerp(torch.zeros_like(gb_texture), gb_texture, alpha)
        gb_texture = gb_texture[...,: 3]
        gb_texture = torch.concat((gb_texture, alpha), dim=-1) #C,H,W,4

        return gb_texture #C,H,W,4_

    def render_RGBD_texture(self,
                            vertices: torch.Tensor,  # V,3 float
                            faces: torch.Tensor,  # F,3 long
                            vt,  # [V,2], range(0, 1)
                            ft,  # [F,3]
                            texture,  # [B,H,W,4] or [H,W,4], RGBA
                            ) -> torch.Tensor:  # C,H,W,4 (RGBD)

        V = vertices.shape[0]
        C = self._mvp.shape[0]
        faces = faces.type(torch.int32)

        # Homogeneous coordinates
        vert_hom = torch.cat((vertices, torch.ones(V, 1, device=vertices.device)), axis=-1)  # V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2, -1)  # C,V,4

        if len(texture.shape) == 3:
            texture = texture.unsqueeze(0)  # Ensure batch dimension if necessary

        # Ensure texture matches batch size
        texture = texture.tile(C, 1, 1, 1)

        # Rasterization
        rast_out, rast_db = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size)  # C,H,W,4
        mask = (rast_out[..., [-1]] > 0)
        alpha = torch.clamp(rast_out[..., -1:], max=1)  # Mask and alpha

        # RGB texture sampling
        v_tex_ndc = vt * 2.0 - 1.0
        t_tex_idx = ft.to(torch.int32)
        gb_uv, _ = dr.interpolate(v_tex_ndc, rast_out, t_tex_idx)
        gb_uv = torch.lerp(torch.full_like(gb_uv, fill_value=-1.0), gb_uv, alpha)
        gb_texture = torch.nn.functional.grid_sample(texture.permute(0, 3, 1, 2), gb_uv, mode='nearest').permute(0, 2,
                                                                                                                 3, 1)
        gb_texture = dr.antialias(gb_texture.contiguous(), rast_out, vertices_clip, faces)

        # Split RGB and Alpha (Depth)
        rgb_texture = gb_texture[..., :3]  # C,H,W,3
        depth_texture = gb_texture[..., 3:]  # C,H,W,1 (Alpha channel used as depth)

        # Combine RGB with Depth (RGBA -> RGBD)
        rgbd_texture = torch.cat((rgb_texture, depth_texture), dim=-1)  # C,H,W,4 (RGB + Depth)

        # Apply alpha mask (visibility)
        rgbd_texture = torch.lerp(torch.zeros_like(rgbd_texture), rgbd_texture, alpha)

        return rgbd_texture  # C,H,W,4 (RGBD)

    # def render_RGB_texture(self,
    #                        vertices: torch.Tensor,  # V,3 float
    #                        faces: torch.Tensor,  # F,3 long
    #                        vt,  # [V,2], range(0, 1]
    #                        ft,  # [F,3]
    #                        texture,  # [B,H,W,3] or [H,W,3]
    #                        ) -> torch.Tensor:  # C,H,W,4
    #
    #     V = vertices.shape[0]
    #     C = self._mvp.shape[0]
    #     faces = faces.type(torch.int32)
    #     vert_hom = torch.cat((vertices, torch.ones(V, 1, device=vertices.device)), axis=-1)  # V,3 -> V,4
    #     vertices_clip = vert_hom @ self._mvp.transpose(-2, -1)  # C,V,4
    #
    #     if len(texture.shape) == 3:
    #         texture = texture.unsqueeze(0)
    #
    #     # 将纹理从 RGB 转换为 RGBA，alpha 值默认为 1.0
    #     if texture.shape[-1] == 3:  # 检查是否为 RGB
    #         alpha_channel = torch.ones((texture.shape[0], texture.shape[1], texture.shape[2], 1),
    #                                    device=texture.device)  # 创建 alpha 通道
    #         rgba_texture = torch.cat((texture, alpha_channel), dim=-1)  # 将 RGB 和 alpha 合并
    #     else:
    #         rgba_texture = texture  # 如果已经是 RGBA，则保持不变
    #
    #     rgba_texture = rgba_texture.tile(C, 1, 1, 1)  # 扩展纹理以匹配视图数
    #
    #     rast_out, rast_db = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size)  # C,H,W,4
    #
    #     # 使用深度信息生成 alpha 通道
    #     mask = (rast_out[..., [-1]] > 0)  # 检查哪些像素是可见的
    #     alpha = mask.float()  # 将可见像素转换为 1.0，其他为 0.0
    #
    #     # rgb texture
    #     v_tex_ndc = vt * 2.0 - 1.0
    #     t_tex_idx = ft.to(torch.int32)
    #     gb_uv, _ = dr.interpolate(v_tex_ndc, rast_out, t_tex_idx)
    #     gb_uv = torch.lerp(torch.full_like(gb_uv, fill_value=-1.0), gb_uv, alpha)
    #
    #     # 使用 RGBA 纹理进行采样
    #     gb_texture = torch.nn.functional.grid_sample(rgba_texture.permute(0, 3, 1, 2), gb_uv, mode='nearest').permute(0, 2, 3, 1)
    #     gb_texture = dr.antialias(gb_texture.contiguous(), rast_out, vertices_clip, faces)
    #     gb_texture = torch.lerp(torch.zeros_like(gb_texture), gb_texture, alpha)
    #
    #     # 使用 alpha 通道信息
    #     # alpha = alpha.unsqueeze(-1).unsqueeze(-1)  # 确保 alpha 为 [C,H,W,1]
    #     # print(gb_texture.shape, alpha.shape)
    #     # gb_texture = torch.cat((gb_texture, alpha), dim=-1)  # C,H,W,4
    #
    #     return gb_texture  # C,H,W,4

class NormalsRenderer:

    _glctx: dr.RasterizeCudaContext = None

    def __init__(
            self,
            mv: torch.Tensor,  # C,4,4
            proj: torch.Tensor,  # C,4,4
            image_size: Tuple[int, int],
            mvp=None,
            device=None,
    ):
        _glctx: dr.RasterizeCudaContext = None

    def __init__(
            self,
            mv: torch.Tensor,  # C,4,4
            proj: torch.Tensor,  # C,4,4
            image_size: tuple[int, int],
    ):
        self._mvp = proj @ mv
        self._mv = mv
        self._proj = proj
        self._image_size = image_size
        self._glctx = dr.RasterizeCudaContext(device='cuda')
        _warmup(self._glctx)

    def render(self,
               vertices: torch.Tensor,  # V,3 float
               normals: torch.Tensor,  # V,3 float
               faces: torch.Tensor,  # F,3 long
               ) -> torch.Tensor:  # C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V, 1, device=vertices.device)), axis=-1)  # V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2, -1)  # C,V,4
        rast_out, _ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False)  # C,H,W,4
        vert_col = (normals + 1) / 2  # V,3
        col, _ = dr.interpolate(vert_col, rast_out, faces)  # C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1)  # C,H,W,1
        col = torch.concat((col, alpha), dim=-1)  # C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces)  # C,H,W,4
        return col  # C,H,W,4

