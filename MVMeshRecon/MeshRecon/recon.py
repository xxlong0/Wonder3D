from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from typing import List
from MeshRecon.remesh import calc_vertex_normals
from MeshRecon.opt import MeshOptimizer
from MeshRecon.func import make_star_cameras_orthographic, make_star_cameras_orthographic_py3d
from MeshRecon.render import NormalsRenderer, Pytorch3DNormalsRenderer
from scripts.utils import to_py3d_mesh, init_target

def reconstruct_stage1(pils: List[Image.Image], steps=100, vertices=None, faces=None, start_edge_len=0.15, end_edge_len=0.005, decay=0.995, return_mesh=True, loss_expansion_weight=0.1, gain=0.1):
    vertices, faces = vertices.to("cuda"), faces.to("cuda")
    assert len(pils) == 4
    mv,proj = make_star_cameras_orthographic(4, 1)
    renderer = NormalsRenderer(mv,proj,list(pils[0].size))
    # cameras = make_star_cameras_orthographic_py3d([0, 270, 180, 90], device="cuda", focal=1., dist=4.0)
    # renderer = Pytorch3DNormalsRenderer(cameras, list(pils[0].size), device="cuda")
    
    target_images = init_target(pils, new_bkgd=(0., 0., 0.)) # 4s
    # 1. no rotate
    target_images = target_images[[0, 3, 2, 1]]

    # 2. init from coarse mesh
    opt = MeshOptimizer(vertices,faces, local_edgelen=False, gain=gain, edge_len_lims=(end_edge_len, start_edge_len))

    vertices = opt.vertices

    mask = target_images[..., -1] < 0.5

    for i in tqdm(range(steps)):
        opt.zero_grad()
        opt._lr *= decay
        normals = calc_vertex_normals(vertices,faces)
        images = renderer.render(vertices,normals,faces)
        
        loss_expand = 0.5 * ((vertices+normals).detach() - vertices).pow(2).mean()
        
        t_mask = images[..., -1] > 0.5
        loss_target_l2 = (images[t_mask] - target_images[t_mask]).abs().pow(2).mean()
        loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()
        
        loss = loss_target_l2 + loss_alpha_target_mask_l2 + loss_expand * loss_expansion_weight
        
        # out of box
        loss_oob = (vertices.abs() > 0.99).float().mean() * 10
        loss = loss + loss_oob
        
        loss.backward()
        opt.step()

        vertices,faces = opt.remesh(poisson=False)

    vertices, faces = vertices.detach(), faces.detach()
    
    if return_mesh:
        return to_py3d_mesh(vertices, faces)
    else:
        return vertices, faces
