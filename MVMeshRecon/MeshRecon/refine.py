from tqdm import tqdm
from PIL import Image
import torch
from typing import List
from MeshRecon.remesh import calc_vertex_normals
from MeshRecon.opt import MeshOptimizer
from MeshRecon.func import make_star_cameras_orthographic, make_star_cameras_orthographic_py3d
from MeshRecon.render import NormalsRenderer, Pytorch3DNormalsRenderer
from scripts.project_mesh import multiview_color_projection, get_cameras_list
from scripts.utils import to_py3d_mesh, from_py3d_mesh, init_target

def run_mesh_refine(vertices, faces, pils: List[Image.Image], steps=100, start_edge_len=0.02, end_edge_len=0.005, decay=0.99, update_normal_interval=10, update_warmup=10, return_mesh=True, process_inputs=True, process_outputs=True):
    if process_inputs:
        vertices = vertices * 2 / 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]
    
    poission_steps = []

    assert len(pils) == 4
    mv,proj = make_star_cameras_orthographic(4, 1)          
    renderer = NormalsRenderer(mv,proj,list(pils[0].size))
    # cameras = make_star_cameras_orthographic_py3d([0, 270, 180, 90], device="cuda", focal=1., dist=4.0)
    # renderer = Pytorch3DNormalsRenderer(cameras, list(pils[0].size), device="cuda")

    target_images = init_target(pils, new_bkgd=(0., 0., 0.)) # 4s
    # 1. no rotate
    target_images = target_images[[0, 3, 2, 1]]

    # 2. init from coarse mesh
    opt = MeshOptimizer(vertices,faces, ramp=5, edge_len_lims=(end_edge_len, start_edge_len), local_edgelen=False, laplacian_weight=0.02)

    vertices = opt.vertices
    alpha_init = None

    mask = target_images[..., -1] < 0.5

    for i in tqdm(range(steps)):
        opt.zero_grad()
        opt._lr *= decay
        normals = calc_vertex_normals(vertices,faces)
        images = renderer.render(vertices,normals,faces)
        if alpha_init is None:
            alpha_init = images.detach()
        
        if i < update_warmup or i % update_normal_interval == 0:
            with torch.no_grad():
                py3d_mesh = to_py3d_mesh(vertices, faces, normals)
                cameras = get_cameras_list(azim_list = [0, 90, 180, 270], device=vertices.device, focal=1.)
                _, _, target_normal = from_py3d_mesh(multiview_color_projection(py3d_mesh, pils, cameras_list=cameras, weights=[2.0, 0.8, 1.0, 0.8], confidence_threshold=0.1, complete_unseen=False, below_confidence_strategy='original', reweight_with_cosangle='linear'))
                target_normal = target_normal * 2 - 1
                target_normal = torch.nn.functional.normalize(target_normal, dim=-1)
                debug_images = renderer.render(vertices,target_normal,faces)
        
        d_mask = images[..., -1] > 0.5
        loss_debug_l2 = (images[..., :3][d_mask] - debug_images[..., :3][d_mask]).pow(2).mean()
        
        loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()
        
        loss = loss_debug_l2 + loss_alpha_target_mask_l2
        
        # out of box
        loss_oob = (vertices.abs() > 0.99).float().mean() * 10
        loss = loss + loss_oob
        
        loss.backward()
        opt.step()
        
        vertices,faces = opt.remesh(poisson=(i in poission_steps))
    
    vertices, faces = vertices.detach(), faces.detach()
    
    if process_outputs:
        vertices = vertices / 2 * 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]

    if return_mesh:
        return to_py3d_mesh(vertices, faces)
    else:
        return vertices, faces
