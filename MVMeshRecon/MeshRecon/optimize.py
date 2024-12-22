import torch
import time
import os
from tqdm import tqdm
import numpy as np
import sys

from PIL import Image
from remeshing.util.typing import * 
import cv2
import numpy as np

from MeshRecon.opt import MeshOptimizer
from MeshRecon.remesh import calc_vertex_normals
from MeshRecon.project_mesh import multiview_color_projection, get_cameras_list, get_cameras_list_py3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex
import pymeshlab
import pymeshlab as ml
from pymeshlab import PercentageValue
from torchvision.utils import save_image


def to_pyml_mesh(vertices,faces):
    m1 = pymeshlab.Mesh(
        vertex_matrix=vertices.cpu().float().numpy().astype(np.float64),
        face_matrix=faces.cpu().long().numpy().astype(np.int32),
    )
    return m1

def meshlab_mesh_to_py3dmesh(mesh: pymeshlab.Mesh) -> Meshes:
    verts = torch.from_numpy(mesh.vertex_matrix()).float()
    faces = torch.from_numpy(mesh.face_matrix()).long()
    colors = torch.from_numpy(mesh.vertex_color_matrix()[..., :3]).float()
    textures = TexturesVertex(verts_features=[colors])
    return Meshes(verts=[verts], faces=[faces], textures=textures)

def simple_clean_mesh(pyml_mesh: ml.Mesh, apply_smooth=True, stepsmoothnum=1, apply_sub_divide=False, sub_divide_threshold=0.25):
    ms = ml.MeshSet()
    ms.add_mesh(pyml_mesh, "cube_mesh")
    
    if apply_smooth:
        # ms.apply_filter("apply_coord_laplacian_smoothing", stepsmoothnum=stepsmoothnum, cotangentweight=False)
        ms.apply_filter("apply_coord_hc_laplacian_smoothing")
    if apply_sub_divide:    # 5s, slow
        ms.apply_filter("meshing_repair_non_manifold_vertices")
        ms.apply_filter("meshing_repair_non_manifold_edges", method='Remove Faces')
        ms.apply_filter("meshing_surface_subdivision_loop", iterations=2, threshold=PercentageValue(sub_divide_threshold))
    return meshlab_mesh_to_py3dmesh(ms.current_mesh())

    
def c2w_to_w2c(c2w):
    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    
    return w2c

def to_py3d_mesh(vertices, faces, normals=None):
    mesh = Meshes(verts=[vertices], faces=[faces], textures=None)
    if normals is None:
        normals = mesh.verts_normals_packed()
    # set normals as vertext colors
    mesh.textures = TexturesVertex(verts_features=[normals / 2 + 0.5])
    return mesh

def from_py3d_mesh(mesh):
    return mesh.verts_list()[0], mesh.faces_list()[0], mesh.textures.verts_features_packed()

def init_target(img_pils, new_bkgd=(0., 0., 0.), device="cuda"):
    # Convert the background color to a PyTorch tensor
    new_bkgd = torch.tensor(new_bkgd, dtype=torch.float32).view(1, 1, 3).to(device)

    # Convert all images to PyTorch tensors and process them
    # imgs = torch.stack([torch.from_numpy(np.array(img, dtype=np.float32)) for img in img_pils]).to(device) / 255
    imgs = img_pils
    img_nps = imgs[..., :3]
    alpha_nps = imgs[..., 3]
    ori_bkgds = img_nps[:, :1, :1]

    # Avoid divide by zero and calculate the original image
    alpha_nps_clamp = torch.clamp(alpha_nps, 1e-6, 1)
    ori_img_nps = (img_nps - ori_bkgds * (1 - alpha_nps.unsqueeze(-1))) / alpha_nps_clamp.unsqueeze(-1)
    ori_img_nps = torch.clamp(ori_img_nps, 0, 1)
    img_nps = torch.where(alpha_nps.unsqueeze(-1) > 0.05,
                          ori_img_nps * alpha_nps.unsqueeze(-1) + new_bkgd * (1 - alpha_nps.unsqueeze(-1)), new_bkgd)

    rgba_img_np = torch.cat([img_nps, alpha_nps.unsqueeze(-1)], dim=-1)
    return rgba_img_np

def geo_aware_mesh_refine(vertices, faces, pils: List[Image.Image], renderer, extrinsics, intrinsics, steps=100, start_edge_len=0.02, end_edge_len=0.005,
                    decay=0.99, update_normal_interval=20, update_warmup=5, return_mesh=False, process_inputs=False,
                    process_outputs=False, b_persp=False):
    if process_inputs:
        vertices = vertices * 2 / 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]

    poission_steps = []

    renderer = renderer
    target_images = init_target(pils, new_bkgd=(0., 0., 0.))  # 4s


    # 2. init from coarse mesh
    opt = MeshOptimizer(vertices, faces, ramp=5,
                        edge_len_lims=(end_edge_len, start_edge_len),
                        local_edgelen=False,
                        laplacian_weight=0.01)

    vertices = opt.vertices
    alpha_init = None

    mask = target_images[..., -1] < 0.5

    for i in tqdm(range(steps)):
        opt.zero_grad()
        opt._lr *= decay
        normals = calc_vertex_normals(vertices, faces)
        images = renderer.render_normal(vertices, normals, faces)
        if alpha_init is None:
            alpha_init = images.detach()

        if i < update_warmup or i % update_normal_interval == 0:
            with torch.no_grad():
                py3d_mesh = to_py3d_mesh(vertices, faces, normals)
                cameras = get_cameras_list_py3d(extrinsics, intrinsics, target_images.shape[-2] , device=vertices.device, b_perspective=b_persp)  #正交相机
                _, _, target_normal = from_py3d_mesh(multiview_color_projection(py3d_mesh, target_images, cameras_list=cameras, weights=None, confidence_threshold=0.2, complete_unseen=False, below_confidence_strategy='original', reweight_with_cosangle='linear', opt_render = renderer))
                target_normal = target_normal * 2 - 1
                target_normal = torch.nn.functional.normalize(target_normal, dim=-1)
                debug_images = renderer.render_normal(vertices, target_normal, faces)

        d_mask = (images[..., -1] > 0.5)
        loss_geo_aware = (images[..., :3][d_mask] - debug_images[..., :3][d_mask]).pow(2).mean()

        loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()

        loss = loss_geo_aware + loss_alpha_target_mask_l2 * 10.

        loss = loss

        loss.backward()
        opt.step()
        vertices, faces = opt.remesh(poisson=(i in poission_steps))

    vertices, faces = vertices.detach(), faces.detach()

    if process_outputs:
        vertices = vertices / 2 * 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]

    if return_mesh:
        return to_py3d_mesh(vertices, faces)
    else:
        return vertices, faces