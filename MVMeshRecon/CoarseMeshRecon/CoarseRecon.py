import os
from PIL import Image
from CoarseMeshRecon.mesh_init import build_mesh, calc_w_over_h, fix_border_with_pymeshlab_fast
from pytorch3d.structures import Meshes, join_meshes_as_scene
import numpy as np
import torch
from pytorch3d.io import load_obj, save_obj

def remove_color(arr):
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    # calc diffs
    base = arr[0, 0]
    diffs = np.abs(arr.astype(np.int32) - base.astype(np.int32)).sum(axis=-1)
    alpha = (diffs <= 80)
    
    arr[alpha] = 255
    alpha = ~alpha
    arr = np.concatenate([arr, alpha[..., None].astype(np.int32) * 255], axis=-1)
    return arr

def simple_remove(imgs):
    """Only works for normal"""
    if not isinstance(imgs, list):
        imgs = [imgs]
        single_input = True
    else:
        single_input = False
    rets = []
    for img in imgs:
        arr = np.array(img)
        arr = remove_color(arr)
        rets.append(Image.fromarray(arr.astype(np.uint8)))
    if single_input:
        return rets[0]
    return rets

def fast_geo(front_normal: Image.Image, back_normal: Image.Image, side_normal: Image.Image, clamp=0., init_type="std", output_path='debug', is_persp=False):
    # if front_normal.mode == "RGB":
    #     front_normal = simple_remove(front_normal, run_sr=False)
    front_normal = front_normal.resize((192, 192))
    # if back_normal.mode == "RGB":
    #     back_normal = simple_remove(back_normal, run_sr=False)
    back_normal = back_normal.resize((192, 192))
    # if side_normal.mode == "RGB":
    #     side_normal = simple_remove(side_normal, run_sr=False)
    side_normal = side_normal.resize((192, 192))
    
    # build mesh with front back projection # ~3s
    side_w_over_h = calc_w_over_h(side_normal)   #  物体宽度和高度的比值
    mesh_front = build_mesh(front_normal, front_normal, clamp_min=clamp, scale=side_w_over_h, init_type=init_type, is_persp=is_persp)
    # build mesh return None for sunken case
    if mesh_front is None:
        print('front mesh is sunken')
        return None

    mesh_back = build_mesh(back_normal, back_normal, is_back=True, clamp_min=-1, scale=side_w_over_h, init_type=init_type, is_persp=is_persp)
    # build mesh return None for sunken case
    if mesh_back is None:
        print('back mesh is sunken')
        return None

    os.makedirs(os.path.join(output_path, 'debug','3d_model'),exist_ok=True)
    save_obj(os.path.join(output_path, 'debug','3d_model','mesh_front.obj'), mesh_front.verts_list()[0], mesh_front.faces_list()[0])
    save_obj(os.path.join(output_path, 'debug','3d_model','mesh_back.obj'), mesh_back.verts_list()[0], mesh_back.faces_list()[0])
    meshes = join_meshes_as_scene([mesh_front, mesh_back])
    meshes = fix_border_with_pymeshlab_fast(meshes, poissson_depth=6, simplification=2000, apply_smooth=True)
    save_obj(os.path.join(output_path, 'debug','3d_model','mesh_init.obj'), meshes.verts_list()[0], meshes.faces_list()[0])
    
    return meshes

def CoarseRecon(front_normal:torch.Tensor, back_normal:torch.Tensor, side_normal:torch.Tensor, output_path, is_persp=False):
    front_normal = Image.fromarray((front_normal.cpu().numpy()*255).astype(np.uint8))
    back_normal = Image.fromarray((back_normal.cpu().numpy()*255).astype(np.uint8))
    side_normal = Image.fromarray((side_normal.cpu().numpy()*255).astype(np.uint8))
    
    meshes = fast_geo(front_normal, back_normal, side_normal, clamp=0., init_type="std", output_path=output_path, is_persp=is_persp)
    if meshes is None:
        return None, None
    verts_list = meshes.verts_list()
    vertices = torch.vstack(verts_list).to('cuda')
    faces = torch.vstack(meshes.faces_list()).to('cuda')
    
    return vertices, faces