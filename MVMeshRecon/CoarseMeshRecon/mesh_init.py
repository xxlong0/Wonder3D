from PIL import Image
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from .utils import meshlab_mesh_to_py3dmesh, py3dmesh_to_meshlab_mesh
import pymeshlab

import cv2

_MAX_THREAD = 8

# rgb and depth to mesh
def get_ortho_ray_directions_origins(W, H, use_pixel_centers=True, device="cuda"):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i).to(device), torch.from_numpy(j).to(device)

    origins = torch.stack([(i/W-0.5)*2, (j/H-0.5)*2 * H / W, torch.zeros_like(i)], dim=-1) # W, H, 3
    directions = torch.stack([torch.zeros_like(i), torch.zeros_like(j), torch.ones_like(i)], dim=-1) # W, H, 3

    return origins, directions


def get_perspective_ray_directions_origins(W, H, fov=49.13, focal_length_mm=35, use_pixel_centers=True, device="cuda"):
    fov = fov * np.pi / 180.0

    sensor_width_mm = 36.0
    focal_length = (W / sensor_width_mm) * focal_length_mm

    # 定义像素中心
    pixel_center = 0.5 if use_pixel_centers else 0

    # 创建网格，表示图像平面上的像素坐标
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i).to(device), torch.from_numpy(j).to(device)

    # 计算射线原点，这里假设为图像平面中心的像素坐标
    origins = torch.stack([(i / W - 0.5) * 2, (j / H - 0.5) * 2 * (H / W), torch.zeros_like(i)], dim=-1)

    # 计算透视投影下的射线方向
    directions = torch.stack([
        (i - W * 0.5) / focal_length,         # x 方向
        -(j - H * 0.5) / focal_length,         # y 方向（负号确保向上为正）
        torch.ones_like(i)                     # z 方向
    ], dim=-1)

    # 归一化方向向量
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    return origins, directions
def erode_depth(mask: np.ndarray, kernel_size=10):
    mask_np = (mask * 255).astype(np.uint8)
    mask_erode_list = []

    for mask_i in mask_np:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask_np_erode = cv2.erode(mask_i, kernel=kernel)
        mask_erode = mask_np_erode / 255.0
        mask_erode_list.append(mask_erode[..., None])  # Add extra dimension to match the shape

    mask_new = np.concatenate(mask_erode_list, axis=0)

    return mask_new

def depth_and_color_to_mesh(rgb_BCHW, pred_HWC, valid_HWC=None, is_back=False, is_persp=True):  # depth-顶点-三角化
    if valid_HWC is None:
        valid_HWC = torch.ones_like(pred_HWC).bool()
    H, W = rgb_BCHW.shape[-2:]
    rgb_BCHW = rgb_BCHW.flip(-2)
    pred_HWC = pred_HWC.flip(0)
    valid_HWC = valid_HWC.flip(0)

    if not is_persp:
        rays_o, rays_d = get_ortho_ray_directions_origins(W, H, device=rgb_BCHW.device)
    else:
        rays_o, rays_d = get_perspective_ray_directions_origins(W, H, device=rgb_BCHW.device)
    verts = rays_o + rays_d * pred_HWC  # [H, W, 3]
    verts = verts.reshape(-1, 3)    # [V, 3]
    indexes = torch.arange(H * W).reshape(H, W).to(rgb_BCHW.device)
    faces1 = torch.stack([indexes[:-1, :-1], indexes[:-1, 1:], indexes[1:, :-1]], dim=-1)
    # faces1_valid = valid_HWC[:-1, :-1] | valid_HWC[:-1, 1:] | valid_HWC[1:, :-1]
    faces1_valid = valid_HWC[:-1, :-1] & valid_HWC[:-1, 1:] & valid_HWC[1:, :-1]
    faces2 = torch.stack([indexes[1:, 1:], indexes[1:, :-1], indexes[:-1, 1:]], dim=-1)
    # faces2_valid = valid_HWC[1:, 1:] | valid_HWC[1:, :-1] | valid_HWC[:-1, 1:]
    faces2_valid = valid_HWC[1:, 1:] & valid_HWC[1:, :-1] & valid_HWC[:-1, 1:]
    faces = torch.cat([faces1[faces1_valid.expand_as(faces1)].reshape(-1, 3), faces2[faces2_valid.expand_as(faces2)].reshape(-1, 3)], dim=0)  # (F, 3)
    colors = (rgb_BCHW[0].permute((1,2,0)) / 2 + 0.5).reshape(-1, 3)  # (V, 3)
    if is_back:
        verts = verts * torch.tensor([-1, 1, -1], dtype=verts.dtype, device=verts.device) 
    
    used_verts = faces.unique()
    old_to_new_mapping = torch.zeros_like(verts[..., 0]).long()
    old_to_new_mapping[used_verts] = torch.arange(used_verts.shape[0], device=verts.device)
    new_faces = old_to_new_mapping[faces]
    mesh = Meshes(verts=[verts[used_verts]], faces=[new_faces], textures=TexturesVertex(verts_features=[colors[used_verts]]))
    return mesh

def normalmap_to_depthmap(normal_np):
    from CoarseMeshRecon.normal_to_height_map import estimate_height_map
    height = estimate_height_map(normal_np, raw_values=True, thread_count=_MAX_THREAD, target_iteration_count=96)
    return height

def transform_back_normal_to_front(normal_pil):
    arr = np.array(normal_pil)  # in [0, 255]
    arr[..., 0] = 255-arr[..., 0]
    arr[..., 2] = 255-arr[..., 2]
    return Image.fromarray(arr.astype(np.uint8))

def calc_w_over_h(normal_pil):
    if isinstance(normal_pil, Image.Image):
        arr = np.array(normal_pil)
    else:
        assert isinstance(normal_pil, np.ndarray)
        arr = normal_pil
    if arr.shape[-1] == 4:
        alpha = arr[..., -1] / 255.
        alpha[alpha >= 0.5] = 1
        alpha[alpha < 0.5] = 0
    else:
        alpha = ~(arr.min(axis=-1) >= 250)
    h_min, w_min = np.min(np.where(alpha), axis=1)
    h_max, w_max = np.max(np.where(alpha), axis=1)
    return (w_max - w_min) / (h_max - h_min) # 物体宽度和高度的比值

def check_sunken_from_depth(depth):
    # cal mean depth for whole depth map
    depth_mean_whole = np.mean(depth)

    # cal mean depth for eroded depth map
    deoth_eroded = erode_depth(depth)
    depth_mean_eroded = np.mean(deoth_eroded)

    # check is sunkun by compare mean depth map
    is_sunken = depth_mean_whole < depth_mean_eroded

    return is_sunken

def build_mesh(normal_pil, rgb_pil, is_back=False, clamp_min=-1, scale=0.3, init_type="std", offset=0, is_persp=False):

    normal_img = np.array(normal_pil)
    rgb_img = np.array(rgb_pil)
    if normal_img.shape[-1] == 4:
        valid_HWC = normal_img[..., [3]] / 255
    elif rgb_img.shape[-1] == 4:
        valid_HWC = rgb_img[..., [3]] / 255
    else:
        raise ValueError("invalid input, either normal or rgb should have alpha channel")
    
    real_height_pix = np.max(np.where(valid_HWC>0.5)[0]) - np.min(np.where(valid_HWC>0.5)[0])  # 物体在图像中height最大值和最小值的差

    heights = normalmap_to_depthmap(normal_img)  # 积分
    is_sunken = check_sunken_from_depth(heights) # test if sunken
    if is_sunken:
        return None

    rgb_BCHW = torch.from_numpy(rgb_img[..., :3] / 255.).permute((2,0,1))[None]
    valid_HWC[valid_HWC < 0.5] = 0
    valid_HWC[valid_HWC >= 0.5] = 1
    valid_HWC = torch.from_numpy(valid_HWC).bool()  # alpha， bool
    if init_type == "std":
        # accurate but not stable
        pred_HWC = torch.from_numpy(heights / heights.max() * (real_height_pix / heights.shape[0]) * scale).float()[..., None]
    elif init_type == "thin":
        heights = heights - heights.min()
        heights = (heights / heights.max() * 0.2)
        pred_HWC = torch.from_numpy(heights * scale).float()[..., None]
    else:
        # stable but not accurate
        heights = heights - heights.min()
        heights = (heights / heights.max() * (1-offset)) + offset # to [0.2, 1]
        pred_HWC = torch.from_numpy(heights * scale).float()[..., None]

    import cv2
    # edge filter
    edge = cv2.Canny((valid_HWC[..., 0] * 255).numpy().astype(np.uint8), 0, 255)
    edge = torch.from_numpy(edge).bool()[..., None]
    pred_HWC[edge] = 0
    
    valid_HWC[pred_HWC < clamp_min] = False
    return depth_and_color_to_mesh(rgb_BCHW.cuda(), pred_HWC.cuda(), valid_HWC.cuda(), is_back, is_persp)

def fix_border_with_pymeshlab_fast(meshes: Meshes, poissson_depth=6, simplification=0, apply_smooth=False, stepsmoothnum=1):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(py3dmesh_to_meshlab_mesh(meshes), "cube_vcolor_mesh")
    if simplification > 0:
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=simplification, preservetopology=True)
    ms.apply_filter('generate_surface_reconstruction_screened_poisson', threads = 6, depth = poissson_depth, preclean = True)
    if simplification > 0:
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=simplification, preservetopology=True)
    if apply_smooth:
        ms.apply_filter("apply_coord_laplacian_smoothing", stepsmoothnum=stepsmoothnum, cotangentweight=False)
    return meshlab_mesh_to_py3dmesh(ms.current_mesh())
