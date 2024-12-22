import json
import math
import os
import random
import cv2
import numpy as np
import torch
from tqdm import tqdm
import rembg
import pyrender
import trimesh
import imageio

import pymeshlab
import pymeshlab as ml
from pymeshlab import PercentageValue

# render with gl camera
def render_glcam(K,
                 Rt,
                 model_in,  # model name or trimesh
                 scale=1.0,
                 std_size=(1000, 1000),
                 flat_shading=False):
    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    # Scene creation
    scene = pyrender.Scene(bg_color=(0.0,0.0,0.0,0.0))

    # Adding objects to the scene
    face_node = scene.add(pr_mesh)

    # Caculate fx fy cx cy from K
    fx, fy = K[0][0] * scale, K[1][1] * scale
    cx, cy = K[0][2] * scale, K[1][2] * scale

    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=0.1, zfar=100000)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = Rt[:3, :3].T
    cam_pose[:3, 3] = -Rt[:3, :3].T.dot(Rt[:, 3])
    scene.add(cam, pose=cam_pose)

    # Set up the light
    # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=15.0)
    light = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=15.0)
    scene.add(light, pose=cam_pose)

    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=std_size[1],
                                   viewport_height=std_size[0],
                                   point_size=1.0)
    color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.RGBA)

    return color

def rendermesh(obj_path, output_path):
    trimesh_model = trimesh.load(obj_path)
    video_name = f'{output_path}/video.gif'
    image_list = []
    intrisic = np.array([[703.3541,   0.    , 256.    ,   0.    ],
    [  0.    , 703.3541, 256.    ,   0.    ],
    [  0.    ,   0.    ,   1.    ,   0.    ],
    [  0.    ,   0.    ,   0.    ,   1.    ]], dtype=np.float32)
    extrinsic = np.array([[ 0.        ,  1.        , -0.        ,  0.        ],
    [-0.08715574,  0.        ,  0.9961947 ,  0.        ],
    [ 0.9961947 , -0.        ,  0.08715574, -2.8000002 ]],
    dtype=np.float32)
    for i in range(0,355,2):
        vertices = trimesh_model.geometry['geometry_0'].vertices
        faces = trimesh_model.geometry['geometry_0'].faces
        vertices = torch.tensor(vertices).cuda().float()
        point_cloud_rot = rotate_point_cloud(vertices, 'z', i)
        rot_mesh  = trimesh.Trimesh(vertices=point_cloud_rot.detach().cpu().numpy(), faces=faces, process=False)
        c_rgb_1 = render_glcam(intrisic, extrinsic, rot_mesh.copy(), std_size=(512, 512))  
        if c_rgb_1.max() > 2.0:
            rgb_np = c_rgb_1[:,:,:3]
        else:
            rgb_np = ((c_rgb_1.detach().cpu().numpy()[0,:,:,:3]) * 255).astype(np.uint8)
        image_list.append(rgb_np)
    imageio.mimsave(video_name, image_list, duration=0.03)
    print('save results to: ',video_name)
    

import pymeshlab
import pymeshlab as ml
from pymeshlab import PercentageValue

def load_mesh_process(mesh_path, shape_init_mesh_up="+y", shape_init_mesh_front="+x", shape_init_params=0.9, device='cuda'):
    import trimesh

    scene = trimesh.load(mesh_path)
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    elif isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate([obj for obj in scene.geometry.values()])
    else:
        raise ValueError(f"Unknown mesh type at {mesh_path}.")
    # mesh.export("debug/images_RGB_debug/debug_01.obj")
    mesh.merge_vertices(merge_tex=False, merge_norm=True)
    
    # move to center
    centroid = mesh.vertices.mean(0)
    mesh.vertices = mesh.vertices - centroid

    if "t-pose" in mesh_path:
        mesh.vertices[:,1] = mesh.vertices[:,1] + 0.4

    # align to up-z and front-x
    dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
    dir2vec = {
        "+x": np.array([1, 0, 0]),
        "+y": np.array([0, 1, 0]),
        "+z": np.array([0, 0, 1]),
        "-x": np.array([-1, 0, 0]),
        "-y": np.array([0, -1, 0]),
        "-z": np.array([0, 0, -1]),
    }
    if (
        shape_init_mesh_up not in dirs
        or shape_init_mesh_front not in dirs
    ):
        raise ValueError(
            f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
        )
    if shape_init_mesh_up[1] == shape_init_mesh_front[1]:
        raise ValueError(
            "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
        )
    z_, x_ = (
        dir2vec[shape_init_mesh_up],
        dir2vec[shape_init_mesh_front],
    )
    y_ = np.cross(z_, x_)
    std2mesh = np.stack([x_, y_, z_], axis=0).T
    mesh2std = np.linalg.inv(std2mesh)

    # scaling
    scale = np.abs(mesh.vertices).max()
    mesh.vertices = mesh.vertices / scale * shape_init_params
    mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T 
    
    print("texture type: ", mesh.visual.kind)
    if  mesh.visual.defined and mesh.visual.kind=="texture" and  mesh.visual.to_color().kind != None:
        v_colors = torch.tensor(np.array(mesh.visual.to_color().vertex_colors)[...,:3] ).to(device)/255
        uv = torch.tensor(mesh.visual.uv, dtype=torch.float32).to(device)
        if isinstance(mesh.visual.material, trimesh.visual.material.PBRMaterial):
            mesh.visual.material = mesh.visual.material.to_simple()
        texture = torch.tensor(np.asarray(mesh.visual.material.image.convert('RGBA'))).to(torch.float32).to(device)/255
        # texture = torch.tensor(np.asarray(mesh.visual.material.image)).to(torch.float32).to(device)/255 #PIL format
    elif (mesh.visual.defined and mesh.visual.kind=="vertex"):
        v_colors = torch.tensor(np.array(mesh.visual.vertex_colors)[...,:3] ).to(device)/255
        # vmapping, indices, uv = xatlas.parametrize(mesh.vertices, mesh.faces)
        # uv = torch.tensor(uv).to(vertices)
        uv = torch.zeros((mesh.vertices.shape[0], 2)).to(device)
        texture = None
    else:
        v_colors = torch.ones((mesh.vertices.shape[0], 3)).to(device)
        # vmapping, indices, uv = xatlas.parametrize(mesh.vertices, mesh.faces)
        # uv = torch.tensor(uv).to(vertices)
        uv = torch.zeros((mesh.vertices.shape[0], 2)).to(device)
        texture = None
        
    uv_f = torch.LongTensor(mesh.faces).to(device)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    
    return torch.FloatTensor(mesh.vertices).to(device), torch.LongTensor(mesh.faces).to(device), v_colors.to(torch.float32), uv, uv_f, texture


def c2w_to_w2c(c2w):
    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    
    return w2c


def convert_proj(K, H, W, near, far):
    return [
        [2 * K[0, 0] / W, -2 * K[0, 1] / W, (W - 2 * K[0, 2]) / W, 0],
        [0, -2 * K[1, 1] / H, (H - 2 * K[1, 2]) / H, 0],
        [0, 0, (-far - near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ]


def load_camera_info(json_path, render_res=512, device='cuda'):
    camera_dict = json.load( open(json_path, "r"))
    assert camera_dict["camera_model"] == "OPENCV"

    frames = camera_dict["frames"]
    frames_proj = []
    frames_c2w = []
    intrinsics = []

    frame_w = render_res
    frame_h = render_res
    scale = frames[0]["w"]/render_res

    c2w_list = []
    for frame in tqdm(frames):
        extrinsic: Float[Tensor, "4 4"] = torch.as_tensor(
            frame["transform_matrix"], dtype=torch.float32
        )
        c2w = extrinsic
        c2w_list.append(c2w)
    c2w_list = torch.stack(c2w_list, dim=0)

    for idx, frame in tqdm(enumerate(frames)):
        intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
        intrinsic[0, 0] = frame["fl_x"] / scale
        intrinsic[1, 1] = frame["fl_y"] / scale
        intrinsic[0, 2] = frame["cx"] / scale
        intrinsic[1, 2] = frame["cy"] / scale

        c2w = c2w_list[idx]

        near = 0.1
        far = 1000.0
        proj = convert_proj(intrinsic, frame_h, frame_w, near, far)
        intrinsic[0, 0] = frame["fl_x"]
        intrinsic[1, 1] = frame["fl_y"]
        intrinsic[0, 2] = frame["cx"]
        intrinsic[1, 2] = frame["cy"]
        
        proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
        intrinsics.append(intrinsic)
        frames_proj.append(proj)
        frames_c2w.append(c2w)

    proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0).to(device)
    c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0).to(device)
    intrinsics: Float[Tensor, "B 4 4"]= torch.stack(intrinsics, dim=0).to(device)
    mv = c2w_to_w2c(c2w)
    
    return mv, proj, intrinsics, c2w


def c_n2w_n(imgs, mv, black_bg=True):
    # imgs: (B,H,W,C)
    # mv: (B,4,4), world to camera
    B,H,W,C = imgs.shape
    img_masks = imgs[..., 3:]
    normals = imgs[...,:3].reshape(B,-1,3)*2-1
    c2w = mv.inverse()
    imgs_world_coord = (normals @ c2w[:,:3,:3].transpose(2,1))[...,:3]
    imgs_world_coord /= torch.norm(imgs_world_coord, dim=-1, keepdim=True)
    imgs_world_coord = imgs_world_coord.reshape(B,H,W,3) * 0.5 + 0.5

    img_masks_nag = (imgs[...,0] < 0.3) & (imgs[...,1] < 0.3) & (imgs[...,2] < 0.3)
    img_masks = (~img_masks_nag[...,None]).to(imgs)
    #imgs = torch.concat([imgs*img_masks, img_masks], dim=-1)

    imgs_world_coord = torch.concat([imgs_world_coord*img_masks, img_masks], dim=-1)
    return imgs_world_coord

def rotate_point_cloud(point_cloud, axis, angle_degrees):
    """
    Rotate a point cloud around a specified axis by a given angle in degrees.

    Parameters:
    point_cloud (torch.Tensor): Nx3 tensor of 3D points.
    axis (str): Axis to rotate around ('x', 'y', or 'z').
    angle_degrees (float): Angle to rotate in degrees.

    Returns:
    torch.Tensor: Rotated point cloud.
    """
    # Convert angle from degrees to radians
    angle = torch.tensor(np.radians(angle_degrees))
    
    if axis == 'x':
        rotation_matrix = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angle), -torch.sin(angle)],
            [0, torch.sin(angle), torch.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = torch.tensor([
            [torch.cos(angle), 0, torch.sin(angle)],
            [0, 1, 0],
            [torch.sin(angle), 0, torch.cos(angle)]
        ])
    elif axis == 'z':
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")
    rotation_matrix = rotation_matrix.cuda().float()
    # Apply rotation
    rotated_point_cloud = torch.matmul(point_cloud, rotation_matrix.T)
    return rotated_point_cloud


def w_n2c_n(imgs, mv, black_bg = True):
    # imgs: (B,H,W,C)
    # mv: (B,4,4), world to camera

    B,H,W,C = imgs.shape
    img_masks = imgs[..., 3:]
    normals = imgs[...,:3].reshape(B, -1, 3) * 2 - 1
    imgs_cam_coord = (normals @ mv[:,:3,:3].transpose(2, 1))[...,:3]
    imgs_cam_coord /= torch.norm(imgs_cam_coord, dim=-1, keepdim=True)
    imgs_cam_coord = imgs_cam_coord.reshape(B,H,W,3) * 0.5 + 0.5
    if img_masks.shape[0] == 0:
        img_masks_nag = (imgs[...,0] < 0.3) & (imgs[...,1] < 0.3) & (imgs[...,2] < 0.3)
        img_masks = (~img_masks_nag[...,None]).to(imgs)

    imgs_cam_coord = torch.concat([imgs_cam_coord*img_masks, img_masks], dim=-1) 
    return imgs_cam_coord


def remove_bg(images):
    # images: [N, H, W, C]
    
    rembg_session = rembg.new_session()
    images_np = images.clamp(0, 1).mul(255.0).detach().cpu().numpy().astype(np.uint8)
    _images = torch.empty((*images.shape[:-1], 4), dtype=images.dtype)
    for i, im in enumerate(images_np):
        im = rembg.remove(im, alpha_matting=True, session=rembg_session)
        # _im = cv2.inpaint(im[:, :, :3], ~im[:, :, [3]], 3.0, cv2.INPAINT_TELEA)
        # im = np.concatenate([_im, im[:, :, [3]]], axis=-1)
        _im = im.copy()
        _im[...,:3] *= _im[:, :, [3]]
        im = _im
        _images[i] = torch.as_tensor(im, dtype=images.dtype).div(255.0)
    images = _images.to(device=images.device)
    
    return images


def erode_mask(mask:torch.Tensor, kernel_size = 5):
    mask_np = (mask.cpu().numpy()*255).astype(np.uint8)
    mask_erode_list = []
    for mask_i in mask_np:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size, kernel_size))
        mask_np_erode = cv2.erode(mask_i, kernel=kernel)
        mask_erode = torch.Tensor(mask_np_erode).to(mask)/255
        mask_erode_list.append(mask_erode[None,..., None])
    mask_new = torch.concat(mask_erode_list, dim=0)
    
    return mask_new
    
    
def rotate_point_cloud(point_cloud, axis, angle_degrees):
    """
    Rotate a point cloud around a specified axis by a given angle in degrees.

    Parameters:
    point_cloud (torch.Tensor): Nx3 tensor of 3D points.
    axis (str): Axis to rotate around ('x', 'y', or 'z').
    angle_degrees (float): Angle to rotate in degrees.

    Returns:
    torch.Tensor: Rotated point cloud.
    """
    # Convert angle from degrees to radians
    angle = torch.tensor(np.radians(angle_degrees))
    
    if axis == 'x':
        rotation_matrix = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angle), -torch.sin(angle)],
            [0, torch.sin(angle), torch.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = torch.tensor([
            [torch.cos(angle), 0, torch.sin(angle)],
            [0, 1, 0],
            [torch.sin(angle), 0, torch.cos(angle)]
        ])
    elif axis == 'z':
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")
    rotation_matrix = rotation_matrix.cuda().float()
    # Apply rotation
    rotated_point_cloud = torch.matmul(point_cloud, rotation_matrix.T)
    return rotated_point_cloud



def mesh_smooth(vertices, faces, stepsmoothnum=1):
    mesh = pymeshlab.Mesh(
        vertex_matrix=vertices.detach().cpu().float().numpy().astype(np.float64),
        face_matrix=faces.detach().cpu().long().numpy().astype(np.int32) )
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh, "cube_vcolor_mesh")

    ms.apply_filter("apply_coord_laplacian_smoothing", stepsmoothnum=stepsmoothnum, cotangentweight=False)
    
    new_mesh = ms.current_mesh()
    smooth_vertices = torch.from_numpy(new_mesh.vertex_matrix()).to(vertices).contiguous()
    smooth_faces = torch.from_numpy(new_mesh.face_matrix()).to(faces).contiguous()
    
    return smooth_vertices, smooth_faces