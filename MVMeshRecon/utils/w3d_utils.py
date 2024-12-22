import os
import numpy as np
import PIL.Image
import cv2
from glob import glob
import torch
import math
from rembg import remove
from .refine_lr_to_sr import sr_front_img

def RT_opengl2opencv(RT):
    # Build the coordinate transform matrix from world to computer vision camera

    R = RT[:3, :3]
    t = RT[:3, 3]

    R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

    R_world2cv = R_bcam2cv @ R
    t_world2cv = R_bcam2cv @ t

    RT = np.concatenate([R_world2cv, t_world2cv[:, None]], 1)

    return RT

def img2normal(img):
    return (img/255.)*2-1

def inv_RT(RT):
    RT_h = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    RT_inv = np.linalg.inv(RT_h)

    return RT_inv[:3, :]

def normal_opengl2opencv(normal):
    H,W,C = np.shape(normal)
    # normal_img = np.reshape(normal, (H*W,C))
    R_bcam2cv = np.array([1, -1, -1], np.float32)
    normal_cv = normal * R_bcam2cv[None, None, :]

    print(np.shape(normal_cv))

    return normal_cv

def camNormal2worldNormal(rot_c2w, camNormal):
    H,W,_ = camNormal.shape
    normal_img = np.matmul(rot_c2w[None, :, :], camNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    return normal_img

def normal2img(normal):
    return np.uint8((normal*0.5+0.5)*255)

def _translation(tx, ty, tz, device):
    translation_matrix = torch.eye(4, device=device)
    translation_matrix[0, 3] = tx
    translation_matrix[1, 3] = ty
    translation_matrix[2, 3] = tz
    return translation_matrix

def _projection(r, device):
    projection_matrix = torch.eye(4, device=device)
    projection_matrix[3, 2] = -r
    return projection_matrix



def _orthographic(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    o = torch.zeros([4,4],device=device)
    o[0,0] = 2/(r-l)
    o[0,3] = -(r+l)/(r-l) # 0
    o[1,1] = 2/(t-b) * (-1 if flip_y else 1)
    o[1,3] = -(t+b)/(t-b) # 0
    o[2,2] = -2/(f-n)
    o[2,3] = -(f+n)/(f-n)
    o[3,3] = 1
    return o # 4,4


def _perspective(fov_y=2*np.arctan(32./70), aspect=1, n=1., f=50., device='cuda', flip_y=True):

    t = n * np.tan((fov_y) / 2)  # 近平面的上边界
    r = t * aspect  # 近平面的右边界
    l, b = -r, -t  # 对称近平面边界

    # 初始化透视投影矩阵
    p = torch.zeros((4, 4), device=device)
    p[0, 0] = 2 * n / (r - l)
    p[1, 1] = 2 * n / (t - b)
    p[0, 2] = -(r + l) / (r - l)  # 水平偏移
    p[1, 2] = (t + b) / (t - b)
    p[2, 2] = -(f + n) / (f - n)  # Z 缩放
    p[2, 3] = -(2 * f * n) / (f - n)  # Z 偏移
    p[3, 2] = -1.0  # W 分量
    p[3, 3] = 0.0  # 保持齐次坐标

    if flip_y:
        p[1, :] *= -1
    return p

def persp_proj(device, fov_x=49.13, ar=1, near=1, far=50, flip_y=True):
    """
    Build a perspective projection matrix.

    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fov_x)
    proj_mat = np.array([[1.0 / np.tan(fov_rad / 2.0), 0, 0, 0],
                      [0, 1.0 * np.float32(ar) / np.tan(fov_rad / 2.0), 0, 0],
                      [0, 0, (near + far) / (near-far), 2 * far * near / (near-far)],
                      [0, 0, -1, 0]])
    proj = torch.tensor(proj_mat, device=device, dtype=torch.float32)

    return proj


def intr_to_proj(intr_mtx:torch.Tensor, near=0.1, far=50, perspective=True):
    proj_mtx = torch.zeros((*intr_mtx.shape[:-2], 4, 4), dtype=intr_mtx.dtype, device=intr_mtx.device)
    if perspective:
        proj_mtx[..., 0, 0] = 2 * intr_mtx[..., 0, 0]
        proj_mtx[..., 1, 1] = - 2 * intr_mtx[..., 1, 1]  # for nvdiffrast
        proj_mtx[..., 2, 2] = -(far + near) / (far - near)
        proj_mtx[..., 2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[..., 3, 2] = -1.0
    else:
        proj_mtx[..., 0, 0] = intr_mtx[..., 0, 0]
        proj_mtx[..., 1, 1] = - intr_mtx[..., 1, 1]  # for nvdiffrast
        proj_mtx[..., 2, 2] = -2.0 / (far - near)
        proj_mtx[..., 3, 3] = 1.0
        proj_mtx[..., 2, 3] = - (far + near) / (far - near)
    return proj_mtx

def fov_to_focal(fov, size):
    # convert fov angle in degree to focal
    return size / np.tan(fov * np.pi / 180.0 / 2.0) / 2.0


def camera_intrinsic_to_opengl_projection(device, intrinsic=None, w=512, h=512, n=0, f=5, flip_y=True):

    if intrinsic==None:
        intrinsic = torch.from_numpy(np.array([
            [5.600000000000000000e+02, 0, 2.560000000000000000e+02, 0],
            [0, 5.600000000000000000e+02, 2.560000000000000000e+02, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).astype(np.float32))

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    proj = torch.from_numpy(np.array([
        [2.*fx/w,   0,    1-2*cx/w,           0],
        [0,    2*fy/h,   -1+2*cy/h,           0],
        [0,         0,(-f-n)/(f-n),-f*n/(f-n)],
        [0,         0,          -1,           0]
    ]).astype(np.float32)).to(device)

    if flip_y:
        proj[1, :] *= -1

    return proj

def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = PIL.Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def load_a_from_pils(front_image=None,rgbs=None,rm_normal=None,imSize=[256,256],view_types=['front', 'front_right', 'right', 'back', 'left', 'front_left'],load_color=True,
                     cam_pose_dir='mv_diffusion_30/data/fixed_poses/nine_views',normal_system='front',crop_size=192):
    all_images = []
    all_normals = []
    all_normals_world = []
    all_masks = []
    all_color_masks = []
    all_poses = []
    all_w2cs = []
    print(os.path.join(cam_pose_dir, '*_%s_RT.txt' % ('front')))
    RT_front = np.loadtxt(glob(os.path.join(cam_pose_dir, '*_%s_RT.txt' % ('front')))[0])  # world2cam matrix
    RT_front_cv = RT_opengl2opencv(RT_front)  # convert normal from opengl to opencv

    for idx, view in enumerate(view_types):
        image = np.array(rgbs[idx].resize(imSize)).copy()[:, :, :3]
        normal = np.array(rm_normal[idx].resize(imSize)).copy()
        normal_mask = normal[:, :, 3]
        normal = normal[:, :, :3]

        gtmask = ~((image[..., 0] / 255 > 0.8) & (image[..., 1] / 255 > 0.8) & (image[..., 2] / 255 > 0.8) & (normal[..., 0] / 255 > 0.8) & (normal[..., 1] / 255 > 0.8) & (normal[..., 2] / 255 > 0.8))
        gtmask = ~((image[..., 0] / 255 > 0.8) & (image[..., 1] / 255 > 0.8) & (image[..., 2] / 255 > 0.8) & (normal[..., 0] / 255 > 0.8) & (normal[..., 1] / 255 > 0.8) & (normal[..., 2] / 255 > 0.8))

        gtmask = np.array(gtmask).astype(np.uint8)

        gtmask = np.array(gtmask * normal_mask).astype(np.uint8)

        RT = np.loadtxt(os.path.join(cam_pose_dir, '000_%s_RT.txt' % (view)))  # world2cam matrix

        normal[gtmask == 0] = [0, 0, 0]

        if load_color:
            all_images.append(image)

        all_masks.append(gtmask)
        all_color_masks.append(gtmask)

        RT_cv = RT_opengl2opencv(RT)  # convert normal from opengl to opencv
        all_poses.append(inv_RT(RT))  # cam2world
        all_w2cs.append(np.vstack((RT, [0, 0, 0, 1])))

        # whether to
        normal_cam_cv = normal_opengl2opencv(normal)

        all_normals.append(normal)
        all_normals_world.append(normal)

        if not load_color:
            all_images = [normal2img(x) for x in all_normals_world]

    image_input = front_image
    image_input = remove(image_input)

    np_front_img = np.asarray(image_input)
    front_res = np_front_img.shape[0]

    if crop_size != -1:
        alpha_np = np.asarray(image_input)[:, :, 3]
        coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
        min_x, min_y = np.min(coords, 0)
        max_x, max_y = np.max(coords, 0)
        ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
        h, w = ref_img_.height, ref_img_.width
        scale = crop_size / max(h, w)
        h_, w_ = int(scale * h), int(scale * w)
        ref_img_ = ref_img_.resize((w_, h_))
        image_input = add_margin(ref_img_, size=imSize[0])
    else:
        image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
        image_input = image_input.resize((imSize))

    if front_res < imSize[0]:
        # do sr
        image_input = sr_front_img(np.asarray(image_input), out_size=imSize[0])
    image_input = np.array(image_input, copy=True)
    alpha_imput = image_input[:, :, 3]

    image_input[:, :, 3] = alpha_imput

    return np.stack(all_images), np.stack(all_masks), np.stack(all_normals), \
        np.stack(all_normals_world), np.stack(all_poses), np.stack(all_w2cs), np.stack(all_color_masks), image_input

def load_a_prediction(root_dir, test_object, imSize, view_types=['front', 'front_right', 'right', 'back', 'left', 'front_left'],
                      load_color=True, cam_pose_dir='../../mvdiffusion/data/fixed_poses/nine_views',
                      normal_system='front', obj_name='2', crop_size=192):
    all_images = []
    all_normals = []
    all_normals_world = []
    all_masks = []
    all_color_masks = []
    all_poses = []
    all_w2cs = []
    print(os.path.join(cam_pose_dir, '*_%s_RT.txt' % ('front')))
    RT_front = np.loadtxt(glob(os.path.join(cam_pose_dir, '*_%s_RT.txt' % ('front')))[0])  # world2cam matrix
    RT_front_cv = RT_opengl2opencv(RT_front)  # convert normal from opengl to opencv

    for idx, view in enumerate(view_types):
        print(os.path.join(root_dir, test_object))
        normal_filepath = os.path.join(root_dir, test_object,  'normals_000_%s.png' % (view))

        # Load key frame
        if load_color:  # use bgr
            rgb_filepath = os.path.join(root_dir, test_object, 'colors', 'rgb_000_%s.png' % (view))
            image = np.array(PIL.Image.open(rgb_filepath).resize(imSize))[:, :, :3]

        normal = np.array(PIL.Image.open(normal_filepath).resize(imSize))
        normal_mask = normal[:, :, 3]

        normal = normal[:, :, :3]
        mask = (normal_mask < 0.5)
        normal_mask[mask] = 0.
        normal_mask[~mask] = 1.0

        gtmask = ~((image[..., 0] / 255 > 0.9) & (image[..., 1] / 255 > 0.9) & (image[..., 2] / 255 > 0.9) & (normal[..., 0] / 255 > 0.9) & (normal[..., 1] / 255 > 0.9) & (normal[..., 2] / 255 > 0.9))
        gtmask = np.array(gtmask).astype(np.uint8)

        gtmask = np.array(gtmask * normal_mask).astype(np.uint8)

        RT = np.loadtxt(os.path.join(cam_pose_dir, '000_%s_RT.txt' % (view)))  # world2cam matrix

        normal[gtmask == 0] = [0, 0, 0]

        if load_color:
            all_images.append(image)

        all_masks.append(gtmask)
        all_color_masks.append(gtmask)
        RT_cv = RT_opengl2opencv(RT)  # convert normal from opengl to opencv
        all_poses.append(inv_RT(RT))  # cam2world
        all_w2cs.append(np.vstack((RT, [0, 0, 0, 1])))

        # whether to
        normal_cam_cv = normal_opengl2opencv(normal)

        if normal_system == 'front':
            print("the loaded normals are defined in the system of front view")
            normal_world = camNormal2worldNormal(inv_RT(RT_front_cv)[:3, :3], (normal_cam_cv))
        elif normal_system == 'self':
            print("the loaded normals are in their independent camera systems")
            normal_world = camNormal2worldNormal(inv_RT(RT_cv)[:3, :3], normal_cam_cv)

        all_normals.append(normal)
        all_normals_world.append(normal)

        if not load_color:
            all_images = [normal2img(x) for x in all_normals_world]

    if obj_name.split('.')[-1] not in ['png', 'jpg']:
        obj_name += '.png'
    image_input = (PIL.Image.open(os.path.join('../example_images', f'{obj_name}')))

    image_input = remove(image_input)

    np_front_img = np.asarray(image_input)
    front_res = np_front_img.shape[0]

    if crop_size != -1:
        alpha_np = np.asarray(image_input)[:, :, 3]
        coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
        min_x, min_y = np.min(coords, 0)
        max_x, max_y = np.max(coords, 0)
        ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
        h, w = ref_img_.height, ref_img_.width
        scale = crop_size / max(h, w)
        h_, w_ = int(scale * h), int(scale * w)
        ref_img_ = ref_img_.resize((w_, h_))
        image_input = add_margin(ref_img_, size=imSize[0])
    else:
        image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
        image_input = image_input.resize((imSize))

    if front_res < imSize[0]:
        # do sr
        image_input = sr_front_img(np.asarray(image_input), out_size=imSize[0])
    image_input = np.array(image_input, copy=True)
    alpha_imput = image_input[:, :, 3]
    # alpha_imput = cv2.erode(alpha_imput, kernel, iterations=1)
    image_input[:, :, 3] = alpha_imput

    return np.stack(all_images), np.stack(all_masks), np.stack(all_normals), \
        np.stack(all_normals_world), np.stack(all_poses), np.stack(all_w2cs), np.stack(all_color_masks), image_input

def make_star_cameras_fixed_angles(distance: float = 10., r: float = None, image_size=[512, 512], device='cuda', angles=None):
    if r is None:
        r = 1 / distance

    # Define the fixed angles in degrees
    if not angles:
        angles = [0, -45, -90, 180, 90, 45]
    angles_rad = [angle * torch.pi / 180.0 for angle in angles]
    C = len(angles)

    mv = torch.eye(4, device=device).repeat(C, 1, 1)

    for i, angle in enumerate(angles_rad):
        rot = torch.eye(3, device=device)
        angle = torch.tensor(angle)
        rot[2, 2] = torch.cos(angle)
        rot[2, 0] = -torch.sin(angle)
        rot[0, 2] = torch.sin(angle)
        rot[0, 0] = torch.cos(angle)

        mv[i, :3, :3] = rot

    # Apply translation
    mv = _translation(0, 0, -distance, device) @ mv

    return mv, _projection(r, device)

def make_wonder3D_cameras(distance: float=10. , r: float=1., image_size = [512, 512], device = 'cuda', cam_type='ortho', angles=None):
    mv, _ = make_star_cameras_fixed_angles(angles=angles)
    if not r:
        r = 1
    if cam_type=='ortho':
        print('making orthogonal cameras')
        return mv, _orthographic(r, device)
    else:
        mv, _ = make_star_cameras_fixed_angles(distance=(1 / np.tan(np.deg2rad(49.13) / 2.0)), angles=angles)
        return mv, _perspective(device=device)