import math
import torch

from .rotation_conversions import lookat_to_matrix


def generate_intrinsics(f_x: float, f_y: float, fov=True, degree=False):
    '''
    f_x, f_y: 
        * focal length divide width/height for perspective camera
        * fov degree or radians for perspective camera
        * scale for orthogonal camera
    intrinsics: [3, 3], normalized
    '''
    if fov:
        if degree:
            f_x = math.radians(f_x)
            f_y = math.radians(f_y)
        f_x_div_W = 2 * math.tan(f_x / 2)
        f_y_div_H = 2 * math.tan(f_y / 2)
    else:
        f_x_div_W = f_x
        f_y_div_H = f_y
    return torch.as_tensor([
        [f_x_div_W, 0.0, 0.5],
        [0.0, f_y_div_H, 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)


def generate_orbit_views_c2ws(num_views: int, radius: float = 1.0, height: float = 0.0, theta_0: float = 0.0, degree=False):
    '''
    c2w_mat: [num_views, 4, 4]
    '''
    if degree:
        theta_0 = math.radians(theta_0)
    projected_radius = math.sqrt(radius ** 2 - height ** 2)
    theta = torch.linspace(theta_0, 2.0 * math.pi, num_views, dtype=torch.float32)
    
    x = projected_radius * torch.cos(theta)
    y = projected_radius * torch.sin(theta)
    z = torch.full((num_views,), fill_value=height, dtype=torch.float32)
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    
    c2w_trans = torch.stack([x, y, z], dim=-1)
    c2w_rot_mat = lookat_to_matrix(torch.nn.functional.normalize(c2w_trans, dim=1))
    zero_zero_zero_one = torch.stack([zero, zero, zero, one], dim=-1)
    c2w_mat = torch.cat([
        torch.cat([c2w_rot_mat, c2w_trans.unsqueeze(-1)], dim=2),
        zero_zero_zero_one.unsqueeze(-2),
    ], dim=1)
    return c2w_mat


def generate_four_views_c2ws(distance: float = 1.0):
    '''
    c2w_mat: [num_views, 4, 4], num_views = 4
        front to back
        right to left
        back to front
        left to right
    '''
    c2w_mat = torch.as_tensor([
        [[1, 0, 0, 0],
        [0, 0, -1, -distance],
        [0, 1, 0, 0],
        [0, 0, 0, 1]],

        [[0, 0, 1, distance],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]],

        [[-1, 0, 0, 0],
        [0, 0, 1, distance],
        [0, 1, 0, 0],
        [0, 0, 0, 1]],

        [[0, 0, -1, -distance],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]],
    ], dtype=torch.float32)
    return c2w_mat


def generate_six_views_c2ws(distance: float = 1.0):
    '''
    c2w_mat: [num_views, 4, 4], num_views = 6
        front to back
        right to left
        back to front
        left to right
        top to down
        down to top
    '''
    c2w_mat = torch.as_tensor([
        [[1, 0, 0, 0],
        [0, 0, -1, -distance],
        [0, 1, 0, 0],
        [0, 0, 0, 1]],

        [[0, 0, 1, distance],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]],

        [[-1, 0, 0, 0],
        [0, 0, 1, distance],
        [0, 1, 0, 0],
        [0, 0, 0, 1]],

        [[0, 0, -1, -distance],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]],

        [[0, 0, -1, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, distance],
        [0, 0, 0, 1]],

        [[0, 0, -1, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, -distance],
        [0, 0, 0, 1]],
    ], dtype=torch.float32)
    return c2w_mat


