import torch
from torch import Tensor


def get_projection_matrix_perspective(intrinsics:Tensor, near: float = 0.01, far: float = 1000.0):
    '''
    intrinsics: [batch_size, 3, 3], normalized

    proj_mtx: [batch_size, 4, 4]
    '''
    batch_size = intrinsics.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32).to(intrinsics)
    proj_mtx[:, 0, 0] = 2 * intrinsics[:, 0, 0]
    proj_mtx[:, 1, 1] = - 2 * intrinsics[:, 1, 1] # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_projection_matrix_orthogonal(intrinsics:Tensor, near: float = 0.01, far: float = 100.0):
    '''
    intrinsics: [batch_size, 3, 3], normalized

    proj_mtx: [batch_size, 4, 4]
    '''
    batch_size = intrinsics.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32).to(intrinsics)
    proj_mtx[:, 0, 0] = intrinsics[:, 0, 0]
    proj_mtx[:, 1, 1] = - intrinsics[:, 0, 0] # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -2.0 / (far - near)
    proj_mtx[:, 3, 3] = 1.0
    proj_mtx[:, 2, 3] = - (far + near) / (far - near)
    return proj_mtx

def get_mvp_matrix(c2w:Tensor, proj_mtx:Tensor):
    '''
    c2w: [batch_size, 4, 4]
    proj_mtx: [batch_size, 4, 4]

    mvp_mtx: [batch_size, 4, 4]
    '''
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx
