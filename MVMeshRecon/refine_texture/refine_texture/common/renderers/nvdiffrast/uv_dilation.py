import numpy as np
import cv2
import torch


def uv_dilation_cv(map_Kd:torch.Tensor, map_mask:torch.Tensor, max_iters=1000):
    '''
    uv map dilation (OpenCV version)

    map_Kd: [N, C, H, W], float32, black background
    map_mask: [N, 1, H, W], bool
    '''
    _map_Kd = map_Kd.permute(0, 2, 3, 1).mul(255).cpu().numpy().astype(np.uint8)
    _map_mask = (~map_mask.to(dtype=torch.bool)).permute(0, 2, 3, 1).mul(255).cpu().numpy().astype(np.uint8)
    for i in range(map_Kd.shape[0]):
        map_Kd[i] = torch.as_tensor(
            cv2.inpaint(_map_Kd[i], _map_mask[i], 2, cv2.INPAINT_TELEA),
            dtype=map_Kd.dtype, device=map_Kd.device,
        ).div(255).permute(2, 0, 1)
    return map_Kd


def uv_dilation(map_Kd:torch.Tensor, map_mask:torch.Tensor, max_iters=1000):
    '''
    uv map dilation (max pool version)

    map_Kd: [N, C, H, W], float32
    map_mask: [N, 1, H, W], bool
    '''
    _map_mask = map_mask.clone().to(dtype=torch.bool)
    _map_Kd = map_Kd * _map_mask
    for iter in range(max_iters):
        if _map_mask.prod() > 0:
            break
        _map_Kd, _map_mask = _uv_dilation(_map_Kd, _map_mask)
    _map_mask = _map_mask.to(dtype=torch.float32)
    map_Kd = torch.clamp(_map_mask * _map_Kd + (1 - _map_mask) * map_Kd, 0.0, 1.0)
    return map_Kd


def _uv_dilation(map_Kd:torch.Tensor, map_mask:torch.Tensor, kernel_size=3):
    '''
    iterable uv map dilation

    map_Kd: [N, C, H, W], float32, black background
    map_mask: [N, 1, H, W], bool
    '''
    kernel_size = 2 * (kernel_size // 2) + 1
    map_mask = map_mask.to(dtype=torch.float32)
    map_mask_dilation = torch.max_pool2d(map_mask, kernel_size, 1, kernel_size // 2)
    map_boundary_mask = (map_mask_dilation - map_mask).to(dtype=torch.bool)
    return torch.where(map_boundary_mask, torch.max_pool2d(map_Kd, kernel_size, 1, kernel_size // 2), map_Kd), map_mask_dilation.to(torch.bool)



