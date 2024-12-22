import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr


############# functions utils #############
def dot(x, y, dim=-1):
    return torch.sum(x * y, dim=dim, keepdim=True)

def length(x, dim=-1, eps=1e-20):
    return torch.sqrt(torch.clamp(dot(x, x, dim=dim), min=eps))

def safe_normalize(x, dim=-1, eps=1e-20):
    return x / length(x, dim=dim, eps=eps)

def reflect(x: torch.Tensor, n: torch.Tensor, dim=-1) -> torch.Tensor:
    return 2 * dot(x, n, dim=dim) * n - x

def scale_tensor(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, torch.Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat

def contract_to_unisphere(x, bbox, unbounded: bool = False):
    if unbounded:
        x = scale_tensor(x, bbox, (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        x = scale_tensor(x, bbox, (0, 1))
    return x

def avg_pool_nhwc(x  : torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

