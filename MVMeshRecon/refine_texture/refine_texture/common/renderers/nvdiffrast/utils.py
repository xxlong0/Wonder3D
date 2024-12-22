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



#----------------------------------------------------------------------------
# Cubemap utility functions
#----------------------------------------------------------------------------

def cube_to_dir(s, x, y):
    # x: origin to front, y: origin to top, z: origin to right
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x  # front
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x  # back
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y    # top
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y  # down
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)   # right
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x) # left
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    '''
    latlong_map: [Hi, Wi, 3]
    res: Ho, Wo
    cubemap: [6, Ho, Wo, 3]

    http://www.paulbourke.net/panorama/cubemaps/
    '''
    device = latlong_map.device
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=device)
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                                indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5  # phi in range(-pi, pi), scale to (0, 1)
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi  # theta in range(0, pi), scale to (0, 1)
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def cubemap_to_latlong(cubemap, res):
    '''
    cubemap: [6, Hi, Wi, 3]
    res: Ho, Wo
    latlong_map: [Ho, Wo, 3]

    NOTE: cubemap_to_latlong and latlong_to_cubemap are not reciprocal
    '''
    device = cubemap.device
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)  # theta in range(0, pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)  # phi in range(-pi, pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = safe_normalize(cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

