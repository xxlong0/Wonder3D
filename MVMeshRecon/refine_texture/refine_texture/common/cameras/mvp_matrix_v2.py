import torch


def intr_to_proj(intr_mtx:torch.Tensor, near=0.01, far=1000.0, perspective=True):
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


def proj_to_intr(proj_mtx:torch.Tensor, perspective=True):
    intr_mtx = torch.zeros((*proj_mtx.shape[:-2], 3, 3), dtype=proj_mtx.dtype, device=proj_mtx.device)
    if perspective:
        intr_mtx[..., 0, 0] = proj_mtx[..., 0, 0] / 2.0
        intr_mtx[..., 1, 1] = - proj_mtx[..., 1, 1] / 2.0  # for nvdiffrast
        intr_mtx[..., 2, 0] = 0.5
        intr_mtx[..., 2, 1] = 0.5
        intr_mtx[..., 2, 2] = 1.0
    else:
        intr_mtx[..., 0, 0] = proj_mtx[..., 0, 0]
        intr_mtx[..., 1, 1] = - proj_mtx[..., 1, 1]  # for nvdiffrast
        intr_mtx[..., 2, 0] = 0.5
        intr_mtx[..., 2, 1] = 0.5
        intr_mtx[..., 2, 2] = 1.0
    return intr_mtx


def c2w_to_w2c(c2w:torch.Tensor):
    # y = Rx + t, x = R_inv(y - t)
    w2c = torch.zeros((*c2w.shape[:-2], 4, 4), dtype=c2w.dtype, device=c2w.device)
    w2c[..., :3, :3] = c2w[..., :3, :3].transpose(-1, -2)
    w2c[..., :3, 3:] = -c2w[..., :3, :3].transpose(-1, -2) @ c2w[..., :3, 3:]
    w2c[..., 3, 3] = 1.0
    return w2c


def get_mvp_mtx(proj_mtx:torch.Tensor, w2c:torch.Tensor):
    return proj_mtx @ w2c


def project(v_pos:torch.Tensor, c2ws:torch.Tensor, intrinsics:torch.Tensor, perspective=True, proj=None):
    v_pos_homo = torch.cat([v_pos, torch.ones_like(v_pos[..., :1])], dim=-1)
    w2cs_mtx = c2w_to_w2c(c2ws)
    if proj is not None:
        proj_mtx = proj[0]
        # print("proj:", proj_mtx, intr_to_proj(intrinsics, perspective=perspective))
    else:
        proj_mtx = intr_to_proj(intrinsics, perspective=perspective)
        # print("intr_to_proj:", proj_mtx, proj)
    mvp_mtx = get_mvp_mtx(proj_mtx, w2cs_mtx)
    v_pos_clip = torch.matmul(v_pos_homo, mvp_mtx.transpose(-1, -2))
    v_depth = v_pos_clip[..., [3]]
    v_pos_ndc = v_pos_clip[..., :2] / v_depth
    return v_pos_ndc, v_depth

def unproject(v_pos_ndc:torch.Tensor, v_depth:torch.Tensor, c2ws:torch.Tensor, intrinsics:torch.Tensor, perspective=True):
    proj_mtx = intr_to_proj(intrinsics, perspective=perspective)
    v_pos_homo = torch.cat([v_pos_ndc * v_depth, torch.zeros_like(v_depth), v_depth], dim=-1)
    v_pos_homo = torch.matmul(v_pos_homo, proj_mtx.inverse())
    v_pos_homo[..., -1] = 1.0
    v_pos_homo = torch.matmul(v_pos_homo, c2ws.transpose(-1, -2))
    v_pos = v_pos_homo[..., :3]
    return v_pos


def discretize(v_pos_ndc:torch.Tensor, H:int, W:int, ndc=True, align_corner=False):
    uf, vf = v_pos_ndc.unbind(-1)
    if ndc:
        uf = uf * 0.5 + 0.5
        vf = vf * 0.5 + 0.5
    if not align_corner:
        ui = torch.floor(uf * W).to(dtype=torch.int64)
        vi = torch.floor(vf * H).to(dtype=torch.int64)
    else:
        ui = torch.floor(uf * (W - 1) + 0.5).to(dtype=torch.int64)
        vi = torch.floor(vf * (H - 1) + 0.5).to(dtype=torch.int64)
    v_pos_pix = torch.stack([ui, vi], dim=-1)
    return v_pos_pix

def undiscretize(v_pos_pix:torch.Tensor, H:int, W:int, ndc=True, align_corner=False):
    ui, vi = v_pos_pix.unbind(-1)
    if not align_corner:
        uf = (ui + 0.5) / W
        vf = (vi + 0.5) / H
    else:
        uf = ui / (W - 1)
        vf = vi / (H - 1)
    if ndc:
        uf = uf * 2.0 - 1.0
        vf = vf * 2.0 - 1.0
    v_pos_ndc = torch.stack([uf, vf], dim=-1)
    return v_pos_ndc


