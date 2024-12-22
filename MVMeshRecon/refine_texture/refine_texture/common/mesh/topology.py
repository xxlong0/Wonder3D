import torch


def dilate_vertex(f_v_idx:torch.Tensor, v_mask:torch.Tensor):
    ...  # TODO


def dilate_edge(f_v_idx:torch.Tensor, e_mask:torch.Tensor):
    ...  # TODO


def dilate_face(f_v_idx:torch.Tensor, f_mask:torch.Tensor, V:int, depth=1):
    if depth <= 0:
        return f_mask
    v_value = torch.zeros((V,), dtype=torch.int64, device=f_v_idx.device)
    f_ones = torch.ones((f_v_idx.shape[0],), dtype=torch.int64, device=f_v_idx.device)
    f_mask_v_idx = torch.masked_select(f_v_idx, f_mask.unsqueeze(-1)).reshape(-1, 3)
    v_value = v_value.scatter_add(0, f_mask_v_idx[:, 0], f_ones).scatter_add(0, f_mask_v_idx[:, 1], f_ones).scatter_add(0, f_mask_v_idx[:, 2], f_ones)
    f_v_value = torch.gather(v_value.unsqueeze(-1).tile(1, 3), 0, f_v_idx)
    f_mask = (f_v_value.sum(dim=-1) > 0)
    return dilate_face(f_v_idx, f_mask, V, depth=depth-1)


def erode_face(f_v_idx:torch.Tensor, f_mask:torch.Tensor, V:int, depth=1):
    return ~dilate_face(f_v_idx, ~f_mask, V, depth=depth)


def dilate_erode_face(f_v_idx:torch.Tensor, f_mask:torch.Tensor, V:int, depth=1):
    return dilate_face(f_v_idx, erode_face(f_v_idx, dilate_face(f_v_idx, f_mask, V, depth=depth), V, depth=2*depth), V, depth=depth)


def get_boundary(f_v_idx:torch.Tensor, V:int):
    e_v_idx_full = torch.cat([f_v_idx[:, [0, 1]], f_v_idx[:, [1, 2]], f_v_idx[:, [2, 0]]], dim=0)
    e_v_idx_sorted = torch.sort(e_v_idx_full, dim=-1).values
    e_v_idx, f_e_idx = torch.unique(e_v_idx_sorted, dim=0, sorted=False, return_inverse=True, return_counts=False)
    f_e_idx = f_e_idx.reshape(-1, 3)
    e_value = torch.zeros((e_v_idx.shape[0],), dtype=torch.int64, device=f_v_idx.device)
    f_ones = torch.ones((f_v_idx.shape[0],), dtype=torch.int64, device=f_v_idx.device)
    e_value = e_value.scatter_add(0, f_e_idx[:, 0], f_ones).scatter_add(0, f_e_idx[:, 1], f_ones).scatter_add(0, f_e_idx[:, 2], f_ones)
    e_mask_boundary = (e_value == 1)
    v_idx_boundary = torch.masked_select(e_v_idx, e_mask_boundary.unsqueeze(-1))
    v_mask_boundary = torch.zeros((V,), dtype=torch.bool, device=f_v_idx.device)
    v_mask_boundary = torch.scatter(v_mask_boundary, 0, v_idx_boundary, 1)
    return v_idx_boundary, v_mask_boundary

