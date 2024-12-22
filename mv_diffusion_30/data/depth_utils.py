import matplotlib
import numpy as np
import torch

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def scale_depth_to_model(depth, camera_type='ortho'):
    """
    Scale depth from the original range.
    """
    assert camera_type == 'ortho' or camera_type == 'persp'
    w, h = depth.shape

    if camera_type == 'ortho':
        original_min = 9000
        original_max = 17000
        target_min = 2000
        target_max = 62000

        mask = depth != 0
        # Scale depth to [0, 1]
        depth_normalized = np.zeros([w, h])
        depth_normalized[mask] = (depth[mask] - original_min) / (original_max - original_min)

        # Scale depth to [2000, 60000]
        scaled_depth = np.zeros([w, h])
        scaled_depth[mask] = depth_normalized[mask] * (target_max - target_min) + target_min

    else:
        original_min = 4000
        original_max = 13000
        target_min = 2000
        target_max = 62000

        mask = depth != 0
        # Scale depth to [0, 1]
        depth_normalized = np.zeros([w, h])
        depth_normalized[mask] = (depth[mask] - original_min) / (original_max - original_min)

        # Scale depth to [2000, 60000]
        scaled_depth = np.zeros([w, h])
        scaled_depth[mask] = depth_normalized[mask] * (target_max - target_min) + target_min

    scaled_depth[scaled_depth > 62000] = 0
    scaled_depth = scaled_depth / 65535. # [0, 1]

    return scaled_depth

def rescale_depth_to_world(scaled_depth, camera_type='ortho'):
    """
    Rescale depth from the scaled range back to the original range.
    """
    assert camera_type == 'ortho' or camera_type == 'persp'
    scaled_depth = scaled_depth * 65535.
    w, h = scaled_depth.shape

    if camera_type == 'ortho':
        original_min = 9000
        original_max = 17000
        target_min = 2000
        target_max = 62000

        mask = scaled_depth != 0
        rescaled_depth_norm = np.zeros([w, h])
        # Rescale depth to [0, 1]
        rescaled_depth_norm[mask] = (scaled_depth[mask] - target_min) / (target_max - target_min)

        # Rescale depth to [9000, 17000]
        rescaled_depth = np.zeros([w, h])
        rescaled_depth[mask] = rescaled_depth_norm[mask] * (original_max - original_min) + original_min

    else:
        original_min = 4000
        original_max = 13000
        target_min = 2000
        target_max = 62000

        mask = scaled_depth != 0
        rescaled_depth_norm = np.zeros([w, h])
        # Rescale depth to [0, 1]
        rescaled_depth_norm[mask] = (scaled_depth[mask] - target_min) / (target_max - target_min)

        # Rescale depth to [9000, 17000]
        rescaled_depth = np.zeros([w, h])
        rescaled_depth[mask] = rescaled_depth_norm[mask] * (original_max - original_min) + original_min

    return rescaled_depth