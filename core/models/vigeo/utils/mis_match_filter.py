import math
import torch
import utils3d
from typing import Literal
from .alignment import align_points_scale_xyz_shift

def filter_depth(
    points_pred,
    points_gt,
    mask,
    focal,
    level: Literal[4, 16, 64] = 16,
    chunk_size: int = 256, # Literal[256, 4096]
    align_resolution: Literal[24, 12, 6] = 12,
    K: int = 4,
    trunc: float = 1.0,
    ratio: float = 1.0,
):

    device, dtype = points_pred.device, points_pred.dtype
    *batch_shape, height, width, _ = points_pred.shape

    points_pred = points_pred.reshape(-1, height, width, 3)
    points_gt = points_gt.reshape(-1, height, width, 3)
    mask = mask.reshape(-1, height, width)
    focal = focal.reshape(-1)

    if focal.numel() == 1 and len(batch_shape) > 0:
        focal = focal.expand(batch_shape[0])

    radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)

    where_mask = torch.where(mask)  # tuple of (batch_idx, i, j), each of shape (N,)
    N_full = where_mask[0].size(0)
    if N_full > 0:
        indices = torch.arange(0, N_full, step=K, device=device)
        where_mask = tuple(wm[indices] for wm in where_mask)

    N = where_mask[0].size(0)

    if N == 0:
        print('No valid points')
        return

    filtered_mask = mask.clone()

    num_chunks = (N + chunk_size - 1) // chunk_size  # ceiling division

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, N)

        patch_batch_idx = where_mask[0][start:end]  # [chunk_size,]
        patch_anchor_i = where_mask[1][start:end]   # [chunk_size,]
        patch_anchor_j = where_mask[2][start:end]   # [chunk_size,]

        patch_i, patch_j = torch.meshgrid(
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        indexing='ij')

        patch_i, patch_j = patch_i + patch_anchor_i[:, None, None], patch_j + patch_anchor_j[:, None, None]
        patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
        patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)

        pred_patch_anchor_points = points_pred[patch_batch_idx, patch_anchor_i, patch_anchor_j]
        pred_patch_radius_3d =  0.5 / level / focal[patch_batch_idx] * pred_patch_anchor_points[:, 2]
        pred_patch_points = points_pred[patch_batch_idx[:, None, None], patch_i, patch_j]
        pred_patch_dist =  (pred_patch_points - pred_patch_anchor_points[:, None, None, :]).norm(dim=-1)
        patch_mask &= mask[patch_batch_idx[:, None, None], patch_i, patch_j]
        patch_mask &= pred_patch_dist <= pred_patch_radius_3d[:, None, None]

        # Pick only non-empty patches
        MINIMUM_POINTS_PER_PATCH = 32
        nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)
        num_nonempty_patches = nonempty[0].shape[0]
        if num_nonempty_patches == 0:
            continue

        patch_batch_idx, patch_i, patch_j = patch_batch_idx[nonempty], patch_i[nonempty], patch_j[nonempty]
        patch_mask = patch_mask[nonempty]

        pred_patch_points = pred_patch_points[nonempty]                         # [num_nonempty_patches, patch_h, patch_w, 3]
        pred_patch_radius_3d = pred_patch_radius_3d[nonempty]                   # [num_nonempty_patches]
        pred_patch_anchor_points = pred_patch_anchor_points[nonempty]           # [num_nonempty_patches, 3]
        gt_patch_points = points_gt[patch_batch_idx[:, None, None], patch_i, patch_j]

        gt_patch_points_lr, pred_patch_points_lr, patch_lr_mask = utils3d.pt.masked_nearest_resize(
            gt_patch_points, pred_patch_points, mask=patch_mask, size=(align_resolution, align_resolution))
        local_scale, local_shift = align_points_scale_xyz_shift(
            gt_patch_points_lr.flatten(-3, -2), pred_patch_points_lr.flatten(-3, -2), patch_lr_mask.flatten(-2) / pred_patch_radius_3d[:, None].add(1e-7), trunc=trunc)
        patch_valid = local_scale > 0
        valid_indices = torch.where(patch_valid)[0]

        if valid_indices.numel() == 0:
            continue

        # Filter all tensors to keep only valid patches
        patch_batch_idx = patch_batch_idx[valid_indices]
        patch_i = patch_i[valid_indices]
        patch_j = patch_j[valid_indices]
        patch_mask = patch_mask[valid_indices]  # ← this is the original mask, not polluted!
        pred_patch_points = pred_patch_points[valid_indices]
        gt_patch_points = gt_patch_points[valid_indices]
        pred_patch_radius_3d = pred_patch_radius_3d[valid_indices]
        local_scale = local_scale[valid_indices]
        local_shift = local_shift[valid_indices]

        gt_patch_points = local_scale[:, None, None, None] * gt_patch_points + local_shift[:, None, None, :]
        residual = (pred_patch_points - gt_patch_points).norm(dim=-1)
        is_outlier = (residual > (pred_patch_radius_3d[:, None, None] * ratio)) & patch_mask

        if is_outlier.any():
            i_valid = patch_i[is_outlier].clamp(0, height - 1)
            j_valid = patch_j[is_outlier].clamp(0, width - 1)
            b_valid = patch_batch_idx[:, None, None].expand_as(is_outlier)[is_outlier]
            filtered_mask[b_valid, i_valid, j_valid] = False

    return filtered_mask.reshape(*batch_shape, height, width)
