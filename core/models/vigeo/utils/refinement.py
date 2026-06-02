import torch

from .mis_match_filter import filter_depth


DEFAULT_FILTER_CONFIGS = (
    (4, 24, 64, 16),
    (16, 12, 2, 256),
    (64, 6, 1, 4096),
)


def multi_scale_filter_depth(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    mask: torch.Tensor,
    focal: torch.Tensor,
    trunc: float = 1.0,
    scale_configs: tuple[tuple[int, int, int, int], ...] = DEFAULT_FILTER_CONFIGS,
) -> torch.Tensor:
    filtered_mask = None
    for level, align_resolution, k_value, chunk_size in scale_configs:
        level_mask = filter_depth(
            points_pred=points_pred,
            points_gt=points_gt,
            mask=mask,
            focal=focal,
            level=level,
            chunk_size=chunk_size,
            align_resolution=align_resolution,
            K=k_value,
            trunc=trunc,
        )
        if level_mask is not None:
            filtered_mask = level_mask if filtered_mask is None else filtered_mask & level_mask

    if filtered_mask is None:
        *batch_shape, height, width, _ = points_pred.shape
        return mask.reshape(*batch_shape, height, width)
    return filtered_mask
