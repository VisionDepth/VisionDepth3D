from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


def read_image(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1)


def read_sparse_depth(path: str | Path) -> torch.Tensor:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D sparse depth map, got shape {depth.shape}")

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.where(depth > 0.0, depth, 0.0).astype(np.float32)
    return torch.from_numpy(depth).unsqueeze(0)


def load_image_sequence(image_paths: Sequence[str | Path]) -> torch.Tensor:
    return torch.stack([read_image(path) for path in image_paths], dim=0)


def load_depth_sequence(depth_paths: Sequence[str | Path]) -> torch.Tensor:
    return torch.stack([read_sparse_depth(path) for path in depth_paths], dim=0)


def load_intrinsic(path: str | Path, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsic = np.load(path).astype(np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsic matrix, got shape {intrinsic.shape}")

    intrinsic = intrinsic.copy()
    if intrinsic[0, 0] > 4.0 or intrinsic[1, 1] > 4.0 or intrinsic[0, 2] > 2.0 or intrinsic[1, 2] > 2.0:
        intrinsic[0, :] /= width
        intrinsic[1, :] /= height

    focal = 1.0 / (1.0 / intrinsic[0, 0] ** 2 + 1.0 / intrinsic[1, 1] ** 2) ** 0.5
    return torch.from_numpy(intrinsic), torch.tensor(focal, dtype=torch.float32)
