# core/adapters/depthanything_adapter.py
import os, re, math
from pathlib import Path
from typing import Tuple, Callable, Dict, Any, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from core.models.depth_anything_v2.dpt import DepthAnythingV2

# Configs from DA-V2 paper/code
_DA2_CFG = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [ 48,  96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [ 96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512,1024,1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536,1536,1536,1536]},
}

_MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Map short spec -> filename in Kijai/DepthAnythingV2-safetensors
_DEFAULT_FILENAMES = {
    "vits_fp16": "depth_anything_v2_vits_fp16.safetensors",
    "vits_fp32": "depth_anything_v2_vits_fp32.safetensors",
    "vitb_fp16": "depth_anything_v2_vitb_fp16.safetensors",
    "vitb_fp32": "depth_anything_v2_vitb_fp32.safetensors",
    "vitl_fp16": "depth_anything_v2_vitl_fp16.safetensors",
    "vitl_fp32": "depth_anything_v2_vitl_fp32.safetensors",
    "vitg_fp32": "depth_anything_v2_vitg_fp32.safetensors",
    # metric variants
    "metric_hypersim_vitl_fp32": "depth_anything_v2_metric_hypersim_vitl_fp32.safetensors",
    "metric_vkitti_vitl_fp32":   "depth_anything_v2_metric_vkitti_vitl_fp32.safetensors",
}

def _snap_kijai_if_needed(filename: str, cache_dir: str) -> str:
    cache_dir = str(Path(cache_dir).expanduser())
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, filename)
    if not os.path.exists(local_path):
        snapshot_download(
            repo_id="Nap/depth_anything_v2_vitg",
            allow_patterns=[f"*{filename}*"],
            local_dir=cache_dir,
            local_dir_use_symlinks=False
        )
    return local_path

def _parse_spec(spec: str) -> Tuple[str, bool, float]:
    """
    Returns (encoder_key, is_metric, max_depth)
    """
    s = spec.lower()
    enc = "vitg" if "vitg" in s else "vitl" if "vitl" in s else "vitb" if "vitb" in s else "vits"
    is_metric = "metric" in s
    max_depth = 20.0 if "hypersim" in s else 80.0
    return enc, is_metric, max_depth

def load_da_v2_adapter(
    spec_or_path: str,
    cache_dir: str,
) -> Tuple[Callable, Dict[str, Any]]:
    """
    spec_or_path:
      - absolute/local path to a *.safetensors file
      - short spec like 'vitg_fp32'
      - 'Kijai/DepthAnythingV2-safetensors:depth_anything_v2_vitg_fp32.safetensors'
    """
    # Resolve a local weight file path
    if os.path.isfile(spec_or_path) and spec_or_path.endswith(".safetensors"):
        weight_path = spec_or_path
    elif ":" in spec_or_path and spec_or_path.split(":", 1)[0].strip().lower().endswith("safetensors"):
        # unlikely form—ignore
        weight_path = spec_or_path
    elif spec_or_path.startswith("Nap/"):
        _, fname = spec_or_path.split(":", 1)
        weight_path = _snap_kijai_if_needed(fname.strip(), cache_dir)
    else:
        fname = _DEFAULT_FILENAMES.get(spec_or_path.lower(), None)
        if fname is None:
            # try to guess e.g. "depth_anything_v2_vitg_fp32.safetensors"
            if spec_or_path.lower().endswith(".safetensors"):
                fname = spec_or_path
            else:
                raise ValueError(f"Unknown DA-V2 spec: {spec_or_path}")
        weight_path = _snap_kijai_if_needed(fname, cache_dir)

    # Infer encoder/metric
    enc, is_metric, max_depth = _parse_spec(Path(weight_path).name)

    cfg = dict(_DA2_CFG[enc])
    if is_metric:
        cfg.update({"is_metric": True, "max_depth": max_depth})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float32 if "fp32" in weight_path.lower() else torch.float16

    # Build model and load weights
    model = DepthAnythingV2(**cfg)
    sd = load_safetensors(weight_path, device="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    # Preprocess
    tfm = transforms.Normalize(mean=_MEAN_STD[0], std=_MEAN_STD[1])

    @torch.no_grad()
    def run(images, inference_size=None):
        if not isinstance(images, list):
            images = [images]
        outputs = []
        for img in images:
            if isinstance(img, Image.Image):
                W, H = img.size
                if inference_size:
                    img = img.resize(inference_size, Image.BICUBIC)
                    W, H = img.size
                # to CHW float
                t = torch.from_numpy(
                    (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                     .view(H, W, 3)
                     .numpy() / 255.0).astype("float32")
                ).permute(2, 0, 1)
            else:
                # assume CHW float tensor [0,1]
                t = img

            # snap to multiple of 14 (DA-V2 requirement)
            h, w = int(t.shape[1]), int(t.shape[2])
            Hs = h - (h % 14)
            Ws = w - (w % 14)
            if (Hs, Ws) != (h, w):
                t = F.interpolate(t.unsqueeze(0), size=(Hs, Ws), mode="bilinear", align_corners=False).squeeze(0)

            t = tfm(t).unsqueeze(0).to(device, dtype=dtype)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda" and dtype==torch.float16)):
                d = model(t)[0]  # (H, W)
            # min-max normalize
            d = (d - d.min()) / (d.max() - d.min() + 1e-6)
            if is_metric:
                d = 1.0 - d
            outputs.append({"predicted_depth": d.detach().cpu()})
        return outputs

    run._is_dav2 = True
    return run, {"is_diffusion": True, "diffusion_kind": "depth", "is_dav2": True}
