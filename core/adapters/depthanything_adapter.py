# core/adapters/depthanything_adapter.py
import os
import math
from pathlib import Path
from typing import Tuple, Callable, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
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

_DEFAULT_REPO_ID = "Kijai/DepthAnythingV2-safetensors"

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

def _snap_kijai_if_needed(filename: str, cache_dir: str, repo_id: str = _DEFAULT_REPO_ID) -> str:
    cache_dir = str(Path(cache_dir).expanduser())
    os.makedirs(cache_dir, exist_ok=True)

    filename = filename.strip()
    local_path = os.path.join(cache_dir, filename)

    if not os.path.exists(local_path):
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[filename, f"**/{filename}"],
            local_dir=cache_dir,
        )

    if os.path.isfile(local_path):
        return local_path

    matches = list(Path(cache_dir).rglob(filename))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(
        f"Failed to locate downloaded Depth Anything V2 weight file: {filename!r} "
        f"in cache dir {cache_dir!r} from repo {repo_id!r}"
    )
    return local_path

def _parse_spec(spec: str) -> Tuple[str, bool, float]:
    """
    Returns (encoder_key, is_metric, max_depth)
    """
    s = spec.lower()

    if "vitg" in s:
        enc = "vitg"
    elif "vitl" in s:
        enc = "vitl"
    elif "vitb" in s:
        enc = "vitb"
    elif "vits" in s:
        enc = "vits"
    else:
        raise ValueError(
            f"Could not infer Depth Anything V2 encoder from weight filename: {spec!r}. "
            "Expected filename to contain one of: vits, vitb, vitl, vitg."
        )

    is_metric = "metric" in s
    max_depth = 20.0 if "hypersim" in s else 80.0
    return enc, is_metric, max_depth

def load_da_v2_adapter(
    spec_or_path: str,
    cache_dir: str,
    use_fp16: bool = False,
) -> Tuple[Callable, Dict[str, Any]]:
    """
    spec_or_path:
      - absolute/local path to a *.safetensors file
      - short spec like 'vitg_fp32'
      - 'Kijai/DepthAnythingV2-safetensors:depth_anything_v2_vitg_fp32.safetensors'
    """
    # Resolve a local weight file path
    spec = str(spec_or_path).strip()
    expanded_path = str(Path(spec).expanduser())

    if expanded_path.lower().endswith(".safetensors") and (
        os.path.isabs(expanded_path) or os.path.dirname(expanded_path)
    ):
        if not os.path.isfile(expanded_path):
            raise FileNotFoundError(f"Depth Anything V2 weight file not found: {expanded_path}")
        weight_path = expanded_path

    elif os.path.isfile(expanded_path) and expanded_path.lower().endswith(".safetensors"):
        weight_path = expanded_path

    elif ":" in spec and "/" in spec.split(":", 1)[0]:
        # Hugging Face form: "repo/name:filename.safetensors"
        repo_id, fname = spec.split(":", 1)
        repo_id = repo_id.strip()
        fname = fname.strip()

        if not fname.lower().endswith(".safetensors"):
            raise ValueError(f"Expected a .safetensors filename in spec: {spec_or_path}")

        weight_path = _snap_kijai_if_needed(fname, cache_dir, repo_id=repo_id)

    else:
        fname = _DEFAULT_FILENAMES.get(spec.lower())
        if fname is None:
            if spec.lower().endswith(".safetensors"):
                fname = Path(spec).name
            else:
                raise ValueError(f"Unknown DA-V2 spec: {spec_or_path}")

        weight_path = _snap_kijai_if_needed(fname, cache_dir)

    # Infer encoder/metric
    enc, is_metric, max_depth = _parse_spec(Path(weight_path).name)

    cfg = dict(_DA2_CFG[enc])
    if is_metric:
        cfg.update({"is_metric": True, "max_depth": max_depth})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use FP16 on CUDA when either:
    # - the checkpoint name is explicitly fp16
    # - the caller/UI requested FP16
    #
    # This lets fp32 safetensors like vitg_fp32 run faster on RTX GPUs.
    use_half = bool(device == "cuda" and (use_fp16 or "fp16" in weight_path.lower()))
    dtype = torch.float16 if use_half else torch.float32

    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    # Build model and load weights
    model = DepthAnythingV2(**cfg)
    sd = load_safetensors(weight_path, device="cpu")
    load_result = model.load_state_dict(sd, strict=False)

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Depth Anything V2 checkpoint did not match the model configuration.\n"
            f"Weight file: {weight_path}\n"
            f"Missing keys: {load_result.missing_keys}\n"
            f"Unexpected keys: {load_result.unexpected_keys}"
        )

    del sd
    model.eval().to(device=device, dtype=dtype)

    if device == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    # Preprocess
    # Preprocess constants on GPU.
    mean_t = torch.tensor(
        _MEAN_STD[0],
        device=device,
        dtype=dtype,
    ).view(1, 3, 1, 1)

    std_t = torch.tensor(
        _MEAN_STD[1],
        device=device,
        dtype=dtype,
    ).view(1, 3, 1, 1)

    @torch.inference_mode()
    def run(images, inference_size=None):
        """
        Batched Depth Anything V2 inference.

        Returns list[{"predicted_depth": tensor_cpu}] for compatibility with
        the existing VD3D depth pipeline.

        Speed improvements over old path:
        - stack frames into a batch
        - one GPU transfer per batch
        - one model call per batch
        - batched output resize when possible
        """
        if not isinstance(images, (list, tuple)):
            images = [images]

        if len(images) == 0:
            return []

        bicubic = getattr(Image, "Resampling", Image).BICUBIC

        infer_w = infer_h = None
        if inference_size:
            infer_w, infer_h = map(int, inference_size)

        tensors = []
        original_sizes = []

        # ------------------------------------------------------------
        # CPU decode/resize/stack
        # ------------------------------------------------------------
        for img in images:
            if isinstance(img, Image.Image):
                img = ImageOps.exif_transpose(img).convert("RGB")
                out_w, out_h = img.size
                original_sizes.append((out_h, out_w))

                if inference_size and img.size != (infer_w, infer_h):
                    img = img.resize((infer_w, infer_h), bicubic)

                arr = np.asarray(img, dtype=np.uint8)
                t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                t = t.float().div_(255.0)

            else:
                if not torch.is_tensor(img):
                    raise TypeError(
                        f"Expected PIL.Image.Image or torch.Tensor, got {type(img)!r}"
                    )

                if img.ndim != 3:
                    raise ValueError(
                        f"Expected CHW tensor with 3 dimensions, got shape {tuple(img.shape)}"
                    )

                if img.shape[0] != 3:
                    raise ValueError(
                        f"Expected CHW RGB tensor with 3 channels, got shape {tuple(img.shape)}"
                    )

                out_h, out_w = int(img.shape[-2]), int(img.shape[-1])
                original_sizes.append((out_h, out_w))

                t = img.detach()

                if t.device.type != "cpu":
                    t = t.cpu()

                if not torch.is_floating_point(t):
                    t = t.float().div_(255.0)
                else:
                    t = t.float()

                # Do not call t.max().item() here; that can sync if input was GPU.
                # VD3D tensors are expected to be 0..1 floats or uint8.

                if inference_size and t.shape[-2:] != (infer_h, infer_w):
                    t = F.interpolate(
                        t.unsqueeze(0),
                        size=(infer_h, infer_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

            tensors.append(t)

        # If no inference_size was given and input images have mixed sizes,
        # fall back to processing each image separately. This keeps image-folder
        # mode safe for mixed-resolution folders.
        shapes = {tuple(t.shape[-2:]) for t in tensors}

        if len(shapes) != 1:
            outputs = []
            for single_img in images:
                outputs.extend(run([single_img], inference_size=inference_size))
            return outputs

        batch = torch.stack(tensors, dim=0)  # [B,3,H,W]

        # Snap to multiple of 14 for ViT/DINOv2 patch safety.
        h, w = int(batch.shape[-2]), int(batch.shape[-1])
        Hs = max(14, math.ceil(h / 14) * 14)
        Ws = max(14, math.ceil(w / 14) * 14)

        if (Hs, Ws) != (h, w):
            batch = F.interpolate(
                batch,
                size=(Hs, Ws),
                mode="bilinear",
                align_corners=False,
            )

        # One GPU transfer for the whole batch.
        batch = batch.to(device=device, dtype=dtype, non_blocking=True)

        try:
            batch = batch.contiguous(memory_format=torch.channels_last)
        except Exception:
            batch = batch.contiguous()

        # Normalize on GPU.
        batch = (batch - mean_t) / std_t

        # ------------------------------------------------------------
        # Batched model inference
        # ------------------------------------------------------------
        with torch.autocast(
            device_type="cuda",
            enabled=(device == "cuda" and dtype == torch.float16),
        ):
            depth = model(batch)

        if isinstance(depth, (tuple, list)):
            depth = depth[0]

        # Normalize shape to [B,H,W].
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        elif depth.ndim == 2:
            depth = depth.unsqueeze(0)

        if depth.ndim != 3:
            raise RuntimeError(
                f"Unexpected Depth Anything V2 output shape: {tuple(depth.shape)}"
            )

        depth = depth.float()

        # ------------------------------------------------------------
        # Resize output back to requested/original sizes
        # ------------------------------------------------------------
        same_target = len(set(original_sizes)) == 1

        if same_target:
            target_h, target_w = original_sizes[0]

            if tuple(depth.shape[-2:]) != (target_h, target_w):
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            if is_metric:
                metric_depth = depth.clamp(min=0)
                predicted_depth = torch.clamp(metric_depth / max_depth, 0.0, 1.0)

                predicted_cpu = predicted_depth.detach().cpu()
                metric_cpu = metric_depth.detach().cpu()

                return [
                    {
                        "predicted_depth": predicted_cpu[i],
                        "metric_depth": metric_cpu[i],
                    }
                    for i in range(predicted_cpu.shape[0])
                ]

            else:
                d_min = depth.amin(dim=(1, 2), keepdim=True)
                d_max = depth.amax(dim=(1, 2), keepdim=True)
                depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)

                depth_cpu = depth_norm.detach().cpu()

                return [
                    {"predicted_depth": depth_cpu[i]}
                    for i in range(depth_cpu.shape[0])
                ]

        # Mixed output sizes. Resize each prediction individually on GPU.
        outputs = []

        for i in range(depth.shape[0]):
            d = depth[i]
            target_h, target_w = original_sizes[i]

            if tuple(d.shape[-2:]) != (target_h, target_w):
                d = F.interpolate(
                    d.unsqueeze(0).unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            if is_metric:
                metric_depth = d.clamp(min=0)
                predicted_depth = torch.clamp(metric_depth / max_depth, 0.0, 1.0)

                outputs.append({
                    "predicted_depth": predicted_depth.detach().cpu(),
                    "metric_depth": metric_depth.detach().cpu(),
                })

            else:
                d_min = d.amin()
                d_max = d.amax()
                d = (d - d_min) / (d_max - d_min + 1e-6)

                outputs.append({"predicted_depth": d.detach().cpu()})

        return outputs

    run._is_dav2 = True
    return run, {"is_diffusion": False, "diffusion_kind": "depth", "is_dav2": True}
