# core/adapters/depthanything_adapter.py
import os
import math
from pathlib import Path
from typing import Tuple, Callable, Dict, Any
from contextlib import nullcontext

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

# Keep the most recently used DA-V2 model alive.
# This prevents accidental per-frame/per-batch reloads from destroying FPS.
_DAV2_MODEL_CACHE: Dict[Tuple[Any, ...], Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]] = {}
_DAV2_MODEL_CACHE_MAX = 1


def clear_da_v2_adapter_cache():
    """
    Optional external cleanup hook for the GUI/worker when switching projects/models.
    Existing active closures still keep their model alive until released.
    """
    _DAV2_MODEL_CACHE.clear()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

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
    use_directml: bool = False,
    device=None,
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

    # ------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------
    # Priority:
    #   1. Explicit device passed by main app
    #   2. DirectML if requested
    #   3. CUDA if available
    #   4. CPU fallback
    if device is not None:
        selected_device = torch.device(device) if isinstance(device, str) else device

    elif use_directml:
        try:
            import torch_directml

            if hasattr(torch_directml, "is_available"):
                try:
                    if not torch_directml.is_available():
                        raise RuntimeError("torch_directml.is_available() returned False")
                except Exception:
                    pass

            selected_device = torch_directml.device()

            # Quick sanity check.
            _ = torch.ones(1).to(selected_device).cpu()

        except Exception as e:
            print(f"⚠️ DA-V2 DirectML requested but unavailable: {e}")
            selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = selected_device

    device_type = getattr(device, "type", str(device))
    is_cuda = device_type == "cuda"
    is_directml = device_type == "privateuseone"

    # Use FP16 only on CUDA.
    # DirectML FP16 can be unstable/unsupported depending on GPU/driver/op coverage.
    use_half = bool(is_cuda and (use_fp16 or "fp16" in weight_path.lower()))
    dtype = torch.float16 if use_half else torch.float32

    if is_cuda:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            # Helps Ampere/Ada/Lovelace cards use TF32 for FP32 matmul paths.
            # This does not affect true FP16 inference, but helps if the user runs FP32.
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        except Exception:
            pass

    # Build/load model, but cache the most recently used model.
    # This protects against caller-side accidental reloads per frame/batch.
    cache_key = (
        os.path.abspath(weight_path),
        enc,
        bool(is_metric),
        float(max_depth),
        str(device),
        str(dtype),
        bool(use_half),
    )

    cached = _DAV2_MODEL_CACHE.get(cache_key)

    if cached is not None:
        model, mean_t, std_t = cached

    else:
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

        if is_cuda:
            try:
                model = model.to(memory_format=torch.channels_last)
            except Exception:
                pass

        # Preprocess constants on target device.
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

        if len(_DAV2_MODEL_CACHE) >= _DAV2_MODEL_CACHE_MAX:
            _DAV2_MODEL_CACHE.clear()

        _DAV2_MODEL_CACHE[cache_key] = (model, mean_t, std_t)
        
    try:
        p = next(model.parameters())
        print(
            f"[DA-V2] READY | weight={Path(weight_path).name} | "
            f"device={p.device} | dtype={p.dtype} | "
            f"use_half={use_half} | backend={'cuda' if is_cuda else ('directml' if is_directml else 'cpu')}",
            flush=True,
        )
    except Exception:
        print(
            f"[DA-V2] READY | weight={Path(weight_path).name} | "
            f"device={device} | dtype={dtype} | use_half={use_half}",
            flush=True,
        )

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

        infer_w = infer_h = None
        if inference_size:
            infer_w, infer_h = map(int, inference_size)

        tensors = []
        original_sizes = []

        # ------------------------------------------------------------
        # Lightweight CPU decode only.
        #
        # Important:
        # - Do not resize PIL frames on CPU.
        # - Do not convert uint8 -> float32 on CPU.
        # - Do not force GPU tensors back to CPU.
        #
        # Resize, dtype conversion, /255, and normalization happen on GPU below.
        # ------------------------------------------------------------
        for img in images:
            if isinstance(img, Image.Image):
                img = ImageOps.exif_transpose(img).convert("RGB")
                out_w, out_h = img.size
                original_sizes.append((out_h, out_w))

                # np.asarray(PIL) can be read-only. Use writable contiguous memory.
                arr = np.array(img, dtype=np.uint8, copy=True)
                t = torch.from_numpy(arr).permute(2, 0, 1)

            elif isinstance(img, np.ndarray):
                arr = np.asarray(img)

                # Remove batch dim if someone passed [1,H,W,C] or [1,C,H,W].
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]

                # CHW -> HWC if needed.
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))

                # Grayscale -> RGB.
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)

                # RGBA -> RGB.
                if arr.ndim == 3 and arr.shape[-1] == 4:
                    arr = arr[..., :3]

                # Single channel -> RGB.
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)

                if arr.ndim != 3 or arr.shape[-1] != 3:
                    raise ValueError(f"Expected RGB ndarray HxWx3, got shape {arr.shape}")

                out_h, out_w = int(arr.shape[0]), int(arr.shape[1])
                original_sizes.append((out_h, out_w))

                if np.issubdtype(arr.dtype, np.floating):
                    if arr.size and float(np.nanmax(arr)) <= 1.0:
                        arr = arr * 255.0
                    arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)

                elif arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)

                if not arr.flags.c_contiguous or not arr.flags.writeable:
                    arr = np.array(arr, dtype=np.uint8, copy=True)
                else:
                    arr = np.ascontiguousarray(arr)

                t = torch.from_numpy(arr).permute(2, 0, 1)

            else:
                if not torch.is_tensor(img):
                    raise TypeError(
                        f"Expected PIL.Image.Image, np.ndarray, or torch.Tensor, got {type(img)!r}"
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

                # Keep tensor on its current device. Moving CUDA tensors back to CPU
                # causes a sync and can destroy render FPS.
                t = img.detach()

            tensors.append(t)

        # Batching requires equal source sizes. Video frames normally satisfy this.
        # Mixed-size folders are kept safe by recursively processing one at a time.
        shapes = {tuple(t.shape[-2:]) for t in tensors}

        if len(shapes) != 1:
            outputs = []
            for single_img in images:
                outputs.extend(run([single_img], inference_size=inference_size))
            return outputs

        # If tensors came from different devices, move them to the inference device
        # before stacking. This is uncommon, but keeps the adapter robust.
        tensor_devices = {t.device for t in tensors}
        if len(tensor_devices) > 1:
            stack_device = device
            tensors = [
                t.to(
                    stack_device,
                    non_blocking=(is_cuda and t.device.type == "cpu"),
                )
                for t in tensors
            ]

        tensors = [
            t.contiguous() if not t.is_contiguous() else t
            for t in tensors
        ]

        batch = torch.stack(tensors, dim=0)  # [B,3,H,W]

        # One transfer for the whole batch. Keep uint8 until it reaches the GPU
        # so CPU does less work and PCIe transfer stays smaller.
        if batch.device != device:
            if is_cuda and batch.device.type == "cpu":
                try:
                    batch = batch.pin_memory()
                except Exception:
                    pass

            batch = batch.to(
                device=device,
                non_blocking=(is_cuda and batch.device.type == "cpu"),
            )

        # Convert and scale on GPU.
        if not torch.is_floating_point(batch):
            batch = batch.to(dtype=dtype)
            batch.div_(255.0)
        else:
            batch = batch.to(dtype=dtype)

        # Decide final model input size.
        cur_h, cur_w = int(batch.shape[-2]), int(batch.shape[-1])

        if inference_size:
            model_h, model_w = infer_h, infer_w
        else:
            # Important:
            # If VisionDepth3D UI is set to "Original", inference_size is None.
            # Do NOT run DA-V2 at full 1080p/4K by default; that will destroy FPS.
            #
            # Default max side 518 matches DA-V2/Depth Anything common inference size.
            # Set VD3D_DAV2_DEFAULT_MAX_SIDE=0 to restore true original-resolution inference.
            try:
                default_max_side = int(os.environ.get("VD3D_DAV2_DEFAULT_MAX_SIDE", "518"))
            except Exception:
                default_max_side = 518

            if default_max_side > 0 and max(cur_h, cur_w) > default_max_side:
                scale = float(default_max_side) / float(max(cur_h, cur_w))
                model_h = max(14, int(round(cur_h * scale)))
                model_w = max(14, int(round(cur_w * scale)))
            else:
                model_h, model_w = cur_h, cur_w

        # Snap to multiple of 14 for ViT/DINOv2 patch safety.
        Hs = max(14, math.ceil(model_h / 14) * 14)
        Ws = max(14, math.ceil(model_w / 14) * 14)

        if (cur_h, cur_w) != (Hs, Ws):
            batch = F.interpolate(
                batch,
                size=(Hs, Ws),
                mode="bilinear",
                align_corners=False,
            )

        if is_cuda:
            try:
                batch = batch.contiguous(memory_format=torch.channels_last)
            except Exception:
                batch = batch.contiguous()
        else:
            batch = batch.contiguous()

        # Normalize on GPU.
        batch = (batch - mean_t) / std_t

        # ------------------------------------------------------------
        # Batched model inference
        # ------------------------------------------------------------
        # CUDA autocast only.
        # DirectML/privateuseone must not use CUDA autocast.
        autocast_ctx = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            )
            if is_cuda and dtype == torch.float16
            else nullcontext()
        )

        with autocast_ctx:
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

            # Do not resize adapter output back to source size here.
            # VD3D main postprocess already resizes depth to final video/image size.
            # Keeping adapter output at model resolution reduces GPU work and GPU->CPU transfer.

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
                # Return raw relative depth. Main VD3D pipeline normalizes later.
                depth_np = depth.detach().cpu().numpy().astype(np.float32, copy=False)

                return [
                    {"predicted_depth": np.ascontiguousarray(depth_np[i])}
                    for i in range(depth_np.shape[0])
                ]

        # Mixed output sizes. Resize each prediction individually on GPU.
        outputs = []

        for i in range(depth.shape[0]):
            d = depth[i]
            target_h, target_w = original_sizes[i]

            # Do not resize adapter output back to source size here.
            # Main VD3D postprocess handles final resizing.
            
            if is_metric:
                metric_depth = d.clamp(min=0)
                predicted_depth = torch.clamp(metric_depth / max_depth, 0.0, 1.0)

                outputs.append({
                    "predicted_depth": predicted_depth.detach().cpu(),
                    "metric_depth": metric_depth.detach().cpu(),
                })

            else:
                # Return raw relative depth. Main VD3D pipeline normalizes later.
                d_np = d.detach().cpu().numpy().astype(np.float32, copy=False)
                outputs.append({"predicted_depth": np.ascontiguousarray(d_np)})

        return outputs

    run._is_dav2 = True
    run._device = str(device)
    run._dtype = str(dtype)
    run._backend = "cuda" if is_cuda else ("directml" if is_directml else "cpu")

    return run, {
        "is_diffusion": False,
        "diffusion_kind": "depth",
        "is_dav2": True,
        "kind": "dav2",
        "device": str(device),
        "dtype": str(dtype),
        "backend": run._backend,
        "is_directml": bool(is_directml),
        "supports_fp16": bool(is_cuda),
    }
