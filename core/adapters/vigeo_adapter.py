# core/adapters/vigeo_adapter.py

import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import os
import sys
from pathlib import Path

def _resolve_torch_device(device=None, use_directml: bool = False):
    """
    Resolve the torch device for ViGeo.

    Priority:
      1. Explicit device passed by VD3D
      2. DirectML if requested
      3. CUDA if available
      4. CPU fallback
    """
    if device is not None:
        if isinstance(device, str):
            return torch.device(device)
        return device

    if use_directml:
        try:
            import torch_directml

            if hasattr(torch_directml, "is_available"):
                try:
                    if not torch_directml.is_available():
                        raise RuntimeError("torch_directml.is_available() returned False")
                except Exception:
                    pass

            dml_device = torch_directml.device()

            # quick sanity check
            _ = torch.ones(1).to(dml_device).cpu()

            return dml_device

        except Exception as e:
            print(f"⚠️ ViGeo DirectML requested but unavailable: {e}")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def _device_backend_name(device) -> str:
    t = getattr(device, "type", str(device))

    if t == "cuda":
        if getattr(torch.version, "hip", None) is not None:
            return "rocm"
        return "cuda"

    if t == "privateuseone":
        return "directml"

    if t == "mps":
        return "mps"

    return "cpu"


def _round_to_multiple(value: int, multiple: int = 14):
    if multiple <= 1:
        return max(1, int(value))

    return max(multiple, int(round(float(value) / float(multiple)) * multiple))


def _resolve_inference_size(
    orig_w: int,
    orig_h: int,
    inference_size=None,
    default_max_side: int = 518,
    multiple: int = 14,
    force_exact_size: bool = False,
):
    """
    ViGeo should preserve aspect ratio by default.

    Important:
      VD3D's common 518x518 preset is square.
      For ViGeo, if the selected size is square and force_exact_size=False,
      treat it as max-side 518 instead of squeezing widescreen video into a square.

    Examples for 1918x800:
      inference_size=(518,518) -> about 518x210
      inference_size=(512,288) -> exact 512x288
      inference_size=None      -> about 518x210
    """
    orig_w = int(orig_w)
    orig_h = int(orig_h)

    if orig_w <= 0 or orig_h <= 0:
        return int(default_max_side), int(default_max_side)

    exact_w = None
    exact_h = None

    if inference_size is not None:
        try:
            exact_w, exact_h = int(inference_size[0]), int(inference_size[1])
        except Exception:
            exact_w, exact_h = None, None

    # If user explicitly forces exact, use exact width/height.
    if force_exact_size and exact_w and exact_h:
        new_w, new_h = exact_w, exact_h

    # If user selected a non-square preset like 512x288 or 910x518, respect it.
    elif exact_w and exact_h and exact_w != exact_h:
        new_w, new_h = exact_w, exact_h

    # If user selected a square preset like 518x518, use it as max-side.
    else:
        max_side = int(exact_w or exact_h or default_max_side or 518)

        if max_side <= 0:
            new_w, new_h = orig_w, orig_h
        else:
            scale = float(max_side) / float(max(orig_w, orig_h))
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))

    if multiple and multiple > 1:
        new_w = _round_to_multiple(new_w, multiple)
        new_h = _round_to_multiple(new_h, multiple)

    return int(new_w), int(new_h)


def _pil_to_rgb_tensor(img, infer_w: int, infer_h: int):
    """
    Converts PIL/np/tensor-ish frame to torch [3,H,W] float32 0..1 on CPU.
    """
    if isinstance(img, Image.Image):
        img = ImageOps.exif_transpose(img)

        if img.mode != "RGB":
            img = img.convert("RGB")

    elif isinstance(img, np.ndarray):
        arr = img

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)

        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise RuntimeError(f"Unexpected numpy image shape for ViGeo: {arr.shape}")

        if arr.shape[2] == 4:
            arr = arr[..., :3]

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr, mode="RGB")

    else:
        raise TypeError(f"Unsupported ViGeo input frame type: {type(img)}")

    orig_w, orig_h = img.size

    if (orig_w, orig_h) != (infer_w, infer_h):
        img = img.resize((infer_w, infer_h), Image.BICUBIC)

    arr = np.array(img, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    return tensor


def _depth_output_to_tensor(depth):
    """
    Converts ViGeo output depth_pred to torch [T,H,W] CPU float32.
    Expected common shape: [T,1,H,W]
    """
    if isinstance(depth, np.ndarray):
        depth_t = torch.from_numpy(depth)
    elif torch.is_tensor(depth):
        depth_t = depth.detach()
    else:
        depth_t = torch.as_tensor(depth)

    depth_t = depth_t.float().cpu()

    if depth_t.ndim == 4:
        # [T,1,H,W]
        if depth_t.shape[1] == 1:
            depth_t = depth_t[:, 0]

        # [T,H,W,1]
        elif depth_t.shape[-1] == 1:
            depth_t = depth_t[..., 0]

        else:
            raise RuntimeError(f"Unexpected ViGeo depth tensor shape: {tuple(depth_t.shape)}")

    elif depth_t.ndim == 3:
        # Already [T,H,W]
        pass

    elif depth_t.ndim == 2:
        # Single frame [H,W]
        depth_t = depth_t.unsqueeze(0)

    else:
        raise RuntimeError(f"Unexpected ViGeo depth tensor shape: {tuple(depth_t.shape)}")

    return depth_t.contiguous()


def _normalize_depth_minmax_fast(d: torch.Tensor) -> torch.Tensor:
    """
    Optional debug normalization.
    Normally VD3D render_depth should normalize after the adapter.
    """
    d = d.float()

    d_min = torch.amin(d)
    d_max = torch.amax(d)

    if float((d_max - d_min).detach().cpu()) < 1e-6:
        return torch.full_like(d, 0.5)

    return ((d - d_min) / (d_max - d_min + 1e-6)).clamp(0.0, 1.0)

def _import_vigeo(cache_dir=None):
    """
    Import ViGeo from:
      1. normal installed package
      2. VD3D_VIGEO_ROOT environment variable
      3. VD3D bundled external/core model folders
      4. local dev path
    """

    try:
        from vigeo import ViGeo
        return ViGeo
    except ModuleNotFoundError as first_error:
        original_error = first_error

    candidates = []

    env_root = os.environ.get("VD3D_VIGEO_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))

    if cache_dir:
        cache_root = Path(cache_dir)
        candidates.extend([
            cache_root / "ViGeo-main",
            cache_root / "ViGeo",
            cache_root / "vigeo",
        ])

    app_root = Path(__file__).resolve().parents[2]

    candidates.extend([
        app_root / "external" / "ViGeo-main",
        app_root / "external" / "ViGeo",
        app_root / "core" / "models" / "ViGeo-main",
        app_root / "core" / "models" / "ViGeo",
        app_root / "core" / "models",
        Path.cwd() / "ViGeo-main",

        # Dev fallback for your current machine.
        Path(r"C:\Users\johna\build\ViGeo-main"),
    ])

    tried = []

    for candidate in candidates:
        if not candidate:
            continue

        candidate = Path(candidate)

        if not candidate.exists():
            continue

        tried.append(str(candidate))

        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

        try:
            from vigeo import ViGeo
            print(f"[ViGeo] Imported from: {candidate}")
            return ViGeo
        except ModuleNotFoundError:
            continue

    raise RuntimeError(
        "Could not import ViGeo. Install it with:\n"
        "  cd C:\\Users\\johna\\build\\ViGeo-main\n"
        "  pip install -e .\n\n"
        "Or set:\n"
        "  set VD3D_VIGEO_ROOT=C:\\Users\\johna\\build\\ViGeo-main\n\n"
        f"Tried paths: {tried}\n"
        f"Original error: {original_error}"
    )

def load_vigeo_adapter(
    spec: str,
    cache_dir: str,
    use_fp16: bool = False,
    use_directml: bool = False,
    device=None,
):
    """
    ViGeo adapter for VD3D.

    spec example:
        "pkqbajng/ViGeo"

    Returns:
        callable, caps

    Notes:
      - The adapter does NOT write videos.
      - VD3D render_depth.py handles video decode, batching, normalization,
        output encoding, progress, and cancel.
      - ViGeo raw output tested as black-near / white-far.
        By default, this adapter flips raw depth with -depth before VD3D
        normalizes it, so VD3D gets white-near / black-far with invert OFF.
    """
    ViGeo = _import_vigeo(cache_dir=cache_dir)

    device = _resolve_torch_device(device=device, use_directml=use_directml)

    device_type = getattr(device, "type", str(device))
    is_cuda = device_type == "cuda"
    is_directml = device_type == "privateuseone"
    backend = _device_backend_name(device)

    # CUDA tuning
    if is_cuda:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        except Exception:
            pass

    # Keep FP16 CUDA-only.
    # Use autocast, but keep weights in their normal loaded dtype for stability.
    use_amp = bool(use_fp16 and is_cuda)

    print(
        f"[ViGeo] Loading {spec} on device={device} "
        f"backend={backend} fp16_autocast={use_amp}"
    )

    try:
        model = ViGeo.from_pretrained(spec, cache_dir=cache_dir)
    except TypeError:
        # Some from_pretrained wrappers do not accept cache_dir.
        model = ViGeo.from_pretrained(spec)

    try:
        model.to(device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to move ViGeo model to {device} backend={backend}. "
            f"If this is DirectML, ViGeo may use unsupported torch-directml ops. "
            f"Original error: {e}"
        ) from e

    model.eval()

    try:
        model.device = device
    except Exception:
        pass

    @torch.inference_mode()
    def vigeo_infer(images, inference_size=None, **kw):
        if not isinstance(images, list):
            images = [images]

        if not images:
            return []

        # ViGeo mode from your standalone script.
        mode = kw.get("mode", "offline")

        # Use selected inference size, but preserve aspect on square presets.
        force_exact_size = bool(kw.get("force_exact_size", False))
        default_max_side = int(kw.get("input_size", kw.get("max_side", 518)) or 518)
        multiple = int(kw.get("multiple", 14) or 14)

        first = images[0]

        if isinstance(first, Image.Image):
            first = ImageOps.exif_transpose(first)
            orig_w, orig_h = first.size
        elif isinstance(first, np.ndarray):
            orig_h, orig_w = first.shape[:2]
        else:
            raise TypeError(f"Unsupported ViGeo first frame type: {type(first)}")

        infer_w, infer_h = _resolve_inference_size(
            orig_w=orig_w,
            orig_h=orig_h,
            inference_size=inference_size,
            default_max_side=default_max_side,
            multiple=multiple,
            force_exact_size=force_exact_size,
        )

        tensors = [
            _pil_to_rgb_tensor(img, infer_w=infer_w, infer_h=infer_h)
            for img in images
        ]

        batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)

        print(f"[ViGeo] infer frames={len(images)} tensor={tuple(batch.shape)} mode={mode}")

        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )

        with ctx:
            output = model.infer(batch, mode=mode)

        if not isinstance(output, dict) or "depth_pred" not in output:
            raise RuntimeError(
                f"ViGeo output did not contain 'depth_pred'. Got keys: "
                f"{list(output.keys()) if isinstance(output, dict) else type(output)}"
            )

        depths_t = _depth_output_to_tensor(output["depth_pred"])

        del batch, output, tensors

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Your standalone test showed ViGeo raw output as black near / white far.
        # Returning -depth flips the ordering without doing per-frame normalization.
        # Then VD3D's main normalizer can still do fixed/temporal normalization.
        invert_raw = bool(kw.get("vigeo_invert_raw", True))

        if invert_raw:
            depths_t = -depths_t

        adapter_normalize = bool(kw.get("adapter_normalize", False))

        outputs = []

        if depths_t.shape[0] != len(images):
            raise RuntimeError(
                f"ViGeo returned {depths_t.shape[0]} depth frames for "
                f"{len(images)} input frames."
            )

        for i in range(depths_t.shape[0]):
            d = depths_t[i].float().cpu()

            if adapter_normalize:
                d = _normalize_depth_minmax_fast(d)

            outputs.append({"predicted_depth": d})

        return outputs

    vigeo_infer._is_vigeo = True
    vigeo_infer._device = str(device)
    vigeo_infer._backend = backend
    vigeo_infer._is_directml = bool(is_directml)

    caps = {
        "kind": "vigeo",
        "has_builtin_processor": True,
        "supports_multi_frame": True,
        "supports_fp16": bool(is_cuda),
        "device": str(device),
        "backend": backend,
        "is_directml": bool(is_directml),
        "skip_warmup": True,
    }

    return vigeo_infer, caps