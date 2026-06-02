# core/adapters/depthanything3_adapter.py
import torch
import numpy as np
from PIL import Image, ImageOps
from contextlib import nullcontext


def _process_res_from_inference_size(inference_size, default=504):
    """
    DA3 uses a scalar process_res.

    Old behavior rounded 518 -> 640, which made "518x518" much slower
    than expected. This version respects the user's selected inference
    size more directly.

    Examples:
      None      -> 504
      504x504   -> 504
      518x518   -> 518
      910x518   -> 910
      1280x720  -> 1280
    """
    if inference_size is None:
        return int(default)

    try:
        w, h = inference_size
        process_res = max(int(w), int(h))
    except Exception:
        return int(default)

    # Keep sane bounds. DA3 can be expensive at very large process_res.
    process_res = max(64, process_res)
    return int(process_res)


def _normalize_depth_minmax_fast(d: torch.Tensor) -> torch.Tensor:
    """
    Optional fast min/max normalization.

    Normally the main VD3D depth pipeline normalizes after the adapter,
    so the adapter should not do expensive percentile normalization by default.
    """
    d = d.float()

    d_min = torch.amin(d)
    d_max = torch.amax(d)

    if float((d_max - d_min).detach().cpu()) < 1e-6:
        return torch.full_like(d, 0.5)

    return ((d - d_min) / (d_max - d_min + 1e-6)).clamp(0.0, 1.0)


def _depths_to_tensor(depths) -> torch.Tensor:
    """
    Converts DA3 Prediction.depth into torch [N,H,W] on CPU.

    Supports:
      - np.ndarray [H,W] or [N,H,W]
      - torch.Tensor [H,W], [N,H,W], [N,1,H,W]
      - list/tuple of arrays/tensors
    """
    if isinstance(depths, (list, tuple)):
        converted = []
        for d in depths:
            if isinstance(d, np.ndarray):
                dt = torch.from_numpy(np.asarray(d))
            elif torch.is_tensor(d):
                dt = d.detach()
            else:
                dt = torch.as_tensor(d)

            if dt.ndim == 3 and dt.shape[0] == 1:
                dt = dt.squeeze(0)

            converted.append(dt.float().cpu())

        if not converted:
            return torch.empty((0, 1, 1), dtype=torch.float32)

        return torch.stack(converted, dim=0)

    if isinstance(depths, np.ndarray):
        depths_t = torch.from_numpy(np.asarray(depths)).float()
    elif torch.is_tensor(depths):
        depths_t = depths.detach().float().cpu()
    else:
        depths_t = torch.as_tensor(depths).float()

    if depths_t.ndim == 2:
        depths_t = depths_t.unsqueeze(0)

    elif depths_t.ndim == 4:
        # Common shape: [N,1,H,W]
        if depths_t.shape[1] == 1:
            depths_t = depths_t[:, 0]

        # Less common shape: [N,H,W,1]
        elif depths_t.shape[-1] == 1:
            depths_t = depths_t[..., 0]

        else:
            raise RuntimeError(f"Unexpected DA3 depth tensor shape: {tuple(depths_t.shape)}")

    if depths_t.ndim != 3:
        raise RuntimeError(f"Unexpected DA3 depth tensor shape after conversion: {tuple(depths_t.shape)}")

    return depths_t.contiguous()

def _resolve_torch_device(device=None, use_directml: bool = False):
    """
    Resolve the torch device for DA3.

    Priority:
      1. Explicit device passed by main app
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
                    # Some torch-directml versions may not behave consistently here.
                    pass

            dml_device = torch_directml.device()

            # Quick sanity check.
            _ = torch.ones(1).to(dml_device).cpu()

            return dml_device

        except Exception as e:
            print(f"⚠️ DA3 DirectML requested but unavailable: {e}")

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

def load_da3_adapter(
    spec: str,
    cache_dir: str,
    use_fp16: bool = False,
    use_directml: bool = False,
    device=None,
):
    from core.models.depth_anything_3.api import DepthAnything3

    device = _resolve_torch_device(device=device, use_directml=use_directml)

    device_type = getattr(device, "type", str(device))
    is_cuda = device_type == "cuda"
    is_directml = device_type == "privateuseone"

    # Keep FP16 CUDA-only.
    # DirectML FP16 can be unstable depending on GPU/driver/op support.
    use_amp = bool(is_cuda and use_fp16)

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

    print(
        f"[DA3] Loading {spec} on device={device} "
        f"backend={_device_backend_name(device)} fp16={use_amp}"
    )

    model = DepthAnything3.from_pretrained(spec, cache_dir=cache_dir)

    # Move the whole wrapper so processors + model are consistent.
    #
    # CUDA: normal .to(cuda)
    # DirectML: .to(privateuseone:0)
    # CPU: .to(cpu)
    try:
        model.to(device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to move DA3 model to {device} "
            f"backend={_device_backend_name(device)}. "
            f"If this is DirectML, this DA3 build may use unsupported ops. "
            f"Original error: {e}"
        ) from e

    model.eval()

    try:
        model.device = device
    except Exception:
        pass

    @torch.inference_mode()
    def da3_infer(images, inference_size=None, **kw):
        if not isinstance(images, list):
            images = [images]

        if not images:
            return []

        # Explicit process_res kw wins.
        # Otherwise respect selected inference_size directly.
        if "process_res" in kw and kw.get("process_res") is not None:
            process_res = int(kw.get("process_res"))
        else:
            process_res = _process_res_from_inference_size(inference_size, default=504)

        process_res_method = kw.get("process_res_method", "upper_bound_resize")

        # Optional safety cap, useful for low VRAM.
        # Example caller kw: process_res_cap=1024
        cap = kw.get("process_res_cap", None)
        if cap is not None:
            process_res = min(int(cap), int(process_res))

        # By default, do NOT normalize in adapter.
        # VD3D main video pipeline normalizes later using temporal/fixed/fast normalizers.
        adapter_normalize = bool(kw.get("adapter_normalize", False))

        cleaned = []

        for im in images:
            if isinstance(im, Image.Image):
                # EXIF transpose avoids rotated-phone-image surprises.
                im = ImageOps.exif_transpose(im)

                if im.mode != "RGB":
                    im = im.convert("RGB")

                cleaned.append(im)

            else:
                # DA3 API normally expects PIL/list input.
                # Keep non-PIL input support in case future callers pass arrays/tensors.
                cleaned.append(im)

        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )

        with ctx:
            pred = model.inference(
                cleaned,
                process_res=int(process_res),
                process_res_method=process_res_method,
                export_dir=None,
                export_format="mini_npz",
            )

        depths = getattr(pred, "depth", None)

        if depths is None:
            raise RuntimeError("DA3 returned Prediction with no .depth")

        depths_t = _depths_to_tensor(depths)

        outputs = []

        # Normal successful batched path.
        if depths_t.shape[0] == len(cleaned):
            for k in range(depths_t.shape[0]):
                d = depths_t[k]

                if adapter_normalize:
                    d = _normalize_depth_minmax_fast(d)

                outputs.append({"predicted_depth": d})

            return outputs

        # Fallback: DA3 returned a mismatch count.
        # Re-run per image, but keep the same optimized/no-percentile behavior.
        outputs = []

        for im in cleaned:
            with ctx:
                pred1 = model.inference(
                    [im],
                    process_res=int(process_res),
                    process_res_method=process_res_method,
                    export_dir=None,
                    export_format="mini_npz",
                )

            d1 = getattr(pred1, "depth", None)

            if d1 is None:
                raise RuntimeError("DA3 per-image fallback returned no .depth")

            d1_t = _depths_to_tensor(d1)

            if d1_t.shape[0] < 1:
                raise RuntimeError("DA3 per-image fallback returned empty depth output")

            d = d1_t[0]

            if adapter_normalize:
                d = _normalize_depth_minmax_fast(d)

            outputs.append({"predicted_depth": d})

        return outputs

    da3_infer._is_da3 = True

    da3_infer._device = str(device)
    da3_infer._backend = _device_backend_name(device)
    da3_infer._is_directml = bool(is_directml)

    caps = {
        "kind": "da3",
        "has_builtin_processor": True,
        "supports_multi_view": True,
        "supports_metric_models": True,
        "supports_fp16": bool(is_cuda),
        "device": str(device),
        "backend": _device_backend_name(device),
        "is_directml": bool(is_directml),
    }

    return da3_infer, caps
