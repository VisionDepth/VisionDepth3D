# core/adapters/depthanything3_adapter.py
import torch
import numpy as np
from PIL import Image
from contextlib import nullcontext

def _process_res_from_inference_size(inference_size, default=504):
    # DA3 default in their API is 504
    if inference_size is None:
        return default
    w, h = inference_size
    max_dim = max(int(w), int(h))
    if max_dim <= 512:  return 512
    if max_dim <= 640:  return 640
    if max_dim <= 768:  return 768
    if max_dim <= 896:  return 896
    if max_dim <= 1024: return 1024
    if max_dim <= 1280: return 1280
    return 1536

def _normalize_depth_percentile(d: torch.Tensor, q_lo=0.02, q_hi=0.98) -> torch.Tensor:
    # d is [H,W]
    flat = d.flatten()
    lo = torch.quantile(flat, q_lo)
    hi = torch.quantile(flat, q_hi)
    d = (d - lo) / (hi - lo + 1e-6)
    return d.clamp(0, 1)

def load_da3_adapter(spec: str, cache_dir: str, use_fp16: bool = False):
    from core.models.depth_anything_3.api import DepthAnything3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda" and use_fp16)

    model = DepthAnything3.from_pretrained(spec, cache_dir=cache_dir)
    # Move the WHOLE wrapper so processors + model are consistent
    model.to(device)
    model.eval()

    try:
        model.device = torch.device(device)
    except Exception:
        pass

    @torch.no_grad()
    def da3_infer(images, inference_size=None, **kw):
        if not isinstance(images, list):
            images = [images]

        # Prefer DA3 defaults unless you override
        default_pr = int(kw.get("process_res", 504))
        process_res = _process_res_from_inference_size(inference_size, default=default_pr)
        process_res_method = kw.get("process_res_method", "upper_bound_resize")

        # If you want a cap, make it optional and higher during quality tests
        cap = kw.get("process_res_cap", None)  # e.g. 1024
        if cap is not None:
            process_res = min(int(cap), int(process_res))

        cleaned = []
        for im in images:
            if isinstance(im, Image.Image):
                if im.mode != "RGB":
                    im = im.convert("RGB")
                cleaned.append(im)
            else:
                cleaned.append(im)

        ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

        # Run as batch (DA3 supports list input)
        with ctx:
            pred = model.inference(
                cleaned,
                process_res=process_res,
                process_res_method=process_res_method,
                export_dir=None,
                export_format="mini_npz",
            )

        depths = getattr(pred, "depth", None)
        if depths is None:
            raise RuntimeError("DA3 returned Prediction with no .depth")

        # Convert to torch [N,H,W]
        if isinstance(depths, np.ndarray):
            depths_t = torch.from_numpy(depths).float()
        else:
            depths_t = depths.detach().float().cpu()

        if depths_t.ndim == 2:
            depths_t = depths_t.unsqueeze(0)

        outputs = []
        for k in range(depths_t.shape[0]):
            d = depths_t[k]
            d = _normalize_depth_percentile(d, q_lo=0.02, q_hi=0.98)
            outputs.append({"predicted_depth": d})

        # If mismatch count, do per-image using SAME normalization
        if len(outputs) != len(cleaned):
            outputs = []
            for im in cleaned:
                with ctx:
                    pred1 = model.inference(
                        [im],
                        process_res=process_res,
                        process_res_method=process_res_method,
                        export_dir=None,
                        export_format="mini_npz",
                    )
                d1 = pred1.depth[0]
                if isinstance(d1, np.ndarray):
                    d1 = torch.from_numpy(d1).float()
                else:
                    d1 = d1.detach().float().cpu()
                d1 = _normalize_depth_percentile(d1, q_lo=0.02, q_hi=0.98)
                outputs.append({"predicted_depth": d1})

        return outputs

    caps = {
        "kind": "da3",
        "has_builtin_processor": True,
        "supports_multi_view": True,
        "supports_metric_models": True,
    }
    return da3_infer, caps
