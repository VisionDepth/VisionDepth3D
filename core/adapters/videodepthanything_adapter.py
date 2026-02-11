# core/adapters/video_depth_anything_adapter.py
import torch
import numpy as np
from PIL import Image

def _frame_to_np(x):
    # Convert PIL → np(H,W,3) uint8 (what VDA expects)
    if isinstance(x, Image.Image):
        if x.mode != "RGB":
            x = x.convert("RGB")
        return np.array(x)

    # Torch tensor → numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        return x

    # Already numpy
    return x


def _pick_encoder_from_repo(repo_id: str) -> str:
    r = (repo_id or "").lower()
    # simple heuristic
    if "small" in r:
        return "vits"
    if "base" in r:
        return "vitb"
    return "vitl"  # Large default

def _ckpt_filename(encoder: str, metric: bool) -> str:
    # matches upstream naming style
    if metric:
        return f"metric_video_depth_anything_{encoder}.pth"
    return f"video_depth_anything_{encoder}.pth"

def _input_size_from_inference_size(inference_size, default=518) -> int:
    # VDA is square input_size in their CLI. Keep it stable.
    if inference_size is None:
        return int(default)
    w, h = inference_size
    m = max(int(w), int(h))
    # clamp to something sane
    return 518 if m >= 518 else 392 if m >= 392 else 256

def load_vda_adapter(spec: str, cache_dir: str, use_fp16: bool = False):
    """
    spec example:
      - "depth-anything/Video-Depth-Anything-Large"
    returns: (callable, caps)
    """
    # your vendored source (or installed package) should expose this
    from core.models.video_depth_anything.video_depth import VideoDepthAnything

    from huggingface_hub import hf_hub_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16_ok = (use_fp16 and device == "cuda")

    metric = ("metric" in (spec or "").lower())

    encoder = _pick_encoder_from_repo(spec)
    ckpt_name = _ckpt_filename(encoder, metric)

    ckpt_path = hf_hub_download(
        repo_id=spec,
        filename=ckpt_name,
        cache_dir=cache_dir,
    )

    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    vda = VideoDepthAnything(**model_configs[encoder], metric=metric)
    sd = torch.load(ckpt_path, map_location="cpu")
    vda.load_state_dict(sd, strict=True)
    vda.to(device).eval()

#    if fp16_ok:
#        vda.half()

    @torch.no_grad()
    def vda_infer(images, inference_size=None, **kw):
        if not isinstance(images, list):
            images = [images]

        frames = []
        for im in images:
            if isinstance(im, Image.Image):
                if im.mode != "RGB":
                    im = im.convert("RGB")
                arr = np.array(im, dtype=np.uint8)  # (H,W,3)
            else:
                arr = np.asarray(im)
                # if someone passed a torch tensor, you may want:
                # if isinstance(im, torch.Tensor): arr = im.detach().cpu().numpy()
            frames.append(arr)

        # ✅ VDA expects (T,H,W,3) array (not list)
        frames_np = np.stack(frames, axis=0)

        input_size = int(kw.get("input_size", 518))
        target_fps = int(kw.get("target_fps", -1))
        fp32 = bool(kw.get("fp32", False))
        

        depths, fps_out = vda.infer_video_depth(
            frames_np,
            target_fps,
            input_size=input_size,
            device=device,
            fp32=fp32,
        )

        # return list of {"predicted_depth": tensor} per frame
        d = np.asarray(depths, dtype=np.float32)
        if d.ndim == 2:
            d = d[None, ...]
        return [{"predicted_depth": torch.from_numpy(d[i]).float()} for i in range(d.shape[0])]

    caps = {
        "kind": "vda",
        "has_builtin_processor": True,
        "supports_multi_view": True,     # sequence model
        "supports_metric_models": True,
        "is_video_model": True,
        "prefers_sequence": True,
    }
    return vda_infer, caps
