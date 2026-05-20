# core/adapters/video_depth_anything_adapter.py

import re
import torch
import numpy as np
from PIL import Image


def _frame_to_np(x):
    """
    Convert input frame to np.ndarray with shape (H, W, 3), dtype uint8.

    VDA expects:
        frames_np: (T, H, W, 3)
        dtype: uint8
        color: RGB
    """

    # PIL image
    if isinstance(x, Image.Image):
        if x.mode != "RGB":
            x = x.convert("RGB")
        return np.array(x, dtype=np.uint8)

    # Torch tensor
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()

        # Remove batch dim if someone passed 1xCxHxW
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]

        arr = x.numpy()

        # CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        # Grayscale -> RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        # RGBA -> RGB
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]

        # Single channel -> RGB
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        # Float 0..1 -> uint8 0..255
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(arr)

    # Numpy or array-like
    arr = np.asarray(x)

    # Remove batch dim if someone passed 1xHxWxC or 1xCxHxW
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    # Grayscale -> RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # RGBA -> RGB
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    # Single channel -> RGB
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"VDA frame must resolve to HxWx3 RGB. Got shape: {arr.shape}")

    # Float 0..1 -> uint8 0..255
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def _pick_encoder_from_repo(repo_id: str) -> str:
    """
    Pick VDA encoder from repo/spec string.
    """
    r = (repo_id or "").lower()

    if "small" in r:
        return "vits"

    if "base" in r:
        return "vitb"

    # Large default
    return "vitl"


def _ckpt_filename(encoder: str, metric: bool) -> str:
    """
    Matches upstream VDA checkpoint naming style.
    """
    if metric:
        return f"metric_video_depth_anything_{encoder}.pth"

    return f"video_depth_anything_{encoder}.pth"


def _input_size_from_inference_size(inference_size=None, default=518) -> int:
    """
    Convert VisionDepth3D inference size into VDA input_size.

    VDA uses a single integer input_size, usually 518.

    Accepts:
        None
        (w, h)
        [w, h]
        "518x518"
        "910x518 (Depth Anything Widescreen)"
        "518 (Video Depth Anything Default)"
        "Original"

    Returns:
        int input_size
    """

    if inference_size is None:
        return int(default)

    # Handle strings
    if isinstance(inference_size, str):
        s = inference_size.strip()

        if not s or s.lower() == "original":
            return int(default)

        # Match 910x518, 518x518, 1024x576, etc.
        match = re.search(r"(\d+)\s*x\s*(\d+)", s)
        if match:
            w = int(match.group(1))
            h = int(match.group(2))

            # For VDA, use the larger side as the input size request,
            # then snap to stable model-friendly choices below.
            requested = max(w, h)
            return _snap_vda_input_size(requested)

        # Match "518 (Video Depth Anything Default)"
        match = re.search(r"^\s*(\d+)", s)
        if match:
            requested = int(match.group(1))
            return _snap_vda_input_size(requested)

        print(f"[VDA][WARN] Could not parse inference_size={inference_size!r}. Using {default}.")
        return int(default)

    # Handle tuple/list like (w, h)
    if isinstance(inference_size, (tuple, list)) and len(inference_size) >= 2:
        w = int(inference_size[0])
        h = int(inference_size[1])
        requested = max(w, h)
        return _snap_vda_input_size(requested)

    # Handle direct int/float
    try:
        requested = int(inference_size)
        return _snap_vda_input_size(requested)
    except Exception:
        print(f"[VDA][WARN] Could not parse inference_size={inference_size!r}. Using {default}.")
        return int(default)


def _snap_vda_input_size(requested: int) -> int:
    """
    Snap requested input size to safe VDA-style sizes.

    VDA official default is 518.
    Higher can be tested, but 518 is the safest baseline.

    Keeping this conservative avoids weird behavior from arbitrary sizes.
    """

    requested = int(requested)

    if requested <= 256:
        return 256

    if requested <= 392:
        return 392

    if requested <= 518:
        return 518

    # Allow higher testing, but keep multiples reasonable.
    # Many transformer/depth pipelines prefer multiples around 14.
    # 518 itself is 37 * 14.
    snapped = int(round(requested / 14) * 14)

    # Clamp to avoid accidentally sending huge sizes from 1080p/4K presets.
    # You can raise this later if you want experimental high-res VDA.
    snapped = max(518, min(snapped, 1036))

    return snapped


def load_vda_adapter(spec: str, cache_dir: str, use_fp16: bool = False):
    """
    spec example:
        "depth-anything/Video-Depth-Anything-Large"

    returns:
        callable, caps
    """

    from core.models.video_depth_anything.video_depth import VideoDepthAnything
    from huggingface_hub import hf_hub_download

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Keeping VDA fp32 by default is safer for temporal consistency.
    # Some depth models can get unstable or slightly different frame-to-frame in fp16.
    fp16_ok = bool(use_fp16 and device == "cuda")

    metric = "metric" in (spec or "").lower()

    encoder = _pick_encoder_from_repo(spec)
    ckpt_name = _ckpt_filename(encoder, metric)

    print(f"[VDA] Loading Video Depth Anything")
    print(f"[VDA] repo={spec}")
    print(f"[VDA] encoder={encoder}")
    print(f"[VDA] metric={metric}")
    print(f"[VDA] checkpoint={ckpt_name}")
    print(f"[VDA] device={device}")

    ckpt_path = hf_hub_download(
        repo_id=spec,
        filename=ckpt_name,
        cache_dir=cache_dir,
    )

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    vda = VideoDepthAnything(**model_configs[encoder], metric=metric)

    sd = torch.load(ckpt_path, map_location="cpu")
    vda.load_state_dict(sd, strict=True)
    vda.to(device).eval()

    # I recommend leaving this off for now.
    # If you later want speed testing, enable use_fp16 and test carefully.
    if fp16_ok:
        print("[VDA][WARN] fp16 requested, but keeping model in fp32 for stability.")
        # vda.half()

    @torch.no_grad()
    def vda_infer(images, inference_size=None, **kw):
        """
        Run Video Depth Anything.

        Important:
            VDA works best when it receives a real sequence, not tiny 4-frame or 8-frame batches.
            If the main depth runner sends short chunks, VDA may look like it skips, pulses, or resets.
        """

        if not isinstance(images, list):
            images = [images]

        frames = [_frame_to_np(im) for im in images]

        if len(frames) == 0:
            return []

        # Make sure all frames match shape.
        first_shape = frames[0].shape
        for i, frame in enumerate(frames):
            if frame.shape != first_shape:
                raise ValueError(
                    f"[VDA] All frames in a VDA sequence must have the same shape. "
                    f"Frame 0 shape={first_shape}, frame {i} shape={frame.shape}"
                )

        # VDA expects (T, H, W, 3), uint8, RGB.
        frames_np = np.stack(frames, axis=0)
        frames_np = np.ascontiguousarray(frames_np, dtype=np.uint8)

        # Use explicit input_size first, otherwise derive from inference_size.
        input_size = int(
            kw.get(
                "input_size",
                _input_size_from_inference_size(inference_size, default=518),
            )
        )

        # Keep original video timing unless caller explicitly overrides.
        target_fps = int(kw.get("target_fps", -1))

        # VDA infer_video_depth uses fp32 flag.
        # For stability, default false unless the caller passes fp32=True.
        fp32 = bool(kw.get("fp32", False))

        sequence_len = frames_np.shape[0]

        if sequence_len < 16:
            print(
                f"[VDA][WARN] Only received {sequence_len} frame(s). "
                f"VDA may look jumpy if processed in tiny chunks. "
                f"Try 32 to 64+ consecutive frames, ideally with overlap."
            )

        print(
            f"[VDA] Running Video Depth Anything | "
            f"frames={sequence_len} | "
            f"frame_shape={frames_np.shape[1]}x{frames_np.shape[2]} | "
            f"input_size={input_size} | "
            f"target_fps={target_fps} | "
            f"fp32={fp32}"
        )

        depths, fps_out = vda.infer_video_depth(
            frames_np,
            target_fps,
            input_size=input_size,
            device=device,
            fp32=fp32,
        )

        d = np.asarray(depths, dtype=np.float32)

        # Normalize output shape to (T, H, W)
        if d.ndim == 2:
            d = d[None, ...]

        if d.ndim != 3:
            raise ValueError(f"[VDA] Expected depth output shape T,H,W. Got {d.shape}")

        if d.shape[0] != sequence_len:
            print(
                f"[VDA][WARN] Output frame count differs from input. "
                f"input={sequence_len}, output={d.shape[0]}, fps_out={fps_out}"
            )

        return [
            {
                "predicted_depth": torch.from_numpy(d[i]).float()
            }
            for i in range(d.shape[0])
        ]

    caps = {
        "kind": "vda",
        "has_builtin_processor": True,
        "supports_multi_view": True,
        "supports_metric_models": True,
        "is_video_model": True,
        "prefers_sequence": True,

        # Helpful hints for your main runner if you decide to use them.
        "recommended_input_size": 518,
        "recommended_sequence_length": 64,
        "minimum_sequence_length": 32,
        "recommended_overlap": 16,
    }

    return vda_infer, caps