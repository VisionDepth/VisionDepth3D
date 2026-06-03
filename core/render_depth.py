import os

# Must be set before importing torch.
# Helps reduce CUDA allocator fragmentation during large video-depth windows.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128"
)

import re
import sys
import time
import uuid
import json
import threading
import subprocess
from tkinter import filedialog, messagebox
import gc
import inspect

# --- add these near your imports ---
from pathlib import Path
import shutil

import traceback, datetime

import numpy as np
import torch
import cv2
import onnxruntime as ort
import matplotlib.cm as cm
from PIL import Image, ImageTk, ImageOps
import platform
from core.ffmpeg_utils import require_tool
from core.debug_flags import debug_print, is_debug_enabled

def hidden_subprocess_kwargs():
    """
    Prevents ffmpeg/ffprobe subprocess console windows from flashing
    in PyInstaller windowed builds on Windows.
    """
    if platform.system().lower() != "windows":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0

    return {
        "startupinfo": startupinfo,
        "creationflags": subprocess.CREATE_NO_WINDOW,
    }
    
# Device setup
# =========================
# Force Hugging Face caches into VD3D /weights
# Must be set BEFORE importing transformers/diffusers
# =========================

def _vd3d_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_VD3D_WEIGHTS = os.path.join(_vd3d_base_dir(), "weights")
os.makedirs(_VD3D_WEIGHTS, exist_ok=True)

# Hugging Face + Transformers + Datasets caches
os.environ.setdefault("HF_HOME", _VD3D_WEIGHTS)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_VD3D_WEIGHTS, "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_VD3D_WEIGHTS, "datasets"))

# Optional: silence symlink warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from transformers import AutoProcessor, AutoModelForDepthEstimation, AutoImageProcessor

#from core.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from core.models.depth_anything_v2.dpt import DepthAnythingV2

global pipe
pipe = None
pipe_type = None
pipe_lock = threading.RLock()

suspend_flag = threading.Event()
cancel_flag = threading.Event()
cancel_requested = threading.Event()
global_session_start_time = None
current_warmup_session = {"id": None}
torch.set_grad_enabled(False)

# near other globals
PIPE_EXTRA_ARGS = {}

def get_active_pipe_snapshot():
    """
    Thread-safe snapshot of the active model callable, type, and extra args.
    """
    with pipe_lock:
        extra = PIPE_EXTRA_ARGS.copy() if isinstance(PIPE_EXTRA_ARGS, dict) else {}
        return pipe, pipe_type, extra


def set_active_pipe(new_pipe, new_pipe_type):
    """
    Thread-safe active model replacement.
    """
    global pipe, pipe_type
    with pipe_lock:
        pipe = new_pipe
        pipe_type = new_pipe_type


def clear_active_pipe():
    """
    Thread-safe active model clear.
    """
    set_active_pipe(None, None)

FFMPEG_CODEC_MAP = {
    # Software (CPU) Encoders
    "H.264 / AVC (libx264 - CPU)": "libx264",
    "H.265 / HEVC (libx265 - CPU)": "libx265",
    "AV1 (libaom - CPU)": "libaom-av1",
    "AV1 (SVT - CPU, faster)": "libsvtav1",
    "MPEG-4 (mp4v - CPU)": "mp4v",
    "XviD (AVI - CPU)": "XVID",
    "DivX (AVI - CPU)": "DIVX",

    # NVIDIA NVENC
    "H.264 / AVC (NVENC - NVIDIA GPU)": "h264_nvenc",
    "H.265 / HEVC (NVENC - NVIDIA GPU)": "hevc_nvenc",
    "AV1 (NVENC - NVIDIA RTX 40+ GPU)": "av1_nvenc",

    # AMD AMF
    "H.264 / AVC (AMF - AMD GPU)": "h264_amf",
    "H.265 / HEVC (AMF - AMD GPU)": "hevc_amf",
    "AV1 (AMF - AMD RDNA3+)": "av1_amf",

    # Intel QSV
    "H.264 / AVC (QSV - Intel GPU)": "h264_qsv",
    "H.265 / HEVC (QSV - Intel GPU)": "hevc_qsv",
    "VP9 (QSV - Intel GPU)": "vp9_qsv",
    "AV1 (QSV - Intel ARC / Gen11+)": "av1_qsv",
}

# ===== Codec helpers =====

def is_opencv_safe_fourcc(ffmpeg_codec: str) -> bool:
    return ffmpeg_codec in ("mp4v", "XVID", "DIVX")
    
def start_ffmpeg_writer(output_path, fps, w, h, ffmpeg_codec):
    ffmpeg_exe = require_tool("ffmpeg")

    cmd = [
        ffmpeg_exe, "-y",
        "-hide_banner", "-loglevel", "error",

        # Bigger queue so stdin bursts do not stall as easily
        "-thread_queue_size", "512",

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(float(fps)),
        "-i", "-",
        "-an",
        "-c:v", ffmpeg_codec,
    ]

    # Codec specific speed tuning
    if ffmpeg_codec in ("libx264", "libx265"):
        cmd += [
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
        ]

    elif ffmpeg_codec in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
        # NVENC speed presets: p1 fastest .. p7 best quality
        cmd += [
            "-preset", "p2",
            "-rc", "vbr",
            "-cq", "19",
            "-pix_fmt", "yuv420p",
        ]

    elif ffmpeg_codec in ("h264_amf", "hevc_amf", "av1_amf"):
        # AMF tuning for AMD GPUs
        cmd += [
            "-quality", "speed",
            "-rc", "vbr_peak",
            "-qvbr_quality_level", "19",
            "-pix_fmt", "yuv420p",
        ]

    else:
        # Safe default
        cmd += ["-pix_fmt", "yuv420p"]
        
    # yuv420p encoders require even dimensions. Pad by at most 1px if needed.
    if int(w) % 2 or int(h) % 2:
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        
    cmd += [output_path]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=10**8,
        **hidden_subprocess_kwargs(),
    )

    # Drain stderr in the background so FFmpeg cannot block if it emits logs/errors.
    # Keep only a bounded amount for diagnostics.
    proc._stderr_chunks = []
    proc._stderr_bytes = 0
    proc._stderr_max_bytes = 1_000_000  # 1 MB is enough for useful error output.

    def _drain_stderr():
        try:
            while True:
                chunk = proc.stderr.read(65536)
                if not chunk:
                    break

                remaining = proc._stderr_max_bytes - proc._stderr_bytes
                if remaining > 0:
                    kept = chunk[:remaining]
                    proc._stderr_chunks.append(kept)
                    proc._stderr_bytes += len(kept)
        except Exception:
            pass

    proc._stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    proc._stderr_thread.start()

    return proc

def request_depth_cancel():
    """
    Called by the global Cancel button when Depth pipeline is running.
    """
    cancel_flag.set()          # keep for legacy callers
    cancel_requested.set()     # this is what the depth loops actually check
    suspend_flag.clear()       # if paused, unpause so loops can exit


def request_depth_pause():
    """
    Called by a Pause button for Depth.
    """
    suspend_flag.set()


def request_depth_resume():
    """
    Called by a Resume button for Depth.
    """
    suspend_flag.clear()


def wait_if_paused(status_label=None):
    """
    Block worker loops while paused, but still allow Cancel.
    Call this inside long loops in background threads.
    """
    while suspend_flag.is_set() and not cancel_requested.is_set():
        if status_label is not None:
            try:
                status_label.after(
                    0,
                    lambda: status_label.config(text="⏸ Paused. Press Resume to continue.")
                )
            except Exception:
                pass
        time.sleep(0.2)

def set_pipe_extra_args(d: dict | None):
    global PIPE_EXTRA_ARGS
    with pipe_lock:
        PIPE_EXTRA_ARGS = d or {}


def _try_get_directml_device():
    """
    Returns a usable torch-directml device, or None.

    torch-directml devices expose as type='privateuseone'.
    """
    try:
        import torch_directml

        if hasattr(torch_directml, "is_available"):
            try:
                if not torch_directml.is_available():
                    return None
            except Exception:
                pass

        dml_device = torch_directml.device()

        # Quick sanity test. Catches broken DirectML installs early.
        _ = torch.ones(1).to(dml_device).cpu()

        return dml_device

    except Exception as e:
        print(f"Depth Estimation: DirectML not available: {e}")
        return None


def pick_torch_device():
    global torch_backend_name

    # Optional override.
    # Useful if the machine has CUDA but you want to test DirectML:
    #   set VD3D_FORCE_DIRECTML=1
    force_dml = os.environ.get("VD3D_FORCE_DIRECTML", "").strip().lower() in {
        "1", "true", "yes", "on"
    }

    if force_dml:
        dml_device = _try_get_directml_device()
        if dml_device is not None:
            torch_backend_name = "DirectML (AMD/Intel)"
            print(f"Depth Estimation: DirectML forced: {dml_device}")
            return dml_device
        print("⚠️ VD3D_FORCE_DIRECTML was set, but DirectML was not usable. Falling back.")

    # CUDA / ROCm-HIP exposed as cuda.
    if torch.cuda.is_available():
        try:
            hip_ver = getattr(torch.version, "hip", None)
            if hip_ver is not None:
                torch_backend_name = "ROCm (AMD)"
                print(f"Depth Estimation: ROCm detected! HIP runtime version: {hip_ver}")
                return torch.device("cuda")
        except Exception:
            pass

        torch_backend_name = "CUDA (NVIDIA)"
        print("Depth Estimation: CUDA")
        return torch.device("cuda")

    # DirectML fallback.
    dml_device = _try_get_directml_device()
    if dml_device is not None:
        torch_backend_name = "DirectML (AMD/Intel)"
        print(f"Depth Estimation: DirectML detected: {dml_device}")
        return dml_device

    # Apple Metal.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch_backend_name = "Metal (Apple)"
        print("Depth Estimation: MPS detected")
        return torch.device("mps")

    torch_backend_name = "CPU"
    print("⚠️ No GPU detected — using CPU")
    return torch.device("cpu")


def torch_device_type(device=None) -> str:
    d = device if device is not None else globals().get("torch_device", None)
    return str(getattr(d, "type", "cpu"))


def is_cuda_device(device=None):
    return torch_device_type(device) == "cuda"


def is_directml_device(device=None):
    return torch_device_type(device) == "privateuseone"


def is_mps_device(device=None):
    return torch_device_type(device) == "mps"


def is_torch_accelerator_device(device=None):
    return is_cuda_device(device) or is_directml_device(device) or is_mps_device(device)


def can_use_fp16_on_device(device=None):
    # Keep FP16 CUDA-only for now.
    # DirectML FP16 is device/op/driver dependent and can cause unsupported-op
    # errors or slow fallback paths.
    return is_cuda_device(device)


def device_display_name():
    name = globals().get("torch_backend_name", None)
    if name:
        return name

    t = torch_device_type()
    if t == "cuda":
        if getattr(torch.version, "hip", None) is not None:
            return "ROCm (AMD)"
        return "CUDA (NVIDIA)"
    if t == "privateuseone":
        return "DirectML (AMD/Intel)"
    if t == "mps":
        return "Metal (Apple)"
    return "CPU"


def offload_available():
    # accelerate CPU/GPU offload is CUDA-centric here.
    # Do not enable this for DirectML.
    try:
        import accelerate
        return is_cuda_device()
    except ImportError:
        return False

# --- tiling config ---
USE_TILED_DEPTH = False   # <- flip off to revert to old behavior
TILE_SIZE       = 512
TILE_PAD        = 32
TILE_DEBUG      = False   # set True to print tile debug info

assert TILE_SIZE > 2*TILE_PAD, "TILE_SIZE must be larger than 2*TILE_PAD"

_EXPECTED_WEIGHT_FILENAMES = {
    "pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt", "flax_model.msgpack"
}

# Known Hugging Face Depth Anything V2 repos -> optimized safetensors adapter specs.
#
# This makes normal menu entries like:
#   "Depth Anything v2 Large": "depth-anything/Depth-Anything-V2-Large-hf"
#
# automatically load through core/adapters/depthanything_adapter.py instead of the
# slower generic Hugging Face Transformers path.
#
# CUDA + FP16 checkbox:
#   Large -> vitl_fp16
#
# CUDA without FP16 / DirectML / CPU:
#   Large -> vitl_fp32
#
# Metric variants currently map to FP32 safetensors.
_DAV2_HF_REPO_TO_ADAPTER_SPEC = {
    "depth-anything/depth-anything-v2-small-hf": {
        "fp16": "vits_fp16",
        "fp32": "vits_fp32",
    },
    "depth-anything/depth-anything-v2-base-hf": {
        "fp16": "vitb_fp16",
        "fp32": "vitb_fp32",
    },
    "depth-anything/depth-anything-v2-large-hf": {
        "fp16": "vitl_fp16",
        "fp32": "vitl_fp32",
    },

    # Metric DA-V2 Large variants.
    # Kijai safetensors available here are FP32.
    "depth-anything/depth-anything-v2-metric-indoor-large-hf": {
        "fp16": "metric_hypersim_vitl_fp32",
        "fp32": "metric_hypersim_vitl_fp32",
    },
    "depth-anything/depth-anything-v2-metric-outdoor-large-hf": {
        "fp16": "metric_vkitti_vitl_fp32",
        "fp32": "metric_vkitti_vitl_fp32",
    },
}

torch_backend_name = "CPU"
torch_device = pick_torch_device()

def active_torch_dtype(use_fp16: bool = False):
    """
    Runtime dtype for active torch backend.
    FP16 is CUDA-only for now.
    """
    return torch.float16 if (bool(use_fp16) and can_use_fp16_on_device(torch_device)) else torch.float32


def model_load_dtype(use_fp16: bool = False):
    """
    dtype passed to from_pretrained().

    DirectML should load as default/FP32.
    CUDA can load FP16 when requested.
    """
    return torch.float16 if (bool(use_fp16) and is_cuda_device(torch_device)) else None


def cleanup_torch_runtime():
    """
    Cleanup for CUDA + DirectML + CPU.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    if is_directml_device(torch_device):
        try:
            import torch_directml
            empty_cache = getattr(torch_directml, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        except Exception:
            pass

    gc.collect()


def call_with_supported_kwargs(fn, *args, **kwargs):
    """
    Calls loader functions while only passing kwargs they support.

    This lets main.py pass:
      device=torch_device
      use_directml=True/False

    without breaking older adapters that do not accept those args yet.
    """
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )

        if not accepts_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in params}

    except Exception:
        pass

    return fn(*args, **kwargs)

def find_onnx_model_dir(base_dir: str) -> str | None:
    """
    Return the directory containing model.onnx.
    Checks base_dir first, then searches recursively.
    """
    direct = os.path.join(base_dir, "model.onnx")
    if os.path.exists(direct):
        return base_dir

    for root, _, files in os.walk(base_dir):
        if "model.onnx" in files:
            return root

    return None

def _ensure_expected_weight_name(local_dir: str | Path) -> str:
    """
    Make a local HF snapshot/folder look like a standard Transformers checkpoint.
    If none of the expected weight filenames exist but exactly one *.safetensors
    exists, copy it to 'model.safetensors'.
    """
    p = Path(local_dir)
    if not p.is_dir():
        return str(p)

    # Already standard?
    if any((p / n).exists() for n in _EXPECTED_WEIGHT_FILENAMES):
        return str(p)

    safes = list(p.glob("*.safetensors"))
    if len(safes) == 1:
        target = p / "model.safetensors"
        try:
            shutil.copy2(safes[0], target)
        except Exception:
            shutil.copyfile(safes[0], target)
    return str(p)


def _is_depthcrafter():
    local_pipe, local_pipe_type, _ = get_active_pipe_snapshot()
    return (local_pipe_type == "depthcrafter") or getattr(local_pipe, "_is_depthcrafter", False)
    
def get_vda_window_settings(batch_size, user_overlap=None):
    """
    Native PyTorch VDA window settings.

    batch_size controls the VDA window size.
    user_overlap controls how many frames are reused between windows.
    """
    try:
        requested = int(batch_size or 8)
    except Exception:
        requested = 8

    window_size = max(2, requested)

    if user_overlap is None:
        overlap = max(1, window_size // 8)
    else:
        try:
            overlap = int(user_overlap)
        except Exception:
            overlap = max(1, window_size // 8)

    overlap = max(0, min(overlap, window_size - 1))
    stride = max(1, window_size - overlap)

    return window_size, overlap, stride

def _is_vda_runtime():
    """
    True for both:
    - native PyTorch Video Depth Anything adapter
    - fixed ONNX Video Depth Anything model
    """
    local_pipe, local_pipe_type, _ = get_active_pipe_snapshot()
    return (
        local_pipe_type == "vda"
        or (
            local_pipe_type == "onnx"
            and getattr(local_pipe, "_is_vda_onnx", False)
        )
    )

def snap_for_vda(w: int, h: int, base: int = 32):
    """Round each dim UP to nearest multiple of `base`."""
    def r(x): return int((int(x) + base - 1) // base * base)
    return (r(w), r(h))


def _hann2d(h, w):
    wy = np.hanning(max(2, h)); wx = np.hanning(max(2, w))
    m = np.outer(wy, wx).astype(np.float32)
    mmax = float(m.max()) if m.size else 1.0
    return m / (mmax + 1e-8)

def _ensure_depth_np(pred):
    """
    Accepts: torch.Tensor [H,W] or [1,H,W], np.ndarray, dict{'predicted_depth': ...}, list[dict]
    Returns: float32 ndarray [H,W]
    """
    # Unwrap dict/list formats first
    if isinstance(pred, list):
        pred = pred[0]
    if isinstance(pred, dict) and "predicted_depth" in pred:
        pred = pred["predicted_depth"]

    # Convert payload
    if isinstance(pred, torch.Tensor):
        arr = pred.detach().cpu().float().numpy()
    elif isinstance(pred, np.ndarray):
        arr = pred.astype(np.float32, copy=False)
    else:
        raise TypeError(f"Unexpected depth type: {type(pred)}")

    # Squeeze to [H,W]
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):   # [C,H,W]
            arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
        elif arr.shape[2] in (1, 3): # [H,W,C]
            arr = arr[..., 0] if arr.shape[2] == 1 else arr.mean(axis=-1)
        else:
            arr = np.squeeze(arr)
    elif arr.ndim > 2:
        arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"Depth must be 2D after squeeze; got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)

def infer_depth_tile(model_call, rgb_np, inference_size, tile=TILE_SIZE, pad=TILE_PAD):
    """
    model_call: callable like your 'pipe' that accepts [PIL] and optional inference_size=(W,H)
    rgb_np: RGB np.array HxWx3
    """
    if rgb_np.dtype != np.uint8:
        rgb_np = rgb_np.astype(np.uint8, copy=False)

    H, W = rgb_np.shape[:2]
    tgtW, tgtH = (inference_size or (W, H))  # inference_size is (W,H)
    img = cv2.resize(rgb_np[:, :, ::-1], (tgtW, tgtH),  # convert to BGR for OpenCV resize, then we flip back when making PIL
                     interpolation=cv2.INTER_AREA if (tgtW < W or tgtH < H) else cv2.INTER_CUBIC)[:, :, ::-1]

    out_accum = np.zeros((tgtH, tgtW), np.float32)
    w_accum   = np.zeros((tgtH, tgtW), np.float32)

    core = max(1, int(tile) - 2*int(pad))
    step = core
    weight_core = np.maximum(_hann2d(core, core), 1e-3)

    for y0 in range(0, tgtH, step):
        for x0 in range(0, tgtW, step):
            y1 = min(y0 + tile, tgtH); x1 = min(x0 + tile, tgtW)
            yp0 = max(0, y0 - pad);    xp0 = max(0, x0 - pad)
            yp1 = min(tgtH, y1 + pad); xp1 = min(tgtW, x1 + pad)

            crop_rgb = img[yp0:yp1, xp0:xp1, :]  # RGB
            
            # ✅ ViT/DINOv2 safe crop size (W,H) = multiples of 14.
            # Keep the original crop size so the prediction can be resized
            # back before extracting the unpadded center region.
            crop_w, crop_h = crop_rgb.shape[1], crop_rgb.shape[0]
            def r14(x): return ((int(x) + 13) // 14) * 14
            cws, chs = r14(crop_w), r14(crop_h)
            if (crop_w, crop_h) != (cws, chs):
                crop_rgb = cv2.resize(crop_rgb, (cws, chs), interpolation=cv2.INTER_CUBIC)
            
            pil      = Image.fromarray(crop_rgb)

            # Call the model with the crop size (W,H)
            try:
                # pass the ViT-safe crop size to the model
                res = model_call([pil], inference_size=(cws, chs))
            except TypeError:
                res = model_call(pil, inference_size=(cws, chs))
            pred_np = _ensure_depth_np(res)

            # If the crop was resized to ViT-safe dimensions, bring the prediction
            # back to the original crop coordinate system before slicing the center.
            if pred_np.shape != (crop_h, crop_w):
                pred_np = cv2.resize(pred_np, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)

            # Center region corresponding to the unpadded core
            yc0 = y0 - yp0; xc0 = x0 - xp0
            yc1 = yc0 + (y1 - y0); xc1 = xc0 + (x1 - x0)
            center = pred_np[max(0,yc0):max(0,yc1), max(0,xc0):max(0,xc1)]

            ch, cw = (y1 - y0), (x1 - x0)
            if center.shape != (ch, cw):
                center = cv2.resize(center, (cw, ch), interpolation=cv2.INTER_CUBIC)

            w = weight_core
            if w.shape != center.shape:
                w = cv2.resize(w, (cw, ch), interpolation=cv2.INTER_CUBIC)

            out_accum[y0:y1, x0:x1] += center * w
            w_accum[y0:y1, x0:x1]   += w

            if TILE_DEBUG and is_debug_enabled():
                debug_print(
                    f"[tile] ({y0}:{y1},{x0}:{x1}) crop={crop_rgb.shape[:2]} "
                    f"center={center.shape} acc={(out_accum[y0:y1, x0:x1].shape)}"
                )

    # Normalize by weights, guard zeros
    depth_tiled = out_accum / np.maximum(w_accum, 1e-8)
    return depth_tiled.astype(np.float32, copy=False)


def _normalize_to_u8(depth_f, out_size, invert=False, pclip=(1.0, 99.0)):
    d = np.asarray(depth_f, dtype=np.float32)
    if not np.isfinite(d).all():
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    lo = np.percentile(d, pclip[0])
    hi = np.percentile(d, pclip[1])
    if hi - lo < 1e-6:
        # fall back to global min–max, or flat mid-gray if still bad
        dmin, dmax = float(d.min()), float(d.max())
        if dmax - dmin < 1e-6:
            u8 = np.full_like(d, 128, dtype=np.uint8)
        else:
            d = (d - dmin) / (dmax - dmin + 1e-6)
            u8 = (d * 255.0).astype(np.uint8)
    else:
        d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        u8 = (d * 255.0).astype(np.uint8)

    if invert:
        u8 = 255 - u8
    return cv2.resize(u8, out_size, interpolation=cv2.INTER_CUBIC)
    
def normalize_depth(depth_f, out_size, invert=False, pclip=(1.0, 99.0), bit_depth=16, median_cleanup=False):
    d = np.asarray(depth_f, dtype=np.float32)
    if not np.isfinite(d).all():
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    # Percentile stretch to 0..1.
    # For very large maps, estimate percentiles from a smaller sample to reduce CPU time.
    sample = d
    if d.size > 512 * 512 and d.shape[0] > 0 and d.shape[1] > 0:
        sample_w = 512
        sample_h = max(1, int(round(512 * d.shape[0] / max(1, d.shape[1]))))
        sample = cv2.resize(d, (sample_w, sample_h), interpolation=cv2.INTER_AREA)

    lo = np.percentile(sample, pclip[0])
    hi = np.percentile(sample, pclip[1])
    if hi - lo < 1e-6:
        dmin, dmax = float(d.min()), float(d.max())
        if dmax - dmin < 1e-6:
            d = np.full_like(d, 0.5, dtype=np.float32)
        else:
            d = (d - dmin) / (dmax - dmin + 1e-6)
    else:
        d = (d - lo) / (hi - lo + 1e-6)
    d = np.clip(d, 0.0, 1.0)

    # safer resize to avoid overshoot at edges
    ow, oh = out_size
    # AREA for downscale, LINEAR for upscale
    interp = cv2.INTER_AREA if (ow < d.shape[1] or oh < d.shape[0]) else cv2.INTER_LINEAR
    d = cv2.resize(d, (ow, oh), interpolation=interp)
    d = np.clip(d, 0.0, 1.0)  # clamp again after resize

    if bit_depth == 16:
        out = (d * 65535.0 + 0.5).astype(np.uint16)
        # Optional tiny clean up of single-pixel specks.
        # Disabled by default to preserve fine depth detail and save CPU time.
        if median_cleanup:
            out = cv2.medianBlur(out, 3)
        return (65535 - out) if invert else out
    else:
        out = (d * 255.0 + 0.5).astype(np.uint8)
        return (255 - out) if invert else out



def _pred_to_np(pred):
    if isinstance(pred, torch.Tensor):
        return pred.detach().cpu().float().numpy()
    return np.asarray(pred, dtype=np.float32)

def _run_pipe_or_tile(images_pil, inference_size=None, **kwargs):
    """
    Returns list[{'predicted_depth': ndarray or tensor}]
    """
    global pipe, pipe_type, PIPE_EXTRA_ARGS

    local_pipe, local_pipe_type, extra_global = get_active_pipe_snapshot()

    if local_pipe is None:
        raise RuntimeError("No depth model is loaded. Please select and load a model first.")

    # --- ONNX special handling: force the warm-up proven size ---
    if local_pipe_type == "onnx":
        good = getattr(local_pipe, "_good_size", None)

        if good is not None:
            inf_w, inf_h = good
        elif inference_size is not None:
            inf_w, inf_h = int(inference_size[0]), int(inference_size[1])
        else:
            inf_w, inf_h = 512, 288

        inf_w, inf_h = snap_for_vda(inf_w, inf_h, base=32)
        inference_size = (inf_w, inf_h)

        try:
            res = local_pipe(images_pil, inference_size=inference_size)
            if isinstance(res, list):
                return res
            elif isinstance(res, dict):
                return [res]
            else:
                return [{"predicted_depth": res}]
        except TypeError:
            outs = []
            for img in images_pil:
                r = local_pipe([img], inference_size=inference_size)
                if isinstance(r, list):
                    r = r[0]
                outs.append(r if isinstance(r, dict) else {"predicted_depth": r})
            return outs

    # ✅ VDA is sequence-based; never tile it
    use_tiled = bool(USE_TILED_DEPTH) and (local_pipe_type not in ("vda", "vigeo"))
    if use_tiled:
        preds = []
        for img in images_pil:
            rgb = np.array(img.convert("RGB"))
            dep = infer_depth_tile(local_pipe, rgb, inference_size, tile=TILE_SIZE, pad=TILE_PAD)
            dep_min, dep_max = float(np.nanmin(dep)), float(np.nanmax(dep))
            debug_print(f"[tile] range min={dep_min:.6f} max={dep_max:.6f}")
            preds.append({"predicted_depth": dep})
        return preds

    # ✅ Merge per-call kwargs (target_fps, input_size, etc) with global args (steps, etc)
    call_kwargs = {}
    call_kwargs.update(extra_global)
    call_kwargs.update(kwargs)

    # Some pipes accept kwargs, HF depth-estimation often doesn't
    forward_ok = local_pipe_type in ("vda", "vigeo", "da3", "dav2", "depthcrafter", "onnx")
    
    # Log what the pipeline is actually receiving
    if inference_size:
        debug_print(
            f"[DEPTH] Running {local_pipe_type} at "
            f"{inference_size[0]}x{inference_size[1]} with {len(images_pil)} frame(s)"
        )
    else:
        debug_print(
            f"[DEPTH] Running {local_pipe_type} at original resolution with {len(images_pil)} frame(s)"
        )
        
    if forward_ok:
        try:
            res = local_pipe(images_pil, inference_size=inference_size, **call_kwargs)
        except TypeError:
            try:
                res = local_pipe(images_pil, **call_kwargs)
            except TypeError:
                try:
                    res = local_pipe(images_pil, inference_size=inference_size)
                except TypeError:
                    res = local_pipe(images_pil)
    else:
        # HF / generic: try global extras only
        try:
            res = local_pipe(images_pil, inference_size=inference_size, **extra_global)
        except TypeError:
            try:
                res = local_pipe(images_pil, inference_size=inference_size)
            except TypeError:
                outs = []
                for img in images_pil:
                    r = local_pipe(img, inference_size=inference_size)
                    if isinstance(r, list):
                        r = r[0]
                    outs.append(r if isinstance(r, dict) else {"predicted_depth": r})
                return outs

    if isinstance(res, list):
        return res
    elif isinstance(res, dict):
        return [res]
    else:
        return [{"predicted_depth": res}]


class TemporalDepthNormalizer:
    """
    Keeps a smooth running [lo, hi] range over time so video depth
    doesn't 'breathe' when scene stats change a bit.
    """
    def __init__(self, pclip=(1.0, 99.0), momentum=0.95):
        self.p_lo, self.p_hi = pclip
        self.m = float(momentum)
        self.lo = None
        self.hi = None

    def __call__(self, depth_f: np.ndarray) -> np.ndarray:
        d = np.asarray(depth_f, dtype=np.float32)
        if not np.isfinite(d).all():
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-frame percentiles
        fr_lo = float(np.percentile(d, self.p_lo))
        fr_hi = float(np.percentile(d, self.p_hi))

        # Init or EMA smooth
        if self.lo is None or self.hi is None:
            self.lo, self.hi = fr_lo, fr_hi
        else:
            self.lo = self.m * self.lo + (1.0 - self.m) * fr_lo
            self.hi = self.m * self.hi + (1.0 - self.m) * fr_hi

        # Guard degenerate
        if self.hi - self.lo < 1e-6:
            dmin, dmax = float(d.min()), float(d.max())
            if dmax - dmin < 1e-6:
                return np.full_like(d, 0.5, dtype=np.float32)
            d = (d - dmin) / (dmax - dmin + 1e-6)
            return np.clip(d, 0.0, 1.0)

        d = (d - self.lo) / (self.hi - self.lo + 1e-6)
        return np.clip(d, 0.0, 1.0)
        
class FixedPercentileNormalizer:
    """Learn percentiles from bootstrap frames, then use them for ALL frames."""
    def __init__(self, pclip=(2.0, 98.0)):
        self.p_lo, self.p_hi = pclip
        self.lo = None
        self.hi = None
        self.locked = False
    
    def learn(self, depth_f):
        d = np.asarray(depth_f, dtype=np.float32)
        if not np.isfinite(d).all():
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        fr_lo = float(np.percentile(d, self.p_lo))
        fr_hi = float(np.percentile(d, self.p_hi))
        if self.lo is None:
            self.lo, self.hi = fr_lo, fr_hi
        else:
            self.lo = min(self.lo, fr_lo)
            self.hi = max(self.hi, fr_hi)
    
    def lock(self):
        if self.lo is None or self.hi is None:
            self.locked = False
            debug_print("⚠️ Depth range lock skipped: no valid bootstrap depth range was learned.")
            return

        self.locked = True
        debug_print(f"🔒 Depth range locked: lo={self.lo:.4f}, hi={self.hi:.4f}")
    
    def __call__(self, depth_f):
        d = np.asarray(depth_f, dtype=np.float32)
        if not np.isfinite(d).all():
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.locked or self.lo is None:
            lo = float(np.percentile(d, self.p_lo))
            hi = float(np.percentile(d, self.p_hi))
        else:
            lo, hi = self.lo, self.hi
        
        if hi - lo < 1e-6:
            return np.full_like(d, 0.5, dtype=np.float32)
        
        d = (d - lo) / (hi - lo + 1e-6)
        return np.clip(d, 0.0, 1.0)

def fast_depth_to_01(depth_f):
    """
    Fast local per-frame normalization.
    Skips scene-level percentile bootstrap.
    Good for speed testing, but may allow depth breathing/flicker.
    """
    d = np.asarray(depth_f, dtype=np.float32)

    if not np.isfinite(d).all():
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    dmin = float(np.min(d))
    dmax = float(np.max(d))

    if dmax - dmin < 1e-6:
        return np.full_like(d, 0.5, dtype=np.float32)

    d = (d - dmin) / (dmax - dmin + 1e-6)
    return np.clip(d, 0.0, 1.0)

def apply_offload_if_supported(model_callable, caps, mode: str):
    """
    mode in {"none","sequential","full"} (your dropdown values)
    Only used for diffusers pipelines on CUDA with accelerate installed.
    """
    if not (caps.get("is_diffusion") and caps.get("supports_offload", False)):
        return

    if not is_cuda_device(torch_device):
        return
        
    try:
        # These are diffusers helpers that rely on accelerate
        if mode == "sequential":
            # progressively moves modules between CPU/GPU
            model_callable.enable_sequential_cpu_offload()
        elif mode == "full":
            # offload entire model when not in use
            model_callable.enable_model_cpu_offload()
        # "none" = do nothing
    except Exception as e:
        print(f"ℹ️ Offload not applied: {e}")


# ---------- Letterbox detection: robust helpers ----------
# ---- utility metrics that letterbox detection depends on ----
def _luma_saturation(frame_bgr: np.ndarray):
    """
    Returns (Y, S) as float32 arrays:
      Y = luma (0..255), computed from RGB
      S = saturation (0..255), from HSV's S channel
    Works with uint8 BGR frames.
    """
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected BGR uint8 image with 3 channels")

    # BGR -> Y (luma) using Rec.709 coefficients
    b = frame_bgr[..., 0].astype(np.float32)
    g = frame_bgr[..., 1].astype(np.float32)
    r = frame_bgr[..., 2].astype(np.float32)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b  # 0..255 range

    # BGR -> HSV -> S (saturation)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32)  # 0..255
    return y, s


def is_scene_cut(prev_gray: np.ndarray, gray: np.ndarray,
                 mad_thresh: float = 28.0,
                 corr_thresh: float = 0.60) -> bool:
    """
    Lightweight scene-cut detector.
    - If mean absolute difference (MAD) is large -> cut.
    - Else, compare 64-bin grayscale histograms; low correlation -> cut.
    """
    if prev_gray is None or gray is None:
        return False
    if prev_gray.shape != gray.shape:
        return True

    # Mean absolute difference
    mad = float(np.mean(np.abs(prev_gray.astype(np.int16) - gray.astype(np.int16))))
    if mad > mad_thresh:
        return True

    # Histogram correlation as a secondary check
    hist1 = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([gray],      [0], None, [64], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
    return corr < corr_thresh


def _row_uniformity_metrics(bgr):
    # Mean/variance per row for luma + saturation (fast)
    y, s = _luma_saturation(bgr)
    y_row_mean = y.mean(axis=1)
    y_row_var  = y.var(axis=1)
    s_row_mean = s.mean(axis=1)
    return y_row_mean, y_row_var, s_row_mean

def _horizontal_edge_density(gray, ksize=3, low=30, high=90):
    edges = cv2.Canny(gray, low, high, apertureSize=ksize, L2gradient=True)
    # Count edges along rows (lower = more likely uniform bars)
    row_edge_density = edges.mean(axis=1)  # 0..255 -> normalize below
    return row_edge_density / 255.0

def detect_letterbox_strict_robust(
    frame_bgr,
    y_thresh=24,
    var_thresh=3.0,
    sat_thresh=10.0,
    max_scan_frac=0.18,
    min_band_frac=0.06,
    edge_max=0.06  # rows with more than ~4% edges are not “bars”
):
    """
    Single-frame guess for (top, bottom) with extra edge-uniformity gate.
    Returns (top, bottom) or (0, 0).
    """
    h, w = frame_bgr.shape[:2]
    if h < 64 or w < 64:
        return 0, 0

    y_mean, y_var, s_mean = _row_uniformity_metrics(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    row_edge = _horizontal_edge_density(gray)

    def scan(side):
        H = int(h * max_scan_frac)
        run = 0
        if side == "top":
            rng = range(0, H)
        else:
            rng = range(h-1, h-1-H, -1)

        for i in rng:
            if (y_mean[i] < y_thresh and
                y_var[i]  < var_thresh and
                s_mean[i] < sat_thresh and
                row_edge[i] <= edge_max):
                run += 1
            else:
                break

        min_band = int(h * min_band_frac)
        if run < min_band:
            run = 0
        if run % 2 == 1:
            run -= 1
        return max(run, 0)

    top = scan("top")
    bot = scan("bottom")
    if top + bot >= h * 0.6:  # absurd
        return 0, 0
    return int(top), int(bot)

def is_near_black_frame(frame_bgr, mean_thresh=18, edge_thresh=0.02):
    # Extended: also check edges; pure black/fades have very few edges
    y, _ = _luma_saturation(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    row_edge = _horizontal_edge_density(gray).mean()
    return float(y.mean()) < mean_thresh and row_edge < edge_thresh

def detect_letterbox_multiframe_confidence(
    cap, original_height, fps, max_seconds=3, samples=9
):
    """
    Probe early frames and return ((top, bottom), confidence in [0..1]).
    Skips blacks & scene cuts. Confidence = fraction of valid samples agreeing
    with the median within a small tolerance.
    """
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception:
        total = 0

    window = min(total, int((fps if fps and fps > 0 else 30) * max_seconds))
    window = max(window, 1)

    tops, bottoms = [], []
    prev_gray = None

    pos_backup = cap.get(cv2.CAP_PROP_POS_FRAMES)
    idxs = np.linspace(0, max(0, window - 1), num=min(samples, window), dtype=int)

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_near_black_frame(frame) or is_scene_cut(prev_gray, gray):
            prev_gray = gray
            continue

        t, b = detect_letterbox_strict_robust(frame)
        if 0 <= t < original_height and 0 <= b < original_height and (t + b) < original_height:
            tops.append(t)
            bottoms.append(b)

        prev_gray = gray

    cap.set(cv2.CAP_PROP_POS_FRAMES, pos_backup or 0)

    if not tops:
        return (0, 0), 0.0

    t_med = int(np.median(tops))
    b_med = int(np.median(bottoms))

    # even
    if t_med % 2: t_med -= 1
    if b_med % 2: b_med -= 1
    t_med = max(t_med, 0); b_med = max(b_med, 0)
    if t_med + b_med >= original_height * 0.6:
        return (0, 0), 0.0

    # Confidence = fraction within ±4px of median (together)
    agree = 0
    for t, b in zip(tops, bottoms):
        if abs(t - t_med) <= 4 and abs(b - b_med) <= 4:
            agree += 1
    confidence = agree / max(1, len(tops))
    return (t_med, b_med), float(confidence)

# ---------- Runtime tracker with locks & hysteresis ----------
class LetterboxTracker:
    """
    Tracks and freezes letterbox bars. Rechecks only at scene cuts on non-black frames.
    States:
      - locked_zero: no bars; never auto-enable from fades
      - locked_bars: (top, bottom) applied; can update on strong evidence at cuts
    """
    def __init__(
        self,
        h,
        fps,
        min_change=8,
        confirm_needed=3,
        max_total_frac=0.35,
        conf_enable=0.7,   # require >=70% agreement to enable bars
        conf_disable=0.6,  # require >=60% agreement to switch to zero
        cooldown_sec=3.0
    ):
        self.h = int(h)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.min_change = int(min_change)
        self.confirm_needed = int(confirm_needed)
        self.max_total_frac = float(max_total_frac)
        self.conf_enable = float(conf_enable)
        self.conf_disable = float(conf_disable)
        self.cooldown_frames = int(self.fps * cooldown_sec)

        # state
        self.top = 0
        self.bot = 0
        self.locked_zero = True   # default: assume no bars
        self.locked_bars = False
        self._cand = (0, 0)
        self._streak = 0
        self._cooldown = 0

        self.prev_gray = None

    def bootstrap(self, cap):
        (t, b), conf = detect_letterbox_multiframe_confidence(cap, self.h, self.fps)
        if conf >= self.conf_enable and (t + b) > 0:
            self.top, self.bot = t, b
            self.locked_bars = True
            self.locked_zero = False
        else:
            self.top, self.bot = 0, 0
            self.locked_zero = True
            self.locked_bars = False
        self._cooldown = self.cooldown_frames
        return self.top, self.bot, (self.locked_bars, self.locked_zero)

    def should_recheck(self, frame_idx):
        # Only recheck when cooldown done
        return self._cooldown <= 0

    def update(self, frame_bgr, frame_idx):
        # count down cooldown
        if self._cooldown > 0:
            self._cooldown -= 1

        # avoid rechecks on black/fades
        if is_near_black_frame(frame_bgr):
            self.prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return self.top, self.bot

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if not is_scene_cut(self.prev_gray, gray):
            self.prev_gray = gray
            return self.top, self.bot

        # Scene cut: we’re allowed to re-evaluate
        self.prev_gray = gray
        if not self.should_recheck(frame_idx):
            return self.top, self.bot

        # Measure bars for this frame (robust)
        mt, mb = detect_letterbox_strict_robust(frame_bgr)

        # sanity caps
        if (mt + mb) > int(self.h * self.max_total_frac):
            mt, mb = 0, 0

        # even px
        if mt % 2: mt -= 1
        if mb % 2: mb -= 1
        mt = max(mt, 0); mb = max(mb, 0)

        # Hysteresis
        change = abs(mt - self.top) + abs(mb - self.bot)
        if change < self.min_change:
            self._streak = 0
            self._cand = (self.top, self.bot)
            return self.top, self.bot

        cand = (mt, mb)
        if cand == self._cand:
            self._streak += 1
        else:
            self._cand = cand
            self._streak = 1

        if self._streak >= self.confirm_needed:
            # If we were locked_zero, only switch to bars if non-zero and plausible
            if self.locked_zero and (mt + mb) > 0:
                self.top, self.bot = mt, mb
                self.locked_zero = False
                self.locked_bars = True
                self._cooldown = self.cooldown_frames
            # If we were locked_bars, allow switch to different bars or zero
            elif self.locked_bars:
                self.top, self.bot = mt, mb
                self.locked_zero = (mt + mb) == 0
                self.locked_bars = (mt + mb) > 0
                self._cooldown = self.cooldown_frames

        return self.top, self.bot


# ---------- Cropping ----------
def crop_by_bars(frame_bgr, top, bottom):
    h = frame_bgr.shape[0]
    top = max(int(top), 0); bottom = max(int(bottom), 0)
    if top + bottom >= h or h <= 0:
        return frame_bgr
    return frame_bgr[top:h-bottom, :, :]

    
def convert_depth_to_grayscale(depth):
    if isinstance(depth, Image.Image):
        depth = np.array(depth).astype(np.float32)
    elif isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().float().numpy()
    elif isinstance(depth, np.ndarray):
        depth = depth.astype(np.float32)
    else:
        raise TypeError(f"Unsupported depth type: {type(depth)}")

    # Handle [C, H, W] or [H, W, C]
    if depth.ndim == 3:
        if depth.shape[0] in {1, 3}:  # [C, H, W]
            depth = depth[0] if depth.shape[0] == 1 else depth.mean(axis=0)
        elif depth.shape[2] in {1, 3}:  # [H, W, C]
            depth = depth[..., 0] if depth.shape[2] == 1 else depth.mean(axis=-1)
    elif depth.ndim != 2:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")

    # Normalize safely to [0, 255]
    depth_min, depth_max = np.min(depth), np.max(depth)
    if np.isnan(depth_min) or np.isnan(depth_max) or depth_max - depth_min < 1e-6:
        print("⚠️ Skipping frame with invalid depth values.")
        return np.zeros_like(depth, dtype=np.uint8)

    norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    return (norm * 255).astype(np.uint8)


# === Setup: Local weights directory ===
def get_weights_dir():
    return _VD3D_WEIGHTS

local_model_dir = get_weights_dir()
os.makedirs(local_model_dir, exist_ok=True)

# === Suppress Hugging Face symlink warnings (esp. on Windows) ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

INFERENCE_RESOLUTIONS = {
    "Original": None,
    "256x256": (256, 256),
    "384x384 (DPT Large / MiDaS v3.0 Default)": (384, 384),
    "504x504 (DA3 Native)": (504, 504),
    "512x512 (BEiT / MiDaS v3.1 Native)": (512, 512),
    "518x518 (Depth Anything / Video Depth Anything Default)": (518, 518),
    "560x560 (Distill-Any-Depth Train Size)": (560, 560),
    "640x640": (640, 640),
    "700x700 (Distill-Any-Depth Repo Example)": (700, 700),
    "768x768 (Marigold Depth v1.1 Diffusion Default)": (768, 768),
    "896x896": (896, 896),
    "1536x1536 (Depth Pro Native)": (1536, 1536),
    "512x288": (512, 288), "640x352": (640, 352),
    "768x432": (768, 432), "896x512": (896, 512),
    "1024x576": (1024, 576), "1152x640": (1152, 640),
    "1280x720": (1280, 720), "1280x768 (LBM Depth Widescreen)": (1280, 768),
    "1344x768": (1344, 768), "1536x864": (1536, 864),
    "1600x896": (1600, 896), "1792x1008": (1792, 1008),
    "1920x1088": (1920, 1088), "1920x512 (LBM Depth Cinematic Wide)": (1920, 512),
    "512x256 (Fastest)": (512, 256), "704x384 (Balanced)": (704, 384),
    "910x518 (Depth Anything Widescreen)": (910, 518),
    "960x540 (Good Quality)": (960, 540),
    "1024x576 (Max Quality)": (1024, 576),
    "1280x720 (720p HD)": (1280, 720),
    "1920x1080 (1080p HD)": (1920, 1080),
}

def load_supported_models():
    models = {
        "  -- Select Model -- ": "  -- Select Model -- ",

        # ViGeo
        "ViGeo": "vigeo:pkqbajng/ViGeo",

        # Marigold
        "Marigold Depth v1.1 (Diffusers)": "diffusers:prs-eth/marigold-depth-v1-1",
        "Marigold Depth v1.0":             "diffusers:prs-eth/marigold-depth-v1-0",

        # Distill-Any-Depth
        "Distill-Any-Depth Large (xingyang1)": "xingyang1/Distill-Any-Depth-Large-hf",
        "Distill-Any-Depth Small (xingyang1)": "xingyang1/Distill-Any-Depth-Small-hf",
        
#        "Deterministic Video Depth":   "FayeHongfeiZhang/DVD",
#        "Distill-Any-Depth Small (keetrap)":   "keetrap/Distill-Any-Depth-Small-hf",

        "Video Depth Anything Large": "vda:depth-anything/Video-Depth-Anything-Large",
        "Video Depth Anything Small": "vda:depth-anything/Video-Depth-Anything-Small",

        # in load_supported_models()
        "Video Depth Anything (ONNX)": "onnx:FuryTMP/Video-Depth-Anything-L-ONNX-512x288",
        "Distill-Any-Depth Large(ONNX)": "onnx:FuryTMP/Distill-Any-Depth-Large-onnx",
        "Distill-Any-Depth Base(ONNX)": "onnx:FuryTMP/Distill-Any-Depth-Base-onnx",
        "Distill-Any-Depth Small(ONNX)": "onnx:FuryTMP/Distill-Any-Depth-Small-onnx",

        "DA3METRIC-LARGE": "da3:depth-anything/DA3METRIC-LARGE",
        "DA3MONO-LARGE": "da3:depth-anything/DA3MONO-LARGE",
        "DA3-LARGE": "da3:depth-anything/DA3-LARGE",
        "DA3-LARGE-1.1": "da3:depth-anything/DA3-LARGE-1.1",                
        "DA3-BASE":               "da3:depth-anything/DA3-BASE",
        "DA3-SMALL":               "da3:depth-anything/DA3-SMALL",
        "DA3-GIANT":              "da3:depth-anything/DA3-GIANT",
        "DA3-GIANT-1.1":              "da3:depth-anything/DA3-GIANT-1.1",
        "DA3NESTED-GIANT-LARGE":              "da3:depth-anything/DA3NESTED-GIANT-LARGE",
        "DA3NESTED-GIANT-LARGE-1.1":              "da3:depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        

        # Depth Anything v2 - Hugging Face Transformers path
        "Depth Anything v2 Large":                 "depth-anything/Depth-Anything-V2-Large-hf",
        "Depth Anything v2 Base":                  "depth-anything/Depth-Anything-V2-Base-hf",
        "Depth Anything v2 Small":                 "depth-anything/Depth-Anything-V2-Small-hf",
        "Depth Anything v2 Metric Indoor (Large)": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "Depth Anything v2 Metric Outdoor (Large)":"depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",

        # Depth Anything v1
        "Depth Anything v1 Large":    "LiheYoung/depth-anything-large-hf",
        "Depth Anything v1 Base":     "LiheYoung/depth-anything-base-hf",
        "Depth Anything v1 Small":    "LiheYoung/depth-anything-small-hf",
#        "Depth Anything v1 ViT-L/14": "LiheYoung/depth_anything_vitl14",
        
        # Prompt Depth
        "Prompt Depth Anything VITS Transparent": "depth-anything/prompt-depth-anything-vits-transparent-hf",
        

        # Other popular models
#        "DA-2 (Haodongli)":            "haodongli/DA-2",
#        "Pixel-Perfect-Depth":         "gangweix/Pixel-Perfect-Depth",
        "LBM Depth":                   "jasperai/LBM_depth",
        "DepthPro (Apple)":            "apple/DepthPro-hf",
        "ZoeDepth (NYU+KITTI)":        "Intel/zoedepth-nyu-kitti",
        "MiDaS 3.0 (DPT-Hybrid)":      "Intel/dpt-hybrid-midas",
        "DPT Large (Intel)":           "Intel/dpt-large",
        "DPT Large (Manojb)":          "Manojb/dpt-large",
        "DPT BEiT Large 512":          "Intel/dpt-beit-large-512",
        "MiDaS v2 (Qualcomm)":         "qualcomm/Midas-V2",

    }


    # ✅ auto-add local folders as “[Local] {folder}”
    for folder in os.listdir(local_model_dir):
        folder_path = os.path.join(local_model_dir, folder)
        if os.path.isdir(folder_path):
            if (os.path.exists(os.path.join(folder_path, "config.json")) or
                os.path.exists(os.path.join(folder_path, "config.yaml")) or
                os.path.exists(os.path.join(folder_path, "model.onnx"))):
                models[f"[Local] {folder}"] = folder_path

    return models

supported_models = load_supported_models()

def _load_flexible_processor(model_dir, cache_dir=None, prefer_fast=True):
    """
    Try AutoProcessor first, then AutoImageProcessor.
    Prefer fast processors when available, but always fall back safely.
    """
    def _try(cls, **kw):
        if cache_dir is not None:
            kw["cache_dir"] = cache_dir
        return cls.from_pretrained(model_dir, **kw)

    # 1) AutoProcessor path
    for use_fast in ([True, False] if prefer_fast else [False, True]):
        try:
            return _try(AutoProcessor, use_fast=use_fast)
        except TypeError:
            # Some processors do not accept use_fast
            try:
                return _try(AutoProcessor)
            except Exception as e:
                last = e
        except Exception as e:
            last = e

    print(f"ℹ️ AutoProcessor failed for {model_dir}: {last}")

    # 2) AutoImageProcessor path
    for use_fast in ([True, False] if prefer_fast else [False, True]):
        try:
            return _try(AutoImageProcessor, use_fast=use_fast)
        except TypeError:
            # Some processors do not accept use_fast
            try:
                return _try(AutoImageProcessor)
            except Exception as e:
                last = e
        except Exception as e:
            last = e

    print(f"❌ AutoImageProcessor also failed for {model_dir}: {last}")
    return None
    
def preferred_onnx_provider():
    available = ort.get_available_providers()

    # Match selected torch backend where possible.
    if is_cuda_device(torch_device):
        for p in ("CUDAExecutionProvider", "ROCMExecutionProvider"):
            if p in available:
                return p

    if is_directml_device(torch_device):
        if "DmlExecutionProvider" in available:
            return "DmlExecutionProvider"
        print("⚠️ DirectML torch backend selected, but ONNX DmlExecutionProvider is not available.")

    if is_mps_device(torch_device) and "CoreMLExecutionProvider" in available:
        return "CoreMLExecutionProvider"

    if "OpenVINOExecutionProvider" in available:
        return "OpenVINOExecutionProvider"

    return "CPUExecutionProvider"

def ensure_model_downloaded(checkpoint, use_fp16: bool = False):
    """
    Handles HF Transformers checkpoints, Diffusers pipelines, DepthCrafter adapters,
    and local/remote ONNX directories. Also normalizes non-standard HF weight names.
    """

    if isinstance(checkpoint, os.PathLike):
        checkpoint = os.fspath(checkpoint)

    if not isinstance(checkpoint, str) or not checkpoint.strip():
        print(f"❌ Invalid checkpoint: {checkpoint!r}")
        return None, None

    checkpoint = checkpoint.strip()

    if checkpoint.startswith("-- Select"):
        print("❌ No model selected.")
        return None, None

    # Decide dtype once at the top.
    # FP16 is CUDA-only. DirectML uses FP32 for compatibility.
    use_fp16 = bool(use_fp16) and can_use_fp16_on_device(torch_device)
    dtype = active_torch_dtype(use_fp16)
    load_dtype = model_load_dtype(use_fp16)

    # --- ViGeo adapter: vigeo:<hf_repo> ---
    if isinstance(checkpoint, str) and checkpoint.startswith("vigeo:"):
        from core.adapters.vigeo_adapter import load_vigeo_adapter

        spec = checkpoint.split(":", 1)[1].strip()

        print(f"🧩 Loading ViGeo adapter for: {spec}")

        try:
            return call_with_supported_kwargs(
                load_vigeo_adapter,
                spec,
                cache_dir=local_model_dir,
                use_fp16=use_fp16,
                use_directml=is_directml_device(torch_device),
                device=torch_device,
            )
        except Exception as e:
            print(f"❌ ViGeo adapter failed: {e}")
            return None, None

    # --- DepthAnything v2 adapter: dav2:<spec> or path to *.safetensors ---
    if isinstance(checkpoint, str) and (checkpoint.startswith("dav2:") or checkpoint.endswith(".safetensors")):
        from core.adapters.depthanything_adapter import load_da_v2_adapter

        spec = checkpoint.split(":", 1)[1].strip() if checkpoint.startswith("dav2:") else checkpoint

        use_dml = is_directml_device(torch_device)
        use_cuda_fp16 = bool(use_fp16) and can_use_fp16_on_device(torch_device)

        print(
            f"🧩 Loading DA-V2 adapter for: {spec} | "
            f"device={device_display_name()} | fp16={use_cuda_fp16} | directml={use_dml}"
        )

        try:
            return call_with_supported_kwargs(
                load_da_v2_adapter,
                spec,
                cache_dir=local_model_dir,
                use_fp16=use_cuda_fp16,
                use_directml=use_dml,
                device=torch_device,
            )
        except Exception as e:
            print(f"❌ DA-V2 adapter failed: {e}")
            return None, None

    # --- Fast-path redirect: Hugging Face DA-V2 repos -> optimized safetensors adapter ---
    #
    # Without this, these menu entries:
    #   depth-anything/Depth-Anything-V2-Large-hf
    #   depth-anything/Depth-Anything-V2-Base-hf
    #   depth-anything/Depth-Anything-V2-Small-hf
    #
    # go through the generic Hugging Face Transformers path.
    #
    # This redirect loads the equivalent Kijai safetensors checkpoint through
    # core/adapters/depthanything_adapter.py instead, which is generally faster
    # and uses the DA-V2 batching/preprocess path.
    #
    # Optional escape hatch:
    #   set VD3D_DISABLE_DAV2_FAST_REDIRECT=1
    #
    # if you ever want to compare against the pure HF path.
    if isinstance(checkpoint, str):
        disable_dav2_redirect = os.environ.get(
            "VD3D_DISABLE_DAV2_FAST_REDIRECT",
            ""
        ).strip().lower() in {"1", "true", "yes", "on"}

        ck_norm = checkpoint.strip().lower()
        dav2_redirect = None if disable_dav2_redirect else _DAV2_HF_REPO_TO_ADAPTER_SPEC.get(ck_norm)

        if dav2_redirect is not None:
            from core.adapters.depthanything_adapter import load_da_v2_adapter

            use_dml = is_directml_device(torch_device)
            use_cuda_fp16 = bool(use_fp16) and can_use_fp16_on_device(torch_device)

            # DirectML uses FP32 for compatibility.
            precision_key = "fp16" if use_cuda_fp16 else "fp32"
            spec = dav2_redirect[precision_key]

            print(
                f"🚀 Redirecting HF DA-V2 repo to optimized DA-V2 adapter | "
                f"repo={checkpoint} | spec={spec} | "
                f"device={device_display_name()} | fp16={use_cuda_fp16} | directml={use_dml}"
            )

            try:
                return call_with_supported_kwargs(
                    load_da_v2_adapter,
                    spec,
                    cache_dir=local_model_dir,
                    use_fp16=use_cuda_fp16,
                    use_directml=use_dml,
                    device=torch_device,
                )
            except Exception as e:
                print(f"❌ HF DA-V2 fast adapter redirect failed: {e}")
                return None, None

    # --- DepthAnything v3 adapter: da3:<hf_repo_or_preset> ---
    
    # --- DepthAnything v3 adapter: da3:<hf_repo_or_preset> ---
    if isinstance(checkpoint, str) and checkpoint.startswith(("da3:", "dav3:")):
        from core.adapters.depthanything3_adapter import load_da3_adapter
        spec = checkpoint.split(":", 1)[1].strip()
        print(f"🧩 Loading DA3 adapter for: {spec}")
        try:
            return call_with_supported_kwargs(
                load_da3_adapter,
                spec,
                cache_dir=local_model_dir,
                use_fp16=use_fp16,
                use_directml=is_directml_device(torch_device),
                device=torch_device,
            )
        except Exception as e:
            print(f"❌ DA3 adapter failed: {e}")
            return None, None

    # --- Video Depth Anything adapter: vda:<hf_repo> ---
    if isinstance(checkpoint, str) and checkpoint.startswith("vda:"):
        from core.adapters.videodepthanything_adapter import load_vda_adapter
        spec = checkpoint.split(":", 1)[1].strip()
        print(f"🧩 Loading VDA adapter for: {spec}")
        try:
            return call_with_supported_kwargs(
                load_vda_adapter,
                spec,
                cache_dir=local_model_dir,
                use_fp16=use_fp16,
                use_directml=is_directml_device(torch_device),
                device=torch_device,
            )
        except Exception as e:
            print(f"❌ VDA adapter failed: {e}")
            return None, None

    # --- LBM Adapter ---
    if isinstance(checkpoint, str) and "lbm" in checkpoint.lower():
        from core.adapters.lbm_adapter import load_lbm_adapter
        print(f"🧩 Loading LBM adapter for: {checkpoint}")
        try:
            return call_with_supported_kwargs(
                load_lbm_adapter,
                spec=checkpoint,
                cache_dir=local_model_dir,
                use_fp16=use_fp16,
                use_directml=is_directml_device(torch_device),
                device=torch_device,
            )
        except Exception as e:
            print(f"❌ LBM adapter failed: {e}")
            return None, None

    # --- Generic ONNX prefix ---
    if isinstance(checkpoint, str) and checkpoint.startswith("onnx:"):
        spec = checkpoint.split(":", 1)[1].strip()

        # Case 1: local directory path
        if os.path.isdir(spec):
            model_base_dir = spec
        else:
            # Case 2: Hugging Face repo id
            from huggingface_hub import snapshot_download

            safe_folder_name = spec.replace("/", "_")
            cache_path = os.path.join(local_model_dir, safe_folder_name)

            model_base_dir = snapshot_download(
                repo_id=spec,
                cache_dir=cache_path,
                local_files_only=False,
            )

        onnx_dir = find_onnx_model_dir(model_base_dir)
        if not onnx_dir:
            print(f"❌ Could not find model.onnx anywhere under: {model_base_dir}")
            return None, None

        provider = preferred_onnx_provider()
        print(f"🧠 Resolved ONNX model directory: {onnx_dir}")
        return load_onnx_model(onnx_dir, device=provider, model_id=spec)
    
    # --- Local path provided ---
    if os.path.isdir(checkpoint):
        # Local ONNX model detection
        onnx_dir = find_onnx_model_dir(checkpoint)
        if onnx_dir:
            provider = preferred_onnx_provider()
            print(f"🧠 Detected ONNX model in {onnx_dir} (provider={provider})")
            return load_onnx_model(onnx_dir, device=provider, model_id=checkpoint)

        # Local HF (tolerant to custom *.safetensors names)
        try:
            fixed_dir = _ensure_expected_weight_name(checkpoint)
            model = AutoModelForDepthEstimation.from_pretrained(
                fixed_dir,
                torch_dtype=load_dtype,
            )

            if is_cuda_device(torch_device):
                try:
                    model = model.to(memory_format=torch.channels_last)
                except Exception:
                    pass

            model.eval()

            processor = _load_flexible_processor(fixed_dir, prefer_fast=True)

            print(f"📂 Loaded local Hugging Face model from {fixed_dir}")
            return model, processor
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            return None, None

    # === Diffusion Model Check (depth-first, then generic) ===
    if isinstance(checkpoint, str) and checkpoint.startswith("diffusers:"):
        model_id = checkpoint.split(":", 1)[1].strip()

        # Diffusers on DirectML is best-effort.
        # Some pipelines may use ops unsupported by torch-directml.
        diffusers_device = torch_device if is_torch_accelerator_device(torch_device) else torch.device("cpu")
        diffusers_dtype = dtype if (is_cuda_device(torch_device) and use_fp16) else torch.float32
        diffusers_variant = "fp16" if (is_cuda_device(torch_device) and diffusers_dtype == torch.float16) else None

        # 1) Try Marigold depth pipeline
        try:
            from diffusers import MarigoldDepthPipeline
            pipe = MarigoldDepthPipeline.from_pretrained(
                model_id,
                variant=diffusers_variant,
                torch_dtype=diffusers_dtype,
                cache_dir=local_model_dir,
            )

            try:
                pipe = pipe.to(diffusers_device)
            except Exception as move_err:
                if is_directml_device(torch_device):
                    print(
                        f"⚠️ Marigold could not move to DirectML: {move_err}\n"
                        "   Falling back to CPU for this diffusion pipeline."
                    )
                    pipe = pipe.to("cpu")
                else:
                    raise

            def diffusion_pipe(images, inference_size=None, **kw):
                steps = int(kw.get("num_inference_steps", 4))
                ensemble = int(kw.get("ensemble_size", 5))
                if not isinstance(images, list):
                    images = [images]
                results = []
                for img in images:
                    if inference_size:
                        img = img.resize(inference_size, Image.BICUBIC)
                    out = pipe(img, num_inference_steps=steps, ensemble_size=ensemble)
                    results.append({"predicted_depth": out.prediction[0]})
                return results

            print(f"🌀 Diffusion depth model loaded: {model_id}")
            diffusion_pipe._is_marigold = True
            diffusion_pipe.image_processor = pipe.image_processor
            caps = {
                "is_diffusion": True,
                "diffusion_kind": "depth",
                "supports_steps": True,
                "supports_offload": True,
            }
            return diffusion_pipe, caps

        except Exception as e_depth:
            print(f"ℹ️ Depth pipeline not available: {e_depth}")

        # 2) Generic DiffusionPipeline (text->image)
        try:
            from diffusers import DiffusionPipeline
            gpipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=diffusers_dtype,
                cache_dir=local_model_dir,
            )

            try:
                gpipe = gpipe.to(diffusers_device)
            except Exception as move_err:
                if is_directml_device(torch_device):
                    print(
                        f"⚠️ Diffusers pipeline could not move to DirectML: {move_err}\n"
                        "   Falling back to CPU for this diffusion pipeline."
                    )
                    gpipe = gpipe.to("cpu")
                else:
                    raise

            def generic_diffusers_call(x, **kw):
                prompt = x if isinstance(x, str) else kw.get("prompt", "VisionDepth3D")
                out = gpipe(prompt, num_inference_steps=int(kw.get("num_inference_steps", 20)))
                return [{"generated_image": out.images[0]}]

            print(f"🎨 Generic diffusers pipeline loaded: {model_id}")
            generic_diffusers_call._is_generic_diffusers = True
            return generic_diffusers_call, {
                "is_diffusion": True,
                "diffusion_kind": "t2i",
                "supports_offload": True,
            }

        except Exception as e_gen:
            print(f"❌ Failed to load any diffusers pipeline: {e_gen}")
            return None, None

    # --- Hugging Face online model (tolerant to custom names) ---
    safe_folder_name = checkpoint.replace("/", "_")
    local_path = os.path.join(local_model_dir, safe_folder_name)

    try:
        # Try standard load first
        model = AutoModelForDepthEstimation.from_pretrained(
            checkpoint,
            cache_dir=local_path,
            torch_dtype=load_dtype,
        )

        if is_cuda_device(torch_device):
            try:
                model = model.to(memory_format=torch.channels_last)
            except Exception:
                pass

        model.eval()
        
        processor = _load_flexible_processor(checkpoint, cache_dir=local_path, prefer_fast=True)
        print(f"⬇️ Downloaded model from Hugging Face: {checkpoint}")
        return model, processor

    except Exception as e1:
        print(f"⚠️ Standard HF load failed, trying normalization: {e1}")
        try:
            from huggingface_hub import snapshot_download
            snap_dir = snapshot_download(
                repo_id=checkpoint,
                cache_dir=local_path,
                local_files_only=False,
            )

            fixed_dir = _ensure_expected_weight_name(snap_dir)

            model = AutoModelForDepthEstimation.from_pretrained(
                fixed_dir,
                torch_dtype=load_dtype,
            )

            if is_cuda_device(torch_device):
                try:
                    model = model.to(memory_format=torch.channels_last)
                except Exception:
                    pass

            model.eval()

            processor = _load_flexible_processor(fixed_dir, cache_dir=local_path, prefer_fast=True)
            if processor is None:
                print("⚠️ Processor missing, but model loaded. Using raw transforms later.")

            print(f"🛠️ Normalized non-standard weights; loaded from {fixed_dir}")
            return model, processor

        except Exception as e2:
            print(f"❌ Failed to load Hugging Face model after normalization: {e2}")
            return None, None


def load_onnx_model(model_dir, device="CUDAExecutionProvider", model_id=None):
    import onnxruntime as ort
    model_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_path):
        print(f"❌ ONNX model not found: {model_path}")
        return None, None

    print(f"🧠 Loading ONNX model from: {model_path}")

    so = ort.SessionOptions()
    # Safe performance
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # keep safe
    # Let ORT decide threads unless you KNOW better
    so.intra_op_num_threads = 0
    so.inter_op_num_threads = 0
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True


    # Multi backend detection, but respect the caller's requested provider.
    # Example: if the caller asks for CPUExecutionProvider, do not silently pick CUDA/DML.
    available = ort.get_available_providers()
    providers = []

    if device and device in available:
        providers.append(device)
    elif device == "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    else:
        gpu_priority = [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "OpenVINOExecutionProvider",
        ]
        for p in gpu_priority:
            if p in available:
                providers.append(p)
                break

    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    print(f"🔧 ONNX Providers selected: {providers}")

    try:
        session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    except Exception as e1:
        print(f"⚠️ ORT init failed ({providers}): {e1}")
        try:
            session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        except Exception as e2:
            print(f"💥 ORT CPU fallback failed: {e2}")
            return None, None

    # Introspect IO
    try:
        in_names = [i.name for i in session.get_inputs()]
        out_names = [o.name for o in session.get_outputs()]
        print(f"🔎 Inputs={in_names} | Outputs={out_names}")
    except Exception:
        pass

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    input_name = input_info.name
    output_name = output_info.name
    input_shape = input_info.shape
    input_rank = len(input_shape)

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    fixed_T = None
    try:
        if input_rank >= 2 and isinstance(input_shape[1], int) and input_rank == 5:
            fixed_T = int(input_shape[1])
    except Exception:
        fixed_T = None

    fixed_HW = None
    model_tag = os.path.basename(os.path.normpath(model_dir)).lower()
    model_id_text = str(model_id or "").lower()
    tag_source = f"{model_tag} {model_id_text}"
    tag_norm = re.sub(r"[\s_\-\/]+", "", tag_source)

    print(f"ONNX model_tag: {model_tag}")
    print(f"ONNX model_id: {model_id_text}")

    if "distillanydepth" in tag_norm:
        fixed_HW = (518, 518)
        print(f"Detected DistillAnyDepth ONNX – forcing fixed input size {fixed_HW}")

    is_vda_onnx = ("videodepthanything" in tag_norm) or ("vda" in tag_norm and input_rank == 5)

    # Pull fixed T/H/W directly from ONNX input shape when available.
    # Expected video ONNX shape is usually [B, T, C, H, W].
    if input_rank == 5:
        if fixed_T is None and isinstance(input_shape[1], int):
            fixed_T = int(input_shape[1])

        if (
            len(input_shape) >= 5
            and isinstance(input_shape[-2], int)
            and isinstance(input_shape[-1], int)
        ):
            fixed_HW = (int(input_shape[-1]), int(input_shape[-2]))  # W, H

    if is_vda_onnx and input_rank == 5:
        if fixed_T is None:
            fixed_T = 8

        # Your converted model name says 512x288, so use that if shape is dynamic.
        if fixed_HW is None:
            fixed_HW = (512, 288)

        print(f"Detected VideoDepthAnything ONNX fixed model: T={fixed_T}, HW={fixed_HW}")

    print(f"Input shape: {input_shape} | Rank: {input_rank} | fixed_T={fixed_T} | fixed_HW={fixed_HW}")

    def _prep_images(images, inference_size):
        arrs = []
        metas = []

        for img in images:
            # Accept PIL.Image or RGB uint8 ndarray.
            if isinstance(img, np.ndarray):
                if img.ndim != 3 or img.shape[2] != 3:
                    raise ValueError(f"Expected RGB ndarray HxWx3, got shape {img.shape}")
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8, copy=False)
                img = Image.fromarray(np.ascontiguousarray(img), mode="RGB")

            elif not isinstance(img, Image.Image):
                raise TypeError(f"ONNX input must be PIL.Image or RGB ndarray, got {type(img)!r}")

            if inference_size:
                W, H = inference_size
                ow, oh = img.size

                # scale that fits inside W,H
                scale = min(W / ow, H / oh)
                nw, nh = int(round(ow * scale)), int(round(oh * scale))

                img_r = img.resize((nw, nh), Image.BICUBIC)

                # compute padding to center
                pad_left   = (W - nw) // 2
                pad_top    = (H - nh) // 2
                pad_right  = W - nw - pad_left
                pad_bottom = H - nh - pad_top

                img_p = ImageOps.expand(img_r, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))

                metas.append((pad_left, pad_top, nw, nh, W, H, ow, oh))
                img = img_p
            else:
                ow, oh = img.size
                metas.append((0, 0, ow, oh, ow, oh, ow, oh))

            x = np.asarray(img, dtype=np.float32) / 255.0
            x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
            arrs.append(x)

        return arrs, metas

    def _unpad_onnx_depth(depth, meta):
        """
        Removes letterbox padding from ONNX depth output and restores the
        original image aspect ratio before the normal VD3D output resize.

        meta:
          pad_left, pad_top, nw, nh, W, H, ow, oh
        """
        pad_left, pad_top, nw, nh, W, H, ow, oh = meta

        arr = _ensure_depth_np(depth)

        # Make sure output depth matches the padded model canvas size.
        if arr.shape != (H, W):
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_CUBIC)

        x0 = int(pad_left)
        y0 = int(pad_top)
        x1 = int(pad_left + nw)
        y1 = int(pad_top + nh)

        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        x1 = max(x0 + 1, min(x1, W))
        y1 = max(y0 + 1, min(y1, H))

        cropped = arr[y0:y1, x0:x1]

        # Restore original pre-letterbox image shape.
        if cropped.shape != (oh, ow):
            cropped = cv2.resize(cropped, (ow, oh), interpolation=cv2.INTER_CUBIC)

        return cropped.astype(np.float32, copy=False)

    def run_onnx(images, inference_size=None):
        # Some models (like DistillAnyDepthBase) only work at one exact size.
        if fixed_HW is not None:
            W, H = fixed_HW
        else:
            if inference_size is None:
                raise ValueError("❌ Must provide inference_size for ONNX.")
            W, H = int(inference_size[0]), int(inference_size[1])
            # Keep your VDA safe snapping for generic models
            W, H = snap_for_vda(W, H, base=32)

        inference_size = (W, H)


        img_batch, metas = _prep_images(images, inference_size)

        if input_rank == 5:
            original_T = len(img_batch)
            T = original_T

            if fixed_T is not None:
                if T < fixed_T:
                    img_batch += [img_batch[-1]] * (fixed_T - T)
                    T = fixed_T
                elif T > fixed_T:
                    raise ValueError(
                        f"ONNX video model expects fixed T={fixed_T}, but got {T}. "
                        f"Set VDA ONNX batch_size to {fixed_T}."
                    )

            input_tensor = np.stack(img_batch, axis=0)[None, ...]  # [1, T, 3, H, W]
            
            
        elif input_rank == 4:
            input_tensor = np.stack(img_batch, axis=0)             # [B, 3, H, W]
        else:
            raise ValueError(f"❌ Unsupported ONNX input rank: {input_rank}")

        output = session.run([output_name], {input_name: input_tensor})[0]
        if input_rank == 5 and output.ndim == 4 and output.shape[0] == 1:
            output = output.squeeze(0)

        if input_rank == 5:
            valid_T = min(original_T, output.shape[0], len(metas))
            outs = []

            for t in range(valid_T):
                depth_fixed = _unpad_onnx_depth(output[t], metas[t])
                outs.append({"predicted_depth": torch.tensor(depth_fixed)})

            return outs

        else:
            valid_B = min(output.shape[0], len(metas))
            outs = []

            for b in range(valid_B):
                depth_fixed = _unpad_onnx_depth(output[b], metas[b])
                outs.append({"predicted_depth": torch.tensor(depth_fixed)})

            return outs
            
    run_onnx._is_marigold = False
    run_onnx._is_vda_onnx = bool(is_vda_onnx)
    run_onnx._fixed_T = fixed_T
    run_onnx._fixed_HW = fixed_HW

    if fixed_HW is not None:
        run_onnx._good_size = fixed_HW

    run_onnx._is_marigold = False
    return run_onnx, {
        "input_rank": input_rank,
        "fixed_T": fixed_T,
        "fixed_HW": fixed_HW,
        "session": session,
        "provider": providers[0] if providers else "CPUExecutionProvider",
        "is_onnx": True,
        "is_vda_onnx": bool(is_vda_onnx),
        "kind": "vda_onnx" if is_vda_onnx else "onnx",
    }


spinner_states = ["⠋", "⠙", "⠸", "⠴", "⠦", "⠇"]


def start_spinner(widget, message="Warming up model..."):
    def spin(index=0):
        if not getattr(widget, "_spinner_running", False):
            return
        state = spinner_states[index % len(spinner_states)]
        widget.config(text=f"{state} {message}")
        widget.after(200, spin, index + 1)

    widget._spinner_running = True
    spin()

def stop_spinner(widget, final_text):
    widget._spinner_running = False
    widget.config(text=final_text)


def update_pipeline(selected_model_var, status_label_widget, inference_res_var, offload_mode_dropdown, inference_steps_entry, fp16_var, *args):
    global pipe
    
    selected_checkpoint = selected_model_var.get()
    checkpoint = supported_models.get(selected_checkpoint, None)

    if (
        not checkpoint
        or str(selected_checkpoint).strip().startswith("-- Select")
        or str(checkpoint).strip().startswith("-- Select")
    ):
        status_label_widget.config(text="⚠️ Please select a valid depth model.")
        return
        
    # Snapshot Tk values on the UI thread before the worker starts.
    # Tk variables/widgets are not safe to read from background threads.
    try:
        selected_inference_res_text = inference_res_var.get()
    except Exception:
        selected_inference_res_text = ""

    try:
        selected_fp16 = bool(fp16_var.get())
    except Exception:
        selected_fp16 = False

    try:
        selected_steps_text = inference_steps_entry.get().strip() if inference_steps_entry is not None else ""
    except Exception:
        selected_steps_text = ""

    try:
        selected_offload_mode = (offload_mode_dropdown.get() or "none").strip().lower() if offload_mode_dropdown is not None else "none"
    except Exception:
        selected_offload_mode = "none"

    session_id = uuid.uuid4().hex
    current_warmup_session["id"] = session_id

    def is_current_session():
        return current_warmup_session.get("id") == session_id


    def warmup_thread():
        global pipe, pipe_type

        try:
            # If this worker is already stale, do not touch the active global pipe.
            if not is_current_session():
                return

            # Release the previous model before loading the next one so CUDA/CPU
            # memory is not held longer than needed during model switching.
            with pipe_lock:
                old_pipe = pipe
                pipe = None
                pipe_type = None

            try:
                del old_pipe
            except Exception:
                pass
                
            cleanup_torch_runtime()

            use_fp16 = selected_fp16 and can_use_fp16_on_device(torch_device)
            dtype = active_torch_dtype(use_fp16)

            print(
                f"[LOAD SETTINGS] selected={selected_checkpoint} | "
                f"checkpoint={checkpoint} | "
                f"ui_fp16={selected_fp16} | resolved_fp16={use_fp16} | "
                f"dtype={dtype} | backend={device_display_name()} | "
                f"torch_device={torch_device}",
                flush=True,
            )

            model_callable, meta = ensure_model_downloaded(checkpoint, use_fp16=use_fp16)
            
            if not is_current_session():
                print(f"ℹ️ Discarding stale model load for: {selected_checkpoint}")
                try:
                    del model_callable
                except Exception:
                    pass
                cleanup_torch_runtime()
                return

            if not model_callable:
                status_label_widget.after(0, lambda: stop_spinner(status_label_widget, f"❌ Failed to load model: {selected_checkpoint}"))
                return

            # Determine execution backend for legacy pipeline-style APIs.
            # DirectML should use direct model.to(torch_device), not pipeline(device=int).
            if is_cuda_device(torch_device):
                device = 0
            elif is_mps_device(torch_device):
                device = "mps"
            elif is_directml_device(torch_device):
                device = torch_device
            else:
                device = -1

            caps = meta if isinstance(meta, dict) else {}
            if not bool(caps.get("supports_steps", False)):
                set_pipe_extra_args({})            
            supports_steps   = bool(caps.get("supports_steps", False))
            supports_offload = bool(caps.get("supports_offload", False))
            is_onnx          = bool(caps.get("is_onnx", False))
            is_diffusion     = bool(caps.get("is_diffusion", False))
            ck = checkpoint or ""
            is_safetensors = isinstance(ck, str) and ck.lower().endswith(".safetensors")
            is_depthpro    = isinstance(ck, str) and ("depthpro" in ck.lower() or "apple/DepthPro-hf".lower() in ck.lower())
            skip_warmup    = bool((caps.get("skip_warmup", False)) or is_safetensors or is_depthpro)

                        
            if is_onnx:
                if not is_current_session():
                    return

                onnx_pipe = model_callable

                status_label_widget.after(
                    0,
                    lambda: start_spinner(status_label_widget, "Warming up ONNX model...")
                )

                warmup_ok = True
                if not skip_warmup:
                    # --- Robust ONNX warm-up (handles stride quirks, fixed T, fixed H/W) ---
                    try:
                        input_rank = caps.get("input_rank", 4)
                        fixed_T    = caps.get("fixed_T", None)
                        fixed_HW   = caps.get("fixed_HW", None)   # (W, H) if fixed by the model
                        warmup_T   = int(fixed_T) if fixed_T is not None else 8

                        # Pull user pref (if any), then snap for VDA (/32)
                        user_res = parse_inference_resolution(selected_inference_res_text, fallback=(512, 288))
                        if user_res is None:
                            user_res = (512, 288)
                        uW, uH = snap_for_vda(user_res[0], user_res[1], base=32)

                        if fixed_HW is not None:
                            sizes = [fixed_HW]
                        else:
                            candidates = [
                                (uW, uH),
                                (512, 288),
                                (640, 360),
                                (768, 432),
                                (896, 504),
                                (960, 544),
                                (1024, 576),
                                (1152, 648),
                                (1280, 720),
                                (1536, 864),
                                (512, 512),
                                (640, 640),
                                (768, 768),
                            ]
                            seen, sizes = set(), []
                            for s in candidates:
                                s = snap_for_vda(s[0], s[1], base=32)
                                if s not in seen:
                                    sizes.append(s)
                                    seen.add(s)

                        last_err = None
                        warmed = False
                        for (W, H) in sizes:
                            try:
                                if input_rank == 5:
                                    dummy_batch = [
                                        Image.new("RGB", (W, H), (127, 127, 127))
                                        for _ in range(warmup_T)
                                    ]
                                else:
                                    dummy_batch = [Image.new("RGB", (W, H), (127, 127, 127))]
                                _ = onnx_pipe(dummy_batch, inference_size=(W, H))
                                print(
                                    f"ONNX model warmed up. Size={(W, H)}, "
                                    f"T={warmup_T if input_rank == 5 else 1}"
                                )
                                warmed = True
                                meta["good_size"] = (W, H)
                                setattr(onnx_pipe, "_good_size", (W, H))
                                break
                            except Exception as err:
                                last_err = err
                                print(f"Warm-up trial failed at {(W, H)}: {err}")

                        if not warmed:
                            raise RuntimeError(
                                f"ONNX warm-up failed for all tried sizes {sizes}. Last error: {last_err}"
                            )

                    except Exception as e:
                        warmup_ok = False
                        if is_current_session():
                            clear_active_pipe()
                        print(f"ONNX warm-up failed: {e}")
                else:
                    print("Skipping ONNX warm-up by request.")

                if not warmup_ok:
                    status_label_widget.after(
                        0,
                        lambda: stop_spinner(
                            status_label_widget,
                            f"❌ ONNX warm-up failed for {selected_checkpoint}. This export looks incompatible."
                        )
                    )
                    return

                if not is_current_session():
                    return

                set_active_pipe(onnx_pipe, "onnx")

                dev_str = meta.get("provider", "CPUExecutionProvider")
                status_label_widget.after(
                    0,
                    lambda: stop_spinner(
                        status_label_widget,
                        f"ONNX model loaded: {selected_checkpoint} (on {dev_str})"
                    )
                )     
                return

            elif is_diffusion:
                kind = caps.get("diffusion_kind", "depth")
                is_dc = (bool(caps.get("is_depthcrafter", False))
                         or getattr(model_callable, "_is_depthcrafter", False))
                         
                if not is_current_session():
                    return

                if is_dc:
                    set_active_pipe(model_callable, "depthcrafter")
                    status_label_widget.after(0, lambda: start_spinner(status_label_widget, "🔄 Getting DepthCrafter ready..."))
                    try:
                        assert callable(model_callable), "DepthCrafter pipe is not callable"
                        print("🔥 DepthCrafter ready (will run during video processing)")
                        status_label_widget.after(0, lambda: stop_spinner(
                            status_label_widget,
                            f"✔️ DepthCrafter loaded: {selected_checkpoint} (device: {device_display_name()})"
                        ))
                    except Exception as e:
                        msg = f"❌ DepthCrafter init failed: {e}"
                        print(msg)
                        status_label_widget.after(0, lambda: stop_spinner(status_label_widget, msg))
                    return

                # Diffusers: depth pipelines (Marigold)
                if kind == "depth" or getattr(model_callable, "_is_marigold", False):
                    set_active_pipe(model_callable, "diffusion_depth")
                    status_label_widget.after(0, lambda: start_spinner(status_label_widget, "🔄 Warming up diffusion depth model..."))
                    if not skip_warmup:
                        try:
                            dummy = Image.new("RGB", (518, 518), (127, 127, 127))
                            _ = model_callable(dummy)
                        except Exception as e:
                            print(f"ℹ️ Depth warm-up skipped: {e}")
                    else:
                        print("⏭️ Skipping diffusion warm-up by request.")
                        
                    # ⬇️ read UI “inference steps” and store for runtime
                    if supports_steps:
                        try:
                            steps_val = max(1, int(selected_steps_text))
                        except Exception:
                            steps_val = 4
                        set_pipe_extra_args({"num_inference_steps": steps_val})
                    else:
                        set_pipe_extra_args({})

                    # update_pipeline(...) in the generic diffusers branch (after warm-up):
                    if supports_offload:
                        apply_offload_if_supported(model_callable, caps, selected_offload_mode)


                    status_label_widget.after(0, lambda: stop_spinner(
                        status_label_widget,
                        f"✅ Diffusion depth loaded: {selected_checkpoint} (device: {device_display_name()})"
                    ))
                    return

                if not is_current_session():
                    return

                # Diffusers: generic pipelines (text-to-image, etc.)
                set_active_pipe(model_callable, "diffusers_generic")
                status_label_widget.after(0, lambda: start_spinner(status_label_widget, "🔄 Warming up diffusers pipeline..."))
                try:
                    _ = model_callable("VisionDepth3D test prompt")
                    print("🔥 Generic diffusers pipeline warmed up with a test prompt")
                except Exception as e:
                    print(f"ℹ️ Generic warm-up skipped: {e}")
                status_label_widget.after(0, lambda: stop_spinner(
                    status_label_widget,
                    f"✅ Diffusers pipeline loaded: {selected_checkpoint} (device: {device_display_name()})"
                ))

            # --- ViGeo adapter callable ---
            is_vigeo = bool(
                caps.get("kind") == "vigeo"
                or getattr(model_callable, "_is_vigeo", False)
            )

            if is_vigeo:
                if not is_current_session():
                    return

                set_active_pipe(model_callable, "vigeo")
                skip_warmup = True

                status_label_widget.after(
                    0,
                    lambda: start_spinner(status_label_widget, "🔄 Loading ViGeo.")
                )

                if not skip_warmup:
                    try:
                        dummy_frames = [
                            Image.new("RGB", (512, 288), (127, 127, 127))
                            for _ in range(4)
                        ]

                        _ = model_callable(
                            dummy_frames,
                            inference_size=(512, 288),
                            mode="offline",
                        )

                        print("🔥 ViGeo warmed up with dummy clip")
                    except Exception as e:
                        print(f"⚠️ ViGeo warm-up failed: {e}")
                else:
                    print("⏭️ Skipping ViGeo warm-up by request.")

                status_label_widget.after(
                    0,
                    lambda: stop_spinner(
                        status_label_widget,
                        f"✅ ViGeo model loaded: {selected_checkpoint} "
                        f"(device: {device_display_name()})"
                    )
                )
                return
                
            # --- Video Depth Anything adapter callable ---
            is_vda = bool(caps.get("kind") == "vda" or caps.get("is_video_model", False))

            if is_vda:
                if not is_current_session():
                    return
                
                set_active_pipe(model_callable, "vda")
                skip_warmup = True
                
                status_label_widget.after(
                    0,
                    lambda: start_spinner(status_label_widget, "🔄 Warming up Video Depth Anything...")
                )

                # VDA is sequence-based. Warm it up with a tiny fake “clip”
                if not skip_warmup:
                    try:
                        # Small, short clip for warmup
                        dummy_frames = [
                            Image.new("RGB", (512, 288), (127, 127, 127))
                            for _ in range(4)
                        ]

                        # Let adapter infer; pass an input_size if you want it explicit
                        _ = model_callable(dummy_frames, inference_size=(512, 288), input_size=518, target_fps=24)

                        print("🔥 VDA warmed up with dummy clip")
                    except Exception as e:
                        print(f"⚠️ VDA warm-up failed: {e}")
                else:
                    print("⏭️ Skipping VDA warm-up by request.")

                status_label_widget.after(
                    0,
                    lambda: stop_spinner(
                        status_label_widget,
                        f"✅ VDA model loaded: {selected_checkpoint} (device: {device_display_name()})"
                    )
                )
                return

            # --- Depth Anything v2 safetensors adapter callable ---
            is_dav2 = bool(caps.get("is_dav2", False) or getattr(model_callable, "_is_dav2", False))

            if is_dav2:
                if not is_current_session():
                    return

                set_active_pipe(model_callable, "dav2")

                status_label_widget.after(
                    0,
                    lambda: start_spinner(status_label_widget, "🔄 Warming up Depth Anything v2...")
                )

                if not skip_warmup:
                    try:
                        dummy = Image.new("RGB", (512, 288), (127, 127, 127))

                        warmup_size = parse_inference_resolution(
                            selected_inference_res_text,
                            fallback=(518, 518),
                        )

                        _ = model_callable([dummy], inference_size=warmup_size)

                        print("🔥 DA-V2 warmed up with dummy frame")
                    except Exception as e:
                        print(f"⚠️ DA-V2 warm-up failed: {e}")
                else:
                    print("⏭️ Skipping DA-V2 warm-up by request.")

                status_label_widget.after(
                    0,
                    lambda: stop_spinner(
                        status_label_widget,
                        f"✅ Depth Anything v2 loaded: {selected_checkpoint} "
                        f"(device: {device_display_name()})"
                    )
                )
                return
            
            # --- Depth Anything v3 adapter callable ---
            is_da3 = bool(caps.get("kind") == "da3" or caps.get("has_builtin_processor", False))

            if is_da3:
                if not is_current_session():
                    return

                set_active_pipe(model_callable, "da3")

                status_label_widget.after(
                    0,
                    lambda: start_spinner(status_label_widget, "🔄 Warming up Depth Anything v3...")
                )

                if not skip_warmup:
                    try:
                        # Warm up DA3 at the selected UI inference resolution instead of
                        # forcing a heavy 756 process_res.
                        dummy = Image.new("RGB", (512, 288), (127, 127, 127))

                        warmup_size = parse_inference_resolution(
                            selected_inference_res_text,
                            fallback=(504, 504),
                        )

                        _ = model_callable([dummy], inference_size=warmup_size)


                        print("🔥 DA3 warmed up with dummy frame")
                    except Exception as e:
                        print(f"⚠️ DA3 warm-up failed: {e}")
                else:
                    print("⏭️ Skipping DA3 warm-up by request.")

                status_label_widget.after(
                    0,
                    lambda: stop_spinner(
                        status_label_widget,
                        f"✅ DA3 model loaded: {selected_checkpoint} (device: {device_display_name()})"
                    )
                )
                return
            
            else:
                processor = meta

                if processor is None:
                    msg = (
                        f"❌ Failed to load processor for {selected_checkpoint}. "
                        "This model cannot run through the generic Hugging Face path."
                    )
                    print(msg)
                    status_label_widget.after(
                        0,
                        lambda m=msg: stop_spinner(status_label_widget, m)
                    )
                    return

                try:
                    first_param = next(model_callable.parameters())
                    model_dtype = str(first_param.dtype)
                    model_device = str(first_param.device)
                except Exception:
                    model_dtype = "unknown"
                    model_device = "unknown"

                print(
                    f"[HF DEPTH] Using generic Hugging Face path | "
                    f"selected={selected_checkpoint} | "
                    f"class={model_callable.__class__.__name__} | "
                    f"device_before_move={model_device} | "
                    f"dtype={model_dtype} | "
                    f"active_backend={device_display_name()}"
                )
                

                # ------------------------------------------------------------
                # Faster generic HuggingFace depth path.
                #
                # Avoid transformers.pipeline("depth-estimation") in the video
                # loop. The pipeline wrapper is convenient but adds overhead.
                # This direct path:
                #   PIL batch -> processor -> CUDA tensors -> model -> depth
                # ------------------------------------------------------------
                try:
                    model_callable.eval()

                    if is_torch_accelerator_device(torch_device):
                        model_callable.to(torch_device)

                    # channels_last is CUDA-only here. Do not force it on DirectML.
                    if is_cuda_device(torch_device):
                        try:
                            model_callable.to(memory_format=torch.channels_last)
                        except Exception:
                            pass
                            
                    try:
                        first_param = next(model_callable.parameters())
                        print(
                            f"[HF DEPTH] Moved model | "
                            f"device={first_param.device} | dtype={first_param.dtype}"
                        )
                    except Exception:
                        pass

                except Exception as e:
                    print(f"⚠️ Could not move HF model to active device cleanly: {e}")

                def hf_batch_safe_pipe(images, inference_size=None, **_):
                    if not isinstance(images, list):
                        images = [images]

                    # Resize only if needed.
                    # In video mode, process_video2() already resized frames with OpenCV
                    # before wrapping them as PIL images. Re-resizing here adds CPU overhead.
                    if inference_size:
                        target_size = (int(inference_size[0]), int(inference_size[1]))
                        bicubic = getattr(Image, "Resampling", Image).BICUBIC

                        images = [
                            img if getattr(img, "size", None) == target_size else img.resize(target_size, bicubic)
                            for img in images
                        ]

                    use_fp16 = selected_fp16 and can_use_fp16_on_device(torch_device)

                    # Processor call. Prefer do_resize=False because we already
                    # resized above. Some processors do not accept do_resize,
                    # so fall back safely.
                    try:
                        inputs = processor(
                            images=images,
                            return_tensors="pt",
                            do_resize=False,
                        )
                    except TypeError:
                        inputs = processor(
                            images=images,
                            return_tensors="pt",
                        )

                    # Move tensor inputs to GPU once.
                    for k, v in list(inputs.items()):
                        if torch.is_tensor(v):
                            if is_cuda_device(torch_device):
                                # channels_last can help conv-heavy models.
                                if v.dim() == 4 and k == "pixel_values":
                                    v = v.contiguous(memory_format=torch.channels_last)
                                inputs[k] = v.to(torch_device, non_blocking=True)
                            else:
                                inputs[k] = v.to(torch_device)

                    with torch.inference_mode():
                        if is_cuda_device(torch_device) and use_fp16:
                            with torch.autocast("cuda", dtype=torch.float16):
                                outputs = model_callable(**inputs)
                        else:
                            outputs = model_callable(**inputs)

                    if hasattr(outputs, "predicted_depth"):
                        depth = outputs.predicted_depth
                    elif isinstance(outputs, dict) and "predicted_depth" in outputs:
                        depth = outputs["predicted_depth"]
                    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                        depth = outputs[0]
                    else:
                        raise RuntimeError(f"HF model output does not contain predicted_depth: {type(outputs)}")

                    # Normalize shape to [B,H,W].
                    if depth.dim() == 4 and depth.shape[1] == 1:
                        depth = depth[:, 0]
                    elif depth.dim() == 2:
                        depth = depth.unsqueeze(0)

                    # Return list[dict] to match existing pipeline contract.
                    return [
                        {"predicted_depth": depth[i].detach()}
                        for i in range(depth.shape[0])
                    ]

                if not is_current_session():
                    return

                set_active_pipe(hf_batch_safe_pipe, "hf")

                status_label_widget.after(0, lambda: start_spinner(
                    status_label_widget, "🔄 Warming up Hugging Face model..."))
                if not skip_warmup:
                    try:
                        dummy = Image.new("RGB", (384, 384), (127, 127, 127))
                        _ = hf_batch_safe_pipe([dummy])
                        print("🔥 Hugging Face pipeline warmed up with dummy frame")
                    except Exception as e:
                        print(f"⚠️ Hugging Face warm-up failed: {e}")
                else:
                    print("⏭️ Skipping HF warm-up (safetensors or DepthPro).")

                status_label_widget.after(0, lambda: stop_spinner(
                    status_label_widget, f"✅ HF model loaded: {selected_checkpoint} (device: {device_display_name()})"))

        except Exception as e:
            tb = "".join(traceback.format_exc())
            print(tb)
            msg = f"💥 Init error: {e}"
            status_label_widget.after(0, lambda m=msg: stop_spinner(status_label_widget, m))


    threading.Thread(target=warmup_thread, daemon=True).start()

#def convert_depthcrafter_tensor_to_gray_sequence(predictions):
    # predictions: Tensor [T, 3, H, W] (after .frames[0])
#    if isinstance(predictions, torch.Tensor):
#        predictions = predictions.detach().cpu().float().numpy()

#    if predictions.ndim == 4 and predictions.shape[1] == 3:
#        print(f"📦 DepthCrafter output: shape={predictions.shape}")
#       res = predictions.mean(1)  # Convert to [T, H, W]
#    elif predictions.ndim == 3:
#        res = predictions
#    else:
#        raise ValueError(f"❌ Unexpected shape for depthcrafter output: {predictions.shape}")

#    d_min, d_max = np.min(res), np.max(res)
#    res = (res - d_min) / (d_max - d_min + 1e-6)
#    res = (res * 255).astype(np.uint8)
#    return res  # shape [T, H, W]


def save_depthcrafter_outputs(depth: np.ndarray, out_path: str, fps: int = 24):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    out_video_path = f"{out_path}_depth.mkv"
    h, w = depth.shape[1], depth.shape[2]

    # Convert to 8-bit grayscale
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth_8bit = (depth_normalized * 255.0).clip(0, 255).astype(np.uint8)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if out_video_path.endswith(".mp4") else cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h), isColor=False)

    print(f"📁 Saving video to: {out_video_path} with shape {depth.shape} @ {fps} FPS")

    for frame in depth_8bit:
        writer.write(frame)

    writer.release()
    print("✅ Depth video saved.")

    # Optionally save raw .npz
    np.savez_compressed(out_path + ".npz", depth=depth)



def round_to_multiple_of_8(x):
    return (x + 7) // 8 * 8
    
def round_to_multiple_of_14(x):
    return ((int(x) + 13) // 14) * 14



def parse_inference_resolution(res_string, fallback=(384, 384)):
    if not res_string:
        return fallback
    return INFERENCE_RESOLUTIONS.get(str(res_string).strip(), fallback)

class _ConstVar:
    """Thread-safe stand-in for Tk variables when only .get() is needed."""
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

def choose_output_directory(output_label_widget, output_dir_var):
    selected_directory = filedialog.askdirectory()
    if selected_directory:
        output_dir_var.set(selected_directory)
        output_label_widget.config(text=f"📁 {selected_directory}")

def get_dynamic_batch_size(base=4, scale_factor=1.0, max_limit=32, reserve_vram_gb=1.0): 
    # DirectML does not expose reliable PyTorch VRAM stats.
    # Keep a conservative fixed default unless the user manually chooses higher.
    if is_directml_device(torch_device):
        return max(1, int(base))

    # GPU Memory-based scaling only when CUDA exists.
    if is_cuda_device(torch_device):
        try:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                props = torch.cuda.get_device_properties(0)
                total_vram = props.total_memory / (1024 ** 3)
                usable_vram = max(0, total_vram - reserve_vram_gb)
                estimated_batch = int(base * usable_vram * scale_factor)
                return min(max(base, estimated_batch), max_limit)
        except Exception as e:
            print(f"⚠️ VRAM query failed: {e}")

    # CPU or MPS fallback — fixed batch to avoid OOM issues
    return base

def process_image_folder(batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, invert_var, root):
    folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
    # reset flags regardless
    cancel_requested.clear()
    suspend_flag.clear()

    if not folder_path:
        status_label.config(text="⚠️ No folder selected.")
        return

    batch_size_value = batch_size_widget.get().strip()
    output_dir_value = output_dir_var.get().strip()
    inference_res_value = inference_res_var.get()
    invert_value = bool(invert_var.get())

    threading.Thread(
        target=process_images_in_folder,
        args=(
            folder_path,
            _ConstVar(batch_size_value),
            _ConstVar(output_dir_value),
            _ConstVar(inference_res_value),
            status_label,
            progress_bar,
            root,
            _ConstVar(invert_value),
        ),
        daemon=True,
    ).start()

def process_images_in_folder(folder_path, batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, root, invert_var):
    output_dir = output_dir_var.get().strip()
    global pipe, pipe_type
    global global_session_start_time
    global_session_start_time = time.time()

    def ui_status(text):
        try:
            root.after(0, lambda t=text: status_label.config(text=t))
        except Exception:
            pass

    def ui_progress(**kwargs):
        try:
            root.after(0, lambda kw=kwargs: progress_bar.config(**kw))
        except Exception:
            pass

    def ui_warn(title, text):
        try:
            root.after(0, lambda: messagebox.showwarning(title, text))
        except Exception:
            pass

    def ui_error(title, text):
        try:
            root.after(0, lambda: messagebox.showerror(title, text))
        except Exception:
            pass

    output_dir = output_dir_var.get().strip()
    if not output_dir:
        ui_warn("Missing Output Folder", "⚠️ Please select an output directory before processing.")
        ui_status("❌ Output directory not selected.")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            ui_error("Folder Creation Failed", f"❌ Could not create output directory:\n{e}")
            ui_status("❌ Failed to create output directory.")
            return

    current_pipe, current_pipe_type, _ = get_active_pipe_snapshot()
    if current_pipe is None:
        ui_status("❌ No depth model loaded. Please select a model first.")
        ui_progress(value=0)
        return


    inference_size = parse_inference_resolution(inference_res_var.get())

    try:
        user_value = batch_size_widget.get().strip()
        batch_size = int(user_value) if user_value else get_dynamic_batch_size()
        if batch_size <= 0:
            raise ValueError
    except Exception:
        batch_size = get_dynamic_batch_size()
        ui_status(f"⚠️ Invalid batch size. Using dynamic batch size: {batch_size}")

    image_files = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        ],
        key=lambda p: natural_sort_key(os.path.basename(p)),
    )

    if not image_files:
        ui_status("⚠️ No image files found.")
        return

    total_images = len(image_files)
    ui_status(f"📂 Processing {total_images} images...")
    ui_progress(maximum=total_images, value=0)

    start_time = time.time()

    for i in range(0, total_images, batch_size):
        wait_if_paused(status_label)

        if cancel_requested.is_set():
            ui_status("❌ Cancelled by user.")
            return

        batch_files = image_files[i:i + batch_size]
        valid_batch_files = []
        images = []
        original_sizes = []

        for file in batch_files:
            try:
                with Image.open(file) as im:
                    img = im.convert("RGB")
                original_sizes.append(img.size)
                images.append(img)
                valid_batch_files.append(file)
            except Exception as e:
                print(f"❌ Failed to open image {file}: {e}")

        if not images:
            continue

        batch_files = valid_batch_files
        print(f"🚀 Running batch of {len(images)} images at {inference_size}")
        predictions = _run_pipe_or_tile(images, inference_size)

        if len(predictions) != len(batch_files):
            print(f"⚠️ Model returned {len(predictions)} predictions for {len(batch_files)} inputs.")

        for j, prediction in enumerate(predictions[:len(batch_files)]):
            wait_if_paused(status_label)

            if cancel_requested.is_set():
                ui_status("❌ Cancelled during batch.")
                return

            file_path = batch_files[j]
            orig_w, orig_h = original_sizes[j]

            try:
                depth_pred = prediction["predicted_depth"]
                TARGET_BITS = 16

                if USE_TILED_DEPTH:
                    out_arr = normalize_depth(
                        depth_pred,
                        (orig_w, orig_h),
                        invert=invert_var.get(),
                        bit_depth=TARGET_BITS
                    )
                    if TARGET_BITS == 16:
                        depth_image = Image.fromarray(out_arr, mode="I;16")
                    else:
                        depth_image = Image.fromarray(out_arr, mode="L")
                else:
                    if getattr(current_pipe, "_is_marigold", False):
                        depth_image = current_pipe.image_processor.export_depth_to_16bit_png(depth_pred)[0]
                        depth_image = depth_image.resize((orig_w, orig_h), Image.BICUBIC)
                        if invert_var.get():
                            arr = np.array(depth_image, dtype=np.uint16)
                            depth_image = Image.fromarray(65535 - arr, mode="I;16")
                    else:
                        depth_f = _pred_to_np(depth_pred).squeeze()
                        out_arr = normalize_depth(
                            depth_f,
                            (orig_w, orig_h),
                            invert=invert_var.get(),
                            bit_depth=TARGET_BITS
                        )
                        if TARGET_BITS == 16:
                            depth_image = Image.fromarray(out_arr, mode="I;16")
                        else:
                            depth_image = Image.fromarray(out_arr, mode="L")

                image_name = os.path.splitext(os.path.basename(file_path))[0]
                output_filename = f"{image_name}_depth.png"
                file_save_path = os.path.join(output_dir, output_filename)
                depth_image.save(file_save_path)

            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                continue

            elapsed_time = time.time() - start_time
            done = i + j + 1
            fps = done / elapsed_time if elapsed_time > 0 else 0
            eta = (total_images - done) / fps if fps > 0 else 0

            root.after(
                0,
                lambda done=done, fps=fps, eta=eta:
                    update_progress(done, total_images, fps, eta, progress_bar, status_label)
            )

    ui_status("✅ All images processed successfully!")
    ui_progress(value=total_images)


def update_progress(processed, total, fps, eta, progress_bar, status_label):
    progress_bar.config(value=processed)

    # Format FPS and ETA
    fps_text = f"{fps:.2f} FPS"
    eta_text = f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}" if eta > 0 else "ETA: --:--:--"
    progress_text = f"📸 Processed: {processed}/{total} | {fps_text} | {eta_text}"

    status_label.config(text=progress_text)


def process_image(file_path, colormap_var, invert_var, output_dir_var, inference_res_var, input_label, output_label, status_label, progress_bar, folder=False):
    global pipe, pipe_type

    def ui_call(fn, *args, **kwargs):
        try:
            status_label.after(0, lambda: fn(*args, **kwargs))
        except Exception:
            pass

    def ui_warn(title, text):
        try:
            status_label.after(0, lambda: messagebox.showwarning(title, text))
        except Exception:
            pass

    def ui_error(title, text):
        try:
            status_label.after(0, lambda: messagebox.showerror(title, text))
        except Exception:
            pass

    def ui_update_previews(image_obj, depth_obj):
        def _update():
            image_disp = image_obj.copy()
            image_disp.thumbnail((480, 270))
            input_img_tk = ImageTk.PhotoImage(image_disp)
            input_label.config(image=input_img_tk)
            input_label.image = input_img_tk

            depth_disp = depth_obj.copy()
            if depth_disp.mode in ("I", "I;16"):
                depth_array = np.array(depth_disp)
                if depth_array.dtype != np.uint16:
                    denom = max(float(depth_array.max() - depth_array.min()), 1e-6)
                    depth_array = (depth_array - depth_array.min()) / denom
                    depth_array = (depth_array * 65535).astype(np.uint16)
                preview_array = (depth_array / 256).astype(np.uint8)
                depth_disp = Image.fromarray(preview_array, mode="L").convert("RGB")

            depth_disp.thumbnail((480, 270))
            depth_img_tk = ImageTk.PhotoImage(depth_disp)
            output_label.config(image=depth_img_tk)
            output_label.image = depth_img_tk

        try:
            status_label.after(0, _update)
        except Exception:
            pass

    output_dir = output_dir_var.get().strip()
    if not output_dir:
        ui_warn("Missing Output Folder", "⚠️ Please select an output directory before saving.")
        ui_call(status_label.config, text="❌ Output directory not selected.")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            ui_error("Folder Creation Failed", f"❌ Could not create output directory:\n{e}")
            ui_call(status_label.config, text="❌ Failed to create output directory.")
            return

    current_pipe, current_pipe_type, _ = get_active_pipe_snapshot()
    if current_pipe is None:
        ui_call(status_label.config, text="❌ No depth model loaded. Please select a model first.")
        return

    with Image.open(file_path) as im:
        image = im.convert("RGB")
        
    original_size = image.size

    inference_size = parse_inference_resolution(inference_res_var.get())
    
    print("📏 Using inference size:", inference_size)
    predictions = _run_pipe_or_tile([image], inference_size)
    if not (isinstance(predictions, list) and "predicted_depth" in predictions[0]):
        raise ValueError("❌ Unexpected prediction format from depth model.")

    try:
        depth_pred = predictions[0]["predicted_depth"]
        colormap_name = colormap_var.get().strip().lower()

        if USE_TILED_DEPTH:
            # ✅ use normalize_depth instead of _normalize_to_u8
            out_arr = normalize_depth(
                depth_pred,
                original_size,
                invert=invert_var.get(),
                bit_depth=16,
            )

            if colormap_name == "default":
                depth_image = Image.fromarray(out_arr, mode=("I;16" if out_arr.dtype == np.uint16 else "L"))
            else:
                # make an 8-bit copy for colormap preview
                preview8 = out_arr if out_arr.dtype == np.uint8 else (out_arr // 256).astype(np.uint8)
                try:
                    cmap = cm.get_cmap(colormap_name)
                    colored = (cmap(preview8.astype(np.float32) / 255.0)[:, :, :3] * 255).astype(np.uint8)
                    depth_image = Image.fromarray(colored)
                except Exception:
                    depth_image = Image.fromarray(preview8)


        else:
            # === Marigold special path ===
            if getattr(current_pipe, "_is_marigold", False):
                if colormap_name == "default":
                    depth_image = current_pipe.image_processor.export_depth_to_16bit_png(depth_pred)[0]
                else:
                    try:
                        depth_image = current_pipe.image_processor.visualize_depth(depth_pred, color_map=colormap_name)[0]
                    except Exception as e:
                        print(f"⚠️ Failed to apply colormap '{colormap_name}', using default. {e}")
                        depth_image = current_pipe.image_processor.visualize_depth(depth_pred)[0]

                depth_image = depth_image.resize(original_size, Image.BICUBIC)
                if invert_var.get():
                    arr = np.array(depth_image)
                    if depth_image.mode == "I;16":
                        depth_image = Image.fromarray(65535 - arr, mode="I;16")
                    else:
                        depth_image = Image.fromarray(255 - arr.astype(np.uint8))

            # === Other HF/ONNX models ===
            else:
                d = _pred_to_np(depth_pred).squeeze()  # float array from model
                out_arr = normalize_depth(
                    d,
                    original_size,
                    invert=invert_var.get(),
                    bit_depth=16,
                )

                if colormap_name == "default":
                    depth_image = Image.fromarray(out_arr, mode=("I;16" if out_arr.dtype == np.uint16 else "L"))
                else:
                    # colormap preview uses an 8-bit copy only
                    preview8 = out_arr if out_arr.dtype == np.uint8 else (out_arr // 256).astype(np.uint8)
                    try:
                        cmap = cm.get_cmap(colormap_name)
                        colored = (cmap(preview8.astype(np.float32) / 255.0)[:, :, :3] * 255).astype(np.uint8)
                        depth_image = Image.fromarray(colored)
                    except ValueError:
                        print(f"⚠️ Unknown colormap '{colormap_name}', defaulting to grayscale.")
                        depth_image = Image.fromarray(preview8)


    except Exception as e:
        print(f"❌ Error extracting depth: {e}")
        return

    if not folder:
        ui_update_previews(image, depth_image)

    image_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{image_name}_depth.png"
    file_save_path = os.path.join(output_dir, output_filename)
    depth_image.save(file_save_path)

    if not folder:
        cancel_requested.clear()
        ui_call(status_label.config, text=f"✅ Image saved: {file_save_path}")
        ui_call(progress_bar.config, value=100)
        ui_call(progress_bar.stop)



def open_image(status_label_widget, progress_bar_widget, colormap_var, invert_var, output_dir_var, inference_res_var, input_label_widget, output_label_widget):
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpeg;*.jpg;*.png")]
    )
    if file_path:
        # DepthCrafter is video-only
        if _is_depthcrafter():
            messagebox.showwarning(
                "DepthCrafter is video-only",
                "DepthCrafter expects a sequence of frames.\nUse the Video mode instead."
            )
            status_label_widget.config(text="⚠️ DepthCrafter is video-only. Use Video.")
            return

        cancel_requested.clear()  # ✅ Reset before starting
        suspend_flag.clear()
        status_label_widget.config(text="🔄 Processing image...")
        progress_bar_widget.start(10)
        colormap_value = colormap_var.get()
        invert_value = bool(invert_var.get())
        output_dir_value = output_dir_var.get().strip()
        inference_res_value = inference_res_var.get()

        colormap_safe = _ConstVar(colormap_value)
        invert_safe = _ConstVar(invert_value)
        output_dir_safe = _ConstVar(output_dir_value)
        inference_res_safe = _ConstVar(inference_res_value)

        def image_worker():
            try:
                process_image(
                    file_path,
                    colormap_safe,
                    invert_safe,
                    output_dir_safe,
                    inference_res_safe,
                    input_label_widget,
                    output_label_widget,
                    status_label_widget,
                    progress_bar_widget,
                )
            finally:
                try:
                    progress_bar_widget.after(0, progress_bar_widget.stop)
                except Exception:
                    pass

        threading.Thread(target=image_worker, daemon=True).start()



def process_video_folder(
    batch_size_widget,
    codec_var,
    inference_steps_entry,
    output_dir_var,
    inference_res_var,
    status_label,
    progress_bar,
    cancel_requested,
    invert_var,
    save_frames=False,
):
    """UI-side launcher: reads Tk values, asks for folder, then starts worker thread."""

    selected_folder = filedialog.askdirectory(title="Select Folder Containing Videos")
    if not selected_folder:
        status_label.config(text="⚠️ No folder selected.")
        return

    cancel_requested.clear()
    suspend_flag.clear()

    try:
        user_value = batch_size_widget.get().strip()
        batch_size = int(user_value) if user_value else get_dynamic_batch_size()
        if batch_size <= 0:
            raise ValueError
    except Exception:
        batch_size = get_dynamic_batch_size()
        status_label.config(
            text=f"⚠️ Invalid batch size. Using dynamic batch size: {batch_size}"
        )

    output_dir = output_dir_var.get().strip() if output_dir_var else ""
    inference_res_text = inference_res_var.get().strip() if inference_res_var else ""
    invert_value = bool(invert_var.get()) if invert_var else False
    ffmpeg_codec = codec_var.get().strip() if codec_var else ""
    inference_steps_value = inference_steps_entry.get().strip() if inference_steps_entry else ""

    threading.Thread(
        target=process_videos_in_folder,
        args=(
            selected_folder,
            batch_size,
            output_dir,
            inference_res_text,
            status_label,
            progress_bar,
            cancel_requested,
            invert_value,
            ffmpeg_codec,
            inference_steps_value,
            save_frames,
        ),
        daemon=True,
    ).start()

def natural_sort_key(filename):
    """Extract numbers from filenames for natural sorting."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", filename)
    ]

def process_videos_in_folder(
    folder_path,
    batch_size,
    output_dir,
    inference_res_text,
    status_label,
    progress_bar,
    cancel_requested,
    invert_value,
    ffmpeg_codec,
    inference_steps_value,
    save_frames=False,
):
    """Worker thread: processes all videos in the selected folder."""

    def ui_status(text):
        try:
            status_label.after(0, lambda t=text: status_label.config(text=t))
        except Exception:
            pass

    def ui_progress(value):
        try:
            progress_bar.after(0, lambda v=value: progress_bar.config(value=v))
        except Exception:
            pass

    if not folder_path or not os.path.isdir(folder_path):
        ui_status("⚠️ No valid folder selected.")
        ui_progress(0)
        return

    output_dir = (output_dir or "").strip()
    if not output_dir:
        ui_status("❌ Output directory not selected.")
        ui_progress(0)
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        ui_status(f"❌ Could not create output directory: {e}")
        ui_progress(0)
        return

    video_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        ui_status("⚠️ No video files found in the selected folder.")
        return

    video_files.sort(key=natural_sort_key)

    ui_status(f"📂 Processing {len(video_files)} videos...")
    global global_session_start_time
    global_session_start_time = time.time()

    total_frames_all = 0
    for f in video_files:
        p = os.path.join(folder_path, f)
        cap_tmp = cv2.VideoCapture(p)
        try:
            total_frames_all += int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap_tmp.release()

    frames_processed_all = 0

    for video_file in video_files:
        wait_if_paused(status_label)

        if cancel_requested.is_set():
            ui_status("🛑 Processing cancelled by user.")
            ui_progress(0)
            return

        video_path = os.path.join(folder_path, video_file)

        processed = process_video2(
            file_path=video_path,
            total_frames_all=total_frames_all,
            frames_processed_all=frames_processed_all,
            batch_size=batch_size,
            output_dir=output_dir,
            inference_res_text=inference_res_text,
            status_label=status_label,
            progress_bar=progress_bar,
            cancel_requested=cancel_requested,
            invert_value=invert_value,
            ffmpeg_codec=ffmpeg_codec,
            inference_steps_value=inference_steps_value,
            save_frames=save_frames,
        )

        if cancel_requested.is_set():
            ui_status("🛑 Processing cancelled by user.")
            ui_progress(0)
            return

        frames_processed_all += processed

    ui_status("✅ All videos processed successfully!")
    ui_progress(100)

def vda_window_blend_weight(i, n, overlap):
    """
    Smoothly blends VDA overlapping window predictions.

    Frames near the start/end of a window get lower weight so the
    shared overlap region blends instead of snapping between windows.
    """
    if overlap <= 0 or n <= 1:
        return 1.0

    weight = 1.0

    # Fade in at the start of the window
    if i < overlap:
        weight = min(weight, (i + 1) / float(overlap + 1))

    # Fade out at the end of the window
    end_pos = n - 1 - i
    if end_pos < overlap:
        weight = min(weight, (end_pos + 1) / float(overlap + 1))

    return max(float(weight), 1e-3)


def postprocess_vda_depth_to_bgr(
    depth_01,
    original_width,
    original_height,
    invert_flag=False,
    ignore_letterbox_bars=False,
    bars=(0, 0),
):
    """
    Converts normalized VDA depth [0..1] into final BGR frame for video writing.
    Keeps the same behavior as the old VDA write loop.
    """
    if depth_01 is None:
        depth_01 = np.full((original_height, original_width), 0.5, dtype=np.float32)

    depth_u8 = (depth_01 * 255.0 + 0.5).astype(np.uint8)

    if invert_flag:
        depth_u8 = 255 - depth_u8

    depth_u8 = cv2.resize(
        depth_u8,
        (original_width, original_height),
        interpolation=cv2.INTER_CUBIC,
    )

    bt, bb = bars

    if ignore_letterbox_bars and (bt or bb):
        top = max(0, int(bt))
        bot = max(0, int(bb))

        if top + bot < original_height:
            core = depth_u8[top:original_height - bot, :]
            neutral = int(np.median(core)) if core.size else 128

            if top > 0:
                depth_u8[:top, :] = neutral

            if bot > 0:
                depth_u8[original_height - bot:, :] = neutral

    return cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

def process_video2(
    file_path,
    total_frames_all,
    frames_processed_all,
    batch_size,
    output_dir,
    inference_res_text,
    status_label,
    progress_bar,
    cancel_requested,
    invert_value,
    ffmpeg_codec,
    inference_steps_value=None,
    window_size=24,
    overlap=25,
    generator=None,
    offload_mode_dropdown=None,
    save_frames=False,
    target_fps=15,
    ignore_letterbox_bars=False,
    prefer_opencv_writer=False,
    disable_scene_normalization=False,
    vda_overlap=None,
):
    
    def ui_set_progress(pct: int):
        try:
            pct = max(0, min(100, int(pct)))
            progress_bar.after(0, lambda v=pct: progress_bar.config(value=v))
        except Exception:
            pass

    def ui_set_status(text: str):
        try:
            status_label.after(0, lambda t=text: status_label.config(text=t))
        except Exception:
            pass

    global pipe, pipe_type
    global global_session_start_time
    
    profile_depth_stages = is_debug_enabled()
    def _profile_sync():
        if profile_depth_stages and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    stage_times = {
        "decode": 0.0,
        "preprocess": 0.0,
        "inference": 0.0,
        "postprocess": 0.0,
        "write": 0.0,
    }

    stage_counts = {
        "frames": 0,
        "batches": 0,
    }

    last_profile_print = time.time()

    # Plain-value normalization for worker thread use
    output_dir = (output_dir or "").strip()
    inference_res_text = (inference_res_text or "").strip()
    ffmpeg_codec = (ffmpeg_codec or "").strip()
    invert_flag = bool(invert_value)

    try:
        inference_steps = int(str(inference_steps_value).strip()) if str(inference_steps_value).strip() else 2
    except Exception:
        inference_steps = 2

    inference_size = parse_inference_resolution(inference_res_text)

    if not output_dir:
        def _warn():
            try:
                messagebox.showwarning("Missing Output Folder", "⚠️ Please select an output directory before processing.")
            except Exception:
                pass
        status_label.after(0, _warn)
        ui_set_status("❌ Output directory not selected.")
        ui_set_progress(0)
        return 0

    # Snapshot the active model once at job start.
    # This also defines current_pipe for the Marigold branch below.
    current_pipe, current_pipe_type, _ = get_active_pipe_snapshot()
    if current_pipe is None:
        ui_set_status("❌ No depth model loaded. Please select a model first.")
        ui_set_progress(0)
        return 0

    print(
        "[DEPTH SETTINGS]",
        f"pipe_type={current_pipe_type}",
        f"inference_size={inference_size}",
        f"batch_size={batch_size}",
        f"codec={ffmpeg_codec}",
        f"invert={invert_flag}",
        f"ignore_letterbox_bars={ignore_letterbox_bars}",
        f"disable_scene_normalization={disable_scene_normalization}",
        f"save_frames={save_frames}",
        f"device={device_display_name()}",
        flush=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    _, input_filename = os.path.split(file_path)
    name, _ = os.path.splitext(input_filename)
    output_filename = f"{name}_depth.mkv"
    output_path = os.path.join(output_dir, output_filename)
    sidecar_path = os.path.splitext(output_path)[0] + ".letterbox.json"

    # === Marigold special path ===
    if hasattr(current_pipe, "image_processor") and hasattr(current_pipe.image_processor, "export_depth_to_16bit_png"):
        print("🎥 Marigold model detected — switching to frame-based 16-bit processing.")

        ffmpeg_exe = require_tool("ffmpeg")

        probe_cap = cv2.VideoCapture(file_path)
        source_fps = probe_cap.get(cv2.CAP_PROP_FPS)
        probe_cap.release()
        if not source_fps or source_fps <= 0:
            source_fps = 24.0

        tmp_frame_dir = os.path.join(output_dir, f"{name}_tmp_frames")
        os.makedirs(tmp_frame_dir, exist_ok=True)

        try:
            extract_result = subprocess.run(
                [ffmpeg_exe, "-y", "-i", file_path, os.path.join(tmp_frame_dir, "frame_%05d.png")],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                **hidden_subprocess_kwargs(),
            )

            if extract_result.returncode != 0:
                err = extract_result.stderr.decode("utf-8", errors="replace").strip()
                print(f"❌ Marigold frame extraction failed:\n{err}")
                ui_set_status("❌ Failed to extract video frames for Marigold.")
                return 0

            source_frame_count = len([
                f for f in os.listdir(tmp_frame_dir)
                if re.fullmatch(r"frame_\d{5}\.png", f)
            ])

            if source_frame_count <= 0:
                print("❌ Marigold frame extraction produced no frames.")
                return 0

            dummy_widget = _ConstVar(str(batch_size))
            dummy_output_var = _ConstVar(tmp_frame_dir)
            dummy_inference_res = _ConstVar(inference_res_text)
            dummy_invert_var = _ConstVar(invert_flag)

            real_root = status_label

            process_images_in_folder(
                tmp_frame_dir,
                batch_size_widget=dummy_widget,
                output_dir_var=dummy_output_var,
                inference_res_var=dummy_inference_res,
                status_label=status_label,
                progress_bar=progress_bar,
                root=real_root,
                invert_var=dummy_invert_var,
            )

            encode_result = subprocess.run(
                [
                    ffmpeg_exe, "-y",
                    "-framerate", str(float(source_fps)),
                    "-start_number", "1",
                    "-i", os.path.join(tmp_frame_dir, "frame_%05d_depth.png"),
                    "-c:v", "ffv1",
                    "-pix_fmt", "gray16le",
                    output_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                **hidden_subprocess_kwargs(),
            )

            if encode_result.returncode != 0:
                err = encode_result.stderr.decode("utf-8", errors="replace").strip()
                print(f"❌ Marigold depth video encode failed:\n{err}")
                ui_set_status("❌ Failed to encode Marigold depth video.")
                return 0

            print(f"✅ Marigold 16-bit depth video saved: {output_path}")
            return source_frame_count

        finally:
            try:
                shutil.rmtree(tmp_frame_dir, ignore_errors=True)
            except Exception as e:
                print(f"⚠️ Failed to remove temporary Marigold frames: {e}")

    # === Open video ===
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        ui_set_status(f"❌ Error: Cannot open {file_path}")
        ui_set_progress(0)
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if not fps or fps <= 0 or not np.isfinite(fps):
        print(f"⚠️ Invalid input FPS reported by OpenCV: {fps}. Falling back to 24 FPS.")
        fps = 24.0

    if original_width <= 0 or original_height <= 0:
        ui_set_status(f"❌ Error: Invalid video dimensions for {file_path}")
        ui_set_progress(0)
        cap.release()
        return 0

    # NOTE: VDA uses its own overlapping sliding-window stride below.
    # Do not skip source frames here unless output timing/repetition is also handled.

    if _is_vda_runtime() and total_frames > 3000:
        msg = (
            f"ℹ️ VDA streaming mode enabled for {total_frames} frames. "
            "Only the current sliding window is kept in memory."
        )
        print(msg)
        ui_set_status(msg)

    # Letterbox tracking
    tracker = LetterboxTracker(original_height, fps)
    bars_top, bars_bottom, (_lb, _lz) = tracker.bootstrap(cap)

    try:
        candidate_in = os.path.splitext(file_path)[0] + ".letterbox.json"
        meta = None
        if os.path.exists(candidate_in):
            with open(candidate_in, "r", encoding="utf-8") as f:
                meta = json.load(f)
        if meta is None:
            sibling = os.path.splitext(os.path.join(os.path.dirname(file_path),
                                       os.path.basename(file_path).replace("_depth", "")))[0] + ".letterbox.json"
            if os.path.exists(sibling):
                with open(sibling, "r", encoding="utf-8") as f:
                    meta = json.load(f)
        if meta is not None:
            t = int(meta.get("top", 0)); b = int(meta.get("bottom", 0))
            if 0 <= t < original_height and 0 <= b < original_height and (t + b) < int(original_height * 0.6):
                tracker.top, tracker.bot = t, b
                tracker.locked_bars = (t + b) > 0
                tracker.locked_zero = (t + b) == 0
                tracker._cooldown = 0
                bars_top, bars_bottom = t, b
                debug_print(f"[VD3D] Sidecar override: top={t} bottom={b}")
    except Exception:
        pass

    if tracker.prev_gray is None and (bars_top + bars_bottom) == 0:
        pos_backup = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
        ok, f = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_backup or 0)
        if ok and not is_near_black_frame(f):
            t2, b2 = detect_letterbox_strict_robust(f)
            if (t2 + b2) > 0:
                tracker.top, tracker.bot = t2, b2
                tracker.locked_bars = True
                tracker.locked_zero = False
                tracker._cooldown = 0
                bars_top, bars_bottom = t2, b2
                debug_print(f"[VD3D] Fallback probe bars: top={t2} bottom={b2}")

    locked_bars = tracker.locked_bars
    locked_zero = tracker.locked_zero
    debug_print(f"[VD3D] Bootstrap bars: top={bars_top} bottom={bars_bottom} | locked_bars={locked_bars} locked_zero={locked_zero}")

    try:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump({"top": int(bars_top), "bottom": int(bars_bottom), "orig_w": int(original_width),
                       "orig_h": int(original_height), "locked_bars": bool(locked_bars),
                       "locked_zero": bool(locked_zero)}, f, indent=2)
    except Exception as e:
        debug_print(f"⚠️ Failed to write letterbox sidecar: {e}")

    print(f"📁 Saving video to: {output_path}")

    # Codec setup
    ffmpeg_codec = FFMPEG_CODEC_MAP.get(ffmpeg_codec, ffmpeg_codec) if ffmpeg_codec else None
    use_opencv = bool(prefer_opencv_writer) and (ffmpeg_codec is None or is_opencv_safe_fourcc(ffmpeg_codec))
    ff_proc = None
    out = None

    if use_opencv:
        if ffmpeg_codec:
            if ffmpeg_codec.lower() == "mp4v": fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            elif ffmpeg_codec.upper() == "XVID": fourcc = cv2.VideoWriter_fourcc(*"XVID")
            elif ffmpeg_codec.upper() == "DIVX": fourcc = cv2.VideoWriter_fourcc(*"DIVX")
            else: fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))
        if not out.isOpened():
            print("⚠️ OpenCV writer failed. Falling back to FFmpeg pipe.")
            out = None
            use_opencv = False

    if not use_opencv:
        if not ffmpeg_codec:
            ffmpeg_codec = "libx264"
        ff_proc = start_ffmpeg_writer(output_path, fps, original_width, original_height, ffmpeg_codec)

    def cleanup_video_handles(cap_obj, out_obj):
        try:
            if cap_obj is not None:
                cap_obj.release()
        except Exception:
            pass

        try:
            if out_obj is not None:
                out_obj.release()
        except Exception:
            pass

        cleanup_torch_runtime()

    frame_output_dir = os.path.join(output_dir, f"{name}_frames")
    if save_frames:
        os.makedirs(frame_output_dir, exist_ok=True)

    frame_count = 0
    write_index = 0
    frames_batch = []
    total_processed_frames = 0
    bars_batch = []

    if current_pipe_type == "onnx" and getattr(current_pipe, "_is_vda_onnx", False):
        fixed_T = int(getattr(current_pipe, "_fixed_T", 8) or 8)
        if batch_size != fixed_T:
            print(f"[VDA-ONNX] Forcing batch_size from {batch_size} to fixed T={fixed_T}")
        batch_size = fixed_T

    if inference_size is not None:
        target_w, target_h = map(int, inference_size)
        interp = cv2.INTER_AREA if (target_w < original_width or target_h < original_height) else cv2.INTER_LINEAR
    else:
        target_w = target_h = None
        interp = None

    if generator is None:
        seed = 42

        # torch.Generator on DirectML/privateuseone is not reliably supported.
        # Use CUDA generator only on CUDA, otherwise CPU generator.
        gen_device = "cuda" if is_cuda_device(torch_device) else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

    global_session_start_time = time.time()
    prev_depth_u8 = None

    # Throttle UI/progress updates. Updating Qt/Tk-compatible proxies every
    # frame creates unnecessary overhead during video inference.
    last_ui_update = 0.0

    # ============================================================
    # DEPTH NORMALIZATION MODE
    # ============================================================
    temp_normalizer = None
    is_vda_model = _is_vda_runtime()

    if disable_scene_normalization:
        debug_print("⚡ Scene normalization disabled. Using fast local per-frame normalization.")

    elif is_vda_model:
        # VDA is sequence-based. Do not run the sampled-frame bootstrap normalizer,
        # because it feeds VDA tiny 2 to 5 frame chunks and can cause bad allocation.
        debug_print("⚡ VDA detected: skipping sampled bootstrap normalizer. Using per-window depth normalization.")

    else:
        temp_normalizer = FixedPercentileNormalizer(pclip=(2.0, 98.0))
        bootstrap_frames = []
        cap_bootstrap = cv2.VideoCapture(file_path)

        bootstrap_samples = min(5, max(3, total_frames // 300))

        if total_frames <= 0:
            bootstrap_indices = np.array([], dtype=int)
        else:
            bootstrap_indices = np.linspace(0, total_frames - 1, bootstrap_samples, dtype=int)

        debug_print(f"🔍 Bootstrapping depth normalizer with {len(bootstrap_indices)} sampled frame(s).")

        for idx in bootstrap_indices:
            cap_bootstrap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap_bootstrap.read()

            if ret:
                if inference_size:
                    frame_rs = cv2.resize(frame, inference_size, interpolation=cv2.INTER_AREA)
                else:
                    frame_rs = frame

                frame_rgb = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGB)
                bootstrap_frames.append(Image.fromarray(frame_rgb))

        cap_bootstrap.release()

        if bootstrap_frames:
            for i in range(0, len(bootstrap_frames), batch_size):
                batch_imgs = bootstrap_frames[i:i + batch_size]

                if cancel_requested.is_set():
                    break

                try:
                    batch_preds = _run_pipe_or_tile(batch_imgs, inference_size)

                    for pred in batch_preds:
                        depth_f = _ensure_depth_np(pred["predicted_depth"])
                        temp_normalizer.learn(depth_f)

                except Exception as e:
                    print(f"⚠️ Bootstrap batch failed: {e}")

        temp_normalizer.lock()

        if temp_normalizer.locked:
            debug_print(
                f"🔒 Depth normalizer locked with range: "
                f"lo={temp_normalizer.lo:.4f}, hi={temp_normalizer.hi:.4f}"
            )

    # Start FPS timing after bootstrap/normalizer setup.
    # Otherwise reported FPS includes the sampled pre-pass and looks artificially low.
    global_session_start_time = time.time()

    # ============================================================
    # MAIN PROCESSING - wrapped in try/finally for cleanup
    # ============================================================
    try:
        if _is_vda_runtime():
            vda_window_size, vda_overlap, vda_stride = get_vda_window_settings(
                batch_size,
                user_overlap=vda_overlap,
            )

            if vda_window_size >= 32:
                print(
                    f"[VDA][WARN] User selected a large VDA window size: {vda_window_size}. "
                    "This may require a lot of VRAM, especially with VDA Large at 518x518."
                )

            print(
                f"[VDA] Using true streaming window mode: "
                f"size={vda_window_size}, overlap={vda_overlap}, stride={vda_stride}"
            )

            # Only keep the current source window plus pending overlap predictions.
            source_window = []       # list of dicts: {"idx": int, "image": PIL.Image, "bars": (top,bottom)}
            pending_depth = {}       # idx -> accumulated weighted depth float32
            pending_weight = {}      # idx -> accumulated weight
            pending_bars = {}        # idx -> bars
            write_index = 0
            source_index = 0
            eof = False

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            def process_vda_window(window_items, is_final_window=False):
                nonlocal write_index, total_processed_frames, last_ui_update

                if not window_items:
                    return

                wait_if_paused(status_label)

                if cancel_requested.is_set():
                    return

                batch_frames = [item["image"] for item in window_items]
                window_start_idx = window_items[0]["idx"]
                window_end_idx = window_items[-1]["idx"]

                debug_print(
                    f"[VDA] Processing streaming window "
                    f"{window_start_idx}-{window_end_idx} ({len(batch_frames)} frames)"
                )

                try:
                    predictions = _run_pipe_or_tile(
                        batch_frames,
                        inference_size,
                        target_fps=int(target_fps) if target_fps else int(fps or 24),
                    )

                except Exception as e:
                    cleanup_torch_runtime()

                    raise RuntimeError(
                        f"VDA streaming window failed at source frame {window_start_idx}. "
                        f"Window had {len(batch_frames)} frame(s). "
                        f"Try a smaller VDA batch/window size, such as 8 or 4, "
                        f"or use Video Depth Anything Small. Original error: {e}"
                    ) from e

                for i, pred in enumerate(predictions):
                    if i >= len(window_items):
                        break

                    item = window_items[i]
                    global_idx = item["idx"]

                    depth_f = _ensure_depth_np(pred["predicted_depth"])

                    if temp_normalizer is not None:
                        depth_01 = temp_normalizer(depth_f)
                    else:
                        depth_01 = fast_depth_to_01(depth_f)

                    depth_01 = np.asarray(depth_01, dtype=np.float32)

                    weight = vda_window_blend_weight(
                        i=i,
                        n=len(predictions),
                        overlap=vda_overlap,
                    )

                    if global_idx not in pending_depth:
                        pending_depth[global_idx] = depth_01 * weight
                        pending_weight[global_idx] = weight
                        pending_bars[global_idx] = item["bars"]
                    else:
                        pending_depth[global_idx] += depth_01 * weight
                        pending_weight[global_idx] += weight

                # After processing a window, only the first stride frames are finalized.
                # The trailing overlap frames must wait for the next window.
                if is_final_window:
                    safe_write_before = window_end_idx + 1
                else:
                    safe_write_before = window_start_idx + vda_stride

                while write_index < safe_write_before and not cancel_requested.is_set():
                    if write_index not in pending_depth:
                        # This should only happen if a model returned fewer frames than expected.
                        depth_01 = np.full(
                            (original_height, original_width),
                            0.5,
                            dtype=np.float32,
                        )
                        bars = pending_bars.get(write_index, (0, 0))
                    else:
                        depth_01 = pending_depth[write_index] / max(
                            pending_weight.get(write_index, 1.0),
                            1e-6,
                        )
                        bars = pending_bars.get(write_index, (0, 0))

                    bgr = postprocess_vda_depth_to_bgr(
                        depth_01=depth_01,
                        original_width=original_width,
                        original_height=original_height,
                        invert_flag=invert_flag,
                        ignore_letterbox_bars=ignore_letterbox_bars,
                        bars=bars,
                    )

                    if use_opencv:
                        out.write(bgr)
                    else:
                        ff_proc.stdin.write(bgr.tobytes())

                    if save_frames:
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(
                            os.path.join(frame_output_dir, f"frame_{write_index:05d}.png"),
                            gray,
                        )

                    pending_depth.pop(write_index, None)
                    pending_weight.pop(write_index, None)
                    pending_bars.pop(write_index, None)

                    write_index += 1
                    total_processed_frames += 1

                now_ui = time.time()

                # Throttle VDA progress/UI updates too.
                # This prevents per-window/per-flush UI overhead.
                if (now_ui - last_ui_update) >= 0.50 or is_final_window:
                    last_ui_update = now_ui

                    elapsed = now_ui - global_session_start_time
                    avg_fps_vda = total_processed_frames / max(elapsed, 1e-6)

                    if total_frames and total_frames > 0:
                        progress = int((min(source_index, total_frames) / total_frames) * 100)
                        remaining = max(0, total_frames - total_processed_frames)
                        eta = remaining / max(avg_fps_vda, 1e-6)

                        ui_set_status(
                            f"VDA Streaming: {total_processed_frames}/{total_frames} | "
                            f"FPS: {avg_fps_vda:.1f} | "
                            f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}"
                        )
                        ui_set_progress(progress)

                    else:
                        ui_set_status(
                            f"VDA Streaming: {total_processed_frames} frames | "
                            f"FPS: {avg_fps_vda:.1f}"
                        )

            while not eof and not cancel_requested.is_set():
                wait_if_paused(status_label)

                # Fill the rolling source window.
                while len(source_window) < vda_window_size and not cancel_requested.is_set():
                    ret, frame = cap.read()

                    if not ret:
                        eof = True
                        break

                    source_index += 1

                    if ignore_letterbox_bars:
                        bt, bb = tracker.update(frame, source_index)
                    else:
                        bt, bb = 0, 0

                    if inference_size:
                        frame_rs = cv2.resize(frame, inference_size, interpolation=cv2.INTER_AREA)
                    else:
                        frame_rs = frame

                    # VDA adapter accepts NumPy RGB uint8 directly.
                    # Avoid NumPy -> PIL -> NumPy round-trip per frame.
                    frame_rgb = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGB)
                    frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)

                    source_window.append({
                        "idx": source_index - 1,
                        "image": frame_rgb,
                        "bars": (bt, bb),
                    })

                if not source_window:
                    break

                # Normal full window, process then keep only overlap frames.
                if len(source_window) >= vda_window_size:
                    process_vda_window(source_window, is_final_window=False)

                    # Drop the finalized stride frames. Keep overlap frames for next window.
                    source_window = source_window[vda_stride:]

                    if source_index % max(1, vda_window_size * 10) == 0:
                        cleanup_torch_runtime()

                # EOF partial window, process everything left and flush all remaining frames.
                elif eof:
                    process_vda_window(source_window, is_final_window=True)
                    source_window.clear()

            # Flush anything that is still pending.
            while pending_depth and not cancel_requested.is_set():
                if write_index not in pending_depth:
                    break

                depth_01 = pending_depth[write_index] / max(
                    pending_weight.get(write_index, 1.0),
                    1e-6,
                )
                bars = pending_bars.get(write_index, (0, 0))

                bgr = postprocess_vda_depth_to_bgr(
                    depth_01=depth_01,
                    original_width=original_width,
                    original_height=original_height,
                    invert_flag=invert_flag,
                    ignore_letterbox_bars=ignore_letterbox_bars,
                    bars=bars,
                )

                if use_opencv:
                    out.write(bgr)
                else:
                    ff_proc.stdin.write(bgr.tobytes())

                if save_frames:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(
                        os.path.join(frame_output_dir, f"frame_{write_index:05d}.png"),
                        gray,
                    )

                pending_depth.pop(write_index, None)
                pending_weight.pop(write_index, None)
                pending_bars.pop(write_index, None)

                write_index += 1
                total_processed_frames += 1

            ui_set_status(f"VDA Done: {total_processed_frames} frames")
            ui_set_progress(100)
            frame_count = total_processed_frames

        else:
            # Non-VDA batch processing.
            #
            # Important:
            # Do not rely on frame_count == total_frames to flush the last batch.
            # OpenCV can report CAP_PROP_FRAME_COUNT as 0 or slightly wrong for
            # some files/streams, which used to drop the final partial batch.
            def flush_non_vda_batch():
                nonlocal prev_depth_u8, write_index, total_processed_frames, last_profile_print

                if not frames_batch:
                    return True

                wait_if_paused(status_label)
                if cancel_requested.is_set():
                    return False

                extra = {}
                if pipe_type == "vda":
                    extra = {
                        "target_fps": int(target_fps) if target_fps and target_fps > 0 else int(fps),
                        "input_size": 518,
                    }

                _profile_sync()
                t_infer = time.perf_counter()

                predictions = _run_pipe_or_tile(frames_batch, inference_size, **extra)

                _profile_sync()
                stage_times["inference"] += time.perf_counter() - t_infer
                stage_counts["batches"] += 1

                if len(predictions) != len(frames_batch):
                    print(
                        f"⚠️ Model returned {len(predictions)} predictions "
                        f"for {len(frames_batch)} video frames."
                    )

                for i, prediction in enumerate(predictions[:len(frames_batch)]):
                    if cancel_requested.is_set():
                        return False

                    try:
                        # -------------------------
                        # Postprocess timing
                        # -------------------------
                        t_post = time.perf_counter()

                        raw_depth = prediction["predicted_depth"]
                        depth_f = _ensure_depth_np(raw_depth).squeeze()

                        if temp_normalizer is not None:
                            depth_01 = temp_normalizer(depth_f)
                        else:
                            depth_01 = fast_depth_to_01(depth_f)

                        depth_u8 = (depth_01 * 255.0 + 0.5).astype(np.uint8)

                        if invert_flag:
                            depth_u8 = 255 - depth_u8

                        depth_u8 = cv2.resize(
                            depth_u8,
                            (original_width, original_height),
                            interpolation=cv2.INTER_CUBIC,
                        )

                        if prev_depth_u8 is None:
                            smoothed_u8 = depth_u8
                        else:
                            smoothed_u8 = cv2.addWeighted(prev_depth_u8, 0.2, depth_u8, 0.8, 0.0)

                        prev_depth_u8 = smoothed_u8

                        bt, bb = bars_batch[i] if i < len(bars_batch) else (0, 0)

                        if ignore_letterbox_bars and (bt or bb):
                            top = max(0, int(bt))
                            bot = max(0, int(bb))

                            if top + bot < original_height:
                                full_gray = smoothed_u8.copy()
                                core = full_gray[top:original_height - bot, :]
                                neutral = int(np.median(core)) if core.size else 0

                                if top > 0:
                                    full_gray[:top, :] = neutral

                                if bot > 0:
                                    full_gray[original_height - bot:, :] = neutral

                                bgr = cv2.cvtColor(full_gray, cv2.COLOR_GRAY2BGR)
                            else:
                                bgr = cv2.cvtColor(smoothed_u8, cv2.COLOR_GRAY2BGR)
                        else:
                            bgr = cv2.cvtColor(smoothed_u8, cv2.COLOR_GRAY2BGR)

                        stage_times["postprocess"] += time.perf_counter() - t_post

                        # -------------------------
                        # Write timing
                        # -------------------------
                        t_write = time.perf_counter()

                        if use_opencv:
                            out.write(bgr)
                        else:
                            ff_proc.stdin.write(bgr.tobytes())

                        if save_frames:
                            cv2.imwrite(
                                os.path.join(frame_output_dir, f"frame_{write_index:05d}.png"),
                                smoothed_u8,
                            )

                        stage_times["write"] += time.perf_counter() - t_write

                        write_index += 1
                        total_processed_frames += 1

                    except Exception as e:
                        print(f"Depth processing error: {e}")

                now_profile = time.time()

                if profile_depth_stages and (now_profile - last_profile_print) >= 10:
                    total_profile = sum(stage_times.values()) or 1e-6

                    debug_print(
                        "[DEPTH PROFILE] "
                        f"frames={stage_counts['frames']} batches={stage_counts['batches']} | "
                        f"decode={stage_times['decode']:.2f}s ({stage_times['decode'] / total_profile * 100:.1f}%) | "
                        f"pre={stage_times['preprocess']:.2f}s ({stage_times['preprocess'] / total_profile * 100:.1f}%) | "
                        f"infer={stage_times['inference']:.2f}s ({stage_times['inference'] / total_profile * 100:.1f}%) | "
                        f"post={stage_times['postprocess']:.2f}s ({stage_times['postprocess'] / total_profile * 100:.1f}%) | "
                        f"write={stage_times['write']:.2f}s ({stage_times['write'] / total_profile * 100:.1f}%)"
                    )

                    last_profile_print = now_profile

                if frame_count % 300 == 0:
                    if is_cuda_device(torch_device):
                        try:
                            if torch.cuda.memory_reserved() > 0.90 * torch.cuda.get_device_properties(0).total_memory:
                                torch.cuda.empty_cache()
                        except Exception:
                            pass

                    if is_directml_device(torch_device):
                        try:
                            import torch_directml
                            empty_cache = getattr(torch_directml, "empty_cache", None)
                            if callable(empty_cache):
                                empty_cache()
                        except Exception:
                            pass

                    gc.collect()

                frames_batch.clear()
                bars_batch.clear()
                return True

            while True:
                wait_if_paused(status_label)
                if cancel_requested.is_set():
                    break

                t_decode = time.perf_counter()
                ret, frame = cap.read()
                stage_times["decode"] += time.perf_counter() - t_decode

                if not ret:
                    # Flush any remaining partial batch at EOF.
                    flush_non_vda_batch()
                    break

                frame_count += 1

                t_pre = time.perf_counter()

                if ignore_letterbox_bars:
                    bars_top, bars_bottom = tracker.update(frame, frame_count)
                else:
                    bars_top, bars_bottom = 0, 0

                if current_pipe_type == "da3":
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    if inference_size is None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rs = cv2.resize(frame, (target_w, target_h), interpolation=interp)
                        frame_rgb = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGB)

                # DA-V2 adapter accepts RGB uint8 NumPy directly.
                # Avoid RGB NumPy -> PIL -> NumPy round-trip per frame.
                if current_pipe_type == "dav2":
                    frames_batch.append(np.ascontiguousarray(frame_rgb, dtype=np.uint8))
                else:
                    frames_batch.append(Image.fromarray(frame_rgb))

                bars_batch.append((bars_top, bars_bottom))

                stage_times["preprocess"] += time.perf_counter() - t_pre
                stage_counts["frames"] += 1

                if len(frames_batch) >= batch_size:
                    if not flush_non_vda_batch():
                        break

                elapsed = time.time() - global_session_start_time
                avg_fps = total_processed_frames / elapsed if elapsed > 0 else 0
                remaining = max(0, total_frames - total_processed_frames) if total_frames > 0 else 0
                eta = remaining / avg_fps if avg_fps > 0 else 0
                progress_den = max(1, int(total_frames_all or total_frames or 1))
                progress = int(((frames_processed_all + total_processed_frames) / progress_den) * 100)

                ui_set_status(
                    f"{frames_processed_all + total_processed_frames}/{total_frames_all} | "
                    f"FPS: {avg_fps:.1f} | "
                    f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}"
                )
                ui_set_progress(progress)

    finally:
        cleanup_video_handles(cap, out)
        
        ffmpeg_error_text = None
        if ff_proc is not None:
            try:
                if ff_proc.stdin:
                    ff_proc.stdin.close()
            except Exception:
                pass

            try:
                ff_proc.wait(timeout=15)
            except Exception:
                try:
                    ff_proc.kill()
                    ff_proc.wait(timeout=5)
                except Exception:
                    pass

            try:
                t = getattr(ff_proc, "_stderr_thread", None)
                if t is not None:
                    t.join(timeout=0.5)
            except Exception:
                pass

            stderr_data = b"".join(getattr(ff_proc, "_stderr_chunks", []) or [])

            if ff_proc.returncode not in (0, None):
                try:
                    ffmpeg_error_text = stderr_data.decode("utf-8", errors="replace").strip()
                except Exception:
                    ffmpeg_error_text = "Unknown FFmpeg error"

        if ffmpeg_error_text:
            ui_set_status(f"FFmpeg encode failed: {os.path.basename(output_path)}")
            print(f"[FFMPEG ERROR] {output_path}\n{ffmpeg_error_text}")
            ui_set_progress(0)
        elif cancel_requested.is_set():
            ui_set_status("Cancelled.")
            ui_set_progress(0)
        else:
            ui_set_status(f"Done: {output_path}")
            ui_set_progress(100)

    return frame_count


def is_av1_encoded(file_path):
    try:
        ffprobe_exe = require_tool("ffprobe")

        result = subprocess.run(
            [
                ffprobe_exe,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=nokey=1:noprint_wrappers=1",
                file_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            **hidden_subprocess_kwargs(),
        )
        codec = result.stdout.strip().lower()
        return "av1" in codec
    except Exception as e:
        print(f"⚠️ Failed to check codec with ffprobe: {e}")
        return False

def open_video(status_label, progress_bar, batch_size_widget, output_dir_var, inference_res_var, invert_var, inference_steps_entry, offload_mode_dropdown, codec_var):
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("All Supported Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv;*.webm;*.mpeg;*.mpg"),
            ("MP4 Files", "*.mp4"),
            ("AVI Files", "*.avi"),
            ("MOV Files", "*.mov"),
            ("MKV Files", "*.mkv"),
            ("FLV Files", "*.flv"),
            ("WMV Files", "*.wmv"),
            ("WebM Files", "*.webm"),
            ("MPEG Files", "*.mpeg;*.mpg"),
            ("All Files", "*.*"),
        ]
    )

    global global_session_start_time
    if global_session_start_time is None:
        global_session_start_time = time.time()

    if file_path:
        # 🔍 Detect AV1 codec
        if is_av1_encoded(file_path):
            messagebox.showwarning(
                "Unsupported AV1 Input",
                "🚫 This video is encoded with AV1.\n\n"
                "This app currently decodes video through OpenCV for depth processing, "
                "and your OpenCV build may not support AV1.\n\n"
                "Please re-encode it to H.264 first, for example:\n\n"
                "ffmpeg -i input.mkv -c:v libx264 -crf 18 -preset veryfast output.mp4"
            )
            status_label.config(text="❌ AV1 input not supported. Re-encode to H.264.")
            return

        cancel_requested.clear()
        suspend_flag.clear()
        status_label.config(text="🔄 Processing video...")
        progress_bar.config(mode="determinate", maximum=100, value=0)

        cap = cv2.VideoCapture(file_path)
        total_frames_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        try:
            user_value = batch_size_widget.get().strip()
            batch_size = int(user_value) if user_value else get_dynamic_batch_size()
            if batch_size <= 0:
                raise ValueError
        except Exception:
            batch_size = get_dynamic_batch_size()
            status_label.config(text=f"⚠️ Invalid batch size. Using dynamic batch size: {batch_size}")

        # Read UI values ON THE MAIN THREAD before launching worker
        output_dir = output_dir_var.get().strip() if output_dir_var else ""
        inference_res_text = inference_res_var.get().strip() if inference_res_var else ""
        invert_value = bool(invert_var.get()) if invert_var else False
        ffmpeg_codec = codec_var.get().strip() if codec_var else ""
        inference_steps_value = inference_steps_entry.get().strip() if inference_steps_entry else ""

        threading.Thread(
            target=process_video2,
            args=(
                file_path,
                total_frames_all,
                0,
                batch_size,
                output_dir,
                inference_res_text,
                status_label,
                progress_bar,
                cancel_requested,
                invert_value,
                ffmpeg_codec,
                inference_steps_value,
            ),
            kwargs={
                "target_fps": -1,
                "ignore_letterbox_bars": False,
                "prefer_opencv_writer": False,
            },
            daemon=True,
        ).start()
        
def _log_ex(exctype, value, tb):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open("vd3d_crash.log", "a", encoding="utf-8") as f:
        f.write(f"\n=== {ts} ===\n")
        traceback.print_exception(exctype, value, tb, file=f)
    print("💥 Unhandled exception; see vd3d_crash.log")

sys.excepthook = _log_ex

if hasattr(threading, "excepthook"):
    def _thread_hook(args):
        _log_ex(args.exc_type, args.exc_value, args.exc_traceback)
    threading.excepthook = _thread_hook

