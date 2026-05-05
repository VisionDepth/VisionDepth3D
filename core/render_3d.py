# render_3D.py
import os, platform, warnings
import time
import cv2
import torch
import numpy as np
import subprocess
import threading
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import onnxruntime as ort
import torch.nn.functional as F
from collections import deque
from scipy.ndimage import gaussian_filter
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur
from core.ffmpeg_blackdetect import detect_black_white_frames
import math
from typing import Iterable, Optional
import platform


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
def pick_torch_device():
    # NVIDIA CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # AMD ROCm (Linux) — presents as 'cuda' via HIP
    try:
        if hasattr(torch, 'hip') and torch.hip.is_available():
            return torch.device("cuda")
    except Exception:
        pass

    # AMD / Intel / Any GPU via DirectML (Windows)
    try:
        import torch_directml
        dml_device = torch_directml.device()
        # Quick test to verify the device works
        _ = torch.zeros(1, device=dml_device)
        return dml_device
    except Exception:
        pass

    # macOS Metal (Apple Silicon / AMD)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # CPU fallback
    return torch.device("cpu")

torch_device = pick_torch_device()
print(f"3D Pipeline running on Torch device: {torch_device.type.upper()}")

# Load ONNX model
#MODEL_PATH = 'weights/backward_warping_model.onnx'
#session = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
#input_name = session.get_inputs()[0].name
#output_name = session.get_outputs()[0].name
#print(f"✅ Loaded ONNX model from {MODEL_PATH} on {onnx_device}")

#Global flags
suspend_flag = threading.Event()
cancel_flag = threading.Event()
process_thread = None 
global_session_start_time = None
ENABLE_DEPTH_ROTO = True           # master toggle
ROTO_NEAR = 1.0                    # 1.0 = screen-near white
ROTO_FAR  = 0.45                   # how dark at edges of subject
ROTO_FEATHER_PX = 12               # edge softness
ROTO_ROUND_GAMMA = 1.2             # >1.0 = rounder center
ROTO_EMA_ALPHA = 0.88              # temporal matte smoothing
ROTO_MASK_DIR = None               # e.g., "mattes/" (PNG per frame) or None if you auto-seg
# Dynamic Floating Window tuning
DFW_MIN_PARALLAX     = 0.010   # do not show any bar below this offset
DFW_MAX_BAR_FRAC     = 0.07    # max bar width as fraction of per-eye width (about 7 percent)
DFW_WIDTH_EASE       = 0.90    # how much to keep previous width (0.9 = very smooth)
DFW_PARALLAX_WEIGHT  = 0.65    # how much the actual parallax drives the bar
DFW_DEPTH_WEIGHT     = 0.35    # how much subject depth offset from mid drives it
DFW_USE_FADE         = True    # use faded mask instead of solid black

SETTINGS_FILE = "settings.json"

# Common Aspect Ratios
aspect_ratios = {
    "Default (16:9)": 16 / 9,
    "CinemaScope (2.39:1)": 2.39,
    "21:9 UltraWide": 21 / 9,
    "4:3 (Classic Films)": 4 / 3,
    "1:1 (Square)": 1 / 1,
    "2.35:1 (Classic Cinematic)": 2.35,
    "2.76:1 (Ultra-Panavision)": 2.76,
}

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

VR180_EQUI_PRESETS = {
    "2048x1024 (Per Eye)": (2048, 1024),
    "3072x1536 (Per Eye)": (3072, 1536),
    "3840x1920 (Per Eye)": (3840, 1920),
    "4096x2048 (Per Eye)": (4096, 2048),
    "5760x2880 (Per Eye)": (5760, 2880),
}

VR180_FLAT_PRESETS = {
    "1280x720 (Working)": (1280, 720),
    "1920x1080 (Working)": (1920, 1080),
    "2560x1440 (Working)": (2560, 1440),
}

def get_video_info_safe(video_path):
    """
    Returns (width, height, fps) using OpenCV first, then ffprobe fallback.
    """
    width = 0
    height = 0
    fps = 0.0

    cap = cv2.VideoCapture(video_path)
    try:
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    # If OpenCV failed, fall back to ffprobe
    if width <= 0 or height <= 0 or fps <= 0:
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=0",
                video_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                **hidden_subprocess_kwargs(),
            )
            info = {}

            for line in result.stdout.splitlines():
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    info[k] = v

            if width <= 0:
                width = int(info.get("width", 0) or 0)
            if height <= 0:
                height = int(info.get("height", 0) or 0)

            def parse_rate(rate_str):
                if not rate_str or rate_str == "0/0":
                    return 0.0
                if "/" in rate_str:
                    a, b = rate_str.split("/", 1)
                    a = float(a)
                    b = float(b)
                    return a / b if b != 0 else 0.0
                return float(rate_str)

            if fps <= 0:
                fps = parse_rate(info.get("avg_frame_rate", "")) or parse_rate(info.get("r_frame_rate", ""))

        except Exception as e:
            print(f"⚠️ ffprobe fallback failed for {video_path}: {e}")

    return width, height, fps

def merge_audio_from_source(final_video, original_video, output_with_audio, start_s=None):
    """
    Muxes the original audio track into the final 3D render without re-encoding.
    Uses two-pass seek to avoid frozen frames at the start.
    """
    if not os.path.exists(original_video) or not os.path.exists(final_video):
        return final_video

    cmd = [
        "ffmpeg", "-y",
        "-i", final_video,
    ]

    # Two-pass seek on the original: fast keyframe seek then accurate decode seek
    if start_s is not None and start_s > 0:
        cmd += ["-ss", str(start_s)]  # fast seek before input
        cmd += ["-i", original_video]
        cmd += ["-ss", str(start_s)]  # accurate seek after input (on audio)
    else:
        cmd += ["-i", original_video]

    cmd += [
        "-map", "0:v:0",      # video from rendered file
        "-map", "1:a:0?",     # audio from original
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        "-fflags", "+shortest",
        output_with_audio
    ]

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **hidden_subprocess_kwargs(),
    )

    if process.returncode != 0:
        print(f"[AUDIO MERGE] ffmpeg failed (code {process.returncode}):")
        print(process.stderr[:500])
        return final_video

    if os.path.exists(output_with_audio) and os.path.getsize(output_with_audio) > 1000:
        try:
            os.remove(final_video)
        except Exception:
            pass
        return output_with_audio

    return final_video



def ffmpeg_rgb48_reader(path, width, height, start_s=None, end_s=None):
    """
    Decode video frames to RGB 16-bit (rgb48le) while explicitly preserving HDR signaling.
    This avoids FFmpeg doing implicit/guessed colorspace conversions on HDR10 sources.

    Returns frames as float32 RGB in [0,1] (still PQ-encoded values, not tonemapped).
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

    # Seek before input for speed (keyframe seek). If you need exact frame-accurate
    # seeking, do a second -ss after -i, but this is usually fine for rendering.
    if start_s is not None:
        cmd += ["-ss", str(float(start_s))]

    cmd += ["-i", path]

    # Clip window
    if end_s is not None and start_s is not None:
        dur = max(0.0, float(end_s) - float(start_s))
        cmd += ["-t", str(dur)]
    elif end_s is not None:
        cmd += ["-to", str(float(end_s))]

    # Force HDR colorspace handling so FFmpeg doesn't guess:
    # - zscale sets primaries/transfer/matrix and preserves PQ/BT.2020
    # - npl=1000 sets nominal peak luminance (helps prevent weird scaling)
    # - format=rgb48le ensures 16-bit RGB output
    vf = (
        "zscale=primaries=bt2020:transfer=smpte2084:matrix=bt2020nc:"
        "range=tv:npl=1000,format=rgb48le"
    )

    cmd += [
        "-an", "-sn", "-dn",
        "-vf", vf,
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",
        "-vsync", "0",
        "-"
    ]

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        bufsize=10**7,
        **hidden_subprocess_kwargs(),
    )
    frame_bytes = int(width) * int(height) * 3 * 2  # 3 channels * 16-bit

    try:
        while True:
            buf = p.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, dtype=np.uint16).reshape((height, width, 3))
            # Still PQ-encoded, just higher precision; do not tonemap here.
            yield (arr.astype(np.float32) / 65535.0).clip(0.0, 1.0)
    finally:
        try:
            if p.stdout:
                p.stdout.close()
        except Exception:
            pass
        p.wait()
        # Optional: surface decode errors
        if p.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg_rgb48_reader: ffmpeg exited with code {p.returncode}")



def ffmpeg_yuv10_reader(path, width, height):
    """
    Yields P010LE frames as float32 RGB in [0,1] with simple 10-bit scaling.
    NOTE: stays in PQ/BT.2020 space; do *not* tone-map to SDR.
    """
    cmd = [
        "ffmpeg","-loglevel","error",
        "-i", path,
        "-f","rawvideo",
        "-pix_fmt","p010le",   # 10-bit 4:2:0
        "-"
    ]
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        **hidden_subprocess_kwargs(),
    )
    stride = width * height * 2 * 3 // 2  # P010 size
    while True:
        buf = p.stdout.read(stride)
        if not buf or len(buf) < stride:
            break
        yuv = np.frombuffer(buf, dtype=np.uint16)
        # reshape to planar P010 (Y full res, UV half res)
        y = (yuv[:width*height].reshape((height, width)) >> 6).astype(np.float32) / 1023.0
        uv = (yuv[width*height:].reshape((height//2, width)) >> 6).astype(np.float32) / 1023.0
        u = uv[:, 0::2]; v = uv[:, 1::2]
        # upsample chroma (nearest is fine here)
        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
        # very simple YUV->RGB for BT.2020 (non-constant luminance)
        # keep in PQ domain (no tone map)
        r = y + 1.4746*(v-0.5)
        g = y - 0.16455*(u-0.5) - 0.57135*(v-0.5)
        b = y + 1.8814*(u-0.5)
        rgb = np.stack([r,g,b], axis=2).clip(0,1).astype(np.float32)
        yield rgb
    p.stdout.close(); p.wait()


def reset_render_state():
    # reset shift EMA
    if hasattr(pixel_shift_cuda, "_shift_ema"):
        pixel_shift_cuda._shift_ema = None

    # reset floating window tracker
    if "floating_window_tracker" in globals():
        floating_window_tracker.prev_offset = 0.0
        floating_window_tracker.frame_counter = 0

    # reset DFW easing
    for k in ("dfw_last_side", "dfw_last_width"):
        if k in globals():
            del globals()[k]

    # reset depth percentile EMA so it learns per render
    global depth_ema_norm
    depth_ema_norm = DepthPercentileEMA(p_lo=0.02, p_hi=0.98, alpha=0.82)

def sculpt_depth_u8(base_depth_u8, mask_u8, *,
                    near=1.0, far=0.4,
                    feather_px=12, round_gamma=1.2):
    """
    base_depth_u8: uint8 [H,W] 0..255 (white = near)
    mask_u8      : uint8 [H,W] 0/255  (255 = inside subject)
    Returns uint8 [H,W] depth with a rounded subject profile blended in.
    """
    mask = (mask_u8 > 127).astype(np.uint8)

    # distance to edge (inside/outside)
    dist_in  = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)

    max_in = max(1.0, float(dist_in.max()))
    r = np.power(np.clip(dist_in / max_in, 0, 1), round_gamma)  # 0 edge → 1 center

    subj = (near * r + far * (1 - r)) * 255.0
    subj_u8 = subj.astype(np.uint8)

    # feather alpha: 0 outside → 1 inside
    alpha = np.clip(dist_out / float(max(1, feather_px)), 0, 1)
    alpha = (1.0 - alpha)  # 1 at subject center, ~0 outside
    alpha3 = alpha  # depth is single-channel

    out = (alpha3 * subj_u8 + (1 - alpha3) * base_depth_u8).astype(np.uint8)
    return out

class MatteEMA:
    """Stabilize matte edges over time to avoid shimmer."""
    def __init__(self, alpha=0.85):
        self.prev = None
        self.alpha = alpha
    def step(self, mask_u8):
        if self.prev is None:
            self.prev = mask_u8.astype(np.float32) / 255.0
        cur = mask_u8.astype(np.float32) / 255.0
        self.prev = self.alpha * self.prev + (1 - self.alpha) * cur
        return (np.clip(self.prev, 0, 1) * 255).astype(np.uint8)


import math
import torch
import torch.nn.functional as F

def build_vr180_equirect_grid(
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
    src_hfov_deg: float = 110.0,
):
    """
    Builds a grid for warping a rectilinear source (normal flat view) into
    a 180-degree equirectangular image (half sphere).

    Equirect domain:
      lon in [-pi/2, +pi/2] across width
      lat in [-pi/2, +pi/2] across height

    Source model:
      simple pinhole perspective with horizontal FOV = src_hfov_deg
      vertical FOV derived from aspect

    Returns:
      grid: [1, out_h, out_w, 2] in grid_sample coords [-1..1]
      valid: [1, 1, out_h, out_w] mask (1 where samples are in front and in bounds)
    """
    device = torch_device if "torch_device" in globals() else "cuda" if torch.cuda.is_available() else "cpu"

    src_w = int(src_w); src_h = int(src_h)
    out_w = int(out_w); out_h = int(out_h)

    # Equirect UV
    u = torch.linspace(0.0, 1.0, out_w, device=device)
    v = torch.linspace(0.0, 1.0, out_h, device=device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")  # [H,W]

    # 180 equirect angles
    lon = (uu - 0.5) * math.pi            # [-pi/2..+pi/2]
    lat = (0.5 - vv) * math.pi            # [+pi/2..-pi/2]

    # Direction vector on unit sphere (camera forward = +Z)
    cos_lat = torch.cos(lat)
    x = cos_lat * torch.sin(lon)
    y = torch.sin(lat)
    z = cos_lat * torch.cos(lon)

    # Perspective projection: x_img = x/z, y_img = y/z
    # Reject anything behind the camera or too close to z=0
    eps = 1e-6
    z_safe = torch.clamp(z, min=eps)
    x_img = x / z_safe
    y_img = y / z_safe

    # FOV mapping
    hfov = math.radians(float(src_hfov_deg))
    hfov = max(min(hfov, math.radians(170.0)), math.radians(10.0))

    # derive vfov from aspect (basic pinhole)
    aspect = src_w / max(src_h, 1)
    vfov = 2.0 * math.atan(math.tan(hfov * 0.5) / max(aspect, 1e-6))

    tan_h = math.tan(hfov * 0.5)
    tan_v = math.tan(vfov * 0.5)

    # Convert to normalized grid_sample coords [-1..1]
    gx = (x_img / tan_h).clamp(-2.0, 2.0)
    gy = (y_img / tan_v).clamp(-2.0, 2.0)

    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1,out_h,out_w,2]

    # Valid mask: in front (z>0) and inside sampling bounds (abs<=1)
    in_front = (z > 0.0).float()
    in_bounds = ((gx.abs() <= 1.0) & (gy.abs() <= 1.0)).float()
    valid = (in_front * in_bounds).unsqueeze(0).unsqueeze(0)  # [1,1,out_h,out_w]

    return grid, valid


def warp_eye_to_vr180_equirect(
    eye_rgb_t: torch.Tensor,   # [3,H,W] float 0..1
    grid: torch.Tensor,        # [1,out_h,out_w,2]
    valid: torch.Tensor,       # [1,1,out_h,out_w]
):
    """
    Warps one eye into VR180 equirect. Keeps everything in float on GPU.
    """

    # ✅ FIX: flip vertical axis (grid_sample uses inverted Y)
    grid = grid.clone()
    grid[..., 1] *= -1

    x = F.grid_sample(
        eye_rgb_t.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )  # [1,3,out_h,out_w]

    # Apply validity mask to hard-black outside the view cone
    x = x * valid
    return x.squeeze(0).clamp(0.0, 1.0)
    
def pack_stereo_tb(left_t: torch.Tensor, right_t: torch.Tensor) -> torch.Tensor:
    # [3,H,W] + [3,H,W] -> [3,2H,W]
    return torch.cat([left_t, right_t], dim=1)

def pack_stereo_sbs(left_t: torch.Tensor, right_t: torch.Tensor) -> torch.Tensor:
    # [3,H,W] + [3,H,W] -> [3,H,2W]
    return torch.cat([left_t, right_t], dim=2)

def parse_timecode(s: str | None) -> float | None:
    """
    'HH:MM:SS', 'MM:SS', 'SS', with optional '.ms'
    Returns seconds as float, or None if blank/invalid.
    """
    if not s or not str(s).strip():
        return None
    s = s.strip()
    # allow H:M:S(.ms) or M:S(.ms) or S(.ms)
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
            return h*3600 + m*60 + sec
        elif len(parts) == 2:
            m = float(parts[0]); sec = float(parts[1])
            return m*60 + sec
        else:
            return float(s)
    except Exception:
        return None



def pad_to_aspect_ratio(image, target_width, target_height, bg_color=(0, 0, 0)):
    """
    Pads the input image to the target resolution without stretching,
    preserving aspect ratio.
    """
    
    h, w = image.shape[:2]
    target_aspect = target_width / target_height
    current_aspect = w / h

    # Step 1: Resize to fit within target while preserving aspect
    if current_aspect > target_aspect:
        # Image is wider than target → match width
        new_w = target_width
        new_h = int(target_width / current_aspect)
    else:
        # Image is taller → match height
        new_h = target_height
        new_w = int(current_aspect * target_height)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 2: Create padded canvas
    padded = np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)

    # Step 3: Center it
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded


# Converters
def frame_to_tensor(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
    return frame_tensor.to(torch_device)

def depth_to_tensor(depth_frame, invert_depth=True):
    depth_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
    depth_tensor = torch.from_numpy(depth_gray).float().unsqueeze(0) / 255.0

    if invert_depth:
        depth_tensor = 1.0 - depth_tensor

    return depth_tensor.to(torch_device)

@torch.no_grad()
def estimate_subject_depth(depth_tensor: torch.Tensor) -> torch.Tensor:
    d = depth_tensor.clamp(0.0, 1.0)
    device = d.device
    _, H, W = d.shape

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )

    center_w = torch.exp(-0.5 * ((yy / 0.55) ** 2 + (xx / 0.70) ** 2))

    dx = F.pad(d[:, :, 1:] - d[:, :, :-1], (1, 0))
    dy = F.pad(d[:, 1:, :] - d[:, :-1, :], (0, 0, 1, 0))
    grad = torch.sqrt(dx.pow(2) + dy.pow(2)).squeeze(0)
    smooth_w = 1.0 - torch.sigmoid(10.0 * (grad - 0.025))

    w = center_w * smooth_w
    vals = d.squeeze(0).reshape(-1)
    weights = w.reshape(-1)

    # nearer-biased weighted percentile, since "subject" is often nearer than background
    sort_idx = torch.argsort(vals)
    vals_sorted = vals[sort_idx]
    w_sorted = weights[sort_idx]
    cdf = torch.cumsum(w_sorted, dim=0) / (w_sorted.sum() + 1e-8)

    # 35th percentile in white-near convention
    idx = torch.searchsorted(cdf, torch.tensor(0.35, device=device))
    idx = torch.clamp(idx, 0, vals_sorted.numel() - 1)
    return vals_sorted[idx]



def enhance_curvature(depth_tensor, strength=0.15):
    """
    Adds a 2D curvature profile to simulate facial/body roundness.
    """
    B, H, W = depth_tensor.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=depth_tensor.device),
        torch.linspace(-1, 1, W, device=depth_tensor.device),
        indexing="ij"
    )
    curvature = 1 - (xx**2 + yy**2)  # peak in center
    curve = curvature.unsqueeze(0).expand(B, -1, -1)
    return depth_tensor + (curve * strength)


# Bilateral smoothing for depth (preserves edges)
def bilateral_smooth_depth(depth_tensor):
    depth_np = depth_tensor.squeeze().cpu().numpy().astype(np.uint8)
    smoothed = cv2.bilateralFilter(depth_np, d=9, sigmaColor=75, sigmaSpace=75)
    smoothed_tensor = torch.from_numpy(smoothed).float().unsqueeze(0) / 255.0
    return smoothed_tensor.to(depth_tensor.device)

# Gradient-aware shift suppression
def suppress_artifacts_with_edge_mask(depth_tensor, total_shift, feather_strength=10.0, edge_threshold=0.02):
    """
    Suppress pixel shift artifacts near sharp depth edges (hair, limbs).
    Returns a softly masked version of total_shift using adaptive edge gradient detection.
    """
    
    dx = torch.abs(F.pad(depth_tensor[:, :, 1:] - depth_tensor[:, :, :-1], (1, 0)))
    dy = torch.abs(F.pad(depth_tensor[:, 1:, :] - depth_tensor[:, :-1, :], (0, 0, 1, 0)))
    grad_mag = torch.sqrt(dx ** 2 + dy ** 2)
  
    edge_mask = torch.sigmoid((grad_mag - edge_threshold) * feather_strength * 5)  # [0, 1]

    smooth_mask = 1.0 - edge_mask
    smooth_mask = F.avg_pool2d(smooth_mask.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)

    return total_shift * smooth_mask

class TemporalDepthFilter:
    def __init__(self, alpha=0.85):
        self.prev_depth = None
        self.alpha = alpha

    def smooth(self, curr_depth):
        if self.prev_depth is None:
            self.prev_depth = curr_depth.clone()
        self.prev_depth = self.alpha * self.prev_depth + (1 - self.alpha) * curr_depth
        return self.prev_depth

class DepthPercentileEMA:
    def __init__(self, p_lo=0.02, p_hi=0.98, alpha=0.90):
        self.p_lo = p_lo
        self.p_hi = p_hi
        self.alpha = alpha
        self._lo = None
        self._hi = None

    def normalize(self, depth_01: torch.Tensor):
        """
        depth_01: [1, H, W] in [0,1] (roughly). Returns normalized depth in [0,1].
        Uses EMA of low/high percentiles to keep range stable across frames.
        """
        assert depth_01.dim() == 3 and depth_01.shape[0] == 1
        d = depth_01.clamp(0, 1)
        
        lo = torch.quantile(d, self.p_lo)
        hi = torch.quantile(d, self.p_hi)
        
        if (hi - lo) < 1e-5:
            return d

        if self._lo is None:
            self._lo, self._hi = lo.detach(), hi.detach()
        else:
            self._lo = self.alpha * self._lo + (1 - self.alpha) * lo.detach()
            self._hi = self.alpha * self._hi + (1 - self.alpha) * hi.detach()

        out = (d - self._lo) / (self._hi - self._lo + 1e-6)
        return out.clamp(0, 1)


def midtone_shape(depth_01: torch.Tensor, gamma=0.85):
    """
    Gentle power curve to allocate more disparity to mid-depths.
    gamma < 1.0 -> more near/mid pop; 0.80–0.95 range is typical.
    """
    return depth_01.clamp(0, 1).pow(gamma)


class ConvergenceEMA:
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.val = None
    def update(self, x):
        self.val = x if self.val is None else (self.alpha * self.val + (1 - self.alpha) * x)
        return self.val


class SubjectDepthEMA:
    def __init__(self, alpha=0.95):
        self.val = None
        self.alpha = alpha
    def update(self, x):
        if self.val is None:
            self.val = x
        else:
            self.val = self.alpha * self.val + (1 - self.alpha) * x
        return self.val

subject_depth_ema = SubjectDepthEMA(alpha=0.80)
depth_ema_norm = DepthPercentileEMA(p_lo=0.02, p_hi=0.98, alpha=0.82)
conv_ema = ConvergenceEMA(alpha=0.90)
MID_GAMMA = 0.90  # 0.80–0.95 works well

def frame16_to_tensor(rgb_float_01):
    """
    rgb_float_01: [H,W,3] float32 0..1 in RGB (PQ-encoded values, but high precision)
    Returns torch [3,H,W] float32 0..1 on device.
    """
    t = torch.from_numpy(rgb_float_01).float().permute(2, 0, 1).contiguous()
    return t.to(torch_device)

def tensor_to_rgb48_bytes(rgb_tensor):
    """
    rgb_tensor: torch [3,H,W] float in [0,1]
    Returns bytes for rgb48le (uint16 little-endian).
    """
    x = rgb_tensor.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    u16 = (x * 65535.0 + 0.5).astype(np.uint16)
    return u16.tobytes()

import torch.nn.functional as F

def tensor_pad_to_aspect_ratio(rgb_t, target_width, target_height):
    """
    Torch equivalent of pad_to_aspect_ratio()
    rgb_t: [3,H,W] RGB float in 0..1
    Returns [3,target_height,target_width]
    """

    C, h, w = rgb_t.shape
    target_aspect = target_width / target_height
    current_aspect = w / h

    # Step 1: resize to fit while preserving aspect
    if current_aspect > target_aspect:
        # wider → match width
        new_w = target_width
        new_h = int(target_width / current_aspect)
    else:
        # taller → match height
        new_h = target_height
        new_w = int(current_aspect * target_height)

    resized = F.interpolate(
        rgb_t.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    # Step 2: padded canvas (black)
    padded = torch.zeros(
        (3, target_height, target_width),
        device=rgb_t.device,
        dtype=rgb_t.dtype
    )

    # Step 3: center it
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    padded[:, y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded.clamp(0.0, 1.0)

def tensor_sharpen(rgb_t, factor=0.0):
    # factor 0 = no sharpen
    if factor <= 1e-6:
        return rgb_t
    # unsharp-ish kernel (simple and stable)
    # conv2d expects [N,C,H,W]
    k = torch.tensor([[0, -1, 0],
                      [-1, 5.0 + float(factor), -1],
                      [0, -1, 0]], device=rgb_t.device, dtype=rgb_t.dtype).view(1,1,3,3)
    x = rgb_t.unsqueeze(0)  # [1,3,H,W]
    # apply per-channel by groups=3
    k3 = k.repeat(3, 1, 1, 1)  # [3,1,3,3]
    y = F.conv2d(x, k3, padding=1, groups=3)
    return y.squeeze(0).clamp(0.0, 1.0)

def tensor_apply_side_mask(rgb_t, side="left", width=40, solid_black=True, fade=False):
    if width <= 0:
        return rgb_t
    C, H, W = rgb_t.shape
    w = min(int(width), W)
    mask = torch.ones((1, H, W), device=rgb_t.device, dtype=rgb_t.dtype)

    if solid_black:
        if side == "left":
            mask[:, :, :w] = 0
        else:
            mask[:, :, W-w:] = 0
    else:
        if fade:
            ramp = torch.linspace(0, 1, w, device=rgb_t.device, dtype=rgb_t.dtype)
            if side == "left":
                mask[:, :, :w] = ramp.view(1, 1, w)
            else:
                mask[:, :, W-w:] = ramp.flip(0).view(1, 1, w)
        else:
            if side == "left":
                mask[:, :, :w] = 0
            else:
                mask[:, :, W-w:] = 0

    return (rgb_t * mask).clamp(0.0, 1.0)

def format_3d_output_torch(left_t, right_t, fmt):
    # left_t/right_t: [3,H,W]
    if fmt in ("Half-SBS", "Full-SBS", "VR"):
        return torch.cat([left_t, right_t], dim=2)  # SBS
    elif fmt == "Passive Interlaced":
        out = left_t.clone()
        out[:, 1::2, :] = right_t[:, 1::2, :]
        return out
    else:
        return torch.cat([left_t, right_t], dim=2)

def tensor_to_frame(tensor):
    frame_cpu = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2BGR)

def detect_black_bars(
    frame_tensor: torch.Tensor,
    threshold: float = 8.0,       # brightness threshold in 0–255 space
    min_bar_height: int = 8,      # ignore tiny bands
    overscan_px: int = 2          # crop a bit *past* the detected edge
):
    """
    Detects top and bottom black bars on a [3, H, W] tensor (0..1 floats).

    Returns (top_crop, bottom_crop) in pixels. We:
      * work in luma (average over channels)
      * scan from top and bottom until rows get brighter than `threshold`
      * only accept bars at least `min_bar_height` high
      * overshoot by `overscan_px` so we remove the transition line too
    """
    if frame_tensor.dim() != 3:
        raise ValueError(f"Expected [3, H, W] tensor, got {frame_tensor.shape}")

    _, H, W = frame_tensor.shape

    # grayscale-ish: average over channels -> [H, W] in 0..1
    gray = frame_tensor.mean(dim=0)
    # mean brightness per row in 0..255
    row_means = (gray.mean(dim=1) * 255.0).cpu()

    # scan from top
    top_idx = 0
    while top_idx < H // 2 and row_means[top_idx] < threshold:
        top_idx += 1

    # scan from bottom
    bottom_idx = H - 1
    while bottom_idx > H // 2 and row_means[bottom_idx] < threshold:
        bottom_idx -= 1

    raw_top_bar = top_idx
    raw_bottom_bar = (H - 1) - bottom_idx

    # If bars are tiny or basically not there, skip cropping
    if raw_top_bar < min_bar_height and raw_bottom_bar < min_bar_height:
        return 0, 0

    # Overscan a couple of pixels inside the picture to kill the bright line
    top_crop = max(0, raw_top_bar - overscan_px)
    bottom_crop = max(0, raw_bottom_bar - overscan_px)

    # Safety: never crop almost everything away
    if top_crop + bottom_crop > H - 16:
        return 0, 0

    return int(top_crop), int(bottom_crop)



def crop_black_bars_torch(frame_tensor, cached_crop=None, threshold=10):
    """
    Crops black bars using cached detection (only detect once).
    - frame_tensor: shape [3, H, W]
    - cached_crop: optional (top, bottom) tuple for reuse
    """
    # If we already have a cached crop, reuse it
    if cached_crop is not None:
        top, bottom = cached_crop
    else:
        top, bottom = detect_black_bars(frame_tensor, threshold)

    if top + bottom >= frame_tensor.shape[1]:
        return frame_tensor, (0, 0)

    cropped = frame_tensor[:, top:frame_tensor.shape[1] - bottom, :]
    return cropped, (top, bottom)


def feather_shift_edges(
    shifted_tensor: torch.Tensor,
    original_tensor: torch.Tensor,
    occ_mask: torch.Tensor | None = None,
    enable_feathering: bool = True
) -> torch.Tensor:
    """
    Blend only in likely disocclusion / warp-stress regions.
    occ_mask should be [H, W] or [1, H, W], with values in [0,1].
    """
    assert shifted_tensor.shape == original_tensor.shape, "Shape mismatch"

    if not enable_feathering or occ_mask is None:
        return shifted_tensor

    if occ_mask.dim() == 2:
        occ_mask = occ_mask.unsqueeze(0)   # [H,W] -> [1,H,W]

    if occ_mask.shape[0] == 1:
        blend_mask = occ_mask.repeat(3, 1, 1)  # [1,H,W] -> [3,H,W]
    else:
        blend_mask = occ_mask

    min_h = min(shifted_tensor.shape[1], blend_mask.shape[1])
    min_w = min(shifted_tensor.shape[2], blend_mask.shape[2])

    blend_mask     = blend_mask[:, :min_h, :min_w]
    shifted_tensor = shifted_tensor[:, :min_h, :min_w]
    original_tensor = original_tensor[:, :min_h, :min_w]

    output_tensor = shifted_tensor * (1.0 - blend_mask) + original_tensor * blend_mask
    return output_tensor.clamp(0.0, 1.0)


def shift_mask(mask_tensor, shift_vals, width):
    H, W = mask_tensor.shape[-2:]

    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

    N, C, H, W = mask_tensor.shape

    # Create grid
    x = torch.linspace(-1, 1, W, device=mask_tensor.device)
    y = torch.linspace(-1, 1, H, device=mask_tensor.device)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((x_grid, y_grid), dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(N, H, W, 2)  # [N, H, W, 2]

    if shift_vals.dim() == 2:
        shift_vals = shift_vals.unsqueeze(0)  # [1, H, W]

    # Make sure shift_vals match batch size
    shift_vals = shift_vals.expand(N, H, W)

    # ✅ Scale shift_vals to grid units
    shift_vals_grid = (shift_vals / (W / 2)).clamp(-1.0, 1.0)  # IMPORTANT ⚡

    grid[..., 0] -= shift_vals_grid  # apply scaled shift

    warped = F.grid_sample(
        mask_tensor, grid,
        mode='bilinear', padding_mode='border', align_corners=True
    )

    return warped.squeeze(0)  # Remove batch dimension

def compute_dynamic_parallax_scale(depth_tensor, min_scale=0.6, max_scale=1.0):
    """
    Adaptive parallax control based on normalized depth variance in center view.
    Returns a scalar float.
    """
    _, H, W = depth_tensor.shape
    center_crop = depth_tensor[:, H//4:H*3//4, W//4:W*3//4]

    # Normalize variance by mean to handle different scene scales
    mean_depth = torch.mean(center_crop)
    variance = torch.var(center_crop)
    norm_var = (variance / (mean_depth + 1e-5)).clamp(0.0, 1.0)

    # Map normalized variance to a smooth parallax scale
    scale = min_scale + (norm_var * (max_scale - min_scale))
    return scale.item()


# --- Enhanced Healing of Warped Areas ---
def heal_missing_pixels(warped_frame, warped_depth, original_frame, edge_mask, heal_strength=0.5):
    """
    Heals gaps after convergence shifting based on depth edges and warp mask,
    with optional selective softening for invisible healing.
    """
    device = warped_frame.device
    warped_gray = warped_frame.mean(dim=0, keepdim=True)  # average over channels
    grad_x = F.pad(warped_gray[:, :, 1:] - warped_gray[:, :, :-1], (1, 0))
    grad_y = F.pad(warped_gray[:, 1:, :] - warped_gray[:, :-1, :], (0, 0, 1, 0))
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    threshold = 0.05  # 🔥 Tune if needed
    missing_mask = (grad_mag > threshold).float()
    missing_mask = F.avg_pool2d(missing_mask.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)
    missing_mask = missing_mask.clamp(0, 1)

    if edge_mask is not None:
        missing_mask = torch.max(missing_mask, edge_mask)

    missing_mask = missing_mask.expand_as(warped_frame)  # [3, H, W]

    healed = (1.0 - heal_strength * missing_mask) * warped_frame + heal_strength * missing_mask * original_frame

    soft_blur = F.avg_pool2d(healed.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
    healed = (1.0 - 0.3 * missing_mask) * healed + 0.3 * missing_mask * soft_blur

    return healed.clamp(0, 1)

# Shift Smoother
class ShiftSmoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev_fg_shift = None
        self.prev_mg_shift = None
        self.prev_bg_shift = None

    def smooth(self, fg_shift, mg_shift, bg_shift):
        if self.prev_fg_shift is None:
            self.prev_fg_shift, self.prev_mg_shift, self.prev_bg_shift = fg_shift, mg_shift, bg_shift
        else:
            self.prev_fg_shift = self.alpha * fg_shift + (1 - self.alpha) * self.prev_fg_shift
            self.prev_mg_shift = self.alpha * mg_shift + (1 - self.alpha) * self.prev_mg_shift
            self.prev_bg_shift = self.alpha * bg_shift + (1 - self.alpha) * self.prev_bg_shift
        return self.prev_fg_shift, self.prev_mg_shift, self.prev_bg_shift
    
class FloatingWindowTracker:
    def __init__(self, alpha=0.85):
        self.prev_offset = 0.0
        self.alpha = alpha
        self.frame_counter = 0  # 🆕 Add a counter

    def smooth_offset(self, current_offset, threshold=0.002):
        delta = abs(current_offset - self.prev_offset)
        if delta < threshold:
            return self.prev_offset  # ignore tiny jitter

        self.prev_offset = self.alpha * self.prev_offset + (1 - self.alpha) * current_offset
        self.frame_counter += 1  # 🆕 Increment each call

        # 🆕 Every 100 updates, clamp to avoid precision drift
        if self.frame_counter >= 100:
            self.prev_offset = max(min(self.prev_offset, 1.0), -1.0)  # clamp to [-1, +1]
            self.frame_counter = 0

        return self.prev_offset

floating_window_tracker = FloatingWindowTracker(alpha=0.88)

class FloatingBarEaser:
    def __init__(self, alpha=0.95):
        self.prev_bar_width = 0
        self.alpha = alpha

    def ease(self, current_width):
        self.prev_bar_width = int(self.alpha * self.prev_bar_width + (1 - self.alpha) * current_width)
        return self.prev_bar_width

bar_easer = FloatingBarEaser(alpha=0.85)

# === POP CURVE HELPERS ===

def _signed_pow(x: torch.Tensor, gamma: float):
    # symmetric contrast around 0
    return torch.sign(x) * (torch.abs(x) ** gamma)

@torch.no_grad()
def shape_depth_for_pop(
    depth_01,
    subject_depth,
    *,
    stretch_lo=0.05,
    stretch_hi=0.95,
    depth_mid=0.50,
    gamma=0.85,
    recenter_strength=0.35,
):
    d = depth_01.clamp(0, 1)
    lo = torch.quantile(d, stretch_lo)
    hi = torch.quantile(d, stretch_hi)

    if (hi - lo) < 1e-5:
        d_stretched = d
    else:
        d_stretched = ((d - lo) / (hi - lo + 1e-6)).clamp(0, 1)

    subj_stretched = ((subject_depth - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    delta = (depth_mid - subj_stretched) * recenter_strength
    centered = (d_stretched + delta).clamp(0, 1)

    shaped = _signed_pow(centered - depth_mid, gamma) + depth_mid
    return shaped.clamp(0, 1)

def compute_occlusion_mask_from_shift(
    shift_vals: torch.Tensor,
    blur_ksize: int = 7,
    occ_threshold: float = 0.02,
    occ_strength: float = 8.0,
    max_mask: float = 0.30,
) -> torch.Tensor:
    """
    Builds a soft mask from displacement gradients instead of depth gradients.
    shift_vals: [H, W] or [1, H, W], normalized grid shift.
    Returns [H, W] in [0, max_mask].
    """
    if shift_vals.dim() == 3:
        shift_vals = shift_vals.squeeze(0)

    shift_grad_x = torch.abs(F.pad(shift_vals[:, 1:] - shift_vals[:, :-1], (1, 0)))
    shift_grad_y = torch.abs(F.pad(shift_vals[1:, :] - shift_vals[:-1, :], (0, 0, 1, 0)))

    occ_mask = torch.clamp(
        (shift_grad_x + shift_grad_y - occ_threshold) * occ_strength,
        0.0,
        1.0
    )

    if blur_ksize > 1:
        occ_mask = F.avg_pool2d(
            occ_mask.unsqueeze(0).unsqueeze(0),
            kernel_size=blur_ksize,
            stride=1,
            padding=blur_ksize // 2
        ).squeeze(0).squeeze(0)

    return (occ_mask * max_mask).clamp(0.0, max_mask)

def compute_warp_validity_mask(grid: torch.Tensor, H: int, W: int, device) -> torch.Tensor:
    """
    Returns per-pixel sampling validity for a warp grid.
    grid: [H,W,2] in grid_sample coords
    returns: [H,W] float, 1 = fully valid, 0 = outside / strongly disoccluded
    """
    ones = torch.ones((1, 1, H, W), device=device, dtype=torch.float32)
    valid = F.grid_sample(
        ones,
        grid.unsqueeze(0),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(0).squeeze(0)
    return valid.clamp(0.0, 1.0)

def rgb_guided_depth_refine(depth_tensor, frame_tensor, kernel_size=5, sigma_color=0.1, sigma_space=5.0):
    """
    Joint bilateral-style refinement: smooth depth using RGB as edge guide.
    depth_tensor: [1, H, W] in 0..1
    frame_tensor: [3, H, W] in 0..1
    Returns refined depth [1, H, W]
    """
    device = depth_tensor.device
    _, H, W = depth_tensor.shape
    
    # Convert RGB to grayscale for edge guidance
    gray = 0.299 * frame_tensor[0] + 0.587 * frame_tensor[1] + 0.114 * frame_tensor[2]  # [H, W]
    
    # Unfold into patches
    pad = kernel_size // 2
    depth_pad = F.pad(depth_tensor.unsqueeze(0), (pad, pad, pad, pad), mode='reflect').squeeze(0)  # [1, H+2p, W+2p]
    gray_pad = F.pad(gray.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='reflect').squeeze(0).squeeze(0)  # [H+2p, W+2p]
    
    # Extract patches: [1, k*k, H, W] for depth, [k*k, H, W] for gray
    depth_patches = F.unfold(depth_pad.unsqueeze(0), kernel_size=kernel_size)  # [1, k*k, H*W]
    gray_patches = F.unfold(gray_pad.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size)  # [1, k*k, H*W]
    
    depth_patches = depth_patches.view(1, kernel_size*kernel_size, H, W)
    gray_patches = gray_patches.view(1, kernel_size*kernel_size, H, W)
    
    # Center pixel for color distance
    gray_center = gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Color weight: how similar is each patch pixel to center?
    color_dist = (gray_patches - gray_center).abs()
    color_weight = torch.exp(-color_dist / (sigma_color + 1e-6))
    
    # Spatial weight: distance from center of kernel
    coords = torch.arange(-pad, pad + 1, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(coords, coords, indexing='ij')
    spatial_dist = (gx**2 + gy**2).sqrt().view(-1, 1, 1).to(device)  # [k*k, 1, 1]
    spatial_weight = torch.exp(-spatial_dist / (2 * sigma_space**2)).unsqueeze(0)  # [1, k*k, 1, 1]
    
    # Combined weight
    weight = color_weight * spatial_weight
    weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
    
    # Weighted average
    refined = (depth_patches * weight).sum(dim=1)  # [1, H, W]
    return refined.clamp(0, 1)

def build_repair_mask(
    validity_mask: torch.Tensor,       # [H,W]
    shift_occ_mask: torch.Tensor,      # [H,W]
    valid_threshold: float = 0.999,
    expand_ksize: int = 5,
    max_mask: float = 1.0,
) -> torch.Tensor:
    """
    Build a repair mask that focuses on actually exposed / invalid regions,
    reinforced by shift-gradient occlusion likelihood.

    validity_mask  < threshold => disocclusion / undersampled region
    shift_occ_mask => where disparity discontinuity is likely
    """
    disocc = (validity_mask < valid_threshold).float()

    # combine true invalidity with shift-edge stress
    repair = torch.max(disocc, shift_occ_mask)

    if expand_ksize > 1:
        repair = F.max_pool2d(
            repair.unsqueeze(0).unsqueeze(0),
            kernel_size=expand_ksize,
            stride=1,
            padding=expand_ksize // 2
        ).squeeze(0).squeeze(0)

    return repair.clamp(0.0, max_mask)


def directional_background_fill(
    warped_tensor: torch.Tensor,       # [3,H,W]
    repair_mask: torch.Tensor,         # [H,W] in [0,1]
    direction: str = "right",
    radius: int = 6,
) -> torch.Tensor:
    """
    Adaptive one-sided fill: keeps pulling background pixels into the
    disocclusion zone until the repair mask is fully covered or max radius
    is reached. Each step blends with decreasing weight for a natural fade.
    """
    assert warped_tensor.dim() == 3 and warped_tensor.shape[0] == 3
    assert repair_mask.dim() == 2
    
    out = warped_tensor.clone()
    C, H, W = out.shape
    
    # Track what's been filled — start with the repair mask
    remaining = repair_mask.clone()
    max_radius = min(radius, W // 2)
    
    for step in range(1, max_radius + 1):
        if remaining.max() < 0.001:
            break  # gap fully covered
            
        shifted = out.clone()
        if direction == "right":
            shifted[:, :, step:] = out[:, :, :-step]
        else:
            shifted[:, :, :-step] = out[:, :, step:]
        
        # Only fill pixels that still need it
        step_weight = max(0.15, 1.0 - (step / max_radius) * 0.85)
        m = (remaining * step_weight).unsqueeze(0)  # [1, H, W]
        out = out * (1.0 - m) + shifted * m
        
        # Check what's still uncovered after this step
        if direction == "right":
            still_uncovered = remaining.clone()
            still_uncovered[:, :-step] = 0
        else:
            still_uncovered = remaining.clone()
            still_uncovered[:, step:] = 0
        remaining = torch.min(remaining, still_uncovered)
    
    return out.clamp(0.0, 1.0)


def repair_disocclusion_regions(
    warped_tensor: torch.Tensor,       # [3,H,W]
    repair_mask: torch.Tensor,         # [H,W]
    protect_mask: torch.Tensor,        # [H,W]
    direction: str,
    blur_ksize: int = 1,
    fill_radius: int = 2,
    repair_strength: float = 0.20,     # ✅ FIX: Reduced from 0.35 to prevent aggressive fill
    protect_dilate_ksize: int = 11,    # ✅ FIX: Increased from 7 to create wider barrier
) -> torch.Tensor:
    """
    Silhouette-safe background repair:
    - create a stronger foreground barrier from both the shift-derived protect mask
      and local image edges
    - forbid repair inside that barrier and add a small safety buffer around it
    - use a shorter one-sided fill so repaired pixels do not smear around ears / hair
    - keep blur minimal so bright contour rims do not turn into halos
    - ✅ FIX #5: Explicit collision prevention between repair and protect masks
    """
    assert warped_tensor.dim() == 3 and warped_tensor.shape[0] == 3
    assert repair_mask.dim() == 2
    assert protect_mask.dim() == 2

    repair_mask = repair_mask.clamp(0.0, 1.0)
    protect_mask = protect_mask.clamp(0.0, 1.0)

    # Strong barrier around the foreground contour.
    protect_barrier = dilate_mask(protect_mask, ksize=protect_dilate_ksize)

    # Extra image-edge barrier from the warped eye itself.
    # This catches bright rims and furry / spiky silhouettes even when the shift mask is soft.
    luma = warped_tensor.mean(dim=0)
    luma_gx = torch.abs(F.pad(luma[:, 1:] - luma[:, :-1], (1, 0)))
    luma_gy = torch.abs(F.pad(luma[1:, :] - luma[:-1, :], (0, 0, 1, 0)))
    luma_edge = torch.sqrt(luma_gx * luma_gx + luma_gy * luma_gy)
    edge_barrier = torch.clamp((luma_edge - 0.030) / 0.070, 0.0, 1.0)
    edge_barrier = dilate_mask(edge_barrier, ksize=max(3, protect_dilate_ksize - 2))

    # Combine barriers and add a wider safety buffer to prevent repair erosion.
    combined_barrier = torch.max(protect_barrier, edge_barrier * 0.85).clamp(0.0, 1.0)
    safety_buffer = dilate_mask(combined_barrier, ksize=5)  # ✅ FIX: 3 → 5

    # Never repair across the preserved contour or right beside it.
    effective_repair = (repair_mask * (1.0 - safety_buffer)).clamp(0.0, 1.0)

    # ============================================================
    # ✅ FIX #5 v3: Simple collision prevention (no squeeze issues)
    # ============================================================
    
    # Build hard barrier by dilating the combined barrier
    hard_barrier = dilate_mask(combined_barrier, ksize=3)
    
    # Soften the barrier edge with a blur so the transition isn't visible
    barrier_soft = F.avg_pool2d(
        hard_barrier.unsqueeze(0).unsqueeze(0),
        kernel_size=5,
        stride=1,
        padding=2
    )
    # Keep as 4D: [1, 1, H, W] — don't squeeze yet
    
    # Reshape effective_repair to 4D for consistent math
    effective_repair_4d = effective_repair.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Multiply in 4D
    effective_repair_4d = effective_repair_4d * (1.0 - barrier_soft)
    
    # Binary barrier in 4D
    binary_barrier = (combined_barrier > 0.01).float().unsqueeze(0).unsqueeze(0)
    binary_barrier = F.max_pool2d(binary_barrier, kernel_size=5, stride=1, padding=2)
    effective_repair_4d = effective_repair_4d * (1.0 - binary_barrier)
    
    # Clamp and squeeze back to 2D
    effective_repair = effective_repair_4d.squeeze(0).squeeze(0).clamp(0.0, 1.0)
    
    # Verify shape is correct
    assert effective_repair.dim() == 2, f"effective_repair should be 2D, got {effective_repair.shape}"
    # ============================================================
    # ✅ FIX #5 v3 END
    # ============================================================

    # 🔍 Diagnostic: verify no overlap between repair and protection
    overlap = (effective_repair * combined_barrier).max().item()
    if overlap > 0.002:
        print(f"⚠️ WARNING: Repair/protect overlap detected: {overlap:.6f}")

    # Keep the repair zone tight so it only touches the newly exposed strip.
    effective_repair = F.avg_pool2d(
        effective_repair.unsqueeze(0).unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1
    ).squeeze(0).squeeze(0)
    effective_repair = (effective_repair * 0.85).clamp(0.0, 1.0)

    filled = directional_background_fill(
        warped_tensor,
        effective_repair,
        direction=direction,
        radius=fill_radius
    )

    if blur_ksize > 1:
        blurred = F.avg_pool2d(
            filled.unsqueeze(0),
            kernel_size=blur_ksize,
            stride=1,
            padding=blur_ksize // 2
        ).squeeze(0)
    else:
        blurred = filled

    m = (effective_repair * float(repair_strength)).clamp(0.0, 1.0).unsqueeze(0)
    out = filled * (1.0 - m) + blurred * m

    # Restore the protected contour directly from the warped image.
    p = combined_barrier.unsqueeze(0)
    out = out * (1.0 - p) + warped_tensor * p

    return out.clamp(0.0, 1.0)

def compute_signed_shift_gradients(shift_vals: torch.Tensor):
    """
    shift_vals: [H,W] or [1,H,W]
    returns gx, gy signed gradients in shift field
    """
    if shift_vals.dim() == 3:
        shift_vals = shift_vals.squeeze(0)

    gx = F.pad(shift_vals[:, 1:] - shift_vals[:, :-1], (1, 0))
    gy = F.pad(shift_vals[1:, :] - shift_vals[:-1, :], (0, 0, 1, 0))
    return gx, gy


def build_one_sided_repair_and_protect_masks(
    shift_vals: torch.Tensor,       # [H,W]
    validity_mask: torch.Tensor,    # [H,W]
    eye: str,                       # "left" or "right"
    grad_threshold: float = 0.0045,
    validity_soft_threshold: float = 0.9985,
    expand_ksize: int = 5,
):
    """
    More conservative silhouette-safe repair/protect masks.
    The goal is to shrink the repair strip and strengthen the foreground barrier
    so spiky outlines do not get a glowing halo during background fill.
    """
    gx, gy = compute_signed_shift_gradients(shift_vals)
    grad_mag = torch.sqrt(gx * gx + gy * gy)

    grad_gate = torch.clamp((grad_mag - grad_threshold) / max(grad_threshold, 1e-6), 0.0, 1.0)
    validity_loss = torch.clamp(
        (validity_soft_threshold - validity_mask) / max(1.0 - validity_soft_threshold, 1e-6),
        0.0, 1.0
    )

    if eye == "left":
        trailing = torch.clamp(gx / max(grad_threshold, 1e-6), 0.0, 1.0)
        leading  = torch.clamp((-gx) / max(grad_threshold, 1e-6), 0.0, 1.0)
    else:
        trailing = torch.clamp((-gx) / max(grad_threshold, 1e-6), 0.0, 1.0)
        leading  = torch.clamp(gx / max(grad_threshold, 1e-6), 0.0, 1.0)

    # Keep repair more local and rely less on validity spillover.
    repair_mask = trailing * grad_gate
    repair_mask = torch.max(repair_mask, repair_mask * 0.50 + validity_loss * 0.20)

    # Stronger leading-side contour protection.
    protect_mask = leading * grad_gate
    protect_mask = torch.max(protect_mask, leading * 0.55)

    if expand_ksize > 1:
        repair_mask = F.max_pool2d(
            repair_mask.unsqueeze(0).unsqueeze(0),
            kernel_size=expand_ksize,
            stride=1,
            padding=expand_ksize // 2
        ).squeeze(0).squeeze(0)

        protect_mask = F.max_pool2d(
            protect_mask.unsqueeze(0).unsqueeze(0),
            kernel_size=expand_ksize,
            stride=1,
            padding=expand_ksize // 2
        ).squeeze(0).squeeze(0)

    repair_mask = (repair_mask * 0.42).clamp(0.0, 1.0)
    protect_mask = (protect_mask * 1.25).clamp(0.0, 1.0)

    return repair_mask, protect_mask

def dilate_mask(mask: torch.Tensor, ksize: int = 3) -> torch.Tensor:
    """
    mask: [H,W] in [0,1]
    returns dilated mask [H,W]
    """
    if ksize <= 1:
        return mask.clamp(0.0, 1.0)

    return F.max_pool2d(
        mask.unsqueeze(0).unsqueeze(0),
        kernel_size=ksize,
        stride=1,
        padding=ksize // 2
    ).squeeze(0).squeeze(0).clamp(0.0, 1.0)

def estimate_edge_window_violation(shift_vals: torch.Tensor, edge_band_px: int = 32):
    """
    Estimate likely frame-edge window violations from the final shift field.

    shift_vals: [H,W] normalized shift map used for actual warp
    Returns:
        left_violation, right_violation

    Interpretation:
      - left_violation: positive shift pressure near left border
      - right_violation: negative shift pressure near right border

    These are not perfect geometric violations, but they are much closer to
    actual edge risk than using only zero_parallax_offset.
    """
    assert shift_vals.dim() == 2

    H, W = shift_vals.shape
    band = min(int(edge_band_px), max(1, W // 4))

    left_band = shift_vals[:, :band]
    right_band = shift_vals[:, W - band:]

    left_violation = torch.relu(left_band).mean()
    right_violation = torch.relu(-right_band).mean()

    return float(left_violation.item()), float(right_violation.item())

def compute_visual_stress_mask(warped_tensor: torch.Tensor,
                               threshold: float = 0.035,
                               expand_ksize: int = 3) -> torch.Tensor:
    luma = warped_tensor.mean(dim=0)
    gx = torch.abs(F.pad(luma[:, 1:] - luma[:, :-1], (1, 0)))
    gy = torch.abs(F.pad(luma[1:, :] - luma[:-1, :], (0, 0, 1, 0)))
    g = torch.sqrt(gx * gx + gy * gy)
    m = torch.clamp((g - threshold) / max(threshold, 1e-6), 0.0, 1.0)

    if expand_ksize > 1:
        m = F.max_pool2d(
            m.unsqueeze(0).unsqueeze(0),
            kernel_size=expand_ksize,
            stride=1,
            padding=expand_ksize // 2
        ).squeeze(0).squeeze(0)

    return m.clamp(0.0, 1.0)

def debug_depth_histogram(depth_tensor, title="Depth"):
    """Print depth distribution to understand where your values actually are"""
    d = depth_tensor.cpu().numpy().flatten()
    print(f"\n🔍 {title} Stats:")
    print(f"   min={d.min():.3f}, max={d.max():.3f}")
    print(f"   mean={d.mean():.3f}, median={np.median(d):.3f}")
    print(f"   10th={np.percentile(d, 10):.3f}, 90th={np.percentile(d, 90):.3f}")
    
    # Bucket counts
    near = (d < 0.3).mean() * 100
    mid = ((d >= 0.3) & (d < 0.7)).mean() * 100
    far = (d >= 0.7).mean() * 100
    print(f"   Near: {near:.1f}%, Mid: {mid:.1f}%, Far: {far:.1f}%")
    print(f"   Shape: {'GOOD separation' if near > 5 and mid > 5 and far > 5 else 'FLAT - poor 3D potential'}")

def pixel_shift_cuda(
    frame_tensor,
    depth_tensor,
    width,
    height,
    fg_shift,
    mg_shift,
    bg_shift,
    blur_ksize=9,
    feather_strength=10.0,
    max_pixel_shift_percent=0.02,
    parallax_balance=0.8,
    zero_parallax_strength=0.0,
    use_subject_tracking=True,
    enable_floating_window=True,
    return_shift_map=True,
    enable_feathering=True,
    enable_edge_masking=True,
    dof_strength=2.0,
    convergence_strength=0.0,
    enable_dynamic_convergence=True,
    depth_pop_gamma=0.85,
    depth_pop_mid=0.50,
    depth_stretch_lo=0.05,
    depth_stretch_hi=0.95,
    fg_pop_multiplier=1.20,
    bg_push_multiplier=1.10,
    subject_lock_strength=0.35,
    return_tensors=False,
    disable_shift_ema=False
):
    width = int(width)
    height = int(height)
    device = frame_tensor.device

    frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
    depth_tensor = F.interpolate(depth_tensor.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)

    if 'enhance_curvature' in globals():
        depth_tensor = enhance_curvature(depth_tensor, strength=0.08)

    depth_tensor = depth_tensor.clamp(0.0, 1.0)
    
    # 🛡️ STRONGER depth refinement for cleaner edges
    depth_tensor = rgb_guided_depth_refine(depth_tensor, frame_tensor, kernel_size=7, sigma_color=0.06, sigma_space=4.0)
    
    # 🛡️ Edge-aware depth blur to prevent sharp transitions
    depth_grad_x = torch.abs(F.pad(depth_tensor[:, :, 1:] - depth_tensor[:, :, :-1], (1, 0)))
    depth_grad_y = torch.abs(F.pad(depth_tensor[:, 1:, :] - depth_tensor[:, :-1, :], (0, 0, 1, 0)))
    depth_edge = (depth_grad_x + depth_grad_y).squeeze(0)
    edge_soft_mask = torch.clamp(depth_edge * 12.0, 0.0, 0.7)
    depth_blurred = tv_gaussian_blur(depth_tensor, kernel_size=5, sigma=1.8)
    depth_tensor = depth_tensor * (1 - edge_soft_mask.unsqueeze(0)) + depth_blurred * edge_soft_mask.unsqueeze(0)

    # assume caller already provides stabilized normalized depth
    d_norm = depth_tensor.clamp(0.0, 1.0)

    # estimate subject from normalized RAW depth, before recenter/pop shaping
    subject_depth_track = estimate_subject_depth(d_norm)
    subject_depth_track = torch.tensor(
        subject_depth_ema.update(subject_depth_track.item()),
        device=device
    )

    # shape a copy for disparity design only
    d_shaped = shape_depth_for_pop(
        d_norm,
        subject_depth_track,
        stretch_lo=depth_stretch_lo,
        stretch_hi=depth_stretch_hi,
        depth_mid=depth_pop_mid,
        gamma=depth_pop_gamma
    )

    # use tracking depth for convergence, not shaped depth
    subject_depth = subject_depth_track
    
    # Broader mid-ground influence for smoother transitions
    near_center = 0.15
    mid_center  = depth_pop_mid
    far_center  = 0.85

    near_sigma = 0.30   # Wider for smoother falloff
    mid_sigma  = 0.35   # Much wider mid zone
    far_sigma  = 0.30   # Wider for smoother falloff

    fg_weight = torch.exp(-0.5 * ((d_shaped - near_center) / near_sigma) ** 2)
    mg_weight = torch.exp(-0.5 * ((d_shaped - mid_center)  / mid_sigma)  ** 2)
    bg_weight = torch.exp(-0.5 * ((d_shaped - far_center)  / far_sigma)  ** 2)

    w_sum = fg_weight + mg_weight + bg_weight + 1e-6
    fg_weight = fg_weight / w_sum
    mg_weight = mg_weight / w_sum
    bg_weight = bg_weight / w_sum

    half_width = width / 2.0

    raw_shift = (fg_weight * fg_shift * fg_pop_multiplier +
                 mg_weight * mg_shift +
                 bg_weight * bg_shift * bg_push_multiplier)

    total_shift = (raw_shift * parallax_balance) / half_width

    zero_parallax_offset = 0.0

    if use_subject_tracking:
        subj = subject_depth.clamp(0.0, 1.0)

        fg_w_subj = torch.exp(-0.5 * ((subj - near_center) / near_sigma) ** 2)
        mg_w_subj = torch.exp(-0.5 * ((subj - mid_center)  / mid_sigma)  ** 2)
        bg_w_subj = torch.exp(-0.5 * ((subj - far_center)  / far_sigma)  ** 2)

        w_subj_sum = fg_w_subj + mg_w_subj + bg_w_subj + 1e-6
        fg_w_subj = fg_w_subj / w_subj_sum
        mg_w_subj = mg_w_subj / w_subj_sum
        bg_w_subj = bg_w_subj / w_subj_sum

        subj_shift = (
            fg_w_subj * fg_shift * fg_pop_multiplier +
            mg_w_subj * mg_shift +
            bg_w_subj * bg_shift * bg_push_multiplier
        )

        zero_parallax_offset = (subj_shift * parallax_balance) / half_width
        zero_parallax_offset = zero_parallax_offset * float(subject_lock_strength)
        zero_parallax_offset = zero_parallax_offset - float(zero_parallax_strength)

        if enable_floating_window:
            depth_bias = torch.abs(subject_depth - 0.5)
            subject_weight = torch.clamp(1.0 - depth_bias * 2.0, 0.4, 1.0)
            zero_parallax_offset = zero_parallax_offset * subject_weight
            zero_parallax_offset = torch.clamp(zero_parallax_offset, -0.30, 0.30)
            zero_parallax_offset = floating_window_tracker.smooth_offset(
                float(zero_parallax_offset.item()),
                threshold=0.001
            )

        total_shift -= zero_parallax_offset
    else:
        zero_parallax_offset = 0.0
        total_shift -= zero_parallax_offset

    disparity_gain = 1.0
    total_shift = total_shift * disparity_gain

    max_shift_px = width * max_pixel_shift_percent
    max_shift_norm = max_shift_px / half_width
    total_shift = torch.clamp(total_shift, -max_shift_norm, max_shift_norm)

    if convergence_strength != 0.0:
        if enable_dynamic_convergence:
            subj_for_conv = subject_depth_track
            convergence_bias = subj_for_conv * convergence_strength
        else:
            convergence_bias = torch.tensor(convergence_strength, device=device)

        conv_smooth = conv_ema.update(convergence_bias.item())
        total_shift -= conv_smooth / half_width

    mask_strength = 0.0 if feather_strength <= 1e-6 else float(np.clip(feather_strength / 10.0, 0.05, 0.3))

    if enable_edge_masking and mask_strength > 0.0:
        occ_mask_shift = compute_occlusion_mask_from_shift(
            total_shift.squeeze(0),
            blur_ksize=7,           # ← Larger blur for softer mask
            occ_threshold=0.025,    # ← Higher threshold = less aggressive
            occ_strength=3.0,       # ← Lower strength = gentler
            max_mask=0.12,          # ← Lower max mask
        ).unsqueeze(0)

        edge_suppressed = total_shift * (1.0 - occ_mask_shift)
        final_shift = (1.0 - mask_strength) * total_shift + mask_strength * edge_suppressed
    else:
        final_shift = total_shift

    # optional EMA
    if disable_shift_ema:
        final_shift = final_shift
    else:
        shift_ema_alpha = 0.65
        if not hasattr(pixel_shift_cuda, "_shift_ema") or pixel_shift_cuda._shift_ema is None:
            pixel_shift_cuda._shift_ema = final_shift.clone()
        else:
            pixel_shift_cuda._shift_ema = (
                shift_ema_alpha * pixel_shift_cuda._shift_ema +
                (1.0 - shift_ema_alpha) * final_shift
            )
        final_shift = pixel_shift_cuda._shift_ema

    shift_vals = final_shift.squeeze(0)
    mask_shift_vals = final_shift.squeeze(0)
    
    edge_violation_left, edge_violation_right = estimate_edge_window_violation(
        shift_vals,
        edge_band_px=max(16, width // 40)
    )

    H, W = d_shaped.shape[1:]
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, W, device=device),
        torch.linspace(-1, 1, H, device=device),
        indexing="xy"
    )
    grid = torch.stack((xx, yy), dim=-1)

    grid_left = grid.clone()
    grid_right = grid.clone()
    grid_left[..., 0] += shift_vals
    grid_right[..., 0] -= shift_vals

    warped_left = F.grid_sample(
        frame_tensor.unsqueeze(0),
        grid_left.unsqueeze(0),
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0)

    warped_right = F.grid_sample(
        frame_tensor.unsqueeze(0),
        grid_right.unsqueeze(0),
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0)

    # 🛡️ STRONGER smear suppression
    shift_grad_x = torch.abs(F.pad(shift_vals[:, 1:] - shift_vals[:, :-1], (1, 0)))
    shift_grad_y = torch.abs(F.pad(shift_vals[1:, :] - shift_vals[:-1, :], (0, 0, 1, 0)))
    shift_grad = shift_grad_x + shift_grad_y
    
    smear_zone = (shift_grad > 0.006).float()  # ← Lower threshold catches more
    smear_zone = F.max_pool2d(
        smear_zone.unsqueeze(0).unsqueeze(0),
        kernel_size=5, stride=1, padding=2    # ← Larger expansion
    ).squeeze(0).squeeze(0)
    
    blend = smear_zone.unsqueeze(0) * 0.60   # ← Stronger blend back to original
    warped_left = warped_left * (1.0 - blend) + frame_tensor * blend
    warped_right = warped_right * (1.0 - blend) + frame_tensor * blend

    # --- per-eye validity / disocclusion analysis ---
    valid_left = compute_warp_validity_mask(grid_left, H, W, device)
    valid_right = compute_warp_validity_mask(grid_right, H, W, device)

    repair_mask_left, protect_mask_left = build_one_sided_repair_and_protect_masks(
        mask_shift_vals,
        valid_left,
        eye="left",
        grad_threshold=0.008,            # ← Lower = more repair
        validity_soft_threshold=0.995,   # ← Higher = more repair
        expand_ksize=7,                  # ← Larger expansion
    )

    repair_mask_right, protect_mask_right = build_one_sided_repair_and_protect_masks(
        mask_shift_vals,
        valid_right,
        eye="right",
        grad_threshold=0.008,
        validity_soft_threshold=0.995,
        expand_ksize=7,
    )

    if enable_feathering:
        # 🛡️ MAXIMUM EDGE REPAIR SETTINGS
        left_blended = repair_disocclusion_regions(
            warped_left,
            repair_mask_left,
            protect_mask_left,
            direction="right",
            blur_ksize=3,              # ← Light blur on repair
            fill_radius=32,            # ← MUCH larger fill
            repair_strength=0.55,      # ← Stronger repair
            protect_dilate_ksize=17    # ← Wider protection
        )

        right_blended = repair_disocclusion_regions(
            warped_right,
            repair_mask_right,
            protect_mask_right,
            direction="left",
            blur_ksize=3,
            fill_radius=32,
            repair_strength=0.55,
            protect_dilate_ksize=17
        )
    else:
        left_blended = warped_left
        right_blended = warped_right

    zero_meta = {
        "shift_map": final_shift.detach().cpu(),
        "subject_depth": float(subject_depth.item()) if torch.is_tensor(subject_depth) else float(subject_depth),
        "zero_parallax_offset": float(zero_parallax_offset) if use_subject_tracking else 0.0,
        "edge_violation_left": float(edge_violation_left),
        "edge_violation_right": float(edge_violation_right),
        "repair_mask_left_mean": float(repair_mask_left.mean().item()),
        "repair_mask_right_mean": float(repair_mask_right.mean().item()),
        "protect_mask_left_mean": float(protect_mask_left.mean().item()),
        "protect_mask_right_mean": float(protect_mask_right.mean().item()),
        "valid_left_min": float(valid_left.min().item()),
        "valid_right_min": float(valid_right.min().item()),
        "valid_left_p01": float(torch.quantile(valid_left, 0.01).item()),
        "valid_right_p01": float(torch.quantile(valid_right, 0.01).item()),
    }

    if return_shift_map:
        if return_tensors:
            return left_blended, right_blended, zero_meta
        return tensor_to_frame(left_blended), tensor_to_frame(right_blended), zero_meta
    else:
        if return_tensors:
            return left_blended, right_blended
        return tensor_to_frame(left_blended), tensor_to_frame(right_blended)

def tensor_pad_to_aspect(t: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    """
    t: [3,H,W] RGB float 0..1
    Pads with black to exactly target_w/target_h, centered.
    """
    C, H, W = t.shape
    out = t
    # resize to fit inside target while preserving aspect
    src_ar = W / max(H, 1)
    dst_ar = target_w / max(target_h, 1)

    if abs(src_ar - dst_ar) > 1e-6:
        if src_ar > dst_ar:
            # too wide, fit width
            new_w = target_w
            new_h = int(round(target_w / src_ar))
        else:
            # too tall, fit height
            new_h = target_h
            new_w = int(round(target_h * src_ar))
    else:
        new_w, new_h = target_w, target_h

    out = F.interpolate(out.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)

    # pad to target
    pad_l = max(0, (target_w - new_w) // 2)
    pad_r = max(0, target_w - new_w - pad_l)
    pad_t = max(0, (target_h - new_h) // 2)
    pad_b = max(0, target_h - new_h - pad_t)

    return F.pad(out, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0.0).clamp(0.0, 1.0)


def tensor_apply_sharpen(t: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    """
    Simple unsharp-style sharpen for tensors [3,H,W] in 0..1.
    """
    if factor <= 0:
        return t
    # light blur
    blur = tv_gaussian_blur(t, kernel_size=3, sigma=1.0)
    out = t + (t - blur) * float(factor)
    return out.clamp(0.0, 1.0)

# Sharpening

def apply_sharpening(frame, factor=1.0):
    # Safer sharpening kernel with brightness normalization
    kernel = np.array([
        [0, -1, 0],
        [-1, 5 + factor, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # Normalize kernel to preserve brightness (sum to ~1)
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel /= kernel_sum

    # Apply and clip result to valid range
    sharpened = cv2.filter2D(frame, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

@torch.no_grad()
def apply_color_grade(
    rgb_tensor: torch.Tensor,      # [3,H,W], float32 in [0,1], RGB
    saturation: float = 1.0,       # 1.0 = no change
    contrast: float = 1.0,         # 1.0 = no change
    brightness: float = 0.0        # additive, -0.5..+0.5 recommended
):
    """
    Fast GPU color grading:
      - saturation: scales chroma around luminance
      - contrast  : symmetric about 0.5
      - brightness: additive offset
    All math in 0..1 RGB space. Clamped at the end.
    """
    assert rgb_tensor.dim() == 3 and rgb_tensor.shape[0] == 3
    # Luminance (Rec.709)
    r, g, b = rgb_tensor[0], rgb_tensor[1], rgb_tensor[2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Saturation: lerp between gray(luma) and original by 'saturation'
    # sat=0 -> gray; sat=1 -> original; sat>1 -> extra chroma
    rgb_sat = torch.stack([
        luma + (r - luma) * saturation,
        luma + (g - luma) * saturation,
        luma + (b - luma) * saturation,
    ], dim=0)

    # Contrast around 0.5 mid-gray
    rgb_con = 0.5 + (rgb_sat - 0.5) * contrast

    # Brightness (additive)
    rgb_bri = rgb_con + brightness

    return rgb_bri.clamp(0.0, 1.0)

@torch.no_grad()
def apply_dof_cuda(
    rgb_tensor: torch.Tensor,
    depth_tensor: torch.Tensor,
    focal_depth: float,
    max_sigma: float = 2.0,
    focus_width: float = 0.35,
    num_levels: int = 5,
):
    """
    Depth-of-field via level-of-detail Gaussian pyramid + per-pixel interpolation.

    rgb_tensor:   [3, H, W], float32 in [0,1]
    depth_tensor: [1, H, W], float32 in [0,1]
    focal_depth:  scalar float or 0-D tensor in [0,1]
    """
    assert rgb_tensor.dim() == 3 and rgb_tensor.shape[0] == 3
    assert depth_tensor.dim() == 3 and depth_tensor.shape[0] == 1
    device = rgb_tensor.device
    C, H, W = rgb_tensor.shape

    # --- 1) per-pixel blur weight based on distance from focal plane ---
    if not torch.is_tensor(focal_depth):
        focal_depth = torch.tensor(float(focal_depth), device=device)
    depth_diff   = torch.abs(depth_tensor - focal_depth)                  # [1,H,W]
    blur_weights = (depth_diff / (focus_width + 1e-6)).clamp(0.0, 1.0)    # [1,H,W]

    # --- 2) build blur levels driven by max_sigma ---
    # levels[0]=0 means "no blur", then linearly up to max_sigma
    levels = torch.linspace(0.0, float(max_sigma), steps=num_levels, device=device)
    blurred_versions = []
    for lvl_idx, sigma in enumerate(levels):
        if float(sigma) == 0.0:
            blurred_versions.append(rgb_tensor)
        else:
            # kernel size ~= 4*sigma + 1, odd
            ksize = int(2 * math.ceil(2 * float(sigma)) + 1)
            blurred_versions.append(tv_gaussian_blur(rgb_tensor, kernel_size=ksize, sigma=float(sigma)))

    # [N,3,H,W]
    stack = torch.stack(blurred_versions, dim=0)  # N=num_levels

    # --- 3) pick the two neighboring levels and lerp between them per pixel ---
    # blur index in [0, N-1]
    N = num_levels
    blur_idx  = (blur_weights * (N - 1)).clamp(0, N - 1 - 1e-6)           # [1,H,W]
    lower_idx = blur_idx.floor().long().clamp(0, N - 2)                   # [1,H,W]
    upper_idx = lower_idx + 1                                             # [1,H,W]
    alpha     = (blur_idx - lower_idx.float()).squeeze(0)                # [H,W]

    # Prepare for gather: move level dimension to the last axis
    # stack_perm: [3,H,W,N]
    stack_perm = stack.permute(1, 2, 3, 0)

    # Indices for gather must match dst shape; make [3,H,W,1]
    li = lower_idx.squeeze(0).unsqueeze(0).expand(3, H, W).unsqueeze(-1)
    ui = upper_idx.squeeze(0).unsqueeze(0).expand(3, H, W).unsqueeze(-1)

    lower_vals = torch.gather(stack_perm, dim=-1, index=li).squeeze(-1)   # [3,H,W]
    upper_vals = torch.gather(stack_perm, dim=-1, index=ui).squeeze(-1)   # [3,H,W]

    # Broadcast alpha to [3,H,W]
    alpha3 = alpha.unsqueeze(0).expand(3, H, W)

    out = (1.0 - alpha3) * lower_vals + alpha3 * upper_vals
    return out.clamp(0.0, 1.0)

# 3D Formats
def format_3d_output(left, right, fmt):
    h, w = left.shape[:2]
    
    if fmt == "Half-SBS":
        return np.hstack((left, right))

    elif fmt == "Full-SBS":
        return np.hstack((left, right))
    
    elif fmt == "VR":
        lw = cv2.resize(left, (1440, 1600))
        rw = cv2.resize(right, (1440, 1600))
        return np.hstack((lw, rw))
    
    elif fmt == "Red-Cyan Anaglyph":
        return generate_anaglyph_3d(left, right, mode="dubois")  # start with halfcolor

    elif fmt == "Passive Interlaced":
        interlaced = np.zeros_like(left)
        interlaced[::2] = left[::2]      # even rows
        interlaced[1::2] = right[1::2]   # odd rows
        return interlaced

    return np.hstack((left, right))  # fallback

def generate_anaglyph_3d(left_bgr, right_bgr, mode="dubois"):
    """
    Inputs are OpenCV BGR. 
    mode="halfcolor" is a simple, high-impact check (Left→Red, Right→Cyan).
    mode="dubois" applies a BGR-adapted Dubois matrix.
    """
    lb, lg, lr = cv2.split(left_bgr)   # B,G,R from LEFT
    rb, rg, rr = cv2.split(right_bgr)  # B,G,R from RIGHT

    if mode == "halfcolor":
        # Left supplies Red, Right supplies Green/Blue (Cyan)
        return cv2.merge([rb, rg, lr])  # B from right, G from right, R from left

    # ---- BGR-adapted Dubois (coefficients reordered for BGR) ----
    # Red   channel is built from LEFT (R,G,B):
    r = 0.1762*lb + 0.5005*lg + 0.4561*lr
    # Green channel is built from RIGHT (R,G,B):
    g = -0.1876*rr + 0.7616*rg + 0.3764*rb
    # Blue  channel is built from RIGHT (R,G,B):
    b =  1.2723*rb - 0.1126*rg - 0.0401*rr

    out = cv2.merge([
        np.clip(b, 0, 1),  # B
        np.clip(g, 0, 1),  # G
        np.clip(r, 0, 1),  # R
    ])
    return (out * 255).astype(np.uint8)


def apply_side_mask(image, side="left", width=40, fade=False, solid_black=True):
    """
    Applies either a faded or solid black mask on one or both edges.
    - fade=True: linear alpha fade
    - solid_black=True: hard opaque black bar (cinema-grade)
    """
    if width <= 0:
        return image

    h, w = image.shape[:2]
    output = image.copy()

    if solid_black:
        # Solid opaque black bar — cinema floating window
        if side == "left":
            output[:, :width] = 0
        else:
            output[:, -width:] = 0
        return output

    # 🩶 Faded style (original)
    mask = np.ones((h, w), np.float32)
    if fade:
        fade_len = min(width, w // 2)
        ramp = np.linspace(0, 1, fade_len)
        if side == "left":
            mask[:, :fade_len] = ramp
        else:
            mask[:, -fade_len:] = ramp[::-1]
    else:
        if side == "left":
            mask[:, :width] = 0
        else:
            mask[:, -width:] = 0

    return (image * mask[..., None]).astype(np.uint8)

    
class FocalDepthTracker:
    def __init__(self, alpha=0.15, deadband=0.03, max_step=0.02):
        self.alpha = float(alpha)
        self.deadband = float(deadband)
        self.max_step = float(max_step)
        self.focal = None

    def reset(self, value=None):
        self.focal = value

    def set_scene_motion(self, motion_metric):
        # motion_metric in [0..1], 0=still, 1=lots of motion
        # still -> alpha ~0.10, busy -> alpha ~0.30
        self.alpha = 0.10 + 0.20 * max(0.0, min(1.0, float(motion_metric)))

    def update(self, candidate):
        c = float(candidate)
        if self.focal is None:
            self.focal = c
            return self.focal
        if abs(c - self.focal) < self.deadband:
            c = self.focal
        new_focal = (1.0 - self.alpha) * self.focal + self.alpha * c
        delta = new_focal - self.focal
        if   delta >  self.max_step: new_focal = self.focal + self.max_step
        elif delta < -self.max_step: new_focal = self.focal - self.max_step
        self.focal = max(0.0, min(1.0, new_focal))
        return self.focal

def compute_motion_metric(prev_d, curr_d):
    if prev_d is None:
        return 0.0
    # mean absolute difference in [0..1] range, clamp for safety
    mad = torch.mean(torch.abs(curr_d - prev_d)).item()
    return max(0.0, min(1.0, mad * 4.0))  # scale a bit to feel responsive


# Render
def render_sbs_3d(
    input_path,
    depth_path,
    output_path,
    selected_codec,
    fps,
    output_width,
    output_height,
    fg_shift,
    mg_shift,
    bg_shift,
    sharpness_factor,
    output_format,
    selected_aspect_ratio,
    aspect_ratios,
    dof_strength,
    feather_strength=0.0,
    blur_ksize=1,
    use_ffmpeg=False,
    preserve_hdr10= False,
    selected_ffmpeg_codec=None,
    crf_value=23,
    nvenc_cq_value=23,
    use_subject_tracking=False,
    use_floating_window=False,
    max_pixel_shift_percent=0.02,
    progress=None,
    progress_label=None,
    suspend_flag=None,
    cancel_flag=None,
    auto_crop_black_bars=False,
    parallax_balance=0.8,
    preserve_original_aspect=False,
    zero_parallax_strength=0.0,
    enable_edge_masking=True,
    enable_feathering=True,
    skip_blank_frames=False,
    original_video_width=None,
    original_video_height=None,
    convergence_strength=0.0,
    enable_dynamic_convergence=True,
    ipd_factor=1.0,
    depth_pop_gamma=0.85,
    depth_pop_mid=0.50,
    depth_stretch_lo=0.05,
    depth_stretch_hi=0.95,
    fg_pop_multiplier=1.20,
    bg_push_multiplier=1.10,
    subject_lock_strength=0.35,
    color_saturation=1.0,
    color_contrast=1.0,
    color_brightness=0.0,
    start_s=None,
    end_s=None,
    eye_mode="sbs",
    vr180_equi_w=None,
    vr180_equi_h=None,
    vr180_flat_w=None,
    vr180_flat_h=None,
    vr180_hfov_deg=110.0,
    disable_shift_ema=False,
):
    reset_render_state()
    cap, dcap = cv2.VideoCapture(input_path), cv2.VideoCapture(depth_path)
    if not cap.isOpened() or not dcap.isOpened():
        return

    hdr_gen = None
    if preserve_hdr10:
        # Use original input dimensions for decode
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        hdr_gen = ffmpeg_rgb48_reader(input_path, src_w, src_h, start_s=start_s, end_s=end_s)

    def read_next_frame():
        """Returns (ret, frame_tensor, frame_bgr_or_None)."""
        if preserve_hdr10:
            try:
                rgb = next(hdr_gen)  # float RGB 0..1
            except StopIteration:
                return False, None, None
            return True, frame16_to_tensor(rgb), None
        else:
            ret, frame_bgr = cap.read()
            if not ret:
                return False, None, None
            return True, frame_to_tensor(frame_bgr), frame_bgr

    # base facts
    total_frames_full = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
    dur_ms = (total_frames_full / max(fps, 1e-6)) * 1000.0

    # resolve clip window
    start_ms = max(0.0, (start_s or 0.0) * 1000.0)
    end_ms = dur_ms if (end_s is None) else min(dur_ms, end_s * 1000.0)
    print(f"[CLIP] start_s={start_s} end_s={end_s}")
    print(f"[CLIP] resolved: start_ms={start_ms:.1f} end_ms={end_ms:.1f}")

    # Guard: clamp and validate
    if start_ms < 0: start_ms = 0.0
    if end_ms > dur_ms: end_ms = dur_ms
    if start_ms >= end_ms - 0.5:
        print("⚠️ Invalid clip window; nothing to render.")
        cap.release(); dcap.release()
        return

    # derive clip frame count for progress
    start_frame_idx = int(round(start_ms / 1000.0 * fps))
    end_frame_idx   = int(round(end_ms   / 1000.0 * fps))
    clip_total_frames = max(0, end_frame_idx - start_frame_idx)
    
    # seek by frames ONCE
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    dcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    # FIRST READ occurs *after* seeking
    ret1, frame_tensor, frame = read_next_frame()
    ret2, depth = dcap.read()

    if not ret1 or not ret2:
        cap.release(); dcap.release()
        return

    global global_session_start_time
    if global_session_start_time is None:
        global_session_start_time = time.time()

    # 🛡️ Validate and fallback selected_ffmpeg_codec BEFORE it's used
    if use_ffmpeg:
        if not selected_ffmpeg_codec or not isinstance(selected_ffmpeg_codec, str) or selected_ffmpeg_codec.strip() == "":
            print("⚠️ No valid FFmpeg codec selected — falling back to libx264.")
            selected_ffmpeg_codec = "libx264"
        elif selected_ffmpeg_codec not in FFMPEG_CODEC_MAP.values():
            print(f"⚠️ Unrecognized codec '{selected_ffmpeg_codec}' — defaulting to libx264.")
            selected_ffmpeg_codec = "libx264"

    # --- Detect Blank Frames ---
    blank_frames = []
    if skip_blank_frames:
        try:
            blank_frames = detect_black_white_frames(
                input_path,
                mode="black",  # or "white"
                duration_threshold=0.1,
                pixel_threshold=0.10,
                cache=True
            )
            blank_frames = set(blank_frames)
        except Exception as e:
            print(f"⚠️ Blank frame detection failed: {e}")
            blank_frames = []
            
    # 🆕 blank frame indices are absolute — offset them for the clip window
    blank_offset = start_frame_idx

    first_frame_tensor = frame_tensor.clone()

    if auto_crop_black_bars:
        # Detect once on first frame
        top_crop, bottom_crop = detect_black_bars(first_frame_tensor)
        cached_crop = (top_crop, bottom_crop)

        # Log just once
        if top_crop > 0 or bottom_crop > 0:
            print(f"Auto-crop detected black bars: top={top_crop}px, bottom={bottom_crop}px")
        else:
            print("Auto-crop: No black bars detected")

        # Apply it to first frame
        first_frame_tensor, _ = crop_black_bars_torch(first_frame_tensor, cached_crop)
    else:
        cached_crop = (0, 0)

    target_ratio = aspect_ratios.get(selected_aspect_ratio.get(), 16 / 9)

    _, h, w = first_frame_tensor.shape
    current_ratio = w / h
    if abs(current_ratio - target_ratio) > 0.01:
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            w = new_w
        else:
            new_h = int(w / target_ratio)
            h = new_h
        
    if preserve_original_aspect:
        if original_video_width is None or original_video_height is None:
            # fallback to current frame tensor size
            _, h0, w0 = first_frame_tensor.shape
            original_video_width, original_video_height = w0, h0

        resized_width = original_video_width
        resized_height = original_video_height

        if output_format == "Full-SBS":
            per_eye_w = resized_width
            per_eye_h = resized_height
            out_width = per_eye_w * 2
            out_height = per_eye_h

        elif output_format == "Half-SBS":
            per_eye_w = resized_width // 2
            per_eye_h = resized_height
            out_width = resized_width
            out_height = resized_height

        elif output_format == "VR":
            per_eye_w = 1440
            per_eye_h = 1600
            out_width = per_eye_w * 2
            out_height = per_eye_h
            
        elif output_format == "VR180 Equirect (TB)":
            # these are only placeholders; final out size comes from equi_eye_* below
            per_eye_w = flat_eye_w if "flat_eye_w" in locals() else 1920
            per_eye_h = flat_eye_h if "flat_eye_h" in locals() else 1080
            out_width = int(vr180_equi_w) if vr180_equi_w else 3840
            out_height = (int(vr180_equi_h) if vr180_equi_h else 1920) * 2

        elif output_format == "VR180 Equirect (SBS)":
            out_width = (int(vr180_equi_w) if vr180_equi_w else 3840) * 2
            out_height = int(vr180_equi_h) if vr180_equi_h else 1920     

        elif output_format == "Red-Cyan Anaglyph":
            per_eye_w = resized_width
            per_eye_h = resized_height
            out_width = resized_width
            out_height = resized_height

        elif output_format == "Passive Interlaced":
            # IMPORTANT: interlaced is single-frame size (not SBS)
            per_eye_w = resized_width
            per_eye_h = resized_height
            out_width = resized_width
            out_height = resized_height

        else:
            per_eye_w = resized_width
            per_eye_h = resized_height
            out_width = resized_width * 2
            out_height = resized_height

    else:
        resized_height = output_height
        resized_width = int(resized_height * target_ratio)
        if resized_width % 2 != 0:
            resized_width += 1

        if output_format == "Full-SBS":
            per_eye_w, per_eye_h = 1920, 1080
            out_width = per_eye_w * 2
            out_height = per_eye_h

        elif output_format == "Half-SBS":
            per_eye_w = resized_width // 2
            per_eye_h = resized_height
            out_width = resized_width
            out_height = resized_height

        elif output_format == "VR":
            per_eye_w = 1440
            per_eye_h = 1600
            out_width = per_eye_w * 2
            out_height = per_eye_h

        elif output_format == "VR180 Equirect (TB)":
            # these are only placeholders; final out size comes from equi_eye_* below
            per_eye_w = flat_eye_w if "flat_eye_w" in locals() else 1920
            per_eye_h = flat_eye_h if "flat_eye_h" in locals() else 1080
            out_width = int(vr180_equi_w) if vr180_equi_w else 3840
            out_height = (int(vr180_equi_h) if vr180_equi_h else 1920) * 2

        elif output_format == "VR180 Equirect (SBS)":
            out_width = (int(vr180_equi_w) if vr180_equi_w else 3840) * 2
            out_height = int(vr180_equi_h) if vr180_equi_h else 1920
            
        elif output_format == "Red-Cyan Anaglyph":
            # One frame only, not SBS
            per_eye_w = resized_width
            per_eye_h = resized_height
            out_width = resized_width
            out_height = resized_height

        elif output_format == "Passive Interlaced":
            # IMPORTANT: interlaced is single-frame size (not SBS)
            per_eye_w = resized_width
            per_eye_h = resized_height
            out_width = resized_width
            out_height = resized_height

        else:
            per_eye_w = resized_width
            per_eye_h = resized_height
            out_width = resized_width * 2
            out_height = resized_height

    # Accept either raw values or Tk variables (e.g. DoubleVar) for VR180 params
    vr180_equi_w = (vr180_equi_w.get() if hasattr(vr180_equi_w, "get") else vr180_equi_w)
    vr180_equi_h = (vr180_equi_h.get() if hasattr(vr180_equi_h, "get") else vr180_equi_h)
    vr180_flat_w = (vr180_flat_w.get() if hasattr(vr180_flat_w, "get") else vr180_flat_w)
    vr180_flat_h = (vr180_flat_h.get() if hasattr(vr180_flat_h, "get") else vr180_flat_h)
    vr180_hfov_deg = (vr180_hfov_deg.get() if hasattr(vr180_hfov_deg, "get") else vr180_hfov_deg)
    
    # VR180 uses two resolutions:
    # - flat_eye_*: internal rectilinear render size (fast)
    # - equi_eye_*: final per-eye equirect size (2:1)
    vr180_enabled = output_format in ("VR180 Equirect (TB)", "VR180 Equirect (SBS)")

    if vr180_enabled:
        # per-eye equirect output size (2:1)
        if vr180_equi_w is None or vr180_equi_h is None:
            equi_eye_w, equi_eye_h = per_eye_w, per_eye_h
        else:
            equi_eye_w, equi_eye_h = int(vr180_equi_w), int(vr180_equi_h)

        # internal flat working size (performance)
        if vr180_flat_w is None or vr180_flat_h is None:
            flat_eye_w, flat_eye_h = 1920, 1080
        else:
            flat_eye_w, flat_eye_h = int(vr180_flat_w), int(vr180_flat_h)
    else:
        equi_eye_w = per_eye_w
        equi_eye_h = per_eye_h
        flat_eye_w = per_eye_w
        flat_eye_h = per_eye_h



    # --- invariants (fixed for the whole render) ---
    cinema_aspect_ratio = aspect_ratios.get(selected_aspect_ratio.get(), 16/9)
    single_eye = eye_mode in ("left", "right")

    # Fixed per-eye resize target used for every frame:
    if vr180_enabled:
        eye_w = flat_eye_w
        eye_h = flat_eye_h
    else:
        if not preserve_original_aspect:
            eye_w = per_eye_w
            eye_h = int(per_eye_w / cinema_aspect_ratio)
            if eye_h % 2 != 0:
                eye_h += 1
        else:
            eye_w = per_eye_w
            eye_h = per_eye_h

    # Floating window should operate on the internal working width
    width_for_bars = eye_w

    # DOF / Color grading flags don’t change during render
    need_dof   = (dof_strength > 0.0)
    need_color = (
        (color_saturation != 1.0) or
        (color_contrast   != 1.0) or
        (abs(color_brightness) > 1e-6)
    )

    ffmpeg_proc = None
    out = None


    # Force single-eye output size for non-VR180 left/right renders
    if single_eye and not vr180_enabled:
        out_width = int(per_eye_w)
        out_height = int(per_eye_h if preserve_original_aspect else eye_h)

    # --- FORCE final output size for VR180 so FFmpeg matches the frames we write ---
    if vr180_enabled:
        if eye_mode in ("left", "right"):
            out_width  = int(equi_eye_w)
            out_height = int(equi_eye_h)
        else:
            if output_format == "VR180 Equirect (TB)":
                out_width  = int(equi_eye_w)
                out_height = int(equi_eye_h) * 2
            elif output_format == "VR180 Equirect (SBS)":
                out_width  = int(equi_eye_w) * 2
                out_height = int(equi_eye_h)

    if use_ffmpeg:
        ffmpeg_cmd = [
            "ffmpeg","-y",
            "-f","rawvideo","-vcodec","rawvideo",
            "-pix_fmt", "rgb48le" if preserve_hdr10 else "bgr24",
            "-s", f"{out_width}x{out_height}",
            "-r", str(fps),
            "-i","-",
            "-an",
            "-c:v", selected_ffmpeg_codec,
        ]


        is_nvenc = "nvenc" in selected_ffmpeg_codec         # h264_nvenc/hevc_nvenc/av1_nvenc

        if preserve_hdr10:
            ffmpeg_cmd += [
                "-pix_fmt","p010le",
                "-color_range","tv",
                "-colorspace","bt2020nc",
                "-color_primaries","bt2020",
                "-color_trc","smpte2084",
            ]

            if is_nvenc:
                ffmpeg_cmd += [
                    "-preset","p5",               # NVENC preset (p1 fastest…p7 slowest)
                    "-tune","hq",
                    "-rc","vbr",
                    "-cq", str(crf_value),        # you’re using this as “quality” knob
                    "-b:v","0",
                    "-profile:v","main10",
                ]
            elif selected_ffmpeg_codec == "libx265":
                ffmpeg_cmd += [
                    "-preset","slow",
                    "-crf", str(crf_value),
                    "-x265-params",
                    "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc"
                ]
            elif selected_ffmpeg_codec in {"h264_amf", "hevc_amf", "av1_amf"}:
                ffmpeg_cmd += ["-quality", "quality", "-rc", "cqp", "-qp_i", str(crf_value), "-qp_p", str(crf_value)]
            else:
                ffmpeg_cmd += ["-preset","slow","-crf", str(crf_value)]

        else:
            # SDR
            if is_nvenc:
                ffmpeg_cmd += [
                    "-preset","p5",
                    "-tune","hq",
                    "-rc","vbr",
                    "-cq", str(crf_value),   # reuse your CRF slider as NVENC CQ
                    "-b:v","0",
                    "-pix_fmt","yuv420p",
                ]
            elif selected_ffmpeg_codec in {"h264_amf", "hevc_amf", "av1_amf"}:
                ffmpeg_cmd += [
                    "-quality", "quality",
                    "-rc", "cqp",
                    "-qp_i", str(crf_value),
                    "-qp_p", str(crf_value),
                    "-pix_fmt","yuv420p",
                ]
            else:
                ffmpeg_cmd += [
                    "-preset","slow",
                    "-crf", str(crf_value),
                    "-pix_fmt","yuv420p",
                ]

        ffmpeg_cmd.append(output_path)
        print("[OUT]", "format=", output_format, "eye_mode=", eye_mode,
              "out=", out_width, out_height,
              "equi_eye=", equi_eye_w, equi_eye_h,
              "flat_eye=", flat_eye_w, flat_eye_h)
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **hidden_subprocess_kwargs(),
        )


    else:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*selected_codec), fps, (out_width, out_height))
        if not out.isOpened():
            print("❌ OpenCV VideoWriter failed to open. Check codec/fourcc and path.")
            cap.release(); dcap.release()
            return

    start_time = time.time()
    prev_time = time.time()
    fps_values = []
    smoother = ShiftSmoother(0.15)
    global temporal_depth_filter
    temporal_depth_filter = TemporalDepthFilter(alpha=0.5)

    avg_fps = 0
    prev_depth_tensor = None
    focal_tracker = FocalDepthTracker(alpha=0.15, deadband=0.03, max_step=0.02)
    matte_ema = MatteEMA(alpha=ROTO_EMA_ALPHA)
        
    # Decide how many frames to process (for loop + progress)
    total_frames = clip_total_frames if clip_total_frames > 0 else total_frames_full
    zero_parallax_offset = 0.0
    
    # --- VR180 grid cache (build once) ---
    vr180_grid = None
    vr180_valid = None
    if vr180_enabled:
        vr180_grid, vr180_valid = build_vr180_equirect_grid(
            src_w=eye_w, src_h=eye_h,          # flat working size
            out_w=equi_eye_w, out_h=equi_eye_h, # equirect per-eye size
            src_hfov_deg=float(vr180_hfov_deg),
        )
    
    try:
        for idx in range(total_frames):
            if cancel_flag.is_set():
                break

            if idx > 0:
                ret1, frame_tensor, frame = read_next_frame()
                ret2, depth = dcap.read()
                if not ret1 or not ret2:
                    break

            # ⏸ pause handling (must be inside the loop so idx is defined)
            while suspend_flag.is_set():
                if cancel_flag.is_set():
                    break
                try:
                    time.sleep(0.2)
                except KeyboardInterrupt:
                    print("⚡ KeyboardInterrupt during suspend. Forcing cancel.")
                    cancel_flag.set()
                    break
                if progress_label:
                    elapsed = time.time() - global_session_start_time
                    elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
                    percent = (idx / total_frames) * 100
                    eta = (total_frames - idx) / avg_fps if avg_fps > 0 else 0
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                    progress_label.config(
                        text=f"{percent:.2f}% | FPS: {avg_fps:.2f} | Elapsed: {elapsed_str} | ETA: {eta_str} ⏸️ Paused"
                    )
                    progress_label.update()

            if cancel_flag.is_set():
                break

            depth_tensor = depth_to_tensor(depth, invert_depth=True)

            if auto_crop_black_bars:
                # Reuse the first-frame crop for the entire clip so frame and depth
                # stay perfectly aligned and do not jitter.
                frame_tensor, _ = crop_black_bars_torch(frame_tensor, cached_crop)
                depth_tensor, _ = crop_black_bars_torch(depth_tensor, cached_crop)

            _, h, w = frame_tensor.shape
            current_ratio = w / h
            if abs(current_ratio - target_ratio) > 0.01:
                if current_ratio > target_ratio:
                    new_w = int(h * target_ratio)
                    start = (w - new_w) // 2
                    frame_tensor = frame_tensor[:, :, start:start + new_w]
                    depth_tensor = depth_tensor[:, :, start:start + new_w]
                else:
                    new_h = int(w / target_ratio)
                    start = (h - new_h) // 2
                    frame_tensor = frame_tensor[:, start:start + new_h, :]
                    depth_tensor = depth_tensor[:, start:start + new_h, :]

            # resize tensors to fixed per-eye target (computed once)
            frame_tensor = F.interpolate(frame_tensor.unsqueeze(0),
                                         size=(eye_h, eye_w),
                                         mode='bilinear', align_corners=False).squeeze(0)
            depth_tensor = F.interpolate(depth_tensor.unsqueeze(0),
                                         size=(eye_h, eye_w),
                                         mode='bilinear', align_corners=False).squeeze(0)

            # --- Depth-Roto Assist (optional) BEFORE temporal filters ---
            if ENABLE_DEPTH_ROTO:
                # convert current depth to u8
                depth_u8 = (depth_tensor.squeeze(0).clamp(0,1).cpu().numpy() * 255.0).astype(np.uint8)

                # load or generate matte for this frame index
                mask_u8 = None
                if ROTO_MASK_DIR is not None:
                    # build filename from absolute frame index (clip start offset + idx)
                    abs_idx = start_frame_idx + idx
                    mask_path = os.path.join(ROTO_MASK_DIR, f"frame_{abs_idx:06d}.png")
                    if os.path.exists(mask_path):
                        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            # ensure matte matches current tensor size (after crop/resize)
                            m = cv2.resize(m, (eye_w, eye_h), interpolation=cv2.INTER_NEAREST)
                            mask_u8 = m
                            
                # (Optional) if you don’t have external mattes yet, you could auto-seg here.
                # e.g., mask_u8 = my_autoseg(frame_tensor)  # expect 0/255 uint8

                if mask_u8 is not None:
                    mask_u8 = matte_ema.step(mask_u8)  # temporal stabilize matte

                    depth_u8 = sculpt_depth_u8(
                        depth_u8, mask_u8,
                        near=ROTO_NEAR, far=ROTO_FAR,
                        feather_px=ROTO_FEATHER_PX, round_gamma=ROTO_ROUND_GAMMA
                    )
                    # back to tensor [1,H,W] in 0..1
                    depth_tensor = torch.from_numpy(depth_u8).to(frame_tensor.device).float().unsqueeze(0) / 255.0

            # Continue with your existing temporal/percentile normalization
            depth_tensor = temporal_depth_filter.smooth(depth_tensor)
            depth_tensor = depth_ema_norm.normalize(depth_tensor)
            
            fg, mg, bg = smoother.smooth(fg_shift, mg_shift, bg_shift)

            # dynamic IPD scale
            dyn_scale = 1.0
            fg *= dyn_scale; mg *= dyn_scale; bg *= dyn_scale

            shift_meta = {
                "subject_depth": 0.5,
                "zero_parallax_offset": 0.0,
                "edge_violation_left": 0.0,
                "edge_violation_right": 0.0,
                "repair_mask_left_mean": 0.0,
                "repair_mask_right_mean": 0.0,
                "protect_mask_left_mean": 0.0,
                "protect_mask_right_mean": 0.0,
                "valid_left_p01": 1.0,
                "valid_right_p01": 1.0,
            }

            if (blank_offset + idx) in blank_frames:
                print(f"⏩ Skipping blank frame {idx}")
                if preserve_hdr10:
                    # frame is None in HDR mode, so use the current tensor as both eyes
                    left_frame = frame_tensor
                    right_frame = frame_tensor
                else:
                    left_frame = frame
                    right_frame = frame
            else:
                fg_run, mg_run, bg_run = fg, mg, bg
                if ipd_factor != 0.0:
                    fg_run *= ipd_factor
                    mg_run *= ipd_factor
                    bg_run *= ipd_factor

                left_frame, right_frame, shift_meta = pixel_shift_cuda(
                    frame_tensor,
                    depth_tensor,
                    eye_w,
                    eye_h,
                    fg_run,
                    mg_run,
                    bg_run,
                    blur_ksize=blur_ksize,
                    feather_strength=feather_strength,
                    use_subject_tracking=use_subject_tracking,
                    enable_floating_window=use_floating_window,
                    return_shift_map=True,
                    max_pixel_shift_percent=max_pixel_shift_percent,
                    zero_parallax_strength=zero_parallax_strength,
                    enable_edge_masking=enable_edge_masking,
                    enable_feathering=enable_feathering,
                    dof_strength=dof_strength,
                    convergence_strength=convergence_strength,
                    enable_dynamic_convergence=enable_dynamic_convergence,
                    depth_pop_gamma=depth_pop_gamma,
                    depth_pop_mid=depth_pop_mid,
                    depth_stretch_lo=depth_stretch_lo,
                    depth_stretch_hi=depth_stretch_hi,
                    fg_pop_multiplier=fg_pop_multiplier,
                    bg_push_multiplier=bg_push_multiplier,
                    subject_lock_strength=subject_lock_strength,
                    return_tensors=True,
                    disable_shift_ema=disable_shift_ema,
                )
                
                if idx % 24 == 0:
                    print(
                        f"[3DDBG] f={idx} "
                        f"subj={shift_meta.get('subject_depth', 0.0):.3f} "
                        f"zpo={shift_meta.get('zero_parallax_offset', 0.0):.5f} "
                        f"evL={shift_meta.get('edge_violation_left', 0.0):.5f} "
                        f"evR={shift_meta.get('edge_violation_right', 0.0):.5f} "
                        f"rL={shift_meta.get('repair_mask_left_mean', 0.0):.4f} "
                        f"rR={shift_meta.get('repair_mask_right_mean', 0.0):.4f} "
                        f"pL={shift_meta.get('protect_mask_left_mean', 0.0):.4f} "
                        f"pR={shift_meta.get('protect_mask_right_mean', 0.0):.4f} "
                        f"vL01={shift_meta.get('valid_left_p01', 1.0):.4f} "
                        f"vR01={shift_meta.get('valid_right_p01', 1.0):.4f}"
                    )
                
                candidate_focal = estimate_subject_depth(depth_tensor)  # 0..1
                motion_metric   = compute_motion_metric(prev_depth_tensor, depth_tensor)
                focal_tracker.set_scene_motion(motion_metric)
                focal_depth     = focal_tracker.update(candidate_focal)

                if need_dof or need_color:
                    # 1) to tensors once
                    left_t  = left_frame
                    right_t = right_frame

                    # 2) match depth to the eye frame once
                    H, W = left_t.shape[1], left_t.shape[2]
                    depth_for_eye = F.interpolate(
                        depth_tensor.unsqueeze(0), size=(H, W),
                        mode='bilinear', align_corners=False
                    ).squeeze(0)  # -> [1,H,W]

                    # 3) DOF first (if enabled)
                    if need_dof:
                        left_t  = apply_dof_cuda(left_t,  depth_for_eye, focal_depth,
                                                 max_sigma=dof_strength, focus_width=0.35)
                        right_t = apply_dof_cuda(right_t, depth_for_eye, focal_depth,
                                                 max_sigma=dof_strength, focus_width=0.35)

                    # 4) Color grading next (if non-neutral)
                    if need_color:
                        left_t  = apply_color_grade(left_t,
                                                    saturation=color_saturation,
                                                    contrast=color_contrast,
                                                    brightness=color_brightness)
                        right_t = apply_color_grade(right_t,
                                                    saturation=color_saturation,
                                                    contrast=color_contrast,
                                                    brightness=color_brightness)

                    # 5) back to numpy for SDR only
                    if preserve_hdr10:
                        # keep tensors for HDR pipe
                        left_frame  = left_t
                        right_frame = right_t
                    else:
                        left_frame  = tensor_to_frame(left_t)
                        right_frame = tensor_to_frame(right_t)


            subject_depth_val = float(shift_meta.get("subject_depth", 0.5))
            zero_parallax_offset = float(shift_meta.get("zero_parallax_offset", 0.0))

            # --- Dynamic Floating Window (shared compute, HDR + SDR) ---
            dfw_apply = False
            dfw_side = "left"
            dfw_width = 0

            if use_floating_window:
                global dfw_last_side, dfw_last_width

                if "dfw_last_side" not in globals():
                    dfw_last_side = "left"
                    dfw_last_width = 0

                edge_violation_left = float(shift_meta.get("edge_violation_left", 0.0))
                edge_violation_right = float(shift_meta.get("edge_violation_right", 0.0))

                parallax_mag = abs(zero_parallax_offset)
                edge_violation_mag = max(edge_violation_left, edge_violation_right)

                if max(parallax_mag, edge_violation_mag) < DFW_MIN_PARALLAX:
                    target_width = 0
                else:
                    depth_delta = abs(subject_depth_val - 0.5)

                    parallax_delta = (
                        DFW_PARALLAX_WEIGHT * max(parallax_mag, edge_violation_mag) +
                        DFW_DEPTH_WEIGHT   * depth_delta
                    )

                    parallax_delta = min(parallax_delta, 0.12)

                    target_width = int(width_for_bars * parallax_delta)
                    max_bar_px   = int(width_for_bars * DFW_MAX_BAR_FRAC)
                    target_width = max(0, min(target_width, max_bar_px))

                    if edge_violation_left > edge_violation_right:
                        dfw_last_side = "left"
                    elif edge_violation_right > edge_violation_left:
                        dfw_last_side = "right"
                    else:
                        dfw_last_side = "left" if zero_parallax_offset > 0.0 else "right"

                dfw_last_width = int(
                    DFW_WIDTH_EASE * dfw_last_width +
                    (1.0 - DFW_WIDTH_EASE) * target_width
                )

                dfw_side = dfw_last_side
                dfw_width = dfw_last_width
                dfw_apply = (dfw_width > 1)            

            if preserve_hdr10 and not use_ffmpeg:
                raise RuntimeError("HDR10 output requires FFmpeg. OpenCV VideoWriter is SDR-only in this pipeline.")

            if not preserve_hdr10:
                if torch.is_tensor(left_frame):
                    left_frame = tensor_to_frame(left_frame)
                if torch.is_tensor(right_frame):
                    right_frame = tensor_to_frame(right_frame)
                    
            # sharpen & pack
            if preserve_hdr10:
                # left_frame/right_frame are torch tensors [3,H,W] RGB float 0..1

                # 1) Sharpen in tensor space
                left_t  = tensor_apply_sharpen(left_frame,  sharpness_factor)
                right_t = tensor_apply_sharpen(right_frame, sharpness_factor)

                # 2) Size handling before final packing
                if vr180_enabled:
                    # Keep at FLAT working res (eye_w x eye_h) for projection step
                    # If anything drifted, enforce it:
                    left_t  = F.interpolate(left_t.unsqueeze(0),  size=(eye_h, eye_w), mode="bilinear", align_corners=False).squeeze(0)
                    right_t = F.interpolate(right_t.unsqueeze(0), size=(eye_h, eye_w), mode="bilinear", align_corners=False).squeeze(0)

                elif output_format == "Half-SBS":
                    left_t  = F.interpolate(left_t.unsqueeze(0),  size=(per_eye_h, per_eye_w), mode="bilinear", align_corners=False).squeeze(0)
                    right_t = F.interpolate(right_t.unsqueeze(0), size=(per_eye_h, per_eye_w), mode="bilinear", align_corners=False).squeeze(0)

                else:
                    left_t  = tensor_pad_to_aspect(left_t,  per_eye_w, per_eye_h)
                    right_t = tensor_pad_to_aspect(right_t, per_eye_w, per_eye_h)

                # 3) Dynamic Floating Window, apply in tensor space
                if dfw_apply:
                    left_t  = tensor_apply_side_mask(
                        left_t, side=dfw_side, width=dfw_width,
                        fade=DFW_USE_FADE, solid_black=(not DFW_USE_FADE)
                    )
                    right_t = tensor_apply_side_mask(
                        right_t, side=dfw_side, width=dfw_width,
                        fade=DFW_USE_FADE, solid_black=(not DFW_USE_FADE)
                    )

                # 3.5) VR180 projection (flat -> equirect per eye)
                if vr180_enabled:
                    left_t  = warp_eye_to_vr180_equirect(left_t,  vr180_grid, vr180_valid)
                    right_t = warp_eye_to_vr180_equirect(right_t, vr180_grid, vr180_valid)
                    
                # 4) Final pack as tensor
                if eye_mode == "left":
                    final_tensor = left_t
                elif eye_mode == "right":
                    final_tensor = right_t
                else:
                    if output_format == "VR180 Equirect (TB)":
                        final_tensor = pack_stereo_tb(left_t, right_t)
                    elif output_format == "VR180 Equirect (SBS)":
                        final_tensor = pack_stereo_sbs(left_t, right_t)
                    else:
                        final_tensor = torch.cat([left_t, right_t], dim=2)  # your existing SBS


                # Optional: if you really need Passive Interlaced in HDR, do it in tensor space
                if (eye_mode == "sbs") and (output_format == "Passive Interlaced"):
                    # interlace rows: even rows left, odd rows right, output is single-eye size
                    H, W2 = final_tensor.shape[1], final_tensor.shape[2]
                    W = W2 // 2
                    left_eye  = final_tensor[:, :, :W]
                    right_eye = final_tensor[:, :, W:]
                    inter = left_eye.clone()
                    inter[:, 1::2, :] = right_eye[:, 1::2, :]
                    final_tensor = inter

            else:
                # SDR numpy path
                left_sharp  = apply_sharpening(left_frame, sharpness_factor)
                right_sharp = apply_sharpening(right_frame, sharpness_factor)

                if vr180_enabled:
                    # always flat working size for projection
                    left_out  = cv2.resize(left_sharp,  (eye_w, eye_h), interpolation=cv2.INTER_AREA)
                    right_out = cv2.resize(right_sharp, (eye_w, eye_h), interpolation=cv2.INTER_AREA)
                else:
                    if output_format == "Full-SBS":
                        left_out  = pad_to_aspect_ratio(left_sharp,  per_eye_w, per_eye_h)
                        right_out = pad_to_aspect_ratio(right_sharp, per_eye_w, per_eye_h)
                    elif output_format == "Half-SBS":
                        left_out  = cv2.resize(left_sharp,  (per_eye_w, per_eye_h), interpolation=cv2.INTER_AREA)
                        right_out = cv2.resize(right_sharp, (per_eye_w, per_eye_h), interpolation=cv2.INTER_AREA)
                    else:
                        left_out  = pad_to_aspect_ratio(left_sharp,  per_eye_w, per_eye_h)
                        right_out = pad_to_aspect_ratio(right_sharp, per_eye_w, per_eye_h)

                # Dynamic Floating Window stays the same for SDR
                if dfw_apply:
                    if DFW_USE_FADE:
                        left_out  = apply_side_mask(left_out,  side=dfw_side, width=dfw_width, fade=True,  solid_black=False)
                        right_out = apply_side_mask(right_out, side=dfw_side, width=dfw_width, fade=True,  solid_black=False)
                    else:
                        left_out  = apply_side_mask(left_out,  side=dfw_side, width=dfw_width, fade=False, solid_black=True)
                        right_out = apply_side_mask(right_out, side=dfw_side, width=dfw_width, fade=False, solid_black=True)


                # Decide output in SDR path
                if vr180_enabled:
                    # project flat -> equirect and pack, and set final
                    left_t  = frame_to_tensor(left_out)
                    right_t = frame_to_tensor(right_out)

                    left_t  = F.interpolate(left_t.unsqueeze(0),  size=(eye_h, eye_w), mode="bilinear", align_corners=False).squeeze(0)
                    right_t = F.interpolate(right_t.unsqueeze(0), size=(eye_h, eye_w), mode="bilinear", align_corners=False).squeeze(0)

                    left_t  = warp_eye_to_vr180_equirect(left_t,  vr180_grid, vr180_valid)
                    right_t = warp_eye_to_vr180_equirect(right_t, vr180_grid, vr180_valid)

                    if eye_mode == "left":
                        final_t = left_t
                    elif eye_mode == "right":
                        final_t = right_t
                    else:
                        if output_format == "VR180 Equirect (TB)":
                            final_t = pack_stereo_tb(left_t, right_t)
                        else:
                            final_t = pack_stereo_sbs(left_t, right_t)

                    final = tensor_to_frame(final_t)

                else:
                    if eye_mode == "left":
                        final = left_out
                    elif eye_mode == "right":
                        final = right_out
                    else:
                        final = format_3d_output(left_out, right_out, output_format)

            # write frame
            if use_ffmpeg:
                try:
                    if preserve_hdr10:
                        # ✅ HDR10 path: write 16-bit RGB (rgb48le) to ffmpeg stdin
                        # Expectation: you must be generating a float RGB tensor [3,H,W] in 0..1
                        # (example name: final_tensor). If you only have `final` as uint8 BGR,
                        # you are NOT preserving HDR10.
                        ffmpeg_proc.stdin.write(tensor_to_rgb48_bytes(final_tensor))
                    else:
                        # SDR path: write 8-bit BGR
                        ffmpeg_proc.stdin.write(final.astype(np.uint8).tobytes())

                except Exception as e:
                    print(f"❌ FFmpeg write error: {e}")
                    break
            else:
                out.write(final)
                
            if end_s is not None:
                cur_abs_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if cur_abs_idx >= end_frame_idx:
                    break
               
            # progress / fps
            percent = ((idx + 1) / max(total_frames, 1)) * 100.0
            elapsed = time.time() - global_session_start_time
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))

            curr_time = time.time()
            delta = curr_time - prev_time
            if delta > 0:
                fps_values.append(1.0 / delta)
                if len(fps_values) > 10:
                    fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

            if progress:
                progress["value"] = percent
                progress.update()
            remaining_frames = total_frames - (idx + 1)
            eta = remaining_frames / avg_fps if avg_fps > 0 else 0
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))

            if progress_label:
                progress_label.config(
                    text=f"{percent:.2f}% | FPS: {avg_fps:.2f} | Elapsed: {elapsed_str} | ETA: {eta_str}"
                )
                
            prev_depth_tensor = depth_tensor.detach()
            prev_time = curr_time

        # ✅ final progress update (inside try)
        if progress:
            progress["value"] = 100
            progress.update()
        if progress_label:
            elapsed = time.time() - global_session_start_time
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
            progress_label.config(
                text=f"100.00% | FPS: {avg_fps:.2f} | Elapsed: {elapsed_str} | ETA: 00:00:00"
            )

    except Exception as e:
        print(f"❌ Render crashed: {e}")

    finally:
        cap.release(); dcap.release()
        if use_ffmpeg and ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
            except:
                pass
            try:
                if cancel_flag.is_set():
                    ffmpeg_proc.kill()
                else:
                    ffmpeg_proc.wait(timeout=5)
            except:
                pass
        elif out is not None:
            try:
                out.release()
            except:
                pass

        torch.cuda.empty_cache()
        if global_session_start_time is not None:
            total_time = time.time() - global_session_start_time
            print(f"✅ Render complete in {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
            global_session_start_time = None

        return output_path  

def render_sbs_3d_image(
    input_image_path: str,
    depth_image_path: str,
    output_image_path: str,
    fg_shift: float,
    mg_shift: float,
    bg_shift: float,
    sharpness_factor: float,
    output_format: str,
    selected_aspect_ratio,
    aspect_ratios,
    feather_strength: float = 0.0,
    blur_ksize: int = 1,
    use_subject_tracking: bool = False,
    use_floating_window: bool = False,
    max_pixel_shift_percent: float = 0.02,
    auto_crop_black_bars: bool = False,
    parallax_balance: float = 0.8,
    zero_parallax_strength: float = 0.0,
    enable_edge_masking: bool = True,
    enable_feathering: bool = True,
    dof_strength: float = 0.0,
    convergence_strength: float = 0.0,
    enable_dynamic_convergence: bool = True,
    ipd_factor: float = 1.0,
    depth_pop_gamma: float = 0.85,
    depth_pop_mid: float = 0.50,
    depth_stretch_lo: float = 0.05,
    depth_stretch_hi: float = 0.95,
    fg_pop_multiplier: float = 1.20,
    bg_push_multiplier: float = 1.10,
    subject_lock_strength: float = 1.00,
    color_saturation: float = 1.0,
    color_contrast: float = 1.0,
    color_brightness: float = 0.0,
    eye_mode: str = "sbs",
    disable_shift_ema: bool = False,
):
    reset_render_state()

    """
    Single image version of render_sbs_3d.
    Runs pixel_shift_cuda with the same depth shaping, parallax logic, and
    floating window as the video path, then writes a single 3D frame to disk.
    """

    # Support Tk variables or plain Python types
    def _val(v):
        return v.get() if hasattr(v, "get") else v

    fg_shift           = float(_val(fg_shift))
    mg_shift           = float(_val(mg_shift))
    bg_shift           = float(_val(bg_shift))
    sharpness_factor   = float(_val(sharpness_factor))
    feather_strength   = float(_val(feather_strength))
    blur_ksize         = int(_val(blur_ksize))
    use_subject_tracking = bool(_val(use_subject_tracking))
    use_floating_window  = bool(_val(use_floating_window))
    max_pixel_shift_percent = float(_val(max_pixel_shift_percent))
    auto_crop_black_bars   = bool(_val(auto_crop_black_bars))
    parallax_balance       = float(_val(parallax_balance))
    zero_parallax_strength = float(_val(zero_parallax_strength))
    enable_edge_masking    = bool(_val(enable_edge_masking))
    enable_feathering      = bool(_val(enable_feathering))
    dof_strength           = float(_val(dof_strength))
    convergence_strength   = float(_val(convergence_strength))
    enable_dynamic_convergence = bool(_val(enable_dynamic_convergence))
    ipd_factor             = float(_val(ipd_factor))
    depth_pop_gamma        = float(_val(depth_pop_gamma))
    depth_pop_mid          = float(_val(depth_pop_mid))
    depth_stretch_lo       = float(_val(depth_stretch_lo))
    depth_stretch_hi       = float(_val(depth_stretch_hi))
    fg_pop_multiplier      = float(_val(fg_pop_multiplier))
    bg_push_multiplier     = float(_val(bg_push_multiplier))
    subject_lock_strength  = float(_val(subject_lock_strength))
    color_saturation       = float(_val(color_saturation))
    color_contrast         = float(_val(color_contrast))
    color_brightness       = float(_val(color_brightness))
    output_format          = _val(output_format)

    # Resolve aspect ratio key from Tk StringVar or plain string
    if hasattr(selected_aspect_ratio, "get"):
        ar_key = selected_aspect_ratio.get()
    else:
        ar_key = selected_aspect_ratio
    target_ratio = aspect_ratios.get(ar_key, 16.0 / 9.0)

    # Load images
    frame = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_image_path, cv2.IMREAD_COLOR)

    if frame is None:
        print(f"❌ Could not read input image: {input_image_path}")
        return None
    if depth is None:
        print(f"❌ Could not read depth image: {depth_image_path}")
        return None

    frame_tensor = frame_to_tensor(frame)
    depth_tensor = depth_to_tensor(depth)

    # Optional black bar crop (same logic as video path)
    cached_crop = (0, 0)
    if auto_crop_black_bars:
        top_crop, bottom_crop = detect_black_bars(frame_tensor)
        cached_crop = (top_crop, bottom_crop)
        frame_tensor, _ = crop_black_bars_torch(frame_tensor, cached_crop)
        depth_tensor, _ = crop_black_bars_torch(depth_tensor, cached_crop)


    # Crop to selected cinema aspect ratio
    _, h, w = frame_tensor.shape
    current_ratio = w / h
    if abs(current_ratio - target_ratio) > 0.01:
        if current_ratio > target_ratio:
            # frame is wider than target, crop left/right
            new_w = int(h * target_ratio)
            start = (w - new_w) // 2
            frame_tensor = frame_tensor[:, :, start:start + new_w]
            depth_tensor = depth_tensor[:, :, start:start + new_w]
        else:
            # frame is taller than target, crop top/bottom
            new_h = int(w / target_ratio)
            start = (h - new_h) // 2
            frame_tensor = frame_tensor[:, start:start + new_h, :]
            depth_tensor = depth_tensor[:, start:start + new_h, :]

    resized_height = frame_tensor.shape[1]
    resized_width  = frame_tensor.shape[2]

    # For stills we preserve original per eye aspect
    if output_format == "Full-SBS":
        per_eye_w = resized_width
        per_eye_h = resized_height
        out_width = per_eye_w * 2
        out_height = per_eye_h
    elif output_format == "Half-SBS":
        per_eye_w = resized_width // 2
        per_eye_h = resized_height
        out_width = resized_width
        out_height = resized_height
    elif output_format == "VR":
        per_eye_w = 1440
        per_eye_h = 1600
        out_width = per_eye_w * 2
        out_height = per_eye_h
    elif output_format == "Red-Cyan Anaglyph":
        per_eye_w = resized_width
        per_eye_h = resized_height
        out_width = resized_width
        out_height = resized_height
    elif output_format == "Passive Interlaced":
        per_eye_w = resized_width
        per_eye_h = resized_height
        out_width = resized_width
        out_height = resized_height
    else:
        per_eye_w = resized_width
        per_eye_h = resized_height
        out_width = resized_width * 2
        out_height = resized_height

    eye_w = per_eye_w
    eye_h = per_eye_h

    # Floating window math should always use per-eye width (VR, SBS, single-eye all consistent)
    width_for_bars = per_eye_w

    need_dof = (dof_strength > 0.0)
    need_color = (
        (color_saturation != 1.0) or
        (color_contrast != 1.0) or
        (abs(color_brightness) > 1e-6)
    )

    # Resize tensors to per eye target
    frame_tensor = F.interpolate(
        frame_tensor.unsqueeze(0),
        size=(eye_h, eye_w),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    depth_tensor = F.interpolate(
        depth_tensor.unsqueeze(0),
        size=(eye_h, eye_w),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    # Optional depth roto, same style as video, using frame index 0
    matte_ema = MatteEMA(alpha=ROTO_EMA_ALPHA)
    if ENABLE_DEPTH_ROTO:
        depth_u8 = (depth_tensor.squeeze(0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        mask_u8 = None
        if ROTO_MASK_DIR is not None:
            mask_path = os.path.join(ROTO_MASK_DIR, "frame_000000.png")
            if os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    m = cv2.resize(m, (eye_w, eye_h), interpolation=cv2.INTER_NEAREST)
                    mask_u8 = m

        if mask_u8 is not None:
            mask_u8 = matte_ema.step(mask_u8)
            depth_u8 = sculpt_depth_u8(
                depth_u8,
                mask_u8,
                near=ROTO_NEAR,
                far=ROTO_FAR,
                feather_px=ROTO_FEATHER_PX,
                round_gamma=ROTO_ROUND_GAMMA,

            )
            depth_tensor = torch.from_numpy(depth_u8).to(frame_tensor.device).float().unsqueeze(0) / 255.0

    # Temporal smoothing is not needed on a single still, but we keep the same
    # normalization path so depth range behaves like video
    local_temporal = TemporalDepthFilter(alpha=0.5)
    depth_tensor = local_temporal.smooth(depth_tensor)
    depth_tensor = depth_ema_norm.normalize(depth_tensor)

    # Shift smoothing and dynamic parallax scale, same as video
    smoother = ShiftSmoother(alpha=0.15)
    fg, mg, bg = smoother.smooth(fg_shift, mg_shift, bg_shift)

    try:
        dyn_scale = compute_dynamic_parallax_scale(depth_tensor, min_scale=0.90, max_scale=1.15)
    except Exception:
        dyn_scale = 1.0

    fg *= dyn_scale
    mg *= dyn_scale
    bg *= dyn_scale

    if ipd_factor != 0.0:
        fg *= ipd_factor
        mg *= ipd_factor
        bg *= ipd_factor

    # Run your CUDA pixel shift exactly like the video pipeline
    left_frame, right_frame, shift_meta = pixel_shift_cuda(
        frame_tensor,
        depth_tensor,
        eye_w,
        eye_h,
        fg,
        mg,
        bg,
        blur_ksize=blur_ksize,
        feather_strength=feather_strength,
        use_subject_tracking=use_subject_tracking,
        enable_floating_window=use_floating_window,
        return_shift_map=True,
        return_tensors=True,
        max_pixel_shift_percent=max_pixel_shift_percent,
        zero_parallax_strength=zero_parallax_strength,
        enable_edge_masking=enable_edge_masking,
        enable_feathering=enable_feathering,
        dof_strength=dof_strength,
        convergence_strength=convergence_strength,
        enable_dynamic_convergence=enable_dynamic_convergence,
        depth_pop_gamma=depth_pop_gamma,
        depth_pop_mid=depth_pop_mid,
        depth_stretch_lo=depth_stretch_lo,
        depth_stretch_hi=depth_stretch_hi,
        fg_pop_multiplier=fg_pop_multiplier,
        bg_push_multiplier=bg_push_multiplier,
        subject_lock_strength=subject_lock_strength,
        disable_shift_ema=disable_shift_ema,
    )

    # Optional DOF and color grade, same order as video
    if need_dof or need_color:
        left_t = frame_to_tensor(left_frame)
        right_t = frame_to_tensor(right_frame)

        H, W = left_t.shape[1], left_t.shape[2]
        depth_for_eye = F.interpolate(
            depth_tensor.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        if need_dof:
            focal_depth = estimate_subject_depth(depth_tensor)
            left_t = apply_dof_cuda(
                left_t,
                depth_for_eye,
                focal_depth,
                max_sigma=dof_strength,
                focus_width=0.35,
            )
            right_t = apply_dof_cuda(
                right_t,
                depth_for_eye,
                focal_depth,
                max_sigma=dof_strength,
                focus_width=0.35,
            )

        if need_color:
            left_t = apply_color_grade(
                left_t,
                saturation=color_saturation,
                contrast=color_contrast,
                brightness=color_brightness,
            )
            right_t = apply_color_grade(
                right_t,
                saturation=color_saturation,
                contrast=color_contrast,
                brightness=color_brightness,
            )

        left_frame = tensor_to_frame(left_t)
        right_frame = tensor_to_frame(right_t)

    # Sharpen and size per eye
    left_sharp = apply_sharpening(left_frame, sharpness_factor)
    right_sharp = apply_sharpening(right_frame, sharpness_factor)

    if output_format == "Full-SBS":
        left_out = pad_to_aspect_ratio(left_sharp, per_eye_w, per_eye_h)
        right_out = pad_to_aspect_ratio(right_sharp, per_eye_w, per_eye_h)
    elif output_format == "Half-SBS":
        left_out = cv2.resize(left_sharp, (per_eye_w, per_eye_h), interpolation=cv2.INTER_AREA)
        right_out = cv2.resize(right_sharp, (per_eye_w, per_eye_h), interpolation=cv2.INTER_AREA)
    elif output_format in ("VR", "Red-Cyan Anaglyph", "Passive Interlaced"):
        left_out = pad_to_aspect_ratio(left_sharp, per_eye_w, per_eye_h)
        right_out = pad_to_aspect_ratio(right_sharp, per_eye_w, per_eye_h)
    else:
        left_out = pad_to_aspect_ratio(left_sharp, per_eye_w, per_eye_h)
        right_out = pad_to_aspect_ratio(right_sharp, per_eye_w, per_eye_h)

    # Dynamic floating window, same logic as video (one frame)
    if use_floating_window:
        global dfw_last_side, dfw_last_width

        if "dfw_last_side" not in globals():
            dfw_last_side = "left"
            dfw_last_width = 0

        subject_depth_val = float(shift_meta.get("subject_depth", 0.5))
        zero_parallax_offset = float(shift_meta.get("zero_parallax_offset", 0.0))

        edge_violation_left = float(shift_meta.get("edge_violation_left", 0.0))
        edge_violation_right = float(shift_meta.get("edge_violation_right", 0.0))

        parallax_mag = abs(zero_parallax_offset)
        edge_violation_mag = max(edge_violation_left, edge_violation_right)

        if max(parallax_mag, edge_violation_mag) < DFW_MIN_PARALLAX:
            target_width = 0
        else:
            depth_delta = abs(subject_depth_val - 0.5)

            parallax_delta = (
                DFW_PARALLAX_WEIGHT * max(parallax_mag, edge_violation_mag)
                + DFW_DEPTH_WEIGHT * depth_delta
            )
            parallax_delta = min(parallax_delta, 0.12)

            target_width = int(width_for_bars * parallax_delta)
            max_bar_px = int(width_for_bars * DFW_MAX_BAR_FRAC)
            target_width = max(0, min(target_width, max_bar_px))

            if edge_violation_left > edge_violation_right:
                dfw_last_side = "left"
            elif edge_violation_right > edge_violation_left:
                dfw_last_side = "right"
            else:
                dfw_last_side = "left" if zero_parallax_offset > 0.0 else "right"

        dfw_last_width = int(
            DFW_WIDTH_EASE * dfw_last_width
            + (1.0 - DFW_WIDTH_EASE) * target_width
        )

        if dfw_last_width > 1:
            if DFW_USE_FADE:
                left_out = apply_side_mask(
                    left_out,
                    side=dfw_last_side,
                    width=dfw_last_width,
                    fade=True,
                    solid_black=False,
                )
                right_out = apply_side_mask(
                    right_out,
                    side=dfw_last_side,
                    width=dfw_last_width,
                    fade=True,
                    solid_black=False,
                )
            else:
                left_out = apply_side_mask(
                    left_out,
                    side=dfw_last_side,
                    width=dfw_last_width,
                    fade=False,
                    solid_black=True,
                )
                right_out = apply_side_mask(
                    right_out,
                    side=dfw_last_side,
                    width=dfw_last_width,
                    fade=False,
                    solid_black=True,
                )

    # Pick eye mode and format
    if eye_mode == "left":
        final = left_out
    elif eye_mode == "right":
        final = right_out
    else:
        final = format_3d_output(left_out, right_out, output_format)

    # Make sure final matches desired output size
    if final.shape[1] != out_width or final.shape[0] != out_height:
        final = cv2.resize(final, (out_width, out_height), interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_image_path, final.astype(np.uint8))
    print(f"✅ Saved 3D image to {output_image_path}")
    return output_image_path


def select_input_video(
    input_video_path,
    video_thumbnail_label,
    video_specs_label,
    update_aspect_preview,
    original_video_width,
    original_video_height
):


    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
    if not video_path:
        return

    input_video_path.set(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # ✅ Now this works without needing to import GUI.py
    original_video_width.set(width)
    original_video_height.set(height)
    
    current_video_width = width
    current_video_height = height
    
    ret, frame = cap.read()
    cap.release()

    if ret:
        THUMB_W, THUMB_H = 160, 90

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            video_thumbnail_label.config(image=img_tk)
            video_thumbnail_label.image = img_tk


        video_specs_label.config(text=f"Video Info:\nResolution: {width}x{height}\nFPS: {fps:.2f}")
    else:
        video_specs_label.config(text="Video Info:\nUnable to extract details")

    # ✅ Call the UI update function
    update_aspect_preview()


def select_output_video(output_sbs_video_path):
    output_sbs_video_path.set(
        filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("MKV files", "*.mkv"),
                ("AVI files", "*.avi"),
            ],
        )
    )


def select_depth_map(selected_depth_map, depth_map_label):
    depth_map_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
    )
    if not depth_map_path:
        return

    selected_depth_map.set(depth_map_path)
    depth_map_label.config(
        text=f"Selected Depth Map:\n{os.path.basename(depth_map_path)}"
    )
    
    
def process_video(
    input_video_path,
    selected_depth_map,
    output_sbs_video_path,
    selected_codec,
    fg_shift,
    mg_shift,
    bg_shift,
    sharpness_factor,
    output_format,
    selected_aspect_ratio,
    aspect_ratios,
    feather_strength,
    blur_ksize,
    progress,
    progress_label,
    suspend_flag,
    cancel_flag,
    use_ffmpeg,
    preserve_hdr10,
    selected_ffmpeg_codec,
    crf_value,
    nvenc_cq_value,
    use_subject_tracking,
    use_floating_window,
    max_pixel_shift,
    auto_crop_black_bars,
    parallax_balance,
    preserve_original_aspect,
    zero_parallax_strength,
    enable_edge_masking,
    enable_feathering,
    skip_blank_frames,
    dof_strength,
    convergence_strength,
    enable_dynamic_convergence,
    depth_pop_gamma,
    depth_pop_mid,
    depth_stretch_lo,
    depth_stretch_hi,
    fg_pop_multiplier,
    bg_push_multiplier,
    subject_lock_strength,
    color_saturation,
    color_contrast,
    color_brightness,
    ipd_value=0.0,
    start_s=None,
    end_s=None,
    eye_mode="sbs",
    output_override=None,
    keep_original_audio=False,
    vr180_equi_preset=None,
    vr180_flat_preset=None,
    vr180_hfov_deg=None,
    vr180_equi_w_var=None,
    vr180_equi_h_var=None,
    vr180_flat_w_var=None,
    vr180_flat_h_var=None,
    vr180_hfov_deg_var=None,
    disable_shift_ema=False,
):


    global original_video_width, original_video_height

    input_path = input_video_path.get()
    depth_path = selected_depth_map.get()
    output_path = (output_override or output_sbs_video_path.get())

    if not input_path or not output_path or not depth_path:
        messagebox.showerror(
            "Error", "Please select input video, depth map, and output path."
        )
        return
   
    codec_key = selected_ffmpeg_codec.get() if hasattr(selected_ffmpeg_codec, "get") else selected_ffmpeg_codec
    codec_val = FFMPEG_CODEC_MAP.get(codec_key, "libx264")

    hdr_on = preserve_hdr10.get() if hasattr(preserve_hdr10, "get") else bool(preserve_hdr10)
    
    width, height, fps = get_video_info_safe(input_path)

    if width <= 0 or height <= 0:
        messagebox.showerror("Error", "Unable to retrieve video dimensions from the input video.")
        return

    if fps <= 0:
        messagebox.showerror("Error", "Unable to retrieve FPS from the input video.")
        return    

    def _get(v, default=None):
        try:
            if v is None:
                return default
            if hasattr(v, "get"):
                v = v.get()
            if v is None:
                return default
            if isinstance(v, str) and not v.strip():
                return default
            return v
        except Exception:
            return default

    def _get_int(v, default):
        v = _get(v, None)
        try:
            return int(v)
        except Exception:
            return int(default)

    def _get_float(v, default):
        v = _get(v, None)
        try:
            return float(v)
        except Exception:
            return float(default)


    # 🧠 Save original dimensions globally
    original_video_width = width
    original_video_height = height
    
    # 🔄 Determine aspect ratio
    aspect_ratio = aspect_ratios.get(selected_aspect_ratio.get(), 16 / 9)
    format_selected = output_format.get()

    # eye mode comes from process_video() argument
    if hasattr(eye_mode, "get"):
        eye_mode = eye_mode.get()
    eye_mode = (eye_mode or "sbs").strip().lower()
    if eye_mode == "both":
        eye_mode = "sbs"

    # VR180 manual entries
    equi_w = equi_h = None
    flat_w = flat_h = None
    hfov   = 110.0

    if format_selected in ("VR180 Equirect (TB)", "VR180 Equirect (SBS)"):
        equi_w = _get_int(vr180_equi_w_var, 3840)
        equi_h = _get_int(vr180_equi_h_var, 1920)
        flat_w = _get_int(vr180_flat_w_var, 1920)
        flat_h = _get_int(vr180_flat_h_var, 1080)
        hfov   = _get_float(vr180_hfov_deg_var, 110.0)

        # clamp some sane limits
        hfov = max(60.0, min(140.0, hfov))

        # enforce 2:1 per-eye for equirect
        if equi_w < 256 or equi_h < 128:
            equi_w, equi_h = 3840, 1920
        if abs((equi_w / max(equi_h, 1)) - 2.0) > 0.05:
            # auto-correct height to keep 2:1
            equi_h = max(1, equi_w // 2)

        # keep flat working size reasonable
        flat_w = max(320, flat_w)
        flat_h = max(240, flat_h)

    # Calculate output dimensions based on selected format
    if preserve_original_aspect.get():
        output_width = width
        output_height = height
    else:
        if format_selected == "Full-SBS":
            output_width = width * 2
            output_height = height

        elif format_selected == "Half-SBS":
            output_width = width
            output_height = height

        elif format_selected == "Passive Interlaced":
            # same size as original frame (not SBS)
            output_width = width
            output_height = height

        elif format_selected == "VR":
            output_width = 4096
            output_height = int(output_width / aspect_ratio)

        elif format_selected == "VR180 Equirect (TB)":
            output_width = int(equi_w)
            output_height = int(equi_h) * 2

        elif format_selected == "VR180 Equirect (SBS)":
            output_width = int(equi_w) * 2
            output_height = int(equi_h)

        else:
            output_width = width
            output_height = int(output_width / aspect_ratio)


    # 🟢 Start progress
    progress["value"] = 0
    progress_label.config(text="0%")
    progress.update()

    final_render_path = None

    # 🔥 Start render process
    if format_selected in [
        "Full-SBS",
        "Half-SBS",
        "VR",
        "VR180 Equirect (TB)",
        "VR180 Equirect (SBS)",
        "Red-Cyan Anaglyph",
        "Passive Interlaced",
    ]:
        final_render_path = render_sbs_3d(
            input_path,
            depth_path,
            output_path,
            selected_codec.get(),
            fps,
            output_width,
            output_height,
            fg_shift.get(),
            mg_shift.get(),
            bg_shift.get(),
            sharpness_factor.get(),
            format_selected,
            selected_aspect_ratio,
            aspect_ratios,
            feather_strength=feather_strength.get(),
            blur_ksize=blur_ksize.get(),
            use_ffmpeg=use_ffmpeg.get(),
            preserve_hdr10=bool(hdr_on),
            selected_ffmpeg_codec=codec_val,
            crf_value=crf_value.get(),
            nvenc_cq_value=(nvenc_cq_value.get() if hasattr(nvenc_cq_value, "get") else nvenc_cq_value),
            use_subject_tracking=use_subject_tracking.get(),
            use_floating_window=use_floating_window.get(),
            max_pixel_shift_percent=max_pixel_shift.get(),
            progress=progress,
            progress_label=progress_label,
            suspend_flag=suspend_flag,
            cancel_flag=cancel_flag,
            auto_crop_black_bars=auto_crop_black_bars.get(),
            parallax_balance=parallax_balance.get(),
            preserve_original_aspect=preserve_original_aspect.get(),
            zero_parallax_strength=zero_parallax_strength.get(),
            enable_edge_masking=enable_edge_masking.get(),
            enable_feathering=enable_feathering.get(),
            skip_blank_frames=skip_blank_frames.get(),
            dof_strength=dof_strength.get(),
            original_video_width=width,
            original_video_height=height,
            convergence_strength=convergence_strength.get(),
            enable_dynamic_convergence=enable_dynamic_convergence.get(),
            ipd_factor=ipd_value,
            depth_pop_gamma=depth_pop_gamma.get(),
            depth_pop_mid=depth_pop_mid.get(),
            depth_stretch_lo=depth_stretch_lo.get(),
            depth_stretch_hi=depth_stretch_hi.get(),
            fg_pop_multiplier=fg_pop_multiplier.get(),
            bg_push_multiplier=bg_push_multiplier.get(),
            subject_lock_strength=subject_lock_strength.get(),
            color_saturation=(color_saturation.get() if hasattr(color_saturation, 'get') else color_saturation),
            color_contrast=(color_contrast.get() if hasattr(color_contrast, 'get') else color_contrast),
            color_brightness=(color_brightness.get() if hasattr(color_brightness, 'get') else color_brightness),
            start_s=start_s,
            end_s=end_s,
            eye_mode=eye_mode,
            vr180_equi_w=equi_w,
            vr180_equi_h=equi_h,
            vr180_flat_w=flat_w,
            vr180_flat_h=flat_h,
            vr180_hfov_deg=hfov,
            disable_shift_ema=disable_shift_ema,
        )

    if not final_render_path:
        return output_path  # safety fallback

    # 🔊 Inject original audio if toggle enabled
    if keep_original_audio:
        print("🔊 Merging original audio into final render…")

        base, ext = os.path.splitext(final_render_path)
        merged_output = base + "_audio" + ext  # keep .mkv/.mp4/.mov etc

        final_render_path = merge_audio_from_source(final_render_path, input_path, merged_output, start_s=start_s)
        print("🎧 Audio merge done!")

    return final_render_path



def render_with_ffmpeg(
    frame_generator: Iterable[np.ndarray],
    output_path: str,
    width: int,
    height: int,
    fps: float,
    codec_name: str = "libx264",
    crf: int = 23,
    nvenc_cq: int = 23,
    preset: str = "slow",
) -> None:
    """
    Stream raw BGR frames to FFmpeg via stdin and encode to a video file.

    Parameters
    ----------
    frame_generator : iterable of np.ndarray
        Yields frames shaped (H, W, 3) in BGR24.
    output_path : str
        Destination video file path (e.g., "out.mp4").
    width, height : int
        Expected frame size. Mismatched frames are skipped (not resized).
    fps : float
        Output frame rate.
    codec_name : str
        FFmpeg encoder (e.g., "libx264", "libx265", "h264_nvenc", "hevc_nvenc").
    crf : int
        CRF value for libx264/libx265.
    nvenc_cq : int
        CQ value for NVENC encoders (used with -rc vbr and -b:v 0).
    preset : str
        Encoder preset (e.g., "slow", "medium", "p5" for NVENC).
    """

    # Base command (reading raw BGR24 frames from stdin)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-an",
        "-c:v", codec_name,
        "-pix_fmt", "yuv420p",   # SDR default; change upstream if you do HDR
        output_path
    ]


    # Codec-dependent quality flags
    if codec_name.startswith("libx"):
        ix = ffmpeg_cmd.index("-pix_fmt")
        ffmpeg_cmd[ix:ix] = ["-preset", preset, "-crf", str(crf)]
    elif "nvenc" in codec_name:
        ix = ffmpeg_cmd.index("-pix_fmt")
        ffmpeg_cmd[ix:ix] = ["-preset", preset, "-cq", str(nvenc_cq)]
        ffmpeg_cmd += ["-b:v", "0"]  # constant-quality style for NVENC
    elif codec_name in {"h264_amf", "hevc_amf", "av1_amf"}:
        ix = ffmpeg_cmd.index("-pix_fmt")
        ffmpeg_cmd[ix:ix] = ["-quality", "quality", "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf)]

    print(f"🚀 Launching FFmpeg render: {codec_name} | CRF: {crf} | NVENC CQ: {nvenc_cq} ➜ {output_path}")

    try:
        with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE) as proc:
            assert proc.stdin is not None, "FFmpeg stdin not available."

            for idx, frame in enumerate(frame_generator):
                if frame is None:
                    print(f"⚠️ Frame {idx} is None — skipping.")
                    continue

                h, w = frame.shape[:2]
                if (w != width) or (h != height):
                    print(f"⚠️ Frame {idx} has incorrect shape: {w}x{h} (expected {width}x{height}) — skipping.")
                    continue

                # Ensure uint8 BGR
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8, copy=False)

                proc.stdin.write(frame.tobytes())

            # Close stdin so ffmpeg can finalize/flush
            proc.stdin.close()
            proc.wait()

            if proc.returncode == 0:
                print("✅ FFmpeg render complete.")
            else:
                print(f"⚠️ FFmpeg exited with code {proc.returncode}. Check logs above.")

    except Exception as e:
        print(f"❌ FFmpeg render failed: {e}")
