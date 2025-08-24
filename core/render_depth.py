import os
import re
import sys
import time
import uuid
import json
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import gc

import traceback, datetime

import numpy as np
import torch
import cv2
import onnxruntime as ort
import matplotlib.cm as cm
from PIL import Image, ImageTk, ImageOps

from transformers import AutoProcessor, AutoModelForDepthEstimation, pipeline
from diffusers import EulerDiscreteScheduler, AutoencoderKL

from safetensors.torch import load_file

# Custom modules
from core.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from diffusers.configuration_utils import ConfigMixin
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from core.depthcrafter_adapter import load_depthcrafter_adapter, run_depthcrafter_inference


global pipe
pipe = None
pipe_type = None 
suspend_flag = threading.Event()
cancel_flag = threading.Event()
cancel_requested = threading.Event()
global_session_start_time = None
current_warmup_session = {"id": None}
torch.set_grad_enabled(False)

        
# --- tiling config ---
USE_TILED_DEPTH = False   # <- flip off to revert to old behavior
TILE_SIZE       = 512
TILE_PAD        = 32
TILE_DEBUG      = False   # set True to print tile debug info

assert TILE_SIZE > 2*TILE_PAD, "TILE_SIZE must be larger than 2*TILE_PAD"

def _is_depthcrafter():
    return (globals().get("pipe_type", None) == "depthcrafter") or getattr(globals().get("pipe", None), "_is_depthcrafter", False)

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
    weight_core = _hann2d(core, core)

    for y0 in range(0, tgtH, step):
        for x0 in range(0, tgtW, step):
            y1 = min(y0 + tile, tgtH); x1 = min(x0 + tile, tgtW)
            yp0 = max(0, y0 - pad);    xp0 = max(0, x0 - pad)
            yp1 = min(tgtH, y1 + pad); xp1 = min(tgtW, x1 + pad)

            crop_rgb = img[yp0:yp1, xp0:xp1, :]  # RGB
            
            # ✅ ViT/DINOv2 safe crop size (W,H) = multiples of 14
            cw, ch = crop_rgb.shape[1], crop_rgb.shape[0]
            def r14(x): return ((int(x) + 13) // 14) * 14
            cws, chs = r14(cw), r14(ch)
            if (cw, ch) != (cws, chs):
                crop_rgb = cv2.resize(crop_rgb, (cws, chs), interpolation=cv2.INTER_CUBIC)

            
            pil      = Image.fromarray(crop_rgb)

            # Call the model with the crop size (W,H)
            try:
                # pass the ViT-safe crop size to the model
                res = model_call([pil], inference_size=(cws, chs))
            except TypeError:
                res = model_call(pil, inference_size=(cws, chs))
            pred_np = _ensure_depth_np(res)

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

            if TILE_DEBUG:
                print(f"[tile] ({y0}:{y1},{x0}:{x1}) crop={crop_rgb.shape[:2]} center={center.shape} "
                      f"acc={(out_accum[y0:y1, x0:x1].shape)}")

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

def _pred_to_np(pred):
    if isinstance(pred, torch.Tensor):
        return pred.detach().cpu().float().numpy()
    return np.asarray(pred, dtype=np.float32)

def _run_pipe_or_tile(images_pil, inference_size):
    """
    Returns list[{'predicted_depth': ndarray or tensor}]
    """
    global pipe, pipe_type

    # --- ONNX special handling: force the warm-up proven size ---
    if pipe_type == "onnx":
        # 1) Prefer the size that succeeded during warm-up
        good = getattr(pipe, "_good_size", None)

        # 2) Pick a base size: good size > user-provided > safe default
        if good is not None:
            inf_w, inf_h = good
        elif inference_size is not None:
            inf_w, inf_h = int(inference_size[0]), int(inference_size[1])
        else:
            # conservative safe default for VDA-style exports
            inf_w, inf_h = 512, 288

        # 3) Snap to multiples of 32 (keeps all intermediate shapes aligned)
        inf_w, inf_h = snap_for_vda(inf_w, inf_h, base=32)
        inference_size = (inf_w, inf_h)

        # 4) Run the ONNX callable
        try:
            res = pipe(images_pil, inference_size=inference_size)
            if isinstance(res, list):
                return res
            elif isinstance(res, dict):
                return [res]
            else:
                return [{"predicted_depth": res}]
        except TypeError:
            outs = []
            for img in images_pil:
                r = pipe(img, inference_size=inference_size)
                if isinstance(r, list):
                    r = r[0]
                outs.append(r if isinstance(r, dict) else {"predicted_depth": r})
            return outs
    if USE_TILED_DEPTH:
        preds = []
        for img in images_pil:
            rgb = np.array(img.convert("RGB"))
            dep = infer_depth_tile(pipe, rgb, inference_size, tile=TILE_SIZE, pad=TILE_PAD)
            dep_min, dep_max = float(np.nanmin(dep)), float(np.nanmax(dep))
            print(f"[tile] range min={dep_min:.6f} max={dep_max:.6f}")
            preds.append({"predicted_depth": dep})
        return preds

    # Non-tiled
    try:
        res = pipe(images_pil, inference_size=inference_size)
        if isinstance(res, list):
            return res
        elif isinstance(res, dict):
            return [res]
        else:
            return [{"predicted_depth": res}]
    except TypeError:
        outs = []
        for img in images_pil:
            r = pipe(img, inference_size=inference_size)
            if isinstance(r, list):
                r = r[0]
            outs.append(r if isinstance(r, dict) else {"predicted_depth": r})
        return outs


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
    y_thresh=16,
    var_thresh=3.0,
    sat_thresh=6.0,
    max_scan_frac=0.25,
    min_band_frac=0.06,
    edge_max=0.04  # rows with more than ~4% edges are not “bars”
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


# ---------- Cropping (unchanged, with guard) ----------
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
    if getattr(sys, 'frozen', False):
        # PyInstaller bundle: use dir of the .exe
        base_path = os.path.dirname(sys.executable)
    else:
        # Source run: use the parent of the current file (core/ -> project root)
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    return os.path.join(base_path, "weights")

local_model_dir = get_weights_dir()
os.makedirs(local_model_dir, exist_ok=True)

# === Suppress Hugging Face symlink warnings (esp. on Windows) ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

INFERENCE_RESOLUTIONS = {
    "Original": None,

    # General square resolutions
    "256x256": (256, 256),
    "384x384": (384, 384),
    "448x448": (448, 448),
    "512x512 (VDA)": (512, 512),
    "576x576": (576, 576),
    "640x640": (640, 640),
    "704x704": (704, 704),
    "768x768": (768, 768),
    "832x832": (832, 832),
    "896x896": (896, 896),
    "960x960": (960, 960),
    "1024x1024": (1024, 1024),

    # ViT/DINOV2-safe resolutions (multiples of 14)
    "512x288":  (512, 288),
    "640x352":  (640, 352),
    "768x432":  (768, 432),
    "896x512":  (896, 512),
    "1024x576": (1024, 576),
    "1152x640": (1152, 640),
    "1280x720": (1280, 720),   # OK, both /32
    "1344x768": (1344, 768),
    "1536x864": (1536, 864),
    "1600x896": (1600, 896),
    "1792x1008":(1792, 1008),
    "1920x1088":(1920, 1088),  # NOTE: 1088 instead of 1080
    # Squares / general
    "256x256":  (256, 256),
    "384x384":  (384, 384),
    "512x512":  (512, 512),
    "640x640":  (640, 640),
    "768x768":  (768, 768),
    "896x896":  (896, 896),
    "1024x1024":(1024, 1024),

    # Widescreen & cinematic
    "512x256 (DC-Fastest)": (512, 256),
    "704x384 (DC-Balanced)": (704, 384),
    "910x518 (Depth Anything)": (910, 518),
    "960x540 (DC-Good Quality)": (960, 540),
    "1024x576 (DC-Max Quality)": (1024, 576),

    # Portrait / vertical or special use
    "912x912": (912, 912),
    "920x1080": (920, 1080),  # vertical

    # Experimental 16:9 upscales
    "1280x720 (720p HD)": (1280, 720),
    "1920x1080 (1080p HD)": (1920, 1080),
}

def load_supported_models():
    models = {
        "  -- Select Model -- ": "  -- Select Model -- ",
        "Marigold Depth (Diffusers)": "diffusers:prs-eth/marigold-depth-v1-1",
        "DepthCrafter (Custom)": "depthcrafter:weights/DepthCrafter",
        "Distil-Any-Depth-Large": "xingyang1/Distill-Any-Depth-Large-hf",
        "Distil-Any-Depth-Small": "xingyang1/Distill-Any-Depth-Small-hf",
        "keetrap-Distil-Any-Depth-Large": "keetrap/Distil-Any-Depth-Large-hf",
        "keetrap-Distil-Any-Depth-Small": "keetrap/Distill-Any-Depth-Small-hf",
        "Depth Anything V2 Large": "depth-anything/Depth-Anything-V2-Large-hf",
        "Depth Anything V2 Base": "depth-anything/Depth-Anything-V2-Base-hf",
        "Depth Anything V2 Small": "depth-anything/Depth-Anything-V2-Small-hf",
        "Depth Anything V1 Large": "LiheYoung/depth-anything-large-hf",
        "Depth Anything V1 Base": "LiheYoung/depth-anything-base-hf",
        "Depth Anything V1 Small": "LiheYoung/depth-anything-small-hf",
        "vitl14": "LiheYoung/depth_anything_vitl14",
        "V2-Metric-Indoor-Large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "V2-Metric-Outdoor-Large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "DepthPro": "apple/DepthPro-hf",
        "marigold-depth-v1-0": "prs-eth/marigold-depth-v1-0",
        "ZoeDepth": "Intel/zoedepth-nyu-kitti",
        "MiDaS 3.0": "Intel/dpt-hybrid-midas",
        "DPT-Large": "Intel/dpt-large",
        "Manojb - DPT-Large": "Manojb/dpt-large",
        "dpt-beit-large-512": "Intel/dpt-beit-large-512",
        "Midas-V2": "qualcomm/Midas-V2",
        "Video Depth Anything (ONNX)": "videodepthanything:VideoDepthAnything",
        #    ^ label seen in the dropdown          ^ relative to local_model_dir
    }

    # ✅ auto-add local folders as “[Local] {folder}”
    for folder in os.listdir(local_model_dir):
        folder_path = os.path.join(local_model_dir, folder)
        if os.path.isdir(folder_path):
            if (os.path.exists(os.path.join(folder_path, "config.json")) or
                os.path.exists(os.path.join(folder_path, "model.onnx"))):
                models[f"[Local] {folder}"] = folder_path

    return models

supported_models = load_supported_models()

def ensure_model_downloaded(checkpoint):
    """
    Handles both Hugging Face checkpoints and local ONNX directories.
    """
    # --- Custom alias: videodepthanything:<relative-or-abs-path> ---
    if isinstance(checkpoint, str) and checkpoint.startswith("videodepthanything:"):
        rel = checkpoint.split(":", 1)[1].strip()
        model_dir = rel if os.path.isabs(rel) else os.path.join(local_model_dir, rel)
        provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        print(f"🧠 Resolving VDA ONNX at: {model_dir} (provider={provider})")
        return load_onnx_model(model_dir, device=provider)

    # (optional) generic onnx: prefix
    if isinstance(checkpoint, str) and checkpoint.startswith("onnx:"):
        rel = checkpoint.split(":", 1)[1].strip()
        model_dir = rel if os.path.isabs(rel) else os.path.join(local_model_dir, rel)
        provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        return load_onnx_model(model_dir, device=provider)

    
    if os.path.isdir(checkpoint):
        # Local ONNX model detection
        if os.path.exists(os.path.join(checkpoint, "model.onnx")):
            provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            print(f"🧠 Detected ONNX model in {checkpoint} (provider={provider})")
            return load_onnx_model(checkpoint, device=provider)


        # Local Hugging Face model
        try:
            model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
            processor = AutoProcessor.from_pretrained(checkpoint)
            print(f"📂 Loaded local Hugging Face model from {checkpoint}")
            return model, processor
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            return None, None
    
    # === Diffusion Model Check ===
    if checkpoint.startswith("diffusers:"):
        from diffusers import MarigoldDepthPipeline
        model_id = checkpoint.replace("diffusers:", "")
        try:
            pipe = MarigoldDepthPipeline.from_pretrained(
                model_id,
                variant="fp16",
                torch_dtype=torch.float16,
                cache_dir=local_model_dir 
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            def diffusion_pipe(images, inference_size=None):
                if not isinstance(images, list):
                    images = [images]
                results = []
                for img in images:
                    if inference_size:
                        img = img.resize(inference_size, Image.BICUBIC)
                    result = pipe(img, num_inference_steps=4, ensemble_size=5)
                    results.append({"predicted_depth": result.prediction[0]})

                return results

            print(f"🌀 Diffusion depth model loaded: {model_id}")
            diffusion_pipe._is_marigold = True  # ✅ Add this!
            diffusion_pipe.image_processor = pipe.image_processor 
            return diffusion_pipe, {"is_diffusion": True}

        except Exception as e:
            print(f"❌ Failed to load diffusion depth model: {e}")
            return None, None
        
    if checkpoint.startswith("depthcrafter:"):
        model_id = checkpoint.replace("depthcrafter:", "")
        if not os.path.isdir(model_id):
            print(f"❌ DepthCrafter folder not found: {model_id}")
            return None, None
        try:
            dc_pipe, dc_meta = load_depthcrafter_adapter(model_id)
            if not isinstance(dc_meta, dict):
                dc_meta = {}
            setattr(dc_pipe, "_is_depthcrafter", True)
            dc_meta.update({"is_depthcrafter": True, "is_diffusion": True})
            print(f"🧩 DepthCrafter adapter loaded: {model_id}")
            return dc_pipe, dc_meta
        except Exception as e:
            print(f"❌ Failed to load DepthCrafter adapter: {e}")
            return None, None



    
    # Hugging Face online model
    safe_folder_name = checkpoint.replace("/", "_")
    local_path = os.path.join(local_model_dir, safe_folder_name)
    try:
        model = AutoModelForDepthEstimation.from_pretrained(checkpoint, cache_dir=local_path)
        processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=local_path)
        print(f"⬇️ Downloaded model from Hugging Face: {checkpoint}")
        return model, processor
    except Exception as e:
        print(f"❌ Failed to load Hugging Face model: {e}")
        return None, None


def load_onnx_model(model_dir, device="CUDAExecutionProvider"):
    import onnxruntime as ort
    model_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_path):
        print(f"❌ ONNX model not found: {model_path}")
        return None, None

    print(f"🧠 Loading ONNX model from: {model_path}")

    # Try with ALL graph optimizations disabled (prevents bad rename/fusion issues)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # Optional: make behavior more deterministic
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    # Build provider list
    providers = []
    if device == "CUDAExecutionProvider" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    # Attempt 1: disabled opts + chosen providers
    try:
        session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    except Exception as e1:
        print(f"⚠️ ORT init failed (opts disabled, {providers}): {e1}")
        # Attempt 2: CPU only, still with opts disabled
        try:
            session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        except Exception as e2:
            print(f"💥 ORT CPU fallback failed (opts disabled): {e2}")
            # As a last resort, try default options (may still fail on bad exports)
            try:
                session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            except Exception as e3:
                print(f"💥 ORT final fallback failed: {e3}")
                return None, None

    # Introspect IO (handy if the export expects multiple inputs)
    try:
        in_names = [i.name for i in session.get_inputs()]
        out_names = [o.name for o in session.get_outputs()]
        print(f"🔎 Inputs={in_names} | Outputs={out_names}")
    except Exception:
        pass

    # Pull basic shape info from the FIRST input (your current runner assumes 1 input)
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    input_name = input_info.name
    output_name = output_info.name
    input_shape = input_info.shape
    input_rank = len(input_shape)

    # ImageNet normalization for VDA / DINOv2
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Detect fixed T for video models (e.g., [1, 32, 3, H, W])
    fixed_T = None
    try:
        if input_rank >= 2 and isinstance(input_shape[1], int):
            fixed_T = int(input_shape[1])
    except Exception:
        fixed_T = None

    print(f"🔎 Input shape: {input_shape} | Rank: {input_rank} | fixed_T={fixed_T}")

    def _prep_images(images, inference_size):
        arrs = []
        for img in images:
            if inference_size:
                img = img.resize(inference_size, Image.BICUBIC)
            x = np.asarray(img, dtype=np.float32) / 255.0  # HWC
            x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()  # CHW
            arrs.append(x)
        return arrs

    def run_onnx(images, inference_size=None):
        if inference_size is None:
            raise ValueError("❌ Must provide inference_size for ONNX.")

        # 🔧 VDA stride safety: snap both dims to /32 and even
        W, H = int(inference_size[0]), int(inference_size[1])
        W, H = snap_for_vda(W, H, base=32)
        inference_size = (W, H)

        img_batch = _prep_images(images, inference_size)

        if input_rank == 5:
            T = len(img_batch)
            if fixed_T is not None and T != fixed_T:
                if T < fixed_T:
                    img_batch += [img_batch[-1]] * (fixed_T - T)
                else:
                    img_batch = img_batch[:fixed_T]
                T = fixed_T
            input_tensor = np.stack(img_batch, axis=0)[None, ...]  # [1, T, 3, H, W]
        elif input_rank == 4:
            input_tensor = np.stack(img_batch, axis=0)             # [B, 3, H, W]
        else:
            raise ValueError(f"❌ Unsupported ONNX input rank: {input_rank}")

        output = session.run([output_name], {input_name: input_tensor})[0]
        if input_rank == 5 and output.ndim == 4 and output.shape[0] == 1:
            output = output.squeeze(0)

        if input_rank == 5:
            return [{"predicted_depth": torch.tensor(output[t])} for t in range(output.shape[0])]
        else:
            return [{"predicted_depth": torch.tensor(output[b])} for b in range(output.shape[0])]


    run_onnx._is_marigold = False
    return run_onnx, {
        "input_rank": input_rank,
        "fixed_T": fixed_T,
        "session": session,
        "provider": providers[0] if providers else "CPUExecutionProvider",
        "is_onnx": True
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


def update_pipeline(selected_model_var, status_label_widget, inference_res_var, offload_mode_dropdown, *args):
    global pipe
    
    selected_checkpoint = selected_model_var.get()
    checkpoint = supported_models.get(selected_checkpoint, None)


    def warmup_thread():
        try:
            model_callable, processor_or_metadata = ensure_model_downloaded(checkpoint)
            if not model_callable:
                status_label_widget.after(0, lambda: stop_spinner(
                    status_label_widget, f"❌ Failed to load model: {selected_checkpoint}"))
                return

            device = 0 if torch.cuda.is_available() else -1
            is_onnx = isinstance(processor_or_metadata, dict) and processor_or_metadata.get("is_onnx", False)
            is_diffusion = isinstance(processor_or_metadata, dict) and processor_or_metadata.get("is_diffusion", False)

            global pipe, pipe_type

            if is_onnx:
                pipe = model_callable
                pipe_type = "onnx"

                status_label_widget.after(0, lambda: start_spinner(status_label_widget, "🔄 Warming up ONNX model..."))

                # --- Robust ONNX warm-up (handles stride quirks & fixed T) ---
                try:
                    input_rank = processor_or_metadata.get("input_rank", 4)
                    fixed_T    = processor_or_metadata.get("fixed_T", None)
                    warmup_T   = int(fixed_T) if fixed_T is not None else 8

                    # Pull user pref (if any), then snap for VDA (/32)
                    user_res = parse_inference_resolution(inference_res_var.get(), fallback=(512, 288))
                    if user_res is None:
                        user_res = (512, 288)
                    uW, uH = snap_for_vda(user_res[0], user_res[1], base=32)

                    # Known good, /32-aligned options (stop at first that runs)
                    candidates = [
                        (uW, uH),            # user's snapped choice first
                        (512, 288),
                        (640, 360),
                        (768, 432),
                        (896, 504),
                        (960, 544),
                        (1024, 576),
                        (1152, 648),
                        (1280, 720),
                        (1536, 864),
                        (512, 512),          # square options if the net allows
                        (640, 640),
                        (768, 768),
                    ]

                    seen, sizes = set(), []
                    for s in candidates:
                        s = snap_for_vda(s[0], s[1], base=32)
                        if s not in seen:
                            sizes.append(s); seen.add(s)

                    last_err = None
                    warmed = False
                    for (W, H) in sizes:
                        try:
                            if input_rank == 5:
                                dummy_batch = [Image.new("RGB", (W, H), (127, 127, 127)) for _ in range(warmup_T)]
                            else:
                                dummy_batch = [Image.new("RGB", (W, H), (127, 127, 127))]
                            _ = pipe(dummy_batch, inference_size=(W, H))
                            print(f"🔥 ONNX model warmed up. Size={(W, H)}, T={warmup_T if input_rank==5 else 1}")
                            warmed = True
                            # (Optional) remember a good size for runtime defaults:
                            processor_or_metadata["good_size"] = (W, H)
                            break
                        except Exception as err:
                            last_err = err
                            print(f"⚠️ Warm-up trial failed at {(W, H)}: {err}")

                    if not warmed:
                        raise RuntimeError(f"ONNX warm-up failed for all tried sizes {sizes}. Last error: {last_err}")

                except Exception as e:
                    print(f"⚠️ ONNX warm-up failed: {e}")



                dev_str = "CUDA" if str(device).lower() in ("cuda", "0", "gpu") else "CPU"
                status_label_widget.after(0, lambda: stop_spinner(
                    status_label_widget, f"✅ ONNX model loaded: {selected_checkpoint} (on {dev_str})"))

            elif is_diffusion:
                is_dc = (isinstance(processor_or_metadata, dict) and processor_or_metadata.get("is_depthcrafter", False)) \
                        or getattr(model_callable, "_is_depthcrafter", False)

                if is_dc:
                    pipe = model_callable
                    pipe_type = "depthcrafter"

                    status_label_widget.after(0, lambda: start_spinner(status_label_widget, "🔄 Getting DepthCrafter ready..."))
                    try:
                        # No real inference here — DC expects sequences; just verify object shape minimally
                        assert callable(pipe), "DepthCrafter pipe is not callable"
                        print("🔥 DepthCrafter ready (will run during video processing)")
                        status_label_widget.after(0, lambda: stop_spinner(
                            status_label_widget,
                            f"✅ DepthCrafter loaded: {selected_checkpoint} (device: {'CUDA' if device == 0 else 'CPU'})"
                        ))
                    except Exception as e:
                        msg = f"❌ DepthCrafter init failed: {e}"
                        print(msg)
                        status_label_widget.after(0, lambda: stop_spinner(status_label_widget, msg))
                    return

                # Other diffusion models (e.g., Marigold)
                pipe = model_callable
                pipe_type = "diffusion"
                status_label_widget.after(0, lambda: start_spinner(status_label_widget, "🔄 Warming up diffusion model..."))
                try:
                    dummy = Image.new("RGB", (518, 518), (127, 127, 127))
                    _ = pipe(dummy)
                    print("🔥 Diffusion model warmed up with dummy image")
                except Exception as e:
                    print(f" Diffusion warm-up skipped: {e}")
                status_label_widget.after(0, lambda: stop_spinner(
                    status_label_widget,
                    f"✅ Diffusion model loaded: {selected_checkpoint} (Running on {'CUDA' if device == 0 else 'CPU'})"
                ))


            else:
                processor = processor_or_metadata
                raw_pipe = pipeline(
                    "depth-estimation",
                    model=model_callable,
                    image_processor=processor,
                    device=device
                )

                def hf_batch_safe_pipe(images, inference_size=None):
                    if inference_size:
                        images = [img.resize(inference_size, Image.BICUBIC) for img in images]
                    return raw_pipe(images) if isinstance(images, list) else [raw_pipe(images)]

                pipe = hf_batch_safe_pipe
                pipe_type = "hf"
                status_label_widget.after(0, lambda: start_spinner(
                    status_label_widget, "🔄 Warming up Hugging Face model..."))

                try:
                    dummy = Image.new("RGB", (384, 384), (127, 127, 127))
                    _ = pipe([dummy])
                    print("🔥 Hugging Face pipeline warmed up with dummy frame")
                except Exception as e:
                    print(f"⚠️ Hugging Face warm-up failed: {e}")

                status_label_widget.after(0, lambda: stop_spinner(
                    status_label_widget, f"✅ HF model loaded: {selected_checkpoint} (Running on {'CUDA' if device == 0 else 'CPU'})"))

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
    return INFERENCE_RESOLUTIONS.get(res_string.strip(), fallback)


def choose_output_directory(output_label_widget, output_dir_var):
    selected_directory = filedialog.askdirectory()
    if selected_directory:
        output_dir_var.set(selected_directory)
        output_label_widget.config(text=f"📁 {selected_directory}")

def get_dynamic_batch_size(base=4, scale_factor=1.0, max_limit=32, reserve_vram_gb=1.0):
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024 ** 3)
        usable_vram = max(0, total_vram - reserve_vram_gb)
        estimated_batch = int(base * usable_vram * scale_factor)
        return min(estimated_batch, max_limit)
    return base

def process_image_folder(batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, invert_var, root):
    folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
    if not folder_path:
        cancel_requested.clear()  # ✅ Reset before starting
        status_label.config(text="⚠️ No folder selected.")
        return

    threading.Thread(
        target=process_images_in_folder,
        args=(folder_path, batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, root, invert_var),
        daemon=True,
    ).start()


def process_images_in_folder(folder_path, batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, root, invert_var):
    output_dir = output_dir_var.get().strip()
    global pipe, pipe_type
    global global_session_start_time
    global_session_start_time = time.time()

    if not output_dir:
        messagebox.showwarning("Missing Output Folder", "⚠️ Please select an output directory before processing.")
        root.after(10, lambda: status_label.config(text="❌ Output directory not selected."))
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            messagebox.showerror("Folder Creation Failed", f"❌ Could not create output directory:\n{e}")
            root.after(10, lambda: status_label.config(text="❌ Failed to create output directory."))
            return

    inference_size = parse_inference_resolution(inference_res_var.get())

    try:
        user_value = batch_size_widget.get().strip()
        batch_size = int(user_value) if user_value else get_dynamic_batch_size()
        if batch_size <= 0:
            raise ValueError
    except Exception:
        batch_size = get_dynamic_batch_size()
        status_label.config(text=f"⚠️ Invalid batch size. Using dynamic batch size: {batch_size}")

    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    ]

    if not image_files:
        root.after(10, lambda: status_label.config(text="⚠️ No image files found."))
        return

    total_images = len(image_files)
    root.after(10, lambda: status_label.config(text=f"📂 Processing {total_images} images..."))
    root.after(10, lambda: progress_bar.config(maximum=total_images, value=0))

    start_time = time.time()

    for i in range(0, total_images, batch_size):
        if cancel_requested.is_set():
            root.after(10, lambda: status_label.config(text="❌ Cancelled by user."))
            return

        batch_files = image_files[i:i + batch_size]
        images = []
        original_sizes = []

        for file in batch_files:
            img = Image.open(file).convert("RGB")
            original_sizes.append(img.size)
            images.append(img)  # let the tiler/pipe handle resizing

        print(f"🚀 Running batch of {len(images)} images at {inference_size}")
        predictions = _run_pipe_or_tile(images, inference_size)

        for j, prediction in enumerate(predictions):
            if cancel_requested.is_set():
                root.after(10, lambda: status_label.config(text="❌ Cancelled during batch."))
                return

            file_path = batch_files[j]
            orig_w, orig_h = original_sizes[j]

            try:
                depth_pred = prediction["predicted_depth"]

                if USE_TILED_DEPTH:
                    # tiler returns float32 ndarray 
                    depth_np = _normalize_to_u8(depth_pred, (orig_w, orig_h), invert=invert_var.get())
                    depth_image = Image.fromarray(depth_np)
                else:
                    # original behavior
                    if getattr(pipe, "_is_marigold", False):
                        depth_image = pipe.image_processor.export_depth_to_16bit_png(depth_pred)[0]
                        depth_image = depth_image.resize((orig_w, orig_h), Image.BICUBIC)
                        if invert_var.get():
                            arr = np.array(depth_image, dtype=np.uint16)
                            depth_image = Image.fromarray(65535 - arr, mode="I;16")
                    else:
                        depth_norm = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min() + 1e-6)
                        depth_np = (depth_norm.squeeze() * 255).astype(np.uint8) if isinstance(depth_pred, np.ndarray) \
                                   else (depth_norm.squeeze().cpu().numpy() * 255).astype(np.uint8)
                        if invert_var.get(): depth_np = 255 - depth_np
                        depth_np = cv2.resize(depth_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                        depth_image = Image.fromarray(depth_np)

                image_name = os.path.splitext(os.path.basename(file_path))[0]
                output_filename = f"{image_name}_depth.png"
                file_save_path = os.path.join(output_dir, output_filename)
                depth_image.save(file_save_path)
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                continue


            elapsed_time = time.time() - start_time
            fps = (i + j + 1) / elapsed_time if elapsed_time > 0 else 0
            eta = (total_images - (i + j + 1)) / fps if fps > 0 else 0

            status_label.after(10, lambda i=i + j + 1, fps=fps, eta=eta: update_progress(i, total_images, fps, eta, progress_bar, status_label))

    root.after(10, lambda: status_label.config(text="✅ All images processed successfully!"))
    root.after(10, lambda: progress_bar.config(value=progress_bar["maximum"]))


def update_progress(processed, total, fps, eta, progress_bar, status_label):
    progress_bar.config(value=processed)

    # Format FPS and ETA
    fps_text = f"{fps:.2f} FPS"
    eta_text = f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}" if eta > 0 else "ETA: --:--:--"
    progress_text = f"📸 Processed: {processed}/{total} | {fps_text} | {eta_text}"

    status_label.config(text=progress_text)


def process_image(file_path, colormap_var, invert_var, output_dir_var, inference_res_var, input_label, output_label, status_label, progress_bar, folder=False):
    global pipe, pipe_type
    image = Image.open(file_path).convert("RGB")
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

            depth_np = _normalize_to_u8(depth_pred, image.size, invert=invert_var.get())
            if colormap_name == "default":
                depth_image = Image.fromarray(depth_np)
            else:
                try:
                    cmap = cm.get_cmap(colormap_name)
                    colored = (cmap(depth_np.astype(np.float32)/255.0)[:, :, :3] * 255).astype(np.uint8)
                    depth_image = Image.fromarray(colored)
                except Exception:
                    depth_image = Image.fromarray(depth_np)

        else:
            # === Marigold special path ===
            if getattr(pipe, "_is_marigold", False):
                if colormap_name == "default":
                    depth_image = pipe.image_processor.export_depth_to_16bit_png(depth_pred)[0]
                else:
                    try:
                        depth_image = pipe.image_processor.visualize_depth(depth_pred, color_map=colormap_name)[0]
                    except Exception as e:
                        print(f"⚠️ Failed to apply colormap '{colormap_name}', using default. {e}")
                        depth_image = pipe.image_processor.visualize_depth(depth_pred)[0]

                depth_image = depth_image.resize(original_size, Image.BICUBIC)
                if invert_var.get():
                    arr = np.array(depth_image)
                    if depth_image.mode == "I;16":
                        depth_image = Image.fromarray(65535 - arr, mode="I;16")
                    else:
                        depth_image = Image.fromarray(255 - arr.astype(np.uint8))

            # === Other HF/ONNX models ===
            else:
                d = _pred_to_np(depth_pred).squeeze()
                d = (d - d.min()) / (d.max() - d.min() + 1e-6)
                depth_np = (d * 255).astype(np.uint8)
                if invert_var.get(): depth_np = 255 - depth_np
                depth_np = cv2.resize(depth_np, original_size, interpolation=cv2.INTER_CUBIC)

                if colormap_name == "default":
                    depth_image = Image.fromarray(depth_np)
                else:
                    try:
                        cmap = cm.get_cmap(colormap_name)
                        colored = (cmap(depth_np.astype(np.float32)/255.0)[:, :, :3] * 255).astype(np.uint8)
                        depth_image = Image.fromarray(colored)
                    except ValueError:
                        print(f"⚠️ Unknown colormap '{colormap_name}', defaulting to grayscale.")
                        depth_image = Image.fromarray(depth_np)

    except Exception as e:
        print(f"❌ Error extracting depth: {e}")
        return

    output_dir = output_dir_var.get().strip()
    if not output_dir:
        messagebox.showwarning("Missing Output Folder", "⚠️ Please select an output directory before saving.")
        status_label.config(text="❌ Output directory not selected.")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            messagebox.showerror("Folder Creation Failed", f"❌ Could not create output directory:\n{e}")
            status_label.config(text="❌ Failed to create output directory.")
            return

    if not folder:
        image_disp = image.copy()
        image_disp.thumbnail((480, 270))
        input_img_tk = ImageTk.PhotoImage(image_disp)
        input_label.config(image=input_img_tk)
        input_label.image = input_img_tk  # ✅ Prevent garbage collection

        depth_disp = depth_image.copy()

        # ✅ Handle 16-bit grayscale preview safely
        if depth_disp.mode in ("I", "I;16"):
            depth_array = np.array(depth_disp)

            if depth_array.dtype != np.uint16:
                # Normalize to 16-bit range first if needed
                depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
                depth_array = (depth_array * 65535).astype(np.uint16)

            # Downscale to 8-bit for preview display
            preview_array = (depth_array / 256).astype(np.uint8)
            depth_disp = Image.fromarray(preview_array, mode="L").convert("RGB")

        depth_disp.thumbnail((480, 270))
        depth_img_tk = ImageTk.PhotoImage(depth_disp)
        output_label.config(image=depth_img_tk)
        output_label.image = depth_img_tk  # ✅ Prevent garbage collection


    image_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{image_name}_depth.png"
    file_save_path = os.path.join(output_dir, output_filename)
    depth_image.save(file_save_path)

    if not folder:
        cancel_requested.clear()
        status_label.config(text=f"✅ Image saved: {file_save_path}")
        progress_bar.config(value=100)
        progress_bar.stop()  



def open_image(status_label_widget, progress_bar_widget, colormap_var, invert_var, output_dir_var, inference_res_var, input_label_widget, output_label_widget):
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpeg;*.jpg;*.png")]
    )
    if file_path:
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
        status_label_widget.config(text="🔄 Processing image...")
        progress_bar_widget.start(10)
        threading.Thread(
            target=lambda: [
                process_image(
                    file_path,
                    colormap_var,
                    invert_var,
                    output_dir_var,
                    inference_res_var,  # ✅ Pass the selected inference resolution
                    input_label_widget,
                    output_label_widget,
                    status_label_widget,
                    progress_bar_widget,
                ),
                progress_bar_widget.stop()
            ],
            daemon=True,
        ).start()



def process_video_folder(
    folder_path,
    batch_size_widget,
    inference_steps_entry,
    output_dir_var,
    inference_res_var,
    status_label,
    progress_bar,
    cancel_requested,
    invert_var
):
    """Runs folder selection in main thread and launches processing in background."""

    selected_folder = filedialog.askdirectory(title="Select Folder Containing Videos")
    if not selected_folder:
        cancel_requested.clear()
        status_label.config(text="⚠️ No folder selected.")
        return

    # Get batch size on the main thread
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

    # Launch the actual processing in background thread
    threading.Thread(
        target=process_videos_in_folder,
        args=(
            selected_folder,
            batch_size,
            output_dir_var,
            inference_res_var,
            status_label,
            progress_bar,
            cancel_requested,
            invert_var,
            inference_steps_entry,
        ),
        daemon=True
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
    output_dir_var,
    inference_res_var,
    status_label,
    progress_bar,
    cancel_requested,
    invert_var,
    inference_steps_entry,
    save_frames=False,
):

    """Processes all video files in the selected folder in the correct numerical order."""
    video_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        status_label.config(text="⚠️ No video files found in the selected folder.")
        return
    
    video_files.sort(key=natural_sort_key)

    status_label.config(text=f"📂 Processing {len(video_files)} videos...")
    global global_session_start_time
    global_session_start_time = time.time()


    total_frames_all = sum(
        int(cv2.VideoCapture(os.path.join(folder_path, f)).get(cv2.CAP_PROP_FRAME_COUNT))
        for f in video_files
    )
    frames_processed_all = 0

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        processed = process_video2(
            video_path,
            total_frames_all,
            frames_processed_all,
            batch_size,
            output_dir_var,
            inference_res_var,
            status_label,
            progress_bar,
            cancel_requested,
            invert_var,
            inference_steps_entry,
            save_frames
        )

        if cancel_requested.is_set():
            status_label.config(text="🛑 Processing cancelled by user.")
            progress_bar.config(value=0)
            return

        frames_processed_all += processed

    status_label.config(text="✅ All videos processed successfully!")
    progress_bar.config(value=100)

def process_video2(
    file_path,
    total_frames_all,
    frames_processed_all,
    batch_size,
    output_dir_var,
    inference_res_var,
    status_label,
    progress_bar,
    cancel_requested,
    invert_var,
    inference_steps_entry=None,
    window_size=24,
    overlap=25,
    generator=None,
    offload_mode_dropdown=None,
    save_frames=False,
    target_fps=15,
    ignore_letterbox_bars=False,
    
      
):
    global pipe, pipe_type
    global global_session_start_time

    # Detect output directory from UI
    output_dir = output_dir_var.get().strip()
    if not output_dir:
        messagebox.showwarning("Missing Output Folder", "⚠️ Please select an output directory before processing.")
        status_label.config(text="❌ Output directory not selected.")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    input_dir, input_filename = os.path.split(file_path)
    name, _ = os.path.splitext(input_filename)
    output_filename = f"{name}_depth.mkv"
    output_path = os.path.join(output_dir, output_filename)

    # ✅ Special case for Marigold (16-bit export path)
    if hasattr(pipe, "image_processor") and hasattr(pipe.image_processor, "export_depth_to_16bit_png"):
        print("🎥 Marigold model detected — switching to frame-based 16-bit processing.")

        tmp_frame_dir = os.path.join(output_dir, f"{name}_tmp_frames")
        os.makedirs(tmp_frame_dir, exist_ok=True)

        # === 1. Extract raw frames from video
        extract_cmd = [
            "ffmpeg", "-y", "-i", file_path,
            os.path.join(tmp_frame_dir, "frame_%05d.png")
        ]
        subprocess.run(extract_cmd)

        # === 2. Process images into depth maps (same folder)
        dummy_widget = tk.StringVar(value=str(batch_size))
        dummy_output_var = tk.StringVar(value=tmp_frame_dir)
        dummy_root = tk.Tk(); dummy_root.withdraw()

        process_images_in_folder(
            tmp_frame_dir,
            batch_size_widget=dummy_widget,
            output_dir_var=dummy_output_var,
            inference_res_var=inference_res_var,
            status_label=status_label,
            progress_bar=progress_bar,
            root=dummy_root,
            invert_var=invert_var
        )

        # === 3. Encode depth frames to video using FFmpeg
        encode_cmd = [
            "ffmpeg", "-y", "-framerate", "24",  # fallback FPS
            "-i", os.path.join(tmp_frame_dir, "frame_%05d_depth.png"),
            "-c:v", "ffv1", "-pix_fmt", "gray16le",
            output_path
        ]
        subprocess.run(encode_cmd)

        print(f"✅ Marigold 16-bit depth video saved: {output_path}")
        return len(os.listdir(tmp_frame_dir))

    # === Fallback: non-Marigold default behavior ===
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        status_label.config(text=f"❌ Error: Cannot open {file_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ... after you read fps/original_w/h
    tracker = LetterboxTracker(original_height, fps)

    bars_top, bars_bottom, (locked_bars, locked_zero) = tracker.bootstrap(cap)
    print(f"[VD3D] Bootstrap bars: top={bars_top} bottom={bars_bottom} | "
          f"locked_bars={locked_bars} locked_zero={locked_zero}")

    # (Optional) write sidecar using bootstrapped values:
    try:
        sidecar = os.path.splitext(output_path)[0] + ".letterbox.json"
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump({
                "top": int(bars_top), "bottom": int(bars_bottom),
                "orig_w": int(original_width), "orig_h": int(original_height),
                "locked_bars": bool(locked_bars), "locked_zero": bool(locked_zero)
            }, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write letterbox sidecar: {e}")


    print(f"📁 Saving video to: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))
    if not out.isOpened():
        print(f"❌ Failed to open video writer for {output_filename}")
        return 0

    try:
        sidecar = os.path.splitext(output_path)[0] + ".letterbox.json"
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump({
                "top": bars_top, "bottom": bars_bottom,
                "orig_w": original_width, "orig_h": original_height
            }, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write letterbox sidecar: {e}")


    frame_output_dir = os.path.join(output_dir, f"{name}_frames")
    if save_frames:
        os.makedirs(frame_output_dir, exist_ok=True)

    frame_count = 0
    write_index = 0
    frames_batch = []
    total_processed_frames = 0
    
    inference_size = parse_inference_resolution(inference_res_var.get())
    resize_required = inference_size is not None
    
    try:
        inference_steps = int(inference_steps_entry.get().strip())
    except:
        inference_steps = 2       
    try:
        offload_mode = offload_mode_dropdown.get().strip()
    except Exception:
        offload_mode = "sequential" 

        
    window_size = 24
    overlap = 25
    
    if generator is None:
        seed = 42
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)        
    
    global_session_start_time = time.time()
    previous_depth = None
    
    # Define it here, before you start reading frames
    while True:
        if cancel_requested.is_set():
            print("🛑 Cancel requested before frame read.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if ignore_letterbox_bars:
            # Let the tracker decide; it rechecks only at scene cuts (non-black) with hysteresis
            bars_top, bars_bottom = tracker.update(frame, frame_count)
        else:
            bars_top, bars_bottom = 0, 0

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(frame_rgb)

        if resize_required:
            pil_image = pil_image.resize(inference_size, Image.BICUBIC)
        frames_batch.append(pil_image)


        if len(frames_batch) == batch_size or frame_count == total_frames:
            if cancel_requested.is_set():
                print("🛑 Cancel requested before inference.")
                break
            
                
            if _is_depthcrafter():
                # === DepthCrafter sequence inference ===
                if target_fps != -1 and fps > target_fps:
                    stride = max(1, round(fps / target_fps))
                    print(f"🎚️ Frame stride enabled: {stride}")
                    frames_input = frames_batch[::stride]
                else:
                    frames_input = frames_batch

                if len(frames_input) < window_size:
                    window_size = len(frames_input)
                    print(f"⚠️ Adjusting window_size to {window_size} due to short batch")

                print("Running DepthCrafter inference with:")
                print(f"  frames={len(frames_input)}  steps={inference_steps}")
                print(f"  resolution={inference_size}  window_size={window_size} overlap={overlap}")

                predictions = run_depthcrafter_inference(
                    pipe,
                    frames_input,
                    inference_size=inference_size,
                    steps=inference_steps,
                    window_size=window_size,
                    overlap=overlap,
                    offload_mode=offload_mode
                )
                if predictions is None or len(predictions) == 0:
                    print("❌ Inference failed. No depth frames collected.")
                    return frame_count

                if cancel_requested.is_set():
                    print("🛑 Cancelled before saving.")
                    return frame_count

                name, _ = os.path.splitext(os.path.basename(file_path))
                save_depthcrafter_outputs(
                    predictions,
                    os.path.join(output_dir, name),
                    target_fps if target_fps > 0 else int(fps)
                )

                # progress UI
                for i in range(predictions.shape[0]):
                    progress = int(((frames_processed_all + total_processed_frames + i + 1) / total_frames_all) * 100)
                    progress_bar["value"] = progress
                    progress_bar.update_idletasks()

                    elapsed = time.time() - global_session_start_time
                    avg_fps = (frames_processed_all + total_processed_frames + i + 1) / max(elapsed, 1e-6)
                    eta = (total_frames_all - (frames_processed_all + total_processed_frames + i + 1)) / max(avg_fps, 1e-6)

                    status_label.config(
                        text=f"🎬 {frames_processed_all + total_processed_frames + i + 1}/{total_frames_all} frames | "
                             f"FPS: {avg_fps:.2f} | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
                             f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))} | Processing: {name}"
                    )
                    status_label.update_idletasks()

                total_processed_frames += predictions.shape[0]
                frames_batch.clear()
                continue


            else:
                predictions = _run_pipe_or_tile(frames_batch, inference_size)


            assert isinstance(predictions, list), "Expected list of predictions from pipeline"
            
            for i, prediction in enumerate(predictions):
                if cancel_requested.is_set():
                    print("🛑 Cancelled during batch write.")
                    status_label.config(text="🛑 Cancelled during batch.")
                    cap.release()
                    out.release()
                    return frame_count

                try:
                    raw_depth = prediction["predicted_depth"]
                    # Convert to uint8 depth (full frame or tiler)
                    if USE_TILED_DEPTH or isinstance(raw_depth, np.ndarray):
                        depth_u8 = _normalize_to_u8(raw_depth, (original_width, original_height), invert=invert_var.get())
                    else:
                        # old path: tensor/other -> grayscale (your helper)
                        depth_u8 = convert_depth_to_grayscale(raw_depth)
                        if invert_var.get():
                            depth_u8 = 255 - depth_u8
                        depth_u8 = cv2.resize(depth_u8, (original_width, original_height), interpolation=cv2.INTER_CUBIC)

                    # letterbox handling (unchanged)
                    if ignore_letterbox_bars and (bars_top or bars_bottom):
                        core_h = original_height - bars_top - bars_bottom
                        if core_h <= 0:
                            bars_top = bars_bottom = 0
                            core_h = original_height
                        gray_core = cv2.resize(depth_u8, (original_width, core_h), interpolation=cv2.INTER_CUBIC)
                        neutral = int(np.median(gray_core)) if gray_core.size else 0
                        full_gray = np.full((original_height, original_width), neutral, dtype=np.uint8)
                        full_gray[bars_top:bars_top+core_h, :] = gray_core
                        bgr = cv2.cvtColor(full_gray, cv2.COLOR_GRAY2BGR)
                        to_save = full_gray
                    else:
                        bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
                        to_save = depth_u8

                    out.write(bgr)
                    if save_frames:
                        frame_filename = os.path.join(frame_output_dir, f"frame_{write_index:05d}.png")
                        cv2.imwrite(frame_filename, to_save)
                    write_index += 1
                    total_processed_frames += 1


                except Exception as e:
                    print(f"⚠️ Depth processing error: {e}")
            
            if frame_count % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()


            frames_batch.clear()

        elapsed = time.time() - global_session_start_time
        avg_fps = total_processed_frames / elapsed if elapsed > 0 else 0
        remaining_frames = total_frames - total_processed_frames
        eta = remaining_frames / avg_fps if avg_fps > 0 else 0

        progress = int(((frames_processed_all + total_processed_frames) / total_frames_all) * 100)
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))

        status_label.config(
            text=f"🎬 {frames_processed_all + total_processed_frames}/{total_frames_all} frames | "
                 f"FPS: {avg_fps:.2f} | Elapsed: {elapsed_str} | ETA: {eta_str} | Processing: {name}"
        )

        status_label.update_idletasks()

    cap.release()
    out.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()



    if cancel_requested.is_set():
        print("🛑 Cancelled: Video not fully processed.")
    else:
        print(f"✅ Video saved: {output_path}")
        
    return frame_count

def is_av1_encoded(file_path):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=nokey=1:noprint_wrappers=1",
                file_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        codec = result.stdout.strip().lower()
        return "av1" in codec
    except Exception as e:
        print(f"⚠️ Failed to check codec with ffprobe: {e}")
        return False

def open_video(status_label, progress_bar, batch_size_widget, output_dir_var, inference_res_var, invert_var, inference_steps_entry, offload_mode_dropdown, ):
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
                "🚫 This video is encoded with AV1, which is not supported by OpenCV in this application.\n\n"
                "Please re-encode it to H.264 using:\n\nffmpeg -i input.mkv -c:v libx264 output.mp4"
            )
            status_label.config(text="❌ AV1 input not supported. Re-encode to H.264.")
            return
            
        offload_mode = offload_mode_dropdown.get() if offload_mode_dropdown else "none"

        cancel_requested.clear()
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

        threading.Thread(target=process_video2, args=(
            file_path,
            total_frames_all,
            0,
            batch_size,
            output_dir_var,
            inference_res_var,
            status_label,
            progress_bar,
            cancel_requested,
            invert_var,
            inference_steps_entry,
        ),
        kwargs={
            "offload_mode_dropdown": offload_mode_dropdown,
            "target_fps": 15,
            "ignore_letterbox_bars": True,
        }
    ).start()

def _log_ex(exctype, value, tb):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open("vd3d_crash.log", "a", encoding="utf-8") as f:
        f.write(f"\n=== {ts} ===\n")
        traceback.print_exception(exctype, value, tb, file=f)
    print("💥 Unhandled exception; see vd3d_crash.log")

sys.excepthook = _log_ex

# Python 3.8+: catch in threads too
if hasattr(threading, "excepthook"):
    def _thread_hook(args):
        _log_ex(args.exc_type, args.exc_value, args.exc_traceback)
    threading.excepthook = _thread_hook
