#!/usr/bin/env python3
"""
VD3D Live: capture → Depth Anything v2 → Pixel-Shift CUDA → SBS (window + optional virtual cam)

Hotkeys:
  f  = fullscreen toggle
  m  = cycle view (Passthrough → Depth → 3D-SBS)
  q/ESC = quit
"""

import os, sys, time, argparse, platform, threading
from collections import deque
from typing import Tuple

import cv2
import numpy as np
# PIL optional; safe to keep import
try:
    from PIL import Image  # noqa: F401
except Exception:
    pass

# ---- Torch & transformers
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from transformers import AutoProcessor, AutoModelForDepthEstimation

# --- Optional virtual camera
try:
    import pyvirtualcam
    HAVE_VCAM = True
except Exception:
    HAVE_VCAM = False

# --- Screen capture (optional)
try:
    import mss
except Exception:
    mss = None

# --- Pixel shift CUDA from your repo
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.getcwd())
    from core.render_3d import pixel_shift_cuda
    HAVE_PIXEL_SHIFT = True
except Exception as e:
    print(f"⚠️ pixel_shift_cuda not available: {e}\n   Falling back to duplicate SBS.")
    HAVE_PIXEL_SHIFT = False
    pixel_shift_cuda = None  # type: ignore


# -------------------- Helpers / Capture -------------------- #
def api_from_name(name: str) -> int:
    table = {
        "any": cv2.CAP_ANY,
        "auto": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "ffmpeg": cv2.CAP_FFMPEG,
    }
    return table.get(name, cv2.CAP_ANY)


# Thread-safe MSS grabber (create mss() in the thread that calls read())
class ScreenGrabber:
    """
    Lightweight screen capture using mss with thread-local handles.
    monitor_index: 1-based (1 = primary). 0 = all monitors bounding box.
    region: optional (x, y, w, h) crop inside the chosen monitor box.
    max_fps: soft cap; we sleep to avoid over-capturing.
    """
    def __init__(self, monitor_index=1, region=None, max_fps=60):
        if mss is None:
            raise RuntimeError("mss is not installed. Run: pip install mss")

        # Probe monitors once to get base rect, but don't keep handles
        with mss.mss() as probe:
            monitors = probe.monitors  # idx 0 = virtual all, 1..N = physical
            if monitor_index < 0 or monitor_index >= len(monitors):
                monitor_index = 1
            base = monitors[monitor_index]
        if region:
            x, y, w, h = region
            self.box = {
                "left": base["left"] + int(x),
                "top": base["top"] + int(y),
                "width": int(w),
                "height": int(h),
            }
        else:
            self.box = {
                "left": base["left"],
                "top": base["top"],
                "width": base["width"],
                "height": base["height"],
            }
        self.max_fps = max(1, int(max_fps)) if max_fps else None
        self._last = 0.0
        self._tls = threading.local()

    def _get_sct(self):
        sct = getattr(self._tls, "sct", None)
        if sct is None:
            self._tls.sct = mss.mss()
            sct = self._tls.sct
        return sct

    def read(self):
        # throttle
        if self.max_fps:
            now = time.perf_counter()
            wait = (1.0 / self.max_fps) - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.perf_counter()
        sct = self._get_sct()
        shot = sct.grab(self.box)  # BGRA
        frame = np.asarray(shot)[:, :, :3]  # drop alpha → BGR
        return True, frame

    def release(self):
        try:
            sct = getattr(self._tls, "sct", None)
            if sct:
                sct.close()
                self._tls.sct = None
        except Exception:
            pass


def _warm_read_ok(cap, tries=12, sleep_s=0.02, diag=False):
    ok, frame = False, None
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0 and frame.mean() > 1.0:
            if diag:
                h, w = frame.shape[:2]
                print(f"[diag] warm frame ok: {w}x{h}, mean={float(frame.mean()):.1f}")
            return True
        time.sleep(sleep_s)
    if diag:
        print("[diag] warm frames all blank or empty")
    return False


def _negotiate_dshow(cap, args, diag=False):
    """
    Try common FourCCs and toggles to avoid black frames on dshow.
    Returns True if frames look valid, else False.
    """
    # Always request RGB conversion on dshow
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        if diag: print("[diag] set CONVERT_RGB=1")
    except Exception:
        pass

    # candidates to try
    fourccs = []
    if getattr(args, "fourcc", ""):
        fourccs.append(args.fourcc.upper())
    fourccs += ["MJPG", "YUY2", "NV12"]

    fps_candidates = [float(args.fps)] if getattr(args, "fps", None) else []
    fps_candidates += [None]

    for fcc in fourccs:
        try:
            four = cv2.VideoWriter_fourcc(*fcc)
            cap.set(cv2.CAP_PROP_FOURCC, four)
            if diag: print(f"[diag] try FOURCC={fcc}")
        except Exception:
            pass

        if getattr(args, "width", None):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
        if getattr(args, "height", None):
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))

        for fps in fps_candidates:
            if fps is not None:
                try:
                    cap.set(cv2.CAP_PROP_FPS, float(fps))
                    if diag: print(f"[diag] try FPS={fps}")
                except Exception:
                    pass

            if _warm_read_ok(cap, diag=diag):
                if diag:
                    gw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    gh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    gf = cap.get(cv2.CAP_PROP_FPS)
                    print(f"[diag] dshow negotiated: {gw}x{gh} @ {gf:.2f}fps FOURCC={fcc}")
                return True

        # clear FPS for next loop
        try:
            cap.set(cv2.CAP_PROP_FPS, 0)
        except Exception:
            pass

    return False


def open_capture(args):
    """
    Unified capture:
      - screen via MSS if --source startswith 'screen'
      - device via OpenCV otherwise (with fallback chain)
    """
    # ----- decide screen vs device -----
    source_raw = getattr(args, "source", "device")
    source = str(source_raw).strip().lower() if source_raw is not None else "device"

    # ----- screen via MSS -----
    if source.startswith("screen"):
        parts = source.split(":")
        mon_idx = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 1
        region = tuple(args.crop) if getattr(args, "crop", None) else None
        grab = ScreenGrabber(
            monitor_index=mon_idx,
            region=region,
            max_fps=getattr(args, "capture_fps", 60),
        )
        box = grab.box
        print(
            f"🖥️  Screen capture: monitor={mon_idx} {box['width']}x{box['height']} @ ~{getattr(args, 'capture_fps', 60)}fps"
            + (f" crop=({box['left']},{box['top']},{box['width']},{box['height']})" if region else "")
        )
        return grab  # exposes read()/release()

    # ----- device via OpenCV (webcam/capture card) -----
    diag = getattr(args, "diag", False)

    req = (args.backend or "any").lower()
    if req in ("dshow", "msmf", "ffmpeg"):
        order = [req, "dshow", "msmf", "any"]  # try requested first, then fallbacks
    else:
        order = ["dshow", "msmf", "any"]

    cap = None

    # Try by device name first if provided (Windows)
    if getattr(args, "dshow_name", None) and platform.system() == "Windows":
        label = args.dshow_name
        if req == "ffmpeg" and not label.lower().startswith("video="):
            label = f"video={label}"
        for be in order:
            pref = api_from_name(be)
            cap = cv2.VideoCapture(label, pref)
            if diag:
                print(f"[diag] try name '{label}' backend={be} -> opened={cap.isOpened()}")
            if cap.isOpened():
                break

    # Fallback to index
    if cap is None or not cap.isOpened():
        idx = int(getattr(args, "device_index", 0))
        for be in order:
            pref = api_from_name(be)
            cap = cv2.VideoCapture(idx, pref)
            if diag:
                print(f"[diag] try index {idx} backend={be} -> opened={cap.isOpened()}")
            if cap.isOpened():
                break

    # Last chance: CAP_ANY
    if cap is None or not cap.isOpened():
        idx = int(getattr(args, "device_index", 0))
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if diag:
            print(f"[diag] final CAP_ANY open -> opened={cap.isOpened()}")

    if cap is None or not cap.isOpened():
        raise RuntimeError("Failed to open capture (try --scan, different --device-index, --backend, or --dshow-name).")

    # Request mode (best-effort)
    if getattr(args, "width", None):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    if getattr(args, "height", None):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    if getattr(args, "fps", None):
        try:
            cap.set(cv2.CAP_PROP_FPS, float(args.fps))
        except Exception:
            pass

    if getattr(args, "fourcc", ""):
        try:
            four = cv2.VideoWriter_fourcc(*args.fourcc.upper())
            cap.set(cv2.CAP_PROP_FOURCC, four)
            print(f"🎞️ Requested FOURCC {args.fourcc.upper()}")
        except Exception:
            pass

    # Try to force RGB conversion if supported
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except Exception:
        pass

    # Low latency
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Negotiate dshow if early frames are blank
    if platform.system() == "Windows" and req in ("dshow", "any", "auto"):
        if not _warm_read_ok(cap, diag=diag):
            if diag:
                print("[diag] first frames are blank, negotiating dshow formats...")
            _negotiate_dshow(cap, args, diag=diag)

    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📷 Opened camera: {got_w}x{got_h} @ {got_fps:.2f}fps (backend={req})")
    return cap


def scan_devices():
    print("🔎 Scanning indices 0..9 on dshow/msmf")
    for backend in ("dshow", "msmf"):
        pref = api_from_name(backend)
        for idx in range(10):
            cap = cv2.VideoCapture(idx, pref)
            ok, _ = (cap.read() if cap.isOpened() else (False, None))
            print(f"{backend:5s} idx {idx}: {'OK' if ok else 'fail'}")
            cap.release()


def start_latest_capture(cap):
    """
    Dedicated reader thread keeps the newest frame only.
    Display loop never blocks on cap.read().
    Works with cv2.VideoCapture *and* ScreenGrabber.
    """
    q = deque(maxlen=1)
    stop = threading.Event()

    def _reader():
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            q.append(frame)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return q, stop


# -------------------- Depth model (fast path) -------------------- #
def make_da2(model_id: str, use_fp16: bool):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available. Install torch to run the depth model.")
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    dtype = torch.float16 if (use_fp16 and CUDA_AVAILABLE) else torch.float32
    proc = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    if CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+
    return model, proc, device


def depth_from_frame_fast(model, proc, device: str, frame_bgr: np.ndarray, inference_size: Tuple[int, int]) -> np.ndarray:
    """Works with DA-v2 (predicted_depth [B,H,W]) and also [B,1,H,W] models."""
    h, w = frame_bgr.shape[:2]
    iw, ih = inference_size
    small = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_AREA) if (w, h) != (iw, ih) else frame_bgr
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    amp_dtype = (torch.float16 if next(model.parameters()).dtype == torch.float16 else torch.float32)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=amp_dtype):
        inputs = proc(images=rgb, return_tensors="pt").to(device)
        out = model(**inputs).predicted_depth      # DA2: [B,H,W]
        if out.ndim == 4:                          # other models: [B,1,H,W]
            out = out[:, 0]                        # -> [B,H,W]

        # upscale on GPU for quality/speed
        pred = torch.nn.functional.interpolate(out.unsqueeze(1), size=(h, w),
                                               mode="bicubic", align_corners=False).squeeze(1)
        depth = pred[0].float().cpu().numpy()

    # percentile stretch → 0..1
    lo, hi = np.percentile(depth, 1), np.percentile(depth, 99)
    if hi - lo < 1e-6:
        return np.full_like(depth, 0.5, dtype=np.float32)
    return np.clip((depth - lo) / (hi - lo), 0, 1).astype(np.float32)


# -------------------- Utilities -------------------- #
def sbs_pack(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    if left_bgr.shape != right_bgr.shape:
        h = min(left_bgr.shape[0], right_bgr.shape[0])
        w = min(left_bgr.shape[1], right_bgr.shape[1])
        left_bgr  = cv2.resize(left_bgr,  (w, h))
        right_bgr = cv2.resize(right_bgr, (w, h))
    return np.hstack([left_bgr, right_bgr])


def apply_preset(args):
    if args.preset == "console1080":
        args.infer_w, args.infer_h = 576, 320
        args.depth_fps = 12
    elif args.preset == "webcam720":
        args.infer_w, args.infer_h = 512, 288
        args.depth_fps = 12
    elif args.preset == "lowlat":
        args.infer_w, args.infer_h = 512, 288
        args.depth_fps = 15
    # "default" → keep user values


# -------------------- Main runtime -------------------- #
def run_live(args, external_stop: threading.Event | None = None):
    """Main live loop: capture -> depth -> pixel shift -> preview/virtualcam"""
    # Open capture & start reader
    cap = open_capture(args)
    frame_q, stop_cap = start_latest_capture(cap)

    # Quick warm-up to fail fast if nothing is arriving
    first = None
    t0 = time.time()
    while time.time() - t0 < 2.0:
        if frame_q:
            first = frame_q[-1]
            break
        time.sleep(0.01)
    if first is None or first.size == 0:
        raise RuntimeError(
            "No frames arriving from capture. Try --backend msmf or --fourcc MJPG or a different --device-index."
        )

    print(f"🔥 CUDA available: {CUDA_AVAILABLE} | Using {'cuda' if CUDA_AVAILABLE else 'cpu'}")

    # Depth model
    model, proc, device = make_da2(args.model, args.fp16)

    # UI / Preview setup
    win = "VD3D Live"
    source_is_screen = isinstance(getattr(args, "source", "device"), str) and str(args.source).lower().startswith("screen")
    show_preview = not getattr(args, "no_preview", False)

    if source_is_screen and not getattr(args, "force_preview", False) and not getattr(args, "mask_preview", False):
        print(
            "🔇 Preview disabled to avoid screen-capture feedback. Use --mask-preview or move the window to another monitor, or pass --force-preview to override."
        )
        show_preview = False

    if show_preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, getattr(args, "preview_w", 960), getattr(args, "preview_h", 540))
        cv2.moveWindow(win, getattr(args, "preview_x", 60), getattr(args, "preview_y", 60))

    fullscreen = False
    view_mode = 2 if args.sbs else 0  # 0=passthrough, 1=depth, 2=SBS

    # Timing state
    fps_ema = None
    t_last = time.time()

    # Depth scheduling
    depth01 = None
    depth_last_t = 0.0
    depth_period = 1.0 / max(1e-3, args.depth_fps)

    # Smoothing state
    ema_alpha = float(args.ema)
    depth_ema = None if args.smooth else None

    # Virtual cam
    vcam = None

    print("▶️  Streaming… (f=fullscreen, m=mode, q=quit)")
    while True:
        # allow GUI to stop us
        if external_stop is not None and external_stop.is_set():
            break

        # --- get latest frame from the reader queue ---
        try:
            frame = frame_q[-1]
        except IndexError:
            time.sleep(0.002)
            continue
        if frame is None or frame.size == 0:
            continue



        # If capturing screen and masking preview, blank that rect to break recursion
        if source_is_screen and getattr(args, "mask_preview", False) and hasattr(cap, "box"):
            rel_x = getattr(args, "preview_x", 60) - cap.box["left"]
            rel_y = getattr(args, "preview_y", 60) - cap.box["top"]
            rw, rh = int(getattr(args, "preview_w", 960)), int(getattr(args, "preview_h", 540))
            H, W = frame.shape[:2]
            x0, y0 = max(0, rel_x), max(0, rel_y)
            x1, y1 = min(W, rel_x + rw), min(H, rel_y + rh)
            if x1 > x0 and y1 > y0:
                frame[y0:y1, x0:x1] = 0

        now = time.time()

        # Recompute depth at target rate
        if (depth01 is None) or (now - depth_last_t >= depth_period):
            depth_new = depth_from_frame_fast(model, proc, device, frame, (args.infer_w, args.infer_h))
            depth_last_t = now
            if args.smooth:
                if depth_ema is None:
                    depth_ema = depth_new
                else:
                    depth_ema = ema_alpha * depth_ema + (1.0 - ema_alpha) * depth_new
                # mild denoise for temporal stability
                depth01 = cv2.medianBlur((depth_ema * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
            else:
                depth01 = depth_new

        # Choose view
        if view_mode == 0:
            out_bgr = frame
        elif view_mode == 1:
            d8 = (depth01 * 255.0).astype(np.uint8)
            out_bgr = cv2.applyColorMap(d8, cv2.COLORMAP_VIRIDIS)
        else:
            if HAVE_PIXEL_SHIFT and pixel_shift_cuda is not None and CUDA_AVAILABLE:
                frm_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frm_t = torch.from_numpy(frm_rgb).float().permute(2, 0, 1) / 255.0
                d_t = torch.from_numpy(depth01).float().unsqueeze(0)
                frm_t = frm_t.to("cuda"); d_t = d_t.to("cuda")
                h, w = frame.shape[:2]
                left, right = pixel_shift_cuda(
                    frm_t, d_t, w, h,
                    args.fg_shift, args.mg_shift, args.bg_shift,
                    blur_ksize=9, feather_strength=12.0,
                    return_shift_map=False,
                    enable_feathering=True,
                    enable_edge_masking=True,
                )
                out_bgr = sbs_pack(left, right)
            else:
                out_bgr = sbs_pack(frame, frame)

        # Lazy-init virtual cam
        if vcam is None and args.virtualcam and HAVE_VCAM:
            out_h, out_w = out_bgr.shape[:2]
            vcam = pyvirtualcam.Camera(width=out_w, height=out_h, fps=args.vcam_fps,
                                       fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"📡 Virtual camera started: {vcam.device}")

        # FPS overlay
        dt = now - t_last
        t_last = now
        inst = (1.0 / dt) if dt > 0 else 0.0
        fps_ema = inst if fps_ema is None else (0.9 * fps_ema + 0.1 * inst)
        cv2.putText(out_bgr, f"{fps_ema:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 3)
        cv2.putText(out_bgr, f"{fps_ema:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        # Show + send
        if show_preview:
            cv2.imshow(win, out_bgr)
        if vcam is not None:
            vcam.send(out_bgr)
            vcam.sleep_until_next_frame()

        # Hotkeys
        key = (cv2.waitKey(1) & 0xFF) if show_preview else 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('f') and show_preview:
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, prop)
        elif key == ord('m'):
            view_mode = (view_mode + 1) % 3


    # Cleanup
    if vcam is not None:
        vcam.close()
    stop_cap.set()
    try:
        cap.release()
    except Exception:
        pass
    if show_preview:
        cv2.destroyAllWindows()


# -------------------- CLI -------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="VD3D Live: external input → DA2 depth → pixel-shift SBS")
    p.add_argument("--scan", action="store_true", default=False, help="Scan indices 0..9 on dshow/msmf and exit")
    p.add_argument("--backend", choices=["any", "dshow", "msmf", "ffmpeg", "auto"], default="any", help="Capture backend")
    p.add_argument("--fourcc", type=str, default="", help="Request pixel format (e.g. MJPG, YUY2, NV12)")
    p.add_argument("--device-index", type=int, default=0, help="VideoCapture index")
    p.add_argument("--dshow-name", type=str, default=None, help="Device name (Windows). For ffmpeg, 'video=' is added.")
    p.add_argument("--width", type=int, default=1280, help="Requested capture width")
    p.add_argument("--height", type=int, default=720, help="Requested capture height")
    p.add_argument("--fps", type=int, default=60, help="Requested capture FPS")

    p.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--fp16", action="store_true", help="Load model in FP16 on CUDA")
    p.add_argument("--infer-w", type=int, default=640, help="Inference width (depth net input)")
    p.add_argument("--infer-h", type=int, default=352, help="Inference height (depth net input)")
    p.add_argument("--depth-fps", type=float, default=12.0, help="Depth update rate; display stays unlocked")

    p.add_argument("--sbs", action="store_true", help="Start in 3D side-by-side view")
    p.add_argument("--fg-shift", type=float, default=20.0, help="Foreground pixel shift")
    p.add_argument("--mg-shift", type=float, default=10.0, help="Midground pixel shift")
    p.add_argument("--bg-shift", type=float, default=-14.0, help="Background pixel shift (neg pulls back)")

    p.add_argument("--smooth", dest="smooth", action="store_true", help="Enable temporal smoothing")
    p.add_argument("--no-smooth", dest="smooth", action="store_false", help="Disable temporal smoothing")
    p.set_defaults(smooth=True)
    p.add_argument("--ema", type=float, default=0.25, help="EMA alpha (0..1); higher=more smoothing")
    p.add_argument("--preset", choices=["default","console1080","webcam720","lowlat"], default="default",
                   help="Quick tuning presets")

    p.add_argument("--virtualcam", action="store_true", help="Send output to a virtual webcam (pyvirtualcam)")
    p.add_argument("--vcam-fps", type=int, default=30, help="Virtual webcam FPS")

    # screen & preview controls
    p.add_argument("--source", type=str, default="device",
                   help="device | screen | screen:N  (N = 0 all, 1 primary, 2 second, ...)")
    p.add_argument("--capture-fps", type=int, default=60,
                   help="Max screen capture FPS when --source is screen")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X","Y","W","H"),
                   help="Crop region for screen capture (relative to chosen monitor)")

    p.add_argument("--no-preview", action="store_true",
                   help="Disable on-screen preview window.")
    p.add_argument("--force-preview", action="store_true",
                   help="Force preview even if capturing the same monitor.")
    p.add_argument("--preview-x", type=int, default=60)
    p.add_argument("--preview-y", type=int, default=60)
    p.add_argument("--preview-w", type=int, default=960)
    p.add_argument("--preview-h", type=int, default=540)
    p.add_argument("--mask-preview", action="store_true",
                   help="When capturing screen, mask the preview window area in the captured frame to prevent recursion.")

    p.add_argument("--diag", action="store_true", help="Verbose capture diagnostics")
    p.add_argument("--gui", action="store_true", help="Open the VD3D Live GUI and exit CLI mode")
    return p

# ---------- Minimal Dark-Mode GUI for VD3D Live ----------
import tkinter as tk
from tkinter import ttk

def launch_live_gui(parent: tk.Tk | None = None):
    """
    Launches a small dark-mode control window that starts/stops run_live(...) in a background thread.
    Call this from your main app:  from vd3d_live import launch_live_gui; launch_live_gui()
    """
    # ---- build window ----
    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("VD3D External 3D")
    win.geometry("540x570")
    win.configure(bg="#121212")
    win.attributes("-topmost", False)

    # ---- simple dark theme ----
    style = ttk.Style(win)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", background="#121212", foreground="#e6e6e6")
    style.configure("TLabel", background="#121212", foreground="#e6e6e6")
    style.configure("TFrame", background="#121212")
    style.configure("TCheckbutton", background="#121212", foreground="#e6e6e6")
    style.configure("TEntry", fieldbackground="#1E1E1E", foreground="#e6e6e6")
    style.configure("TCombobox", fieldbackground="#1E1E1E", background="#1E1E1E", foreground="#e6e6e6")
    style.map("TButton",
        background=[("active", "#2c2c2c"), ("!disabled", "#1e1e1e")],
        foreground=[("!disabled", "#f5f5f5")]
    )

    # ---- variables ----
    backend_v     = tk.StringVar(value="any")      # any|dshow|msmf|ffmpeg|auto
    source_v      = tk.StringVar(value="device")   # device | screen | screen:1
    index_v       = tk.IntVar(value=0)
    dshow_name_v  = tk.StringVar(value="")
    fourcc_v      = tk.StringVar(value="")
    width_v       = tk.IntVar(value=1280)
    height_v      = tk.IntVar(value=720)
    fps_v         = tk.IntVar(value=60)

    model_v       = tk.StringVar(value="depth-anything/Depth-Anything-V2-Small-hf")
    fp16_v        = tk.BooleanVar(value=True)
    infer_w_v     = tk.IntVar(value=512)
    infer_h_v     = tk.IntVar(value=288)
    depth_fps_v   = tk.DoubleVar(value=12.0)

    sbs_v         = tk.BooleanVar(value=True)
    fg_shift_v    = tk.DoubleVar(value=7.0)
    mg_shift_v    = tk.DoubleVar(value=1.5)
    bg_shift_v    = tk.DoubleVar(value=-3.5)

    smooth_v      = tk.BooleanVar(value=True)
    ema_v         = tk.DoubleVar(value=0.55)
    preset_v      = tk.StringVar(value="default")  # default|console1080|webcam720|lowlat

    vcam_v        = tk.BooleanVar(value=False)
    vcam_fps_v    = tk.IntVar(value=30)

    no_preview_v  = tk.BooleanVar(value=False)
    force_prev_v  = tk.BooleanVar(value=False)
    mask_prev_v   = tk.BooleanVar(value=False)
    prev_x_v      = tk.IntVar(value=60)
    prev_y_v      = tk.IntVar(value=60)
    prev_w_v      = tk.IntVar(value=960)
    prev_h_v      = tk.IntVar(value=540)

    diag_v        = tk.BooleanVar(value=False)

    # ---- layout helpers ----
    def L(f, text): return ttk.Label(f, text=text)
    def E(f, textvar, w=10): 
        e = ttk.Entry(f, textvariable=textvar, width=w); return e
    def C(f, text, var): 
        c = ttk.Checkbutton(f, text=text, variable=var); return c
    def CB(f, textvar, values, w=14):
        cb = ttk.Combobox(f, textvariable=textvar, values=values, width=w, state="readonly"); return cb

    root = ttk.Frame(win)
    root.pack(fill="both", expand=True, padx=12, pady=12)

    # ---- sections ----
    sec_cap = ttk.LabelFrame(root, text="Capture", padding=10)
    sec_cap.pack(fill="x", expand=False, pady=(0,10))

    L(sec_cap, "Backend").grid(row=0, column=0, sticky="w"); CB(sec_cap, backend_v, ["any","dshow","msmf","ffmpeg","auto"]).grid(row=0, column=1, padx=6)
    L(sec_cap, "Source").grid(row=0, column=2, sticky="w"); CB(sec_cap, source_v, ["device","screen","screen:1","screen:2","screen:0"]).grid(row=0, column=3, padx=6)
    L(sec_cap, "Index").grid(row=1, column=0, sticky="w"); E(sec_cap, index_v, 6).grid(row=1, column=1, sticky="w")
    L(sec_cap, "dshow name").grid(row=1, column=2, sticky="w"); E(sec_cap, dshow_name_v, 18).grid(row=1, column=3, sticky="w")
    L(sec_cap, "FOURCC").grid(row=2, column=0, sticky="w"); E(sec_cap, fourcc_v, 8).grid(row=2, column=1, sticky="w")
    L(sec_cap, "WxH@FPS").grid(row=2, column=2, sticky="w")
    whf = ttk.Frame(sec_cap); whf.grid(row=2, column=3, sticky="w")
    E(whf, width_v, 6).pack(side="left"); ttk.Label(whf, text="x").pack(side="left", padx=3)
    E(whf, height_v, 6).pack(side="left"); ttk.Label(whf, text="@").pack(side="left", padx=3)
    E(whf, fps_v, 6).pack(side="left")

    sec_depth = ttk.LabelFrame(root, text="Depth Model", padding=10)
    sec_depth.pack(fill="x", expand=False, pady=(0,10))
    L(sec_depth, "Model ID").grid(row=0, column=0, sticky="w"); E(sec_depth, model_v, 34).grid(row=0, column=1, columnspan=3, sticky="we")
    C(sec_depth, "FP16", fp16_v).grid(row=1, column=0, sticky="w")
    L(sec_depth, "Infer WxH").grid(row=1, column=1, sticky="w")
    inf = ttk.Frame(sec_depth); inf.grid(row=1, column=2, sticky="w")
    E(inf, infer_w_v, 6).pack(side="left"); ttk.Label(inf, text="x").pack(side="left", padx=3)
    E(inf, infer_h_v, 6).pack(side="left")
    L(sec_depth, "Depth FPS").grid(row=1, column=3, sticky="w"); E(sec_depth, depth_fps_v, 6).grid(row=1, column=4, sticky="w")

    sec_3d = ttk.LabelFrame(root, text="Stereo & Parallax", padding=10)
    sec_3d.pack(fill="x", expand=False, pady=(0,10))
    C(sec_3d, "Start in SBS view", sbs_v).grid(row=0, column=0, sticky="w")
    L(sec_3d, "FG/MG/BG shift").grid(row=0, column=1, sticky="w")
    sh = ttk.Frame(sec_3d); sh.grid(row=0, column=2, sticky="w")
    E(sh, fg_shift_v, 6).pack(side="left"); ttk.Label(sh, text="/").pack(side="left", padx=2)
    E(sh, mg_shift_v, 6).pack(side="left"); ttk.Label(sh, text="/").pack(side="left", padx=2)
    E(sh, bg_shift_v, 6).pack(side="left")

    sec_smooth = ttk.LabelFrame(root, text="Smoothing / Presets", padding=10)
    sec_smooth.pack(fill="x", expand=False, pady=(0,10))
    C(sec_smooth, "Temporal smoothing", smooth_v).grid(row=0, column=0, sticky="w")
    L(sec_smooth, "EMA α").grid(row=0, column=1, sticky="w"); E(sec_smooth, ema_v, 6).grid(row=0, column=2, sticky="w")
    L(sec_smooth, "Preset").grid(row=0, column=3, sticky="w"); CB(sec_smooth, preset_v, ["default","console1080","webcam720","lowlat"]).grid(row=0, column=4, sticky="w")

    sec_out = ttk.LabelFrame(root, text="Output", padding=10)
    sec_out.pack(fill="x", expand=False, pady=(0,10))
    C(sec_out, "Virtual cam", vcam_v).grid(row=0, column=0, sticky="w")
    L(sec_out, "VCam FPS").grid(row=0, column=1, sticky="w"); E(sec_out, vcam_fps_v, 6).grid(row=0, column=2, sticky="w")

    sec_prev = ttk.LabelFrame(root, text="Preview", padding=10)
    sec_prev.pack(fill="x", expand=False, pady=(0,10))
    C(sec_prev, "Disable preview", no_preview_v).grid(row=0, column=0, sticky="w")
    C(sec_prev, "Force preview (screen cap)", force_prev_v).grid(row=0, column=1, sticky="w")
    C(sec_prev, "Mask preview rect (screen cap)", mask_prev_v).grid(row=0, column=2, sticky="w")
    L(sec_prev, "Pos (x,y)").grid(row=1, column=0, sticky="w")
    pos = ttk.Frame(sec_prev); pos.grid(row=1, column=1, sticky="w")
    E(pos, prev_x_v, 6).pack(side="left"); ttk.Label(pos, text=",").pack(side="left", padx=2)
    E(pos, prev_y_v, 6).pack(side="left")
    L(sec_prev, "Size (w,h)").grid(row=1, column=2, sticky="w")
    siz = ttk.Frame(sec_prev); siz.grid(row=1, column=3, sticky="w")
    E(siz, prev_w_v, 6).pack(side="left"); ttk.Label(siz, text=",").pack(side="left", padx=2)
    E(siz, prev_h_v, 6).pack(side="left")

    C(root, "Diagnostics (verbose capture)", diag_v).pack(anchor="w")

    btns = ttk.Frame(root); btns.pack(fill="x", pady=(12,0))
    start_btn = ttk.Button(btns, text="Start")
    stop_btn  = ttk.Button(btns, text="Stop", state="disabled")
    start_btn.pack(side="left", padx=(0,8))
    stop_btn.pack(side="left")

    status = ttk.Label(root, text="Idle", anchor="w")
    status.pack(fill="x", pady=(12,0))

    # ---- threading glue ----
    run_thread: threading.Thread | None = None
    stop_event: threading.Event | None = None

    def to_args():
        # Build a minimal Namespace-like object expected by run_live/apply_preset
        class A: pass
        a = A()
        a.backend      = backend_v.get()
        a.source       = source_v.get()
        a.device_index = index_v.get()
        a.dshow_name   = (dshow_name_v.get().strip() or None)
        a.fourcc       = fourcc_v.get().strip()
        a.width        = width_v.get()
        a.height       = height_v.get()
        a.fps          = fps_v.get()
        a.model        = model_v.get()
        a.fp16         = bool(fp16_v.get())
        a.infer_w      = infer_w_v.get()
        a.infer_h      = infer_h_v.get()
        a.depth_fps    = float(depth_fps_v.get())
        a.sbs          = bool(sbs_v.get())
        a.fg_shift     = float(fg_shift_v.get())
        a.mg_shift     = float(mg_shift_v.get())
        a.bg_shift     = float(bg_shift_v.get())
        a.smooth       = bool(smooth_v.get())
        a.ema          = float(ema_v.get())
        a.preset       = preset_v.get()
        a.virtualcam   = bool(vcam_v.get())
        a.vcam_fps     = int(vcam_fps_v.get())
        a.no_preview   = bool(no_preview_v.get())
        a.force_preview= bool(force_prev_v.get())
        a.mask_preview = bool(mask_prev_v.get())
        a.preview_x    = int(prev_x_v.get())
        a.preview_y    = int(prev_y_v.get())
        a.preview_w    = int(prev_w_v.get())
        a.preview_h    = int(prev_h_v.get())
        a.capture_fps  = int(fps_v.get())  # reuse for screen cap throttle
        a.crop         = None
        a.diag         = bool(diag_v.get())
        return a

    def start():
        nonlocal run_thread, stop_event
        if run_thread and run_thread.is_alive():
            return
        args = to_args()
        apply_preset(args)  # use your preset helper
        stop_event = threading.Event()
        status.config(text="Starting…")
        start_btn.config(state="disabled")
        stop_btn.config(state="normal")

        def _run():
            try:
                run_live(args, external_stop=stop_event)
            except Exception as e:
                status.config(text=f"Error: {e}")
            finally:
                start_btn.config(state="normal")
                stop_btn.config(state="disabled")
                status.config(text="Idle")

        run_thread = threading.Thread(target=_run, daemon=True)
        run_thread.start()

    def stop():
        nonlocal stop_event
        if stop_event is not None:
            stop_event.set()

    start_btn.config(command=start)
    stop_btn.config(command=stop)

    def on_close():
        stop()
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)
    if parent is None:
        win.mainloop()

if __name__ == "__main__":
    # Optional: DPI awareness to match pixels 1:1 on Windows
    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args()
    
    # Open GUI by default if no CLI args, or when --gui is passed
    if args.gui or len(sys.argv) == 1:
        launch_live_gui()
        sys.exit(0)



    if args.diag:
        print("[diag] argv:", " ".join(sys.argv))
        print("[diag] parsed args.source:", getattr(args, "source", None))
        print("[diag] running file:", __file__)

    if args.scan:
        scan_devices()
        sys.exit(0)

    apply_preset(args)

    try:
        run_live(args)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)
        
        


