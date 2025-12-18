#!/usr/bin/env python3
"""
VD3D Live (GUI only): capture -> Depth Anything v2 -> Pixel-Shift CUDA -> SBS

Hotkeys (when preview window is visible):
  f      fullscreen toggle
  m      cycle view (Passthrough -> Depth -> 3D-SBS)
  q / ESC  quit
"""

import os, sys, time, platform, threading, subprocess, shlex
from collections import deque
from typing import Tuple
from threading import Thread, Lock

import cv2
import numpy as np

from flask import Flask, Response
from argparse import Namespace

# -------------------- GUI -------------------- #
import tkinter as tk
from tkinter import ttk
import threading as _threading
from tkinter import messagebox


# PIL optional
try:
    from PIL import Image  # noqa: F401
except Exception:
    pass

# ---- Torch & transformers
try:
    import torch
    from transformers import AutoModelForDepthEstimation
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    torch = None  # type: ignore
    AutoModelForDepthEstimation = None  # type: ignore
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

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

# Ensure project root on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Pixel shift CUDA from your repo
HAVE_PIXEL_SHIFT = False
pixel_shift_cuda = None
try:
    from core.render_3d import pixel_shift_cuda
    HAVE_PIXEL_SHIFT = True
    print("External 3D Pipeline: pixel_shift_cuda loaded from core.render_3d")
except Exception as e:
    print(f"⚠️ pixel_shift_cuda not available: {e}\n   Falling back to duplicate SBS.")

# -------------------- Helpers / Capture -------------------- #
def api_from_name(name: str) -> int:
    name = (name or "any").lower()
    table = {
        "any": cv2.CAP_ANY,
        "auto": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "ffmpeg": cv2.CAP_FFMPEG,
    }
    return table.get(name, cv2.CAP_ANY)


class ScreenGrabber:
    """
    Lightweight screen capture using mss with thread-local handles.
    monitor_index: 1-based (1 = primary). 0 = all monitors bounding box.
    region: optional (x, y, w, h) crop inside the chosen monitor box.
    max_fps: soft cap; we sleep to avoid over-capturing.
    """
    def __init__(self, monitor_index=1, region=None, max_fps=60, diag=False):
        if mss is None:
            raise RuntimeError("mss is not installed. Run: pip install mss")

        if diag:
            print(f"[screen] init monitor_index={monitor_index} region={region} max_fps={max_fps}")

        with mss.mss() as probe:
            monitors = probe.monitors
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
        if self.max_fps:
            now = time.perf_counter()
            wait = (1.0 / self.max_fps) - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.perf_counter()

        sct = self._get_sct()
        shot = sct.grab(self.box)  # BGRA
        frame = np.asarray(shot)[:, :, :3]
        return True, frame

    def release(self):
        try:
            sct = getattr(self._tls, "sct", None)
            if sct:
                sct.close()
                self._tls.sct = None
        except Exception:
            pass


def open_capture(args):
    """
    Very simple, robust capture opener:
      - source starts with "screen" -> ScreenGrabber + mss
      - source == "device" -> cv2.VideoCapture on given backend + index
    """
    diag = bool(getattr(args, "diag", False))
    source_raw = getattr(args, "source", "device")
    source = str(source_raw).strip().lower() if source_raw is not None else "device"

    # ---- Screen capture path ----
    if source.startswith("screen"):
        parts = source.split(":")
        mon_idx = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 1
        region = tuple(args.crop) if getattr(args, "crop", None) else None
        grab = ScreenGrabber(
            monitor_index=mon_idx,
            region=region,
            max_fps=getattr(args, "capture_fps", 60),
            diag=diag,
        )
        box = grab.box
        print(
            f"🖥️  Screen capture: monitor={mon_idx} {box['width']}x{box['height']} "
            f"@ ~{getattr(args, 'capture_fps', 60)}fps"
        )

        # Test one grab so run_live sees a valid first frame
        ok, frame = grab.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("ScreenGrabber opened but no frames returned on first read.")
        if diag:
            h, w = frame.shape[:2]
            print(f"[screen] first frame ok: {w}x{h}")
        return grab

    # ---- Device (camera / capture card) path ----
    backend_name = (args.backend or "any")
    api = api_from_name(backend_name)
    idx = int(getattr(args, "device_index", 0))

    if diag:
        print(f"[capture] opening device index={idx} backend={backend_name} (api={api})")

    cap = cv2.VideoCapture(idx, api)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open capture index {idx} with backend={backend_name} (api={api}).")

    # Minimal property setup; don't over-negotiate
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

    # Force RGB conversion where supported
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except Exception:
        pass

    # Test one read right here
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        cap.release()
        raise RuntimeError(
            f"No frames arriving from capture index={idx} backend={backend_name} "
            f"(api={api}). Try another backend or index."
        )

    h, w = frame.shape[:2]
    fps_reported = cap.get(cv2.CAP_PROP_FPS)
    print(f"📷 Opened camera: {w}x{h} @ {fps_reported:.2f}fps (backend={backend_name})")

    # We keep the capture open and reuse it; run_live will start a reader thread.
    return cap


def start_latest_capture(cap):
    """
    Reader thread that always keeps the most recent frame in a deque(maxlen=1).
    Works with both cv2.VideoCapture and ScreenGrabber (must have .read()).
    """
    q = deque(maxlen=1)
    stop = threading.Event()

    def _reader():
        while not stop.is_set():
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None

            if not ok or frame is None or frame.size == 0:
                time.sleep(0.005)
                continue
            q.append(frame)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return q, stop

# -------------------- Depth model -------------------- #
def make_da2(model_id: str, use_fp16: bool):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available.")
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    dtype = torch.float16 if (use_fp16 and CUDA_AVAILABLE) else torch.float32

    if device == "cuda":
        print(f"🧠 Loading depth model on CUDA ({model_id})...")
    else:
        print(f"🧠 Loading depth model on CPU ({model_id})...")

    model = AutoModelForDepthEstimation.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device).eval()
    torch.set_grad_enabled(False)
    if CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    try:
        cfg = model.config.vision_config
        mean = torch.tensor(
            getattr(cfg, "image_mean", [0.5, 0.5, 0.5]),
            device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            getattr(cfg, "image_std", [0.5, 0.5, 0.5]),
            device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
    except Exception:
        mean = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32).view(1, 3, 1, 1)

    return model, (mean, std), device


_STAGING = {"inp": None, "rgb_small": None}


def depth_from_frame_fast(model, norm, device: str,
                          frame_bgr: np.ndarray,
                          inference_size: Tuple[int, int]) -> np.ndarray:
    iw, ih = inference_size
    h, w = frame_bgr.shape[:2]

    small = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    t_inp = _STAGING["inp"]
    if t_inp is None or tuple(t_inp.shape[-2:]) != (ih, iw):
        t_inp = torch.empty((1, 3, ih, iw), device=device, dtype=torch.float32)
        _STAGING["inp"] = t_inp

    t_inp.copy_(
        torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(
            device=device, dtype=torch.float32
        ),
        non_blocking=True,
    )
    mean, std = norm
    t_inp = (t_inp / 255.0 - mean) / std

    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=getattr(next(model.parameters()), "dtype", torch.float16)
    ):
        out = model(pixel_values=t_inp).predicted_depth
        if out.ndim == 4:
            out = out[:, 0]
        pred = torch.nn.functional.interpolate(
            out.unsqueeze(1).float(),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        depth = pred[0]

    d = depth.flatten()
    lo = torch.quantile(d, 0.01)
    hi = torch.quantile(d, 0.99)
    depth01 = torch.clamp((depth - lo) / (hi - lo + 1e-6), 0, 1)

    return depth01.detach().cpu().numpy().astype(np.float32)

# -------------------- Utilities -------------------- #
def sbs_pack_gpu_rgb(left_t: torch.Tensor, right_t: torch.Tensor) -> np.ndarray:
    sbs = torch.cat([left_t, right_t], dim=2)
    sbs8 = (sbs.mul(255).clamp(0, 255).byte()).permute(1, 2, 0).contiguous()
    return cv2.cvtColor(sbs8.cpu().numpy(), cv2.COLOR_RGB2BGR)


def sbs_pack_gpu_bgr(left_t: torch.Tensor, right_t: torch.Tensor) -> np.ndarray:
    sbs = torch.cat([left_t, right_t], dim=2)
    sbs8 = (sbs.mul(255).clamp(0, 255).byte()).permute(1, 2, 0).contiguous()
    return sbs8.cpu().numpy()


class MJPEGStreamer:
    def __init__(self, bind_host: str, bind_port: int):
        self.app = Flask("vd3d_mjpeg")
        self._lock = Lock()
        self._jpeg = None

        @self.app.route("/video.mjpg")
        def video():
            def gen():
                boundary = b"--frame\r\n"
                while True:
                    with self._lock:
                        buf = self._jpeg
                    if buf is not None:
                        yield boundary
                        yield b"Content-Type: image/jpeg\r\n"
                        yield b"Cache-Control: no-cache\r\n\r\n"
                        yield buf + b"\r\n"
                    else:
                        yield b""
            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        self._host = bind_host
        self._port = bind_port
        self._thread = Thread(
            target=lambda: self.app.run(
                host=self._host, port=self._port, threaded=True, use_reloader=False
            ),
            daemon=True,
        )

    def start(self):
        self._thread.start()

    def push_bgr(self, frame_bgr):
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self._lock:
                self._jpeg = jpg.tobytes()

# -------------------- Main runtime -------------------- #
def run_live(args, external_stop: threading.Event | None = None):
    print("[run_live] Received args:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    diag = bool(getattr(args, "diag", False))

    try:
        cap = open_capture(args)
    except Exception as e:
        print(f"[run_live] ERROR opening capture: {e}")
        raise

    frame_q, stop_cap = start_latest_capture(cap)

    if diag:
        print("[diag] args:", vars(args))

    # Wait briefly for first frame
    first = None
    t0 = time.time()
    while time.time() - t0 < 2.0:
        if frame_q:
            first = frame_q[-1]
            break
        time.sleep(0.01)
    if first is None or first.size == 0:
        stop_cap.set()
        try:
            cap.release()
        except Exception:
            pass
        raise RuntimeError("No frames arriving from capture")

    print(f"🔥 CUDA available: {CUDA_AVAILABLE} | Using {'cuda' if CUDA_AVAILABLE else 'cpu'}")

    model, proc, device = make_da2(args.model, args.fp16)

    win = "VD3D Live"
    source_is_screen = isinstance(getattr(args, "source", "device"), str) and str(
        args.source
    ).lower().startswith("screen")
    show_preview = not getattr(args, "no_preview", False)

    if source_is_screen and not getattr(args, "force_preview", False) and not getattr(
        args, "mask_preview", False
    ):
        print(
            "🔇 Preview disabled to avoid screen-capture feedback. "
            "Use mask preview or move the window to another monitor."
        )
        show_preview = False

    if show_preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            win, getattr(args, "preview_w", 960), getattr(args, "preview_h", 540)
        )
        cv2.moveWindow(
            win, getattr(args, "preview_x", 60), getattr(args, "preview_y", 60)
        )

    fullscreen = False
    view_mode = 2 if args.sbs else 0

    fps_ema = None
    t_last = time.time()

    depth01 = None
    depth_last_t = 0.0
    depth_period = 1.0 / max(1e-3, args.depth_fps)

    ema_alpha = float(args.ema)
    depth_ema = None

    streamer = None
    if getattr(args, "http_stream", None):
        try:
            host, port = args.http_stream.split(":")
            streamer = MJPEGStreamer(host, int(port))
            streamer.start()
            print(f"🌐 MJPEG streaming at http://{host}:{port}/video.mjpg")
        except Exception as e:
            print(f"⚠️ Failed to start MJPEG server: {e}")
            streamer = None

    vcam = None
    audio_proc = None
    if getattr(args, "audio_device", None):
        d = max(0, int(getattr(args, "audio_delay_ms", 0)))
        adelay = f"{d}|{d}"
        ff_cmd = [
            "ffplay",
            "-loglevel", "error",
            "-f", "dshow",
            "-i", f"audio={args.audio_device}",
            "-nodisp",
            "-af", f"adelay={adelay}",
        ]
        try:
            audio_proc = subprocess.Popen(ff_cmd)
            if diag:
                print(f"[diag] audio monitor started: {' '.join(ff_cmd)}")
        except Exception as e:
            print(f"⚠️ Could not start audio monitor: {e}")


    print("▶️  Streaming… (f=fullscreen, m=mode, q=quit)")
    out_bgr = first

    while True:
        if external_stop is not None and external_stop.is_set():
            break

        try:
            frame = frame_q[-1]
        except IndexError:
            time.sleep(0.002)
            continue

        fourcc_req = (getattr(args, "fourcc", "") or "").upper()
        if frame.ndim == 3 and frame.shape[2] == 2 and fourcc_req == "YUY2":
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)

        if getattr(args, "force_bgr_swap", False):
            frame = frame[..., ::-1].copy()
        elif not getattr(args, "no_capture_swap", False):
            if frame.ndim == 3 and frame.shape[2] == 3:
                g_mean = float(frame[..., 1].mean())
                rb_mean = 0.5 * (
                    float(frame[..., 0].mean()) + float(frame[..., 2].mean())
                )
                if rb_mean > 0 and g_mean < 0.35 * rb_mean:
                    frame = frame[..., ::-1].copy()

        if frame is None or frame.size == 0:
            continue

        # Mask preview rectangle in screen-capture mode if requested
        if source_is_screen and getattr(args, "mask_preview", False) and hasattr(
            cap, "box"
        ):
            rel_x = getattr(args, "preview_x", 60) - cap.box["left"]
            rel_y = getattr(args, "preview_y", 60) - cap.box["top"]
            rw, rh = int(getattr(args, "preview_w", 960)), int(
                getattr(args, "preview_h", 540)
            )
            H, W = frame.shape[:2]
            x0, y0 = max(0, rel_x), max(0, rel_y)
            x1, y1 = min(W, rel_x + rw), min(H, rel_y + rh)
            if x1 > x0 and y1 > y0:
                frame[y0:y1, x0:x1] = 0

        now = time.time()

        # Depth update
        if (depth01 is None) or (now - depth_last_t >= depth_period):
            depth_new = depth_from_frame_fast(
                model, proc, device, frame, (args.infer_w, args.infer_h)
            )
            depth_last_t = now
            if args.smooth:
                if depth_ema is None:
                    depth_ema = depth_new
                else:
                    depth_ema = (1.0 - ema_alpha) * depth_ema + ema_alpha * depth_new
                depth01 = cv2.medianBlur(
                    (depth_ema * 255).astype(np.uint8), 3
                ).astype(np.float32) / 255.0
            else:
                depth01 = depth_new

        # View modes
        if view_mode == 0:
            out_bgr = frame
        elif view_mode == 1:
            d8 = (depth01 * 255.0).astype(np.uint8)
            out_bgr = cv2.applyColorMap(d8, cv2.COLORMAP_VIRIDIS)
        else:
            if HAVE_PIXEL_SHIFT and pixel_shift_cuda is not None and CUDA_AVAILABLE:
                if getattr(args, "pixelshift_rgb", False):
                    base = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    base = frame

                frm_t = torch.from_numpy(base).permute(2, 0, 1).to(
                    "cuda", dtype=torch.float32
                ).div_(255.0)
                d_t = torch.from_numpy(depth01).to(
                    "cuda", dtype=torch.float32
                ).unsqueeze(0)

                h, w = frame.shape[:2]
                left, right = pixel_shift_cuda(
                    frm_t,
                    d_t,
                    w,
                    h,
                    args.fg_shift,
                    args.mg_shift,
                    args.bg_shift,
                    blur_ksize=9,
                    feather_strength=12.0,
                    return_shift_map=False,
                    enable_feathering=True,
                    enable_edge_masking=True,
                )

                if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                    if left.device.type != "cuda":
                        left = left.to("cuda")
                    if right.device.type != "cuda":
                        right = right.to("cuda")

                    if left.dim() == 3 and left.shape[0] != 3:
                        left = left.permute(2, 0, 1)
                        right = right.permute(2, 0, 1)

                    if getattr(args, "pixelshift_rgb", False):
                        out_bgr = sbs_pack_gpu_rgb(left, right)
                    else:
                        out_bgr = sbs_pack_gpu_bgr(left, right)
                else:
                    if left.dtype != np.uint8:
                        left_u8 = np.clip(left * 255.0, 0, 255).astype(np.uint8)
                        right_u8 = np.clip(right * 255.0, 0, 255).astype(np.uint8)
                    else:
                        left_u8, right_u8 = left, right

                    if left_u8.ndim == 3 and left_u8.shape[0] == 3:
                        left_u8 = np.transpose(left_u8, (1, 2, 0))
                        right_u8 = np.transpose(right_u8, (1, 2, 0))

                    if getattr(args, "pixelshift_rgb", False):
                        left_bgr = cv2.cvtColor(left_u8, cv2.COLOR_RGB2BGR)
                        right_bgr = cv2.cvtColor(right_u8, cv2.COLOR_RGB2BGR)
                    else:
                        left_bgr, right_bgr = left_u8, right_u8

                    out_bgr = np.hstack([left_bgr, right_bgr])
            else:
                out_bgr = np.hstack([frame, frame])

        # Virtual cam
        if vcam is None and args.virtualcam and HAVE_VCAM:
            out_h, out_w = out_bgr.shape[:2]
            vcam = pyvirtualcam.Camera(
                width=out_w,
                height=out_h,
                fps=args.vcam_fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            print(f"📡 Virtual camera started: {vcam.device}")

        # FPS overlay
        dt = now - t_last
        t_last = now
        inst = (1.0 / dt) if dt > 0 else 0.0
        fps_ema = inst if fps_ema is None else (0.9 * fps_ema + 0.1 * inst)
        cv2.putText(
            out_bgr,
            f"{fps_ema:.1f} FPS",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            3,
        )
        cv2.putText(
            out_bgr,
            f"{fps_ema:.1f} FPS",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
        )

        if show_preview:
            cv2.imshow(win, out_bgr)
        if vcam is not None:
            vcam.send(out_bgr)
            vcam.sleep_until_next_frame()

        key = (cv2.waitKey(1) & 0xFF) if show_preview else 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("f") and show_preview:
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, prop)
        elif key == ord("m"):
            view_mode = (view_mode + 1) % 3
        elif key == ord("c"):
            out_bgr = out_bgr[..., ::-1].copy()

    # Flush one last frame to MJPEG if requested
    if streamer is not None:
        streamer.push_bgr(out_bgr)

    if audio_proc is not None:
        try:
            audio_proc.terminate()
        except Exception:
            pass
    if vcam is not None:
        vcam.close()
    stop_cap.set()
    try:
        cap.release()
    except Exception:
        pass
    if show_preview:
        cv2.destroyAllWindows()

# -------------------- Live GUI -------------------- #

class LiveGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("VD3D Live – GUI")
        self.master.geometry("475x725")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        self.worker_thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None

        self._build_vars()
        self._build_ui()

    # ---------- Tk Variables ---------- #
    def _build_vars(self):
        # Capture
        default_backend = "msmf" if os.name == "nt" else "any"
        self.source_var = tk.StringVar(value="device")           # "device" or "screen:1"
        self.device_index_var = tk.IntVar(value=0)
        self.backend_var = tk.StringVar(value=default_backend)
        self.capture_fps_var = tk.IntVar(value=60)
        self.width_var = tk.IntVar(value=0)                      # 0 = auto
        self.height_var = tk.IntVar(value=0)                     # 0 = auto
        self.cam_fps_var = tk.IntVar(value=0)                    # 0 = no explicit FPS
        self.fourcc_var = tk.StringVar(value="")
        self.no_capture_swap_var = tk.BooleanVar(value=False)
        self.force_bgr_swap_var = tk.BooleanVar(value=False)

        # Depth / model
        self.model_var = tk.StringVar(
            value="depth-anything/Depth-Anything-V2-Small-hf"
        )
        self.fp16_var = tk.BooleanVar(value=CUDA_AVAILABLE)
        self.infer_w_var = tk.IntVar(value=448)
        self.infer_h_var = tk.IntVar(value=256)
        self.depth_fps_var = tk.DoubleVar(value=8.0)
        self.smooth_var = tk.BooleanVar(value=True)
        self.ema_var = tk.DoubleVar(value=0.4)

        # 3D / Pixel-shift
        self.sbs_var = tk.BooleanVar(value=True)
        self.fg_shift_var = tk.DoubleVar(value=7.0)
        self.mg_shift_var = tk.DoubleVar(value=3.0)
        self.bg_shift_var = tk.DoubleVar(value=-5.0)
        self.pixelshift_rgb_var = tk.BooleanVar(value=False)

        # Preview / window
        self.preview_var = tk.BooleanVar(value=True)
        self.force_preview_var = tk.BooleanVar(value=False)
        self.mask_preview_var = tk.BooleanVar(value=False)
        self.preview_x_var = tk.IntVar(value=60)
        self.preview_y_var = tk.IntVar(value=60)
        self.preview_w_var = tk.IntVar(value=960)
        self.preview_h_var = tk.IntVar(value=540)

        # Output
        self.http_stream_var = tk.StringVar(value="")            # e.g. "127.0.0.1:8080"
        self.audio_device_var = tk.StringVar(value="")
        self.audio_delay_var = tk.IntVar(value=0)
        self.virtualcam_var = tk.BooleanVar(value=False)
        self.vcam_fps_var = tk.IntVar(value=30)

        # Misc
        self.diag_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Idle")

    # ---------- UI Layout ---------- #
    def _build_ui(self):
        main = ttk.Frame(self.master)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Capture frame
        cap_frame = ttk.LabelFrame(main, text="Capture")
        cap_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(cap_frame, text="Source").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            cap_frame,
            textvariable=self.source_var,
            values=["device", "screen:1", "screen:2", "screen:0"],
            width=12,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=3, pady=2)

        ttk.Label(cap_frame, text="Device index").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            cap_frame, from_=0, to=16, textvariable=self.device_index_var, width=5
        ).grid(row=0, column=3, sticky="w", padx=3, pady=2)

        ttk.Label(cap_frame, text="Backend").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            cap_frame,
            textvariable=self.backend_var,
            values=["any", "dshow", "msmf", "ffmpeg"],
            width=12,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=3, pady=2)

        ttk.Label(cap_frame, text="Capture FPS").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(
            cap_frame, from_=1, to=240, textvariable=self.capture_fps_var, width=5
        ).grid(row=1, column=3, sticky="w", padx=3, pady=2)

        ttk.Label(cap_frame, text="Width").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            cap_frame, from_=0, to=7680, textvariable=self.width_var, width=7
        ).grid(row=2, column=1, sticky="w", padx=3, pady=2)

        ttk.Label(cap_frame, text="Height").grid(row=2, column=2, sticky="w")
        ttk.Spinbox(
            cap_frame, from_=0, to=4320, textvariable=self.height_var, width=7
        ).grid(row=2, column=3, sticky="w", padx=3, pady=2)

        ttk.Label(cap_frame, text="Camera FPS").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(
            cap_frame, from_=0, to=240, textvariable=self.cam_fps_var, width=7
        ).grid(row=3, column=1, sticky="w", padx=3, pady=2)

        ttk.Label(cap_frame, text="FOURCC").grid(row=3, column=2, sticky="w")
        ttk.Entry(cap_frame, textvariable=self.fourcc_var, width=8).grid(
            row=3, column=3, sticky="w", padx=3, pady=2
        )

        ttk.Checkbutton(
            cap_frame,
            text="Force BGR swap",
            variable=self.force_bgr_swap_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Checkbutton(
            cap_frame,
            text="Disable auto swap",
            variable=self.no_capture_swap_var,
        ).grid(row=4, column=2, columnspan=2, sticky="w", pady=2)

        # Depth / model frame
        depth_frame = ttk.LabelFrame(main, text="Depth / Model")
        depth_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(depth_frame, text="Model ID").grid(row=0, column=0, sticky="w")
        ttk.Entry(depth_frame, textvariable=self.model_var, width=50).grid(
            row=0, column=1, columnspan=3, sticky="we", padx=3, pady=2
        )

        ttk.Checkbutton(
            depth_frame, text="Use FP16 (if CUDA)", variable=self.fp16_var
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(depth_frame, text="Infer W").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            depth_frame, from_=128, to=2048, textvariable=self.infer_w_var, width=7
        ).grid(row=2, column=1, sticky="w", padx=3, pady=2)

        ttk.Label(depth_frame, text="Infer H").grid(row=2, column=2, sticky="w")
        ttk.Spinbox(
            depth_frame, from_=64, to=2048, textvariable=self.infer_h_var, width=7
        ).grid(row=2, column=3, sticky="w", padx=3, pady=2)

        ttk.Label(depth_frame, text="Depth FPS").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(
            depth_frame, from_=1, to=60, increment=1,
            textvariable=self.depth_fps_var,
            width=7,
        ).grid(row=3, column=1, sticky="w", padx=3, pady=2)

        ttk.Checkbutton(
            depth_frame, text="Smooth (EMA + median)", variable=self.smooth_var
        ).grid(row=3, column=2, sticky="w", pady=2)

        ttk.Label(depth_frame, text="EMA α").grid(row=4, column=0, sticky="w")
        ttk.Spinbox(
            depth_frame, from_=0.05, to=1.0, increment=0.05,
            textvariable=self.ema_var,
            width=7,
        ).grid(row=4, column=1, sticky="w", padx=3, pady=2)

        # 3D / pixel shift frame
        ps_frame = ttk.LabelFrame(main, text="3D / Pixel Shift")
        ps_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Checkbutton(
            ps_frame, text="Enable SBS 3D", variable=self.sbs_var
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Checkbutton(
            ps_frame, text="PixelShift RGB input", variable=self.pixelshift_rgb_var
        ).grid(row=0, column=2, columnspan=2, sticky="w", pady=2)

        ttk.Label(ps_frame, text="FG shift").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(
            ps_frame, from_=-100, to=100, increment=1,
            textvariable=self.fg_shift_var,
            width=7,
        ).grid(row=1, column=1, sticky="w", padx=3, pady=2)

        ttk.Label(ps_frame, text="MG shift").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(
            ps_frame, from_=-100, to=100, increment=1,
            textvariable=self.mg_shift_var,
            width=7,
        ).grid(row=1, column=3, sticky="w", padx=3, pady=2)

        ttk.Label(ps_frame, text="BG shift").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            ps_frame, from_=-100, to=100, increment=1,
            textvariable=self.bg_shift_var,
            width=7,
        ).grid(row=2, column=1, sticky="w", padx=3, pady=2)

        # Preview / output frame
        out_frame = ttk.LabelFrame(main, text="Preview / Output")
        out_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Checkbutton(
            out_frame, text="Show preview window", variable=self.preview_var
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Checkbutton(
            out_frame, text="Force preview (screen src)", variable=self.force_preview_var
        ).grid(row=0, column=2, columnspan=2, sticky="w", pady=2)

        ttk.Checkbutton(
            out_frame, text="Mask preview region in screen capture",
            variable=self.mask_preview_var,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=2)

        ttk.Label(out_frame, text="Preview X").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            out_frame, from_=-3000, to=3000, textvariable=self.preview_x_var, width=7
        ).grid(row=2, column=1, sticky="w", padx=3, pady=2)

        ttk.Label(out_frame, text="Preview Y").grid(row=2, column=2, sticky="w")
        ttk.Spinbox(
            out_frame, from_=-3000, to=3000, textvariable=self.preview_y_var, width=7
        ).grid(row=2, column=3, sticky="w", padx=3, pady=2)

        ttk.Label(out_frame, text="Preview W").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(
            out_frame, from_=100, to=4096, textvariable=self.preview_w_var, width=7
        ).grid(row=3, column=1, sticky="w", padx=3, pady=2)

        ttk.Label(out_frame, text="Preview H").grid(row=3, column=2, sticky="w")
        ttk.Spinbox(
            out_frame, from_=100, to=4096, textvariable=self.preview_h_var, width=7
        ).grid(row=3, column=3, sticky="w", padx=3, pady=2)

        ttk.Label(out_frame, text="HTTP stream (host:port)").grid(
            row=4, column=0, sticky="w"
        )
        ttk.Entry(out_frame, textvariable=self.http_stream_var, width=18).grid(
            row=4, column=1, sticky="w", padx=3, pady=2
        )

        ttk.Label(out_frame, text="Audio device").grid(row=5, column=0, sticky="w")
        ttk.Entry(out_frame, textvariable=self.audio_device_var, width=24).grid(
            row=5, column=1, sticky="w", padx=3, pady=2
        )

        ttk.Label(out_frame, text="Audio delay ms").grid(
            row=5, column=2, sticky="w"
        )
        ttk.Spinbox(
            out_frame, from_=0, to=5000, textvariable=self.audio_delay_var, width=7
        ).grid(row=5, column=3, sticky="w", padx=3, pady=2)

        ttk.Checkbutton(
            out_frame, text="Virtual camera", variable=self.virtualcam_var
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(out_frame, text="VCam FPS").grid(row=6, column=2, sticky="w")
        ttk.Spinbox(
            out_frame, from_=1, to=120, textvariable=self.vcam_fps_var, width=7
        ).grid(row=6, column=3, sticky="w", padx=3, pady=2)

        ttk.Checkbutton(
            out_frame, text="Diagnostic logging", variable=self.diag_var
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=2)

        # Bottom controls
        bottom = ttk.Frame(main)
        bottom.grid(row=4, column=0, sticky="ew", padx=5, pady=(10, 0))
        bottom.columnconfigure(1, weight=1)

        self.status_label = ttk.Label(bottom, textvariable=self.status_var)
        self.status_label.grid(row=0, column=0, sticky="w")

        btn_frame = ttk.Frame(bottom)
        btn_frame.grid(row=0, column=1, sticky="e")

        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.start_live)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = ttk.Button(
            btn_frame, text="Stop", command=self.stop_live, state="disabled"
        )
        self.stop_btn.grid(row=0, column=1, padx=5)

        ttk.Button(btn_frame, text="Close", command=self.on_close).grid(
            row=0, column=2, padx=5
        )

    # ---------- Arg building ---------- #
    def _build_args(self) -> Namespace:
        # Convert IntVars where 0 = None for optional properties
        width = self.width_var.get()
        width = width if width > 0 else None
        height = self.height_var.get()
        height = height if height > 0 else None
        cam_fps = self.cam_fps_var.get()
        cam_fps = cam_fps if cam_fps > 0 else None
        http_stream = self.http_stream_var.get().strip() or None
        audio_device = self.audio_device_var.get().strip() or None

        args_dict = {
            # Capture / source
            "source": self.source_var.get(),
            "device_index": int(self.device_index_var.get()),
            "backend": self.backend_var.get(),
            "capture_fps": int(self.capture_fps_var.get()),
            "width": width,
            "height": height,
            "fps": cam_fps,
            "fourcc": self.fourcc_var.get().strip(),
            "force_bgr_swap": bool(self.force_bgr_swap_var.get()),
            "no_capture_swap": bool(self.no_capture_swap_var.get()),
            "crop": None,  # hook for future GUI crop controls

            # Depth / model
            "model": self.model_var.get().strip(),
            "fp16": bool(self.fp16_var.get()),
            "infer_w": int(self.infer_w_var.get()),
            "infer_h": int(self.infer_h_var.get()),
            "depth_fps": float(self.depth_fps_var.get()),
            "smooth": bool(self.smooth_var.get()),
            "ema": float(self.ema_var.get()),

            # 3D / pixel shift
            "sbs": bool(self.sbs_var.get()),
            "fg_shift": float(self.fg_shift_var.get()),
            "mg_shift": float(self.mg_shift_var.get()),
            "bg_shift": float(self.bg_shift_var.get()),
            "pixelshift_rgb": bool(self.pixelshift_rgb_var.get()),

            # Preview / window
            "no_preview": not bool(self.preview_var.get()),
            "force_preview": bool(self.force_preview_var.get()),
            "mask_preview": bool(self.mask_preview_var.get()),
            "preview_x": int(self.preview_x_var.get()),
            "preview_y": int(self.preview_y_var.get()),
            "preview_w": int(self.preview_w_var.get()),
            "preview_h": int(self.preview_h_var.get()),

            # Output / streaming / audio / vcam
            "http_stream": http_stream,
            "audio_device": audio_device,
            "audio_delay_ms": int(self.audio_delay_var.get()),
            "virtualcam": bool(self.virtualcam_var.get()),
            "vcam_fps": int(self.vcam_fps_var.get()),

            # Misc
            "diag": bool(self.diag_var.get()),
        }

        return Namespace(**args_dict)

    # ---------- Control handlers ---------- #
    def start_live(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return  # already running

        try:
            args = self._build_args()
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to build arguments:\n{e}")
            return

        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._worker_run,
            args=(args,),
            daemon=True,
        )
        self.worker_thread.start()
        self.status_var.set("Running…")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._poll_worker()

    def _worker_run(self, args: Namespace):
        import traceback

        try:
            run_live(args, external_stop=self.stop_event)
        except Exception as e:
            traceback.print_exc()
            # capture the message BEFORE leaving `except` so the closure can use it
            msg = str(e)

            def _show_err(msg=msg):
                messagebox.showerror("VD3D Live error", msg)

            try:
                self.master.after(0, _show_err)
            except Exception:
                pass


    def _poll_worker(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.master.after(500, self._poll_worker)
        else:
            self.worker_thread = None
            self.stop_event = None
            self.status_var.set("Idle")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

    def stop_live(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_var.set("Stopping…")

    def on_close(self):
        self.stop_live()
        # Give worker a moment to exit
        self.master.after(300, self.master.destroy)


def launch_live_gui():
    """
    Entry point for opening the live GUI.

    If there is already a Tk root (e.g. from your main VisionDepth3D UI),
    this will open a Toplevel. Otherwise it will create a root and start
    its own mainloop.
    """
    root = tk._default_root
    if root is None:
        root = tk.Tk()
        LiveGUI(root)
        root.mainloop()
    else:
        win = tk.Toplevel(root)
        LiveGUI(win)
