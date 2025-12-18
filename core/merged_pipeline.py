import os
import re
import sys
import time
import threading
import numpy as np
import cv2
import onnxruntime as ort
from tkinter import messagebox, filedialog
from tqdm import tqdm
import subprocess
from queue import Queue
from tkinter.simpledialog import askstring
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



suspend_flag = threading.Event()
cancel_flag = threading.Event()
progress_bar = None
status_label = None

# ✅ Get absolute path to resource (for PyInstaller compatibility)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS2  # ✅ Corrected for PyInstaller
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# ✅ ONNX session options with graph optimization
session_options = ort.SessionOptions()
session_options.log_severity_level = 3
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = max(1, os.cpu_count() // 2)
session_options.inter_op_num_threads = 1

# ✅ ONNX Execution Provider fallback logic
available_providers = ort.get_available_providers()

if "CUDAExecutionProvider" in available_providers:
    device = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    device = ["CPUExecutionProvider"]

provider_txt = "CUDA" if "CUDAExecutionProvider" in device else "CPU-only"
print(f"Frametool Upscaler ONNX: {provider_txt}")


# ✅ Load RIFE
rife_path = resource_path(os.path.join("weights", "RIFE_fp32.onnx"))

try:
    rife_session = ort.InferenceSession(rife_path, sess_options=session_options, providers=device)
    print("FPS Upscale Model: RIFE model loaded.")
except Exception as e:
    print(f"❌ Failed to load RIFE model: {e}")
    rife_session = None

esrgan_session = None  # ONNX ESRGAN / other ONNX SR
srresnet_model = None  # PyTorch SRResNet
srresnet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# which backend is currently active: "onnx", "srresnet", or "none"
UPSCALE_BACKEND = "none"

# =========================
#  SRResNet Upscaler (PyTorch, .pth)
# =========================

class SRResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return residual + out


class SRResNet(nn.Module):
    def __init__(self, num_blocks: int = 16, upscale_factor: int = 4):
        super().__init__()

        self.upscale_factor = upscale_factor

        # Initial feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.res_blocks = nn.Sequential(*[SRResBlock(64) for _ in range(num_blocks)])

        # Conv after residuals
        self.conv_res = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_res = nn.BatchNorm2d(64)

        # Upsampling (PixelShuffle)
        up_layers = []
        for _ in range(int(math.log2(upscale_factor))):
            up_layers += [
                nn.Conv2d(64, 256, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ]
        self.upsample = nn.Sequential(*up_layers)

        # Final output
        self.conv_out = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        residual = x
        x = self.res_blocks(x)
        x = self.bn_res(self.conv_res(x))
        x = x + residual
        x = self.upsample(x)
        x = self.conv_out(x)
        return x


def normalize_frame(img, target_size=None):
    """
    Return uint8 BGR frame.
    If target_size is given, resize to (w,h).
    Handles None, grayscale, BGRA, float.
    """
    if img is None:
        raise ValueError("normalize_frame: got None image")

    # Channels
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Type/range
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.astype(np.uint8)

    if target_size is not None:
        w, h = target_size
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    return img

def ui_set_status(widget, text):
    try:
        widget.after(0, lambda: widget.configure(text=text))
    except Exception:
        pass

def ui_set_progress(progressbar, value, maximum=100.0):
    def _apply():
        try:
            # ok for ttk.Progressbar (and Scale ignores unknown kwargs)
            try:
                progressbar.configure(mode="determinate", maximum=maximum)
            except Exception:
                pass
            progressbar.configure(value=value)
        except Exception:
            pass
    try:
        progressbar.after(0, _apply)
    except Exception:
        pass

# put this near your other globals
_last_ui_push = {"t": 0.0}

def update_progress(done, total, start):
    global _last_ui_push
    if not progress_bar or not status_label:
        return

    # math
    now = time.monotonic()
    elapsed = max(1e-6, time.time() - start)
    fps = done / elapsed
    remaining = max(0, total - done)
    eta_secs = (remaining / fps) if fps > 0 else None
    pct = 0.0 if not total else (done / total) * 100.0
    pct = max(0.0, min(100.0, pct))  # clamp

    # debounce UI updates: 50 ms, always allow the final tick
    if pct < 100.0 and (now - _last_ui_push["t"] < 0.05):
        return
    _last_ui_push["t"] = now

    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_secs)) if eta_secs is not None else "--:--"

    def _apply():
        try:
            # ttk.Progressbar supports maximum/mode/value
            try:
                progress_bar.configure(mode="determinate", maximum=100.0)
            except Exception:
                # if it's not ttk.Progressbar (e.g., a Scale), ignore
                pass

            # both ttk.Progressbar and Scale understand "value"
            progress_bar.configure(value=pct)

            status_label.configure(
                text=f"Progress: {done}/{total} | FPS: {fps:.2f} | ETA: {eta_str}"
            )
        except Exception:
            # widget might be destroyed during shutdown, ignore
            pass

    # marshal to Tk main thread
    try:
        progress_bar.after(0, _apply)
    except Exception:
        pass

def _validate_frame_bytes(frame, width, height):
    if frame is None:
        return False, "frame=None"
    if not isinstance(frame, np.ndarray):
        return False, f"type={type(frame)}"
    if frame.dtype != np.uint8:
        return False, f"dtype={frame.dtype}"
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False, f"shape={frame.shape}"
    if frame.shape[1] != width or frame.shape[0] != height:
        return False, f"size={frame.shape[1]}x{frame.shape[0]} expected={width}x{height}"
    return True, "ok"

def _frame_to_bytes(frame):
    # ensure contiguous BGR24 for rawvideo
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    return frame.tobytes()

def _frame_loader(file_list, target_size=None):
    q = Queue(maxsize=8)
    stop = object()

    def _worker():
        for fp in file_list:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if target_size:
                img = cv2.resize(
                    img,
                    target_size,
                    interpolation=cv2.INTER_AREA if img.shape[1] > target_size[0] else cv2.INTER_CUBIC
                )
            q.put(img)
        q.put(stop)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        item = q.get()
        if item is stop:
            break
        yield item

def select_video_and_generate_frames(set_folder_callback=None, merged_progress=None, merged_status=None):
    video_path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")]
    )
    if not video_path:
        return

    output_root = filedialog.askdirectory(title="Select Folder to Save Extracted Frames")
    if not output_root:
        return

    image_format = askstring("Image Format", "Enter image format to save (e.g., png, jpg):")
    valid_formats = ["png", "jpg", "jpeg", "bmp", "webp"]
    if not image_format or image_format.lower() not in valid_formats:
        messagebox.showerror("Invalid Format", "Please enter a valid format like png, jpg, etc.")
        return
    image_format = image_format.lower()

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_root, f"{base_name}_frames")
    os.makedirs(output_folder, exist_ok=True)

    output_pattern = os.path.join(output_folder, f"frame_%05d.{image_format}")

    def extract_thread():
        def start_spinner():
            if merged_progress and merged_status:
                merged_progress.config(mode="indeterminate")
                merged_progress.start()
                merged_status.config(text="⏳ Extracting frames...")

        def stop_spinner(success):
            if merged_progress and merged_status:
                merged_progress.stop()
                merged_progress.config(mode="determinate")
                if success:
                    merged_status.config(text="✅ Extraction complete.")
                    messagebox.showinfo("Done", f"✅ Frames saved to:\n{output_folder}")
                    if set_folder_callback:
                        set_folder_callback(output_folder)
                else:
                    merged_status.config(text="❌ Extraction failed.")
                    messagebox.showerror("Error", "❌ FFmpeg frame extraction failed.")

        if merged_progress:
            merged_progress.after(0, start_spinner)

        print(f"🚀 Running FFmpeg to extract frames from: {video_path}")
        print(f"📁 Saving to: {output_folder}")

        command = [
            "ffmpeg", "-y",
            "-hwaccel", "auto",
            "-i", video_path,
            "-q:v", "2",
            output_pattern
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if merged_progress:
            merged_progress.after(0, lambda: stop_spinner(result.returncode == 0))


    threading.Thread(target=extract_thread, daemon=True).start()


def select_output_file(output_path_var):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".avi",
        filetypes=[("AVI Files", "*.avi"), ("MP4 Files", "*.mp4"), ("All Files", "*.*")]
    )
    if file_path:
        output_path_var.set(file_path)

def select_frames_folder(path_var):
    folder = filedialog.askdirectory()
    if folder:
        path_var.set(folder)

def extract_frame_number(filename):
    match = re.search(r"(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else float("inf")

def natural_sort(files):
    return sorted(files, key=extract_frame_number)

def concatenate_images(frame1, frame2):
    return np.concatenate((frame1.astype(np.float32) / 255.0, frame2.astype(np.float32) / 255.0), axis=2)

def preprocess_rife(frame):
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame.astype(np.float32)

def run_rife(frame1, frame2, multiplier):
    if not rife_session:
        return []

    merged = concatenate_images(frame1, frame2)
    tensor = preprocess_rife(merged)
    batch_tensor = np.repeat(tensor, repeats=multiplier - 1, axis=0)

    try:
        output = rife_session.run(None, {rife_session.get_inputs()[0].name: batch_tensor})[0]
        output = np.clip(output, 0, 1)
        output = np.transpose(output, (0, 2, 3, 1))
        return [(frame * 255).astype(np.uint8) for frame in output]
    except Exception as e:
        print(f"❌ RIFE inference error: {e}")
        return []

def preprocess_esr(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

def postprocess_esr(tensor):
    tensor = np.squeeze(tensor, axis=0)
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = np.clip(tensor, 0, 1) * 255.0
    return cv2.cvtColor(tensor.astype(np.uint8), cv2.COLOR_RGB2BGR)

def blend_images(original, upscaled, mode="OFF"):
    if mode == "OFF":
        return upscaled
    alpha_map = {"LOW": 0.85, "MEDIUM": 0.5, "HIGH": 0.25}
    alpha = alpha_map.get(mode.upper(), 1.0)
    return cv2.addWeighted(upscaled, alpha, original, 1 - alpha, 0)

def init_upscaler(model_path: str, enable_upscale: bool):
    """
    Decide which backend to use based on model_path extension:
      - .onnx -> ONNX / ESRGAN
      - .pth  -> PyTorch SRResNet super-res
    """
    global esrgan_session, srresnet_model, UPSCALE_BACKEND

    esrgan_session = None
    srresnet_model = None
    UPSCALE_BACKEND = "none"

    if not enable_upscale or not model_path:
        print("Upscaler disabled.")
        return

    # Make sure we resolve correctly inside PyInstaller bundle too
    model_path = resource_path(model_path)
    ext = os.path.splitext(model_path)[1].lower()

    if ext == ".onnx":
        try:
            print(f"📦 Loading ONNX upscaler from {model_path}")
            esrgan_session = ort.InferenceSession(
                model_path, sess_options=session_options, providers=device
            )
            UPSCALE_BACKEND = "onnx"
            mode_txt = "CUDA" if "CUDAExecutionProvider" in device else "CPU"
            print(f"⚡ ONNX upscaler ready [{mode_txt}]")
        except Exception as e:
            UPSCALE_BACKEND = "none"
            print(f"❌ Failed to load ONNX upscaler: {e}")

    elif ext == ".pth":
        try:
            print(f"📦 Loading SRResNet (.pth) upscaler from {model_path}")
            model = SRResNet(num_blocks=16, upscale_factor=4)
            state = torch.load(model_path, map_location=srresnet_device)
            model.load_state_dict(state)
            model.to(srresnet_device)
            model.eval()
            srresnet_model = model
            UPSCALE_BACKEND = "srresnet"
            mode_txt = "CUDA" if srresnet_device.type == "cuda" else "CPU"
            print(f"⚡ SRResNet upscaler ready [{mode_txt}]")
        except Exception as e:
            UPSCALE_BACKEND = "none"
            print(f"❌ Failed to load SRResNet model: {e}")

    else:
        print(f"❌ Unknown upscaler model extension: {ext}. Supported: .onnx, .pth")
        UPSCALE_BACKEND = "none"


def _run_srresnet(frame_bgr: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Run your trained SRResNet on a single BGR uint8 frame.
    Returns a BGR uint8 image (native 4x HR from the model).
    """
    global srresnet_model, srresnet_device
    if srresnet_model is None:
        return frame_bgr

    # BGR uint8 -> RGB [0,1] tensor
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))  # C,H,W
    tensor = torch.from_numpy(rgb).unsqueeze(0).to(srresnet_device)

    with torch.no_grad():
        sr = srresnet_model(tensor)

    sr = sr.clamp(0.0, 1.0).cpu().numpy()[0]
    sr = np.transpose(sr, (1, 2, 0))  # H,W,C
    sr = (sr * 255.0).round().astype(np.uint8)
    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
    return sr


def run_esrgan(frame,
               blend_mode="OFF",
               input_res_pct=100,
               model_name="RealESR_Gx4_fp16",
               target_size=None,
               tile=None,
               tile_pad=8):
    """
    Generic upscaler entrypoint.

    Backends:
      - ONNX ESRGAN (UPSCALE_BACKEND == "onnx")
      - PyTorch SRResNet (.pth) (UPSCALE_BACKEND == "srresnet")

    Always returns a BGR uint8 frame matching target_size if given.
    """
    global esrgan_session, srresnet_model, UPSCALE_BACKEND
    if frame is None:
        return frame

    # --- helpers ---
    def _to_bgr_uint8(img):
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255)
            if img.max() <= 1.0:
                img = img * 255.0
            img = img.astype(np.uint8)
        return img

    def _fit_size(img, wh):
        if img is None:
            return None
        w, h = wh
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        return img

    frame = _to_bgr_uint8(frame)
    original = frame.copy()

    # No upscaler loaded
    backend_has_model = (
        (UPSCALE_BACKEND == "onnx" and esrgan_session is not None) or
        (UPSCALE_BACKEND == "srresnet" and srresnet_model is not None)
    )
    if not backend_has_model:
        out = original
        if target_size:
            out = _fit_size(out, target_size)
        return out if blend_mode == "OFF" else blend_images(original, out, mode=blend_mode)

    # Optional pre-scale (applies to both backends)
    if input_res_pct != 100:
        h, w = frame.shape[:2]
        new_w = max(1, int(w * input_res_pct / 100))
        new_h = max(1, int(h * input_res_pct / 100))
        frame = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if input_res_pct < 100 else cv2.INTER_CUBIC,
        )

    # --- Actual backend inference ---
    if UPSCALE_BACKEND == "srresnet":
        # SRResNet path: use its native 4x output
        upscaled = _run_srresnet(frame, scale=4)

        upscaled = _to_bgr_uint8(upscaled)

        # If a specific output size was requested, resize there directly
        if target_size:
            upscaled = _fit_size(upscaled, target_size)
            original_for_blend = _fit_size(original, target_size)
        else:
            h_hr, w_hr = upscaled.shape[:2]
            original_for_blend = _fit_size(original, (w_hr, h_hr))

        return blend_images(original_for_blend, upscaled, mode=blend_mode)

    # ---- ONNX Real-ESRGAN branch ----
    if tile:
        upscaled = _esrgan_tiled(frame, tile, tile_pad)
    else:
        tensor = preprocess_esr(frame)
        try:
            output = esrgan_session.run(
                None, {esrgan_session.get_inputs()[0].name: tensor}
            )[0]
            upscaled = postprocess_esr(output)
        except Exception as e:
            print(f"❌ ESRGAN failed: {e}")
            out = original
            if target_size:
                out = _fit_size(out, target_size)
            return out if blend_mode == "OFF" else blend_images(original, out, mode=blend_mode)

    upscaled = _to_bgr_uint8(upscaled)

    # 2x vs 4x heuristic for ONNX models
    scale = 2 if "x2" in model_name.lower() else 4

    upscaled = cv2.resize(
        upscaled,
        (frame.shape[1] * scale, frame.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )

    upscaled = _fit_size(upscaled, (original.shape[1], original.shape[0]))

    if target_size:
        upscaled = _fit_size(upscaled, target_size)
        original_for_blend = _fit_size(original, target_size)
    else:
        original_for_blend = original

    return blend_images(original_for_blend, upscaled, mode=blend_mode)



def _esrgan_tiled(img, tile, pad):
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(h, y + tile + pad), min(w, x + tile + pad)
            crop = img[y0:y1, x0:x1]
            t = preprocess_esr(crop)
            pred = esrgan_session.run(None, {esrgan_session.get_inputs()[0].name: t})[0]
            up = postprocess_esr(pred)
            # place center region
            yc0, xc0 = y - y0, x - x0
            yc1, xc1 = yc0 + min(tile, h - y), xc0 + min(tile, w - x)
            out[y:y+min(tile, h - y), x:x+min(tile, w - x)] = up[yc0:yc1, xc0:xc1]
    return out

def start_merged_pipeline(settings, progress_widget, status_label_widget):
    global progress_bar, status_label, esrgan_session
    progress_bar = progress_widget
    status_label = status_label_widget

    frames_dir = settings["frames_folder"]
    output_path = settings["output_file"]
    codec = settings["codec"]
    width, height = settings["width"], settings["height"]
    fps = settings["fps"]
    fps_mult = settings["fps_multiplier"]
    enable_rife = settings["enable_rife"]
    enable_upscale = settings["enable_upscale"]
    blend_mode = settings.get("blend_mode", "OFF")
    input_res_pct = settings.get("input_res_pct", 100)
    model_path = settings.get("model_path", "weights/RealESR_Gx4_fp16.onnx")

    # Initialize upscaler (ONNX or SRResNet) based on model_path extension
    init_upscaler(model_path, enable_upscale)

    files = natural_sort([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not files:
        messagebox.showerror("Error", "No frames found in selected folder.")
        return

    output_fps = fps * fps_mult if enable_rife else fps
    video = start_ffmpeg_writer(output_path, width, height, output_fps, settings["codec"])
    start = time.time()

    target_size = (width, height)

    # 🔥 Important: keep native resolution for the SR model / RIFE
    file_iter = _frame_loader(files, None)


    prev = next(file_iter, None)
    if prev is None:
        messagebox.showerror("Error", "No readable frames.")
        try:
            video.stdin.close()
        except Exception:
            pass
        video.wait()
        return

    total_src = len(files)

    for i, curr in enumerate(file_iter, start=1):
        if cancel_flag.is_set():
            break

        if enable_rife:
            interpolated = run_rife(prev, curr, fps_mult)
            if enable_upscale:
                interpolated = [run_esrgan(f, blend_mode, input_res_pct, target_size=target_size) for f in interpolated]
                curr_proc = run_esrgan(curr, blend_mode, input_res_pct, target_size=target_size)
            else:
                interpolated = [cv2.resize(f, target_size) for f in interpolated]
                curr_proc = cv2.resize(curr, target_size)

            for f in interpolated:
                video.stdin.write(f.tobytes())
        else:
            curr_proc = run_esrgan(curr, blend_mode, input_res_pct, target_size=target_size) if enable_upscale else cv2.resize(curr, target_size)

        video.stdin.write(curr_proc.tobytes())
        prev = curr

        update_progress(i + 1, total_src, start)

    try:
        video.stdin.close()
    except Exception:
        pass
    video.wait()

    update_progress(total_src, total_src, start)
    try:
        status_label.after(0, lambda: status_label.configure(text="✅ Processing Complete!"))
    except Exception:
        pass


MAX_QUEUE_SIZE = 16

def start_threaded_pipeline(settings, progress_widget, status_label_widget):
    global progress_bar, status_label, esrgan_session
    progress_bar = progress_widget
    status_label = status_label_widget
    
    try:
        progress_bar.after(0, lambda: progress_bar.configure(mode="determinate",
                                                             maximum=100.0, value=0.0))
        status_label.after(0, lambda: status_label.configure(text="Preparing..."))
    except Exception:
        pass
    
    frames_dir   = settings["frames_folder"]
    output_path  = settings["output_file"]
    codec        = settings["codec"]
    width        = settings["width"]
    height       = settings["height"]
    fps          = settings["fps"]
    fps_mult     = settings["fps_multiplier"]
    enable_rife  = settings["enable_rife"]
    enable_up    = settings["enable_upscale"]
    blend_mode   = settings.get("blend_mode", "OFF")
    input_res_pct= settings.get("input_res_pct", 100)
    model_path   = settings.get("model_path", "weights/RealESR_Gx4_fp16.onnx")

    ui_set_status(status_label, "Preparing...")
    output_fps = fps * fps_mult if enable_rife else fps
    video = start_ffmpeg_writer(output_path, width, height, output_fps, codec)
    start = time.time()
    target_size = (width, height)

    # Initialize upscaler (ONNX or SRResNet) based on model_path extension
    init_upscaler(model_path, enable_up)

    # Sorted frame list
    files = natural_sort([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    total_src = len(files)
    if total_src < 2:
        ui_set_status(status_label, "Not enough frames to process.")
        return

    interpolation_queue = Queue(MAX_QUEUE_SIZE)
    upscaling_queue = Queue(MAX_QUEUE_SIZE)


    cancel_local = threading.Event()

    def rife_worker():
        try:
            prev = cv2.imread(files[0]);  prev = normalize_frame(prev, target_size)
            for idx in range(1, len(files)):
                if cancel_flag.is_set() or cancel_local.is_set():
                    break
                curr = cv2.imread(files[idx]);  curr = normalize_frame(curr, target_size)

                interpolated = run_rife(prev, curr, fps_mult) if enable_rife else []
                if interpolated:
                    interpolated = [normalize_frame(f, target_size) for f in interpolated]

                # enqueue the WHOLE segment: prev, interps, curr
                interpolation_queue.put((idx, prev, curr, interpolated))
                prev = curr

            # tail sentinel; prev is the last frame seen
            interpolation_queue.put(("END", prev, None, []))
        except Exception as e:
            print("rife_worker error:", e)
            interpolation_queue.put(("END", None, None, []))

    def esrgan_worker():
        try:
            processed = 0
            frame_id = 0
            while True:
                idx, prev, curr, interpolated = interpolation_queue.get()
                if idx == "END":
                    # write the very last prev (final frame) once
                    if prev is not None:
                        if (prev.shape[1], prev.shape[0]) != target_size:
                            prev = normalize_frame(prev, target_size)
                        final_prev = run_esrgan(prev, blend_mode, input_res_pct, target_size=target_size) if enable_up else prev
                        upscaling_queue.put((frame_id, final_prev)); frame_id += 1
                    upscaling_queue.put(("END", None))
                    break

                # last-ditch safety only — rife_worker already normalized everything
                if (prev.shape[1], prev.shape[0]) != target_size:
                    prev = normalize_frame(prev, target_size)
                if (curr.shape[1], curr.shape[0]) != target_size:
                    curr = normalize_frame(curr, target_size)
                if interpolated and any((f.shape[1], f.shape[0]) != target_size for f in interpolated):
                    interpolated = [normalize_frame(f, target_size) for f in interpolated]

                # upscale (if enabled)
                if enable_up:
                    prev_proc = run_esrgan(prev, blend_mode, input_res_pct, target_size=target_size)
                    inter_proc = [run_esrgan(f, blend_mode, input_res_pct, target_size=target_size) for f in interpolated]
                    curr_proc = run_esrgan(curr, blend_mode, input_res_pct, target_size=target_size)
                else:
                    prev_proc = prev
                    inter_proc = interpolated
                    curr_proc = curr

                # IMPORTANT: correct order
                upscaling_queue.put((frame_id, prev_proc)); frame_id += 1
                for f in inter_proc:
                    upscaling_queue.put((frame_id, f)); frame_id += 1
                upscaling_queue.put((frame_id, curr_proc)); frame_id += 1

                processed += 1
                update_progress(processed, total_src, start)
        except Exception as e:
            print("esrgan_worker error:", e)
            upscaling_queue.put(("END", None))


    def writer_worker():
        try:
            buffer = {}
            expected_id = 0
            wrote = 0

            while True:
                # If ffmpeg died, stop cleanly and print why
                if video.poll() is not None:
                    try:
                        err = video.stderr.read().decode(errors="ignore") if video.stderr else ""
                    except Exception:
                        err = ""
                    print("[writer] ffmpeg exited early.")
                    if err:
                        print(err.strip()[:1200])
                    break

                fid, frame = upscaling_queue.get()

                if fid == "END":
                    # flush any remainder in order
                    for k in sorted(buffer.keys()):
                        ok, why = _validate_frame_bytes(buffer[k], width, height)
                        if not ok:
                            print(f"[writer] drop bad buffered frame {k}: {why}")
                            continue
                        try:
                            video.stdin.write(_frame_to_bytes(buffer[k]))
                            wrote += 1
                        except (BrokenPipeError, OSError, ValueError) as e:
                            print(f"[writer] flush write error: {e}")
                            break
                    break

                buffer[fid] = frame

                # write any ready frames in sequence
                while expected_id in buffer:
                    f = buffer.pop(expected_id)
                    ok, why = _validate_frame_bytes(f, width, height)
                    if not ok:
                        print(f"[writer] drop frame {expected_id}: {why}")
                        expected_id += 1
                        continue
                    try:
                        video.stdin.write(_frame_to_bytes(f))
                        wrote += 1
                    except (BrokenPipeError, OSError, ValueError) as e:
                        print(f"[writer] write error on {expected_id}: {e}")
                        # try to read stderr for context, then stop
                        try:
                            err = video.stderr.read().decode(errors="ignore") if video.stderr else ""
                            if err:
                                print(err.strip()[:1200])
                        except Exception:
                            pass
                        return
                    expected_id += 1

            # close stdin only once at the end
            try:
                video.stdin.close()
            except Exception:
                pass

        except Exception as e:
            print("writer_worker error:", e)

def _encoder_args(codec: str, width: int, height: int):
    c = (codec or "").lower()
    if c in {"h264_nvenc","hevc_nvenc","av1_nvenc"}:
        args = [
            "-c:v", c,
            "-preset", "p4",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "19",
            "-rc-lookahead", "20",
            "-bf:v", "3",
            "-pix_fmt", "yuv420p",
        ]
        if c != "av1_nvenc":
            args += ["-b_ref_mode", "middle"]
        return args
        
    # --- AMD AMF ---
    if c in {"h264_amf", "hevc_amf", "av1_amf"}:
        return ["-c:v", c, "-quality", "speed", "-pix_fmt", "yuv420p"]

    # --- Intel QSV ---
    if c in {"h264_qsv", "hevc_qsv", "vp9_qsv", "av1_qsv"}:
        # global_quality is QSV’s CRF-like knob; lower = better
        return ["-c:v", c, "-global_quality", "23", "-pix_fmt", "yuv420p"]

    # --- CPU encoders ---
    if c == "libx264":
        return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    if c == "libx265":
        return ["-c:v", "libx265", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
    if c == "libaom-av1":
        return ["-c:v", "libaom-av1", "-cpu-used", "6", "-crf", "32", "-b:v", "0", "-pix_fmt", "yuv420p"]
    if c == "libsvtav1":
        return ["-c:v", "libsvtav1", "-preset", "6", "-crf", "28", "-pix_fmt", "yuv420p"]
    if c in {"mp4v", "xvid", "divx"}:
        return ["-c:v", c, "-qscale:v", "2", "-pix_fmt", "yuv420p"]  # old-school MPEG-4 style

    # --- Fallback: pass through whatever was requested, or libx264 ---
    return ["-c:v", (c or "libx264"), "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]

def start_ffmpeg_writer(output_path, width, height, fps, codec):
    base = [
        "ffmpeg","-y",
        "-f","rawvideo","-vcodec","rawvideo",
        "-pix_fmt","bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i","-",
    ]
    enc = _encoder_args(codec, width, height)
    cmd = base + enc + [output_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

