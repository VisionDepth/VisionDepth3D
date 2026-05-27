# merged_pipeline.py

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
import queue
import gc

import platform
from core.ffmpeg_utils import require_tool
from core.debug_flags import debug_print, is_debug_enabled

def hidden_subprocess_kwargs():
    if platform.system().lower() != "windows":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0

    return {
        "startupinfo": startupinfo,
        "creationflags": subprocess.CREATE_NO_WINDOW,
    }

def start_stderr_drain_thread(proc, keep_last=8000):
    """
    Drains proc.stderr so FFmpeg cannot block on a full stderr pipe.
    Keeps only the last chunk for error/debug reporting.
    """
    chunks = []

    def _reader():
        try:
            while True:
                data = proc.stderr.readline()
                if not data:
                    break

                if isinstance(data, bytes):
                    data = data.decode("utf-8", errors="replace")

                chunks.append(data)

                joined = "".join(chunks)
                if len(joined) > keep_last:
                    chunks[:] = [joined[-keep_last:]]

        except Exception:
            pass

    if proc is not None and proc.stderr is not None:
        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    return chunks

suspend_flag = threading.Event()
cancel_flag = threading.Event()
progress_bar = None
status_label = None


def app_root():
    """
    Folder beside the EXE when frozen, or current script folder in dev.
    This is where user/runtime weights should live.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(".")


def bundle_root():
    """
    PyInstaller temporary bundle folder when frozen.
    Falls back to app_root() in normal dev runs.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return app_root()


def resolve_model_path(*parts):
    """
    Prefer the real app folder first:
        VisionDepth3D/weights/...
    Then fall back to bundled PyInstaller data:
        VisionDepth3D/_internal/weights/...
    """
    rel = os.path.join(*parts)

    app_path = os.path.join(app_root(), rel)
    if os.path.exists(app_path):
        return app_path

    bundled_path = os.path.join(bundle_root(), rel)
    if os.path.exists(bundled_path):
        return bundled_path

    # Return the app-side path by default so logs show the expected runtime location
    return app_path


# ✅ Get absolute path to resource (for PyInstaller compatibility)
#def resource_path(relative_path):
#    try:
#        base_path = sys._MEIPASS  # ✅ Corrected for PyInstaller
#    except AttributeError:
#        base_path = os.path.abspath(".")

#    return os.path.join(base_path, relative_path)

# =========================
# Force Hugging Face caches into VD3D /weights
# =========================
_VD3D_WEIGHTS = os.path.join(app_root(), "weights")
os.makedirs(_VD3D_WEIGHTS, exist_ok=True)

os.environ.setdefault("HF_HOME", _VD3D_WEIGHTS)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_VD3D_WEIGHTS, "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_VD3D_WEIGHTS, "datasets"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ✅ ONNX session options with graph optimization
session_options = ort.SessionOptions()
session_options.log_severity_level = 3
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_cpu_count = os.cpu_count() or 2
session_options.intra_op_num_threads = max(1, _cpu_count // 2)
session_options.inter_op_num_threads = 1

# ✅ ONNX Execution Provider fallback logic
available_providers = ort.get_available_providers()
debug_print(f"Available ONNX providers: {available_providers}")

if "CUDAExecutionProvider" in available_providers:
    device = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    provider_txt = "CUDA (NVIDIA)"
elif "ROCMExecutionProvider" in available_providers:
    device = ["ROCMExecutionProvider", "CPUExecutionProvider"]
    provider_txt = "ROCm (AMD)"
elif "DmlExecutionProvider" in available_providers:
    device = ["DmlExecutionProvider", "CPUExecutionProvider"]
    provider_txt = "DirectML (AMD/Intel)"
else:
    device = ["CPUExecutionProvider"]
    provider_txt = "CPU-only"

debug_print(f"Frametool Upscaler ONNX: {provider_txt}")

# ✅ Load RIFE
rife_session = None
rife_model_path = None
rife_model_id = None

esrgan_session = None  # ONNX ESRGAN / other ONNX SR
srresnet_model = None  # PyTorch SRResNet
if torch.cuda.is_available():
    srresnet_device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    srresnet_device = torch.device("mps")
else:
    srresnet_device = torch.device("cpu")

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

def wait_for_ffmpeg_writer(proc, timeout=300):
    """
    Waits for FFmpeg to finalize the written video.
    Prevents silent infinite hangs at 100 percent.
    """
    try:
        return_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass

        stderr_tail = "".join(getattr(proc, "_vd3d_stderr_chunks", []))[-4000:]
        raise RuntimeError(
            "FFmpeg writer timed out while finalizing the video.\n\n"
            f"{stderr_tail}"
        )

    if return_code != 0:
        stderr_tail = "".join(getattr(proc, "_vd3d_stderr_chunks", []))[-4000:]
        raise RuntimeError(
            f"FFmpeg writer failed with code {return_code}.\n\n"
            f"{stderr_tail}"
        )

    return return_code

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
    elapsed = max(1e-6, now - start)
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

def _abort_ffmpeg_writer(proc):
    """
    Best-effort cleanup for FFmpeg when processing fails before normal finalization.
    """
    try:
        if proc is not None and proc.stdin is not None:
            proc.stdin.close()
    except Exception:
        pass

    try:
        wait_for_ffmpeg_writer(proc, timeout=30)
    except Exception as e:
        debug_print(f"⚠️ FFmpeg abort/finalize failed: {e}")


def _write_validated_frame(proc, frame, width, height, label="frame"):
    ok, why = _validate_frame_bytes(frame, width, height)
    if not ok:
        _abort_ffmpeg_writer(proc)
        raise RuntimeError(f"Invalid {label} for FFmpeg: {why}")

    try:
        proc.stdin.write(_frame_to_bytes(frame))
    except Exception as e:
        _abort_ffmpeg_writer(proc)
        raise RuntimeError(f"Failed writing {label} to FFmpeg: {e}") from e

def _frame_loader(file_list, target_size=None, max_queue=8):
    """
    Generator that loads frames in a background thread and yields them.
    - Uses a bounded queue to limit RAM
    - Supports cancellation via global cancel_flag
    - Avoids deadlock by using timeouts on put/get
    """
    q = Queue(maxsize=max_queue)
    stop = object()

    def _worker():
        try:
            for fp in file_list:
                if cancel_flag.is_set():
                    break

                img = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                if target_size:
                    img = cv2.resize(
                        img,
                        target_size,
                        interpolation=cv2.INTER_AREA if img.shape[1] > target_size[0] else cv2.INTER_CUBIC
                    )

                # Put with timeout so we can observe cancel_flag and not deadlock
                while not cancel_flag.is_set():
                    try:
                        q.put(img, timeout=0.25)
                        break
                    except queue.Full:
                        continue
        finally:
            # Always try to signal end
            while True:
                try:
                    q.put(stop, timeout=0.25)
                    break
                except queue.Full:
                    # If consumer died, we don't want to hang forever
                    if cancel_flag.is_set():
                        break
                    continue

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        if cancel_flag.is_set():
            break
        try:
            item = q.get(timeout=0.25)
        except queue.Empty:
            continue

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

        debug_print(f"🚀 Running FFmpeg to extract frames from: {video_path}")
        debug_print(f"📁 Saving to: {output_folder}")

        ffmpeg_exe = require_tool("ffmpeg")
        command = [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-hwaccel", "auto",
            "-i", video_path,
            "-q:v", "2",
            output_pattern,
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            **hidden_subprocess_kwargs(),
        )

        if result.returncode != 0:
            debug_print(f"❌ FFmpeg frame extraction failed:\n{result.stderr[-4000:]}")
            
        if merged_progress:
            merged_progress.after(0, lambda: stop_spinner(result.returncode == 0))


    threading.Thread(target=extract_thread, daemon=True).start()


def select_output_file(output_path_var):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".mkv",
        filetypes=[("MKV Files", "*.mkv"), ("MP4 Files", "*.mp4"), ("AVI Files", "*.avi"), ("All Files", "*.*")]
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

# =========================
# Pause / Resume / Stop Controls (Upscale Pipeline)
# =========================

def _upscale_wait_if_paused():
    """
    Call this often inside loops.
    If paused, block until resumed.
    Cancel always wins.
    """
    while suspend_flag.is_set():
        if cancel_flag.is_set():
            return False
        time.sleep(0.05)
    return not cancel_flag.is_set()


def request_upscale_pause(progress_widget=None, status_widget=None):
    suspend_flag.set()
    if status_widget is not None:
        ui_set_status(status_widget, "Paused")


def request_upscale_resume(progress_widget=None, status_widget=None):
    suspend_flag.clear()
    if status_widget is not None:
        ui_set_status(status_widget, "Resuming...")


def request_upscale_stop(progress_widget=None, status_widget=None):
    # Stop means cancel the job and force-unpause so threads can exit
    cancel_flag.set()
    suspend_flag.clear()
    if status_widget is not None:
        ui_set_status(status_widget, "Stopping...")
        
def resolve_runtime_model(model_ref: str) -> str | None:
    """
    Supported forms:

      upscale:FuryTMP/RealESR_Gx4_fp16
      upscale:FuryTMP/BSRGANx2_fp16
      rife:FuryTMP/RIFE_fp32
      weights/RealESR_Gx4_fp16.onnx
      C:/absolute/path/model.onnx
    """
    if not model_ref:
        return None

    model_ref = str(model_ref).strip()

    # ---- Hugging Face repo selectors ----
    if model_ref.startswith("upscale:"):
        repo_id = model_ref[len("upscale:"):].strip()
        if not repo_id:
            debug_print(f"❌ Invalid upscale model ref: {model_ref}")
            return None

        filename = repo_id.rstrip("/").split("/")[-1] + ".onnx"
        return ensure_hf_file(repo_id, filename, local_subdir="weights")

    if model_ref.startswith("rife:"):
        repo_id = model_ref[len("rife:"):].strip()
        if not repo_id:
            debug_print(f"❌ Invalid RIFE model ref: {model_ref}")
            return None

        filename = repo_id.rstrip("/").split("/")[-1] + ".onnx"
        return ensure_hf_file(repo_id, filename, local_subdir="weights")

    # ---- Local path fallback ----
    model_ref = os.path.normpath(model_ref)

    if os.path.isabs(model_ref):
        return model_ref if os.path.exists(model_ref) else None

    resolved = resolve_model_path(model_ref)
    return resolved if os.path.exists(resolved) else None

def ensure_hf_file(repo_id: str, filename: str, local_subdir: str = "weights") -> str | None:
    """
    Download a single file from Hugging Face into VisionDepth3D/weights if missing.
    Returns the local file path, or None on failure.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        debug_print(f"❌ huggingface_hub not available: {e}")
        return None

    local_dir = os.path.join(app_root(), local_subdir)
    os.makedirs(local_dir, exist_ok=True)

    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        debug_print(f"✅ Model already exists: {local_path}")
        return local_path

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir
        )
        debug_print(f"⬇️ Downloaded {filename} from Hugging Face to: {downloaded}")
        return downloaded
    except Exception as e:
        debug_print(f"❌ Failed to download {filename} from {repo_id}: {e}")
        return None


def load_rife_model(model_ref: str):
    global rife_session, rife_model_path, rife_model_id

    rife_session = None
    rife_model_path = None
    rife_model_id = model_ref

    if not model_ref:
        debug_print("⚠️ No RIFE model selected.")
        return False

    resolved_path = resolve_runtime_model(model_ref)

    if not resolved_path or not os.path.exists(resolved_path):
        debug_print(f"❌ RIFE model file not found: {resolved_path}")
        return False

    try:
        rife_session = ort.InferenceSession(
            resolved_path,
            sess_options=session_options,
            providers=device
        )
        rife_model_path = resolved_path
        debug_print(f"✅ RIFE model loaded: {resolved_path}")
        return True
    except Exception as e:
        debug_print(f"❌ Failed to load RIFE model session: {e}")
        rife_session = None
        rife_model_path = None
        return False
        
def preprocess_rife(frame):
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame.astype(np.float32)


def _linear_intermediate_frames(frame1, frame2, count):
    """
    Timing-safe fallback.
    These are not true AI interpolation frames, but they preserve duration
    if RIFE fails or returns the wrong number of frames.
    """
    frames = []

    if count <= 0:
        return frames

    frame1 = normalize_frame(frame1)
    frame2 = normalize_frame(frame2, (frame1.shape[1], frame1.shape[0]))

    for i in range(count):
        alpha = (i + 1) / (count + 1)
        blended = cv2.addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0)
        frames.append(blended.astype(np.uint8))

    return frames


def _run_rife_middle(frame1, frame2):
    """
    Run RIFE once and return one middle frame between frame1 and frame2.
    Most simple RIFE ONNX exports produce one middle/interpolated frame.
    """
    if not rife_session:
        return None

    try:
        frame1 = normalize_frame(frame1)
        frame2 = normalize_frame(frame2, (frame1.shape[1], frame1.shape[0]))

        merged = concatenate_images(frame1, frame2)
        tensor = preprocess_rife(merged)

        input_name = rife_session.get_inputs()[0].name
        output = rife_session.run(None, {input_name: tensor})[0]
        output = np.clip(output, 0, 1)

        # Usually [1, 3, H, W]
        if output.ndim == 4:
            frame = output[0]
        else:
            frame = output

        # CHW to HWC
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))

        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return normalize_frame(frame, (frame1.shape[1], frame1.shape[0]))

    except Exception as e:
        debug_print(f"❌ RIFE middle-frame inference error: {e}")
        return None

    finally:
        gc.collect()


def _rife_recursive_between(frame1, frame2, levels):
    """
    Recursive interpolation.

    levels=1:
        1 in-between frame, for 2x

    levels=2:
        3 in-between frames, for 4x

    levels=3:
        7 in-between frames, for 8x
    """
    if levels <= 0:
        return []

    mid = _run_rife_middle(frame1, frame2)

    if mid is None:
        return []

    left = _rife_recursive_between(frame1, mid, levels - 1)
    right = _rife_recursive_between(mid, frame2, levels - 1)

    return left + [mid] + right


def run_rife(frame1, frame2, multiplier):
    """
    Always returns exactly multiplier - 1 intermediate frames.

    This is critical:
    2x needs 1 in-between frame
    4x needs 3 in-between frames
    8x needs 7 in-between frames

    If output_fps is multiplied but these frames are missing,
    the video plays too fast.
    """
    multiplier = int(multiplier)
    expected = max(0, multiplier - 1)

    if expected <= 0:
        return []

    level_map = {
        2: 1,
        4: 2,
        8: 3,
    }

    levels = level_map.get(multiplier)

    if not rife_session or levels is None:
        debug_print(f"⚠️ RIFE unavailable or unsupported multiplier {multiplier}. Using timing fallback.")
        return _linear_intermediate_frames(frame1, frame2, expected)

    frames = _rife_recursive_between(frame1, frame2, levels)

    if len(frames) != expected:
        debug_print(
            f"⚠️ RIFE returned {len(frames)} frames, expected {expected}. "
            "Using timing-safe fallback frames."
        )
        frames = _linear_intermediate_frames(frame1, frame2, expected)

    # Final guarantee. Never return too few or too many.
    if len(frames) < expected:
        frames.extend(_linear_intermediate_frames(frame1, frame2, expected - len(frames)))

    return frames[:expected]

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
    """
    Blend model output with the original frame.

    UI meaning:
      OFF    = 100% model output
      LOW    = 75% model / 25% original
      MEDIUM = 50% model / 50% original
      HIGH   = 25% model / 75% original

    The original frame is resized to match the upscaled/output frame before blending.
    """
    if upscaled is None:
        return original

    mode = str(mode or "OFF").upper()

    if mode == "OFF":
        return normalize_frame(upscaled)

    if original is None:
        return normalize_frame(upscaled)

    upscaled = normalize_frame(upscaled)

    original = normalize_frame(
        original,
        target_size=(upscaled.shape[1], upscaled.shape[0]),
    )

    model_alpha_map = {
        "LOW": 0.75,
        "MEDIUM": 0.50,
        "HIGH": 0.25,
    }

    model_alpha = model_alpha_map.get(mode, 1.0)

    if model_alpha >= 1.0:
        return upscaled

    original_alpha = 1.0 - model_alpha

    return cv2.addWeighted(upscaled, model_alpha, original, original_alpha, 0)

def init_upscaler(model_path: str, enable_upscale: bool):
    """
    Decide which backend to use based on model_path extension:
      - .onnx -> ONNX / ESRGAN
      - .pth  -> PyTorch SRResNet super-res

    model_path can be:
      - local relative path
      - local absolute path
      - Hugging Face selector, e.g. upscale:FuryTMP/RealESR_Gx4_fp16
    """
    global esrgan_session, srresnet_model, UPSCALE_BACKEND

    esrgan_session = None
    srresnet_model = None
    UPSCALE_BACKEND = "none"

    if not enable_upscale or not model_path:
        debug_print("Upscaler disabled.")
        return False

    resolved_model_path = resolve_runtime_model(model_path)
    if not resolved_model_path:
        debug_print(f"❌ Failed to resolve upscaler model: {model_path}")
        return False

    debug_print(f"Upscaler path resolved to: {resolved_model_path}")
    ext = os.path.splitext(resolved_model_path)[1].lower()

    if ext == ".onnx":
        try:
            debug_print(f"Loading ONNX upscaler from {resolved_model_path}")
            esrgan_session = ort.InferenceSession(
                resolved_model_path,
                sess_options=session_options,
                providers=device
            )
            UPSCALE_BACKEND = "onnx"
            debug_print(f"ONNX upscaler ready [{provider_txt}]")
            return True
        except Exception as e:
            UPSCALE_BACKEND = "none"
            debug_print(f"❌ Failed to load ONNX upscaler: {e}")
            return False

    elif ext == ".pth":
        try:
            debug_print(f"Loading SRResNet (.pth) upscaler from {resolved_model_path}")
            model = SRResNet(num_blocks=16, upscale_factor=4)
            state = torch.load(resolved_model_path, map_location=srresnet_device)
            model.load_state_dict(state)
            model.to(srresnet_device)
            model.eval()
            srresnet_model = model
            UPSCALE_BACKEND = "srresnet"
            if srresnet_device.type == "cuda":
                mode_txt = "CUDA" if not getattr(torch.version, "hip", None) else "ROCm"
            elif srresnet_device.type == "mps":
                mode_txt = "Metal"
            else:
                mode_txt = "CPU"
            debug_print(f"SRResNet upscaler ready [{mode_txt}]")
            return True
        except Exception as e:
            UPSCALE_BACKEND = "none"
            debug_print(f"❌ Failed to load SRResNet model: {e}")
            return False

    else:
        debug_print(f"⚠️ Unknown upscaler model extension: {ext}. Supported: .onnx, .pth")
        UPSCALE_BACKEND = "none"
        return False



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
        if srresnet_device.type == "cuda" and not getattr(torch.version, "hip", None):
            with torch.autocast("cuda", dtype=torch.float16):
                sr = srresnet_model(tensor)
        else:
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

    UI behavior preserved:

      input_res_pct controls the frame size sent into the model.

      Example with 1920x1080 source and x4 model:
        100% -> model input 1920x1080 -> model output 7680x4320 -> resize to target_size
         75% -> model input 1440x810  -> model output 5760x3240 -> resize to target_size
         50% -> model input 960x540   -> model output 3840x2160 -> resize to target_size
         25% -> model input 480x270   -> model output 1920x1080 -> resize to target_size

    Blend behavior:
      OFF    = full model output
      LOW    = 75% model / 25% original
      MEDIUM = 50% model / 50% original
      HIGH   = 25% model / 75% original
    """
    global esrgan_session, srresnet_model, UPSCALE_BACKEND

    if frame is None:
        return frame

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

    def _model_scale_from_name(name):
        n = str(name or "").lower()

        if "x2" in n or "2x" in n:
            return 2
        if "x3" in n or "3x" in n:
            return 3
        if "x4" in n or "4x" in n:
            return 4

        # Most RealESRGAN/ESRGAN models in this project are x4 unless specified.
        return 4

    frame = _to_bgr_uint8(frame)
    original = frame.copy()

    backend_has_model = (
        (UPSCALE_BACKEND == "onnx" and esrgan_session is not None) or
        (UPSCALE_BACKEND == "srresnet" and srresnet_model is not None)
    )

    if not backend_has_model:
        out = original
        if target_size:
            out = _fit_size(out, target_size)
        return out if str(blend_mode or "OFF").upper() == "OFF" else blend_images(original, out, mode=blend_mode)

    # ------------------------------------------------------------
    # Apply UI input percentage BEFORE model inference.
    # This is the behavior your UI describes.
    # ------------------------------------------------------------
    frame_for_model = frame

    try:
        pct = int(input_res_pct)
    except Exception:
        pct = 100

    pct = max(1, min(100, pct))

    if pct != 100:
        h, w = frame_for_model.shape[:2]
        new_w = max(1, int(round(w * pct / 100.0)))
        new_h = max(1, int(round(h * pct / 100.0)))

        frame_for_model = cv2.resize(
            frame_for_model,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if pct < 100 else cv2.INTER_CUBIC,
        )

        if is_debug_enabled():
            debug_print(
                f"[UPSCALE INPUT%] {pct}%: "
                f"{w}x{h} -> model input {new_w}x{new_h}"
            )

    # ------------------------------------------------------------
    # SRResNet branch
    # ------------------------------------------------------------
    if UPSCALE_BACKEND == "srresnet":
        try:
            upscaled = _run_srresnet(frame_for_model, scale=4)
            upscaled = _to_bgr_uint8(upscaled)

        except Exception as e:
            debug_print(f"❌ SRResNet failed: {e}")
            out = original
            if target_size:
                out = _fit_size(out, target_size)
            return out if str(blend_mode or "OFF").upper() == "OFF" else blend_images(original, out, mode=blend_mode)

        if target_size:
            upscaled = _fit_size(upscaled, target_size)
            original_for_blend = _fit_size(original, target_size)
        else:
            h_hr, w_hr = upscaled.shape[:2]
            original_for_blend = _fit_size(original, (w_hr, h_hr))

        return blend_images(original_for_blend, upscaled, mode=blend_mode)

    # ------------------------------------------------------------
    # ONNX ESRGAN branch
    # ------------------------------------------------------------
    scale = _model_scale_from_name(model_name)

    try:
        if tile:
            upscaled = _esrgan_tiled(frame_for_model, tile, tile_pad)
        else:
            tensor = preprocess_esr(frame_for_model)
            input_name = esrgan_session.get_inputs()[0].name
            output = esrgan_session.run(None, {input_name: tensor})[0]
            upscaled = postprocess_esr(output)

    except Exception as e:
        debug_print(f"❌ ESRGAN failed: {e}")
        out = original
        if target_size:
            out = _fit_size(out, target_size)
        return out if str(blend_mode or "OFF").upper() == "OFF" else blend_images(original, out, mode=blend_mode)

    upscaled = _to_bgr_uint8(upscaled)

    # Some ONNX models return full scaled output.
    # Some exports may return same-size output. If same/smaller, force expected scale.
    h0, w0 = frame_for_model.shape[:2]
    h1, w1 = upscaled.shape[:2]

    if h1 <= h0 and w1 <= w0:
        upscaled = cv2.resize(
            upscaled,
            (w0 * scale, h0 * scale),
            interpolation=cv2.INTER_CUBIC,
        )

    if is_debug_enabled():
        h2, w2 = upscaled.shape[:2]
        debug_print(
            f"[UPSCALE MODEL] backend={UPSCALE_BACKEND} scale_guess={scale} "
            f"model_input={w0}x{h0} model_output={w2}x{h2} "
            f"target={target_size if target_size else 'native'} blend={blend_mode}"
        )

    # Final output always matches the UI output resolution if target_size is provided.
    if target_size:
        upscaled = _fit_size(upscaled, target_size)
        original_for_blend = _fit_size(original, target_size)
    else:
        h_hr, w_hr = upscaled.shape[:2]
        original_for_blend = _fit_size(original, (w_hr, h_hr))

    return blend_images(original_for_blend, upscaled, mode=blend_mode)



def _esrgan_tiled(img, tile, pad):
    """
    Run ONNX ESRGAN in tiles.

    Correctly handles models that return scaled output, e.g. 2x or 4x.
    """
    global esrgan_session

    if esrgan_session is None:
        return img

    if tile is None or int(tile) <= 0:
        return img

    tile = int(tile)
    pad = int(pad or 0)

    h, w = img.shape[:2]
    input_name = esrgan_session.get_inputs()[0].name

    out = None
    scale_x = None
    scale_y = None

    for y in range(0, h, tile):
        for x in range(0, w, tile):
            tile_h = min(tile, h - y)
            tile_w = min(tile, w - x)

            y0 = max(0, y - pad)
            x0 = max(0, x - pad)
            y1 = min(h, y + tile_h + pad)
            x1 = min(w, x + tile_w + pad)

            crop = img[y0:y1, x0:x1]
            tensor = preprocess_esr(crop)

            pred = esrgan_session.run(None, {input_name: tensor})[0]
            up = postprocess_esr(pred)

            crop_h, crop_w = crop.shape[:2]
            up_h, up_w = up.shape[:2]

            this_scale_y = up_h / float(crop_h)
            this_scale_x = up_w / float(crop_w)

            if out is None:
                scale_y = this_scale_y
                scale_x = this_scale_x

                out_h = max(1, int(round(h * scale_y)))
                out_w = max(1, int(round(w * scale_x)))

                out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

            # Source region inside padded crop, converted to upscaled coordinates.
            src_y0 = int(round((y - y0) * this_scale_y))
            src_x0 = int(round((x - x0) * this_scale_x))
            src_y1 = int(round((y - y0 + tile_h) * this_scale_y))
            src_x1 = int(round((x - x0 + tile_w) * this_scale_x))

            # Destination region in final output.
            dst_y0 = int(round(y * scale_y))
            dst_x0 = int(round(x * scale_x))
            dst_y1 = int(round((y + tile_h) * scale_y))
            dst_x1 = int(round((x + tile_w) * scale_x))

            patch = up[src_y0:src_y1, src_x0:src_x1]

            expected_w = dst_x1 - dst_x0
            expected_h = dst_y1 - dst_y0

            if patch.shape[1] != expected_w or patch.shape[0] != expected_h:
                patch = cv2.resize(
                    patch,
                    (expected_w, expected_h),
                    interpolation=cv2.INTER_CUBIC,
                )

            out[dst_y0:dst_y1, dst_x0:dst_x1] = patch

    return out if out is not None else img
    

def merge_audio_from_source_video(rendered_video_path, source_video_path, output_with_audio_path):
    if not source_video_path or not os.path.exists(source_video_path):
        return rendered_video_path

    if not rendered_video_path or not os.path.exists(rendered_video_path):
        return rendered_video_path

    ffmpeg_exe = require_tool("ffmpeg")

    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-y",
        "-i", rendered_video_path,
        "-i", source_video_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        output_with_audio_path,
    ]

    debug_print("[AUDIO MERGE CMD]", " ".join(str(x) for x in cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        **hidden_subprocess_kwargs(),
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:] or "Audio merge failed.")

    return output_with_audio_path
   
def start_merged_pipeline(settings, progress_widget, status_label_widget):
    global progress_bar, status_label, esrgan_session
    progress_bar = progress_widget
    status_label = status_label_widget

    cancel_flag.clear()
    suspend_flag.clear()

    frames_dir = settings["frames_folder"]
    output_path = settings["output_file"]
    codec = settings["codec"]
    width, height = settings["width"], settings["height"]
    fps = settings["fps"]
    fps_mult = settings["fps_multiplier"]
    enable_rife = settings["enable_rife"]
    rife_model = settings.get("rife_model", "rife:FuryTMP/RIFE_fp32")
    enable_upscale = settings["enable_upscale"]
    blend_mode = settings.get("blend_mode", "OFF")
    input_res_pct = settings.get("input_res_pct", 100)
    model_path = settings.get("model_path", "upscale:FuryTMP/RealESR_Gx4_fp16")

    # Initialize upscaler (ONNX or SRResNet)
    if enable_upscale:
        if not init_upscaler(model_path, enable_upscale):
            messagebox.showerror(
                "Upscaler Error",
                f"Failed to load/download upscaler model:\n{model_path}"
            )
            return

    # Initialize RIFE only if needed
    if enable_rife:
        if not load_rife_model(rife_model):
            messagebox.showerror(
                "RIFE Error",
                f"Failed to load/download RIFE model:\n{rife_model}"
            )
            return

    files = natural_sort([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not files:
        messagebox.showerror("Error", "No frames found in selected folder.")
        return

    output_fps = fps * fps_mult if enable_rife else fps
    video = start_ffmpeg_writer(output_path, width, height, output_fps, codec)
    start = time.monotonic()

    target_size = (width, height)

    # Keep native resolution internally for SR / RIFE
    file_iter = _frame_loader(files, None)

    prev = next(file_iter, None)
    if prev is None:
        messagebox.showerror("Error", "No readable frames.")
        try:
            video.stdin.close()
        except Exception:
            pass
        wait_for_ffmpeg_writer(video)
        return

    total_src = len(files)

    # Write the first source frame so output duration matches the input timeline.
    if enable_upscale:
        prev_proc = run_esrgan(
            prev,
            blend_mode,
            input_res_pct,
            model_name=model_path,
            target_size=target_size
        )
    else:
        prev_proc = cv2.resize(prev, target_size)

    _write_validated_frame(video, prev_proc, width, height, "first frame")
    
    del prev_proc
    gc.collect()

    for i, curr in enumerate(file_iter, start=1):
        if cancel_flag.is_set():
            break
        if not _upscale_wait_if_paused():
            break

        if enable_rife:
            if not _upscale_wait_if_paused():
                break

            interpolated = run_rife(prev, curr, fps_mult)
            expected_interpolated = int(fps_mult) - 1

            if is_debug_enabled() and i <= 10:
                debug_print(
                    f"[RIFE CHECK] pair={i} multiplier={fps_mult} "
                    f"expected={expected_interpolated} got={len(interpolated)}"
                )

            if len(interpolated) != expected_interpolated:
                debug_print(
                    f"⚠️ RIFE returned {len(interpolated)} frames, "
                    f"expected {expected_interpolated}. Timing will be wrong unless fallback is used."
                )

            if enable_upscale:
                if not _upscale_wait_if_paused():
                    break

                interpolated = [
                    run_esrgan(f, blend_mode, input_res_pct, model_name=model_path, target_size=target_size)
                    for f in interpolated
                ]

                if not _upscale_wait_if_paused():
                    break

                curr_proc = run_esrgan(
                    curr,
                    blend_mode,
                    input_res_pct,
                    model_name=model_path,
                    target_size=target_size
                )
            else:
                interpolated = [cv2.resize(f, target_size) for f in interpolated]
                curr_proc = cv2.resize(curr, target_size)

            for f in interpolated:
                if cancel_flag.is_set():
                    break
                if not _upscale_wait_if_paused():
                    break
                _write_validated_frame(video, f, width, height, "interpolated frame")

        else:
            if enable_upscale:
                if not _upscale_wait_if_paused():
                    break

                curr_proc = run_esrgan(
                    curr,
                    blend_mode,
                    input_res_pct,
                    model_name=model_path,
                    target_size=target_size
                )
            else:
                curr_proc = cv2.resize(curr, target_size)

        if cancel_flag.is_set():
            break
        if not _upscale_wait_if_paused():
            break

        _write_validated_frame(video, curr_proc, width, height, "current frame")
        
        prev = curr

        update_progress(i + 1, total_src, start)

    if cancel_flag.is_set():
        ui_set_status(status_label, "Stopped.")
        try:
            video.stdin.close()
        except Exception:
            pass

        try:
            wait_for_ffmpeg_writer(video)
        except Exception as e:
            debug_print(f"⚠️ FFmpeg finalize after stop failed: {e}")

        return

    ui_set_status(status_label, "Finalizing video...")
    ui_set_progress(progress_bar, 99.0)

    try:
        video.stdin.close()
    except Exception:
        pass

    wait_for_ffmpeg_writer(video)

    final_output_path = output_path

    if settings.get("keep_original_audio", False) and settings.get("input_video_file"):
        try:
            base, ext = os.path.splitext(output_path)
            audio_output_path = base + "_audio" + ext

            ui_set_status(status_label, "Merging original audio...")

            merged_path = merge_audio_from_source_video(
                output_path,
                settings.get("input_video_file"),
                audio_output_path,
            )

            # Replace silent output with audio version if possible.
            try:
                os.replace(merged_path, output_path)
                final_output_path = output_path
            except Exception:
                final_output_path = merged_path

        except Exception as exc:
            debug_print(f"⚠️ Audio merge failed: {exc}")
            ui_set_status(
                status_label,
                f"Processing complete, but audio merge failed: {exc}"
            )

    update_progress(total_src, total_src, start)
    ui_set_progress(progress_bar, 100.0)
    ui_set_status(status_label, "✅ Processing Complete!")
    
MAX_QUEUE_SIZE = 16
END_SEG = ("END", None, None, [])
END_FRM = ("END", None)

def q_put(q, item, cancel_evt, timeout=0.25):
    while not cancel_flag.is_set() and not cancel_evt.is_set():
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            # normal: queue is full, retry until cancel
            continue
        except Exception as e:
            # not normal: surface the real bug
            debug_print(f"[q_put] unexpected error: {e}")
            cancel_evt.set()
            return False
    return False

def q_get(q, cancel_evt, timeout=0.25):
    while not cancel_flag.is_set() and not cancel_evt.is_set():
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            # normal: nothing ready yet, retry until cancel
            continue
        except Exception as e:
            debug_print(f"[q_get] unexpected error: {e}")
            cancel_evt.set()
            return None
    return None

def start_threaded_pipeline(settings, progress_widget, status_label_widget):
    global progress_bar, status_label, esrgan_session
    progress_bar = progress_widget
    status_label = status_label_widget

    cancel_flag.clear()
    suspend_flag.clear()

    try:
        progress_bar.after(0, lambda: progress_bar.configure(
            mode="determinate", maximum=100.0, value=0.0
        ))
        status_label.after(0, lambda: status_label.configure(text="Preparing..."))
    except Exception:
        pass

    frames_dir    = settings["frames_folder"]
    output_path   = settings["output_file"]
    codec         = settings["codec"]
    width         = settings["width"]
    height        = settings["height"]
    fps           = settings["fps"]
    fps_mult      = settings["fps_multiplier"]
    enable_rife   = settings["enable_rife"]
    enable_up     = settings["enable_upscale"]
    blend_mode    = settings.get("blend_mode", "OFF")
    input_res_pct = settings.get("input_res_pct", 100)
    rife_model    = settings.get("rife_model", "rife:FuryTMP/RIFE_fp32")
    model_path    = settings.get("model_path", "upscale:FuryTMP/RealESR_Gx4_fp16")

    ui_set_status(status_label, "Preparing...")

    output_fps = fps * fps_mult if enable_rife else fps
    target_size = (width, height)
    work_size = None

    files = natural_sort([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    total_src = len(files)
    total_pairs = max(1, total_src - 1)

    if total_src < 2:
        ui_set_status(status_label, "⚠️ Not enough frames to process.")
        return

    if enable_up:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not init_upscaler(model_path, enable_up):
            ui_set_status(status_label, "❌ Failed to load/download upscaler model.")
            return

    if enable_rife:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not load_rife_model(rife_model):
            ui_set_status(status_label, "❌ Failed to load/download RIFE model.")
            return

    video = start_ffmpeg_writer(output_path, width, height, output_fps, codec)
    start = time.monotonic()

    # Small queues. Keep them tight so RAM does not balloon.
    segment_queue = Queue(maxsize=4)
    write_queue = Queue(maxsize=8)

    cancel_local = threading.Event()
    END_SEG = ("END", None, None)
    END_WRITE = ("END", None)
    
    
    # Write/enqueue the first source frame so threaded output matches
    # the non-threaded pipeline timeline.
    try:
        first = cv2.imread(files[0], cv2.IMREAD_COLOR)
        first = normalize_frame(first, work_size)

        if enable_up:
            first_proc = run_esrgan(
                first,
                blend_mode,
                input_res_pct,
                model_name=model_path,
                target_size=target_size,
            )
        else:
            first_proc = cv2.resize(first, target_size)

        ok, why = _validate_frame_bytes(first_proc, width, height)
        if not ok:
            raise ValueError(f"Initial frame is invalid: {why}")

        write_queue.put_nowait(("FRAME", first_proc))

    except Exception as e:
        debug_print(f"⚠️ Failed to prepare first frame: {e}")
        ui_set_status(status_label, f"❌ Failed to prepare first frame: {e}")
        try:
            video.stdin.close()
        except Exception:
            pass
        try:
            wait_for_ffmpeg_writer(video)
        except Exception:
            pass
        return

    def reader_rife_worker():
        """
        Reads frames and performs only the RIFE stage.
        Sends:
            (pair_index, curr_frame, interpolated_frames)
        in natural order.
        """
        try:
            prev = cv2.imread(files[0], cv2.IMREAD_COLOR)
            prev = normalize_frame(prev, work_size)

            for idx in range(1, len(files)):
                if cancel_flag.is_set() or cancel_local.is_set():
                    break
                if not _upscale_wait_if_paused():
                    break

                curr = cv2.imread(files[idx], cv2.IMREAD_COLOR)
                curr = normalize_frame(curr, work_size)

                if enable_rife:
                    interpolated = run_rife(prev, curr, fps_mult)
                    expected_interpolated = int(fps_mult) - 1

                    if is_debug_enabled() and idx <= 10:
                        debug_print(
                            f"[RIFE CHECK THREADED] pair={idx} multiplier={fps_mult} "
                            f"expected={expected_interpolated} got={len(interpolated)}"
                        )
                    
                else:
                    interpolated = []

                if not q_put(segment_queue, (idx, curr, interpolated), cancel_local):
                    break

                prev = curr

        except Exception as e:
            debug_print(f"⚠️ reader_rife_worker error: {e}")
            cancel_local.set()
        finally:
            q_put(segment_queue, END_SEG, cancel_local)

    def process_worker():
        """
        Consumes ordered segments, performs upscale/resize, and emits finished
        frames in final output order. No reordering dict needed.
        """
        try:
            processed = 0

            while True:
                if cancel_flag.is_set() or cancel_local.is_set():
                    break
                if not _upscale_wait_if_paused():
                    break

                item = q_get(segment_queue, cancel_local)
                if item is None:
                    break

                idx, curr, interpolated = item
                if idx == "END":
                    break

                # Process interpolated frames first
                if enable_up:
                    inter_proc = [
                        run_esrgan(
                            f,
                            blend_mode,
                            input_res_pct,
                            model_name=model_path,
                            target_size=target_size
                        )
                        for f in interpolated
                    ]
                    curr_proc = run_esrgan(
                        curr,
                        blend_mode,
                        input_res_pct,
                        model_name=model_path,
                        target_size=target_size
                    )
                else:
                    inter_proc = [cv2.resize(f, target_size) for f in interpolated]
                    curr_proc = cv2.resize(curr, target_size)

                for f in inter_proc:
                    if cancel_flag.is_set() or cancel_local.is_set():
                        break
                    if not _upscale_wait_if_paused():
                        break
                    if not q_put(write_queue, ("FRAME", f), cancel_local):
                        break

                if cancel_flag.is_set() or cancel_local.is_set():
                    break
                if not _upscale_wait_if_paused():
                    break

                if not q_put(write_queue, ("FRAME", curr_proc), cancel_local):
                    break

                processed += 1
                update_progress(processed, total_pairs, start)

        except Exception as e:
            debug_print(f"⚠️ process_worker error: {e}")
            cancel_local.set()
        finally:
            q_put(write_queue, END_WRITE, cancel_local)

    def writer_worker():
        """
        Writes already-ordered finished frames directly to ffmpeg.
        """
        try:
            while True:
                if cancel_flag.is_set() or cancel_local.is_set():
                    break

                item = q_get(write_queue, cancel_local)
                if item is None:
                    break

                kind, payload = item
                if kind == "END":
                    break

                frame = payload
                ok, why = _validate_frame_bytes(frame, width, height)
                if not ok:
                    debug_print(f"[writer] invalid frame, aborting: {why}")
                    cancel_local.set()
                    break

                try:
                    video.stdin.write(_frame_to_bytes(frame))
                except Exception as e:
                    debug_print(f"[writer] write error: {e}")
                    cancel_local.set()
                    break

        except Exception as e:
            debug_print(f"⚠️ writer_worker error: {e}")
            cancel_local.set()
        finally:
            try:
                video.stdin.close()
            except Exception:
                pass
            try:
                wait_for_ffmpeg_writer(video)
            except Exception as e:
                debug_print(f"⚠️ FFmpeg writer finalize failed: {e}")
                cancel_local.set()

    t_read = threading.Thread(target=reader_rife_worker, daemon=True)
    t_proc = threading.Thread(target=process_worker, daemon=True)
    t_wrt  = threading.Thread(target=writer_worker, daemon=True)

    debug_print(">>> start_threaded_pipeline called")
    t_read.start()
    t_proc.start()
    t_wrt.start()

    ui_set_status(status_label, "Running threaded pipeline...")

    def _wait_finish():
        t_read.join()
        t_proc.join()
        t_wrt.join()

        was_cancelled = cancel_flag.is_set()
        failed = cancel_local.is_set() and not was_cancelled

        if was_cancelled:
            ui_set_status(status_label, "Stopped.")
            return

        if failed:
            ui_set_status(status_label, "❌ Processing failed. See debug log for details.")
            return

        audio_warning = None

        if settings.get("keep_original_audio", False) and settings.get("input_video_file"):
            try:
                base, ext = os.path.splitext(output_path)
                audio_output_path = base + "_audio" + ext

                ui_set_status(status_label, "Merging original audio...")
                merged_path = merge_audio_from_source_video(
                    output_path,
                    settings.get("input_video_file"),
                    audio_output_path,
                )

                try:
                    os.replace(merged_path, output_path)
                except Exception:
                    # If replacement fails, at least leave the audio version on disk.
                    audio_warning = f"Audio merged to separate file: {merged_path}"

            except Exception as exc:
                debug_print(f"⚠️ Audio merge failed: {exc}")
                audio_warning = f"Processing complete, but audio merge failed: {exc}"

        update_progress(total_pairs, total_pairs, start)
        ui_set_progress(progress_bar, 100.0)

        if audio_warning:
            ui_set_status(status_label, audio_warning)
        else:
            ui_set_status(status_label, "Processing Complete!")

    threading.Thread(target=_wait_finish, daemon=True).start()
    
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
    ffmpeg_exe = require_tool("ffmpeg")

    base = [
        ffmpeg_exe,
        "-hide_banner",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{int(width)}x{int(height)}",
        "-r", str(float(fps)),
        "-i", "-",
        "-an",
    ]

    enc = _encoder_args(codec, width, height)

    ext = os.path.splitext(output_path)[1].lower()
    mux_args = []

    if ext in {".mp4", ".mov", ".m4v"}:
        mux_args += ["-movflags", "+faststart"]

    cmd = base + enc + mux_args + [output_path]

    debug_print("[FPS UPSCALE FFMPEG CMD]", " ".join(str(x) for x in cmd))

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
        **hidden_subprocess_kwargs(),
    )

    proc._vd3d_stderr_chunks = start_stderr_drain_thread(proc)
    return proc
