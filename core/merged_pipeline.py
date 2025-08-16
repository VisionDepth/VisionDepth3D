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

print(f"🧠 ONNX will use providers: {device}")



# ✅ Load RIFE
rife_path = resource_path(os.path.join("weights", "RIFE_fp32.onnx"))

try:
    rife_session = ort.InferenceSession(rife_path, sess_options=session_options, providers=device)
    print("✅ RIFE model loaded.")
except Exception as e:
    print(f"❌ Failed to load RIFE model: {e}")
    rife_session = None

esrgan_session = None  # Lazy-load ESRGAN

def update_progress(done, total, start):
    if not progress_bar or not status_label:
        return
    elapsed = time.time() - start
    fps = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / fps if fps > 0 else float("inf")
    eta_fmt = time.strftime("%H:%M:%S", time.gmtime(eta)) if eta != float("inf") else "--:--"
    pct = (done / total) * 100 if total else 0

    # ✅ marshal UI updates to the main thread
    try:
        progress_bar.after(0, lambda: progress_bar.configure(value=pct))
        status_label.after(0, lambda: status_label.configure(
            text=f"Progress: {done}/{total} | FPS: {fps:.2f} | ETA: {eta_fmt}"
        ))
    except Exception:
        pass


from queue import Queue

def _frame_loader(file_list, target_size):
    q = Queue(maxsize=8)
    stop = object()

    def _worker():
        for fp in file_list:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if target_size:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA if img.shape[1] > target_size[0] else cv2.INTER_CUBIC)
            q.put(img)
        q.put(stop)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        item = q.get()
        if item is stop:
            break
        yield item


from tkinter.simpledialog import askstring

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
    
def run_esrgan(frame, blend_mode="OFF", input_res_pct=100, model_name="RealESR_Gx4_fp16",
               target_size=None, tile=None, tile_pad=8):
    global esrgan_session
    if not esrgan_session:
        return frame

    original = frame
    if input_res_pct != 100:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * input_res_pct / 100), int(h * input_res_pct / 100)), interpolation=cv2.INTER_AREA if input_res_pct < 100 else cv2.INTER_CUBIC)

    if tile:
        upscaled = _esrgan_tiled(frame, tile, tile_pad)
    else:
        tensor = preprocess_esr(frame)
        try:
            output = esrgan_session.run(None, {esrgan_session.get_inputs()[0].name: tensor})[0]
            upscaled = postprocess_esr(output)
        except Exception as e:
            print(f"❌ ESRGAN failed: {e}")
            return original

    scale = 2 if "x2" in model_name.lower() else 4
    upscaled = cv2.resize(upscaled, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    upscaled = cv2.resize(upscaled, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
    if target_size:
        upscaled = cv2.resize(upscaled, target_size, interpolation=cv2.INTER_CUBIC)
    return blend_images(original, upscaled, mode=blend_mode)

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

    if enable_upscale:
        if not os.path.exists(model_path):
            print(f"\u274C ESRGAN model missing: {model_path}")
            esrgan_session = None
        else:
            esrgan_session = ort.InferenceSession(model_path, sess_options=session_options, providers=device)

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

    # ---- prefetch-driven loop ----
    target_size = (width, height)

    # If upscaling is enabled, let run_esrgan handle resizing to target_size.
    # Otherwise, have the loader resize to target_size up-front to hide I/O latency.
    file_iter = _frame_loader(files, None if enable_upscale else target_size)

    # prime the first frame
    prev = next(file_iter, None)
    if prev is None:
        messagebox.showerror("Error", "No readable frames.")
        try:
            video.stdin.close()
        except Exception:
            pass
        video.wait()
        return

    if enable_upscale:
        prev = run_esrgan(prev, blend_mode, input_res_pct, target_size=target_size)

    # write first frame
    video.stdin.write(prev.tobytes())

    # total frames we’ll report progress against
    # (if RIFE is enabled, we conceptually process pairs; progress uses #source frames)
    total_src = len(files)

    for i, curr in enumerate(file_iter, start=1):
        if cancel_flag.is_set():
            break

        if enable_upscale:
            curr_proc = run_esrgan(curr, blend_mode, input_res_pct, target_size=target_size)
        else:
            curr_proc = curr  # already resized by loader

        if enable_rife:
            # interpolate between prev and curr, then write curr
            inter = run_rife(prev, curr_proc, fps_mult)
            for f in inter:
                video.stdin.write(f.tobytes())

        video.stdin.write(curr_proc.tobytes())
        prev = curr_proc

        # progress reports against source frames consumed
        update_progress(i + 1, total_src if not enable_rife else total_src - 1, start)

    # if cancelled, still close ffmpeg cleanly
    try:
        video.stdin.close()
    except Exception:
        pass
    video.wait()

    # final progress / status
    final_total = total_src if not enable_rife else total_src - 1
    update_progress(final_total, final_total, start)
    try:
        status_label.after(0, lambda: status_label.configure(text="✅ Processing Complete!"))
    except Exception:
        pass



def _encoder_args(codec: str, width: int, height: int):
    codec = (codec or "").lower()

    if "nvenc" in codec:  # NVIDIA
        # Good quality/speed balance: CQ with moderate preset
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p4",             # p1 best quality .. p7 fastest
            "-tune", "hq",
            "-rc", "vbr",                # or "constqp" for fixed QP
            "-cq", "19",                 # ~CRF-like; lower=better
            "-rc-lookahead", "20",
            "-bf:v", "3",
            "-b_ref_mode", "middle",
            "-pix_fmt", "yuv420p"
        ]
    else:  # libx264
        return [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",                # quality target
            "-pix_fmt", "yuv420p"
        ]

def start_ffmpeg_writer(output_path, width, height, fps, codec):
    base = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-"
    ]
    enc = _encoder_args(codec, width, height)
    command = base + enc + [output_path]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


