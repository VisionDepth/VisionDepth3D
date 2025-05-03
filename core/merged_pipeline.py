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

# ✅ ONNX Execution Provider fallback logic
available_providers = ort.get_available_providers()

if "TensorrtExecutionProvider" in available_providers:
    device = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
elif "CUDAExecutionProvider" in available_providers:
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
    progress_bar["value"] = (done / total) * 100
    elapsed = time.time() - start
    fps = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / fps if fps > 0 else float("inf")
    eta_fmt = time.strftime("%H:%M:%S", time.gmtime(eta)) if eta != float("inf") else "--:--"
    status_label["text"] = f"Progress: {done}/{total} | FPS: {fps:.2f} | ETA: {eta_fmt}"

def select_video_and_generate_frames(set_folder_callback=None):
    video_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")]
    )
    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "❌ Unable to open selected video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join("frames", f"{base_name}_frames")
    os.makedirs(output_folder, exist_ok=True)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f"frame_{i:05d}.png"), frame)

    cap.release()
    messagebox.showinfo("Done", f"✅ Extracted {total_frames} frames to:\n{output_folder}")
    if set_folder_callback:
        set_folder_callback(output_folder)

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
    
def run_esrgan(frame, blend_mode="OFF", input_res_pct=100, model_name="RealESR_Gx4_fp16", target_size=None):
    global esrgan_session
    if not esrgan_session:
        return frame

    original = frame.copy()
    if input_res_pct != 100:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * input_res_pct / 100), int(h * input_res_pct / 100)), interpolation=cv2.INTER_CUBIC)

    tensor = preprocess_esr(frame)
    try:
        output = esrgan_session.run(None, {esrgan_session.get_inputs()[0].name: tensor})[0]
        upscaled = postprocess_esr(output)

        scale = 2 if "x2" in model_name.lower() else 4
        upscaled = cv2.resize(upscaled, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        upscaled = cv2.resize(upscaled, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)

        # 🔁 Force to output resolution if specified
        if target_size:
            upscaled = cv2.resize(upscaled, target_size, interpolation=cv2.INTER_CUBIC)

        return blend_images(original, upscaled, mode=blend_mode)

    except Exception as e:
        print(f"❌ ESRGAN failed: {e}")
        return original

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

    total = len(files)
    frame_count = total - 1 if enable_rife else total

    for i in range(frame_count):
        if cancel_flag.is_set():
            break

        f1 = cv2.imread(files[i])
        f1 = run_esrgan(f1, blend_mode, input_res_pct, target_size=(width, height)) if enable_upscale else cv2.resize(f1, (width, height))
        video.stdin.write(f1.tobytes())

        if enable_rife and i + 1 < total:
            f2 = cv2.imread(files[i + 1])
            f2 = run_esrgan(f2, blend_mode, input_res_pct, target_size=(width, height)) if enable_upscale else cv2.resize(f2, (width, height))
            interpolated = run_rife(f1, f2, fps_mult)
            for frame in interpolated:
                video.stdin.write(frame.tobytes())

        update_progress(i + 1, frame_count, start)

    if not enable_rife:
        last_frame = cv2.imread(files[-1])
        last_frame = run_esrgan(last_frame, blend_mode, input_res_pct, target_size=(width, height)) if enable_upscale else cv2.resize(last_frame, (width, height))
        video.stdin.write(last_frame.tobytes())

    video.stdin.close()
    video.wait()

    update_progress(frame_count, frame_count, start)
    status_label["text"] = "\u2705 Processing Complete!"


def start_ffmpeg_writer(output_path, width, height, fps, codec):
    command = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", codec,
        "-preset", "p5" if "nvenc" in codec else "medium",
        "-b:v", "10M",
        "-maxrate", "20M",
        "-bufsize", "40M",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)

