import os
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import onnxruntime as ort
import re

# Global flags
suspend_flag = threading.Event()
cancel_flag = threading.Event()

# GUI Progress
upscale_progress = None
upscale_status_label = None

available_providers = ort.get_available_providers()
device = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]
print(f"✅ Using ONNX Execution Provider: {device}")

session_options = ort.SessionOptions()
session_options.log_severity_level = 3

MODEL_PATH = os.path.join("weights", "RealESR_Gx4_fp16.onnx")
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Real-ESRGAN model not found at {MODEL_PATH}")
    sys.exit(1)

print(f"✅ Loading Real-ESRGAN ONNX model from: {MODEL_PATH}")
esr_session = ort.InferenceSession(MODEL_PATH, sess_options=session_options, providers=device)

esrgan_session = esr_session

def preprocess_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

def postprocess_output(output):
    output = np.squeeze(output, axis=0)
    output = np.transpose(output, (1, 2, 0))
    output = np.clip(output, 0, 1) * 255.0
    return cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)

def blend_images(original, upscaled, mode="OFF"):
    if mode == "OFF":
        return upscaled
    alpha_map = {"LOW": 0.85, "MEDIUM": 0.5, "HIGH": 0.25}
    alpha = alpha_map.get(mode.upper(), 1.0)
    return cv2.addWeighted(upscaled, alpha, original, 1 - alpha, 0)

def update_progress(processed, total, start_time):
    if upscale_progress:
        upscale_progress["value"] = (processed / total) * 100

    elapsed = time.time() - start_time
    fps = processed / elapsed if elapsed > 0 else 0
    eta = (total - processed) / fps if fps > 0 else float("inf")
    eta_fmt = time.strftime("%H:%M:%S", time.gmtime(eta)) if eta != float("inf") else "--:--"
    if upscale_status_label:
        upscale_status_label["text"] = f"Upscaled: {processed}/{total} | FPS: {fps:.2f} | ETA: {eta_fmt}"

def upscale_frames(
    frames_folder,
    output_path,
    output_width,
    output_height,
    fps,
    codec_str,
    esrgan_session,
    progress_widget,
    status_label_widget,
    batch_size=1,
    input_res_pct=100,
    blend_mode="OFF",
    save_frames_only=False,
    generate_video=True,
    save_frames_folder="upscaled_frames"
):

    global upscale_progress, upscale_status_label
    upscale_progress = progress_widget
    upscale_status_label = status_label_widget
    upscale_progress["value"] = 0
    upscale_status_label["text"] = "🚀 Starting batch upscale..."

    BATCH_SIZE = batch_size

    def natural_sort_key(filename):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

    files = sorted([
        os.path.join(frames_folder, f)
        for f in os.listdir(frames_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ], key=lambda x: natural_sort_key(os.path.basename(x)))

    if not files:
        messagebox.showerror("Error", "❌ No image frames found!")
        return

    if generate_video:
        codec = cv2.VideoWriter_fourcc(*codec_str)
        out = cv2.VideoWriter(output_path, codec, fps, (output_width, output_height))

    if save_frames_only:
        os.makedirs(save_frames_folder, exist_ok=True)

    total = len(files)
    start_time = time.time()

    for i in range(0, total, BATCH_SIZE):
        if cancel_flag.is_set():
            break
        while suspend_flag.is_set():
            time.sleep(0.5)

        batch_files = files[i:i + BATCH_SIZE]
        input_batch = []
        original_frames = []

        for filepath in batch_files:
            frame = cv2.imread(filepath)
            if frame is None:
                print(f"⚠️ Skipping unreadable file: {filepath}")
                continue
            original_frames.append(frame)
            tensor = preprocess_image(frame)
            input_batch.append(tensor)

        if not input_batch:
            continue

        batch_input = np.vstack(input_batch).astype(np.float32)

        try:
            onnx_input = {esrgan_session.get_inputs()[0].name: batch_input}
            batch_output = esrgan_session.run(None, onnx_input)[0]
        except Exception as e:
            print(f"❌ ONNX Runtime Error: {e}")
            break

        for j, output in enumerate(batch_output):
            result = postprocess_output(np.expand_dims(output, axis=0))
            original = original_frames[j]
            if original is not None and result.shape == original.shape:
                result = blend_images(original, result, mode=blend_mode)

            if (result.shape[1], result.shape[0]) != (output_width, output_height):
                result = cv2.resize(result, (output_width, output_height), interpolation=cv2.INTER_CUBIC)

            if generate_video:
                out.write(result)

            if save_frames_only:
                output_filename = f"frame_{i + j:06d}.jpeg"
                cv2.imwrite(os.path.join(save_frames_folder, output_filename), result)

        update_progress(i + len(batch_files), total, start_time)

    if generate_video:
        out.release()

    upscale_status_label["text"] = "✅ Batch upscaling complete!"
    print("🎉 All frames upscaled and outputs generated!")
