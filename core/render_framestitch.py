import os
import re
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import cv2
from tqdm import tqdm
import onnxruntime as ort

#Global flags
suspend_flag = threading.Event()
cancel_flag = threading.Event()

# Available codecs (Fastest first)
CODECS = {
    "XVID (Good Compatibility)": "XVID",
    "MJPG (Motion JPEG)": "MJPG",
    "MP4V (Standard MPEG-4)": "MP4V",
    "DIVX (Older Compatibility)": "DIVX",
}

# ✅ Load ONNX models once at startup
available_providers = ort.get_available_providers()
device = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]
print(f"✅ Using ONNX Execution Providers: {device}")

# ✅ Set SessionOptions to reduce noise
session_options = ort.SessionOptions()
session_options.log_severity_level = 3  # 0 = VERBOSE, 1 = INFO, 2 = WARNING, 3 = ERROR

weights_path = "weights/"
RIFE_MODEL_PATH = os.path.join(weights_path, "RIFE_fp32.onnx")

if not os.path.exists(RIFE_MODEL_PATH):
    print(f"❌ ERROR: Model file not found at {RIFE_MODEL_PATH}")
    sys.exit(1)

print(f"✅ Loading ONNX model from {RIFE_MODEL_PATH} on {device}")
rife_session = ort.InferenceSession(RIFE_MODEL_PATH, sess_options=session_options, providers=device)

# ✅ Progress Bar and ETA
vdstitch_progress = None
vdstitch_status_label = None

def update_progress(progress, total, start_time):
    if vdstitch_progress:
        vdstitch_progress["value"] = (progress / total) * 100
    
    elapsed_time = time.time() - start_time
    current_fps = progress / elapsed_time if elapsed_time > 0 else 0
    remaining_frames = total - progress
    eta_seconds = (remaining_frames / current_fps) if current_fps > 0 else float("inf")
    eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if eta_seconds != float("inf") else "N/A"
    
    if vdstitch_status_label:
        vdstitch_status_label["text"] = f"Processed: {progress}/{total} | FPS: {current_fps:.2f} | ETA: {eta_formatted}"

def select_video_and_generate_frames():
    video_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov")],
        title="Select a video to extract frames"
    )
    
    if not video_path:
        print("❌ No video selected.")
        return

    # Automatically name the output folder based on the video file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📽 Extracting frames from: {video_path}")
    print(f"💾 Saving to folder: {output_folder}")
    
    frame_number = 0
    with tqdm(total=total_frames, desc="Extracting Frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(output_folder, f"frame_{frame_number:04d}.jpeg")
            cv2.imwrite(frame_path, frame)
            frame_number += 1
            pbar.update(1)

    cap.release()
    print("✅ Frame extraction complete.")


def select_frames_folder(frames_folder_var):
    folder = filedialog.askdirectory()
    if folder:
        frames_folder_var.set(folder)


def select_output_file(output_file_var):
    file = filedialog.asksaveasfilename(
        defaultextension=".mkv",
        filetypes=[("MKV files", "*.mkv"), ("MP4 files", "*.mp4")],
    )
    if file:
        output_file_var.set(file)

def start_processing(
    enable_fps_interpolation,
    frames_folder,
    output_file,
    width,
    height,
    fps,
    fps_multiplier,
    selected_codec,
    progress_widget,       # ✅ new
    status_label_widget    # ✅ new
):
    global vdstitch_progress, vdstitch_status_label
    vdstitch_progress = progress_widget
    vdstitch_status_label = status_label_widget

    # (optional: reset the UI before starting)
    vdstitch_progress["value"] = 0
    vdstitch_status_label["text"] = "Starting interpolation..."
    folder = frames_folder.get()
    output = output_file.get()
    codec_str = CODECS.get(selected_codec.get(), "XVID")  # ✅ Ensure codec is retrieved properly
    
    if not folder or not os.path.exists(folder):
        messagebox.showerror("Error", "Please select a valid frames folder!")
        return
    
    if not output:
        messagebox.showerror("Error", "Please specify an output file!")
        return
    
    threading.Thread(
        target=process_video3,
        args=(
            bool(enable_fps_interpolation.get()),
            folder,
            output,
            int(width.get()),
            int(height.get()),
            float(fps.get()),
            int(fps_multiplier.get()),
            codec_str
        ),
        daemon=True,
    ).start()


def extract_frame_number(filename):
    match = re.search(r"(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else float("inf")

def concatenate_images(frame1, frame2):
    frame1 = frame1.astype(np.float32) / 255.0
    frame2 = frame2.astype(np.float32) / 255.0
    return np.concatenate((frame1, frame2), axis=2)

def preprocess_frame(frame):
    frame = np.transpose(frame, (2, 0, 1))  # Convert HWC to CHW
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension (B, C, H, W)
    return frame.astype(np.float32)

def run_onnx_inference(image):
    onnx_input_name = rife_session.get_inputs()[0].name
    print(f"🔍 Running ONNX inference with input shape: {image.shape}")  # Debugging output
    onnx_output = rife_session.run(None, {onnx_input_name: image})
    return onnx_output[0]

def postprocess_output(onnx_output):
    onnx_output = np.squeeze(onnx_output, axis=0)  # Remove batch dimension if present
    onnx_output = np.clip(onnx_output, 0, 1)
    onnx_output = np.transpose(onnx_output, (1, 2, 0))
    return (onnx_output * 255).astype(np.uint8)

def interpolate_frames(frame1, frame2, fps_mult):
    interpolated_frames = []
    for _ in range(fps_mult - 1):
        concatenated = concatenate_images(frame1, frame2)
        processed_input = preprocess_frame(concatenated)
        onnx_output = run_onnx_inference(processed_input)
        interpolated_frames.append(postprocess_output(onnx_output))
    return interpolated_frames

def process_video3(enable_interpolation, frames_folder, output_path, width, height, original_fps, fps_mult, codec_str):
    frames = sorted(
        [os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if os.path.isfile(os.path.join(frames_folder, f))],
        key=extract_frame_number,
    )

    if not frames:
        messagebox.showerror("Error", "No frames found in the selected folder!")
        return

    codec = cv2.VideoWriter_fourcc(*codec_str)

    # ✅ Use fps_multiplier instead of manual FPS calculations
    adjusted_fps = original_fps * fps_mult if enable_interpolation else original_fps

    first_frame = cv2.imread(frames[0])
    is_color = len(first_frame.shape) == 3
    video = cv2.VideoWriter(output_path, codec, adjusted_fps, (width, height), isColor=True)

    total_frames = len(frames)
    start_time = time.time()

    for index in range(total_frames - 1):
        frame1 = cv2.imread(frames[index])
        frame2 = cv2.imread(frames[index + 1])

        frame1_resized = cv2.resize(frame1, (width, height), interpolation=cv2.INTER_LINEAR)
        frame2_resized = cv2.resize(frame2, (width, height), interpolation=cv2.INTER_LINEAR)

        video.write(frame1_resized)

        if enable_interpolation:
            interpolated_frames = interpolate_frames(frame1_resized, frame2_resized, fps_mult)
            for interpolated_frame in interpolated_frames:
                video.write(interpolated_frame)

        update_progress(index + 1, total_frames, start_time)

    video.write(cv2.resize(cv2.imread(frames[-1]), (width, height), interpolation=cv2.INTER_LINEAR))
    
    video.release()
    update_progress(total_frames, total_frames, start_time)
    print(f"✅ Processing Complete! Output FPS: {adjusted_fps}")