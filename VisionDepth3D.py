import shutil
import tkinter as tk
from tkinter import (
    ttk,
    filedialog,
    Label,
    Button,
    OptionMenu,
    StringVar,
    BooleanVar,
    Entry,
    messagebox,
)
from tqdm import tqdm
from PIL import Image, ImageTk, ImageOps
from transformers import AutoProcessor, AutoModelForDepthEstimation
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import time
import threading
import webbrowser
import json
import subprocess
import onnxruntime as ort  # ONNX Inference
from collections import deque
import matplotlib.cm as cm
from transformers import pipeline
import torch  # Ensure PyTorch is imported
import re
import os
import sys


# ✅ Get absolute path to resource (for PyInstaller compatibility)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # ✅ Corrected for PyInstaller
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


sys.path.append("modules")

# ✅ Set base paths correctly (for when running as .exe)
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# ✅ Define paths for models, assets, and weights
models_path = os.path.join(base_path, "models")
assets_path = os.path.join(base_path, "assets")
weights_path = os.path.join(base_path, "weights")

print(f"📂 Models Path: {models_path}")
print(f"📂 Assets Path: {assets_path}")
print(f"📂 Weights Path: {weights_path}")

# ✅ Properly detect ONNX execution providers
available_providers = ort.get_available_providers()
device = (
    "CUDAExecutionProvider"
    if "CUDAExecutionProvider" in available_providers
    else "CPUExecutionProvider"
)

# ✅ Load ONNX model safely
MODEL_PATH = os.path.join(weights_path, "backward_warping_model.onnx")

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
    sys.exit(1)  # ✅ Exit safely if model is missing

print(f"✅ Loading ONNX model from {MODEL_PATH} on {device}")

# ✅ Initialize ONNX session
session = ort.InferenceSession(MODEL_PATH, providers=[device])

# ✅ Extract input & output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"✅ ONNX Model Loaded Successfully!")

# ✅ Global threading flags
suspend_flag = threading.Event()  # ✅ Better for threading-based pausing
cancel_flag = threading.Event()


# ✅ Initialize `pipe` globally to prevent NameError
pipe = None  # Set to None at the start

frame_gpu = cv2.cuda_GpuMat()
depth_gpu = cv2.cuda_GpuMat()


def format_3d_output(left_frame, right_frame, output_format):
    """Formats the 3D output according to the user's selection."""
    height, width = left_frame.shape[:2]

    if output_format == "Full-SBS":
        return np.hstack((left_frame, right_frame))  # Full-SBS (3840x1080)

    elif output_format == "Half-SBS":
        left_resized = cv2.resize(
            left_frame, (width // 2, height), interpolation=cv2.INTER_LANCZOS4
        )
        right_resized = cv2.resize(
            right_frame, (width // 2, height), interpolation=cv2.INTER_LANCZOS4
        )
        return np.hstack((left_resized, right_resized))  # Half-SBS (1920x1080)

    elif output_format == "Full-OU":
        return np.vstack((left_frame, right_frame))  # 1920x2160

    elif output_format == "Half-OU":
        left_resized = cv2.resize(
            left_frame, (width, height // 2), interpolation=cv2.INTER_LANCZOS4
        )
        right_resized = cv2.resize(
            right_frame, (width, height // 2), interpolation=cv2.INTER_LANCZOS4
        )
        return np.vstack((left_resized, right_resized))  # Half-OU (1920x1080)

    elif output_format == "VR180":
        left_resized = cv2.resize(
            left_frame, (1440, 1600), interpolation=cv2.INTER_LANCZOS4
        )
        right_resized = cv2.resize(
            right_frame, (1440, 1600), interpolation=cv2.INTER_LANCZOS4
        )
        return np.vstack((left_resized, right_resized))  # VR180 (2880x1600)

    elif output_format == "VR360":
        left_resized = cv2.resize(
            left_frame, (1920, 960), interpolation=cv2.INTER_LANCZOS4
        )
        right_resized = cv2.resize(
            right_frame, (1920, 960), interpolation=cv2.INTER_LANCZOS4
        )
        return np.vstack((left_resized, right_resized))  # VR360 (3840x960)

    else:
        print(f"⚠ Unknown output format '{output_format}', defaulting to SBS.")
        return np.hstack((left_frame, right_frame))  # Default to Full-SBS


def generate_anaglyph_3d(left_frame, right_frame):
    """Creates a properly balanced True Red-Cyan Anaglyph 3D effect."""

    # Convert frames to float to prevent overflow during merging
    left_frame = left_frame.astype(np.float32)
    right_frame = right_frame.astype(np.float32)

    # Extract color channels
    left_r, left_g, left_b = cv2.split(left_frame)
    right_r, right_g, right_b = cv2.split(right_frame)

    # Merge the corrected Red-Cyan channels (based on optimized anaglyph conversion)
    anaglyph = cv2.merge(
        [
            right_b * 0.6,
            right_g * 0.7,
            left_r * 0.9,  # ✅ Slight reduction to balance intensity
        ]
    )

    # Clip values to ensure valid pixel range
    anaglyph = np.clip(anaglyph, 0, 255).astype(np.uint8)

    return anaglyph


def apply_aspect_ratio_crop(frame, aspect_ratio, is_full_ou=False):
    """Crops the frame to the selected aspect ratio while maintaining width.
    If is_full_ou is True, handles Full-OU stacking correctly.
    """
    height, width = frame.shape[:2]

    # ✅ Debug: Print original frame size
    print(f"🟢 Original Frame Size: {width}x{height}")

    if is_full_ou:
        # ✅ Split into two halves
        half_height = height // 2
        top_half = frame[:half_height, :]
        bottom_half = frame[half_height:, :]

        # ✅ Crop both halves separately
        top_cropped = apply_aspect_ratio_crop(top_half, aspect_ratio, is_full_ou=False)
        bottom_cropped = apply_aspect_ratio_crop(
            bottom_half, aspect_ratio, is_full_ou=False
        )

        # ✅ Stack them back together
        cropped_frame = np.vstack((top_cropped, bottom_cropped))
    else:
        # ✅ Calculate the correct target height
        target_height = int(width / aspect_ratio)
        if target_height >= height:
            print("✅ No cropping needed. Returning original frame.")
            return frame  # No cropping needed

        crop_y = (height - target_height) // 2
        cropped_frame = frame[crop_y : crop_y + target_height, :]

        # ✅ Debug: Show cropped frame size
        print(
            f"✅ Cropped Frame Size: {width}x{target_height} (Aspect Ratio: {aspect_ratio})"
        )

    # ✅ Ensure correct final resizing
    final_frame = cv2.resize(
        cropped_frame, (width, cropped_frame.shape[0]), interpolation=cv2.INTER_AREA
    )

    # ✅ Debug: Check resized frame size
    print(f"🔵 Resized Frame Size: {final_frame.shape[1]}x{final_frame.shape[0]}")

    return final_frame


# Initialize a history buffer to track black bar detections
black_bar_history = deque(maxlen=10)  # Stores the last 10 frames
history_threshold = 5  # Require at least 5 consecutive detections before cropping


def remove_black_bars(frame, reference_crop=None):
    """Removes black bars from the frame consistently using the first frame's crop values."""
    if reference_crop is not None:
        x, y, w, h = reference_crop
        return frame[y : y + h, x : x + w], reference_crop

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("⚠ No black bars detected. Returning original frame.")
        return frame, (0, 0, frame.shape[1], frame.shape[0])

    # ✅ Largest contour is assumed to be the actual frame
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return frame[y : y + h, x : x + w], (x, y, w, h)


def remove_white_edges(image):
    mask = (image[:, :, 0] > 240) & (image[:, :, 1] > 240) & (image[:, :, 2] > 240)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)

    # Fill white regions with a median blur
    blurred = cv2.medianBlur(image, 5)
    image[mask.astype(bool)] = blurred[mask.astype(bool)]
    return image


def correct_convergence_shift(
    left_frame,
    right_frame,
    depth_map,
    session,
    input_name,
    output_name,
    bg_threshold=3.0,
):
    """Optimized: Uses CUDA for warping, artifact removal, and halo reduction."""

    # ✅ Upload depth map to CUDA
    depth_gpu = cv2.cuda_GpuMat()
    depth_gpu.upload(depth_map.astype(np.float32))

    # ✅ Apply Edge-Preserving Filtering (Reduces halos)
    depth_gpu = cv2.cuda.bilateralFilter(depth_gpu, 9, 75, 75)

    # ✅ Normalize depth map
    depth_gpu = cv2.cuda.normalize(
        depth_gpu, alpha=0.1, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # ✅ Convert back to CPU for ONNX inference
    depth_normalized = depth_gpu.download()

    # ✅ Run ONNX model to get warp matrix
    warp_input = np.array([[np.mean(depth_normalized)]], dtype=np.float32)
    warp_params = (
        session.run([output_name], {input_name: warp_input})[0]
        .reshape(3, 3)
        .astype(np.float32)
    )

    h, w = left_frame.shape[:2]

    # ✅ Convert frames to CUDA
    left_gpu = cv2.cuda_GpuMat()
    right_gpu = cv2.cuda_GpuMat()
    left_gpu.upload(left_frame)
    right_gpu.upload(right_frame)

    # ✅ Apply warping
    corrected_left_gpu = cv2.cuda.warpPerspective(
        left_gpu, warp_params, (w, h), flags=cv2.INTER_CUBIC
    )
    corrected_right_gpu = cv2.cuda.warpPerspective(
        right_gpu, warp_params, (w, h), flags=cv2.INTER_CUBIC
    )

    # ✅ Convert back to CPU for inpainting
    corrected_left = corrected_left_gpu.download()
    corrected_right = corrected_right_gpu.download()

    # ✅ Detect black pixels and halos for inpainting
    mask_left = cv2.inRange(corrected_left, (0, 0, 0), (10, 10, 10))
    mask_right = cv2.inRange(corrected_right, (0, 0, 0), (10, 10, 10))

    edges_left = cv2.Canny(depth_normalized.astype(np.uint8), 30, 100)
    edges_right = cv2.Canny(depth_normalized.astype(np.uint8), 30, 100)

    kernel = np.ones((5, 5), np.uint8)
    edges_left = cv2.dilate(edges_left, kernel, iterations=2)
    edges_right = cv2.dilate(edges_right, kernel, iterations=2)

    mask_left = cv2.bitwise_or(mask_left, edges_left)
    mask_right = cv2.bitwise_or(mask_right, edges_right)

    # ✅ Apply inpainting with a larger radius
    inpainted_left = cv2.inpaint(
        corrected_left, mask_left, inpaintRadius=15, flags=cv2.INPAINT_NS
    )
    inpainted_right = cv2.inpaint(
        corrected_right, mask_right, inpaintRadius=15, flags=cv2.INPAINT_NS
    )

    return inpainted_left, inpainted_right


def render_sbs_3d(
    input_video,
    depth_video,
    output_video,
    codec,
    fps,
    width,
    height,
    fg_shift,
    mg_shift,
    bg_shift,
    sharpness_factor,
    output_format,
    aspect_ratio_value,
    delay_time=1 / 30,
    blend_factor=0.5,
    progress=None,
    progress_label=None,
    suspend_flag=None,
    cancel_flag=None,
):
    frame_delay = int(fps * delay_time)
    frame_buffer = []
    original_cap = cv2.VideoCapture(input_video)
    depth_cap = cv2.VideoCapture(depth_video)

    # 🔥 Attempt to read the first frame to determine height based on aspect ratio
    ret, first_frame = original_cap.read()
    if not ret:
        print("❌ Error: Unable to read the first frame from the input video.")
        return

    # ✅ Aspect ratio crop to determine final height
    aspect_ratio_value = aspect_ratios.get(selected_aspect_ratio.get(), 16 / 9)
    cropped_frame = apply_aspect_ratio_crop(first_frame, aspect_ratio_value)
    height = cropped_frame.shape[0]  # Update height based on cropped frame

    # 🎬 Initialize VideoWriter with correct height (before loop)
    output_width = width if output_format == "Half-SBS" else width * 2

    # ✅ Updated codec map including lossless & standard options
    codec_map = {
        "mp4v": cv2.VideoWriter_fourcc(*"mp4v"),  # Standard MPEG-4
        "XVID": cv2.VideoWriter_fourcc(*"XVID"),  # XviD (Good for AVI format)
        "DIVX": cv2.VideoWriter_fourcc(*"DIVX"),  # DivX (Older AVI format)
        # 🔹 Lossless Codecs
        "FFV1": cv2.VideoWriter_fourcc(*"FFV1"),  # FFmpeg Lossless (Best quality)
        "H264": cv2.VideoWriter_fourcc(*"avc1"),  # H.264 Lossless Mode (CPU-based)
        "HEVC": cv2.VideoWriter_fourcc(*"HEVC"),  # H.265 Lossless Mode (CPU-based)
        "LAGS": cv2.VideoWriter_fourcc(*"LAGS"),  # Lagarith Lossless
        "ULRG": cv2.VideoWriter_fourcc(*"ULRG"),  # UT Video Lossless
        "MJPG": cv2.VideoWriter_fourcc(*"MJPG"),  # Near-lossless Motion JPEG
    }

    # ✅ Validate and select codec
    if codec not in codec_map:
        print(f"⚠ Warning: Unknown codec '{codec}', defaulting to 'mp4v'")
        codec = "mp4v"  # Default to MPEG-4

    # ✅ Use the mapped FourCC code for OpenCV VideoWriter
    out = cv2.VideoWriter(
        output_video,
        codec_map[codec],
        fps,
        (width * 2 if output_format == "Full-SBS" else width, height),
    )

    if not original_cap.isOpened() or not depth_cap.isOpened():
        print("❌ Error: Unable to open input or depth video.")
        return

    # Reset video capture to the beginning after reading the first frame
    original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    depth_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()
    prev_time = start_time
    fps_values = []  # Store last few FPS values for smoothing
    reference_crop = None

    for frame_idx in range(total_frames):
        if cancel_flag and cancel_flag.is_set():
            print("❌ Rendering canceled.")
            break

        while suspend_flag and suspend_flag.is_set():
            print("⏸ Rendering paused...")
            time.sleep(0.5)

        ret1, original_frame = original_cap.read()
        ret2, depth_frame = depth_cap.read()
        if not ret1 or not ret2:
            break

        percentage = (frame_idx / total_frames) * 100
        elapsed_time = time.time() - start_time

        # ✅ **Calculate FPS with Moving Average**
        curr_time = time.time()
        frame_time = curr_time - prev_time  # Time taken for one frame

        if frame_time > 0:
            fps_calc = 1.0 / frame_time  # FPS based on actual frame time
            fps_values.append(fps_calc)

        if len(fps_values) > 10:  # Keep last 10 FPS values for smoothing
            fps_values.pop(0)

        avg_fps = (
            sum(fps_values) / len(fps_values) if fps_values else 0
        )  # Compute average FPS

        # ✅ **Update Progress Bar and FPS Display**
        if progress:
            progress["value"] = percentage
            progress.update()
        if progress_label:
            progress_label.config(
                text=f"{percentage:.2f}% | FPS: {avg_fps:.2f} | Elapsed: {time.strftime('%M:%S', time.gmtime(elapsed_time))}"
            )

        prev_time = curr_time  # Update previous frame time

        # ✅ Consistent black bar removal - handled once & reused
        if reference_crop is None:
            original_frame, reference_crop = remove_black_bars(original_frame)
        else:
            x, y, w, h = reference_crop
            original_frame = original_frame[y : y + h, x : x + w]

        frame_resized = cv2.resize(
            original_frame, (width, height), interpolation=cv2.INTER_AREA
        )
        depth_frame_resized = cv2.resize(depth_frame, (width, height))

        left_frame, right_frame = frame_resized.copy(), frame_resized.copy()

        # Pulfrich effect adjustments
        blend_factor = (
            min(0.5, blend_factor + 0.05) if len(frame_buffer) else blend_factor
        )

        # ✅ Upload depth frame to GPU
        depth_gpu = cv2.cuda_GpuMat()
        depth_gpu.upload(depth_frame)

        # ✅ Convert to grayscale (CUDA version)
        depth_gpu = cv2.cuda.cvtColor(depth_gpu, cv2.COLOR_BGR2GRAY)

        # ✅ Convert to float32 (CUDA version)
        depth_gpu = depth_gpu.convertTo(cv2.CV_32F)

        # ✅ Normalize Depth (Ensures values are within expected range)
        depth_gpu = cv2.cuda.normalize(
            depth_gpu, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

        # ✅ Apply CUDA Box Filter (Replaces Bilateral Filter)
        box_filter = cv2.cuda.createBoxFilter(depth_gpu.type(), -1, (5, 5))
        depth_gpu = box_filter.apply(depth_gpu)

        # ✅ Apply CUDA Gaussian Blur (For smoother transitions)
        gaussian_filter = cv2.cuda.createGaussianFilter(depth_gpu.type(), -1, (5, 5), 0)
        depth_gpu = gaussian_filter.apply(depth_gpu)

        # ✅ Download processed depth map for CPU-based remapping
        depth_cpu = depth_gpu.download()

        # ✅ Resize depth_cpu to match the cropped frame size
        depth_cpu_resized = cv2.resize(
            depth_cpu, (width, height), interpolation=cv2.INTER_AREA
        )

        # ✅ Compute shift values using the resized depth map
        shift_vals_fg = (-depth_cpu_resized * fg_shift).astype(np.float32)
        shift_vals_mg = (-depth_cpu_resized * mg_shift).astype(np.float32)
        shift_vals_bg = (depth_cpu_resized * bg_shift).astype(np.float32)

        # ✅ Ensure new_x_left and new_x_right have the correct shape (height, width)
        x_coords = np.tile(
            np.arange(width), (height, 1)
        )  # Create a 2D array of x-coordinates

        new_x_left = np.clip(
            x_coords + shift_vals_fg + shift_vals_mg + shift_vals_bg, 0, width - 1
        )
        new_x_right = np.clip(
            x_coords - shift_vals_fg - shift_vals_mg - shift_vals_bg, 0, width - 1
        )

        # ✅ Generate y-coordinate mapping
        map_y = np.repeat(np.arange(height).reshape(-1, 1), width, axis=1).astype(
            np.float32
        )

        map_x_left = new_x_left.astype(np.float32)  # No need to reshape anymore
        map_x_right = new_x_right.astype(np.float32)

        # ✅ CUDA Remap Implementation
        frame_gpu = cv2.cuda_GpuMat()
        frame_gpu.upload(frame_resized)

        map_x_left_gpu = cv2.cuda_GpuMat()
        map_x_right_gpu = cv2.cuda_GpuMat()
        map_y_gpu = cv2.cuda_GpuMat()

        map_x_left_gpu.upload(map_x_left)
        map_x_right_gpu.upload(map_x_right)
        map_y_gpu.upload(map_y)

        # ✅ CUDA-Based Remapping
        left_frame_gpu = cv2.cuda.remap(
            frame_gpu, map_x_left_gpu, map_y_gpu, interpolation=cv2.INTER_CUBIC
        )
        right_frame_gpu = cv2.cuda.remap(
            frame_gpu, map_x_right_gpu, map_y_gpu, interpolation=cv2.INTER_CUBIC
        )

        # ✅ Download Processed Frames
        left_frame = left_frame_gpu.download()
        right_frame = right_frame_gpu.download()

        # Buffer logic
        frame_buffer.append((left_frame, right_frame))
        if len(frame_buffer) > frame_delay:
            delayed_left_frame, delayed_right_frame = frame_buffer.pop(0)
        else:
            delayed_left_frame, delayed_right_frame = left_frame, right_frame

        # Create Pulfrich effect
        blended_left_frame = cv2.addWeighted(
            delayed_left_frame, blend_factor, left_frame, 1 - blend_factor, 0
        )
        sharpen_kernel = np.array(
            [[0, -1, 0], [-1, 5 + float(sharpness_factor), -1], [0, -1, 0]]
        )
        left_sharp = cv2.filter2D(blended_left_frame, -1, sharpen_kernel)
        right_sharp = cv2.filter2D(right_frame, -1, sharpen_kernel)

        left_sharp_resized = cv2.resize(left_sharp, (width // 2, height))
        right_sharp_resized = cv2.resize(right_sharp, (width // 2, height))

        sbs_frame = format_3d_output(left_sharp, right_sharp, output_format)

        # ✅ Dynamically check expected size based on output format
        expected_width = (
            width * 2 if output_format == "Full-SBS" else width
        )  # 7680 for Full-SBS
        expected_height = height  # Height stays the same

        h, w = sbs_frame.shape[:2]
        if (w, h) != (expected_width, expected_height):
            print(
                f"⚠ Warning: Frame size mismatch! Expected: {expected_width}x{expected_height}, Got: {w}x{h}"
            )
            sbs_frame = cv2.resize(
                sbs_frame,
                (expected_width, expected_height),
                interpolation=cv2.INTER_AREA,
            )

        # ✅ Write frame and track success
        try:
            out.write(sbs_frame)
        except Exception as e:
            print(f"❌ Error while writing frame: {e}")

    original_cap.release()
    depth_cap.release()
    out.release()
    print("3D video generated successfully.")


def render_ou_3d(
    input_video,
    depth_video,
    output_video,
    codec,
    fps,
    width,
    height,
    fg_shift,
    mg_shift,
    bg_shift,
    sharpness_factor,
    output_format,
    aspect_ratio_value,
    delay_time=1 / 30,
    blend_factor=0.5,
    progress=None,
    progress_label=None,
    suspend_flag=None,
    cancel_flag=None,
):
    frame_delay = int(fps * delay_time)
    frame_buffer = []
    original_cap = cv2.VideoCapture(input_video)
    depth_cap = cv2.VideoCapture(depth_video)

    # 🔥 Read first frame for aspect ratio analysis
    ret, first_frame = original_cap.read()
    if not ret:
        print("❌ Error: Unable to read the first frame.")
        return

    # ✅ Get aspect ratio & resize first frame
    aspect_ratio_value = aspect_ratios.get(selected_aspect_ratio.get(), 16 / 9)
    cropped_frame = apply_aspect_ratio_crop(first_frame, aspect_ratio_value)
    height = cropped_frame.shape[0]

    # ✅ Set output height correctly for Half-OU
    output_height = height * 2 if output_format == "Full-OU" else height

    # ✅ Select correct codec
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, output_height))

    if not out.isOpened():
        print("❌ ERROR: VideoWriter failed to initialize. Check codec or resolution.")
        return

    total_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reference_crop = None
    start_time, prev_time = time.time(), time.time()
    fps_values = []  # Store last few FPS values for smoothing

    for frame_idx in range(total_frames):
        if cancel_flag and cancel_flag.is_set():
            print("❌ Rendering canceled.")
            break
        while suspend_flag and suspend_flag.is_set():
            print("⏸ Rendering paused...")
            time.sleep(0.5)

        ret1, original_frame = original_cap.read()
        ret2, depth_frame = depth_cap.read()
        if not ret1 or not ret2:
            break

        percentage = (frame_idx / total_frames) * 100
        elapsed_time = time.time() - start_time

        # ✅ Calculate FPS
        curr_time = time.time()
        frame_time = curr_time - prev_time
        if frame_time > 0:
            fps_calc = 1.0 / frame_time
            fps_values.append(fps_calc)
        if len(fps_values) > 10:
            fps_values.pop(0)
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

        # ✅ Update progress bar and FPS display
        if progress:
            progress["value"] = percentage
            progress.update()
        if progress_label:
            progress_label.config(
                text=f"{percentage:.2f}% | FPS: {avg_fps:.2f} | Elapsed: {time.strftime('%M:%S', time.gmtime(elapsed_time))}"
            )
        prev_time = curr_time

        # ✅ Remove black bars
        if reference_crop is None:
            original_frame, reference_crop = remove_black_bars(original_frame)
        else:
            x, y, w, h = reference_crop
            original_frame = original_frame[y : y + h, x : x + w]

        frame_resized = cv2.resize(
            original_frame, (width, height), interpolation=cv2.INTER_AREA
        )
        depth_frame_resized = cv2.resize(depth_frame, (width, height))

        top_frame, bottom_frame = frame_resized.copy(), frame_resized.copy()

        # ✅ Upload depth frame to GPU
        depth_gpu = cv2.cuda_GpuMat()
        depth_gpu.upload(depth_frame_resized)
        depth_gpu = cv2.cuda.cvtColor(depth_gpu, cv2.COLOR_BGR2GRAY)
        depth_gpu = depth_gpu.convertTo(cv2.CV_32F)
        depth_gpu = cv2.cuda.normalize(
            depth_gpu, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

        box_filter = cv2.cuda.createBoxFilter(depth_gpu.type(), -1, (5, 5))
        depth_gpu = box_filter.apply(depth_gpu)
        gaussian_filter = cv2.cuda.createGaussianFilter(depth_gpu.type(), -1, (5, 5), 0)
        depth_gpu = gaussian_filter.apply(depth_gpu)
        depth_cpu = depth_gpu.download()
        depth_cpu_resized = cv2.resize(
            depth_cpu, (width, height), interpolation=cv2.INTER_AREA
        )

        shift_vals_fg = (-depth_cpu_resized * fg_shift).astype(np.float32)
        shift_vals_mg = (-depth_cpu_resized * mg_shift).astype(np.float32)
        shift_vals_bg = (depth_cpu_resized * bg_shift).astype(np.float32)

        map_y = np.repeat(np.arange(height).reshape(-1, 1), width, axis=1).astype(
            np.float32
        )
        new_y_top = np.clip(
            map_y + shift_vals_fg + shift_vals_mg + shift_vals_bg, 0, height - 1
        )
        new_y_bottom = np.clip(
            map_y - shift_vals_fg - shift_vals_mg - shift_vals_bg, 0, height - 1
        )

        frame_gpu = cv2.cuda_GpuMat()
        frame_gpu.upload(frame_resized)
        map_y_top_gpu, map_y_bottom_gpu = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        map_y_top_gpu.upload(new_y_top)
        map_y_bottom_gpu.upload(new_y_bottom)
        map_x_gpu = cv2.cuda_GpuMat()
        map_x_gpu.upload(np.tile(np.arange(width), (height, 1)).astype(np.float32))

        top_frame_gpu = cv2.cuda.remap(
            frame_gpu, map_x_gpu, map_y_top_gpu, interpolation=cv2.INTER_CUBIC
        )
        bottom_frame_gpu = cv2.cuda.remap(
            frame_gpu, map_x_gpu, map_y_bottom_gpu, interpolation=cv2.INTER_CUBIC
        )

        top_frame, bottom_frame = top_frame_gpu.download(), bottom_frame_gpu.download()
        sharpen_kernel = np.array(
            [[0, -1, 0], [-1, 5 + sharpness_factor, -1], [0, -1, 0]]
        )
        top_sharp = cv2.filter2D(top_frame, -1, sharpen_kernel)
        bottom_sharp = cv2.filter2D(bottom_frame, -1, sharpen_kernel)

        ou_frame = format_3d_output(top_sharp, bottom_sharp, output_format)

        expected_height = height * 2 if output_format == "Full-OU" else height
        if ou_frame.shape[:2] != (expected_height, width):
            ou_frame = cv2.resize(
                ou_frame, (width, expected_height), interpolation=cv2.INTER_AREA
            )

        out.write(ou_frame)

    original_cap.release()
    depth_cap.release()
    out.release()
    print(f"🎬 Full-OU 3D video generated successfully: {output_video}")


def start_processing_thread():
    global process_thread
    cancel_flag.clear()  # Reset cancel state
    suspend_flag.clear()  # Ensure it's not paused
    process_thread = threading.Thread(target=process_video, daemon=True)
    process_thread.start()


def select_input_video():
    video_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
    )
    if not video_path:
        return

    input_video_path.set(video_path)

    # Extract video specs
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read the first frame to generate a thumbnail
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Convert the frame to an image compatible with Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((300, 200))  # Resize thumbnail
        img_tk = ImageTk.PhotoImage(img)

        # Update the GUI
        video_thumbnail_label.config(image=img_tk)
        video_thumbnail_label.image = (
            img_tk  # Save a reference to prevent garbage collection
        )

        video_specs_label.config(
            text=f"Video Info:\nResolution: {width}x{height}\nFPS: {fps:.2f}"
        )
    else:
        video_specs_label.config(text="Video Info:\nUnable to extract details")


def update_thumbnail(thumbnail_path):
    thumbnail_image = Image.open(thumbnail_path)
    thumbnail_image = thumbnail_image.resize(
        (250, 100), Image.LANCZOS
    )  # Adjust the size as needed
    thumbnail_photo = ImageTk.PhotoImage(thumbnail_image)
    video_thumbnail_label.config(image=thumbnail_photo)
    video_thumbnail_label.image = thumbnail_photo


def select_output_video():
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


def select_depth_map():
    depth_map_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
    )
    if not depth_map_path:
        return

    selected_depth_map.set(depth_map_path)
    depth_map_label.config(
        text=f"Selected Depth Map:\n{os.path.basename(depth_map_path)}"
    )


def process_video():
    if (
        not input_video_path.get()
        or not output_sbs_video_path.get()
        or not selected_depth_map.get()
    ):
        messagebox.showerror(
            "Error", "Please select input video, depth map, and output path."
        )
        return

    cap = cv2.VideoCapture(input_video_path.get())
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        messagebox.showerror("Error", "Unable to retrieve FPS from the input video.")
        return

    progress["value"] = 0
    progress_label.config(text="0%")
    progress.update()

    # Call rendering function based on format
    if output_format.get() in ["Full-OU", "Half-OU"]:
        render_ou_3d(
            input_video_path.get(),
            selected_depth_map.get(),
            output_sbs_video_path.get(),
            selected_codec.get(),
            fps,
            width,
            height,
            fg_shift.get(),
            mg_shift.get(),
            bg_shift.get(),
            sharpness_factor.get(),
            output_format.get(),
            aspect_ratios.get(selected_aspect_ratio.get()),
            delay_time=delay_time.get(),
            blend_factor=blend_factor.get(),
            progress=progress,
            progress_label=progress_label,
            suspend_flag=suspend_flag,  # Pass suspend_flag
            cancel_flag=cancel_flag,  # Pass cancel_flag
        )
    else:
        render_sbs_3d(
            input_video_path.get(),
            selected_depth_map.get(),
            output_sbs_video_path.get(),
            selected_codec.get(),
            fps,
            width,
            height,
            fg_shift.get(),
            mg_shift.get(),
            bg_shift.get(),
            sharpness_factor.get(),
            output_format.get(),
            aspect_ratios.get(selected_aspect_ratio.get()),
            delay_time=delay_time.get(),
            blend_factor=blend_factor.get(),
            progress=progress,
            progress_label=progress_label,
            suspend_flag=suspend_flag,  # Pass suspend_flag
            cancel_flag=cancel_flag,  # Pass cancel_flag
        )

    if not cancel_flag.is_set():
        progress["value"] = 100
        progress_label.config(text="100%")
        progress.update()
        print("✅ Processing complete.")


def suspend_processing():
    """Pauses the processing loop safely."""
    suspend_flag.set()  # This will cause processing to pause
    print("⏸ Processing Suspended!")


def resume_processing():
    """Resumes the processing loop safely."""
    suspend_flag.clear()  # Processing will continue from where it left off
    print("▶ Processing Resumed!")


def cancel_processing():
    """Cancels processing completely."""
    cancel_flag.set()
    suspend_flag.clear()  # Ensure no accidental resume
    print("❌ Processing canceled.")


# Define SETTINGS_FILE at the top of the script
SETTINGS_FILE = "settings.json"


def open_github():
    """Opens the GitHub repository in a web browser."""
    webbrowser.open_new(
        "https://github.com/VisionDepth/VisionDepth3D"
    )  # Replace with your actual GitHub URL


def reset_settings():
    """Resets all sliders and settings to default values"""
    fg_shift.set(6.0)  # Default divergence shift
    mg_shift.set(3.0)  # Default depth transition
    bg_shift.set(-4.0)  # Default convergence shift
    sharpness_factor.set(0.2)
    blend_factor.set(0.6)
    delay_time.set(1 / 30)
    output_format = tk.StringVar(value="Full-SBS")

    messagebox.showinfo("Settings Reset", "All values have been restored to defaults!")


# -----------------------
# Global Variables & Setup
# -----------------------

# --- Window Setup ---
root = tk.Tk()
root.title("VisionDepth3D Video Generator")
root.geometry("1080x780")

background_image = Image.open(resource_path(os.path.join("assets", "Background.png")))
background_image = background_image.resize((1080, 780), Image.LANCZOS)
bg_image = ImageTk.PhotoImage(background_image)

root.bg_image = bg_image  # keep a persistent reference
background_label = tk.Label(root, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# ---Depth Estimation---

# ✅ Define supported Hugging Face models
supported_models = {
    "Distil-Any-Depth-Large": "keetrap/Distil-Any-Depth-Large-hf",
    "Distil-Any-Depth-Small": "keetrap/Distill-Any-Depth-Small-hf",
    "rock-depth-ai": "justinsoberano/rock-depth-ai",
    "Depth Anything V2 Large": "depth-anything/Depth-Anything-V2-Large-hf",
    "Depth Anything V2 Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth Anything V2 Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth Anything V1 Large": "LiheYoung/Depth-Anything-V2-Large",
    "Depth Anything V1 Base": "LiheYoung/depth-anything-base-hf",
    "Depth Anything V1 Small": "LiheYoung/depth-anything-small-hf",
    "V2-Metric-Indoor-Large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "V2-Metric-Outdoor-Large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "DA_vitl14": "LiheYoung/depth_anything_vitl14",
    "DA_vits14": "LiheYoung/depth_anything_vits14",
    "DepthPro": "apple/DepthPro-hf",
    "ZoeDepth": "Intel/zoedepth-nyu-kitti",
    "MiDaS 3.0": "Intel/dpt-hybrid-midas",
    "DPT-Large": "Intel/dpt-large",
    "DinoV2": "facebook/dpt-dinov2-small-kitti",
    "dpt-beit-large-512": "Intel/dpt-beit-large-512",
}

# ✅ Define local model storage path
local_model_dir = os.path.expanduser(
    "~/.cache/huggingface/models/"
)  # Correct local path

selected_model = tk.StringVar(root, value="Distil-Any-Depth-Large")
colormap_var = tk.StringVar(root, value="Default")
invert_var = tk.BooleanVar(root, value=False)
save_frames_var = tk.BooleanVar(value=False)
output_dir = ""

# -----------------------
# Functions
# -----------------------


def ensure_model_downloaded(checkpoint):
    """Ensures the model is downloaded locally before loading."""
    local_path = os.path.join(
        local_model_dir, checkpoint.replace("/", "_")
    )  # Convert to valid folder name

    if not os.path.exists(local_path):
        print(f"📥 Downloading model: {checkpoint} ... This may take a few minutes.")
        try:
            # ✅ Download model & processor to local storage
            AutoModelForDepthEstimation.from_pretrained(
                checkpoint, cache_dir=local_path
            )
            AutoProcessor.from_pretrained(checkpoint, cache_dir=local_path)
            print(f"✅ Model downloaded successfully: {checkpoint}")
        except Exception as e:
            print(f"❌ Failed to download model: {str(e)}")
            return None  # Prevent using a broken model

    return local_path  # Return the local path to be used


def update_pipeline(*args):
    """Loads the depth estimation model from local cache."""
    global pipe

    selected_checkpoint = selected_model.get()
    checkpoint = supported_models.get(selected_checkpoint, None)

    if checkpoint is None:
        status_label.config(text=f"⚠️ Error: Model '{selected_checkpoint}' not found.")
        return

    try:
        # ✅ Ensure model is available locally
        local_model_path = ensure_model_downloaded(checkpoint)
        if not local_model_path:
            status_label.config(text=f"❌ Failed to load model: {selected_checkpoint}")
            return

        # ✅ Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ Load image processor & model locally
        processor = AutoProcessor.from_pretrained(
            checkpoint, cache_dir=local_model_path
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            checkpoint, cache_dir=local_model_path
        ).to(device)

        # ✅ Load the depth-estimation pipeline
        pipe = pipeline(
            "depth-estimation", model=model, device=device, image_processor=processor
        )

        # ✅ Update status label to show only the model name
        status_label.config(
            text=f"✅ Model loaded: {selected_checkpoint} (Running on {device.upper()})"
        )
        status_label.update_idletasks()  # Force label update in Tkinter

    except Exception as e:
        status_label.config(text=f"❌ Model loading failed: {str(e)}")
        status_label.update_idletasks()  # Ensure GUI updates


def choose_output_directory():
    global output_dir
    selected_directory = filedialog.askdirectory()
    if selected_directory:
        output_dir = selected_directory
        output_dir_label.config(text=f"📁 {output_dir}")


def process_image_folder():
    """Opens a folder dialog and processes all image files inside it in a background thread."""
    folder_path = filedialog.askdirectory(title="Select Folder Containing Images")

    if not folder_path:
        status_label.config(text="⚠️ No folder selected.")
        return

    # ✅ Run processing in a separate thread
    threading.Thread(
        target=process_images_in_folder, args=(folder_path,), daemon=True
    ).start()


def process_images_in_folder(folder_path):
    """Processes images in the folder in a separate thread (to prevent GUI freezing) with FPS and ETA tracking."""

    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    ]

    if not image_files:
        root.after(
            10,
            lambda: status_label.config(
                text="⚠️ No image files found in the selected folder."
            ),
        )
        return

    total_images = len(image_files)
    root.after(
        10, lambda: status_label.config(text=f"📂 Processing {total_images} images...")
    )
    root.after(10, lambda: progress_bar.config(maximum=total_images, value=0))

    start_time = time.time()  # Track the start time

    # ✅ Process each image and track FPS / ETA
    for i, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, image_file)
        process_image(image_path, folder=True)

        elapsed_time = time.time() - start_time  # Time since start
        fps = i / elapsed_time if elapsed_time > 0 else 0  # Calculate FPS
        eta = (total_images - i) / fps if fps > 0 else 0  # Estimate time remaining

        # ✅ Update progress and status dynamically
        root.after(
            10, lambda i=i, fps=fps, eta=eta: update_progress(i, total_images, fps, eta)
        )

    # ✅ Ensure progress reaches 100%
    root.after(
        10, lambda: status_label.config(text="✅ All images processed successfully!")
    )
    root.after(10, lambda: progress_bar.config(value=progress_bar["maximum"]))


def update_progress(processed, total, fps, eta):
    """Updates the progress bar and displays frames status, FPS, and ETA."""
    progress_bar.config(value=processed)

    # Format FPS and ETA
    fps_text = f"{fps:.2f} FPS"
    eta_text = f"ETA: {int(eta)}s" if eta > 0 else "ETA: --s"
    progress_text = f"📸 Processed: {processed}/{total} | {fps_text} | {eta_text}"

    status_label.config(text=progress_text)


def process_image(file_path, folder=False):
    """Processes a single image file and saves the depth-mapped version."""
    image = Image.open(file_path)
    predictions = pipe(image)

    if "predicted_depth" in predictions:
        raw_depth = predictions["predicted_depth"]
        depth_norm = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min())
        depth_np = depth_norm.squeeze().detach().cpu().numpy()
        grayscale = (depth_np * 255).astype("uint8")

        cmap_choice = colormap_var.get()
        if cmap_choice == "Default":
            depth_image = predictions["depth"]
        else:
            cmap = cm.get_cmap(cmap_choice.lower())
            colored = cmap(grayscale / 255.0)
            colored = (colored[:, :, :3] * 255).astype(np.uint8)
            depth_image = Image.fromarray(colored)
    else:
        depth_image = predictions["depth"]

    if invert_var.get():
        print("Inversion enabled")
        depth_image = ImageOps.invert(depth_image.convert("RGB"))

    # ✅ If processing a single image, update GUI preview
    if not folder:
        image_disp = image.copy()
        image_disp.thumbnail((480, 270))
        photo_input = ImageTk.PhotoImage(image_disp)
        input_label.config(image=photo_input)
        input_label.image = photo_input

        depth_disp = depth_image.copy()
        depth_disp.thumbnail((480, 270))
        photo_depth = ImageTk.PhotoImage(depth_disp)
        output_label.config(image=photo_depth)
        output_label.image = photo_depth

    # ✅ Save output to the selected directory
    image_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{image_name}_depth.png"
    file_save_path = (
        os.path.join(output_dir, output_filename) if output_dir else output_filename
    )
    depth_image.save(file_save_path)

    if not folder:
        status_label.config(text=f"✅ Image saved: {file_save_path}")
        progress_bar.config(value=100)


def open_image():
    """Opens a file dialog to select a single image and process it."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpeg;*.jpg;*.png")]
    )
    if file_path:
        status_label.config(text="🔄 Processing image...")
        progress_bar.start(10)
        threading.Thread(
            target=lambda: [process_image(file_path), progress_bar.stop()], daemon=True
        ).start()


def process_video_folder():
    """Opens a folder dialog and processes all video files inside it in a background thread."""
    folder_path = filedialog.askdirectory(title="Select Folder Containing Videos")

    if not folder_path:
        status_label.config(text="⚠️ No folder selected.")
        return

    # ✅ Run processing in a separate thread
    thread = threading.Thread(target=process_videos_in_folder, args=(folder_path,))
    thread.start()


def natural_sort_key(filename):
    """Extract numbers from filenames for natural sorting."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", filename)
    ]


def process_videos_in_folder(folder_path):
    """Processes all video files in the selected folder in the correct numerical order."""
    video_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        status_label.config(text="⚠️ No video files found in the selected folder.")
        return

    # ✅ Sort videos numerically
    video_files.sort(key=natural_sort_key)

    status_label.config(text=f"📂 Processing {len(video_files)} videos...")

    total_frames_all = sum(
        cv2.VideoCapture(os.path.join(folder_path, f)).get(cv2.CAP_PROP_FRAME_COUNT)
        for f in video_files
    )
    frames_processed_all = 0

    # ✅ Ensure BATCH_SIZE is properly defined
    try:
        BATCH_SIZE = int(batch_size_entry.get().strip())
        if BATCH_SIZE <= 0:
            raise ValueError
    except ValueError:
        BATCH_SIZE = 8  # Default batch size
        status_label.config(text="⚠️ Invalid batch size. Using default batch size (8).")

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        frames_processed_all += process_video2(
            video_path, total_frames_all, frames_processed_all, BATCH_SIZE
        )  # ✅ Pass BATCH_SIZE

    status_label.config(text="✅ All videos processed successfully!")
    progress_bar.config(value=100)


def process_video2(file_path, total_frames_all, frames_processed_all, BATCH_SIZE):
    """Processes a single video file and updates the progress bar correctly."""

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        status_label.config(text=f"❌ Error: Cannot open {file_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_dir, input_filename = os.path.split(file_path)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_depth.mp4"
    output_path = os.path.join(output_dir, output_filename)

    print(f"📁 Saving video to: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

    if not out.isOpened():
        print(f"❌ Failed to open video writer for {output_filename}")
        return 0

    save_frames = save_frames_var.get()  # Check if saving frames is enabled
    frames_batch = []
    frame_count = 0
    start_time = time.time()

    frame_output_dir = os.path.join(output_dir, f"{name}_frames")
    if save_frames and not os.path.exists(frame_output_dir):
        os.makedirs(frame_output_dir)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames_batch.append(pil_image)

        if len(frames_batch) == BATCH_SIZE or frame_count == total_frames:
            predictions = pipe(frames_batch)

            for i, prediction in enumerate(predictions):
                raw_depth = prediction["predicted_depth"]
                depth_norm = (raw_depth - raw_depth.min()) / (
                    raw_depth.max() - raw_depth.min()
                )
                depth_np = depth_norm.squeeze().detach().cpu().numpy()
                grayscale = (depth_np * 255).astype("uint8")

                depth_frame = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
                depth_frame = cv2.resize(depth_frame, (original_width, original_height))
                out.write(depth_frame)

                if save_frames:
                    frame_filename = os.path.join(
                        frame_output_dir,
                        f"frame_{frame_count - len(predictions) + i + 1}.png",
                    )
                    cv2.imwrite(frame_filename, depth_frame)

            frames_batch.clear()

        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        frames_left = total_frames_all - (frames_processed_all + frame_count)
        estimated_time_remaining = frames_left / avg_fps if avg_fps > 0 else 0

        progress = int(((frames_processed_all + frame_count) / total_frames_all) * 100)
        progress_bar.config(value=progress)
        status_label.config(
            text=f"🎬 {frame_count}/{total_frames} frames | FPS: {avg_fps:.2f} | "
            f"ETA: {time.strftime('%M:%S', time.gmtime(estimated_time_remaining))} | "
            f"Processing: {name}"
        )
        status_label.update_idletasks()

    cap.release()
    out.release()

    print(f"✅ Video saved: {output_path}")

    return frame_count


def open_video():
    file_path = filedialog.askopenfilename(
        filetypes=[
            (
                "All Supported Video Files",
                "*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv;*.webm;*.mpeg;*.mpg",
            ),
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

    if file_path:
        status_label.config(text="🔄 Processing video...")
        progress_bar.config(mode="determinate", maximum=100, value=0)

        # ✅ Get total frame count before starting processing
        cap = cv2.VideoCapture(file_path)
        total_frames_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ✅ Ensure batch size is properly retrieved from GUI
        try:
            BATCH_SIZE = int(batch_size_entry.get().strip())
            if BATCH_SIZE <= 0:
                raise ValueError
        except ValueError:
            BATCH_SIZE = 8  # Default batch size
            status_label.config(
                text="⚠️ Invalid batch size. Using default batch size (8)."
            )

        # ✅ Start processing in a background thread
        threading.Thread(
            target=process_video2,
            args=(file_path, total_frames_all, 0, BATCH_SIZE),  # Pass all required args
            daemon=True,
        ).start()


# --- VDStitch Contents ---

# Available codecs (Fastest first)
CODECS = {
    "XVID (Good Compatibility)": "XVID",
    "MJPG (Motion JPEG)": "MJPG",
    "MP4V (Standard MPEG-4)": "MP4V",
    "DIVX (Older Compatibility)": "DIVX",
}

# Variables
frames_folder = tk.StringVar()
output_file = tk.StringVar()
width = tk.IntVar(value=640)
height = tk.IntVar(value=480)
fps = tk.DoubleVar(value=23.976)
selected_codec = tk.StringVar(value="XVID")


def select_frames_folder():
    folder = filedialog.askdirectory()
    if folder:
        frames_folder.set(folder)


def select_output_file():
    file = filedialog.asksaveasfilename(
        defaultextension=".mkv",
        filetypes=[("MKV files", "*.mkv"), ("MP4 files", "*.mp4")],
    )
    if file:
        output_file.set(file)


def start_processing():
    if not frames_folder.get() or not os.path.exists(frames_folder.get()):
        messagebox.showerror("Error", "Please select a valid frames folder!")
        return

    if not output_file.get():
        messagebox.showerror("Error", "Please specify an output file!")
        return

    # ✅ Run in a separate thread
    threading.Thread(target=process_video3, daemon=True).start()


def process_video3():
    folder = frames_folder.get()
    frames = sorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ]
    )

    if not frames:
        messagebox.showerror("Error", "No frames found in the selected folder!")
        return

    width_value, height_value = int(width.get()), int(
        height.get()
    )  # ✅ Ensure integer values
    fps_value = float(fps.get())  # ✅ Ensure float value
    codec_str = CODECS.get(selected_codec.get(), "XVID")  # ✅ Fix: No 'self'
    codec = cv2.VideoWriter_fourcc(*codec_str)

    first_frame = cv2.imread(frames[0])
    if first_frame is None:
        messagebox.showerror("Error", "Failed to load first frame. Check file format!")
        return

    is_color = len(first_frame.shape) == 3
    print(f"🖼 Frame Type: {'RGB' if is_color else 'Grayscale'}")

    video = cv2.VideoWriter(
        output_file.get(), codec, fps_value, (width_value, height_value), isColor=True
    )

    if not video.isOpened():
        messagebox.showerror(
            "Error", "Failed to open video writer. Check codec and output path!"
        )
        return

    total_frames = len(frames)
    vdstitch_progress["maximum"] = total_frames
    vdstitch_progress["value"] = 0
    vdstitch_progress.update_idletasks()

    def process_frame(frame_path):
        frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            return None

        frame_resized = cv2.resize(
            frame, (width_value, height_value), interpolation=cv2.INTER_LINEAR
        )

        # ✅ Auto-detect Grayscale and Convert to 3-Channel if needed
        if len(frame_resized.shape) == 2:  # Single-channel grayscale
            frame_resized = cv2.cvtColor(
                frame_resized, cv2.COLOR_GRAY2BGR
            )  # Convert grayscale to RGB

        return frame_resized

    def update_progress(current_frame):
        """Safely update progress bar from the main thread"""
        vdstitch_progress["value"] = current_frame
        vdstitch_progress.update_idletasks()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, frame in enumerate(executor.map(process_frame, frames), start=1):
            if frame is not None:
                video.write(frame)
                root.after(
                    10, lambda: vdstitch_progress.config(value=index)
                )  # ✅ Update VDStitch progress bar

    video.release()

    # ✅ Ensure progress reaches 100% at the end
    root.after(
        10, lambda: vdstitch_progress.config(value=vdstitch_progress["maximum"])
    )  # ✅ VDStitch bar updates
    root.after(
        10, lambda: messagebox.showinfo("Success", "Video processing completed!")
    )


# ---GUI Setup---

# --- Notebook for Tabs ---
tab_control = ttk.Notebook(root)
tab_control.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8, relheight=0.8)

# --- Depth Estimation GUI ---
depth_estimation_frame = tk.Frame(tab_control)
tab_control.add(depth_estimation_frame, text="Depth Estimation")

# Use the depth estimation tab’s content frame as the parent
depth_content_frame = tk.Frame(depth_estimation_frame, highlightthickness=0, bd=0)
depth_content_frame.pack(fill="both", expand=True)

# Sidebar Frame inside depth_content_frame
sidebar = tk.Frame(depth_content_frame, bg="#1c1c1c", width=240)
sidebar.pack(side="left", fill="y")

# Main Content Frame inside depth_content_frame
main_content = tk.Frame(depth_content_frame, bg="#2b2b2b")
main_content.pack(side="right", fill="both", expand=True)

# --- RIFE FPS Interpolation Tab (Coming Soon Placeholder) ---
VDStitch = tk.Frame(tab_control)  # Create a new frame stitch tab
tab_control.add(VDStitch, text="VDStitch")  # Add to notebook

# --- RIFE FPS Interpolation Tab (Coming Soon Placeholder) ---
# VDStitch = tk.Frame(tab_control)  # Create a new frame stitch tab
# tab_control.add(VDStitch, text="VDStitch")  # Add to notebook

# Centered Label saying "RIFE Coming Soon"
# VDStitch_placeholder_label = tk.Label(VDStitch, text="🛠️ VDStitch - Coming Soon!",
#                                  font=("Arial", 16, "bold"), fg="gray")
# VDStitch_placeholder_label.pack(expand=True)  # Center the label

# --- 3D Video Generator Tab ---
visiondepth_frame = tk.Frame(tab_control)
tab_control.add(visiondepth_frame, text="3D Video Generator")

# ✅ Same styled content_frame for VisionDepth3D tab
visiondepth_content_frame = tk.Frame(visiondepth_frame, highlightthickness=0, bd=0)
visiondepth_content_frame.pack(fill="both", expand=True)

# --- RIFE FPS Interpolation Tab (Coming Soon Placeholder) ---
rife_fps_frame = tk.Frame(tab_control)  # Create a new frame for RIFE
tab_control.add(rife_fps_frame, text="RIFE FPS Interpolation")  # Add to notebook

# Centered Label saying "RIFE Coming Soon"
rife_placeholder_label = tk.Label(
    rife_fps_frame,
    text="🛠️ RIFE FPS Interpolation - Coming Soon!",
    font=("Arial", 16, "bold"),
    fg="gray",
)
rife_placeholder_label.pack(expand=True)  # Center the label


# --- Sidebar Content ---
tk.Label(sidebar, text="🛠️ Model", bg="#1c1c1c", fg="white", font=("Arial", 11)).pack(
    pady=5
)
model_dropdown = ttk.Combobox(
    sidebar,
    textvariable=selected_model,
    values=list(supported_models.keys()),
    state="readonly",
    width=22,
)
model_dropdown.pack(pady=5)
model_dropdown.bind("<<ComboboxSelected>>", update_pipeline)

output_dir_label = tk.Label(
    sidebar, text="📂 Output Dir: None", bg="#1c1c1c", fg="white", wraplength=200
)
output_dir_label.pack(pady=5)
tk.Button(
    sidebar, text="Choose Directory", command=choose_output_directory, width=20
).pack(pady=5)

tk.Label(sidebar, text="🎨 Colormap:", bg="#1c1c1c", fg="white").pack(pady=5)
colormap_dropdown = ttk.Combobox(
    sidebar,
    textvariable=colormap_var,
    values=["Default", "Magma", "Viridis", "Inferno", "Plasma", "Gray"],
    state="readonly",
    width=22,
)
colormap_dropdown.pack(pady=5)

invert_checkbox = tk.Checkbutton(
    sidebar, text="🌑 Invert Depth", variable=invert_var, bg="#1c1c1c", fg="white"
)
invert_checkbox.pack(pady=5)

save_frames_checkbox = tk.Checkbutton(
    sidebar, text=" Save Frames", variable=save_frames_var, bg="#1c1c1c", fg="white"
)
save_frames_checkbox.pack(pady=5)

tk.Label(sidebar, text="💾 Filename (Image):", bg="#1c1c1c", fg="white").pack(pady=5)
output_name_entry = tk.Entry(sidebar, width=22)
output_name_entry.insert(0, "depth_map.png")
output_name_entry.pack(pady=5)

tk.Label(sidebar, text="📦 Batch Size (Frames):", bg="#1c1c1c", fg="white").pack(pady=5)
batch_size_entry = tk.Entry(sidebar, width=22)
batch_size_entry.insert(0, "8")
batch_size_entry.pack(pady=5)

tk.Label(sidebar, text="🖼️ Video Resolution (w,h):", bg="#1c1c1c", fg="white").pack(
    pady=5
)
resolution_entry = tk.Entry(sidebar, width=22)
resolution_entry.insert(0, "")
resolution_entry.pack(pady=5)

progress_bar = ttk.Progressbar(sidebar, mode="determinate", length=180)
progress_bar.pack(pady=10)
status_label = tk.Label(
    sidebar, text="🔋 Ready", bg="#1c1c1c", fg="white", width=30, wraplength=200
)
status_label.pack(pady=5)

depth_map_label_depth = tk.Label(
    sidebar, text="Previous Depth Map: None", justify="left", wraplength=200
)
depth_map_label_depth.pack(pady=5)

# --- Depth Content: Image previews ---
# --- Top Frame: For the original image ---
top_frame = tk.Frame(main_content, bg="#2b2b2b")
top_frame.pack(pady=10)

input_label = tk.Label(top_frame, text="🖼️ Input Image", bg="#2b2b2b", fg="white")
input_label.pack()  # No side=, so it stacks vertically

# --- Middle Frame: For the buttons ---
button_frame = tk.Frame(main_content, bg="#2b2b2b")
button_frame.pack(pady=10)

tk.Button(
    button_frame,
    text="🖼️ Process Image",
    command=open_image,
    width=25,
    bg="#4a4a4a",
    fg="white",
).pack(pady=2)
tk.Button(
    button_frame,
    text="🖼️ Process Image Folder",
    command=process_image_folder,
    width=25,
    bg="#4a4a4a",
    fg="white",
).pack(pady=2)
tk.Button(
    button_frame,
    text="🎥 Process Video",
    command=open_video,
    width=25,
    bg="#4a4a4a",
    fg="white",
).pack(pady=2)
tk.Button(
    button_frame,
    text="📂 Select Video Folder",
    command=process_video_folder,
    width=25,
    bg="#4a4a4a",
    fg="white",
).pack(pady=2)

# --- Bottom Frame: For the depth map ---
bottom_frame = tk.Frame(main_content, bg="#2b2b2b")
bottom_frame.pack(pady=10)

output_label = tk.Label(bottom_frame, text="🌊 Depth Map", bg="#2b2b2b", fg="white")
output_label.pack()


# --- FrameStitch Contents ---

# --- VDStitch Sidebar (Dark Theme) ---
VDStitch.configure(bg="#1c1c1c")  # Set background for the entire tab

tk.Label(VDStitch, text="📂 Select Frames Folder:", bg="#1c1c1c", fg="white").pack(
    anchor="w", padx=10, pady=5
)
tk.Entry(
    VDStitch,
    textvariable=frames_folder,
    width=50,
    bg="#2b2b2b",
    fg="white",
    insertbackground="white",
    relief="flat",
).pack(padx=10, pady=2)
tk.Button(
    VDStitch,
    text="Browse",
    command=select_frames_folder,
    bg="#4a4a4a",
    fg="white",
    relief="flat",
).pack(pady=2)

tk.Label(VDStitch, text="🎥 Select Output Video File:", bg="#1c1c1c", fg="white").pack(
    anchor="w", padx=10, pady=5
)
tk.Entry(
    VDStitch,
    textvariable=output_file,
    width=50,
    bg="#2b2b2b",
    fg="white",
    insertbackground="white",
    relief="flat",
).pack(padx=10, pady=2)
tk.Button(
    VDStitch,
    text="Save As",
    command=select_output_file,
    bg="#4a4a4a",
    fg="white",
    relief="flat",
).pack(pady=2)

tk.Label(
    VDStitch, text="🖼️ Resolution (Width x Height):", bg="#1c1c1c", fg="white"
).pack(anchor="w", padx=10, pady=5)
frame_res = tk.Frame(VDStitch, bg="#1c1c1c")
frame_res.pack()
tk.Entry(
    frame_res,
    textvariable=width,
    width=10,
    bg="#2b2b2b",
    fg="white",
    insertbackground="white",
    relief="flat",
).pack(side="left", padx=5)
tk.Label(frame_res, text="x", bg="#1c1c1c", fg="white").pack(side="left")
tk.Entry(
    frame_res,
    textvariable=height,
    width=10,
    bg="#2b2b2b",
    fg="white",
    insertbackground="white",
    relief="flat",
).pack(side="left", padx=5)

tk.Label(VDStitch, text="🎞 Select Codec:", bg="#1c1c1c", fg="white").pack(
    anchor="w", padx=10, pady=5
)
codec_menu = ttk.Combobox(
    VDStitch, textvariable=selected_codec, values=list(CODECS.keys()), state="readonly"
)
codec_menu.pack(padx=10, pady=2)

tk.Label(VDStitch, text="⏱ Frame Rate (FPS):", bg="#1c1c1c", fg="white").pack(
    anchor="w", padx=10, pady=5
)
tk.Entry(
    VDStitch,
    textvariable=fps,
    width=10,
    bg="#2b2b2b",
    fg="white",
    insertbackground="white",
    relief="flat",
).pack(padx=10, pady=2)

# Processing Button with Dark Theme
process_btn = tk.Button(
    VDStitch,
    text="▶ Start Processing",
    command=start_processing,
    bg="green",
    fg="white",
    relief="flat",
)
process_btn.pack(pady=10)

# Styled Progress Bar
vdstitch_progress = ttk.Progressbar(VDStitch, length=300, mode="determinate")
vdstitch_progress.pack(pady=10)


# ---3D Generator Frame Contents ---

input_video_path = tk.StringVar()
selected_depth_map = tk.StringVar()
output_sbs_video_path = tk.StringVar()
selected_codec = tk.StringVar(value="mp4v")
fg_shift = tk.DoubleVar(value=6.0)
mg_shift = tk.DoubleVar(value=3.0)
bg_shift = tk.DoubleVar(value=-4.0)
sharpness_factor = tk.DoubleVar(value=0.2)
blend_factor = tk.DoubleVar(value=0.6)
delay_time = tk.DoubleVar(value=1 / 30)
output_format = tk.StringVar(value="Full-SBS")

# Dictionary of Common Aspect Ratios
aspect_ratios = {
    "Default (16:9)": 16 / 9,
    "CinemaScope (2.39:1)": 2.39,
    "21:9 UltraWide": 21 / 9,
    "4:3 (Classic Films)": 4 / 3,
    "1:1 (Square)": 1 / 1,
    "2.35:1 (Classic Cinematic)": 2.35,
    "2.76:1 (Ultra-Panavision)": 2.76,
}

# Tkinter Variable to Store Selected Aspect Ratio
selected_aspect_ratio = tk.StringVar(value="Default (16:9)")


def save_settings():
    """Saves all current settings to a JSON file"""
    settings = {
        "selected_codec": selected_codec.get(),
        "fg_shift": fg_shift.get(),
        "mg_shift": mg_shift.get(),
        "bg_shift": bg_shift.get(),
        "sharpness_factor": sharpness_factor.get(),
        "blend_factor": blend_factor.get(),
        "delay_time": delay_time.get(),
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)


def load_settings():
    """Loads settings from the JSON file, if available"""
    if os.path.exists(SETTINGS_FILE):  # Now SETTINGS_FILE is properly defined
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
            fg_shift.set(settings.get("fg_shift", 6.0))
            mg_shift.set(settings.get("mg_shift", 3.0))
            bg_shift.set(settings.get("bg_shift", -4.0))
            sharpness_factor.set(settings.get("sharpness_factor", 0.2))
            blend_factor.set(settings.get("blend_factor", 0.6))
            delay_time.set(settings.get("delay_time", 1 / 30))


# Ensure SETTINGS_FILE is defined before calling load_settings()
load_settings()


# ✅ Updated codec options: Standard + Lossless Codecs
codec_options = [
    # 🔹 Standard Codecs
    "mp4v",  # MPEG-4 (Good for MP4 format, widely supported)
    "XVID",  # XviD (Best for AVI format)
    "DIVX",  # DivX (Older AVI format)
    # 🔹 Lossless Codecs
    "FFV1",  # FFmpeg Lossless (Best quality)
    "LAGS",  # Lagarith Lossless
    "ULRG",  # UT Video Lossless
    "MJPG",  # Near-lossless Motion JPEG
]

# Layout frames
top_widgets_frame = tk.LabelFrame(
    visiondepth_content_frame, text="Video Info", padx=10, pady=10
)
top_widgets_frame.grid(row=0, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

# Thumbnail
video_thumbnail_label = tk.Label(
    top_widgets_frame, text="No Thumbnail", bg="white", width=20, height=5
)
video_thumbnail_label.grid(row=0, column=0, padx=10, pady=5)

video_specs_label = tk.Label(
    top_widgets_frame, text="Resolution: N/A\nFPS: N/A", justify="left"
)
video_specs_label.grid(row=0, column=1, padx=10, pady=5)

depth_map_label = tk.Label(
    top_widgets_frame, text="Depth Map (3D): None", justify="left", wraplength=200
)
depth_map_label.grid(row=1, column=1, padx=10, pady=5)

progress = ttk.Progressbar(
    top_widgets_frame, orient="horizontal", length=300, mode="determinate"
)
progress.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

progress_label = tk.Label(top_widgets_frame, text="0%", font=("Arial", 10))
progress_label.grid(row=1, column=2, padx=10, pady=5, sticky="ew")


# Processing Options
options_frame = tk.LabelFrame(
    visiondepth_content_frame, text="Processing Options", padx=10, pady=10
)
options_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

tk.Label(options_frame, text="Codec").grid(row=0, column=0, sticky="w")
codec_menu = tk.OptionMenu(options_frame, selected_codec, *codec_options)
codec_menu.grid(row=0, column=1, sticky="ew")

# Aspect Ratio Selection Dropdown
tk.Label(options_frame, text="Aspect Ratio").grid(row=0, column=2, sticky="w")
aspect_ratio_menu = tk.OptionMenu(
    options_frame, selected_aspect_ratio, *aspect_ratios.keys()
)
aspect_ratio_menu.grid(row=0, column=3, padx=5, sticky="ew")


tk.Label(options_frame, text="Divergence Shift").grid(row=1, column=0, sticky="w")
tk.Scale(
    options_frame,
    from_=0,
    to=20,
    resolution=0.5,
    orient=tk.HORIZONTAL,
    variable=fg_shift,
).grid(row=1, column=1, sticky="ew")

tk.Label(options_frame, text="Sharpness Factor").grid(row=1, column=2, sticky="w")
tk.Scale(
    options_frame,
    from_=-1,
    to=1,
    resolution=0.1,
    orient=tk.HORIZONTAL,
    variable=sharpness_factor,
).grid(row=1, column=3, sticky="ew")

tk.Label(options_frame, text="Depth Transition").grid(row=2, column=0, sticky="w")
tk.Scale(
    options_frame,
    from_=-5,
    to=10,
    resolution=0.5,
    orient=tk.HORIZONTAL,
    variable=mg_shift,
).grid(row=2, column=1, sticky="ew")

tk.Label(options_frame, text="Blend Factor").grid(row=2, column=2, sticky="w")
tk.Scale(
    options_frame,
    from_=0.1,
    to=1.0,
    resolution=0.1,
    orient=tk.HORIZONTAL,
    variable=blend_factor,
).grid(row=2, column=3, sticky="ew")

tk.Label(options_frame, text="Convergence Shift").grid(row=3, column=0, sticky="w")
tk.Scale(
    options_frame,
    from_=-20,
    to=0,
    resolution=0.5,
    orient=tk.HORIZONTAL,
    variable=bg_shift,
).grid(row=3, column=1, sticky="ew")

tk.Label(options_frame, text="Delay Time (seconds)").grid(row=3, column=2, sticky="w")
tk.Scale(
    options_frame,
    from_=1 / 50,
    to=1 / 20,
    resolution=0.001,
    orient=tk.HORIZONTAL,
    variable=delay_time,
).grid(row=3, column=3, sticky="ew")

reset_button = tk.Button(
    options_frame,
    text="Reset to Defaults",
    command=reset_settings,
    bg="#8B0000",
    fg="white",
)
reset_button.grid(row=3, column=4, columnspan=2, pady=10)

# File Selection
tk.Button(
    visiondepth_content_frame, text="Select Input Video", command=select_input_video
).grid(row=3, column=0, pady=5, sticky="ew")
tk.Entry(visiondepth_content_frame, textvariable=input_video_path, width=50).grid(
    row=3, column=1, pady=5, padx=5
)

tk.Button(
    visiondepth_content_frame, text="Select Depth Map", command=select_depth_map
).grid(row=4, column=0, pady=5, sticky="ew")
tk.Entry(visiondepth_content_frame, textvariable=selected_depth_map, width=50).grid(
    row=4, column=1, pady=5, padx=5
)

tk.Button(
    visiondepth_content_frame, text="Select Output Video", command=select_output_video
).grid(row=5, column=0, pady=5, sticky="ew")
tk.Entry(visiondepth_content_frame, textvariable=output_sbs_video_path, width=50).grid(
    row=5, column=1, pady=5, padx=5
)

# Frame to Hold Buttons and Format Selection in a Single Row
button_frame = tk.Frame(visiondepth_content_frame)
button_frame.grid(row=6, column=0, columnspan=5, pady=10, sticky="w")

# 3D Format Label and Dropdown (Inside button_frame)
tk.Label(button_frame, text="3D Format").pack(side="left", padx=5)

option_menu = tk.OptionMenu(
    button_frame,
    output_format,
    "Full-SBS",
    "Half-SBS",
    "Full-OU",
    "Half-OU",
    "Red-Cyan Anaglyph",
    "VR",
)
option_menu.config(width=10)  # Adjust width to keep consistent look
option_menu.pack(side="left", padx=5)

# Buttons Inside button_frame to Keep Everything on One Line
start_button = tk.Button(
    button_frame,
    text="Generate 3D Video",
    command=process_video,
    bg="green",
    fg="white",
)
start_button.pack(side="left", padx=5)

suspend_button = tk.Button(
    button_frame, text="Suspend", command=suspend_processing, bg="orange", fg="black"
)
suspend_button.pack(side="left", padx=5)

resume_button = tk.Button(
    button_frame, text="Resume", command=resume_processing, bg="blue", fg="white"
)
resume_button.pack(side="left", padx=5)

cancel_button = tk.Button(
    button_frame, text="Cancel", command=cancel_processing, bg="red", fg="white"
)
cancel_button.pack(side="left", padx=5)


# Load the GitHub icon from assets
github_icon_path = resource_path(os.path.join("assets", "github_Logo.png"))

# Ensure the file exists before trying to open it
if not os.path.exists(github_icon_path):
    print(f"❌ ERROR: Missing github_Logo.png at {github_icon_path}")
    sys.exit(1)  # Exit to prevent crashing

github_icon = Image.open(github_icon_path)
github_icon = github_icon.resize((15, 15), Image.LANCZOS)  # Resize to fit UI
github_icon_tk = ImageTk.PhotoImage(github_icon)

# Create the clickable GitHub icon button
github_button = tk.Button(
    visiondepth_content_frame,
    image=github_icon_tk,
    command=open_github,
    borderwidth=0,
    bg="white",
    cursor="hand2",
)
github_button.image = github_icon_tk  # Keep a reference to prevent garbage collection
github_button.grid(row=7, column=0, pady=10, padx=5, sticky="w")  # Adjust positioning


# Load previous settings (if they exist)
load_settings()

# Ensure settings are saved when the program closes
root.protocol("WM_DELETE_WINDOW", lambda: (save_settings(), root.destroy()))

root.mainloop()
