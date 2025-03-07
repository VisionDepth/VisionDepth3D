import os
import sys
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, Label, Button, OptionMenu, StringVar, BooleanVar, Entry
from tqdm import tqdm
from PIL import Image, ImageTk, ImageOps
from transformers import AutoProcessor, AutoModelForDepthEstimation
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
import torch

sys.path.append("modules")

# Automatically detect device (CUDA, CPU, etc.)
device = "CUDAExecutionProvider" if ort.get_device() == "GPU" else "CPUExecutionProvider"

# Load the trained ONNX model
MODEL_PATH = 'weights/backward_warping_model.onnx'
session = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Extract input & output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"✅ Loaded ONNX model from {MODEL_PATH} on {device}")

# Define global flags
suspend_flag = threading.Event()  # ✅ Better for threading-based pausing
cancel_flag = threading.Event()


# ✅ Initialize `pipe` globally to prevent NameError
pipe = None  # Set to None at the start


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS2
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def format_3d_output(left_frame, right_frame, output_format):
    """Formats the 3D output according to the user's selection."""
    height, width = left_frame.shape[:2]
    
    if output_format == "Red-Cyan Anaglyph":
        print("🎨 Generating Red-Cyan Anaglyph 3D frames...")
        final_frame = generate_anaglyph_3d(left_frame, right_frame)
        return final_frame  # ✅ Return the generated anaglyph frame


    elif output_format == "Full-SBS":
        # ✅ Check if frames are already the correct width, otherwise fix them
        expected_width = width * 2
        if left_frame.shape[1] != width or right_frame.shape[1] != width:
            print("🔄 Resizing left and right frames to match Full-SBS format...")
            left_frame = cv2.resize(left_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            right_frame = cv2.resize(right_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)

        sbs_frame = np.hstack((left_frame, right_frame))  # Final size should be 7680x1608
        print(f"✅ Full-SBS Frame Created: {sbs_frame.shape}")
        return sbs_frame

    elif output_format == "Half-SBS":
        # ✅ Apply selected aspect ratio before resizing
        aspect_ratio_value = aspect_ratios.get(selected_aspect_ratio.get(), 16/9)
        left_frame = apply_aspect_ratio_crop(left_frame, aspect_ratio_value)
        right_frame = apply_aspect_ratio_crop(right_frame, aspect_ratio_value)

        # ✅ Resize for Half-SBS: 960xHeight per eye (1920x1080 total for 1080p)
        half_width = left_frame.shape[1] // 2
        height = left_frame.shape[0]
        left_resized = cv2.resize(left_frame, (half_width, height), interpolation=cv2.INTER_LANCZOS4)
        right_resized = cv2.resize(right_frame, (half_width, height), interpolation=cv2.INTER_LANCZOS4)

        sbs_half_frame = np.hstack((left_resized, right_resized))
        print(f"✅ Half-SBS Frame Created: {sbs_half_frame.shape}")
        return sbs_half_frame


    elif output_format == "Full-OU":
            return np.vstack((left_frame, right_frame))  # 1920x2160

    elif output_format == "Half-OU":
        # ✅ Half-OU = 1920x540 per eye (for passive 3D TVs)
        target_height = height // 2
        left_resized = cv2.resize(left_frame, (width, 540), interpolation=cv2.INTER_LANCZOS4)
        right_resized = cv2.resize(right_frame, (width, 540), interpolation=cv2.INTER_LANCZOS4)

        return np.vstack((left_resized, right_resized))  # 1920x1080

    elif output_format == "VR":
        # ✅ VR Headsets require per-eye aspect ratio correction (e.g., Oculus)
        vr_width = 1440  # Per-eye resolution for Oculus Quest 2
        vr_height = 1600  # Adjusted height
        left_resized = cv2.resize(left_frame, (vr_width, vr_height), interpolation=cv2.INTER_LANCZOS4)
        right_resized = cv2.resize(right_frame, (vr_width, vr_height), interpolation=cv2.INTER_LANCZOS4)

        return np.hstack((left_resized, right_resized))  # 2880x1600

    else:
        print(f"⚠ Warning: Unknown output format '{output_format}', defaulting to SBS.")
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
    anaglyph = cv2.merge([
        right_b * 0.6,  
        right_g * 0.7,  
        left_r * 0.9    # ✅ Slight reduction to balance intensity
    ])


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
        bottom_cropped = apply_aspect_ratio_crop(bottom_half, aspect_ratio, is_full_ou=False)

        # ✅ Stack them back together
        cropped_frame = np.vstack((top_cropped, bottom_cropped))
    else:
        # ✅ Calculate the correct target height
        target_height = int(width / aspect_ratio)
        if target_height >= height:
            print("✅ No cropping needed. Returning original frame.")
            return frame  # No cropping needed

        crop_y = (height - target_height) // 2
        cropped_frame = frame[crop_y:crop_y + target_height, :]

        # ✅ Debug: Show cropped frame size
        print(f"✅ Cropped Frame Size: {width}x{target_height} (Aspect Ratio: {aspect_ratio})")

    # ✅ Ensure correct final resizing
    final_frame = cv2.resize(cropped_frame, (width, cropped_frame.shape[0]), interpolation=cv2.INTER_AREA)

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
        return frame[y:y+h, x:x+w], reference_crop

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("⚠ No black bars detected. Returning original frame.")
        return frame, (0, 0, frame.shape[1], frame.shape[0])

    # ✅ Largest contour is assumed to be the actual frame
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return frame[y:y+h, x:x+w], (x, y, w, h)
    
def remove_white_edges(image):
    mask = (image[:, :, 0] > 240) & (image[:, :, 1] > 240) & (image[:, :, 2] > 240)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    
    # Fill white regions with a median blur
    blurred = cv2.medianBlur(image, 5)
    image[mask.astype(bool)] = blurred[mask.astype(bool)]
    return image
    
def correct_convergence_shift(left_frame, right_frame, depth_map, session, input_name, output_name, bg_threshold=3.0):
    """
    Adjusts convergence shift using depth-based warp parameters from an ONNX model.

    Parameters:
        left_frame (np.ndarray): Left-eye frame.
        right_frame (np.ndarray): Right-eye frame.
        depth_map (np.ndarray): Normalized depth map.
        session (onnxruntime.InferenceSession): ONNX model session.
        input_name (str): ONNX model input tensor name.
        output_name (str): ONNX model output tensor name.
        bg_threshold (float): Threshold for background separation.

    Returns:
        np.ndarray: Adjusted left frame.
        np.ndarray: Adjusted right frame.
    """

    # Normalize depth map to range [0.1, 1.0]
    depth_map = cv2.normalize(depth_map, None, 0.1, 1.0, cv2.NORM_MINMAX)
    background_mask = (depth_map >= bg_threshold).astype(np.uint8)

    # 🔥 Use ONNX model for warp prediction
    warp_input = np.array([[np.mean(depth_map)]], dtype=np.float32)  # Ensure float32 input
    warp_params = session.run([output_name], {input_name: warp_input})[0].reshape(3, 3)

    h, w = left_frame.shape[:2]

    # ✅ Apply perspective warping with bicubic interpolation
    corrected_left = cv2.warpPerspective(left_frame, warp_params, (w, h), flags=cv2.INTER_CUBIC)
    corrected_right = cv2.warpPerspective(right_frame, warp_params, (w, h), flags=cv2.INTER_CUBIC)

    # ✅ Resize with bicubic interpolation to avoid artifacts
    corrected_left = cv2.resize(corrected_left, (w, h), interpolation=cv2.INTER_CUBIC)
    corrected_right = cv2.resize(corrected_right, (w, h), interpolation=cv2.INTER_CUBIC)
    background_mask = cv2.resize(background_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # ✅ Soft blend background mask to reduce harsh edges
    background_mask = cv2.GaussianBlur(background_mask.astype(np.float32), (5, 5), 2.0)

    blended_left = background_mask[..., None] * corrected_left + (1 - background_mask[..., None]) * left_frame
    blended_right = background_mask[..., None] * corrected_right + (1 - background_mask[..., None]) * right_frame

    return blended_left.astype(np.uint8), blended_right.astype(np.uint8)




def render_sbs_3d(input_video, depth_video, output_video, codec, fps, width, height, fg_shift, mg_shift, bg_shift,
                  sharpness_factor, output_format, aspect_ratio_value, delay_time=1/30, blend_factor=0.5, progress=None, progress_label=None,
                  suspend_flag=None, cancel_flag=None):
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
    aspect_ratio_value = aspect_ratios.get(selected_aspect_ratio.get(), 16/9)
    cropped_frame = apply_aspect_ratio_crop(first_frame, aspect_ratio_value)
    height = cropped_frame.shape[0]  # Update height based on cropped frame


    # 🎬 Initialize VideoWriter with correct height (before loop)
    output_width = width if output_format == "Half-SBS" else width * 2
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*codec), fps, (output_width, height))

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

        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0  # Compute average FPS

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
            original_frame = original_frame[y:y+h, x:x+w]

        frame_resized = cv2.resize(original_frame, (width, height), interpolation=cv2.INTER_AREA)
        depth_frame_resized = cv2.resize(depth_frame, (width, height))
        
        left_frame, right_frame = frame_resized.copy(), frame_resized.copy()

        # Pulfrich effect adjustments
        blend_factor = min(0.5, blend_factor + 0.05) if len(frame_buffer) else blend_factor

        # Process Depth frame
        depth_map_gray = cv2.cvtColor(depth_frame_resized, cv2.COLOR_BGR2GRAY)
        depth_normalized = cv2.normalize(depth_map_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_filtered = cv2.bilateralFilter(depth_normalized, d=5, sigmaColor=50, sigmaSpace=50)
        depth_normalized = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
        depth_normalized = depth_filtered / 255.0             
        
        # Apply convergence correction BEFORE doing depth-based shifts
        left_frame, right_frame = correct_convergence_shift(left_frame, right_frame, depth_normalized, session, input_name, output_name)

               
        # Generate y-coordinate mapping
        map_y = np.repeat(np.arange(height).reshape(-1, 1), width, axis=1).astype(np.float32)
        
        # Depth-based pixel shift
        for y in range(height):
            fg_shift_val = fg_shift  # Foreground shift (e.g. 12)
            mg_shift_val = mg_shift  # Midground shift (e.g. 6)
            bg_shift_val = bg_shift  # Background shift (e.g. -2)

            # **Original depth-based mapping**
            shift_vals_fg = (-depth_normalized[y, :] * fg_shift_val).astype(np.float32)
            shift_vals_mg = (-depth_normalized[y, :] * mg_shift_val).astype(np.float32)
            shift_vals_bg = (depth_normalized[y, :] * bg_shift_val).astype(np.float32)

            # Final x-mapping
            new_x_left = np.clip(np.arange(width) + shift_vals_fg + shift_vals_mg + shift_vals_bg, 0, width - 1)
            new_x_right = np.clip(np.arange(width) - shift_vals_fg - shift_vals_mg - shift_vals_bg, 0, width - 1)

            # Reshape for remapping
            map_x_left = new_x_left.reshape(1, -1).astype(np.float32)
            map_x_right = new_x_right.reshape(1, -1).astype(np.float32)

            # Apply remapping (ensuring correct interpolation)
            left_frame[y] = cv2.remap(frame_resized, map_x_left, map_y[y].reshape(1, -1), interpolation=cv2.INTER_CUBIC)
            right_frame[y] = cv2.remap(frame_resized, map_x_right, map_y[y].reshape(1, -1), interpolation=cv2.INTER_CUBIC)
        
        # Buffer logic
        frame_buffer.append((left_frame, right_frame))
        if len(frame_buffer) > frame_delay:
            delayed_left_frame, delayed_right_frame = frame_buffer.pop(0)
        else:
            delayed_left_frame, delayed_right_frame = left_frame, right_frame

        # Create Pulfrich effect
        blended_left_frame = cv2.addWeighted(delayed_left_frame, blend_factor, left_frame, 1 - blend_factor, 0)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5 + float(sharpness_factor), -1], [0, -1, 0]])
        left_sharp = cv2.filter2D(blended_left_frame, -1, sharpen_kernel)
        right_sharp = cv2.filter2D(right_frame, -1, sharpen_kernel)
        
        left_sharp_resized = cv2.resize(left_sharp, (width // 2, height))
        right_sharp_resized = cv2.resize(right_sharp, (width // 2, height))

        sbs_frame = format_3d_output(left_sharp, right_sharp, output_format)

        # ✅ Dynamically check expected size based on output format
        expected_width = width * 2 if output_format == "Full-SBS" else width  # 7680 for Full-SBS
        expected_height = height  # Height stays the same

        h, w = sbs_frame.shape[:2]
        if (w, h) != (expected_width, expected_height):
            print(f"⚠ Warning: Frame size mismatch! Expected: {expected_width}x{expected_height}, Got: {w}x{h}")
            sbs_frame = cv2.resize(sbs_frame, (expected_width, expected_height), interpolation=cv2.INTER_AREA)

        # ✅ Write frame and track success
        try:
            out.write(sbs_frame)
        except Exception as e:
            print(f"❌ Error while writing frame: {e}")

    original_cap.release()
    depth_cap.release()
    out.release()
    print("3D video generated successfully.")
    
def render_ou_3d(input_video, depth_video, output_video, codec, fps, width, height,
                 fg_shift, mg_shift, bg_shift, sharpness_factor, output_format, aspect_ratio_value,
                 delay_time=1/30, blend_factor=0.5, progress=None, progress_label=None,
                 suspend_flag=None, cancel_flag=None):
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
    aspect_ratio_value = aspect_ratios.get(selected_aspect_ratio.get(), 16/9)
    cropped_frame = apply_aspect_ratio_crop(first_frame, aspect_ratio_value)
    height = cropped_frame.shape[0]

    # ✅ Set output height correctly for Half-OU
    output_height = height * 2 if output_format == "Full-OU" else height

    # 🔥 DEBUG: Print codec info
    print(f"📝 Selected Codec: {codec} | FPS: {fps} | Resolution: {width}x{output_height}")

    # ✅ Select correct codec
    fourcc = cv2.VideoWriter_fourcc(*codec)

   # ✅ Initialize VideoWriter
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

        # ✅ **Calculate FPS with Moving Average**
        curr_time = time.time()
        frame_time = curr_time - prev_time  # Time taken for one frame

        if frame_time > 0:
            fps_calc = 1.0 / frame_time  # FPS based on actual frame time
            fps_values.append(fps_calc)

        if len(fps_values) > 10:  # Keep last 10 FPS values for smoothing
            fps_values.pop(0)

        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0  # Compute average FPS

        # ✅ **Update Progress Bar and FPS Display**
        if progress:
            progress["value"] = percentage
            progress.update()
        if progress_label:
            progress_label.config(
                text=f"{percentage:.2f}% | FPS: {avg_fps:.2f} | Elapsed: {time.strftime('%M:%S', time.gmtime(elapsed_time))}"
            )

        prev_time = curr_time  # Update previous frame time

        # ✅ Black bar removal once, reused
        if reference_crop is None:
            original_frame, reference_crop = remove_black_bars(original_frame)
        else:
            x, y, w, h = reference_crop
            original_frame = original_frame[y:y + h, x:x + w]

        # ✅ Resize frames for consistent processing
        frame_resized = cv2.resize(original_frame, (width, height), interpolation=cv2.INTER_AREA)
        depth_frame_resized = cv2.resize(depth_frame, (width, height))

        top_frame, bottom_frame = frame_resized.copy(), frame_resized.copy()

        # ✅ Depth normalization
        depth_map_gray = cv2.cvtColor(depth_frame_resized, cv2.COLOR_BGR2GRAY)
        depth_normalized = cv2.normalize(depth_map_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_filtered = cv2.bilateralFilter(depth_normalized, d=5, sigmaColor=50, sigmaSpace=50)
        depth_normalized = cv2.GaussianBlur(depth_filtered, (5, 5), 0) / 255.0

        # ✅ Depth-based pixel shifting for OU
        map_y = np.repeat(np.arange(height).reshape(-1, 1), width, axis=1).astype(np.float32)
        
        for y in range(height):
            fg_shift_val = fg_shift  # Foreground shift (e.g. 12)
            mg_shift_val = mg_shift  # Midground shift (e.g. 6)
            bg_shift_val = bg_shift  # Background shift (e.g. -2)

            # **Original depth-based mapping**
            shift_vals_fg = (-depth_normalized[y, :] * fg_shift_val).astype(np.float32)
            shift_vals_mg = (-depth_normalized[y, :] * mg_shift_val).astype(np.float32)
            shift_vals_bg = (depth_normalized[y, :] * bg_shift_val).astype(np.float32)

            # Final x-mapping
            new_x_top = np.clip(np.arange(width) + shift_vals_fg + shift_vals_mg + shift_vals_bg, 0, width - 1)
            new_x_bottom = np.clip(np.arange(width) - shift_vals_fg - shift_vals_mg - shift_vals_bg, 0, width - 1)

            # Convert mappings to proper format for remapping
            map_x_top = new_x_top.reshape(1, -1).astype(np.float32)
            map_x_bottom = new_x_bottom.reshape(1, -1).astype(np.float32)

            # ✅ Apply depth-based remapping (FIXED VARIABLE NAME)
            top_frame[y] = cv2.remap(frame_resized, map_x_top, map_y[y].reshape(1, -1), interpolation=cv2.INTER_CUBIC)
            bottom_frame[y] = cv2.remap(frame_resized, map_x_bottom, map_y[y].reshape(1, -1), interpolation=cv2.INTER_CUBIC)

        # ✅ Apply sharpening filter
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5 + sharpness_factor, -1], [0, -1, 0]])
        top_sharp = cv2.filter2D(top_frame, -1, sharpen_kernel)
        bottom_sharp = cv2.filter2D(bottom_frame, -1, sharpen_kernel)

        # ✅ Ensure the output format function returns a valid frame
        ou_frame = format_3d_output(top_sharp, bottom_sharp, output_format)

        if ou_frame is None:
            print("❌ Error: format_3d_output() returned None!")
            return

        # ✅ Dynamically check expected size based on output format
        expected_height = height * 2 if output_format == "Full-OU" else height
        expected_width = width  # width stays the same
        
        h, w = ou_frame.shape[:2]
        if (w, h) != (expected_width, expected_height):
            print(f"⚠ Warning: Resizing output to match expected Full-OU dimensions ({expected_width}x{expected_height})")
            ou_frame = cv2.resize(ou_frame, (expected_width, expected_height), interpolation=cv2.INTER_AREA)

        # ✅ Final debug check before writing the frame
        if ou_frame is None or ou_frame.shape[0] == 0 or ou_frame.shape[1] == 0:
            print("❌ Fatal Error: OU Frame is invalid before writing to file!")
            return  # Avoid writing a blank/broken frame

        out.write(ou_frame)  # 📝 Now safe to write


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
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
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
        video_thumbnail_label.image = img_tk  # Save a reference to prevent garbage collection

        video_specs_label.config(
            text=f"Video Info:\nResolution: {width}x{height}\nFPS: {fps:.2f}"
        )
    else:
        video_specs_label.config(
            text="Video Info:\nUnable to extract details"
        )

def update_thumbnail(thumbnail_path):
    thumbnail_image = Image.open(thumbnail_path)
    thumbnail_image = thumbnail_image.resize((250, 100), Image.LANCZOS)  # Adjust the size as needed
    thumbnail_photo = ImageTk.PhotoImage(thumbnail_image)
    video_thumbnail_label.config(image=thumbnail_photo)
    video_thumbnail_label.image = thumbnail_photo

def select_output_video():
    output_sbs_video_path.set(filedialog.asksaveasfilename(
        defaultextension=".mp4",
        filetypes=[("MP4 files", "*.mp4"), ("MKV files", "*.mkv"), ("AVI files", "*.avi")]
    ))

def select_depth_map():
    depth_map_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
    if not depth_map_path:
        return

    selected_depth_map.set(depth_map_path)
    depth_map_label.config(text=f"Selected Depth Map:\n{os.path.basename(depth_map_path)}")


def process_video():
    if not input_video_path.get() or not output_sbs_video_path.get() or not selected_depth_map.get():
        messagebox.showerror("Error", "Please select input video, depth map, and output path.")
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
            cancel_flag=cancel_flag     # Pass cancel_flag
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
            cancel_flag=cancel_flag     # Pass cancel_flag
        )

    if not cancel_flag.is_set():
        progress["value"] = 100
        progress_label.config(text="100%")
        progress.update()
        print("✅ Processing complete.")

def suspend_processing():
    """ Pauses the processing loop safely. """
    suspend_flag.set()  # This will cause processing to pause
    print("⏸ Processing Suspended!")

def resume_processing():
    """ Resumes the processing loop safely. """
    suspend_flag.clear()  # Processing will continue from where it left off
    print("▶ Processing Resumed!")


def cancel_processing():
    """ Cancels processing completely. """
    cancel_flag.set()
    suspend_flag.clear()  # Ensure no accidental resume
    print("❌ Processing canceled.")
    
# Define SETTINGS_FILE at the top of the script
SETTINGS_FILE = "settings.json"

def open_github():
    """Opens the GitHub repository in a web browser."""
    webbrowser.open_new("https://github.com/VisionDepth/VisionDepth3D")  # Replace with your actual GitHub URL

def reset_settings():
    """ Resets all sliders and settings to default values """
    fg_shift.set(6.0)  # Default divergence shift
    mg_shift.set(3.0)  # Default depth transition
    bg_shift.set(-4.0)  # Default convergence shift
    sharpness_factor.set(0.2)
    blend_factor.set(0.6)
    delay_time.set(1/30)
    output_format = tk.StringVar(value="Full-SBS")
    
    messagebox.showinfo("Settings Reset", "All values have been restored to defaults!")

#---Depth Estimation---

# -----------------------
# Global Variables & Setup
# -----------------------

# --- Window Setup ---
root = tk.Tk()
root.title("VisionDepth3D Video Generator")
root.geometry("1080x780")

background_image = Image.open(resource_path("assets/Background.png"))
background_image = background_image.resize((1080, 780), Image.LANCZOS)
bg_image = ImageTk.PhotoImage(background_image)

root.bg_image = bg_image  # keep a persistent reference
background_label = tk.Label(root, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


supported_models = {
    "Distil-Any-Depth-Large": "keetrap/Distil-Any-Depth-Large-hf",
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
    "dpt-beit-large-512": "Intel/dpt-beit-large-512"
}

selected_model = StringVar(root, value="Distil-Any-Depth-Large")
colormap_var = StringVar(root, value="Default")
invert_var = BooleanVar(root, value=False)
output_dir = ""

# -----------------------
# Functions
# -----------------------

def update_pipeline(*args):
    global pipe

    selected_checkpoint = selected_model.get()
    checkpoint = supported_models.get(selected_checkpoint, None)

    if checkpoint is None:
        status_label.config(text=f"⚠️ Error: Model '{selected_checkpoint}' not found.")
        return

    try:
        # ✅ Ensure device is set correctly for PyTorch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ Explicitly load the image processor
        processor = AutoProcessor.from_pretrained(checkpoint)

        # ✅ Load the depth-estimation pipeline
        pipe = pipeline("depth-estimation", model=checkpoint, device=device, image_processor=processor)

        status_label.config(text=f"✅ Model updated: {selected_checkpoint} (Running on {device.upper()})")

    except Exception as e:
        status_label.config(text=f"❌ Model loading failed: {str(e)}")


def choose_output_directory():
    global output_dir
    selected_directory = filedialog.askdirectory()
    if selected_directory:
        output_dir = selected_directory
        output_dir_label.config(text=f"📁 {output_dir}")

def process_image(file_path):
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

    image_disp = image.copy()
    image_disp.thumbnail((600, 600))
    photo_input = ImageTk.PhotoImage(image_disp)
    input_label.config(image=photo_input)
    input_label.image = photo_input

    depth_disp = depth_image.copy()
    depth_disp.thumbnail((600, 600))
    photo_depth = ImageTk.PhotoImage(depth_disp)
    output_label.config(image=photo_depth)
    output_label.image = photo_depth

    output_filename = output_name_entry.get().strip() or "depth_map.png"
    file_save_path = os.path.join(output_dir, output_filename) if output_dir else output_filename
    depth_image.save(file_save_path)
    status_label.config(text=f"✅ Image saved: {file_save_path}")
    progress_bar.config(value=100)

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpeg;*.jpg;*.png")])
    if file_path:
        status_label.config(text="🔄 Processing image...")
        progress_bar.start(10)
        threading.Thread(target=lambda: [process_image(file_path), progress_bar.stop()], daemon=True).start()

def process_video2(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        status_label.config(text="❌ Error: Cannot open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Handle resolution input (downscale for processing)
    res_input = resolution_entry.get().strip()
    target_size = None
    if res_input:
        try:
            width, height = map(int, res_input.split(','))
            target_size = (width, height)
        except:
            status_label.config(text="⚠️ Invalid resolution input. Using original size.")
    else:
        target_size = (original_width, original_height)

    # Handle batch size input
    try:
        BATCH_SIZE = int(batch_size_entry.get().strip())
        if BATCH_SIZE <= 0:
            raise ValueError
    except ValueError:
        BATCH_SIZE = 8  # Default batch size
        status_label.config(text="⚠️ Invalid batch size. Using default batch size (8).")

    # Output file setup
    input_dir, input_filename = os.path.split(file_path)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_depth{ext}"
    output_path = os.path.join(output_dir, output_filename) if output_dir else output_filename

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if ext.lower() == ".mp4" else cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

    # -----------------------
    # 🚀 Batch Processing Setup
    # -----------------------
    frames_batch = []
    frame_count = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Downscale frame if needed
        if target_size and target_size != (original_width, original_height):
            frame = cv2.resize(frame, target_size)

        # Convert frame for pipeline
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        frames_batch.append(pil_image)

        # Process batch when full or at the end
        if len(frames_batch) == BATCH_SIZE or frame_count == total_frames:
            predictions = pipe(frames_batch)

            for prediction in predictions:
                raw_depth = prediction["predicted_depth"]
                depth_norm = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min())
                depth_np = depth_norm.squeeze().detach().cpu().numpy()
                grayscale = (depth_np * 255).astype("uint8")

                cmap_choice = colormap_var.get()
                if cmap_choice == "Default":
                    depth_image = prediction["depth"]
                else:
                    cmap = cm.get_cmap(cmap_choice.lower())
                    colored = cmap(grayscale / 255.0)
                    colored = (colored[:, :, :3] * 255).astype(np.uint8)
                    depth_image = Image.fromarray(colored)

                if invert_var.get():
                    depth_image = ImageOps.invert(depth_image.convert("RGB"))

                # Convert to OpenCV format
                depth_frame = np.array(depth_image)
                depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_RGB2BGR)

                # **Upscale back to original resolution before saving**
                if target_size and target_size != (original_width, original_height):
                    depth_frame = cv2.resize(depth_frame, (original_width, original_height))

                out.write(depth_frame)

            frames_batch.clear()  # Clear the batch for the next set of frames

        # Progress Update
        progress = int((frame_count / total_frames) * 100)
        progress_bar.config(value=progress)
        status_label.config(text=f"🎬 Processing video: {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    status_label.config(text=f"✅ Video saved: {output_path}")
    progress_bar.config(value=100)


def open_video():
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
            ("All Files", "*.*")
        ]
    )

    if file_path:
        status_label.config(text="🔄 Processing video...")
        progress_bar.config(mode="determinate", maximum=100, value=0)
        threading.Thread(target=process_video2, args=(file_path,), daemon=True).start()



# --- Notebook for Tabs ---
tab_control = ttk.Notebook(root)
tab_control.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.6, relheight=0.8)

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
rife_placeholder_label = tk.Label(rife_fps_frame, text="🛠️ RIFE FPS Interpolation - Coming Soon!", 
                                  font=("Arial", 16, "bold"), fg="gray")
rife_placeholder_label.pack(expand=True)  # Center the label



# --- Sidebar Content ---
tk.Label(sidebar, text="🛠️ Model", bg="#1c1c1c", fg="white", font=("Arial", 11)).pack(pady=5)
model_dropdown = ttk.Combobox(sidebar, textvariable=selected_model, values=list(supported_models.keys()), state="readonly", width=22)
model_dropdown.pack(pady=5)
model_dropdown.bind("<<ComboboxSelected>>", update_pipeline)

output_dir_label = tk.Label(sidebar, text="📂 Output Dir: None", bg="#1c1c1c", fg="white", wraplength=200)
output_dir_label.pack(pady=5)
tk.Button(sidebar, text="Choose Directory", command=choose_output_directory, width=20).pack(pady=5)

tk.Label(sidebar, text="🎨 Colormap:", bg="#1c1c1c", fg="white").pack(pady=5)
colormap_dropdown = ttk.Combobox(sidebar, textvariable=colormap_var, values=["Default", "Magma", "Viridis", "Inferno", "Plasma", "Gray"], state="readonly", width=22)
colormap_dropdown.pack(pady=5)

invert_checkbox = tk.Checkbutton(sidebar, text="🌑 Invert Depth", variable=invert_var, bg="#1c1c1c", fg="white")
invert_checkbox.pack(pady=5)

tk.Label(sidebar, text="💾 Filename (Image):", bg="#1c1c1c", fg="white").pack(pady=5)
output_name_entry = tk.Entry(sidebar, width=22)
output_name_entry.insert(0, "depth_map.png")
output_name_entry.pack(pady=5)

# Batch Size Input
tk.Label(sidebar, text="📦 Batch Size (Frames):", bg="#1c1c1c", fg="white").pack(pady=5)
batch_size_entry = tk.Entry(sidebar, width=22)
batch_size_entry.insert(0, "8")
batch_size_entry.pack(pady=5)

tk.Label(sidebar, text="🖼️ Video Resolution (w,h):", bg="#1c1c1c", fg="white").pack(pady=5)
resolution_entry = tk.Entry(sidebar, width=22)
resolution_entry.insert(0, "")
resolution_entry.pack(pady=5)

progress_bar = ttk.Progressbar(sidebar, mode="determinate", length=180)
progress_bar.pack(pady=10)
status_label = tk.Label(sidebar, text="🔋 Ready", bg="#1c1c1c", fg="white")
status_label.pack(pady=5)

depth_map_label_depth = tk.Label(sidebar, text="Depth Map (Depth Estimation): None", justify="left", wraplength=200)
depth_map_label_depth.pack(pady=5)

# --- Main Content: Image previews ---
frame_preview = tk.Frame(main_content, bg="#2b2b2b")
frame_preview.pack(pady=10)
input_label = tk.Label(frame_preview, text="🖼️ Input Image", bg="#2b2b2b", fg="white", width=40, height=15)
input_label.pack(side="left", padx=10)
output_label = tk.Label(frame_preview, text="🌊 Depth Map", bg="#2b2b2b", fg="white", width=40, height=15)
output_label.pack(side="right", padx=10)

# --- Main Content: Buttons ---
tk.Button(main_content, text="🖼️ Process Image", command=open_image, width=25, bg="#4a4a4a", fg="white").pack(pady=5)
tk.Button(main_content, text="🎥 Process Video", command=open_video, width=25, bg="#4a4a4a", fg="white").pack(pady=5)


input_video_path = tk.StringVar()
selected_depth_map = tk.StringVar()
output_sbs_video_path = tk.StringVar()
selected_codec = tk.StringVar(value="mp4v")  
fg_shift = tk.DoubleVar(value=6.0)
mg_shift = tk.DoubleVar(value=3.0)
bg_shift = tk.DoubleVar(value=-4.0)
sharpness_factor = tk.DoubleVar(value=0.2)
blend_factor = tk.DoubleVar(value=0.6)
delay_time = tk.DoubleVar(value=1/30)
output_format = tk.StringVar(value="Full-SBS")

# Dictionary of Common Aspect Ratios
aspect_ratios = {
    "Default (16:9)": 16/9,
    "CinemaScope (2.39:1)": 2.39,
    "21:9 UltraWide": 21/9,
    "4:3 (Classic Films)": 4/3,
    "1:1 (Square)": 1/1,
    "2.35:1 (Classic Cinematic)": 2.35,
    "2.76:1 (Ultra-Panavision)": 2.76
}

# Tkinter Variable to Store Selected Aspect Ratio
selected_aspect_ratio = tk.StringVar(value="Default (16:9)")



def save_settings():
    """ Saves all current settings to a JSON file """
    settings = {
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
    """ Loads settings from the JSON file, if available """
    if os.path.exists(SETTINGS_FILE):  # Now SETTINGS_FILE is properly defined
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
            fg_shift.set(settings.get("fg_shift", 6.0))
            mg_shift.set(settings.get("mg_shift", 3.0))
            bg_shift.set(settings.get("bg_shift", -4.0))
            sharpness_factor.set(settings.get("sharpness_factor", 0.2))
            blend_factor.set(settings.get("blend_factor", 0.6))
            delay_time.set(settings.get("delay_time", 1/30))

# Ensure SETTINGS_FILE is defined before calling load_settings()
load_settings()


# Codec options
codec_options = ["mp4v", "H264", "XVID", "DIVX"]

# Layout frames
top_widgets_frame = tk.LabelFrame(visiondepth_content_frame, text="Video Info", padx=10, pady=10)
top_widgets_frame.grid(row=0, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

# Thumbnail
video_thumbnail_label = tk.Label(top_widgets_frame, text="No Thumbnail", bg="white", width=20, height=5)
video_thumbnail_label.grid(row=0, column=0, padx=10, pady=5)

video_specs_label = tk.Label(top_widgets_frame, text="Resolution: N/A\nFPS: N/A", justify="left")
video_specs_label.grid(row=0, column=1, padx=10, pady=5)

depth_map_label = tk.Label(top_widgets_frame, text="Depth Map (3D): None", justify="left", wraplength=200)
depth_map_label.grid(row=1, column=1, padx=10, pady=5)

progress = ttk.Progressbar(top_widgets_frame, orient="horizontal", length=300, mode="determinate")
progress.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

progress_label = tk.Label(top_widgets_frame, text="0%", font=("Arial", 10))
progress_label.grid(row=1, column=2, padx=10, pady=5, sticky="ew")


# Processing Options
options_frame = tk.LabelFrame(visiondepth_content_frame, text="Processing Options", padx=10, pady=10)
options_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

tk.Label(options_frame, text="Codec").grid(row=0, column=0, sticky="w")
codec_menu = tk.OptionMenu(options_frame, selected_codec, *codec_options)
codec_menu.grid(row=0, column=1, sticky="ew")

# Aspect Ratio Selection Dropdown
tk.Label(options_frame, text="Aspect Ratio").grid(row=0, column=2, sticky="w")
aspect_ratio_menu = tk.OptionMenu(options_frame, selected_aspect_ratio, *aspect_ratios.keys())
aspect_ratio_menu.grid(row=0, column=3, padx=5, sticky="ew")


tk.Label(options_frame, text="Divergence Shift").grid(row=1, column=0, sticky="w")
tk.Scale(options_frame, from_=0, to=20, resolution=0.5, orient=tk.HORIZONTAL, variable=fg_shift).grid(row=1, column=1, sticky="ew")

tk.Label(options_frame, text="Sharpness Factor").grid(row=1, column=2, sticky="w")
tk.Scale(options_frame, from_=-1, to=1, resolution=0.1, orient=tk.HORIZONTAL, variable=sharpness_factor).grid(row=1, column=3, sticky="ew")

tk.Label(options_frame, text="Depth Transition").grid(row=2, column=0, sticky="w")
tk.Scale(options_frame, from_=-5, to=10, resolution=0.5, orient=tk.HORIZONTAL, variable=mg_shift).grid(row=2, column=1, sticky="ew")

tk.Label(options_frame, text="Blend Factor").grid(row=2, column=2, sticky="w")
tk.Scale(options_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=blend_factor).grid(row=2, column=3, sticky="ew")

tk.Label(options_frame, text="Convergence Shift").grid(row=3, column=0, sticky="w")
tk.Scale(options_frame, from_=-20, to=0, resolution=0.5, orient=tk.HORIZONTAL, variable=bg_shift).grid(row=3, column=1, sticky="ew")

tk.Label(options_frame, text="Delay Time (seconds)").grid(row=3, column=2, sticky="w")
tk.Scale(options_frame, from_=1/50, to=1/20, resolution=0.001, orient=tk.HORIZONTAL, variable=delay_time).grid(row=3, column=3, sticky="ew")

reset_button = tk.Button(options_frame, text="Reset to Defaults", command=reset_settings, bg="#8B0000", fg="white")
reset_button.grid(row=3, column=4, columnspan=2, pady=10)

# File Selection
tk.Button(visiondepth_content_frame, text="Select Input Video", command=select_input_video).grid(row=3, column=0, pady=5, sticky="ew")
tk.Entry(visiondepth_content_frame, textvariable=input_video_path, width=50).grid(row=3, column=1, pady=5, padx=5)

tk.Button(visiondepth_content_frame, text="Select Depth Map", command=select_depth_map).grid(row=4, column=0, pady=5, sticky="ew")
tk.Entry(visiondepth_content_frame, textvariable=selected_depth_map, width=50).grid(row=4, column=1, pady=5, padx=5)

tk.Button(visiondepth_content_frame, text="Select Output Video", command=select_output_video).grid(row=5, column=0, pady=5, sticky="ew")
tk.Entry(visiondepth_content_frame, textvariable=output_sbs_video_path, width=50).grid(row=5, column=1, pady=5, padx=5)

# Frame to Hold Buttons and Format Selection in a Single Row
button_frame = tk.Frame(visiondepth_content_frame)
button_frame.grid(row=6, column=0, columnspan=5, pady=10, sticky="w")

# 3D Format Label and Dropdown (Inside button_frame)
tk.Label(button_frame, text="3D Format").pack(side="left", padx=5)

option_menu = tk.OptionMenu(button_frame, output_format, "Full-SBS", "Half-SBS", "Full-OU", "Half-OU", "Red-Cyan Anaglyph", "VR")
option_menu.config(width=10)  # Adjust width to keep consistent look
option_menu.pack(side="left", padx=5)

# Buttons Inside button_frame to Keep Everything on One Line
start_button = tk.Button(button_frame, text="Generate 3D Video", command=process_video, bg="green", fg="white")
start_button.pack(side="left", padx=5)

suspend_button = tk.Button(button_frame, text="Suspend", command=suspend_processing, bg="orange", fg="black")
suspend_button.pack(side="left", padx=5)

resume_button = tk.Button(button_frame, text="Resume", command=resume_processing, bg="blue", fg="white")
resume_button.pack(side="left", padx=5)

cancel_button = tk.Button(button_frame, text="Cancel", command=cancel_processing, bg="red", fg="white")
cancel_button.pack(side="left", padx=5)



# Load the GitHub icon from assets
github_icon_path = resource_path("assets/github_Logo.png")
github_icon = Image.open(github_icon_path)
github_icon = github_icon.resize((15, 15), Image.LANCZOS)  # Resize to fit UI
github_icon_tk = ImageTk.PhotoImage(github_icon)

# Create the clickable GitHub icon button
github_button = tk.Button(visiondepth_content_frame, image=github_icon_tk, command=open_github, borderwidth=0, bg="white", cursor="hand2")
github_button.image = github_icon_tk  # Keep a reference to prevent garbage collection
github_button.grid(row=7, column=0, pady=10, padx=5, sticky="w")  # Adjust positioning


# Load previous settings (if they exist)
load_settings()

# Ensure settings are saved when the program closes
root.protocol("WM_DELETE_WINDOW", lambda: (save_settings(), root.destroy()))

root.mainloop()
