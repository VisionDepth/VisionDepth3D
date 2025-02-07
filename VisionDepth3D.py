import os
import sys
import shutil
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from tqdm import tqdm
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from threading import Thread
import webbrowser
import json
import subprocess
from tensorflow import keras
from moviepy.editor import VideoFileClip, AudioFileClip
import imageio
from collections import deque

# Load the trained divergence correction model
MODEL_PATH = 'weights/backward_warping_model.keras'
model = keras.models.load_model(MODEL_PATH)

# Define global flags
suspend_flag = threading.Event()  # ✅ Better for threading-based pausing
cancel_flag = threading.Event()
suspend_flag = threading.Event()

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
        return generate_anaglyph_3d(left_frame, right_frame)  # 🔴🔵
    
    if output_format == "Full-SBS":
        return np.hstack((left_frame, right_frame))  # 3840x1080

    elif output_format == "Half-SBS":
        # ✅ Half-SBS = 960x1080 per eye (for passive 3D TVs)
        half_width = width // 2
        left_resized = cv2.resize(left_frame, (960, height), interpolation=cv2.INTER_LANCZOS4)
        right_resized = cv2.resize(right_frame, (960, height), interpolation=cv2.INTER_LANCZOS4)

        return np.hstack((left_resized, right_resized))  # 1920x1080

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
        right_b * 0.7,  # Reduce blue intensity to avoid overpowering cyan
        right_g * 0.8,  # Slightly stronger green to balance cyan
        left_r * 1.0    # Keep red at full strength
    ])

    # Clip values to ensure valid pixel range
    anaglyph = np.clip(anaglyph, 0, 255).astype(np.uint8)

    return anaglyph



def apply_aspect_ratio_crop(frame, aspect_ratio):
    """Crops the frame to the selected aspect ratio while maintaining width."""
    height, width = frame.shape[:2]
    target_height = int(width / aspect_ratio)  # Calculate new height for the given aspect ratio

    # If the target height is already within the current height, no cropping needed
    if target_height >= height:
        return frame

    # Calculate cropping margins (center crop)
    crop_y = (height - target_height) // 2
    cropped_frame = frame[crop_y:crop_y + target_height, :]

    print(f"✅ Aspect Ratio Applied | {width}x{target_height} ({aspect_ratio})")
    return cropped_frame


# Initialize cropping size variables globally
global crop_x, crop_y, crop_w, crop_h
crop_x, crop_y, crop_w, crop_h = None, None, None, None

def remove_black_bars(frame):
    global crop_x, crop_y, crop_w, crop_h

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return frame, 0, 0, frame.shape[1], frame.shape[0]

    # Detect cropping region only once
    if crop_x is None or crop_w is None:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        crop_x, crop_y, crop_w, crop_h = x, y, w, h

    return frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w], crop_x, crop_y, crop_w, crop_h

def remove_white_edges(image):
    mask = (image[:, :, 0] > 240) & (image[:, :, 1] > 240) & (image[:, :, 2] > 240)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    
    # Fill white regions with a median blur
    blurred = cv2.medianBlur(image, 5)
    image[mask.astype(bool)] = blurred[mask.astype(bool)]
    return image
    
def calculate_depth_intensity(depth_map):
    depth_gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    avg_depth = np.mean(depth_gray)
    depth_variance = np.var(depth_gray)
    
def correct_convergence_shift(left_frame, right_frame, depth_map, model, bg_threshold=0.3):
    """
    Applies backward warping correction to background objects to enhance depth perception.

    - Uses the trained model to simulate depth by warping background regions inward.
    - Applies correction only to background areas.

    Parameters:
        left_frame (numpy array): Left eye frame.
        right_frame (numpy array): Right eye frame.
        depth_map (numpy array): Depth map normalized between 0-1.
        model (Keras model): Trained warp correction model.
        bg_threshold (float): Depth threshold to define the background.

    Returns:
        corrected_left, corrected_right (numpy arrays): Frames with corrected convergence shift.
    """
    
    global previous_warp_params  # Ensure it's globally accessible

    # ✅ Ensure depth is normalized
    depth_map = cv2.normalize(depth_map, None, 0.1, 1.0, cv2.NORM_MINMAX)

    # ✅ Create a binary mask for background (where depth is farther)
    background_mask = (depth_map >= bg_threshold).astype(np.uint8)

    # ✅ Initialize previous_warp_params if not set
    if 'previous_warp_params' not in globals():
        previous_warp_params = None

    # ✅ Optimize: Call the model only when needed
    if previous_warp_params is None:
        warp_input = np.array([[bg_threshold]])  # Using depth threshold as input
        previous_warp_params = model.predict(warp_input).reshape(3, 3)

    warp_params = previous_warp_params

    # ✅ Apply warp transformation ONLY to background areas
    h, w = left_frame.shape[:2]
    corrected_left = cv2.warpPerspective(left_frame, warp_params, (w, h))
    corrected_right = cv2.warpPerspective(right_frame, warp_params, (w, h))

    # ✅ Mask out non-background areas to prevent unnecessary corrections
    corrected_left = np.where(background_mask[..., None], corrected_left, left_frame)
    corrected_right = np.where(background_mask[..., None], corrected_right, right_frame)

    return corrected_left, corrected_right


def render_sbs_3d(input_video, depth_video, output_video, codec, fps, width, height, 
                  fg_shift, mg_shift, bg_shift, delay_time=None, 
                  blend_factor=0.5, progress=None, progress_label=None, suspend_flag=None, cancel_flag=None):

        # If no delay_time is passed, use the default value
    if delay_time is None:
        delay_time = 1/30  # ✅ Default value only applied if None
    

    # Frame delay buffer
    frame_delay = int(fps * delay_time)
    frame_buffer = deque(maxlen=frame_delay + 1)
    
    # ✅ Scene change tracking
    scene_change_detected = False  # 🔥 Initialize scene change flag
    scene_stability_counter = 0    # 🔥 Counter for scene stability
    prev_frame_gray = None         # 🔥 Stores previous grayscale frame for change detection
    rolling_diff = []              # 🔥 Rolling buffer for scene difference calculations
    

    # Determine output width based on format
    if output_format.get() == "Half-SBS":
        output_width = width // 2  # Half-SBS has half the width
    else:
        output_width = width

    # Preallocate memory for shift maps
    temp_output = output_video.rsplit('.', 1)[0] + '_temp.' + output_video.rsplit('.', 1)[1]

    # Initialize FFmpeg video writer
    writer = imageio.get_writer(temp_output, fps=fps, macro_block_size=1, 
                               codec='libx264', ffmpeg_params=['-crf', '18'])

    # Open video sources
    original_cap = cv2.VideoCapture(input_video)
    depth_cap = cv2.VideoCapture(depth_video)

    try:
        total_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = time.time()

        for frame_idx in range(total_frames):
            if cancel_flag and cancel_flag.is_set():
                print("❌ Rendering canceled.")
                break

            while suspend_flag and suspend_flag.is_set():
                print("⏸ Rendering paused...")
                time.sleep(0.5)
                
            # Read frames
            ret1, original_frame = original_cap.read()
            ret2, depth_frame = depth_cap.read()
            if not ret1 or not ret2:
                break
            
            elapsed_time = time.time() - start_time
            fps_current = frame_idx / elapsed_time if elapsed_time > 0 else 0
            remaining_time = (total_frames - frame_idx) / fps_current if fps_current > 0 else 0


            # ✅ Update progress bar and label
            progress_percentage = (frame_idx / total_frames) * 100
            if progress:
                progress["value"] = progress_percentage
                progress.update_idletasks()  # ✅ Force GUI update

            if progress_label:
                # Convert remaining time (in seconds) to MM:SS format
                time_remaining_str = time.strftime("%M:%S", time.gmtime(remaining_time))

                # Calculate estimated completion time (ETA)
                estimated_completion_time = datetime.now() + timedelta(seconds=remaining_time)
                completion_time_str = estimated_completion_time.strftime("%I:%M %p")  # 12-hour format with AM/PM

                # Update Progress Label
                progress_label.config(
                    text=f"Processing: {progress_percentage:.2f}% | FPS: {fps_current:.2f} | Time Left: {time_remaining_str} | ETA: {completion_time_str}"
                )


            # Remove black bars
            cropped_frame, x, y, w, h = remove_black_bars(original_frame)
            cropped_resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_AREA)
            depth_frame_resized = cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_AREA)


            # **✅ Process Depth frame**
            depth_map_gray = cv2.cvtColor(depth_frame_resized, cv2.COLOR_BGR2GRAY)
            depth_normalized = cv2.normalize(depth_map_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_filtered = cv2.bilateralFilter(depth_normalized, d=5, sigmaColor=50, sigmaSpace=50)
            depth_normalized = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
            depth_normalized = depth_filtered / 255.0  

            # **Ensure left and right frames are fresh copies**
            left_frame, right_frame = cropped_resized_frame.copy(), cropped_resized_frame.copy()

            # **✅ Apply Convergence Correction BEFORE Depth-Based Shifts**
            left_frame, right_frame = correct_convergence_shift(left_frame, right_frame, depth_normalized, model)

            # **✅ Pulfrich effect adjustments (Re-Added)**
            blend_factor = min(0.5, blend_factor + 0.05) if len(frame_buffer) else blend_factor

            # **✅ Depth-based pixel shift logic**
            map_y = np.repeat(np.arange(height).reshape(-1, 1), width, axis=1).astype(np.float32)

            for y in range(height):
                fg_shift_val = fg_shift
                mg_shift_val = mg_shift
                bg_shift_val = bg_shift

                # Compute pixel shift values based on depth
                shift_vals_fg = np.clip(-depth_normalized[y, :] * fg_shift, -10, 10).astype(np.float32)
                shift_vals_mg = np.clip(-depth_normalized[y, :] * mg_shift, -5, 5).astype(np.float32)
                shift_vals_bg = np.clip(depth_normalized[y, :] * bg_shift, -2, 2).astype(np.float32)

                # Final x-mapping for remapping
                new_x_left = np.clip(np.arange(width) + shift_vals_fg + shift_vals_mg + shift_vals_bg, 0, width - 1)
                new_x_right = np.clip(np.arange(width) - shift_vals_fg - shift_vals_mg - shift_vals_bg, 0, width - 1)

                # Convert mappings to proper format
                map_x_left = new_x_left.reshape(1, -1).astype(np.float32)
                map_x_right = new_x_right.reshape(1, -1).astype(np.float32)

                # Apply depth-based remapping
                left_frame[y] = cv2.remap(cropped_resized_frame, map_x_left, map_y[y].reshape(1, -1), interpolation=cv2.INTER_CUBIC)
                right_frame[y] = cv2.remap(cropped_resized_frame, map_x_right, map_y[y].reshape(1, -1), interpolation=cv2.INTER_CUBIC)

            if scene_change_detected and scene_stability_counter > 5:
                print(f"🎬 Scene change detected at frame {frame_idx}, diff: {avg_diff_score:.2f}")

                # Reset Pulfrich buffer but fill it with duplicate frames to prevent flickering
                frame_buffer.clear()
                for _ in range(frame_delay):  
                    frame_buffer.append((left_frame.copy(), right_frame.copy()))

            if len(frame_buffer) < frame_delay:
                delayed_left_frame, delayed_right_frame = left_frame, right_frame  # Don't use buffer if not enough frames
            else:
                delayed_left_frame, delayed_right_frame = frame_buffer.popleft()

            # **✅ Create Pulfrich effect**
            blended_left_frame = cv2.addWeighted(delayed_left_frame, min(0.3, blend_factor), left_frame, 1 - min(0.3, blend_factor), 0)
            sharpen_kernel = np.array([[0, -1, 0], [-1, 5 + sharpness_factor.get(), -1], [0, -1, 0]])
            left_sharp = cv2.filter2D(blended_left_frame, -1, sharpen_kernel)
            right_sharp = cv2.filter2D(right_frame, -1, sharpen_kernel)

            # **✅ Resize for Half-SBS**
            left_sharp_resized = cv2.resize(left_sharp, (width // 2, height)) if output_format.get() == "Half-SBS" else left_sharp
            right_sharp_resized = cv2.resize(right_sharp, (width // 2, height)) if output_format.get() == "Half-SBS" else right_sharp
            
            # Apply Aspect Ratio Crop Before Resizing
            if selected_aspect_ratio.get() != "Default (16:9)":
                aspect_ratio_value = aspect_ratios[selected_aspect_ratio.get()]
                left_frame = apply_aspect_ratio_crop(left_frame, aspect_ratio_value)
                right_frame = apply_aspect_ratio_crop(right_frame, aspect_ratio_value)

            # ✅ Format the 3D output properly
            sbs_frame = format_3d_output(left_frame, right_frame, output_format.get())

            # ✅ Convert to RGB (FFmpeg requires RGB format)
            sbs_frame_rgb = cv2.cvtColor(sbs_frame, cv2.COLOR_BGR2RGB)

            # ✅ Write frame using FFmpeg writer
            writer.append_data(sbs_frame_rgb)


    finally:
        # ✅ Release video resources safely
        if original_cap.isOpened():
            original_cap.release()
        if depth_cap.isOpened():
            depth_cap.release()
        try:
            writer.close()  # ✅ Ensure the writer is closed properly
        except Exception as e:
            print(f"⚠ Warning: Error closing video writer: {e}")

        # ✅ Ensure progress bar reaches 100% if not canceled
        if not cancel_flag.is_set():
            progress["value"] = 100
            progress_label.config(text="✅ Processing complete!")
            progress.update_idletasks()

            print("✅ Half-SBS video generated successfully.")
        else:
            progress_label.config(text="❌ Processing Canceled.")
            progress["value"] = 0
            progress.update_idletasks()
            print("❌ Video processing was canceled by the user.")

        print("🎬 Rendering process finalized. All resources cleaned up.")
        
def update_progress(frame_idx, total_frames, start_time, progress, progress_label):
    percentage = (frame_idx / total_frames) * 100
    if progress:
        progress["value"] = percentage
        progress.update()

    if progress_label:
        elapsed_time = time.time() - start_time
        time_per_frame = elapsed_time / (frame_idx + 1)
        fps = 1.0 / time_per_frame  # Calculate frames per second
        time_remaining = time_per_frame * (total_frames - frame_idx)

        # ✅ Convert Time Left to MM:SS format
        time_remaining_str = time.strftime("%M:%S", time.gmtime(time_remaining))
        elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))

        # ✅ Update Label in a cleaner format
        progress_label.config(
            text=f"Processing: {percentage:.2f}% | FPS: {fps:.2f} | Time Left: {time_remaining_str}"
        )

    
def render_ou_3d(input_video, depth_video, output_video, codec, fps, width, height, 
                 fg_shift, mg_shift, bg_shift, delay_time=1/30, 
                 blend_factor=0.5, progress=None, progress_label=None, suspend_flag=None, cancel_flag=None):
    
    # Frame delay buffer
    frame_delay = int(fps * delay_time)
    frame_buffer = deque(maxlen=frame_delay + 1)

    # Prepare intermediate output file
    temp_output = output_video.rsplit('.', 1)[0] + '_temp.' + output_video.rsplit('.', 1)[1]

    # Initialize video writer
    writer = imageio.get_writer(temp_output, fps=fps, macro_block_size=1, 
                               codec='libx264', ffmpeg_params=['-crf', '18'])

    # Open video sources
    original_cap = cv2.VideoCapture(input_video)
    depth_cap = cv2.VideoCapture(depth_video)

    try:
        total_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = time.time()

        for frame_idx in range(total_frames):
            if cancel_flag and cancel_flag.is_set():
                print("❌ Rendering canceled.")
                break

            while suspend_flag and suspend_flag.is_set():
                print("⏸ Rendering paused...")
                time.sleep(0.5)

            # Read frames
            ret1, original_frame = original_cap.read()
            ret2, depth_frame = depth_cap.read()
            if not ret1 or not ret2:
                break

            elapsed_time = time.time() - start_time
            fps_current = frame_idx / elapsed_time if elapsed_time > 0 else 0
            remaining_time = (total_frames - frame_idx) / fps_current if fps_current > 0 else 0

            # ✅ Update progress
            progress_percentage = (frame_idx / total_frames) * 100
            if progress:
                progress["value"] = progress_percentage
                progress.update_idletasks()

            if progress_label:
                estimated_completion_time = datetime.now() + timedelta(seconds=remaining_time)
                completion_time_str = estimated_completion_time.strftime("%I:%M %p")

                progress_label.config(
                    text=f"Processing: {progress_percentage:.2f}% | FPS: {fps_current:.2f} | Time Left: {time_remaining_str} | ETA: {completion_time_str}"
                )

            # Remove black bars
            cropped_frame, x, y, w, h = remove_black_bars(original_frame)
            cropped_resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_AREA)
            depth_frame_resized = cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_AREA)

            # ✅ Process Depth frame
            depth_map_gray = cv2.cvtColor(depth_frame_resized, cv2.COLOR_BGR2GRAY)
            depth_normalized = cv2.normalize(depth_map_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_filtered = cv2.bilateralFilter(depth_normalized, d=5, sigmaColor=50, sigmaSpace=50)
            depth_normalized = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
            depth_normalized = depth_filtered / 255.0  

            # ✅ Ensure left and right frames are fresh copies
            top_frame, bottom_frame = cropped_resized_frame.copy(), cropped_resized_frame.copy()

            # ✅ Apply Convergence Correction BEFORE Depth-Based Shifts
            top_frame, bottom_frame = correct_convergence_shift(top_frame, bottom_frame, depth_normalized, model)

            # ✅ Depth-based pixel shift logic
            map_y = np.repeat(np.arange(height).reshape(-1, 1), width, axis=1).astype(np.float32)

            for y in range(height):
                # Compute pixel shift values based on depth
                shift_vals_fg = np.clip(-depth_normalized[y, :] * fg_shift, -10, 10).astype(np.float32)
                shift_vals_mg = np.clip(-depth_normalized[y, :] * mg_shift, -5, 5).astype(np.float32)
                shift_vals_bg = np.clip(depth_normalized[y, :] * bg_shift, -2, 2).astype(np.float32)

                # Final x-mapping for remapping
                new_x_top = np.clip(np.arange(width) + shift_vals_fg + shift_vals_mg + shift_vals_bg, 0, width - 1)
                new_x_bottom = np.clip(np.arange(width) - shift_vals_fg - shift_vals_mg - shift_vals_bg, 0, width - 1)

                # Convert mappings to proper format
                map_x_top = new_x_top.reshape(1, -1).astype(np.float32)
                map_x_bottom = new_x_bottom.reshape(1, -1).astype(np.float32)

                # Apply depth-based remapping
                top_frame[y] = cv2.remap(cropped_resized_frame, map_x_top, map_y[y].reshape(1, -1), interpolation=cv2.INTER_CUBIC)
                bottom_frame[y] = cv2.remap(cropped_resized_frame, map_x_bottom, map_y[y].reshape(1, -1), interpolation=cv2.INTER_CUBIC)

            # Apply Aspect Ratio Crop Before Resizing
            if selected_aspect_ratio.get() != "Default (16:9)":
                aspect_ratio_value = aspect_ratios[selected_aspect_ratio.get()]
                top_frame = apply_aspect_ratio_crop(top_frame, aspect_ratio_value)
                bottom_frame = apply_aspect_ratio_crop(bottom_frame, aspect_ratio_value)

            # ✅ Resize for Half-OU
            if output_format.get() == "Half-OU":
                half_height = height // 2
                top_frame = cv2.resize(top_frame, (width, half_height), interpolation=cv2.INTER_LANCZOS4)
                bottom_frame = cv2.resize(bottom_frame, (width, half_height), interpolation=cv2.INTER_LANCZOS4)

            # ✅ Format the 3D output properly
            ou_frame = format_3d_output(top_frame, bottom_frame, output_format.get())

            # ✅ Convert frame to RGB (FFmpeg requires RGB)
            ou_frame_rgb = cv2.cvtColor(ou_frame, cv2.COLOR_BGR2RGB)

            # ✅ Write frame using FFmpeg writer
            writer.append_data(ou_frame_rgb)

            print(f"🖼 Frame {frame_idx + 1}/{total_frames} | Format: {output_format.get()} | Frame Size: {ou_frame.shape}")

    finally:
        # ✅ Release video resources safely
        if original_cap.isOpened():
            original_cap.release()
        if depth_cap.isOpened():
            depth_cap.release()
        try:
            writer.close()  # ✅ Ensure the writer is closed properly
        except Exception as e:
            print(f"⚠ Warning: Error closing video writer: {e}")

        # ✅ Ensure progress bar reaches 100% if not canceled
        if not cancel_flag.is_set():
            progress["value"] = 100
            progress_label.config(text="✅ Processing complete!")
            progress.update_idletasks()
            print("✅ Over-Under 3D video generated successfully.")
        else:
            progress_label.config(text="❌ Processing Canceled.")
            progress["value"] = 0
            progress.update_idletasks()
            print("❌ Video processing was canceled by the user.")

        print("🎬 Rendering process finalized. All resources cleaned up.")
        

def start_processing_thread():
    global process_thread
    cancel_flag.clear()  # Reset cancel state
    suspend_flag.clear()  # Ensure it's not paused

    # Create a new thread for video processing
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

    # ✅ Run UI updates with `root.after()` to avoid freezing
    root.after(0, progress_label.config, {"text": "Processing... 0%"})
    root.after(0, progress["value"], 0)
    
    def update_ui_progress(value, text):
        root.after(0, progress_label.config, {"text": text})
        root.after(0, progress["value"], value)

    # Call rendering function based on format
    if output_format.get() == "Full-OU" or output_format.get() == "Half-OU":
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
            delay_time.get(),
            progress=progress,
            progress_label=progress_label,
            suspend_flag=suspend_flag,
            cancel_flag=cancel_flag
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
            delay_time.get(),
            progress=progress,
            progress_label=progress_label,
            suspend_flag=suspend_flag,
            cancel_flag=cancel_flag
        )

    if not cancel_flag.is_set():
        update_ui_progress(100, "✅ Processing Complete!")
        print("✅ Processing complete.")
    else:
        update_ui_progress(0, "❌ Processing Canceled.")
        print("❌ Processing canceled.")


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

    messagebox.showinfo("Settings Reset", "All values have been restored to defaults!")

root = tk.Tk()
root.title("VisionDepth3D Video Generator")
root.geometry("1080x780")

background_image = Image.open(resource_path("assets/Background.png"))
background_image = background_image.resize((1080, 780), Image.LANCZOS)
bg_image = ImageTk.PhotoImage(background_image)

background_label = tk.Label(root, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

content_frame = tk.Frame(root, highlightthickness=0, bd=0)
content_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.6, relheight=0.8)

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
output_format = tk.StringVar(value="SBS")
cinemascope_enabled = tk.BooleanVar(value=False)  # Checkbox variable
ou_type = tk.StringVar(value="Half-OU")


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
top_widgets_frame = tk.LabelFrame(content_frame, text="Video Info", padx=10, pady=10)
top_widgets_frame.grid(row=0, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

# Thumbnail
video_thumbnail_label = tk.Label(top_widgets_frame, text="No Thumbnail", bg="white", width=20, height=5)
video_thumbnail_label.grid(row=0, column=0, padx=10, pady=5)

video_specs_label = tk.Label(top_widgets_frame, text="Resolution: N/A\nFPS: N/A", justify="left")
video_specs_label.grid(row=0, column=1, padx=10, pady=5)

depth_map_label = tk.Label(top_widgets_frame, text="Depth Map: None", justify="left")
depth_map_label.grid(row=1, column=1, padx=10, pady=5)

progress = ttk.Progressbar(top_widgets_frame, orient="horizontal", length=300, mode="determinate")
progress.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

progress_label = tk.Label(top_widgets_frame, text="0%", font=("Arial", 10))
progress_label.grid(row=1, column=2, padx=10, pady=5, sticky="ew")


# Processing Options
options_frame = tk.LabelFrame(content_frame, text="Processing Options", padx=10, pady=10)
options_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

tk.Label(options_frame, text="Codec").grid(row=0, column=0, sticky="w")
codec_menu = tk.OptionMenu(options_frame, selected_codec, *codec_options)
codec_menu.grid(row=0, column=1, sticky="ew")

# Aspect Ratio Selection Dropdown
tk.Label(options_frame, text="Aspect Ratio").grid(row=0, column=2, sticky="w")

aspect_ratio_menu = tk.OptionMenu(options_frame, selected_aspect_ratio, *aspect_ratios.keys())
aspect_ratio_menu.grid(row=0, column=3, padx=5, sticky="ew")

reset_button = tk.Button(options_frame, text="Reset to Defaults", command=reset_settings, bg="#8B0000", fg="white")
reset_button.grid(row=0, column=4, columnspan=2, pady=10)


tk.Label(options_frame, text="Divergence Shift").grid(row=1, column=0, sticky="w")
tk.Scale(options_frame, from_=0, to=15, resolution=0.5, orient=tk.HORIZONTAL, variable=fg_shift).grid(row=1, column=1, sticky="ew")

tk.Label(options_frame, text="Sharpness Factor").grid(row=1, column=2, sticky="w")
tk.Scale(options_frame, from_=-1, to=1, resolution=0.1, orient=tk.HORIZONTAL, variable=sharpness_factor).grid(row=1, column=3, sticky="ew")

tk.Label(options_frame, text="Depth Transition").grid(row=2, column=0, sticky="w")
tk.Scale(options_frame, from_=0, to=8, resolution=0.5, orient=tk.HORIZONTAL, variable=mg_shift).grid(row=2, column=1, sticky="ew")

tk.Label(options_frame, text="Blend Factor").grid(row=2, column=2, sticky="w")
tk.Scale(options_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=blend_factor).grid(row=2, column=3, sticky="ew")

tk.Label(options_frame, text="Convergence Shift").grid(row=3, column=0, sticky="w")
tk.Scale(options_frame, from_=-5, to=0, resolution=0.5, orient=tk.HORIZONTAL, variable=bg_shift).grid(row=3, column=1, sticky="ew")

tk.Label(options_frame, text="Delay Time (seconds)").grid(row=3, column=2, sticky="w")
tk.Scale(options_frame, from_=1/50, to=1/20, resolution=0.001, orient=tk.HORIZONTAL, variable=delay_time).grid(row=3, column=3, sticky="ew")

# File Selection
tk.Button(content_frame, text="Select Input Video", command=select_input_video).grid(row=3, column=0, pady=5, sticky="ew")
tk.Entry(content_frame, textvariable=input_video_path, width=50).grid(row=3, column=1, pady=5, padx=5)

tk.Button(content_frame, text="Select Depth Map", command=select_depth_map).grid(row=4, column=0, pady=5, sticky="ew")
tk.Entry(content_frame, textvariable=selected_depth_map, width=50).grid(row=4, column=1, pady=5, padx=5)

tk.Button(content_frame, text="Select Output Video", command=select_output_video).grid(row=5, column=0, pady=5, sticky="ew")
tk.Entry(content_frame, textvariable=output_sbs_video_path, width=50).grid(row=5, column=1, pady=5, padx=5)

# Frame to Hold Buttons and Format Selection in a Single Row
button_frame = tk.Frame(content_frame)
button_frame.grid(row=6, column=0, columnspan=5, pady=10, sticky="w")

# 3D Format Label and Dropdown (Inside button_frame)
tk.Label(button_frame, text="3D Format").pack(side="left", padx=5)

option_menu = tk.OptionMenu(button_frame, output_format, "Half-SBS", "Full-SBS", "Half-OU", "Full-OU", "Red-Cyan Anaglyph", "Interlaced 3D")
option_menu.config(width=10)  # Adjust width to keep consistent look
option_menu.pack(side="left", padx=5)

# Buttons Inside button_frame to Keep Everything on One Line
start_button = tk.Button(button_frame, text="Generate 3D Video", command=start_processing_thread, bg="green", fg="white")
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
github_button = tk.Button(content_frame, image=github_icon_tk, command=open_github, borderwidth=0, bg="white", cursor="hand2")
github_button.image = github_icon_tk  # Keep a reference to prevent garbage collection
github_button.grid(row=7, column=0, pady=10, padx=5, sticky="w")  # Adjust positioning


# Load previous settings (if they exist)
load_settings()

# Ensure settings are saved when the program closes
root.protocol("WM_DELETE_WINDOW", lambda: (save_settings(), root.destroy()))

root.mainloop()
