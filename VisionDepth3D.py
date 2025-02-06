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
import time
import threading
from threading import Thread
import webbrowser
import json
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip
import imageio
from collections import deque

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

    if output_format == "SBS":
        return np.hstack((left_frame, right_frame))  # Full Side-by-Side

    elif output_format == "Half-SBS":
        # ✅ Resize both frames to HALF WIDTH while keeping full height
        half_width = width // 2
        left_resized = cv2.resize(left_frame, (half_width, height), interpolation=cv2.INTER_LANCZOS4)
        right_resized = cv2.resize(right_frame, (half_width, height), interpolation=cv2.INTER_LANCZOS4)

        # ✅ Stack resized frames into Half-SBS format
        return np.hstack((left_resized, right_resized))

    elif output_format == "Full-OU":
        return np.vstack((left_frame, right_frame))  # Full Over-Under

    elif output_format == "Half-OU":
        # ✅ Force exact 16:9 scaling for Half-OU
        target_height = width * 9 // 16 // 2  # Half the correct height

        # ✅ Resize both frames
        left_resized = cv2.resize(left_frame, (width, target_height), interpolation=cv2.INTER_LANCZOS4)
        right_resized = cv2.resize(right_frame, (width, target_height), interpolation=cv2.INTER_LANCZOS4)

        # ✅ Ensure black padding is present (Bigscreen VR requires it)
        padding_height = int(target_height * 0.02)  # 2% padding
        padding_bar = np.zeros((padding_height, width, 3), dtype=np.uint8)

        # ✅ Stack frames with padding
        return np.vstack((left_resized, padding_bar, right_resized))

    elif output_format == "Interlaced 3D":
        interlaced_frame = np.zeros_like(left_frame)
        interlaced_frame[::2] = left_frame[::2]  # Odd lines from left eye
        interlaced_frame[1::2] = right_frame[1::2]  # Even lines from right eye
        return interlaced_frame

    else:
        print(f"⚠ Warning: Unknown output format '{output_format}', defaulting to SBS.")
        return np.hstack((left_frame, right_frame))  # Default to SBS

def apply_cinemascope_crop(frame, format_type):
    """ 
    Crops frame to 2.39:1 aspect ratio while keeping width the same.
    Applies ONLY to Full-SBS and Half-SBS formats.
    """
    if format_type not in ["SBS", "Half-SBS"]:
        return frame  # ❌ Skip for Over-Under formats

    height, width = frame.shape[:2]
    target_height = int(width / 2.39)  # Maintain width, crop height

    print(f"🎬 Applying Cinemascope | Format: {format_type} | Original: {width}x{height} → Target: {width}x{target_height}")

    if target_height < height:
        crop_y = (height - target_height) // 2  # Center crop
        cropped_frame = frame[crop_y:crop_y + target_height, :]

        # ✅ Special handling for Half-SBS
        if format_type == "Half-SBS":
            cropped_frame = cv2.resize(cropped_frame, (width // 2, target_height), interpolation=cv2.INTER_LANCZOS4)

        return cropped_frame  # ✅ Correctly cropped & resized

    return frame  # Return unchanged if already 2.39:1


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


def render_sbs_3d(input_video, depth_video, output_video, codec, fps, width, height, 
                 fg_shift, mg_shift, bg_shift, delay_time=1/30, 
                 blend_factor=0.5, progress=None, progress_label=None, suspend_flag=None, cancel_flag=None):
    
    # Frame delay buffer
    frame_delay = int(fps * delay_time)
    frame_buffer = deque(maxlen=frame_delay + 1)
    
    # Precompute reusable components
    height_range = np.arange(height, dtype=np.float32)
    width_range = np.arange(width, dtype=np.float32)
    
    # Preallocate memory for shift maps
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

            # Convert frame to RGB
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            
            # Depth processing
            depth_gray = cv2.cvtColor(cv2.resize(depth_frame, (width, height)), cv2.COLOR_BGR2GRAY)

            # Smoothen depth map to remove harsh edges causing haloing
            depth_normalized = cv2.GaussianBlur(depth_gray, (9, 9), 0)

            # Normalize depth
            depth_normalized = cv2.normalize(depth_normalized, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            # Compute depth-based shifts with softened weights to avoid over-shifting
            combined_shift = (depth_normalized * (-0.7 * fg_shift - 0.5 * mg_shift) + (1 - depth_normalized) * (0.5 * bg_shift))

            # Ensure mapping coordinates are the same shape as original_frame
            map_x_left = np.clip(np.tile(width_range, (height, 1)) + combined_shift, 10, width - 10).astype(np.float32)
            map_x_right = np.clip(np.tile(width_range, (height, 1)) - combined_shift, 10, width - 10).astype(np.float32)
            map_y = np.tile(height_range[:, np.newaxis], (1, width)).astype(np.float32)
            
            # Remap the frames properly
            left_shifted = cv2.remap(original_frame, map_x_left, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            right_shifted = cv2.remap(original_frame, map_x_right, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # Edge-aware inpainting for missing pixels
            mask = (combined_shift > 0).astype(np.uint8) * 255
            left_shifted = cv2.inpaint(left_shifted, mask, 3, cv2.INPAINT_TELEA)
            right_shifted = cv2.inpaint(right_shifted, mask, 3, cv2.INPAINT_TELEA)

            # Frame buffering and blending
            frame_buffer.append((left_shifted, right_shifted))
            delayed_left, delayed_right = frame_buffer[0] if len(frame_buffer) > frame_delay else (left_shifted, right_shifted)

            # Vectorized blending (original)
            blended_left = cv2.addWeighted(delayed_left, blend_factor, left_shifted, 1 - blend_factor, 0)
            blended_right = cv2.addWeighted(delayed_right, blend_factor, right_shifted, 1 - blend_factor, 0)

            # 🔥 Apply Motion-Aware Smoothing to Reduce Ghosting 🔥
            if frame_idx > 0:
                prev_blended_left = cv2.addWeighted(frame_buffer[-1][0], 0.3, blended_left, 0.7, 0)
                prev_blended_right = cv2.addWeighted(frame_buffer[-1][1], 0.3, blended_right, 0.7, 0)
            else:
                prev_blended_left, prev_blended_right = blended_left, blended_right

            # Format output with the smoothed frames
            sbs_frame = format_3d_output(prev_blended_left, prev_blended_right, output_format.get())

            # Write frame to output
            writer.append_data(sbs_frame)


            # Update progress bar
            update_progress(frame_idx, total_frames, start_time, progress, progress_label)

    finally:
        writer.close()
        original_cap.release()
        depth_cap.release()

    # Add audio back to the final output
    command = [
        'ffmpeg',
        '-i', temp_output,  
        '-i', input_video,  
        '-c:v', 'copy',    
        '-c:a', 'aac',    
        '-map', '0:v:0',   
        '-map', '1:a:0',    
        '-y', output_video  
    ]
    
    try:
        subprocess.run(command, check=True)
    finally:
        if os.path.exists(temp_output):
            os.remove(temp_output)
    
    print("✅ Optimized SBS video generation complete.")


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

        elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
        time_remaining_str = time.strftime("%M:%S", time.gmtime(time_remaining))

        progress_label.config(
            text=f"{percentage:.2f}% | FPS: {fps:.2f} | Elapsed: {elapsed_time_str} | Remaining: {time_remaining_str}"
        )

def render_ou_3d(input_video, depth_video, output_video, codec, fps, width, height, 
                 fg_shift, mg_shift, bg_shift, delay_time=1/30, 
                 blend_factor=0.5, progress=None, progress_label=None, suspend_flag=None, cancel_flag=None):
    
    # Frame delay buffer for Pulfrich effect
    frame_delay = int(fps * delay_time)
    frame_buffer = deque(maxlen=frame_delay + 1)
    
    # Prepare intermediate output
    temp_output = output_video.rsplit('.', 1)[0] + '_temp.' + output_video.rsplit('.', 1)[1]
    
    # Initialize video writer
    writer = imageio.get_writer(temp_output, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    
    # Open video sources
    original_cap = cv2.VideoCapture(input_video)
    depth_cap = cv2.VideoCapture(depth_video)
    
    try:
        total_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = time.time()

        # ✅ Load warp model parameters
        warp_params = model.predict(np.array([[0.3]])).reshape(3, 3)

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

            # Convert frame to RGB
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

            # Process Depth frame
            depth_normalized = cv2.normalize(
                cv2.cvtColor(cv2.resize(depth_frame, (width, height)), cv2.COLOR_BGR2GRAY),
                None, 0, 1, cv2.NORM_MINMAX, dtype=np.float32
            )
            depth_normalized = cv2.GaussianBlur(depth_normalized, (5, 5), 0)

            # Compute depth-based shifts
            combined_shift = (depth_normalized * (-fg_shift - mg_shift) + (1 - depth_normalized) * bg_shift)
            
            # Create mapping coordinates
            width_range = np.arange(width, dtype=np.float32)
            map_x_top = np.clip(width_range + combined_shift, 0, width-1)
            map_x_bottom = np.clip(width_range - combined_shift, 0, width-1)

            # Apply sharpness adjustment before resizing
            top_shifted = apply_sharpness(top_shifted, sharpness_factor.get())
            bottom_shifted = apply_sharpness(bottom_shifted, sharpness_factor.get())

            # Handle Half-OU with exact 16:9 scaling
            if output_format.get() == "Half-OU":
                half_height = width * 9 // 16 // 2  # Ensure correct Half-OU aspect ratio

                # ✅ Properly scale both frames
                top_half = cv2.resize(top_shifted, (width, half_height), interpolation=cv2.INTER_LANCZOS4)
                bottom_half = cv2.resize(bottom_shifted, (width, half_height), interpolation=cv2.INTER_LANCZOS4)

                # ✅ Add black padding to ensure alignment
                padding_height = int(half_height * 0.02)
                padding_bar = np.zeros((padding_height, width, 3), dtype=np.uint8)

                # ✅ Stack properly
                ou_frame = np.vstack((top_half, padding_bar, bottom_half))
            else:
                # ✅ Full-OU (no resizing, stack directly)
                ou_frame = np.vstack((top_shifted, bottom_shifted))
                            
            # Write frame to output
            writer.append_data(ou_frame)

            # Update progress bar
            update_progress(frame_idx, total_frames, start_time, progress, progress_label)

    finally:
        writer.close()
        original_cap.release()
        depth_cap.release()

    # Add audio back to the final output
    command = [
        'ffmpeg',
        '-i', temp_output,  
        '-i', input_video,  
        '-c:v', 'copy',    
        '-c:a', 'aac',    
        '-map', '0:v:0',   
        '-map', '1:a:0',    
        '-y', output_video  
    ]
    
    try:
        subprocess.run(command, check=True)
    finally:
        if os.path.exists(temp_output):
            os.remove(temp_output)
    
    print("✅ Optimized Over-Under 3D video generation complete.")

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
    if output_format.get() == "Over-Under":
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

def save_settings():
    """ Saves all current settings to a JSON file """
    settings = {
        "fg_shift": fg_shift.get(),
        "mg_shift": mg_shift.get(),
        "bg_shift": bg_shift.get(),
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

cinemascope_checkbox = tk.Checkbutton(options_frame, text="CinemaScope", variable=cinemascope_enabled)
cinemascope_checkbox.grid(row=0, column=2, columnspan=2, pady=5, sticky="w")

reset_button = tk.Button(options_frame, text="Reset to Defaults", command=reset_settings, bg="#8B0000", fg="white")
reset_button.grid(row=0, column=3, columnspan=2, pady=10)

tk.Label(options_frame, text="Divergence Shift").grid(row=1, column=0, sticky="w")
tk.Scale(options_frame, from_=0, to=30, resolution=0.5, orient=tk.HORIZONTAL, variable=fg_shift).grid(row=1, column=1, sticky="ew", padx=5)

tk.Label(options_frame, text="Depth Transition").grid(row=2, column=0, sticky="w")
tk.Scale(options_frame, from_=0, to=15, resolution=0.5, orient=tk.HORIZONTAL, variable=mg_shift).grid(row=2, column=1, sticky="ew", padx=5)

tk.Label(options_frame, text="Convergence Shift").grid(row=3, column=0, sticky="w")
tk.Scale(options_frame, from_=-20, to=0, resolution=0.5, orient=tk.HORIZONTAL, variable=bg_shift).grid(row=3, column=1, sticky="ew", padx=5)

tk.Label(options_frame, text="Blend Factor").grid(row=1, column=2, sticky="w")
tk.Scale(options_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=blend_factor).grid(row=1, column=3, sticky="ew", padx=5)

tk.Label(options_frame, text="Delay Time (seconds)").grid(row=2, column=2, sticky="w")
tk.Scale(options_frame, from_=1/50, to=1/20, resolution=0.001, orient=tk.HORIZONTAL, variable=delay_time).grid(row=2, column=3, sticky="ew", padx=5)

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

option_menu = tk.OptionMenu(button_frame, output_format, "SBS", "Half-SBS", "Half-OU", "Full-OU", "Interlaced 3D")
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
github_button = tk.Button(content_frame, image=github_icon_tk, command=open_github, borderwidth=0, bg="white", cursor="hand2")
github_button.image = github_icon_tk  # Keep a reference to prevent garbage collection
github_button.grid(row=7, column=0, pady=10, padx=5, sticky="w")  # Adjust positioning


# Load previous settings (if they exist)
load_settings()

# Ensure settings are saved when the program closes
root.protocol("WM_DELETE_WINDOW", lambda: (save_settings(), root.destroy()))

root.mainloop()
