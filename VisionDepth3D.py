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
from moviepy.editor import VideoFileClip, AudioFileClip
import json
import subprocess


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
    if output_format == "SBS":
        return np.hstack((left_frame, right_frame))  # Standard Side-by-Side

    elif output_format == "Over-Under":
        if ou_type.get() == "Half-OU":
            # Resize frames to Half-OU size while keeping proper scaling
            top_half = cv2.resize(left_frame, (width, height // 2), interpolation=cv2.INTER_LANCZOS4)
            bottom_half = cv2.resize(right_frame, (width, height // 2), interpolation=cv2.INTER_LANCZOS4)

            # Ensure the padding bar is always present in Half-OU
            padding_height = int(height * 0.02)  # Adjust percentage to control black bar size
            padding_bar = np.zeros((padding_height, width, 3), dtype=np.uint8)  # Black bar

            # Stack the frames with the black bar in the middle
            return np.vstack((top_half, padding_bar, bottom_half))

        else:  # Full-OU
            return np.vstack((left_frame, right_frame))  # Full frames, no padding bar
     # Stack top and bottom frames

    elif output_format == "Interlaced 3D":
        interlaced_frame = np.zeros_like(left_frame)
        interlaced_frame[::2] = left_frame[::2]  # Odd lines from left eye
        interlaced_frame[1::2] = right_frame[1::2]  # Even lines from right eye
        return interlaced_frame

    else:
        print(f"⚠ Warning: Unknown output format '{output_format}', defaulting to SBS.")
        return np.hstack((left_frame, right_frame))  # Default to SBS

def apply_cinemascope_crop(frame):
    """ Crops frame to 2.39:1 aspect ratio while keeping width the same. """
    height, width = frame.shape[:2]
    target_height = int(width / 2.39)  # Maintain width, crop height
    
    if target_height < height:
        crop_y = (height - target_height) // 2  # Center crop
        cropped_frame = frame[crop_y:crop_y + target_height, :]
        
        # If Half-OU, ensure aspect ratio matches expected dimensions
        if output_format.get() == "Half-OU":
            cropped_frame = cv2.resize(cropped_frame, (width, height // 2), interpolation=cv2.INTER_LANCZOS4)
        
        return cropped_frame
    return frame  # Return unchanged if already 2.39:1

# Function to detect and remove black bars
def remove_black_bars(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return frame, 0, 0, frame.shape[1], frame.shape[0]

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    if w < frame.shape[1] * 0.5 or h < frame.shape[0] * 0.5:
        return frame, 0, 0, frame.shape[1], frame.shape[0]

    return frame[y:y+h, x:x+w], x, y, w, h
    
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
    
    # Define threshold for close-ups vs. wide shots
    if depth_variance < 500:  
        return "close-up"
    else:
        return "wide"



def render_sbs_3d(input_video, depth_video, output_video, codec, fps, width, height, fg_shift, mg_shift, bg_shift,
                  sharpness_factor, delay_time=1/30, blend_factor=0.5, progress=None, progress_label=None):
    frame_delay = int(fps * delay_time)  # Number of frames corresponding to the delay time
    frame_buffer = []  # Buffer to store frames for temporal delay
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    original_cap = cv2.VideoCapture(input_video)
    depth_cap = cv2.VideoCapture(depth_video)

    prev_frame_gray = None
    rolling_diff = []
    max_rolling_frames = 5  # Number of frames to average for adaptive threshold

    print("Creating Half SBS video with Pulfrich effect, blending, and black bar removal")
  
    # Start tracking time
    start_time = time.time()

    # Get total number of frames
    total_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(total_frames):
        ret1, original_frame = original_cap.read()
        ret2, depth_frame = depth_cap.read()
        if not ret1 or not ret2:
            break

        # Calculate percentage
        percentage = (frame_idx / total_frames) * 100

        # Update progress bar
        if progress:
            progress["value"] = percentage
            progress.update()

        # Update percentage label
        if progress_label:
            progress_label.config(text=f"{percentage:.2f}%")

        # Calculate elapsed time and time remaining
        elapsed_time = time.time() - start_time
        if frame_idx > 0:
            time_per_frame = elapsed_time / frame_idx
            time_remaining = time_per_frame * (total_frames - frame_idx)
        else:
            time_remaining = 0  # No estimate initially

        # Format time values as MM:SS
        elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
        time_remaining_str = time.strftime("%M:%S", time.gmtime(time_remaining))

        # Update the progress bar and label
        if progress_label:
            progress_label.config(
                text=f"{percentage:.2f}% | Elapsed: {elapsed_time_str} | Remaining: {time_remaining_str}"
            )

        # Remove black bars
        cropped_frame, x, y, w, h = remove_black_bars(original_frame)
        cropped_resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_AREA)
        depth_frame_resized = cv2.resize(depth_frame, (width, height))

        left_frame, right_frame = cropped_resized_frame, cropped_resized_frame

        # Convert to grayscale for scene change detection
        current_frame_gray = cv2.cvtColor(cropped_resized_frame, cv2.COLOR_BGR2GRAY)

         # Depth analysis
        depth_scene_type = calculate_depth_intensity(depth_frame)     
        
        corrected_left, corrected_right = left_frame, right_frame  # No IPD adjustment
        
        # Pulfrich effect adjustments
        blend_factor = min(0.5, blend_factor + 0.05) if len(frame_buffer) else blend_factor

        # Process Depth frame
        depth_map_gray = cv2.cvtColor(depth_frame_resized, cv2.COLOR_BGR2GRAY)
        depth_normalized = cv2.normalize(depth_map_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_filtered = cv2.bilateralFilter(depth_normalized, d=5, sigmaColor=50, sigmaSpace=50)
        depth_normalized = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
        depth_normalized = depth_filtered / 255.0             
        
        # Ensure left and right frames are fresh copies of the cropped frame
        left_frame, right_frame = cropped_resized_frame.copy(), cropped_resized_frame.copy()

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
            left_frame[y] = cv2.remap(cropped_resized_frame, map_x_left, map_y[y].reshape(1, -1), interpolation=cv2.INTER_CUBIC)
            right_frame[y] = cv2.remap(cropped_resized_frame, map_x_right, map_y[y].reshape(1, -1), interpolation=cv2.INTER_CUBIC)
        
        if cinemascope_enabled.get():
            left_frame = apply_cinemascope_crop(left_frame)
            right_frame = apply_cinemascope_crop(right_frame)
            
        # Buffer logic
        frame_buffer.append((left_frame, right_frame))
        if len(frame_buffer) > frame_delay:
            delayed_left_frame, delayed_right_frame = frame_buffer.pop(0)
        else:
            delayed_left_frame, delayed_right_frame = left_frame, right_frame

        # Create Pulfrich effect
        blended_left_frame = cv2.addWeighted(delayed_left_frame, blend_factor, left_frame, 1 - blend_factor, 0)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5 + sharpness_factor, -1], [0, -1, 0]])
        left_sharp = cv2.filter2D(blended_left_frame, -1, sharpen_kernel)
        right_sharp = cv2.filter2D(right_frame, -1, sharpen_kernel)
        
        left_sharp_resized = cv2.resize(left_sharp, (width // 2, height))
        right_sharp_resized = cv2.resize(right_sharp, (width // 2, height))

        sbs_frame = format_3d_output(left_sharp_resized, right_sharp_resized, output_format.get())
        out.write(sbs_frame)

    original_cap.release()
    depth_cap.release()
    out.release()
    print("Half SBS video generated successfully.")

def render_ou_3d(input_video, depth_video, output_video, codec, fps, width, height, fg_shift, mg_shift, bg_shift,
                  sharpness_factor, delay_time=1/30, blend_factor=0.5, progress=None, progress_label=None):

    frame_delay = int(fps * delay_time)  # Number of frames corresponding to the delay time
    frame_buffer = []  # Buffer to store frames for temporal delay
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    original_cap = cv2.VideoCapture(input_video)
    depth_cap = cv2.VideoCapture(depth_video)

    prev_frame_gray = None
    rolling_diff = []
    scene_change_detected = False
    scene_stability_counter = 0  # Helps prevent false positives

    print("Creating Over-Under 3D video with Pulfrich effect, blending, and black bar removal")

    # Start tracking time
    start_time = time.time()
    total_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(total_frames):
        # Read frame
        ret1, original_frame = original_cap.read()
        ret2, depth_frame = depth_cap.read()
        if not ret1 or not ret2:
            break

        # 🎯 Scene classification (close-up or wide)
        scene_type = smooth_scene_type(detect_closeup(original_frame))
        depth_scene_type = calculate_depth_intensity(depth_frame)

        # 🛠 Update depth shifts dynamically
        fg_shift, mg_shift, bg_shift = update_depth_shifts(scene_type, fg_shift, mg_shift, bg_shift)

        # Update progress bar
        percentage = (frame_idx / total_frames) * 100
        if progress:
            progress["value"] = percentage
            progress.update()
        if progress_label:
            progress_label.config(text=f"{percentage:.2f}%")

        # Scene change detection
        current_frame_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        if prev_frame_gray is not None:
            diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
            diff_score = np.mean(diff)

            rolling_diff.append(diff_score)
            if len(rolling_diff) > 8:  # Keep rolling buffer size at 8
                rolling_diff.pop(0)

            avg_diff_score = np.mean(rolling_diff)
            adaptive_threshold = 40 if avg_diff_score < 80 else 70
            scene_change_detected = avg_diff_score > adaptive_threshold

            # 🎬 Handle scene change
            if scene_change_detected and scene_stability_counter > 5:
                print(f"🎬 Scene change detected at frame {frame_idx}, diff: {avg_diff_score:.2f}")
                frame_buffer.clear()  # Reset Pulfrich buffer

                # Smooth blending instead of abrupt cuts
                blend_factor = np.clip(blend_factor * 0.9, 0.3, 0.7)  # Adjust transition smoothness

                # **Update depth shifts dynamically**
                fg_shift, mg_shift, bg_shift = update_depth_shifts(scene_type, fg_shift, mg_shift, bg_shift)

                scene_stability_counter = 0  # Reset stability counter
            else:
                scene_stability_counter += 1  # Increase stability count

        prev_frame_gray = current_frame_gray.copy()

        # Process Depth frame
        depth_map_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
        depth_normalized = cv2.normalize(depth_map_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_filtered = cv2.bilateralFilter(depth_normalized, d=5, sigmaColor=50, sigmaSpace=50)
        depth_normalized = depth_filtered / 255.0

        # Prepare top and bottom frames
        top_frame, bottom_frame = original_frame.copy(), original_frame.copy()

        # Apply depth-based shifts
        for y in range(height):
            shift_vals_fg = (-depth_normalized[y, :] * fg_shift).astype(np.float32)
            shift_vals_mg = (-depth_normalized[y, :] * mg_shift).astype(np.float32)
            shift_vals_bg = (depth_normalized[y, :] * bg_shift).astype(np.float32)

            new_x_top = np.clip(np.arange(width) + shift_vals_fg + shift_vals_mg + shift_vals_bg, 0, width - 1)
            new_x_bottom = np.clip(np.arange(width) - shift_vals_fg - shift_vals_mg - shift_vals_bg, 0, width - 1)

            map_x_top = new_x_top.reshape(1, -1).astype(np.float32)
            map_x_bottom = new_x_bottom.reshape(1, -1).astype(np.float32)

            top_frame[y] = cv2.remap(original_frame, map_x_top, np.full_like(map_x_top, y), 
                                     interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            bottom_frame[y] = cv2.remap(original_frame, map_x_bottom, np.full_like(map_x_bottom, y), 
                                        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Pulfrich effect
        frame_buffer.append((top_frame, bottom_frame))
        if len(frame_buffer) > frame_delay:
            delayed_top_frame, delayed_bottom_frame = frame_buffer.pop(0)
        else:
            delayed_top_frame, delayed_bottom_frame = top_frame, bottom_frame

        blended_top_frame = cv2.addWeighted(delayed_top_frame, blend_factor, top_frame, 1 - blend_factor, 0)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5 + sharpness_factor, -1], [0, -1, 0]])
        top_sharp = cv2.filter2D(blended_top_frame, -1, sharpen_kernel)
        bottom_sharp = cv2.filter2D(bottom_frame, -1, sharpen_kernel)

        # Apply Cinemascope crop if enabled
        if cinemascope_enabled.get():
            top_sharp = apply_cinemascope_crop(top_sharp)
            bottom_sharp = apply_cinemascope_crop(bottom_sharp)

        # Format the 3D output properly
        ou_frame = format_3d_output(top_sharp, bottom_sharp, output_format.get())

        out.write(ou_frame)

    original_cap.release()
    depth_cap.release()
    out.release()
    print("✅ Over-Under 3D video generated successfully.")

def start_processing_thread():
    thread = Thread(target=process_video)
    thread.start()

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
            sharpness_factor.get(),
            delay_time=delay_time.get(),
            blend_factor=blend_factor.get(),
            progress=progress,
            progress_label=progress_label
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
            delay_time=delay_time.get(),
            blend_factor=blend_factor.get(),
            progress=progress,
            progress_label=progress_label
        )
            
    # Set progress bar to 100% after rendering
    progress["value"] = 100
    progress_label.config(text="100%")
    progress.update()

SETTINGS_FILE = "settings.json"

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
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
            fg_shift.set(settings.get("fg_shift", 6.0))
            mg_shift.set(settings.get("mg_shift", 3.0))
            bg_shift.set(settings.get("bg_shift", -4.0))
            sharpness_factor.set(settings.get("sharpness_factor", 0.2))
            blend_factor.set(settings.get("blend_factor", 0.6))
            delay_time.set(settings.get("delay_time", 1/30))


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

# Start Processing Button
start_button = tk.Button(top_widgets_frame, text="Generate 3D SBS Video", command=process_video, bg="green", fg="white")
start_button.grid(row=2, column=2, columnspan=2, pady=10)

# Processing Options
options_frame = tk.LabelFrame(content_frame, text="Processing Options", padx=10, pady=10)
options_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

tk.Label(options_frame, text="Codec").grid(row=0, column=0, sticky="w")
codec_menu = tk.OptionMenu(options_frame, selected_codec, *codec_options)
codec_menu.grid(row=0, column=1, sticky="ew")

cinemascope_checkbox = tk.Checkbutton(options_frame, text="CinemaScope Mode (2.39:1)", variable=cinemascope_enabled)
cinemascope_checkbox.grid(row=0, column=2, columnspan=2, pady=5, sticky="w")


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

tk.Label(content_frame, text="3D Format").grid(row=6, column=0, sticky="w")
option_menu = tk.OptionMenu(content_frame, output_format, "SBS", "Half-OU", "Full-OU", "Interlaced 3D",)
option_menu.config(width=12)  # Adjust width to fit the text without stretching
option_menu.grid(row=6, column=1, pady=5, padx=5, sticky="w")  # Align left without stretching

# Load the GitHub icon from assets
github_icon_path = resource_path("assets/github_Logo.png")
github_icon = Image.open(github_icon_path)
github_icon = github_icon.resize((15, 15), Image.LANCZOS)  # Resize to fit UI
github_icon_tk = ImageTk.PhotoImage(github_icon)

# Create the clickable GitHub icon button
github_button = tk.Button(content_frame, image=github_icon_tk, command=open_github, borderwidth=0, bg="white", cursor="hand2")
github_button.image = github_icon_tk  # Keep a reference to prevent garbage collection
github_button.grid(row=7, column=0, pady=10, padx=5, sticky="w")  # Adjust positioning

reset_button = tk.Button(content_frame, text="Reset to Defaults", command=reset_settings, bg="#8B0000", fg="white")
reset_button.grid(row=6, column=1, columnspan=2, pady=10)

# Load previous settings (if they exist)
load_settings()

# Ensure settings are saved when the program closes
root.protocol("WM_DELETE_WINDOW", lambda: (save_settings(), root.destroy()))

root.mainloop()
