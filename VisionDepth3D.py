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
from tensorflow import keras
from threading import Thread

# Load the trained divergence correction model
MODEL_PATH = 'weights/backward_warping_model.keras'
model = keras.models.load_model(MODEL_PATH)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Audio Transfer Function
def transferAudio(sourceVideo, targetVideo):
    tempAudioFileName = "./temp/audio.mkv"
    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")
    os.system(f'ffmpeg -y -i "{sourceVideo}" -c:a copy -vn "{tempAudioFileName}"')

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    os.system(f'ffmpeg -y -i "{targetNoAudio}" -i "{tempAudioFileName}" -c copy "{targetVideo}"')

    if os.path.getsize(targetVideo) == 0:
        tempAudioFileName = "./temp/audio.m4a"
        os.system(f'ffmpeg -y -i "{sourceVideo}" -c:a aac -b:a 160k -vn "{tempAudioFileName}"')
        os.system(f'ffmpeg -y -i "{targetNoAudio}" -i "{tempAudioFileName}" -c copy "{targetVideo}"')
        if os.path.getsize(targetVideo) == 0:
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio.")
        else:
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    shutil.rmtree("temp")  # Cleanup temp files

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

def detect_closeup(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) > 0:
        return "close-up"
    else:
        return "wide"

previous_scene_types = []

def smooth_scene_type(scene_type):
    global previous_scene_types
    previous_scene_types.append(scene_type)
    
    if len(previous_scene_types) > 10:  # Keep last 10 frames
        previous_scene_types.pop(0)
    
    return max(set(previous_scene_types), key=previous_scene_types.count)

def calculate_depth_intensity(depth_map):
    # Check if the depth map is grayscale or not
    if len(depth_map.shape) == 3 and depth_map.shape[2] == 3:
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    avg_depth = np.mean(depth_map)
    depth_variance = np.var(depth_map)
    
    # Define threshold for close-ups vs. wide shots
    return "close-up" if depth_variance < 500 else "wide"


def apply_convergence_shift(image, shift_value, direction='left'):
    height, width = image.shape[:2]
    M = np.float32([[1, 0, shift_value], [0, 1, 0]])
    shifted_image = cv2.warpAffine(image, M, (width, height))
    if direction == 'left':
        return shifted_image[:, :width - shift_value]
    else:
        return shifted_image[:, shift_value:]

def correct_divergence_shift(left_frame, right_frame, divergence_value):
    # Ensure divergence value is within trained range
    divergence_value = max(0.0, min(divergence_value, 5.0))  
    
    # Prepare input for the model
    divergence_input = np.array([[divergence_value]])
    
    # Predict warp parameters
    warp_params = model.predict(divergence_input).reshape(3, 3)

    # Apply warp transformation
    h, w = left_frame.shape[:2]
    corrected_left = cv2.warpPerspective(left_frame, warp_params, (w, h))
    corrected_right = cv2.warpPerspective(right_frame, warp_params, (w, h))

    # Create a mask for black regions after warping
    mask_left = cv2.cvtColor(corrected_left, cv2.COLOR_BGR2GRAY) == 0
    mask_right = cv2.cvtColor(corrected_right, cv2.COLOR_BGR2GRAY) == 0

    # Increase dilation to cover missed edges
    kernel = np.ones((5,5), np.uint8)  # Increase kernel size for stronger effect
    mask_left = cv2.dilate(mask_left.astype(np.uint8), kernel, iterations=3)
    mask_right = cv2.dilate(mask_right.astype(np.uint8), kernel, iterations=3)

    # Apply inpainting with adjusted mask
    corrected_left = cv2.inpaint(corrected_left, mask_left, 5, cv2.INPAINT_TELEA)
    corrected_right = cv2.inpaint(corrected_right, mask_right, 5, cv2.INPAINT_TELEA)

    return corrected_left, corrected_right

def render_sbs_3d(input_video, depth_video, output_video, codec, fps, width, height, fg_shift, mg_shift, bg_shift,
                  sharpness_factor, convergence_shift=0, divergence_shift=0, ipd_offset=0.0, ipd_mode='manual',
                  delay_time=1/30, blend_factor=0.5, progress=None, progress_label=None, batch_size=10, vram_limit=4):
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

    # Batch processing
    for frame_idx in range(0, total_frames, batch_size):
        batch_frames = []
        batch_depth_frames = []

        for i in range(batch_size):
            ret1, original_frame = original_cap.read()
            ret2, depth_frame = depth_cap.read()
            if not ret1 or not ret2:
                break

            batch_frames.append(original_frame)
            batch_depth_frames.append(depth_frame)

        # Process each frame in the batch
        for i in range(len(batch_frames)):
            original_frame = batch_frames[i]
            depth_frame = batch_depth_frames[i]

            # Calculate percentage
            percentage = ((frame_idx + i) / total_frames) * 100

            # Update progress bar
            if progress:
                progress["value"] = percentage
                progress.update()

            # Update percentage label
            if progress_label:
                progress_label.config(text=f"{percentage:.2f}%")

            # Calculate elapsed time and time remaining
            elapsed_time = time.time() - start_time
            if frame_idx + i > 0:
                time_per_frame = elapsed_time / (frame_idx + i)
                time_remaining = time_per_frame * (total_frames - (frame_idx + i))
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

            # Convert to grayscale for scene change detection
            current_frame_gray = cv2.cvtColor(cropped_resized_frame, cv2.COLOR_BGR2GRAY)

            # Scene change detection
            if prev_frame_gray is not None:
                diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
                diff_score = np.sum(diff) / (width * height)

                rolling_diff.append(diff_score)
                if len(rolling_diff) > max_rolling_frames:
                    rolling_diff.pop(0)
                avg_diff_score = np.mean(rolling_diff)

                adaptive_threshold = 50 if avg_diff_score < 100 else 75
                if avg_diff_score > adaptive_threshold:
                    print(f"Scene change detected at frame {frame_idx + i} with diff {avg_diff_score:.2f}")
                    frame_buffer.clear()  # Clear buffer for Pulfrich effect
                    blend_factor = max(0.1, blend_factor - 0.2)  # Reduce blending for scene change

            prev_frame_gray = current_frame_gray       

            # Scene classification (close-up or wide)
            scene_type = smooth_scene_type(detect_closeup(original_frame))                     
            left_frame, right_frame = cropped_resized_frame, cropped_resized_frame

            # Depth analysis
            depth_scene_type = calculate_depth_intensity(depth_frame)
            
            # Determine IPD Offset
            if ipd_mode == 'dynamic':
                ipd_value = dynamic_ipd_adjustment(depth_frame_resized)
            else:
                ipd_value = ipd_offset  # Use manual slider value

            # Apply divergence correction using the trained model and inpainting
            corrected_left, corrected_right = correct_divergence_shift(
                cropped_resized_frame, cropped_resized_frame, divergence_shift)

            # Apply IPD shift to left and right frames
            corrected_left, corrected_right = apply_ipd_offset(corrected_left, corrected_right, ipd_value)
            
            # Pulfrich effect adjustments
            blend_factor = min(0.5, blend_factor + 0.05) if len(frame_buffer) else blend_factor

            # Process Depth frame
            depth_map_gray = cv2.cvtColor(depth_frame_resized, cv2.COLOR_BGR2GRAY)
            depth_normalized = cv2.normalize(depth_map_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_filtered = cv2.bilateralFilter(depth_normalized, d=5, sigmaColor=50, sigmaSpace=50)
            depth_normalized = depth_filtered / 255.0             
            
            # Apply depth-based shifts
            left_frame, right_frame = cropped_resized_frame.copy(), cropped_resized_frame.copy()
            for y in range(height):
                fg_shift_val = fg_shift
                mg_shift_val = mg_shift
                bg_shift_val = bg_shift
                if convergence_shift > 0:
                    bg_shift_val += int(bg_shift_val * convergence_shift)
                if divergence_shift > 0:
                    fg_shift_val += int(fg_shift_val * divergence_shift)

                shift_vals_fg = (depth_normalized[y, :] * fg_shift_val).astype(np.int32)
                shift_vals_mg = (depth_normalized[y, :] * mg_shift_val).astype(np.int32)
                shift_vals_bg = (depth_normalized[y, :] * bg_shift_val).astype(np.int32)
                new_x_left = np.clip(np.arange(width) + shift_vals_fg + shift_vals_mg + shift_vals_bg, 0, width - 1)
                new_x_right = np.clip(np.arange(width) - shift_vals_fg - shift_vals_mg - shift_vals_bg, 0, width - 1)
                left_frame[y, new_x_left] = cropped_resized_frame[y, np.arange(width)]
                right_frame[y, new_x_right] = cropped_resized_frame[y, np.arange(width)]

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

            # Combine into SBS format
            sbs_frame = np.hstack((left_sharp_resized, right_sharp_resized))
            out.write(sbs_frame)

    original_cap.release()
    depth_cap.release()
    out.release()
    print("Half SBS video generated successfully.")

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

     # Get VRAM limit and batch size from the GUI inputs
    vram_limit = vram_limit_var.get()
    batch_size = batch_size_var.get()  # Use batch_size_var.get() to retrieve the value

    if vram_limit <= 0:
        messagebox.showerror("Error", "VRAM limit must be greater than 0.")
        return

    if batch_size <= 0:
        messagebox.showerror("Error", "Batch size must be greater than 0.")
        return
        
    # Render the video with batch processing and VRAM limit
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
        convergence_shift=convergence_shift.get(),
        divergence_shift=divergence_shift.get(),
        delay_time=delay_time.get(),
        ipd_offset=ipd_offset.get(),
        ipd_mode=ipd_mode.get(),
        blend_factor=blend_factor.get(),
        progress=progress,
        progress_label=progress_label,
        batch_size=batch_size,  # Adjust batch size as needed
        vram_limit=vram_limit  # Pass VRAM limit to the render function
    )

    # Set progress bar to 100% after rendering
    progress["value"] = 100
    progress_label.config(text="100%")
    progress.update()

    # Add Audio to the Generated SBS Video
    try:
        transferAudio(input_video_path.get(), output_sbs_video_path.get())
        progress_label.config(text="Complete")  # Indicate completion
        print("Audio transfer complete.")
    except Exception as e:
        print(f"Audio transfer failed: {e}")
        progress_label.config(text="Audio Error")  # Indicate an error
        messagebox.showwarning("Warning", "The video was generated without audio.")

def dynamic_ipd_adjustment(depth_frame, base_ipd=3.0):
    depth_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
    avg_depth = np.mean(depth_gray)  # Calculate average depth

    # Adjust IPD based on depth (scale IPD dynamically based on depth)
    dynamic_ipd = base_ipd * (1 + (avg_depth / 255.0))
    return max(0.5, min(dynamic_ipd, 10.0))  # Keep within reasonable bounds

def apply_ipd_offset(left_frame, right_frame, ipd_value):
    width = left_frame.shape[1]
    shift_pixels = int(ipd_value * width / 100)  # Convert IPD to pixel shift

    # Shift left and right images
    left_frame_shifted = np.roll(left_frame, shift_pixels, axis=1)
    right_frame_shifted = np.roll(right_frame, -shift_pixels, axis=1)

    return left_frame_shifted, right_frame_shifted

def update_ipd_mode(*args):
    if ipd_mode.get() == "manual":
        ipd_slider.config(state="normal")
    else:
        ipd_slider.config(state="disabled")

previous_ipd_values = []

def smoothed_ipd_adjustment(depth_frame, base_ipd=3.0):
    global previous_ipd_values
    new_ipd = dynamic_ipd_adjustment(depth_frame, base_ipd)

    if len(previous_ipd_values) > 5:
        previous_ipd_values.pop(0)

    previous_ipd_values.append(new_ipd)
    smoothed_ipd = np.mean(previous_ipd_values)
    return smoothed_ipd

def reset_ipd():
    ipd_offset.set(3.0)  # Reset to default

# GUI Setup
root = tk.Tk()
root.title("VisionDepth3D Video Generator")
root.geometry("1090x920")

background_image = Image.open(resource_path("assets\\Background.png"))
background_image = background_image.resize((1090,920), Image.LANCZOS)
bg_image = ImageTk.PhotoImage(background_image)

background_label = tk.Label(root, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

content_frame = tk.Frame(root, highlightthickness=0, bd=0)
content_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.7, relheight=0.8)

# Variables
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
convergence_shift = tk.DoubleVar(value=0.0)
divergence_shift = tk.DoubleVar(value=0.0)
ipd_mode = tk.StringVar(value="manual")
ipd_offset = tk.DoubleVar(value=3.0)
vram_limit_var = tk.DoubleVar(value=4.0)  # Default VRAM limit in GB
batch_size_var = tk.IntVar(value=10)  # Default batch size is 10

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

# VRAM Limit Input
tk.Label(top_widgets_frame, text="VRAM Limit (GB):").grid(row=2, column=0, sticky="w")
vram_limit_entry = tk.Entry(top_widgets_frame, textvariable=vram_limit_var, width=8)
vram_limit_entry.grid(row=2, column=1, sticky="w")

tk.Label(top_widgets_frame, text="Batch Size:").grid(row=3, column=0, sticky="w")
batch_size_entry = tk.Entry(top_widgets_frame, textvariable=batch_size_var, width=8)
batch_size_entry.grid(row=3, column=1, sticky="w")

# Start Processing Button
start_button = tk.Button(top_widgets_frame, text="Generate 3D SBS Video", command=process_video, bg="green", fg="white")
start_button.grid(row=6, column=3, columnspan=2, padx=10, pady=5, sticky="n")  # Corrected line

# Processing Options
options_frame = tk.LabelFrame(content_frame, text="Processing Options", padx=10, pady=10)
options_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

tk.Label(options_frame, text="Codec").grid(row=0, column=0, sticky="w")
codec_menu = tk.OptionMenu(options_frame, selected_codec, *codec_options)
codec_menu.grid(row=0, column=1, sticky="ew")

tk.Label(options_frame, text="Foreground Shift").grid(row=1, column=0, sticky="w")
tk.Entry(options_frame, textvariable=fg_shift, width=8).grid(row=1, column=1, sticky="w")

tk.Label(options_frame, text="Midground Shift").grid(row=2, column=0, sticky="w")
tk.Entry(options_frame, textvariable=mg_shift, width=8).grid(row=2, column=1, sticky="w")

tk.Label(options_frame, text="Background Shift").grid(row=3, column=0, sticky="w")
tk.Entry(options_frame, textvariable=bg_shift, width=8).grid(row=3, column=1, sticky="w")

tk.Label(options_frame, text="Sharpness Factor").grid(row=1, column=2, sticky="w")
tk.Scale(options_frame, from_=-1, to=1, resolution=0.1, orient=tk.HORIZONTAL, variable=sharpness_factor).grid(row=1, column=3)

tk.Label(options_frame, text="Blend Factor").grid(row=2, column=2, sticky="w")
tk.Scale(options_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=blend_factor).grid(row=2, column=3)

tk.Label(options_frame, text="Convergence Shift").grid(row=4, column=0, sticky="w")
tk.Entry(options_frame, textvariable=convergence_shift, width=8).grid(row=4, column=1, sticky="w")

tk.Label(options_frame, text="Divergence Shift").grid(row=5, column=0, sticky="w")
tk.Entry(options_frame, textvariable=divergence_shift, width=8).grid(row=5, column=1, sticky="w")

tk.Label(options_frame, text="Delay Time (seconds)").grid(row=3, column=2, sticky="w")
tk.Scale(options_frame, from_=1/50, to=1/20, resolution=0.001, orient=tk.HORIZONTAL, variable=delay_time).grid(row=3, column=3)

# IPD Controls
ipd_frame = tk.LabelFrame(content_frame, text="IPD Adjustment", padx=10, pady=10)
ipd_frame.grid(row=2, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

tk.Label(ipd_frame, text="Mode:").grid(row=0, column=0, sticky="w")
ipd_mode_menu = ttk.Combobox(ipd_frame, textvariable=ipd_mode, values=["manual", "dynamic"], state="readonly")
ipd_mode_menu.grid(row=0, column=1)
ipd_mode_menu.bind("<<ComboboxSelected>>", lambda e: update_ipd_mode())

tk.Label(ipd_frame, text="Manual Offset:").grid(row=1, column=0, sticky="w")
ipd_slider = tk.Scale(ipd_frame, from_=0.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, variable=ipd_offset)
ipd_slider.grid(row=1, column=1)

reset_button = tk.Button(ipd_frame, text="Reset IPD", command=reset_ipd)
reset_button.grid(row=1, column=2, padx=5)

# File Selection
tk.Button(content_frame, text="Select Input Video", command=select_input_video).grid(row=3, column=0, pady=5, sticky="ew")
tk.Entry(content_frame, textvariable=input_video_path, width=50).grid(row=3, column=1, pady=5, padx=5)

tk.Button(content_frame, text="Select Depth Map", command=select_depth_map).grid(row=4, column=0, pady=5, sticky="ew")
tk.Entry(content_frame, textvariable=selected_depth_map, width=50).grid(row=4, column=1, pady=5, padx=5)

tk.Button(content_frame, text="Select Output Video", command=select_output_video).grid(row=5, column=0, pady=5, sticky="ew")
tk.Entry(content_frame, textvariable=output_sbs_video_path, width=50).grid(row=5, column=1, pady=5, padx=5)


root.mainloop()
