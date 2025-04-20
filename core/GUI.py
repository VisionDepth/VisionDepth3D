# ── Standard Library ─────────────────────────────
import os
import sys
import cv2
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from threading import Event
from audio import launch_audio_gui

# ── External Libraries ───────────────────────────
from PIL import Image, ImageTk
import torch.nn.functional as F
import numpy as np
import re
import webbrowser

# ── VisionDepth3D Custom Modules ────────────────
# 3D Rendering
from render_3d import (
    render_sbs_3d,
    format_3d_output,
    frame_to_tensor,
    depth_to_tensor,
    tensor_to_frame,
    pixel_shift_cuda,
    generate_anaglyph_3d,
    apply_sharpening,
    select_input_video,
    select_depth_map,
    select_output_video,
    process_video,
)

# Depth Estimation
from render_depth import (
    ensure_model_downloaded,
    update_pipeline,
    open_image,
    open_video,
    choose_output_directory,
    process_image,
    process_image_folder,
    process_images_in_folder,
    process_videos_in_folder,
    update_progress,
    cancel_requested,
)

from merged_pipeline import (
    start_merged_pipeline,
    select_video_and_generate_frames,
    select_output_file,
    select_frames_folder, 
)

# Video Player
from VDPlayer import (
    load_video,
    seek_video,
    play,
    pause_video,
    stop_video,
    open_fullscreen,
)

from preview_gui import open_3d_preview_window


# At the top of GUI.py
cancel_requested = threading.Event()

suspend_flag = Event()
cancel_flag = Event()
SETTINGS_FILE = "settings.json"



# ✅ Get absolute path to resource (for PyInstaller compatibility)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # ✅ Corrected for PyInstaller
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def save_settings():
    """Saves all current settings to a JSON file"""
    settings = {
        "input_video_path": input_video_path.get(),
        "selected_depth_map": selected_depth_map.get(),
        "output_sbs_video_path": output_sbs_video_path.get(),
        "ffmpeg_codec": selected_ffmpeg_codec.get(),
        "crf_value": crf_value.get(),
        "output_format": output_format.get(),

        # Depth 3D settings
        "fg_shift": fg_shift.get(),
        "mg_shift": mg_shift.get(),
        "bg_shift": bg_shift.get(),
        "sharpness_factor": sharpness_factor.get(),
        "blend_factor": blend_factor.get(),
        "delay_time": delay_time.get(),
        "feather_strength": feather_strength.get(),
        "blur_ksize": blur_ksize.get(),

        # 🆕 Additional toggle/sliders for 3D convergence behavior
        "parallax_balance": parallax_balance.get(),
        "max_pixel_shift": max_pixel_shift.get(),
        "use_subject_tracking": use_subject_tracking.get(),
        "use_floating_window": use_floating_window.get(),
        "auto_crop_black_bars": auto_crop_black_bars.get(),
        "preserve_original_aspect": preserve_original_aspect.get(),
        "convergence_offset": convergence_offset.get(),
    }

    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)
    print("💾 Settings saved.")


def load_settings():
    """Loads settings from the JSON file, if available"""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)

            input_video_path.set(settings.get("input_video_path", ""))
            selected_depth_map.set(settings.get("selected_depth_map", ""))
            output_sbs_video_path.set(settings.get("output_sbs_video_path", ""))

            selected_ffmpeg_codec.set(settings.get("ffmpeg_codec", "libx264"))
            crf_value.set(settings.get("crf_value", 23))
            output_format.set(settings.get("output_format", "Full-SBS"))

            # Depth shift params
            fg_shift.set(settings.get("fg_shift", 6.0))
            mg_shift.set(settings.get("mg_shift", 3.0))
            bg_shift.set(settings.get("bg_shift", -4.0))
            sharpness_factor.set(settings.get("sharpness_factor", 0.2))
            blend_factor.set(settings.get("blend_factor", 0.6))
            delay_time.set(settings.get("delay_time", 1 / 30))
            feather_strength.set(settings.get("feather_strength", 9.0))
            blur_ksize.set(settings.get("blur_ksize", 6))

            # 🆕 Additional render settings
            parallax_balance.set(settings.get("parallax_balance", 0.8))
            max_pixel_shift.set(settings.get("max_pixel_shift", 0.02))
            use_subject_tracking.set(settings.get("use_subject_tracking", True))
            use_floating_window.set(settings.get("use_floating_window", True))
            auto_crop_black_bars.set(settings.get("auto_crop_black_bars", False))
            preserve_original_aspect.set(settings.get("preserve_original_aspect", False))
            convergence_offset.set(settings.get("convergence_offset", 0.01))


        print("✅ Settings loaded from file.")


def reset_settings():
    """Resets all GUI values and UI elements to their default states."""

    # 🎬 File Paths and Codecs
    input_video_path.set("")
    selected_depth_map.set("")
    output_sbs_video_path.set("")
    selected_codec.set("mp4v")
    selected_ffmpeg_codec.set("libx264")
    output_format.set("Full-SBS")

    # 🧠 3D Shifting Parameters
    fg_shift.set(6.0)
    mg_shift.set(3.0)
    bg_shift.set(-4.0)

    # ✨ Visual Enhancements
    sharpness_factor.set(0.2)
    blend_factor.set(0.6)
    delay_time.set(1 / 30)

    # 🧼 Edge Cleanup
    feather_strength.set(9.0)
    blur_ksize.set(6)

    # 🎛️ Advanced Stereo Controls
    parallax_balance.set(0.8)
    max_pixel_shift.set(0.02)

    # 🟢 Toggles
    use_subject_tracking.set(False)
    use_floating_window.set(False)
    auto_crop_black_bars.set(False)
    preserve_original_aspect.set(False)
    convergence_offset.set(0.01)

    # 🎥 CRF for FFmpeg
    crf_value.set(23)

    # 🖼️ UI Resets
    try:
        video_thumbnail_label.config(image="", text="No preview")
        video_thumbnail_label.image = None
        video_specs_label.config(text="Video Info:\nResolution: -\nFPS: -")
    except Exception as e:
        print(f"⚠️ GUI reset skipped: {e}")

    # 🔁 Reset aspect preview if available
    try:
        update_aspect_preview()
    except Exception as e:
        print(f"⚠️ Aspect preview reset skipped: {e}")

    messagebox.showinfo("Settings Reset", "✅ All settings and preview panels reset to default!")


def cancel_processing():
    global cancel_flag, suspend_flag, cancel_requested  # Include all used flags
    cancel_flag.set()
    cancel_requested.set()  # For depth processing cancellation
    suspend_flag.clear()
    print("❌ Processing canceled (all systems).")


def suspend_processing():
    global suspend_flag
    suspend_flag.set()
    print("⏸ Processing Suspended!")

def resume_processing():
    global suspend_flag
    suspend_flag.clear()
    print("▶ Processing Resumed!")

def grab_frame_from_video(video_path, frame_idx=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def generate_preview_image(preview_type, left, right, shift_map, w, h):
    if preview_type == "Passive Interlaced":
        interlaced = np.zeros_like(left)
        interlaced[::2] = left[::2]
        interlaced[1::2] = right[1::2]
        return interlaced

    elif preview_type == "HSBS":
        half_w = w // 2
        left_resized = cv2.resize(left, (half_w, h))
        right_resized = cv2.resize(right, (half_w, h))
        return np.hstack((left_resized, right_resized))

    elif preview_type == "Shift Heatmap":
        shift_np = shift_map.cpu().numpy()
        shift_norm = cv2.normalize(shift_np, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(shift_norm.astype(np.uint8), cv2.COLORMAP_JET)

    elif preview_type == "Shift Heatmap (Abs)":
        shift_abs = np.abs(shift_map.cpu().numpy())
        shift_norm = cv2.normalize(shift_abs, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(shift_norm.astype(np.uint8), cv2.COLORMAP_JET)

    elif preview_type == "Shift Heatmap (Clipped ±5px)":
        shift_np = shift_map.cpu().numpy()
        max_disp = 5.0
        shift_clipped = np.clip(shift_np, -max_disp, max_disp)
        shift_norm = ((shift_clipped + max_disp) / (2 * max_disp)) * 255
        return cv2.applyColorMap(shift_norm.astype(np.uint8), cv2.COLORMAP_JET)

    elif preview_type == "Left-Right Diff":
        diff = cv2.absdiff(left, right)
        return diff

    elif preview_type == "Feather Blend":
        return left

    elif preview_type == "Feather Mask":
        shift_np = shift_map.cpu().numpy()
        feather_mask = np.clip(np.abs(shift_np) * 50, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(feather_mask, cv2.COLORMAP_BONE)

    elif preview_type == "Red-Blue Anaglyph":
        left = left.astype(np.uint8)
        right = right.astype(np.uint8)
        red_channel = left[:, :, 2]
        green_channel = right[:, :, 1]
        blue_channel = right[:, :, 0]
        anaglyph = cv2.merge((blue_channel, green_channel, red_channel))
        return anaglyph

    elif preview_type == "Overlay Arrows":
        # Simplified useful visual using arrows to debug horizontal displacement
        debug = left.copy()
        shift_np = shift_map.cpu().numpy()
        step = 20
        for y in range(0, h, step):
            for x in range(0, w, step):
                dx = int(shift_np[y, x] * 10)
                if abs(dx) > 1:
                    cv2.arrowedLine(debug, (x, y), (x + dx, y), (0, 255, 0), 1, tipLength=0.3)
        return debug

    return None



def preview_passive_3d_frame():
    input_path = input_video_path.get()
    depth_path = selected_depth_map.get()

    if not os.path.exists(input_path) or not os.path.exists(depth_path):
        messagebox.showerror("Missing Input", "Please load both input video and depth map.")
        return

    frame_idx = frame_to_preview_var.get()

    input_frame = grab_frame_from_video(input_path, frame_idx)
    depth_frame = grab_frame_from_video(depth_path, frame_idx)

    if input_frame is None or depth_frame is None:
        messagebox.showerror("Frame Error", "Unable to extract frames from videos.")
        return

    h, w = input_frame.shape[:2]
    frame_tensor = F.interpolate(
        frame_to_tensor(input_frame).unsqueeze(0),
        size=(h, w), mode='bilinear', align_corners=False
    ).squeeze(0)

    depth_tensor = F.interpolate(
        depth_to_tensor(depth_frame).unsqueeze(0),
        size=(h, w), mode='bilinear', align_corners=False
    ).squeeze(0)

    left, right, shift_map = pixel_shift_cuda(
        frame_tensor, depth_tensor, w, h,
        fg_shift.get(), mg_shift.get(), bg_shift.get(),
        blur_ksize=blur_ksize.get(),
        feather_strength=feather_strength.get(),
        return_shift_map=True,
        use_subject_tracking=use_subject_tracking.get(),
        enable_floating_window=use_floating_window.get()
    )

    preview_type = preview_mode.get()
    preview_img = generate_preview_image(preview_type, left, right, shift_map, w, h)


    # Save and open preview image
    if preview_img is not None:
        safe_preview_name = re.sub(r'[^a-zA-Z0-9_]', '', preview_type.replace(' ', '_').lower())
        preview_path = f"preview_{safe_preview_name}.png"
        cv2.imwrite(preview_path, preview_img)
        os.startfile(preview_path)
        print(f"✅ {preview_type} preview saved to: {preview_path}")
    else:
        messagebox.showwarning("Preview Error", f"Could not generate preview for: {preview_type}")

def update_aspect_preview(*args):
    try:
        ratio = aspect_ratios[selected_aspect_ratio.get()]
        format_selected = output_format.get()

        # 👇 Use .get() to access live values
        width = original_video_width.get()
        height = original_video_height.get()

        if format_selected == "Full-SBS":
            base_width = width * 2
        elif format_selected == "Half-SBS":
            base_width = width
        elif format_selected == "VR":
            base_width = 4096
            height = int(base_width / ratio)
        else:
            base_width = width
            height = int(base_width / ratio)

        aspect_preview_label.config(
            text=f"🧮 {base_width}x{height} ({ratio:.2f}:1)"
        )
    except Exception as e:
        aspect_preview_label.config(text="❌ Invalid Aspect Ratio")
        print(f"[Aspect Preview Error] {e}")


# ---GUI Setup---

# -----------------------
# Global Variables & Setup
# -----------------------

# --- Window Setup ---
root = tk.Tk()
root.title("VisionDepth3D Video Generator")
root.geometry("885x850")

# --- Notebook for Tabs ---
tab_control = ttk.Notebook(root)
tab_control.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)

# --- Depth Estimation GUI ---
depth_estimation_frame = tk.Frame(tab_control)
tab_control.add(depth_estimation_frame, text="Depth Estimation")

# Use the depth estimation tab’s content frame as the parent
depth_content_frame = tk.Frame(depth_estimation_frame, highlightthickness=0, bd=0)
depth_content_frame.pack(fill="both", expand=True)

# Sidebar Frame inside depth_content_frame
sidebar = tk.Frame(depth_content_frame, bg="#1c1c1c", width=250)
sidebar.pack(side="left", fill="y")

# Main Content Frame inside depth_content_frame
main_content = tk.Frame(depth_content_frame, bg="#2b2b2b")
main_content.pack(side="right", fill="both", expand=True)

# --- 3D Video Generator Tab ---
visiondepth_frame = tk.Frame(tab_control)
tab_control.add(visiondepth_frame, text="3D Video Generator")

# ✅ Same styled content_frame for VisionDepth3D tab
visiondepth_content_frame = tk.Frame(visiondepth_frame, highlightthickness=0, bd=0, bg="#1c1c1c")
visiondepth_content_frame.pack(fill="both", expand=True)

FrameTools3D = tk.Frame(tab_control, bg="#1c1c1c")
tab_control.add(FrameTools3D, text="📽 FrameTools 3D")

# --- VisionDepth3D Player Tab Setup ---
preview_Video = tk.Frame(tab_control)
tab_control.add(preview_Video, text="VDPlayer")

# Colors
bg_main = "#1e1e1e"
bg_controls = "#292929"
accent_color = "#4dd0e1"
fg_text = "white"

# Content Frame
player_content_frame = tk.Frame(preview_Video, bg=bg_main)
player_content_frame.pack(fill="both", expand=True)

# Top Controls
top_controls = tk.Frame(player_content_frame, bg=bg_controls, height=50)
top_controls.pack(fill="x", pady=(10, 5))

load_button = tk.Button(
    top_controls,
    text="📂 Load Video",
    command=lambda: load_video(video_frame, seek_bar, timestamp_label),
    bg=accent_color,
    fg="black",
    font=("Segoe UI", 10, "bold"),
    padx=12, pady=4,
    relief="flat",
    cursor="hand2",
    activebackground="#00acc1"
)
load_button.pack(side="left", padx=10)

# Placeholder Video Area
video_frame = tk.Label(
    player_content_frame,
    text="🎞️ No video loaded",
    bg="#121212",
    fg="gray",
    font=("Helvetica", 13, "italic"),
    width=80,
    height=20,
    anchor="center",
    justify="center"
)
video_frame.pack(pady=10)

# Style ttk elements
style = ttk.Style()
style.theme_use('clam')
style.configure(
    "TScale",
    background=bg_main,
    troughcolor="#444444",
    sliderthickness=10,
    sliderlength=14
)

# Scrubber
seek_bar = ttk.Scale(
    player_content_frame,
    from_=0, to=100,
    orient="horizontal",
    length=600,
    command=seek_video
)
seek_bar.pack(pady=5)

# Timestamp
timestamp_label = tk.Label(
    player_content_frame,
    text="00:00 / 00:00",
    bg=bg_main,
    fg=fg_text,
    font=("Segoe UI", 10)
)
timestamp_label.pack(pady=(0, 8))

# Playback Controls
bottom_controls = tk.Frame(player_content_frame, bg=bg_controls, height=40)
bottom_controls.pack(fill="x", pady=(0, 12))

def make_button(text, cmd):
    return tk.Button(
        bottom_controls,
        text=text,
        command=cmd,
        bg=accent_color,
        fg="black",
        font=("Segoe UI", 10, "bold"),
        padx=10,
        pady=3,
        relief="flat",
        cursor="hand2",
        activebackground="#00acc1"
    )

play_btn = make_button("▶ Play", lambda: play(video_frame, seek_bar, timestamp_label))
pause_btn = make_button("⏸ Pause", lambda: pause_video(video_frame, seek_bar, timestamp_label))
stop_btn = make_button("⏹ Stop", lambda: stop_video(video_frame, seek_bar, timestamp_label))
fullscreen_btn = make_button("🖥 Fullscreen", lambda: open_fullscreen(video_frame))

for btn in [play_btn, pause_btn, stop_btn, fullscreen_btn]:
    btn.pack(side="left", padx=10)

# Optional Status Label
status_bar = tk.Label(
    player_content_frame,
    text="🔋 Ready",
    bg=bg_main,
    fg="gray",
    font=("Segoe UI", 9, "italic"),
    anchor="w"
)
status_bar.pack(fill="x", padx=15, pady=(0, 5))


# --- Depth Content ---

# ✅ Define supported Hugging Face models
supported_models = {
    "Distil-Any-Depth-Large": "keetrap/Distil-Any-Depth-Large-hf",
    "Distil-Any-Depth-Small": "keetrap/Distill-Any-Depth-Small-hf",
    "Depth Anything V2 Large": "depth-anything/Depth-Anything-V2-Large-hf",
    "Depth Anything V2 Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth Anything V2 Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth Anything V1 Large": "LiheYoung/depth-anything-large-hf",
    "Depth Anything V1 Base": "LiheYoung/depth-anything-base-hf",
    "Depth Anything V1 Small": "LiheYoung/depth-anything-small-hf",
    "V2-Metric-Indoor-Large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "V2-Metric-Outdoor-Large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "DepthPro": "apple/DepthPro-hf",
    "marigold-depth-v1-0": "prs-eth/marigold-depth-v1-0",
    "ZoeDepth": "Intel/zoedepth-nyu-kitti",
    "MiDaS 3.0": "Intel/dpt-hybrid-midas",
    "DPT-Large": "Intel/dpt-large",
    "dpt-beit-large-512": "Intel/dpt-beit-large-512",
    "security_model": "nagayama0706/security_model",
}



selected_model = tk.StringVar(root, value="Distil-Any-Depth-Large")
colormap_var = tk.StringVar(root, value="Default")
invert_var = tk.BooleanVar(root, value=False)
save_frames_var = tk.BooleanVar(value=False)
output_dir = tk.StringVar(value="")



tk.Label(sidebar, text="Model", bg="#1c1c1c", fg="white", font=("Arial", 11)).pack(
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
model_dropdown.bind(
    "<<ComboboxSelected>>",
    lambda event: update_pipeline(selected_model, status_label)
)


output_dir_label = tk.Label(
    sidebar, text="Output Dir: None", bg="#1c1c1c", fg="white", wraplength=200
)
output_dir_label.pack(pady=5)
tk.Button(
    sidebar,
    text="Choose Directory",
    command=lambda: choose_output_directory(output_dir_label, output_dir),
    width=20
).pack(pady=5)


tk.Label(sidebar, text="Colormap:", bg="#1c1c1c", fg="white").pack(pady=5)
colormap_dropdown = ttk.Combobox(
    sidebar,
    textvariable=colormap_var,
    values=["Default", "Magma", "Viridis", "Inferno", "Plasma", "Gray"],
    state="readonly",
    width=22,
)
colormap_dropdown.pack(pady=5)

invert_checkbox = tk.Checkbutton(
    sidebar, text="Invert Depth", variable=invert_var, bg="#1c1c1c", fg="white"
)
invert_checkbox.pack(pady=5)

save_frames_checkbox = tk.Checkbutton(
    sidebar, text=" Save Frames", variable=save_frames_var, bg="#1c1c1c", fg="white"
)
save_frames_checkbox.pack(pady=5)

tk.Label(sidebar, text="Batch Size (Frames):", bg="#1c1c1c", fg="white").pack(pady=5)

batch_size_entry = tk.Entry(sidebar, width=22)
batch_size_entry.insert(0, "8")  # Default value
batch_size_entry.pack(pady=5)


# ✅ Add event listener to update batch size dynamically
def update_batch_size(*args):
    try:
        batch_size = int(batch_size_entry.get().strip())
        if batch_size <= 0:
            raise ValueError
        status_label.config(text=f"🔄 Batch Size Updated: {batch_size}")
    except ValueError:
        status_label.config(text="⚠️ Invalid batch size. Using default (8).")


batch_size_entry.bind("<Return>", update_batch_size)  # Update on "Enter" key press
batch_size_entry.bind("<FocusOut>", update_batch_size)  # Update when user clicks away


#tk.Label(sidebar, text="Video Resolution (w,h):", bg="#1c1c1c", fg="white").pack(
#    pady=5
#)
#resolution_entry = tk.Entry(sidebar, width=22)
#resolution_entry.insert(0, "")
#resolution_entry.pack(pady=5)

progress_bar = ttk.Progressbar(sidebar, mode="determinate", length=180)
progress_bar.pack(pady=10)
status_label = tk.Label(
    sidebar, text="Ready", bg="#1c1c1c", fg="white", width=30, wraplength=200
)
status_label.pack(pady=5)


# --- Depth Content: Image previews ---
# --- Top Frame: For the original image ---
top_frame = tk.Frame(main_content, bg="#2b2b2b")
top_frame.pack(pady=10)

input_label = tk.Label(top_frame, text="Input Image", bg="#2b2b2b", fg="white")
input_label.pack()  # No side=, so it stacks vertically

# --- Middle Frame: For the buttons ---
button_frame = tk.Frame(main_content, bg="#2b2b2b")
button_frame.pack(pady=10)

tk.Button(
    button_frame,
    text="Process Image",
    command=lambda: open_image(
        status_label,
        progress_bar,
        colormap_var,
        invert_var,
        output_dir,
        input_label,
        output_label
    ),
    width=25,
    bg="#4a4a4a",
    fg="white",
).pack(pady=2)


tk.Button(
    button_frame,
    text="Process Image Folder",
    command=lambda: process_image_folder(
    batch_size_entry,
    output_dir,
    status_label,
    progress_bar,
    root
),
    width=25,
    bg="#4a4a4a",
    fg="white",
).pack(pady=2)

tk.Button(
    button_frame,
    text="Process Video",
    command=lambda: open_video(status_label, progress_bar, batch_size_entry, output_dir, invert_var),
    width=25,
    bg="#4a4a4a",
    fg="white",
).pack(pady=2)

tk.Button(
    button_frame,
    text="Select Video Folder",
    command=lambda: process_videos_in_folder(
        filedialog.askdirectory(),  # folder_path from dialog
        batch_size_entry,
        output_dir,
        status_label,
        progress_bar,
        cancel_requested
    ),
    width=25,
    bg="#4a4a4a",
    fg="white",
).pack(pady=2)


# --- Bottom Frame: For the depth map ---
bottom_frame = tk.Frame(main_content, bg="#2b2b2b")
bottom_frame.pack(pady=10)

output_label = tk.Label(bottom_frame, text="Depth Map", bg="#2b2b2b", fg="white")
output_label.pack()


# 🧠 Variables
ft3d_frames_folder = tk.StringVar()
ft3d_output_file = tk.StringVar()
ft3d_width = tk.IntVar(value=1920)
ft3d_height = tk.IntVar(value=804)
ft3d_fps = tk.DoubleVar(value=23.976)
ft3d_codec = tk.StringVar(value="XVID")
ft3d_enable_rife = tk.BooleanVar(value=True)
ft3d_enable_upscale = tk.BooleanVar(value=False)
ft3d_fps_multiplier = tk.IntVar(value=2)
ft3d_blend_mode = tk.StringVar(value="OFF")
ft3d_input_res_pct = tk.IntVar(value=100)
ft3d_selected_model = tk.StringVar(value="RealESR_Gx4_fp16")



REAL_ESRGAN_MODELS = {
    "RealESR_Gx4_fp16": "weights/RealESR_Gx4_fp16.onnx",
    "RealESRGAN_x4_fp16": "weights/RealESRGANx4_fp16.onnx",
    "RealESR_Animex4_fp16": "weights/RealESR_Animex4_fp16.onnx",
    "BSRGANx2_fp16": "weights/BSRGANx2_fp16.onnx",
    "BSRGANx4_fp16": "weights/BSRGANx4_fp16.onnx"
}


# 🎛️ Common settings
COMMON_FPS = [23.976, 24, 30, 48, 60, 120]
FPS_MULTIPLIERS = [2, 4, 8]
CODECS = {
    "XVID (Good Compatibility)": "XVID",
    "MJPG (Motion JPEG)": "MJPG",
    "MP4V (Standard MPEG-4)": "MP4V",
    "DIVX (Older Compatibility)": "DIVX",
}

# ─── FrameTools3D GUI Layout ──────────────────────────────────────────────
# Extract Button
tk.Button(
    FrameTools3D,
    text="Extract Frames from Video",
    command=lambda: select_video_and_generate_frames(ft3d_frames_folder.set),
    bg="green",
    fg="white",
    relief="flat"
).pack(pady=10)

# Input / Output Group
io_frame = tk.LabelFrame(FrameTools3D, text="Input / Output", bg="#1c1c1c", fg="white")
io_frame.pack(fill="x", padx=10, pady=4)

tk.Label(io_frame, text="Frames Folder:", bg="#1c1c1c", fg="white").pack(anchor="w", padx=10, pady=(6, 2))
tk.Entry(io_frame, textvariable=ft3d_frames_folder, width=50, bg="#2b2b2b", fg="white", insertbackground="white").pack(pady=2)
tk.Button(io_frame, text="Browse", command=lambda: select_frames_folder(ft3d_frames_folder), bg="#4a4a4a", fg="white").pack(pady=2)

tk.Label(io_frame, text="Output Video File:", bg="#1c1c1c", fg="white").pack(anchor="w", padx=10, pady=(6, 2))
tk.Entry(io_frame, textvariable=ft3d_output_file, width=50, bg="#2b2b2b", fg="white", insertbackground="white").pack(pady=2)
tk.Button(io_frame, text="Save As", command=lambda: select_output_file(ft3d_output_file), bg="#4a4a4a", fg="white").pack(pady=2)

# Processing Options
proc_frame = tk.LabelFrame(FrameTools3D, text="⚙️ Processing Options", bg="#1c1c1c", fg="white")
proc_frame.pack(fill="x", padx=10, pady=4)

tk.Checkbutton(proc_frame, text="Enable RIFE Interpolation", variable=ft3d_enable_rife, bg="#1c1c1c", fg="white", selectcolor="#2b2b2b").pack(anchor="w", padx=10, pady=2)
tk.Checkbutton(proc_frame, text="Enable Real-ESRGAN Upscale", variable=ft3d_enable_upscale, bg="#1c1c1c", fg="white", selectcolor="#2b2b2b").pack(anchor="w", padx=10, pady=2)

#Output Settings
out_frame = tk.LabelFrame(FrameTools3D, text="Output Settings", bg="#1c1c1c", fg="white")
out_frame.pack(fill="x", padx=10, pady=4)

res_box = tk.Frame(out_frame, bg="#1c1c1c")
res_box.pack(anchor="w", padx=10, pady=4)
tk.Label(res_box, text="Resolution (WxH):", bg="#1c1c1c", fg="white").pack(side="left")
tk.Entry(res_box, textvariable=ft3d_width, width=6, bg="#2b2b2b", fg="white", insertbackground="white").pack(side="left", padx=4)
tk.Label(res_box, text="x", bg="#1c1c1c", fg="white").pack(side="left")
tk.Entry(res_box, textvariable=ft3d_height, width=6, bg="#2b2b2b", fg="white", insertbackground="white").pack(side="left", padx=4)

def combo_row(parent, label_text, var, values):
    row = tk.Frame(parent, bg="#1c1c1c")
    row.pack(anchor="w", padx=10, pady=4, fill="x")
    tk.Label(row, text=label_text, bg="#1c1c1c", fg="white", width=22, anchor="w").pack(side="left")
    ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=20).pack(side="left")

combo_row(out_frame, "Original FPS:", ft3d_fps, COMMON_FPS)
combo_row(out_frame, "FPS Interpolation Multiplier:", ft3d_fps_multiplier, FPS_MULTIPLIERS)
combo_row(out_frame, "Output Codec:", ft3d_codec, list(CODECS.keys()))

# ESRGAN Settings
esrgan_frame = tk.LabelFrame(FrameTools3D, text="ESRGAN Settings", bg="#1c1c1c", fg="white")
esrgan_frame.pack(fill="x", padx=10, pady=4)

combo_row(esrgan_frame, "AI Blending:", ft3d_blend_mode, ["OFF", "LOW", "MEDIUM", "HIGH"])
combo_row(esrgan_frame, "Input Resolution %:", ft3d_input_res_pct, [25, 50, 75, 100])
combo_row(esrgan_frame, "Model Selection:", ft3d_selected_model, list(REAL_ESRGAN_MODELS.keys()))

# ▶️ Start Button
tk.Button(
    FrameTools3D,
    text="▶ Start Processing",
    bg="green", fg="white", relief="flat",
    command=lambda: threading.Thread(
        target=start_merged_pipeline,
        args=(
            {
                "frames_folder": ft3d_frames_folder.get(),
                "output_file": ft3d_output_file.get(),
                "width": ft3d_width.get(),
                "height": ft3d_height.get(),
                "fps": ft3d_fps.get(),
                "fps_multiplier": ft3d_fps_multiplier.get(),
                "codec": CODECS.get(ft3d_codec.get(), "XVID"),
                "enable_rife": ft3d_enable_rife.get(),
                "enable_upscale": ft3d_enable_upscale.get(),
                "blend_mode": ft3d_blend_mode.get(),
                "input_res_pct": ft3d_input_res_pct.get(),
                "model_path": REAL_ESRGAN_MODELS.get(ft3d_selected_model.get(), "weights/RealESR_Gx4_fp16")
            },
            merged_progress,
            merged_status,
        ),
        daemon=True
    ).start()
).pack(pady=12)

# 📊 Progress Bar
merged_progress = ttk.Progressbar(FrameTools3D, orient="horizontal", length=300, mode="determinate")
merged_progress.pack(pady=6)

merged_status = tk.Label(FrameTools3D, text="Waiting to start...", bg="#1c1c1c", fg="white")
merged_status.pack()


# ---3D Generator Frame Contents ---

# Dark Theme Styling
STYLE_BG = "#1c1c1c"
STYLE_ENTRY = "#2b2b2b"
STYLE_FG = "white"
STYLE_TROUGH = "#444"

input_video_path = tk.StringVar()
selected_depth_map = tk.StringVar()
output_sbs_video_path = tk.StringVar()
selected_codec = tk.StringVar(value="XVID")
fg_shift = tk.DoubleVar(value=6.5)
mg_shift = tk.DoubleVar(value=-1.5)
bg_shift = tk.DoubleVar(value=-12.0)
sharpness_factor = tk.DoubleVar(value=0.2)
blend_factor = tk.DoubleVar(value=0.6)
delay_time = tk.DoubleVar(value=1 / 30)
output_format = tk.StringVar(value="Full-SBS")
blur_ksize = tk.IntVar(value=1)
feather_strength = tk.DoubleVar(value=0.0)
selected_ffmpeg_codec = tk.StringVar(value="h264_nvenc")
crf_value = tk.IntVar(value=23)
use_ffmpeg = tk.BooleanVar(value=False)
use_subject_tracking = tk.BooleanVar(value=True)
use_floating_window = tk.BooleanVar(value=True)
preview_mode = tk.StringVar(value="Passive Interlaced")
frame_to_preview_var = tk.IntVar(value=6478)
original_video_width = tk.IntVar(value=1920)
original_video_height = tk.IntVar(value=1080)
preserve_content = tk.BooleanVar(value=True)
max_pixel_shift = tk.DoubleVar(value=0.02)
auto_crop_black_bars = tk.BooleanVar(value=True)
parallax_balance = tk.DoubleVar(value=0.8)
preserve_original_aspect = tk.BooleanVar(value=False)
nvenc_cq_value = tk.IntVar(value=23)
convergence_offset = tk.DoubleVar(value=0.01)


load_settings()

aspect_ratios = {
    "Default (16:9)": 16 / 9,
    "Classic (4:3)": 4 / 3,
    "Square (1:1)": 1.0,
    "Vertical 9:16": 9 / 16,
    "Instagram 4:5": 4 / 5,
    "CinemaScope (2.39:1)": 2.39,
    "Anamorphic (2.35:1)": 2.35,
    "Modern Cinema (2.40:1)": 2.40,
    "Ultra Panavision (2.76:1)": 2.76,
    "Academy Flat (1.85:1)": 1.85,
    "European Flat (1.66:1)": 1.66,
    "21:9 UltraWide": 21 / 9,
    "32:9 SuperWide": 32 / 9,
    "2:1 (Modern Hybrid)": 2.0,
}

selected_aspect_ratio = tk.StringVar(value="Default (16:9)")

codec_options = ["mp4v", "XVID", "DIVX"]

FFMPEG_CODEC_MAP = {
    "H.264 / AVC (libx264)": "libx264",
    "H.265 / HEVC (libx265)": "libx265",
    "MPEG-4 (mp4v)": "mp4v",
    "XviD (AVI - CPU)": "XVID",
    "DivX (AVI - CPU)": "DIVX",
    "AVC (NVENC GPU)": "h264_nvenc",
    "HEVC / H.265 (NVENC GPU)": "hevc_nvenc",
    "AV1 (CPU)": "libaom-av1",
    "AV1 (NVIDIA)": "av1_nvenc",
}

# Layout frames

top_widgets_frame = tk.LabelFrame(
    visiondepth_content_frame,
    text="Video Info",
    bg="#1c1c1c",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    labelanchor="nw",
    padx=10,
    pady=10
)
top_widgets_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")


# Thumbnail
video_thumbnail_label = tk.Label(
    top_widgets_frame, text="No Thumbnail", bg="#1c1c1c", fg="white"
)
video_thumbnail_label.grid(row=0, column=0, padx=10, pady=5)

video_specs_label = tk.Label(
    top_widgets_frame, text="Resolution: N/A\nFPS: N/A", justify="left", bg="#1c1c1c", fg="white"
)
video_specs_label.grid(row=0, column=1, padx=10, pady=5)

aspect_preview_label = tk.Label(top_widgets_frame, text="", font=("Segoe UI", 8, "italic"), bg="#1c1c1c", fg="white")
aspect_preview_label.grid(row=1, column=0, sticky="w", padx=5)

# 🔁 Bind aspect ratio dropdown to preview label
selected_aspect_ratio.trace_add("write", update_aspect_preview)
update_aspect_preview()

depth_map_label = tk.Label(
    top_widgets_frame, text="Depth Map (3D): None", bg="#1c1c1c", fg="white", justify="left", wraplength=200
)
depth_map_label.grid(row=1, column=1, padx=10, pady=5)

progress = ttk.Progressbar(
    top_widgets_frame, orient="horizontal", length=300, mode="determinate"
)
progress.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

progress_label = tk.Label(top_widgets_frame, text="0%", font=("Arial", 10), bg="#1c1c1c", fg="white")
progress_label.grid(row=1, column=2, padx=10, pady=5, sticky="ew")


# Processing Options
options_frame = tk.LabelFrame(
    visiondepth_content_frame,
    text="Processing Options",
    bg="#1c1c1c",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    labelanchor="nw",
    padx=10,
    pady=10
)
options_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")


# Ensure uniform spacing
for i in range(4):
    options_frame.columnconfigure(i, weight=1)

# Row 0
tk.Checkbutton(
    options_frame,
    text="Preserve Original Aspect Ratio", bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=preserve_original_aspect
).grid(row=0, column=0, sticky="w", padx=5)

tk.Checkbutton(
    options_frame,
    text="Auto Crop Black Bars", bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=auto_crop_black_bars,
    anchor="e",
    justify="left"
).grid(row=0, column=1, sticky="w", padx=5)

tk.Checkbutton(
    options_frame,
    text="Stabilize Zero-Parallax (center-depth)", bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=use_subject_tracking,
    anchor="w",
    justify="left"
).grid(row=0, column=2, sticky="w", padx=5)

tk.Checkbutton(
    options_frame,
    text="Enable Floating Window (DFW)", bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=use_floating_window,
    anchor="e",
    justify="left"
).grid(row=0, column=3, sticky="e", padx=5)

# Row 1
tk.Checkbutton(
    options_frame,
    text="Use FFmpeg Renderer", bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=use_ffmpeg,
    anchor="w",
    justify="left"
).grid(row=1, column=0, sticky="w", padx=5)

# Row 2
tk.Label(options_frame, text="Foreground Shift", bg="#1c1c1c", fg="white").grid(row=2, column=0, sticky="w")
tk.Scale(options_frame, from_=0, to=15, resolution=0.5, orient=tk.HORIZONTAL, variable=fg_shift, bg="#1c1c1c", fg="white")\
    .grid(row=2, column=1, sticky="ew")

tk.Label(options_frame, text="Midground Shift", bg="#1c1c1c", fg="white").grid(row=2, column=2, sticky="w")
tk.Scale(options_frame, from_=-5, to=5, resolution=0.5, orient=tk.HORIZONTAL, variable=mg_shift, bg="#1c1c1c", fg="white")\
    .grid(row=2, column=3, sticky="ew")

# Row 3
tk.Label(options_frame, text="Background Shift", bg="#1c1c1c", fg="white").grid(row=3, column=0, sticky="w")
tk.Scale(options_frame, from_=-15, to=0, resolution=0.5, orient=tk.HORIZONTAL, variable=bg_shift, bg="#1c1c1c", fg="white")\
    .grid(row=3, column=1, sticky="ew")

tk.Label(options_frame, text="Sharpness Factor", bg="#1c1c1c", fg="white").grid(row=3, column=2, sticky="w")
tk.Scale(options_frame, from_=-1, to=1, resolution=0.1, orient=tk.HORIZONTAL, variable=sharpness_factor, bg="#1c1c1c", fg="white")\
    .grid(row=3, column=3, sticky="ew")

#Row 4
tk.Label(options_frame, text="convergence offset", bg="#1c1c1c", fg="white").grid(row=4, column=0, sticky="w")
tk.Scale(options_frame, from_=-0.05, to=0.05, resolution=0.001, orient=tk.HORIZONTAL, variable=convergence_offset, length=200, bg="#1c1c1c", fg="white")\
    .grid(row=4, column=1, sticky="ew")
 
tk.Label(options_frame, text="Parallax Balance", bg="#1c1c1c", fg="white").grid(row=4, column=2, sticky="w")
tk.Scale(
    options_frame,
    from_=0.0,
    to=1.0,
    resolution=0.05,
    orient="horizontal",
    variable=parallax_balance,
    bg="#1c1c1c", fg="white"
).grid(row=4, column=3, sticky="ew")

#Row 5
tk.Label(options_frame, text="Max Pixel Shift (%)", bg="#1c1c1c", fg="white").grid(row=5, column=0, sticky="w")
tk.Scale(options_frame, from_=0.005, to=0.10, resolution=0.005, orient=tk.HORIZONTAL, variable=max_pixel_shift, length=200, bg="#1c1c1c", fg="white")\
    .grid(row=5, column=1, sticky="ew")   

# File Selection
tk.Button(
    visiondepth_content_frame,
    text="Select Input Video", bg="#1c1c1c", fg="white",
    command=lambda: select_input_video(
        input_video_path,
        video_thumbnail_label,
        video_specs_label,
        update_aspect_preview,
        original_video_width,
        original_video_height
    )


).grid(row=3, column=0, pady=5, sticky="ew")

tk.Entry(visiondepth_content_frame, textvariable=input_video_path, width=50, bg="#1c1c1c", fg="white").grid(
    row=3, column=1, pady=5, padx=5
)

tk.Button(
    visiondepth_content_frame,
    text="Select Depth Map", bg="#1c1c1c", fg="white",
    command=lambda: select_depth_map(selected_depth_map, depth_map_label)
).grid(row=4, column=0, pady=5, sticky="ew")

tk.Entry(visiondepth_content_frame, textvariable=selected_depth_map, width=50, bg="#1c1c1c", fg="white").grid(
    row=4, column=1, pady=5, padx=5
)

tk.Button(
    visiondepth_content_frame,
    text="Select Output Video", bg="#1c1c1c", fg="white",
    command=lambda: select_output_video(output_sbs_video_path)
).grid(row=5, column=0, pady=5, sticky="ew")

tk.Entry(visiondepth_content_frame, textvariable=output_sbs_video_path, width=50,  bg="#1c1c1c", fg="white").grid(
    row=5, column=1, pady=5, padx=5
)

# Frame to Hold Buttons and Format Selection in a Single Row
button_frame = tk.Frame(visiondepth_content_frame, bg="#1c1c1c")
button_frame.grid(row=7, column=0, columnspan=5, pady=10, sticky="w")

# 3D Format Label and Dropdown (Inside button_frame)
tk.Label(button_frame, text="3D Format", bg="#1c1c1c", fg="white").pack(side="left", padx=5)

option_menu = tk.OptionMenu(
    button_frame,
    output_format,
    "Full-SBS",
    "Half-SBS",
    "VR",
    "Red-Cyan Anaglyph",
    "Passive Interlaced",
)
option_menu.config(width=10)  # Adjust width to keep consistent look
option_menu.pack(side="left", padx=5)

# Buttons Inside button_frame to Keep Everything on One Line
start_button = tk.Button(
    button_frame,
    text="Generate 3D Video",
    bg="green",
    fg="white",
    command=lambda: (
        save_settings(),
        process_video(
            input_video_path,
            selected_depth_map,
            output_sbs_video_path,
            selected_codec,
            fg_shift,
            mg_shift,
            bg_shift,
            sharpness_factor,
            output_format,
            selected_aspect_ratio,
            aspect_ratios,
            feather_strength,    
            blur_ksize,           
            progress,
            progress_label,
            suspend_flag,
            cancel_flag,
            use_ffmpeg,
            selected_ffmpeg_codec,
            crf_value,
            use_subject_tracking,
            use_floating_window,
            max_pixel_shift,
            auto_crop_black_bars,
            parallax_balance,
            preserve_original_aspect,
        )
    )
)

start_button.pack(side="left", padx=5)


preview_button = tk.Button(
    button_frame,
    text="Open Preview",
    command=lambda: open_3d_preview_window(
        input_video_path,
        selected_depth_map,
        fg_shift,
        mg_shift,
        bg_shift,
        blur_ksize,
        feather_strength,
        use_subject_tracking,
        use_floating_window
    )
)
preview_button.pack(side="left", padx=5)


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

# Row 7 - Reset button centered
reset_button = tk.Button(
    button_frame, text="Reset to Defaults", command=reset_settings, bg="#8B0000", fg="white"
)
reset_button.pack(side="left", padx=5)

# 🔲 Encoding Settings Group
encoding_frame = tk.LabelFrame(
    visiondepth_content_frame,
    text="Encoding Settings",
    bg="#1c1c1c",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    labelanchor="nw",
    padx=10,
    pady=10
)
encoding_frame.grid(row=8, column=0, columnspan=5, padx=10, pady=10, sticky="ew")

# Make columns evenly resize
for i in range(6):
    encoding_frame.columnconfigure(i, weight=1)

# 🧮 Aspect Ratio
tk.Label(encoding_frame, text="Aspect Ratio:", bg="#1c1c1c", fg="white").grid(row=0, column=0, sticky="w", padx=5)
tk.OptionMenu(encoding_frame, selected_aspect_ratio, *aspect_ratios.keys())\
    .grid(row=0, column=1, sticky="ew", padx=5)

# 🧰 FFmpeg Codec
tk.Label(encoding_frame, text="FFmpeg Codec:", bg="#1c1c1c", fg="white").grid(row=0, column=2, sticky="w", padx=5)
tk.OptionMenu(encoding_frame, selected_ffmpeg_codec, *FFMPEG_CODEC_MAP.keys())\
    .grid(row=0, column=3, sticky="ew", padx=5)

# 🎞️ Codec
tk.Label(encoding_frame, text="Codec:", bg="#1c1c1c", fg="white").grid(row=0, column=4, sticky="w", padx=5)
tk.OptionMenu(encoding_frame, selected_codec, *codec_options)\
    .grid(row=0, column=5, sticky="ew", padx=5)

# 📉 CRF
tk.Label(encoding_frame, text="CRF", bg="#1c1c1c", fg="white").grid(row=1, column=0, sticky="w", padx=5)
tk.Scale(encoding_frame, from_=0, to=51, resolution=1,
         orient=tk.HORIZONTAL, variable=crf_value, length=150,
         bg="#2b2b2b", fg="white", troughcolor="#444")\
    .grid(row=1, column=1, columnspan=2, sticky="ew", padx=5)

# 🚀 NVENC CQ
tk.Label(encoding_frame, text="NVENC CQ", bg="#1c1c1c", fg="white").grid(row=1, column=3, sticky="w", padx=5)
tk.Scale(encoding_frame, from_=0, to=51, resolution=1,
         orient=tk.HORIZONTAL, variable=nvenc_cq_value, length=150,
         bg="#2b2b2b", fg="white", troughcolor="#444")\
    .grid(row=1, column=4, columnspan=2, sticky="ew", padx=5)


# Row 9 – Icon Buttons + Audio Tool
def open_github():
    """Opens the GitHub repository in a web browser."""
    webbrowser.open_new("https://github.com/VisionDepth/VisionDepth3D")

def open_aspect_ratio_CheatSheet():
    """Opens the Aspect Ratio Cheat Sheet."""
    webbrowser.open_new("https://www.wearethefirehouse.com/aspect-ratio-cheat-sheet")

# Load GitHub icon
github_icon_path = resource_path(os.path.join("assets", "github_Logo.png"))
if not os.path.exists(github_icon_path):
    print(f"❌ ERROR: Missing github_Logo.png at {github_icon_path}")
    sys.exit(1)

github_icon = Image.open(github_icon_path).resize((15, 15), Image.LANCZOS)
github_icon_tk = ImageTk.PhotoImage(github_icon)

# Load CheatSheet icon
CheatSheet_icon_path = resource_path(os.path.join("assets", "cheatsheet.png"))
if not os.path.exists(CheatSheet_icon_path):
    print(f"❌ ERROR: Missing cheatsheet.png at {CheatSheet_icon_path}")
    sys.exit(1)

CheatSheet_icon = Image.open(CheatSheet_icon_path).resize((15, 15), Image.LANCZOS)
CheatSheet_icon_tk = ImageTk.PhotoImage(CheatSheet_icon)

# 🔹 Combine GitHub, Cheat Sheet, and Audio Tool into one frame
bottom_links_frame = tk.Frame(visiondepth_content_frame, bg="#1c1c1c")
bottom_links_frame.grid(row=9, column=0, columnspan=6, sticky="w", padx=10, pady=10)

# GitHub Button
github_button = tk.Button(
    bottom_links_frame,
    image=github_icon_tk,
    command=open_github,
    borderwidth=0,
    bg="white",
    cursor="hand2"
)
github_button.image = github_icon_tk
github_button.pack(side="left", padx=5)

# Cheat Sheet Button
CheatSheet_button = tk.Button(
    bottom_links_frame,
    image=CheatSheet_icon_tk,
    command=open_aspect_ratio_CheatSheet,
    borderwidth=0,
    bg="white",
    cursor="hand2"
)
CheatSheet_button.image = CheatSheet_icon_tk
CheatSheet_button.pack(side="left", padx=5)

# 🎵 Audio Tool Button (text button next to icons)
audio_tool_button = tk.Button(
    bottom_links_frame,
    text="🎵 Audio Tool", bg="#1c1c1c", fg="white",
    command=launch_audio_gui
)
audio_tool_button.pack(side="left", padx=10)

# Ensure settings are saved when the program closes
def on_exit():
    save_settings()        # 💾 Save settings
    root.destroy()         # ❌ Close GUI

root.protocol("WM_DELETE_WINDOW", on_exit)


root.mainloop()
