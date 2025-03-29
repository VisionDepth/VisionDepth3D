# ── Standard Library ─────────────────────────────
import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Event
from audio import launch_audio_gui

# ── External Libraries ───────────────────────────
from PIL import Image, ImageTk

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
    open_github,
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
)

# RIFE Frame Interpolation
from render_framestitch import (
    start_processing,
    select_video_and_generate_frames,
    select_frames_folder,
    select_output_file,
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

def cancel_processing():
    global cancel_flag, suspend_flag  # 🛠 This line fixes the scope error
    cancel_flag.set()
    suspend_flag.clear()
    print("❌ Processing canceled.")

def suspend_processing():
    global suspend_flag
    suspend_flag.set()
    print("⏸ Processing Suspended!")

def resume_processing():
    global suspend_flag
    suspend_flag.clear()
    print("▶ Processing Resumed!")


# ---GUI Setup---

# -----------------------
# Global Variables & Setup
# -----------------------

# --- Window Setup ---
root = tk.Tk()
root.title("VisionDepth3D Video Generator")
root.geometry("1080x790")

background_image = Image.open(resource_path(os.path.join("assets", "Background.png")))
background_image = background_image.resize((1080, 790), Image.LANCZOS)
bg_image = ImageTk.PhotoImage(background_image)

root.bg_image = bg_image  # keep a persistent reference
background_label = tk.Label(root, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# --- Notebook for Tabs ---
tab_control = ttk.Notebook(root)
tab_control.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8, relheight=1.0)

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
visiondepth_content_frame = tk.Frame(visiondepth_frame, highlightthickness=0, bd=0)
visiondepth_content_frame.pack(fill="both", expand=True)

# --- VDStitch Interpolation Tab ---
VDStitch = tk.Frame(tab_control)  # Create a new frame stitch tab
tab_control.add(VDStitch, text="VDStitch+RIFE")  # Add to notebook

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

# --- RIFE FPS Interpolation Tab (Coming Soon Placeholder) ---
RealESRGAN = tk.Frame(tab_control)  # Create a new frame stitch tab
tab_control.add(RealESRGAN, text="Real-ESRGAN")  # Add to notebook

# Centered Label saying "Real-ESRGAN Coming Soon"
RealESRGAN_placeholder_label = tk.Label(
    RealESRGAN,
    text="🛠️ Real-ESRGAN - Coming Soon!",
    font=("Arial", 16, "bold"),
    fg="gray",
)
RealESRGAN_placeholder_label.pack(expand=True)  # Center the label


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
}



selected_model = tk.StringVar(root, value="Distil-Any-Depth-Large")
colormap_var = tk.StringVar(root, value="Default")
invert_var = tk.BooleanVar(root, value=False)
save_frames_var = tk.BooleanVar(value=False)
output_dir = tk.StringVar(value="")



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
model_dropdown.bind(
    "<<ComboboxSelected>>",
    lambda event: update_pipeline(selected_model, status_label)
)


output_dir_label = tk.Label(
    sidebar, text="📂 Output Dir: None", bg="#1c1c1c", fg="white", wraplength=200
)
output_dir_label.pack(pady=5)
tk.Button(
    sidebar,
    text="Choose Directory",
    command=lambda: choose_output_directory(output_dir_label, output_dir),
    width=20
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

tk.Label(sidebar, text="📦 Batch Size (Frames):", bg="#1c1c1c", fg="white").pack(pady=5)

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

cancel_btn = tk.Button(
    sidebar,
    text="❌ Cancel Processing",
    command=lambda: cancel_requested.set(True),
    bg="red",
    fg="white"
)
cancel_btn.pack(pady=5)


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
    text="🖼️ Process Image Folder",
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
    text="🎥 Process Video",
    command=lambda: open_video(status_label, progress_bar, batch_size_entry, output_dir),
    width=25,
    bg="#4a4a4a",
    fg="white",
).pack(pady=2)
tk.Button(
    button_frame,
    text="📂 Select Video Folder",
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

output_label = tk.Label(bottom_frame, text="🌊 Depth Map", bg="#2b2b2b", fg="white")
output_label.pack()

# --- VDStitch Sidebar (Dark Theme) ---

# --- VDStitch Contents ---

# Variables
frames_folder = tk.StringVar()
output_file = tk.StringVar()
width = tk.IntVar(value=1920)
height = tk.IntVar(value=800)
fps = tk.DoubleVar(value=24)
selected_codec = tk.StringVar(value="XVID")
enable_fps_interpolation = tk.BooleanVar(value=False)
target_fps = tk.DoubleVar(value=60)
fps_multiplier = tk.IntVar(value=2)

# Available codecs (Fastest first)
CODECS = {
    "XVID (Good Compatibility)": "XVID",
    "MJPG (Motion JPEG)": "MJPG",
    "MP4V (Standard MPEG-4)": "MP4V",
    "DIVX (Older Compatibility)": "DIVX",
}

# Common FPS Values
COMMON_FPS = [23.976, 24, 30, 48, 60, 120]
MULTIPLIERS = [2, 4, 8]


VDStitch.configure(bg="#1c1c1c")  # Set background for the entire tab

# Generate Frames Button (Extract frames from video)
generate_frames_btn = tk.Button(
    VDStitch,
    text="🎞 Generate Frames from Video",
    command=select_video_and_generate_frames,
    bg="green",
    fg="white",
    relief="flat"
)
generate_frames_btn.pack(pady=10)

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
    command=lambda: select_frames_folder(frames_folder),  # 👈 Pass it in
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
    command=lambda: select_output_file(output_file),  # 👈 Pass it
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

# ✅ Update GUI for FPS Selection & Interpolation Multiplier
tk.Label(VDStitch, text="🎞 Original FPS:", bg="#1c1c1c", fg="white").pack(anchor="w", padx=10, pady=5)
fps_menu = ttk.Combobox(VDStitch, textvariable=fps, values=COMMON_FPS, state="readonly")
fps_menu.pack(padx=10, pady=2)
fps_menu.current(0)

# ✅ FPS Interpolation Checkbox
fps_checkbox = tk.Checkbutton(
    VDStitch,
    text="Enable RIFE FPS Interpolation",
    variable=enable_fps_interpolation,
    bg="#1c1c1c",
    fg="white",
    selectcolor="#2b2b2b",
)
fps_checkbox.pack(anchor="w", padx=10, pady=5)

tk.Label(VDStitch, text="⏩ Interpolation Multiplier:", bg="#1c1c1c", fg="white").pack(anchor="w", padx=10, pady=5)
fps_mult_menu = ttk.Combobox(VDStitch, textvariable=fps_multiplier, values=MULTIPLIERS, state="readonly")
fps_mult_menu.pack(padx=10, pady=2)
fps_mult_menu.current(0)

# Processing Button with Dark Theme
process_btn = tk.Button(
    VDStitch,
    text="▶ Start Processing",
    command=lambda: start_processing(
        enable_fps_interpolation,
        frames_folder,
        output_file,
        width,
        height,
        fps,
        fps_multiplier,
        selected_codec,
        vdstitch_progress,      # ✅ progress bar
        vdstitch_status_label            # ✅ status label
    ),

    bg="green",
    fg="white",
    relief="flat",
)

process_btn.pack(pady=10)



# ✅ Define the progress bar before updating it
vdstitch_progress = ttk.Progressbar(
    VDStitch, orient="horizontal", length=300, mode="determinate"
)
vdstitch_progress.pack(pady=10)

vdstitch_status_label = tk.Label(
    VDStitch,
    text="Waiting to start...",
    bg="#1c1c1c",
    fg="white",
    font=("Segoe UI", 10),
)
vdstitch_status_label.pack()

# ---3D Generator Frame Contents ---

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

# ✅ Updated codec options: Standard + Lossless Codecs
codec_options = [
    # 🔹 Standard Codecs
    "mp4v",  # MPEG-4 (Good for MP4 format, widely supported)
    "XVID",  # XviD (Best for AVI format)
    "DIVX",  # DivX (Older AVI format)
]

# Layout frames
top_widgets_frame = tk.LabelFrame(
    visiondepth_content_frame, text="Video Info", padx=10, pady=10
)
top_widgets_frame.grid(row=0, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

# Thumbnail
video_thumbnail_label = tk.Label(
    top_widgets_frame, text="No Thumbnail", bg="white"
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
    visiondepth_content_frame,
    text="Select Input Video",
    command=lambda: select_input_video(input_video_path, video_thumbnail_label, video_specs_label)
).grid(row=3, column=0, pady=5, sticky="ew")

tk.Entry(visiondepth_content_frame, textvariable=input_video_path, width=50).grid(
    row=3, column=1, pady=5, padx=5
)

tk.Button(
    visiondepth_content_frame,
    text="Select Depth Map",
    command=lambda: select_depth_map(selected_depth_map, depth_map_label)
).grid(row=4, column=0, pady=5, sticky="ew")

tk.Entry(visiondepth_content_frame, textvariable=selected_depth_map, width=50).grid(
    row=4, column=1, pady=5, padx=5
)

tk.Button(
    visiondepth_content_frame,
    text="Select Output Video",
    command=lambda: select_output_video(output_sbs_video_path)
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
    "Red-Cyan Anaglyph",
)
option_menu.config(width=10)  # Adjust width to keep consistent look
option_menu.pack(side="left", padx=5)

# Buttons Inside button_frame to Keep Everything on One Line
start_button = tk.Button(
    button_frame,
    text="Generate 3D Video",
    bg="green",
    fg="white",
    command=lambda: process_video(
        input_video_path,
        selected_depth_map,
        output_sbs_video_path,
        selected_codec,
        fg_shift,
        mg_shift,
        bg_shift,
        sharpness_factor,
        blend_factor,
        delay_time,
        output_format,
        selected_aspect_ratio,   # 👈 pass this!
        aspect_ratios,           # 👈 and this!
        progress,
        progress_label,
        suspend_flag,
        cancel_flag,
    )



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

tk.Button(
    visiondepth_content_frame,
    text="🎵 Audio Tool",
    command=launch_audio_gui).grid(row=7, column=1, pady=10, padx=5, sticky="w")


# Ensure settings are saved when the program closes
def on_exit():
    stop_audio()           # 🔇 Kill ffplay audio process
    save_settings()        # 💾 Save settings
    root.destroy()         # ❌ Close GUI

root.protocol("WM_DELETE_WINDOW", on_exit)


root.mainloop()
