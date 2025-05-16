# ── Standard Library ─────────────────────────────
import os
import sys
import cv2
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from threading import Event
from core.audio import launch_audio_gui

# ── External Libraries ───────────────────────────
from PIL import Image, ImageTk
import torch.nn.functional as F
import numpy as np
import re
import webbrowser
import glob

# ── VisionDepth3D Custom Modules ────────────────
# 3D Rendering
from core.render_3d import (
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
from core.render_depth import (
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

from core.merged_pipeline import (
    start_merged_pipeline,
    select_video_and_generate_frames,
    select_output_file,
    select_frames_folder, 
    start_ffmpeg_writer,
)

from core.preview_gui import open_3d_preview_window
# At the top of GUI.py
cancel_requested = threading.Event()

process_thread = None 
suspend_flag = Event()
cancel_flag = Event()
SETTINGS_FILE = "settings.json"
translations = {}
current_language = "en"
tooltip_refs = {}

def set_language(lang_code):
    global current_language
    current_language = lang_code
    load_language(lang_code)
    refresh_ui_labels()
    save_settings()  # Persist the language selection

def load_language(lang_code):
    global translations
    try:
        path = f"languages/{lang_code}.json"
        with open(path, "r", encoding="utf-8") as f:
            translations = json.load(f)
            print(f"✅ Loaded '{lang_code}' with {len(translations)} keys from {path}")
    except Exception as e:
        print(f"⚠️ Failed to load language '{lang_code}': {e}")
        translations = {}

def t(key):
    return translations.get(key, key)

# ✅ Load default language before building GUI
load_language("en")

# ✅ Get absolute path to resource (for PyInstaller compatibility)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# ✅ Force include core/ into path
core_dir = resource_path("core")
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

    
# ✅ Inject DLL directory into PATH
def inject_dll_directory():
    dll_dir = resource_path("dlls")
    if os.path.isdir(dll_dir):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    else:
        print(f"[Warning] DLL folder not found: {dll_dir}")

# 🟢 Call this before any ONNX/TensorRT/CUDA init
inject_dll_directory()


def save_settings():
    settings = {name: var.get() for name, var in gui_variables.items()}

    if root.winfo_exists():
        settings["window_geometry"] = root.geometry()

    settings["language"] = current_language

    # ✅ Save input/depth video paths
    if "input_video_path" in globals() and hasattr(input_video_path, "get"):
        settings["input_video_path"] = input_video_path.get()
    if "selected_depth_map" in globals() and hasattr(selected_depth_map, "get"):
        settings["selected_depth_map"] = selected_depth_map.get()

    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)

    print("💾 Settings saved.")

    
def prompt_and_save_preset():
    file_path = filedialog.asksaveasfilename(
        title="Save Preset As",
        defaultextension=".json",
        filetypes=[("JSON Files", "*.json")],
        initialdir=PRESET_DIR,
        initialfile="custom_preset.json"
    )

    if file_path:
        preset_name = os.path.basename(file_path)
        save_current_preset(preset_name)


def reset_settings():
    """Resets all GUI values and UI elements to their default states."""

    # 🎬 File Paths and Codecs
    input_video_path.set("")
    selected_depth_map.set("")
    output_sbs_video_path.set("")
    selected_codec.set("mp4v")
    selected_ffmpeg_codec.set("H.264 / AVC (libx264)")
    output_format.set("Full-SBS")

    # 🧠 3D Shifting Parameters
    fg_shift.set(5.0)
    mg_shift.set(2.0)
    bg_shift.set(6.0)

    # ✨ Visual Enhancements
    sharpness_factor.set(1.0)
    blend_factor.set(0.0)
    delay_time.set(1 / 30)

    # 🧼 Edge Cleanup
    feather_strength.set(0.0)
    blur_ksize.set(1)

    # 🎛️ Advanced Stereo Controls
    parallax_balance.set(0.80)
    max_pixel_shift.set(0.20)
    dof_strength.set(2.0)

    # 🟢 Toggles
    use_subject_tracking.set(False)
    use_floating_window.set(False)
    auto_crop_black_bars.set(False)
    preserve_original_aspect.set(False)
    zero_parallax_strength.set(0.000)
    skip_blank_frames.set(False)
    convergence_strength.set(0.0),
    enable_dynamic_convergence.set(True),

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
    global cancel_flag, suspend_flag, cancel_requested, process_thread
    cancel_flag.set()
    cancel_requested.set()
    cancel_requested.clear()
    suspend_flag.clear()

    # 🔥 Reset the thread if it's no longer running
    if process_thread is not None and not process_thread.is_alive():
        print("🧼 Cleaning up finished thread...")
        process_thread = None

    print("❌ Processing canceled (all systems).")


def suspend_processing():
    global suspend_flag
    suspend_flag.set()
    print("⏸ Processing Suspended!")

def resume_processing():
    global suspend_flag
    suspend_flag.clear()
    print("▶ Processing Resumed!")

def handle_generate_3d():
    global process_thread
    try:
        if process_thread is None or not process_thread.is_alive():
            print("🚀 Starting new 3D processing thread...")
            cancel_flag.clear()
            suspend_flag.clear()
            process_thread = threading.Thread(target=lambda: process_video(
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
                zero_parallax_strength,
                enable_edge_masking,
                enable_feathering,
                skip_blank_frames,
                dof_strength,
                convergence_strength,
                enable_dynamic_convergence,
            ), daemon=True)
            process_thread.start()
        else:
            print("⚠️ 3D processing already running! Use Suspend/Resume/Cancel.")
    except Exception as e:
        print(f"❌ Error starting 3D processing: {e}")


def grab_frame_from_video(video_path, frame_idx=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


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

# ✅ Simple Tooltip Helper for Tkinter
class CreateToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tip_window or not self.text:
            return
        x = y = 0
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=4, ipady=2)

    def hide_tooltip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


# ---GUI Setup---

# -----------------------
# Global Variables & Setup
# -----------------------

# --- Window Setup ---
root = tk.Tk()
root.title("VisionDepth3D v3.1.9")
root.geometry("885x860")

# --- Menu Bar Setup ---
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# 🌐 Language Menu
language_menu = tk.Menu(menu_bar, tearoff=0)
lang_var = tk.StringVar(value="en")

# Add language options
for code, label in [("en", "English"), ("fr", "Français"), ("de", "German"), ("es", "Español"), ]:
    language_menu.add_command(
        label=label,
        command=lambda c=code: (
            load_language(c),
            refresh_ui_labels(),
            lang_var.set(c)
        )
    )

menu_bar.add_cascade(label="🌐 Language", menu=language_menu)

# Optional: File and Help menus
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Exit", command=root.quit)
menu_bar.add_cascade(label="File", menu=file_menu)

help_menu = tk.Menu(menu_bar, tearoff=0)
help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "VisionDepth3D v3.1.8"))
menu_bar.add_cascade(label="Help", menu=help_menu)


# --- Notebook for Tabs ---
tab_control = ttk.Notebook(root)
tab_control.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)

# --- Depth Estimation GUI ---
depth_estimation_frame = tk.Frame(tab_control)
tab_control.add(depth_estimation_frame, text="Depth Estimation")  # Initial text
depth_tab_index = tab_control.index("end") - 1  # Save the tab index

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
visiondepth_tab_index = tab_control.index("end") - 1

visiondepth_content_frame = tk.Frame(visiondepth_frame, highlightthickness=0, bd=0, bg="#1c1c1c")
visiondepth_content_frame.pack(fill="both", expand=True)

FrameTools3D = tk.Frame(tab_control, bg="#1c1c1c")
tab_control.add(FrameTools3D, text="FrameTools")
frametools_tab_index = tab_control.index("end") - 1

# --- Colors ---
bg_main = "#1e1e1e"
bg_controls = "#292929"
accent_color = "#4dd0e1"
fg_text = "white"

# --- Depth Content ---

local_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))
os.makedirs(local_model_dir, exist_ok=True)

def load_supported_models():
    models = {
        "  -- Select Model -- ": "  -- Select Model -- ",
        "Distill Any Depth Large": "model.onnx",
        "Distill Any Depth Base": "model.onnx",
        "Distill Any Depth Small": "model.onnx",
        "Distill-Any-Depth-Large": "xingyang1/Distill-Any-Depth-Large-hf",
        "Distill-Any-Depth-Small": "xingyang1/Distill-Any-Depth-Small-hf",
        "keetrap-Distil-Any-Depth-Large": "keetrap/Distil-Any-Depth-Large-hf",
        "keetrap-Distil-Any-Depth-Small": "keetrap/Distil-Any-Depth-Small-hf",
        "Video Depth Anything": "model.onnx",
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

    # ✅ Add local models from weights directory
    for folder_name in os.listdir(local_model_dir):
        folder_path = os.path.join(local_model_dir, folder_name)
        if os.path.isdir(folder_path):
            has_config = os.path.exists(os.path.join(folder_path, "config.json"))
            has_onnx = os.path.exists(os.path.join(folder_path, "model.onnx"))

            if has_config or has_onnx:
                display_name = f"[Local] {folder_name}"
                models[display_name] = folder_path


    return models

# Load models at start
supported_models = load_supported_models()

selected_model = tk.StringVar(root, value="-- Select Model --")
colormap_var = tk.StringVar(root, value="Default")
invert_var = tk.BooleanVar(root, value=False)
save_frames_var = tk.BooleanVar(value=False)
output_dir = tk.StringVar(value="")


selected_model_label = tk.Label(
    sidebar, text=t("Model"), bg="#1c1c1c",
    fg="white", font=("Arial", 11)
)
selected_model_label.pack(pady=5)


def refresh_model_dropdown():
    global supported_models
    supported_models = load_supported_models()  # Reload model list
    model_dropdown['values'] = list(supported_models.keys())

# Bind refresh on dropdown click (before dropdown opens)
model_dropdown = ttk.Combobox(
    sidebar,
    textvariable=selected_model,
    values=list(supported_models.keys()),
    state="readonly",
    width=22,
)
model_dropdown.pack(pady=5)

# Bind event that fires *before* selection
model_dropdown.bind("<Button-1>", lambda event: refresh_model_dropdown())

# Bind event for when user selects a model
model_dropdown.bind(
    "<<ComboboxSelected>>",
    lambda event: update_pipeline(selected_model, status_label)
)


output_dir_label = tk.Label(
    sidebar, text=t("Output Dir: None"), bg="#1c1c1c", fg="white", wraplength=200
)
output_dir_label.pack(pady=5)

output_dir_button = tk.Button(
    sidebar,
    text=t("Choose Directory"),
    command=lambda: choose_output_directory(output_dir_label, output_dir),
    width=20
)
output_dir_button.pack(pady=5)

colormap_label = tk.Label(
    sidebar, text=t("Colormap:"), bg="#1c1c1c", fg="white"
)
colormap_label.pack(pady=5)

colormap_dropdown = ttk.Combobox(
    sidebar,
    textvariable=colormap_var,
    values=["Default", "Magma", "Viridis", "Inferno", "Plasma", "Gray"],
    state="readonly",
    width=22,
)
colormap_dropdown.pack(pady=5)

invert_checkbox = tk.Checkbutton(
    sidebar, text=t("Invert Depth"), variable=invert_var, bg="#1c1c1c", fg="white", selectcolor="#2b2b2b"
)
invert_checkbox.pack(pady=5)

save_frames_checkbox = tk.Checkbutton(
    sidebar, text=t(" Save Frames"), variable=save_frames_var, bg="#1c1c1c", fg="white",  selectcolor="#2b2b2b"
)
save_frames_checkbox.pack(pady=5)

batch_size_label = tk.Label(
    sidebar, text=t("Batch Size (Frames):"),
    bg="#1c1c1c", fg="white"
)

batch_size_label.pack(pady=5)

batch_size_entry = tk.Entry(sidebar, width=22)
batch_size_entry.insert(0, "8")  # Default value
batch_size_entry.pack(pady=5)


# ✅ Add event listener to update batch size dynamically
def update_batch_size(*args):
    try:
        batch_size = int(batch_size_entry.get().strip())
        if batch_size <= 0:
            raise ValueError
        status_label.config(text=t(f"🔄 Batch Size Updated: {batch_size}"))
    except ValueError:
        status_label.config(text=t("⚠️ Invalid batch size. Using default (8)."))


batch_size_entry.bind("<Return>", update_batch_size)  # Update on "Enter" key press
batch_size_entry.bind("<FocusOut>", update_batch_size)  # Update when user clicks away

# Add a dropdown for inference resolution
inference_res_label = tk.Label(
    sidebar,
    text=t("Inference Resolution:"),
    bg="#1c1c1c",
    fg="white",
    font=("Arial", 11)
)
inference_res_label.pack(pady=5)

inference_res_var = tk.StringVar(value="Original")  # default value
inference_res_dropdown = tk.OptionMenu(
    sidebar,
    inference_res_var,
    "Original", "256x256", "384x384", "448x448", "512x512", "518x518", "912x912", "1024x1024", "920x1080"
)
inference_res_dropdown.config(bg="#2e2e2e", fg="white", highlightthickness=0, font=("Arial", 10))
inference_res_dropdown.pack(pady=5)


progress_bar = ttk.Progressbar(sidebar, mode="determinate", length=180)
progress_bar.pack(pady=10)
status_label = tk.Label(
    sidebar, text=t("Ready"), bg="#1c1c1c", fg="white", width=30, wraplength=200
)
status_label.pack(pady=5)

# --- Depth Content: Image previews ---
# --- Top Frame: For the original image ---
top_frame = tk.Frame(main_content, bg="#2b2b2b")
top_frame.pack(pady=10)

input_label = tk.Label(top_frame, text=t("Input Image"), bg="#2b2b2b", fg="white")
input_label.pack()  # No side=, so it stacks vertically

# --- Middle Frame: For the buttons ---
button_frame = tk.Frame(main_content, bg="#2b2b2b")
button_frame.pack(pady=10)

process_image_button = tk.Button(
    button_frame,
    text=t("Process Image"),
    command=lambda: open_image(
        status_label,
        progress_bar,
        colormap_var,
        invert_var,
        output_dir,
        inference_res_var,
        input_label,
        output_label
    ),
    width=25,
    bg="#4a4a4a",
    fg="white",
)
process_image_button.pack(pady=2)


process_image_folder_button = tk.Button(
    button_frame,
    text=t("Process Image Folder"),
    command=lambda: process_image_folder(
        batch_size_entry,
        output_dir,
        inference_res_var,
        status_label,
        progress_bar,
        root
    ),
    width=25,
    bg="#4a4a4a",
    fg="white",
)
process_image_folder_button.pack(pady=2)

process_video_button = tk.Button(
    button_frame,
    text=t("Process Video"),
    command=lambda: open_video(
        status_label,
        progress_bar,
        batch_size_entry,
        output_dir,
        inference_res_var,
        invert_var
    ),
    width=25,
    bg="#4a4a4a",
    fg="white",
)
process_video_button.pack(pady=2)

process_video_folder_button = tk.Button(
    button_frame,
    text=t("Process Video Folder"),
    command=lambda: process_videos_in_folder(
        filedialog.askdirectory(),  # folder_path from dialog
        batch_size_entry,
        output_dir,
        inference_res_var,
        status_label,
        progress_bar,
        cancel_requested,
        invert_var,
        save_frames_var.get()  # Optional, if used
    ),
    width=25,
    bg="#4a4a4a",
    fg="white",
)
process_video_folder_button.pack(pady=2)

# --- Bottom Frame: For the depth map ---
bottom_frame = tk.Frame(main_content, bg="#2b2b2b")
bottom_frame.pack(pady=10)

output_label = tk.Label(bottom_frame, text=t("Depth Map"), bg="#2b2b2b", fg="white")
output_label.pack()


# 🧠 Variables
ft3d_frames_folder = tk.StringVar()
ft3d_output_file = tk.StringVar()
ft3d_width = tk.IntVar(value=1920)
ft3d_height = tk.IntVar(value=804)
ft3d_fps = tk.DoubleVar(value=23.976)
ft3d_codec = tk.StringVar(value="AVC (NVENC GPU)")
ft3d_enable_rife = tk.BooleanVar(value=True)
ft3d_enable_upscale = tk.BooleanVar(value=False)
ft3d_fps_multiplier = tk.IntVar(value=2)
ft3d_blend_mode = tk.StringVar(value="OFF")
ft3d_input_res_pct = tk.IntVar(value=100)
ft3d_selected_model = tk.StringVar(value="VD-GAN")



REAL_ESRGAN_MODELS = {
    "RealESR_Gx4_fp16": "weights/RealESR_Gx4_fp16.onnx",
    "RealESRGAN_x4_fp16": "weights/RealESRGANx4_fp16.onnx",
    "RealESR_Animex4_fp16": "weights/RealESR_Animex4_fp16.onnx",
    "BSRGANx2_fp16": "weights/BSRGANx2_fp16.onnx",
    "BSRGANx4_fp16": "weights/BSRGANx4_fp16.onnx"
}


# 🎛️ Common settings
COMMON_FPS = [
    23.976,  # NTSC Film (24000/1001)
    24.0,    # Digital Cinema / Blu-ray
    25.0,    # PAL Standard
    29.97,   # NTSC Broadcast (30000/1001)
    30.0,    # Web video / mobile
    48.0,    # HFR Cinema (The Hobbit)
    50.0,    # PAL HFR (Broadcast, EU)
    59.94,   # NTSC HFR (60000/1001) — TVs & consoles
    60.0,    # High framerate (PC, streaming)
    72.0,    # Rare — some projectors / experimental VR
    90.0,    # VR headsets (Oculus Quest, HTC Vive)
    96.0,
    100.0,   # HFR mobile / 100Hz screens
    119.88,  # NTSC-style 120 (120000/1001)
    120.0,   # Ultra HFR (gaming, ProMotion)
    144.0,   # High-refresh PC monitors
    165.0,   # Gaming monitors
    240.0    # Max HFR (eSports displays, slow-mo)
]


FPS_MULTIPLIERS = [2, 4, 8]
FFMPEG_CODEC_MAP = {
    # Software (CPU) Encoders
    "H.264 / AVC (libx264 - CPU)": "libx264",
    "H.265 / HEVC (libx265 - CPU)": "libx265",
    "AV1 (libaom - CPU)": "libaom-av1",
    "AV1 (SVT - CPU, faster)": "libsvtav1",
    "MPEG-4 (mp4v - CPU)": "mp4v",
    "XviD (AVI - CPU)": "XVID",
    "DivX (AVI - CPU)": "DIVX",

    # NVIDIA NVENC
    "H.264 / AVC (NVENC - NVIDIA GPU)": "h264_nvenc",
    "H.265 / HEVC (NVENC - NVIDIA GPU)": "hevc_nvenc",
    "AV1 (NVENC - NVIDIA RTX 40+ GPU)": "av1_nvenc",

    # AMD AMF
    "H.264 / AVC (AMF - AMD GPU)": "h264_amf",
    "H.265 / HEVC (AMF - AMD GPU)": "hevc_amf",
    "AV1 (AMF - AMD RDNA3+)": "av1_amf",

    # Intel QSV
    "H.264 / AVC (QSV - Intel GPU)": "h264_qsv",
    "H.265 / HEVC (QSV - Intel GPU)": "hevc_qsv",
    "VP9 (QSV - Intel GPU)": "vp9_qsv",
    "AV1 (QSV - Intel ARC / Gen11+)": "av1_qsv",
}



# ─── FrameTools3D GUI Layout ──────────────────────────────────────────────

# Extract Button
extract_frames_button = tk.Button(
    FrameTools3D,
    text=t("Extract Frames from Video"),
    command=lambda: select_video_and_generate_frames(ft3d_frames_folder.set),
    bg="green",
    fg="white",
    relief="flat"
)
extract_frames_button.pack(pady=10)

# Input / Output Group
io_frame = tk.LabelFrame(FrameTools3D, text=t("Input / Output"), bg="#1c1c1c", fg="white")
io_frame.pack(fill="x", padx=10, pady=4)

frames_folder_label = tk.Label(
    io_frame, text=t("Frames Folder:"),
    bg="#1c1c1c", fg="white"
)
frames_folder_label.pack(anchor="w", padx=10, pady=(6, 2))

tk.Entry(io_frame, textvariable=ft3d_frames_folder, width=50, bg="#2b2b2b", fg="white", insertbackground="white").pack(pady=2)
browse_button = tk.Button(
    io_frame, text=t("Browse"),
    command=lambda: select_frames_folder(ft3d_frames_folder),
    bg="#4a4a4a", fg="white"
)
browse_button.pack(pady=2)

output_video_file_label = tk.Label(
    io_frame, text=t("Output Video File:"),
    bg="#1c1c1c", fg="white"
)
output_video_file_label.pack(anchor="w", padx=10, pady=(6, 2))

tk.Entry(io_frame, textvariable=ft3d_output_file, width=50, bg="#2b2b2b", fg="white", insertbackground="white").pack(pady=2)

save_as_button = tk.Button(
    io_frame, text=t("Save As"), command=lambda: select_output_file(ft3d_output_file),
    bg="#4a4a4a", fg="white"
)
save_as_button.pack(pady=2)

# Processing Options
proc_frame = tk.LabelFrame(FrameTools3D, text=t("⚙️ Processing Options"), bg="#1c1c1c", fg="white")
proc_frame.pack(fill="x", padx=10, pady=4)

RIFE_FPS_button = tk.Checkbutton(
    proc_frame, text=t("Enable RIFE Interpolation"),
    variable=ft3d_enable_rife, bg="#1c1c1c", fg="white",
    selectcolor="#2b2b2b"
)
RIFE_FPS_button.pack(anchor="w", padx=10, pady=2)

esrgan_button = tk.Checkbutton(
    proc_frame, text=t("Enable Real-ESRGAN Upscale"),
    variable=ft3d_enable_upscale, bg="#1c1c1c", fg="white",
    selectcolor="#2b2b2b"
)
esrgan_button.pack(anchor="w", padx=10, pady=2)

# Output Settings
out_frame = tk.LabelFrame(FrameTools3D, text=t("Output Settings"), bg="#1c1c1c", fg="white")
out_frame.pack(fill="x", padx=10, pady=4)

res_box = tk.Frame(out_frame, bg="#1c1c1c")
res_box.pack(anchor="w", padx=10, pady=4)

resolution_label = tk.Label(res_box, text=t("Resolution (WxH):"), bg="#1c1c1c", fg="white")
resolution_label.pack(side="left")

tk.Entry(res_box, textvariable=ft3d_width, width=6, bg="#2b2b2b", fg="white", insertbackground="white").pack(side="left", padx=4)
tk.Label(res_box, text="x", bg="#1c1c1c", fg="white").pack(side="left")
tk.Entry(res_box, textvariable=ft3d_height, width=6, bg="#2b2b2b", fg="white", insertbackground="white").pack(side="left", padx=4)

# Helper function that returns label for tooltip use
def combo_row(parent, label_text, var, values):
    row = tk.Frame(parent, bg="#1c1c1c")
    row.pack(anchor="w", padx=10, pady=4, fill="x")
    label = tk.Label(row, text=label_text, bg="#1c1c1c", fg="white", width=22, anchor="w")
    label.pack(side="left")
    ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=20).pack(side="left")
    return label

# Output Setting Rows
original_fps_label = combo_row(out_frame, t("Original FPS:"), ft3d_fps, COMMON_FPS)
fps_multi_label = combo_row(out_frame, t("FPS Interpolation Multiplier:"), ft3d_fps_multiplier, FPS_MULTIPLIERS)
selected_ffmpeg_codec_frametools_label = combo_row(out_frame, t("FFmpeg Output Codec:"), ft3d_codec, list(FFMPEG_CODEC_MAP.keys()))

# ESRGAN Settings
esrgan_frame = tk.LabelFrame(FrameTools3D, text=t("ESRGAN Settings"), bg="#1c1c1c", fg="white")
esrgan_frame.pack(fill="x", padx=10, pady=4)

ai_blend_select = combo_row(esrgan_frame, t("AI Blending:"), ft3d_blend_mode, ["OFF", "LOW", "MEDIUM", "HIGH"])
input_res_pct_label = combo_row(esrgan_frame, t("Input Resolution %:"), ft3d_input_res_pct, [25, 50, 75, 100])
model_select = combo_row(esrgan_frame, t("Model Selection:"), ft3d_selected_model, list(REAL_ESRGAN_MODELS.keys()))


# ▶️ Start Button
start_processing_button = tk.Button(
    FrameTools3D,
    text=t("▶ Start Processing"),
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
                "codec": FFMPEG_CODEC_MAP.get(ft3d_codec.get(), "h264_nvenc"),
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
)
start_processing_button.pack(pady=12)

# 📊 Progress Bar
merged_progress = ttk.Progressbar(FrameTools3D, orient="horizontal", length=300, mode="determinate")
merged_progress.pack(pady=6)

merged_status = tk.Label(FrameTools3D, text=t("Waiting to start..."), bg="#1c1c1c", fg="white")
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
fg_shift = tk.DoubleVar(value=4.5)
mg_shift = tk.DoubleVar(value=-1.5)
bg_shift = tk.DoubleVar(value=-6.0)
sharpness_factor = tk.DoubleVar(value=0.2)
blend_factor = tk.DoubleVar(value=0.6)
delay_time = tk.DoubleVar(value=1 / 30)
output_format = tk.StringVar(value="Full-SBS")
blur_ksize = tk.IntVar(value=1)
feather_strength = tk.DoubleVar(value=0.0)
selected_ffmpeg_codec = tk.StringVar(value="H.264 / AVC (libx264 - CPU)")
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
zero_parallax_strength = tk.DoubleVar(value=0.01)
enable_edge_masking = tk.BooleanVar(value=True)
enable_feathering = tk.BooleanVar(value=True)
skip_blank_frames = tk.BooleanVar()
dof_strength = tk.DoubleVar(value=2.0)  # Default strength (sigma)
enable_dynamic_convergence = tk.BooleanVar(value=True)
convergence_strength = tk.DoubleVar(value=0.0)




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
    # Software (CPU) Encoders
    "H.264 / AVC (libx264 - CPU)": "libx264",
    "H.265 / HEVC (libx265 - CPU)": "libx265",
    "AV1 (libaom - CPU)": "libaom-av1",
    "AV1 (SVT - CPU, faster)": "libsvtav1",
    "MPEG-4 (mp4v - CPU)": "mp4v",
    "XviD (AVI - CPU)": "XVID",
    "DivX (AVI - CPU)": "DIVX",

    # NVIDIA NVENC
    "H.264 / AVC (NVENC - NVIDIA GPU)": "h264_nvenc",
    "H.265 / HEVC (NVENC - NVIDIA GPU)": "hevc_nvenc",
    "AV1 (NVENC - NVIDIA RTX 40+ GPU)": "av1_nvenc",

    # AMD AMF
    "H.264 / AVC (AMF - AMD GPU)": "h264_amf",
    "H.265 / HEVC (AMF - AMD GPU)": "hevc_amf",
    "AV1 (AMF - AMD RDNA3+)": "av1_amf",

    # Intel QSV
    "H.264 / AVC (QSV - Intel GPU)": "h264_qsv",
    "H.265 / HEVC (QSV - Intel GPU)": "hevc_qsv",
    "VP9 (QSV - Intel GPU)": "vp9_qsv",
    "AV1 (QSV - Intel ARC / Gen11+)": "av1_qsv",
}


# 🧠 Master list of all variables that should be saved
gui_variables = {
    "input_video_path": input_video_path,
    "selected_depth_map": selected_depth_map,
    "output_sbs_video_path": output_sbs_video_path,
    "selected_codec": selected_codec,
    "selected_ffmpeg_codec": selected_ffmpeg_codec,
    "use_ffmpeg": use_ffmpeg,
    "crf_value": crf_value,
    "nvenc_cq_value": nvenc_cq_value,
    "output_format": output_format,
    "preview_mode": preview_mode,
    "frame_to_preview_var": frame_to_preview_var,
    "fg_shift": fg_shift,
    "mg_shift": mg_shift,
    "bg_shift": bg_shift,
    "sharpness_factor": sharpness_factor,
    "blend_factor": blend_factor,
    "delay_time": delay_time,
    "blur_ksize": blur_ksize,
    "feather_strength": feather_strength,
    "parallax_balance": parallax_balance,
    "max_pixel_shift": max_pixel_shift,
    "use_subject_tracking": use_subject_tracking,
    "use_floating_window": use_floating_window,
    "auto_crop_black_bars": auto_crop_black_bars,
    "preserve_original_aspect": preserve_original_aspect,
    "zero_parallax_strength": zero_parallax_strength,
    "enable_edge_masking": enable_edge_masking,
    "enable_feathering": enable_feathering,
    "skip_blank_frames": skip_blank_frames,
    "selected_aspect_ratio": selected_aspect_ratio,
    "original_video_width": original_video_width,
    "original_video_height": original_video_height,
    "preserve_content": preserve_content,
    "dof_strength": dof_strength,
    "convergence_strength": convergence_strength,
    "enable_dynamic_convergence": enable_dynamic_convergence,

}


# Layout frames

top_widgets_frame = tk.LabelFrame(
    visiondepth_content_frame,
    text=t("Video Info"),
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
    top_widgets_frame, text=t("No Thumbnail"), bg="#1c1c1c", fg="white"
)
video_thumbnail_label.grid(row=0, column=0, padx=10, pady=5)

video_specs_label = tk.Label(
    top_widgets_frame, text=t("Resolution: N/A\nFPS: N/A"), justify="left", bg="#1c1c1c", fg="white"
)
video_specs_label.grid(row=0, column=1, padx=10, pady=5)

aspect_preview_label = tk.Label(top_widgets_frame, text="", font=("Segoe UI", 8, "italic"), bg="#1c1c1c", fg="white")
aspect_preview_label.grid(row=1, column=0, sticky="w", padx=5)

# 🔁 Bind aspect ratio dropdown to preview label
selected_aspect_ratio.trace_add("write", update_aspect_preview)
update_aspect_preview()

depth_map_label = tk.Label(
    top_widgets_frame, text=t("Depth Map (3D): None"), bg="#1c1c1c", fg="white", justify="left", wraplength=200
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
    text=t("Processing Options"),
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
preserve_aspect_checkbox = tk.Checkbutton(
    options_frame,
    text=t("Preserve Original Aspect Ratio"), bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=preserve_original_aspect
)
preserve_aspect_checkbox.grid(row=0, column=0, sticky="w", padx=5)

auto_crop_checkbox = tk.Checkbutton(
    options_frame,
    text=t("Auto Crop Black Bars"), bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=auto_crop_black_bars,
    anchor="e",
    justify="left"
)
auto_crop_checkbox.grid(row=0, column=1, sticky="w", padx=5)

use_subject_tracking_checkbox = tk.Checkbutton(
    options_frame,
    text=t("Stabilize Zero-Parallax (center-depth)"), bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=use_subject_tracking,
    anchor="w",
    justify="left"
)
use_subject_tracking_checkbox.grid(row=0, column=2, sticky="w", padx=5)

use_dfw_checkbox = tk.Checkbutton(
    options_frame,
    text=t("Enable Floating Window (DFW)"), bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=use_floating_window,
    anchor="e",
    justify="left"
)
use_dfw_checkbox.grid(row=0, column=3, sticky="e", padx=5)

# Row 1
use_ffmpeg_checkbox = tk.Checkbutton(
    options_frame,
    text=t("Use FFmpeg Renderer"), bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=use_ffmpeg,
    anchor="w",
    justify="left"
)
use_ffmpeg_checkbox.grid(row=1, column=0, sticky="w", padx=5)

enable_edge_checkbox = tk.Checkbutton(
    options_frame,
    text=t("Enable Edge Masking"), bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=enable_edge_masking,
    anchor="w",
    justify="left"
)
enable_edge_checkbox.grid(row=1, column=1, sticky="w", padx=5)

enable_feathering_checkbox = tk.Checkbutton(
    options_frame,
    text=t("Enable Feathering"), bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=enable_feathering,
    anchor="w",
    justify="left"
)
enable_feathering_checkbox.grid(row=1, column=2, sticky="w", padx=5)


skip_blank_frames_checkbox = tk.Checkbutton(
    options_frame,
    text=t("Skip Blank/White Frames"), bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=skip_blank_frames,
    anchor="w",
    justify="left"
)
skip_blank_frames_checkbox.grid(row=1, column=3, sticky="w", padx=5)

enable_dynamic_convergence_checkbox = tk.Checkbutton(
    options_frame,
    text=t("Enable Dynamic Convergence"), bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=enable_dynamic_convergence,
    anchor="w",
    justify="left"
)
enable_dynamic_convergence_checkbox.grid(row=2, column=2, sticky="w", padx=5)



# Row 2
fg_shift_label = tk.Label(
    options_frame,
    text=t("Foreground Shift"),
    bg="#1c1c1c",
    fg="white"
)
fg_shift_label.grid(row=2, column=0, sticky="w")

tk.Scale(
    options_frame,
    from_=0,
    to=30,
    resolution=0.5,
    orient=tk.HORIZONTAL,
    variable=fg_shift,
    bg="#1c1c1c", fg="white"
).grid(row=2, column=1, sticky="ew")


# Row 3

mg_shift_label = tk.Label(
    options_frame,
    text=t("Midground Shift"),
    bg="#1c1c1c",
    fg="white"
)
mg_shift_label.grid(row=3, column=0, sticky="w")

tk.Scale(
    options_frame, 
    from_=-10, to=10,
    resolution=0.5,
    orient=tk.HORIZONTAL, variable=mg_shift,
    bg="#1c1c1c", fg="white"
).grid(row=3, column=1, sticky="ew")

convergence_strength_label = tk.Label(
    options_frame,
    text=t("Convergence Strength"),
    bg="#1c1c1c", fg="white"
)
convergence_strength_label.grid(row=3, column=2, sticky="w")

tk.Scale(
    options_frame, from_=-0.05, to=0.05, resolution=0.001, orient=tk.HORIZONTAL,
    variable=convergence_strength, length=200, bg="#1c1c1c", fg="white"
).grid(row=3, column=3, sticky="ew")


sharpness_factor_label = tk.Label(
    options_frame,
    text=t("Sharpness Factor"),
    bg="#1c1c1c", fg="white"
)
sharpness_factor_label.grid(row=4, column=2, sticky="w")

tk.Scale(
    options_frame,
    from_=-1, to=1,
    resolution=0.1, orient=tk.HORIZONTAL, 
    variable=sharpness_factor, bg="#1c1c1c", fg="white"
).grid(row=4, column=3, sticky="ew")

#Row 4
bg_shift_label = tk.Label(
    options_frame,
    text=t("Background Shift"),
    bg="#1c1c1c", fg="white"
)
bg_shift_label.grid(row=4, column=0, sticky="w")

tk.Scale(
    options_frame,
    from_=-20, to=0, 
    resolution=0.5, orient=tk.HORIZONTAL,
    variable=bg_shift, bg="#1c1c1c", fg="white"
).grid(row=4, column=1, sticky="ew")
 
parallax_balance_label = tk.Label(
    options_frame,
    text=t("Parallax Balance"),
    bg="#1c1c1c", fg="white"
)
parallax_balance_label.grid(row=5, column=2, sticky="w")

tk.Scale(
    options_frame,
    from_=0.0,
    to=1.0,
    resolution=0.05,
    orient="horizontal",
    variable=parallax_balance,
    bg="#1c1c1c", fg="white"
).grid(row=5, column=3, sticky="ew")

#Row 5
zero_parallax_strength_label = tk.Label(
    options_frame,
    text=t("Zero Parallax Strength"),
    bg="#1c1c1c", fg="white"
)
zero_parallax_strength_label.grid(row=5, column=0, sticky="w")

tk.Scale(
    options_frame, from_=-0.05, to=0.05, resolution=0.001, orient=tk.HORIZONTAL,
    variable=zero_parallax_strength, length=200, bg="#1c1c1c", fg="white"
).grid(row=5, column=1, sticky="ew")

max_pixel_shift_label = tk.Label(
    options_frame,
    text=t("Max Pixel Shift (%)"),
    bg="#1c1c1c", fg="white"
)
max_pixel_shift_label.grid(row=6, column=0, sticky="w")

tk.Scale(
    options_frame,
    from_=0.005, to=0.10,
    resolution=0.005,
    orient=tk.HORIZONTAL,
    variable=max_pixel_shift,
    length=200, bg="#1c1c1c", fg="white"
).grid(row=6, column=1, sticky="ew")   

dof_strength_label = tk.Label(
    options_frame,
    text=t("DoF Strength"),
    bg="#1c1c1c", fg="white"
)
dof_strength_label.grid(row=6, column=2, sticky="w")

tk.Scale(
    options_frame,
    from_=0.0, to=5.0,
    resolution=0.1,
    orient=tk.HORIZONTAL,
    variable=dof_strength,
    length=200,
    bg="#1c1c1c", fg="white"
).grid(row=6, column=3, sticky="ew")


# File Selection
select_input_video_button = tk.Button(
    visiondepth_content_frame,
    text=t("Select Input Video"),
    bg="#2c2c2c", fg="white",
    activebackground="#444444", activeforeground="white",
    relief="groove", bd=2,
    command=lambda: select_input_video(
        input_video_path,
        video_thumbnail_label,
        video_specs_label,
        update_aspect_preview,
        original_video_width,
        original_video_height
    )
)
select_input_video_button.grid(row=3, column=0, pady=5, sticky="ew")

tk.Entry(
    visiondepth_content_frame,
    textvariable=input_video_path,
    width=50,
    bg="#2c2c2c", fg="white",
    insertbackground="white",
    relief="groove", bd=2
).grid(row=3, column=1, pady=5, padx=5)

select_depth_map_button = tk.Button(
    visiondepth_content_frame,
    text=t("Select Depth Map"),
    bg="#2c2c2c", fg="white",
    activebackground="#444444", activeforeground="white",
    relief="groove", bd=2,
    command=lambda: select_depth_map(selected_depth_map, depth_map_label)
)
select_depth_map_button.grid(row=4, column=0, pady=5, sticky="ew")

tk.Entry(
    visiondepth_content_frame,
    textvariable=selected_depth_map,
    width=50,
    bg="#2c2c2c", fg="white",
    insertbackground="white",
    relief="groove", bd=2
).grid(row=4, column=1, pady=5, padx=5)

select_output_video_button = tk.Button(
    visiondepth_content_frame,
    text=t("Select Output Video"),
    bg="#2c2c2c", fg="white",
    activebackground="#444444", activeforeground="white",
    relief="groove", bd=2,
    command=lambda: select_output_video(output_sbs_video_path)
)
select_output_video_button.grid(row=5, column=0, pady=5, sticky="ew")

tk.Entry(
    visiondepth_content_frame,
    textvariable=output_sbs_video_path,
    width=50,
    bg="#2c2c2c", fg="white",
    insertbackground="white",
    relief="groove", bd=2
).grid(row=5, column=1, pady=5, padx=5)


# Frame to Hold Buttons and Format Selection in a Single Row
button_frame = tk.Frame(visiondepth_content_frame, bg="#1c1c1c")
button_frame.grid(row=7, column=0, columnspan=5, pady=10, sticky="w")

# 3D Format Label and Dropdown (Inside button_frame)
format_button = tk.Label(
    button_frame, text=t("3D Format"),
    bg="#1c1c1c", fg="white"
)
format_button.pack(side="left", padx=5)

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
    text=t("Generate 3D Video"),
    bg="green",
    fg="white",
    command=lambda: (
        save_settings(),
        handle_generate_3d()
    )
)

start_button.pack(side="left", padx=5)

preview_button = tk.Button(
    button_frame,
    text=t("Open Preview"),
    command=lambda: open_3d_preview_window(
        input_video_path,
        selected_depth_map,
        fg_shift,
        mg_shift,
        bg_shift,
        blur_ksize,
        feather_strength,
        use_subject_tracking,
        use_floating_window,
        zero_parallax_strength,
        parallax_balance,
        enable_edge_masking,
        enable_feathering,
        sharpness_factor,
        max_pixel_shift,
        dof_strength,
        convergence_strength,
        enable_dynamic_convergence,

    )

)

preview_button.pack(side="left", padx=5)

suspend_button = tk.Button(
    button_frame, text=t("Suspend"), command=suspend_processing, bg="orange", fg="black"
)
suspend_button.pack(side="left", padx=5)

resume_button = tk.Button(
    button_frame, text=t("Resume"), command=resume_processing, bg="blue", fg="white"
)
resume_button.pack(side="left", padx=5)

cancel_button = tk.Button(
    button_frame, text=t("Cancel"), command=cancel_processing, bg="red", fg="white"
)
cancel_button.pack(side="left", padx=5)

# Row 7 - Reset button centered
reset_button = tk.Button(
    button_frame, text=t("Reset to Defaults"), command=reset_settings, bg="#8B0000", fg="white"
)
reset_button.pack(side="left", padx=5)

save_preset_button = tk.Button(
    button_frame, text=t("Save Preset"), bg="#1c1c1c", fg="white",
    command=prompt_and_save_preset
)
save_preset_button.pack(side="left", padx=5)


# 🔲 Encoding Settings Group
encoding_frame = tk.LabelFrame(
    visiondepth_content_frame,
    text=t("Encoding Settings"),
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
selected_aspect_ratio_label = tk.Label(
    encoding_frame,
    text=t("Aspect Ratio:"),
    bg="#1c1c1c",
    fg="white"
)
selected_aspect_ratio_label.grid(row=0, column=0, sticky="w", padx=5)

tk.OptionMenu(
    encoding_frame,
    selected_aspect_ratio,
    *aspect_ratios.keys()
).grid(row=0, column=1, sticky="ew", padx=5)

# 🧰 FFmpeg Codec
selected_ffmpeg_codec_label = tk.Label(
    encoding_frame,
    text=t("FFmpeg Codec:"),
    bg="#1c1c1c", fg="white"
)
selected_ffmpeg_codec_label.grid(row=0, column=2, sticky="w", padx=5)

tk.OptionMenu(
    encoding_frame,
    selected_ffmpeg_codec,
    *FFMPEG_CODEC_MAP.keys()
).grid(row=0, column=3, sticky="ew", padx=5)

# 🎞️ Codec
selected_codec_label = tk.Label(
    encoding_frame,
    text=t("Codec:"),
    bg="#1c1c1c",
    fg="white"
)
selected_codec_label.grid(row=0, column=4, sticky="w", padx=5)

tk.OptionMenu(
    encoding_frame,
    selected_codec,
    *codec_options
).grid(row=0, column=5, sticky="ew", padx=5)

# 📉 CRF
crf_value_label = tk.Label(
    encoding_frame,
    text=t("CRF"),
    bg="#1c1c1c",
    fg="white"
)
crf_value_label.grid(row=1, column=0, sticky="w", padx=5)

tk.Scale(
    encoding_frame,
    from_=0,
    to=51,
    resolution=1,
    orient=tk.HORIZONTAL,
    variable=crf_value, 
    length=150,
    bg="#2b2b2b",
    fg="white", 
    troughcolor="#444"
).grid(row=1, column=1, columnspan=2, sticky="ew", padx=5)

# 🚀 NVENC CQ
nvenc_cq_value_label = tk.Label(
    encoding_frame,
    text=t("NVENC CQ"),
    bg="#1c1c1c",
    fg="white"
)
nvenc_cq_value_label.grid(row=1, column=3, sticky="w", padx=5)

tk.Scale(
    encoding_frame,
    from_=0,
    to=51,
    resolution=1,
    orient=tk.HORIZONTAL,
    variable=nvenc_cq_value,
    length=150,
    bg="#2b2b2b",
    fg="white",
    troughcolor="#444"
).grid(row=1, column=4, columnspan=2, sticky="ew", padx=5)


# Row 9 – Icon Buttons + Audio Tool
def open_github():
    """Opens the GitHub repository in a web browser."""
    webbrowser.open_new("https://github.com/VisionDepth/VisionDepth3D")

def open_aspect_ratio_CheatSheet():
    """Opens the Aspect Ratio Cheat Sheet."""
    webbrowser.open_new("https://www.wearethefirehouse.com/aspect-ratio-cheat-sheet")

# Load GitHub icon
github_icon_path = resource_path(os.path.join("assets", "github.png"))
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
    text=t("🎵 Audio Tool"), bg="#1c1c1c", fg="white",
    command=launch_audio_gui
)
audio_tool_button.pack(side="left", padx=10)

preset_var = tk.StringVar()
preset_menu = ttk.Combobox(bottom_links_frame, textvariable=preset_var)
preset_menu['values'] = ["Balanced Depth", "IMAX Depth", "Pop-Out 3D"]
preset_menu.set("Select Preset")
preset_menu.bind("<<ComboboxSelected>>", lambda e: apply_preset(preset_var.get()))
preset_menu.pack(side="left", padx=10)

# -- Depth Estimation Tab --
tooltip_refs["Model"] = CreateToolTip(model_dropdown, t("Tooltip.Model"))
tooltip_refs["OutputDirLabel"] = CreateToolTip(output_dir_label, t("Tooltip.OutputDirLabel"))
tooltip_refs["OutputDirButton"] = CreateToolTip(output_dir_button, t("Tooltip.OutputDirButton"))
tooltip_refs["ColormapLabel"] = CreateToolTip(colormap_label, t("Tooltip.ColormapLabel"))
tooltip_refs["ColormapDropdown"] = CreateToolTip(colormap_dropdown, t("Tooltip.ColormapDropdown"))
tooltip_refs["InvertCheckbox"] = CreateToolTip(invert_checkbox, t("Tooltip.InvertCheckbox"))
tooltip_refs["SaveFramesCheckbox"] = CreateToolTip(save_frames_checkbox, t("Tooltip.SaveFramesCheckbox"))
tooltip_refs["BatchSizeEntry"] = CreateToolTip(batch_size_entry, t("Tooltip.BatchSizeEntry"))
tooltip_refs["InputLabel"] = CreateToolTip(input_label, t("Tooltip.InputLabel"))
tooltip_refs["DepthLabel"] = CreateToolTip(output_label, t("Tooltip.DepthLabel"))
tooltip_refs["ProcessImage"] = CreateToolTip(process_image_button, t("Tooltip.ProcessImage"))
tooltip_refs["ProcessImageFolder"] = CreateToolTip(process_image_folder_button, t("Tooltip.ProcessImageFolder"))
tooltip_refs["ProcessVideo"] = CreateToolTip(process_video_button, t("Tooltip.ProcessVideo"))
tooltip_refs["ProcessVideoFolder"] = CreateToolTip(process_video_folder_button, t("Tooltip.ProcessVideoFolder"))


# -- 3D Render Tab --
tooltip_refs["StartButton"] = CreateToolTip(start_button, t("Tooltip.StartButton"))
tooltip_refs["PreviewButton"] = CreateToolTip(preview_button, t("Tooltip.PreviewButton"))
tooltip_refs["suspend_button"] = CreateToolTip(suspend_button, t("Tooltip.SuspendButton"))
tooltip_refs["SuspendButton"] = CreateToolTip(resume_button, t("Tooltip.ResumeButton"))
tooltip_refs["ResumeButton"] = CreateToolTip(cancel_button, t("Tooltip.CancelButton"))
tooltip_refs["ResetButton"] = CreateToolTip(reset_button, t("Tooltip.ResetButton"))

tooltip_refs["OptionMenu"] = CreateToolTip(option_menu, t("Tooltip.OptionMenu"))
tooltip_refs["AspectPreview"] = CreateToolTip(aspect_preview_label, t("Tooltip.AspectPreview"))

# Sliders
tooltip_refs["FGShift"] = CreateToolTip(fg_shift_label, t("Tooltip.FGShift"))
tooltip_refs["mg_shift_label"] = CreateToolTip(mg_shift_label, t("Tooltip.MGShift"))
tooltip_refs["MGShift"] = CreateToolTip(bg_shift_label, t("Tooltip.BGShift"))
tooltip_refs["Sharpness"] = CreateToolTip(sharpness_factor_label, t("Tooltip.Sharpness"))
tooltip_refs["ZeroParallaxStrength"] = CreateToolTip(zero_parallax_strength_label, t("Tooltip.ZeroParallaxStrength"))
tooltip_refs["ParallaxBalance"] = CreateToolTip(parallax_balance_label, t("Tooltip.ParallaxBalance"))
tooltip_refs["MaxPixelShift"] = CreateToolTip(max_pixel_shift_label, t("Tooltip.MaxPixelShift"))
tooltip_refs["DOFStrength"] = CreateToolTip(dof_strength_label, t("Tooltip.DOFStrength"))
tooltip_refs["ConvergenceStrength"] = CreateToolTip(convergence_strength_label, t("Tooltip.ConvergenceStrength"))

# Checkboxes
tooltip_refs["PreserveAspect"] = CreateToolTip(preserve_aspect_checkbox, t("Tooltip.PreserveAspect"))
tooltip_refs["AutoCrop"] = CreateToolTip(auto_crop_checkbox, t("Tooltip.AutoCrop"))
tooltip_refs["SubjectTracking"] = CreateToolTip(use_subject_tracking_checkbox, t("Tooltip.SubjectTracking"))
tooltip_refs["FloatingWindow"] = CreateToolTip(use_dfw_checkbox, t("Tooltip.FloatingWindow"))
tooltip_refs["EdgeMasking"] = CreateToolTip(enable_edge_checkbox, t("Tooltip.EdgeMasking"))
tooltip_refs["Feathering"] = CreateToolTip(enable_feathering_checkbox, t("Tooltip.Feathering"))
tooltip_refs["SkipBlankFrames"] = CreateToolTip(skip_blank_frames_checkbox, t("Tooltip.SkipBlankFrames"))
tooltip_refs["UseFFmpeg"] = CreateToolTip(use_ffmpeg_checkbox, t("Tooltip.UseFFmpeg"))
tooltip_refs["EnableDynConvergence"] = CreateToolTip(enable_dynamic_convergence_checkbox, t("Tooltip.EnableDynConvergence"))


# Encoding
tooltip_refs["CRF"] = CreateToolTip(crf_value_label, t("Tooltip.CRF"))
tooltip_refs["NVENCCQ"] = CreateToolTip(nvenc_cq_value_label, t("Tooltip.NVENCCQ"))
tooltip_refs["SelectedCodec"] = CreateToolTip(selected_codec_label, t("Tooltip.SelectedCodec"))
tooltip_refs["FFmpegCodec"] = CreateToolTip(selected_ffmpeg_codec_label, t("Tooltip.FFmpegCodec"))
tooltip_refs["AspectRatio"] = CreateToolTip(selected_aspect_ratio_label, t("Tooltip.AspectRatio"))

# -- FrameTool Tips --
tooltip_refs["ExtractFrames"] = CreateToolTip(extract_frames_button, t("Tooltip.ExtractFrames"))
tooltip_refs["RIFE"] = CreateToolTip(RIFE_FPS_button, t("Tooltip.RIFE"))
tooltip_refs["ESRGAN"] = CreateToolTip(esrgan_button, t("Tooltip.ESRGAN"))
tooltip_refs["Resolution"] = CreateToolTip(resolution_label, t("Tooltip.Resolution"))
tooltip_refs["OriginalFPS"] = CreateToolTip(original_fps_label, t("Tooltip.OriginalFPS"))
tooltip_refs["FPSMultiplier"] = CreateToolTip(fps_multi_label, t("Tooltip.FPSMultiplier"))
tooltip_refs["AIBlend"] = CreateToolTip(ai_blend_select, t("Tooltip.AIBlend"))
tooltip_refs["InputResPct"] = CreateToolTip(input_res_pct_label, t("Tooltip.InputResPct"))
tooltip_refs["ModelSelect"] = CreateToolTip(model_select, t("Tooltip.ModelSelect"))

PRESET_DIR = "presets"
os.makedirs(PRESET_DIR, exist_ok=True)

def refresh_ui_labels():
    tab_control.tab(depth_tab_index, text=t("Depth Estimation"))
    tab_control.tab(visiondepth_tab_index, text=t("3D Video Generator"))
    tab_control.tab(frametools_tab_index, text=t("FrameTools"))
    selected_model_label.config(text=t("Model"))
    output_dir_label.config(text=t("Output Dir: None"))
    output_dir_button.config(text=t("Choose Directory"))
    colormap_label.config(text=t("Colormap:"))
    invert_checkbox.config(text=t("Invert Depth"))
    save_frames_checkbox.config(text=t("Save Frames"))
    batch_size_label.config(text=t("Batch Size (Frames):"))
    inference_res_label.config(text=t("Inference Resolution:"))
    status_label.config(text=t("Ready"))
    input_label.config(text=t("Input Image"))
    output_label.config(text=t("Depth Map"))
    process_image_button.config(text=t("Process Image"))
    process_image_folder_button.config(text=t("Process Image Folder"))
    process_video_button.config(text=t("Process Video"))
    process_video_folder_button.config(text=t("Process Video Folder"))

    # 3D Render Tab Labels
    video_thumbnail_label.config(text=t("No Thumbnail"))
    video_specs_label.config(text=t("Resolution: N/A\nFPS: N/A"))
    depth_map_label.config(text=t("Depth Map (3D): None"))
    audio_tool_button.config(text=t("🎵 Audio Tool"))
    select_input_video_button.config(text=t("Select Input Video"))
    select_depth_map_button.config(text=t("Select Depth Map"))
    select_output_video_button.config(text=t("Select Output Video"))
    format_button.config(text=t("3D Format"))
    start_button.config(text=t("Generate 3D Video"))
    preview_button.config(text=t("Open Preview"))
    suspend_button.config(text=t("Suspend"))
    resume_button.config(text=t("Resume"))
    cancel_button.config(text=t("Cancel"))
    reset_button.config(text=t("Reset to Defaults"))
    save_preset_button.config(text=t("Save Preset"))

    # Sliders
    fg_shift_label.config(text=t("Foreground Shift"))
    mg_shift_label.config(text=t("Midground Shift"))
    bg_shift_label.config(text=t("Background Shift"))
    sharpness_factor_label.config(text=t("Sharpness Factor"))
    zero_parallax_strength_label.config(text=t("Zero Parallax Strength"))
    parallax_balance_label.config(text=t("Parallax Balance"))
    max_pixel_shift_label.config(text=t("Max Pixel Shift %"))
    dof_strength_label.config(text=t("DOF Strength"))
    convergence_strength_label.config(text=t("Convergence Strength"))

    # Toggles and Checkboxes
    preserve_aspect_checkbox.config(text=t("Preserve Original Aspect Ratio"))
    auto_crop_checkbox.config(text=t("Auto Crop Black Bars"))
    use_subject_tracking_checkbox.config(text=t("Stabilize Zero-Parallax (center-depth)"))
    use_dfw_checkbox.config(text=t("Enable Floating Window (DFW)"))
    use_ffmpeg_checkbox.config(text=t("Use FFmpeg Renderer"))
    enable_edge_checkbox.config(text=t("Enable Edge Masking"))
    enable_feathering_checkbox.config(text=t("Enable Feathering"))
    skip_blank_frames_checkbox.config(text=t("Skip Blank/White Frames"))
    enable_dynamic_convergence_checkbox.config(text=t("Enable Dynamic Convergence"))

    # Encoding Settings
    selected_aspect_ratio_label.config(text=t("Aspect Ratio:"))
    selected_ffmpeg_codec_label.config(text=t("FFmpeg Codec:"))
    selected_codec_label.config(text=t("Codec:"))
    crf_value_label.config(text=t("CRF"))
    nvenc_cq_value_label.config(text=t("NVENC CQ"))

    encoding_frame.config(text=t("Encoding Settings"))
    options_frame.config(text=t("Processing Options"))
    top_widgets_frame.config(text=t("Video Info"))


    # FrameTools Tab
    extract_frames_button.config(text=t("Extract Frames from Video"))
    io_frame.config(text=t("Input / Output"))
    frames_folder_label.config(text=t("Frames Folder:"))
    browse_button.config(text=t("Browse"))
    output_video_file_label.config(text=t("Output Video File:"))
    save_as_button.config(text=t("Save As"))

    proc_frame.config(text=t("⚙️ Processing Options"))
    RIFE_FPS_button.config(text=t("Enable RIFE Interpolation"))
    esrgan_button.config(text=t("Enable Real-ESRGAN Upscale"))

    out_frame.config(text=t("Output Settings"))
    resolution_label.config(text=t("Resolution (WxH):"))
    original_fps_label.config(text=t("Original FPS:"))
    fps_multi_label.config(text=t("FPS Interpolation Multiplier:"))
    selected_ffmpeg_codec_frametools_label.config(text=t("FFmpeg Output Codec:"))

    esrgan_frame.config(text=t("ESRGAN Settings"))
    ai_blend_select.config(text=t("AI Blending:"))
    input_res_pct_label.config(text=t("Input Resolution %:"))
    model_select.config(text=t("Model Selection:"))
    start_processing_button.config(text=t("▶ Start Processing"))

    # ✅ Update tooltip text for active language
    for key, tooltip in tooltip_refs.items():
        if tooltip:
            tooltip.text = t(f"Tooltip.{key}")

def get_all_presets():
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(PRESET_DIR, "*.json"))]

preset_menu['values'] = get_all_presets()

def apply_preset(preset_name):
    path = os.path.join(PRESET_DIR, f"{preset_name}.json")

    if not os.path.exists(path):
        print(f"❌ Preset not found: {path}")
        return

    with open(path, 'r') as f:
        config = json.load(f)

    fg_shift.set(config.get("fg_shift", 8.0))
    mg_shift.set(config.get("mg_shift", -3.0))
    bg_shift.set(config.get("bg_shift", -6.0))
    zero_parallax_strength.set(config.get("zero_parallax_strength", 0.0))
    max_pixel_shift.set(config.get("max_pixel_shift", 0.02))
    parallax_balance.set(config.get("parallax_balance", 0.8))
    sharpness_factor.set(config.get("sharpness_factor", 1.0))
    dof_strength.set(config.get("dof_strength", 2.0))
    convergence_strength.set(config.get("convergence_strength", 0.0))
    
    use_ffmpeg.set(config.get("use_ffmpeg", False))
    enable_feathering.set(config.get("enable_feathering", True))
    enable_edge_masking.set(config.get("enable_edge_masking", True))
    use_floating_window.set(config.get("use_floating_window", True))
    auto_crop_black_bars.set(config.get("auto_crop_black_bars", False))
    skip_blank_frames.set(config.get("skip_blank_frames", False))
    enable_dynamic_convergence.set(config.get("enable_dynamic_convergence", True))

    print(f"✅ Applied preset: {preset_name}")


def save_current_preset(name="custom_preset.json"):
    preset = {
        "fg_shift": fg_shift.get(),
        "mg_shift": mg_shift.get(),
        "bg_shift": bg_shift.get(),
        "zero_parallax_strength": zero_parallax_strength.get(),
        "max_pixel_shift": max_pixel_shift.get(),
        "parallax_balance": parallax_balance.get(),
        "sharpness_factor": sharpness_factor.get(),
        "use_ffmpeg": use_ffmpeg.get(),
        "enable_feathering": enable_feathering.get(),
        "enable_edge_masking": enable_edge_masking.get(),
        "use_floating_window": use_floating_window.get(),
        "auto_crop_black_bars": auto_crop_black_bars.get(),
        "skip_blank_frames": skip_blank_frames.get(),
        "dof_strength": dof_strength.get(),
        "convergence_strength": convergence_strength.get(),
        "enable_dynamic_convergence": enable_dynamic_convergence.get(),
    }

    path = os.path.join(PRESET_DIR, name)
    with open(path, 'w') as f:
        json.dump(preset, f, indent=4)

    print(f"💾 Preset saved: {name}")
    preset_menu['values'] = get_all_presets()
    preset_menu.set(os.path.splitext(name)[0])


def load_settings():
    global current_language
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error loading settings: {e}")
            return

        for name, value in settings.items():
            if name in gui_variables:
                try:
                    gui_variables[name].set(value)
                except Exception as e:
                    print(f"⚠️ Failed to set variable '{name}': {e}")

        # ✅ Restore input video path and refresh preview
        if "input_video_path" in settings and "input_video_path" in globals():
            try:
                input_video_path.set(settings["input_video_path"])
                if os.path.exists(input_video_path.get()):
                    select_input_video(
                        input_video_path,
                        video_thumbnail_label,
                        video_specs_label,
                        update_aspect_preview,
                        original_video_width,
                        original_video_height
                    )
            except Exception as e:
                print(f"⚠️ Could not restore input path: {e}")

        # ✅ Restore depth map and label
        if "selected_depth_map" in settings and "selected_depth_map" in globals():
            try:
                selected_depth_map.set(settings["selected_depth_map"])
                if os.path.exists(selected_depth_map.get()):
                    depth_map_label.config(
                        text=f"Selected Depth Map:\n{os.path.basename(selected_depth_map.get())}"
                    )
            except Exception as e:
                print(f"⚠️ Could not restore depth path: {e}")

        if "language" in settings:
            current_language = settings["language"]
            load_language(current_language)

        if "window_geometry" in settings:
            root.geometry(settings["window_geometry"])

        print("✅ Settings loaded from file.")

# Ensure settings are saved when the program closes
def on_exit():
    save_settings()
    root.destroy() # ❌ Close GUI

root.protocol("WM_DELETE_WINDOW", on_exit)


root.mainloop()
