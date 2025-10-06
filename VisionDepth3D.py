# ── Standard Library ─────────────────────────────
import os, platform, warnings
import sys
import cv2
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from threading import Event
from core.audio import launch_audio_gui
import queue


# ── External Libraries ───────────────────────────
from PIL import Image, ImageTk
import torch.nn.functional as F
import numpy as np
import re
import webbrowser
import glob
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


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
    parse_timecode,
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
    start_threaded_pipeline,
    select_video_and_generate_frames,
    select_output_file,
    select_frames_folder, 
    start_ffmpeg_writer,
)

# DB.py exports you already have
from core.DB import (
    lighten_beta,
    FramesWorker,
    VideosWorker,
    TORCH_CUDA,
)

from core.vd3d_live import launch_live_gui
from core.preview_gui import open_3d_preview_window
from core.models.depth_anything_v2.dpt import DepthAnythingV2

# At the top of GUI.py
cancel_requested = threading.Event()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
process_thread = None 
suspend_flag = Event()
cancel_flag = Event()
SETTINGS_FILE = "settings.json"
translations = {}
current_language = "en"
tooltip_refs = {}

# global dict to store menu references and indices
MENUS = {"file_menu": None, "help_menu": None,
         "file_btn": None, "help_btn": None, "lang_btn": None,
         "FILE_IDX": {}, "HELP_IDX": {}}


# --- VisionDepth3D Links ---
VD_WEBSITE   = "https://visiondepth.github.io/VisionDepth3D/"
VD_GITHUB    = "https://github.com/VisionDepth/VisionDepth3D"
VD_METHOD    = "https://github.com/VisionDepth/VisionDepth3D/blob/Main-Stable/VisionDepth3D_Method.md"
VD_RELEASES  = "https://github.com/VisionDepth/VisionDepth3D/releases"
VD_ISSUES    = "https://github.com/VisionDepth/VisionDepth3D/issues"
VD_REDDIT    = "https://www.reddit.com/r/VisionDepth3D/"

if platform.system() == "Windows":
    # If the env var is set anywhere, drop it so PyTorch won't warn
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    # Also hide the specific warning just in case
    warnings.filterwarnings(
        "ignore",
        message=".*expandable_segments not supported on this platform.*",
        category=UserWarning,
    )

# --- add near the top with other imports ---
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

def _menu_set(menu, idx, label, accel=None):
    """Update a menu entry label with optional accelerator text aligned right."""
    if not menu or idx is None:
        return
    if accel:
        # \t is the magic: Tkinter aligns everything after \t flush right
        menu.entryconfig(idx, label=f"{label}\t{accel}")
    else:
        menu.entryconfig(idx, label=label)

def open_github():
    """Opens the GitHub repository in a web browser."""
    webbrowser.open_new("https://github.com/VisionDepth/VisionDepth3D")

def open_aspect_ratio_CheatSheet():
    """Opens the Aspect Ratio Cheat Sheet."""
    webbrowser.open_new("https://www.wearethefirehouse.com/aspect-ratio-cheat-sheet")

def ui_select_input_video():
    path = select_input_video(
        input_video_path,
        video_thumbnail_label,
        video_specs_label,
        update_aspect_preview,     # <- your function/callback
        original_video_width,
        original_video_height,
    )
    # if your select_* returns a path you can save after:
    try: 
        if path: save_settings()
    except Exception:
        pass

def ui_select_depth_map():
    path = select_depth_map(
        selected_depth_map,        # adjust to your signature
        depth_map_label,         # e.g. if you have one
    )
    try:
        if path: save_settings()
    except Exception:
        pass

def ui_select_output_path():
    path = output_sbs_video_path(
        selected_depth_map,       
        depth_map_label,         
    )
    try:
        if path: save_settings()
    except Exception:
        pass

def _apply_preset_config(config: dict):
    """Apply a preset config dict to GUI variables (shared by all loaders)."""
    def _clamp(v, lo, hi): return max(lo, min(hi, v))

    fg_shift.set(float(config.get("fg_shift", 8.0)))
    mg_shift.set(float(config.get("mg_shift", -3.0)))
    bg_shift.set(float(config.get("bg_shift", -6.0)))
    zero_parallax_strength.set(float(config.get("zero_parallax_strength", 0.0)))
    max_pixel_shift.set(float(config.get("max_pixel_shift", 0.02)))
    parallax_balance.set(float(config.get("parallax_balance", 0.8)))
    sharpness_factor.set(float(config.get("sharpness_factor", 1.0)))
    dof_strength.set(float(config.get("dof_strength", 2.0)))
    convergence_strength.set(float(config.get("convergence_strength", 0.0)))

    use_ffmpeg.set(bool(config.get("use_ffmpeg", False)))
    enable_feathering.set(bool(config.get("enable_feathering", True)))
    enable_edge_masking.set(bool(config.get("enable_edge_masking", True)))
    use_floating_window.set(bool(config.get("use_floating_window", True)))
    auto_crop_black_bars.set(bool(config.get("auto_crop_black_bars", False)))
    skip_blank_frames.set(bool(config.get("skip_blank_frames", False)))
    enable_dynamic_convergence.set(bool(config.get("enable_dynamic_convergence", True)))

    gamma = float(config.get("depth_pop_gamma", 0.85))
    depth_pop_gamma.set(_clamp(gamma, 0.70, 1.20))

    mid = float(config.get("depth_pop_mid", 0.50))
    depth_pop_mid.set(_clamp(mid, 0.0, 1.0))

    lo = float(config.get("depth_stretch_lo", 0.05))
    hi = float(config.get("depth_stretch_hi", 0.95))
    lo = _clamp(lo, 0.0, 1.0); hi = _clamp(hi, 0.0, 1.0)
    if hi <= lo: lo, hi = 0.05, 0.95
    depth_stretch_lo.set(lo); depth_stretch_hi.set(hi)

    fg_mul = float(config.get("fg_pop_multiplier", 1.20))
    bg_mul = float(config.get("bg_push_multiplier", 1.10))
    subject_lock = float(config.get("subject_lock_strength", 1.00))
    fg_pop_multiplier.set(_clamp(fg_mul, 0.5, 2.0))
    bg_push_multiplier.set(_clamp(bg_mul, 0.5, 2.0))
    subject_lock_strength.set(_clamp(subject_lock, 0.0, 2.0))

    # Optional color grading
    sat = float(config.get("saturation", 1.0))
    con = float(config.get("contrast",   1.0))
    bri = float(config.get("brightness", 0.0))
    saturation.set(_clamp(sat, 0.0, 2.0))
    contrast.set(_clamp(con, 0.0, 2.0))
    brightness.set(_clamp(bri, -0.5, 0.5))

    # IPD / stereo separation (optional)
    try:
        ipd_on  = bool(config.get("ipd_enabled", False))
        ipd_val = _clamp(float(config.get("ipd_factor", 1.00)), 0.50, 1.50)
        if 'ipd_enabled_var' in globals(): ipd_enabled_var.set(ipd_on)
        if 'ipd_factor_var'  in globals(): ipd_factor_var.set(ipd_val)
        try:
            if 'ipd_slider' in globals():
                ipd_slider.config(state=('normal' if ipd_on else 'disabled'))
            if 'ipd_value_lbl' in globals():
                ipd_value_lbl.config(text=f"{ipd_val:.2f}x")
            if '_on_ipd_slider' in globals():
                _on_ipd_slider()
        except Exception:
            pass
    except Exception as e:
        print(f"⚠️ IPD restore skipped: {e}")

    # Sync any text entries if present
    try:
        pop_mid_entry.delete(0, tk.END);      pop_mid_entry.insert(0, f"{depth_pop_mid.get():.2f}")
        stretch_lo_entry.delete(0, tk.END);   stretch_lo_entry.insert(0, f"{depth_stretch_lo.get():.2f}")
        stretch_hi_entry.delete(0, tk.END);   stretch_hi_entry.insert(0, f"{depth_stretch_hi.get():.2f}")
    except Exception:
        pass

def apply_preset(preset_name: str):
    """Load preset by name from PRESET_DIR and apply."""
    path = os.path.join(PRESET_DIR, f"{preset_name}.json")
    if not os.path.exists(path):
        messagebox.showerror("Preset", f"Preset not found:\n{path}")
        print(f"❌ Preset not found: {path}")
        return
    with open(path, 'r', encoding="utf-8") as f:
        config = json.load(f)
    _apply_preset_config(config)
    print(f"✅ Applied preset: {preset_name}")
    
def load_preset_dialog():
    """Pick any preset JSON (defaulting to PRESET_DIR) and apply it."""
    initial = PRESET_DIR if os.path.isdir(PRESET_DIR) else os.getcwd()
    path = filedialog.askopenfilename(
        title="Load Preset",
        initialdir=initial,
        filetypes=[("Preset JSON", "*.json"), ("All files", "*.*")]
    )
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        _apply_preset_config(config)
        # If it was inside PRESET_DIR, reflect the name in any preset selector UI
        try:
            base = os.path.splitext(os.path.basename(path))[0]
            if 'preset_var' in globals():
                preset_var.set(base)
        except Exception:
            pass
        print(f"✅ Loaded preset from file: {path}")
    except Exception as e:
        messagebox.showerror("Load Preset", f"Failed to load preset:\n{e}")
        

def handle_generate_3d():
    global process_thread, is_rendering
    try:
        if process_thread is not None and process_thread.is_alive():
            print("⚠️ 3D processing already running! Use Suspend/Resume/Cancel.")
            return

        print("🚀 Starting new 3D processing thread...")
        cancel_flag.clear()
        suspend_flag.clear()
        is_rendering = True

        preserve_hdr10 = bool(preserve_hdr10_var.get())

        # --- small helpers ---
        def _get_time(v):
            try:
                s = v.get().strip()
                return parse_timecode(s) if s else None
            except Exception:
                return None

        def _get_num(varname, default=0.0):
            return globals()[varname].get() if varname in globals() else default

        # clip window
        start_s = _get_time(clip_start_var) if 'clip_start_var' in globals() else None
        end_s   = _get_time(clip_end_var)   if 'clip_end_var'   in globals() else None
        if start_s is not None and end_s is not None:
            if end_s <= start_s:
                end_s = start_s + end_s
            if end_s <= start_s:
                end_s = start_s + 0.001

        # colors
        sat = (color_saturation.get() if 'color_saturation' in globals() else saturation.get())
        con = (color_contrast.get()   if 'color_contrast'   in globals() else contrast.get())
        bri = (color_brightness.get() if 'color_brightness' in globals() else brightness.get())

        # IPD
        ipd_value = 0.0
        if 'ipd_enabled_var' in globals() and ipd_enabled_var.get():
            ipd_value = _get_num('ipd_factor_var', 1.0)

        # 👇 read output-mode from the UI ("sbs"|"left"|"right"|"both")
        eye_mode = stereo_out_var.get().strip().lower()

        # 👇 derive output names up-front
        base_out = output_sbs_video_path.get()
        base, ext = os.path.splitext(base_out)
        left_out  = f"{base}_LEFT{ext}"
        right_out = f"{base}_RIGHT{ext}"

        # what we will run
        if eye_mode == "sbs":
            jobs = [("sbs", base_out)]
        elif eye_mode == "left":
            jobs = [("left", left_out)]
        elif eye_mode == "right":
            jobs = [("right", right_out)]
        elif eye_mode == "both":
            jobs = [("left", left_out), ("right", right_out)]
        else:
            # fallback
            jobs = [("sbs", base_out)]

        def run_and_clear_flag():
            nonlocal start_s, end_s, sat, con, bri, ipd_value, preserve_hdr10, eye_mode
            created = []
            try:
                for mode, out_path in jobs:
                    print(f"▶️ Render pass: {mode} → {out_path}")

                    # NOTE: process_video must accept eye_mode + output_override (see patch below)
                    out_path_done = process_video(
                        input_video_path,
                        selected_depth_map,
                        output_sbs_video_path,   # still pass your Tk var (unused if output_override)
                        selected_codec,
                        fg_shift, mg_shift, bg_shift,
                        sharpness_factor,
                        output_format,
                        selected_aspect_ratio, aspect_ratios,
                        feather_strength, blur_ksize,
                        progress, progress_label,
                        suspend_flag, cancel_flag,
                        use_ffmpeg, preserve_hdr10,
                        selected_ffmpeg_codec, crf_value,
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
                        depth_pop_gamma, depth_pop_mid,
                        depth_stretch_lo, depth_stretch_hi,
                        fg_pop_multiplier, bg_push_multiplier,
                        subject_lock_strength,
                        sat, con, bri,
                        ipd_value,
                        start_s, end_s,
                        eye_mode=mode,              # 👈 NEW
                        output_override=out_path    # 👈 NEW
                    )
                    if out_path_done:
                        created.append(out_path_done)
                    if cancel_flag.is_set():
                        break

                # UI notify
                ui_root = progress_label.winfo_toplevel()
                if created:
                    msg = "Created file(s):\n" + "\n".join(created)
                    ui_root.after(0, lambda: messagebox.showinfo("3D Render", msg))
                else:
                    ui_root.after(0, lambda: messagebox.showwarning("3D Render", "Finished, but no outputs were created."))

            except Exception as e:
                print(f"❌ Error during 3D processing: {e}")
            finally:
                is_rendering = False

        process_thread = threading.Thread(target=run_and_clear_flag, daemon=True)
        process_thread.start()

    except Exception as e:
        print(f"❌ Error starting 3D processing: {e}")


        
def handle_open_preview():
    # optional: mirror Start button behavior
    try:
        save_settings()
    except Exception:
        pass

    # call the same preview function with the same args as your button
    return open_3d_preview_window(
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
        depth_pop_gamma,
        depth_pop_mid,
        depth_stretch_lo,
        depth_stretch_hi,
        fg_pop_multiplier,
        bg_push_multiplier,
        subject_lock_strength,
        saturation,
        contrast,
        brightness,
    )


def _val(v):
    # Only call .get() on actual tk.Variable instances; otherwise return as-is.
    return v.get() if isinstance(v, tk.Variable) else v

def rendering_in_progress():
    return is_rendering

def is_render_done():
    global process_thread
    return process_thread is None or not process_thread.is_alive()


def set_language(lang_code):
    global current_language
    current_language = lang_code
    load_language(lang_code)
    refresh_ui_labels()
    refresh_menu_labels()
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

# Load default language before building GUI
load_language("en")

# Get absolute path to resource (for PyInstaller compatibility)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

#Force include core/ into path
core_dir = resource_path("core")
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

#Force include languages/ into path
languages_dir = resource_path("languages")
if languages_dir not in sys.path:
    sys.path.insert(0, languages_dir)

#Inject DLL directory into PATH
#def inject_dll_directory():
#    dll_dir = resource_path("dlls")
#    if os.path.isdir(dll_dir):
#        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
#    else:
#        print(f"[Warning] DLL folder not found: {dll_dir}")

# 🟢 Call this before any ONNX/TensorRT/CUDA init
#inject_dll_directory()


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



def load_settings():
    global current_language
    if not os.path.exists(SETTINGS_FILE):
        return

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

    # ✅ Restore input video path and refresh thumbnail + video info
    input_path = settings.get("input_video_path", "")
    if input_path and os.path.exists(input_path):
        input_video_path.set(input_path)
        try:
            cap = cv2.VideoCapture(input_path)
            ret, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((128, 72), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                video_thumbnail_label.config(image=img_tk)
                video_thumbnail_label.image = img_tk  # Retain reference

                original_video_width.set(width)
                original_video_height.set(height)
                video_specs_label.config(
                    text=f"Resolution: {width}x{height}\nFPS: {fps:.2f}" if fps > 0 else "FPS: Unknown"
                )

                update_aspect_preview()
        except Exception as e:
            print(f"⚠️ Could not render video thumbnail: {e}")

    # ✅ Restore depth map path and label
    depth_path = settings.get("selected_depth_map", "")
    if depth_path and os.path.exists(depth_path):
        selected_depth_map.set(depth_path)
        try:
            depth_map_label.config(
                text=f"Selected Depth Map:\n{os.path.basename(depth_path)}"
            )
        except Exception as e:
            print(f"⚠️ Could not render depth label: {e}")

    # ✅ Restore language and window size
    if "language" in settings:
        current_language = settings["language"]
        load_language(current_language)

    if "window_geometry" in settings:
        root.geometry(settings["window_geometry"])

    print("✅ Settings loaded from file.")

def reset_settings():
    """Resets all GUI values and UI elements to their default states."""

    # 🎬 File Paths and Codecs
    input_video_path.set("")
    selected_depth_map.set("")
    output_sbs_video_path.set("")
    selected_codec.set("mp4v")
    selected_ffmpeg_codec.set("H.264 / AVC (libx264 - CPU)")
    output_format.set("Full-SBS")

    # 🧠 3D Shifting Parameters
    fg_shift.set(8.0)
    mg_shift.set(1.5)
    bg_shift.set(-2.5)

    # --- NEW: Pop & Subject Controls ---
    depth_pop_gamma.set(0.85)          # mid-contrast gamma curve for depth
    depth_pop_mid.set(0.50)            # mid depth pivot point
    depth_stretch_lo.set(0.05)         # lower depth compression bound
    depth_stretch_hi.set(0.95)         # upper depth compression bound
    fg_pop_multiplier.set(1.20)        # extra push-out for FG
    bg_push_multiplier.set(1.10)       # extra push-back for BG
    subject_lock_strength.set(1.00)    # subject tracking lock weight

    # ✨ Visual Enhancements
    sharpness_factor.set(0.2)


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
    convergence_strength.set(0.0)
    enable_dynamic_convergence.set(True)

    # 🎥 CRF for FFmpeg
    crf_value.set(23)
    
    # 🎨 Color Grading
    saturation.set(1.00)   # 0.00..2.00
    contrast.set(1.00)     # 0.00..2.00
    brightness.set(0.00)   # -0.50..+0.50

    

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
        
    # --- IPD controls ---
    ipd_enabled_var.set(True)      # or False if you want it off by default
    ipd_factor_var.set(1.00)
    try:
        ipd_readout.config(text=f"{ipd_factor_var.get():.2f}x" if ipd_enabled_var.get() else "OFF")
    except Exception:
        pass

    # --- Clip window (start/end) ---
    try:
        clip_start_var.set("")     # clears the UI entries
        clip_end_var.set("")
    except Exception:
        pass

    # --- Toggles that weren’t reset yet ---
    use_ffmpeg.set(False)              # you currently don’t reset this
    enable_edge_masking.set(True)      # whichever default you want
    enable_feathering.set(True)        # whichever default you want

    # --- Aspect ratio choice (if you expose it in UI) ---
    try:
        selected_aspect_ratio.set("Default (16:9)")
    except Exception:
        pass

    # --- Progress & flags ---
    progress["value"] = 0
    progress_label.config(text="0%")
    cancel_flag.clear()
    suspend_flag.clear()

    # --- Original video dims cache (if these are tk variables in your UI) ---
    try:
        original_video_width.set(0)
        original_video_height.set(0)
    except Exception:
        pass


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

is_rendering = False  # Make sure this is defined globally at the top of your script

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

class CreateToolTip:
    def __init__(self, widget, text_provider):
        self.widget = widget
        self.text_provider = text_provider
        self.tip_window = None
        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")  # hide on click

    def _get_text(self):
        try:
            val = self.text_provider()
            return (val or "").strip()
        except Exception:
            return ""

    def _on_enter(self, _):
        txt = self._get_text()
        if not txt:
            return
        self.show_tooltip(txt)

    def _on_leave(self, _):
        self.hide_tooltip()

    def show_tooltip(self, text):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)
        tw.wm_geometry(f"+{x}+{y}")
        frm = tk.Frame(tw, bg="#2a2a2a", bd=1, highlightthickness=0)
        lbl = tk.Label(frm, text=text, bg="#2a2a2a", fg="white",
                       font=("Segoe UI", 9), justify="left", padx=8, pady=5)
        frm.pack()
        lbl.pack()
        self.tip_window = tw

    def hide_tooltip(self):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

# ---GUI Setup---

# -----------------------
# Global Variables & Setup
# -----------------------

# ---- Dark theme helpers (once at startup) ----
BG_MAIN      = "#1e1e1e"
BG_CONTROLS  = "#292929"
FG_TEXT      = "white"
ACCENT_COLOR = "#4dd0e1"

from tkinter import ttk  # ensure ttk imported

def setup_dark_ttk(root):
    style = ttk.Style(root)
    style.theme_use("clam")

    # Base frame bg for ttk
    style.configure("VD3D.TFrame", background=BG_MAIN)

    # Notebook + Tabs
    style.configure("VD3D.TNotebook",
        background=BG_MAIN,
        borderwidth=0,
    )
    style.configure("VD3D.TNotebook.Tab",
        background=BG_CONTROLS,
        foreground=FG_TEXT,
        padding=(10, 6),
        borderwidth=0,
    )
    style.map("VD3D.TNotebook.Tab",
        background=[("selected", "#333333"), ("active", "#3a3a3a")],
        foreground=[("selected", ACCENT_COLOR), ("active", FG_TEXT)],
    )
    
    # 👉 add a few generic dark styles used in the Blender tab
    style.configure("VD3D.TLabel",       background=BG_MAIN,    foreground=FG_TEXT)
    style.configure("VD3D.TLabelframe",  background=BG_MAIN,    foreground=FG_TEXT, borderwidth=0)
    style.configure("VD3D.TLabelframe.Label", background=BG_MAIN, foreground=FG_TEXT)
    style.configure("VD3D.TButton",      background=BG_CONTROLS, foreground=FG_TEXT, padding=(10,4))
    style.map("VD3D.TButton",
              background=[("active","#3a3a3a")], foreground=[("active", FG_TEXT)])
    style.configure("VD3D.TCheckbutton", background=BG_MAIN,    foreground=FG_TEXT)
    style.configure("VD3D.TRadiobutton", background=BG_MAIN,    foreground=FG_TEXT)
    style.configure("VD3D.TEntry",       fieldbackground=BG_CONTROLS, foreground=FG_TEXT, insertcolor=FG_TEXT)
    style.configure("VD3D.Horizontal.TScale", background=BG_MAIN, troughcolor=BG_CONTROLS)
    style.configure("VD3D.Preview.TLabel", background=BG_CONTROLS, foreground=FG_TEXT, relief="groove")

    # make tk widgets default dark too
    root.option_add("*foreground", FG_TEXT)
    root.option_add("*background", BG_MAIN)
    root.option_add("*Entry.background", BG_CONTROLS)

    # Scrollbars
    style.configure("VD.Vertical.TScrollbar",
                    background=BG_CONTROLS, troughcolor=BG_MAIN,
                    bordercolor=BG_MAIN, arrowcolor=FG_TEXT,
                    lightcolor=BG_MAIN, darkcolor=BG_MAIN)
    style.configure("VD.Horizontal.TScrollbar",
                    background=BG_CONTROLS, troughcolor=BG_MAIN,
                    bordercolor=BG_MAIN, arrowcolor=FG_TEXT,
                    lightcolor=BG_MAIN, darkcolor=BG_MAIN)

    # Progressbar (optional)
    style.configure("VD.Horizontal.TProgressbar",
                    background=ACCENT_COLOR, troughcolor=BG_CONTROLS)



# ---------- Reusable dark scrollable container ----------
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *, vscroll=True, hscroll=False,
                 bg=BG_MAIN, inner_bg=BG_MAIN, **kwargs):
        super().__init__(parent, **kwargs)

        # canvas draws the background
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0, bg=bg)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # inner content frame (tk.Frame so bg actually shows)
        self.inner = tk.Frame(self.canvas, bg=inner_bg)

        # scrollbars (ttk, dark style)
        self.vsb = None
        self.hsb = None
        if vscroll:
            self.vsb = ttk.Scrollbar(self, orient="vertical",
                                     command=self.canvas.yview,
                                     style="VD.Vertical.TScrollbar")
            self.canvas.configure(yscrollcommand=self.vsb.set)
            self.vsb.grid(row=0, column=1, sticky="ns")
        if hscroll:
            self.hsb = ttk.Scrollbar(self, orient="horizontal",
                                     command=self.canvas.xview,
                                     style="VD.Horizontal.TScrollbar")
            self.canvas.configure(xscrollcommand=self.hsb.set)
            self.hsb.grid(row=1, column=0, sticky="ew")

        # put inner frame into canvas
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # keep scrollregion current + make inner match width on resize
        self.inner.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self._win, width=e.width)
        )

        # mousewheel: only while hovered
        self._bind_hover_scroll()
        self.configure(style="VD3D.TFrame")  # ensure bg matches

    def _bind_hover_scroll(self):
        self.canvas.bind("<Enter>", self._mw_bind)
        self.canvas.bind("<Leave>", self._mw_unbind)
        self.inner.bind("<Enter>", self._mw_bind)
        self.inner.bind("<Leave>", self._mw_unbind)
        for seq in ("<Button-4>", "<Button-5>"):  # Linux
            self.canvas.bind(seq, self._on_mousewheel)
            self.inner.bind(seq, self._on_mousewheel)
        self.bind("<Destroy>", lambda _e: self._mw_unbind())

    def _mw_bind(self, _e=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_wheel)

    def _mw_unbind(self, _e=None):
        try:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Shift-MouseWheel>")
        except Exception:
            pass

    def _on_mousewheel(self, event):
        if event.num in (4, 5):
            delta = -1 if event.num == 4 else 1
        else:
            delta = int(-event.delta / 120)
        self.canvas.yview_scroll(delta, "units")

    def _on_shift_wheel(self, event):
        delta = int(-event.delta / 120) if event.delta else 0
        self.canvas.xview_scroll(delta, "units")

# --- Window Setup ---
root = tk.Tk()
root.title("VisionDepth3D v3.6")
root.geometry("1734x792+131+119")
root.resizable(True, True)

# Apply dark ttk styles once
setup_dark_ttk(root)

# Root grid
root.grid_rowconfigure(0, weight=0)  # header
root.grid_rowconfigure(1, weight=1)  # notebook
root.grid_columnconfigure(0, weight=1)

# OPTIONAL: these affect tk.Menu dropdown styling; not required for custom header
MENU_BG        = "#1e1e1e"
MENU_FG        = "white"
MENU_ACTIVE_BG = "#333333"
MENU_ACTIVE_FG = "#4dd0e1"
MENU_FONT      = ("Segoe UI", 9)
root.option_add("*Menu.background",       MENU_BG)
root.option_add("*Menu.foreground",       MENU_FG)
root.option_add("*Menu.activeBackground", MENU_ACTIVE_BG)
root.option_add("*Menu.activeForeground", MENU_ACTIVE_FG)
root.option_add("*Menu.relief",           "flat")
root.option_add("*Menu.borderWidth",      1)
root.option_add("*Menu.font",             MENU_FONT)
root.option_add("*tearOff",               False)

# --- Dark Header Bar (custom, replaces native menubar) ---
HEADER_BG  = "#1e1e1e"
HEADER_FG  = "white"
HEADER_HOV = "#2a2a2a"
ACCENT     = "#4dd0e1"


def _open(url: str):
    try:
        webbrowser.open_new(url)
    except Exception:
        messagebox.showerror("Open Link", f"Couldn't open:\n{url}")

def open_website():  _open(VD_WEBSITE)
def open_github():   _open(VD_GITHUB)
def open_method():   _open(VD_METHOD)
def open_releases(): _open(VD_RELEASES)
def open_issues():   _open(VD_ISSUES)
def open_reddit():   _open(VD_REDDIT)


from tkinter import ttk

def apply_vd3d_dark_styles(root):
    BG  = "#1c1c1c"   # window/frames
    FG  = "white"
    ACC = "#2a2a2a"   # hover
    TR  = "#333333"   # trough/borders
    TXT = "#151515"   # text widgets background

    style = ttk.Style(root)
    style.theme_use("clam")  # enables colorable elements on Windows

    # base + labels
    style.configure(".", background=BG, foreground=FG)
    style.configure("VD3D.TLabel", background=BG, foreground=FG)

    # frames / labelframes (kill light borders)
    style.configure("VD3D.TFrame", background=BG)
    style.configure("VD3D.TLabelframe",
                    background=BG,
                    bordercolor=BG, lightcolor=BG, darkcolor=BG,
                    relief="flat", borderwidth=1)
    style.configure("VD3D.TLabelframe.Label", background=BG, foreground=FG)

    # preview label (no border)
    style.configure("VD3D.Preview.TLabel", background=BG, borderwidth=0, relief="flat")

    # buttons/entries/radios/checks to match
    style.configure("VD3D.TButton", background=ACC, foreground=FG, bordercolor=TR)
    style.map("VD3D.TButton", background=[("active", TR), ("!active", ACC)])
    style.configure("VD3D.TEntry", fieldbackground=TXT, foreground=FG, bordercolor=TR)
    style.configure("VD3D.TRadiobutton", background=BG, foreground=FG)
    style.configure("VD3D.TCheckbutton", background=BG, foreground=FG)

    # sliders + progress
    style.configure("VD3D.Horizontal.TScale", background=BG, troughcolor=TR)
    style.configure("VD3D.Horizontal.TProgressbar", background=ACC, troughcolor=TR)

    # scrollbars (vertical + horizontal)
    style.configure("VD3D.Vertical.TScrollbar",
                    background=TR, troughcolor=BG,
                    bordercolor=BG, lightcolor=BG, darkcolor=BG,
                    arrowcolor=FG)
    style.map("VD3D.Vertical.TScrollbar",
              background=[("active", ACC), ("!active", TR)])
    style.configure("VD3D.Horizontal.TScrollbar",
                    background=TR, troughcolor=BG,
                    bordercolor=BG, lightcolor=BG, darkcolor=BG,
                    arrowcolor=FG)
    style.map("VD3D.Horizontal.TScrollbar",
              background=[("active", ACC), ("!active", TR)])

    # kill white focus rings on tk widgets globally
    root.option_add("*highlightThickness", 0)
    root.option_add("*BorderWidth", 0)
    root.option_add("*Text.highlightThickness", 0)
    root.option_add("*Text.borderWidth", 0)
    root.option_add("*Canvas.highlightThickness", 0)
    root.option_add("*Canvas.borderWidth", 0)
    
    # --- tk.Listbox dark defaults (since Listbox isn't ttk) ---
    root.option_add("*Listbox.background", "#151515")
    root.option_add("*Listbox.foreground", "white")
    root.option_add("*Listbox.selectBackground", "#3a3a3a")
    root.option_add("*Listbox.selectForeground", "white")
    root.option_add("*Listbox.highlightThickness", 1)      # subtle focus ring
    root.option_add("*Listbox.highlightBackground", "#333333")
    root.option_add("*Listbox.highlightColor", "#4dd0e1")  # accent on focus
    root.option_add("*Listbox.borderwidth", 0)
    root.option_add("*Listbox.relief", "flat")


# call once at startup, right after setup_dark_ttk(root)
apply_vd3d_dark_styles(root)

def vd3d_listbox(parent, height=6):
    # wrapper gets the dark bg via VD3D.TFrame
    wrap = ttk.Frame(parent, style="VD3D.TFrame")
    lb = tk.Listbox(wrap, height=height)
    lb.grid(row=0, column=0, sticky="nsew")
    sb = ttk.Scrollbar(wrap, orient="vertical",
                       command=lb.yview, style="VD.Vertical.TScrollbar")
    sb.grid(row=0, column=1, sticky="ns", padx=(4,0))
    lb.configure(yscrollcommand=sb.set)
    wrap.columnconfigure(0, weight=1)
    wrap.rowconfigure(0, weight=1)
    return wrap, lb

    
def build_dark_header(root, on_language_change):
    style = ttk.Style(root); style.theme_use("clam")
    style.configure("VD.Menu.TMenubutton",
                    background=HEADER_BG, foreground=HEADER_FG,
                    padding=(10,4), relief="flat")
    style.map("VD.Menu.TMenubutton",
              background=[("active", HEADER_HOV)],
              foreground=[("active", HEADER_FG)])

    hdr = tk.Frame(root, bg=HEADER_BG)
    hdr.grid_columnconfigure(99, weight=1)  # spacer stretch


    def mk_menu():
        return tk.Menu(hdr, tearoff=False, bg=HEADER_BG, fg=HEADER_FG,
                       activebackground=HEADER_HOV, activeforeground=ACCENT,
                       relief="flat", borderwidth=1)

    # Language
    lang_var = tk.StringVar(value="en")
    lang_btn = ttk.Menubutton(hdr, text="🌐 Language", style="VD.Menu.TMenubutton")
    lang_menu = mk_menu()
    for code, label in [("en","English"),("fr","Français"),("de","German"),
                        ("es","Español"),("ja","Japanese")]:
        lang_menu.add_radiobutton(
            label=label, variable=lang_var, value=code,
            command=lambda c=code: (lang_var.set(c), on_language_change(c))
        )
    lang_btn["menu"] = lang_menu
    lang_btn.grid(row=0, column=0, padx=(8,0), pady=4)

    # File Menu
    file_btn = ttk.Menubutton(hdr, text="File", style="VD.Menu.TMenubutton")
    file_menu = mk_menu()

    _menu_add(file_menu, "Save Settings",            save_settings,           "Ctrl+S")
    MENUS["FILE_IDX"]["save_settings"] = file_menu.index("end")

    _menu_add(file_menu, "Save Preset As…",          prompt_and_save_preset,  "Ctrl+Shift+S")
    MENUS["FILE_IDX"]["save_preset_as"] = file_menu.index("end")

    _menu_add(file_menu, "Load Settings",            load_settings,           "Ctrl+L")
    MENUS["FILE_IDX"]["load_settings"] = file_menu.index("end")

    _menu_add(file_menu, "Load Preset…",             load_preset_dialog,      "Ctrl+P")
    MENUS["FILE_IDX"]["load_preset"] = file_menu.index("end")

    file_menu.add_separator()

    _menu_add(file_menu, "Video",                    ui_select_input_video,   "Ctrl+I")
    MENUS["FILE_IDX"]["video"] = file_menu.index("end")

    _menu_add(file_menu, "Depth Map",                ui_select_depth_map,     "Ctrl+D")
    MENUS["FILE_IDX"]["depth_map"] = file_menu.index("end")

    _menu_add(file_menu, "Output Path",              select_output_video,     "Ctrl+O")
    MENUS["FILE_IDX"]["output_path"] = file_menu.index("end")

    _menu_add(file_menu, "Generate 3D",              handle_generate_3d,      "Shift+Enter")
    MENUS["FILE_IDX"]["generate_3d"] = file_menu.index("end")

    _menu_add(file_menu, "Cancel",                   cancel_processing,       "Esc")
    MENUS["FILE_IDX"]["cancel"] = file_menu.index("end")

    file_menu.add_separator()

    _menu_add(file_menu, "Audio Tool",               launch_audio_gui,        "Ctrl+A")
    MENUS["FILE_IDX"]["audio_tool"] = file_menu.index("end")

    file_menu.add_separator()

    _menu_add(file_menu, "Reset to Defaults",        reset_settings,          "Ctrl+R")
    MENUS["FILE_IDX"]["reset_defaults"] = file_menu.index("end")

    _menu_add(file_menu, "Exit",                     root.quit,               "Ctrl+Q")
    MENUS["FILE_IDX"]["exit"] = file_menu.index("end")

    file_btn["menu"] = file_menu
    file_btn.grid(row=0, column=1, padx=(8,0), pady=4)


    # Help
    help_btn = ttk.Menubutton(hdr, text="Help", style="VD.Menu.TMenubutton")
    help_menu = mk_menu()

    # About
    help_menu.add_command(
        label=t("Help.About"),        # localized caption only
        accelerator="F1",             # shortcut shown on the right
        command=lambda: messagebox.showinfo(
            "About VisionDepth3D",
            (
                "VisionDepth3D v3.6\n"
                "----------------------------\n"
                "A hybrid real-time 2D-to-3D conversion suite for cinema and VR.\n\n"
                "Features:\n"
                " • Depth map blending (multi-model)\n"
                " • Depth-weighted parallax shifting\n"
                " • Scene-aware stereo rendering\n"
                " • CUDA acceleration\n"
                " • Real-time preview & batch processing\n\n"
                "Website: " + VD_WEBSITE + "\n"
                "GitHub:  " + VD_GITHUB  + "\n"
                "© 2025 VisionDepth3D"
            )
        )
    )
    MENUS["HELP_IDX"]["about"] = help_menu.index("end")

    


    _menu_add(help_menu, t("Help.OfficialWebsite"), open_website,               "F2"); MENUS["HELP_IDX"]["website"] = help_menu.index("end")
    _menu_add(help_menu, t("Help.Reddit"),          open_reddit,                "F3"); MENUS["HELP_IDX"]["reddit"]  = help_menu.index("end")
    _menu_add(help_menu, t("Help.Docs"),            open_method,                "F4"); MENUS["HELP_IDX"]["docs"]    = help_menu.index("end")
    _menu_add(help_menu, t("Help.StarGithub"),      open_github,                "F5"); MENUS["HELP_IDX"]["star"]    = help_menu.index("end")

    help_menu.add_separator()

    _menu_add(help_menu, t("Help.CheckUpdates"),    open_releases,              "F6"); MENUS["HELP_IDX"]["updates"] = help_menu.index("end")
    _menu_add(help_menu, t("Help.ReportBug"),       open_issues,                "F7"); MENUS["HELP_IDX"]["report"]  = help_menu.index("end")
    _menu_add(help_menu, t("Help.AspectCheat"),     open_aspect_ratio_CheatSheet,"F8");MENUS["HELP_IDX"]["aspect"]  = help_menu.index("end")
    _menu_add(help_menu, t("Help.PreviewGUI"),      handle_open_preview,        "F9"); MENUS["HELP_IDX"]["preview"] = help_menu.index("end")
    
    # --- GPU Diagnostics (Help menu) ---
    def _run_gpu_diag():
        import gpu_diag
        rpt = gpu_diag.gpu_diagnostics()  # no return_text
        try:
            root.clipboard_clear()
            root.clipboard_append(rpt)
        except Exception:
            pass


    _menu_add(help_menu, t("Help.GPUDiagnostics"), _run_gpu_diag, "F10")
    MENUS["HELP_IDX"]["gpu_diag"] = help_menu.index("end")


    help_btn["menu"] = help_menu
    help_btn.grid(row=0, column=2, padx=(8,0), pady=4)
        

    MENUS["file_menu"] = file_menu
    MENUS["file_btn"]  = file_btn
    MENUS["help_menu"] = help_menu
    MENUS["help_btn"]  = help_btn
    MENUS["lang_btn"]  = lang_btn

    # divider
    tk.Frame(hdr, bg="#2a2a2a", height=1).grid(row=1, column=0, columnspan=100, sticky="ew")
    return hdr
    
def refresh_menu_labels():
    fm, hm = MENUS["file_menu"], MENUS["help_menu"]
    f, h = MENUS["FILE_IDX"], MENUS["HELP_IDX"]

    # Menubutton captions
    if MENUS["lang_btn"]: MENUS["lang_btn"].config(text=t("Menu.Language"))
    if MENUS["file_btn"]: MENUS["file_btn"].config(text=t("Menu.File"))
    if MENUS["help_btn"]: MENUS["help_btn"].config(text=t("Menu.Help"))

    # File
    _menu_set(fm, f["save_settings"],  t("Menu.SaveSettings"),   "Ctrl+S")
    _menu_set(fm, f["save_preset_as"], t("Menu.SavePresetAs"),   "Ctrl+Shift+S")
    _menu_set(fm, f["load_settings"],  t("Menu.LoadSettings"),   "Ctrl+L")
    _menu_set(fm, f["load_preset"],    t("Menu.LoadPreset"),     "Ctrl+P")
    _menu_set(fm, f["video"],          t("Menu.Video"),          "Ctrl+I")
    _menu_set(fm, f["depth_map"],      t("Menu.DepthMap"),       "Ctrl+D")
    _menu_set(fm, f["output_path"],    t("Menu.OutputPath"),     "Ctrl+O")
    _menu_set(fm, f["generate_3d"],    t("Menu.Generate3D"),     "Shift+Enter")
    _menu_set(fm, f["cancel"],         t("Menu.Cancel"),         "Esc")
    _menu_set(fm, f["audio_tool"],     t("Menu.AudioTool"),      "Ctrl+A")
    _menu_set(fm, f["reset_defaults"], t("Menu.ResetDefaults"),  "Ctrl+R")
    _menu_set(fm, f["exit"],           t("Menu.Exit"),           "Ctrl+Q")

    # Help
    _menu_set(hm, h["about"],   t("Help.About"),           "F1")
    _menu_set(hm, h["website"], t("Help.OfficialWebsite"), "F2")
    _menu_set(hm, h["reddit"],  t("Help.Reddit"),          "F3")
    _menu_set(hm, h["docs"],    t("Help.Docs"),            "F4")
    _menu_set(hm, h["star"],    t("Help.StarGithub"),      "F5")
    _menu_set(hm, h["updates"], t("Help.CheckUpdates"),    "F6")
    _menu_set(hm, h["report"],  t("Help.ReportBug"),       "F7")
    _menu_set(hm, h["aspect"],  t("Help.AspectCheat"),     "F8")
    _menu_set(hm, h["preview"], t("Help.PreviewGUI"),      "F9")
    _menu_set(hm, h["gpu_diag"], t("Help.GPUDiagnostics"), "F10")

    
def _menu_add(menu, label, command, accel=None):
    # Create a command entry with a proper accelerator column
    menu.add_command(label=label, accelerator=(accel or ""), command=command)

def _menu_set(menu, index, label, accel=None):
    # Update an existing entry (used by refresh_menu_labels)
    try:
        menu.entryconfig(index, label=label)
        menu.entryconfig(index, accelerator=(accel or ""))
    except Exception:
        pass


def _on_language_change(code):
    load_language(code)
    refresh_ui_labels()
    refresh_menu_labels()


header = build_dark_header(root, _on_language_change)
header.grid(row=0, column=0, sticky="ew")

# Shortcuts

root.bind_all("<Control-q>", lambda e: root.quit())
root.bind_all("<F1>", lambda e: messagebox.showinfo("About", "VisionDepth3D v3.5\n"
                "----------------------------\n"
                "A hybrid real-time 2D-to-3D conversion suite for cinema and VR.\n\n"
                "Features:\n"
                " • 25+ AI depth estimation models\n"
                " • Depth-weighted parallax shifting\n"
                " • Scene-aware stereo rendering\n"
                " • CUDA acceleration\n"
                " • Real-time preview & batch processing\n\n"
                "Created by: Johnathan Carpenter\n"
                "Website: https://github.com/VisionDepth/VisionDepth3D\n"
                "© 2025 VisionDepth3D. All rights reserved.",))
root.bind_all("<F2>", lambda e: open_website())
root.bind_all("<F3>", lambda e: open_reddit())
root.bind_all("<F4>", lambda e: open_github())
root.bind_all("<F5>", lambda e: open_method())
root.bind_all("<F6>", lambda e: open_releases())
root.bind_all("<F7>", lambda e: open_issues())
root.bind_all("<F8>", lambda e: open_aspect_ratio_CheatSheet())
root.bind_all("<F9>", lambda e: handle_open_preview())
root.bind_all("<F10>", lambda e: _run_gpu_diag())

# File operations
root.bind_all("<Control-i>", lambda e: ui_select_input_video())         # Input video
root.bind_all("<Control-d>", lambda e: ui_select_depth_map())           # Depth map
root.bind_all("<Control-o>", lambda e: select_output_video(output_sbs_video_path))  # Output file

# Presets
root.bind_all("<Control-s>", lambda e: save_settings())                 # Save current settings
root.bind_all("<Control-l>", lambda e: load_settings())                 # Load saved settings
root.bind_all("<Control-Shift-S>", lambda e: prompt_and_save_preset())  # Save as preset
root.bind_all("<Control-p>", lambda e: load_preset_dialog())            # Load preset

# Render
root.bind_all("<Shift-Return>", lambda e: handle_generate_3d())                   # Start 3D render
root.bind_all("<Escape>", lambda e: cancel_processing())                # Cancel render

# Extras
root.bind_all("<Control-r>", lambda e: reset_settings())                # Reset
root.bind_all("<Control-a>", lambda e: launch_audio_gui())              # Audio GUI


# Arrow key navigation between tabs
root.bind_all("<Left>",  lambda e: tab_control.select(frametools_tab))
root.bind_all("<Up>",    lambda e: tab_control.select(depth_estimation_frame))
root.bind_all("<Right>", lambda e: tab_control.select(depth_blend_frame))
root.bind_all("<Down>",  lambda e: tab_control.select(visiondepth_frame))



# --- Notebook for Tabs (single instance, dark) ---
tab_control = ttk.Notebook(root, style="VD3D.TNotebook")
tab_control.grid(row=1, column=0, sticky="nsew")

# --- FrameTools Tab ---
frametools_tab = ttk.Frame(tab_control, style="VD3D.TFrame")
tab_control.add(frametools_tab, text="FrameTools")
frametools_tab_index = tab_control.index("end") - 1

ft_scroll = ScrollableFrame(frametools_tab, vscroll=True, hscroll=False,
                            bg=BG_MAIN, inner_bg=BG_MAIN)
ft_scroll.pack(fill="both", expand=True)
frametools_inner = ft_scroll.inner   # <-- use this for your widgets, not for select()

# --- Depth Estimation GUI ---
depth_estimation_frame = ttk.Frame(tab_control, style="VD3D.TFrame")
tab_control.add(depth_estimation_frame, text="Depth Estimation")
depth_tab_index = tab_control.index("end") - 1

depth_content_frame = tk.Frame(depth_estimation_frame, bg=BG_MAIN, highlightthickness=0, bd=0)
depth_content_frame.pack(fill="both", expand=True)

sidebar = tk.Frame(depth_content_frame, bg=BG_MAIN, width=320)
sidebar.pack(side="left", fill="y")

main_content = tk.Frame(depth_content_frame, bg=BG_CONTROLS)
main_content.pack(side="right", fill="both", expand=True)

# --- Depth Blender Tab (scrollable, dark) ---
depth_blend_frame = ttk.Frame(tab_control, style="VD3D.TFrame")
tab_control.add(depth_blend_frame, text="Depth-Blender")
depth_blend_index = tab_control.index("end") - 1

scroll_area = ScrollableFrame(depth_blend_frame, vscroll=True, hscroll=False,
                              bg=BG_MAIN, inner_bg=BG_MAIN)
scroll_area.pack(fill="both", expand=True)
depthblend_content_frame = scroll_area.inner  # parent for your 3D widgets


# --- 3D Video Generator Tab (scrollable, dark) ---
visiondepth_frame = ttk.Frame(tab_control, style="VD3D.TFrame")
tab_control.add(visiondepth_frame, text="3D Video Generator")
visiondepth_tab_index = tab_control.index("end") - 1

scroll_area = ScrollableFrame(visiondepth_frame, vscroll=False, hscroll=False,
                              bg=BG_MAIN, inner_bg=BG_MAIN)
scroll_area.pack(fill="both", expand=True)
visiondepth_content_frame = scroll_area.inner  # parent for your 3D widgets

parent = visiondepth_content_frame

left_col  = ttk.Frame(parent, style="VD3D.TFrame")
right_col = ttk.Frame(parent, style="VD3D.TFrame")

left_col.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=10)
right_col.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=10)

# allow both columns to expand horizontally, but let only the right grow tall
parent.grid_columnconfigure(0, weight=1, minsize=680)   # tweak minsize to taste
parent.grid_columnconfigure(1, weight=1)
parent.grid_rowconfigure(0, weight=1)

left_col.grid_rowconfigure(99, weight=1)

# your two columns inside visiondepth_content_frame
visiondepth_content_frame.grid_columnconfigure(0, weight=1)
visiondepth_content_frame.grid_columnconfigure(1, weight=1)

def relayout(event=None):
    w = visiondepth_content_frame.winfo_width()
    if w < 1400:
        # stack right under left
        right_col.grid(row=1, column=0, sticky="nsew")
        visiondepth_content_frame.grid_columnconfigure(1, weight=0)
    else:
        # two columns side by side
        right_col.grid(row=0, column=1, sticky="nsew")
        visiondepth_content_frame.grid_columnconfigure(1, weight=1)

visiondepth_content_frame.bind("<Configure>", relayout)
relayout()

# --- Depthblend Content ---
class DepthBlenderTab(ttk.Frame):
    """
    Frame-embedded version of your Depth Blender 'App'.
    Minimal changes from App(tk.Tk):
      - Inherit from ttk.Frame, not tk.Tk
      - No title/geometry/minsize calls
      - Key binds go to the toplevel
      - Everything else stays the same
    """
    def __init__(self, parent):
        super().__init__(parent)
        # ---- state ----
        self.mode = tk.StringVar(value="frames")
        self.overwrite_v2 = tk.BooleanVar(value=True)
        self.v1_path = tk.StringVar()
        self.v2_path = tk.StringVar()
        self.out_path = tk.StringVar()
        self.w_var = tk.StringVar()
        self.h_var = tk.StringVar()
        self.use_gpu = tk.BooleanVar(value=TORCH_CUDA)

        # params
        self.white_strength = tk.DoubleVar(value=1.0)
        self.blur_k = tk.IntVar(value=35)
        self.clip_limit = tk.DoubleVar(value=2.0)
        self.tile_grid = tk.IntVar(value=8)
        self.bf_d = tk.IntVar(value=12)
        self.bf_sigmaColor = tk.IntVar(value=75)
        self.bf_sigmaSpace = tk.IntVar(value=75)

        # preview infra
        self._preview_lock = threading.Lock()
        self._preview_thread = None
        self._preview_after = None
        self._preview_imgtk = None  # keep ref

        # scrubber
        self.preview_index = tk.IntVar(value=0)
        self.preview_max   = tk.IntVar(value=0)
        self._idx_scale = None

        # queues / worker
        self.qlog = queue.Queue()
        self.qprog = queue.Queue()
        self.stop_evt = threading.Event()
        self.worker = None

        self._build_ui()

        # key binds on the toplevel so arrows work even if focus is elsewhere
        top = self.winfo_toplevel()
        top.bind("<Left>",  lambda e: self._nudge_preview(-1))
        top.bind("<Right>", lambda e: self._nudge_preview(+1))

        # poller
        self.after(100, self._poll)

    # ---------- UI ----------
    def _build_ui(self):
        BG_PAD = {"padx": 10, "pady": 6}

        # Title
        self.title_lbl = ttk.Label(self, text="Depth Blender", style="VD3D.TLabel")
        self.title_lbl.pack(fill="x", **BG_PAD)


        # Top split: left controls / right preview
        top = ttk.Frame(self, style="VD3D.TFrame"); top.pack(fill="both", expand=True, **BG_PAD)
        left = ttk.Frame(top, style="VD3D.TFrame"); left.pack(side="left", fill="y", padx=6, pady=0)
        right = ttk.Frame(top, style="VD3D.TFrame"); right.pack(side="left", fill="both", expand=True, padx=6, pady=0)

        # --- Preview area (Canvas with subtle border + placeholder) ---
        self.preview_lbl = ttk.Label(
            right, text="Preview (scrubbable):",
            style="VD3D.TLabel"
            )
        self.preview_lbl.pack(anchor="w")
        border = tk.Frame(right, bg="#2a2a2a", highlightthickness=0)
        border.pack(fill="both", expand=True, padx=6, pady=6)

        self.preview_canvas = tk.Canvas(
            border, bg="#1c1c1c", highlightthickness=0, bd=0, width=640, height=360
        )
        self.preview_canvas.pack(fill="both", expand=True, padx=1, pady=1)
        self._preview_canvas_img = None
        self.preview_canvas.bind("<Configure>", lambda e: self._redraw_preview())
        self._draw_preview_placeholder()

        # -------------------- LEFT COLUMN --------------------
        # --- Mode ---
        self.mode_frame = ttk.LabelFrame(left, text="Mode", style="VD3D.TLabelframe")
        self.mode_frame.pack(fill="x", padx=6, pady=6)

        self.rb_frames = ttk.Radiobutton(
            self.mode_frame,
            text="Folders (frames)",
            variable=self.mode,
            value="frames",
            command=self._toggle_mode, style="VD3D.TRadiobutton"
        )
        self.rb_frames.grid(row=0, column=0, sticky="w", padx=6, pady=4)

        self.rb_videos = ttk.Radiobutton(
            self.mode_frame,
            text="Videos",
            variable=self.mode,
            value="videos",
            command=self._toggle_mode,
            style="VD3D.TRadiobutton"
        )
        self.rb_videos.grid(row=0, column=1, sticky="w", padx=12, pady=4)

        # --- GPU ---
        self.gpu_row = ttk.Frame(left, style="VD3D.TFrame")
        self.gpu_row.pack(fill="x", padx=6, pady=0)

        self.gpu_btn = ttk.Checkbutton(
            self.gpu_row,
            text="Use GPU (PyTorch CUDA)",
            variable=self.use_gpu,
            command=lambda: self._schedule_preview(120),
            style="VD3D.TCheckbutton"
        )
        self.gpu_btn.pack(anchor="w")

        # --- Inputs ---
        self.paths = ttk.LabelFrame(left, text="Inputs", style="VD3D.TLabelframe")
        self.paths.pack(fill="x", padx=6, pady=6)

        self.v1_lbl = ttk.Label(self.paths, text="V1 path:", style="VD3D.TLabel")
        self.v1_lbl.grid(row=0, column=0, sticky="e")

        ttk.Entry(self.paths, textvariable=self.v1_path, width=40, style="VD3D.TEntry")\
           .grid(row=0, column=1, sticky="we", padx=6)

        self.v1_browse = ttk.Button(self.paths, text="Browse…", command=self._browse_v1, style="VD3D.TButton")
        self.v1_browse.grid(row=0, column=2, padx=4)

        self.v2_lbl = ttk.Label(self.paths, text="V2 path:", style="VD3D.TLabel")   # <- separate create
        self.v2_lbl.grid(row=1, column=0, sticky="e")                           # <- then grid

        ttk.Entry(self.paths, textvariable=self.v2_path, width=40, style="VD3D.TEntry")\
           .grid(row=1, column=1, sticky="we", padx=6)

        self.v2_browse = ttk.Button(self.paths, text="Browse…", command=self._browse_v2, style="VD3D.TButton")
        self.v2_browse.grid(row=1, column=2, padx=4)

        self.paths.grid_columnconfigure(1, weight=1)

        # --- Output ---
        self.outf = ttk.LabelFrame(left, text="Output", style="VD3D.TLabelframe")
        self.outf.pack(fill="x", padx=6, pady=6)  # <- self.outf

        self.chk_over = ttk.Checkbutton(
            self.outf,
            text="Overwrite V2 (frames mode only)",
            variable=self.overwrite_v2,
            command=self._toggle_out_controls,
            style="VD3D.TCheckbutton"
        )
        self.chk_over.grid(row=0, column=0, sticky="w", padx=6)

        self.out_lbl = ttk.Label(self.outf, text="Output path/file:", style="VD3D.TLabel")  # keep a ref if you want a tooltip
        self.out_lbl.grid(row=1, column=0, sticky="e")

        ttk.Entry(self.outf, textvariable=self.out_path, width=40, style="VD3D.TEntry")\
           .grid(row=1, column=1, sticky="we", padx=6)

        self.out_browse = ttk.Button(self.outf, text="Browse…", command=self._browse_out, style="VD3D.TButton")
        self.out_browse.grid(row=1, column=2, padx=4)

        self.outf.grid_columnconfigure(1, weight=1)

        # --- Final Size (optional) ---
        self.sizef = ttk.LabelFrame(left, text="Final Size (optional)", style="VD3D.TLabelframe")
        self.sizef.pack(fill="x", padx=6, pady=6)

        self.lbl_w = ttk.Label(self.sizef, text="Width:", style="VD3D.TLabel")
        self.lbl_w.grid(row=0, column=0, sticky="e")

        ttk.Entry(self.sizef, textvariable=self.w_var, width=8, style="VD3D.TEntry")\
           .grid(row=0, column=1, sticky="w", padx=6)

        self.lbl_h = ttk.Label(self.sizef, text="Height:", style="VD3D.TLabel")
        self.lbl_h.grid(row=0, column=2, sticky="e")

        ttk.Entry(self.sizef, textvariable=self.h_var, width=8, style="VD3D.TEntry")\
           .grid(row=0, column=3, sticky="w", padx=6)

        self.lbl_keep = ttk.Label(self.sizef, text="(Leave blank to keep source)", style="VD3D.TLabel")
        self.lbl_keep.grid(row=0, column=4, sticky="w", padx=12)


        # Blend Parameters
        self.parms = ttk.LabelFrame(left, text="Blend Parameters (preview live)", style="VD3D.TLabelframe")
        self.parms.pack(fill="x", padx=6, pady=6)

        self.lbl_white, self.scale_white, _ = self._add_slider(self.parms, "White Strength",        0.0, 2.0, self.white_strength,  0)
        self.lbl_blur,  self.scale_blur,  _ = self._add_slider(self.parms, "Feather Blur (kernel)", 1,   99,  self.blur_k,          1)
        self.lbl_clahe, self.scale_clahe, _ = self._add_slider(self.parms, "CLAHE Clip Limit",      0.5, 4.0, self.clip_limit,      2)
        self.lbl_tiles, self.scale_tiles, _ = self._add_slider(self.parms, "CLAHE Tile Grid",       2,   32,  self.tile_grid,       3)
        self.lbl_bfd,   self.scale_bfd,   _ = self._add_slider(self.parms, "Bilateral d",           1,   25,  self.bf_d,            4)
        self.lbl_sigmaC,self.scale_sigmaC,_ = self._add_slider(self.parms, "Bilateral sigmaColor",  1,   200, self.bf_sigmaColor,   5)
        self.lbl_sigmaS,self.scale_sigmaS,_ = self._add_slider(self.parms, "Bilateral sigmaSpace",  1,   200, self.bf_sigmaSpace,   6)

        self.parms.grid_columnconfigure(1, weight=1)

        # Scrubber
        self.scrub = ttk.LabelFrame(left, text="Preview Frame", style="VD3D.TLabelframe")
        self.scrub.pack(fill="x", padx=6, pady=6)
        self._idx_scale = ttk.Scale(
            self.scrub, from_=0, to=0, orient="horizontal",
            command=lambda _=None: self._schedule_preview(50),
            variable=self.preview_index, style="VD3D.Horizontal.TScale"
        )
        self._idx_scale.grid(row=0, column=0, sticky="we", padx=6, pady=4)
        ttk.Label(self.scrub, textvariable=self.preview_index, width=6, style="VD3D.TLabel")\
            .grid(row=0, column=1, sticky="e", padx=6)
        btns = ttk.Frame(self.scrub, style="VD3D.TFrame"); btns.grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        self.prev_btn = ttk.Button(btns, text="⟨ Prev", style="VD3D.TButton", command=lambda: self._nudge_preview(-1))
        self.prev_btn.pack(side="left", padx=2)
        self.next_btn = ttk.Button(btns, text="Next ⟩", style="VD3D.TButton", command=lambda: self._nudge_preview(+1))
        self.next_btn .pack(side="left", padx=2)
        self.scrub.grid_columnconfigure(0, weight=1)

        # Buttons
        btns2 = ttk.Frame(left, style="VD3D.TFrame"); btns2.pack(fill="x", padx=6, pady=6)
        self.btn_preview = ttk.Button(btns2, text="Preview Now", style="VD3D.TButton", command=self._preview_now)
        self.btn_start   = ttk.Button(btns2, text="Start Batch", style="VD3D.TButton", command=self._start)
        self.btn_stop    = ttk.Button(btns2, text="Stop",        style="VD3D.TButton", command=self._stop, state="disabled")
        self.btn_preview.grid(row=0, column=0, padx=4)
        self.btn_start.grid(row=0, column=1, padx=4)
        self.btn_stop.grid(row=0, column=2, padx=4)

        # Progress + Log
        pf = ttk.Frame(self, style="VD3D.TFrame"); pf.pack(fill="x")
        self.prog = ttk.Progressbar(pf, mode="determinate", style="VD3D.Horizontal.TProgressbar"); self.prog.pack(fill="x")
        self.prog_lbl = ttk.Label(pf, text="Progress: 0/0", style="VD3D.TLabel"); self.prog_lbl.pack(anchor="w")

        lf = ttk.LabelFrame(self, text="Log", style="VD3D.TLabelframe"); lf.pack(fill="both", expand=True, padx=6, pady=6)
        # Text + optional scrollbar
        self.log = tk.Text(lf, height=10, wrap="word", relief="flat",
                           bg="#151515", fg="white", insertbackground="white", highlightthickness=0, bd=0)
        self.log.pack(fill="both", expand=True, side="left")
        sy = ttk.Scrollbar(lf, orient="vertical", style="VD3D.Vertical.TScrollbar", command=self.log.yview)
        self.log.configure(yscrollcommand=sy.set)
        sy.pack(side="right", fill="y")
        
        tooltip_refs["DB.V1Path"]   = CreateToolTip(self.v1_lbl,    lambda: t("Tooltip.DB.V1Path"))
        tooltip_refs["DB.V2Path"]   = CreateToolTip(self.v2_lbl,    lambda: t("Tooltip.DB.V2Path"))
        tooltip_refs["DB.OutPath"]  = CreateToolTip(self.outf,      lambda: t("Tooltip.DB.OutPath"))
        tooltip_refs["DB.Width"]    = CreateToolTip(self.lbl_w,     lambda: t("Tooltip.DB.Width"))
        tooltip_refs["DB.Height"]   = CreateToolTip(self.lbl_h,     lambda: t("Tooltip.DB.Height"))
        tooltip_refs["DB.KeepSrc"]  = CreateToolTip(self.lbl_keep,  lambda: t("Tooltip.DB.KeepSource"))
        tooltip_refs["DB.White"]    = CreateToolTip(self.lbl_white, lambda: t("Tooltip.DB.WhiteStrength"))
        tooltip_refs["DB.Blur"]     = CreateToolTip(self.lbl_blur,  lambda: t("Tooltip.DB.FeatherKernel"))
        tooltip_refs["DB.CLAHE"]    = CreateToolTip(self.lbl_clahe, lambda: t("Tooltip.DB.CLAHEClip"))
        tooltip_refs["DB.Tiles"]    = CreateToolTip(self.lbl_tiles, lambda: t("Tooltip.DB.CLAHETiles"))
        tooltip_refs["DB.BFd"]      = CreateToolTip(self.lbl_bfd,   lambda: t("Tooltip.DB.BilateralD"))
        tooltip_refs["DB.SigmaC"]   = CreateToolTip(self.lbl_sigmaC,lambda: t("Tooltip.DB.SigmaColor"))
        tooltip_refs["DB.SigmaS"]   = CreateToolTip(self.lbl_sigmaS,lambda: t("Tooltip.DB.SigmaSpace"))
        
        tooltip_refs["DB.PreviewFrame"]   = CreateToolTip(self.scrub,lambda: t("Tooltip.DB.PreviewFrame"))
        tooltip_refs["DB.BlendParms"]   = CreateToolTip(self.parms,lambda: t("Tooltip.DB.BlendParms"))
        tooltip_refs["DB.ModeTitle"]   = CreateToolTip(self.mode_frame,lambda: t("DB.ModeTitle"))
        tooltip_refs["DB.TitleLbl"]   = CreateToolTip(self.title_lbl,lambda: t("DB.TitleLbl"))
        

    def refresh_labels(self):
        # tiny helper
        def _cfg(w, **kw):
            try: w.config(**kw)
            except Exception: pass

        _cfg(self.title_lbl,    text=t("Depth Blender"))
        _cfg(self.paths,    text=t("Inputs"))
        _cfg(self.sizef,    text=t("Final Size (optional)"))
        _cfg(self.parms,    text=t("Blend Parameters (preview live)"))
        _cfg(self.preview_lbl,    text=t("Preview (scrubbable):"))
        _cfg(self.scrub,    text=t("Preview Frame"))
        _cfg(self.btn_preview,    text=t("Preview Now"))
        _cfg(self.btn_start,    text=t("Start Batch"))
        _cfg(self.btn_stop,    text=t("Stop"))
        _cfg(self.prev_btn,    text=t("⟨ Prev"))
        _cfg(self.next_btn,    text=t("Next ⟩"))
        _cfg(self.btn_stop,    text=t("Stop"))
        
        
        
        _cfg(self.v1_lbl,    text=t("V1 path:"))
        _cfg(self.v2_lbl,    text=t("V2 path:"))
        _cfg(self.outf,      text=t("Output"))
        _cfg(self.out_lbl,   text=t("Output path/file:"))
        _cfg(self.chk_over,  text=t("Overwrite V2 (frames mode only)"))

        _cfg(self.lbl_w,     text=t("Width:"))
        _cfg(self.lbl_h,     text=t("Height:"))
        _cfg(self.lbl_keep,  text=t("(Leave blank to keep source)"))

        _cfg(self.mode_frame, text=t("Mode"))
        _cfg(self.rb_frames,  text=t("Folders (frames)"))
        _cfg(self.rb_videos,  text=t("Videos"))

        _cfg(self.gpu_btn,    text=t("Use GPU (PyTorch CUDA)"))

        # Parameter section title lives on the labelframe itself:
        # Find the LabelFrame by keeping a handle if you need to localize its 'text' too.
        # Example (if you kept it): _cfg(self.parms_frame, text=t("Blend Parameters (preview live)"))

        _cfg(self.lbl_white,  text=t("White Strength"))
        _cfg(self.lbl_blur,   text=t("Feather Blur (kernel)"))
        _cfg(self.lbl_clahe,  text=t("CLAHE Clip Limit"))
        _cfg(self.lbl_tiles,  text=t("CLAHE Tile Grid"))
        _cfg(self.lbl_bfd,    text=t("Bilateral d"))
        _cfg(self.lbl_sigmaC, text=t("Bilateral sigmaColor"))
        _cfg(self.lbl_sigmaS, text=t("Bilateral sigmaSpace"))


    def _add_slider(self, parent, label, mn, mx, var, row):
        lbl = ttk.Label(parent, text=label, style="VD3D.TLabel")
        lbl.grid(row=row, column=0, sticky="w", padx=6)
        s = ttk.Scale(parent, from_=mn, to=mx, orient="horizontal",
                      command=lambda _=None: self._schedule_preview(120), variable=var,
                      style="VD3D.Horizontal.TScale")
        s.grid(row=row, column=1, sticky="we", padx=6)
        parent.grid_columnconfigure(1, weight=1)
        val = ttk.Label(parent, textvariable=var, style="VD3D.TLabel")
        val.grid(row=row, column=2, sticky="e", padx=6)
        return lbl, s, val

    def _build_progress_and_log(self, parent):
        pf = ttk.Frame(parent); pf.pack(fill="x")
        self.prog = ttk.Progressbar(pf, mode="determinate"); self.prog.pack(fill="x")
        self.prog_lbl = ttk.Label(pf, text="Progress: 0/0"); self.prog_lbl.pack(anchor="w")
        lf = ttk.LabelFrame(parent, text="Log"); lf.pack(fill="both", expand=True, padx=6, pady=6)
        self.log = tk.Text(lf, height=10, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

    # -------- browsers / toggles / helpers ----------
    def _browse_v1(self):
        if self.mode.get() == "frames":
            p = filedialog.askdirectory(title="Select V1 frames folder")
        else:
            p = filedialog.askopenfilename(title="Select V1 video",
                                           filetypes=[("Video", "*.mp4;*.mov;*.mkv;*.avi"), ("All", "*.*")])
        if p:
            self.v1_path.set(p)
            self._infer_mode_from_paths()
            self._update_preview_bounds()
            self._schedule_preview(0)

    def _browse_v2(self):
        if self.mode.get() == "frames":
            p = filedialog.askdirectory(title="Select V2 frames folder (base)")
        else:
            p = filedialog.askopenfilename(title="Select V2 video (base)",
                                           filetypes=[("Video", "*.mp4;*.mov;*.mkv;*.avi"), ("All", "*.*")])
        if p:
            self.v2_path.set(p)
            self._infer_mode_from_paths()
            self._update_preview_bounds()
            self._schedule_preview(0)

    def _browse_out(self):
        if self.mode.get() == "frames":
            if not self.overwrite_v2.get():
                p = filedialog.askdirectory(title="Select output frames folder")
                if p: self.out_path.set(p)
            else:
                messagebox.showinfo("Output", "Overwrite V2 is on. No separate output folder needed.")
        else:
            p = filedialog.asksaveasfilename(title="Save output video as", defaultextension=".mp4",
                                             filetypes=[("MP4", "*.mp4"), ("All", "*.*")])
            if p: self.out_path.set(p)
            
    def _path_is_file(self, p: str) -> bool:
        return bool(p) and os.path.isfile(p)

    def _path_is_dir(self, p: str) -> bool:
        return bool(p) and os.path.isdir(p)

    def _infer_mode_from_paths(self):
        v1p = self.v1_path.get().strip()
        v2p = self.v2_path.get().strip()
        if any(self._path_is_file(p) for p in (v1p, v2p)):
            if self.mode.get() != "videos":
                self.mode.set("videos")
                self._toggle_mode()
        elif all(self._path_is_dir(p) for p in (v1p, v2p)) and v1p and v2p:
            if self.mode.get() != "frames":
                self.mode.set("frames")
                self._toggle_mode()

    def _toggle_mode(self):
        if self.mode.get() == "videos":
            self.chk_over.state(["disabled"])
        else:
            self.chk_over.state(["!disabled"])
        self._toggle_out_controls()
        self._update_preview_bounds()
        self._schedule_preview(0)

    def _toggle_out_controls(self):
        pass

    # ---------- scrubber ----------
    def _nudge_preview(self, delta):
        cur = int(self.preview_index.get())
        mx  = int(self.preview_max.get())
        new = max(0, min(mx, cur + int(delta)))
        if new != cur:
            self.preview_index.set(new)
            self._schedule_preview(50)

    def _update_preview_bounds(self):
        mx = 0
        if self.mode.get() == "frames":
            v1p, v2p = self.v1_path.get().strip(), self.v2_path.get().strip()
            if not (self._path_is_dir(v1p) and self._path_is_dir(v2p)):
                self.preview_max.set(0)
                if self._idx_scale is not None:
                    self._idx_scale.configure(to=0)
                return
            n1 = len([f for f in os.listdir(v1p) if f.lower().endswith(".png")])
            n2 = len([f for f in os.listdir(v2p) if f.lower().endswith(".png")])
            mx = max(0, min(n1, n2) - 1)
        else:
            v2p = self.v2_path.get().strip()
            if self._path_is_file(v2p):
                cap = cv2.VideoCapture(v2p)
                if cap.isOpened():
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
                    mx = max(0, total - 1)
                cap.release()
        self.preview_max.set(mx)
        if self._idx_scale is not None:
            self._idx_scale.configure(to=mx)
        self.preview_index.set(min(int(self.preview_index.get()), mx))


    # ---------- start/stop ----------
    def _start(self):
        mode = self.mode.get()
        v1, v2 = self.v1_path.get().strip(), self.v2_path.get().strip()
        if not v1 or not v2:
            messagebox.showerror("Missing paths", "Please select both V1 and V2 paths.")
            return

        out_w = int(self.w_var.get()) if self.w_var.get().strip().isdigit() else None
        out_h = int(self.h_var.get()) if self.h_var.get().strip().isdigit() else None

        params = {
            "white_strength": float(self.white_strength.get()),
            "blur_k": int(self.blur_k.get()),
            "clip_limit": float(self.clip_limit.get()),
            "tile_grid": int(self.tile_grid.get()),
            "bf_d": int(self.bf_d.get()),
            "bf_sigmaColor": int(self.bf_sigmaColor.get()),
            "bf_sigmaSpace": int(self.bf_sigmaSpace.get()),
        }

        self.stop_evt.clear()
        self.btn_start.config(state="disabled"); self.btn_stop.config(state="normal")
        self._set_prog(0, 0); self._log("Starting...")

        if mode == "frames":
            ow = self.overwrite_v2.get()
            out_mode = "overwrite_v2" if ow else "output_folder"
            out_path = self.out_path.get().strip()
            if not ow and not out_path:
                messagebox.showerror("Missing output folder", "Pick an output folder or enable Overwrite V2.")
                self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled"); return
            self.worker = FramesWorker(
                v1, v2, out_mode, out_path, out_w, out_h,
                self.qlog, self.qprog, self.stop_evt,
                use_gpu=self.use_gpu.get(), params=params
            )
        else:
            out_file = self.out_path.get().strip()
            if not out_file:
                messagebox.showerror("Missing output file", "Choose where to save the output video.")
                self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled"); return
            self.worker = VideosWorker(
                v1, v2, out_file, out_w, out_h,
                self.qlog, self.qprog, self.stop_evt,
                use_gpu=self.use_gpu.get(), params=params
            )

        self.worker.start()

    def _stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_evt.set()
            self._log("Stopping requested...")

    # ---------- preview ----------
    def _schedule_preview(self, delay_ms=200):
        if self._preview_after is not None:
            try: self.after_cancel(self._preview_after)
            except Exception: pass
        self._preview_after = self.after(int(max(0, delay_ms)), self._preview_now)

    def _preview_now(self):
        with self._preview_lock:
            if self._preview_thread and self._preview_thread.is_alive():
                return
            self._preview_thread = threading.Thread(target=self._compute_preview, daemon=True)
            self._preview_thread.start()

    def _compute_preview(self):
        try:
            self._infer_mode_from_paths()
            v1p, v2p = self.v1_path.get().strip(), self.v2_path.get().strip()
            if self.mode.get() == "frames" and (not self._path_is_dir(v1p) or not self._path_is_dir(v2p)):
                self._log("Preview: set to Frames mode but paths are files. Switching to Videos or pick folders.")
                return
            if self.mode.get() == "videos" and (not self._path_is_file(v1p) or not self._path_is_file(v2p)):
                self._log("Preview: set to Videos mode but paths are folders. Switching to Frames or pick files.")
                return
            if not v1p or not v2p:
                return
            idx = int(self.preview_index.get())
            if self.mode.get() == "frames":
                v1_files = sorted([f for f in os.listdir(v1p) if f.lower().endswith(".png")])
                v2_files = sorted([f for f in os.listdir(v2p) if f.lower().endswith(".png")])
                if not v1_files or not v2_files: return
                idx = max(0, min(idx, min(len(v1_files), len(v2_files)) - 1))
                v1 = cv2.imread(os.path.join(v1p, v1_files[idx]), cv2.IMREAD_GRAYSCALE)
                v2 = cv2.imread(os.path.join(v2p, v2_files[idx]), cv2.IMREAD_GRAYSCALE)
                if v1 is None or v2 is None: return
                if v1.shape != v2.shape:
                    v2 = cv2.resize(v2, (v1.shape[1], v1.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                cap1, cap2 = cv2.VideoCapture(v1p), cv2.VideoCapture(v2p)
                if not cap1.isOpened() or not cap2.isOpened():
                    if cap1: cap1.release()
                    if cap2: cap2.release()
                    return
                cap1.set(cv2.CAP_PROP_POS_FRAMES, idx)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok1, fr1 = cap1.read(); ok2, fr2 = cap2.read()
                cap1.release(); cap2.release()
                if not ok1 or not ok2: return
                v1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
                v2 = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY)
                if v1.shape != v2.shape:
                    v2 = cv2.resize(v2, (v1.shape[1], v1.shape[0]), interpolation=cv2.INTER_AREA)

            params = {
                "white_strength": float(self.white_strength.get()),
                "blur_k": int(self.blur_k.get()),
                "clip_limit": float(self.clip_limit.get()),
                "tile_grid": int(self.tile_grid.get()),
                "bf_d": int(self.bf_d.get()),
                "bf_sigmaColor": int(self.bf_sigmaColor.get()),
                "bf_sigmaSpace": int(self.bf_sigmaSpace.get()),
            }
            out = lighten_beta(
                v1, v2,
                clip_limit=params["clip_limit"],
                tile_grid=(params["tile_grid"], params["tile_grid"]),
                d=params["bf_d"],
                sC=params["bf_sigmaColor"],
                sS=params["bf_sigmaSpace"],
                blur_k=params["blur_k"],
                white_strength=params["white_strength"],
                use_gpu=self.use_gpu.get()
            )

            vis_v2 = cv2.cvtColor(v2, cv2.COLOR_GRAY2BGR)
            vis_out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            panel = np.hstack([_put_label(vis_v2, "V2 Base"),
                               _put_label(vis_out, f"Blended Preview (idx {idx})")])
            panel = _resize_max(panel, max_w=840, max_h=520)

            im = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(im)
            self._preview_imgtk = imgtk
            self.preview_canvas.after(0, lambda: self._blit_preview_image(imgtk))


        except Exception:
            pass
    def _draw_preview_placeholder(self):
        c = self.preview_canvas
        if not c: return
        c.delete("all")
        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
        # light checker just to signal "this is a viewport"
        tile = 20
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                if ((x//tile) + (y//tile)) % 2 == 0:
                    c.create_rectangle(x, y, x+tile, y+tile, fill="#1a1a1a", outline="")
        c.create_text(w//2, h//2, text="No preview", fill="#777", font=("Segoe UI", 11))

    def _redraw_preview(self):
        # re-center last image when canvas resizes
        if getattr(self, "_preview_imgtk", None) is None:
            self._draw_preview_placeholder()
            return
        self._blit_preview_image(self._preview_imgtk)

    def _blit_preview_image(self, imgtk):
        c = self.preview_canvas
        if not c: return
        c.delete("all")
        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
        self._preview_canvas_img = c.create_image(w//2, h//2, image=imgtk, anchor="center")

    # ---------- utils ----------
    def _set_prog(self, done, total):
        self.prog["maximum"] = max(total, 1)
        self.prog["value"] = done
        self.prog_lbl.config(text=f"Progress: {done}/{total}")

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _poll(self):
        try:
            while True:
                self._log(self.qlog.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                d, t = self.qprog.get_nowait()
                self._set_prog(d, t)
        except queue.Empty:
            pass
        if self.worker and not self.worker.is_alive():
            self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
        self.after(100, self._poll)
        
# Mount the Depth Blender UI inside the tab
depth_blender_ui = DepthBlenderTab(depthblend_content_frame)
depth_blender_ui.pack(fill="both", expand=True)
# --- Depth Content ---

local_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))
os.makedirs(local_model_dir, exist_ok=True)

def load_supported_models():
    models = {
        "  -- Select Model -- ": "  -- Select Model -- ",

        # Marigold
        "Marigold Depth v1.1 (Diffusers)": "diffusers:prs-eth/marigold-depth-v1-1",
        "Marigold Depth v1.0":              "prs-eth/marigold-depth-v1-0",

        # Distill-Any-Depth
        "Distill-Any-Depth Large (xingyang1)": "xingyang1/Distill-Any-Depth-Large-hf",
        "Distill-Any-Depth Small (xingyang1)": "xingyang1/Distill-Any-Depth-Small-hf",
        "Distill-Any-Depth Large (keetrap)":   "keetrap/Distill-Any-Depth-Large-hf",
        "Distill-Any-Depth Small (keetrap)":   "keetrap/Distill-Any-Depth-Small-hf",

        # Depth Anything v2
        "Depth Anything v2 Large":                 "depth-anything/Depth-Anything-V2-Large-hf",
        "Depth Anything v2 Base":                  "depth-anything/Depth-Anything-V2-Base-hf",
        "Depth Anything v2 Small":                 "depth-anything/Depth-Anything-V2-Small-hf",
        "Depth Anything v2 Metric Indoor (Large)": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "Depth Anything v2 Metric Outdoor (Large)":"depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "Depth Anything v2 Giant (safetensors)": "dav2:vitg_fp32",

        # Depth Anything v1
        "Depth Anything v1 Large":    "LiheYoung/depth-anything-large-hf",
        "Depth Anything v1 Base":     "LiheYoung/depth-anything-base-hf",
        "Depth Anything v1 Small":    "LiheYoung/depth-anything-small-hf",
        "Depth Anything v1 ViT-L/14": "LiheYoung/depth_anything_vitl14",
        
        # Prompt Depth
        "Prompt Depth Anything VITS Transparent": "depth-anything/prompt-depth-anything-vits-transparent-hf",
        

        # Other popular models
        "DepthPro (Apple)":            "apple/DepthPro-hf",
        "ZoeDepth (NYU+KITTI)":        "Intel/zoedepth-nyu-kitti",
        "MiDaS 3.0 (DPT-Hybrid)":      "Intel/dpt-hybrid-midas",
        "DPT Large (Intel)":           "Intel/dpt-large",
        "DPT Large (Manojb)":          "Manojb/dpt-large",
        "DPT BEiT Large 512":          "Intel/dpt-beit-large-512",
        "MiDaS v2 (Qualcomm)":         "qualcomm/Midas-V2",

        # Local ONNX wrapper
        "Video Depth Anything (ONNX)": "videodepthanything:VideoDepthAnything",
    }


    # ✅ auto-add local folders as “[Local] {folder}”
    for folder in os.listdir(local_model_dir):
        folder_path = os.path.join(local_model_dir, folder)
        if os.path.isdir(folder_path):
            if (os.path.exists(os.path.join(folder_path, "config.json")) or
                os.path.exists(os.path.join(folder_path, "model.onnx"))):
                models[f"[Local] {folder}"] = folder_path

    return models

supported_models = load_supported_models()

selected_model = tk.StringVar(root, value="-- Select Model --")
colormap_var = tk.StringVar(root, value="Default")
invert_var = tk.BooleanVar(root, value=False)
save_frames_var = tk.BooleanVar(value=False)
output_dir = tk.StringVar(value="")
# near other UI vars
use_fp16_var = tk.BooleanVar(value=CUDA_AVAILABLE)  # default ON only if CUDA present

INFERENCE_RESOLUTIONS = {
    "Original": None,

    # General square resolutions
    "256x256": (256, 256),
    "384x384": (384, 384),
    "448x448": (448, 448),
    "512x512 (VDA)": (512, 512),
    "576x576": (576, 576),
    "640x640": (640, 640),
    "704x704": (704, 704),
    "768x768": (768, 768),
    "832x832": (832, 832),
    "896x896": (896, 896),
    "960x960": (960, 960),
    "1024x1024": (1024, 1024),

    # ViT/DINOV2-safe resolutions (multiples of 14)
    "512x288":  (512, 288),
    "640x352":  (640, 352),
    "768x432":  (768, 432),
    "896x512":  (896, 512),
    "1024x576": (1024, 576),
    "1152x640": (1152, 640),
    "1280x720": (1280, 720),   # OK, both /32
    "1344x768": (1344, 768),
    "1536x864": (1536, 864),
    "1600x896": (1600, 896),
    "1792x1008":(1792, 1008),
    "1920x1088":(1920, 1088),  # NOTE: 1088 instead of 1080
    # Squares / general
    "256x256":  (256, 256),
    "384x384":  (384, 384),
    "512x512":  (512, 512),
    "640x640":  (640, 640),
    "768x768":  (768, 768),
    "896x896":  (896, 896),
    "1024x1024":(1024, 1024),

    # Widescreen & cinematic
    "512x256 (DC-Fastest)": (512, 256),
    "704x384 (DC-Balanced)": (704, 384),
    "910x518 (Depth Anything)": (910, 518),
    "960x540 (DC-Good Quality)": (960, 540),
    "1024x576 (DC-Max Quality)": (1024, 576),

    # Portrait / vertical or special use
    "912x912": (912, 912),
    "920x1080": (920, 1080),  # vertical

    # Experimental 16:9 upscales
    "1280x720 (720p HD)": (1280, 720),
    "1920x1080 (1080p HD)": (1920, 1080),
}



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
    width=30,
    style="VD3D.TEntry",
)
model_dropdown.pack(pady=5)

# Bind event that fires *before* selection
model_dropdown.bind("<Button-1>", lambda event: refresh_model_dropdown())

# Bind event for when user selects a model
model_dropdown.bind(
    "<<ComboboxSelected>>",
    lambda event: update_pipeline(selected_model, status_label, inference_res_var, offload_mode_dropdown, inference_steps_entry, use_fp16_var)

)


output_dir_label = tk.Label(
    sidebar, text=t("Output Dir: None"), bg="#1c1c1c", fg="white", wraplength=200
)
output_dir_label.pack(pady=5)

output_dir_button = ttk.Button(
    sidebar,
    text=t("Choose Directory"),
    command=lambda: choose_output_directory(output_dir_label, output_dir),
    width=20,
    style="VD3D.TButton"
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
    style="VD3D.TEntry",
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

# Label (styled)
inference_steps_label = ttk.Label(
    sidebar,
    text=t("Inference Steps:"),
    style="VD3D.TLabel"
)
inference_steps_label.pack(pady=5)

# Entry (styled)
inference_steps_entry = ttk.Entry(
    sidebar,
    width=22,
    style="VD3D.TEntry"
)
inference_steps_entry.insert(0, "5")  # Default value
inference_steps_entry.pack(pady=5)

def update_inference_steps(*args):
    try:
        steps = int(inference_steps_entry.get().strip())
        if steps <= 0:
            raise ValueError
        status_label.config(text=t(f"🔄 Inference Steps Updated: {steps}"))
    except ValueError:
        status_label.config(text=t("⚠️ Invalid step count. Using default (5)."))

inference_steps_entry.bind("<Return>", update_inference_steps)
inference_steps_entry.bind("<FocusOut>", update_inference_steps)

# Label (styled)
batch_size_label = ttk.Label(
    sidebar,
    text=t("Batch Size (Frames):"),
    style="VD3D.TLabel"
)
batch_size_label.pack(pady=5)

# Entry (styled)
batch_size_entry = ttk.Entry(
    sidebar,
    width=22,
    style="VD3D.TEntry"
)
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
inference_res_label = ttk.Label(
    sidebar,
    text=t("Inference Resolution:"),
    style="VD3D.TLabel"
)
inference_res_label.pack(pady=5)

# Variable for the dropdown
inference_res_var = tk.StringVar(value="Original")  # default value


inference_res_dropdown = ttk.Combobox(
    sidebar,
    textvariable=inference_res_var,  # bind the variable
    values=list(INFERENCE_RESOLUTIONS.keys()),  # show these options
    state="readonly",
    style="VD3D.TEntry"
)
inference_res_dropdown.set("Original")  # set default value
inference_res_dropdown.pack(pady=5)


offload_mode_label = ttk.Label(
    sidebar,
    text="CPU Offload Mode",
    style="VD3D.TLabel"
)
offload_mode_label.pack(pady=5)

offload_mode_dropdown = ttk.Combobox(
    sidebar,
    values=["none", "model", "vae", "unet", "sequential"],
    state="readonly",
    width=20,
    style="VD3D.TEntry"
)

offload_mode_dropdown.set("none")
offload_mode_dropdown.pack()

float_16_btn = tk.Checkbutton(
    sidebar,
    text="Use float16 (CUDA)",
    variable=use_fp16_var,
    bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
)
float_16_btn.pack()


progress_bar = ttk.Progressbar(
    sidebar,
    mode="determinate",
    style="VD3D.Horizontal.TProgressbar",
    length=180
)
progress_bar.pack(pady=10)
status_label = tk.Label(
    sidebar, text=t("Ready"), bg="#1c1c1c", fg="white", width=30, wraplength=200
)
status_label.pack(pady=5)

cancel_depth_button = tk.Button(
    sidebar, text=t("Cancel"), command=cancel_processing, bg="red", fg="white"
)
cancel_depth_button.pack(pady=5)

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
        invert_var,
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
        invert_var,
        inference_steps_entry,
        offload_mode_dropdown,
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



# ─── frametools_inner GUI Layout ──────────────────────────────────────────────



# 🎥 Scene Detection
scene_threshold_var = tk.DoubleVar(value=30.0)
scene_output_format = tk.StringVar(value="mkv")
scene_detect_label = tk.Label(
    frametools_inner,
    text="🎥 Scene Detection (PySceneDetect)",
    font=("Segoe UI", 12, "bold"), background="#1c1c1c",
    foreground="white"
)
scene_detect_label.pack(pady=(20, 5))

scene_detect_threshold = tk.Label(
    frametools_inner,
    text="Sensitivity Threshold (lower = more cuts):",
    background="#1c1c1c", foreground="white"
)
scene_detect_threshold.pack()

scene_detect_slider = ttk.Scale(
    frametools_inner,
    from_=10, to=80,
    variable=scene_threshold_var,
    style="VD3D.Horizontal.TScale",
    length=300
)
scene_detect_slider.pack(pady=5)

def run_scene_detect():
    global scene_output_format
    video_path = filedialog.askopenfilename(title="Select Video for Scene Detection", filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not video_path:
        return
    output_folder = filedialog.askdirectory(title="Select Output Folder for Scenes")
    if not output_folder:
        return
    threshold = scene_threshold_var.get()
    ext = scene_output_format.get().strip().lower()
    if ext not in ["mp4", "mov", "avi", "mkv"]:
        messagebox.showerror("Invalid Format", f"Unsupported output format: {ext}")
        return
    merged_status.config(text="⏳ Detecting scenes...")
    merged_progress.start()
    
    def scene_thread():
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        import os, subprocess

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video)

        scene_list = scene_manager.get_scene_list()
        fps = video.frame_rate

        for i, (start, end) in enumerate(scene_list):
            start_frame = start.get_frames()
            end_frame = end.get_frames()
            start_time = start_frame / fps
            duration = (end_frame - start_frame) / fps

            scene_filename = os.path.join(output_folder, f"scene_{i+1:03d}.{ext}")
            command = [
                "ffmpeg", "-y", "-hwaccel", "auto",
                "-i", video_path,
                "-ss", f"{start_time:.3f}", "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "128k",
                scene_filename
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        merged_progress.stop()
        merged_status.config(text=f"✅ Exported {len(scene_list)} scenes as videos.")
        messagebox.showinfo("Done", f"✅ Exported {len(scene_list)} scenes to:\n{output_folder}")

    threading.Thread(target=scene_thread, daemon=True).start()

detect_scenes_button = ttk.Button(
    frametools_inner,
    text="Detect Scenes & Extract",
    style="VD3D.TButton",
    command=run_scene_detect
)
detect_scenes_button.pack(pady=10)

# Input / Output Group
io_frame = tk.LabelFrame(frametools_inner, text=t("Input / Output"), bg="#1c1c1c", fg="white")
io_frame.pack(fill="x", padx=10, pady=4)

# Extract Button


extract_frames_button = ttk.Button(
    io_frame,
    style="VD3D.TButton",
    text=t("Extract Frames from Video"),
    command=lambda: select_video_and_generate_frames(
        ft3d_frames_folder.set,
        merged_progress,
        merged_status
    )
)
extract_frames_button.pack(pady=10)



frames_folder_label = ttk.Label(
    io_frame,
    text=t("Frames Folder:"),
    style="VD3D.TLabel"
)
frames_folder_label.pack(anchor="w", padx=10, pady=(6, 2))

frames_folder_entry = ttk.Entry(
    io_frame,
    textvariable=ft3d_frames_folder,
    width=50,
    style="VD3D.TEntry"
)
frames_folder_entry.pack(pady=2)

browse_button = ttk.Button(
    io_frame, text=t("Browse"),
    command=lambda: select_frames_folder(ft3d_frames_folder),
    style="VD3D.TButton"
)
browse_button.pack(pady=2)

output_video_file_label = ttk.Label(
    io_frame,
    text=t("Output Video File:"),
    style="VD3D.TLabel"
    
)
output_video_file_label.pack(anchor="w", padx=10, pady=(6, 2))

output_video_ft = ttk.Entry(
    io_frame,
    textvariable=ft3d_output_file,
    width=50,
    style="VD3D.TEntry"

)
output_video_ft.pack(pady=2)

save_as_button = ttk.Button(
    io_frame,
    text=t("Save As"),
    command=lambda: select_output_file(ft3d_output_file),
    style="VD3D.TButton"
)
save_as_button.pack(pady=2)

# Processing Options
proc_frame = ttk.LabelFrame(
    frametools_inner,
    text=t("Processing Options"),
    style="VD3D.TLabelframe"
)
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
out_frame = tk.LabelFrame(frametools_inner, text=t("Output Settings"), bg="#1c1c1c", fg="white")
out_frame.pack(fill="x", padx=10, pady=4)

res_box = tk.Frame(out_frame, bg="#1c1c1c")
res_box.pack(anchor="w", padx=10, pady=4)

resolution_label = tk.Label(res_box, text=t("Resolution (WxH):"), bg="#1c1c1c", fg="white")
resolution_label.pack(side="left")

resolution_width_entry = ttk.Entry(
    res_box,
    width=6,
    textvariable=ft3d_width,
    style="VD3D.TEntry"
)
resolution_width_entry.pack(side="left", padx=4)

tk.Label(res_box, text="x", bg="#1c1c1c", fg="white").pack(side="left")

resolution_height_entry = ttk.Entry(
    res_box,
    textvariable=ft3d_height,
    width=6,
    style="VD3D.TEntry"
)
resolution_height_entry.pack(side="left", padx=4)

# Helper function that returns label for tooltip use
def combo_row(parent, label_text, var, values):
    row = tk.Frame(parent, bg="#1c1c1c")
    row.pack(anchor="w", padx=10, pady=4, fill="x")
    label = tk.Label(row, text=label_text, bg="#1c1c1c", fg="white", width=22, anchor="w")
    label.pack(side="left")
    ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=20, style="VD3D.TEntry").pack(side="left")
    return label

# Output Setting Rows
original_fps_label = combo_row(out_frame, t("Original FPS:"), ft3d_fps, COMMON_FPS)
fps_multi_label = combo_row(out_frame, t("FPS Interpolation Multiplier:"), ft3d_fps_multiplier, FPS_MULTIPLIERS)
selected_ffmpeg_codec_frametools_label = combo_row(out_frame, t("FFmpeg Output Codec:"), ft3d_codec, list(FFMPEG_CODEC_MAP.keys()))

# ESRGAN Settings
esrgan_frame = tk.LabelFrame(frametools_inner, text=t("ESRGAN Settings"), bg="#1c1c1c", fg="white")
esrgan_frame.pack(fill="x", padx=10, pady=4)

ai_blend_select = combo_row(esrgan_frame, t("AI Blending:"), ft3d_blend_mode, ["OFF", "LOW", "MEDIUM", "HIGH"])
input_res_pct_label = combo_row(esrgan_frame, t("Input Resolution %:"), ft3d_input_res_pct, [25, 50, 75, 100])
model_select = combo_row(esrgan_frame, t("Model Selection:"), ft3d_selected_model, list(REAL_ESRGAN_MODELS.keys()))


# ▶️ Start Button
start_processing_button = tk.Button(
    frametools_inner,
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

threaded_processing_button = tk.Button(
    frametools_inner,
    text="⚡ Threaded RIFE + ESRGAN",
    bg="#007acc", fg="white", relief="flat",
    activebackground="#005f99", activeforeground="white",
    command=lambda: threading.Thread(
        target=start_threaded_pipeline,  # ✅ corrected name here
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
            merged_status
        ),
        daemon=True
    ).start()
)
threaded_processing_button.pack(pady=(0, 12))



# 📊 Progress Bar
merged_progress = ttk.Progressbar(
    frametools_inner,
    style="VD3D.Horizontal.TProgressbar",
    length=300,
    mode="determinate"
)
merged_progress.pack(pady=6)

merged_status = tk.Label(frametools_inner, text=t("Waiting to start..."), bg="#1c1c1c", fg="white")
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
fg_shift = tk.DoubleVar(value=8.0)
mg_shift = tk.DoubleVar(value=1.5)
bg_shift = tk.DoubleVar(value=-2.5)
sharpness_factor = tk.DoubleVar(value=0.2)
output_format = tk.StringVar(value="Full-SBS")
blur_ksize = tk.IntVar(value=1)
feather_strength = tk.DoubleVar(value=0.0)
preserve_hdr10_var = tk.BooleanVar(value=False)
selected_ffmpeg_codec = tk.StringVar(value="")
crf_value = tk.IntVar(value=23)
use_ffmpeg = tk.BooleanVar(value=False)
use_subject_tracking = tk.BooleanVar(value=True)
use_floating_window = tk.BooleanVar(value=True)
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
depth_pop_gamma       = tk.DoubleVar(value=0.85)
depth_pop_mid         = tk.DoubleVar(value=0.50)
depth_stretch_lo      = tk.DoubleVar(value=0.05)
depth_stretch_hi      = tk.DoubleVar(value=0.95)
fg_pop_multiplier     = tk.DoubleVar(value=1.20)
bg_push_multiplier    = tk.DoubleVar(value=1.10)
subject_lock_strength = tk.DoubleVar(value=1.00)
saturation = tk.DoubleVar(value=1.00)   # 0.00..2.00
contrast   = tk.DoubleVar(value=1.00)   # 0.00..2.00
brightness = tk.DoubleVar(value=0.00)   # -0.50..+0.50


# --- Clip range (optional) ---
clip_start_var = tk.StringVar(value="")   # e.g. "00:01:23.500" or "83.5"
clip_end_var   = tk.StringVar(value="")   # leave blank to go to end

# --- IPD UI state ---
ipd_enabled_var = tk.BooleanVar(value=True)   # toggle
ipd_factor_var  = tk.DoubleVar(value=1.00)    # 0.5..1.5 typical

stereo_out_var   = tk.StringVar(value="sbs")   # sbs | left | right | both
delete_fsbs_var  = tk.BooleanVar(value=False)  # delete full SBS after split
preserve_hdr10_var = tk.BooleanVar(value=False)


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
    "fg_shift": fg_shift,
    "mg_shift": mg_shift,
    "bg_shift": bg_shift,
    "sharpness_factor": sharpness_factor,
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
    "depth_pop_gamma": depth_pop_gamma,
    "depth_pop_mid": depth_pop_mid,
    "depth_stretch_lo": depth_stretch_lo,
    "depth_stretch_hi": depth_stretch_hi,
    "fg_pop_multiplier": fg_pop_multiplier,
    "bg_push_multiplier": bg_push_multiplier,
    "subject_lock_strength": subject_lock_strength,
    "saturation": saturation,
    "contrast": contrast,
    "brightness": brightness,
    # --- IPD controls ---
    "ipd_enabled_var": ipd_enabled_var,
    "ipd_factor_var": ipd_factor_var,

    # --- Clip window (start/end) ---
    "clip_start_var": clip_start_var,
    "clip_end_var": clip_end_var,
    "preserve_hdr10": preserve_hdr10_var,
    "stereo_out_var": stereo_out_var
    
    

}


def clear_clip():
    clip_start_var.set("")
    clip_end_var.set("")

# Layout frames

top_widgets_frame = tk.LabelFrame(
    right_col,
    text=t("Video Info"),
    bg="#1c1c1c",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    labelanchor="nw",
    padx=10,
    pady=10
)
top_widgets_frame.grid(row=0, column=0, sticky="new")


# Thumbnail
video_thumbnail_label = tk.Label(
    top_widgets_frame,
    text=t("No Thumbnail"),
    bg="#1c1c1c", fg="white"
)
video_thumbnail_label.grid(row=0, column=0, padx=10, pady=5)

video_specs_label = tk.Label(
    top_widgets_frame,
    text=t("Resolution: N/A\nFPS: N/A"),
    justify="left", bg="#1c1c1c", fg="white"
)
video_specs_label.grid(row=0, column=1, padx=10, pady=5)

aspect_preview_label = tk.Label(
    top_widgets_frame,
    text="",
    font=("Segoe UI", 8, "italic"),
    bg="#1c1c1c", fg="white")
aspect_preview_label.grid(row=1, column=0, sticky="w", padx=5)

# 🔁 Bind aspect ratio dropdown to preview label
selected_aspect_ratio.trace_add("write", update_aspect_preview)
update_aspect_preview()

depth_map_label = tk.Label(
    top_widgets_frame,
    text=t("Depth Map (3D): None"),
    bg="#1c1c1c", fg="white",
    justify="left", wraplength=200
)
depth_map_label.grid(row=1, column=1, padx=10, pady=5)

progress = ttk.Progressbar(
    top_widgets_frame,
    style="VD3D.Horizontal.TProgressbar",
    length=300,
    mode="determinate"
)
progress.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

progress_label = tk.Label(top_widgets_frame, text="0%", font=("Arial", 10), bg="#1c1c1c", fg="white")
progress_label.grid(row=1, column=2, padx=10, pady=5, sticky="ew")


preview_button = ttk.Button(
    top_widgets_frame,
    text=t("Open Preview"),
    command=lambda: handle_open_preview(),
    style="VD3D.TButton",
)

preview_button.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

# Audio Tool Button
audio_tool_button = ttk.Button(
    top_widgets_frame,
    text=t("Audio Tool"),
    command=launch_audio_gui,
    style="VD3D.TButton",
)
audio_tool_button.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

live_button = ttk.Button(
    top_widgets_frame,
    text=t("VD3D External Input 3D (WIP)"),
    command=launch_live_gui,
    style="VD3D.TButton",
)
live_button.grid(row=2, column=2, padx=10, pady=5, sticky="ew")


# Processing Options
options_frame = tk.LabelFrame(
    left_col,
    text=t("Pop & Subject Controls"),
    bg="#1c1c1c",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    labelanchor="nw",
    padx=10,
    pady=10
)
options_frame.grid(row=1, column=0, sticky="new", pady=(8, 0))

# Ensure uniform spacing
for i in range(4):
    options_frame.columnconfigure(i, weight=1)

# Row 0
pop_gamma_label = tk.Label(
    options_frame,
    text=t("Depth Pop Gamma"),
    bg="#1c1c1c", fg="white"
    )
pop_gamma_label.grid(row=0, column=0, sticky="w")

pop_gamma_scale = tk.Scale(
    options_frame,
    from_=0.70, to=1.20,
    resolution=0.01, orient=tk.HORIZONTAL,
    variable=depth_pop_gamma,
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
)
pop_gamma_scale.grid(row=0, column=1, sticky="ew")

pop_mid_label = ttk.Label(
    options_frame,
    text=t("Pop Mid (0..1)"),
    style="VD3D.TLabel"
)
pop_mid_label.grid(row=0, column=2, sticky="w")

pop_mid_entry = ttk.Entry(
    options_frame,
    width=8,
    style="VD3D.TEntry"
)
pop_mid_entry.insert(0, f"{depth_pop_mid.get():.2f}")
pop_mid_entry.grid(row=0, column=3, sticky="w")

# Row 1
stretch_lo_label = ttk.Label(
    options_frame,
    text=t("Stretch Lo"),
    style="VD3D.TLabel"
)
stretch_lo_label.grid(row=1, column=0, sticky="w")

stretch_lo_entry = ttk.Entry(
    options_frame,
    width=8,
    style="VD3D.TEntry"
)
stretch_lo_entry.insert(0, f"{depth_stretch_lo.get():.2f}")
stretch_lo_entry.grid(row=1, column=1, sticky="w")

stretch_hi_label = ttk.Label(
    options_frame,
    text=t("Stretch Hi"),
    style="VD3D.TLabel"
)
stretch_hi_label.grid(row=1, column=2, sticky="w")

stretch_hi_entry = ttk.Entry(
    options_frame,
    width=8,
    style="VD3D.TEntry"
)
stretch_hi_entry.insert(0, f"{depth_stretch_hi.get():.2f}")
stretch_hi_entry.grid(row=1, column=3, sticky="w")

# Row 2
fg_pop_label = ttk.Label(
    options_frame,
    text=t("FG Pop ×"),
    style="VD3D.TLabel"
)
fg_pop_label.grid(row=2, column=0, sticky="w")

fg_pop_scale = tk.Scale(
    options_frame,
    from_=1.00, to=1.60,
    resolution=0.01, orient=tk.HORIZONTAL,
    variable=fg_pop_multiplier,
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
)
fg_pop_scale.grid(row=2, column=1, sticky="ew")

bg_push_label = tk.Label(
    options_frame,
    text=t("BG Push ×"),
    bg="#1c1c1c", fg="white"
)
bg_push_label.grid(row=2, column=2, sticky="w")
bg_push_scale = tk.Scale(
    options_frame,
    from_=1.00, to=1.40,
    resolution=0.01, orient=tk.HORIZONTAL,
    variable=bg_push_multiplier, 
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
)
bg_push_scale.grid(row=2, column=3, sticky="ew")

# Row 3
subj_lock_label = tk.Label(
    options_frame,
    text=t("Subject Lock"),
    bg="#1c1c1c", fg="white"
)
subj_lock_label.grid(row=3, column=0, sticky="w")

subj_lock_scale = tk.Scale(
    options_frame,
    from_=0.00, to=2.00,
    resolution=0.05, orient=tk.HORIZONTAL,
    variable=subject_lock_strength,
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
)
subj_lock_scale.grid(row=3, column=1, sticky="ew")


def _commit_pop_entries():
    try:
        depth_pop_mid.set(float(pop_mid_entry.get()))
        lo = float(stretch_lo_entry.get())
        hi = float(stretch_hi_entry.get())
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
        if hi <= lo:
            messagebox.showwarning(t("Invalid Input"), t("Stretch Hi must be greater than Stretch Lo."))
            return
        depth_stretch_lo.set(lo)
        depth_stretch_hi.set(hi)
    except ValueError:
        messagebox.showwarning(t("Invalid Input"), t("Use numeric values for Mid/Lo/Hi (0..1)."))

apply_entries_btn = ttk.Button(
    options_frame,
    text=t("Apply Entries"),
    command=_commit_pop_entries,
    style="VD3D.TButton"
)
apply_entries_btn.grid(row=3, column=3, sticky="e")

# (optional) Enter-to-commit
pop_mid_entry.bind("<Return>", lambda _e: _commit_pop_entries())
stretch_lo_entry.bind("<Return>", lambda _e: _commit_pop_entries())
stretch_hi_entry.bind("<Return>", lambda _e: _commit_pop_entries())



# Row 4
fg_shift_label = tk.Label(
    options_frame,
    text=t("Foreground Shift"),
    bg="#1c1c1c",
    fg="white"
)
fg_shift_label.grid(row=4, column=0, sticky="w")

tk.Scale(
    options_frame,
    from_=0,
    to=30,
    resolution=0.5,
    orient=tk.HORIZONTAL,
    variable=fg_shift,
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
).grid(row=4, column=1, sticky="ew")

convergence_strength_label = tk.Label(
    options_frame,
    text=t("Convergence Strength"),
    bg="#1c1c1c", fg="white"
)
convergence_strength_label.grid(row=4, column=2, sticky="w")

tk.Scale(
    options_frame,
    from_=-0.05,
    to=0.05,
    resolution=0.001,
    orient=tk.HORIZONTAL,
    variable=convergence_strength,
    length=200, bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
).grid(row=4, column=3, sticky="ew")

# Row 5

mg_shift_label = tk.Label(
    options_frame,
    text=t("Midground Shift"),
    bg="#1c1c1c",
    fg="white"
)
mg_shift_label.grid(row=5, column=0, sticky="w")

tk.Scale(
    options_frame, 
    from_=-10, to=10,
    resolution=0.5,
    orient=tk.HORIZONTAL, variable=mg_shift,
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
).grid(row=5, column=1, sticky="ew")


sharpness_factor_label = tk.Label(
    options_frame,
    text=t("Sharpness Factor"),
    bg="#1c1c1c", fg="white"
)
sharpness_factor_label.grid(row=5, column=2, sticky="w")

tk.Scale(
    options_frame,
    from_=-1, to=1,
    resolution=0.1, orient=tk.HORIZONTAL, 
    variable=sharpness_factor, bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
).grid(row=5, column=3, sticky="ew")

#Row 6
bg_shift_label = tk.Label(
    options_frame,
    text=t("Background Shift"),
    bg="#1c1c1c", fg="white"
)
bg_shift_label.grid(row=6, column=0, sticky="w")

tk.Scale(
    options_frame,
    from_=-20, to=0, 
    resolution=0.5, orient=tk.HORIZONTAL,
    variable=bg_shift, bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
).grid(row=6, column=1, sticky="ew")
 
parallax_balance_label = tk.Label(
    options_frame,
    text=t("Parallax Balance"),
    bg="#1c1c1c", fg="white"
)
parallax_balance_label.grid(row=6, column=2, sticky="w")

tk.Scale(
    options_frame,
    from_=0.0,
    to=1.0,
    resolution=0.05,
    orient="horizontal",
    variable=parallax_balance,
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
    
).grid(row=6, column=3, sticky="ew")

#Row 7
zero_parallax_strength_label = tk.Label(
    options_frame,
    text=t("Zero Parallax Strength"),
    bg="#1c1c1c", fg="white"
)
zero_parallax_strength_label.grid(row=7, column=0, sticky="w")

tk.Scale(
    options_frame,
    from_=-0.05, to=0.05,
    resolution=0.001, orient=tk.HORIZONTAL,
    variable=zero_parallax_strength,
    length=200, bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
    
).grid(row=7, column=1, sticky="ew")

max_pixel_shift_label = tk.Label(
    options_frame,
    text=t("Max Pixel Shift (%)"),
    bg="#1c1c1c", fg="white"
)
max_pixel_shift_label.grid(row=7, column=2, sticky="w")

tk.Scale(
    options_frame,
    from_=0.005, to=0.10,
    resolution=0.005,
    orient=tk.HORIZONTAL,
    variable=max_pixel_shift,
    length=200, bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
).grid(row=7, column=3, sticky="ew")   

dof_strength_label = tk.Label(
    options_frame,
    text=t("DoF Strength"),
    bg="#1c1c1c", fg="white"
)
dof_strength_label.grid(row=8, column=0, sticky="w")

tk.Scale(
    options_frame,
    from_=0.0, to=5.0,
    resolution=0.1,
    orient=tk.HORIZONTAL,
    variable=dof_strength,
    length=200,
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
).grid(row=8, column=1, sticky="ew")

# --- Row 8 : IPD Scale (match existing tk widgets/theme) ---
ipd_label = tk.Label(
    options_frame,
    text=t("Stereo Scaling (IPD)"),
    bg="#1c1c1c", fg="white"
)
ipd_label.grid(row=9, column=0, sticky="w")

ipd_slider = tk.Scale(
    options_frame,
    from_=0.50, to=1.50,
    resolution=0.01,
    orient=tk.HORIZONTAL,
    variable=ipd_factor_var,
    length=200,
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
)
ipd_slider.grid(row=9, column=1, sticky="ew")

ipd_readout = tk.Label(
    options_frame,
    text="1.00x",
    bg="#1c1c1c", fg="white"
)
ipd_readout.grid(row=9, column=2, sticky="w")

def _ipd_update(*_):
    # Keep slider enabled so color/style matches other sliders
    ipd_readout.config(
        text=(f"{ipd_factor_var.get():.2f}x" if ipd_enabled_var.get() else "OFF")
    )

ipd_factor_var.trace_add("write", _ipd_update)
ipd_enabled_var.trace_add("write", _ipd_update)
_ipd_update()

preset_var = tk.StringVar(value="Select Preset")

preset_menu = ttk.Combobox(
    options_frame,
    textvariable=preset_var,
    values=["Balanced Depth", "IMAX Depth", "Pop-Out 3D"],
    state="readonly",          # prevent typing
    width=18                   # tweak to taste
)
preset_menu.grid(row=8, column=3, sticky="e", padx=10, pady=4)

def on_preset_selected(event=None):
    sel = preset_var.get()
    if sel != "Select Preset":
        apply_preset(sel)

preset_menu.bind("<<ComboboxSelected>>", on_preset_selected)

save_preset_button = ttk.Button(
    options_frame,
    text=t("Save Preset"),
    style="VD3D.TButton",
    command=prompt_and_save_preset
)
save_preset_button.grid(row=9, column=3, sticky="e")

pop_frame = tk.LabelFrame(
    left_col,
    text=t("Processing Options"),
    bg="#1c1c1c", fg="white",
    font=("Segoe UI", 10, "bold"),
    labelanchor="nw", padx=10, pady=10
)
pop_frame.grid(row=0, column=0, sticky="new") 

for i in range(4):
    pop_frame.columnconfigure(i, weight=0)


# Row 0
preserve_aspect_checkbox = tk.Checkbutton(
    pop_frame,
    text=t("Preserve Original Aspect Ratio"),
    bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=preserve_original_aspect
)
preserve_aspect_checkbox.grid(row=0, column=0, sticky="w", padx=5)

auto_crop_checkbox = tk.Checkbutton(
    pop_frame,
    text=t("Auto Crop Black Bars"),
    bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=auto_crop_black_bars,
    anchor="e",
    justify="left"
)
auto_crop_checkbox.grid(row=0, column=1, sticky="w", padx=5)

use_subject_tracking_checkbox = tk.Checkbutton(
    pop_frame,
    text=t("Stabilize Zero-Parallax (center-depth)"),
    bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=use_subject_tracking,
    anchor="w",
    justify="left"
)
use_subject_tracking_checkbox.grid(row=0, column=2, sticky="w", padx=5)

# Row 1

skip_blank_frames_checkbox = tk.Checkbutton(
    pop_frame,
    text=t("Skip Blank/White Frames"), bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=skip_blank_frames,
    anchor="w",
    justify="left"
)
skip_blank_frames_checkbox.grid(row=1, column=0, sticky="w", padx=5)


enable_edge_checkbox = tk.Checkbutton(
    pop_frame,
    text=t("Enable Edge Masking"),
    bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=enable_edge_masking,
    anchor="w",
    justify="left"
)
enable_edge_checkbox.grid(row=1, column=1, sticky="w", padx=5)

enable_feathering_checkbox = tk.Checkbutton(
    pop_frame,
    text=t("Enable Feathering"),
    bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=enable_feathering,
    anchor="w",
    justify="left"
)
enable_feathering_checkbox.grid(row=1, column=2, sticky="w", padx=5)


enable_dynamic_convergence_checkbox = tk.Checkbutton(
    pop_frame,
    text=t("Enable Dynamic Convergence"),
    bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=enable_dynamic_convergence,
    anchor="w",
    justify="left"
)
enable_dynamic_convergence_checkbox.grid(row=3, column=0, sticky="w", padx=5)

ipd_toggle = tk.Checkbutton(
    pop_frame,
    text=t("Enable Stereo Scaling (IPD)"),
    variable=ipd_enabled_var,
    bg="#1c1c1c", fg="white",
    selectcolor="#2b2b2b",
    activebackground="#1c1c1c",
    activeforeground="white",
    highlightthickness=0
)
ipd_toggle.grid(row=3, column=1, sticky="w", padx=5)

use_dfw_checkbox = tk.Checkbutton(
    pop_frame,
    text=t("Enable Floating Window (DFW)"),
    bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=use_floating_window,
    anchor="e",
    justify="left"
)
use_dfw_checkbox.grid(row=3, column=2, sticky="w", padx=5)


color_frame = tk.LabelFrame(
    left_col,
    text=t("Color Grading"),
    bg="#1c1c1c", fg="white",
    font=("Segoe UI", 10, "bold"),
    labelanchor="nw", padx=10, pady=10
)
color_frame.grid(row=2, column=0, sticky="new", pady=(8, 0))

for i in range(4):
    color_frame.columnconfigure(i, weight=1)

# Saturation
saturation_label = tk.Label(
    color_frame, text=t("Saturation"),
    bg="#1c1c1c", fg="white")
saturation_label.grid(row=0, column=0, sticky="w")

tk.Scale(
    color_frame,
    from_=0.0, to=2.0,
    resolution=0.05, orient=tk.HORIZONTAL,
    variable=saturation,
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
).grid(row=0, column=1, sticky="ew")

# Contrast
contrast_label = tk.Label(
    color_frame,text=t("Contrast"),
    bg="#1c1c1c", fg="white")
contrast_label.grid(row=0, column=2, sticky="w")

tk.Scale(
    color_frame,
    from_=0.0, to=2.0,
    resolution=0.05, orient=tk.HORIZONTAL,
    variable=contrast,
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
).grid(row=0, column=3, sticky="ew")

# Brightness
brightness_label = tk.Label(
    color_frame, text=t("Brightness"),
    bg="#1c1c1c", fg="white")
brightness_label.grid(row=1, column=0, sticky="w")

tk.Scale(
    color_frame,
    from_=-0.5, to=0.5,
    resolution=0.01, orient=tk.HORIZONTAL,
    variable=brightness,
    bg="#1c1c1c", fg="white",
    cursor="sb_h_double_arrow"
).grid(row=1, column=1, sticky="ew")

# Optional reset
def _reset_color_grade():
    saturation.set(1.00); contrast.set(1.00); brightness.set(0.00)
color_reset_button = tk.Button(
    color_frame,
    text=t("Reset"),
    cursor="hand2",
    command=_reset_color_grade,
          bg="#2c2c2c",
          fg="white",
          activebackground="#444444",
          activeforeground="white"
)
color_reset_button.grid(row=1, column=3, sticky="e")

# 🔲 Encoding Settings Group
encoding_frame = tk.LabelFrame(
    right_col,
    text=t("Encoding Settings"),
    bg="#1c1c1c",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    labelanchor="nw",
    padx=10,
    pady=10
)
encoding_frame.grid(row=2, column=0, sticky="new", pady=(8, 0))

# Make columns evenly resize & give a minimum so controls don't squash
for i in range(6):
    encoding_frame.columnconfigure(i, weight=1, minsize=110)

# ───────── Row 0: Stereo output + Delete SBS ─────────
StereoOutput_label = tk.Label(
    encoding_frame,
    text="Left/Right Output",
    bg="#1c1c1c",
    fg="white"
)
StereoOutput_label.grid(row=0, column=0, sticky="w", padx=6, pady=4)

tk.OptionMenu(
    encoding_frame,
    stereo_out_var,
    "sbs", "left", "right", "both"
).grid(row=0, column=1, sticky="ew", padx=6, pady=4)

use_ffmpeg_checkbox = tk.Checkbutton(
    encoding_frame,
    text=t("Use FFmpeg Renderer"),
    bg="#1c1c1c", fg="white", selectcolor="#2b2b2b",
    variable=use_ffmpeg,
    anchor="w",
    justify="left"
)
use_ffmpeg_checkbox.grid(row=0, column=2, sticky="w", padx=5)


DeleteSBS_label = tk.Checkbutton(
    encoding_frame,
    text="Delete SBS after",
    variable=delete_fsbs_var,
    bg="#1c1c1c",
    fg="white",
    activebackground="#1c1c1c",
    selectcolor="#1c1c1c"
)
DeleteSBS_label.grid(row=0, column=3, columnspan=2, sticky="w", padx=5)

hdr_checkbox = tk.Checkbutton(
    encoding_frame,
    text="Preserve HDR10",
    variable=preserve_hdr10_var,
    onvalue=True,
    offvalue=False,
    bg="#1c1c1c",
    fg="white",
    activebackground="#1c1c1c",
    selectcolor="#1c1c1c",
    anchor="w",
    justify="left"
)
hdr_checkbox.grid(row=0, column=4, columnspan=2, sticky="w", padx=5)


# ───────── Row 1: Aspect • FFmpeg Codec • Codec ─────────
selected_aspect_ratio_label = tk.Label(
    encoding_frame,
    text=t("Aspect Ratio:"),
    bg="#1c1c1c",
    fg="white"
)
selected_aspect_ratio_label.grid(row=1, column=0, sticky="w", padx=6, pady=4)

tk.OptionMenu(
    encoding_frame,
    selected_aspect_ratio,
    *aspect_ratios.keys()
).grid(row=1, column=1, sticky="ew", padx=6, pady=4)

selected_ffmpeg_codec_label = tk.Label(
    encoding_frame,
    text=t("FFmpeg Codec:"),
    bg="#1c1c1c",
    fg="white"
)
selected_ffmpeg_codec_label.grid(row=1, column=2, sticky="w", padx=6, pady=4)

tk.OptionMenu(
    encoding_frame,
    selected_ffmpeg_codec,
    *FFMPEG_CODEC_MAP.keys()
).grid(row=1, column=3, sticky="ew", padx=6, pady=4)

selected_codec_label = tk.Label(
    encoding_frame,
    text=t("Codec:"),
    bg="#1c1c1c",
    fg="white"
)
selected_codec_label.grid(row=1, column=4, sticky="w", padx=6, pady=4)

tk.OptionMenu(
    encoding_frame,
    selected_codec,
    *codec_options
).grid(row=1, column=5, sticky="ew", padx=6, pady=4)

# ───────── Row 2: CRF • NVENC CQ ─────────
crf_value_label = tk.Label(
    encoding_frame,
    text=t("CRF"),
    bg="#1c1c1c",
    fg="white"
)
crf_value_label.grid(row=2, column=0, sticky="w", padx=6, pady=6)

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
    troughcolor="#444",
    highlightthickness=0,
    bd=0
).grid(row=2, column=1, columnspan=2, sticky="ew", padx=6, pady=6)

nvenc_cq_value_label = tk.Label(
    encoding_frame,
    text=t("NVENC CQ"),
    bg="#1c1c1c",
    fg="white"
)
nvenc_cq_value_label.grid(row=2, column=3, sticky="w", padx=6, pady=6)

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
    troughcolor="#444",
    highlightthickness=0,
    bd=0
).grid(row=2, column=4, columnspan=2, sticky="ew", padx=6, pady=6)


# --- Clip Range UI ---
clip_frame = tk.LabelFrame(
    right_col,
    text="Clip Range (optional)",
    bg="#1c1c1c",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    labelanchor="nw",
    padx=10,
    pady=10
)
clip_frame.grid(row=4, column=0, padx=8, pady=8, sticky="we")  # adjust placement


def _time_validate(s: str) -> bool:
    """
    Allow digits, colon, dot, spaces so users can type 'HH:MM:SS(.ms)', 'MM:SS(.ms)', or 'SS(.ms)'.
    Actual parsing is done by parse_timecode(); this just keeps the entry clean-ish.
    """
    return bool(re.match(r'^[0-9:\.\s]*$', s))

vcmd = (clip_frame.register(_time_validate), "%P")

tk.Label(clip_frame, text="Start (HH:MM:SS[.ms] or seconds):").grid(row=0, column=0, sticky="w", padx=6, pady=4)
start_entry = tk.Entry(clip_frame, textvariable=clip_start_var, width=18, validate="key", validatecommand=vcmd)
start_entry.grid(row=0, column=1, sticky="w", padx=6, pady=4)

tk.Label(clip_frame, text="End (HH:MM:SS[.ms] or seconds):").grid(row=1, column=0, sticky="w", padx=6, pady=4)
end_entry = tk.Entry(clip_frame, textvariable=clip_end_var, width=18, validate="key", validatecommand=vcmd)
end_entry.grid(row=1, column=1, sticky="w", padx=6, pady=4)

btns = tk.Frame(clip_frame)
btns.grid(row=0, column=2, rowspan=2, padx=6, pady=4, sticky="e")
tk.Button(btns, text="Clear", command=clear_clip).grid(row=0, column=0, padx=4)

# ── INPUT SOURCES (own frame) ──────────────────────────────────────────────
inputs_frame = tk.LabelFrame(
    right_col,
    text=t("Input Sources"),
    bg="#1c1c1c",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    labelanchor="nw",
    padx=10, pady=10
)

inputs_frame.grid(row=1, column=0, sticky="new", pady=(8, 0))

for i in range(6):
    inputs_frame.columnconfigure(i, weight=1)

# --- Mode Toggle (inside inputs_frame) ---
mode = tk.StringVar(value="Single")
mode_label = tk.Label(
    inputs_frame,
    text="Mode:",
    bg="#1e1e1e",
    fg="white"
)
mode_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)

ttk.Combobox(
    inputs_frame,
    textvariable=mode,
    style="VD3D.TEntry",
    state="readonly",
    values=["Single", "Batch"])\
.grid(row=0, column=1, pady=5, padx=5, sticky="w")

# --- Frame Containers (children of inputs_frame) ---
single_frame = tk.Frame(inputs_frame, bg="#1e1e1e")
batch_frame  = tk.Frame(inputs_frame, bg="#1e1e1e")
single_frame.grid(row=1, column=0, columnspan=6, sticky="ew")
batch_frame.grid(row=1, column=0, columnspan=6, sticky="ew")
batch_frame.grid_remove()

# --- Single Input Fields ---
select_input_video_button = ttk.Button(
    single_frame, text=t("Select Input Video"),
    command=lambda: select_input_video(
        input_video_path, video_thumbnail_label, video_specs_label,
        update_aspect_preview, original_video_width, original_video_height
    ),
    style="VD3D.TButton"
)
select_input_video_button.grid(row=0, column=0, pady=5, sticky="ew")

ttk.Entry(
    single_frame,
    textvariable=input_video_path,
    style="VD3D.TEntry",
    width=50,

).grid(row=0, column=1, pady=5, padx=5, sticky="ew")

select_depth_map_button = ttk.Button(
    single_frame, text=t("Select Depth Map"),
    command=lambda: select_depth_map(selected_depth_map, depth_map_label),
    style="VD3D.TButton",
)
select_depth_map_button.grid(row=1, column=0, pady=5, sticky="ew")

ttk.Entry(
    single_frame,
    textvariable=selected_depth_map,
    width=50,
    style="VD3D.TEntry",
    
).grid(row=1, column=1, pady=5, padx=5, sticky="ew")

select_output_video_button = ttk.Button(
    single_frame, text=t("Select Output Video"),
    command=lambda: select_output_video(output_sbs_video_path),
    style="VD3D.TButton",
)
select_output_video_button.grid(row=2, column=0, pady=5, sticky="ew")

ttk.Entry(
    single_frame,
    textvariable=output_sbs_video_path,
    width=50,
    style="VD3D.TEntry",

).grid(row=2, column=1, pady=5, padx=5, sticky="ew")

# --- Batch Input Fields ---
def add_to_listbox(listbox, filetypes):
    files = filedialog.askopenfilenames(filetypes=filetypes)
    for f in files:
        listbox.insert(tk.END, f)

input_video_listbox = tk.Listbox(batch_frame, selectmode=tk.SINGLE, height=5, bg="#1e1e1e", fg="white")
input_video_listbox.grid(row=0, column=1, pady=5, padx=5, sticky="ew")

batch_video_button = ttk.Button(
    batch_frame, text="+ Add Video",
    command=lambda: add_to_listbox(input_video_listbox, filetypes=[("Video files", "*.mp4 *.mkv *.mov")]),
    style="VD3D.TButton"
)
batch_video_button.grid(row=0, column=0, pady=5, sticky="ew")

depth_map_listbox = tk.Listbox(batch_frame, selectmode=tk.SINGLE, height=5, bg="#1e1e1e", fg="white")
depth_map_listbox.grid(row=1, column=1, pady=5, padx=5, sticky="ew")

batch_depth_button = ttk.Button(
    batch_frame, text="+ Add Depth Map",
    command=lambda: add_to_listbox(depth_map_listbox,
    filetypes=[("Video files", "*.mp4 *.mkv *.mov")]),
    style="VD3D.TButton"

)
batch_depth_button.grid(row=1, column=0, pady=5, sticky="ew")

# --- Toggle Mode Visibility ---
def toggle_mode(*args):
    if mode.get() == "Single":
        batch_frame.grid_remove()
        single_frame.grid()
    else:
        single_frame.grid_remove()
        batch_frame.grid()

mode.trace_add("write", toggle_mode)
toggle_mode()  # Init

# --- Batch helpers (unchanged; can stay here) ---
output_batch_folder = ""

def select_output_batch_folder():
    global output_batch_folder
    folder = filedialog.askdirectory(title="Select Output Folder for 3D Batch")
    if folder:
        output_batch_folder = folder
    else:
        messagebox.showerror("Error", "You must select an output folder to proceed.")

batch_queue = []

def start_batch_processing():
    global batch_queue
    if input_video_listbox.size() != depth_map_listbox.size():
        messagebox.showerror("Mismatch", "Videos and depth maps must match in count.")
        return
    select_output_batch_folder()
    if not output_batch_folder:
        return
    video_paths = input_video_listbox.get(0, tk.END)
    depth_paths = depth_map_listbox.get(0, tk.END)
    batch_queue = list(zip(video_paths, depth_paths))
    process_next_in_batch()

def process_next_in_batch():
    global is_rendering, batch_queue, output_batch_folder
    if not batch_queue:
        print("✅ All batch renders complete.")
        return
    if not is_render_done():
        # use the top-level container to schedule; either works
        visiondepth_content_frame.after(1000, process_next_in_batch)
        return
    video, depth = batch_queue.pop(0)
    input_video_path.set(video)
    selected_depth_map.set(depth)
    scene_num = len(input_video_listbox.get(0, tk.END)) - len(batch_queue)
    output_name = f"sbs-scene-{scene_num:03}.mkv"
    output_path = os.path.join(output_batch_folder, output_name)
    output_sbs_video_path.set(output_path)
    print(f"🎬 Rendering {output_name}...")
    handle_generate_3d()
    visiondepth_content_frame.after(1000, process_next_in_batch)

# Frame to Hold Buttons and Format Selection in a Single Row
button_frame = tk.Frame(right_col, bg="#1c1c1c")
button_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

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
option_menu.config(width=10, cursor="hand2")  # Adjust width to keep consistent look
option_menu.pack(side="left", padx=5)

# Buttons Inside button_frame to Keep Everything on One Line
start_button = tk.Button(
    button_frame,
    text=t("Generate 3D Video"),
    bg="green",
    fg="white",
    cursor="hand2",
    command=lambda: (
        save_settings(),
        handle_generate_3d()
    )
)

start_button.pack(side="left", padx=5)

batch_start_button = tk.Button(
    button_frame,
    text=t("Start Batch Render"),
    bg="purple",
    fg="white",
    cursor="hand2",
    command=lambda: (
        save_settings(),
        start_batch_processing()
    )
)
batch_start_button.pack(side="left", padx=5)


suspend_button = tk.Button(
    button_frame,
    text=t("Suspend"),
    command=suspend_processing,
    bg="orange", fg="black",
    cursor="hand2"
)
suspend_button.pack(side="left", padx=5)

resume_button = tk.Button(
    button_frame,
    text=t("Resume"),
    command=resume_processing,
    bg="blue", fg="white",
    cursor="hand2"
)
resume_button.pack(side="left", padx=5)

cancel_button = tk.Button(
    button_frame,
    text=t("Cancel"),
    command=cancel_processing,
    bg="red", fg="white",
    cursor="hand2"
)
cancel_button.pack(side="left", padx=5)

# Row 7 - Reset button centered
reset_button = tk.Button(
    button_frame,
    text=t("Reset to Defaults"),
    command=reset_settings,
    bg="#8B0000", fg="white",
    cursor="hand2"
)
reset_button.pack(side="left", padx=5)


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
bottom_links_frame.grid(row=3, column=0, columnspan=6, sticky="w", padx=10, pady=10)

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

def on_render_finished(created_files: list[str]):
    if not created_files:
        messagebox.showwarning("Stereo output", "Finished, but no outputs were created.")
        return
    msg = "Output:\n" + "\n".join(created_files)
    print(msg)
    messagebox.showinfo("Stereo output", msg)
    # Optional: open folder of first file
    try:
        import os, subprocess, sys
        folder = os.path.dirname(created_files[0])
        if sys.platform.startswith("win"):
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder])
        else:
            subprocess.run(["xdg-open", folder])
    except Exception:
        pass


# give inner groups uniform column stretch (change 4 if you use more columns)
for grp in (options_frame, pop_frame, color_frame):
    for c in range(4):
        grp.grid_columnconfigure(c, weight=1)

right_col.grid_rowconfigure(99, weight=1)

# same uniform stretch on the right groups
for grp in (top_widgets_frame, inputs_frame, encoding_frame):
    for c in range(4):
        grp.grid_columnconfigure(c, weight=1)

# -- Depth Estimation Tab --
tooltip_refs["Model"] = CreateToolTip(model_dropdown, lambda: t("Tooltip.Model"))
tooltip_refs["OutputDirLabel"] = CreateToolTip(output_dir_label, lambda: t("Tooltip.OutputDirLabel"))
tooltip_refs["OutputDirButton"] = CreateToolTip(output_dir_button, lambda: t("Tooltip.OutputDirButton"))
tooltip_refs["ColormapLabel"] = CreateToolTip(colormap_label, lambda: t("Tooltip.ColormapLabel"))
tooltip_refs["ColormapDropdown"] = CreateToolTip(colormap_dropdown, lambda: t("Tooltip.ColormapDropdown"))
tooltip_refs["InvertCheckbox"] = CreateToolTip(invert_checkbox, lambda: t("Tooltip.InvertCheckbox"))
tooltip_refs["SaveFramesCheckbox"] = CreateToolTip(save_frames_checkbox, lambda: t("Tooltip.SaveFramesCheckbox"))
tooltip_refs["BatchSizeEntry"] = CreateToolTip(batch_size_entry, lambda: t("Tooltip.BatchSizeEntry"))
tooltip_refs["InputLabel"] = CreateToolTip(input_label, lambda: t("Tooltip.InputLabel"))
tooltip_refs["InferenceSteps"] = CreateToolTip(inference_steps_label, lambda: t("Tooltip.InferenceSteps"))
tooltip_refs["DepthLabel"] = CreateToolTip(output_label, lambda: t("Tooltip.DepthLabel"))
tooltip_refs["CPUMode"] = CreateToolTip(offload_mode_label, lambda: t("Tooltip.CPUMode"))
tooltip_refs["ProcessImage"] = CreateToolTip(process_image_button, lambda: t("Tooltip.ProcessImage"))
tooltip_refs["ProcessImageFolder"] = CreateToolTip(process_image_folder_button, lambda: t("Tooltip.ProcessImageFolder"))
tooltip_refs["ProcessVideo"] = CreateToolTip(process_video_button, lambda: t("Tooltip.ProcessVideo"))
tooltip_refs["ProcessVideoFolder"] = CreateToolTip(process_video_folder_button, lambda: t("Tooltip.ProcessVideoFolder"))


# -- 3D Render Tab --
tooltip_refs["StartButton"] = CreateToolTip(start_button, lambda: t("Tooltip.StartButton"))
tooltip_refs["PreviewButton"] = CreateToolTip(preview_button, lambda: t("Tooltip.PreviewButton"))
tooltip_refs["Suspendbutton"] = CreateToolTip(suspend_button, lambda: t("Tooltip.SuspendButton"))
tooltip_refs["ResumeButton"] = CreateToolTip(resume_button, lambda: t("Tooltip.ResumeButton"))
tooltip_refs["CancelButton"] = CreateToolTip(cancel_button, lambda: t("Tooltip.CancelButton"))
tooltip_refs["ResetButton"] = CreateToolTip(reset_button, lambda: t("Tooltip.ResetButton"))
tooltip_refs["ColorResetButton"] = CreateToolTip(color_reset_button, lambda: t("Tooltip.ColorResetButton"))

tooltip_refs["OptionMenu"] = CreateToolTip(option_menu, lambda: t("Tooltip.OptionMenu"))
tooltip_refs["AspectPreview"] = CreateToolTip(aspect_preview_label, lambda: t("Tooltip.AspectPreview"))

# Sliders
tooltip_refs["FGShift"] = CreateToolTip(fg_shift_label, lambda: t("Tooltip.FGShift"))
tooltip_refs["MGShift"] = CreateToolTip(mg_shift_label, lambda: t("Tooltip.MGShift"))
tooltip_refs["BGShift"] = CreateToolTip(bg_shift_label, lambda: t("Tooltip.BGShift"))
tooltip_refs["Sharpness"] = CreateToolTip(sharpness_factor_label, lambda: t("Tooltip.Sharpness"))
tooltip_refs["ZeroParallaxStrength"] = CreateToolTip(zero_parallax_strength_label, lambda: t("Tooltip.ZeroParallaxStrength"))
tooltip_refs["ParallaxBalance"] = CreateToolTip(parallax_balance_label, lambda: t("Tooltip.ParallaxBalance"))
tooltip_refs["MaxPixelShift"] = CreateToolTip(max_pixel_shift_label, lambda: t("Tooltip.MaxPixelShift"))
tooltip_refs["DOFStrength"] = CreateToolTip(dof_strength_label, lambda: t("Tooltip.DOFStrength"))
tooltip_refs["ConvergenceStrength"] = CreateToolTip(convergence_strength_label, lambda: t("Tooltip.ConvergenceStrength"))

# Checkboxes
tooltip_refs["PreserveAspect"]  = CreateToolTip(preserve_aspect_checkbox, lambda: t("Tooltip.PreserveAspect"))
tooltip_refs["AutoCrop"]        = CreateToolTip(auto_crop_checkbox, lambda: t("Tooltip.AutoCrop"))
tooltip_refs["SubjectTracking"] = CreateToolTip(use_subject_tracking_checkbox, lambda: t("Tooltip.SubjectTracking"))
tooltip_refs["FloatingWindow"]  = CreateToolTip(use_dfw_checkbox, lambda: t("Tooltip.FloatingWindow"))
tooltip_refs["EdgeMasking"]     = CreateToolTip(enable_edge_checkbox, lambda: t("Tooltip.EdgeMasking"))
tooltip_refs["Feathering"]      = CreateToolTip(enable_feathering_checkbox, lambda: t("Tooltip.Feathering"))
tooltip_refs["SkipBlankFrames"] = CreateToolTip(skip_blank_frames_checkbox, lambda: t("Tooltip.SkipBlankFrames"))
tooltip_refs["UseFFmpeg"]       = CreateToolTip(use_ffmpeg_checkbox, lambda: t("Tooltip.UseFFmpeg"))
tooltip_refs["EnableDynConvergence"] = CreateToolTip(enable_dynamic_convergence_checkbox, lambda: t("Tooltip.EnableDynConvergence"))

tooltip_refs["PopGamma"]        = CreateToolTip(pop_gamma_label, lambda: t("Tooltip.PopGamma"))
tooltip_refs["PopMid"]          = CreateToolTip(pop_mid_label,   lambda: t("Tooltip.PopMid"))
tooltip_refs["StretchLo"]       = CreateToolTip(stretch_lo_label, lambda: t("Tooltip.StretchLo"))
tooltip_refs["StretchHi"]       = CreateToolTip(stretch_hi_label, lambda: t("Tooltip.StretchHi"))
tooltip_refs["FGPop"]           = CreateToolTip(fg_pop_label,    lambda: t("Tooltip.FGPop"))
tooltip_refs["BGPush"]          = CreateToolTip(bg_push_label,   lambda: t("Tooltip.BGPush"))
tooltip_refs["SubjectLock"]     = CreateToolTip(subj_lock_label, lambda: t("Tooltip.SubjectLock"))
tooltip_refs["ApplyEntries"] = CreateToolTip(apply_entries_btn, lambda: t("Tooltip.ApplyEntries"))

tooltip_refs["Saturation"] = CreateToolTip(saturation_label, lambda: t("Tooltip.Saturation"))
tooltip_refs["Contrast"]   = CreateToolTip(contrast_label,   lambda: t("Tooltip.Contrast"))
tooltip_refs["Brightness"] = CreateToolTip(brightness_label, lambda: t("Tooltip.Brightness"))

# Encoding
tooltip_refs["CRF"] = CreateToolTip(crf_value_label, lambda: t("Tooltip.CRF"))
tooltip_refs["NVENCCQ"] = CreateToolTip(nvenc_cq_value_label, lambda: t("Tooltip.NVENCCQ"))
tooltip_refs["SelectedCodec"] = CreateToolTip(selected_codec_label, lambda: t("Tooltip.SelectedCodec"))
tooltip_refs["FFmpegCodec"] = CreateToolTip(selected_ffmpeg_codec_label, lambda: t("Tooltip.FFmpegCodec"))
tooltip_refs["AspectRatio"] = CreateToolTip(selected_aspect_ratio_label, lambda: t("Tooltip.AspectRatio"))

# --- IPD Controls ---
tooltip_refs["IPDShift"] = CreateToolTip(ipd_label, lambda: t("Tooltip.IPDShift"))
tooltip_refs["EnableIPD"] = CreateToolTip(ipd_toggle, lambda: t("Tooltip.EnableIPD"))

tooltip_refs["AddVideo"] = CreateToolTip(batch_video_button, lambda: t("Tooltip.AddVideo"))
tooltip_refs["AddDepthMap"] = CreateToolTip(batch_depth_button, lambda: t("Tooltip.AddDepthMap"))
tooltip_refs["StartBatchRender"] = CreateToolTip(batch_start_button, lambda: t("Tooltip.StartBatchRender"))
tooltip_refs["ModeSelect"] = CreateToolTip(mode_label, lambda: t("Tooltip.ModeSelect"))

# -- FrameTool Tips --
tooltip_refs["ExtractFrames"] = CreateToolTip(extract_frames_button, lambda: t("Tooltip.ExtractFrames"))
tooltip_refs["RIFE"] = CreateToolTip(RIFE_FPS_button, lambda: t("Tooltip.RIFE"))
tooltip_refs["ESRGAN"] = CreateToolTip(esrgan_button, lambda: t("Tooltip.ESRGAN"))
tooltip_refs["Resolution"] = CreateToolTip(resolution_label, lambda: t("Tooltip.Resolution"))
tooltip_refs["OriginalFPS"] = CreateToolTip(original_fps_label, lambda: t("Tooltip.OriginalFPS"))
tooltip_refs["FPSMultiplier"] = CreateToolTip(fps_multi_label, lambda: t("Tooltip.FPSMultiplier"))
tooltip_refs["AIBlend"] = CreateToolTip(ai_blend_select, lambda: t("Tooltip.AIBlend"))
tooltip_refs["InputResPct"] = CreateToolTip(input_res_pct_label, lambda: t("Tooltip.InputResPct"))
tooltip_refs["ModelSelect"] = CreateToolTip(model_select, lambda: t("Tooltip.ModelSelect"))
tooltip_refs["SceneDetect"] = CreateToolTip(scene_detect_label,        lambda: t("Tooltip.SceneDetect"))
tooltip_refs["SceneDetectThreshold"] = CreateToolTip(scene_detect_threshold, lambda: t("Tooltip.SceneDetectThreshold"))
tooltip_refs["DetectScenesExtract"] = CreateToolTip(detect_scenes_button,     lambda: t("Tooltip.DetectScenesExtract"))
tooltip_refs["LROutput"] = CreateToolTip(StereoOutput_label,        lambda: t("Tooltip.LROutput"))
tooltip_refs["DeleteOutputbtn"] = CreateToolTip(DeleteSBS_label,        lambda: t("Tooltip.DeleteOutputbtn"))
tooltip_refs["HDRbtn"] = CreateToolTip(hdr_checkbox,        lambda: t("Tooltip.HDRbtn"))
tooltip_refs["Livebtn"] = CreateToolTip(live_button,        lambda: t("Tooltip.Livebtn"))
tooltip_refs["Audiobtn"] = CreateToolTip(audio_tool_button,        lambda: t("Tooltip.Audiobtn"))



PRESET_DIR = "presets"
os.makedirs(PRESET_DIR, exist_ok=True)



def refresh_ui_labels():
    # small safety helper so missing widgets don't crash refresh
    def _cfg(w, **kw):
        try:
            w.config(**kw)
        except Exception:
            pass

    # Tabs
    tab_control.tab(depth_tab_index, text=t("Depth Estimation"))
    tab_control.tab(visiondepth_tab_index, text=t("3D Video Generator"))
    tab_control.tab(frametools_tab_index, text=t("FrameTools"))

    # Depth tab
    _cfg(selected_model_label, text=t("Model"))
    _cfg(output_dir_label, text=t("Output Dir: None"))
    _cfg(output_dir_button, text=t("Choose Directory"))
    _cfg(colormap_label, text=t("Colormap:"))
    _cfg(invert_checkbox, text=t("Invert Depth"))
    _cfg(save_frames_checkbox, text=t("Save Frames"))
    _cfg(batch_size_label, text=t("Batch Size (Frames):"))
    _cfg(inference_res_label, text=t("Inference Resolution:"))
    _cfg(inference_steps_label, text=t("Inference Steps:"))
    _cfg(status_label, text=t("Ready"))
    _cfg(offload_mode_label, text=t("CPU Offload Mode"))
    _cfg(input_label, text=t("Input Image"))
    _cfg(output_label, text=t("Depth Map"))
    _cfg(process_image_button, text=t("Process Image"))
    _cfg(process_image_folder_button, text=t("Process Image Folder"))
    _cfg(process_video_button, text=t("Process Video"))
    _cfg(process_video_folder_button, text=t("Process Video Folder"))
    

    # 3D Render tab — existing controls
    _cfg(video_thumbnail_label, text=t("No Thumbnail"))
    _cfg(video_specs_label, text=t("Resolution: N/A\nFPS: N/A"))
    _cfg(depth_map_label, text=t("Depth Map (3D): None"))
    _cfg(audio_tool_button, text=t("Audio Tool"))
    _cfg(select_input_video_button, text=t("Select Input Video"))
    _cfg(select_depth_map_button, text=t("Select Depth Map"))
    _cfg(select_output_video_button, text=t("Select Output Video"))
    _cfg(format_button, text=t("3D Format"))
    _cfg(start_button, text=t("Generate 3D Video"))
    _cfg(batch_start_button, text=t("Start Batch Render"))
    _cfg(preview_button, text=t("Open Preview"))
    _cfg(suspend_button, text=t("Suspend"))
    _cfg(resume_button, text=t("Resume"))
    _cfg(cancel_button, text=t("Cancel"))
    _cfg(cancel_depth_button, text=t("Cancel"))
    _cfg(reset_button, text=t("Reset to Defaults"))
    _cfg(color_reset_button, text=t("Reset"))
    _cfg(inputs_frame, text=t("Input Sources"))
    _cfg(mode_label, text=t("Mode:"))
    
    _cfg(save_preset_button, text=t("Save Preset"))
    _cfg(batch_video_button, text=t("+ Add Video"))
    _cfg(batch_depth_button, text=t("+ Add Depth Map"))
    

    # Parallax/quality sliders (existing)
    _cfg(fg_shift_label, text=t("Foreground Shift"))
    _cfg(mg_shift_label, text=t("Midground Shift"))
    _cfg(bg_shift_label, text=t("Background Shift"))
    _cfg(sharpness_factor_label, text=t("Sharpness Factor"))
    _cfg(zero_parallax_strength_label, text=t("Zero Parallax Strength"))
    _cfg(parallax_balance_label, text=t("Parallax Balance"))
    _cfg(max_pixel_shift_label, text=t("Max Pixel Shift %"))
    _cfg(dof_strength_label, text=t("DOF Strength"))
    _cfg(convergence_strength_label, text=t("Convergence Strength"))

    # ✅ NEW: “Pop & Subject Controls” group + labels
    _cfg(pop_gamma_label, text=t("Depth Pop Gamma"))
    _cfg(pop_mid_label, text=t("Pop Mid (0..1)"))
    _cfg(stretch_lo_label, text=t("Stretch Lo"))
    _cfg(stretch_hi_label, text=t("Stretch Hi"))
    _cfg(fg_pop_label, text=t("FG Pop ×"))
    _cfg(bg_push_label, text=t("BG Push ×"))
    _cfg(subj_lock_label, text=t("Subject Lock"))
    _cfg(apply_entries_btn, text=t("Apply Entries"))

    # Toggles / checkboxes (existing)
    _cfg(preserve_aspect_checkbox, text=t("Preserve Original Aspect Ratio"))
    _cfg(auto_crop_checkbox, text=t("Auto Crop Black Bars"))
    _cfg(use_subject_tracking_checkbox, text=t("Stabilize Zero-Parallax (center-depth)"))
    _cfg(use_dfw_checkbox, text=t("Enable Floating Window (DFW)"))
    _cfg(use_ffmpeg_checkbox, text=t("Use FFmpeg Renderer"))
    _cfg(enable_edge_checkbox, text=t("Enable Edge Masking"))
    _cfg(enable_feathering_checkbox, text=t("Enable Feathering"))
    _cfg(skip_blank_frames_checkbox, text=t("Skip Blank/White Frames"))
    _cfg(enable_dynamic_convergence_checkbox, text=t("Enable Dynamic Convergence"))

    # Encoding settings
    _cfg(selected_aspect_ratio_label, text=t("Aspect Ratio:"))
    _cfg(selected_ffmpeg_codec_label, text=t("FFmpeg Codec:"))
    _cfg(selected_codec_label, text=t("Codec:"))
    _cfg(crf_value_label, text=t("CRF"))
    _cfg(nvenc_cq_value_label, text=t("NVENC CQ"))
    _cfg(encoding_frame, text=t("Encoding Settings"))
    _cfg(options_frame, text=t("Processing Options"))
    _cfg(top_widgets_frame, text=t("Video Info"))
    _cfg(StereoOutput_label, text=t("Left/Right Output"))
    _cfg(DeleteSBS_label, text=t("Delete SBS after"))
    _cfg(hdr_checkbox, text=t("Preserve HDR10"))

    # FrameTools tab
    _cfg(extract_frames_button, text=t("Extract Frames from Video"))
    _cfg(io_frame, text=t("Input / Output"))
    _cfg(frames_folder_label, text=t("Frames Folder:"))
    _cfg(browse_button, text=t("Browse"))
    _cfg(output_video_file_label, text=t("Output Video File:"))
    _cfg(save_as_button, text=t("Save As"))

    _cfg(proc_frame, text=t("Processing Options"))
    _cfg(RIFE_FPS_button, text=t("Enable RIFE Interpolation"))
    _cfg(esrgan_button, text=t("Enable Real-ESRGAN Upscale"))

    _cfg(out_frame, text=t("Output Settings"))
    _cfg(resolution_label, text=t("Resolution (WxH):"))
    _cfg(original_fps_label, text=t("Original FPS:"))
    _cfg(fps_multi_label, text=t("FPS Interpolation Multiplier:"))
    _cfg(selected_ffmpeg_codec_frametools_label, text=t("FFmpeg Output Codec:"))

    _cfg(esrgan_frame, text=t("ESRGAN Settings"))
    _cfg(ai_blend_select, text=t("AI Blending:"))
    _cfg(input_res_pct_label, text=t("Input Resolution %:"))
    _cfg(model_select, text=t("Model Selection:"))
    _cfg(start_processing_button, text=t("▶ Start Processing"))
    _cfg(merged_status, text=t("Waiting to start..."))
    
    _cfg(scene_detect_label, text=t("🎥 Scene Detection (PySceneDetect)"))
    _cfg(scene_detect_threshold, text=t("Sensitivity Threshold (lower = more cuts):"))
    _cfg(detect_scenes_button, text=t("Detect Scenes & Extract"))
    
    # 🎨 Color Grading (new group)
    _cfg(color_frame, text=t("Color Grading"))
    _cfg(saturation_label, text=t("Saturation"))
    _cfg(contrast_label,   text=t("Contrast"))
    _cfg(brightness_label, text=t("Brightness"))
       
        # 🔀 Stereo Separation (IPD)
    _cfg(ipd_toggle, text=t("Enable Stereo Scaling (IPD)"))
    _cfg(ipd_label, text=t("Stereo Scaling (IPD)"))
 
    _cfg(pop_frame, text=t("Pop & Subject Controls"))    
    _cfg(ipd_label, text=t("Stereo Scaling (IPD)"))
    
    _cfg(live_button, text=t("VD3D External Input 3D (WIP)"))
    
    try:
        depth_blender_ui.refresh_labels()
    except Exception:
        pass

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

    # --- existing ---
    fg_shift.set(float(config.get("fg_shift", 8.0)))
    mg_shift.set(float(config.get("mg_shift", 1.5)))
    bg_shift.set(float(config.get("bg_shift", -2.5)))
    zero_parallax_strength.set(float(config.get("zero_parallax_strength", 0.0)))
    max_pixel_shift.set(float(config.get("max_pixel_shift", 0.02)))
    parallax_balance.set(float(config.get("parallax_balance", 0.8)))
    sharpness_factor.set(float(config.get("sharpness_factor", 1.0)))
    dof_strength.set(float(config.get("dof_strength", 2.0)))
    convergence_strength.set(float(config.get("convergence_strength", 0.0)))

    use_ffmpeg.set(bool(config.get("use_ffmpeg", False)))
    enable_feathering.set(bool(config.get("enable_feathering", True)))
    enable_edge_masking.set(bool(config.get("enable_edge_masking", True)))
    use_floating_window.set(bool(config.get("use_floating_window", True)))
    auto_crop_black_bars.set(bool(config.get("auto_crop_black_bars", False)))
    skip_blank_frames.set(bool(config.get("skip_blank_frames", False)))
    enable_dynamic_convergence.set(bool(config.get("enable_dynamic_convergence", True)))

    # --- NEW: pop & subject controls (with sane clamps) ---
    def _clamp(v, lo, hi): return max(lo, min(hi, v))

    gamma = float(config.get("depth_pop_gamma", 0.85))
    depth_pop_gamma.set(_clamp(gamma, 0.70, 1.20))

    mid = float(config.get("depth_pop_mid", 0.50))
    depth_pop_mid.set(_clamp(mid, 0.0, 1.0))

    lo = float(config.get("depth_stretch_lo", 0.05))
    hi = float(config.get("depth_stretch_hi", 0.95))
    lo = _clamp(lo, 0.0, 1.0)
    hi = _clamp(hi, 0.0, 1.0)
    if hi <= lo:
        lo, hi = 0.05, 0.95
    depth_stretch_lo.set(lo)
    depth_stretch_hi.set(hi)

    fg_mul = float(config.get("fg_pop_multiplier", 1.20))
    bg_mul = float(config.get("bg_push_multiplier", 1.10))
    subject_lock = float(config.get("subject_lock_strength", 1.00))
    fg_pop_multiplier.set(_clamp(fg_mul, 0.5, 2.0))
    bg_push_multiplier.set(_clamp(bg_mul, 0.5, 2.0))
    subject_lock_strength.set(_clamp(subject_lock, 0.0, 2.0))

    # --- NEW: color grading (backward compatible defaults) ---
    sat = float(config.get("saturation", 1.0))
    con = float(config.get("contrast",   1.0))
    bri = float(config.get("brightness", 0.0))
    saturation.set(_clamp(sat, 0.0, 2.0))
    contrast.set(_clamp(con, 0.0, 2.0))
    brightness.set(_clamp(bri, -0.5, 0.5))
    
        # --- NEW: IPD / stereo separation (backward compatible) ---
    try:
        ipd_on  = bool(config.get("ipd_enabled", False))
        ipd_val = float(config.get("ipd_factor", 1.00))

        # clamp to your slider's range
        def _clamp(v, lo, hi): return max(lo, min(hi, v))
        ipd_val = _clamp(ipd_val, 0.50, 1.50)

        # set vars if they exist
        if 'ipd_enabled_var' in globals(): ipd_enabled_var.set(ipd_on)
        if 'ipd_factor_var'  in globals(): ipd_factor_var.set(ipd_val)

        # update UI bits if present
        try:
            if 'ipd_slider' in globals():
                ipd_slider.config(state=('normal' if ipd_on else 'disabled'))
            if 'ipd_value_lbl' in globals():
                ipd_value_lbl.config(text=f"{ipd_val:.2f}x")
            # if you wired a trace callback like _on_ipd_slider(), call it:
            if '_on_ipd_slider' in globals():
                _on_ipd_slider()
        except Exception:
            pass
    except Exception as e:
        print(f"⚠️ IPD restore skipped: {e}")

    # 🧽 Sync entry fields (if present) so UI reflects loaded preset immediately
    try:
        pop_mid_entry.delete(0, tk.END);      pop_mid_entry.insert(0, f"{depth_pop_mid.get():.2f}")
        stretch_lo_entry.delete(0, tk.END);   stretch_lo_entry.insert(0, f"{depth_stretch_lo.get():.2f}")
        stretch_hi_entry.delete(0, tk.END);   stretch_hi_entry.insert(0, f"{depth_stretch_hi.get():.2f}")
    except Exception:
        pass

    print(f"✅ Applied preset: {preset_name}")


def save_current_preset(name="custom_preset.json"):
    preset = {
        # --- existing ---
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

        # --- pop & subject controls ---
        "depth_pop_gamma": depth_pop_gamma.get(),
        "depth_pop_mid": depth_pop_mid.get(),
        "depth_stretch_lo": depth_stretch_lo.get(),
        "depth_stretch_hi": depth_stretch_hi.get(),
        "fg_pop_multiplier": fg_pop_multiplier.get(),
        "bg_push_multiplier": bg_push_multiplier.get(),
        "subject_lock_strength": subject_lock_strength.get(),

        # --- color grading ---
        "saturation": saturation.get(),
        "contrast":   contrast.get(),
        "brightness": brightness.get(),
        
        # --- IPD / stereo separation ---
        "ipd_enabled": ipd_enabled_var.get(),
        "ipd_factor":  ipd_factor_var.get(),


        # Optional: version tag helps future migrations
        "preset_version": "3.5"
    }

    path = os.path.join(PRESET_DIR, name)
    with open(path, 'w') as f:
        json.dump(preset, f, indent=4)

    print(f"💾 Preset saved: {name}")
    preset_menu['values'] = get_all_presets()
    preset_menu.set(os.path.splitext(name)[0])






# Ensure settings are saved when the program closes
def on_exit():
    save_settings()
    root.destroy() # ❌ Close GUI

root.protocol("WM_DELETE_WINDOW", on_exit)

load_settings() 

root.mainloop()


