import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn.functional as F
import json
import os
from render_3d import (
    frame_to_tensor,
    depth_to_tensor,
    pixel_shift_cuda,
    apply_sharpening,
)
from preview_utils import grab_frame_from_video, generate_preview_image

SETTINGS_FILE = "preview_settings.json"

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {}

def open_3d_preview_window(input_video_path, selected_depth_map,
                           fg_shift, mg_shift, bg_shift,
                           blur_ksize, feather_strength,
                           use_subject_tracking, use_floating_window, convergence_offset):
    settings = load_settings()

    preview_win = tk.Toplevel()
    preview_win.title("🌀 Live 3D Preview")
    preview_win.geometry("1010x800")

    cap = cv2.VideoCapture(input_video_path.get())
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 10000
    cap.release()

    canvas = tk.Canvas(preview_win)
    scrollbar = tk.Scrollbar(preview_win, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    inner_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")
    inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    top_controls_frame = tk.Frame(inner_frame)
    top_controls_frame.pack(side="top", fill="x")

    left_dummy_top = tk.Label(top_controls_frame)
    left_dummy_top.pack(side='left', fill='x', expand=True)

    width_label = tk.Label(top_controls_frame, text="Width:")
    width_label.pack(side="left")
    width_entry = tk.Entry(top_controls_frame, width=5)
    width_entry.insert(0, settings.get('width', '960'))
    width_entry.pack(side="left")

    height_label = tk.Label(top_controls_frame, text="Height:")
    height_label.pack(side="left")
    height_entry = tk.Entry(top_controls_frame, width=5)
    height_entry.insert(0, settings.get('height', '540'))
    height_entry.pack(side="left", padx=(0, 20))

    apply_size_button = tk.Button(top_controls_frame, text="Apply Size", command=lambda: apply_size())
    apply_size_button.pack(side="left", padx=10)

    preview_type_var = tk.StringVar(value=settings.get('preview_type', "Red-Blue Anaglyph"))
    preview_dropdown = tk.OptionMenu(top_controls_frame, preview_type_var, *[
        "Red-Blue Anaglyph", "HSBS", "Shift Heatmap", "Overlay Arrows", "Feather Mask", "Left-Right Diff"
    ])
    preview_dropdown.pack(side="left")

    right_dummy_top = tk.Label(top_controls_frame)
    right_dummy_top.pack(side='left', fill='x', expand=True)

    preview_frame = tk.Frame(inner_frame)
    preview_frame.pack(side="top", fill="x")
    preview_canvas = tk.Label(preview_frame)
    preview_canvas.pack()

    shift_sliders_frame = tk.Frame(inner_frame)
    shift_sliders_frame.pack(side="top", fill="x")
    fg_slider = tk.Scale(shift_sliders_frame, from_=0, to=15, orient="horizontal", label="FG Shift", resolution=0.5, length=200)
    fg_slider.pack(side="left", padx=10)
    mg_slider = tk.Scale(shift_sliders_frame, from_=-10, to=10, orient="horizontal", label="MG Shift", resolution=0.5, length=200)
    mg_slider.pack(side="left", padx=10)
    bg_slider = tk.Scale(shift_sliders_frame, from_=0, to=-10, orient="horizontal", label="BG Shift", resolution=0.5, length=200)
    bg_slider.pack(side="left", padx=10)

    feather_sliders_frame = tk.Frame(inner_frame)
    feather_sliders_frame.pack(side="top", fill="x")
    feather_strength_slider = tk.Scale(feather_sliders_frame, from_=0, to=20, orient="horizontal", label="Feather Strength", resolution=0.5, length=200)
    feather_strength_slider.pack(side="left", padx=10)
    blur_ksize_slider = tk.Scale(feather_sliders_frame, from_=1, to=15, orient="horizontal", label="Feather Blur Size", resolution=1, length=200)
    blur_ksize_slider.pack(side="left", padx=10)
    sharpening_slider = tk.Scale(feather_sliders_frame, from_=-1, to=1, orient="horizontal", label="Sharpen Factor", resolution=0.1, length=200)
    sharpening_slider.pack(side="left", padx=10)
    max_shift_slider = tk.Scale(feather_sliders_frame, from_=0.005, to=0.10, orient="horizontal", label="Max Pixel Shift (%)", resolution=0.005, length=200)
    max_shift_slider.pack(side="left", padx=10)
    convergence_slider = tk.Scale(
        feather_sliders_frame, from_=-0.05, to=0.05,
        orient="horizontal", label="Convergence Offset",
        resolution=0.001, length=200
    )
    convergence_slider.pack(side="left", padx=10)
     

    frame_slider_frame = tk.Frame(inner_frame)
    frame_slider_frame.pack(side="top")
    prev_button = tk.Button(frame_slider_frame, text="⏪ Prev", command=lambda: step_frame(-1))
    prev_button.pack(side="left", padx=(10, 5))
    frame_slider = tk.Scale(frame_slider_frame, from_=0, to=total_frames - 1, orient="horizontal", label="Frame", length=800)
    frame_slider.pack(side="left")
    next_button = tk.Button(frame_slider_frame, text="Next ⏩", command=lambda: step_frame(1))
    next_button.pack(side="left", padx=(5, 10))

    def update_preview(*args):
        input_path = input_video_path.get()
        depth_path = selected_depth_map.get()
        if not input_path or not depth_path:
            return

        frame_idx = frame_slider.get()
        frame = grab_frame_from_video(input_path, frame_idx)
        depth = grab_frame_from_video(depth_path, frame_idx)
        if frame is None or depth is None:
            messagebox.showerror("Frame Error", "Unable to grab frame from video.")
            return

        h, w = frame.shape[:2]
        frame_tensor = F.interpolate(frame_to_tensor(frame).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        depth_tensor = F.interpolate(depth_to_tensor(depth).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

        left, right, shift_map = pixel_shift_cuda(
            frame_tensor, depth_tensor, w, h,
            fg_slider.get(), mg_slider.get(), bg_slider.get(),
            blur_ksize=blur_ksize_slider.get(),
            feather_strength=feather_strength_slider.get(),
            return_shift_map=True,
            use_subject_tracking=use_subject_tracking.get(),
            enable_floating_window=use_floating_window.get(),
            max_pixel_shift_percent=max_shift_slider.get(),
            convergence_offset=convergence_slider.get()
        )


        preview_img = generate_preview_image(preview_type_var.get(), left, right, shift_map, w, h)
        if preview_img is not None:
            preview_img = apply_sharpening(preview_img, sharpening_slider.get())
            img_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img = pil_img.resize((int(width_entry.get()), int(height_entry.get())), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(pil_img)
            preview_canvas.config(image=img_tk)
            preview_canvas.image = img_tk

    def step_frame(delta):
        frame_slider.set(max(0, min(frame_slider.get() + delta, total_frames - 1)))
        update_preview()

    def apply_size():
        try:
            preview_width = int(width_entry.get())
            preview_height = int(height_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid width or height.")
            return
        frame_slider.config(length=preview_width - 120)
        preview_win.geometry(f"{preview_width + 30}x{330 + preview_height}")
        update_preview()

    def on_close():
        settings = {
            'width': width_entry.get(),
            'height': height_entry.get(),
            'preview_type': preview_type_var.get(),
            'fg_shift': fg_slider.get(),
            'mg_shift': mg_slider.get(),
            'bg_shift': bg_slider.get(),
            'blur_ksize': blur_ksize_slider.get(),
            'feather_strength': feather_strength_slider.get(),
            'sharpening': sharpening_slider.get(),
            'max_pixel_shift': max_shift_slider.get(),
            'convergence_offset': convergence_slider.get(),
        }
        save_settings(settings)
        preview_win.destroy()

    preview_win.protocol("WM_DELETE_WINDOW", on_close)

    fg_slider.set(settings.get('fg_shift', fg_shift.get()))
    mg_slider.set(settings.get('mg_shift', mg_shift.get()))
    bg_slider.set(settings.get('bg_shift', bg_shift.get()))
    blur_ksize_slider.set(settings.get('blur_ksize', blur_ksize.get()))
    feather_strength_slider.set(settings.get('feather_strength', feather_strength.get()))
    sharpening_slider.set(settings.get('sharpening', 0.2))
    max_shift_slider.set(settings.get('max_pixel_shift', 0.02))
    convergence_slider.set(settings.get('convergence_offset', convergence_offset.get()))
    frame_slider.set(0)

    fg_slider.config(command=update_preview)
    mg_slider.config(command=update_preview)
    bg_slider.config(command=update_preview)
    blur_ksize_slider.config(command=update_preview)
    feather_strength_slider.config(command=update_preview)
    sharpening_slider.config(command=update_preview)
    max_shift_slider.config(command=update_preview)
    frame_slider.config(command=update_preview)
    preview_type_var.trace_add("write", lambda *_: update_preview())
    convergence_slider.config(command=update_preview)

    preview_win.protocol("WM_DELETE_WINDOW", on_close)
