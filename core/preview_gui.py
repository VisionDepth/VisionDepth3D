import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn.functional as F
# ✅ Correct: Import specific functions from your module
from render_3d import (
    frame_to_tensor,
    depth_to_tensor,
    pixel_shift_cuda,
)

from preview_utils import grab_frame_from_video, generate_preview_image


def open_3d_preview_window(input_video_path, selected_depth_map,
                           fg_shift, mg_shift, bg_shift,
                           blur_ksize, feather_strength,
                           use_subject_tracking, use_floating_window):

    preview_win = tk.Toplevel()
    preview_win.title("🌀 Live 3D Preview")

    preview_canvas = tk.Label(preview_win)
    preview_canvas.pack()

    fg_slider = tk.Scale(preview_win, from_=0, to=15, orient="horizontal", label="FG Shift", resolution=0.5)
    mg_slider = tk.Scale(preview_win, from_=-10, to=10, orient="horizontal", label="MG Shift", resolution=0.5)
    bg_slider = tk.Scale(preview_win, from_=0, to=-10, orient="horizontal", label="BG Shift", resolution=0.5)
    frame_slider = tk.Scale(preview_win, from_=0, to=10000, orient="horizontal", label="Frame")

    fg_slider.set(fg_shift.get())
    mg_slider.set(mg_shift.get())
    bg_slider.set(bg_shift.get())
    frame_slider.set(0)

    fg_slider.pack()
    mg_slider.pack()
    bg_slider.pack()
    frame_slider.pack()

    preview_type_var = tk.StringVar(value="Red-Blue Anaglyph")
    preview_dropdown = tk.OptionMenu(preview_win, preview_type_var, *[
        "Red-Blue Anaglyph", "HSBS", "Shift Heatmap", "Overlay Arrows", "Feather Mask", "Left-Right Diff"
    ])
    preview_dropdown.pack()

    def update_preview(*args):
        input_path = input_video_path.get()
        depth_path = selected_depth_map.get()
        if not input_path or not depth_path:
            return

        frame_idx = frame_slider.get()
        fg = fg_slider.get()
        mg = mg_slider.get()
        bg = bg_slider.get()

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
            fg, mg, bg,
            blur_ksize=blur_ksize.get(),
            feather_strength=feather_strength.get(),
            return_shift_map=True,
            use_subject_tracking=use_subject_tracking.get(),
            enable_floating_window=use_floating_window.get()
        )

        preview_type = preview_type_var.get()
        preview_img = generate_preview_image(preview_type, left, right, shift_map, w, h)
        if preview_img is not None:
            img_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img = pil_img.resize((960, 540), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(pil_img)
            preview_canvas.config(image=img_tk)
            preview_canvas.image = img_tk

    # Hook slider events to refresh
    fg_slider.config(command=update_preview)
    mg_slider.config(command=update_preview)
    bg_slider.config(command=update_preview)
    frame_slider.config(command=update_preview)
    preview_type_var.trace_add("write", lambda *_: update_preview())

    update_preview()
