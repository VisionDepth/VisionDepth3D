import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn.functional as F
import json
import os
import datetime
from tkinter import filedialog

from core.render_3d import (
    frame_to_tensor,
    depth_to_tensor,
    pixel_shift_cuda,
    apply_sharpening,
    tensor_to_frame,
    pad_to_aspect_ratio,
    format_3d_output,
    apply_color_grade,   # color grading
)
from core.preview_utils import grab_frame_from_video, generate_preview_image

SETTINGS_FILE = "settings.json"

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_frame(capture, frame_idx):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = capture.read()
    return frame if ret else None

def open_3d_preview_window(
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
    # pop/subject vars (use main UI vars directly)
    depth_pop_gamma,
    depth_pop_mid,
    depth_stretch_lo,
    depth_stretch_hi,
    fg_pop_multiplier,
    bg_push_multiplier,
    subject_lock_strength,
    # color vars (use main UI vars directly)
    saturation,
    contrast,
    brightness,
):
    settings = load_settings()
    # --- IPD preview state ---
    ipd_enabled = tk.BooleanVar(value=bool(settings.get("ipd_enabled", True)))
    ipd_scale   = tk.DoubleVar(value=float(settings.get("ipd_scale", 1.00)))


    preview_win = tk.Toplevel()
    preview_win.title("Live 3D Preview")
    preview_win.geometry("1010x1000")

    global preview_cap
    preview_img = None
    frame = None
    depth = None
    preview_cap = cv2.VideoCapture(input_video_path.get())
    depth_cap = cv2.VideoCapture(selected_depth_map.get())

    input_total  = int(preview_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if preview_cap.isOpened() else 0
    depth_total  = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if depth_cap.isOpened() else 0

    if input_total <= 0:
        input_total = 1
    total_frames = min(v for v in (input_total, depth_total) if v > 0) if (input_total > 0 and depth_total > 0) else input_total
    total_frames = max(1, total_frames)
    
    # ----- Paned layout: top (preview) / bottom (scrollable controls) -----
    preview_win.minsize(980, 640)
    preview_win.rowconfigure(0, weight=1)
    preview_win.columnconfigure(0, weight=1)

    paned = tk.PanedWindow(preview_win, orient="vertical", sashwidth=6, sashrelief="flat")
    paned.grid(row=0, column=0, sticky="nsew")

    # Top area: reserved image holder + frame slider
    top_area = tk.Frame(paned)
    top_area.columnconfigure(0, weight=1)
    paned.add(top_area, minsize=280)

    initial_w = int(settings.get('width', '960')) if str(settings.get('width', '960')).isdigit() else 960
    initial_h = int(settings.get('height', '540')) if str(settings.get('height', '540')).isdigit() else 540

    img_holder = tk.Frame(top_area, height=initial_h + 8)
    img_holder.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
    img_holder.pack_propagate(False)

    preview_canvas = tk.Label(img_holder, bg="#111", bd=1, relief="flat", width=initial_w, height=initial_h)
    preview_canvas.pack(anchor="center")

    frame_slider = tk.Scale(
        top_area, from_=0, to=total_frames - 1, orient="horizontal",
        label="Frame", length=800
    )
    frame_slider.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

    # Auto-fit slider length
    def _resize_slider(ev=None):
        try:
            w = top_area.winfo_width()
            frame_slider.configure(length=max(100, w - 40))
        except Exception:
            pass
    preview_win.bind("<Configure>", _resize_slider)

    # Bottom area: scrollable controls
    bottom_area = tk.Frame(paned)
    paned.add(bottom_area)

    class ScrollableFrame(tk.Frame):
        def __init__(self, parent, *args, **kwargs):
            super().__init__(parent, *args, **kwargs)
            self.canvas = tk.Canvas(self, highlightthickness=0)
            self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.vsb.set)

            self.inner = tk.Frame(self.canvas)
            self.inner.bind(
                "<Configure>",
                lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            )
            self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

            self.canvas.grid(row=0, column=0, sticky="nsew")
            self.vsb.grid(row=0, column=1, sticky="ns")
            self.rowconfigure(0, weight=1)
            self.columnconfigure(0, weight=1)

            def _resize_inner(event):
                self.canvas.itemconfig(self._win, width=event.width)
            self.canvas.bind("<Configure>", _resize_inner)

            # Mouse wheel
            def _on_wheel(event):
                delta = event.delta
                if delta == 0 and hasattr(event, "num"):  # X11
                    delta = -120 if event.num == 5 else 120
                self.canvas.yview_scroll(int(-delta / 120), "units")

            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.canvas.bind_all(seq, _on_wheel, add="+")

    scroll = ScrollableFrame(bottom_area)
    scroll.pack(fill="both", expand=True)

    control_container = scroll.inner

    top_controls_frame = tk.LabelFrame(control_container, text="Preview Controls", padx=10, pady=5)
    top_controls_frame.pack(pady=(0, 10), anchor="center")

    width_label = tk.Label(top_controls_frame, text="Width:")
    width_label.grid(row=0, column=0)
    width_entry = tk.Entry(top_controls_frame, width=5)
    width_entry.insert(0, str(initial_w))
    width_entry.grid(row=0, column=1, padx=(0, 10))

    height_label = tk.Label(top_controls_frame, text="Height:")
    height_label.grid(row=0, column=2)
    height_entry = tk.Entry(top_controls_frame, width=5)
    height_entry.insert(0, str(initial_h))
    height_entry.grid(row=0, column=3, padx=(0, 10))
    preview_job = None  

    def update_preview_debounced(*args):
        nonlocal preview_job
        if preview_job is not None:
            preview_win.after_cancel(preview_job)
        preview_job = preview_win.after(150, update_preview_now)  # debounce

    def apply_size():
        try:
            preview_width = int(width_entry.get())
            preview_height = int(height_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid width or height.")
            return
        img_holder.configure(height=preview_height + 8)
        preview_canvas.config(width=preview_width, height=preview_height)
        _resize_slider()
        update_preview_debounced()

    apply_size_button = tk.Button(top_controls_frame, text="Apply Size", command=apply_size)
    apply_size_button.grid(row=0, column=4, padx=(0, 10))

    save_button = tk.Button(top_controls_frame, text="Save Preview", command=lambda: save_preview(preview_img))
    save_button.grid(row=0, column=8, padx=(0, 10))

    preview_type_var = tk.StringVar(value=settings.get('preview_type', "HSBS"))
    preview_dropdown = tk.OptionMenu(top_controls_frame, preview_type_var, *[
        "Passive Interlaced",
        "Red-Blue Anaglyph",
        "HSBS",
        "Shift Heatmap",
        "Shift Heatmap (Abs)",
        "Shift Heatmap (Clipped ±5px)",
        "Overlay Arrows",
        "Left-Right Diff",
        "Feather Mask",
        "Feather Blend"
    ])
    preview_dropdown.grid(row=0, column=5, padx=(0, 10))

    edge_masking_checkbox = tk.Checkbutton(top_controls_frame, text="Edge Masking", variable=enable_edge_masking)
    edge_masking_checkbox.grid(row=0, column=6)

    feathering_checkbox = tk.Checkbutton(top_controls_frame, text="Feathering", variable=enable_feathering)
    feathering_checkbox.grid(row=0, column=7)

    shift_frame = tk.LabelFrame(control_container, text="Depth Shift Settings", padx=10, pady=5)
    shift_frame.pack(pady=(0, 10), anchor="center")

    fg_slider = tk.Scale(shift_frame, from_=0, to=30, resolution=0.5, orient="horizontal", label="FG Shift", variable=fg_shift, length=200)
    fg_slider.grid(row=0, column=0, padx=10)

    mg_slider = tk.Scale(shift_frame, from_=-10, to=10, resolution=0.5, orient="horizontal", label="MG Shift", variable=mg_shift, length=200)
    mg_slider.grid(row=0, column=1, padx=10)

    bg_slider = tk.Scale(shift_frame, from_=-20, to=0, resolution=0.5, orient="horizontal", label="BG Shift", variable=bg_shift, length=200)
    bg_slider.grid(row=0, column=2, padx=10)

    feather_frame = tk.LabelFrame(control_container, text="Parallax Control", padx=10, pady=5)
    feather_frame.pack(pady=(0, 10), anchor="center")

    feather_frame.columnconfigure((0, 1, 2, 3), weight=1, pad=10)
    feather_frame.columnconfigure((4, 5, 6, 7), weight=1, pad=10)

    # Row 0
    tk.Label(feather_frame, text="Sharpen").grid(row=0, column=0, sticky="w")
    sharpness_slider = tk.Scale(
        feather_frame, from_=-1, to=1, resolution=0.1, orient="horizontal",
        variable=sharpness_factor, length=150, showvalue=True,
        command=lambda _: update_preview_debounced()
    )
    sharpness_slider.grid(row=0, column=1, sticky="w")

    tk.Label(feather_frame, text="Max Pixel Shift (%)").grid(row=0, column=2, sticky="w")
    max_shift_entry = tk.Entry(feather_frame, width=8)
    max_shift_entry.insert(0, str(max_pixel_shift.get()))
    max_shift_entry.grid(row=0, column=3, sticky="w")

    # Row 1
    tk.Label(feather_frame, text="DoF Strength").grid(row=1, column=0, sticky="w")
    dof_strength_entry = tk.Entry(feather_frame, width=8)
    dof_strength_entry.insert(0, str(dof_strength.get()))
    dof_strength_entry.grid(row=1, column=1, sticky="w")

    tk.Label(feather_frame, text="Convergence Strength").grid(row=1, column=2, sticky="w")
    convergence_slider = tk.Scale(
        feather_frame, from_=-0.05, to=0.05, resolution=0.001, orient="horizontal",
        variable=convergence_strength, length=150, showvalue=True,
        command=lambda _: update_preview_debounced()
    )
    convergence_slider.grid(row=1, column=3, sticky="w")

    # Row 2
    zero_parallax_strength_slider = tk.Scale(
        feather_frame, from_=-0.05, to=0.05,
        resolution=0.001, orient="horizontal",
        label="Zero Parallax Strength", variable=zero_parallax_strength,
        length=200, command=lambda _: update_preview_debounced()
    )
    zero_parallax_strength_slider.grid(row=2, column=0, columnspan=2, sticky="w")

    parallax_balance_slider = tk.Scale(
        feather_frame, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
        label="Parallax Balance", variable=parallax_balance, length=200,
        command=lambda _: update_preview_debounced()
    )
    parallax_balance_slider.grid(row=2, column=2, columnspan=2, sticky="w")

    # --- Stereo Separation (IPD) ---
    ipd_frame = tk.LabelFrame(control_container, text="Stereo Separation (IPD)", padx=10, pady=8)
    ipd_frame.pack(pady=(0, 10), anchor="center")

    ipd_chk = tk.Checkbutton(ipd_frame, text="Enable Stereo Scaling (IPD)", variable=ipd_enabled,
                             command=lambda: update_preview_debounced())
    ipd_chk.grid(row=0, column=0, sticky="w", padx=4)

    tk.Label(ipd_frame, text="Stereo Scaling (IPD)").grid(row=0, column=1, sticky="w", padx=(12,4))
    ipd_slider = tk.Scale(ipd_frame, from_=0.50, to=1.50, resolution=0.01, orient="horizontal",
                          variable=ipd_scale, length=220, command=lambda _=None: update_preview_debounced())
    ipd_slider.grid(row=0, column=2, sticky="w", padx=4)

    pop_frame = tk.LabelFrame(control_container, text="Pop & Subject Controls", padx=10, pady=8)
    pop_frame.pack(pady=(0, 10), anchor="center")

    # Row 0
    tk.Label(pop_frame, text="Depth Pop Gamma").grid(row=0, column=0, sticky="w")
    gamma_slider = tk.Scale(pop_frame, from_=0.70, to=1.20, resolution=0.01,
                            orient="horizontal", variable=depth_pop_gamma, length=180,
                            command=lambda _: update_preview_debounced())
    gamma_slider.grid(row=0, column=1, padx=8, sticky="w")

    tk.Label(pop_frame, text="Pop Mid (0..1)").grid(row=0, column=2, sticky="w")
    pop_mid_entry = tk.Entry(pop_frame, width=6)
    pop_mid_entry.insert(0, f"{depth_pop_mid.get():.2f}")
    pop_mid_entry.grid(row=0, column=3, sticky="w")

    # Row 1
    tk.Label(pop_frame, text="Stretch Lo").grid(row=1, column=0, sticky="w")
    stretch_lo_entry = tk.Entry(pop_frame, width=6)
    stretch_lo_entry.insert(0, f"{depth_stretch_lo.get():.2f}")
    stretch_lo_entry.grid(row=1, column=1, sticky="w")

    tk.Label(pop_frame, text="Stretch Hi").grid(row=1, column=2, sticky="w")
    stretch_hi_entry = tk.Entry(pop_frame, width=6)
    stretch_hi_entry.insert(0, f"{depth_stretch_hi.get():.2f}")
    stretch_hi_entry.grid(row=1, column=3, sticky="w")

    # Row 2
    tk.Label(pop_frame, text="FG Pop ×").grid(row=2, column=0, sticky="w")
    fg_pop_slider = tk.Scale(pop_frame, from_=1.00, to=1.60, resolution=0.01,
                             orient="horizontal", variable=fg_pop_multiplier, length=180,
                             command=lambda _: update_preview_debounced())
    fg_pop_slider.grid(row=2, column=1, padx=8, sticky="w")

    tk.Label(pop_frame, text="BG Push ×").grid(row=2, column=2, sticky="w")
    bg_push_slider = tk.Scale(pop_frame, from_=1.00, to=1.40, resolution=0.01,
                              orient="horizontal", variable=bg_push_multiplier, length=180,
                              command=lambda _: update_preview_debounced())
    bg_push_slider.grid(row=2, column=3, padx=8, sticky="w")

    # Row 3
    tk.Label(pop_frame, text="Subject Lock").grid(row=3, column=0, sticky="w")
    subj_lock_slider = tk.Scale(pop_frame, from_=0.00, to=2.00, resolution=0.05,
                                orient="horizontal", variable=subject_lock_strength, length=180,
                                command=lambda _: update_preview_debounced())
    subj_lock_slider.grid(row=3, column=1, padx=8, sticky="w")

    # --- Color Grading UI ---
    color_frame = tk.LabelFrame(control_container, text="Color Grading", padx=10, pady=8)
    color_frame.pack(pady=(0, 10), anchor="center")
    color_frame.columnconfigure((0,1,2,3), weight=1)

    tk.Label(color_frame, text="Saturation").grid(row=0, column=0, sticky="w")
    sat_slider = tk.Scale(
        color_frame, from_=0.0, to=2.0, resolution=0.05, orient="horizontal",
        variable=saturation, length=180,
        command=lambda _=None: update_preview_debounced()
    )
    sat_slider.grid(row=0, column=1, padx=8, sticky="w")

    tk.Label(color_frame, text="Contrast").grid(row=0, column=2, sticky="w")
    con_slider = tk.Scale(
        color_frame, from_=0.0, to=2.0, resolution=0.05, orient="horizontal",
        variable=contrast, length=180,
        command=lambda _=None: update_preview_debounced()
    )
    con_slider.grid(row=0, column=3, padx=8, sticky="w")

    tk.Label(color_frame, text="Brightness").grid(row=1, column=0, sticky="w")
    bri_slider = tk.Scale(
        color_frame, from_=-0.5, to=0.5, resolution=0.01, orient="horizontal",
        variable=brightness, length=180,
        command=lambda _=None: update_preview_debounced()
    )
    bri_slider.grid(row=1, column=1, padx=8, sticky="w")

    # Commit button for the Entry fields
    def _commit_pop_entries():
        try:
            depth_pop_mid.set(float(pop_mid_entry.get()))
            lo = float(stretch_lo_entry.get())
            hi = float(stretch_hi_entry.get())
            lo = max(0.0, min(1.0, lo))
            hi = max(0.0, min(1.0, hi))
            if hi <= lo:
                messagebox.showwarning("Invalid Input", "Stretch Hi must be > Stretch Lo.")
                return
            depth_stretch_lo.set(lo)
            depth_stretch_hi.set(hi)
            update_preview_debounced()
        except ValueError:
            messagebox.showwarning("Invalid Input", "Enter numeric values for Mid/Lo/Hi in 0..1.")

    tk.Button(pop_frame, text="Apply Entries", command=_commit_pop_entries).grid(row=3, column=3, sticky="e")

    def update_max_shift(*_):
        try:
            val = float(max_shift_entry.get())
            max_pixel_shift.set(val)
            update_preview_debounced()
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid float for max pixel shift.")

    def update_dof_strength(*_):
        try:
            val = float(dof_strength_entry.get())
            dof_strength.set(val)
            update_preview_debounced()
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid float for DoF Strength.")

    def save_preview(image_to_save):
        nonlocal frame, depth
        if image_to_save is None or frame is None or depth is None:
            messagebox.showinfo("No Preview", "Nothing to save yet.")
            return

        preview_type = preview_type_var.get().replace(" ", "_")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"preview_{preview_type}_{timestamp}"

        save_dir = filedialog.askdirectory(title="Select folder to save preview and references")
        if not save_dir:
            return

        preview_path = os.path.join(save_dir, base_name + "_preview.png")
        input_path   = os.path.join(save_dir, base_name + "_input.png")
        depth_path   = os.path.join(save_dir, base_name + "_depth.png")

        cv2.imwrite(preview_path, image_to_save)
        cv2.imwrite(input_path, frame)
        cv2.imwrite(depth_path, depth)
        messagebox.showinfo("Saved", f"Files saved to:\n{save_dir}")

    dof_strength_entry.bind("<KeyRelease>", update_dof_strength)
    sharpness_slider.bind("<KeyRelease>", lambda _e: update_preview_debounced())
    max_shift_entry.bind("<KeyRelease>", update_max_shift)

    def update_convergence_strength(*_):
        try:
            val = float(convergence_slider.get())
            convergence_strength.set(val)
            update_preview_debounced()
        except ValueError:
            messagebox.showwarning("Invalid Input", "Enter a valid float for convergence strength.")

    convergence_slider.bind("<KeyRelease>", update_convergence_strength)

    def update_preview_now():
        nonlocal preview_job
        nonlocal preview_img, frame, depth
        preview_job = None 

        try:
            frame_idx = frame_slider.get()
            input_path = input_video_path.get()
            depth_path = selected_depth_map.get()
        except (tk.TclError, AttributeError):
            return

        if not input_path or not depth_path:
            return

        frame = get_frame(preview_cap, frame_idx)

        depth = None
        if depth_cap and depth_cap.isOpened():
            if depth_total > 0 and frame_idx >= depth_total:
                frame_idx = depth_total - 1
            depth = get_frame(depth_cap, frame_idx)

        if frame is None:
            messagebox.showerror("Frame Error", "Unable to grab frame from INPUT video at this index.")
            return
        if depth is None:
            messagebox.showerror("Frame Error", "Unable to grab frame from DEPTH video at this index. (Depth is shorter or unreadable at this position.)")
            return

        h, w = frame.shape[:2]
        frame_tensor = F.interpolate(frame_to_tensor(frame).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        depth_tensor = F.interpolate(depth_to_tensor(depth).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

        try:
            max_pixel_shift_val = float(max_pixel_shift.get())
            zero_parallax_strength_val = float(zero_parallax_strength.get())
            parallax_balance_val = float(parallax_balance.get())
            dof_strength_val = float(dof_strength.get())
            sharpness_val = float(sharpness_factor.get())
            convergence_strength_val = float(convergence_strength.get())
            dynamic_convergence_enabled = enable_dynamic_convergence.get()
        except ValueError:
            messagebox.showwarning("Input Error", "One or more numeric settings are invalid.")
            return
        # Build IPD-aware layer shifts
        fg_val = float(fg_shift.get())
        mg_val = float(mg_shift.get())
        bg_val = float(bg_shift.get())

        if ipd_enabled.get():
            factor = float(ipd_scale.get())
            fg_val *= factor
            mg_val *= factor
            bg_val *= factor
        # Call pixel_shift_cuda with convergence and pop controls
        left_tensor, right_tensor, shift_map = pixel_shift_cuda(
            frame_tensor, depth_tensor, w, h,
            fg_val, mg_val, bg_val, 
            return_shift_map=True,
            use_subject_tracking=use_subject_tracking.get(),
            enable_floating_window=use_floating_window.get(),
            max_pixel_shift_percent=max_pixel_shift_val,
            zero_parallax_strength=zero_parallax_strength_val,
            parallax_balance=parallax_balance_val,
            enable_edge_masking=enable_edge_masking.get(),
            enable_feathering=enable_feathering.get(),
            dof_strength=dof_strength_val,
            convergence_strength=convergence_strength_val,
            enable_dynamic_convergence=dynamic_convergence_enabled,
            # pop controls
            depth_pop_gamma=depth_pop_gamma.get(),
            depth_pop_mid=depth_pop_mid.get(),
            depth_stretch_lo=depth_stretch_lo.get(),
            depth_stretch_hi=depth_stretch_hi.get(),
            fg_pop_multiplier=fg_pop_multiplier.get(),
            bg_push_multiplier=bg_push_multiplier.get(),
            subject_lock_strength=subject_lock_strength.get(),
        )

        # Convert tensors to frames
        left_frame = tensor_to_frame(left_tensor) if isinstance(left_tensor, torch.Tensor) else left_tensor
        right_frame = tensor_to_frame(right_tensor) if isinstance(right_tensor, torch.Tensor) else right_tensor

        # Sharpen per-eye
        left_frame = apply_sharpening(left_frame, sharpness_val)
        right_frame = apply_sharpening(right_frame, sharpness_val)

        # Color grade per-eye
        lt = frame_to_tensor(left_frame)   # [3,H,W] float 0..1
        rt = frame_to_tensor(right_frame)

        lt = apply_color_grade(
            lt,
            saturation=saturation.get(),
            contrast=contrast.get(),
            brightness=brightness.get()
        )
        rt = apply_color_grade(
            rt,
            saturation=saturation.get(),
            contrast=contrast.get(),
            brightness=brightness.get()
        )

        left_frame  = tensor_to_frame(lt)
        right_frame = tensor_to_frame(rt)

        # Format output
        preview_mode = preview_type_var.get()
        if preview_mode in ("HSBS", "Half-SBS"):
            preview_img = format_3d_output(left_frame, right_frame, "Half-SBS")
        elif preview_mode in (
            "Red-Blue Anaglyph", "Passive Interlaced", "Overlay Arrows",
            "Shift Heatmap",  "Shift Heatmap (Abs)", "Shift Heatmap (Clipped ±5px)",
            "Feather Mask", "Feather Blend", "Left-Right Diff", "Shift Map Grayscale",
        ):
            preview_img = generate_preview_image(preview_mode, left_frame, right_frame, shift_map, w, h)
        else:
            preview_img = generate_preview_image(preview_mode, left_tensor, right_tensor, shift_map, w, h)

        # Resize and render
        if preview_img is not None:
            img_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            try:
                preview_width = int(width_entry.get())
                preview_height = int(height_entry.get())
            except ValueError:
                preview_width, preview_height = img_rgb.shape[1], img_rgb.shape[0]

            pil_img = Image.fromarray(img_rgb).resize((preview_width, preview_height), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(pil_img)
            preview_canvas.config(image=img_tk)
            preview_canvas.image = img_tk

    def on_close():
        settings = {
            'width': width_entry.get(),
            'height': height_entry.get(),
            'preview_type': preview_type_var.get(),
            'fg_shift': fg_shift.get(),
            'mg_shift': mg_shift.get(),
            'bg_shift': bg_shift.get(),
            'blur_ksize': blur_ksize.get(),
            'feather_strength': feather_strength.get(),
            'sharpening': sharpness_factor.get(),
            'max_pixel_shift': max_pixel_shift.get(),
            'zero_parallax_strength': zero_parallax_strength.get(),
            'parallax_balance': parallax_balance.get(),
            'enable_edge_masking': enable_edge_masking.get(),
            'enable_feathering': enable_feathering.get(),
            'dof_strength': dof_strength.get(),
            'convergence_strength': convergence_strength.get(),
            'enable_dynamic_convergence': enable_dynamic_convergence.get(),
            'depth_pop_gamma': depth_pop_gamma.get(),
            'depth_pop_mid': depth_pop_mid.get(),
            'depth_stretch_lo': depth_stretch_lo.get(),
            'depth_stretch_hi': depth_stretch_hi.get(),
            'fg_pop_multiplier': fg_pop_multiplier.get(),
            'bg_push_multiplier': bg_push_multiplier.get(),
            'subject_lock_strength': subject_lock_strength.get(),
            # persist color grade
            'saturation': saturation.get(),
            'contrast':   contrast.get(),
            'brightness': brightness.get(),
            'ipd_enabled': ipd_enabled.get(),
            'ipd_scale':   ipd_scale.get(),

        }
        save_settings(settings)
        try:
            if preview_cap: preview_cap.release()
        except: pass
        try:
            if depth_cap: depth_cap.release()
        except: pass
        preview_win.destroy()

    preview_win.protocol("WM_DELETE_WINDOW", on_close)

    # Reactive bindings
    frame_slider.config(command=update_preview_debounced)
    fg_slider.config(command=update_preview_debounced)
    mg_slider.config(command=update_preview_debounced)
    bg_slider.config(command=update_preview_debounced)
    zero_parallax_strength_slider.config(command=update_preview_debounced)
    parallax_balance_slider.config(command=update_preview_debounced)
    convergence_slider.bind("<KeyRelease>", update_convergence_strength)
    preview_type_var.trace_add("write", lambda *_: update_preview_debounced())
    enable_edge_masking.trace_add("write", lambda *_: update_preview_debounced())
    enable_feathering.trace_add("write", lambda *_: update_preview_debounced())
    enable_dynamic_convergence.trace_add("write", lambda *_: update_preview_debounced())
    sharpness_factor.trace_add("write", lambda *_: update_preview_debounced())
    depth_pop_gamma.trace_add("write", lambda *_: update_preview_debounced())
    fg_pop_multiplier.trace_add("write", lambda *_: update_preview_debounced())
    bg_push_multiplier.trace_add("write", lambda *_: update_preview_debounced())
    subject_lock_strength.trace_add("write", lambda *_: update_preview_debounced())
    ipd_enabled.trace_add("write", lambda *_: update_preview_debounced())
    ipd_scale.trace_add("write",   lambda *_: update_preview_debounced())

    # color vars already trigger updates via their Scale command callbacks

    pop_mid_entry.bind("<Return>", lambda _e: _commit_pop_entries())
    stretch_lo_entry.bind("<Return>", lambda _e: _commit_pop_entries())
    stretch_hi_entry.bind("<Return>", lambda _e: _commit_pop_entries())

    _resize_slider()
    apply_size()
