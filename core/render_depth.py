import os
import re
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import torch
import cv2
import matplotlib.cm as cm

from transformers import AutoProcessor, AutoModelForDepthEstimation, pipeline

global pipe
suspend_flag = threading.Event()
cancel_flag = threading.Event()
cancel_requested = threading.Event()

# ---Depth Estimation---

# ✅ Define local model storage path
local_model_dir = os.path.expanduser(
    "~/.cache/huggingface/models/"
)  # Correct local path

# -----------------------
# Functions
# -----------------------

supported_models = {
    "Distil-Any-Depth-Large": "xingyang1/Distill-Any-Depth-Large-hf",
    "Distil-Any-Depth-Small": "xingyang1/Distill-Any-Depth-Small-hf",
    "Depth Anything V2 Large": "depth-anything/Depth-Anything-V2-Large-hf",
    "Depth Anything V2 Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth Anything V2 Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth Anything V1 Large": "LiheYoung/depth-anything-large-hf",
    "Depth Anything V1 Base": "LiheYoung/depth-anything-base-hf",
    "Depth Anything V1 Small": "LiheYoung/depth-anything-small-hf",
    "V2-Metric-Indoor-Large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "V2-Metric-Outdoor-Large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "DA_vitl14": "LiheYoung/depth_anything_vitl14",
    "DA_vits14": "LiheYoung/depth_anything_vits14",
    "DepthPro": "apple/DepthPro-hf",
    "Sura": "Seraph19/Sura",
    "rock-depth-ai": "justinsoberano/rock-depth-ai",
    "ZoeDepth": "Intel/zoedepth-nyu-kitti",
    "MiDaS 3.0": "Intel/dpt-hybrid-midas",
    "DPT-Large": "Intel/dpt-large",
    "DinoV2": "facebook/dpt-dinov2-small-kitti",
    "dpt-beit-large-512": "Intel/dpt-beit-large-512",
}

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def ensure_model_downloaded(checkpoint):
    """Ensures the model is downloaded locally before loading."""
    local_path = os.path.join(
        local_model_dir, checkpoint.replace("/", "_")
    )  # Convert to valid folder name

    if not os.path.exists(local_path):
        print(f"📥 Downloading model: {checkpoint} ... This may take a few minutes.")
        try:
            # ✅ Download model & processor to local storage
            AutoModelForDepthEstimation.from_pretrained(
                checkpoint, cache_dir=local_path
            )
            AutoProcessor.from_pretrained(checkpoint, cache_dir=local_path)
            print(f"✅ Model downloaded successfully: {checkpoint}")
        except Exception as e:
            print(f"❌ Failed to download model: {str(e)}")
            return None  # Prevent using a broken model

    return local_path  # Return the local path to be used


def update_pipeline(selected_model_var, status_label_widget, *args):
    """Loads the depth estimation model from local cache."""
    global pipe

    selected_checkpoint = selected_model_var.get()
    checkpoint = supported_models.get(selected_checkpoint, None)

    if checkpoint is None:
        status_label_widget.config(text=f"⚠️ Error: Model '{selected_checkpoint}' not found.")
        return

    try:
        # ✅ Ensure model is available locally
        local_model_path = ensure_model_downloaded(checkpoint)
        if not local_model_path:
            status_label_widget.config(text=f"❌ Failed to load model: {selected_checkpoint}")
            return

        # ✅ Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ Load image processor & model locally
        processor = AutoProcessor.from_pretrained(
            checkpoint,
            cache_dir=local_model_path,
            use_fast=True  # ✅ Enables fast processor and suppresses future warning
        )

        model = AutoModelForDepthEstimation.from_pretrained(
            checkpoint,
            cache_dir=local_model_path,
        ).to(device)

        # ✅ Load the depth-estimation pipeline
        pipe = pipeline(
            "depth-estimation", model=model, device=device, image_processor=processor
        )

        # ✅ Update status label to show only the model name
        status_label_widget.config(
            text=f"✅ Model loaded: {selected_checkpoint} (Running on {device.upper()})"
        )
        status_label_widget.update_idletasks()  # Force label update in Tkinter

    except Exception as e:
        status_label_widget.config(text=f"❌ Model loading failed: {str(e)}")
        status_label_widget.update_idletasks()  # Ensure GUI updates


def choose_output_directory(output_label_widget, output_dir_var):
    selected_directory = filedialog.askdirectory()
    if selected_directory:
        output_dir_var.set(selected_directory)
        output_label_widget.config(text=f"📁 {selected_directory}")

def get_dynamic_batch_size(base=4, scale_factor=1.0, max_limit=32, reserve_vram_gb=1.0):
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024 ** 3)
        usable_vram = max(0, total_vram - reserve_vram_gb)
        estimated_batch = int(base * usable_vram * scale_factor)
        return min(estimated_batch, max_limit)
    return base

def process_image_folder(batch_size_widget, output_dir_var, status_label, progress_bar, root):
    folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
    if not folder_path:
        cancel_requested.clear()  # ✅ Reset before starting
        status_label.config(text="⚠️ No folder selected.")
        return

    threading.Thread(
        target=process_images_in_folder,
        args=(folder_path, batch_size_widget, output_dir_var, status_label, progress_bar, root, cancel_requested),
        daemon=True,
    ).start()

def process_images_in_folder(folder_path, batch_size_widget, output_dir_var, status_label, progress_bar, root, cancel_requested):
    try:
        user_value = batch_size_widget.get().strip()
        batch_size = int(user_value) if user_value else get_dynamic_batch_size()
        if batch_size <= 0:
            raise ValueError
    except Exception:
        batch_size = get_dynamic_batch_size()
        status_label.config(text=f"⚠️ Invalid batch size. Using dynamic batch size: {batch_size}")
        
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    ]

    if not image_files:
        root.after(10, lambda: status_label.config(text="⚠️ No image files found."))
        return

    total_images = len(image_files)
    root.after(
        10, lambda: status_label.config(text=f"📂 Processing {total_images} images...")
    )
    root.after(10, lambda: progress_bar.config(maximum=total_images, value=0))

    start_time = time.time()

    # ✅ Process in batches using dynamically set batch size
    for i in range(0, total_images, batch_size):
        if cancel_requested.is_set():
            root.after(10, lambda: status_label.config(text="❌ Cancelled by user."))
            return

        batch_files = image_files[i : i + batch_size]

        # Load all images into memory
        images = [Image.open(file).convert("RGB") for file in batch_files]

        # ✅ Run batch inference (GPU is utilized more efficiently)
        print(f"🚀 Running batch of {len(images)} images")
        predictions = pipe(images)


        for j, prediction in enumerate(predictions):
            if cancel_requested.is_set():
                root.after(10, lambda: status_label.config(text="❌ Cancelled during batch."))
                return

            file_path = batch_files[j]
            raw_depth = prediction["predicted_depth"]

            # Normalize depth
            depth_norm = (raw_depth - raw_depth.min()) / (
                raw_depth.max() - raw_depth.min()
            )
            depth_tensor = depth_norm.squeeze()

            # ✅ Convert to NumPy before saving
            depth_np = (depth_tensor * 255).cpu().numpy().astype(np.uint8)

            # Save output
            image_name = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"{image_name}_depth.png"
            file_save_path = (
                os.path.join(output_dir_var.get(), output_filename)
                if output_dir_var
                else output_filename
            )
            Image.fromarray(depth_np).save(file_save_path)

            # ✅ Update progress dynamically
            elapsed_time = time.time() - start_time
            fps = (i + j + 1) / elapsed_time if elapsed_time > 0 else 0
            eta = (total_images - (i + j + 1)) / fps if fps > 0 else 0

            status_label.after(
                10,
                lambda i=i + j + 1, fps=fps, eta=eta: update_progress(
                    i, total_images, fps, eta, progress_bar, status_label
                ),
            )

    root.after(
        10, lambda: status_label.config(text="✅ All images processed successfully!")
    )
    root.after(10, lambda: progress_bar.config(value=progress_bar["maximum"]))


def update_progress(processed, total, fps, eta, progress_bar, status_label):
    progress_bar.config(value=processed)

    # Format FPS and ETA
    fps_text = f"{fps:.2f} FPS"
    eta_text = f"ETA: {int(eta)}s" if eta > 0 else "ETA: --s"
    progress_text = f"📸 Processed: {processed}/{total} | {fps_text} | {eta_text}"

    status_label.config(text=progress_text)


def process_image(file_path, colormap_var, invert_var, output_dir_var, input_label, output_label, status_label, progress_bar, folder=False):
    """Processes a single image file and saves the depth-mapped version."""
    image = Image.open(file_path)
    predictions = pipe(image)

    if "predicted_depth" in predictions:
        raw_depth = predictions["predicted_depth"]
        depth_norm = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min())
        depth_tensor = depth_norm.squeeze()  # Keep it on GPU
        depth_np = depth_tensor.mul(
            255
        ).byte()  # Direct conversion (GPU stays utilized)

        cmap_choice = colormap_var.get()
        if cmap_choice == "Default":
            depth_image = predictions["depth"]
        else:
            cmap = cm.get_cmap(cmap_choice.lower())
            depth_np_float = depth_tensor.cpu().numpy() / 255.0  # Normalize for colormap
            colored = cmap(depth_np_float)
            colored = (colored[:, :, :3] * 255).astype(np.uint8)
            depth_image = Image.fromarray(colored)

    else:
        depth_image = predictions["depth"]

    if invert_var.get():
        print("Inversion enabled")
        depth_image = ImageOps.invert(depth_image.convert("RGB"))

    # ✅ If processing a single image, update GUI preview
    if not folder:
        image_disp = image.copy()
        image_disp.thumbnail((480, 270))
        photo_input = ImageTk.PhotoImage(image_disp)
        input_label.config(image=photo_input)
        input_label.image = photo_input

        depth_disp = depth_image.copy()
        depth_disp.thumbnail((480, 270))
        photo_depth = ImageTk.PhotoImage(depth_disp)
        output_label.config(image=photo_depth)
        output_label.image = photo_depth

    # ✅ Save output to the selected directory
    image_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{image_name}_depth.png"
    file_save_path = (
        os.path.join(output_dir_var.get(), output_filename)
        if output_dir_var.get()
        else output_filename
    )
    depth_image.save(file_save_path)

    if not folder:
        cancel_requested.clear()  # ✅ Reset before starting
        status_label.config(text=f"✅ Image saved: {file_save_path}")
        progress_bar.config(value=100)


def open_image(status_label_widget, progress_bar_widget, colormap_var, invert_var, output_dir_var, input_label_widget, output_label_widget):
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpeg;*.jpg;*.png")]
    )
    if file_path:
        cancel_requested.clear()  # ✅ Reset before starting
        status_label_widget.config(text="🔄 Processing image...")
        progress_bar_widget.start(10)
        threading.Thread(
            target=lambda: [
                process_image(
                    file_path,
                    colormap_var,
                    invert_var,
                    output_dir_var,
                    input_label_widget,
                    output_label_widget,
                    status_label_widget,
                    progress_bar_widget,
                ),
                progress_bar_widget.stop()
            ],
            daemon=True,
        ).start()


def process_video_folder(folder_path, batch_size_widget, output_dir_var, status_label, progress_bar, cancel_requested):
    """Opens a folder dialog and processes all video files inside it in a background thread."""
    folder_path = filedialog.askdirectory(title="Select Folder Containing Videos")

    if not folder_path:
        cancel_requested.clear()  # ✅ Reset before starting
        status_label.config(text="⚠️ No folder selected.")
        return

    # ✅ Run processing in a separate thread
    threading.Thread(
        target=process_videos_in_folder,
        args=(folder_path, batch_size_widget, output_dir_var, status_label, progress_bar, cancel_requested),
        daemon=True,
    ).start()



def natural_sort_key(filename):
    """Extract numbers from filenames for natural sorting."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", filename)
    ]

def process_videos_in_folder(
    folder_path,
    batch_size_widget,
    output_dir_var,
    status_label,
    progress_bar,
    cancel_requested,
    invert_var,
    save_frames=False,    # Optional
):

    """Processes all video files in the selected folder in the correct numerical order."""
    video_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        status_label.config(text="⚠️ No video files found in the selected folder.")
        return

    # ✅ Sort videos numerically
    video_files.sort(key=natural_sort_key)

    status_label.config(text=f"📂 Processing {len(video_files)} videos...")

    total_frames_all = sum(
        int(cv2.VideoCapture(os.path.join(folder_path, f)).get(cv2.CAP_PROP_FRAME_COUNT))
        for f in video_files
    )
    frames_processed_all = 0

    # ✅ Retrieve batch size safely from widget
    try:
        user_value = batch_size_widget.get().strip()
        batch_size = int(user_value) if user_value else get_dynamic_batch_size()
        if batch_size <= 0:
            raise ValueError
    except Exception:
        batch_size = get_dynamic_batch_size()
        status_label.config(text=f"⚠️ Invalid batch size. Using dynamic batch size: {batch_size}")

    # ✅ Process each video with updated pipeline
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        processed = process_video2(
            video_path,
            total_frames_all,
            frames_processed_all,
            batch_size,
            output_dir_var,
            status_label,
            progress_bar,
            cancel_requested,
            invert_var
        )

        # If cancelled mid-batch, exit early
        if cancel_requested.is_set():
            status_label.config(text="🛑 Processing cancelled by user.")
            progress_bar.config(value=0)
            return

        frames_processed_all += processed

    status_label.config(text="✅ All videos processed successfully!")
    progress_bar.config(value=100)
    
def process_video2(
    file_path,
    total_frames_all,
    frames_processed_all,
    batch_size,
    output_dir_var,
    status_label,
    progress_bar,
    cancel_requested,
    invert_var,
    save_frames=False
):
    """Processes a single video file and updates the progress bar correctly."""

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        status_label.config(text=f"❌ Error: Cannot open {file_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = output_dir_var.get()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_dir, input_filename = os.path.split(file_path)
    name, _ = os.path.splitext(input_filename)
    output_filename = f"{name}_depth.mkv"
    output_path = os.path.join(output_dir, output_filename)

    print(f"📁 Saving video to: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

    if not out.isOpened():
        print(f"❌ Failed to open video writer for {output_filename}")
        return 0

    frame_output_dir = os.path.join(output_dir, f"{name}_frames")
    if save_frames and not os.path.exists(frame_output_dir):
        os.makedirs(frame_output_dir)

    frame_count = 0
    start_time = time.time()
    frames_batch = []

    while True:
        if cancel_requested.is_set():
            print("🛑 Cancel requested before frame read.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames_batch.append(pil_image)

        if len(frames_batch) == batch_size or frame_count == total_frames:
            if cancel_requested.is_set():
                print("🛑 Cancel requested before inference.")
                break

            predictions = pipe(frames_batch)

            for i, prediction in enumerate(predictions):
                if cancel_requested.is_set():
                    print("🛑 Cancelled during batch write.")
                    status_label.config(text="🛑 Cancelled during batch.")
                    cap.release()
                    out.release()
                    return frame_count

                raw_depth = prediction["predicted_depth"]
                depth_tensor = ((raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min())).squeeze()

                if invert_var.get():
                    print("🔁 Invert depth selected.")
                    depth_tensor = 1.0 - depth_tensor

                depth_tensor = depth_tensor.mul(255).byte()
                depth_np = depth_tensor.cpu().numpy().astype("uint8")
                depth_frame = cv2.cvtColor(depth_np, cv2.COLOR_GRAY2BGR)
                depth_frame = cv2.resize(depth_frame, (original_width, original_height))
                out.write(depth_frame)

                if save_frames:
                    frame_idx = frame_count - len(predictions) + i + 1
                    frame_filename = os.path.join(frame_output_dir, f"frame_{frame_idx}.png")
                    cv2.imwrite(frame_filename, depth_frame)

            frames_batch.clear()

        # 🟡 Progress Bar Update
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        remaining_frames = total_frames_all - (frames_processed_all + frame_count)
        eta = remaining_frames / avg_fps if avg_fps > 0 else 0

        progress = int(((frames_processed_all + frame_count) / total_frames_all) * 100)
        progress_bar.config(value=progress)
        status_label.config(
            text=f"🎬 {frame_count}/{total_frames} frames | FPS: {avg_fps:.2f} | "
                 f"ETA: {time.strftime('%M:%S', time.gmtime(eta))} | Processing: {name}"
        )
        status_label.update_idletasks()

    cap.release()
    out.release()

    if cancel_requested.is_set():
        print("🛑 Cancelled: Video not fully processed.")
    else:
        print(f"✅ Video saved: {output_path}")

    return frame_count




def open_video(status_label, progress_bar, batch_size_widget, output_dir_var, invert_var):
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("All Supported Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv;*.webm;*.mpeg;*.mpg"),
            ("MP4 Files", "*.mp4"),
            ("AVI Files", "*.avi"),
            ("MOV Files", "*.mov"),
            ("MKV Files", "*.mkv"),
            ("FLV Files", "*.flv"),
            ("WMV Files", "*.wmv"),
            ("WebM Files", "*.webm"),
            ("MPEG Files", "*.mpeg;*.mpg"),
            ("All Files", "*.*"),
        ]
    )

    if file_path:
        cancel_requested.clear()  # ✅ Reset before starting
        status_label.config(text="🔄 Processing video...")
        progress_bar.config(mode="determinate", maximum=100, value=0)

        cap = cv2.VideoCapture(file_path)
        total_frames_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        try:
            user_value = batch_size_widget.get().strip()
            batch_size = int(user_value) if user_value else get_dynamic_batch_size()
            if batch_size <= 0:
                raise ValueError
        except Exception:
            batch_size = get_dynamic_batch_size()
            status_label.config(text=f"⚠️ Invalid batch size. Using dynamic batch size: {batch_size}")

        threading.Thread(
            target=process_video2,
            args=(file_path, total_frames_all, 0, batch_size, output_dir_var, status_label, progress_bar, cancel_requested, invert_var),
            daemon=True
        ).start()

