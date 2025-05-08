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
import onnxruntime as ort
import subprocess

from transformers import AutoProcessor, AutoModelForDepthEstimation, pipeline

global pipe
suspend_flag = threading.Event()
cancel_flag = threading.Event()
cancel_requested = threading.Event()
global_session_start_time = None


# === Setup: Local weights directory ===
local_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))
os.makedirs(local_model_dir, exist_ok=True)

# === Suppress Hugging Face symlink warnings (esp. on Windows) ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def load_supported_models():
    models = {
        "  -- Select Model -- ": "  -- Select Model -- ",
        "Video Depth Anything": os.path.join(local_model_dir, "Video Depth Anything"),
        "Distill Any Depth Base": os.path.join(local_model_dir, "Distill Any Depth Base"),
        "Distil-Any-Depth-Large": "xingyang1/Distill-Any-Depth-Large-hf",
        "Distil-Any-Depth-Small": "xingyang1/Distill-Any-Depth-Small-hf",
        "keetrap-Distil-Any-Depth-Large": "keetrap/Distil-Any-Depth-Large-hf",
        "keetrap-Distil-Any-Depth-Small": "keetrap/Distill-Any-Depth-Small-hf",
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

    # Add local folders that look like model directories
    for folder in os.listdir(local_model_dir):
        folder_path = os.path.join(local_model_dir, folder)
        if os.path.isdir(folder_path):
            has_config = os.path.exists(os.path.join(folder_path, "config.json"))
            has_onnx = os.path.exists(os.path.join(folder_path, "model.onnx"))
            
            if has_config or has_onnx:
                models[f"[Local] {folder}"] = folder_path


    return models
    
supported_models = load_supported_models()

def ensure_model_downloaded(checkpoint):
    """
    Handles both Hugging Face checkpoints and local ONNX directories.
    """
    if os.path.isdir(checkpoint):
        # Local ONNX model detection
        if os.path.exists(os.path.join(checkpoint, "model.onnx")):
            print(f"🧠 Detected ONNX model in {checkpoint}")
            return load_onnx_model(checkpoint)

        # Local Hugging Face model
        try:
            model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
            processor = AutoProcessor.from_pretrained(checkpoint)
            print(f"📂 Loaded local Hugging Face model from {checkpoint}")
            return model, processor
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            return None, None

    # Hugging Face online model
    safe_folder_name = checkpoint.replace("/", "_")
    local_path = os.path.join(local_model_dir, safe_folder_name)
    try:
        model = AutoModelForDepthEstimation.from_pretrained(checkpoint, cache_dir=local_path)
        processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=local_path)
        print(f"⬇️ Downloaded model from Hugging Face: {checkpoint}")
        return model, processor
    except Exception as e:
        print(f"❌ Failed to load Hugging Face model: {e}")
        return None, None

def load_onnx_model(model_dir):
    model_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_path):
        print(f"❌ ONNX model file not found in {model_dir}")
        return None, None

    session = ort.InferenceSession(
        model_path,
        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_rank = len(input_shape)

    print(f"🔍 ONNX input shape: {input_shape} (Rank {input_rank})")

    def onnx_pipe(images):
        preds = []

        if input_rank == 5:
            # === Temporal model: expects (1, 32, 3, H, W)
            if len(images) != 32:
                if len(images) > 32:
                    raise ValueError("ONNX model requires exactly 32 frames per inference.")
                print(f"⚠️ Padding {len(images)} to 32 frames.")
                images += [images[-1]] * (32 - len(images))

            img_batch = [np.array(img.resize((518, 518))).astype(np.float32).transpose(2, 0, 1) / 255.0 for img in images]
            input_tensor = np.stack(img_batch)[None, ...]  # (1, 32, 3, H, W)

            result = session.run([output_name], {input_name: input_tensor})[0]  # (1, 32, H, W)
            result = result.squeeze(0)

            for i in range(32):
                preds.append({"predicted_depth": torch.tensor(result[i])})

        elif input_rank == 4:
            # === Spatial model: expects (N, 3, H, W)
            img_batch = [np.array(img.resize((518, 518))).astype(np.float32).transpose(2, 0, 1) / 255.0 for img in images]
            input_tensor = np.stack(img_batch)  # (N, 3, H, W)

            result = session.run([output_name], {input_name: input_tensor})[0]  # (N, H, W)

            for i in range(len(images)):
                preds.append({"predicted_depth": torch.tensor(result[i])})
        else:
            raise ValueError(f"❌ Unsupported input rank: {input_rank}")

        return preds

    return onnx_pipe, None




def update_pipeline(selected_model_var, status_label_widget, *args):
    global pipe

    selected_checkpoint = selected_model_var.get()
    checkpoint = supported_models.get(selected_checkpoint, None)

    if checkpoint is None:
        status_label_widget.config(text=f"⚠️ Error: Model '{selected_checkpoint}' not found.")
        return

    try:
        model, processor = ensure_model_downloaded(checkpoint)
        if not model:
            status_label_widget.config(text=f"❌ Failed to load model: {selected_checkpoint}")
            return

        device = 0 if torch.cuda.is_available() else -1

        if callable(model):
            # ONNX function pipeline
            pipe = model
            status_label_widget.config(text=f"✅ ONNX model loaded: {selected_checkpoint}")
        else:
            pipe = pipeline(
                "depth-estimation",
                model=model,
                image_processor=processor,
                device=device
            )
            status_label_widget.config(
                text=f"✅ HF model loaded: {selected_checkpoint} (Running on {'CUDA' if device == 0 else 'CPU'})"
            )

        status_label_widget.update_idletasks()

    except Exception as e:
        status_label_widget.config(text=f"❌ Model loading failed: {str(e)}")
        status_label_widget.update_idletasks()


def parse_inference_resolution(res_string, fallback=(384, 384)):
    res_string = res_string.strip().lower()
    if "original" in res_string or res_string == "--":
        return None
    try:
        return tuple(map(int, res_string.split("x")))
    except Exception:
        return fallback


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

def process_image_folder(batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, root):
    folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
    if not folder_path:
        cancel_requested.clear()  # ✅ Reset before starting
        status_label.config(text="⚠️ No folder selected.")
        return

    threading.Thread(
        target=process_images_in_folder,
        args=(folder_path, batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, root, cancel_requested),
        daemon=True,
    ).start()


def process_images_in_folder(folder_path, batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, root, cancel_requested):
    output_dir = output_dir_var.get().strip()
    global global_session_start_time
    if global_session_start_time is None:
        global_session_start_time = time.time()

    if not output_dir:
        messagebox.showwarning("Missing Output Folder", "⚠️ Please select an output directory before processing.")
        root.after(10, lambda: status_label.config(text="❌ Output directory not selected."))
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            messagebox.showerror("Folder Creation Failed", f"❌ Could not create output directory:\n{e}")
            root.after(10, lambda: status_label.config(text="❌ Failed to create output directory."))
            return

    # ✅ Get and parse resolution from dropdown
    inference_size = parse_inference_resolution(inference_res_var.get())
    
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
    root.after(10, lambda: status_label.config(text=f"📂 Processing {total_images} images..."))
    root.after(10, lambda: progress_bar.config(maximum=total_images, value=0))

    start_time = time.time()

    # ✅ Process in batches using resized input images
    for i in range(0, total_images, batch_size):
        if cancel_requested.is_set():
            root.after(10, lambda: status_label.config(text="❌ Cancelled by user."))
            return

        batch_files = image_files[i:i + batch_size]

        images = []
        original_sizes = []
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            original_sizes.append(img.size)
            images.append(img.resize(inference_size, Image.BICUBIC))

        print(f"🚀 Running batch of {len(images)} images at {inference_size}")
        predictions = pipe(images)

        for j, prediction in enumerate(predictions):
            if cancel_requested.is_set():
                root.after(10, lambda: status_label.config(text="❌ Cancelled during batch."))
                return

            file_path = batch_files[j]
            raw_depth = prediction["predicted_depth"]
            depth_norm = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min())
            depth_tensor = depth_norm.squeeze()
            depth_np = (depth_tensor * 255).cpu().numpy().astype(np.uint8)

            # ✅ Resize depth map back to original resolution
            orig_w, orig_h = original_sizes[j]
            depth_np = cv2.resize(depth_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            image_name = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"{image_name}_depth.png"
            file_save_path = os.path.join(output_dir, output_filename)
            Image.fromarray(depth_np).save(file_save_path)

            elapsed_time = time.time() - start_time
            fps = (i + j + 1) / elapsed_time if elapsed_time > 0 else 0
            eta = (total_images - (i + j + 1)) / fps if fps > 0 else 0

            status_label.after(10, lambda i=i + j + 1, fps=fps, eta=eta: update_progress(i, total_images, fps, eta, progress_bar, status_label))

    root.after(10, lambda: status_label.config(text="✅ All images processed successfully!"))
    root.after(10, lambda: progress_bar.config(value=progress_bar["maximum"]))



def update_progress(processed, total, fps, eta, progress_bar, status_label):
    progress_bar.config(value=processed)

    # Format FPS and ETA
    fps_text = f"{fps:.2f} FPS"
    eta_text = f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}" if eta > 0 else "ETA: --:--:--"
    progress_text = f"📸 Processed: {processed}/{total} | {fps_text} | {eta_text}"

    status_label.config(text=progress_text)


def process_image(file_path, colormap_var, invert_var, output_dir_var, inference_res_var, input_label, output_label, status_label, progress_bar, folder=False):
    """Processes a single image file and saves the depth-mapped version."""
    image = Image.open(file_path).convert("RGB")
    original_size = image.size

    inference_size = parse_inference_resolution(inference_res_var.get())

    # ✅ Resize input image for inference if required
    if inference_size:
        image_resized = image.resize(inference_size, Image.BICUBIC)
    else:
        image_resized = image.copy()  # Keep original resolution

    predictions = pipe(image_resized)

    if "predicted_depth" in predictions:
        raw_depth = predictions["predicted_depth"]
        depth_norm = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min())
        depth_tensor = depth_norm.squeeze()
        depth_np = (depth_tensor * 255).cpu().numpy().astype(np.uint8)

        # ✅ Resize depth map back to original resolution
        depth_np = cv2.resize(depth_np, original_size, interpolation=cv2.INTER_CUBIC)

        cmap_choice = colormap_var.get()
        if cmap_choice == "Default":
            depth_image = Image.fromarray(depth_np)
        else:
            cmap = cm.get_cmap(cmap_choice.lower())
            depth_np_float = depth_np.astype(np.float32) / 255.0
            colored = cmap(depth_np_float)
            colored = (colored[:, :, :3] * 255).astype(np.uint8)
            depth_image = Image.fromarray(colored)
    else:
        depth_image = predictions["depth"]

    if invert_var.get():
        print("Inversion enabled")
        depth_image = ImageOps.invert(depth_image.convert("RGB"))

    output_dir = output_dir_var.get().strip()
    if not output_dir:
        messagebox.showwarning("Missing Output Folder", "⚠️ Please select an output directory before saving.")
        status_label.config(text="❌ Output directory not selected.")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            messagebox.showerror("Folder Creation Failed", f"❌ Could not create output directory:\n{e}")
            status_label.config(text="❌ Failed to create output directory.")
            return

    # ✅ Update preview in GUI
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
    file_save_path = os.path.join(output_dir, output_filename)
    depth_image.save(file_save_path)

    if not folder:
        cancel_requested.clear()
        status_label.config(text=f"✅ Image saved: {file_save_path}")
        progress_bar.config(value=100)

def open_image(status_label_widget, progress_bar_widget, colormap_var, invert_var, output_dir_var, inference_res_var, input_label_widget, output_label_widget):
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
                    inference_res_var,  # ✅ Pass the selected inference resolution
                    input_label_widget,
                    output_label_widget,
                    status_label_widget,
                    progress_bar_widget,
                ),
                progress_bar_widget.stop()
            ],
            daemon=True,
        ).start()



def process_video_folder(folder_path, batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, cancel_requested):
    """Opens a folder dialog and processes all video files inside it in a background thread."""
    folder_path = filedialog.askdirectory(title="Select Folder Containing Videos")

    if not folder_path:
        cancel_requested.clear()  # ✅ Reset before starting
        status_label.config(text="⚠️ No folder selected.")
        return

    # ✅ Run processing in a separate thread with resolution
    threading.Thread(
        target=process_videos_in_folder,
        args=(folder_path, batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, cancel_requested),
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
    inference_res_var,
    status_label,
    progress_bar,
    cancel_requested,
    invert_var,
    save_frames=False,
):
    """Processes all video files in the selected folder in the correct numerical order."""
    video_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        status_label.config(text="⚠️ No video files found in the selected folder.")
        return
    
    video_files.sort(key=natural_sort_key)

    status_label.config(text=f"📂 Processing {len(video_files)} videos...")
    global global_session_start_time
    if global_session_start_time is None:
        global_session_start_time = time.time()

    total_frames_all = sum(
        int(cv2.VideoCapture(os.path.join(folder_path, f)).get(cv2.CAP_PROP_FRAME_COUNT))
        for f in video_files
    )
    frames_processed_all = 0

    try:
        user_value = batch_size_widget.get().strip()
        batch_size = int(user_value) if user_value else get_dynamic_batch_size()
        if batch_size <= 0:
            raise ValueError
    except Exception:
        batch_size = get_dynamic_batch_size()
        status_label.config(text=f"⚠️ Invalid batch size. Using dynamic batch size: {batch_size}")

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        processed = process_video2(
            video_path,
            total_frames_all,
            frames_processed_all,
            batch_size,
            output_dir_var,
            inference_res_var,
            status_label,
            progress_bar,
            cancel_requested,
            invert_var,
            save_frames
        )

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
    inference_res_var,
    status_label,
    progress_bar,
    cancel_requested,
    invert_var,
    save_frames=False
):
    global pipe
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        status_label.config(text=f"❌ Error: Cannot open {file_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = output_dir_var.get().strip()
    if not output_dir:
        messagebox.showwarning("Missing Output Folder", "⚠️ Please select an output directory before processing.")
        status_label.config(text="❌ Output directory not selected.")
        return 0

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            messagebox.showerror("Folder Creation Failed", f"❌ Could not create output directory:\n{e}")
            status_label.config(text="❌ Failed to create output directory.")
            return 0

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
    frames_batch = []

    inference_size = parse_inference_resolution(inference_res_var.get())

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
        
        if inference_size:
            pil_image = pil_image.resize(inference_size, Image.BICUBIC)

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

        global global_session_start_time
        elapsed = time.time() - global_session_start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        remaining_frames = total_frames_all - (frames_processed_all + frame_count)
        eta = remaining_frames / avg_fps if avg_fps > 0 else 0

        progress = int(((frames_processed_all + frame_count) / total_frames_all) * 100)
        progress_bar.config(value=progress)
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))

        status_label.config(
            text=f"🎬 {frame_count}/{total_frames} frames | FPS: {avg_fps:.2f} | "
                 f"Elapsed: {elapsed_str} | ETA: {eta_str} | Processing: {name}"
        )
        status_label.update_idletasks()

    cap.release()
    out.release()

    if cancel_requested.is_set():
        print("🛑 Cancelled: Video not fully processed.")
    else:
        print(f"✅ Video saved: {output_path}")

    return frame_count

def is_av1_encoded(file_path):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=nokey=1:noprint_wrappers=1",
                file_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        codec = result.stdout.strip().lower()
        return "av1" in codec
    except Exception as e:
        print(f"⚠️ Failed to check codec with ffprobe: {e}")
        return False


def open_video(status_label, progress_bar, batch_size_widget, output_dir_var, inference_res_var, invert_var):
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

    global global_session_start_time
    if global_session_start_time is None:
        global_session_start_time = time.time()

    if file_path:
        # 🔍 Detect AV1 codec
        if is_av1_encoded(file_path):
            messagebox.showwarning(
                "Unsupported AV1 Input",
                "🚫 This video is encoded with AV1, which is not supported by OpenCV in this application.\n\n"
                "Please re-encode it to H.264 using:\n\nffmpeg -i input.mkv -c:v libx264 output.mp4"
            )
            status_label.config(text="❌ AV1 input not supported. Re-encode to H.264.")
            return

        cancel_requested.clear()
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
            args=(file_path, total_frames_all, 0, batch_size, output_dir_var, inference_res_var, status_label, progress_bar, cancel_requested, invert_var),
            daemon=True
        ).start()


