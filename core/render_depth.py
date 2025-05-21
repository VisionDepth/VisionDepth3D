import os
import re
import sys
import time
import uuid
import json
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import gc

import numpy as np
import torch
import cv2
import onnxruntime as ort
import matplotlib.cm as cm
from PIL import Image, ImageTk, ImageOps

from transformers import AutoProcessor, AutoModelForDepthEstimation, pipeline
from diffusers import EulerDiscreteScheduler, AutoencoderKL

from safetensors.torch import load_file

# Custom modules
from core.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from core.depth_crafter_ppl import DepthCrafterPipeline
from diffusers.configuration_utils import ConfigMixin
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

global pipe
pipe = None
pipe_type = None  # Add this
suspend_flag = threading.Event()
cancel_flag = threading.Event()
cancel_requested = threading.Event()
global_session_start_time = None
current_warmup_session = {"id": None}


# === Setup: Local weights directory ===
def get_weights_dir():
    if getattr(sys, 'frozen', False):
        # PyInstaller bundle: use dir of the .exe
        base_path = os.path.dirname(sys.executable)
    else:
        # Source run: use the parent of the current file (core/ -> project root)
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    return os.path.join(base_path, "weights")

local_model_dir = get_weights_dir()
os.makedirs(local_model_dir, exist_ok=True)

# === Suppress Hugging Face symlink warnings (esp. on Windows) ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def load_supported_models():
    models = {
        "  -- Select Model -- ": "  -- Select Model -- ",
        "Marigold Depth (Diffusers)": "diffusers:prs-eth/marigold-depth-v1-1",
        "DepthCrafter (Custom)": "depthcrafter:weights/DepthCrafter",
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
    
    # === Diffusion Model Check ===
    if checkpoint.startswith("diffusers:"):
        from diffusers import MarigoldDepthPipeline
        model_id = checkpoint.replace("diffusers:", "")
        try:
            pipe = MarigoldDepthPipeline.from_pretrained(
                model_id,
                variant="fp16",
                torch_dtype=torch.float16,
                cache_dir=local_model_dir  # ✅ Force download to your custom weights folder
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            def diffusion_pipe(images, inference_size=None):
                if not isinstance(images, list):
                    images = [images]
                results = []
                for img in images:
                    result = pipe(img, num_inference_steps=4, ensemble_size=5)
                    results.append({"predicted_depth": result.prediction[0]})

                return results

            print(f"🌀 Diffusion depth model loaded: {model_id}")
            diffusion_pipe._is_marigold = True  # ✅ Add this!
            diffusion_pipe.image_processor = pipe.image_processor 
            return diffusion_pipe, {"is_diffusion": True}

        except Exception as e:
            print(f"❌ Failed to load diffusion depth model: {e}")
            return None, None
    
    if checkpoint.startswith("depthcrafter:"):
        model_id = checkpoint.replace("depthcrafter:", "")
        return load_depthcrafter_pipeline(model_id, inference_steps=5, offload_mode="none")


    
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

def get_required_frames_from_unet(unet):
    try:
        sample_size = unet.config.get("sample_size", [16, 64, 64])
        if isinstance(sample_size, (list, tuple)) and len(sample_size) >= 1:
            return sample_size[0]  # temporal size
    except Exception as e:
        print(f"⚠️ Could not determine required frame count: {e}")
    return 16  # fallback


def load_depthcrafter_pipeline(weights_dir, inference_size=(512, 256), inference_steps=5, offload_mode="none"):
    unet_config_path = os.path.join(weights_dir, "unet_config.json")
    with open(unet_config_path, "r") as f:
        config_dict = json.load(f)

    unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_config(config_dict)
    state_dict = load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
    unet.load_state_dict(state_dict)

    scheduler_config_path = os.path.join(weights_dir, "scheduler_config.json")

    with open(scheduler_config_path, "r") as f:
        scheduler_config = json.load(f)

    scheduler = EulerDiscreteScheduler.from_config(scheduler_config)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    
    if torch.cuda.is_available():
        vae = vae.to(dtype=torch.float16)


    # 🔥 REQUIRED: load image encoder + feature extractor
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    feature_extractor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    # ✅ Move all modules to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if offload_mode == "model":
        unet.to("cpu")
        vae.to("cpu")
        image_encoder.to("cpu")
    elif offload_mode == "unet":
        unet.to("cpu")
        vae.to(device)
        image_encoder.to(device)
    elif offload_mode == "vae":
        vae.to("cpu")
        unet.to(device)
        image_encoder.to(device)
    elif offload_mode == "sequential":
        # All on GPU, but assume user will call modules one-by-one
        unet.to("cuda")
        vae.to("cuda")
        image_encoder.to("cuda")
    else:  # offload_mode == "none"
        unet.to(device)
        vae.to(device)
        image_encoder.to(device)

    pipe = DepthCrafterPipeline(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
)
    pipe.offload_mode = offload_mode
    
    # 🔍 Determine required number of frames dynamically
    required_frames = get_required_frames_from_unet(unet)


    def run_depthcrafter(images, inference_size=(512, 256), inference_steps=inference_steps):
        # Resize to force multiple of 64
        width = max(64, round(inference_size[0] / 64) * 64)
        height = max(64, round(inference_size[1] / 64) * 64)

        # Convert PIL images → torch tensors (B, C, H, W)
        frames = [
            torch.from_numpy(np.array(img.resize((width, height), Image.BICUBIC))).permute(2, 0, 1).float() / 255.0
            for img in images
        ]

        # Stack into a video tensor [T, C, H, W]
        video_tensor = torch.stack(frames).half().to("cuda")

        # Pad/crop for required number of frames
        if len(video_tensor) < required_frames:
            pad_count = required_frames - len(video_tensor)
            video_tensor = torch.cat([video_tensor] + [video_tensor[-1:].clone()] * pad_count, dim=0)
        elif len(video_tensor) > required_frames:
            video_tensor = video_tensor[:required_frames]

        # Run the model
        with torch.no_grad():
            result = pipe(
                video=video_tensor,
                height=height,
                width=width,
                output_type="latent",  # 🧠 GET LATENT instead of np
                num_inference_steps=inference_steps,
                guidance_scale=1.0,
                window_size=4,
                overlap=1,
                track_time=False
            )


        frames_out = result.frames[0]  # [T, C, H, W] or [T, H, W, C] → still latent

        # Stack to tensor
        frames_out_cpu = [f.detach().cpu() if isinstance(f, torch.Tensor) else torch.from_numpy(f) for f in frames_out]
        latents_tensor = torch.stack(frames_out_cpu).to(pipe.vae.device).half()

        if latents_tensor.ndim == 4 and latents_tensor.shape[1] == 3:
            pass  # already [T, 3, H, W]
        elif latents_tensor.ndim == 4 and latents_tensor.shape[-1] == 3:
            latents_tensor = latents_tensor.permute(0, 3, 1, 2)  # [T, H, W, C] → [T, C, H, W]
        else:
            print(f"⚠️ Unexpected latent shape: {latents_tensor.shape}")

        # Decode via VAE
        decoded_tensor = pipe.vae.decode(latents_tensor).sample  # [T, 3, H, W]
        del latents_tensor
        gc.collect()
        torch.cuda.empty_cache()

        decoded_np = decoded_tensor.detach().cpu().numpy()

        depth_frames = []
        for i in range(decoded_np.shape[0]):
            d = decoded_np[i]  # [3, H, W]
            if d.shape[0] == 3:
                d = d.mean(axis=0)  # Collapse channels → [H, W]
            norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
            depth_frames.append(torch.from_numpy(norm).float())

            print(f"🧪 Decoded Frame {i}: min={d.min():.4f}, max={d.max():.4f}, shape={d.shape}")

        return [{"predicted_depth": depth_frames[i], "latent_depth": frames_out[i]} for i in range(len(depth_frames))]


    run_depthcrafter.original_pipe = pipe
    run_depthcrafter._is_marigold = False
    return run_depthcrafter, {"is_diffusion": True}


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
    
    def onnx_pipe(images, inference_size=None):
        if inference_size is None:
            raise ValueError("❌ ONNX model requires explicit inference_size.")

        preds = []

        if input_rank == 5:
            if len(images) != 32:
                if len(images) > 32:
                    raise ValueError("ONNX model requires exactly 32 frames per inference.")
                print(f"⚠️ Padding {len(images)} to 32 frames.")
                images += [images[-1]] * (32 - len(images))

            img_batch = [np.array(img.resize(inference_size)).astype(np.float32).transpose(2, 0, 1) / 255.0 for img in images]
            input_tensor = np.stack(img_batch)[None, ...]
            result = session.run([output_name], {input_name: input_tensor})[0]
            result = result.squeeze(0)

            for i in range(32):
                preds.append({"predicted_depth": torch.tensor(result[i])})

        elif input_rank == 4:
            img_batch = [np.array(img.resize(inference_size)).astype(np.float32).transpose(2, 0, 1) / 255.0 for img in images]
            input_tensor = np.stack(img_batch)
            result = session.run([output_name], {input_name: input_tensor})[0]

            for i in range(len(images)):
                preds.append({"predicted_depth": torch.tensor(result[i])})
        else:
            raise ValueError(f"❌ Unsupported input rank: {input_rank}")

        return preds

    onnx_pipe._is_marigold = False
    return onnx_pipe, {
        "is_onnx": True,
        "input_rank": input_rank,
        "session": session
    }



spinner_states = ["⠋", "⠙", "⠸", "⠴", "⠦", "⠇"]
def start_spinner(widget, message="Warming up model..."):
    def spin(index=0):
        if not getattr(widget, "_spinner_running", False):
            return
        state = spinner_states[index % len(spinner_states)]
        widget.config(text=f"{state} {message}")
        widget.after(200, spin, index + 1)

    widget._spinner_running = True
    spin()

def stop_spinner(widget, final_text):
    widget._spinner_running = False
    widget.config(text=final_text)

def update_pipeline(selected_model_var, status_label_widget, inference_res_var, offload_mode_dropdown, *args):
    global pipe

    selected_checkpoint = selected_model_var.get()
    checkpoint = supported_models.get(selected_checkpoint, None)


    def warmup_thread():
        try:
            model_callable, processor_or_metadata = ensure_model_downloaded(checkpoint)
            if not model_callable:
                status_label_widget.after(0, lambda: stop_spinner(
                    status_label_widget, f"❌ Failed to load model: {selected_checkpoint}"))
                return

            device = 0 if torch.cuda.is_available() else -1
            is_onnx = isinstance(processor_or_metadata, dict) and processor_or_metadata.get("is_onnx", False)
            is_diffusion = isinstance(processor_or_metadata, dict) and processor_or_metadata.get("is_diffusion", False)

            global pipe, pipe_type

            if is_onnx:
                pipe = model_callable
                pipe_type = "onnx"

                status_label_widget.after(0, lambda: start_spinner(status_label_widget, "🔄 Warming up ONNX model..."))

                try:
                    input_rank = processor_or_metadata.get("input_rank", 4)
                    dummy_res = parse_inference_resolution(inference_res_var.get(), fallback=(518, 518))

                    if dummy_res is None:
                        dummy_res = (518, 518)  # fallback if user selected "Original" or invalid value

                    dummy_res = tuple(round_to_multiple_of_8(x) for x in dummy_res)

                    dummy_batch = [Image.new("RGB", dummy_res, (127, 127, 127))] * (32 if input_rank == 5 else 1)

                    _ = pipe(dummy_batch, inference_size=dummy_res)
                    print("🔥 ONNX model warmed up.")
                except Exception as e:
                    print(f"⚠️ ONNX warm-up failed: {e}")

                status_label_widget.after(0, lambda: stop_spinner(
                    status_label_widget, f"✅ ONNX model loaded: {selected_checkpoint} (on {'CUDA' if device == 0 else 'CPU'})"))


            elif is_diffusion:
                if hasattr(model_callable, "__call__") and hasattr(model_callable, "original_pipe"):
                    status_label_widget.after(0, lambda: start_spinner(status_label_widget, "🔄 Warming up DepthCrafter model..."))
                    # ✅ DepthCrafter — special constructor required
                    print("📦 Loading DepthCrafter model with custom params...")

                    try:
                        inference_steps = int(inference_steps_entry.get().strip())
                    except:
                        inference_steps = 5

                    try:
                        res_str = inference_res_var.get().split(" ")[0]  # "512x256 (DC-Fastest)" → "512x256"
                        w, h = [int(x) for x in res_str.split("x")]
                        inference_size = (w, h)
                    except Exception:
                        inference_size = (512, 256)

                    offload_mode = offload_mode_dropdown.get()

                    # Reload the pipeline with GUI options
                    pipe, _ = load_depthcrafter_pipeline(
                        checkpoint.replace("depthcrafter:", ""),
                        inference_steps=inference_steps,
                        offload_mode=offload_mode,
                        inference_size=inference_size
                    )
                    pipe_type = "diffusion"

                    status_label_widget.after(0, lambda: stop_spinner(
                        status_label_widget,
                        f"✅ DepthCrafter model loaded: {selected_checkpoint} (inference steps: {inference_steps}, {inference_size}, offload: {offload_mode})"
                    ))
                    return

                else:
                    # ✅ Other diffusion model (e.g. Marigold)
                    pipe = model_callable
                    pipe_type = "diffusion"

                    status_label_widget.after(0, lambda: start_spinner(
                        status_label_widget, "🔄 Warming up diffusion model..."))

                    try:
                        dummy = Image.new("RGB", (512, 512), (127, 127, 127))
                        _ = pipe(dummy)
                        print("🔥 Diffusion model warmed up with dummy image")
                    except Exception as e:
                        print(f"⚠️ Diffusion warm-up failed: {e}")

                    status_label_widget.after(0, lambda: stop_spinner(
                        status_label_widget,
                        f"✅ Diffusion model loaded: {selected_checkpoint} (Running on {'CUDA' if device == 0 else 'CPU'})"
                    ))

            else:
                processor = processor_or_metadata
                raw_pipe = pipeline(
                    "depth-estimation",
                    model=model_callable,
                    image_processor=processor,
                    device=device
                )

                def hf_batch_safe_pipe(images, **kwargs):
                    return raw_pipe(images) if isinstance(images, list) else [raw_pipe(images)]

                pipe = hf_batch_safe_pipe
                pipe_type = "hf"
                status_label_widget.after(0, lambda: start_spinner(
                    status_label_widget, "🔄 Warming up Hugging Face model..."))

                try:
                    dummy = Image.new("RGB", (384, 384), (127, 127, 127))
                    _ = pipe([dummy])
                    print("🔥 Hugging Face pipeline warmed up with dummy frame")
                except Exception as e:
                    print(f"⚠️ Hugging Face warm-up failed: {e}")

                status_label_widget.after(0, lambda: stop_spinner(
                    status_label_widget, f"✅ HF model loaded: {selected_checkpoint} (Running on {'CUDA' if device == 0 else 'CPU'})"))

        except Exception as e:
            err_msg = str(e)
            print(f"❌ Model loading failed: {e}")
            status_label_widget.after(0, lambda: stop_spinner(
                status_label_widget, f"❌ Model loading failed: {err_msg}"))


    threading.Thread(target=warmup_thread, daemon=True).start()

def round_to_multiple_of_8(x):
    return (x + 7) // 8 * 8


def parse_inference_resolution(res_string, fallback=(384, 384)):
    if not res_string or not isinstance(res_string, str):
        return fallback

    res_string = res_string.strip().lower()
    if "original" in res_string or res_string in {"--", "none", "auto"}:
        return None

    try:
        parts = res_string.replace("×", "x").split("x")
        if len(parts) == 2:
            return tuple(int(p) for p in parts)
    except Exception:
        pass

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

def process_image_folder(batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, invert_var, root):
    folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
    if not folder_path:
        cancel_requested.clear()  # ✅ Reset before starting
        status_label.config(text="⚠️ No folder selected.")
        return

    threading.Thread(
        target=process_images_in_folder,
        args=(folder_path, batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, root, invert_var),
        daemon=True,
    ).start()


def process_images_in_folder(folder_path, batch_size_widget, output_dir_var, inference_res_var, status_label, progress_bar, root, invert_var):
    output_dir = output_dir_var.get().strip()
    global pipe, pipe_type
    global global_session_start_time
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
            if inference_size:
                images.append(img.resize(inference_size, Image.BICUBIC))
            else:
                images.append(img)

        print(f"🚀 Running batch of {len(images)} images at {inference_size}")
        predictions = pipe(images, inference_size)


        for j, prediction in enumerate(predictions):
            if cancel_requested.is_set():
                root.after(10, lambda: status_label.config(text="❌ Cancelled during batch."))
                return

            file_path = batch_files[j]
            orig_w, orig_h = original_sizes[j]

            try:
                depth_tensor = prediction["predicted_depth"]

                # 🔄 Check for Marigold 16-bit support
                if getattr(pipe, "_is_marigold", False):
                    # ✅ Save 16-bit PNG from Marigold
                    depth_image = pipe.image_processor.export_depth_to_16bit_png(depth_tensor)[0]
                    depth_image = depth_image.resize((orig_w, orig_h), Image.BICUBIC)

                    if invert_var.get():
                        print("🌀 Inverting 16-bit depth for:", file_path)
                        depth_array = np.array(depth_image, dtype=np.uint16)
                        depth_array = 65535 - depth_array  # Manual inversion
                        depth_image = Image.fromarray(depth_array, mode="I;16")

                else:
                    # 🧠 Fallback to 8-bit normalized path
                    depth_norm = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())

                    if isinstance(depth_tensor, np.ndarray):
                        depth_np = (depth_norm.squeeze() * 255).astype(np.uint8)
                    else:
                        depth_np = (depth_norm.squeeze() * 255).cpu().numpy().astype(np.uint8)

                    if invert_var.get():
                        print("🌀 Inverting 8-bit depth for:", file_path)
                        depth_np = 255 - depth_np

                    depth_np = cv2.resize(depth_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                    depth_image = Image.fromarray(depth_np)

                # 💾 Save
                image_name = os.path.splitext(os.path.basename(file_path))[0]
                output_filename = f"{image_name}_depth.png"
                file_save_path = os.path.join(output_dir, output_filename)
                depth_image.save(file_save_path)

            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                continue

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
    global pipe, pipe_type
    image = Image.open(file_path).convert("RGB")
    original_size = image.size

    inference_size = parse_inference_resolution(inference_res_var.get())
    image_resized = image.resize(inference_size, Image.BICUBIC) if inference_size else image.copy()

    predictions = pipe([image], inference_size)
    
    if not (isinstance(predictions, list) and "predicted_depth" in predictions[0]):
        raise ValueError("❌ Unexpected prediction format from depth model.")

    depth_tensor = predictions[0]["predicted_depth"]

    try:
        colormap_name = colormap_var.get().strip().lower()

        if getattr(pipe, "_is_marigold", False):
            if colormap_name == "default":
                # Export raw 16-bit grayscale
                depth_image = pipe.image_processor.export_depth_to_16bit_png(depth_tensor)[0]
            else:
                # Export colorized RGB image
                try:
                    depth_image = pipe.image_processor.visualize_depth(depth_tensor, color_map=colormap_name)[0]
                except Exception as e:
                    print(f"⚠️ Failed to apply colormap '{colormap_name}', using default colormap. Error: {e}")
                    depth_image = pipe.image_processor.visualize_depth(depth_tensor)[0]

            depth_image = depth_image.resize(original_size, Image.BICUBIC)

            if invert_var.get():
                print("🌀 Inverting Marigold depth image")
                depth_array = np.array(depth_image)
                if depth_image.mode == "I;16":
                    depth_array = 65535 - depth_array
                    depth_image = Image.fromarray(depth_array, mode="I;16")
                else:
                    depth_array = 255 - depth_array
                    depth_image = Image.fromarray(depth_array.astype(np.uint8))

        else:
            # === Fallback for other models ===
            depth_norm = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
            depth_tensor = depth_norm.squeeze()

            if isinstance(depth_tensor, torch.Tensor):
                depth_np = (depth_tensor * 255).cpu().numpy().astype(np.uint8)
            else:
                depth_np = (depth_tensor * 255).astype(np.uint8)

            if invert_var.get():
                print("🌀 Inverting 8-bit depth for single image")
                depth_np = 255 - depth_np

            depth_np = cv2.resize(depth_np, original_size, interpolation=cv2.INTER_CUBIC)

            if colormap_name == "default":
                depth_image = Image.fromarray(depth_np)
            else:
                try:
                    cmap = cm.get_cmap(colormap_name)
                    colored = cmap(depth_np.astype(np.float32) / 255.0)
                    colored = (colored[:, :, :3] * 255).astype(np.uint8)
                    depth_image = Image.fromarray(colored)
                except ValueError:
                    print(f"⚠️ Unknown colormap: {colormap_name}, defaulting to grayscale.")
                    depth_image = Image.fromarray(depth_np)

    except Exception as e:
        print(f"❌ Error extracting depth: {e}")
        return

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

    if not folder:
        image_disp = image.copy()
        image_disp.thumbnail((480, 270))
        input_img_tk = ImageTk.PhotoImage(image_disp)
        input_label.config(image=input_img_tk)
        input_label.image = input_img_tk  # ✅ Prevent garbage collection

        depth_disp = depth_image.copy()

        # ✅ Handle 16-bit grayscale preview safely
        if depth_disp.mode in ("I", "I;16"):
            depth_array = np.array(depth_disp)

            if depth_array.dtype != np.uint16:
                # Normalize to 16-bit range first if needed
                depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
                depth_array = (depth_array * 65535).astype(np.uint16)

            # Downscale to 8-bit for preview display
            preview_array = (depth_array / 256).astype(np.uint8)
            depth_disp = Image.fromarray(preview_array, mode="L").convert("RGB")

        depth_disp.thumbnail((480, 270))
        depth_img_tk = ImageTk.PhotoImage(depth_disp)
        output_label.config(image=depth_img_tk)
        output_label.image = depth_img_tk  # ✅ Prevent garbage collection


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
    global pipe, pipe_type
    global global_session_start_time

    # Detect output directory from UI
    output_dir = output_dir_var.get().strip()
    if not output_dir:
        messagebox.showwarning("Missing Output Folder", "⚠️ Please select an output directory before processing.")
        status_label.config(text="❌ Output directory not selected.")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    input_dir, input_filename = os.path.split(file_path)
    name, _ = os.path.splitext(input_filename)
    output_filename = f"{name}_depth.mkv"
    output_path = os.path.join(output_dir, output_filename)

    # ✅ Special case for Marigold (16-bit export path)
    if hasattr(pipe, "image_processor") and hasattr(pipe.image_processor, "export_depth_to_16bit_png"):
        print("🎥 Marigold model detected — switching to frame-based 16-bit processing.")

        tmp_frame_dir = os.path.join(output_dir, f"{name}_tmp_frames")
        os.makedirs(tmp_frame_dir, exist_ok=True)

        # === 1. Extract raw frames from video
        extract_cmd = [
            "ffmpeg", "-y", "-i", file_path,
            os.path.join(tmp_frame_dir, "frame_%05d.png")
        ]
        subprocess.run(extract_cmd)

        # === 2. Process images into depth maps (same folder)
        dummy_widget = tk.StringVar(value=str(batch_size))
        dummy_output_var = tk.StringVar(value=tmp_frame_dir)
        dummy_root = tk.Tk(); dummy_root.withdraw()

        process_images_in_folder(
            tmp_frame_dir,
            batch_size_widget=dummy_widget,
            output_dir_var=dummy_output_var,
            inference_res_var=inference_res_var,
            status_label=status_label,
            progress_bar=progress_bar,
            root=dummy_root,
            cancel_requested=cancel_requested,
            invert_var=invert_var
        )

        # === 3. Encode depth frames to video using FFmpeg
        encode_cmd = [
            "ffmpeg", "-y", "-framerate", "24",  # fallback FPS
            "-i", os.path.join(tmp_frame_dir, "frame_%05d_depth.png"),
            "-c:v", "ffv1", "-pix_fmt", "yuv420p16le",
            output_path
        ]
        subprocess.run(encode_cmd)

        print(f"✅ Marigold 16-bit depth video saved: {output_path}")
        return len(os.listdir(tmp_frame_dir))

    # === Fallback: non-Marigold default behavior ===
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        status_label.config(text=f"❌ Error: Cannot open {file_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📁 Saving video to: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))
    if not out.isOpened():
        print(f"❌ Failed to open video writer for {output_filename}")
        return 0

    frame_output_dir = os.path.join(output_dir, f"{name}_frames")
    if save_frames:
        os.makedirs(frame_output_dir, exist_ok=True)

    frame_count = 0
    frames_batch = []
    total_processed_frames = 0
    
    inference_size = parse_inference_resolution(inference_res_var.get())
    if inference_size is not None:
        inference_size = (
            round_to_multiple_of_8(inference_size[0]),
            round_to_multiple_of_8(inference_size[1])
        )

    if inference_size is None:
        inference_size = (512, 512)  # or whatever your ONNX default is

    global_session_start_time = time.time()
    previous_depth = None

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

            predictions = pipe(frames_batch, inference_size=inference_size)
            
            assert isinstance(predictions, list), "Expected list of predictions from pipeline"
            
            for i, prediction in enumerate(predictions):
                if cancel_requested.is_set():
                    print("🛑 Cancelled during batch write.")
                    status_label.config(text="🛑 Cancelled during batch.")
                    cap.release()
                    out.release()
                    return frame_count

                try:
                    raw_depth = prediction["predicted_depth"]

                    def convert_depth_to_grayscale(depth):
                        if isinstance(depth, Image.Image):
                            depth = np.array(depth).astype(np.float32)

                        elif isinstance(depth, torch.Tensor):
                            depth = depth.detach().cpu().float().numpy()

                        elif isinstance(depth, np.ndarray):
                            depth = depth.astype(np.float32)
                        else:
                            raise TypeError(f"Unsupported depth type: {type(depth)}")

                        # Handle [C, H, W] or [H, W, C]
                        if depth.ndim == 3:
                            if depth.shape[0] in {1, 3}:  # [C, H, W]
                                depth = depth[0] if depth.shape[0] == 1 else depth.mean(axis=0)
                            elif depth.shape[2] in {1, 3}:  # [H, W, C]
                                depth = depth[..., 0] if depth.shape[2] == 1 else depth.mean(axis=-1)

                        elif depth.ndim != 2:
                            raise ValueError(f"Unexpected depth shape: {depth.shape}")

                        # Normalize safely to [0, 255]
                        depth_min, depth_max = np.min(depth), np.max(depth)
                        if np.isnan(depth_min) or np.isnan(depth_max) or depth_max - depth_min < 1e-6:
                            print("⚠️ Skipping frame with invalid depth values.")
                            return np.zeros_like(depth, dtype=np.uint8)

                        norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)
                        return (norm * 255).astype(np.uint8)

                    # Process list of frames or single frame
                    if isinstance(raw_depth, list):
                        for depth_frame_tensor in raw_depth:
                            gray = convert_depth_to_grayscale(depth_frame_tensor)
                            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                            bgr = cv2.resize(bgr, (original_width, original_height))
                            out.write(bgr)

                            if save_frames:
                                frame_filename = os.path.join(frame_output_dir, f"frame_{frame_count:05d}.png")
                                cv2.imwrite(frame_filename, gray)

                            total_processed_frames += 1
                    else:
                        gray = convert_depth_to_grayscale(raw_depth)
                        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        bgr = cv2.resize(bgr, (original_width, original_height))
                        out.write(bgr)

                        if save_frames:
                            frame_filename = os.path.join(frame_output_dir, f"frame_{frame_count:05d}.png")
                            cv2.imwrite(frame_filename, gray)

                        total_processed_frames += 1

                except Exception as e:
                    print(f"⚠️ Depth processing error: {e}")


                    
            if frame_count % 20 == 0:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()

            frames_batch.clear()

        elapsed = time.time() - global_session_start_time
        avg_fps = total_processed_frames / elapsed if elapsed > 0 else 0
        remaining_frames = total_frames_all - (frames_processed_all + total_processed_frames)
        eta = remaining_frames / avg_fps if avg_fps > 0 else 0

        progress = int(((frames_processed_all + total_processed_frames) / total_frames_all) * 100)
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))

        status_label.config(
            text=f"🎬 {frames_processed_all + total_processed_frames}/{total_frames_all} frames | "
                 f"FPS: {avg_fps:.2f} | Elapsed: {elapsed_str} | ETA: {eta_str} | Processing: {name}"
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


