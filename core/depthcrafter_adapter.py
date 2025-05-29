import os
import numpy as np
import torch
import cv2
import json
from safetensors.torch import load_file

from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from .depth_crafter_ppl import DepthCrafterPipeline
from .unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from diffusers.configuration_utils import ConfigMixin



def load_depthcrafter_adapter(weights_dir, offload_mode="sequential"):
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
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    feature_extractor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    pipe = DepthCrafterPipeline(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if offload_mode == "model":
        pipe.unet.to("cpu")
        pipe.vae.to("cpu")
        pipe.image_encoder.to("cpu")
    elif offload_mode == "unet":
        pipe.unet.to("cpu")
        pipe.vae.to(device)
        pipe.image_encoder.to(device)
    elif offload_mode == "vae":
        pipe.vae.to("cpu")
        pipe.unet.to(device)
        pipe.image_encoder.to(device)
    elif offload_mode == "sequential":
        pipe.unet.to("cuda")
        pipe.vae.to(dtype=pipe.vae.dtype, device="cuda")
        pipe.image_encoder.to("cuda")
    else:  # default
        pipe.unet.to(device)
        pipe.vae.to(dtype=pipe.vae.dtype, device=device)
        pipe.image_encoder.to(device)
    
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    return pipe, {"is_diffusion": True}


def run_depthcrafter_inference(
    pipe,
    frames,
    inference_size=(512, 256),
    steps=2,
    window_size=24,
    overlap=25,
    offload_mode="sequential",
):
    width, height = inference_size

    if not frames:
        print("❌ No input frames received by DepthCrafter.")
        return None

    # Convert list of PIL or np.ndarray to single np array [T, H, W, C]
    try:
        video = np.stack([
            np.asarray(f).astype("float32") / 255.0 for f in frames
        ])  # [T, H, W, C]
    except Exception as e:
        print(f"❌ Failed to prepare input frames: {e}")
        return None

    try:
        result = pipe(
            video=video,
            height=height,
            width=width,
            num_inference_steps=steps,
            window_size=window_size,
            overlap=overlap,
            output_type="np",  # guarantees numpy output
            return_dict=True
        )
    except Exception as e:
        print(f"❌ DepthCrafter inference failed: {e}")
        return None

    frames_out = result.frames if result and hasattr(result, "frames") else None

    if frames_out is None or len(frames_out) == 0:
        print("❌ No frames returned from DepthCrafter.")
        return None

    # Original repo takes channel mean across last dim [T, H, W, 3] → [T, H, W]
    depth = frames_out[0].mean(axis=-1).astype("float32")  # normalized already

    return depth  # shape: [T, H, W]

