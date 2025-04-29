import os
import time
import cv2
import torch
import numpy as np
import subprocess
import threading
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import onnxruntime as ort
import torch.nn.functional as F
from collections import deque
from scipy.ndimage import gaussian_filter
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur
from ffmpeg_blackdetect import detect_black_white_frames



# Device setup
#onnx_device = "CUDAExecutionProvider" if ort.get_device() == "GPU" else "CPUExecutionProvider"
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 CUDA available: {torch.cuda.is_available()} | Running on {torch_device}")

# Load ONNX model
#MODEL_PATH = 'weights/backward_warping_model.onnx'
#session = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
#input_name = session.get_inputs()[0].name
#output_name = session.get_outputs()[0].name
#print(f"✅ Loaded ONNX model from {MODEL_PATH} on {onnx_device}")

#Global flags
suspend_flag = threading.Event()
cancel_flag = threading.Event()
process_thread = None 

# Common Aspect Ratios
aspect_ratios = {
    "Default (16:9)": 16 / 9,
    "CinemaScope (2.39:1)": 2.39,
    "21:9 UltraWide": 21 / 9,
    "4:3 (Classic Films)": 4 / 3,
    "1:1 (Square)": 1 / 1,
    "2.35:1 (Classic Cinematic)": 2.35,
    "2.76:1 (Ultra-Panavision)": 2.76,
}

FFMPEG_CODEC_MAP = {
    # 🔹 Software (CPU) Codecs
    "H.264 / AVC (libx264)": "libx264",          # Standard CPU-based H.264
    "H.265 / HEVC (libx265)": "libx265",         # Better compression, slower
    "MPEG-4 (mp4v)": "mp4v",                     # Legacy MPEG-4 Part 2
    "XviD (AVI - CPU)": "XVID",                  # Good for AVI containers
    "DivX (AVI - CPU)": "DIVX",                  # Older compatibility

    # 🔹 NVIDIA NVENC (GPU) Codecs
    "AVC (NVENC GPU)": "h264_nvenc",             # Hardware-accelerated H.264
    "HEVC / H.265 (NVENC GPU)": "hevc_nvenc",    # Hardware-accelerated H.265

    # 🔹 Optional / Experimental
    "AV1 (CPU)": "libaom-av1",
    "AV1 (NVIDIA)": "av1_nvenc",  # Supported on newer RTX GPUs
}


def pad_to_aspect_ratio(image, target_width, target_height, bg_color=(0, 0, 0)):
    """
    Pads the input image to the target resolution without stretching,
    preserving aspect ratio.
    """
    
    h, w = image.shape[:2]
    target_aspect = target_width / target_height
    current_aspect = w / h

    # Step 1: Resize to fit within target while preserving aspect
    if current_aspect > target_aspect:
        # Image is wider than target → match width
        new_w = target_width
        new_h = int(target_width / current_aspect)
    else:
        # Image is taller → match height
        new_h = target_height
        new_w = int(current_aspect * target_height)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 2: Create padded canvas
    padded = np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)

    # Step 3: Center it
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded


# Converters
def frame_to_tensor(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
    return frame_tensor.to(torch_device)

def depth_to_tensor(depth_frame):
    depth_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
    depth_tensor = torch.from_numpy(depth_gray).float().unsqueeze(0) / 255.0
    return depth_tensor.to(torch_device)

def estimate_subject_depth(depth_tensor):
    """
    More robust subject depth estimator using center-weighted and histogram analysis.
    """
    _, H, W = depth_tensor.shape
    center_crop = depth_tensor[
        :, H // 4 : H * 3 // 4, W // 4 : W * 3 // 4
    ]  # Focus on center 50%

    flat = center_crop.flatten()
    filtered = flat[(flat > 0.1) & (flat < 0.85)]

    if filtered.numel() < 1:
        return torch.tensor(0.5, device=depth_tensor.device)

    # Histogram-based mode estimation (most frequent depth)
    hist = torch.histc(filtered, bins=64, min=0.0, max=1.0)
    peak_bin = torch.argmax(hist)
    bin_width = 1.0 / 64
    subject_depth = (peak_bin.float() + 0.5) * bin_width

    return subject_depth

def enhance_curvature(depth_tensor, strength=0.08):
    """
    Adds a 2D curvature profile to simulate facial/body roundness.
    """
    B, H, W = depth_tensor.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=depth_tensor.device),
        torch.linspace(-1, 1, W, device=depth_tensor.device),
        indexing="ij"
    )
    curvature = 1 - (xx**2 + yy**2)  # peak in center
    curve = curvature.unsqueeze(0).expand(B, -1, -1)
    return depth_tensor + (curve * strength)


# Bilateral smoothing for depth (preserves edges)
def bilateral_smooth_depth(depth_tensor):
    depth_np = depth_tensor.squeeze().cpu().numpy().astype(np.uint8)
    smoothed = cv2.bilateralFilter(depth_np, d=9, sigmaColor=75, sigmaSpace=75)
    smoothed_tensor = torch.from_numpy(smoothed).float().unsqueeze(0) / 255.0
    return smoothed_tensor.to(depth_tensor.device)

# Gradient-aware shift suppression
def suppress_artifacts_with_edge_mask(depth_tensor, total_shift, feather_strength=10.0, edge_threshold=0.02):
    """
    Suppress pixel shift artifacts near sharp depth edges (hair, limbs).
    Returns a softly masked version of total_shift using adaptive edge gradient detection.
    """
    # Compute depth gradient (H, W)
    dx = torch.abs(F.pad(depth_tensor[:, :, 1:] - depth_tensor[:, :, :-1], (1, 0)))
    dy = torch.abs(F.pad(depth_tensor[:, 1:, :] - depth_tensor[:, :-1, :], (0, 0, 1, 0)))
    grad_mag = torch.sqrt(dx ** 2 + dy ** 2)

    # Use a sigmoid function for smooth masking
    edge_mask = torch.sigmoid((grad_mag - edge_threshold) * feather_strength * 5)  # [0, 1]

    # Invert mask and smooth
    smooth_mask = 1.0 - edge_mask
    smooth_mask = F.avg_pool2d(smooth_mask.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)

    # Apply mask to suppress shift near edges
    return total_shift * smooth_mask


# Optional temporal depth filter class
class TemporalDepthFilter:
    def __init__(self, alpha=0.85):
        self.prev_depth = None
        self.alpha = alpha

    def smooth(self, curr_depth):
        if self.prev_depth is None:
            self.prev_depth = curr_depth.clone()
        self.prev_depth = self.alpha * self.prev_depth + (1 - self.alpha) * curr_depth
        return self.prev_depth

def tensor_to_frame(tensor):
    frame_cpu = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2BGR)

def detect_black_bars(frame_tensor, threshold=10):
    """
    Automatically detects black bars on top and bottom of a frame tensor.
    Returns: (top_crop, bottom_crop) in pixels
    """
    frame_np = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)

    h = gray.shape[0]
    top_crop, bottom_crop = 0, 0

    # Scan from top
    for i in range(h):
        if np.mean(gray[i]) > threshold:
            top_crop = i
            break

    # Scan from bottom
    for i in range(h - 1, -1, -1):
        if np.mean(gray[i]) > threshold:
            bottom_crop = h - i - 1
            break

    return top_crop, bottom_crop

def crop_black_bars_torch(frame_tensor, top, bottom):
    """
    Crops black bars vertically using PyTorch tensors.
    - frame_tensor: shape [3, H, W]
    Returns: cropped tensor
    """
    if top + bottom >= frame_tensor.shape[1]:
        return frame_tensor  # prevent invalid crop
    return frame_tensor[:, top:frame_tensor.shape[1] - bottom, :]

def feather_shift_edges(
    shifted_tensor: torch.Tensor,
    original_tensor: torch.Tensor,
    depth_tensor: torch.Tensor,
    blur_ksize: int = 7,
    feather_strength: float = 10.0,
    enable_feathering: bool = True
) -> torch.Tensor:
    """
    Blends shifted frame with original frame based on depth edge gradients.
    Helps reduce hard-edge ghosting artifacts in 3D rendering.
    """
    assert shifted_tensor.shape == original_tensor.shape, "Shape mismatch"
    assert depth_tensor.dim() == 3, "Depth tensor must be [1, H, W]"

    if not enable_feathering:
        return shifted_tensor  # 🔥 skip blending and return shifted

    # Compute depth gradient magnitude
    grad_x = F.pad(depth_tensor[:, :, 1:] - depth_tensor[:, :, :-1], (1, 0))
    grad_y = F.pad(depth_tensor[:, 1:, :] - depth_tensor[:, :-1, :], (0, 0, 1, 0))
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize & exaggerate gradients into mask
    edge_mask = torch.clamp(grad_mag * feather_strength, 0.0, 1.0)

    # Apply blur for smooth feathering
    blurred_mask = F.avg_pool2d(
        edge_mask.unsqueeze(0),
        kernel_size=blur_ksize,
        stride=1,
        padding=blur_ksize // 2
    ).squeeze(0)

    # Expand to match 3 channels (C=3, H, W)
    blend_mask = blurred_mask.repeat(3, 1, 1)

    min_h = min(shifted_tensor.shape[1], blend_mask.shape[1])
    min_w = min(shifted_tensor.shape[2], blend_mask.shape[2])

    blend_mask = blend_mask[:, :min_h, :min_w]
    shifted_tensor = shifted_tensor[:, :min_h, :min_w]
    original_tensor = original_tensor[:, :min_h, :min_w]

    output_tensor = shifted_tensor * (1.0 - blend_mask) + original_tensor * blend_mask

    return output_tensor.clamp(0.0, 1.0)


def shift_mask(mask_tensor, shift_vals, width):
    H, W = mask_tensor.shape[-2:]

    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

    N, C, H, W = mask_tensor.shape

    # Create grid
    x = torch.linspace(-1, 1, W, device=mask_tensor.device)
    y = torch.linspace(-1, 1, H, device=mask_tensor.device)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((x_grid, y_grid), dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(N, H, W, 2)  # [N, H, W, 2]

    if shift_vals.dim() == 2:
        shift_vals = shift_vals.unsqueeze(0)  # [1, H, W]

    # Make sure shift_vals match batch size
    shift_vals = shift_vals.expand(N, H, W)

    # ✅ Scale shift_vals to grid units
    shift_vals_grid = (shift_vals / (W / 2)).clamp(-1.0, 1.0)  # IMPORTANT ⚡

    grid[..., 0] -= shift_vals_grid  # apply scaled shift

    warped = F.grid_sample(
        mask_tensor, grid,
        mode='bilinear', padding_mode='border', align_corners=True
    )

    return warped.squeeze(0)  # Remove batch dimension

def compute_dynamic_parallax_scale(depth_tensor, min_scale=0.6, max_scale=1.0):
    """
    Dynamically adjusts parallax strength based on scene depth.
    - Closer shots get full parallax.
    - Wider, flatter shots get reduced parallax to avoid warping.
    """
    center_crop = depth_tensor[:, depth_tensor.shape[1]//4:3*depth_tensor.shape[1]//4,
                                  depth_tensor.shape[2]//4:3*depth_tensor.shape[2]//4]
    depth_variance = torch.var(center_crop)

    # Heuristic: less variance = flatter scene = reduce parallax
    if depth_variance < 0.001:
        scale = min_scale
    elif depth_variance > 0.01:
        scale = max_scale
    else:
        scale = min_scale + (depth_variance - 0.001) / (0.01 - 0.001) * (max_scale - min_scale)
        scale = torch.clamp(scale, min_scale, max_scale)
    return scale.item()

# --- Enhanced Healing of Warped Areas ---
def heal_missing_pixels(warped_frame, warped_depth, original_frame, edge_mask, heal_strength=0.5):
    """
    Heals gaps after convergence shifting based on depth edges and warp mask,
    with optional selective softening for invisible healing.
    """
    device = warped_frame.device
    warped_gray = warped_frame.mean(dim=0, keepdim=True)  # average over channels
    grad_x = F.pad(warped_gray[:, :, 1:] - warped_gray[:, :, :-1], (1, 0))
    grad_y = F.pad(warped_gray[:, 1:, :] - warped_gray[:, :-1, :], (0, 0, 1, 0))
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    threshold = 0.05  # 🔥 Tune if needed
    missing_mask = (grad_mag > threshold).float()
    missing_mask = F.avg_pool2d(missing_mask.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)
    missing_mask = missing_mask.clamp(0, 1)

    if edge_mask is not None:
        missing_mask = torch.max(missing_mask, edge_mask)

    missing_mask = missing_mask.expand_as(warped_frame)  # [3, H, W]

    # Basic healing: blend original into missing areas
    healed = (1.0 - heal_strength * missing_mask) * warped_frame + heal_strength * missing_mask * original_frame

    # 💡 BONUS: Apply slight blur *only* on healed areas for better invisibility
    soft_blur = F.avg_pool2d(healed.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
    healed = (1.0 - 0.3 * missing_mask) * healed + 0.3 * missing_mask * soft_blur

    return healed.clamp(0, 1)


# Shift Smoother
class ShiftSmoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev_fg_shift = None
        self.prev_mg_shift = None
        self.prev_bg_shift = None

    def smooth(self, fg_shift, mg_shift, bg_shift):
        if self.prev_fg_shift is None:
            self.prev_fg_shift, self.prev_mg_shift, self.prev_bg_shift = fg_shift, mg_shift, bg_shift
        else:
            self.prev_fg_shift = self.alpha * fg_shift + (1 - self.alpha) * self.prev_fg_shift
            self.prev_mg_shift = self.alpha * mg_shift + (1 - self.alpha) * self.prev_mg_shift
            self.prev_bg_shift = self.alpha * bg_shift + (1 - self.alpha) * self.prev_bg_shift
        return self.prev_fg_shift, self.prev_mg_shift, self.prev_bg_shift
    
class FloatingWindowTracker:
    def __init__(self, alpha=0.85):
        self.prev_offset = 0.0
        self.alpha = alpha
        self.frame_counter = 0  # 🆕 Add a counter

    def smooth_offset(self, current_offset, threshold=0.002):
        delta = abs(current_offset - self.prev_offset)
        if delta < threshold:
            return self.prev_offset  # ignore tiny jitter

        self.prev_offset = self.alpha * self.prev_offset + (1 - self.alpha) * current_offset
        self.frame_counter += 1  # 🆕 Increment each call

        # 🆕 Every 100 updates, clamp to avoid precision drift
        if self.frame_counter >= 100:
            self.prev_offset = max(min(self.prev_offset, 1.0), -1.0)  # clamp to [-1, +1]
            self.frame_counter = 0

        return self.prev_offset

floating_window_tracker = FloatingWindowTracker(alpha=0.97)

class FloatingBarEaser:
    def __init__(self, alpha=0.95):
        self.prev_bar_width = 0
        self.alpha = alpha

    def ease(self, current_width):
        self.prev_bar_width = int(self.alpha * self.prev_bar_width + (1 - self.alpha) * current_width)
        return self.prev_bar_width

bar_easer = FloatingBarEaser(alpha=0.85)

def pixel_shift_cuda(
    frame_tensor,
    depth_tensor,
    width,
    height,
    fg_shift,
    mg_shift,
    bg_shift,
    blur_ksize=9,
    feather_strength=10.0,
    max_pixel_shift_percent=0.02,
    parallax_balance=0.8,
    convergence_offset=0.0,
    use_subject_tracking=True,
    enable_floating_window=True,
    return_shift_map=True,
    enable_feathering=True,
    enable_edge_masking=True,
):
    device = frame_tensor.device

    frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
    depth_tensor = F.interpolate(depth_tensor.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)

    if 'enhance_curvature' in globals():
        depth_tensor = enhance_curvature(depth_tensor, strength=0.08)

    depth_tensor = depth_tensor.clamp(0.0, 1.0)

    fg_weight = (1.0 - depth_tensor)
    mg_weight = (1.0 - (depth_tensor - 0.4).abs() * 2.5)
    bg_weight = depth_tensor

    half_width = width / 2.0

    raw_shift = (-depth_tensor * fg_weight * fg_shift +
                 -depth_tensor * mg_weight * mg_shift +
                 depth_tensor * bg_weight * bg_shift)

    total_shift = (raw_shift * parallax_balance) / half_width

    if use_subject_tracking:
        subject_depth = estimate_subject_depth(depth_tensor)
        adjusted_depth = subject_depth * parallax_balance

        depth_flat = depth_tensor.view(-1)
        depth_std = torch.std(depth_flat)
        scaling_factor = torch.clamp(depth_std * 5.0, 0.5, 1.5).item()
        fg_shift *= scaling_factor
        mg_shift *= scaling_factor
        bg_shift *= scaling_factor

        zero_parallax_offset = (
            (-adjusted_depth * fg_shift) +
            (-adjusted_depth * mg_shift) +
            (adjusted_depth * bg_shift)
        ) / half_width

        if enable_floating_window:
            subject_weight = torch.clamp(1.0 - subject_depth * 2.5, 0.25, 1.0)
            zero_parallax_offset *= subject_weight
            zero_parallax_offset = torch.clamp(zero_parallax_offset, -0.3, 0.3)
            zero_parallax_offset = floating_window_tracker.smooth_offset(zero_parallax_offset.item(), threshold=0.002)

        total_shift -= zero_parallax_offset

    total_shift += convergence_offset

    # --- Fix applied here ---
    if convergence_offset != 0.0:
        with torch.no_grad():
            convergence_grid = -torch.full_like(depth_tensor, convergence_offset).squeeze(0)

            dx = F.pad(depth_tensor[:, :, 1:] - depth_tensor[:, :, :-1], (1, 0))
            dy = F.pad(depth_tensor[:, 1:, :] - depth_tensor[:, :-1, :], (0, 0, 1, 0))
            grad_mag = torch.sqrt(dx ** 2 + dy ** 2)

            edge_strength = np.clip(abs(convergence_offset) * 80.0, 30.0, 80.0)
            edge_mask_orig = torch.sigmoid((grad_mag - 0.02) * edge_strength)
            edge_mask_orig = F.avg_pool2d(edge_mask_orig.unsqueeze(0), kernel_size=7, stride=1, padding=3).squeeze(0)

            soft_shift = convergence_grid * (1.0 - edge_mask_orig)

            smoothness_mask = (grad_mag < 0.05).float()
            smoothness_mask = F.avg_pool2d(smoothness_mask.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)

            blurred_shift = F.avg_pool2d(soft_shift.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)
            soft_shift = soft_shift * (1.0 - smoothness_mask) + blurred_shift * smoothness_mask

            warped_depth = shift_mask(depth_tensor, soft_shift, width)
            warped_frame = shift_mask(frame_tensor, soft_shift, width)

            # UPDATE depth_tensor to warped_depth immediately
            depth_tensor = warped_depth
            frame_tensor = warped_frame

    warped_depth = depth_tensor

    # --- From this point onwards, ONLY use warped_depth ---

    max_shift_px = width * max_pixel_shift_percent
    max_shift_norm = max_shift_px / half_width
    total_shift = torch.clamp(total_shift, -max_shift_norm, max_shift_norm)

    mask_strength = np.clip(feather_strength / 10.0, 0.05, 0.3)

    if enable_edge_masking:
        edge_suppressed = suppress_artifacts_with_edge_mask(warped_depth, total_shift, feather_strength)
        final_shift = (1.0 - mask_strength) * total_shift + mask_strength * edge_suppressed
    else:
        final_shift = total_shift

    shift_vals = final_shift.squeeze(0)

    H, W = depth_tensor.shape[1:]
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, W, device=device),
        torch.linspace(-1, 1, H, device=device),
        indexing="xy"
    )
    grid = torch.stack((xx, yy), dim=-1)

    grid_left = grid.clone()
    grid_right = grid.clone()
    grid_left[..., 0] += shift_vals
    grid_right[..., 0] -= shift_vals

    left = F.grid_sample(frame_tensor.unsqueeze(0), grid_left.unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True).squeeze(0)
    right = F.grid_sample(frame_tensor.unsqueeze(0), grid_right.unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True).squeeze(0)

    if enable_feathering:
        left_blended = feather_shift_edges(left, frame_tensor, warped_depth, blur_ksize, feather_strength, enable_feathering)
        right_blended = feather_shift_edges(right, frame_tensor, warped_depth, blur_ksize, feather_strength, enable_feathering)
    else:
        left_blended = left
        right_blended = right

    if return_shift_map:
        return tensor_to_frame(left_blended), tensor_to_frame(right_blended), final_shift.detach().cpu()
    else:
        return tensor_to_frame(left_blended), tensor_to_frame(right_blended)
        
# Sharpening

def apply_sharpening(frame, factor=1.0):
    # Safer sharpening kernel with brightness normalization
    kernel = np.array([
        [0, -1, 0],
        [-1, 5 + factor, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # Normalize kernel to preserve brightness (sum to ~1)
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel /= kernel_sum

    # Apply and clip result to valid range
    sharpened = cv2.filter2D(frame, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# 3D Formats
def format_3d_output(left, right, fmt):
    h, w = left.shape[:2]
    
    if fmt == "Half-SBS":
        return np.hstack((left, right))

    elif fmt == "Full-SBS":
        return np.hstack((left, right))
    
    elif fmt == "VR":
        lw = cv2.resize(left, (1440, 1600))
        rw = cv2.resize(right, (1440, 1600))
        return np.hstack((lw, rw))
    
    elif fmt == "Red-Cyan Anaglyph":
        return generate_anaglyph_3d(left, right)
    
    elif fmt == "Passive Interlaced":
        interlaced = np.zeros_like(left)
        interlaced[::2] = left[::2]      # even rows
        interlaced[1::2] = right[1::2]   # odd rows
        return interlaced

    return np.hstack((left, right))  # fallback

def generate_anaglyph_3d(left_frame, right_frame):
    """
    Generates a Dubois-style Red-Cyan anaglyph for better color accuracy and depth.
    """
    left = left_frame.astype(np.float32) / 255.0
    right = right_frame.astype(np.float32) / 255.0

    l_r, l_g, l_b = cv2.split(left)
    r_r, r_g, r_b = cv2.split(right)

    # Dubois-style anaglyph transform
    red = 0.4561 * l_r + 0.5005 * l_g + 0.1762 * l_b
    green = 0.3764 * r_r + 0.7616 * r_g - 0.1876 * r_b
    blue = -0.0401 * r_r - 0.1126 * r_g + 1.2723 * r_b

    anaglyph = cv2.merge([
        np.clip(red, 0, 1),
        np.clip(green, 0, 1),
        np.clip(blue, 0, 1)
    ])

    return (anaglyph * 255).astype(np.uint8)

def apply_side_mask(image, side="left", width=40):
    h, w = image.shape[:2]
    mask = np.ones((h, w, 3), dtype=np.uint8) * 255
    if side == "left":
        mask[:, :width] = 0
    elif side == "right":
        mask[:, w - width:] = 0
    return cv2.bitwise_and(image, mask)

# Render
def render_sbs_3d(
    input_path,
    depth_path,
    output_path,
    selected_codec,
    fps,
    output_width,
    output_height,
    fg_shift,
    mg_shift,
    bg_shift,
    sharpness_factor,
    output_format,
    selected_aspect_ratio,
    aspect_ratios,
    feather_strength=0.0,
    blur_ksize=1,
    use_ffmpeg=False,
    selected_ffmpeg_codec=None,
    crf_value=23,
    use_subject_tracking=False,
    use_floating_window=False,
    max_pixel_shift_percent=0.02,
    progress=None,
    progress_label=None,
    suspend_flag=None,
    cancel_flag=None,
    auto_crop_black_bars=False,
    parallax_balance=0.8,
    preserve_original_aspect=False,
    convergence_offset=0.0,
    enable_edge_masking=True,
    enable_feathering=True,
    skip_blank_frames=False,
    original_video_width=None,
    original_video_height=None,
):

    cap, dcap = cv2.VideoCapture(input_path), cv2.VideoCapture(depth_path)
    if not cap.isOpened() or not dcap.isOpened(): return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret1, frame = cap.read()
    ret2, depth = dcap.read()
    if not ret1 or not ret2: return

    # --- Detect Blank Frames ---
    blank_frames = []
    if skip_blank_frames:
        try:
            blank_frames = detect_black_white_frames(
                input_path,
                mode="black",  # or "white" — if you later expose to user
                duration_threshold=0.1,
                pixel_threshold=0.10,
                cache=True
            )
            blank_frames = set(blank_frames)  # For fast lookup
        except Exception as e:
            print(f"⚠️ Blank frame detection failed: {e}")
            blank_frames = []


    
    first_frame_tensor = frame_to_tensor(frame)

    if auto_crop_black_bars:
        top_crop, bottom_crop = detect_black_bars(first_frame_tensor)
        print(f"Auto-crop: Top {top_crop}px | Bottom {bottom_crop}px")
        first_frame_tensor = crop_black_bars_torch(first_frame_tensor, top_crop, bottom_crop)
    else:
        top_crop, bottom_crop = 0, 0

    target_ratio = aspect_ratios.get(selected_aspect_ratio.get(), 16 / 9)

    _, h, w = first_frame_tensor.shape
    current_ratio = w / h
    if abs(current_ratio - target_ratio) > 0.01:
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            w = new_w
        else:
            new_h = int(w / target_ratio)
            h = new_h
        
    if preserve_original_aspect:
        resized_width = original_video_width
        resized_height = original_video_height

        if output_format == "Full-SBS":
            per_eye_w = resized_width
            per_eye_h = resized_height
            out_width = per_eye_w * 2
            out_height = per_eye_h
        elif output_format == "Half-SBS":
            per_eye_w = resized_width // 2
            per_eye_h = resized_height
            out_width = resized_width
            out_height = resized_height
        elif output_format == "VR":
            per_eye_w = 1440
            per_eye_h = 1600
            out_width = per_eye_w * 2
            out_height = per_eye_h
        else:
            per_eye_w = resized_width
            per_eye_h = resized_height
            out_width = resized_width * 2
            out_height = resized_height
    else:
        resized_height = output_height
        resized_width = int(resized_height * target_ratio)
        if resized_width % 2 != 0:
            resized_width += 1

        if output_format == "Full-SBS":
            per_eye_w, per_eye_h = 1920, 1080
            out_width = per_eye_w * 2
            out_height = per_eye_h
        elif output_format == "Half-SBS":
            per_eye_w = resized_width // 2
            per_eye_h = resized_height
            out_width = resized_width
            out_height = resized_height
        elif output_format == "VR":
            per_eye_w = 1440
            per_eye_h = 1600
            out_width = per_eye_w * 2
            out_height = per_eye_h
        else:
            per_eye_w = resized_width
            per_eye_h = resized_height
            out_width = resized_width * 2
            out_height = resized_height


    if use_ffmpeg:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{out_width}x{out_height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", selected_ffmpeg_codec,
            "-preset", "slow",
            "-pix_fmt", "yuv420p"
        ]

        if selected_ffmpeg_codec.startswith("libx"):
            ffmpeg_cmd += ["-crf", str(crf_value)]
        elif "nvenc" in selected_ffmpeg_codec:
            ffmpeg_cmd += ["-cq", str(crf_value), "-b:v", "0"]

        ffmpeg_cmd.append(output_path)
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    else:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*selected_codec), fps, (out_width, out_height))

    start_time = time.time()
    prev_time = time.time()
    fps_values = []
    smoother = ShiftSmoother(0.15)
    global temporal_depth_filter
    temporal_depth_filter = TemporalDepthFilter(alpha=0.5)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    dcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for idx in range(total_frames):
        if cancel_flag.is_set():
            break

        while suspend_flag.is_set():
            if cancel_flag.is_set():
                break
            try:
                time.sleep(0.2)
            except KeyboardInterrupt:
                print("⚡ KeyboardInterrupt during suspend. Forcing cancel.")
                cancel_flag.set()
                break
            if progress:
                progress.update()
            if progress_label:
                progress_label.update()

            
        ret1, frame = cap.read()
        ret2, depth = dcap.read()
        if not ret1 or not ret2: break

        frame_tensor = frame_to_tensor(frame)
        depth_tensor = depth_to_tensor(depth)

        if auto_crop_black_bars:
            top_crop, bottom_crop = detect_black_bars(frame_tensor)
            print(f"🔪 Cropping top: {top_crop}px, bottom: {bottom_crop}px")
            frame_tensor = crop_black_bars_torch(frame_tensor, top_crop, bottom_crop)
            depth_tensor = crop_black_bars_torch(depth_tensor, top_crop, bottom_crop)

        _, h, w = frame_tensor.shape
        current_ratio = w / h
        if abs(current_ratio - target_ratio) > 0.01:
            if current_ratio > target_ratio:
                new_w = int(h * target_ratio)
                start = (w - new_w) // 2
                frame_tensor = frame_tensor[:, :, start:start + new_w]
                depth_tensor = depth_tensor[:, :, start:start + new_w]
            else:
                new_h = int(w / target_ratio)
                start = (h - new_h) // 2
                frame_tensor = frame_tensor[:, start:start + new_h, :]
                depth_tensor = depth_tensor[:, start:start + new_h, :]

        # Only recalculate target_eye_h if we're NOT preserving the original aspect
        if not preserve_original_aspect:
            cinema_aspect_ratio = aspect_ratios.get(selected_aspect_ratio.get(), 16 / 9)
            target_eye_w = per_eye_w
            target_eye_h = int(per_eye_w / cinema_aspect_ratio)
            if target_eye_h % 2 != 0:
                target_eye_h += 1
        else:
            # ✅ Use exact dimensions without recalculation
            target_eye_w = per_eye_w
            target_eye_h = per_eye_h


        frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=(target_eye_h, target_eye_w), mode='bilinear', align_corners=False).squeeze(0)
        depth_tensor = F.interpolate(depth_tensor.unsqueeze(0), size=(target_eye_h, target_eye_w), mode='bilinear', align_corners=False).squeeze(0)

        fg, mg, bg = smoother.smooth(fg_shift, mg_shift, bg_shift)
        if output_format in ["Full-SBS", "Half-SBS", "VR", "Red-Cyan Anaglyph", "Passive Interlaced"]:
            bg *= 2.0

        if idx in blank_frames:
            # 🔥 Detected blank frame: skip pixel shifting
            print(f"⏩ Skipping blank frame {idx}")
            left_frame = frame
            right_frame = frame
        else:
            left_frame, right_frame = pixel_shift_cuda(
                frame_tensor, depth_tensor, resized_width, resized_height,
                fg, mg, bg,
                blur_ksize=blur_ksize,
                feather_strength=feather_strength,
                use_subject_tracking=use_subject_tracking,
                enable_floating_window=use_floating_window,
                return_shift_map=False,
                max_pixel_shift_percent=max_pixel_shift_percent,
                convergence_offset=convergence_offset,
                enable_edge_masking=enable_edge_masking,
                enable_feathering=enable_feathering
            )

        subject_depth = estimate_subject_depth(depth_tensor)
        zero_parallax_offset = ((-subject_depth * fg) + (-subject_depth * mg) + (subject_depth * bg)) / (resized_width / 2)

        if use_floating_window and use_subject_tracking:
            shift_thresh = 0.005
            raw_bar_width = int(abs(zero_parallax_offset) * resized_width * 0.75)
            smoothed_bar_width = bar_easer.ease(raw_bar_width)
            bar_width = max(min(smoothed_bar_width, 80), 0)
            if zero_parallax_offset > shift_thresh:
                left_frame = apply_side_mask(left_frame, side="right", width=bar_width)
                right_frame = apply_side_mask(right_frame, side="right", width=bar_width)
            elif zero_parallax_offset < -shift_thresh:
                left_frame = apply_side_mask(left_frame, side="left", width=bar_width)
                right_frame = apply_side_mask(right_frame, side="left", width=bar_width)

        left_sharp = apply_sharpening(left_frame, sharpness_factor)
        right_sharp = apply_sharpening(right_frame, sharpness_factor)

        if output_format == "Full-SBS":
            left_out = pad_to_aspect_ratio(left_sharp, per_eye_w, per_eye_h)
            right_out = pad_to_aspect_ratio(right_sharp, per_eye_w, per_eye_h)
        elif output_format == "Half-SBS":
            left_out = cv2.resize(left_sharp, (per_eye_w, per_eye_h), interpolation=cv2.INTER_AREA)
            right_out = cv2.resize(right_sharp, (per_eye_w, per_eye_h), interpolation=cv2.INTER_AREA)
        else:
            left_out = pad_to_aspect_ratio(left_sharp, per_eye_w, per_eye_h)
            right_out = pad_to_aspect_ratio(right_sharp, per_eye_w, per_eye_h)

        final = format_3d_output(left_out, right_out, output_format)

        if use_ffmpeg:
            try:
                ffmpeg_proc.stdin.write(final.astype(np.uint8).tobytes())
            except Exception as e:
                print(f"❌ FFmpeg write error: {e}")
                break
        else:
            out.write(final)

        percent = (idx / total_frames) * 100
        elapsed = time.time() - start_time
        curr_time = time.time()
        delta = curr_time - prev_time
        if delta > 0:
            fps_values.append(1.0 / delta)
            if len(fps_values) > 10:
                fps_values.pop(0)
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

        if progress:
            progress["value"] = percent
            progress.update()
        if progress_label:
            progress_label.config(text=f"{percent:.2f}% | FPS: {avg_fps:.2f} | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
        prev_time = curr_time

    cap.release()
    dcap.release()

    if use_ffmpeg:
        try:
            ffmpeg_proc.stdin.close()
        except Exception as e:
            print(f"⚠️ Error closing FFmpeg stdin: {e}")

        try:
            if cancel_flag.is_set():
                ffmpeg_proc.kill()
            else:
                ffmpeg_proc.wait()
        except Exception as e:
            print(f"⚠️ Error terminating FFmpeg: {e}")
    else:
        out.release()

    torch.cuda.empty_cache()



def start_processing_thread():
    global process_thread
    cancel_flag.clear()  # ✅ Reset cancel state
    suspend_flag.clear()  # Ensure it's not paused
    process_thread = threading.Thread(target=process_video, daemon=True)
    process_thread.start()



def select_input_video(
    input_video_path,
    video_thumbnail_label,
    video_specs_label,
    update_aspect_preview,
    original_video_width,
    original_video_height
):


    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
    if not video_path:
        return

    input_video_path.set(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # ✅ Now this works without needing to import GUI.py
    original_video_width.set(width)
    original_video_height.set(height)
    
    current_video_width = width
    current_video_height = height
    
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((300, 200))
        img_tk = ImageTk.PhotoImage(img)

        video_thumbnail_label.config(image=img_tk)
        video_thumbnail_label.image = img_tk

        video_specs_label.config(text=f"Video Info:\nResolution: {width}x{height}\nFPS: {fps:.2f}")
    else:
        video_specs_label.config(text="Video Info:\nUnable to extract details")

    # ✅ Call the UI update function
    update_aspect_preview()

def update_thumbnail(thumbnail_path):
    thumbnail_image = Image.open(thumbnail_path)
    thumbnail_image = thumbnail_image.resize(
        (300, 250), Image.LANCZOS
    )  # Adjust the size as needed
    thumbnail_photo = ImageTk.PhotoImage(thumbnail_image)
    video_thumbnail_label.config(image=thumbnail_photo)
    video_thumbnail_label.image = thumbnail_photo


def select_output_video(output_sbs_video_path):
    output_sbs_video_path.set(
        filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("MKV files", "*.mkv"),
                ("AVI files", "*.avi"),
            ],
        )
    )


def select_depth_map(selected_depth_map, depth_map_label):
    depth_map_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
    )
    if not depth_map_path:
        return

    selected_depth_map.set(depth_map_path)
    depth_map_label.config(
        text=f"Selected Depth Map:\n{os.path.basename(depth_map_path)}"
    )

def process_video(
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
    convergence_offset,
    enable_edge_masking,
    enable_feathering,
    skip_blank_frames, 
):

    global original_video_width, original_video_height

    input_path = input_video_path.get()
    depth_path = selected_depth_map.get()
    output_path = output_sbs_video_path.get()

    if not input_path or not output_path or not depth_path:
        messagebox.showerror(
            "Error", "Please select input video, depth map, and output path."
        )
        return

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        messagebox.showerror("Error", "Unable to retrieve FPS from the input video.")
        return

    # 🧠 Save original dimensions globally
    original_video_width = width
    original_video_height = height
    
    # 🔄 Determine aspect ratio
    aspect_ratio = aspect_ratios.get(selected_aspect_ratio.get(), 16 / 9)
    format_selected = output_format.get()

    # 🧩 Calculate output dimensions based on selected format
    if preserve_original_aspect.get():
        output_width = width
        output_height = height
    else:
        if format_selected == "Full-SBS":
            output_width = width * 2
            output_height = height
        elif format_selected == "Half-SBS":
            output_width = width
            output_height = height
        elif format_selected == "VR":
            output_width = 4096
            output_height = int(output_width / aspect_ratio)
        else:
            output_width = width
            output_height = int(output_width / aspect_ratio)


    # 🟢 Start progress
    progress["value"] = 0
    progress_label.config(text="0%")
    progress.update()

    # 🔥 Start render process
    if format_selected in ["Full-SBS", "Half-SBS", "Red-Cyan Anaglyph", "Passive Interlaced"]:
        render_sbs_3d(
            input_path,
            depth_path,
            output_path,
            selected_codec.get(),
            fps,
            output_width,
            output_height,
            fg_shift.get(),
            mg_shift.get(),
            bg_shift.get(),
            sharpness_factor.get(),
            format_selected,
            selected_aspect_ratio,
            aspect_ratios,
            feather_strength=feather_strength.get(),
            blur_ksize=blur_ksize.get(),
            use_ffmpeg=use_ffmpeg.get(),
            selected_ffmpeg_codec=FFMPEG_CODEC_MAP[selected_ffmpeg_codec.get()],
            crf_value=crf_value.get(),
            use_subject_tracking=use_subject_tracking.get(),
            use_floating_window=use_floating_window.get(),
            max_pixel_shift_percent=max_pixel_shift.get(),
            progress=progress,
            progress_label=progress_label,
            suspend_flag=suspend_flag,
            cancel_flag=cancel_flag,
            auto_crop_black_bars=auto_crop_black_bars.get(),
            parallax_balance=parallax_balance.get(),
            preserve_original_aspect=preserve_original_aspect.get(),
            convergence_offset=convergence_offset.get(),
            enable_edge_masking=enable_edge_masking.get(),
            enable_feathering=enable_feathering.get(),
            skip_blank_frames=skip_blank_frames.get(),
        )




# Define SETTINGS_FILE at the top of the script
SETTINGS_FILE = "settings.json"
def render_with_ffmpeg(
    frame_generator,
    output_path,
    width,
    height,
    fps,
    codec_name="libx264",
    crf=23,
    nvenc_cq=23,
    preset="slow"
):
    """
    Stream raw frames to FFmpeg using stdin to encode with H.264/H.265 or NVENC.
    """
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", codec_name,
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        output_path
    ]

    # 🔁 Codec-dependent quality option
    if codec_name.startswith("libx"):
        ffmpeg_cmd.insert(ffmpeg_cmd.index("-pix_fmt"), "-crf")
        ffmpeg_cmd.insert(ffmpeg_cmd.index("-crf") + 1, str(crf))
    elif "nvenc" in codec_name:
        ffmpeg_cmd.insert(ffmpeg_cmd.index("-pix_fmt"), "-cq")
        ffmpeg_cmd.insert(ffmpeg_cmd.index("-cq") + 1, str(nvenc_cq))
        ffmpeg_cmd += ["-b:v", "0"]  # ✅ Important for NVENC constant quality

    print(f"🚀 Launching FFmpeg render: {codec_name} | CRF: {crf} | NVENC CQ: {nvenc_cq} ➜ {output_path}")
    try:
        with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE) as proc:
            for idx, frame in enumerate(frame_generator):
                if frame is None:
                    print(f"⚠️ Frame {idx} is None — skipping.")
                    continue
                h, w = frame.shape[:2]
                if (w != width or h != height):
                    print(f"⚠️ Frame {idx} has incorrect shape: {w}x{h} (expected {width}x{height}) — skipping.")
                    continue
                proc.stdin.write(frame.astype(np.uint8).tobytes())

            proc.stdin.close()
            proc.wait()
            print("✅ FFmpeg render complete.")
    except Exception as e:
        print(f"❌ FFmpeg render failed: {e}")

