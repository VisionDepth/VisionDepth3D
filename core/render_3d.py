import os
import time
import cv2
import torch
import numpy as np
import threading
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import onnxruntime as ort
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from collections import deque

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

# Converters

def frame_to_tensor(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
    return frame_tensor.to(torch_device)

def depth_to_tensor(depth_frame):
    depth_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
    depth_tensor = torch.from_numpy(depth_gray).float().unsqueeze(0) / 255.0
    return depth_tensor.to(torch_device)

# Bilateral smoothing for depth (preserves edges)
def bilateral_smooth_depth(depth_tensor):
    depth_np = depth_tensor.squeeze().cpu().numpy().astype(np.uint8)
    smoothed = cv2.bilateralFilter(depth_np, d=9, sigmaColor=75, sigmaSpace=75)
    smoothed_tensor = torch.from_numpy(smoothed).float().unsqueeze(0) / 255.0
    return smoothed_tensor.to(depth_tensor.device)

# Gradient-aware shift suppression
def suppress_artifacts_with_edge_mask(depth_tensor, total_shift, threshold=0.15):
    dx = torch.abs(F.pad(depth_tensor[:, :, 1:] - depth_tensor[:, :, :-1], (1, 0)))
    dy = torch.abs(F.pad(depth_tensor[:, 1:, :] - depth_tensor[:, :-1, :], (0, 0, 1, 0)))
    grad_mag = torch.sqrt(dx**2 + dy**2)
    edge_mask = (grad_mag > threshold).float()
    return total_shift * (1.0 - edge_mask)

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

# Pixel Shift

def pixel_shift_cuda(frame_tensor, depth_tensor, width, height, fg_shift, mg_shift, bg_shift):
    # Resize inputs
    frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
    depth_tensor = F.interpolate(depth_tensor.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)

    # ✅ Step 1: Blur depth to reduce artifacts
    #depth_tensor = bilateral_smooth_depth(depth_tensor)

    # Optional: temporal smoothing
    global temporal_depth_filter
    #depth_tensor = temporal_depth_filter.smooth(depth_tensor)


    # ✅ Step 2: Compute pixel shift
    fg_shift_tensor = (-depth_tensor * fg_shift) / (width / 2)
    mg_shift_tensor = (-depth_tensor * mg_shift) / (width / 2)
    bg_shift_tensor = (depth_tensor * bg_shift) / (width / 2)
    total_shift = fg_shift_tensor + mg_shift_tensor + bg_shift_tensor

    # Clamp and mask sharp edges
    max_shift_px = width * 0.05
    max_shift_norm = max_shift_px / (width / 2)
    total_shift = torch.clamp(total_shift, -max_shift_norm, max_shift_norm)

    # Apply gradient-aware artifact suppression
    #total_shift = suppress_artifacts_with_edge_mask(depth_tensor, total_shift)

    # Generate remap grid
    H, W = depth_tensor.shape[1], depth_tensor.shape[2]
    x = torch.linspace(-1, 1, W, device=frame_tensor.device).repeat(H, 1)
    y = torch.linspace(-1, 1, H, device=frame_tensor.device).unsqueeze(1).repeat(1, W)
    grid = torch.stack((x, y), dim=2)

    # Shift x-coordinates
    shift_vals = total_shift.squeeze(0)
    grid_left = grid.clone()
    grid_right = grid.clone()
    grid_left[:, :, 0] += shift_vals
    grid_right[:, :, 0] -= shift_vals

    # Use 'reflection' padding to reduce outer edge distortion
    left = F.grid_sample(frame_tensor.unsqueeze(0), grid_left.unsqueeze(0),
                         mode='bilinear', padding_mode='reflection', align_corners=True).squeeze(0)

    right = F.grid_sample(frame_tensor.unsqueeze(0), grid_right.unsqueeze(0),
                          mode='bilinear', padding_mode='reflection', align_corners=True).squeeze(0)

    return tensor_to_frame(left), tensor_to_frame(right)



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

#def correct_convergence_shift_torch(left_tensor, right_tensor, depth_tensor, session, input_name, output_name, bg_threshold=0.3):
#    depth_np = depth_tensor.squeeze().cpu().numpy()
#    depth_norm = cv2.normalize(depth_np, None, 0.1, 1.0, cv2.NORM_MINMAX)
#    background_mask = (depth_norm >= bg_threshold).astype(np.float32)

#    warp_input = np.array([[np.mean(depth_norm)]], dtype=np.float32)
#    warp_matrix = session.run([output_name], {input_name: warp_input})[0].reshape(3, 3)
#    warp_matrix = torch.from_numpy(warp_matrix).float().to(left_tensor.device)

#    b, h, w = 1, left_tensor.shape[1], left_tensor.shape[2]
#    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
#    ones = torch.ones_like(xx)
#    base_grid = torch.stack([xx, yy, ones], dim=0).reshape(3, -1).to(left_tensor.device).float()

#    warped_grid = torch.matmul(warp_matrix, base_grid.clone())
#    warped_grid = warped_grid.clone()  # Fix memory overlap issue
#   warped_grid = warped_grid / warped_grid[2:3, :].clone()

#    x_warp = (warped_grid[0] / (w - 1)) * 2 - 1
#    y_warp = (warped_grid[1] / (h - 1)) * 2 - 1
#    grid = torch.stack((x_warp, y_warp), dim=1).reshape(h, w, 2)

#    warped_left = F.grid_sample(left_tensor.unsqueeze(0), grid.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
#    warped_right = F.grid_sample(right_tensor.unsqueeze(0), grid.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()

#   warped_left_img = tensor_to_frame(warped_left)
#   warped_right_img = tensor_to_frame(warped_right)

#    mask_left = cv2.inRange(warped_left_img, (0, 0, 0), (10, 10, 10))
#    mask_right = cv2.inRange(warped_right_img, (0, 0, 0), (10, 10, 10))

#    inpainted_left = cv2.inpaint(warped_left_img, mask_left, 10, cv2.INPAINT_NS)
#    inpainted_right = cv2.inpaint(warped_right_img, mask_right, 10, cv2.INPAINT_NS)

#    return inpainted_left, inpainted_right



# 3D Format

def format_3d_output(left, right, fmt):
    h, w = left.shape[:2]
    if fmt == "Half-SBS":
        lw = cv2.resize(left, (w // 2, h))
        rw = cv2.resize(right, (w // 2, h))
        return np.hstack((lw, rw))
    elif fmt == "Full-SBS":
        return np.hstack((left, right))
    elif fmt == "VR":
        lw = cv2.resize(left, (1440, 1600))
        rw = cv2.resize(right, (1440, 1600))
        return np.hstack((lw, rw))
    elif fmt == "Red-Cyan Anaglyph":
        return generate_anaglyph_3d(left, right)
    return np.hstack((left, right))

def generate_anaglyph_3d(left_frame, right_frame):
    """Creates a properly balanced True Red-Cyan Anaglyph 3D effect."""
    left_frame = left_frame.astype(np.float32)
    right_frame = right_frame.astype(np.float32)

    left_r, left_g, left_b = cv2.split(left_frame)
    right_r, right_g, right_b = cv2.split(right_frame)

    anaglyph = cv2.merge([
        right_b * 0.6,
        right_g * 0.7,
        left_r * 0.9
    ])
    anaglyph = np.clip(anaglyph, 0, 255).astype(np.uint8)
    return anaglyph


# Render

def render_sbs_3d(input_path, depth_path, output_path, codec, fps, width, height, fg_shift, mg_shift, bg_shift,
                  sharpness_factor, output_format, selected_aspect_ratio, aspect_ratios, delay_time=1/30,
                  blend_factor=0.5, progress=None, progress_label=None,
                  suspend_flag=None, cancel_flag=None):

    cap, dcap = cv2.VideoCapture(input_path), cv2.VideoCapture(depth_path)
    if not cap.isOpened() or not dcap.isOpened(): return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_width = width if output_format == "Half-SBS" else width * 2
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, (out_width, height))

    smoother = ShiftSmoother(0.15)
    frame_buffer, start_time, prev_time, fps_values = [], time.time(), time.time(), []
    frame_delay = int(fps * delay_time)
    global temporal_depth_filter
    temporal_depth_filter = TemporalDepthFilter(alpha=0.85)


    for idx in range(total_frames):
        if cancel_flag and cancel_flag.is_set(): break
        while suspend_flag and suspend_flag.is_set(): time.sleep(0.5)

        ret1, frame = cap.read()
        ret2, depth = dcap.read()
        if not ret1 or not ret2: break

        frame = cv2.resize(frame, (width, height))
        depth = cv2.resize(depth, (width, height))
        frame_tensor, depth_tensor = frame_to_tensor(frame), depth_to_tensor(depth)

        fg, mg, bg = fg_shift, mg_shift, bg_shift  # use exact slider values
        fg, mg, bg = smoother.smooth(fg, mg, bg)

        left_frame, right_frame = pixel_shift_cuda(frame_tensor, depth_tensor, width, height, fg, mg, bg)

        # Apply ONNX warp correction (torch version)
        #left_img, right_img = correct_convergence_shift_torch(
        #    frame_tensor, frame_tensor, depth_tensor, session, input_name, output_name
        #) 

        # Then apply pixel shift CUDA after this
        #frame_tensor = frame_to_tensor(left_img)  # 🔁 Convert corrected image back to tensor
        #left_frame, right_frame = pixel_shift_cuda(frame_tensor, depth_tensor, width, height, fg, mg, bg)

        frame_buffer.append((left_frame, right_frame))

        delayed = frame_buffer.pop(0)[0] if len(frame_buffer) > frame_delay else left_frame
        blended_left = cv2.addWeighted(delayed, blend_factor, left_frame, 1 - blend_factor, 0)

        left_sharp = apply_sharpening(blended_left, sharpness_factor)
        right_sharp = apply_sharpening(right_frame, sharpness_factor)

        final = format_3d_output(left_sharp, right_sharp, output_format)
        out.write(final)

        percent = (idx / total_frames) * 100
        elapsed = time.time() - start_time
        curr_time = time.time()
        delta = curr_time - prev_time
        if delta > 0:
            fps_values.append(1.0 / delta)
            if len(fps_values) > 10: fps_values.pop(0)
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

        if progress: progress["value"] = percent; progress.update()
        if progress_label:
            progress_label.config(
                text=f"{percent:.2f}% | FPS: {avg_fps:.2f} | Elapsed: {time.strftime('%M:%S', time.gmtime(elapsed))}"
            )
        prev_time = curr_time

    cap.release(); dcap.release(); out.release()
    print("🎉 3D video rendering complete.")


    

def start_processing_thread():
    global process_thread
    cancel_flag.clear()  # Reset cancel state
    suspend_flag.clear()  # Ensure it's not paused
    process_thread = threading.Thread(target=process_video, daemon=True)
    process_thread.start()


def select_input_video(input_video_path, video_thumbnail_label, video_specs_label):
    video_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
    )

    if not video_path:
        return

    input_video_path.set(video_path)
    
    # Extract video specs
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read the first frame to generate a thumbnail
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Convert the frame to an image compatible with Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((300, 200))  # Resize thumbnail
        img_tk = ImageTk.PhotoImage(img)

        # Update the GUI
        video_thumbnail_label.config(image=img_tk)
        video_thumbnail_label.image = (
            img_tk  # Save a reference to prevent garbage collection
        )

        video_specs_label.config(
            text=f"Video Info:\nResolution: {width}x{height}\nFPS: {fps:.2f}"
        )
    else:
        video_specs_label.config(text="Video Info:\nUnable to extract details")


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
    blend_factor,
    delay_time,
    output_format,
    selected_aspect_ratio,   # 👈 now passed
    aspect_ratios,           # 👈 also passed
    progress_bar,
    progress_label,
    suspend_flag,
    cancel_flag,
):


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

    progress_bar["value"] = 0
    progress_label.config(text="0%")
    progress_bar.update()

    output_format = output_format.get()
    aspect_ratio = aspect_ratios.get(selected_aspect_ratio.get(), 16 / 9)

    # Run rendering
    if output_format in ["Full-SBS", "Half-SBS"]:
        render_sbs_3d(
            input_path,
            depth_path,
            output_path,
            selected_codec.get(),
            fps,
            width,
            height,
            fg_shift.get(),
            mg_shift.get(),
            bg_shift.get(),
            sharpness_factor.get(),
            output_format,
            selected_aspect_ratio,
            aspect_ratios,
            delay_time=delay_time.get(),
            blend_factor=blend_factor.get(),
            progress=progress_bar,
            progress_label=progress_label,
            suspend_flag=suspend_flag,
            cancel_flag=cancel_flag,
        )

    if not cancel_flag.is_set():
        progress_bar["value"] = 100
        progress_label.config(text="100%")
        progress_bar.update()
        print("✅ Processing complete.")


# Define SETTINGS_FILE at the top of the script
SETTINGS_FILE = "settings.json"


def open_github():
    """Opens the GitHub repository in a web browser."""
    webbrowser.open_new(
        "https://github.com/VisionDepth/VisionDepth3D"
    )  # Replace with your actual GitHub URL
    

