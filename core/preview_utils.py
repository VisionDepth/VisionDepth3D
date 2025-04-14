import cv2
import torch
import torch.nn.functional as F
import numpy as np
import re
from render_3d import pixel_shift_cuda
from render_3d import frame_to_tensor, depth_to_tensor


def grab_frame_from_video(video_path, frame_idx=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def generate_preview_image(preview_type, left, right, shift_map, w, h):
    if preview_type == "Passive Interlaced":
        interlaced = np.zeros_like(left)
        interlaced[::2] = left[::2]
        interlaced[1::2] = right[1::2]
        return interlaced

    elif preview_type == "HSBS":
        half_w = w // 2
        left_resized = cv2.resize(left, (half_w, h))
        right_resized = cv2.resize(right, (half_w, h))
        return np.hstack((left_resized, right_resized))

    elif preview_type == "Shift Heatmap":
        shift_np = shift_map.cpu().numpy()
        shift_norm = cv2.normalize(shift_np, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(shift_norm.astype(np.uint8), cv2.COLORMAP_JET)

    elif preview_type == "Shift Heatmap (Abs)":
        shift_abs = np.abs(shift_map.cpu().numpy())
        shift_norm = cv2.normalize(shift_abs, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(shift_norm.astype(np.uint8), cv2.COLORMAP_JET)

    elif preview_type == "Shift Heatmap (Clipped ±5px)":
        shift_np = shift_map.cpu().numpy()
        max_disp = 5.0
        shift_clipped = np.clip(shift_np, -max_disp, max_disp)
        shift_norm = ((shift_clipped + max_disp) / (2 * max_disp)) * 255
        return cv2.applyColorMap(shift_norm.astype(np.uint8), cv2.COLORMAP_JET)

    elif preview_type == "Left-Right Diff":
        diff = cv2.absdiff(left, right)
        return diff

    elif preview_type == "Feather Blend":
        return left

    elif preview_type == "Feather Mask":
        shift_np = shift_map.cpu().numpy()
        feather_mask = np.clip(np.abs(shift_np) * 50, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(feather_mask, cv2.COLORMAP_BONE)

    elif preview_type == "Red-Blue Anaglyph":
        red_channel = left[:, :, 2]
        green_channel = right[:, :, 1]
        blue_channel = right[:, :, 0]
        anaglyph = cv2.merge((blue_channel, green_channel, red_channel))
        return anaglyph

    elif preview_type == "Overlay Arrows":
        debug = left.copy()
        shift_np = shift_map.cpu().numpy()
        step = 20
        for y in range(0, h, step):
            for x in range(0, w, step):
                dx = int(shift_np[y, x] * 10)
                if abs(dx) > 1:
                    cv2.arrowedLine(debug, (x, y), (x + dx, y), (0, 255, 0), 1, tipLength=0.3)
        return debug

    return None
