# === Drop-In FFmpeg Black/White Frame Detector for VisionDepth3D ===

import subprocess
import re
import os
import json

def get_video_fps(input_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        rate = result.stdout.strip()
        num, den = map(int, rate.split('/'))
        return num / den
    except Exception as e:
        print(f"[Warning] Failed to get FPS with ffprobe: {e}")
        return 30  # fallback fps

def detect_black_white_frames(input_path, mode="black", duration_threshold=0.1, pixel_threshold=0.10, cache=True):
    """
    Detects black or white frames using FFmpeg blackdetect filter.

    Args:
        input_path (str): Path to the input video.
        mode (str): 'black' or 'white' detection.
        duration_threshold (float): Minimum duration to trigger detection.
        pixel_threshold (float): Pixel threshold for detection.
        cache (bool): Use cached results if available.

    Returns:
        List[int]: List of frame indices to skip pixel shifting.
    """
    cache_file = input_path + ".blankcache.json"
    if cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            pass

    fps = get_video_fps(input_path)
    skip_frames = []

    if mode == "black":
        filter_cmd = f"blackdetect=d={duration_threshold}:pix_th={pixel_threshold}"
    elif mode == "white":
        filter_cmd = f"lutrgb='r=max(val\,240):g=max(val\,240):b=max(val\,240)',blackdetect=d={duration_threshold}:pix_th={pixel_threshold}"
    else:
        raise ValueError("mode must be 'black' or 'white'")

    command = [
        "ffmpeg", "-i", input_path,
        "-vf", filter_cmd,
        "-an", "-f", "null", "-"
    ]

    try:
        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        output = result.stderr

        matches = re.findall(r"black_start:(\d+\.\d+)", output)
        for time_sec in matches:
            frame_idx = int(float(time_sec) * fps)
            skip_frames.append(frame_idx)

        if cache:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(skip_frames, f)
            except:
                pass

        return sorted(skip_frames)

    except Exception as e:
        print(f"[Warning] FFmpeg frame detect failed: {e}")
        return []

# === Example of integration ===

# Somewhere before starting render_3d processing:
# blank_frames = detect_black_white_frames(input_video_path, mode="black")

# Then inside render loop:
# if idx in blank_frames:
#     # Skip shifting logic, direct copy or minimal processing
# else:
#     # Perform normal 3D pixel shifting

# === Optional Enhancements ===
# - Toggle black vs white detection by user checkbox
# - Add sensitivity slider for duration_threshold and pixel_threshold
# - Manual override to edit skip frame list

# === Notes ===
# - FFmpeg + FFprobe must be installed and in PATH
# - Detection is very fast (~10–30 seconds for full movies)
# - Minimal overhead, huge benefit for real-time 3D quality during scene fades
# - Automatic caching prevents re-scanning same video every time
