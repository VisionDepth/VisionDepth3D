import os
import shutil
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from tqdm import tqdm
from PIL import Image, ImageTk
import cv2
import numpy as np
import time

# Audio Transfer Function
def transferAudio(sourceVideo, targetVideo):
    tempAudioFileName = "./temp/audio.mkv"
    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")
    os.system(f'ffmpeg -y -i "{sourceVideo}" -c:a copy -vn "{tempAudioFileName}"')

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    os.system(f'ffmpeg -y -i "{targetNoAudio}" -i "{tempAudioFileName}" -c copy "{targetVideo}"')

    if os.path.getsize(targetVideo) == 0:
        tempAudioFileName = "./temp/audio.m4a"
        os.system(f'ffmpeg -y -i "{sourceVideo}" -c:a aac -b:a 160k -vn "{tempAudioFileName}"')
        os.system(f'ffmpeg -y -i "{targetNoAudio}" -i "{tempAudioFileName}" -c copy "{targetVideo}"')
        if os.path.getsize(targetVideo) == 0:
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio.")
        else:
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    shutil.rmtree("temp")  # Cleanup temp files


# Function to detect and remove black bars
def remove_black_bars(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return frame, 0, 0, frame.shape[1], frame.shape[0]

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    if w < frame.shape[1] * 0.5 or h < frame.shape[0] * 0.5:
        return frame, 0, 0, frame.shape[1], frame.shape[0]

    return frame[y:y+h, x:x+w], x, y, w, h


def render_sbs_3d(input_video, depth_video, output_video, codec, fps, width, height, fg_shift, mg_shift, bg_shift,
                  sharpness_factor, delay_time=1/30, blend_factor=0.5, progress=None, progress_label=None):
    frame_delay = int(fps * delay_time)  # Number of frames corresponding to the delay time
    frame_buffer = []  # Buffer to store frames for temporal delay
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    original_cap = cv2.VideoCapture(input_video)
    depth_cap = cv2.VideoCapture(depth_video)

    prev_frame_gray = None
    rolling_diff = []
    max_rolling_frames = 5  # Number of frames to average for adaptive threshold
    
    print("Creating Half SBS video with Pulfrich effect, blending, and black bar removal")
    
    # Start tracking time
    start_time = time.time()

    # Get total number of frames
    total_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(total_frames):
        ret1, original_frame = original_cap.read()
        ret2, depth_frame = depth_cap.read()
        if not ret1 or not ret2:
            break

        # Calculate percentage
        percentage = (frame_idx / total_frames) * 100

        # Update progress bar
        if progress:
            progress["value"] = percentage
            progress.update()

        # Update percentage label
        if progress_label:
            progress_label.config(text=f"{percentage:.2f}%")

        # Calculate elapsed time and time remaining
        elapsed_time = time.time() - start_time
        if frame_idx > 0:
            time_per_frame = elapsed_time / frame_idx
            time_remaining = time_per_frame * (total_frames - frame_idx)
        else:
            time_remaining = 0  # No estimate initially

        # Format time values as MM:SS
        elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
        time_remaining_str = time.strftime("%M:%S", time.gmtime(time_remaining))

        # Update the progress bar and label
        if progress_label:
            progress_label.config(
                text=f"{percentage:.2f}% | Elapsed: {elapsed_time_str} | Remaining: {time_remaining_str}"
            )

        # Remove black bars
        cropped_frame, x, y, w, h = remove_black_bars(original_frame)
        cropped_resized_frame = cv2.resize(cropped_frame, (width, height))
        depth_frame_resized = cv2.resize(depth_frame, (width, height))

        # Convert to grayscale for scene change detection
        current_frame_gray = cv2.cvtColor(cropped_resized_frame, cv2.COLOR_BGR2GRAY)

        # Scene change detection
        if prev_frame_gray is not None:
            diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
            diff_score = np.sum(diff) / (width * height)

            rolling_diff.append(diff_score)
            if len(rolling_diff) > max_rolling_frames:
                rolling_diff.pop(0)
            avg_diff_score = np.mean(rolling_diff)

            adaptive_threshold = 50 if avg_diff_score < 100 else 75
            if avg_diff_score > adaptive_threshold:
                print(f"Scene change detected at frame {frame_idx} with diff {avg_diff_score:.2f}")
                frame_buffer.clear()  # Clear buffer for Pulfrich effect
                blend_factor = max(0.1, blend_factor - 0.2)  # Reduce blending for scene change

        prev_frame_gray = current_frame_gray

        # Pulfrich effect adjustments
        blend_factor = min(0.5, blend_factor + 0.05) if len(frame_buffer) else blend_factor

        # Process Depth frame
        depth_map_gray = cv2.cvtColor(depth_frame_resized, cv2.COLOR_BGR2GRAY)
        depth_normalized = cv2.normalize(depth_map_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_filtered = cv2.bilateralFilter(depth_normalized, d=5, sigmaColor=50, sigmaSpace=50)
        depth_normalized = depth_filtered / 255.0

        # Apply depth-based shifts
        left_frame, right_frame = cropped_resized_frame.copy(), cropped_resized_frame.copy()
        for y in range(height):
            shift_vals = (depth_normalized[y, :] * fg_shift).astype(np.int32)
            new_x_left = np.clip(np.arange(width) + shift_vals, 0, width - 1)
            new_x_right = np.clip(np.arange(width) - shift_vals, 0, width - 1)
            left_frame[y, new_x_left] = cropped_resized_frame[y, np.arange(width)]
            right_frame[y, new_x_right] = cropped_resized_frame[y, np.arange(width)]

        # Buffer logic
        frame_buffer.append((left_frame, right_frame))
        if len(frame_buffer) > frame_delay:
            delayed_left_frame, delayed_right_frame = frame_buffer.pop(0)
        else:
            delayed_left_frame, delayed_right_frame = left_frame, right_frame

        # Create Pulfrich effect
        blended_left_frame = cv2.addWeighted(delayed_left_frame, blend_factor, left_frame, 1 - blend_factor, 0)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5 + sharpness_factor, -1], [0, -1, 0]])
        left_sharp = cv2.filter2D(blended_left_frame, -1, sharpen_kernel)
        right_sharp = cv2.filter2D(right_frame, -1, sharpen_kernel)

        left_sharp_resized = cv2.resize(left_sharp, (width // 2, height))
        right_sharp_resized = cv2.resize(right_sharp, (width // 2, height))

        # Combine into SBS format
        sbs_frame = np.hstack((left_sharp_resized, right_sharp_resized))
        out.write(sbs_frame)

    original_cap.release()
    depth_cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Half SBS video generated successfully.")


def select_input_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
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
        video_thumbnail_label.image = img_tk  # Save a reference to prevent garbage collection

        video_specs_label.config(
            text=f"Video Info:\nResolution: {width}x{height}\nFPS: {fps:.2f}"
        )
    else:
        video_specs_label.config(
            text="Video Info:\nUnable to extract details"
        )


def select_output_video():
    output_sbs_video_path.set(filedialog.asksaveasfilename(
        defaultextension=".mp4",
        filetypes=[("MP4 files", "*.mp4"), ("MKV files", "*.mkv"), ("AVI files", "*.avi")]
    ))


def process_video():
    if not input_video_path.get() or not output_sbs_video_path.get():
        messagebox.showerror("Error", "Please select both input and output paths.")
        return

    cap = cv2.VideoCapture(input_video_path.get())
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        messagebox.showerror("Error", "Unable to retrieve FPS from the input video.")
        return

    os.makedirs("Depthmaps", exist_ok=True)
    input_video_name = os.path.basename(input_video_path.get())
    depth_output_path = os.path.join("Depthmaps", f"depth_{input_video_name}")

    progress["value"] = 0
    progress_label.config(text="0%")
    progress.update()

    # Render the video
    render_sbs_3d(
        input_video_path.get(),
        depth_output_path,
        output_sbs_video_path.get(),
        selected_codec.get(),
        fps,
        width,
        height,
        fg_shift.get(),
        mg_shift.get(),
        bg_shift.get(),
        sharpness_factor.get(),
        delay_time=delay_time.get(),
        blend_factor=blend_factor.get(),
        progress=progress,
        progress_label=progress_label
    )

    # Set progress bar to 100% after rendering
    progress["value"] = 100
    progress_label.config(text="100%")
    progress.update()

    # Add Audio to the Generated SBS Video
    try:
        transferAudio(input_video_path.get(), output_sbs_video_path.get())
        progress_label.config(text="Complete")  # Indicate completion
        print("Audio transfer complete.")
    except Exception as e:
        print(f"Audio transfer failed: {e}")
        progress_label.config(text="Audio Error")  # Indicate an error
        messagebox.showwarning("Warning", "The video was generated without audio.")

root = tk.Tk()
root.title("3D Video Generator")
root.geometry("1280x920")

background_image = Image.open("C:/Users/johna/build/Program/assets/Background.png")
background_image = background_image.resize((1280, 920), Image.LANCZOS)
bg_image = ImageTk.PhotoImage(background_image)

background_label = tk.Label(root, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

content_frame = tk.Frame(root, highlightthickness=0, bd=0)
content_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.3, relheight=0.8)

input_video_path = tk.StringVar()
output_sbs_video_path = tk.StringVar()
selected_codec = tk.StringVar(value="mp4v")
fg_shift = tk.DoubleVar(value=6.0)
mg_shift = tk.DoubleVar(value=3.0)
bg_shift = tk.DoubleVar(value=-4.0)
sharpness_factor = tk.DoubleVar(value=0.2)
blend_factor = tk.DoubleVar(value=0.6)
delay_time = tk.DoubleVar(value=1/30)

top_frame = tk.Frame(content_frame)
top_frame.pack(pady=10)

tk.Label(top_frame, text="Codec", fg="black").pack()
tk.Entry(top_frame, textvariable=selected_codec, width=10, bg="white").pack(pady=5)

tk.Label(top_frame, text="Foreground Shift", fg="black").pack()
tk.Entry(top_frame, textvariable=fg_shift, width=5, bg="white").pack()

tk.Label(top_frame, text="Midground Shift", fg="black").pack()
tk.Entry(top_frame, textvariable=mg_shift, width=5, bg="white").pack()

tk.Label(top_frame, text="Background Shift", fg="black").pack()
tk.Entry(top_frame, textvariable=bg_shift, width=5, bg="white").pack()

tk.Label(top_frame, text="Sharpness Factor", fg="black").pack()
tk.Scale(top_frame, from_=-1, to=1, resolution=0.1, orient=tk.HORIZONTAL, variable=sharpness_factor).pack()

tk.Label(top_frame, text="Blend Factor (Pulfrich Effect)", fg="black").pack()
tk.Scale(top_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=blend_factor).pack()

tk.Label(top_frame, text="Delay Time (seconds)", fg="black").pack()
tk.Scale(top_frame, from_=1/50, to=1/20, resolution=0.001, orient=tk.HORIZONTAL, variable=delay_time).pack()

tk.Button(top_frame, text="Select Input Video", command=select_input_video).pack(pady=5)
tk.Entry(top_frame, textvariable=input_video_path, width=50).pack(pady=5)

# Add a frame for video info
video_info_frame = tk.Frame(content_frame)
video_info_frame.pack(pady=10)

video_thumbnail_label = tk.Label(video_info_frame, text="No Thumbnail", bg="white", width=10, height=4)
video_thumbnail_label.grid(row=0, column=0, padx=10, pady=5)

video_specs_label = tk.Label(video_info_frame, text="Video Info:\nResolution: N/A\nFPS: N/A", anchor="w", justify="left")
video_specs_label.grid(row=0, column=1, padx=10, pady=5)


tk.Button(top_frame, text="Select Output Video", command=select_output_video).pack(pady=5)
tk.Entry(top_frame, textvariable=output_sbs_video_path, width=50).pack(pady=5)

tk.Button(top_frame, text="Generate 3D SBS Video", command=process_video, bg="white").pack(pady=10)

progress = ttk.Progressbar(content_frame, orient="horizontal", length=300, mode="determinate")
progress.pack(pady=10)

progress_label = tk.Label(content_frame, text="0%", font=("Arial", 10))
progress_label.pack(pady=5)

root.mainloop()
