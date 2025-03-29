import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import time
import subprocess
import platform
import signal
import threading
import os

# ───────────────────────────── Globals ─────────────────────────────
cap = None
is_playing = False
playback_loop_id = None
fullscreen_window = None
fullscreen_label = None
audio_process = None
current_video_path = None
last_frame_time = 0

# ───────────────────────────── Utils ───────────────────────────────
def format_time(ms):
    total_seconds = int(ms / 1000)
    return f"{total_seconds // 60:02}:{total_seconds % 60:02}"

def resize_frame(frame, max_w, max_h):
    h, w, _ = frame.shape
    scale = min(max_w / w, max_h / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

def update_seek_and_time(seek, label):
    if not cap:
        return
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_time = (total_frames / fps) * 1000
    seek.config(to=total_frames)
    seek.set(current_frame)
    label.config(text=f"{format_time(current_time)} / {format_time(total_time)}")

# ───────────────────────────── Core Player ───────────────────────────────
def load_video(video_frame, seek, label):
    global cap, is_playing, current_video_path, last_frame_time

    # 🔁 Reset previous state (in case user loads new video after Stop)
    if cap:
        try:
            cap.release()
        except:
            pass
        cap = None

    is_playing = False
    last_frame_time = 0

    file = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
    if file:
        current_video_path = file
        cap = cv2.VideoCapture(file)

        if not cap or not cap.isOpened():
            print("❌ Could not open video.")
            return

        is_playing = True
        threading.Thread(target=play_audio_with_ffplay, args=(file,), daemon=True).start()
        play_video(video_frame, seek, label)


def play(video_frame, seek, label):
    global is_playing, last_frame_time
    if not cap:
        return
    is_playing = True
    last_frame_time = time.time()
    threading.Thread(target=play_audio, args=(current_video_path,), daemon=True).start()
    play_video(video_frame, seek, label)

def pause_video(video_frame, seek, label):
    global is_playing, playback_loop_id
    is_playing = False
    
    if playback_loop_id:
        video_frame.after_cancel(playback_loop_id)
        playback_loop_id = None

def stop_video(video_frame, seek, label):
    global cap, is_playing, playback_loop_id
    is_playing = False
    

    if playback_loop_id:
        try:
            video_frame.after_cancel(playback_loop_id)
        except Exception:
            pass
        playback_loop_id = None

    if cap:
        try:
            cap.release()
        except Exception:
            pass
        cap = None

    # Reset visual
    video_frame.config(
        image='',
        text="🎞️ No video loaded",
        font=("Helvetica", 13, "italic"),
        fg="gray",
        anchor="center"
    )

    if seek:
        seek.set(0)
    if label:
        label.config(text="00:00 / 00:00")

    close_fullscreen()

def play_video(video_frame, seek, label):
    global cap, is_playing, playback_loop_id, last_frame_time
    if not cap or not is_playing:
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    delay = 1000 // int(fps)
    ret, frame = cap.read()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = resize_frame(frame, 800, 450)
        img = Image.fromarray(resized)
        imgtk = ImageTk.PhotoImage(image=img)
        video_frame.imgtk = imgtk
        video_frame.config(image=imgtk, width=resized.shape[1], height=resized.shape[0])
        update_seek_and_time(seek, label)
        playback_loop_id = video_frame.after(delay, lambda: play_video(video_frame, seek, label))
    else:
        stop_video(video_frame, seek, label)

def seek_video(val):
    if cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(val)))

# ───────────────────────────── Fullscreen ───────────────────────────────
def open_fullscreen(video_frame):
    global fullscreen_window, fullscreen_label, is_playing
    if not cap:
        return

    fullscreen_window = tk.Toplevel()
    fullscreen_window.attributes("-fullscreen", True)
    fullscreen_window.configure(bg="black")
    fullscreen_label = tk.Label(fullscreen_window, bg="black")
    fullscreen_label.pack(expand=True)
    fullscreen_window.bind("<Escape>", lambda e: close_fullscreen())
    is_playing = True
    play_fullscreen()

def play_fullscreen():
    global cap, is_playing, fullscreen_label
    if cap and is_playing and fullscreen_label and fullscreen_label.winfo_exists():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = resize_frame(frame, fullscreen_window.winfo_screenwidth(), fullscreen_window.winfo_screenheight())
            img = Image.fromarray(resized)
            imgtk = ImageTk.PhotoImage(image=img)
            fullscreen_label.imgtk = imgtk
            fullscreen_label.config(image=imgtk)
            fullscreen_label.after(30, play_fullscreen)
        else:
            close_fullscreen()

def close_fullscreen():
    global fullscreen_window
    if fullscreen_window and fullscreen_window.winfo_exists():
        fullscreen_window.destroy()
        fullscreen_window = None
