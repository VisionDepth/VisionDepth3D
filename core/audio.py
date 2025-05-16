import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import subprocess
import shutil

def get_ffmpeg_path():
    """Get the path to ffmpeg executable."""
    path = shutil.which("ffmpeg")
    if not path:
        messagebox.showerror("FFmpeg Not Found", "❌ FFmpeg is not installed or not in system PATH.")
        raise FileNotFoundError("FFmpeg not found in system PATH.")
    return path

def rip_audio(source_path, output_audio_path, audio_codec='copy', bitrate=None):
    try:
        ffmpeg_path = get_ffmpeg_path()
        cmd = [ffmpeg_path, "-y", "-i", source_path, "-map", "0:a", "-c:a", audio_codec]

        if bitrate and audio_codec != "copy":
            cmd.extend(["-b:a", bitrate])

        cmd.append(output_audio_path)

        subprocess.run(cmd, check=True)
        messagebox.showinfo("Success", f"✅ Audio ripped to:\n{output_audio_path}")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"❌ FFmpeg failed:\n{e}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def attach_audio(video_path, audio_path, output_path):
    try:
        ffmpeg_path = get_ffmpeg_path()
        cmd = [
            ffmpeg_path, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            output_path
        ]

        subprocess.run(cmd, check=True)
        messagebox.showinfo("Success", f"✅ Audio attached to video:\n{output_path}")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"❌ FFmpeg failed:\n{e}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def launch_audio_gui():
    root = tk.Toplevel()
    root.title("🎵 Audio Ripper & Attacher")
    root.geometry("500x500")

    # Paths
    source_video = tk.StringVar()
    ripped_audio = tk.StringVar()
    rendered_video = tk.StringVar()
    output_video = tk.StringVar()
    bitrate = tk.StringVar(value="192k")
    codec = tk.StringVar(value="aac")

    def browse_video(var): var.set(filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mkv *.avi")]))
    def browse_audio(): ripped_audio.set(filedialog.askopenfilename(filetypes=[("Audio", "*.aac *.wav *.mp3")]))
    def browse_save_audio(): ripped_audio.set(filedialog.asksaveasfilename(defaultextension=".aac"))
    def browse_output_video(): output_video.set(filedialog.asksaveasfilename(defaultextension=".mp4"))

    # --- Ripping Section ---
    tk.Label(root, text="🎬 Source Video:").pack()
    tk.Entry(root, textvariable=source_video, width=50).pack()
    tk.Button(root, text="Browse", command=lambda: browse_video(source_video)).pack()

    tk.Label(root, text="💾 Save Audio As:").pack()
    tk.Entry(root, textvariable=ripped_audio, width=50).pack()
    tk.Button(root, text="Save As", command=browse_save_audio).pack()

    tk.Label(root, text="🎚 Codec:").pack()
    ttk.Combobox(root, textvariable=codec, values=["copy", "aac", "ac3", "eac3", "mp3", "flac", "wav"]).pack()

    tk.Label(root, text="🎛 Bitrate:").pack()
    ttk.Combobox(root, textvariable=bitrate, values=["128k", "192k", "256k", "320k"]).pack()

    tk.Button(root, text="🎧 Rip Audio", command=lambda: rip_audio(source_video.get(), ripped_audio.get(), codec.get(), bitrate.get())).pack(pady=5)

    # --- Attaching Section ---
    tk.Label(root, text="🎥 Rendered 3D Video:").pack()
    tk.Entry(root, textvariable=rendered_video, width=50).pack()
    tk.Button(root, text="Browse", command=lambda: browse_video(rendered_video)).pack()

    tk.Label(root, text="🎼 Audio File to Attach:").pack()
    tk.Entry(root, textvariable=ripped_audio, width=50).pack()
    tk.Button(root, text="Browse", command=browse_audio).pack()

    tk.Label(root, text="💾 Output Video With Audio:").pack()
    tk.Entry(root, textvariable=output_video, width=50).pack()
    tk.Button(root, text="Save As", command=browse_output_video).pack()

    tk.Button(root, text="📎 Attach Audio", command=lambda: attach_audio(rendered_video.get(), ripped_audio.get(), output_video.get())).pack(pady=10)

    root.mainloop()
