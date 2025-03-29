import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import ffmpeg

def rip_audio(source_path, output_audio_path, audio_codec='aac', bitrate='192k'):
    try:
        (
            ffmpeg
            .input(source_path)
            .output(output_audio_path, **{'q:a': 0, 'map': 'a', 'acodec': audio_codec, 'b:a': bitrate})
            .run(overwrite_output=True)
        )
        messagebox.showinfo("Success", f"Audio ripped to:\n{output_audio_path}")
    except ffmpeg.Error as e:
        error_output = e.stderr.decode(errors='ignore') if e.stderr else str(e)
        messagebox.showerror("Error", error_output)

def attach_audio(video_path, audio_path, output_path):
    try:
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(audio_path)

        (
            ffmpeg
            .output(input_video, input_audio, output_path, **{
                'c:v': 'copy',
                'c:a': 'aac',
                'map': '0:v:0',
                'map': '1:a:0'
            })
            .run(overwrite_output=True)
        )
        messagebox.showinfo("Success", f"Audio attached to video:\n{output_path}")
    except ffmpeg.Error as e:
        error_output = e.stderr.decode(errors='ignore') if e.stderr else str(e)
        messagebox.showerror("Error", error_output)


def launch_audio_gui():
    root = tk.Toplevel()
    root.title("🎵 Audio Ripper & Attacher")
    root.geometry("500x400")

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
    ttk.Combobox(root, textvariable=codec, values=["aac", "mp3", "wav"]).pack()

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
