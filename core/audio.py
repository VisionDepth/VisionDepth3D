import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess, threading, json, shutil, time

# ---------- FFprobe / progress helpers ----------

def ffprobe_duration(path):
    """Return duration in seconds (float) or None."""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", path
        ], stderr=subprocess.STDOUT)
        js = json.loads(out.decode("utf-8", errors="ignore"))
        dur = js.get("format", {}).get("duration")
        return float(dur) if dur else None
    except Exception:
        return None

def run_ffmpeg_async(cmd, title="Working…", expected_secs=None, on_done=None):
    """Run FFmpeg in a thread with a tiny progress window."""
    win = tk.Toplevel()
    win.title(title)
    win.resizable(False, False)
    ttk.Label(win, text=title).pack(padx=12, pady=(12, 6))
    pb = ttk.Progressbar(win, length=340)
    pb.pack(padx=12, pady=(0, 6))
    status = ttk.Label(win, text="")
    status.pack(padx=12, pady=(0, 12))

    show_percent = any(arg == "-progress" for arg in cmd)
    if expected_secs and show_percent:
        pb.config(mode="determinate", maximum=100, value=0)
    else:
        pb.config(mode="indeterminate")
        pb.start(10)

    def worker():
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=True
            )
            last_time = 0.0
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if not line:
                    time.sleep(0.02); continue

                if show_percent and expected_secs:
                    if "out_time_ms=" in line:
                        try:
                            ms = float(line.strip().split("out_time_ms=")[-1])
                            last_time = ms / 1_000_000.0
                        except:
                            pass
                    elif "out_time=" in line:
                        try:
                            ts = line.strip().split("out_time=")[-1]
                            h, m, s = ts.split(":")
                            last_time = float(h)*3600 + float(m)*60 + float(s)
                        except:
                            pass

                    if expected_secs > 0:
                        pct = max(0, min(100, (last_time / expected_secs) * 100.0))
                        pb.after(0, lambda p=pct: pb.config(value=p))
                        status.after(0, lambda t=last_time: status.config(
                            text=f"{t:0.1f}s / {expected_secs:0.1f}s"
                        ))
            rc = proc.wait()
        except Exception:
            rc = -1
        finally:
            def close_and_callback():
                try: win.destroy()
                except: pass
                if on_done: on_done(rc == 0)
            win.after(0, close_and_callback)

    threading.Thread(target=worker, daemon=True).start()

def _ffmpeg():
    path = shutil.which("ffmpeg")
    if not path:
        messagebox.showerror("FFmpeg Not Found",
            "FFmpeg is not installed or not in PATH.")
        raise FileNotFoundError("ffmpeg not found")
    return "ffmpeg"

# ---------- Core ops ----------

def rip_audio(source_path, out_audio_path, codec_choice="copy", bitrate_kbps=None):
    """
    Extract audio. Default is stream copy (fast).
    codec_choice: 'copy', 'aac', 'mp3', 'opus', 'flac', etc.
    bitrate_kbps: int or None (ignored for 'copy').
    """
    if not source_path or not out_audio_path:
        messagebox.showerror("Audio Rip", "Select a source and output path.")
        return

    if codec_choice == "copy":
        cmd = [
            _ffmpeg(), "-y",
            "-i", source_path,
            "-vn",
            "-map", "0:a:0",
            "-c:a", "copy",
            "-progress", "pipe:1",
            out_audio_path
        ]
    else:
        a_opts = ["-c:a", codec_choice]
        if bitrate_kbps:
            a_opts += ["-b:a", f"{int(bitrate_kbps)}k"]
        cmd = [
            _ffmpeg(), "-y",
            "-i", source_path,
            "-vn",
            "-map", "0:a:0",
            *a_opts,
            "-progress", "pipe:1",
            out_audio_path
        ]

    dur = ffprobe_duration(source_path)  # container duration is fine here
    run_ffmpeg_async(
        cmd, title="Extracting audio…", expected_secs=dur,
        on_done=lambda ok: messagebox.showinfo("Audio Rip", "Done!" if ok else "Failed.")
    )

def attach_audio(video_path, audio_path, out_path,
                 offset_sec=0.0, force_reencode=False,
                 vcodec="copy", acodec="copy"):
    """
    Mux external audio onto a video.
    offset_sec: positive delays audio; negative advances it.
    Default is stream copy for both tracks (fast).
    """
    if not video_path or not audio_path or not out_path:
        messagebox.showerror("Attach Audio", "Provide video, audio, and output paths.")
        return

    # Inputs (with optional offset applied to audio input)
    cmd = [_ffmpeg(), "-y", "-i", video_path]
    if abs(offset_sec) > 1e-6:
        cmd += ["-itsoffset", str(offset_sec)]
    cmd += ["-i", audio_path]

    vopt = ["-c:v", vcodec if force_reencode else "copy"]
    aopt = ["-c:a", acodec if force_reencode else "copy"]

    cmd += [
        "-map", "0:v:0",
        "-map", "1:a:0",
        *vopt, *aopt,
        "-shortest",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        out_path
    ]

    dur = ffprobe_duration(video_path)
    mode = "Re-encode" if (force_reencode or vcodec != "copy" or acodec != "copy") else "Stream copy"

    run_ffmpeg_async(
        cmd, title=f"Muxing audio… ({mode})", expected_secs=dur,
        on_done=lambda ok: messagebox.showinfo("Attach Audio", "Done!" if ok else "Failed.")
    )

# ---------- GUI ----------

def launch_audio_gui(parent=None):
    root = tk.Toplevel(parent) if parent else tk.Tk()
    root.title("Audio Ripper & Attacher v3")
    root.geometry("560x560")

    # ✅ bind variables to THIS window
    source_video     = tk.StringVar(master=root)
    ripped_audio_out = tk.StringVar(master=root)
    attach_video_in  = tk.StringVar(master=root)
    attach_audio_in  = tk.StringVar(master=root)
    output_video_out = tk.StringVar(master=root)
    offset           = tk.DoubleVar(master=root, value=0.0)

    rip_codec_ui     = tk.StringVar(master=root, value="Copy (no re-encode)")
    rip_bitrate_ui   = tk.StringVar(master=root, value="192")
    attach_force_reencode = tk.BooleanVar(master=root, value=False)

    # File pickers
    def pick_video(var):
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov")])
        if p: var.set(p)

    def pick_audio(var):
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.aac *.m4a *.mp3 *.wav *.flac *.opus *.ac3 *.eac3")])
        if p: var.set(p)

    def save_audio(var):
        p = filedialog.asksaveasfilename(defaultextension=".aac",
            filetypes=[("AAC", "*.aac"), ("M4A", "*.m4a"), ("MP3", "*.mp3"),
                       ("WAV", "*.wav"), ("FLAC", "*.flac"), ("Opus", "*.opus")])
        if p: var.set(p)

    def save_video(var):
        p = filedialog.asksaveasfilename(defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("MKV", "*.mkv"), ("MOV", "*.mov")])
        if p: var.set(p)

    # Layout
    pad = dict(padx=10, pady=4, sticky="ew")
    root.columnconfigure(1, weight=1)

    # --- Ripping section ---
    ttk.Label(root, text="Source Video (rip from):").grid(row=0, column=0, **pad)
    ttk.Entry(root, textvariable=source_video).grid(row=0, column=1, **pad)
    ttk.Button(root, text="Browse", command=lambda: pick_video(source_video)).grid(row=0, column=2, padx=6, pady=4)

    ttk.Label(root, text="Save Ripped Audio As:").grid(row=1, column=0, **pad)
    ttk.Entry(root, textvariable=ripped_audio_out).grid(row=1, column=1, **pad)
    ttk.Button(root, text="Save As", command=lambda: save_audio(ripped_audio_out)).grid(row=1, column=2, padx=6, pady=4)

    ttk.Label(root, text="Codec").grid(row=2, column=0, **pad)
    rip_codec_menu = ttk.Combobox(root, textvariable=rip_codec_ui, state="readonly", values=[
        "Copy (no re-encode)", "aac", "mp3", "opus", "flac", "wav", "ac3", "eac3"
    ])
    rip_codec_menu.grid(row=2, column=1, **pad)

    ttk.Label(root, text="Bitrate (kbps)").grid(row=3, column=0, **pad)
    rip_bitrate_entry = ttk.Entry(root, textvariable=rip_bitrate_ui, width=8)
    rip_bitrate_entry.grid(row=3, column=1, sticky="w", padx=10, pady=4)

    def on_codec_change(*_):
        copying = rip_codec_ui.get().lower().startswith("copy")
        rip_bitrate_entry.config(state=("disabled" if copying else "normal"))
    rip_codec_menu.bind("<<ComboboxSelected>>", on_codec_change)
    on_codec_change()

    ttk.Button(root, text="Rip Audio", command=lambda: (
        rip_audio(
            source_video.get(),
            ripped_audio_out.get(),
            "copy" if rip_codec_ui.get().lower().startswith("copy") else rip_codec_ui.get().lower(),
            None if rip_codec_ui.get().lower().startswith("copy") else int(rip_bitrate_ui.get() or "192")
        ))
    ).grid(row=4, column=1, pady=8)

    ttk.Separator(root).grid(row=5, column=0, columnspan=3, sticky="ew", pady=8)

    # --- Attaching section ---
    ttk.Label(root, text=" Video to Attach Onto:").grid(row=6, column=0, **pad)
    ttk.Entry(root, textvariable=attach_video_in).grid(row=6, column=1, **pad)
    ttk.Button(root, text="Browse", command=lambda: pick_video(attach_video_in)).grid(row=6, column=2, padx=6, pady=4)

    ttk.Label(root, text=" Audio File to Attach:").grid(row=7, column=0, **pad)
    ttk.Entry(root, textvariable=attach_audio_in).grid(row=7, column=1, **pad)
    ttk.Button(root, text="Browse", command=lambda: pick_audio(attach_audio_in)).grid(row=7, column=2, padx=6, pady=4)

    # --- Offset controls ---
    ttk.Label(root, text="Audio Offset (sec)").grid(row=8, column=0, padx=10, pady=4, sticky="ew")

    def _on_offset_move(v):
        offset_label.config(text=f"{float(v):+.2f} s")
    offset_scale = ttk.Scale(root, variable=offset, from_=-10.0, to=10.0,
                             orient=tk.HORIZONTAL, command=_on_offset_move, length=280)
    offset_scale.grid(row=8, column=1, padx=10, pady=4, sticky="ew")

    offset_label = ttk.Label(root, text=f"{offset.get():+.2f} s", width=8)
    offset_label.grid(row=8, column=2, padx=6, pady=4)

    # small nudge buttons
    def bump(d):
        val = round(offset.get() + d, 3)
        offset.set(val)
        offset_label.config(text=f"{val:+.2f} s")

    btns = ttk.Frame(root)
    btns.grid(row=9, column=1, padx=10, pady=(0,6), sticky="w")
    for txt, d in [("-1.00", -1.0), ("-0.25", -0.25), ("+0.25", +0.25), ("+1.00", +1.0)]:
        ttk.Button(btns, text=txt, width=6, command=lambda d=d: bump(d)).pack(side="left", padx=4)

    offset_box = ttk.Entry(root, width=8)
    offset_box.insert(0, f"{offset.get():.2f}")
    offset_box.grid(row=9, column=2, padx=6, pady=(0,6))
    def _apply_box(_evt=None):
        try:
            val = float(offset_box.get()); offset.set(val)
            offset_label.config(text=f"{val:+.2f} s")
        except:
            pass
    offset_box.bind("<Return>", _apply_box)
    def _sync_box(*_):
        offset_box.delete(0, tk.END); offset_box.insert(0, f"{offset.get():.2f}")
    offset.trace_add("write", lambda *_: _sync_box())

    # --- Output path (moved to its own row to avoid overlap) ---
    ttk.Label(root, text=" Output Video:").grid(row=10, column=0, **pad)   # ← FIXED row
    ttk.Entry(root, textvariable=output_video_out).grid(row=10, column=1, **pad)
    ttk.Button(root, text="Save As", command=lambda: save_video(output_video_out)).grid(row=10, column=2, padx=6, pady=4)

    ttk.Checkbutton(root, text="Force re-encode (avoid unless necessary)",
                    variable=attach_force_reencode).grid(row=11, column=1, sticky="w", padx=10, pady=4)

    ttk.Button(root, text=" Attach Audio", command=lambda: (
        attach_audio(
            attach_video_in.get(),
            attach_audio_in.get(),
            output_video_out.get(),
            offset_sec=float(offset.get() or 0.0),
            force_reencode=attach_force_reencode.get()
        ))
    ).grid(row=12, column=1, pady=10)

    try:
        root.transient(parent)
        root.grab_set()
    except:
        pass
    if parent is None:
        root.mainloop()
