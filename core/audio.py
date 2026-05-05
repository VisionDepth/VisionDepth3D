from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess, json, threading, shutil
from pathlib import Path
import re

VIDEO_EXT = (".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm")
AUDIO_EXT = (".aac", ".m4a", ".mp3", ".wav", ".flac", ".opus", ".ac3", ".eac3", ".dts")

# ---------- ffprobe helpers ----------

def _which_ffmpeg():
    p = shutil.which("ffmpeg")
    if not p:
        messagebox.showerror("FFmpeg not found", "FFmpeg is not installed or not in PATH.")
        raise FileNotFoundError("ffmpeg not found")
    return p  # return the actual resolved path

def ffprobe_has_audio(path: str) -> bool:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "json", path],
            stderr=subprocess.STDOUT
        )
        js = json.loads(out.decode("utf-8", "ignore"))
        return bool(js.get("streams"))
    except Exception:
        return False

def ffprobe_duration(path: str) -> float | None:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            stderr=subprocess.STDOUT
        )
        js = json.loads(out.decode("utf-8", "ignore"))
        d = js.get("format", {}).get("duration")
        return float(d) if d else None
    except Exception:
        return None

def ffprobe_audio_codec(path: str) -> str | None:
    """Return codec_name for a:0 or None."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "json", path],
            stderr=subprocess.STDOUT
        )
        js = json.loads(out.decode("utf-8", "ignore"))
        streams = js.get("streams", [])
        if streams:
            return streams[0].get("codec_name")
    except Exception:
        pass
    return None

# ---------- async runner ----------

def run_ffmpeg_async(cmd: list[str], title="Working…", expect_secs=None, on_done=None):
    win = tk.Toplevel()
    win.title(title)
    win.resizable(False, False)

    ttk.Label(win, text=title).pack(padx=12, pady=(12, 6))
    pb = ttk.Progressbar(win, length=380, mode="indeterminate")
    pb.pack(padx=12, pady=(0, 8))
    pb.start(10)
    outbox = tk.Text(win, height=6, width=72)
    outbox.pack(padx=12, pady=(0, 8))
    outbox.config(state="disabled")

    def _append(line):
        outbox.config(state="normal")
        outbox.insert("end", line)
        outbox.see("end")
        outbox.config(state="disabled")

    def worker():
        rc = -1
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                win.after(0, _append, line)
            rc = proc.wait()
        except Exception as e:
            win.after(0, _append, f"\nERROR: {e}\n")
        finally:
            def done():
                try:
                    pb.stop()
                    win.destroy()
                except:
                    pass
                if on_done:
                    on_done(rc == 0)
            win.after(0, done)

    threading.Thread(target=worker, daemon=True).start()

# ---------- matching / concat ----------

def _last_digits(s: str) -> str | None:
    m = re.search(r'(\d+)(?!.*\d)', s)
    return m.group(1) if m else None

def best_audio_match_for_video(video_stem: str, audio_files: list[Path]) -> Path | None:
    exact = [a for a in audio_files if a.stem == video_stem]
    if exact:
        return exact[0]
    pref = [a for a in audio_files if a.stem.startswith(video_stem + "_")]
    if pref:
        pref.sort(key=lambda p: len(p.stem))
        return pref[0]
    vnum = _last_digits(video_stem)
    if vnum is None:
        return None
    vn = int(vnum.lstrip("0") or "0")
    cands = []
    for a in audio_files:
        anum = _last_digits(a.stem)
        if anum is None:
            continue
        if int(anum.lstrip("0") or "0") == vn:
            cands.append((abs(len(a.stem) - len(video_stem)), a))
    if cands:
        cands.sort(key=lambda t: t[0])
        return cands[0][1]
    return None

def ffprobe_video_params(path: str) -> tuple[int,int,str]:
    """(w, h, fps_expr) from r_frame_rate; fps_expr keeps '24000/1001' when present."""
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=width,height,r_frame_rate","-of","json", path],
            stderr=subprocess.STDOUT
        )
        js = json.loads(out.decode("utf-8","ignore"))
        st = js["streams"][0]
        w = int(st.get("width", 0)); h = int(st.get("height", 0))
        rfr = st.get("r_frame_rate", "0/1") or "0/1"
        # ffmpeg accepts fps=fps=NUM/DEN directly
        return w, h, rfr
    except Exception:
        return 0, 0, "0/1"

def _normalize_preset(vcodec: str, preset: str) -> str:
    """Map NVENC-style p1..p7 to x264/x265 when needed, and vice-versa."""
    preset = (preset or "").lower()
    if "nvenc" in vcodec:
        if preset in ("slow","medium","fast"):
            return {"slow":"p7","medium":"p5","fast":"p3"}[preset]
        return preset or "p5"
    # software encoders
    if preset.startswith("p") and preset[1:].isdigit():   # p1..p7 picked with x264
        return "medium"
    return preset or "medium"

def build_gapless_filter(inputs_have_audio: list[bool], w:int, h:int, fps_expr:str) -> str:
    """
    Make all inputs uniform: fps/size/pix_fmt + gapless audio. Works across AV1/H.264/etc.
    """
    n = len(inputs_have_audio)
    parts = []
    vlabels, alabels = [], []

    # If we couldn't probe, don't force size/fps (let ffmpeg infer)
    vfix = []
    if w > 0 and h > 0:
        vfix.append(f"scale={w}:{h}:flags=bicubic")
    if fps_expr and fps_expr != "0/1":
        vfix.append(f"fps=fps={fps_expr}")
    vfix.append("format=yuv420p")            # normalize pix fmt for concat
    vfix.append("settb=AVTB,setpts=PTS-STARTPTS")

    for i in range(n):
        parts.append(f"[{i}:v:0]{','.join(vfix)}[v{i}]")
        vlabels.append(f"[v{i}]")
        if inputs_have_audio[i]:
            parts.append(f"[{i}:a:0]aresample=48000,asetpts=PTS-STARTPTS[a{i}]")
            alabels.append(f"[a{i}]")

    if len(alabels) == n and n > 0:
        parts.append("".join(vlabels + alabels) + f"concat=n={n}:v=1:a=1[v][a]")
    else:
        parts.append("".join(vlabels) + f"concat=n={n}:v=1:a=0[v]")
    return ";".join(parts)

def _stitch(self, files: list[Path]):
    files = [Path(p) for p in files if Path(p).exists()]
    if not files:
        messagebox.showerror("Stitch", "No per-clip outputs to stitch.")
        return

    have_audio = [ffprobe_has_audio(str(p)) for p in files]
    # Use the first clip as the canonical size/fps
    w, h, fps_expr = ffprobe_video_params(str(files[0]))
    filter_graph = build_gapless_filter(have_audio, w, h, fps_expr)

    cmd = [_which_ffmpeg(), "-y"]
    for p in files:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", filter_graph, "-map", "[v]"]
    if all(have_audio):
        cmd += ["-map", "[a]"]

    fv = self.final_vcodec.get()
    cq = self.final_crf.get().strip() or "18"
    preset = _normalize_preset(fv, self.final_preset.get())
    fa = self.final_acodec.get()
    fbr = self.final_abitrate.get().strip() or "192k"

    if fv == "auto":
        fv = "libx264"
    if "nvenc" in fv:
        cmd += ["-c:v", fv, "-cq", cq, "-preset", preset]
    else:
        cmd += ["-c:v", fv, "-crf", cq, "-preset", preset]

    if all(have_audio):
        cmd += ["-c:a", fa, "-b:a", fbr, "-ar", "48000"]

    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", self.stitched_out.get()]

    total = sum((ffprobe_duration(str(p)) or 0.0) for p in files)
    run_ffmpeg_async(
        cmd, title="Stitching final video…", expect_secs=total,
        on_done=lambda ok: messagebox.showinfo(
            "Stitch", "Final render complete." if ok else "Stitch failed. Check inputs/codecs.")
    )


# ---------- GUI ----------

class AudioUI:
    def __init__(self, parent=None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Audio Ripper & Attacher")
        self.root.geometry("950x560")

        # model
        self.mode         = tk.StringVar(self.root, value="rip")
        self.audio_mode   = tk.StringVar(self.root, value="folder")
        self.audio_folder = tk.StringVar(self.root)
        self.single_audio = tk.StringVar(self.root)
        self.offset       = tk.DoubleVar(self.root, value=0.0)

        self.force_clip  = tk.BooleanVar(self.root, value=False)
        self.vcodec_clip = tk.StringVar(self.root, value="copy")
        self.acodec_clip = tk.StringVar(self.root, value="copy")

        self.rip_codec   = tk.StringVar(self.root, value="copy")
        self.rip_bitrate = tk.StringVar(self.root, value="192")

        self.final_vcodec  = tk.StringVar(self.root, value="auto")
        self.final_crf     = tk.StringVar(self.root, value="18")
        self.final_preset  = tk.StringVar(self.root, value="slow")
        self.final_acodec  = tk.StringVar(self.root, value="aac")
        self.final_abitrate= tk.StringVar(self.root, value="192k")

        self.delivery     = tk.StringVar(self.root, value="both")
        self.perclip_out  = tk.StringVar(self.root)
        self.stitched_out = tk.StringVar(self.root)
        self.rip_out      = tk.StringVar(self.root)

        self._build_ui()
        self._toggle_single_row()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        # Step 1
        src = ttk.LabelFrame(main, text="Step 1 — Sources")
        src.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        src.columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(src, height=10, selectmode=tk.EXTENDED)
        sb = ttk.Scrollbar(src, orient="vertical", command=self.listbox.yview)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        src.rowconfigure(0, weight=1)
        row = ttk.Frame(src); row.grid(row=1, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Button(row, text="Add Files", command=self.add_files).pack(side="left", padx=4)
        ttk.Button(row, text="Add Folder", command=self.add_folder).pack(side="left", padx=4)
        ttk.Button(row, text="Up", command=lambda: self._move(-1)).pack(side="left", padx=4)
        ttk.Button(row, text="Down", command=lambda: self._move(+1)).pack(side="left", padx=4)
        ttk.Button(row, text="Remove", command=self.remove_sel).pack(side="left", padx=4)
        ttk.Button(row, text="Clear", command=self.clear_all).pack(side="left", padx=4)

        # Step 2
        opts = ttk.LabelFrame(main, text="Step 2 — Operation & Options")
        opts.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for i in range(6):
            opts.columnconfigure(i, weight=1)
        ttk.Label(opts, text="Operation").grid(row=0, column=0, sticky="e")
        m = ttk.Combobox(opts, textvariable=self.mode, state="readonly",
                         values=["rip", "attach", "attach_stitch"], width=16)
        m.grid(row=0, column=1, sticky="w")

        row_at = ttk.Frame(opts); row_at.grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 2))
        ttk.Label(row_at, text="Audio source").pack(side="left")
        ttk.Radiobutton(row_at, text="Auto-match from folder", value="folder",
                        variable=self.audio_mode, command=self._toggle_single_row).pack(side="left", padx=8)
        ttk.Radiobutton(row_at, text="Single audio for all", value="single",
                        variable=self.audio_mode, command=self._toggle_single_row).pack(side="left", padx=8)

        row_af = ttk.Frame(opts); row_af.grid(row=2, column=0, columnspan=6, sticky="ew")
        ttk.Label(row_af, text="Audio folder").pack(side="left")
        self.ent_audio_folder = ttk.Entry(row_af, textvariable=self.audio_folder, width=58)
        self.ent_audio_folder.pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row_af, text="Browse", command=self.pick_audio_folder).pack(side="left", padx=2)

        row_sf = ttk.Frame(opts); row_sf.grid(row=3, column=0, columnspan=6, sticky="ew")
        ttk.Label(row_sf, text="Single audio").pack(side="left")
        self.ent_single_audio = ttk.Entry(row_sf, textvariable=self.single_audio, width=58)
        self.ent_single_audio.pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row_sf, text="Browse", command=self.pick_single_audio).pack(side="left", padx=2)
        self._row_single = row_sf
        self._row_folder = row_af

        ttk.Label(opts, text="Audio offset (sec)").grid(row=4, column=0, sticky="e")
        ttk.Scale(opts, variable=self.offset, from_=-10.0, to=10.0,
                  orient=tk.HORIZONTAL, length=280).grid(row=4, column=1, sticky="w", padx=6)

        row_clip = ttk.Frame(opts); row_clip.grid(row=5, column=0, columnspan=6, sticky="w", pady=(6, 2))
        ttk.Checkbutton(row_clip, text="Force re-encode (per-clip attach)",
                        variable=self.force_clip).pack(side="left")
        ttk.Label(row_clip, text="vcodec").pack(side="left", padx=(12, 4))
        ttk.Combobox(row_clip, textvariable=self.vcodec_clip, state="readonly",
                     values=["copy", "libx264", "libx265", "h264_nvenc", "hevc_nvenc"], width=12).pack(side="left")
        ttk.Label(row_clip, text="acodec").pack(side="left", padx=(12, 4))
        ttk.Combobox(row_clip, textvariable=self.acodec_clip, state="readonly",
                     values=["copy", "aac", "mp3", "opus", "flac", "ac3", "eac3"], width=10).pack(side="left")

        row_rip = ttk.Frame(opts); row_rip.grid(row=6, column=0, columnspan=6, sticky="w", pady=(6, 2))
        ttk.Label(row_rip, text="Rip codec").pack(side="left")
        ttk.Combobox(row_rip, textvariable=self.rip_codec, state="readonly",
                     values=["copy", "aac", "mp3", "opus", "flac", "wav", "ac3", "eac3"], width=10).pack(side="left", padx=(6, 0))
        ttk.Label(row_rip, text="bitrate (kbps)").pack(side="left", padx=(12, 4))
        ttk.Entry(row_rip, textvariable=self.rip_bitrate, width=6).pack(side="left")

        row_final = ttk.Frame(opts); row_final.grid(row=7, column=0, columnspan=6, sticky="w", pady=(6, 2))
        ttk.Label(row_final, text="Final vcodec").pack(side="left")
        ttk.Combobox(row_final, textvariable=self.final_vcodec, state="readonly",
                     values=["auto", "libx264", "libx265", "h264_nvenc", "hevc_nvenc"], width=12).pack(side="left", padx=(6, 0))
        ttk.Label(row_final, text="CRF/CQ").pack(side="left", padx=(12, 4))
        ttk.Entry(row_final, textvariable=self.final_crf, width=4).pack(side="left")
        ttk.Label(row_final, text="preset").pack(side="left", padx=(12, 4))
        ttk.Combobox(row_final, textvariable=self.final_preset, state="readonly",
                     values=["slow", "medium", "fast", "p1", "p2", "p3", "p4", "p5", "p6", "p7"], width=8).pack(side="left")
        ttk.Label(row_final, text="acodec").pack(side="left", padx=(12, 4))
        ttk.Combobox(row_final, textvariable=self.final_acodec, state="readonly",
                     values=["aac", "opus", "mp3", "flac", "ac3", "eac3"], width=8).pack(side="left")
        ttk.Label(row_final, text="bitrate").pack(side="left", padx=(12, 4))
        ttk.Entry(row_final, textvariable=self.final_abitrate, width=8).pack(side="left")

        # Step 3
        out = ttk.LabelFrame(main, text="Step 3 — Output & Actions")
        out.grid(row=2, column=0, sticky="ew")
        for i in range(6):
            out.columnconfigure(i, weight=1)

        row_pc = ttk.Frame(out); row_pc.grid(row=0, column=0, columnspan=6, sticky="ew")
        ttk.Label(row_pc, text="Per-clip output folder").pack(side="left")
        self.ent_perclip = ttk.Entry(row_pc, textvariable=self.perclip_out, width=70)
        self.ent_perclip.pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row_pc, text="Browse", command=self.pick_perclip_out).pack(side="left")

        row_st = ttk.Frame(out); row_st.grid(row=1, column=0, columnspan=6, sticky="ew")
        ttk.Label(row_st, text="Final stitched output").pack(side="left")
        self.ent_stitched = ttk.Entry(row_st, textvariable=self.stitched_out, width=70)
        self.ent_stitched.pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row_st, text="Save As", command=self.pick_stitched_out).pack(side="left")

        row_rp = ttk.Frame(out); row_rp.grid(row=2, column=0, columnspan=6, sticky="ew")
        ttk.Label(row_rp, text="Rip output folder").pack(side="left")
        self.ent_ripout = ttk.Entry(row_rp, textvariable=self.rip_out, width=70)
        self.ent_ripout.pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row_rp, text="Browse", command=self.pick_rip_out).pack(side="left")

        row_btn = ttk.Frame(out); row_btn.grid(row=3, column=0, columnspan=6, sticky="w", pady=(6, 0))
        ttk.Button(row_btn, text="Preview", command=self.preview).pack(side="left", padx=4)
        ttk.Button(row_btn, text="Run", command=self.run).pack(side="left", padx=4)

    # ---------- list actions ----------

    def add_files(self):
        sel = filedialog.askopenfilenames(
            parent=self.root, title="Add video files",
            filetypes=[("Video files", [f"*{e}" for e in VIDEO_EXT]), ("All files", "*.*")]
        )
        files = self.root.tk.splitlist(sel)
        existing = set(self.listbox.get(0, tk.END))
        for p in files:
            if p and p not in existing:
                self.listbox.insert(tk.END, p)

    def add_folder(self):
        d = filedialog.askdirectory(parent=self.root, title="Add all videos from folder")
        if not d:
            return
        paths = sorted([str(p) for p in Path(d).glob("*") if p.suffix.lower() in VIDEO_EXT])
        existing = set(self.listbox.get(0, tk.END))
        for p in paths:
            if p not in existing:
                self.listbox.insert(tk.END, p)

    def _move(self, delta):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        for idx in (sel if delta < 0 else reversed(sel)):
            new = idx + delta
            if new < 0 or new >= self.listbox.size():
                continue
            txt = self.listbox.get(idx)
            self.listbox.delete(idx)
            self.listbox.insert(new, txt)
            self.listbox.selection_set(new)

    def remove_sel(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)

    def clear_all(self):
        self.listbox.delete(0, tk.END)

    # ---------- pickers ----------

    def pick_audio_folder(self):
        d = filedialog.askdirectory(parent=self.root, title="Choose audio folder")
        if d:
            self.audio_folder.set(d)
            self.ent_audio_folder.delete(0, tk.END)
            self.ent_audio_folder.insert(0, d)

    def pick_single_audio(self):
        f = filedialog.askopenfilename(
            parent=self.root, title="Choose audio file",
            filetypes=[("Audio files", [f"*{e}" for e in AUDIO_EXT]), ("All files", "*.*")]
        )
        if f:
            self.single_audio.set(f)
            self.ent_single_audio.delete(0, tk.END)
            self.ent_single_audio.insert(0, f)

    def pick_perclip_out(self):
        d = filedialog.askdirectory(parent=self.root, title="Choose per-clip output folder")
        if d:
            self.perclip_out.set(d)
            self.ent_perclip.delete(0, tk.END)
            self.ent_perclip.insert(0, d)

    def pick_stitched_out(self):
        f = filedialog.asksaveasfilename(
            parent=self.root, title="Save final stitched video",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("MKV", "*.mkv"), ("All files", "*.*")]
        )
        if f:
            self.stitched_out.set(f)
            self.ent_stitched.delete(0, tk.END)
            self.ent_stitched.insert(0, f)

    def pick_rip_out(self):
        d = filedialog.askdirectory(parent=self.root, title="Choose rip output folder")
        if d:
            self.rip_out.set(d)
            self.ent_ripout.delete(0, tk.END)
            self.ent_ripout.insert(0, d)

    def _toggle_single_row(self):
        single = (self.audio_mode.get() == "single")
        state_single = ("normal" if single else "disabled")
        state_folder = ("disabled" if single else "normal")
        for w in self._row_single.winfo_children():
            try:
                w.configure(state=state_single)
            except:
                pass
        for w in self._row_folder.winfo_children():
            try:
                w.configure(state=state_folder)
            except:
                pass

    # ---------- preview / run ----------

    def preview(self):
        vids = [self.listbox.get(i) for i in range(self.listbox.size())]
        if not vids:
            messagebox.showerror("Preview", "Add videos in Step 1.")
            return
        m = self.mode.get()
        lines = [f"Mode: {m}"]
        if m in ("attach", "attach_stitch"):
            if self.audio_mode.get() == "single":
                lines.append(f"Single audio: {self.single_audio.get() or '(not set)'}")
            else:
                af = self.audio_folder.get().strip()
                lines.append(f"Audio folder: {af or '(not set)'}")
        if m == "rip":
            lines.append(f"Rip -> {self.rip_out.get() or '(not set)'}")
        else:
            lines.append(f"Per-clip -> {self.perclip_out.get() or '(disabled)'}")
            lines.append(f"Final    -> {self.stitched_out.get() or '(disabled)'}")

        if m in ("attach", "attach_stitch"):
            audios = []
            if self.audio_mode.get() == "folder":
                af = self.audio_folder.get().strip()
                if af:
                    aroot = Path(af)
                    if aroot.exists():
                        audios = [p for p in aroot.glob("*") if p.suffix.lower() in AUDIO_EXT]
            lines.append("\nPairs:")
            for v in vids:
                vstem = Path(v).stem
                if self.audio_mode.get() == "single":
                    a = self.single_audio.get() or "(MISSING)"
                else:
                    mpath = best_audio_match_for_video(vstem, audios)
                    a = str(mpath) if mpath else "(NO MATCH)"
                lines.append(f"  {Path(v).name}  <--  {Path(a).name if a not in ('(MISSING)', '(NO MATCH)') else a}")
        messagebox.showinfo("Preview", "\n".join(lines))

    def run(self):
        try:
            _which_ffmpeg()
        except FileNotFoundError:
            return

        vids = [self.listbox.get(i) for i in range(self.listbox.size())]
        if not vids:
            messagebox.showerror("Run", "Add videos in Step 1.")
            return
        mode = self.mode.get()

        if mode == "rip":
            outdir_s = self.rip_out.get().strip()
            if not outdir_s:
                messagebox.showerror("Run", "Choose a Rip output folder.")
                return
            outdir = Path(outdir_s)
            outdir.mkdir(parents=True, exist_ok=True)
            self._run_rip(vids, outdir)
            return

        # attach / attach+stitch
        if self.audio_mode.get() == "single":
            a = self.single_audio.get().strip()
            if not a or not Path(a).exists():
                messagebox.showerror("Run", "Choose a valid Single audio file.")
                return
        else:
            af = self.audio_folder.get().strip()
            if not af:
                messagebox.showerror("Run", "Choose an Audio folder to auto-match.")
                return

        pcout_s = self.perclip_out.get().strip()
        if not pcout_s:
            messagebox.showerror("Run", "Choose a Per-clip output folder.")
            return
        Path(pcout_s).mkdir(parents=True, exist_ok=True)

        if mode == "attach_stitch":
            if not self.stitched_out.get().strip():
                messagebox.showerror("Run", "Choose the Final stitched output file.")
                return

        self._run_attach_and_maybe_stitch(vids, mode)

    # ---------- ops ----------

    def _run_rip(self, vids: list[str], outdir: Path):
        codec = self.rip_codec.get().strip().lower()
        br = self.rip_bitrate.get().strip()
        ff = _which_ffmpeg()
        cmds = []

        for v in vids:
            if not ffprobe_has_audio(v):
                messagebox.showwarning("Rip", f"{Path(v).name} has no audio. Skipping.")
                continue

            stem = Path(v).stem
            fmt_args: list[str] = []
            out_ext = ".m4a"

            if codec == "copy":
                src_c = (ffprobe_audio_codec(v) or "").lower()
                # map codec to extension; default to .mka when unknown
                ext_map = {
                    "aac": (".m4a", None),
                    "alac": (".m4a", None),
                    "mp3": (".mp3", None),
                    "opus": (".opus", None),
                    "flac": (".flac", None),
                    "ac3": (".ac3", None),
                    "eac3": (".eac3", None),
                    "dca": (".dts", None),        # DTS
                    "dts": (".dts", None),
                    "pcm_s16le": (".wav", None),
                    "pcm_s24le": (".wav", None),
                    "pcm_s32le": (".wav", None),
                }
                out_ext, force_fmt = ext_map.get(src_c, (".mka", "matroska"))
                if force_fmt:
                    fmt_args = ["-f", force_fmt]
                out = outdir / f"{stem}_{src_c or 'copy'}{out_ext}"
                cmd = [ff, "-y", "-i", v, "-vn", "-map", "0:a:0", "-c:a", "copy"] + fmt_args + [str(out)]
            else:
                # choose container by chosen codec
                if codec == "aac":
                    out_ext = ".m4a"
                elif codec == "mp3":
                    out_ext = ".mp3"
                elif codec == "opus":
                    out_ext = ".opus"
                elif codec == "flac":
                    out_ext = ".flac"
                elif codec == "wav":
                    out_ext = ".wav"
                elif codec == "ac3":
                    out_ext = ".ac3"
                elif codec == "eac3":
                    out_ext = ".eac3"
                else:
                    out_ext = ".mka"
                out = outdir / f"{stem}_{codec}{out_ext}"
                cmd = [ff, "-y", "-i", v, "-vn", "-map", "0:a:0", "-c:a", codec]
                if br:
                    try:
                        cmd += ["-b:a", f"{int(br)}k"]
                    except ValueError:
                        pass
                cmd += [str(out)]
            cmds.append(cmd)

        if not cmds:
            return

        def step(i=0):
            if i >= len(cmds):
                messagebox.showinfo("Rip", f"Extracted {len(cmds)} file(s).")
                return
            vname = Path(vids[i]).name
            run_ffmpeg_async(cmds[i], title=f"Ripping {vname}", on_done=lambda ok, j=i: step(j + 1))

        step(0)

    def _run_attach_and_maybe_stitch(self, vids: list[str], mode: str):
        a_mode = self.audio_mode.get()
        offset = float(self.offset.get() or 0.0)
        force = self.force_clip.get()
        vcodec = self.vcodec_clip.get()
        acodec = self.acodec_clip.get()
        ff = _which_ffmpeg()

        audio_files = []
        single_audio = None
        if a_mode == "folder":
            aroot = Path(self.audio_folder.get().strip())
            if aroot.exists():
                audio_files = [p for p in aroot.glob("*") if p.suffix.lower() in AUDIO_EXT]
        else:
            single_audio = self.single_audio.get().strip()

        perclip_outputs: list[Path] = []

        def attach_one(i=0):
            if i >= len(vids):
                if mode == "attach_stitch":
                    _stitch(self, perclip_outputs)
                else:
                    messagebox.showinfo("Attach", f"Done. Muxed {len(perclip_outputs)} file(s).")
                return

            v = vids[i]
            vstem = Path(v).stem
            if single_audio:
                a = single_audio
            else:
                match = best_audio_match_for_video(vstem, audio_files)
                if not match:
                    messagebox.showwarning("Attach", f"No audio match for {Path(v).name}; skipping.")
                    return attach_one(i + 1)
                a = str(match)

            out_dir = Path(self.perclip_out.get().strip())
            out = out_dir / f"{Path(v).stem}{Path(v).suffix}"
            perclip_outputs.append(out)

            cmd = [ff, "-y", "-i", v]
            if abs(offset) > 1e-6:
                # apply offset to the audio input only
                cmd += ["-itsoffset", str(offset)]
            cmd += ["-i", a, "-map", "0:v:0", "-map", "1:a:0"]
            if force:
                cmd += ["-c:v", vcodec, "-c:a", acodec]
            else:
                cmd += ["-c:v", "copy", "-c:a", "copy"]
            cmd += ["-shortest", "-movflags", "+faststart", str(out)]

            run_ffmpeg_async(cmd, title=f"Attaching -> {Path(v).name}",
                             on_done=lambda ok, j=i: attach_one(j + 1))

        attach_one(0)


# public
def launch_audio_gui(parent=None):
    ui = AudioUI(parent)
    try:
        ui.root.transient(parent)
        ui.root.grab_set()
    except:
        pass
    if parent is None:
        ui.root.mainloop()
