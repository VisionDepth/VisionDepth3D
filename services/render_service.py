import copy
import os
import subprocess
import threading
import time
from pathlib import Path

from core.ffmpeg_utils import require_tool
from models.app_state import AppState
from core.debug_flags import debug_print, is_debug_enabled


from core.render_3d import (
    process_video,
    parse_timecode,
    render_sbs_3d_image,
    merge_audio_from_source,
    FFMPEG_CODEC_MAP,
    hidden_subprocess_kwargs,
)

try:
    from services.keyframe_service import KeyframeService
except ImportError:
    KeyframeService = None

class RenderCancelled(Exception):
    pass
    
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def apply_3d_suffix(base_out: str, output_format: str, eye_mode: str) -> str:
    path = Path(base_out)
    base = str(path.with_suffix(""))
    ext = path.suffix

    fmt = (output_format or "").strip().lower()
    mode = (eye_mode or "").strip().lower()

    format_suffix = ""

    # --- VR180 / packed format naming ---
    if fmt == "vr180 equirect (tb)":
        format_suffix = "_TB_180"
    elif fmt == "vr180 equirect (sbs)":
        format_suffix = "_SBS_180"

    eye_suffix = ""

    if mode == "sbs":
        if fmt == "full-sbs":
            format_suffix = "_LR_Full_SBS"
        elif fmt == "half-sbs":
            format_suffix = "_LR_Half_SBS"
        elif fmt == "vr":
            format_suffix = "_VR"
        elif fmt == "red-cyan anaglyph":
            format_suffix = "_Anaglyph"
        elif fmt == "passive interlaced":
            format_suffix = "_Interlaced"

    elif mode == "left":
        eye_suffix = "_LR_Left"

    elif mode == "right":
        eye_suffix = "_LR_Right"

    elif mode == "both":
        pass

    suffix = f"{format_suffix}{eye_suffix}"

    if not suffix:
        return base_out

    # Avoid doubling the full suffix.
    if base.endswith(suffix):
        return f"{base}{ext}"

    # If user selected a pre-suffixed packed VR180 path, avoid doubling the format suffix.
    if format_suffix and eye_suffix and base.endswith(format_suffix):
        return f"{base}{eye_suffix}{ext}"

    return f"{base}{suffix}{ext}"

class VarAdapter:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class ProgressProxy:
    def __init__(self, callback=None):
        self.value = 0
        self.callback = callback

    def __setitem__(self, key, value):
        if key == "value":
            self.value = value
            if self.callback:
                self.callback(progress=value)

    def __getitem__(self, key):
        if key == "value":
            return self.value
        raise KeyError(key)

    def update(self):
        pass


class ProgressLabelProxy:
    def __init__(self, callback=None):
        self.text = ""
        self.callback = callback

    def config(self, **kwargs):
        if "text" in kwargs:
            self.text = kwargs["text"]
            if self.callback:
                self.callback(status_text=self.text)

    def update(self):
        pass

    def winfo_toplevel(self):
        return self

    def after(self, delay_ms, callback):
        callback()


class RenderService:
    def __init__(self):
        self.progress_callback = None
        self.progress = ProgressProxy(self._emit_progress_update)
        self.progress_label = ProgressLabelProxy(self._emit_progress_update)

        self.suspend_flag = threading.Event()
        self.cancel_flag = threading.Event()

        self.render_start_time = None
        self.last_progress_value = 0

        # Batch progress context
        self._batch_total = None
        self._batch_index = None
        self._batch_start_time = None
        self._batch_rate_label = "FPS"

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def _parse_fps_from_status_text(self, status_text):
        """
        Extracts real render FPS from strings like:
        '12.34% | FPS: 7.21 | Elapsed: 00:01:20 | ETA: 01:40:00'

        Returns float or None.
        """
        if not status_text:
            return None

        text = str(status_text)
        marker = "FPS:"

        if marker not in text:
            return None

        try:
            tail = text.split(marker, 1)[1].strip()
            value_text = tail.split("|", 1)[0].strip()
            value_text = value_text.split()[0].strip()
            return float(value_text)
        except Exception:
            return None

    def _emit_progress_update(self, progress=None, status_text=None):
        if not self.progress_callback:
            return

        now = time.time()
        elapsed = 0.0
        fps_like = 0.0
        eta = None
        rate_label = "FPS"

        if self.render_start_time is not None:
            elapsed = now - self.render_start_time

        current_progress = self.progress.value if progress is None else progress
        callback_progress = float(current_progress)

        if self._batch_total is not None and self._batch_index is not None:
            total = max(1, int(self._batch_total))
            index = max(1, int(self._batch_index))

            inner_progress = max(0.0, min(100.0, float(current_progress))) / 100.0
            completed_items = max(0.0, (index - 1) + inner_progress)

            callback_progress = max(0.0, min(100.0, (completed_items / total) * 100.0))

            if self._batch_start_time is not None:
                elapsed = now - self._batch_start_time

            if elapsed > 0 and completed_items > 0:
                fps_like = completed_items / elapsed
                remaining_items = max(0.0, total - completed_items)
                eta = remaining_items / fps_like if fps_like > 0 else None

            rate_label = self._batch_rate_label or "FPS"

        elif elapsed > 0 and current_progress > 0:
            fps_like = current_progress / elapsed
            remaining = max(0.0, 100.0 - current_progress)
            eta = (remaining / current_progress) * elapsed if current_progress > 0 else None

        final_status_text = status_text if status_text is not None else self.progress_label.text

        # The render backend already formats real frame FPS into status_text.
        # Do not display percent-per-second as FPS.
        parsed_render_fps = self._parse_fps_from_status_text(final_status_text)

        if parsed_render_fps is not None:
            fps_like = parsed_render_fps
            rate_label = "FPS"

        self.progress_callback({
            "progress": float(callback_progress),
            "status_text": final_status_text,
            "elapsed": elapsed,
            "eta": eta,
            "fps_like": fps_like,
            "rate_label": rate_label,
        })

    def request_suspend(self):
        self.suspend_flag.set()

    def request_resume(self):
        self.suspend_flag.clear()

    def request_cancel(self):
        self.cancel_flag.set()

    def reset_flags(self):
        self.suspend_flag.clear()
        self.cancel_flag.clear()
        
    def _load_keyframe_service(self, state: AppState):
        """
        Loads 3D render keyframes from AppState when enabled.
        Returns None when keyframes are disabled or no valid keyframe file is set.
        """
        if not getattr(state, "keyframes_enabled", False):
            return None

        keyframes_path = str(getattr(state, "keyframes_path", "") or "").strip()

        if not keyframes_path:
            print("[3D KEYFRAME] Keyframes enabled, but no keyframe file path was set.")
            return None

        if KeyframeService is None:
            raise RuntimeError("Keyframes are enabled, but services.keyframe_service could not be imported.")

        path = Path(keyframes_path)

        if not path.exists():
            raise FileNotFoundError(f"Keyframe file does not exist: {keyframes_path}")

        service = KeyframeService.load(path)

        if not service.has_keyframes():
            print(f"[3D KEYFRAME] Keyframe file loaded but contains no keyframes: {keyframes_path}")
            return None

        print(f"[3D KEYFRAME] Loaded {len(service.keyframes)} keyframe(s): {keyframes_path}")
        return service

    def _get_aspect_ratios(self):
        return {
            "Default (16:9)": 16 / 9,
            "Classic (4:3)": 4 / 3,
            "Square (1:1)": 1.0,
            "Vertical 9:16": 9 / 16,
            "Instagram 4:5": 4 / 5,
            "CinemaScope (2.39:1)": 2.39,
            "Anamorphic (2.35:1)": 2.35,
            "Modern Cinema (2.40:1)": 2.40,
            "Ultra Panavision (2.76:1)": 2.76,
            "Academy Flat (1.85:1)": 1.85,
            "European Flat (1.66:1)": 1.66,
            "21:9 UltraWide": 21 / 9,
            "32:9 SuperWide": 32 / 9,
            "2:1 (Modern Hybrid)": 2.0,
        }

    def _list_media_files(self, folder_path: str, extensions: set[str]) -> list[Path]:
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return []

        files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]

        return sorted(files, key=lambda p: p.name.lower())

    def _find_matching_depth_file(self, source_file: Path, depth_files: list[Path]) -> Path | None:
        source_stem = source_file.stem.lower()

        for depth_file in depth_files:
            depth_stem = depth_file.stem.lower()

            if depth_stem == source_stem:
                return depth_file

            if depth_stem == f"{source_stem}_depth":
                return depth_file

            if depth_stem.endswith("_depth") and depth_stem[:-len("_depth")] == source_stem:
                return depth_file

        return None

    def _clone_state_for_job(self, state: AppState, input_path: Path, depth_path: Path, output_path: Path):
        job_state = copy.copy(state)
        job_state.input_video_path = str(input_path)
        job_state.depth_map_path = str(depth_path)
        job_state.output_path = str(output_path)
        return job_state

    def _ensure_image_output_path(self, output_path: str) -> str:
        path = Path(output_path)

        if not path.suffix:
            path = path.with_suffix(".png")

        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _temp_sbs_output_path(self, output_path: str) -> str:
        path = Path(output_path)
        ext = path.suffix or ".mp4"
        base = path.with_suffix("")
        return str(base) + ".__vd3d_temp_sbs" + ext

    def _split_eye_output_paths(self, output_path: str, output_format: str):
        left_output = apply_3d_suffix(output_path, output_format, "left")
        right_output = apply_3d_suffix(output_path, output_format, "right")
        return left_output, right_output

    def _get_video_size(self, video_path: str):
        import cv2

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Could not open rendered temp video: {video_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if width <= 0 or height <= 0:
                raise RuntimeError(f"Could not read rendered temp video size: {video_path}")

            return width, height

        finally:
            cap.release()

    def _validate_split_eye_output_format(self, output_format: str):
        fmt = str(output_format or "").strip().lower()

        if fmt in {"red-cyan anaglyph", "passive interlaced"}:
            raise ValueError(
                f"Stereo Output left/right/both is not supported for {output_format}. "
                "Use Full-SBS, Half-SBS, VR, or VR180 SBS/TB."
            )

    def _split_sbs_filter(self, *, side: str, output_format: str, width: int, height: int) -> str:
        fmt = str(output_format).strip().lower()

        if fmt in ("red-cyan anaglyph", "passive interlaced"):
            raise ValueError(
                f"Stereo Output left/right/both is not supported for {output_format}. "
                "Use Full-SBS, Half-SBS, VR, or VR180 SBS/TB."
            )

        # VR180 TB packs left eye on top and right eye on bottom.
        if fmt == "vr180 equirect (tb)":
            half_h = max(2, height // 2)
            y = 0 if side == "left" else half_h
            return f"crop={width}:{half_h}:0:{y}"

        # Normal SBS-style formats pack left/right horizontally.
        half_w = max(2, width // 2)
        x = 0 if side == "left" else half_w
        vf = f"crop={half_w}:{height}:{x}:0"

        # Half-SBS stores each eye squeezed horizontally.
        # After splitting, expand each eye back to a normal mono-eye view.
        if fmt == "half-sbs":
            vf += f",scale={width}:{height}:flags=lanczos"

        return vf

    def _encode_crop(self, *, input_path: str, output_path: str, vf: str, state: AppState):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        codec_key = getattr(state, "selected_ffmpeg_codec", "H.264 / AVC (libx264 - CPU)")
        codec = FFMPEG_CODEC_MAP.get(codec_key, "libx264")

        crf = int(getattr(state, "crf_value", 18))
        nvenc_cq = int(getattr(state, "nvenc_cq_value", crf))

        ffmpeg_exe = require_tool("ffmpeg")

        cmd = [
            ffmpeg_exe,
            "-hide_banner",
            "-nostats",
            "-loglevel", "error",
            "-y",
            "-i", input_path,
            "-vf", vf,
            "-an",
            "-c:v", codec,
        ]

        if "nvenc" in codec:
            cmd += [
                "-preset", "p5",
                "-tune", "hq",
                "-rc", "vbr",
                "-cq", str(nvenc_cq),
                "-b:v", "0",
                "-pix_fmt", "yuv420p",
            ]
        elif codec in {"libx264", "libx265"}:
            cmd += [
                "-preset", "slow",
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
            ]
        elif codec in {"h264_amf", "hevc_amf", "av1_amf"}:
            cmd += [
                "-quality", "quality",
                "-rc", "cqp",
                "-qp_i", str(crf),
                "-qp_p", str(crf),
                "-pix_fmt", "yuv420p",
            ]
        else:
            cmd += [
                "-q:v", "2",
                "-pix_fmt", "yuv420p",
            ]

        cmd += [
            "-movflags", "+faststart",
            str(output),
        ]

        print("[SPLIT EYE CMD]", " ".join(str(x) for x in cmd))

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            **hidden_subprocess_kwargs(),
        )

        stderr = ""

        while True:
            if self.cancel_flag.is_set():
                process.kill()
                _, stderr = process.communicate()
                stderr = stderr or ""

                try:
                    if output.exists():
                        output.unlink()
                except Exception as exc:
                    print(f"[SPLIT EYE] Could not delete partial output after cancel: {exc}")

                raise RenderCancelled("Render cancelled while splitting eye output.")

            try:
                _, stderr = process.communicate(timeout=0.25)
                stderr = stderr or ""
                break
            except subprocess.TimeoutExpired:
                continue

        if process.returncode != 0:
            try:
                if output.exists():
                    output.unlink()
            except Exception as exc:
                print(f"[SPLIT EYE] Could not delete failed split output: {exc}")

            raise RuntimeError(
                f"Failed to split eye output:\n{output_path}\n\n{stderr[-4000:]}"
            )

        if not output.exists() or output.stat().st_size <= 1000:
            try:
                if output.exists():
                    output.unlink()
            except Exception as exc:
                print(f"[SPLIT EYE] Could not delete invalid split output: {exc}")

            raise RuntimeError(f"Split eye output was not created correctly: {output_path}")

        return str(output)

    def _merge_original_audio_if_needed(self, *, state: AppState, video_path: str) -> str:
        if not getattr(state, "keep_original_audio", True):
            return video_path

        start_s = parse_timecode(getattr(state, "clip_start", ""))

        video = Path(video_path)
        temp_audio_output = str(video.with_name(f"{video.stem}.__vd3d_audio_merge{video.suffix}"))

        print(f"🔊 Merging original audio into split eye output: {video.name}")

        merged_path = merge_audio_from_source(
            video_path,
            state.input_video_path,
            temp_audio_output,
            start_s=start_s,
        )

        merged = Path(merged_path or temp_audio_output)

        if not merged.exists() or merged.stat().st_size <= 1000:
            raise RuntimeError(f"Audio merge failed or produced an invalid file: {merged}")

        # Keep the user's expected output path. Replace the silent crop with the audio-merged file.
        if merged.resolve() != video.resolve():
            os.replace(str(merged), str(video))

        # Best-effort cleanup if merge_audio_from_source created something extra.
        try:
            temp_path = Path(temp_audio_output)
            if temp_path.exists() and temp_path.resolve() != video.resolve():
                temp_path.unlink()
        except Exception as exc:
            print(f"[SPLIT EYE] Could not clean audio temp file: {exc}")

        print("🎧 Split eye audio merge done!")
        return str(video)

    def _start_split_eye_video_render(self, state: AppState, requested_eye_mode: str) -> list[str]:
        """
        Safer left/right/both output path.

        Instead of rendering left and right as separate full render passes,
        render one normal SBS temp file, then split the finished SBS into
        left/right mono-eye videos.
        """       
        output_format = getattr(state, "output_format", "Full-SBS")
        self._validate_split_eye_output_format(output_format)
        
        temp_output = self._temp_sbs_output_path(state.output_path)
        temp_done = None

        temp_state = copy.copy(state)
        temp_state.stereo_mode = "sbs"
        temp_state.keep_original_audio = False

        outputs = []

        try:
            self.progress_label.config(
                text="Rendering temporary SBS for eye split..."
            )

            temp_done = self._run_process_video(
                state=temp_state,
                eye_mode="sbs",
                resolved_output_path=temp_output,
            )

            if isinstance(temp_done, (list, tuple)):
                temp_done = temp_done[0]

            width, height = self._get_video_size(temp_done)

            left_output, right_output = self._split_eye_output_paths(
                state.output_path,
                output_format,
            )

            if requested_eye_mode == "both" and Path(left_output) == Path(right_output):
                raise RuntimeError(
                    f"Left and right split-eye outputs resolved to the same path: {left_output}"
                )
                
            if requested_eye_mode in ("left", "both"):
                self.progress_label.config(text="Splitting left eye output...")
                vf_left = self._split_sbs_filter(
                    side="left",
                    output_format=output_format,
                    width=width,
                    height=height,
                )
                left_done = self._encode_crop(
                    input_path=temp_done,
                    output_path=left_output,
                    vf=vf_left,
                    state=state,
                )
                left_done = self._merge_original_audio_if_needed(
                    state=state,
                    video_path=left_done,
                )
                outputs.append(left_done)

            if requested_eye_mode in ("right", "both"):
                self.progress_label.config(text="Splitting right eye output...")
                vf_right = self._split_sbs_filter(
                    side="right",
                    output_format=output_format,
                    width=width,
                    height=height,
                )
                right_done = self._encode_crop(
                    input_path=temp_done,
                    output_path=right_output,
                    vf=vf_right,
                    state=state,
                )
                right_done = self._merge_original_audio_if_needed(
                    state=state,
                    video_path=right_done,
                )
                outputs.append(right_done)

        finally:
            try:
                if temp_done and Path(temp_done).exists():
                    Path(temp_done).unlink()
                    print(f"[SPLIT EYE] Deleted temp SBS: {temp_done}")
            except Exception as exc:
                print(f"[SPLIT EYE] Could not delete temp SBS: {exc}")

        if not outputs:
            raise RuntimeError("Split eye render finished, but no outputs were created.")

        self.progress["value"] = 100
        self.progress_label.config(text="100.00% | FPS: 0.00 | Elapsed: 00:00:00 | ETA: 00:00:00")

        return outputs

    def _set_manual_progress(self, value: float, text: str = ""):
        value = max(0.0, min(100.0, float(value)))

        now = time.time()
        elapsed = 0.0
        eta = None
        fps_like = None
        rate_label = "FPS"

        progress = value

        if self.render_start_time is not None:
            elapsed = now - self.render_start_time

        if self._batch_total is not None and self._batch_index is not None:
            total = max(1, int(self._batch_total))
            index = max(1, int(self._batch_index))

            inner_progress = value / 100.0
            completed_frames = max(0.0, (index - 1) + inner_progress)

            progress = (completed_frames / total) * 100.0
            progress = max(0.0, min(100.0, progress))

            if self._batch_start_time is not None:
                elapsed = now - self._batch_start_time

            if elapsed > 0 and completed_frames > 0:
                fps_like = completed_frames / elapsed
                remaining_frames = max(0.0, total - completed_frames)
                eta = remaining_frames / fps_like if fps_like > 0 else None

            rate_label = self._batch_rate_label or "FPS"

        else:
            if elapsed > 0 and progress > 0:
                fps_like = progress / elapsed
                remaining = max(0.0, 100.0 - progress)
                eta = (remaining / progress) * elapsed

        self.progress.value = progress

        if text:
            self.progress_label.text = text

        if self.progress_callback:
            self.progress_callback({
                "progress": float(progress),
                "status_text": text if text else self.progress_label.text,
                "elapsed": elapsed,
                "eta": eta,
                "fps_like": fps_like,
                "rate_label": rate_label,
            })

    def _run_process_video(
        self,
        *,
        state: AppState,
        eye_mode: str,
        resolved_output_path: str,
    ) -> str:
        input_video_path = VarAdapter(state.input_video_path)
        selected_depth_map = VarAdapter(state.depth_map_path)
        output_sbs_video_path = VarAdapter(resolved_output_path)

        selected_codec = VarAdapter(getattr(state, "selected_codec", "XVID"))
        fg_shift = VarAdapter(state.fg_shift)
        mg_shift = VarAdapter(state.mg_shift)
        bg_shift = VarAdapter(state.bg_shift)
        sharpness_factor = VarAdapter(state.sharpness_factor)
        output_format = VarAdapter(state.output_format)
        selected_aspect_ratio = VarAdapter(getattr(state, "selected_aspect_ratio", "Default (16:9)"))

        feather_strength = VarAdapter(getattr(state, "feather_strength", 0.0))
        blur_ksize = VarAdapter(getattr(state, "blur_ksize", 1))

        use_ffmpeg = VarAdapter(getattr(state, "use_ffmpeg", False))
        preserve_hdr10 = VarAdapter(getattr(state, "preserve_hdr10", False))
        selected_ffmpeg_codec = VarAdapter(
            getattr(state, "selected_ffmpeg_codec", "H.264 / AVC (libx264 - CPU)")
        )
        crf_value = VarAdapter(getattr(state, "crf_value", 23))
        nvenc_cq_value = VarAdapter(getattr(state, "nvenc_cq_value", 23))

        use_subject_tracking = VarAdapter(getattr(state, "use_subject_tracking", False))
        use_floating_window = VarAdapter(getattr(state, "use_floating_window", False))
        max_pixel_shift = VarAdapter(getattr(state, "max_pixel_shift", 0.02))
        auto_crop_black_bars = VarAdapter(getattr(state, "auto_crop_black_bars", False))
        parallax_balance = VarAdapter(getattr(state, "parallax_balance", 0.8))
        preserve_original_aspect = VarAdapter(getattr(state, "preserve_original_aspect", False))
        zero_parallax_strength = VarAdapter(getattr(state, "zero_parallax_strength", 0.0))
        enable_edge_masking = VarAdapter(getattr(state, "enable_edge_masking", True))
        enable_feathering = VarAdapter(getattr(state, "enable_feathering", True))

        # Make sure the UI edge repair dropdown reaches render_sbs_3d().
        edge_repair_quality = VarAdapter(getattr(state, "edge_repair_quality", "Fast"))

        skip_blank_frames = VarAdapter(getattr(state, "skip_blank_frames", False))

        # Default DOF must be OFF unless explicitly enabled.
        dof_strength = VarAdapter(getattr(state, "dof_strength", 0.0))
        convergence_strength = VarAdapter(getattr(state, "convergence_strength", 0.0))
        enable_dynamic_convergence = VarAdapter(getattr(state, "enable_dynamic_convergence", True))
        disable_shift_ema = VarAdapter(getattr(state, "disable_shift_ema", False))

        depth_pop_gamma = VarAdapter(getattr(state, "depth_pop_gamma", 0.85))
        depth_pop_mid = VarAdapter(getattr(state, "depth_pop_mid", 0.50))
        depth_stretch_lo = VarAdapter(getattr(state, "depth_stretch_lo", 0.05))
        depth_stretch_hi = VarAdapter(getattr(state, "depth_stretch_hi", 0.95))
        fg_pop_multiplier = VarAdapter(getattr(state, "fg_pop_multiplier", 1.20))
        bg_push_multiplier = VarAdapter(getattr(state, "bg_push_multiplier", 1.10))
        subject_lock_strength = VarAdapter(getattr(state, "subject_lock_strength", 1.00))

        subject_plane_lock_strength = VarAdapter(
            getattr(state, "subject_plane_lock_strength", 0.0)
        )
        subject_plane_lock_width = VarAdapter(
            getattr(state, "subject_plane_lock_width", 0.08)
        )

        foreground_curvature_strength = VarAdapter(
            getattr(state, "foreground_curvature_strength", 0.06)
        )
        color_saturation = VarAdapter(getattr(state, "saturation", 1.0))
        color_contrast = VarAdapter(getattr(state, "contrast", 1.0))
        color_brightness = VarAdapter(getattr(state, "brightness", 0.0))

        ipd_value = getattr(state, "ipd_scale", 1.0) if getattr(state, "ipd_enabled", True) else 0.0

        start_s = parse_timecode(getattr(state, "clip_start", ""))
        end_s = parse_timecode(getattr(state, "clip_end", ""))

        keep_original_audio = getattr(state, "keep_original_audio", True)

        vr180_equi_w = VarAdapter(getattr(state, "vr180_equi_w", 3840))
        vr180_equi_h = VarAdapter(getattr(state, "vr180_equi_h", 1920))
        vr180_flat_w = VarAdapter(getattr(state, "vr180_flat_w", 1920))
        vr180_flat_h = VarAdapter(getattr(state, "vr180_flat_h", 1080))
        vr180_hfov_deg = VarAdapter(getattr(state, "vr180_hfov_deg", 110.0))

        aspect_ratios = self._get_aspect_ratios()
        keyframe_service = self._load_keyframe_service(state)        

        debug_print(
            "[3D SETTINGS]",
            f"format={getattr(state, 'output_format', None)}",
            f"eye_mode={eye_mode}",
            f"use_ffmpeg={getattr(state, 'use_ffmpeg', None)}",
            f"codec={getattr(state, 'selected_ffmpeg_codec', None)}",
            f"dof_strength={getattr(state, 'dof_strength', None)}",
            f"edge_repair_quality={getattr(state, 'edge_repair_quality', None)}",
            f"enable_edge_masking={getattr(state, 'enable_edge_masking', None)}",
            f"enable_feathering={getattr(state, 'enable_feathering', None)}",
            f"use_subject_tracking={getattr(state, 'use_subject_tracking', None)}",
            f"use_floating_window={getattr(state, 'use_floating_window', None)}",
            f"sharpness={getattr(state, 'sharpness_factor', None)}",
            f"preserve_hdr10={getattr(state, 'preserve_hdr10', None)}",
        )

        out_path_done = process_video(
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
            self.progress,
            self.progress_label,
            self.suspend_flag,
            self.cancel_flag,
            use_ffmpeg,
            preserve_hdr10,
            selected_ffmpeg_codec,
            crf_value,
            nvenc_cq_value,
            use_subject_tracking,
            use_floating_window,
            max_pixel_shift,
            auto_crop_black_bars,
            parallax_balance,
            preserve_original_aspect,
            zero_parallax_strength,
            enable_edge_masking,
            enable_feathering,
            skip_blank_frames,
            dof_strength,
            convergence_strength,
            enable_dynamic_convergence,            
            depth_pop_gamma,
            depth_pop_mid,
            depth_stretch_lo,
            depth_stretch_hi,
            fg_pop_multiplier,
            bg_push_multiplier,
            subject_lock_strength,
            subject_plane_lock_strength,
            subject_plane_lock_width,
            foreground_curvature_strength,
            color_saturation,
            color_contrast,
            color_brightness,
            ipd_value=ipd_value,
            start_s=start_s,
            end_s=end_s,
            eye_mode=eye_mode,
            output_override=resolved_output_path,
            keep_original_audio=keep_original_audio,
            vr180_equi_w_var=vr180_equi_w,
            vr180_equi_h_var=vr180_equi_h,
            vr180_flat_w_var=vr180_flat_w,
            vr180_flat_h_var=vr180_flat_h,
            vr180_hfov_deg_var=vr180_hfov_deg,
            disable_shift_ema=disable_shift_ema,
            keyframe_service=keyframe_service,
            edge_repair_quality=edge_repair_quality,
        )

        if self.cancel_flag.is_set():
            raise RenderCancelled("Render cancelled.")

        if not out_path_done:
            raise RuntimeError(f"Render finished for '{eye_mode}', but no output file was returned.")

        return str(Path(out_path_done))

    def _run_process_image(
        self,
        *,
        state: AppState,
        eye_mode: str,
        resolved_output_path: str,
    ) -> str:
        input_image_path = state.input_video_path
        depth_image_path = state.depth_map_path
        resolved_output_path = self._ensure_image_output_path(resolved_output_path)

        if not input_image_path:
            raise ValueError("No input image selected.")

        if not depth_image_path:
            raise ValueError("No depth image selected.")

        if not resolved_output_path:
            raise ValueError("No output image path selected.")

        if self.cancel_flag.is_set():
            raise RenderCancelled("Render cancelled.")

        Path(resolved_output_path).parent.mkdir(parents=True, exist_ok=True)

        self._set_manual_progress(5, f"Rendering 3D image: {Path(input_image_path).name}")

        ipd_factor = getattr(state, "ipd_scale", 1.0) if getattr(state, "ipd_enabled", True) else 0.0

        out_path_done = render_sbs_3d_image(
            input_image_path=input_image_path,
            depth_image_path=depth_image_path,
            output_image_path=resolved_output_path,
            fg_shift=getattr(state, "fg_shift", 0.0),
            mg_shift=getattr(state, "mg_shift", 0.0),
            bg_shift=getattr(state, "bg_shift", 0.0),
            sharpness_factor=getattr(state, "sharpness_factor", 0.0),
            output_format=getattr(state, "output_format", "Full-SBS"),
            selected_aspect_ratio=getattr(state, "selected_aspect_ratio", "Default (16:9)"),
            aspect_ratios=self._get_aspect_ratios(),
            preserve_original_aspect=getattr(state, "preserve_original_aspect", True),
            feather_strength=getattr(state, "feather_strength", 0.0),
            blur_ksize=getattr(state, "blur_ksize", 1),
            use_subject_tracking=getattr(state, "use_subject_tracking", False),
            use_floating_window=getattr(state, "use_floating_window", False),
            max_pixel_shift_percent=getattr(state, "max_pixel_shift", 0.02),
            auto_crop_black_bars=getattr(state, "auto_crop_black_bars", False),
            parallax_balance=getattr(state, "parallax_balance", 0.8),
            zero_parallax_strength=getattr(state, "zero_parallax_strength", 0.0),
            enable_edge_masking=getattr(state, "enable_edge_masking", True),
            enable_feathering=getattr(state, "enable_feathering", True),
            dof_strength=getattr(state, "dof_strength", 0.0),
            convergence_strength=getattr(state, "convergence_strength", 0.0),
            enable_dynamic_convergence=getattr(state, "enable_dynamic_convergence", True),
            ipd_factor=ipd_factor,
            depth_pop_gamma=getattr(state, "depth_pop_gamma", 0.85),
            depth_pop_mid=getattr(state, "depth_pop_mid", 0.50),
            depth_stretch_lo=getattr(state, "depth_stretch_lo", 0.05),
            depth_stretch_hi=getattr(state, "depth_stretch_hi", 0.95),
            fg_pop_multiplier=getattr(state, "fg_pop_multiplier", 1.20),
            bg_push_multiplier=getattr(state, "bg_push_multiplier", 1.10),
            subject_lock_strength=getattr(state, "subject_lock_strength", 1.00),
            subject_plane_lock_strength=getattr(state, "subject_plane_lock_strength", 0.0),
            subject_plane_lock_width=getattr(state, "subject_plane_lock_width", 0.08),
            foreground_curvature_strength=getattr(state, "foreground_curvature_strength", 0.06),
            color_saturation=getattr(state, "saturation", 1.0),
            color_contrast=getattr(state, "contrast", 1.0),
            color_brightness=getattr(state, "brightness", 0.0),
            eye_mode=eye_mode,
            disable_shift_ema=getattr(state, "disable_shift_ema", False),
        )

        if self.cancel_flag.is_set():
            raise RenderCancelled("Render cancelled.")

        if not out_path_done:
            raise RuntimeError(f"Image render finished for '{eye_mode}', but no output file was returned.")

        self._set_manual_progress(100, f"Saved 3D image: {Path(out_path_done).name}")
        return str(Path(out_path_done))

    def start_3d_render(self, state: AppState) -> list[str]:
        mode = getattr(state, "render_mode", "video").strip().lower()

        self.reset_flags()
        self.render_start_time = time.time()
        self.progress.value = 0
        self.progress_label.text = ""

        self._batch_total = None
        self._batch_index = None
        self._batch_start_time = None
        self._batch_rate_label = "FPS"

        if mode == "image":
            return self._start_single_image_render(state)

        if mode == "video_folder":
            return self._start_batch_video_render(state)

        if mode == "image_folder":
            return self._start_image_folder_render(state)

        return self._start_single_video_render(state)
        
    def _start_single_video_render(self, state: AppState) -> list[str]:
        if not state.input_video_path:
            raise ValueError("No input video selected.")
        if not state.depth_map_path:
            raise ValueError("No depth map selected.")
        if not state.output_path:
            raise ValueError("No output path selected.")

        eye_mode = getattr(state, "stereo_mode", "sbs").strip().lower()

        # New safer path:
        # Render once as SBS, then split/crop at the end.
        if eye_mode in ("left", "right", "both"):
            return self._start_split_eye_video_render(state, eye_mode)

        resolved_output_path = apply_3d_suffix(
            state.output_path,
            state.output_format,
            eye_mode,
        )

        out_path_done = self._run_process_video(
            state=state,
            eye_mode=eye_mode,
            resolved_output_path=resolved_output_path,
        )

        if isinstance(out_path_done, (list, tuple)):
            return [str(Path(p)) for p in out_path_done]

        return [str(Path(out_path_done))]

    def _start_single_image_render(self, state: AppState) -> list[str]:
        if not state.input_video_path:
            raise ValueError("No input image selected.")
        if not state.depth_map_path:
            raise ValueError("No depth image selected.")
        if not state.output_path:
            raise ValueError("No output image path selected.")

        eye_mode = getattr(state, "stereo_mode", "sbs").strip().lower()

        if eye_mode == "both":
            left_output = apply_3d_suffix(state.output_path, state.output_format, "left")
            right_output = apply_3d_suffix(state.output_path, state.output_format, "right")

            left_done = self._run_process_image(
                state=state,
                eye_mode="left",
                resolved_output_path=left_output,
            )

            if self.cancel_flag.is_set():
                raise RenderCancelled("Render cancelled.")

            right_done = self._run_process_image(
                state=state,
                eye_mode="right",
                resolved_output_path=right_output,
            )

            return [left_done, right_done]

        resolved_output_path = apply_3d_suffix(
            state.output_path,
            state.output_format,
            eye_mode,
        )

        out_path_done = self._run_process_image(
            state=state,
            eye_mode=eye_mode,
            resolved_output_path=resolved_output_path,
        )

        return [out_path_done]

    def _start_batch_video_render(self, state: AppState) -> list[str]:
        input_dir = Path(state.input_video_path)
        depth_dir = Path(state.depth_map_path)
        output_dir = Path(state.output_path)

        if not input_dir.is_dir():
            raise ValueError("Input video folder does not exist.")
        if not depth_dir.is_dir():
            raise ValueError("Depth video folder does not exist.")

        output_dir.mkdir(parents=True, exist_ok=True)

        video_files = self._list_media_files(str(input_dir), VIDEO_EXTENSIONS)
        depth_files = self._list_media_files(str(depth_dir), VIDEO_EXTENSIONS)

        if not video_files:
            raise ValueError("No video files found in the input folder.")
        if not depth_files:
            raise ValueError("No depth video files found in the depth folder.")

        outputs = []
        total = len(video_files)

        self._batch_total = total
        self._batch_index = 1
        self._batch_start_time = time.time()
        self._batch_rate_label = "Videos/s"

        try:
            for index, video_file in enumerate(video_files, start=1):
                if self.cancel_flag.is_set():
                    raise RenderCancelled("Render cancelled.")

                self._batch_index = index

                depth_file = self._find_matching_depth_file(video_file, depth_files)

                if depth_file is None:
                    print(f"[Batch 3D] Skipping, no matching depth map: {video_file.name}")
                    continue

                base_output = output_dir / f"{video_file.stem}.mp4"

                job_state = self._clone_state_for_job(
                    state,
                    input_path=video_file,
                    depth_path=depth_file,
                    output_path=base_output,
                )

                self.progress_label.config(
                    text=f"Batch video {index}/{total}: {video_file.name}"
                )

                outputs.extend(self._start_single_video_render(job_state))

            if not outputs:
                raise RuntimeError("Batch video render finished, but no files were created.")

            self._batch_index = total
            self._set_manual_progress(100, "Batch video render complete.")
            return outputs

        finally:
            self._batch_total = None
            self._batch_index = None
            self._batch_start_time = None
            self._batch_rate_label = "FPS"

    def _start_image_folder_render(self, state: AppState) -> list[str]:
        input_dir = Path(state.input_video_path)
        depth_dir = Path(state.depth_map_path)
        output_dir = Path(state.output_path)

        if not input_dir.is_dir():
            raise ValueError("Input image folder does not exist.")
        if not depth_dir.is_dir():
            raise ValueError("Depth image folder does not exist.")

        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = self._list_media_files(str(input_dir), IMAGE_EXTENSIONS)
        depth_files = self._list_media_files(str(depth_dir), IMAGE_EXTENSIONS)

        if not image_files:
            raise ValueError("No image files found in the input folder.")
        if not depth_files:
            raise ValueError("No depth image files found in the depth folder.")

        outputs = []
        total = len(image_files)

        self._batch_total = total
        self._batch_index = 1
        self._batch_start_time = time.time()
        self._batch_rate_label = "Images/s"

        try:
            for index, image_file in enumerate(image_files, start=1):
                if self.cancel_flag.is_set():
                    raise RenderCancelled("Render cancelled.")

                self._batch_index = index

                depth_file = self._find_matching_depth_file(image_file, depth_files)

                if depth_file is None:
                    print(f"[Image Folder 3D] Skipping, no matching depth image: {image_file.name}")
                    continue

                base_output = output_dir / f"{image_file.stem}.png"

                job_state = self._clone_state_for_job(
                    state,
                    input_path=image_file,
                    depth_path=depth_file,
                    output_path=base_output,
                )

                self._set_manual_progress(
                    0,
                    f"Image folder render {index}/{total}: {image_file.name}",
                )

                outputs.extend(self._start_single_image_render(job_state))

            if not outputs:
                raise RuntimeError("Image folder render finished, but no files were created.")

            self._batch_index = total
            self._set_manual_progress(100, "Image folder render complete.")
            return outputs

        finally:
            self._batch_total = None
            self._batch_index = None
            self._batch_start_time = None
            self._batch_rate_label = "FPS"
