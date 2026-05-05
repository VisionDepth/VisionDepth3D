import threading
import time
from pathlib import Path

from models.app_state import AppState
from core.render_3d import process_video, parse_timecode

class RenderCancelled(Exception):
    pass

def apply_3d_suffix(base_out: str, output_format: str, eye_mode: str) -> str:
    path = Path(base_out)
    base = str(path.with_suffix(""))
    ext = path.suffix
    
    fmt = output_format.strip().lower()
    mode = eye_mode.strip().lower()

    suffix = ""

    # --- VR180 (DeoVR compliant naming) ---
    if fmt == "vr180 equirect (tb)":
        suffix = "_TB_180"
    elif fmt == "vr180 equirect (sbs)":
        suffix = "_SBS_180"

    # --- Standard Stereo ---
    elif mode == "sbs":
        if fmt == "full-sbs":
            suffix = "_LR_Full_SBS"
        elif fmt == "half-sbs":
            suffix = "_LR_Half_SBS"
        elif fmt == "vr":
            suffix = "_VR"
        elif fmt == "red-cyan anaglyph":
            suffix = "_Anaglyph"
        elif fmt == "passive interlaced":
            suffix = "_Interlaced"

    elif mode == "left":
        suffix = "_LR_Left"

    elif mode == "right":
        suffix = "_LR_Right"

    elif mode == "both":
        pass  # handled elsewhere if you later split outputs

    if not suffix:
        return base_out

    # Avoid doubling suffix if user re-renders to a previously suffixed path
    if base.endswith(suffix):
        return f"{base}{ext}"

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

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def _emit_progress_update(self, progress=None, status_text=None):
        if not self.progress_callback:
            return

        now = time.time()
        elapsed = 0.0
        fps = 0.0
        eta = None

        if self.render_start_time is not None:
            elapsed = now - self.render_start_time

        current_progress = self.progress.value if progress is None else progress

        if elapsed > 0 and current_progress > 0:
            fps_like = current_progress / elapsed
            fps = fps_like
            remaining = max(0.0, 100.0 - current_progress)
            eta = (remaining / current_progress) * elapsed if current_progress > 0 else None

        self.progress_callback({
            "progress": float(current_progress),
            "status_text": status_text if status_text is not None else self.progress_label.text,
            "elapsed": elapsed,
            "eta": eta,
            "fps_like": fps,
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
        skip_blank_frames = VarAdapter(getattr(state, "skip_blank_frames", False))
        dof_strength = VarAdapter(getattr(state, "dof_strength", 2.0))
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

        aspect_ratios = {
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
        )

        if self.cancel_flag.is_set():
            raise RenderCancelled("Render cancelled.")

        if not out_path_done:
            raise RuntimeError(f"Render finished for '{eye_mode}', but no output file was returned.")

        return str(Path(out_path_done))

    def start_3d_render(self, state: AppState) -> list[str]:
        if not state.input_video_path:
            raise ValueError("No input video selected.")
        if not state.depth_map_path:
            raise ValueError("No depth map selected.")
        if not state.output_path:
            raise ValueError("No output path selected.")

        self.reset_flags()
        self.render_start_time = time.time()
        self.progress.value = 0
        self.progress_label.text = ""

        eye_mode = getattr(state, "stereo_mode", "sbs").strip().lower()

        if eye_mode == "both":
            left_output = apply_3d_suffix(state.output_path, state.output_format, "left")
            right_output = apply_3d_suffix(state.output_path, state.output_format, "right")

            left_done = self._run_process_video(
                state=state,
                eye_mode="left",
                resolved_output_path=left_output,
            )

            if self.cancel_flag.is_set():
                raise RenderCancelled("Render cancelled.")

            right_done = self._run_process_video(
                state=state,
                eye_mode="right",
                resolved_output_path=right_output,
            )

            if self.cancel_flag.is_set():
                raise RenderCancelled("Render cancelled.")

            return [left_done, right_done]

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

        return [out_path_done]