import threading
import time
import re
from pathlib import Path

from models.depth_state import DepthState


class DepthCancelled(Exception):
    pass


class VarAdapter:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value

    def strip(self):
        s = str(self._value).strip()
        if s.isdigit():
            return int(s)
        try:
            return float(s)
        except ValueError:
            return s

    def __str__(self):
        return str(self._value)

    def __bool__(self):
        return bool(self._value)

    def __eq__(self, other):
        return self._value == other

    def __hash__(self):
        return hash(self._value)


class DepthProgressProxy:
    def __init__(self, callback=None):
        self.callback = callback
        self.value = 0
        self._maximum = 100

    def __setitem__(self, key, value):
        if key == "value":
            self.value = value
            if self.callback:
                self.callback(progress=value)

        elif key == "maximum":
            self._maximum = value
            if self.callback:
                self.callback(progress=self.value)

    def __getitem__(self, key):
        if key == "value":
            return self.value
        if key == "maximum":
            return self._maximum
        return None

    def config(self, **kwargs):
        if "maximum" in kwargs:
            self._maximum = kwargs["maximum"]
            if self.callback:
                self.callback(progress=self.value)

        if "value" in kwargs:
            self.value = kwargs["value"]
            if self.callback:
                self.callback(progress=self.value)

    def configure(self, **kwargs):
        self.config(**kwargs)

    def update(self):
        pass

    def update_idletasks(self):
        pass

    def start(self, interval=None):
        pass

    def stop(self):
        pass


class DepthProgressLabelProxy:
    def __init__(self, callback=None):
        self.text = ""
        self.callback = callback

    def config(self, **kwargs):
        if "text" in kwargs:
            self.text = kwargs["text"]
            if self.callback:
                self.callback(status_text=self.text)

    def configure(self, **kwargs):
        self.config(**kwargs)

    def update(self):
        pass

    def update_idletasks(self):
        pass

    def winfo_toplevel(self):
        return self

    def after(self, delay_ms, callback):
        import threading as _threading
        timer = _threading.Timer(delay_ms / 1000.0, callback)
        timer.daemon = True
        timer.start()


class DepthService:
    """Service that wraps the legacy render_depth.process_video2 function
    with the same progress/cancel/suspend pattern used by RenderService."""

    def __init__(self):
        self.progress_callback = None
        self.progress = DepthProgressProxy(self._emit_progress_update)
        self.progress_label = DepthProgressLabelProxy(self._emit_progress_update)

        self.suspend_flag = threading.Event()
        self.cancel_flag = threading.Event()

        self.start_time = None

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def _parse_hms(self, text):
        try:
            parts = str(text).strip().split(":")
            if len(parts) == 3:
                h, m, s = [int(float(p)) for p in parts]
                return h * 3600 + m * 60 + s
            if len(parts) == 2:
                m, s = [int(float(p)) for p in parts]
                return m * 60 + s
        except Exception:
            pass

        return None


    def _parse_depth_status_text(self, text):
        """
        Parses legacy depth status text like:
        56/7188 | FPS: 1.2 | ETA: 01:43:18
        """
        result = {
            "completed_units": None,
            "total_units": None,
            "fps_like": None,
            "eta": None,
        }

        if not text:
            return result

        text = str(text)

        match = re.search(r"(\d+)\s*/\s*(\d+)", text)
        if match:
            result["completed_units"] = float(match.group(1))
            result["total_units"] = float(match.group(2))

        fps_match = re.search(r"FPS\s*:\s*([0-9.]+)", text, re.IGNORECASE)
        if fps_match:
            try:
                result["fps_like"] = float(fps_match.group(1))
            except Exception:
                pass

        eta_match = re.search(r"ETA\s*:\s*([0-9:.]+)", text, re.IGNORECASE)
        if eta_match:
            result["eta"] = self._parse_hms(eta_match.group(1))

        return result

    def _emit_progress_update(self, progress=None, status_text=None):
        if not self.progress_callback:
            return

        now = time.time()
        elapsed = 0.0
        eta = None
        fps_like = None

        if self.start_time is not None:
            elapsed = max(0.0, now - self.start_time)

        active_status_text = status_text if status_text is not None else self.progress_label.text
        parsed = self._parse_depth_status_text(active_status_text)

        raw_value = self.progress.value if progress is None else progress

        try:
            raw_value = float(raw_value or 0.0)
        except Exception:
            raw_value = 0.0

        try:
            maximum = float(getattr(self.progress, "_maximum", 100) or 100)
        except Exception:
            maximum = 100.0

        completed_units = raw_value
        total_units = maximum

        # Prefer parsed frame count from legacy depth status text:
        # "56/7188 | FPS: 1.2 | ETA: 01:43:18"
        if parsed["completed_units"] is not None and parsed["total_units"]:
            completed_units = parsed["completed_units"]
            total_units = parsed["total_units"]
            percent = (completed_units / max(total_units, 1.0)) * 100.0
        elif maximum > 100 and raw_value <= maximum:
            percent = (raw_value / max(maximum, 1.0)) * 100.0
        else:
            percent = raw_value

        percent = max(0.0, min(100.0, float(percent)))

        if parsed["fps_like"] is not None:
            fps_like = parsed["fps_like"]
        elif elapsed > 0 and completed_units > 0:
            fps_like = completed_units / elapsed

        if parsed["eta"] is not None:
            eta = parsed["eta"]
        elif elapsed > 0 and fps_like and fps_like > 0:
            remaining_units = max(0.0, total_units - completed_units)
            eta = remaining_units / fps_like

        self.progress_callback({
            "progress": percent,
            "status_text": active_status_text,
            "elapsed": elapsed,
            "eta": eta,
            "fps_like": fps_like,
            "rate_label": "FPS",
        })
        
    def request_suspend(self):
        try:
            from core.render_depth import request_depth_pause
            request_depth_pause()
        except Exception:
            pass
        self.suspend_flag.set()
        import core.render_depth
        core.render_depth.suspend_flag.set()

    def request_resume(self):
        try:
            from core.render_depth import request_depth_resume
            request_depth_resume()
        except Exception:
            pass
        self.suspend_flag.clear()
        import core.render_depth
        core.render_depth.suspend_flag.clear()

    def request_cancel(self):
        try:
            from core.render_depth import request_depth_cancel
            request_depth_cancel()
        except Exception:
            pass
        self.cancel_flag.set()
        import core.render_depth
        core.render_depth.cancel_flag.set()
        core.render_depth.cancel_requested.set()

    def reset_flags(self):
        self.suspend_flag.clear()
        self.cancel_flag.clear()

    def start_depth_processing(self, state: DepthState) -> str:
        if not state.input_video_path:
            raise ValueError("No input video selected.")
        if not state.output_dir:
            raise ValueError("No output directory selected.")

        self.reset_flags()
        self.start_time = time.time()
        self.progress.value = 0
        self.progress_label.text = ""

        from core.render_depth import process_video2
        import cv2

        # Get the real frame count for progress display
        cap = cv2.VideoCapture(state.input_video_path)
        real_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 1
        cap.release()
        
        self.progress["maximum"] = max(1, real_total)
        self.progress["value"] = 0

        batch_size_value = state.batch_size
        try:
            if hasattr(batch_size_value, 'get'):
                batch_size = int(batch_size_value.get())
            else:
                batch_size = int(batch_size_value)
        except (ValueError, TypeError):
            batch_size = 4
        output_dir = VarAdapter(state.output_dir)
        inference_res_text = VarAdapter(state.inference_resolution)
        status_label = self.progress_label
        progress_bar = self.progress
        cancel_requested = self.cancel_flag
        invert_value = VarAdapter(state.invert_depth)
        ffmpeg_codec = VarAdapter(state.codec)
        inference_steps_value = VarAdapter(str(state.inference_steps))
        offload_mode_dropdown = VarAdapter(state.offload_mode)
        target_fps = state.target_fps
        ignore_letterbox_bars = state.ignore_letterbox_bars
        prefer_opencv_writer = state.prefer_opencv_writer
        save_frames = state.save_frames

        try:
            process_video2(
                file_path=state.input_video_path,
                total_frames_all=real_total,
                frames_processed_all=0,
                batch_size=batch_size,
                output_dir=output_dir,
                inference_res_text=inference_res_text,
                status_label=status_label,
                progress_bar=progress_bar,
                cancel_requested=cancel_requested,
                invert_value=invert_value,
                ffmpeg_codec=ffmpeg_codec,
                inference_steps_value=inference_steps_value,
                offload_mode_dropdown=offload_mode_dropdown,
                save_frames=save_frames,
                target_fps=target_fps,
                ignore_letterbox_bars=ignore_letterbox_bars,
                prefer_opencv_writer=prefer_opencv_writer,
            )
        except Exception as e:
            if self.cancel_flag.is_set():
                raise DepthCancelled("Depth processing cancelled.")
            raise RuntimeError(f"Depth processing failed: {e}")

        if self.cancel_flag.is_set():
            raise DepthCancelled("Depth processing cancelled.")

        input_stem = Path(state.input_video_path).stem
        output_path = Path(state.output_dir) / f"{input_stem}_depth.mkv"
        return str(output_path)
