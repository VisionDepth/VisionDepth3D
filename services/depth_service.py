import threading
import time
import re
import logging
from pathlib import Path

from models.depth_state import DepthState

logger = logging.getLogger(__name__)


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
        return str(self._value).strip()

    def __str__(self):
        return str(self._value)

    def __bool__(self):
        return bool(self._value)

    def __eq__(self, other):
        if isinstance(other, VarAdapter):
            return self._value == other._value
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
        
    def cget(self, key):
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

    def cget(self, key):
        if key == "text":
            return self.text
        return None

    def after(self, delay_ms, callback=None, *args):
        """
        Tk-compatible .after() replacement for the PySide6 worker bridge.

        Important:
        Do NOT spawn threading.Timer per frame. The depth worker already runs
        in a background QThread, and config() emits a Qt signal safely through
        the controller callback.
        """
        if callback is None:
            return None

        try:
            callback(*args)
        except Exception:
            logger.exception("Depth progress proxy after() callback failed")

        return None

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

    def after(self, delay_ms, callback=None, *args):
        """
        Tk-compatible .after() replacement for the PySide6 worker bridge.

        Important:
        The old version spawned threading.Timer for every status update.
        process_video2 can call this every frame, which creates massive
        overhead and can make depth rendering crawl.

        We execute immediately in the depth worker thread. The callback calls
        config(), which emits progress through the service callback/Qt signal.
        """
        if callback is None:
            return None

        try:
            callback(*args)
        except Exception:
            logger.exception("Depth progress label after() callback failed")

        return None

    def after_cancel(self, timer):
        return None

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

    def _call_legacy_control(self, function_name):
        try:
            import core.render_depth as render_depth
            function = getattr(render_depth, function_name, None)
            if callable(function):
                function()
        except Exception:
            logger.debug("Failed to call core.render_depth.%s", function_name, exc_info=True)

    def _set_legacy_event(self, event_name, action):
        try:
            import core.render_depth as render_depth
            event = getattr(render_depth, event_name, None)
            method = getattr(event, action, None)
            if callable(method):
                method()
        except Exception:
            logger.debug(
                "Failed to %s core.render_depth.%s",
                action,
                event_name,
                exc_info=True,
            )

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

        payload = {
            "progress": percent,
            "status_text": active_status_text,
            "elapsed": elapsed,
            "eta": eta,
            "fps_like": fps_like,
            "rate_label": "FPS",
        }

        try:
            self.progress_callback(payload)
        except Exception:
            logger.exception("Depth progress callback failed")
        
    def request_suspend(self):
        self._call_legacy_control("request_depth_pause")
        self.suspend_flag.set()
        self._set_legacy_event("suspend_flag", "set")

    def request_resume(self):
        self._call_legacy_control("request_depth_resume")
        self.suspend_flag.clear()
        self._set_legacy_event("suspend_flag", "clear")

    def request_cancel(self):
        self._call_legacy_control("request_depth_cancel")
        self.cancel_flag.set()
        self._set_legacy_event("cancel_flag", "set")
        self._set_legacy_event("cancel_requested", "set")

    def reset_flags(self):
        self.suspend_flag.clear()
        self.cancel_flag.clear()

        # Important: core.render_depth uses module-level flags too.
        # If these are not cleared, a new depth job can immediately cancel.
        self._set_legacy_event("suspend_flag", "clear")
        self._set_legacy_event("cancel_flag", "clear")
        self._set_legacy_event("cancel_requested", "clear")

    def start_depth_processing(self, state: DepthState) -> str:
        if not state.input_video_path:
            raise ValueError("No input video selected.")
        if not state.output_dir:
            raise ValueError("No output directory selected.")
        
        input_path = Path(state.input_video_path)
        if not input_path.is_file():
            raise ValueError(f"Input video does not exist: {input_path}")

        output_dir_path = Path(state.output_dir)
        if output_dir_path.exists() and not output_dir_path.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir_path}")

        try:
            output_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Could not create output directory: {output_dir_path}") from e

        self.reset_flags()
        self.start_time = time.time()
        self.progress.value = 0
        self.progress_label.text = ""

        from core.render_depth import process_video2
        import cv2

        # Get the real frame count for progress display.
        # Fail early if OpenCV cannot open the video; otherwise progress is misleading.
        cap = cv2.VideoCapture(str(input_path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open input video: {input_path}")

            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            real_total = int(frame_count) if frame_count and frame_count > 0 else 1
        finally:
            cap.release()
        
        self.progress["maximum"] = max(1, real_total)
        self.progress["value"] = 0

        batch_size_value = state.batch_size
        try:
            if hasattr(batch_size_value, "get"):
                batch_size = int(batch_size_value.get())
            else:
                batch_size = int(batch_size_value)
        except (ValueError, TypeError):
            batch_size = 4

        batch_size = max(1, batch_size)

        try:
            vda_overlap = int(getattr(state, "vda_overlap", 4))
        except (ValueError, TypeError):
            vda_overlap = 4

        # Overlap must be lower than the active window size.
        vda_overlap = max(0, min(vda_overlap, batch_size - 1))

        output_dir = VarAdapter(str(output_dir_path))
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
        disable_scene_normalization = getattr(state, "disable_scene_normalization", False)

        try:
            result_path = process_video2(
                file_path=str(input_path),
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
                disable_scene_normalization=disable_scene_normalization,
                vda_overlap=vda_overlap,
            )
        except Exception as e:
            if self.cancel_flag.is_set():
                raise DepthCancelled("Depth processing cancelled.") from e
            raise RuntimeError(f"Depth processing failed: {e}") from e

        if self.cancel_flag.is_set():
            raise DepthCancelled("Depth processing cancelled.")

        if isinstance(result_path, (str, Path)) and result_path:
            return str(result_path)

        input_stem = input_path.stem
        output_path = output_dir_path / f"{input_stem}_depth.mkv"
        return str(output_path)
