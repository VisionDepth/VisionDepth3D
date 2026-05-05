import threading
import time
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
        self.value = 0
        self.callback = callback
        self._maximum = 100
        self._mode = "determinate"

    def __setitem__(self, key, value):
        if key == "value":
            self.value = value
            if self.callback:
                self.callback(progress=value)

    def __getitem__(self, key):
        if key == "value":
            return self.value
        if key == "maximum":
            return self._maximum
        raise KeyError(key)

    def config(self, **kwargs):
        changed = False
        if "value" in kwargs:
            self.value = kwargs["value"]
            changed = True
        if "maximum" in kwargs:
            self._maximum = kwargs["maximum"]
        if "mode" in kwargs:
            self._mode = kwargs["mode"]
        if changed and self.callback:
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

    def _emit_progress_update(self, progress=None, status_text=None):
        if not self.progress_callback:
            return

        now = time.time()
        elapsed = 0.0
        fps = 0.0
        eta = None

        if self.start_time is not None:
            elapsed = now - self.start_time

        current_progress = self.progress.value if progress is None else progress

        if elapsed > 0 and current_progress > 0:
            remaining = max(0.0, 100.0 - current_progress)
            fps = current_progress / elapsed
            eta = (remaining / current_progress) * elapsed if current_progress > 0 else None

        self.progress_callback({
            "progress": float(current_progress),
            "status_text": status_text if status_text is not None else self.progress_label.text,
            "elapsed": elapsed,
            "eta": eta,
            "fps_like": fps,
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