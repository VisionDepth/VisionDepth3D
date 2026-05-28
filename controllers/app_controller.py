from PySide6.QtCore import QObject, Signal, QThread

from models.app_state import AppState
from models.depth_state import DepthState
from services.settings_service import SettingsService
from services.render_service import RenderService, RenderCancelled
from services.preview_service import PreviewService
from services.preset_service import PresetService
from services.depth_service import DepthService, DepthCancelled

from services.language_service import LanguageService
from core.vd3d_live import run_live
from core.debug_flags import debug_print, is_debug_enabled

import copy

class _DepthVar:
    """Minimal Tkinter Variable-compatible wrapper for depth pipeline args."""
    def __init__(self, value):
        self._value = value
    def get(self):
        return self._value
    def set(self, value):
        self._value = value
    def after(self, *args, **kwargs):
        pass
    def config(self, *args, **kwargs):
        pass
    def winfo_toplevel(self):
        return self
    def __bool__(self):
        return bool(self._value)


class RenderWorker(QObject):
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, render_service: RenderService, state: AppState):
        super().__init__()
        self.render_service = render_service
        self.state = state

    def run(self):
        try:
            debug_print(
                "[RENDER WORKER STATE]",
                f"format={getattr(self.state, 'output_format', None)}",
                f"stereo_mode={getattr(self.state, 'stereo_mode', None)}",
                f"use_ffmpeg={getattr(self.state, 'use_ffmpeg', None)}",
                f"codec={getattr(self.state, 'selected_ffmpeg_codec', None)}",
                f"dof_strength={getattr(self.state, 'dof_strength', None)}",
                f"edge_repair_quality={getattr(self.state, 'edge_repair_quality', None)}",
                f"enable_edge_masking={getattr(self.state, 'enable_edge_masking', None)}",
                f"enable_feathering={getattr(self.state, 'enable_feathering', None)}",
                f"use_subject_tracking={getattr(self.state, 'use_subject_tracking', None)}",
                f"use_floating_window={getattr(self.state, 'use_floating_window', None)}",
                f"sharpness={getattr(self.state, 'sharpness_factor', None)}",
                f"preserve_hdr10={getattr(self.state, 'preserve_hdr10', None)}",
            )

            outputs = self.render_service.start_3d_render(self.state)
            self.finished.emit(outputs)

        except RenderCancelled:
            self.failed.emit("__CANCELLED__")
        except Exception as e:
            self.failed.emit(str(e))

class DepthWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, depth_service: DepthService, state: DepthState):
        super().__init__()
        self.depth_service = depth_service
        self.state = state

    def run(self):
        try:
            output = self.depth_service.start_depth_processing(self.state)
            self.finished.emit(output)
        except DepthCancelled:
            self.failed.emit("__CANCELLED__")
        except Exception as e:
            self.failed.emit(str(e))


class AppController(QObject):
    state_changed = Signal(str, object)
    settings_loaded = Signal()

    # 3D Render signals
    render_started = Signal()
    render_finished = Signal(list)
    render_failed = Signal(str)
    render_cancelled = Signal()
    render_suspended = Signal()
    render_resumed = Signal()
    render_progress = Signal(dict)

    # Preview signals
    preview_requested = Signal()
    preview_updated = Signal(object)
    preview_failed = Signal(str)

    # Depth processing signals
    depth_started = Signal()
    depth_finished = Signal(str)
    depth_failed = Signal(str)
    depth_cancelled = Signal()
    depth_suspended = Signal()
    depth_resumed = Signal()
    depth_progress_updated = Signal(dict)
    depth_model_ready = Signal()

    # Live 3D signals
    live_started = Signal()
    live_finished = Signal()
    live_failed = Signal(str)
    live_status = Signal(str)

    # Language signals
    language_changed = Signal(str)

    def __init__(
        self,
        state: AppState,
        depth_state: DepthState,
        settings_service: SettingsService,
        render_service: RenderService,
        depth_service: DepthService,
        preview_service: PreviewService,
        preset_service: PresetService,
        language_service: LanguageService,
        live_service,
    ):
        super().__init__()
        self.state = state
        self.depth_state = depth_state
        self.settings_service = settings_service
        self.render_service = render_service
        self.depth_service = depth_service
        self.preview_service = preview_service
        self.preset_service = preset_service
        self.language_service = language_service
        self.live_service = live_service

        # Connect language change signal
        self.language_service.language_changed.connect(self._on_language_changed)
        # Connect Live 3D service signals
        self.live_service.started.connect(self.live_started.emit)
        self.live_service.finished.connect(self.live_finished.emit)
        self.live_service.failed.connect(self.live_failed.emit)
        self.live_service.status.connect(self.live_status.emit)

        self.render_thread = None
        self.render_worker = None
        self.depth_thread = None
        self.depth_worker = None
        self._model_ready = False

        # Processing state trackers
        self._render_running = False
        self._render_suspended = False
        self._depth_running = False
        self._depth_suspended = False

    # ── Generic state access ──
    def set_state(self, key: str, value):
        if not hasattr(self.state, key):
            raise AttributeError(f"AppState has no field named '{key}'")
        setattr(self.state, key, value)
        self.state_changed.emit(key, value)
        self.save_settings()

    def get_state(self, key: str):
        if not hasattr(self.state, key):
            raise AttributeError(f"AppState has no field named '{key}'")
        return getattr(self.state, key)

    # ── Settings ──
    def load_settings(self):
        loaded = self.settings_service.load()
        if not loaded:
            return

        for key, value in loaded.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

        self.settings_loaded.emit()

    def save_settings(self):
        self.settings_service.save(self.state.to_dict())

    # ── 3D Render ──
    def start_render(self):
        if self.render_thread is not None and self.render_thread.isRunning():
            self.render_failed.emit("A render is already running.")
            return

        self._render_running = True
        self._render_suspended = False

        self.render_service.set_progress_callback(
            lambda payload: self.render_progress.emit(payload)
        )

        # Freeze settings for this render.
        # Prevents UI changes from mutating AppState while render thread is active.
        try:
            render_state = copy.deepcopy(self.state)
        except Exception:
            render_state = copy.copy(self.state)

        self.render_started.emit()

        self.render_thread = QThread()
        self.render_worker = RenderWorker(self.render_service, render_state)
        self.render_worker.moveToThread(self.render_thread)

        self.render_thread.started.connect(self.render_worker.run)

        self.render_worker.finished.connect(self._on_render_finished)
        self.render_worker.failed.connect(self._on_render_failed)

        self.render_worker.finished.connect(self.render_thread.quit)
        self.render_worker.failed.connect(self.render_thread.quit)

        self.render_worker.finished.connect(self.render_worker.deleteLater)
        self.render_worker.failed.connect(self.render_worker.deleteLater)

        self.render_thread.finished.connect(self._cleanup_render_thread)
        self.render_thread.finished.connect(self.render_thread.deleteLater)

        self.render_thread.start()

    def _on_render_finished(self, outputs):
        self._render_running = False
        self._render_suspended = False
        self.render_finished.emit(outputs)

    def _on_render_failed(self, error):
        self._render_running = False
        self._render_suspended = False

        if error == "__CANCELLED__":
            self.render_cancelled.emit()
        else:
            self.render_failed.emit(error)

    def _cleanup_render_thread(self):
        self.render_thread = None
        self.render_worker = None

    def suspend_render(self):
        if not self._render_running:
            return

        self._render_suspended = True
        self.render_service.request_suspend()
        self.render_suspended.emit()

    def resume_render(self):
        if not self._render_running:
            return

        self._render_suspended = False
        self.render_service.request_resume()
        self.render_resumed.emit()

    def cancel_render(self):
        self.render_service.request_cancel()

        # Important: if render is paused, wake it so cancel can finish.
        if self._render_suspended:
            self.render_service.request_resume()

    # ── Preview ──
    def request_preview(self):
        self.preview_requested.emit()

    def open_preview_sources(self):
        if not self.state.input_video_path:
            self.preview_failed.emit("No input video selected.")
            return 0

        if not self.state.depth_map_path:
            self.preview_failed.emit("No depth map selected.")
            return 0

        try:
            total = self.preview_service.open_sources(
                self.state.input_video_path,
                self.state.depth_map_path,
            )
            return total
        except Exception as e:
            self.preview_failed.emit(str(e))
            return 0

    def update_preview(self):
        try:
            result = self.preview_service.generate_preview(
                state=self.state,
                frame_idx=self.state.preview_frame_index,
                preview_mode=self.state.preview_mode,
                ipd_enabled=self.state.ipd_enabled,
                ipd_scale=self.state.ipd_scale,
            )
            self.preview_updated.emit(result)
        except Exception as e:
            self.preview_failed.emit(str(e))

    # ── Depth Processing ──
    def start_depth_processing(self):
        if self.depth_thread is not None and self.depth_thread.isRunning():
            self.depth_failed.emit("Depth processing is already running.")
            return

        self._depth_running = True
        self._depth_suspended = False

        self.depth_service.set_progress_callback(
            lambda payload: self.depth_progress_updated.emit(payload)
        )

        self.depth_started.emit()

        self.depth_thread = QThread()
        self.depth_worker = DepthWorker(self.depth_service, self.depth_state)
        self.depth_worker.moveToThread(self.depth_thread)

        self.depth_thread.started.connect(self.depth_worker.run)

        self.depth_worker.finished.connect(self._on_depth_finished)
        self.depth_worker.failed.connect(self._on_depth_failed)

        self.depth_worker.finished.connect(self.depth_thread.quit)
        self.depth_worker.failed.connect(self.depth_thread.quit)

        self.depth_worker.finished.connect(self.depth_worker.deleteLater)
        self.depth_worker.failed.connect(self.depth_worker.deleteLater)

        self.depth_thread.finished.connect(self._cleanup_depth_thread)
        self.depth_thread.finished.connect(self.depth_thread.deleteLater)

        self.depth_thread.start()

    def _on_depth_finished(self, output_path):
        self._depth_running = False
        self._depth_suspended = False
        self.depth_finished.emit(output_path)

    def _on_depth_failed(self, error):
        self._depth_running = False
        self._depth_suspended = False

        if error == "__CANCELLED__":
            self.depth_cancelled.emit()
        else:
            self.depth_failed.emit(error)

    def _cleanup_depth_thread(self):
        self.depth_thread = None
        self.depth_worker = None

    def suspend_depth(self):
        if not self._depth_running:
            return

        self._depth_suspended = True
        self.depth_service.request_suspend()
        self.depth_suspended.emit()

    def resume_depth(self):
        if not self._depth_running:
            return

        self._depth_suspended = False
        self.depth_service.request_resume()
        self.depth_resumed.emit()

    def cancel_depth(self):
        self.depth_service.request_cancel()

        # Important: if depth processing is paused, wake it so cancel can finish.
        if self._depth_suspended:
            self.depth_service.request_resume()
            
    def start_depth_image(self):
        import threading
        import tkinter as tk
        from core.render_depth import process_image

        if not self.depth_state.input_video_path:
            self.depth_failed.emit("No input image selected.")
            return
        if not self.depth_state.output_dir:
            self.depth_failed.emit("No output directory selected.")
            return

        self.depth_started.emit()

        def _run():
            try:
                root = tk.Tk()
                root.withdraw()
                print(f"[IMAGE DEBUG] invert_depth={self.depth_state.invert_depth}")

                process_image(
                    file_path=self.depth_state.input_video_path,
                    colormap_var=_DepthVar(self.depth_state.colormap),
                    invert_var=_DepthVar(self.depth_state.invert_depth),
                    output_dir_var=_DepthVar(self.depth_state.output_dir),
                    inference_res_var=_DepthVar(self.depth_state.inference_resolution),
                    input_label=None,
                    output_label=None,
                    status_label=None,
                    progress_bar=None,
                    folder=True,
                )
                root.destroy()
                self.depth_finished.emit(self.depth_state.output_dir)
            except Exception as e:
                self.depth_failed.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    def start_depth_image_folder(self):
        import threading
        import tkinter as tk
        from core.render_depth import process_images_in_folder

        if not self.depth_state.input_video_path:
            self.depth_failed.emit("No input folder selected.")
            return
        if not self.depth_state.output_dir:
            self.depth_failed.emit("No output directory selected.")
            return

        self.depth_started.emit()

        def _run():
            try:
                root = tk.Tk()
                root.withdraw()

                process_images_in_folder(
                    folder_path=self.depth_state.input_video_path,
                    batch_size_widget=_DepthVar(str(self.depth_state.batch_size)),
                    output_dir_var=_DepthVar(self.depth_state.output_dir),
                    inference_res_var=_DepthVar(self.depth_state.inference_resolution),
                    status_label=None,
                    progress_bar=None,
                    root=root,
                    invert_var=_DepthVar(self.depth_state.invert_depth),
                )
                root.destroy()
                self.depth_finished.emit(self.depth_state.output_dir)
            except Exception as e:
                self.depth_failed.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    def load_depth_model(self, model_key: str):
        import threading
        from core.render_depth import update_pipeline, load_supported_models

        supported = load_supported_models()
        checkpoint = supported.get(model_key)

        if checkpoint is None:
            return

        class _Var:
            def __init__(self, val):
                self._val = val
            def get(self): return self._val
            def set(self, v): self._val = v
            def after(self, *args, **kwargs): pass
            def config(self, *args, **kwargs): pass
            def winfo_toplevel(self): return self

        def _warmup():
            try:
                update_pipeline(
                    _Var(model_key),
                    _Var(""),
                    _Var(self.depth_state.inference_resolution),
                    _Var(self.depth_state.offload_mode),
                    _Var(str(self.depth_state.inference_steps)),
                    _Var(self.depth_state.use_fp16),
                )
                print(f"Depth model load initiated: {model_key}")
            except Exception as e:
                print(f"Depth model warmup failed: {e}")

        threading.Thread(target=_warmup, daemon=True).start()

    # ── Presets ──
    def list_presets(self) -> list[str]:
        return self.preset_service.list_presets()

    def save_preset(self, filename: str):
        return self.preset_service.save_preset(self.state, filename)

    def load_preset(self, path: str):
        config = self.preset_service.load_preset_file(path)
        ignored = self.preset_service.apply_preset_to_state(self.state, config)

        for key, value in self.state.to_dict().items():
            self.state_changed.emit(key, value)

        self.settings_loaded.emit()
        self.save_settings()

        return ignored

    # ── Live 3D ──
    def start_live_3d(self, args):
        self.live_service.start(args)

    def stop_live_3d(self):
        self.live_service.stop()

    def is_live_3d_running(self) -> bool:
        return self.live_service.is_running()
                
    # ── Language ──

    @property
    def translations(self) -> dict:
        return self.language_service.translations

    @property
    def current_language(self) -> str:
        return self.language_service.current_language

    def t(self, key: str, default: str = "") -> str:
        return self.language_service.t(key, default)

    def set_language(self, code: str):
        self.language_service.set_language(code)

    def available_languages(self) -> dict:
        return self.language_service.available_languages()

    def _on_language_changed(self, code: str):
        self.language_changed.emit(code)
