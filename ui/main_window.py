import os
import sys
import time
import webbrowser

from PySide6.QtCore import Qt, QObject, Signal, QUrl
from PySide6.QtGui import QIcon, QPalette, QActionGroup, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QMessageBox,
    QLabel,
    QPushButton,
    QSplitter,
)

from ui.pages.depth_generation_page import DepthGenerationPage
from ui.pages.depth_blender_page import DepthBlenderPage
from ui.pages.fps_upscale_page import FpsUpscalePage
from ui.pages.live_3d_page import Live3DPage
from services.theme_service import ThemeService
from ui.dialogs.theme_creator_dialog import ThemeCreatorDialog
from core.debug_flags import set_debug_enabled

import psutil


def resource_path(relative_path):
    """
    Works in normal Python mode and PyInstaller bundled mode.
    """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from ui.queue_dock import JobQueueDock
from ui.pages.stereo_generator_page import StereoGeneratorPage


class _StreamEmitter(QObject):
    text_received = Signal(str)

    def __init__(self):
        super().__init__()
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            self.buffer = lines.pop()
            for line in lines:
                if line.strip():
                    self.text_received.emit(line)

    def flush(self):
        if self.buffer.strip():
            self.text_received.emit(self.buffer)
            self.buffer = ""

    def isatty(self):
        return False  # Not a terminal — prevents libraries from trying TTY-specific features


class PlaceholderPage(QWidget):
    def __init__(self, title: str):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)


class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.nav_buttons = {}

        self.theme_service = ThemeService()
        self.current_theme_id = getattr(self.controller.state, "selected_theme", "dark")

        # App/window icon
        icon_path = resource_path("resources/icons/logo.ico")
        app_icon = QIcon(icon_path)

        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(app_icon)

        self.setWindowIcon(app_icon)

        self._build_menu_bar()

        self.setWindowTitle("VisionDepth3D")
        self.setMinimumSize(900, 600)

        # Debug stream capture
        self._debug_emitter = _StreamEmitter()
        self._debug_emitter.text_received.connect(self._on_debug_text)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._debug_active = False

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Top bar ---
        top_bar = QWidget()
        top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(14, 10, 14, 10)
        top_layout.setSpacing(10)

        self.app_title = QLabel("VisionDepth3D")
        self.app_title.setObjectName("AppTitle")

        self.gpu_label = QLabel("")
        self.gpu_label.setObjectName("GpuLabel")

        self.btn_stereo = QPushButton(self._t("3D Generator"))
        self.btn_depth = QPushButton(self._t("Depth Engine"))
        self.btn_blend = QPushButton(self._t("Depth Blender"))
        self.btn_frame = QPushButton(self._t("FPS/Upscale"))
        self.btn_live = QPushButton(self._t("Live 3D"))
        self.debug_btn = QPushButton(self._t("Debug"))
        self.debug_btn.setCheckable(True)
        self.debug_btn.setObjectName("TopNavButton")

        for key, btn in [
            ("stereo", self.btn_stereo),
            ("depth", self.btn_depth),
            ("blend", self.btn_blend),
            ("frame", self.btn_frame),
            ("live", self.btn_live),
        ]:
            btn.setCheckable(True)
            btn.setObjectName("TopNavButton")
            self.nav_buttons[key] = btn

        top_layout.addWidget(self.app_title)
        top_layout.addSpacing(18)
        top_layout.addWidget(self.btn_stereo)
        top_layout.addWidget(self.btn_depth)
        top_layout.addWidget(self.btn_blend)
        top_layout.addWidget(self.btn_frame)
        top_layout.addWidget(self.btn_live)
        top_layout.addWidget(self.debug_btn)
        top_layout.addSpacing(150)
        top_layout.addWidget(self.gpu_label)
        top_layout.addStretch()

        root.addWidget(top_bar)

        # --- Main page area + bottom queue in resizable splitter ---
        self.content_splitter = QSplitter(Qt.Vertical)
        self.content_splitter.setChildrenCollapsible(False)

        self.pages = QStackedWidget()
        self.queue = JobQueueDock()
        if hasattr(self.queue, "set_translator"):
            self.queue.set_translator(self._t)

        self.content_splitter.addWidget(self.pages)
        self.content_splitter.addWidget(self.queue)

        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 0)
        self.content_splitter.setSizes([780, 140])

        root.addWidget(self.content_splitter, 1)

        self.stereo_page = StereoGeneratorPage(controller)
        self.depth_page = DepthGenerationPage(controller)
        self.blend_page = DepthBlenderPage(controller)
        self.blend_page.progress_updated.connect(self._on_blend_progress)
        self.frame_page = FpsUpscalePage(controller)
        self.frame_page.progress_updated.connect(self._on_blend_progress)
        self.live_page = Live3DPage(controller)

        self.page_map = {
            "stereo": self.stereo_page,
            "depth": self.depth_page,
            "blend": self.blend_page,
            "frame": self.frame_page,
            "live": self.live_page,
        }

        for page in self.page_map.values():
            self.pages.addWidget(page)

        self._bind_events()
        self._switch_page("stereo")
        self._apply_styles()
        self._apply_page_themes()
        self._detect_gpu()
        self.refresh_shell_labels()

    def _bind_events(self):
        self.btn_stereo.clicked.connect(lambda: self._switch_page("stereo"))
        self.btn_depth.clicked.connect(lambda: self._switch_page("depth"))
        self.btn_blend.clicked.connect(lambda: self._switch_page("blend"))
        self.btn_frame.clicked.connect(lambda: self._switch_page("frame"))
        self.btn_live.clicked.connect(lambda: self._switch_page("live"))
        self.debug_btn.toggled.connect(self._toggle_debug)

        self.controller.render_started.connect(self._on_render_started)
        self.controller.render_finished.connect(self._on_render_finished)
        self.controller.render_failed.connect(self._on_render_failed)

        self.controller.render_suspended.connect(self._on_render_suspended)
        self.controller.render_resumed.connect(self._on_render_resumed)
        self.controller.render_cancelled.connect(self._on_render_cancelled)
        self.controller.render_progress.connect(self._on_render_progress)

        self.controller.depth_started.connect(self._on_depth_started)
        self.controller.depth_finished.connect(self._on_depth_finished)
        self.controller.depth_failed.connect(self._on_depth_failed)
        self.controller.depth_cancelled.connect(self._on_depth_cancelled)
        self.controller.depth_suspended.connect(self._on_depth_suspended)
        self.controller.depth_resumed.connect(self._on_depth_resumed)
        self.controller.depth_progress_updated.connect(self._on_depth_progress)

    # ── Debug toggle ──
    def _toggle_debug(self, checked):
        set_debug_enabled(checked)

        self._debug_active = checked
        self.queue.set_log_visible(checked)

        if checked:
            sys.stdout = self._debug_emitter
            sys.stderr = self._debug_emitter
            self.queue.add_message(self._t("Debug output enabled"))
        else:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self.queue.add_message(self._t("Debug output disabled"))

    def _on_debug_text(self, text):
        if self._debug_active:
            self.queue.add_message(text)

    # ── Page switching ──
    def _switch_page(self, key: str):
        page = self.page_map.get(key)
        if not page:
            return
        self.pages.setCurrentWidget(page)
        for btn_key, btn in self.nav_buttons.items():
            btn.setChecked(btn_key == key)
        
        # Free GPU memory when switching away from heavy tabs
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # ── Render callbacks ──
    def _on_render_started(self):
        self.queue.reset_progress()
        self.queue.set_status(self._t("Render started..."))
        self.queue.set_telemetry("")

    def _on_render_finished(self, outputs: list):
        self.queue.set_progress(100)
        self.queue.set_status(self._t("Render finished."))
        self.queue.add_message(self._t("Render finished."))

        QMessageBox.information(
            self,
            self._t("Render Complete"),
            self._t("Created files:") + "\n" + "\n".join(outputs),
        )

    def _on_render_failed(self, error: str):
        self.queue.set_status(f"{self._t('Render failed:')} {error}")
        self.queue.add_message(f"{self._t('Render failed:')} {error}")

        QMessageBox.critical(
            self,
            self._t("Render Failed"),
            error,
        )

    def _on_render_suspended(self):
        self.queue.set_status(self._t("Render suspended."))

    def _on_render_resumed(self):
        self.queue.set_status(self._t("Render resumed."))

    def _on_render_cancelled(self):
        self.queue.set_status(self._t("Render cancelled."))
        self.queue.add_message(self._t("Render cancelled."))

    # ── Depth callbacks ──
    def _on_depth_started(self):
        self.queue.reset_progress()
        self.queue.set_status(self._t("Depth processing started..."))
        self.queue.set_telemetry("")
        self.queue.add_message(self._t("Depth processing started..."))

    def _on_depth_finished(self, output_path: str):
        self.queue.set_progress(100)
        self.queue.set_status(self._t("Depth processing finished."))
        self.queue.add_message(f"{self._t('Depth output:')} {output_path}")

    def _on_depth_failed(self, error: str):
        self.queue.set_status(f"{self._t('Depth failed:')} {error}")
        self.queue.add_message(f"{self._t('Depth failed:')} {error}")

    def _on_depth_cancelled(self):
        self.queue.set_status(self._t("Depth cancelled."))
        self.queue.add_message(self._t("Depth cancelled."))

    def _on_depth_suspended(self):
        self.queue.set_status(self._t("Depth suspended."))
        self.queue.add_message(self._t("Depth suspended."))

    def _on_depth_resumed(self):
        self.queue.set_status(self._t("Depth resumed."))
        self.queue.add_message(self._t("Depth resumed."))

    def _get_system_stats_text(self):
        now = time.monotonic()

        if (
            hasattr(self, "_last_queue_stat_poll")
            and hasattr(self, "_last_queue_stats_text")
            and (now - self._last_queue_stat_poll) <= 0.5
        ):
            return self._last_queue_stats_text

        self._last_queue_stat_poll = now

        stats = self._get_system_stats()
        gpu_text = f"{stats['gpu']:.0f}%" if stats["gpu"] is not None else "N/A"
        vram_text = f"{stats['vram']:.0f}%" if stats["vram"] is not None else "N/A"

        self._last_queue_stats_text = (
            f"CPU: {stats['cpu']:.0f}% | RAM: {stats['ram']:.0f}% | "
            f"GPU: {gpu_text} | VRAM: {vram_text}"
        )

        return self._last_queue_stats_text


    def _update_queue_progress_line(self, payload, default_rate_label="FPS"):
        progress = float(payload.get("progress", 0.0) or 0.0)
        progress = max(0.0, min(100.0, progress))

        elapsed = payload.get("elapsed", None)
        eta = payload.get("eta", None)
        fps_like = payload.get("fps_like", None)
        rate_label = payload.get("rate_label", default_rate_label)

        line_parts = [f"{progress:.2f}%"]

        if fps_like is not None:
            try:
                line_parts.append(f"{rate_label}: {float(fps_like):.2f}")
            except Exception:
                pass

        if elapsed is not None:
            line_parts.append(f"Elapsed: {self._format_seconds(elapsed)}")

        if eta is not None:
            line_parts.append(f"ETA: {self._format_seconds(eta)}")

        self.queue.set_progress(progress)
        self.queue.set_status(" | ".join(line_parts))
        self.queue.set_telemetry(self._get_system_stats_text())

    # ── Progress ──
    def _on_blend_progress(self, payload):
        self._update_queue_progress_line(payload, default_rate_label="FPS")
        
    def _on_render_progress(self, payload):
        self._update_queue_progress_line(payload, default_rate_label="FPS")

    def _on_depth_progress(self, payload):
        payload = dict(payload or {})

        # Make depth render use the same unified queue display as
        # 3D render, FPS/Upscale, frame extraction, scene detection, etc.
        payload.setdefault("rate_label", "FPS")

        self._update_queue_progress_line(payload, default_rate_label="FPS")

    def _format_seconds(self, seconds):
        seconds = max(0, int(seconds or 0))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ── System stats ──
    def _get_system_stats(self):
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        gpu = None
        vram = None
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu = util.gpu
                vram = (mem.used / mem.total) * 100 if mem.total else 0
            except Exception:
                pass
        return {"cpu": cpu, "ram": ram, "gpu": gpu, "vram": vram}

    def _detect_gpu(self):
        gpu_name = "CPU"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    gpu_name += " (ROCm)"
        except Exception:
            pass
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            gpu_name = name
        except Exception:
            pass
        self.gpu_label.setText(f"\U0001f5a5 {gpu_name}")


    def _t(self, key: str) -> str:
        translator = getattr(self.controller, "t", None)
        if callable(translator):
            return translator(key)
        return key

    def _set_action_text(self, action, key: str):
        action.setText(self._t(key))

    def refresh_shell_labels(self):
        # Top navigation
        self.btn_stereo.setText(self._t("3D Generator"))
        self.btn_depth.setText(self._t("Depth Engine"))
        self.btn_blend.setText(self._t("Depth Blender"))
        self.btn_frame.setText(self._t("FPS/Upscale"))
        self.btn_live.setText(self._t("Live 3D"))
        self.debug_btn.setText(self._t("Debug"))

        # Menus
        self.file_menu.setTitle(self._t("File"))
        self.help_menu.setTitle(self._t("Help"))
        self.lang_menu.setTitle(self._t("Language"))
        
        if hasattr(self, "theme_menu"):
            self.theme_menu.setTitle(self._t("Themes"))
            
        # Theme submenu actions
        if hasattr(self, "reload_themes_action"):
            self.reload_themes_action.setText(self._t("Reload Themes"))

        if hasattr(self, "create_theme_action"):
            self.create_theme_action.setText(self._t("Create Theme..."))

        # Built-in theme names
        if hasattr(self, "_theme_actions"):
            themes = self.theme_service.available_themes()

            for theme_id, action in self._theme_actions.items():
                theme = themes.get(theme_id, {})
                theme_name = theme.get("name", theme_id.title())
                action.setText(self._t(theme_name))

        # File actions
        self._set_action_text(self.save_preset_action, "Save Preset As…")
        self._set_action_text(self.load_preset_action, "Load Preset…")
        self._set_action_text(self.select_video_action, "Select Input Video")
        self._set_action_text(self.select_depth_action, "Select Depth Map")
        self._set_action_text(self.exit_action, "Exit")

        # Help actions
        self._set_action_text(self.about_action, "About VisionDepth3D")
        self._set_action_text(self.user_guide_action, "User Guide")
        self._set_action_text(self.website_action, "Official Website")
        self._set_action_text(self.github_action, "GitHub Repository")
        self._set_action_text(self.docs_action, "Documentation / Method")
        self._set_action_text(self.issues_action, "Report a Bug")
        self._set_action_text(self.gpu_diag_action, "GPU Diagnostics")

        # Queue dock
        if hasattr(self.queue, "refresh_labels"):
            self.queue.refresh_labels()

    def _create_theme_dialog(self):
        user_themes_dir = getattr(
            self.theme_service,
            "user_themes_dir",
            getattr(self.theme_service, "themes_dir", None),
        )

        if not user_themes_dir:
            QMessageBox.warning(
                self,
                self._t("Themes"),
                self._t("Could not locate the user themes folder."),
            )
            return

        dialog = ThemeCreatorDialog(
            themes_dir=user_themes_dir,
            parent=self,
            base_theme=self._theme(),
        )

        if dialog.exec() != ThemeCreatorDialog.Accepted:
            return

        self.theme_service.reload()
        self._rebuild_theme_menu()

        theme_id = getattr(dialog, "saved_theme_id", None)

        if theme_id:
            self._set_theme(theme_id)

        QMessageBox.information(
            self,
            self._t("Theme Created"),
            self._t("Theme created and applied successfully."),
        )

    def _rebuild_theme_menu(self):
        if not hasattr(self, "theme_menu"):
            return

        self.theme_menu.clear()
        self.theme_service.reload()

        self._theme_actions = {}
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)

        for theme_id, theme in self.theme_service.available_themes().items():
            theme_name = theme.get("name", theme_id.title())

            # Translate built-in theme names if a translation exists.
            # Custom user theme names will stay unchanged if no translation exists.
            action = self.theme_menu.addAction(self._t(theme_name))

            action.setCheckable(True)
            action.setChecked(theme_id == self.current_theme_id)
            action.triggered.connect(lambda checked=False, tid=theme_id: self._set_theme(tid))

            self._theme_group.addAction(action)
            self._theme_actions[theme_id] = action

        self.theme_menu.addSeparator()

        self.reload_themes_action = self.theme_menu.addAction(self._t("Reload Themes"))
        self.reload_themes_action.triggered.connect(self._rebuild_theme_menu)

        self.create_theme_action = self.theme_menu.addAction(self._t("Create Theme..."))
        self.create_theme_action.triggered.connect(self._create_theme_dialog)
        
    def _build_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        self.file_menu = menubar.addMenu(self._t("File"))

        self.save_preset_action = self.file_menu.addAction(self._t("Save Preset As…"))
        self.save_preset_action.triggered.connect(
            self.controller.save_preset_dialog
            if hasattr(self.controller, "save_preset_dialog")
            else lambda: None
        )

        self.load_preset_action = self.file_menu.addAction(self._t("Load Preset…"))
        self.load_preset_action.triggered.connect(lambda: None)

        self.theme_menu = self.file_menu.addMenu(self._t("Themes"))
        self._rebuild_theme_menu()

        self.file_menu.addSeparator()

        self.select_video_action = self.file_menu.addAction(self._t("Select Input Video"))
        self.select_video_action.triggered.connect(lambda: self._switch_page("stereo"))

        self.select_depth_action = self.file_menu.addAction(self._t("Select Depth Map"))
        self.select_depth_action.triggered.connect(lambda: self._switch_page("stereo"))

        self.file_menu.addSeparator()

        self.exit_action = self.file_menu.addAction(self._t("Exit"))
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)

        # Help menu
        self.help_menu = menubar.addMenu(self._t("Help"))

        self.about_action = self.help_menu.addAction(self._t("About VisionDepth3D"))
        self.about_action.triggered.connect(self._show_about_dialog)

        self.help_menu.addSeparator()

        self.user_guide_action = self.help_menu.addAction(self._t("User Guide"))
        self.user_guide_action.triggered.connect(
            lambda: webbrowser.open(
                "https://github.com/VisionDepth/VisionDepth3D/blob/Main-Stable/UserGuide.md"
            )
        )

        self.website_action = self.help_menu.addAction(self._t("Official Website"))
        self.website_action.triggered.connect(
            lambda: webbrowser.open("https://visiondepth.github.io/VisionDepth3D/")
        )

        self.github_action = self.help_menu.addAction(self._t("GitHub Repository"))
        self.github_action.triggered.connect(
            lambda: webbrowser.open("https://github.com/VisionDepth/VisionDepth3D")
        )

        self.docs_action = self.help_menu.addAction(self._t("Documentation / Method"))
        self.docs_action.triggered.connect(
            lambda: webbrowser.open(
                "https://github.com/VisionDepth/VisionDepth3D/blob/Main-Stable/VisionDepth3D_Method.md"
            )
        )

        self.issues_action = self.help_menu.addAction(self._t("Report a Bug"))
        self.issues_action.triggered.connect(
            lambda: webbrowser.open("https://github.com/VisionDepth/VisionDepth3D/issues")
        )

        self.help_menu.addSeparator()

        self.gpu_diag_action = self.help_menu.addAction(self._t("GPU Diagnostics"))
        self.gpu_diag_action.triggered.connect(self._run_gpu_diagnostics)

        # Language menu
        self.lang_menu = menubar.addMenu(self._t("Language"))
        self._lang_actions = {}

        for code, name in self.controller.available_languages().items():
            action = self.lang_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(code == "en")
            action.triggered.connect(lambda checked, c=code: self.controller.set_language(c))
            self._lang_actions[code] = action

        self.controller.language_changed.connect(self._on_language_changed)

    def _show_about_dialog(self):
        QMessageBox.about(
            self,
            self._t("About VisionDepth3D"),
            (
                f"{self._t('VisionDepth3D v4.1.1')}\n\n"
                f"{self._t('A hybrid 2D-to-3D conversion suite for cinema and VR.')}\n\n"
                f"{self._t('Features:')}\n"
                f" • {self._t('Depth map blending (multi-model)')}\n"
                f" • {self._t('Depth-weighted parallax shifting')}\n"
                f" • {self._t('Scene-aware stereo rendering')}\n"
                f" • {self._t('CUDA / DirectML / ROCm acceleration')}\n"
                f" • {self._t('Real-time preview & batch processing')}\n\n"
                "Website: https://visiondepth.github.io/VisionDepth3D/\n"
                "GitHub: https://github.com/VisionDepth/VisionDepth3D\n"
                "© 2026 VisionDepth3D"
            )
        )

    def _run_gpu_diagnostics(self):
        try:
            from gpu_diag import gpu_diagnostics
        except Exception:
            try:
                from core.gpu_diag import gpu_diagnostics
            except Exception as e:
                QMessageBox.warning(
                    self,
                    self._t("GPU Diagnostics"),
                    self._t("Could not load GPU diagnostics:") + f"\n{e}"
                )
                return

        try:
            report = gpu_diagnostics(return_text=True)

            QMessageBox.information(
                self,
                self._t("GPU Diagnostics"),
                report,
            )

        except Exception as e:
            QMessageBox.warning(
                self,
                self._t("GPU Diagnostics"),
                self._t("Could not detect GPU:") + f"\n{e}"
            )
    def _on_language_changed(self, code: str):
        # Update checked state in the language menu
        for c, action in self._lang_actions.items():
            action.setChecked(c == code)

        # Refresh shell/top bar/menu/queue labels
        self.refresh_shell_labels()

        # Refresh all page labels
        for page in self.page_map.values():
            if hasattr(page, 'refresh_labels'):
                page.refresh_labels()

    def _system_accent_hex(self) -> str:
        app = QApplication.instance()

        if app is not None:
            color = app.palette().color(QPalette.ColorRole.Highlight)
            if color.isValid():
                return color.name()

        return "#3b82f6"

    def _theme(self):
        return self.theme_service.get_theme(self.current_theme_id)

    def _theme_colors(self):
        return self._theme().get("colors", {})

    def _set_theme(self, theme_id: str):
        self.current_theme_id = theme_id

        if hasattr(self.controller.state, "selected_theme"):
            self.controller.state.selected_theme = theme_id
            self.controller.save_settings()

        for tid, action in getattr(self, "_theme_actions", {}).items():
            action.setChecked(tid == theme_id)

        self._apply_styles()
        self._apply_page_themes()

    def _apply_page_themes(self):
        theme = self._theme()

        if hasattr(self.queue, "apply_theme"):
            self.queue.apply_theme(theme)

        for page in getattr(self, "page_map", {}).values():
            if hasattr(page, "apply_theme"):
                page.apply_theme(theme)

    # ── Styles ──
    def _apply_styles(self):
        accent = self._system_accent_hex()

        colors = self._theme_colors()

        bg = colors.get("bg", "#0b0f14")
        topbar = colors.get("topbar", "#0f141a")
        panel = colors.get("panel", "#111821")
        panel_2 = colors.get("panel_2", "#0d131b")
        panel_3 = colors.get("panel_3", "#151d29")
        border = colors.get("border", "#263445")
        border_soft = colors.get("border_soft", "#2d3b4f")
        text = colors.get("text", "#e6edf3")
        text_bright = colors.get("text_bright", "#f0f6fc")
        muted = colors.get("muted", "#8b949e")
        accent = colors.get("accent", "#2f81f7")
        accent_text = colors.get("accent_text", "#ffffff")
        danger = colors.get("danger", "#ff7b72")
        danger_bg = colors.get("danger_bg", "#2b1518")
        preview_bg = colors.get("preview_bg", panel_2)

        theme = self._theme()
        stylesheet = theme.get("_qss_template", "")

        if not stylesheet:
            stylesheet = """
                QWidget {
                    background-color: __BG__;
                    color: __TEXT__;
                }

                QMainWindow {
                    background-color: __BG__;
                }

                QWidget#TopBar {
                    background-color: __TOPBAR__;
                    border-bottom: 1px solid __BORDER__;
                }

                QLabel#AppTitle {
                    color: __TEXT_BRIGHT__;
                    font-size: 15px;
                    font-weight: 800;
                }

                QLabel#GpuLabel {
                    color: __ACCENT__;
                }

                QPushButton#TopNavButton {
                    background-color: __PANEL_3__;
                    border: 1px solid __BORDER_SOFT__;
                    border-radius: 9px;
                    padding: 8px 14px;
                    color: __TEXT_BRIGHT__;
                    font-weight: 600;
                }

                QPushButton#TopNavButton:hover {
                    border: 1px solid __ACCENT__;
                    background-color: __PANEL__;
                }

                QPushButton#TopNavButton:checked {
                    background-color: __ACCENT__;
                    border: 1px solid __ACCENT__;
                    color: __ACCENT_TEXT__;
                }

                QMenuBar {
                    background: __TOPBAR__;
                    color: __TEXT__;
                    border-bottom: 1px solid __BORDER__;
                    padding: 2px 8px;
                }

                QMenuBar::item {
                    padding: 6px 12px;
                    border-radius: 6px;
                }

                QMenuBar::item:selected {
                    background: __PANEL_3__;
                }

                QMenu {
                    background: __PANEL__;
                    color: __TEXT__;
                    border: 1px solid __BORDER__;
                    border-radius: 8px;
                    padding: 4px;
                }

                QMenu::item {
                    padding: 8px 32px 8px 16px;
                    border-radius: 4px;
                }

                QMenu::item:selected {
                    background: __PANEL_3__;
                    color: __ACCENT__;
                }

                QMenu::separator {
                    height: 1px;
                    background: __BORDER__;
                    margin: 4px 8px;
                }

                QSplitter::handle {
                    background: __BORDER__;
                }

                QScrollArea {
                    background: __BG__;
                    border: none;
                }

                QProgressBar {
                    background: __PANEL_2__;
                    color: __TEXT__;
                    border: 1px solid __BORDER_SOFT__;
                    border-radius: 8px;
                    text-align: center;
                    padding: 2px;
                }

                QProgressBar::chunk {
                    background: __ACCENT__;
                    border-radius: 6px;
                }

                QListWidget {
                    background: __PANEL_2__;
                    color: __TEXT__;
                    border: 1px solid __BORDER__;
                    border-radius: 8px;
                }
            """

        replacements = {
            "__BG__": bg,
            "__TOPBAR__": topbar,
            "__PANEL__": panel,
            "__PANEL_2__": panel_2,
            "__PANEL_3__": panel_3,
            "__BORDER__": border,
            "__BORDER_SOFT__": border_soft,
            "__TEXT__": text,
            "__TEXT_BRIGHT__": text_bright,
            "__MUTED__": muted,
            "__ACCENT__": accent,
            "__ACCENT_TEXT__": accent_text,
            "__DANGER__": danger,
            "__DANGER_BG__": danger_bg,
            "__PREVIEW_BG__": preview_bg,
        }

        for key, value in replacements.items():
            stylesheet = stylesheet.replace(key, value)

        self.setStyleSheet(stylesheet)
