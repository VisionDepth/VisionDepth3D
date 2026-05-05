import os
import sys
import time
import webbrowser

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QIcon
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

        self.controller.depth_progress_updated.connect(self._on_depth_progress)

    # ── Debug toggle ──
    def _toggle_debug(self, checked):
        if checked:
            self._debug_active = True
            sys.stdout = self._debug_emitter
            sys.stderr = self._debug_emitter
            self.queue.add_message(self._t("Debug output enabled"))
        else:
            self._debug_active = False
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
        self.queue.add_message(self._t("Render started..."))

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
        QMessageBox.critical(self, self._t("Render Failed"), error)

    def _on_render_suspended(self):
        self.queue.set_status(self._t("Render suspended."))
        self.queue.add_message(self._t("Render suspended."))

    def _on_render_resumed(self):
        self.queue.set_status(self._t("Render resumed."))
        self.queue.add_message(self._t("Render resumed."))

    def _on_render_cancelled(self):
        self.queue.set_status(self._t("Render cancelled."))
        self.queue.add_message(self._t("Render cancelled."))

    # ── Progress ──
    def _on_blend_progress(self, payload):
        progress = payload.get("progress", 0.0)
        status_text = payload.get("status_text", "")
        self.queue.set_progress(progress)
        if status_text:
            self.queue.set_status(status_text)

    def _on_render_progress(self, payload):
        stats = self._get_system_stats()
        progress = payload.get("progress", 0.0)
        status_text = payload.get("status_text", "")

        gpu_text = f"{stats['gpu']:.0f}%" if stats["gpu"] is not None else "N/A"
        vram_text = f"{stats['vram']:.0f}%" if stats["vram"] is not None else "N/A"
        telemetry = (
            f"CPU: {stats['cpu']:.0f}% | RAM: {stats['ram']:.0f}% | "
            f"GPU: {gpu_text} | VRAM: {vram_text}"
        )

        self.queue.set_progress(progress)
        if status_text:
            self.queue.set_status(status_text)
        else:
            self.queue.set_status(f"{self._t('Progress:')} {progress:.1f}%")
        self.queue.set_telemetry(telemetry)

    def _on_depth_progress(self, payload):
        progress = payload.get("progress", 0.0)
        status_text = payload.get("status_text", "")
        now = time.monotonic()

        if not hasattr(self, '_last_stat_poll') or (now - self._last_stat_poll) > 0.5:
            self._last_stat_poll = now
            stats = self._get_system_stats()
            gpu_text = f"{stats['gpu']:.0f}%" if stats["gpu"] is not None else "N/A"
            vram_text = f"{stats['vram']:.0f}%" if stats["vram"] is not None else "N/A"
            telemetry = (
                f"CPU: {stats['cpu']:.0f}% | RAM: {stats['ram']:.0f}% | "
                f"GPU: {gpu_text} | VRAM: {vram_text}"
            )
            self.queue.set_telemetry(telemetry)

        self.queue.set_progress(progress)
        if status_text:
            self.queue.set_status(status_text)

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

        # File actions
        self._set_action_text(self.save_preset_action, "Save Preset As…")
        self._set_action_text(self.load_preset_action, "Load Preset…")
        self._set_action_text(self.select_video_action, "Select Input Video")
        self._set_action_text(self.select_depth_action, "Select Depth Map")
        self._set_action_text(self.exit_action, "Exit")

        # Help actions
        self._set_action_text(self.about_action, "About VisionDepth3D")
        self._set_action_text(self.website_action, "Official Website")
        self._set_action_text(self.github_action, "GitHub Repository")
        self._set_action_text(self.docs_action, "Documentation / Method")
        self._set_action_text(self.issues_action, "Report a Bug")
        self._set_action_text(self.gpu_diag_action, "GPU Diagnostics")

        # Queue dock
        if hasattr(self.queue, "refresh_labels"):
            self.queue.refresh_labels()

    def _build_menu_bar(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background: #0f141a;
                color: #e8ecf1;
                border-bottom: 1px solid #28303a;
                padding: 2px 8px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                border-radius: 6px;
            }
            QMenuBar::item:selected {
                background: #1a2230;
            }
            QMenu {
                background: #0f141a;
                color: #e8ecf1;
                border: 1px solid #28303a;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 32px 8px 16px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #1a2230;
                color: #4dd0e1;
            }
            QMenu::separator {
                height: 1px;
                background: #28303a;
                margin: 4px 8px;
            }
        """)

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
                f"{self._t('VisionDepth3D v4.0')}\n\n"
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
            import torch
            info = f"PyTorch: {torch.__version__}\n"
            info += f"CUDA Available: {torch.cuda.is_available()}\n"
            if torch.cuda.is_available():
                info += f"CUDA Version: {torch.version.cuda}\n"
                info += f"GPU: {torch.cuda.get_device_name(0)}\n"
                info += f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n"
            info += f"\nDevice: {self.gpu_label.text()}"
            QMessageBox.information(self, self._t("GPU Diagnostics"), info)
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

    # ── Styles ──
    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background: #111418; }
            QWidget { color: #e8ecf1; font-family: Segoe UI; font-size: 10pt; }
            #TopBar { background: #0f141a; border-bottom: 1px solid #28303a; }
            #AppTitle { font-size: 12pt; font-weight: 600; padding-right: 8px; }
            #GpuLabel { color: #4dd0e1; font-size: 9pt; padding-left: 4px; padding-right: 8px; }
            QPushButton#TopNavButton {
                background: #1a2230; border: 1px solid #2f3947;
                border-radius: 10px; padding: 8px 14px;
            }
            QPushButton#TopNavButton:hover { background: #283142; }
            QPushButton#TopNavButton:checked {
                background: #243246; border: 1px solid #3b4d63;
            }
            #PreviewFrame {
                background: #0d1117; border: 1px solid #28303a; border-radius: 14px;
            }
            QGroupBox {
                background: #171c23; border: 1px solid #28303a;
                border-radius: 14px; margin-top: 12px; padding-top: 10px; font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
            QPushButton {
                background: #202734; border: 1px solid #2f3947;
                border-radius: 10px; padding: 8px 12px;
            }
            QPushButton:hover { background: #283142; }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
                background: #0f141a; border: 1px solid #2a3440;
                border-radius: 10px; padding: 8px;
            }
        """)