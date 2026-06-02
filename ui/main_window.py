import os
import sys
import time
import webbrowser

from PySide6.QtCore import Qt, QObject, Signal, QUrl, QRect, QTimer
from PySide6.QtGui import QIcon, QPalette, QActionGroup, QDesktopServices, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QStackedWidget,
    QMessageBox,
    QLabel,
    QPushButton,
    QSplitter,
    QDialog,
    QScrollArea,
    QTextEdit,
    QFrame,
)

from ui.pages.depth_generation_page import DepthGenerationPage
from ui.pages.depth_blender_page import DepthBlenderPage
from ui.pages.fps_upscale_page import FpsUpscalePage
from ui.pages.live_3d_page import Live3DPage
from services.theme_service import ThemeService
from ui.dialogs.theme_creator_dialog import ThemeCreatorDialog
from core.debug_flags import set_debug_enabled

import psutil
import subprocess

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

        self.app_title = QLabel()
        self.app_title.setObjectName("AppLogo")
        self.app_title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.app_title.setStyleSheet("background: transparent; border: none; padding: 0px; margin: 0px;")
        self._app_logo_path = resource_path("resources/icons/NewVD3D-Logo.png")
        self._app_logo_height = 42
        self._set_app_logo()

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
        top_layout.addSpacing(8)
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
        self.content_splitter.setSizes([700, 150])

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

    def bring_to_front_once(self):
        """
        Bring the main VD3D window to the front once after startup.

        This avoids making the app permanently always-on-top.
        It only asks Windows/Qt to raise and activate the main window after loading.
        """
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized)
        self.raise_()
        self.activateWindow()

        # Windows sometimes ignores the first activation if another window had focus.
        # Temporarily set always-on-top, then remove it right away.
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.show()
        self.raise_()
        self.activateWindow()

        def remove_top_hint():
            self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
            self.show()
            self.raise_()
            self.activateWindow()

        QTimer.singleShot(250, remove_top_hint)

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
        self.queue.set_status_key("Render started...")
        self.queue.set_telemetry("")
        
    def _on_render_finished(self, outputs: list):
        self.queue.set_progress(100)
        self.queue.set_status_key("Render finished.")
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
        self.queue.set_status_key("Render suspended.")

    def _on_render_resumed(self):
        self.queue.set_status_key("Render resumed.")

    def _on_render_cancelled(self):
        self.queue.set_status_key("Render cancelled.")
        self.queue.add_message(self._t("Render cancelled."))

    # ── Depth callbacks ──
    def _on_depth_started(self):
        self.queue.reset_progress()
        self.queue.set_status_key("Depth processing started...")
        self.queue.set_telemetry("")
        self.queue.add_message(self._t("Depth processing started..."))

    def _on_depth_finished(self, output_path: str):
        self.queue.set_progress(100)
        self.queue.set_status_key("Depth processing finished.")
        self.queue.add_message(f"{self._t('Depth output:')} {output_path}")

    def _on_depth_failed(self, error: str):
        self.queue.set_status(f"{self._t('Depth failed:')} {error}")
        self.queue.add_message(f"{self._t('Depth failed:')} {error}")

    def _on_depth_cancelled(self):
        self.queue.set_status_key("Depth cancelled.")
        self.queue.add_message(self._t("Depth cancelled."))

    def _on_depth_suspended(self):
        self.queue.set_status_key("Depth suspended.")
        self.queue.add_message(self._t("Depth suspended."))

    def _on_depth_resumed(self):
        self.queue.set_status_key("Depth resumed.")
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
        payload = dict(payload or {})

        progress = float(payload.get("progress", 0.0) or 0.0)
        progress = max(0.0, min(100.0, progress))

        done = payload.get("done", None)
        total = payload.get("total", None)

        elapsed = payload.get("elapsed", None)
        eta = payload.get("eta", None)
        fps_like = payload.get("fps_like", None)
        rate_label = payload.get("rate_label", default_rate_label)

        # Original unified VD3D queue format:
        # 119/9557 | FPS: 3.10 | Elapsed: 00:00:39 | ETA: 00:50:46
        line_parts = []

        if done is not None and total is not None:
            line_parts.append(f"{int(done)}/{int(total)}")
        else:
            line_parts.append(f"{progress:.2f}%")

        if fps_like is not None:
            try:
                line_parts.append(f"{rate_label}: {float(fps_like):.2f}")
            except Exception:
                pass

        if elapsed is not None:
            line_parts.append(f"Elapsed: {self._format_seconds(elapsed)}")

        if eta is not None:
            line_parts.append(f"ETA: {self._format_seconds(eta)}")

        status_line = " | ".join(line_parts)

        # 3D render already sends a real frame-FPS status string:
        # "12.34% | FPS: 7.21 | Elapsed: ... | ETA: ..."
        #
        # Prefer that over the generic fps_like field, because fps_like may be
        # percent-per-second for progress-only updates.
        status_text = str(payload.get("status_text") or "").strip()

        if "FPS:" in status_text and ("Elapsed:" in status_text or "ETA:" in status_text):
            status_line = status_text

        self.queue.set_progress(progress)
        self.queue.set_status(status_line)
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

    def _get_windows_system_gpu_name(self):
        """
        Returns the real Windows display GPU name using PowerShell/CIM.
        This detects AMD / Intel / NVIDIA even when CUDA/NVML is unavailable.
        """
        if sys.platform != "win32":
            return None

        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                (
                    "Get-CimInstance Win32_VideoController | "
                    "Where-Object { $_.Name -and $_.Name -notmatch 'Microsoft Basic Display|Remote Display|Parsec|Virtual' } | "
                    "Select-Object -ExpandProperty Name"
                ),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            names = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]

            if not names:
                return None

            # Prefer real discrete GPUs when multiple adapters are listed.
            priority = ("nvidia", "geforce", "rtx", "gtx", "radeon", "amd", "intel arc")
            for key in priority:
                for name in names:
                    if key in name.lower():
                        return name

            return names[0]

        except Exception:
            return None

    def _get_active_compute_backend_name(self):
        """
        Returns the backend VD3D/PyTorch is likely able to use.
        This is not always the same as the physical system GPU.
        """
        try:
            import torch

            if torch.cuda.is_available():
                if getattr(torch.version, "hip", None) is not None:
                    return "ROCm"
                return "CUDA"

            try:
                import torch_directml
                dml_device = torch_directml.device()
                _ = torch.ones(1).to(dml_device).cpu()
                return "DirectML"
            except Exception:
                pass

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "Metal"

        except Exception:
            pass

        return "CPU"

    def _detect_gpu(self):
        system_gpu_name = None

        # 1. Try Windows system GPU detection first.
        # This catches AMD / Intel / NVIDIA even if VD3D is not using that GPU yet.
        system_gpu_name = self._get_windows_system_gpu_name()

        # 2. Try NVIDIA NVML as a fallback.
        if not system_gpu_name and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                system_gpu_name = str(name)
            except Exception:
                pass

        # 3. Try PyTorch CUDA name as another fallback.
        if not system_gpu_name:
            try:
                import torch
                if torch.cuda.is_available():
                    system_gpu_name = torch.cuda.get_device_name(0)
                    if getattr(torch.version, "hip", None) is not None:
                        system_gpu_name += " (ROCm)"
            except Exception:
                pass

        if not system_gpu_name:
            system_gpu_name = "No dedicated GPU detected"

        backend_name = self._get_active_compute_backend_name()

        self.gpu_label.setText(f"\U0001f5a5 {system_gpu_name} | {backend_name}")


    def _t(self, key: str) -> str:
        translator = getattr(self.controller, "t", None)
        if callable(translator):
            return translator(key)
        return key

    def _set_action_text(self, action, key: str):
        action.setText(self._t(key))

    def _trim_transparent_pixmap(self, pixmap: QPixmap, alpha_threshold: int = 8) -> QPixmap:
        """
        Crops transparent padding from a logo PNG so the QLabel only takes up
        the real visible logo size.
        """
        if pixmap.isNull():
            return pixmap

        try:
            try:
                fmt = QImage.Format.Format_ARGB32
            except AttributeError:
                fmt = QImage.Format_ARGB32

            image = pixmap.toImage().convertToFormat(fmt)

            w = image.width()
            h = image.height()

            min_x = w
            min_y = h
            max_x = -1
            max_y = -1

            for y in range(h):
                for x in range(w):
                    if image.pixelColor(x, y).alpha() > alpha_threshold:
                        if x < min_x:
                            min_x = x
                        if y < min_y:
                            min_y = y
                        if x > max_x:
                            max_x = x
                        if y > max_y:
                            max_y = y

            if max_x < min_x or max_y < min_y:
                return pixmap

            rect = QRect(
                min_x,
                min_y,
                max_x - min_x + 1,
                max_y - min_y + 1,
            )

            return QPixmap.fromImage(image.copy(rect))

        except Exception as e:
            print(f"Logo trim failed: {e}")
            return pixmap


    def _set_app_logo(self):
        """
        Loads the transparent VD3D logo into the top-left app title area.
        Crops transparent padding so it does not push the tabs away.
        """
        if not hasattr(self, "app_title"):
            return

        logo_path = getattr(
            self,
            "_app_logo_path",
            resource_path("resources/icons/NewVD3D-Logo.png"),
        )

        pixmap = QPixmap(logo_path)

        if pixmap.isNull():
            self.app_title.setObjectName("AppTitle")
            self.app_title.setText(self._t("VisionDepth3D"))
            self.app_title.setFixedSize(120, 36)
            return

        pixmap = self._trim_transparent_pixmap(pixmap)

        target_h = int(getattr(self, "_app_logo_height", 42))

        scaled = pixmap.scaledToHeight(
            target_h,
            Qt.SmoothTransformation,
        )

        self.app_title.setObjectName("AppLogo")
        self.app_title.setText("")
        self.app_title.setToolTip(self._t("VisionDepth3D"))
        self.app_title.setPixmap(scaled)

        # This is the important part:
        # QLabel becomes the exact size of the scaled cropped logo.
        self.app_title.setFixedSize(scaled.width(), scaled.height())

    def refresh_shell_labels(self):
        # App title / window title
        self.setWindowTitle(self._t("VisionDepth3D"))
        if hasattr(self, "app_title"):
            self._set_app_logo()

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
                f"{self._t('VisionDepth3D v4.2.1')}\n\n"
                f"{self._t('A hybrid 2D-to-3D conversion suite for cinema and VR.')}\n\n"
                f"{self._t('Features:')}\n"
                f" • {self._t('Depth map blending (multi-model)')}\n"
                f" • {self._t('Depth-weighted parallax shifting')}\n"
                f" • {self._t('Scene-aware stereo rendering')}\n"
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
            self._show_gpu_diagnostics_dialog(report)

        except Exception as e:
            QMessageBox.warning(
                self,
                self._t("GPU Diagnostics"),
                self._t("Could not detect GPU:") + f"\n{e}"
            )

    def _parse_gpu_report(self, report: str) -> dict:
        data = {}

        for line in str(report or "").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()

        return data

    def _diag_badge_style(self, value: str) -> str:
        v = str(value or "").strip().lower()

        if v in ("yes", "true", "ok") or " ok in " in v:
            bg = "#065f46"
            fg = "#d1fae5"
        elif v in ("no", "false", "none", "(not found)", "missing", "not checked"):
            bg = "#7f1d1d"
            fg = "#fee2e2"
        elif "failed" in v or "not found" in v or "missing" in v:
            bg = "#7f1d1d"
            fg = "#fee2e2"
        elif "maybe" in v or "unavailable" in v:
            bg = "#78350f"
            fg = "#fef3c7"
        else:
            bg = "#374151"
            fg = "#e5e7eb"

        return (
            f"background-color: {bg};"
            f"color: {fg};"
            "border-radius: 8px;"
            "padding: 5px 8px;"
            "font-weight: 700;"
        )

    def _add_gpu_diag_card(self, parent_layout, title: str, rows: list, data: dict):
        card = QFrame()
        card.setObjectName("GpuDiagCard")
        card.setStyleSheet("""
            QFrame#GpuDiagCard {
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 14px;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        title_label = QLabel(self._t(title))
        title_label.setStyleSheet("""
            color: #f9fafb;
            font-size: 15px;
            font-weight: 800;
        """)
        layout.addWidget(title_label)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        for row, key in enumerate(rows):
            value = data.get(key, self._t("Not checked"))

            name_label = QLabel(self._t(key))
            name_label.setStyleSheet("color: #d1d5db; font-size: 12px;")
            name_label.setMinimumWidth(170)

            value_label = QLabel(str(value))
            value_label.setWordWrap(True)
            value_label.setStyleSheet(self._diag_badge_style(str(value)))

            grid.addWidget(name_label, row, 0)
            grid.addWidget(value_label, row, 1)

        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        parent_layout.addWidget(card)
    def _show_gpu_diagnostics_dialog(self, report: str):
        data = self._parse_gpu_report(report)
        active_backend = data.get("Active VD3D backend", "Unknown")

        dialog = QDialog(self)
        dialog.setWindowTitle(self._t("GPU Diagnostics"))
        dialog.resize(860, 760)
        dialog.setMinimumSize(760, 560)

        dialog.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #e5e7eb;
            }

            QScrollArea {
                background-color: #0f172a;
                border: none;
            }

            QTextEdit {
                background-color: #020617;
                color: #d1d5db;
                border: 1px solid #374151;
                border-radius: 10px;
                padding: 8px;
                font-family: Consolas;
                font-size: 10pt;
            }

            QPushButton {
                background-color: #374151;
                color: #ffffff;
                border: none;
                border-radius: 9px;
                padding: 9px 14px;
                font-weight: 700;
            }

            QPushButton:hover {
                background-color: #4b5563;
            }

            QPushButton#PrimaryButton {
                background-color: #2563eb;
            }

            QPushButton#PrimaryButton:hover {
                background-color: #1d4ed8;
            }
        """)

        root_layout = QVBoxLayout(dialog)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        header = QWidget()
        header.setStyleSheet("background-color: #020617;")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 16, 18, 14)
        header_layout.setSpacing(6)

        title = QLabel(self._t("VisionDepth3D GPU Diagnostics"))
        title.setStyleSheet("color: #f8fafc; font-size: 22px; font-weight: 900;")
        header_layout.addWidget(title)

        backend = QLabel(f"{self._t('Active Backend:')} {active_backend}")
        backend.setStyleSheet("color: #93c5fd; font-size: 13px; font-weight: 800;")
        header_layout.addWidget(backend)

        root_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(14, 12, 14, 12)
        content_layout.setSpacing(10)

        self._add_gpu_diag_card(content_layout, "Backend Summary", [
            "GPU Diagnostics version",
            "Active VD3D backend",
            "PyTorch",
        ], data)

        self._add_gpu_diag_card(content_layout, "CUDA / NVIDIA", [
            "CUDA available",
            "torch.version.cuda",
            "Device count",
            "Device 0",
            "cuDNN enabled",
            "CUDA matmul OK in",
            "NVIDIA driver",
            "NVENC encoders listed",
        ], data)

        self._add_gpu_diag_card(content_layout, "DirectML", [
            "DirectML available",
            "DirectML device",
            "DirectML matmul OK in",
            "DirectML error",
        ], data)

        self._add_gpu_diag_card(content_layout, "FFmpeg Tools", [
            "FFmpeg found",
            "FFmpeg source",
            "FFmpeg executable test",
            "FFprobe found",
            "FFprobe source",
            "FFprobe executable test",
        ], data)

        self._add_gpu_diag_card(content_layout, "Install Paths", [
            "App base",
            "Bundle base",
            "FFmpeg path",
            "FFprobe path",
        ], data)

        raw_label = QLabel(self._t("Raw Diagnostic Log"))
        raw_label.setStyleSheet("color: #f9fafb; font-size: 15px; font-weight: 800; margin-top: 8px;")
        content_layout.addWidget(raw_label)

        raw = QTextEdit()
        raw.setReadOnly(True)
        raw.setPlainText(report or "")
        raw.setMinimumHeight(180)
        content_layout.addWidget(raw)

        content_layout.addStretch()
        scroll.setWidget(content)

        root_layout.addWidget(scroll, 1)

        button_bar = QWidget()
        button_bar.setStyleSheet("background-color: #020617;")
        button_layout = QHBoxLayout(button_bar)
        button_layout.setContentsMargins(18, 12, 18, 12)

        copy_btn = QPushButton(self._t("Copy Report"))
        copy_btn.setObjectName("PrimaryButton")

        close_btn = QPushButton(self._t("Close"))
       

        def copy_report():
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(report or "")

        copy_btn.clicked.connect(copy_report)
        close_btn.clicked.connect(dialog.accept)

        button_layout.addWidget(copy_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)

        root_layout.addWidget(button_bar)

        dialog.exec()

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
                
                QLabel#AppLogo {
                    background-color: transparent;
                    border: none;
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
