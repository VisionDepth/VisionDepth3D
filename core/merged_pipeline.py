# ui/pages/fps_upscale_page.py
import os
import threading
import traceback
import time
import json

from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QScrollArea,
    QGroupBox,
    QLineEdit,
    QSlider,
    QMessageBox,
    QInputDialog,
    QFrame,
    QGridLayout,
    QProgressBar,
    QSizePolicy,
    QSplitter,
)

from ui.styles.page_theme import apply_unified_page_theme

import platform
import subprocess


def hidden_subprocess_kwargs():
    if platform.system().lower() != "windows":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0

    return {
        "startupinfo": startupinfo,
        "creationflags": subprocess.CREATE_NO_WINDOW,
    }

COMMON_FPS = [
    23.976, 24.0, 25.0, 29.97, 29.976, 30.0, 48.0, 50.0,
    59.94, 60.0, 72.0, 90.0, 100.0, 119.88, 120.0,
    144.0, 165.0, 239.76, 239.808, 240.0,
]

FPS_MULTIPLIERS = [2, 4, 8]

FFMPEG_CODEC_MAP = {
    "H.264 / AVC (libx264 - CPU)": "libx264",
    "H.265 / HEVC (libx265 - CPU)": "libx265",
    "AV1 (libaom - CPU)": "libaom-av1",
    "AV1 (SVT - CPU, faster)": "libsvtav1",
    "MPEG-4 (mp4v - CPU)": "mp4v",
    "XviD (AVI - CPU)": "XVID",
    "DivX (AVI - CPU)": "DIVX",
    "H.264 / AVC (NVENC - NVIDIA GPU)": "h264_nvenc",
    "H.265 / HEVC (NVENC - NVIDIA GPU)": "hevc_nvenc",
    "AV1 (NVENC - NVIDIA RTX 40+ GPU)": "av1_nvenc",
    "H.264 / AVC (AMF - AMD GPU)": "h264_amf",
    "H.265 / HEVC (AMF - AMD GPU)": "hevc_amf",
    "AV1 (AMF - AMD RDNA3+)": "av1_amf",
    "H.264 / AVC (QSV - Intel GPU)": "h264_qsv",
    "H.265 / HEVC (QSV - Intel GPU)": "hevc_qsv",
    "VP9 (QSV - Intel GPU)": "vp9_qsv",
    "AV1 (QSV - Intel ARC / Gen11+)": "av1_qsv",
}


def _load_upscaler_models():
    return {
        "RealESR (Balanced)": "upscale:FuryTMP/RealESR_Gx4_fp16",
        "RealESRGAN (Sharp)": "upscale:FuryTMP/RealESRGANx4_fp16",
        "RealESR Anime": "upscale:FuryTMP/RealESR_Animex4_fp16",
        "BSRGAN x2": "upscale:FuryTMP/BSRGANx2_fp16",
        "BSRGAN x4": "upscale:FuryTMP/BSRGANx4_fp16",
    }


def _load_rife_models():
    return {
        "RIFE FP32": "rife:FuryTMP/RIFE_fp32",
        "RIFE v4.9": "rife:FuryTMP/RIFE_v4.9"
    }


class SectionHeader(QWidget):
    def __init__(self, title: str):
        super().__init__()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 8, 2, 2)
        layout.setSpacing(8)

        self.label = QLabel(title)
        self.label.setObjectName("SectionHeaderLabel")

        layout.addWidget(self.label)
        layout.addStretch()

    def set_text(self, text: str):
        self.label.setText(text)
        
class ModernCard(QFrame):
    def __init__(self, title: str = "", subtitle: str = ""):
        super().__init__()
        self.setObjectName("ModernCard")

        self.title_label = None
        self.subtitle_label = None

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 16)
        self.layout.setSpacing(10)

        if title:
            self.title_label = QLabel(title)
            self.title_label.setObjectName("CardTitle")
            self.layout.addWidget(self.title_label)

        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setObjectName("CardSubtitle")
            self.subtitle_label.setWordWrap(True)
            self.layout.addWidget(self.subtitle_label)

    def set_title_text(self, text: str):
        if self.title_label is not None:
            self.title_label.setText(text)

    def set_subtitle_text(self, text: str):
        if self.subtitle_label is not None:
            self.subtitle_label.setText(text)

class _TkProgressProxy:
    """Mimics Tkinter progress/status widgets for merged_pipeline compatibility."""
    def __init__(self, callback=None):
        self._callback = callback
        self._value = 0.0
        self._maximum = 100.0
        self._status_text = ""
        self._start_time = time.time()
        self._fps_like = None
        self._eta = None

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

    def _parse_status_metrics(self, text):
        """
        Parses merged_pipeline status strings like:
        Progress: 123/500 | FPS: 3.85 | ETA: 00:02:31
        """
        self._fps_like = None
        self._eta = None

        if not text:
            return

        parts = [p.strip() for p in str(text).split("|")]

        for part in parts:
            lower = part.lower()

            if lower.startswith("fps:"):
                try:
                    self._fps_like = float(part.split(":", 1)[1].strip())
                except Exception:
                    pass

            elif lower.startswith("eta:"):
                eta_text = part.split(":", 1)[1].strip()
                self._eta = self._parse_hms(eta_text)

    def _progress_percent(self):
        maximum = max(1e-6, float(self._maximum or 100.0))
        return max(0.0, min(100.0, (float(self._value) / maximum) * 100.0))

    def _emit(self):
        if not self._callback:
            return

        elapsed = max(0.0, time.time() - self._start_time)
        progress = self._progress_percent()

        fps_like = self._fps_like
        eta = self._eta

        # Fallback if merged_pipeline did not provide FPS/ETA text yet.
        if fps_like is None and elapsed > 0 and progress > 0:
            fps_like = progress / elapsed

        if eta is None and elapsed > 0 and progress > 0:
            remaining = max(0.0, 100.0 - progress)
            eta = (remaining / progress) * elapsed

        self._callback({
            "progress": progress,
            "status_text": self._status_text,
            "elapsed": elapsed,
            "eta": eta,
            "fps_like": fps_like,
            "rate_label": "FPS",
        })

    def config(self, **kw):
        if "maximum" in kw:
            try:
                self._maximum = float(kw["maximum"])
            except Exception:
                self._maximum = 100.0

        if "value" in kw:
            try:
                self._value = float(kw["value"])
            except Exception:
                self._value = 0.0

        if "text" in kw:
            self._status_text = str(kw["text"])
            self._parse_status_metrics(self._status_text)

        self._emit()

    def configure(self, **kw):
        self.config(**kw)

    def after(self, ms, fn):
        import threading as _th
        t = _th.Timer(ms / 1000.0, fn)
        t.daemon = True
        t.start()

    def __setitem__(self, key, value):
        if key == "value":
            try:
                self._value = float(value)
            except Exception:
                self._value = 0.0
            self._emit()

        elif key == "maximum":
            try:
                self._maximum = float(value)
            except Exception:
                self._maximum = 100.0
            self._emit()

    def __getitem__(self, key):
        if key == "maximum":
            return self._maximum
        if key == "value":
            return self._value
        return 0

    def start(self, interval=None):
        pass

    def stop(self):
        pass

    def update_idletasks(self):
        pass

    def update(self):
        pass

    def winfo_toplevel(self):
        return self
        
class FpsUpscalePage(QWidget):
    progress_updated = Signal(dict)
    preview_ready = Signal(list)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._translation_map = []

        self.input_video_file = ""
        self.frames_folder = ""
        self.output_file = ""
        self.keep_original_audio = True

        self.out_width = 1920
        self.out_height = 1080
        self.fps = 23.976
        self.fps_multiplier = 2
        self.codec = "H.264 / AVC (NVENC - NVIDIA GPU)"
        self.enable_rife = True
        self.enable_upscale = False
        self.blend_mode = "OFF"
        self.input_res_pct = 100
        self.upscale_model = "RealESR (Balanced)"
        self.rife_model = "RIFE FP32"
        self.scene_threshold = 30
        self.scene_format = "mkv"

        # FPS/Upscale preview state
        self.preview_images = []
        self.preview_index = 0
        self.preview_zoom = 1.0
        self.preview_focus_side = "sbs"  # sbs, original, preview

        # Preview pan/drag state
        self.preview_dragging = False
        self.preview_drag_start = None
        self.preview_drag_h_start = 0
        self.preview_drag_v_start = 0

        self._build_ui()
        self.progress_updated.connect(self._on_progress_updated)
        self.preview_ready.connect(self._on_preview_ready)

    def _t(self, key: str) -> str:
        translator = getattr(self.controller, "t", None)
        if not callable(translator):
            return key

        translations = getattr(self.controller, "translations", None)
        if translations is None:
            language_service = getattr(self.controller, "language_service", None)
            translations = getattr(language_service, "translations", None)

        aliases = {
            "FPS / Upscale Enhancer": "FPS/Upscale Enhancement",
            "📂 Extract Frames from Video": "Extract Frames from Video",
            "Enable RIFE Frame Interpolation": "Enable RIFE Interpolation",
            "Upscale Model:": "Model Selection:",
            "Blend:": "AI Blending:",
            "Input %:": "Input Resolution %:",
            "Threshold:": "Sensitivity Threshold (lower = more cuts):",
            "🔍 Detect Scenes & Extract": "Detect Scenes & Extract",
            "▶ Start Processing": "▶ Start Processing",
            "⚡ Threaded RIFE + ESRGAN": "Threaded RIFE + ESRGAN",
            "Processing Options": "Processing Options",
            "Output Settings": "Output Settings",
            "Codec:": "FFmpeg Output Codec:",
            "Frames:": "Frames Folder:",
            "Output:": "Output Video File:",
        }

        if isinstance(translations, dict):
            if key in translations:
                return translations[key]

            old_key = aliases.get(key)
            if old_key and old_key in translations:
                return translations[old_key]

            return key

        value = translator(key)
        if value != key:
            return value

        old_key = aliases.get(key)
        if old_key:
            old_value = translator(old_key)
            if old_value != old_key:
                return old_value

        return key

    def _qt_text(self, text: str) -> str:
        text = str(text)
        marker = "\u0000"
        return text.replace("&&", marker).replace("&", "&&").replace(marker, "&&")

    def _register_text(self, widget, key: str):
        self._translation_map.append((widget, key, "text"))
        widget.setText(self._qt_text(self._t(key)))
        
    def _register_title(self, widget, key: str):
        self._translation_map.append((widget, key, "title"))
        if hasattr(widget, "setTitle"):
            widget.setTitle(self._t(key))
        elif hasattr(widget, "set_title_text"):
            widget.set_title_text(self._t(key))
        elif hasattr(widget, "set_text"):
            widget.set_text(self._t(key))

    def _register_subtitle(self, widget, key: str):
        self._translation_map.append((widget, key, "subtitle"))
        if hasattr(widget, "set_subtitle_text"):
            widget.set_subtitle_text(self._t(key))

    def _register_placeholder(self, widget, key: str):
        self._translation_map.append((widget, key, "placeholder"))
        widget.setPlaceholderText(self._t(key))

    def _register_tooltip(self, widget, key: str):
        self._translation_map.append((widget, key, "tooltip"))
        widget.setToolTip(self._t(key))

    def _label(self, key: str) -> QLabel:
        label = QLabel()
        self._register_text(label, key)
        return label

    def _button(self, key: str) -> QPushButton:
        button = QPushButton()
        self._register_text(button, key)
        return button

    def _checkbox(self, key: str) -> QCheckBox:
        checkbox = QCheckBox()
        self._register_text(checkbox, key)
        return checkbox

    def _group(self, key: str) -> QGroupBox:
        group = QGroupBox()
        self._register_title(group, key)
        return group

    def _section(self, key: str) -> SectionHeader:
        section = SectionHeader(self._t(key))
        self._register_title(section, key)
        return section

    def _card(self, title_key: str, subtitle_key: str = "") -> ModernCard:
        card = ModernCard(self._t(title_key), self._t(subtitle_key) if subtitle_key else "")
        self._register_title(card, title_key)
        if subtitle_key:
            self._register_subtitle(card, subtitle_key)
        return card

    def _set_spin_prefixes(self):
        if hasattr(self, "w_spin"):
            self.w_spin.setPrefix(f"{self._t('W:')} ")
        if hasattr(self, "h_spin"):
            self.h_spin.setPrefix(f"{self._t('H:')} ")

    def refresh_labels(self):
        for widget, key, widget_type in self._translation_map:
            try:
                if widget_type == "text":
                    widget.setText(self._qt_text(self._t(key)))
                elif widget_type == "title":
                    if hasattr(widget, "setTitle"):
                        widget.setTitle(self._t(key))
                    elif hasattr(widget, "set_title_text"):
                        widget.set_title_text(self._t(key))
                    elif hasattr(widget, "set_text"):
                        widget.set_text(self._t(key))
                elif widget_type == "subtitle":
                    if hasattr(widget, "set_subtitle_text"):
                        widget.set_subtitle_text(self._t(key))
                elif widget_type == "placeholder":
                    widget.setPlaceholderText(self._t(key))
                elif widget_type == "tooltip":
                    widget.setToolTip(self._t(key))
            except RuntimeError:
                pass

        self._set_spin_prefixes()
        self._refresh_summary()

    def _apply_modern_style(self):
        theme = getattr(self, "_active_theme", None) or {}
        apply_unified_page_theme(self, theme)


    def _build_ui(self):
        self._apply_modern_style()

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(16)

        header = QHBoxLayout()
        header.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        title = QLabel()
        self._register_text(title, "FPS / Upscale Enhancer")
        title.setObjectName("PageTitle")

        subtitle = QLabel()
        self._register_text(
            subtitle,
            "Interpolate frames with RIFE, upscale with ESRGAN, detect scenes, and export final video."
        )
        subtitle.setObjectName("PageSubtitle")

        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        self.state_badge = QLabel()
        self._register_text(self.state_badge, "READY")
        self.state_badge.setObjectName("StatusPill")
        self.state_badge.setAlignment(Qt.AlignCenter)

        header.addLayout(title_col)
        header.addStretch()
        header.addWidget(self.state_badge)

        root.addLayout(header)

        body_scroll = QScrollArea()
        body_scroll.setWidgetResizable(True)
        body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        body_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body_scroll.setFrameShape(QFrame.NoFrame)

        body_widget = QWidget()
        body_widget.setMinimumWidth(1180)

        top_row = QHBoxLayout(body_widget)
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(16)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setMinimumWidth(300)

        left_widget = QWidget()
        left_widget.setMinimumWidth(360)

        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        left_layout.addWidget(self._section("Input / Output"))

        # Source preparation tools
        source_tools_group = self._group("Source Tools")
        source_tools_layout = QVBoxLayout(source_tools_group)
        source_tools_layout.setSpacing(10)

        self.extract_btn = self._button("📂 Extract Frames from Video")
        self.extract_btn.setMinimumHeight(38)
        self.extract_btn.clicked.connect(self._extract_frames)
        source_tools_layout.addWidget(self.extract_btn)

        self.detect_scenes_btn = self._button("🔍 Detect Scenes & Extract")
        self.detect_scenes_btn.setMinimumHeight(38)
        self.detect_scenes_btn.clicked.connect(self._detect_scenes)
        source_tools_layout.addWidget(self.detect_scenes_btn)

        scene_settings_group = self._group("Scene Settings")
        scene_settings_layout = QVBoxLayout(scene_settings_group)
        scene_settings_layout.setSpacing(10)

        scene_thresh_row = QHBoxLayout()
        scene_thresh_row.setSpacing(8)

        scene_thresh_row.addWidget(self._label("Threshold:"))

        self.scene_slider = QSlider(Qt.Horizontal)
        self.scene_slider.setRange(10, 80)
        self.scene_slider.setValue(30)

        self.scene_thresh_label = QLabel("30")
        self.scene_thresh_label.setMinimumWidth(28)
        self.scene_thresh_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.scene_slider.valueChanged.connect(
            lambda v: self.scene_thresh_label.setText(str(v))
        )

        scene_thresh_row.addWidget(self.scene_slider, 1)
        scene_thresh_row.addWidget(self.scene_thresh_label)

        scene_fmt_row = QHBoxLayout()
        scene_fmt_row.setSpacing(8)

        scene_fmt_row.addWidget(self._label("Format:"))

        self.scene_fmt_combo = QComboBox()
        self.scene_fmt_combo.addItems(["mp4", "mov", "avi", "mkv"])
        self.scene_fmt_combo.setCurrentText("mkv")

        scene_fmt_row.addWidget(self.scene_fmt_combo, 1)

        scene_settings_layout.addLayout(scene_thresh_row)
        scene_settings_layout.addLayout(scene_fmt_row)

        source_tools_layout.addWidget(scene_settings_group)
        left_layout.addWidget(source_tools_group)
        
        io_group = self._group("Paths")
        io_layout = QVBoxLayout(io_group)
        io_layout.setSpacing(10)

        input_video_row = QHBoxLayout()
        input_video_row.setSpacing(8)

        input_video_label = self._label("Input Video:")
        input_video_label.setMinimumWidth(76)

        self.input_video_edit = QLineEdit()
        self.input_video_edit.setReadOnly(True)
        self._register_placeholder(self.input_video_edit, "Optional source video for audio...")

        input_video_browse = self._button("Browse")
        input_video_browse.setFixedWidth(78)
        input_video_browse.clicked.connect(self._browse_input_video)

        input_video_row.addWidget(input_video_label)
        input_video_row.addWidget(self.input_video_edit, 1)
        input_video_row.addWidget(input_video_browse)

        frames_row = QHBoxLayout()
        frames_row.setSpacing(8)

        frames_label = self._label("Frames:")
        frames_label.setMinimumWidth(54)

        self.frames_edit = QLineEdit()
        self.frames_edit.setReadOnly(True)
        self._register_placeholder(self.frames_edit, "Select frames folder...")

        frames_browse = self._button("Browse")
        frames_browse.setFixedWidth(78)
        frames_browse.clicked.connect(self._browse_frames)

        frames_row.addWidget(frames_label)
        frames_row.addWidget(self.frames_edit, 1)
        frames_row.addWidget(frames_browse)

        out_row = QHBoxLayout()
        out_row.setSpacing(8)

        output_label = self._label("Output:")
        output_label.setMinimumWidth(54)

        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self._register_placeholder(self.output_edit, "Select output file...")

        out_browse = self._button("Save As")
        out_browse.setFixedWidth(78)
        out_browse.clicked.connect(self._browse_output)

        out_row.addWidget(output_label)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(out_browse)

        self.keep_audio_check = self._checkbox("Keep Original Audio")
        self.keep_audio_check.setChecked(True)

        io_layout.addLayout(input_video_row)
        io_layout.addLayout(frames_row)
        io_layout.addLayout(out_row)
        io_layout.addWidget(self.keep_audio_check)
        
        left_layout.addWidget(io_group)

        toggle_group = self._group("Processing Options")
        toggle_layout = QVBoxLayout(toggle_group)
        toggle_layout.setSpacing(10)

        self.rife_check = self._checkbox("Enable RIFE Frame Interpolation")
        self.rife_check.setChecked(True)

        self.upscale_check = self._checkbox("Enable Real-ESRGAN Upscale")

        toggle_layout.addWidget(self.rife_check)
        toggle_layout.addWidget(self.upscale_check)

        left_layout.addWidget(toggle_group)

        out_group = self._group("Output Settings")
        out_layout = QVBoxLayout(out_group)
        out_layout.setSpacing(10)

        res_row = QHBoxLayout()
        res_row.setSpacing(8)

        res_label = self._label("Resolution:")
        res_label.setMinimumWidth(76)

        self.w_spin = QSpinBox()
        self.w_spin.setRange(256, 7680)
        self.w_spin.setValue(1920)
        self.w_spin.setPrefix(f"{self._t('W:')} ")

        self.h_spin = QSpinBox()
        self.h_spin.setRange(256, 7680)
        self.h_spin.setValue(1080)
        self.h_spin.setPrefix(f"{self._t('H:')} ")

        res_row.addWidget(res_label)
        res_row.addWidget(self.w_spin)
        res_row.addWidget(self.h_spin)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(8)

        fps_label = self._label("FPS:")
        fps_label.setMinimumWidth(76)

        self.fps_combo = QComboBox()
        for f in COMMON_FPS:
            self.fps_combo.addItem(str(f), f)
        self.fps_combo.setCurrentText("23.976")

        self.mult_combo = QComboBox()
        for m in FPS_MULTIPLIERS:
            self.mult_combo.addItem(f"{m}x", m)
        self.mult_combo.setCurrentText("2x")

        settings_row.addWidget(fps_label)
        settings_row.addWidget(self.fps_combo, 1)
        settings_row.addWidget(QLabel("×"))
        settings_row.addWidget(self.mult_combo, 1)

        self.codec_combo = QComboBox()
        self.codec_combo.addItems(list(FFMPEG_CODEC_MAP.keys()))
        self.codec_combo.setCurrentText(self.codec)

        out_layout.addLayout(res_row)
        out_layout.addLayout(settings_row)
        out_layout.addWidget(self._label("Codec:"))
        out_layout.addWidget(self.codec_combo)

        left_layout.addWidget(out_group)

        model_group = self._group("ESRGAN / RIFE Models")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(10)

        model_layout.addWidget(self._label("Upscale Model:"))

        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(list(_load_upscaler_models().keys()))
        self.upscale_combo.setCurrentText(self.upscale_model)
        model_layout.addWidget(self.upscale_combo)

        model_layout.addWidget(self._label("RIFE Model:"))

        self.rife_combo = QComboBox()
        self.rife_combo.addItems(list(_load_rife_models().keys()))
        self.rife_combo.setCurrentText(self.rife_model)
        model_layout.addWidget(self.rife_combo)

        blend_row = QHBoxLayout()
        blend_row.setSpacing(8)

        self.blend_combo = QComboBox()
        self.blend_combo.addItems(["OFF", "LOW", "MEDIUM", "HIGH"])
        self.blend_combo.setCurrentText("OFF")

        self.res_pct_combo = QComboBox()
        self.res_pct_combo.addItems(["25", "50", "75", "100"])
        self.res_pct_combo.setCurrentText("100")

        blend_row.addWidget(self._label("Blend:"))
        blend_row.addWidget(self.blend_combo, 1)
        blend_row.addWidget(self._label("Input %:"))
        blend_row.addWidget(self.res_pct_combo, 1)

        model_layout.addLayout(blend_row)
        left_layout.addWidget(model_group)
        left_layout.addStretch()

        left_scroll.setWidget(left_widget)

        center_widget = QWidget()
        center_widget.setMinimumWidth(420)
        center_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(12)

        render_card = self._card(
            "Render Plan",
            "Current export settings, pipeline state, and job progress."
        )
        
        render_grid = QGridLayout()
        render_grid.setHorizontalSpacing(18)
        render_grid.setVerticalSpacing(8)

        self.summary_resolution = QLabel("1920 × 1080")
        self.summary_fps = QLabel("23.976 FPS × 2")
        self.summary_codec = QLabel("NVENC H.264")
        self.summary_pipeline = QLabel("RIFE enabled, ESRGAN off")
        self.summary_model = QLabel("RealESR Balanced / RIFE FP32")

        for label in (
            self.summary_resolution,
            self.summary_fps,
            self.summary_codec,
            self.summary_pipeline,
            self.summary_model,
        ):
            label.setObjectName("MutedLabel")
            label.setWordWrap(True)
            label.setMinimumWidth(120)

        def add_summary_item(row, col, title_key, value_label, col_span=1):
            title = self._label(title_key)
            title.setObjectName("MutedLabel")

            box = QVBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(2)
            box.addWidget(title)
            box.addWidget(value_label)

            wrapper = QWidget()
            wrapper.setLayout(box)

            render_grid.addWidget(wrapper, row, col, 1, col_span)

        add_summary_item(0, 0, "Resolution:", self.summary_resolution)
        add_summary_item(0, 1, "Frame Rate:", self.summary_fps)
        add_summary_item(0, 2, "Codec:", self.summary_codec)
        add_summary_item(1, 0, "Pipeline:", self.summary_pipeline, 2)
        add_summary_item(1, 2, "Models:", self.summary_model)

        render_grid.setColumnStretch(0, 1)
        render_grid.setColumnStretch(1, 1)
        render_grid.setColumnStretch(2, 1)

        render_card.layout.addLayout(render_grid)

        render_card.setMaximumHeight(150)
        center_layout.addWidget(render_card, 0)

        preview_card = self._card(
            "Preview / Job Output",
            "Generate sample previews from the beginning, middle, and end of the frame folder."
        )

        self.preview_placeholder = QLabel()
        self._register_text(self.preview_placeholder, "No preview loaded yet")
        self.preview_placeholder.setObjectName("PreviewPlaceholder")
        self.preview_placeholder.setAlignment(Qt.AlignCenter)
        self.preview_placeholder.setMinimumSize(640, 460)
        self.preview_placeholder.setWordWrap(True)
        self.preview_placeholder.setMouseTracking(True)
        self.preview_placeholder.setCursor(Qt.CursorShape.OpenHandCursor)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setFrameShape(QFrame.NoFrame)
        self.preview_scroll.setAlignment(Qt.AlignCenter)
        self.preview_scroll.setMinimumHeight(460)
        self.preview_scroll.setWidget(self.preview_placeholder)
        self.preview_scroll.viewport().setMouseTracking(True)
        self.preview_scroll.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

        self.preview_placeholder.installEventFilter(self)
        self.preview_scroll.viewport().installEventFilter(self)

        preview_nav = QHBoxLayout()
        preview_nav.setSpacing(8)

        self.generate_preview_btn = self._button("Generate Preview")
        self.generate_preview_btn.setMinimumHeight(32)
        self.generate_preview_btn.clicked.connect(self._generate_preview)

        self.preview_prev_btn = self._button("Previous")
        self.preview_prev_btn.setMinimumHeight(32)
        self.preview_prev_btn.clicked.connect(self._preview_previous)

        self.preview_counter_label = QLabel("Preview 0 / 0")
        self.preview_counter_label.setObjectName("MutedLabel")
        self.preview_counter_label.setAlignment(Qt.AlignCenter)

        self.preview_next_btn = self._button("Next")
        self.preview_next_btn.setMinimumHeight(32)
        self.preview_next_btn.clicked.connect(self._preview_next)

        self.preview_zoom_label = QLabel("100%")
        self.preview_zoom_label.setObjectName("MutedLabel")
        self.preview_zoom_label.setAlignment(Qt.AlignCenter)
        self.preview_zoom_label.setMinimumWidth(58)

        self.preview_hint_label = QLabel(self._t("Mouse wheel over Original or Preview to zoom."))
        self.preview_hint_label.setObjectName("MutedLabel")
        self.preview_hint_label.setAlignment(Qt.AlignCenter)

        self.preview_reset_btn = self._button("Reset View")
        self.preview_reset_btn.setMinimumHeight(32)
        self.preview_reset_btn.clicked.connect(self._preview_zoom_reset)

        preview_nav.addWidget(self.generate_preview_btn, 2)
        preview_nav.addWidget(self.preview_prev_btn, 1)
        preview_nav.addWidget(self.preview_counter_label, 1)
        preview_nav.addWidget(self.preview_next_btn, 1)
        preview_nav.addWidget(self.preview_zoom_label)
        preview_nav.addWidget(self.preview_reset_btn, 1)

        preview_card.layout.addWidget(self.preview_scroll, 1)
        preview_card.layout.addWidget(self.preview_hint_label)
        preview_card.layout.addLayout(preview_nav)

        center_layout.addWidget(preview_card, 1)

        center_layout.setStretch(0, 0)
        center_layout.setStretch(1, 1)

        right_widget = QWidget()
        right_widget.setMinimumWidth(260)

        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        info_card = self._card("Session Info")

        self.info_label = QLabel()
        self._register_text(
            self.info_label,
            "Select a frames folder and output file to begin.\n\n"
            "• RIFE interpolates new frames between existing ones\n"
            "• ESRGAN upscales each frame to a higher resolution\n"
            "• Threaded mode can process RIFE and ESRGAN in parallel"
        )
        self.info_label.setObjectName("MutedLabel")
        self.info_label.setWordWrap(True)

        info_card.layout.addWidget(self.info_label)
        right_layout.addWidget(info_card)

        self.status_label = QLabel(self._t("Ready"))
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)

        right_layout.addWidget(self.status_label)

        actions_card = self._card("Actions")

        self.start_btn = self._button("▶ Start Processing")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.setMinimumHeight(42)
        self.start_btn.clicked.connect(self._start_processing)

        self.threaded_btn = self._button("⚡ Threaded RIFE + ESRGAN")
        self.threaded_btn.setMinimumHeight(40)
        self.threaded_btn.clicked.connect(self._start_threaded)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.suspend_btn = QPushButton("⏸")
        self.suspend_btn.setObjectName("IconButton")
        self._register_tooltip(self.suspend_btn, "Pause")
        self.suspend_btn.clicked.connect(self._pause)

        self.resume_btn = QPushButton("▶")
        self.resume_btn.setObjectName("IconButton")
        self._register_tooltip(self.resume_btn, "Resume")
        self.resume_btn.clicked.connect(self._resume)

        self.cancel_btn = QPushButton("⏹")
        self.cancel_btn.setObjectName("DangerButton")
        self._register_tooltip(self.cancel_btn, "Stop")
        self.cancel_btn.clicked.connect(self._stop)

        ctrl_row.addWidget(self.suspend_btn)
        ctrl_row.addWidget(self.resume_btn)
        ctrl_row.addWidget(self.cancel_btn)

        actions_card.layout.addWidget(self.start_btn)
        actions_card.layout.addWidget(self.threaded_btn)
        actions_card.layout.addLayout(ctrl_row)

        right_layout.addWidget(actions_card)
        right_layout.addStretch()

        # Resizable FPS/Upscale layout:
        # left settings | center preview/render plan | right session/actions
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        self.main_splitter.addWidget(left_scroll)
        self.main_splitter.addWidget(center_widget)
        self.main_splitter.addWidget(right_widget)

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)

        self.main_splitter.setSizes([380, 950, 330])

        top_row.addWidget(self.main_splitter, 1)

        body_scroll.setWidget(body_widget)
        root.addWidget(body_scroll, 1)

        self._connect_summary_signals()
        self._refresh_summary()

    def _connect_summary_signals(self):
        self.w_spin.valueChanged.connect(self._refresh_summary)
        self.h_spin.valueChanged.connect(self._refresh_summary)
        self.fps_combo.currentTextChanged.connect(self._refresh_summary)
        self.mult_combo.currentTextChanged.connect(self._refresh_summary)
        self.codec_combo.currentTextChanged.connect(self._refresh_summary)
        self.rife_check.toggled.connect(self._refresh_summary)
        self.upscale_check.toggled.connect(self._refresh_summary)
        self.upscale_combo.currentTextChanged.connect(self._refresh_summary)
        self.rife_combo.currentTextChanged.connect(self._refresh_summary)
        self.blend_combo.currentTextChanged.connect(self._refresh_summary)
        self.res_pct_combo.currentTextChanged.connect(self._refresh_summary)

    def _short_codec_label(self, codec_name: str) -> str:
        if "NVENC" in codec_name:
            if "H.265" in codec_name:
                return "NVENC H.265"
            if "AV1" in codec_name:
                return "NVENC AV1"
            return "NVENC H.264"

        if "AMF" in codec_name:
            if "H.265" in codec_name:
                return "AMD AMF H.265"
            if "AV1" in codec_name:
                return "AMD AMF AV1"
            return "AMD AMF H.264"

        if "QSV" in codec_name:
            if "H.265" in codec_name:
                return "Intel QSV H.265"
            if "AV1" in codec_name:
                return "Intel QSV AV1"
            if "VP9" in codec_name:
                return "Intel QSV VP9"
            return "Intel QSV H.264"

        if "libx265" in codec_name or "H.265" in codec_name:
            return "CPU H.265"
        if "libx264" in codec_name or "H.264" in codec_name:
            return "CPU H.264"
        if "AV1" in codec_name:
            return "CPU AV1"

        return codec_name

    def _refresh_summary(self):
        width = self.w_spin.value()
        height = self.h_spin.value()
        fps = self.fps_combo.currentData()
        mult = self.mult_combo.currentData()

        rife_state = self._t("RIFE enabled") if self.rife_check.isChecked() else self._t("RIFE off")
        upscale_state = self._t("ESRGAN enabled") if self.upscale_check.isChecked() else self._t("ESRGAN off")
        
        blend = self.blend_combo.currentText()
        input_pct = self.res_pct_combo.currentText()

        self.summary_resolution.setText(f"{width} × {height}")
        self.summary_fps.setText(f"{fps} FPS × {mult}")
        self.summary_codec.setText(self._short_codec_label(self.codec_combo.currentText()))
        self.summary_pipeline.setText(
            f"{rife_state}, {upscale_state}, {self._t('blend')} {blend}, {self._t('input')} {input_pct}%"
        )

        if getattr(self, "preview_images", None):
            self._show_preview_index()
            return

        if self.frames_folder or self.output_file:
            frames_text = self.frames_folder if self.frames_folder else self._t("No frames folder selected")
            output_text = self.output_file if self.output_file else self._t("No output file selected")
            self.preview_placeholder.setText(
                f"{self._t('Frames:')}\n{frames_text}\n\n{self._t('Output:')}\n{output_text}"
            )
        else:
            self.preview_placeholder.setText(self._t("No preview loaded yet"))

    def _set_state(self, state_text: str):
        self.state_badge.setText(self._t(state_text).upper())

    def apply_theme(self, theme: dict):
        self._active_theme = theme
        self._apply_modern_style()

    def _set_input_video_file(self, path: str):
        self.input_video_file = path
        self.input_video_edit.setText(path)
        self._refresh_summary()

    def _browse_input_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._t("Select Input Video"),
            "",
            "Video (*.mp4 *.mkv *.avi *.mov *.webm);;All Files (*.*)",
        )

        if path:
            self._set_input_video_file(path)
            self.status_label.setText(self._t("Input video selected."))
            self._set_state("Ready")

    def _set_frames_folder(self, path: str):
        self.frames_folder = path
        self.frames_edit.setText(path)
        self._clear_preview_cache()
        self._refresh_summary()
        
    def _set_output_file(self, path: str):
        self.output_file = path
        self.output_edit.setText(path)
        self._refresh_summary()

    def _browse_frames(self):
        path = QFileDialog.getExistingDirectory(self, self._t("Select Frames Folder"))
        if path:
            self._set_frames_folder(path)
            self.status_label.setText(self._t("Frames folder selected."))
            self._set_state("Ready")

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._t("Save Output Video"),
            "",
            "MKV (*.mkv);;MP4 (*.mp4);;MOV (*.mov);;AVI (*.avi);;All (*.*)"
        )

        if path:
            self._set_output_file(path)
            self.status_label.setText(self._t("Output file selected."))
            self._set_state("Ready")

    def _preview_frame_files(self):
        if not self.frames_folder or not os.path.isdir(self.frames_folder):
            return []

        files = [
            os.path.join(self.frames_folder, name)
            for name in os.listdir(self.frames_folder)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]

        try:
            from core.merged_pipeline import natural_sort
            return natural_sort(files)
        except Exception:
            return sorted(files)

    def _pick_preview_files(self, count=5):
        files = self._preview_frame_files()

        if not files:
            return []

        if len(files) <= count:
            return files

        positions = [0.0, 0.25, 0.50, 0.75, 0.95]
        indexes = []

        for pos in positions:
            idx = int(round((len(files) - 1) * pos))
            idx = max(0, min(len(files) - 1, idx))
            indexes.append(idx)

        # keep order, remove duplicates
        picked = []
        seen = set()

        for idx in indexes:
            if idx not in seen:
                picked.append(files[idx])
                seen.add(idx)

        return picked

    def _clear_preview_cache(self):
        self.preview_images = []
        self.preview_index = 0
        self.preview_zoom = 1.0
        self.preview_focus_side = "sbs"

        if hasattr(self, "preview_counter_label"):
            self.preview_counter_label.setText("Preview 0 / 0")

        if hasattr(self, "preview_zoom_label"):
            self.preview_zoom_label.setText("100%")

        if hasattr(self, "preview_placeholder"):
            self.preview_placeholder.clear()
            self.preview_placeholder.setText(self._t("No preview loaded yet"))
            self.preview_placeholder.setMinimumSize(640, 460)

    def _on_preview_ready(self, image_paths):
        self.preview_images = list(image_paths or [])
        self.preview_index = 0
        self._show_preview_index()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if getattr(self, "preview_images", None):
            self._show_preview_index()

    def _preview_source_pixmap(self):
        if not getattr(self, "preview_images", None):
            return QPixmap()

        self.preview_index = max(0, min(self.preview_index, len(self.preview_images) - 1))
        path = self.preview_images[self.preview_index]
        pixmap = QPixmap(path)

        if pixmap.isNull():
            return pixmap

        # At 100%, show the normal side-by-side comparison.
        if self.preview_zoom <= 1.01 or self.preview_focus_side == "sbs":
            return pixmap

        half_w = pixmap.width() // 2

        if self.preview_focus_side == "original":
            return pixmap.copy(0, 0, half_w, pixmap.height())

        if self.preview_focus_side == "preview":
            return pixmap.copy(half_w, 0, pixmap.width() - half_w, pixmap.height())

        return pixmap

    def _show_preview_index(self):
        if not getattr(self, "preview_images", None):
            if hasattr(self, "preview_counter_label"):
                self.preview_counter_label.setText("Preview 0 / 0")
            return

        pixmap = self._preview_source_pixmap()

        if pixmap.isNull():
            self.preview_placeholder.clear()
            self.preview_placeholder.setText(self._t("Could not load preview image."))
            return

        viewport_size = self.preview_scroll.viewport().size()

        if viewport_size.width() <= 0 or viewport_size.height() <= 0:
            viewport_size = self.preview_placeholder.size()

        base_scale = min(
            viewport_size.width() / max(1, pixmap.width()),
            viewport_size.height() / max(1, pixmap.height()),
        )

        base_scale = max(0.01, base_scale)

        target_w = max(1, int(pixmap.width() * base_scale * self.preview_zoom))
        target_h = max(1, int(pixmap.height() * base_scale * self.preview_zoom))

        scaled = pixmap.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.preview_placeholder.clear()
        self.preview_placeholder.setPixmap(scaled)
        self.preview_placeholder.setFixedSize(scaled.size())

        self.preview_counter_label.setText(
            f"Preview {self.preview_index + 1} / {len(self.preview_images)}"
        )

        if hasattr(self, "preview_zoom_label"):
            side_text = ""
            if self.preview_focus_side == "original" and self.preview_zoom > 1.01:
                side_text = " Original"
            elif self.preview_focus_side == "preview" and self.preview_zoom > 1.01:
                side_text = " Preview"

            self.preview_zoom_label.setText(f"{int(self.preview_zoom * 100)}%{side_text}")

    def _event_point(self, event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _preview_side_from_pos(self, label_pos):
        # At normal fit view, the comparison image is Original | Preview.
        # Left half = original, right half = generated preview.
        label_w = max(1, self.preview_placeholder.width())

        if label_pos.x() < label_w / 2:
            return "original"

        return "preview"

    def _clamp01(self, value):
        return max(0.0, min(1.0, float(value)))

    def _preview_anchor_from_pos(self, label_pos):
        """
        Returns:
            side, x_ratio, y_ratio

        x_ratio/y_ratio represent the mouse position inside the visible image/side,
        so after zoom we can scroll back to that same pixel area.
        """
        label_w = max(1, self.preview_placeholder.width())
        label_h = max(1, self.preview_placeholder.height())

        x = self._clamp01(label_pos.x() / label_w)
        y = self._clamp01(label_pos.y() / label_h)

        # When fitted at 100%, the displayed image is Original | Preview.
        # Left half zooms original, right half zooms preview.
        if self.preview_zoom <= 1.01 or self.preview_focus_side == "sbs":
            if x < 0.5:
                side = "original"
                side_x = self._clamp01(x / 0.5)
            else:
                side = "preview"
                side_x = self._clamp01((x - 0.5) / 0.5)

            return side, side_x, y

        # Already zoomed into one side. Keep that side.
        return self.preview_focus_side, x, y

    def _scroll_preview_to_anchor(self, anchor_x, anchor_y, viewport_point):
        """
        After zooming, scroll so the pixel under the mouse stays under the mouse.
        """
        hbar = self.preview_scroll.horizontalScrollBar()
        vbar = self.preview_scroll.verticalScrollBar()

        content_w = max(1, self.preview_placeholder.width())
        content_h = max(1, self.preview_placeholder.height())

        target_x = int((anchor_x * content_w) - viewport_point.x())
        target_y = int((anchor_y * content_h) - viewport_point.y())

        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), target_x)))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), target_y)))
        
    def eventFilter(self, obj, event):
        preview_label = getattr(self, "preview_placeholder", None)
        preview_scroll = getattr(self, "preview_scroll", None)
        preview_viewport = preview_scroll.viewport() if preview_scroll is not None else None

        if obj not in (preview_label, preview_viewport):
            return super().eventFilter(obj, event)

        if not getattr(self, "preview_images", None):
            return super().eventFilter(obj, event)

        # Convert mouse position to viewport + label coordinates.
        raw_point = self._event_point(event) if hasattr(event, "pos") or hasattr(event, "position") else None

        def _map_point_between(source_widget, target_widget, point):
            if point is None or source_widget is None or target_widget is None:
                return None

            global_point = source_widget.mapToGlobal(point)
            return target_widget.mapFromGlobal(global_point)

        def to_viewport_point(point):
            if point is None:
                return None

            if obj is preview_viewport:
                return point

            return _map_point_between(preview_label, preview_viewport, point)

        def to_label_point(point):
            if point is None:
                return None

            if obj is preview_label:
                return point

            return _map_point_between(preview_viewport, preview_label, point)

        # Mouse wheel zoom
        if event.type() == QEvent.Type.Wheel:
            label_point = to_label_point(raw_point)
            viewport_point = to_viewport_point(raw_point)

            if label_point is None or viewport_point is None:
                return super().eventFilter(obj, event)

            side, anchor_x, anchor_y = self._preview_anchor_from_pos(label_point)

            # Pick side on first zoom from 100%.
            if self.preview_zoom <= 1.01:
                self.preview_focus_side = side

            delta = event.angleDelta().y()

            if delta > 0:
                self.preview_zoom = min(6.0, self.preview_zoom * 1.25)
            else:
                self.preview_zoom = max(1.0, self.preview_zoom / 1.25)

            if self.preview_zoom <= 1.01:
                self.preview_zoom = 1.0
                self.preview_focus_side = "sbs"
                self._show_preview_index()
            else:
                self._show_preview_index()
                self._scroll_preview_to_anchor(anchor_x, anchor_y, viewport_point)

            event.accept()
            return True

        # Click + drag pan
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self.preview_dragging = True
            self.preview_drag_start = to_viewport_point(raw_point)

            hbar = self.preview_scroll.horizontalScrollBar()
            vbar = self.preview_scroll.verticalScrollBar()

            self.preview_drag_h_start = hbar.value()
            self.preview_drag_v_start = vbar.value()

            self.preview_placeholder.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.preview_scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

            event.accept()
            return True

        if event.type() == QEvent.Type.MouseMove and self.preview_dragging:
            current_point = to_viewport_point(raw_point)

            if current_point is None or self.preview_drag_start is None:
                return True

            dx = current_point.x() - self.preview_drag_start.x()
            dy = current_point.y() - self.preview_drag_start.y()

            hbar = self.preview_scroll.horizontalScrollBar()
            vbar = self.preview_scroll.verticalScrollBar()

            hbar.setValue(self.preview_drag_h_start - dx)
            vbar.setValue(self.preview_drag_v_start - dy)

            event.accept()
            return True

        if event.type() in (QEvent.Type.MouseButtonRelease, QEvent.Type.Leave):
            if self.preview_dragging:
                self.preview_dragging = False
                self.preview_drag_start = None

                self.preview_placeholder.setCursor(Qt.CursorShape.OpenHandCursor)
                self.preview_scroll.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

                event.accept()
                return True

        return super().eventFilter(obj, event)

    def _preview_zoom_reset(self):
        self.preview_zoom = 1.0
        self.preview_focus_side = "sbs"
        self.preview_dragging = False
        self.preview_drag_start = None

        if hasattr(self, "preview_zoom_label"):
            self.preview_zoom_label.setText("100%")

        if hasattr(self, "preview_placeholder"):
            self.preview_placeholder.setCursor(Qt.CursorShape.OpenHandCursor)

        if hasattr(self, "preview_scroll"):
            self.preview_scroll.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

        self._show_preview_index()
            
    def _preview_previous(self):
        if not self.preview_images:
            return

        self.preview_index = (self.preview_index - 1) % len(self.preview_images)
        self._show_preview_index()

    def _preview_next(self):
        if not self.preview_images:
            return

        self.preview_index = (self.preview_index + 1) % len(self.preview_images)
        self._show_preview_index()

    def _generate_preview(self):
        frame_files = self._pick_preview_files(count=5)

        if not frame_files:
            QMessageBox.warning(
                self,
                self._t("Missing frames"),
                self._t("Select a frames folder with images first."),
            )
            return

        settings = self._get_settings()

        preview_dir = os.path.join(self.frames_folder, "_vd3d_preview")
        os.makedirs(preview_dir, exist_ok=True)

        self._clear_preview_cache()
        self.status_label.setText(self._t("Generating preview..."))
        self._set_state("Preview")

        self._emit_job_progress(
            progress=0,
            status_text=self._t("Preparing preview frames..."),
            start_time=time.time(),
            completed_units=0,
            total_units=len(frame_files),
            rate_label="Frames/s",
            state="Preview",
        )

        def _run():
            start_time = time.time()

            try:
                import cv2
                import numpy as np
                from core.merged_pipeline import init_upscaler, run_esrgan

                target_size = (settings["width"], settings["height"])
                model_path = settings["model_path"]
                enable_upscale = bool(settings["enable_upscale"])

                if enable_upscale:
                    self._emit_job_progress(
                        progress=2,
                        status_text=self._t("Loading preview upscaler model..."),
                        start_time=start_time,
                        completed_units=0,
                        total_units=len(frame_files),
                        rate_label="Frames/s",
                        state="Preview",
                    )

                    if not init_upscaler(model_path, True):
                        raise RuntimeError(
                            f"{self._t('Failed to load/download upscaler model:')} {model_path}"
                        )

                output_paths = []
                total = len(frame_files)

                for index, frame_path in enumerate(frame_files, start=1):
                    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)

                    if img is None:
                        continue

                    original = cv2.resize(
                        img,
                        target_size,
                        interpolation=cv2.INTER_AREA if img.shape[1] > target_size[0] else cv2.INTER_CUBIC,
                    )

                    if enable_upscale:
                        processed = run_esrgan(
                            img,
                            settings["blend_mode"],
                            settings["input_res_pct"],
                            model_name=model_path,
                            target_size=target_size,
                        )
                    else:
                        processed = original.copy()

                    if processed is None:
                        processed = original.copy()

                    if processed.shape[:2] != original.shape[:2]:
                        processed = cv2.resize(processed, target_size, interpolation=cv2.INTER_CUBIC)

                    comparison = np.hstack([original, processed])

                    # Small readable labels
                    cv2.putText(
                        comparison,
                        "Original",
                        (24, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        comparison,
                        "Preview",
                        (target_size[0] + 24, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    out_path = os.path.join(preview_dir, f"preview_{index:02d}.jpg")
                    cv2.imwrite(out_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    output_paths.append(out_path)

                    progress = (index / max(total, 1)) * 100.0

                    self._emit_job_progress(
                        progress=progress,
                        status_text=f"{self._t('Generating preview')} {index}/{total}",
                        start_time=start_time,
                        completed_units=index,
                        total_units=total,
                        rate_label="Frames/s",
                        state="Preview",
                    )

                if not output_paths:
                    raise RuntimeError(self._t("No preview frames were created."))

                self.preview_ready.emit(output_paths)

                self._emit_job_progress(
                    progress=100,
                    status_text=self._t("Preview complete."),
                    start_time=start_time,
                    completed_units=len(output_paths),
                    total_units=len(output_paths),
                    rate_label="Frames/s",
                    state="Done",
                    notify_complete=False,
                )

                self.status_label.setText(
                    self._t("Preview generated. Use Previous / Next to compare samples.")
                )

            except Exception as exc:
                self._emit_job_progress(
                    progress=0,
                    status_text=f"{self._t('Preview failed:')} {exc}",
                    start_time=start_time,
                    completed_units=0,
                    total_units=max(len(frame_files), 1),
                    rate_label="Frames/s",
                    state="Error",
                    notify_error=True,
                    message_title=self._t("Preview Failed"),
                    message_text=str(exc),
                    traceback_text=traceback.format_exc(),
                )

        threading.Thread(target=_run, daemon=True).start()

    def _format_seconds(self, seconds):
        seconds = max(0, int(seconds or 0))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _emit_job_progress(
        self,
        *,
        progress=0.0,
        status_text="",
        start_time=None,
        completed_units=None,
        total_units=None,
        rate_label="FPS",
        state=None,
        notify_complete=False,
        notify_error=False,
        message_title="",
        message_text="",
        traceback_text=None,
    ):
        now = time.time()
        elapsed = 0.0
        eta = None
        fps_like = None

        if start_time is not None:
            elapsed = now - start_time

        if (
            start_time is not None
            and completed_units is not None
            and total_units is not None
            and completed_units > 0
        ):
            fps_like = completed_units / max(elapsed, 1e-6)
            remaining = max(0.0, total_units - completed_units)
            eta = remaining / fps_like if fps_like > 0 else None

        self.progress_updated.emit({
            "progress": float(max(0.0, min(100.0, progress))),
            "status_text": status_text,
            "elapsed": elapsed,
            "eta": eta,
            "fps_like": fps_like,
            "rate_label": rate_label,
            "state": state,
            "notify_complete": notify_complete,
            "notify_error": notify_error,
            "message_title": message_title,
            "message_text": message_text,
            "traceback": traceback_text,
        })

    def _parse_rate(self, rate_str):
        if not rate_str or rate_str == "0/0":
            return 0.0

        try:
            if "/" in rate_str:
                a, b = rate_str.split("/", 1)
                a = float(a)
                b = float(b)
                return a / b if b else 0.0

            return float(rate_str)
        except Exception:
            return 0.0

    def _probe_video_frame_count(self, video_path):
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,duration,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            video_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            **hidden_subprocess_kwargs(),
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffprobe failed.")

        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []

        if not streams:
            raise RuntimeError("No video stream found.")

        stream = streams[0]

        for key in ("nb_read_frames", "nb_frames"):
            value = stream.get(key)
            if value and str(value).isdigit():
                count = int(value)
                if count > 0:
                    return count

        duration = float(stream.get("duration") or 0.0)
        fps = self._parse_rate(stream.get("avg_frame_rate")) or self._parse_rate(stream.get("r_frame_rate"))

        if duration > 0 and fps > 0:
            return max(1, int(duration * fps))

        return 1

    def _extract_frames(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            self._t("Select Video to Extract Frames"),
            "",
            "Video (*.mp4 *.mkv *.avi *.mov *.webm);;All Files (*.*)",
        )

        if not video_path:
            return
        
        self._set_input_video_file(video_path)

        output_dir = QFileDialog.getExistingDirectory(
            self,
            self._t("Select Output Folder for Frames"),
        )

        if not output_dir:
            return

        image_format, ok = QInputDialog.getItem(
            self,
            self._t("Select Frame Format"),
            self._t("Choose image format for extracted frames:"),
            ["png", "jpg", "jpeg", "bmp", "webp"],
            0,
            False,
        )

        if not ok or not image_format:
            return

        image_format = image_format.lower().strip()

        frames_dir = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(video_path))[0] + f"_{image_format}_frames",
        )
        
        os.makedirs(frames_dir, exist_ok=True)

        self._set_frames_folder(frames_dir)
        self.status_label.setText(self._t("Extracting frames..."))
        self._set_state("Extract")

        self._emit_job_progress(
            progress=0,
            status_text=self._t("Preparing frame extraction..."),
            start_time=time.time(),
            completed_units=0,
            total_units=1,
            state="Extract",
        )

        def _run():
            start_time = time.time()

            try:
                total_frames = self._probe_video_frame_count(video_path)
                output_pattern = os.path.join(frames_dir, f"frame_%06d.{image_format}")

                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-y",
                    "-i",
                    video_path,
                    "-vsync",
                    "0",
                ]

                if image_format in ("jpg", "jpeg", "webp"):
                    cmd += ["-q:v", "2"]

                cmd += [
                    output_pattern,
                    "-progress",
                    "pipe:1",
                    "-nostats",
                ]

                print("[FRAME EXTRACT CMD]", " ".join(str(x) for x in cmd))

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    **hidden_subprocess_kwargs(),
                )

                current_frame = 0

                if proc.stdout:
                    for line in proc.stdout:
                        line = line.strip()

                        if line.startswith("frame="):
                            try:
                                current_frame = int(line.split("=", 1)[1])
                            except Exception:
                                continue

                            progress = (current_frame / max(total_frames, 1)) * 100.0

                            self._emit_job_progress(
                                progress=progress,
                                status_text=f"{self._t('Extracting frames')} {current_frame}/{total_frames}",
                                start_time=start_time,
                                completed_units=current_frame,
                                total_units=total_frames,
                                rate_label="FPS",
                                state="Extract",
                            )

                return_code = proc.wait()

                stderr_text = ""
                try:
                    if proc.stderr:
                        stderr_text = proc.stderr.read()
                except Exception:
                    pass

                if return_code != 0:
                    raise RuntimeError(stderr_text[-4000:] or f"ffmpeg exited with code {return_code}")

                extracted_count = len([
                    name for name in os.listdir(frames_dir)
                    if name.lower().endswith((".png", ".jpg", ".jpeg"))
                ])

                self._emit_job_progress(
                    progress=100,
                    status_text=f"{self._t('Frame extraction complete.')} {extracted_count} {self._t('frames extracted.')}",
                    start_time=start_time,
                    completed_units=max(extracted_count, total_frames),
                    total_units=max(extracted_count, total_frames, 1),
                    rate_label="FPS",
                    state="Done",
                    notify_complete=True,
                    message_title=self._t("Frame Extraction Complete"),
                    message_text=f"{self._t('Extracted frames to:')}\n{frames_dir}",
                )

            except Exception as exc:
                self._emit_job_progress(
                    progress=0,
                    status_text=f"{self._t('Frame extraction failed:')} {exc}",
                    start_time=start_time,
                    completed_units=0,
                    total_units=1,
                    rate_label="FPS",
                    state="Error",
                    notify_error=True,
                    message_title=self._t("Frame Extraction Failed"),
                    message_text=str(exc),
                    traceback_text=traceback.format_exc(),
                )

        threading.Thread(target=_run, daemon=True).start()

    def _get_settings(self):
        return {
            "input_video_file": self.input_video_file,
            "keep_original_audio": self.keep_audio_check.isChecked(),
            "frames_folder": self.frames_folder,
            "output_file": self.output_file,
            "width": self.w_spin.value(),
            "height": self.h_spin.value(),
            "fps": self.fps_combo.currentData(),
            "fps_multiplier": self.mult_combo.currentData(),
            "codec": FFMPEG_CODEC_MAP.get(
                self.codec_combo.currentText(),
                "h264_nvenc"
            ),
            "enable_rife": self.rife_check.isChecked(),
            "enable_upscale": self.upscale_check.isChecked(),
            "blend_mode": self.blend_combo.currentText(),
            "input_res_pct": int(self.res_pct_combo.currentText()),
            "rife_model": _load_rife_models().get(
                self.rife_combo.currentText(),
                "rife:FuryTMP/RIFE_fp32"
            ),
            "model_path": _load_upscaler_models().get(
                self.upscale_combo.currentText(),
                "upscale:FuryTMP/RealESR_Gx4_fp16"
            ),
        }

    def _validate_paths(self) -> bool:
        if not self.frames_folder or not self.output_file:
            QMessageBox.warning(
                self,
                self._t("Missing paths"),
                self._t("Select a frames folder and output file first.")
            )
            return False

        return True

    def _start_processing(self):
        if not self._validate_paths():
            return

        settings = self._get_settings()

        self.status_label.setText(self._t("Processing..."))
        self._set_state("Running")

        self.progress_updated.emit({
            "progress": 0,
            "status_text": self._t("Starting standard pipeline...")
        })

        def _run():
            try:
                from core.merged_pipeline import start_merged_pipeline

                proxy = _TkProgressProxy(
                    lambda p: self.progress_updated.emit(p)
                )
                start_merged_pipeline(settings, proxy, proxy)

                self.progress_updated.emit({
                    "progress": 100,
                    "status_text": self._t("Done.")
                })

            except Exception as exc:
                self.progress_updated.emit({
                    "progress": 0,
                    "status_text": f"{self._t('Error:')} {exc}",
                    "state": "Error",
                    "traceback": traceback.format_exc(),
                })

        threading.Thread(target=_run, daemon=True).start()

    def _start_threaded(self):
        if not self._validate_paths():
            return

        settings = self._get_settings()

        self.status_label.setText(self._t("Processing with threaded RIFE + ESRGAN..."))
        self._set_state("Running")

        self.progress_updated.emit({
            "progress": 0,
            "status_text": self._t("Starting threaded pipeline...")
        })

        def _run():
            try:
                from core.merged_pipeline import start_threaded_pipeline

                proxy = _TkProgressProxy(
                    lambda p: self.progress_updated.emit(p)
                )
                start_threaded_pipeline(settings, proxy, proxy)

                self.progress_updated.emit({
                    "progress": 100,
                    "status_text": self._t( "Done.")
                })

            except Exception as exc:
                self.progress_updated.emit({
                    "progress": 0,
                    "status_text": f"{self._t('Error:')} {exc}",
                    "state": "Error",
                    "traceback": traceback.format_exc(),
                })

        threading.Thread(target=_run, daemon=True).start()

    def _pause(self):
        from core.merged_pipeline import request_upscale_pause

        request_upscale_pause()

        self.status_label.setText(self._t("Paused"))
        self._set_state("Paused")

    def _resume(self):
        from core.merged_pipeline import request_upscale_resume

        request_upscale_resume()

        self.status_label.setText(self._t("Resuming..."))
        self._set_state("Running")

    def _stop(self):
        from core.merged_pipeline import request_upscale_stop

        request_upscale_stop()

        self.status_label.setText(self._t("Stopping..."))
        self._set_state("Stopping")

    def _detect_scenes(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._t("Select Video for Scene Detection"),
            "",
            "Video (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )

        if not path:
            return

        out_dir = QFileDialog.getExistingDirectory(
            self,
            self._t("Select Output Folder for Scenes")
        )

        if not out_dir:
            return

        threshold = self.scene_slider.value()
        fmt = self.scene_fmt_combo.currentText()

        self.status_label.setText(self._t("Detecting scenes..."))
        self._set_state("Detect")

        self.progress_updated.emit({
            "progress": 0,
            "status_text": self._t("Preparing scene detection..."),
            "elapsed": 0,
            "eta": None,
            "fps_like": None,
            "rate_label": "Scenes/s",
            "state": "Detect",
        })

        def _run():
            start_time = time.time()
            total = 0
            exported = 0

            try:
                from scenedetect import open_video, SceneManager
                from scenedetect.detectors import ContentDetector

                self._emit_job_progress(
                    progress=2,
                    status_text=self._t("Scanning video for scene changes..."),
                    start_time=start_time,
                    completed_units=0,
                    total_units=1,
                    rate_label="Scenes/s",
                    state="Detect",
                )

                video = open_video(path)

                sm = SceneManager()
                sm.add_detector(ContentDetector(threshold=threshold))
                sm.detect_scenes(video)

                scenes = sm.get_scene_list()
                fps = video.frame_rate
                total = len(scenes)

                if not scenes:
                    self._emit_job_progress(
                        progress=100,
                        status_text=self._t("No scenes detected."),
                        start_time=start_time,
                        completed_units=1,
                        total_units=1,
                        rate_label="Scenes/s",
                        state="Ready",
                        notify_complete=True,
                        message_title=self._t("Scene Detection Complete"),
                        message_text=self._t("No scenes were detected in this video."),
                    )
                    return

                self._emit_job_progress(
                    progress=10,
                    status_text=f"{self._t('Detected')} {total} {self._t('scenes.')} {self._t('Exporting...')}",
                    start_time=start_time,
                    completed_units=0,
                    total_units=total,
                    rate_label="Scenes/s",
                    state="Export",
                )

                for i, (start, end) in enumerate(scenes, start=1):
                    t0 = start.get_frames() / fps
                    dur = (end.get_frames() - start.get_frames()) / fps
                    out = os.path.join(out_dir, f"scene_{i:03d}.{fmt}")

                    status = (
                        f"{self._t('Exporting scene')} {i}/{total} "
                        f"({dur:.2f}s)"
                    )

                    # progress before this scene starts
                    pre_progress = 10.0 + ((i - 1) / max(total, 1)) * 90.0
                    self._emit_job_progress(
                        progress=pre_progress,
                        status_text=status,
                        start_time=start_time,
                        completed_units=i - 1,
                        total_units=total,
                        rate_label="Scenes/s",
                        state="Export",
                    )

                    cmd = [
                        "ffmpeg",
                        "-hide_banner",
                        "-y",
                        "-ss",
                        f"{t0:.3f}",
                        "-i",
                        path,
                        "-t",
                        f"{dur:.3f}",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "18",
                        "-preset",
                        "fast",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        out,
                    ]

                    print("[SCENE EXPORT CMD]", " ".join(str(x) for x in cmd))

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        **hidden_subprocess_kwargs(),
                    )

                    if result.returncode != 0:
                        raise RuntimeError(
                            f"Failed exporting scene {i}/{total}:\n"
                            f"{result.stderr[-4000:]}"
                        )

                    exported = i
                    progress = 10.0 + (exported / max(total, 1)) * 90.0

                    self._emit_job_progress(
                        progress=progress,
                        status_text=f"{self._t('Exported scene')} {exported}/{total}",
                        start_time=start_time,
                        completed_units=exported,
                        total_units=total,
                        rate_label="Scenes/s",
                        state="Export",
                    )

                self._emit_job_progress(
                    progress=100,
                    status_text=f"{self._t('Exported')} {exported} {self._t('scenes.')}",
                    start_time=start_time,
                    completed_units=exported,
                    total_units=total,
                    rate_label="Scenes/s",
                    state="Done",
                    notify_complete=True,
                    message_title=self._t("Scene Detection Complete"),
                    message_text=(
                        f"{self._t('Exported')} {exported} {self._t('scenes.')}\n\n"
                        f"{out_dir}"
                    ),
                )

            except Exception as exc:
                self._emit_job_progress(
                    progress=0,
                    status_text=f"{self._t('Scene detection error:')} {exc}",
                    start_time=start_time,
                    completed_units=exported,
                    total_units=max(total, 1),
                    rate_label="Scenes/s",
                    state="Error",
                    notify_error=True,
                    message_title=self._t("Scene Detection Failed"),
                    message_text=str(exc),
                    traceback_text=traceback.format_exc(),
                )

        threading.Thread(target=_run, daemon=True).start()
        
    def _on_progress_updated(self, payload: dict):
        status_text = payload.get("status_text")
        state = payload.get("state")

        if status_text:
            self.status_label.setText(status_text)

        if state:
            self._set_state(state)
        elif payload.get("progress") == 100:
            self._set_state("Done")

        tb = payload.get("traceback")
        if tb:
            print(tb)

        if payload.get("notify_complete"):
            QMessageBox.information(
                self,
                payload.get("message_title") or self._t("Done"),
                payload.get("message_text") or status_text or self._t("Done."),
            )

        if payload.get("notify_error"):
            QMessageBox.critical(
                self,
                payload.get("message_title") or self._t("Error"),
                payload.get("message_text") or status_text or self._t("Error"),
            )
