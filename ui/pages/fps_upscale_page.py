# ui/pages/fps_upscale_page.py
import os
import threading
import traceback

from PySide6.QtCore import Qt, Signal
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
    QFrame,
    QGridLayout,
    QProgressBar,
    QSizePolicy,
)

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
    23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0,
    59.94, 60.0, 72.0, 90.0, 100.0, 119.88, 120.0,
    144.0, 165.0, 240.0,
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
        self._value = 0

    def config(self, **kw):
        if "text" in kw and self._callback:
            self._callback({"status_text": str(kw["text"])})

    def configure(self, **kw):
        self.config(**kw)

    def after(self, ms, fn):
        import threading as _th
        t = _th.Timer(ms / 1000.0, fn)
        t.daemon = True
        t.start()

    def __setitem__(self, key, value):
        if key == "value":
            self._value = value
            if self._callback:
                self._callback({"progress": float(value)})

    def __getitem__(self, key):
        if key == "maximum":
            return 100
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

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._translation_map = []

        self.frames_folder = ""
        self.output_file = ""

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

        self._build_ui()
        self.progress_updated.connect(self._on_progress_updated)

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

    def _register_text(self, widget, key: str):
        self._translation_map.append((widget, key, "text"))
        widget.setText(self._t(key))

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
                    widget.setText(self._t(key))
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
        self.setStyleSheet("""
            QWidget {
                background-color: #0b0f14;
                color: #e6edf3;
                font-family: "Segoe UI";
                font-size: 13px;
            }

            QLabel {
                background: transparent;
            }

            QLabel#PageTitle {
                font-size: 22px;
                font-weight: 800;
                color: #f0f6fc;
            }

            QLabel#PageSubtitle {
                font-size: 13px;
                color: #8b949e;
            }

            QLabel#SectionHeaderLabel {
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.5px;
                color: #7d8590;
                text-transform: uppercase;
            }

            QLabel#CardTitle {
                font-size: 14px;
                font-weight: 800;
                color: #f0f6fc;
            }

            QLabel#CardSubtitle {
                color: #8b949e;
                font-size: 12px;
            }

            QLabel#MutedLabel {
                color: #8b949e;
            }

            QLabel#StatusPill {
                background-color: #102033;
                border: 1px solid #1f6feb;
                border-radius: 12px;
                padding: 6px 10px;
                color: #9ecbff;
                font-weight: 700;
            }

            QLabel#PreviewPlaceholder {
                background-color: #0d131b;
                border: 1px dashed #2d3b4f;
                border-radius: 16px;
                color: #8b949e;
                font-size: 14px;
            }

            QLabel#StatusLabel {
                color: #d7dce2;
                padding: 12px 14px;
                border: 1px solid #263445;
                border-radius: 12px;
                background-color: #0f1724;
                font-size: 13px;
            }

            QFrame#ModernCard {
                background-color: #111821;
                border: 1px solid #263445;
                border-radius: 16px;
            }

            QGroupBox {
                background-color: #111821;
                border: 1px solid #263445;
                border-radius: 15px;
                margin-top: 12px;
                padding: 14px;
                font-weight: 800;
                color: #f0f6fc;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 7px;
                color: #f0f6fc;
            }

            QLineEdit,
            QComboBox,
            QSpinBox {
                background-color: #0d131b;
                border: 1px solid #2d3b4f;
                border-radius: 10px;
                padding: 8px 10px;
                color: #e6edf3;
                min-height: 22px;
            }

            QLineEdit:hover,
            QComboBox:hover,
            QSpinBox:hover {
                border: 1px solid #3d8bfd;
            }

            QLineEdit:focus,
            QComboBox:focus,
            QSpinBox:focus {
                border: 1px solid #58a6ff;
            }

            QLineEdit:read-only {
                color: #aab6c5;
            }

            QComboBox::drop-down {
                border: none;
                width: 26px;
            }

            QPushButton {
                background-color: #1b2533;
                border: 1px solid #2d3b4f;
                border-radius: 11px;
                padding: 9px 14px;
                color: #f0f6fc;
                font-weight: 700;
            }

            QPushButton:hover {
                background-color: #243247;
                border: 1px solid #3d8bfd;
            }

            QPushButton:pressed {
                background-color: #172033;
            }

            QPushButton#PrimaryButton {
                background-color: #2f81f7;
                border: 1px solid #58a6ff;
                color: white;
                font-weight: 800;
            }

            QPushButton#PrimaryButton:hover {
                background-color: #1f6feb;
            }

            QPushButton#DangerButton {
                background-color: #2b1518;
                border: 1px solid #6e2630;
                color: #ffb4b4;
            }

            QPushButton#DangerButton:hover {
                background-color: #421b22;
                border: 1px solid #ff7b72;
            }

            QPushButton#IconButton {
                min-height: 34px;
                font-size: 14px;
            }

            QCheckBox {
                spacing: 8px;
                color: #e6edf3;
                background: transparent;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 1px solid #3b4658;
                background-color: #0d131b;
            }

            QCheckBox::indicator:hover {
                border: 1px solid #58a6ff;
            }

            QCheckBox::indicator:checked {
                background-color: #2f81f7;
                border: 1px solid #58a6ff;
            }

            QSlider::groove:horizontal {
                height: 6px;
                background: #0d131b;
                border: 1px solid #2d3b4f;
                border-radius: 3px;
            }

            QSlider::handle:horizontal {
                background: #58a6ff;
                border: 1px solid #9ecbff;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }

            QProgressBar {
                background-color: #0d131b;
                border: 1px solid #2d3b4f;
                border-radius: 10px;
                height: 18px;
                text-align: center;
                color: #f0f6fc;
                font-weight: 700;
            }

            QProgressBar::chunk {
                background-color: #2f81f7;
                border-radius: 9px;
            }

            QScrollArea {
                border: none;
                background: transparent;
            }

            QScrollBar:vertical {
                background: #0b0f14;
                width: 10px;
                margin: 2px;
            }

            QScrollBar::handle:vertical {
                background: #263445;
                border-radius: 5px;
                min-height: 32px;
            }

            QScrollBar::handle:vertical:hover {
                background: #34465e;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

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
        left_scroll.setMinimumWidth(390)
        left_scroll.setMaximumWidth(460)

        left_widget = QWidget()
        left_widget.setMinimumWidth(360)

        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        left_layout.addWidget(self._section("Input / Output"))

        self.extract_btn = self._button("📂 Extract Frames from Video")
        self.extract_btn.setMinimumHeight(38)
        self.extract_btn.clicked.connect(self._extract_frames)
        left_layout.addWidget(self.extract_btn)

        io_group = self._group("Paths")
        io_layout = QVBoxLayout(io_group)
        io_layout.setSpacing(10)

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

        io_layout.addLayout(frames_row)
        io_layout.addLayout(out_row)
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

        scene_group = self._group("Scene Detection")
        scene_layout = QVBoxLayout(scene_group)
        scene_layout.setSpacing(10)

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

        self.detect_scenes_btn = self._button("🔍 Detect Scenes & Extract")
        self.detect_scenes_btn.setMinimumHeight(38)
        self.detect_scenes_btn.clicked.connect(self._detect_scenes)

        scene_layout.addLayout(scene_thresh_row)
        scene_layout.addLayout(scene_fmt_row)
        scene_layout.addWidget(self.detect_scenes_btn)

        left_layout.addWidget(scene_group)
        left_layout.addStretch()

        left_scroll.setWidget(left_widget)
        top_row.addWidget(left_scroll)

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
        render_grid.setHorizontalSpacing(14)
        render_grid.setVerticalSpacing(10)
        render_grid.setColumnStretch(1, 1)

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

        render_grid.addWidget(self._label("Resolution:"), 0, 0)
        render_grid.addWidget(self.summary_resolution, 0, 1)

        render_grid.addWidget(self._label("Frame Rate:"), 1, 0)
        render_grid.addWidget(self.summary_fps, 1, 1)

        render_grid.addWidget(self._label("Codec:"), 2, 0)
        render_grid.addWidget(self.summary_codec, 2, 1)

        render_grid.addWidget(self._label("Pipeline:"), 3, 0)
        render_grid.addWidget(self.summary_pipeline, 3, 1)

        render_grid.addWidget(self._label("Models:"), 4, 0)
        render_grid.addWidget(self.summary_model, 4, 1)

        render_card.layout.addLayout(render_grid)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        render_card.layout.addWidget(self.progress_bar)

        center_layout.addWidget(render_card)

        preview_card = self._card(
            "Preview / Job Output",
            "This space can be used for a preview frame, render logs, or diagnostics later."
        )

        self.preview_placeholder = QLabel()
        self._register_text(self.preview_placeholder, "No preview loaded yet")
        self.preview_placeholder.setObjectName("PreviewPlaceholder")
        self.preview_placeholder.setAlignment(Qt.AlignCenter)
        self.preview_placeholder.setMinimumHeight(260)
        self.preview_placeholder.setWordWrap(True)

        preview_card.layout.addWidget(self.preview_placeholder, 1)

        center_layout.addWidget(preview_card, 1)

        top_row.addWidget(center_widget, 1)

        right_widget = QWidget()
        right_widget.setMinimumWidth(330)
        right_widget.setMaximumWidth(380)

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

        top_row.addWidget(right_widget)

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

    def _set_frames_folder(self, path: str):
        self.frames_folder = path
        self.frames_edit.setText(path)
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

    def _extract_frames(self):
        from core.merged_pipeline import select_video_and_generate_frames

        self.status_label.setText(self._t("Opening frame extraction workflow..."))
        self._set_state("Extract")

        select_video_and_generate_frames(
            set_folder_callback=lambda p: self._set_frames_folder(p)
        )

        self.status_label.setText(self._t("Frame extraction folder updated."))
        self._set_state("Ready")

    def _get_settings(self):
        return {
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
        self.progress_bar.setValue(0)
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
        self.progress_bar.setValue(0)
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
            "Video (*.mp4 *.avi *.mov *.mkv);;All (*.*)"
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
        self.progress_bar.setValue(0)
        self._set_state("Detect")

        def _run():
            try:
                from scenedetect import open_video, SceneManager
                from scenedetect.detectors import ContentDetector
                import subprocess

                video = open_video(path)

                sm = SceneManager()
                sm.add_detector(ContentDetector(threshold=threshold))
                sm.detect_scenes(video)

                scenes = sm.get_scene_list()
                fps = video.frame_rate

                if not scenes:
                    self.progress_updated.emit({
                        "progress": 100,
                        "status_text": self._t("No scenes detected."),
                        "state": "Ready",
                    })
                    return

                total = len(scenes)

                for i, (start, end) in enumerate(scenes):
                    t0 = start.get_frames() / fps
                    dur = (end.get_frames() - start.get_frames()) / fps
                    out = os.path.join(out_dir, f"scene_{i + 1:03d}.{fmt}")

                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-hwaccel",
                            "auto",
                            "-i",
                            path,
                            "-ss",
                            f"{t0:.3f}",
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
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        **hidden_subprocess_kwargs(),
                    )

                    progress = int(((i + 1) / total) * 100)

                    self.progress_updated.emit({
                        "progress": 0,
                        "status_text": self._t("Starting threaded pipeline...")
                    })

                self.progress_updated.emit({
                    "progress": 100,
                    "status_text": self._t("Done.")
                })

            except Exception as exc:
                self.progress_updated.emit({
                    "progress": 100,
                    "status_text": f"{self._t('Exported')} {total} {self._t('scenes.')}",
                    "state": "Ready",
                    "traceback": traceback.format_exc(),
                })

        threading.Thread(target=_run, daemon=True).start()

    def _on_progress_updated(self, payload: dict):
        progress = payload.get("progress")
        status_text = payload.get("status_text")
        state = payload.get("state")

        if progress is not None:
            try:
                self.progress_bar.setValue(int(progress))
            except ValueError:
                pass

        if status_text:
            self.status_label.setText(status_text)

        if state:
            self._set_state(state)
        elif progress == 100:
            self._set_state("Done")

        tb = payload.get("traceback")
        if tb:
            print(tb)
