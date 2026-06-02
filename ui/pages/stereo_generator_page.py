from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QDoubleSpinBox,
    QSlider,
    QSplitter,
    QMessageBox,
    QCheckBox,
    QGridLayout,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QDialog,
    QGroupBox,
    QFormLayout,
    QTabWidget,
)
from ui.widgets.file_picker_row import FilePickerRow
from ui.widgets.parameter_card import ParameterCard
from ui.widgets.preview_panel import PreviewPanel
from ui.styles.page_theme import apply_unified_page_theme

import os
from pathlib import Path

from services.keyframe_service import (
    KeyframeService,
    create_default_keyframe_path,
)

ASPECT_RATIO_OPTIONS = [
    "Default (16:9 / 1.78:1)",
    "Classic (4:3 / 1.33:1)",
    "Square (1:1 / 1.00:1)",
    "Vertical 9:16 / 0.56:1",
    "Instagram 4:5 / 0.80:1",
    "3:2 Photography / 1.50:1",
    "5:4 / 1.25:1",
    "7:5 / 1.40:1",
    "Academy Flat (1.85:1)",
    "European Flat (1.66:1)",
    "2:1 (Modern Hybrid)",
    "CinemaScope (2.39:1)",
    "Anamorphic (2.35:1)",
    "Modern Cinema (2.40:1)",
    "Ultra Panavision (2.76:1)",
    "21:9 UltraWide / 2.33:1",
    "32:9 SuperWide / 3.56:1",
]

VR180_EQUI_PRESETS = {
    "2048x1024 (Per Eye)": (2048, 1024),
    "3072x1536 (Per Eye)": (3072, 1536),
    "3840x1920 (Per Eye)": (3840, 1920),
    "4096x2048 (Per Eye)": (4096, 2048),
    "5760x2880 (Per Eye)": (5760, 2880),
}

VR180_FLAT_PRESETS = {
    "1280x720 (Working)": (1280, 720),
    "1920x1080 (Working)": (1920, 1080),
    "2560x1440 (Working)": (2560, 1440),
}

ENCODING_PRESETS = {
    "Compatibility Mode - Plays Everywhere": {
        "description": "Safest output for TVs, projectors, media boxes, VLC, Plex, Jellyfin, and general playback.",
        "use_ffmpeg": True,
        "ffmpeg_codec": "H.264 / AVC (NVENC - NVIDIA GPU)",
        "basic_codec": "mp4v",
        "container": "mp4",
        "extension": ".mp4",
        "pixel_format": "yuv420p",
        "encoder_preset": "p5",
        "crf": 20,
        "nvenc_cq": 20,
        "audio_mode": "copy",
        "keep_audio": True,
        "preserve_hdr10": False,
        "warning": "",
    },

    "Fast Preview - Quick Test": {
        "description": "Fast render for checking depth, alignment, pop-out, and scene comfort.",
        "use_ffmpeg": True,
        "ffmpeg_codec": "H.264 / AVC (NVENC - NVIDIA GPU)",
        "basic_codec": "mp4v",
        "container": "mp4",
        "extension": ".mp4",
        "pixel_format": "yuv420p",
        "encoder_preset": "p2",
        "crf": 26,
        "nvenc_cq": 26,
        "audio_mode": "copy",
        "keep_audio": True,
        "preserve_hdr10": False,
        "warning": "Lower quality. Use this for testing, not final renders.",
    },

    "Balanced Final - NVIDIA": {
        "description": "Good default final render for NVIDIA users. Balanced speed, quality, and compatibility.",
        "use_ffmpeg": True,
        "ffmpeg_codec": "H.264 / AVC (NVENC - NVIDIA GPU)",
        "basic_codec": "mp4v",
        "container": "mp4",
        "extension": ".mp4",
        "pixel_format": "yuv420p",
        "encoder_preset": "p5",
        "crf": 20,
        "nvenc_cq": 20,
        "audio_mode": "copy",
        "keep_audio": True,
        "preserve_hdr10": False,
        "warning": "",
    },

    "High Quality Final - NVIDIA": {
        "description": "Cleaner final H.264 output with strong compatibility. Larger files than Balanced.",
        "use_ffmpeg": True,
        "ffmpeg_codec": "H.264 / AVC (NVENC - NVIDIA GPU)",
        "basic_codec": "mp4v",
        "container": "mp4",
        "extension": ".mp4",
        "pixel_format": "yuv420p",
        "encoder_preset": "p7",
        "crf": 18,
        "nvenc_cq": 18,
        "audio_mode": "copy",
        "keep_audio": True,
        "preserve_hdr10": False,
        "warning": "",
    },

    "Small File - HEVC": {
        "description": "Smaller files with good quality for modern devices.",
        "use_ffmpeg": True,
        "ffmpeg_codec": "H.265 / HEVC (NVENC - NVIDIA GPU)",
        "basic_codec": "mp4v",
        "container": "mp4",
        "extension": ".mp4",
        "pixel_format": "yuv420p",
        "encoder_preset": "p5",
        "crf": 24,
        "nvenc_cq": 24,
        "audio_mode": "aac",
        "keep_audio": True,
        "preserve_hdr10": False,
        "warning": "Some TVs, projectors, and media boxes may not decode HEVC correctly.",
    },

    "4K / Full-SBS High Quality - HEVC": {
        "description": "High quality HEVC for large Full-SBS, 4K-style, or headset-focused renders.",
        "use_ffmpeg": True,
        "ffmpeg_codec": "H.265 / HEVC (NVENC - NVIDIA GPU)",
        "basic_codec": "mp4v",
        "container": "mkv",
        "extension": ".mkv",
        "pixel_format": "yuv420p",
        "encoder_preset": "p7",
        "crf": 18,
        "nvenc_cq": 18,
        "audio_mode": "copy",
        "keep_audio": True,
        "preserve_hdr10": False,
        "warning": "Use Compatibility Mode if playback looks corrupted on a media box or projector.",
    },



    "Archive Master - Large File": {
        "description": "Large high-quality master render for keeping or re-encoding later.",
        "use_ffmpeg": True,
        "ffmpeg_codec": "H.264 / AVC (NVENC - NVIDIA GPU)",
        "basic_codec": "mp4v",
        "container": "mkv",
        "extension": ".mkv",
        "pixel_format": "yuv420p",
        "encoder_preset": "p7",
        "crf": 16,
        "nvenc_cq": 16,
        "audio_mode": "copy",
        "keep_audio": True,
        "preserve_hdr10": False,
        "warning": "Large file size.",
    },

    "CPU Compatibility - Slow": {
        "description": "CPU H.264 encode. Slower, but useful if GPU encoding is unavailable.",
        "use_ffmpeg": True,
        "ffmpeg_codec": "H.264 / AVC (libx264 - CPU)",
        "basic_codec": "mp4v",
        "container": "mp4",
        "extension": ".mp4",
        "pixel_format": "yuv420p",
        "encoder_preset": "medium",
        "crf": 20,
        "nvenc_cq": 20,
        "audio_mode": "copy",
        "keep_audio": True,
        "preserve_hdr10": False,
        "warning": "Slower because it uses CPU encoding.",
    },

    "Custom": {
        "description": "Manual encoder settings.",
    },
}


VR180_RENDER_PRESETS = {
    "Standard VR180": {
        "description": "Balanced VR180 setting.",
        "output_format": "VR180 Equirect (SBS)",
        "hfov": 110,
        "equi": (3840, 1920),
        "flat": (1920, 1080),
    },
    "Quest / Mobile Friendly": {
        "description": "Lower resolution VR180 for easier playback.",
        "output_format": "VR180 Equirect (SBS)",
        "hfov": 100,
        "equi": (2048, 1024),
        "flat": (1280, 720),
    },
    "High Quality VR180": {
        "description": "Sharper VR180 output.",
        "output_format": "VR180 Equirect (SBS)",
        "hfov": 110,
        "equi": (4096, 2048),
        "flat": (2560, 1440),
    },
    "Max Quality VR180": {
        "description": "Large VR180 output for high quality exports.",
        "output_format": "VR180 Equirect (SBS)",
        "hfov": 110,
        "equi": (5760, 2880),
        "flat": (2560, 1440),
    },
    "Custom": {
        "description": "Manual VR180 settings.",
    },
}

SIMPLE_3D_ADVANCED_PRESETS = {
    "Comfortable Cinema": {
        "fg_shift": -8.9,
        "mg_shift": -2.5,
        "bg_shift": 2.9,
        "max_pixel_shift": 0.071,
        "zero_parallax_strength": -0.005,
        "parallax_balance": 0.37,
        "convergence_strength": 0.05,
        "depth_pop_gamma": 0.94,
        "depth_pop_mid": 0.45,
        "depth_stretch_lo": 0.06,
        "depth_stretch_hi": 1.0,
        "fg_pop_multiplier": 1.07,
        "bg_push_multiplier": 1.03,
        "subject_lock_strength": 0.85,
        "subject_plane_lock_strength": 0.28,
        "subject_plane_lock_width": 0.12,
        "subject_screen_plane": 0.0,
        "foreground_curvature_strength": 0.06,
        "sharpness_factor": 0.2,
        "dof_strength": 0.3,
        "use_subject_tracking": True,
        "enable_dynamic_convergence": True,
        "use_floating_window": False,
        "enable_edge_masking": True,
        "enable_feathering": True,
        "disable_shift_ema": True,
        "edge_repair_quality": "High",
    },

    "Close-Up Safe": {
        "fg_shift": -6.4,
        "mg_shift": -2.1,
        "bg_shift": 2.2,
        "max_pixel_shift": 0.053,
        "zero_parallax_strength": -0.003,
        "parallax_balance": 0.63,
        "convergence_strength": 0.259,
        "depth_pop_gamma": 0.98,
        "depth_pop_mid": 0.5,
        "depth_stretch_lo": 0.05,
        "depth_stretch_hi": 0.96,
        "fg_pop_multiplier": 1.0,
        "bg_push_multiplier": 1.0,
        "subject_lock_strength": 0.32,
        "subject_plane_lock_strength": 0.28,
        "subject_plane_lock_width": 0.12,
        "subject_screen_plane": 0.0,
        "foreground_curvature_strength": 0.01,
        "sharpness_factor": 0.2,
        "dof_strength": 0.0,
        "use_subject_tracking": True,
        "enable_dynamic_convergence": False,
        "use_floating_window": False,
        "enable_edge_masking": True,
        "enable_feathering": True,
        "disable_shift_ema": True,
        "edge_repair_quality": "Fast",
    },

    "Wide / Deep Scene": {
        "fg_shift": -9.0,
        "mg_shift": -3.0,
        "bg_shift": 2.4,
        "max_pixel_shift": 0.071,
        "zero_parallax_strength": 0.005,
        "parallax_balance": 0.38,
        "convergence_strength": 0.058,
        "depth_pop_gamma": 0.86,
        "depth_pop_mid": 0.5,
        "depth_stretch_lo": 0.05,
        "depth_stretch_hi": 0.95,
        "fg_pop_multiplier": 1.11,
        "bg_push_multiplier": 1.05,
        "subject_lock_strength": 0.9,
        "subject_plane_lock_strength": 0.31,
        "subject_plane_lock_width": 0.13,
        "subject_screen_plane": 0.0,
        "foreground_curvature_strength": 0.07,
        "sharpness_factor": 0.2,
        "dof_strength": 0.3,
        "use_subject_tracking": True,
        "enable_dynamic_convergence": True,
        "use_floating_window": False,
        "enable_edge_masking": True,
        "enable_feathering": True,
        "disable_shift_ema": True,
        "edge_repair_quality": "Fast",
    },

    "Showcase Mode": {
        "fg_shift": -9.9,
        "mg_shift": -3.0,
        "bg_shift": 3.3,
        "max_pixel_shift": 0.071,
        "zero_parallax_strength": -0.01,
        "parallax_balance": 1.0,
        "convergence_strength": 0.007,
        "depth_pop_gamma": 0.87,
        "depth_pop_mid": 0.5,
        "depth_stretch_lo": 0.05,
        "depth_stretch_hi": 0.95,
        "fg_pop_multiplier": 1.11,
        "bg_push_multiplier": 1.05,
        "subject_lock_strength": 0.9,
        "subject_plane_lock_strength": 0.28,
        "subject_plane_lock_width": 0.12,
        "subject_screen_plane": 0.0,
        "foreground_curvature_strength": 0.07,
        "sharpness_factor": 0.2,
        "dof_strength": 0.3,
        "use_subject_tracking": True,
        "enable_dynamic_convergence": True,
        "use_floating_window": False,
        "enable_edge_masking": True,
        "enable_feathering": True,
        "disable_shift_ema": True,
        "edge_repair_quality": "Fast",
    },

    "Clean Edge": {
        "fg_shift": -9.0,
        "mg_shift": -3.0,
        "bg_shift": 2.4,
        "max_pixel_shift": 0.071,
        "zero_parallax_strength": 0.008,
        "parallax_balance": 0.35,
        "convergence_strength": 0.006,
        "depth_pop_gamma": 1.0,
        "depth_pop_mid": 0.5,
        "depth_stretch_lo": 0.05,
        "depth_stretch_hi": 0.95,
        "fg_pop_multiplier": 1.11,
        "bg_push_multiplier": 1.05,
        "subject_lock_strength": 0.95,
        "subject_plane_lock_strength": 0.28,
        "subject_plane_lock_width": 0.12,
        "subject_screen_plane": 0.0,
        "foreground_curvature_strength": 0.07,
        "sharpness_factor": 0.2,
        "dof_strength": 0.3,
        "use_subject_tracking": True,
        "enable_dynamic_convergence": True,
        "use_floating_window": False,
        "enable_edge_masking": True,
        "enable_feathering": True,
        "disable_shift_ema": True,
        "edge_repair_quality": "Fast",
    },
}

SIMPLE_3D_PRESETS = {
    "Comfortable Cinema": {
        "description": "Balanced visible depth with safer comfort for most movies.",
        "strength": 60,
        "pop": 45,
        "comfort": 65,
        "stability": 70,
        "screen_depth": 50,
        "subject_plane": 0,
    },
    "Strong Pop-Out": {
        "description": "More foreground punch for action scenes and demo clips.",
        "strength": 78,
        "pop": 88,
        "comfort": 45,
        "stability": 60,
        "screen_depth": 42,
        "subject_plane": 0,
    },
    "Deep Background": {
        "description": "Pushes scenery and wide shots deeper into the screen.",
        "strength": 78,
        "pop": 40,
        "comfort": 55,
        "stability": 55,
        "screen_depth": 62,
        "subject_plane": 0,
    },
    "Close-Up Safe": {
        "description": "Gentler settings for faces, dialogue, and subject-heavy shots.",
        "strength": 50,
        "pop": 35,
        "comfort": 80,
        "stability": 90,
        "screen_depth": 48,
        "subject_plane": 0,
    },
    "VR Comfortable": {
        "description": "Lower strain settings for headset viewing while keeping visible depth.",
        "strength": 48,
        "pop": 35,
        "comfort": 88,
        "stability": 85,
        "screen_depth": 55,
        "subject_plane": 0,
    },
    "Wide / Deep Scene": {
        "description": "Stronger depth for wide shots and scenery while staying inside the safe shift range.",
        "strength": 70,
        "pop": 40,
        "comfort": 55,
        "stability": 55,
        "screen_depth": 60,
        "subject_plane": 0,
    },

    "Clean Edge": {
        "description": "Strong depth with safer edge handling based on the clean edge preset.",
        "strength": 68,
        "pop": 35,
        "comfort": 65,
        "stability": 80,
        "screen_depth": 45,
        "subject_plane": 0,
    },

    "Showcase Mode": {
        "description": "Stronger depth for testing and show-off clips.",
        "strength": 88,
        "pop": 78,
        "comfort": 35,
        "stability": 50,
        "screen_depth": 45,
        "subject_plane": 0,
    },   
    "Custom": {
        "description": "Manual simple control settings.",
        "strength": 60,
        "pop": 45,
        "comfort": 65,
        "stability": 70,
        "screen_depth": 50,
        "subject_plane": 0,
    },
}

class StereoGeneratorPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._translation_map = []
        self._last_preview_result = None
        self._last_preview_pixmap = None
        self._fullscreen_dialog = None
        self._fullscreen_label = None
        self._keyframe_service = None
        self._applying_3d_assistant = False
        self._applying_encoding_preset = False
        self._applying_simple_3d_adjustment = False
        self._active_3d_style_key = "Custom"
        self._active_3d_style_base = {}
        self._advanced_3d_base_dirty = True
        
        self._preview_debounce = QTimer()
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(50)
        self._preview_debounce.timeout.connect(self.controller.update_preview)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # Left column
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setMinimumWidth(280)

        left_container = QWidget()
        left_col = QVBoxLayout(left_container)
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(12)
        
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItem(self._t("Single Video Render"), "video")
        self.render_mode_combo.addItem(self._t("3D Image Render"), "image")
        self.render_mode_combo.addItem(self._t("Batch Video Folder Render"), "video_folder")
        self.render_mode_combo.addItem(self._t("Image Folder Render"), "image_folder")

        self.input_row = FilePickerRow("Input Video", "Select source video...")
        self.depth_row = FilePickerRow("Depth Map", "Select depth video...")
        self.output_row = FilePickerRow("Output", "Choose output file...")

        self._register_file_row(self.input_row, "Input Video", "Select source video...")
        self._register_file_row(self.depth_row, "Depth Map", "Select depth video...")
        self._register_file_row(self.output_row, "Output", "Choose output file...")

        io_card = ParameterCard("Sources")
        self._register_title(io_card, "Sources")
        io_card.inner_layout.addWidget(self._label("Render Mode"))
        io_card.inner_layout.addWidget(self.render_mode_combo)
        io_card.inner_layout.addWidget(self.input_row)
        io_card.inner_layout.addWidget(self.depth_row)
        io_card.inner_layout.addWidget(self.output_row)
        
        preset_card = ParameterCard("Presets")
        self._register_title(preset_card, "Presets")

        self.preset_combo = QComboBox()
        self.preset_combo.addItem(self._t("Select Preset"))

        self.load_preset_btn = self._button("Load Preset")
        self.save_preset_btn = self._button("Save Preset")

        preset_card.inner_layout.addWidget(self.preset_combo)
        preset_card.inner_layout.addWidget(self.load_preset_btn)
        preset_card.inner_layout.addWidget(self.save_preset_btn)
        
        # --- Actions ---
        action_card = ParameterCard("Actions")
        self._register_title(action_card, "Actions")

        self.preview_btn = self._button("Load Preview Sources")
        self.encoding_settings_btn = self._button("Output & Encoding...")
        self.processing_settings_btn = self._button("Processing Options...")
        self.render_btn = self._button("Start Render")
        self.suspend_btn = self._button("Suspend")
        self.resume_btn = self._button("Resume")
        self.cancel_btn = self._button("Cancel")

        action_card.inner_layout.addWidget(self.preview_btn)
        action_card.inner_layout.addWidget(self.encoding_settings_btn)
        action_card.inner_layout.addWidget(self.processing_settings_btn)
        action_card.inner_layout.addWidget(self.render_btn)
        action_card.inner_layout.addWidget(self.suspend_btn)
        action_card.inner_layout.addWidget(self.resume_btn)
        action_card.inner_layout.addWidget(self.cancel_btn)

        format_card = ParameterCard("Output & Encoding")
        self._register_title(format_card, "Output & Encoding")

        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems([
            "Full-SBS",
            "Half-SBS",
            "VR",
            "VR180 Equirect (TB)",
            "VR180 Equirect (SBS)",
            "Red-Cyan Anaglyph",
            "Passive Interlaced",
        ])

        self.stereo_out_combo = QComboBox()
        self.stereo_out_combo.addItems(["sbs", "left", "right", "both"])

        self.use_ffmpeg_check = self._checkbox("Use FFmpeg Renderer")
        self.keep_original_audio_check = self._checkbox("Keep Original Audio")
        self.preserve_hdr10_check = self._checkbox("Preserve HDR10")
        
        self.ffmpeg_codec_combo = QComboBox()
        self.ffmpeg_codec_combo.addItems([
            "H.264 / AVC (libx264 - CPU)",
            "H.265 / HEVC (libx265 - CPU)",
            "AV1 (libaom - CPU)",
            "AV1 (SVT - CPU, faster)",
            "MPEG-4 (mp4v - CPU)",
            "XviD (AVI - CPU)",
            "DivX (AVI - CPU)",
            "H.264 / AVC (NVENC - NVIDIA GPU)",
            "H.265 / HEVC (NVENC - NVIDIA GPU)",
            "AV1 (NVENC - NVIDIA RTX 40+ GPU)",
            "H.264 / AVC (AMF - AMD GPU)",
            "H.265 / HEVC (AMF - AMD GPU)",
            "AV1 (AMF - AMD RDNA3+)",
            "H.264 / AVC (QSV - Intel GPU)",
            "H.265 / HEVC (QSV - Intel GPU)",
            "VP9 (QSV - Intel GPU)",
            "AV1 (QSV - Intel ARC / Gen11+)",
        ])

        self.basic_codec_combo = QComboBox()
        self.basic_codec_combo.addItems(["mp4v", "XVID", "DIVX"])

        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)

        self.nvenc_cq_spin = QSpinBox()
        self.nvenc_cq_spin.setRange(0, 51)
        
        self.encoding_preset_combo = QComboBox()
        self.encoding_preset_combo.addItems(list(ENCODING_PRESETS.keys()))
        self.encoding_preset_combo.setCurrentText("Compatibility Mode - Plays Everywhere")

        self.encoding_preset_desc = QLabel("")
        self.encoding_preset_desc.setWordWrap(True)

        self.active_encoder_label = QLabel("")
        self.active_encoder_label.setWordWrap(True)
        
        self.encoding_preset_details = QLabel("")
        self.encoding_preset_details.setWordWrap(True)

        self.encoding_preset_warning = QLabel("")
        self.encoding_preset_warning.setWordWrap(True)
        self.encoding_preset_warning.setStyleSheet("color: #ffb86c; font-weight: 600;")

        self.encoding_advanced_btn = QPushButton(self._qt_text(self._t("Advanced Encoding Settings")))
        self.encoding_advanced_btn.setCheckable(True)
        self.encoding_advanced_btn.setChecked(False)

        self.vr180_render_preset_combo = QComboBox()
        self.vr180_render_preset_combo.addItems(list(VR180_RENDER_PRESETS.keys()))
        self.vr180_render_preset_combo.setCurrentText("Standard VR180")

        self.vr180_preset_desc = QLabel("")
        self.vr180_preset_desc.setWordWrap(True)

        self.vr180_advanced_btn = QPushButton(self._qt_text(self._t("Advanced VR180 Settings")))
        self.vr180_advanced_btn.setCheckable(True)
        self.vr180_advanced_btn.setChecked(False)

        format_card.inner_layout.addWidget(self._label("Output Format"))
        format_card.inner_layout.addWidget(self.output_format_combo)

        format_card.inner_layout.addWidget(self._label("Stereo Output"))
        format_card.inner_layout.addWidget(self.stereo_out_combo)

        format_card.inner_layout.addWidget(self.use_ffmpeg_check)
        format_card.inner_layout.addWidget(self.keep_original_audio_check)
        format_card.inner_layout.addWidget(self.preserve_hdr10_check)

        format_card.inner_layout.addWidget(self._label("FFmpeg Codec"))
        format_card.inner_layout.addWidget(self.ffmpeg_codec_combo)

        format_card.inner_layout.addWidget(self._label("Basic Codec"))
        format_card.inner_layout.addWidget(self.basic_codec_combo)

        format_card.inner_layout.addWidget(self._label("CRF"))
        format_card.inner_layout.addWidget(self.crf_spin)

        format_card.inner_layout.addWidget(self._label("NVENC CQ"))
        format_card.inner_layout.addWidget(self.nvenc_cq_spin)

        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItems(ASPECT_RATIO_OPTIONS)

        self.vr180_hfov_slider = QSlider(Qt.Horizontal)
        self.vr180_hfov_slider.setRange(60, 140)
        self.vr180_hfov_value = QLabel("110")

        self.vr180_equi_w_spin = QSpinBox()
        self.vr180_equi_w_spin.setRange(512, 16384)
        self.vr180_equi_h_spin = QSpinBox()
        self.vr180_equi_h_spin.setRange(512, 16384)

        self.vr180_flat_w_spin = QSpinBox()
        self.vr180_flat_w_spin.setRange(512, 16384)
        self.vr180_flat_h_spin = QSpinBox()
        self.vr180_flat_h_spin.setRange(512, 16384)

        self.vr180_equi_preset_combo = QComboBox()
        self.vr180_equi_preset_combo.addItems(list(VR180_EQUI_PRESETS.keys()))

        self.vr180_flat_preset_combo = QComboBox()
        self.vr180_flat_preset_combo.addItems(list(VR180_FLAT_PRESETS.keys()))

        format_card.inner_layout.addWidget(self._label("Aspect Ratio"))
        format_card.inner_layout.addWidget(self.aspect_ratio_combo)

        format_card.inner_layout.addWidget(self._label("VR180 HFOV"))
        format_card.inner_layout.addWidget(self.vr180_hfov_slider)
        format_card.inner_layout.addWidget(self.vr180_hfov_value)

        format_card.inner_layout.addWidget(self._label("VR180 Equirect Width"))
        format_card.inner_layout.addWidget(self.vr180_equi_w_spin)
        format_card.inner_layout.addWidget(self._label("VR180 Equirect Height"))
        format_card.inner_layout.addWidget(self.vr180_equi_h_spin)

        format_card.inner_layout.addWidget(self._label("VR180 Flat Width"))
        format_card.inner_layout.addWidget(self.vr180_flat_w_spin)
        format_card.inner_layout.addWidget(self._label("VR180 Flat Height"))
        format_card.inner_layout.addWidget(self.vr180_flat_h_spin)

        format_card.inner_layout.addWidget(self._label("VR180 Equirect Preset"))
        format_card.inner_layout.addWidget(self.vr180_equi_preset_combo)

        format_card.inner_layout.addWidget(self._label("VR180 Flat Preset"))
        format_card.inner_layout.addWidget(self.vr180_flat_preset_combo)

        processing_card = ParameterCard("Processing Options")
        self._register_title(processing_card, "Processing Options")

        self.preserve_aspect_check = self._checkbox("Preserve Original Aspect Ratio")
        self.auto_crop_check = self._checkbox("Auto Crop Black Bars")
        self.subject_tracking_check = self._checkbox("Stabilize Screen Plane")
        self.skip_blank_check = self._checkbox("Skip Blank/White Frames")
        self.edge_masking_check = self._checkbox("Enable Edge Masking")
        self.feathering_check = self._checkbox("Enable Feathering")
        self.edge_repair_quality_combo = QComboBox()
        self.edge_repair_quality_combo.addItems([
            "Off",
            "Fast",
            "Balanced",
            "High",
            "Showcase",
        ])
        self.edge_repair_quality_combo.setCurrentText(
            getattr(self.controller.state, "edge_repair_quality", "Balanced")
        )
        self.dynamic_convergence_check = self._checkbox("Enable Dynamic Convergence")
        self.floating_window_check = self._checkbox("Enable Floating Window (DFW)")
        self.disable_shift_ema_check = self._checkbox("Disable Shift EMA (Debug)")

        self.clip_start_edit = QLineEdit()
        self.clip_end_edit = QLineEdit()
        self.clip_start_edit.setPlaceholderText("Start: HH:MM:SS.ms or seconds")
        self.clip_end_edit.setPlaceholderText("End: HH:MM:SS.ms or seconds")

        self.clear_clip_btn = self._button("Clear Clip Range")

        processing_card.inner_layout.addWidget(self.preserve_aspect_check)
        processing_card.inner_layout.addWidget(self.auto_crop_check)
        processing_card.inner_layout.addWidget(self.subject_tracking_check)
        processing_card.inner_layout.addWidget(self.skip_blank_check)
        processing_card.inner_layout.addWidget(self.edge_masking_check)
        processing_card.inner_layout.addWidget(self.feathering_check)

        processing_card.inner_layout.addWidget(self._label("Edge Repair Quality"))
        processing_card.inner_layout.addWidget(self.edge_repair_quality_combo)

        processing_card.inner_layout.addWidget(self.dynamic_convergence_check)
        processing_card.inner_layout.addWidget(self.floating_window_check)
        processing_card.inner_layout.addWidget(self.disable_shift_ema_check)

        processing_card.inner_layout.addWidget(self._label("Clip Start"))
        processing_card.inner_layout.addWidget(self.clip_start_edit)
        processing_card.inner_layout.addWidget(self._label("Clip End"))
        processing_card.inner_layout.addWidget(self.clip_end_edit)
        processing_card.inner_layout.addWidget(self.clear_clip_btn)
        
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems([
            "Red-Blue Anaglyph",
            "Passive Interlaced",
            "HSBS",
            "Shift Heatmap",
            "Shift Heatmap (Abs)",
            "Shift Heatmap (Clipped ±5px)",
            "Overlay Arrows",
            "Left-Right Diff",
            "Feather Mask",
            "Feather Blend",
        ])

        self.ipd_enabled_check = self._checkbox("Enable Stereo Scaling (IPD)")
        self.ipd_scale = QDoubleSpinBox()
        self.ipd_scale.setRange(0.50, 1.50)
        self.ipd_scale.setDecimals(2)
        self.ipd_scale.setSingleStep(0.01)

        self.show_guides_check = self._checkbox("Show Convergence Guides")

        left_col.addWidget(io_card)
        left_col.addWidget(preset_card)
        left_col.addWidget(action_card)
        left_col.addStretch()

        left_scroll.setWidget(left_container)

        # Center
        center_col = QVBoxLayout()
        center_col.setSpacing(10)

        self.preview_meta_label = QLabel()
        self.preview_meta_label.setWordWrap(True)
        self.preview_meta_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._apply_theme_styles()

        self.preview_panel = PreviewPanel()
        self.preview_panel.set_translator(self._t)

        # Build frame_card BEFORE the splitter so it exists when referenced
        frame_card = ParameterCard("Frame Preview")
        self._register_title(frame_card, "Frame Preview")
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_label = QLabel(f"{self._t('Frame')}: 0 / 0")
        self.prev_frame_btn = self._button("Previous Frame")
        self.next_frame_btn = self._button("Next Frame")

        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 120)
        self.frame_step_spin.setValue(1)
        self.frame_step_spin.setToolTip("How many frames to move with Previous / Next")
        self.refresh_preview_btn = self._button("Refresh Preview")
        self.save_preview_btn = self._button("Save Preview Image")
        self.fullscreen_preview_btn = self._button("Fullscreen Preview")

        frame_card.inner_layout.addWidget(self.frame_label)
        frame_card.inner_layout.addWidget(self.frame_slider)

        frame_nav_layout = QGridLayout()
        frame_nav_layout.setHorizontalSpacing(10)
        frame_nav_layout.setVerticalSpacing(8)

        frame_nav_layout.addWidget(self.prev_frame_btn, 0, 0)
        frame_nav_layout.addWidget(self.next_frame_btn, 0, 1)
        frame_nav_layout.addWidget(self._label("Step"), 1, 0)
        frame_nav_layout.addWidget(self.frame_step_spin, 1, 1)

        frame_card.inner_layout.addLayout(frame_nav_layout)

        # --- Keyframe Shot Controls ---
        # Compact workflow:
        # choose frame -> tune settings -> add/update keyframe.
        frame_card.inner_layout.addSpacing(10)

        self.keyframe_hint_label = self._label(
            "Keyframes control settings from this frame until the next keyframe."
        )
        self.keyframe_hint_label.setWordWrap(True)
        frame_card.inner_layout.addWidget(self.keyframe_hint_label)

        keyframe_layout = QGridLayout()
        keyframe_layout.setHorizontalSpacing(10)
        keyframe_layout.setVerticalSpacing(8)

        self.keyframes_enabled_check = self._checkbox("Enable 3D Keyframes")

        self.keyframes_path_edit = QLineEdit()
        self.keyframes_path_edit.setReadOnly(True)
        self._register_placeholder(
            self.keyframes_path_edit,
            "No keyframe file selected."
        )

        self.create_keyframes_btn = self._button("Create Keyframe File")
        self.load_keyframes_btn = self._button("Load Keyframe File")
        self.save_keyframes_btn = self._button("Save Keyframes")
        self.keyframe_file_btn = self._button("Keyframe File...")

        self.keyframe_label_edit = QLineEdit()
        self._register_placeholder(
            self.keyframe_label_edit,
            "Shot label, example: Close, Wide, Pan Safe"
        )

        self.keyframe_transition_spin = QSpinBox()
        self.keyframe_transition_spin.setRange(0, 240)
        self.keyframe_transition_spin.setValue(24)
        self.keyframe_transition_spin.setToolTip("Frames used to blend into this keyframe. Use 0 for an instant cut.")

        self.keyframe_transition_combo = QComboBox()
        self._rebuild_key_combo(
            self.keyframe_transition_combo,
            ["cut", "linear", "smoothstep"],
            selected_key="smoothstep",
        )

        self.keyframe_combo = QComboBox()
        self.keyframe_combo.setMinimumWidth(220)
        self.keyframe_combo.setToolTip("Shows the frame range controlled by each keyframe.")

        self.add_keyframe_btn = self._button("Add / Update Current Shot")
        self.delete_keyframe_btn = self._button("Delete")
        self.apply_keyframe_btn = self._button("Apply")

        keyframe_layout.addWidget(self.keyframes_enabled_check, 0, 0, 1, 2)
        keyframe_layout.addWidget(self.keyframe_file_btn, 0, 2)

        keyframe_layout.addWidget(self._label("Shot Label"), 1, 0)
        keyframe_layout.addWidget(self.keyframe_label_edit, 1, 1, 1, 2)

        keyframe_layout.addWidget(self._label("Blend"), 2, 0)
        keyframe_layout.addWidget(self.keyframe_transition_spin, 2, 1)
        keyframe_layout.addWidget(self.keyframe_transition_combo, 2, 2)

        keyframe_layout.addWidget(self._label("Shot Ranges"), 3, 0)
        keyframe_layout.addWidget(self.keyframe_combo, 3, 1, 1, 2)

        keyframe_layout.addWidget(self.add_keyframe_btn, 4, 0, 1, 3)
        keyframe_layout.addWidget(self.apply_keyframe_btn, 5, 0)
        keyframe_layout.addWidget(self.delete_keyframe_btn, 5, 1)

        frame_card.inner_layout.addLayout(keyframe_layout)

        # --- Preview controls ---
        frame_card.inner_layout.addSpacing(10)

        self.preview_settings_btn = self._button("Preview Settings...")

        preview_button_layout = QGridLayout()
        preview_button_layout.setHorizontalSpacing(10)
        preview_button_layout.setVerticalSpacing(8)
        preview_button_layout.addWidget(self.refresh_preview_btn, 0, 0)
        preview_button_layout.addWidget(self.save_preview_btn, 0, 1)
        preview_button_layout.addWidget(self.fullscreen_preview_btn, 1, 0)
        preview_button_layout.addWidget(self.preview_settings_btn, 1, 1)

        frame_card.inner_layout.addLayout(preview_button_layout)

        # Scrollable center: preview always visible, frame card scrolls below it
        center_scroll = QScrollArea()
        center_scroll.setWidgetResizable(True)
        center_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        center_scroll.setFrameShape(QScrollArea.NoFrame)

        center_container = QWidget()
        center_inner = QVBoxLayout(center_container)
        center_inner.setContentsMargins(0, 0, 0, 0)
        center_inner.setSpacing(10)

        center_inner.addWidget(self.preview_meta_label)
        center_inner.addWidget(self.preview_panel, 1)  # preview stretches
        center_inner.addWidget(frame_card)

        center_scroll.setWidget(center_container)
        center_col.addWidget(center_scroll)

        # --- Stereo Shift ---
        shift_card = ParameterCard("Stereo Shift")
        self._register_title(shift_card, "Stereo Shift")

        shift_layout = QGridLayout()
        shift_layout.setHorizontalSpacing(10)
        shift_layout.setVerticalSpacing(12)

        self.fg_shift_slider = QSlider(Qt.Horizontal)
        self.mg_shift_slider = QSlider(Qt.Horizontal)
        self.bg_shift_slider = QSlider(Qt.Horizontal)

        self.fg_shift_slider.setRange(-200, 200)   # -20.0 to 20.0
        self.mg_shift_slider.setRange(-100, 100)   # -10.0 to 10.0
        self.bg_shift_slider.setRange(-200, 200)   # -20.0 to 20.0

        self.fg_shift_value = QLabel("0.00")
        self.mg_shift_value = QLabel("0.00")
        self.bg_shift_value = QLabel("0.00")

        shift_layout.addWidget(self._label("Foreground Shift"), 0, 0)
        shift_layout.addWidget(self.fg_shift_slider, 1, 0)
        shift_layout.addWidget(self.fg_shift_value, 1, 1)

        shift_layout.addWidget(self._label("Midground Shift"), 2, 0)
        shift_layout.addWidget(self.mg_shift_slider, 3, 0)
        shift_layout.addWidget(self.mg_shift_value, 3, 1)

        shift_layout.addWidget(self._label("Background Shift"), 4, 0)
        shift_layout.addWidget(self.bg_shift_slider, 5, 0)
        shift_layout.addWidget(self.bg_shift_value, 5, 1)

        shift_card.inner_layout.addLayout(shift_layout)

        # --- Depth & Parallax ---
        depth_card = ParameterCard("Depth & Parallax Controls")
        self._register_title(depth_card, "Depth & Parallax Controls")

        depth_layout = QGridLayout()
        depth_layout.setHorizontalSpacing(10)
        depth_layout.setVerticalSpacing(12)

        self.convergence_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.parallax_balance_slider = QSlider(Qt.Horizontal)
        self.zero_parallax_slider = QSlider(Qt.Horizontal)
        self.max_pixel_shift_slider = QSlider(Qt.Horizontal)
        self.dof_slider = QSlider(Qt.Horizontal)

        # scaled integer ranges
        self.convergence_slider.setRange(-1000, 1000)       # -1.000 to 1.000
        self.sharpness_slider.setRange(-100, 100)           # -1.0 to 1.0
        self.parallax_balance_slider.setRange(0, 100)       # 0.00 to 1.00
        self.zero_parallax_slider.setRange(-50, 50)         # -0.050 to 0.050
        self.max_pixel_shift_slider.setRange(5, 100)        # 0.005 to 0.100
        self.dof_slider.setRange(0, 50)                     # 0.0 to 5.0

        self.convergence_value = QLabel("0.000")
        self.sharpness_value = QLabel("0.0")
        self.parallax_balance_value = QLabel("0.00")
        self.zero_parallax_value = QLabel("0.000")
        self.max_pixel_shift_value = QLabel("0.000")
        self.dof_value = QLabel("0.0")

        depth_layout.addWidget(self._label("Convergence Strength"), 0, 0)
        depth_layout.addWidget(self.convergence_slider, 1, 0)
        depth_layout.addWidget(self.convergence_value, 1, 1)

        depth_layout.addWidget(self._label("Sharpness Factor"), 2, 0)
        depth_layout.addWidget(self.sharpness_slider, 3, 0)
        depth_layout.addWidget(self.sharpness_value, 3, 1)

        depth_layout.addWidget(self._label("Parallax Balance"), 4, 0)
        depth_layout.addWidget(self.parallax_balance_slider, 5, 0)
        depth_layout.addWidget(self.parallax_balance_value, 5, 1)

        depth_layout.addWidget(self._label("Screen Plane Offset"), 6, 0)
        depth_layout.addWidget(self.zero_parallax_slider, 7, 0)
        depth_layout.addWidget(self.zero_parallax_value, 7, 1)

        depth_layout.addWidget(self._label("Max Pixel Shift"), 8, 0)
        depth_layout.addWidget(self.max_pixel_shift_slider, 9, 0)
        depth_layout.addWidget(self.max_pixel_shift_value, 9, 1)

        depth_layout.addWidget(self._label("DoF Strength"), 10, 0)
        depth_layout.addWidget(self.dof_slider, 11, 0)
        depth_layout.addWidget(self.dof_value, 11, 1)

        # IMPORTANT: add the populated layout into the card.
        depth_card.inner_layout.addLayout(depth_layout)

        pop_card = ParameterCard("Pop & Subject Controls")
        self._register_title(pop_card, "Pop & Subject Controls")

        pop_layout = QGridLayout()
        pop_layout.setHorizontalSpacing(10)
        pop_layout.setVerticalSpacing(12)

        self.depth_pop_gamma_slider = QSlider(Qt.Horizontal)
        self.depth_pop_gamma_slider.setRange(70, 120)   # 0.70 to 1.20
        self.depth_pop_gamma_value = QLabel("0.85")

        self.fg_pop_slider = QSlider(Qt.Horizontal)
        self.fg_pop_slider.setRange(100, 160)           # 1.00 to 1.60
        self.fg_pop_value = QLabel("1.20")

        self.bg_push_slider = QSlider(Qt.Horizontal)
        self.bg_push_slider.setRange(100, 140)          # 1.00 to 1.40
        self.bg_push_value = QLabel("1.10")

        self.subject_lock_slider = QSlider(Qt.Horizontal)
        self.subject_lock_slider.setRange(0, 200)       # 0.00 to 2.00
        self.subject_lock_value = QLabel("1.00")

        self.subject_plane_lock_slider = QSlider(Qt.Horizontal)
        self.subject_plane_lock_slider.setRange(0, 100)     # 0.00 to 1.00
        self.subject_plane_lock_value = QLabel("0.00")

        self.subject_plane_width_slider = QSlider(Qt.Horizontal)
        self.subject_plane_width_slider.setRange(1, 30)     # 0.01 to 0.30
        self.subject_plane_width_value = QLabel("0.08")
                
        self.foreground_curvature_slider = QSlider(Qt.Horizontal)
        self.foreground_curvature_slider.setRange(0, 20)    # 0.00 to 0.20
        self.foreground_curvature_value = QLabel("0.06")

        self.pop_mid_edit = QLineEdit()
        self.stretch_lo_edit = QLineEdit()
        self.stretch_hi_edit = QLineEdit()
        self.apply_pop_entries_btn = self._button("Apply Entries")

        pop_layout.addWidget(self._label("Depth Pop Gamma"), 0, 0)
        pop_layout.addWidget(self.depth_pop_gamma_slider, 1, 0)
        pop_layout.addWidget(self.depth_pop_gamma_value, 1, 1)

        pop_layout.addWidget(self._label("Pop Mid (0..1)"), 2, 0)
        pop_layout.addWidget(self.pop_mid_edit, 2, 1)

        pop_layout.addWidget(self._label("Stretch Lo"), 3, 0)
        pop_layout.addWidget(self.stretch_lo_edit, 3, 1)

        pop_layout.addWidget(self._label("Stretch Hi"), 4, 0)
        pop_layout.addWidget(self.stretch_hi_edit, 4, 1)

        pop_layout.addWidget(self._label("FG Pop ×"), 5, 0)
        pop_layout.addWidget(self.fg_pop_slider, 6, 0)
        pop_layout.addWidget(self.fg_pop_value, 6, 1)

        pop_layout.addWidget(self._label("BG Push ×"), 7, 0)
        pop_layout.addWidget(self.bg_push_slider, 8, 0)
        pop_layout.addWidget(self.bg_push_value, 8, 1)

        pop_layout.addWidget(self._label("Subject Lock"), 9, 0)
        pop_layout.addWidget(self.subject_lock_slider, 10, 0)
        pop_layout.addWidget(self.subject_lock_value, 10, 1)

        pop_layout.addWidget(self._label("Subject Plane Lock"), 11, 0)
        pop_layout.addWidget(self.subject_plane_lock_slider, 12, 0)
        pop_layout.addWidget(self.subject_plane_lock_value, 12, 1)

        pop_layout.addWidget(self._label("Subject Lock Width"), 13, 0)
        pop_layout.addWidget(self.subject_plane_width_slider, 14, 0)
        pop_layout.addWidget(self.subject_plane_width_value, 14, 1)

        pop_layout.addWidget(self._label("Foreground Curvature"), 15, 0)
        pop_layout.addWidget(self.foreground_curvature_slider, 16, 0)
        pop_layout.addWidget(self.foreground_curvature_value, 16, 1)

        pop_layout.addWidget(self.apply_pop_entries_btn, 17, 1)

        pop_card.inner_layout.addLayout(pop_layout)
                
        color_card = ParameterCard("Color Grading")
        self._register_title(color_card, "Color Grading")

        color_layout = QGridLayout()
        color_layout.setHorizontalSpacing(10)
        color_layout.setVerticalSpacing(12)

        self.saturation_slider = QSlider(Qt.Horizontal)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.brightness_slider = QSlider(Qt.Horizontal)

        # scaled integer ranges
        self.saturation_slider.setRange(0, 200)    # 0.00 to 2.00
        self.contrast_slider.setRange(0, 200)      # 0.00 to 2.00
        self.brightness_slider.setRange(-50, 50)   # -0.50 to 0.50

        self.saturation_value = QLabel("1.00")
        self.contrast_value = QLabel("1.00")
        self.brightness_value = QLabel("0.00")

        self.color_reset_btn = self._button("Reset")

        color_layout.addWidget(self._label("Saturation"), 0, 0)
        color_layout.addWidget(self.saturation_slider, 1, 0)
        color_layout.addWidget(self.saturation_value, 1, 1)

        color_layout.addWidget(self._label("Contrast"), 2, 0)
        color_layout.addWidget(self.contrast_slider, 3, 0)
        color_layout.addWidget(self.contrast_value, 3, 1)

        color_layout.addWidget(self._label("Brightness"), 4, 0)
        color_layout.addWidget(self.brightness_slider, 5, 0)
        color_layout.addWidget(self.brightness_value, 5, 1)

        color_layout.addWidget(self.color_reset_btn, 6, 1)

        color_card.inner_layout.addLayout(color_layout)                

        # --- 3D Assistant ---
        assistant_card = ParameterCard("3D Assistant")
        self._register_title(assistant_card, "3D Assistant")

        assistant_layout = QGridLayout()
        assistant_layout.setHorizontalSpacing(10)
        assistant_layout.setVerticalSpacing(10)

        self.simple_3d_style_combo = QComboBox()
        self._rebuild_simple_3d_style_combo(selected_key="Custom")

        self.simple_3d_desc = QLabel("")
        self.simple_3d_desc.setWordWrap(True)

        self.simple_strength_slider = QSlider(Qt.Horizontal)
        self.simple_strength_slider.setRange(0, 100)
        self.simple_strength_value = QLabel("50")

        self.simple_pop_slider = QSlider(Qt.Horizontal)
        self.simple_pop_slider.setRange(0, 100)
        self.simple_pop_value = QLabel("50")

        self.simple_comfort_slider = QSlider(Qt.Horizontal)
        self.simple_comfort_slider.setRange(0, 100)
        self.simple_comfort_value = QLabel("70")

        self.simple_stability_slider = QSlider(Qt.Horizontal)
        self.simple_stability_slider.setRange(0, 100)
        self.simple_stability_value = QLabel("60")

        self.simple_screen_depth_slider = QSlider(Qt.Horizontal)
        self.simple_screen_depth_slider.setRange(0, 100)
        self.simple_screen_depth_value = QLabel("50")

        self.simple_subject_plane_slider = QSlider(Qt.Horizontal)
        self.simple_subject_plane_slider.setRange(0, 100)
        self.simple_subject_plane_value = QLabel("0%")

        self.advanced_3d_btn = QPushButton(self._qt_text(self._t("Advanced 3D Controls")))
        self.advanced_3d_btn.setCheckable(True)
        self.advanced_3d_btn.setChecked(False)

        assistant_layout.addWidget(self._label("3D Style"), 0, 0)
        assistant_layout.addWidget(self.simple_3d_style_combo, 0, 1)

        assistant_layout.addWidget(self._label("Summary"), 1, 0)
        assistant_layout.addWidget(self.simple_3d_desc, 1, 1)

        assistant_layout.addWidget(self._label("3D Strength"), 2, 0)
        assistant_layout.addWidget(self.simple_strength_slider, 2, 1)
        assistant_layout.addWidget(self.simple_strength_value, 3, 1)

        assistant_layout.addWidget(self._label("Pop-Out"), 4, 0)
        assistant_layout.addWidget(self.simple_pop_slider, 4, 1)
        assistant_layout.addWidget(self.simple_pop_value, 5, 1)

        assistant_layout.addWidget(self._label("Depth Comfort"), 6, 0)
        assistant_layout.addWidget(self.simple_comfort_slider, 6, 1)
        assistant_layout.addWidget(self.simple_comfort_value, 7, 1)

        assistant_layout.addWidget(self._label("Subject Stability"), 8, 0)
        assistant_layout.addWidget(self.simple_stability_slider, 8, 1)
        assistant_layout.addWidget(self.simple_stability_value, 9, 1)

        assistant_layout.addWidget(self._label("Screen Depth"), 10, 0)
        assistant_layout.addWidget(self.simple_screen_depth_slider, 10, 1)
        assistant_layout.addWidget(self.simple_screen_depth_value, 11, 1)

        assistant_layout.addWidget(self._label("Subject Zero Lock"), 12, 0)
        assistant_layout.addWidget(self.simple_subject_plane_slider, 12, 1)
        assistant_layout.addWidget(self.simple_subject_plane_value, 13, 1)

        assistant_layout.addWidget(self.advanced_3d_btn, 14, 0, 1, 2)

        assistant_card.inner_layout.addLayout(assistant_layout)

        # Right column (scrollable)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setFrameShape(QScrollArea.NoFrame)
        
        right_scroll.setMinimumWidth(300)

        right_container = QWidget()
        right_col = QVBoxLayout(right_container)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(12)

        # Compact right-side tuning panel.
        # Tabs keep the main workflow clean while preserving all advanced controls.
        self.tuning_tabs = QTabWidget()
        self.tuning_tabs.setDocumentMode(True)

        depth_tab = QWidget()
        depth_tab_layout = QVBoxLayout(depth_tab)
        depth_tab_layout.setContentsMargins(0, 0, 0, 0)
        depth_tab_layout.setSpacing(12)
        depth_tab_layout.addWidget(shift_card)
        depth_tab_layout.addWidget(depth_card)
        depth_tab_layout.addStretch()

        subject_tab = QWidget()
        subject_tab_layout = QVBoxLayout(subject_tab)
        subject_tab_layout.setContentsMargins(0, 0, 0, 0)
        subject_tab_layout.setSpacing(12)
        subject_tab_layout.addWidget(pop_card)
        subject_tab_layout.addStretch()

        color_tab = QWidget()
        color_tab_layout = QVBoxLayout(color_tab)
        color_tab_layout.setContentsMargins(0, 0, 0, 0)
        color_tab_layout.setSpacing(12)
        color_tab_layout.addWidget(color_card)
        color_tab_layout.addStretch()

        self.tuning_tabs.addTab(depth_tab, self._qt_text(self._t("Depth")))
        self.tuning_tabs.addTab(subject_tab, self._qt_text(self._t("Subject")))
        self.tuning_tabs.addTab(color_tab, self._qt_text(self._t("Color")))

        right_col.addWidget(assistant_card)
        right_col.addWidget(self.tuning_tabs)

        self.tuning_tabs.setVisible(False)

        right_scroll.setWidget(right_container)
        
        # Resizable 3-column layout
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        center_widget = QWidget()
        center_widget.setLayout(center_col)

        self.main_splitter.addWidget(left_scroll)
        self.main_splitter.addWidget(center_widget)
        self.main_splitter.addWidget(right_scroll)

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)

        self.main_splitter.setSizes([320, 900, 360])

        root.addWidget(self.main_splitter, 1)
        
        self.encoding_dialog = self._build_encoding_dialog()
        self.processing_dialog = self._create_card_dialog(self._t("Processing Options"), processing_card, min_width=500)
        self.keyframe_file_dialog = self._build_keyframe_file_dialog()
        self.preview_settings_dialog = self._build_preview_settings_dialog()

        self._bind_events()
        self._load_initial_state()
        self._refresh_preset_list()
        self._set_render_idle_state()
        self._apply_theme_styles()

    def _t(self, key: str) -> str:
        """
        Translation helper.
        Uses exact keys first, then falls back to older/original JSON keys.
        """
        translator = getattr(self.controller, "t", None)
        if not callable(translator):
            return key

        translations = getattr(self.controller, "translations", None)
        if translations is None:
            language_service = getattr(self.controller, "language_service", None)
            translations = getattr(language_service, "translations", None)

        aliases = {
            "Input Video": "Select Input Video",
            "Depth Map": "Select Depth Map",
            "Output": "Output Path",
            "Choose output file...": "Select Output Video",

            "Frame Preview": "Open Preview",
            "Frame": "Frame",
            "Refresh Preview": "Open Preview",
            "Save Preview Image": "Save Preview",

            "Depth & Parallax": "Depth & Parallax Controls",
            "Pop & Subject Controls": "Depth & Parallax Controls",
            "Stabilize Zero-Parallax": "Use Subject Tracking",

            "Output & Encoding...": "Encoding Settings",
            "Output & Encoding": "Encoding Settings",
            "Processing Options...": "Processing Options",

            "Start Render": "Generate 3D",
            "Start Batch Render": "Start Batch Render",

            "DoF Strength": "DOF Strength",
            "Max Pixel Shift": "Max Pixel Shift %",
            "Basic Codec": "Codec:",
            "FFmpeg Codec": "FFmpeg Codec:",
            "Aspect Ratio": "Aspect Ratio:",
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
        """
        Escape literal ampersands for Qt widgets.

        Qt uses & as a keyboard shortcut marker, so:
        "Output & Encoding" displays wrong.

        This converts it to:
        "Output && Encoding"

        which displays as:
        "Output & Encoding"
        """
        text = str(text)
        marker = "\u0000"
        return text.replace("&&", marker).replace("&", "&&").replace(marker, "&&")

    def _rebuild_combo_translated(self, combo, items):
        """
        Rebuild a combo with translated display text while preserving internal data.
        items = [("Display Key", "internal_value"), ...]
        """
        current_data = combo.currentData()

        combo.blockSignals(True)
        combo.clear()

        for text_key, data_value in items:
            combo.addItem(self._qt_text(self._t(text_key)), data_value)

        index = combo.findData(current_data)
        if index >= 0:
            combo.setCurrentIndex(index)

        combo.blockSignals(False)

    def _combo_key(self, combo, default=""):
        """
        Returns the stable English/internal key for translated combo boxes.
        Falls back to visible text for older combo boxes.
        """
        if combo is None:
            return default

        data = combo.currentData()
        if data is not None:
            return data

        text = combo.currentText()
        return text if text else default


    def _set_combo_by_key(self, combo, key: str):
        """
        Selects a translated combo item by its stored English itemData key.
        Falls back to visible text for older combos.
        """
        if combo is None:
            return

        index = combo.findData(key)
        if index >= 0:
            combo.setCurrentIndex(index)
            return

        combo.setCurrentText(key)


    def _rebuild_key_combo(self, combo, keys, selected_key=None):
        """
        Rebuilds a combo using translated display text while storing
        the original English key in itemData.

        This lets the UI translate without breaking dictionary lookups.
        """
        if combo is None:
            return

        current_key = selected_key
        if current_key is None:
            current_key = combo.currentData()

        if not current_key:
            current_text = combo.currentText()
            current_key = current_text if current_text in keys else None

        combo.blockSignals(True)
        combo.clear()

        for key in keys:
            combo.addItem(self._qt_text(self._t(key)), key)

        index = combo.findData(current_key)
        if index < 0 and keys:
            index = 0

        if index >= 0:
            combo.setCurrentIndex(index)

        combo.blockSignals(False)

    def _refresh_combo_labels(self):
        self._rebuild_combo_translated(
            self.render_mode_combo,
            [
                ("Single Video Render", "video"),
                ("3D Image Render", "image"),
                ("Batch Video Folder Render", "video_folder"),
                ("Image Folder Render", "image_folder"),
            ],
        )

        if hasattr(self, "encoding_preset_combo"):
            self._rebuild_key_combo(
                self.encoding_preset_combo,
                list(ENCODING_PRESETS.keys()),
            )

        if hasattr(self, "vr180_render_preset_combo"):
            self._rebuild_key_combo(
                self.vr180_render_preset_combo,
                list(VR180_RENDER_PRESETS.keys()),
            )

        if hasattr(self, "output_format_combo"):
            self._rebuild_key_combo(
                self.output_format_combo,
                [
                    "Full-SBS",
                    "Half-SBS",
                    "VR",
                    "VR180 Equirect (TB)",
                    "VR180 Equirect (SBS)",
                    "Red-Cyan Anaglyph",
                    "Passive Interlaced",
                ],
            )

        if hasattr(self, "stereo_out_combo"):
            self._rebuild_key_combo(
                self.stereo_out_combo,
                ["sbs", "left", "right", "both"],
            )

        if hasattr(self, "aspect_ratio_combo"):
            self._rebuild_key_combo(
                self.aspect_ratio_combo,
                ASPECT_RATIO_OPTIONS,
            )

        if hasattr(self, "vr180_equi_preset_combo"):
            self._rebuild_key_combo(
                self.vr180_equi_preset_combo,
                list(VR180_EQUI_PRESETS.keys()),
            )

        if hasattr(self, "vr180_flat_preset_combo"):
            self._rebuild_key_combo(
                self.vr180_flat_preset_combo,
                list(VR180_FLAT_PRESETS.keys()),
            )

        if hasattr(self, "edge_repair_quality_combo"):
            self._rebuild_key_combo(
                self.edge_repair_quality_combo,
                ["Off", "Fast", "Balanced", "High", "Showcase"],
            )

        if hasattr(self, "preview_mode_combo"):
            self._rebuild_key_combo(
                self.preview_mode_combo,
                [
                    "Red-Blue Anaglyph",
                    "Passive Interlaced",
                    "HSBS",
                    "Shift Heatmap",
                    "Shift Heatmap (Abs)",
                    "Shift Heatmap (Clipped ±5px)",
                    "Overlay Arrows",
                    "Left-Right Diff",
                    "Feather Mask",
                    "Feather Blend",
                ],
            )

        if hasattr(self, "ffmpeg_codec_combo"):
            self._rebuild_key_combo(
                self.ffmpeg_codec_combo,
                [
                    "H.264 / AVC (libx264 - CPU)",
                    "H.265 / HEVC (libx265 - CPU)",
                    "AV1 (libaom - CPU)",
                    "AV1 (SVT - CPU, faster)",
                    "MPEG-4 (mp4v - CPU)",
                    "XviD (AVI - CPU)",
                    "DivX (AVI - CPU)",
                    "H.264 / AVC (NVENC - NVIDIA GPU)",
                    "H.265 / HEVC (NVENC - NVIDIA GPU)",
                    "AV1 (NVENC - NVIDIA RTX 40+ GPU)",
                    "H.264 / AVC (AMF - AMD GPU)",
                    "H.265 / HEVC (AMF - AMD GPU)",
                    "AV1 (AMF - AMD RDNA3+)",
                    "H.264 / AVC (QSV - Intel GPU)",
                    "H.265 / HEVC (QSV - Intel GPU)",
                    "VP9 (QSV - Intel GPU)",
                    "AV1 (QSV - Intel ARC / Gen11+)",
                ],
            )

        if hasattr(self, "basic_codec_combo"):
            self._rebuild_key_combo(
                self.basic_codec_combo,
                ["mp4v", "XVID", "DIVX"],
            )

        if hasattr(self, "encoding_preset_desc"):
            self._update_encoding_preset_description()

        if hasattr(self, "vr180_preset_desc"):
            self._update_vr180_preset_description()

        if hasattr(self, "active_encoder_label"):
            self._update_active_encoder_label()

    def _refresh_tab_labels(self):
        """
        Refresh labels for QTabWidget tabs.
        These are not QLabel/QPushButton widgets, so they do not go through
        the normal _translation_map system.
        """
        if hasattr(self, "tuning_tabs") and self.tuning_tabs is not None:
            tab_keys = ["Depth", "Subject", "Color"]
            for index, key in enumerate(tab_keys):
                if index < self.tuning_tabs.count():
                    self.tuning_tabs.setTabText(index, self._qt_text(self._t(key)))

    def _refresh_keyframe_labels(self):
        """
        Refresh keyframe widgets that are not normal QLabel/QPushButton text.
        """
        if hasattr(self, "keyframe_transition_combo"):
            current_key = self._combo_key(self.keyframe_transition_combo, "smoothstep")
            self._rebuild_key_combo(
                self.keyframe_transition_combo,
                ["cut", "linear", "smoothstep"],
                selected_key=current_key,
            )

        if hasattr(self, "keyframe_transition_spin"):
            self.keyframe_transition_spin.setToolTip(
                self._qt_text(
                    self._t("Frames used to blend into this keyframe. Use 0 for an instant cut.")
                )
            )

        if hasattr(self, "keyframe_combo"):
            self.keyframe_combo.setToolTip(
                self._qt_text(
                    self._t("Shows the frame range controlled by each keyframe.")
                )
            )
            self._refresh_keyframe_combo()

    def _set_title_text(self, widget, text: str):
        """
        Works with QGroupBox and ParameterCard-like widgets.
        """
        text = self._qt_text(text)

        if hasattr(widget, "setTitle"):
            widget.setTitle(text)
            return

        for attr in ("title_label", "header_label", "label"):
            label = getattr(widget, attr, None)
            if label is not None and hasattr(label, "setText"):
                label.setText(text)
                return

        labels = widget.findChildren(QLabel)
        if labels:
            labels[0].setText(text)

    def _register_text(self, widget, key: str):
        self._translation_map.append((widget, key, "text"))
        widget.setText(self._qt_text(self._t(key)))

    def _register_placeholder(self, widget, key: str):
        self._translation_map.append((widget, key, "placeholder"))
        widget.setPlaceholderText(self._qt_text(self._t(key)))

    def _register_title(self, widget, key: str):
        self._translation_map.append((widget, key, "title"))
        self._set_title_text(widget, self._t(key))

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

    def _register_file_row(self, row, label_key: str, placeholder_key: str):
        """
        Supports FilePickerRow if it exposes common label/edit/button attributes.
        """
        self._translation_map.append((row, label_key, "file_row_label"))
        self._translation_map.append((row, placeholder_key, "file_row_placeholder"))
        self._translation_map.append((row, "Browse", "file_row_browse"))
        self._apply_file_row_translation(row, label_key, placeholder_key)

    def _apply_file_row_translation(self, row, label_key: str, placeholder_key: str):
        label_text = self._t(label_key)
        placeholder_text = self._t(placeholder_key)
        browse_text = self._t("Browse")

        if hasattr(row, "set_label_text"):
            row.set_label_text(label_text)
        elif hasattr(row, "label"):
            row.label.setText(label_text)

        if hasattr(row, "set_placeholder_text"):
            row.set_placeholder_text(placeholder_text)
        elif hasattr(row, "edit"):
            row.edit.setPlaceholderText(placeholder_text)
        elif hasattr(row, "line_edit"):
            row.line_edit.setPlaceholderText(placeholder_text)

        if hasattr(row, "set_browse_text"):
            row.set_browse_text(browse_text)
        elif hasattr(row, "browse_btn"):
            row.browse_btn.setText(browse_text)
        elif hasattr(row, "button"):
            row.button.setText(browse_text)

    def _get_file_row_text(self, row) -> str:
        """
        Safely reads text from FilePickerRow without depending on one exact API name.
        """
        if row is None:
            return ""

        for attr in ("text", "path", "file_path", "value"):
            value = getattr(row, attr, None)
            if isinstance(value, str):
                return value.strip()

        for method_name in ("get_text", "text", "get_path", "path", "get_value"):
            method = getattr(row, method_name, None)
            if callable(method):
                try:
                    value = method()
                    if value is not None:
                        return str(value).strip()
                except Exception:
                    pass

        for attr in ("edit", "line_edit", "path_edit"):
            widget = getattr(row, attr, None)
            if widget is not None and hasattr(widget, "text"):
                try:
                    return str(widget.text()).strip()
                except Exception:
                    pass

        return ""


    def _set_file_row_text(self, row, text: str):
        """
        Safely writes text to FilePickerRow without depending on one exact API name.
        """
        if row is None:
            return

        text = str(text)

        for method_name in ("set_text", "set_path", "set_value"):
            method = getattr(row, method_name, None)
            if callable(method):
                try:
                    method(text)
                    return
                except Exception:
                    pass

        for attr in ("edit", "line_edit", "path_edit"):
            widget = getattr(row, attr, None)
            if widget is not None and hasattr(widget, "setText"):
                try:
                    widget.setText(text)
                    return
                except Exception:
                    pass


    def _get_output_row_text(self) -> str:
        return self._get_file_row_text(getattr(self, "output_row", None))


    def _set_output_row_text(self, text: str):
        self._set_file_row_text(getattr(self, "output_row", None), text)

    def apply_theme(self, theme: dict):
        self._active_theme = theme or {}
        apply_unified_page_theme(self, self._active_theme)
            
    def _apply_theme_styles(self):
        theme = getattr(self, "_active_theme", None) or {}
        colors = theme.get("colors", {}) if isinstance(theme, dict) else {}

        panel = colors.get("panel", "#111821")
        panel_2 = colors.get("panel_2", "#0d131b")
        panel_3 = colors.get("panel_3", "#151d29")
        border = colors.get("border", "#263445")
        border_soft = colors.get("border_soft", "#2d3b4f")
        text = colors.get("text", "#e6edf3")
        text_bright = colors.get("text_bright", "#f0f6fc")
        muted = colors.get("muted", "#8b949e")
        accent = colors.get("accent", "#2f81f7")
        danger = colors.get("danger", "#ff7b72")

        if hasattr(self, "preview_meta_label"):
            self.preview_meta_label.setStyleSheet(f"""
                QLabel {{
                    color: {text};
                    padding: 6px 8px;
                    border: 1px solid {border_soft};
                    border-radius: 8px;
                    background-color: {panel_3};
                }}
            """)

        # Optional: theme this page's scrollbars/sliders/buttons a bit more too.
        self.setStyleSheet(f"""
            QLabel {{
                color: {text};
            }}

            QScrollArea {{
                background: transparent;
                border: none;
            }}

            QComboBox,
            QLineEdit,
            QSpinBox,
            QDoubleSpinBox {{
                background-color: {panel_2};
                border: 1px solid {border_soft};
                border-radius: 8px;
                padding: 6px 8px;
                color: {text};
            }}

            QComboBox:hover,
            QLineEdit:hover,
            QSpinBox:hover,
            QDoubleSpinBox:hover {{
                border: 1px solid {accent};
            }}

            QPushButton {{
                background-color: {panel_3};
                border: 1px solid {border_soft};
                border-radius: 8px;
                padding: 7px 12px;
                color: {text_bright};
                font-weight: 600;
            }}

            QPushButton:hover {{
                background-color: {panel};
                border: 1px solid {accent};
            }}

            QGroupBox {{
                background-color: {panel};
                border: 1px solid {border};
                border-radius: 12px;
                margin-top: 10px;
                padding: 12px;
                color: {text_bright};
                font-weight: 700;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {text_bright};
            }}

            QSlider::groove:horizontal {{
                height: 6px;
                background: {panel_2};
                border: 1px solid {border_soft};
                border-radius: 3px;
            }}

            QSlider::handle:horizontal {{
                background: {accent};
                border: 1px solid {accent};
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}

            QScrollBar:vertical {{
                background: transparent;
                width: 10px;
                margin: 2px;
            }}

            QScrollBar::handle:vertical {{
                background: {border};
                border-radius: 5px;
                min-height: 32px;
            }}

            QScrollBar::handle:vertical:hover {{
                background: {border_soft};
            }}

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    def refresh_labels(self):
        """
        Called by MainWindow when language changes.
        """
        for widget, key, widget_type in self._translation_map:
            try:
                if widget_type == "text":
                    widget.setText(self._qt_text(self._t(key)))

                elif widget_type == "placeholder":
                    widget.setPlaceholderText(self._qt_text(self._t(key)))

                elif widget_type == "title":
                    self._set_title_text(widget, self._t(key))

                elif widget_type == "file_row_label":
                    self._apply_file_row_translation(widget, key, "")

                elif widget_type == "file_row_placeholder":
                    pass

                elif widget_type == "file_row_browse":
                    pass

            except RuntimeError:
                pass

        # Re-apply FilePickerRow translations as full rows
        for item in self._translation_map:
            widget, key, widget_type = item
            if widget_type == "file_row_label":
                if widget is self.input_row:
                    self._apply_file_row_translation(widget, "Input Video", "Select source video...")
                elif widget is self.depth_row:
                    self._apply_file_row_translation(widget, "Depth Map", "Select depth video...")
                elif widget is self.output_row:
                    self._apply_file_row_translation(widget, "Output", "Choose output file...")

        # Refresh dialog window titles
        if hasattr(self, "encoding_dialog") and self.encoding_dialog is not None:
            self.encoding_dialog.setWindowTitle(
                self._qt_text(self._t("Output & Encoding"))
            )

        if hasattr(self, "processing_dialog") and self.processing_dialog is not None:
            self.processing_dialog.setWindowTitle(
                self._qt_text(self._t("Processing Options"))
            )

        self._refresh_combo_labels()
        self._refresh_combo_labels()
        self._refresh_tab_labels()
        self._refresh_keyframe_labels()
        self._rebuild_simple_3d_style_combo()
        self._update_encoding_preset_description()
        self._update_vr180_preset_description()
        self._update_active_encoder_label()

        if hasattr(self, "encoding_advanced_btn"):
            self._set_encoding_advanced_visible(
                bool(self.encoding_advanced_btn.isChecked())
            )

        if hasattr(self, "vr180_advanced_btn"):
            self._set_vr180_advanced_visible(
                bool(self.vr180_advanced_btn.isChecked())
            )

        if hasattr(self, "preview_panel") and self.preview_panel is not None:
            self.preview_panel.set_translator(self._t)
            self.preview_panel.refresh_labels()

        # Refresh manually controlled button text.
        if hasattr(self, "advanced_3d_btn"):
            self._set_advanced_3d_visible(
                bool(self.advanced_3d_btn.isChecked())
            )

        self._apply_render_mode_ui()
        self._update_frame_label()
        self._update_simple_3d_labels()
        self._refresh_preview_meta()

    def _current_render_mode(self) -> str:
        if hasattr(self, "render_mode_combo"):
            return self.render_mode_combo.currentData() or "video"
        return getattr(self.controller.state, "render_mode", "video")

    def _set_combo_by_data(self, combo, data):
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    def _on_render_mode_changed(self, *_args):
        mode = self._current_render_mode()
        print(f"[UI RENDER MODE] {mode}")
        self.controller.set_state("render_mode", mode)
        self._apply_render_mode_ui()
        self._refresh_preview_meta()

    def _apply_render_mode_ui(self):
        mode = self._current_render_mode()

        if mode == "image":
            self._apply_file_row_translation(self.input_row, "Input Image", "Select source image...")
            self._apply_file_row_translation(self.depth_row, "Depth Map Image", "Select depth image...")
            self._apply_file_row_translation(self.output_row, "Output Image", "Choose output image...")
            self.render_btn.setText(self._t("Render 3D Image"))
            self.preview_btn.setEnabled(False)

        elif mode == "video_folder":
            self._apply_file_row_translation(self.input_row, "Input Video Folder", "Select video folder...")
            self._apply_file_row_translation(self.depth_row, "Depth Video Folder", "Select depth video folder...")
            self._apply_file_row_translation(self.output_row, "Output Folder", "Select output folder...")
            self.render_btn.setText(self._t("Start Batch Render"))
            self.preview_btn.setEnabled(False)

        elif mode == "image_folder":
            self._apply_file_row_translation(self.input_row, "Input Image Folder", "Select image folder...")
            self._apply_file_row_translation(self.depth_row, "Depth Image Folder", "Select depth image folder...")
            self._apply_file_row_translation(self.output_row, "Output Folder", "Select output folder...")
            self.render_btn.setText(self._t("Start Image Folder Render"))
            self.preview_btn.setEnabled(False)

        else:
            self._apply_file_row_translation(self.input_row, "Input Video", "Select source video...")
            self._apply_file_row_translation(self.depth_row, "Depth Map", "Select depth video...")
            self._apply_file_row_translation(self.output_row, "Output", "Choose output file...")
            self.render_btn.setText(self._t("Start Render"))
            self.preview_btn.setEnabled(True)

    def _start_render_clicked(self):
        mode = self._current_render_mode()
        print(f"[START CLICKED RENDER MODE] {mode}")
        self.controller.set_state("render_mode", mode)
        self.controller.start_render()

    def _step_preview_frame(self, direction: int):
        """
        Moves the preview frame backward or forward by the step size.
        direction:
            -1 = previous
             1 = next
        """
        step = int(self.frame_step_spin.value()) if hasattr(self, "frame_step_spin") else 1

        current = int(self.frame_slider.value())
        minimum = int(self.frame_slider.minimum())
        maximum = int(self.frame_slider.maximum())

        new_value = current + (step * int(direction))
        new_value = max(minimum, min(maximum, new_value))

        if new_value == current:
            return

        self.frame_slider.setValue(new_value)
        self.controller.set_state("preview_frame_index", new_value)

        if hasattr(self, "_update_frame_label"):
            self._update_frame_label()

        if hasattr(self, "_preview_debounce"):
            self._preview_debounce.start()

    def _clamp(self, value, minimum, maximum):
        return max(minimum, min(maximum, value))


    def _capture_current_advanced_3d_settings(self) -> dict:
        """
        Captures the current advanced 3D controls as the new simple-slider base.

        This lets the user tune advanced controls first, then use the beginner
        sliders as relative trims instead of snapping back to the original style preset.
        """
        def slider_value(name: str, scale: float, default: float = 0.0) -> float:
            slider = getattr(self, name, None)
            if slider is None:
                return float(default)

            try:
                return float(slider.value()) / float(scale)
            except Exception:
                return float(default)

        def check_value(name: str, default: bool = False) -> bool:
            widget = getattr(self, name, None)
            if widget is None:
                return bool(default)

            try:
                return bool(widget.isChecked())
            except Exception:
                return bool(default)

        def combo_key(name: str, default: str = "") -> str:
            combo = getattr(self, name, None)
            if combo is None:
                return str(default)

            try:
                return str(self._combo_key(combo, default))
            except Exception:
                return str(default)

        def line_float(name: str, default: float) -> float:
            edit = getattr(self, name, None)
            if edit is None:
                return float(default)

            try:
                return float(str(edit.text()).strip())
            except Exception:
                return float(default)

        state = getattr(self.controller, "state", None)

        return {
            "fg_shift": slider_value("fg_shift_slider", 10.0, getattr(state, "fg_shift", -8.0)),
            "mg_shift": slider_value("mg_shift_slider", 10.0, getattr(state, "mg_shift", -2.0)),
            "bg_shift": slider_value("bg_shift_slider", 10.0, getattr(state, "bg_shift", 2.5)),

            "max_pixel_shift": slider_value("max_pixel_shift_slider", 1000.0, getattr(state, "max_pixel_shift", 0.045)),
            "zero_parallax_strength": slider_value("zero_parallax_slider", 1000.0, getattr(state, "zero_parallax_strength", 0.0)),
            "parallax_balance": slider_value("parallax_balance_slider", 100.0, getattr(state, "parallax_balance", 0.8)),
            "convergence_strength": slider_value("convergence_slider", 1000.0, getattr(state, "convergence_strength", 0.0)),
            "sharpness_factor": slider_value("sharpness_slider", 100.0, getattr(state, "sharpness_factor", 0.0)),
            "dof_strength": slider_value("dof_slider", 10.0, getattr(state, "dof_strength", 0.0)),

            "depth_pop_gamma": slider_value("depth_pop_gamma_slider", 100.0, getattr(state, "depth_pop_gamma", 0.85)),
            "depth_pop_mid": line_float("pop_mid_edit", getattr(state, "depth_pop_mid", 0.50)),
            "depth_stretch_lo": line_float("stretch_lo_edit", getattr(state, "depth_stretch_lo", 0.05)),
            "depth_stretch_hi": line_float("stretch_hi_edit", getattr(state, "depth_stretch_hi", 0.95)),
            "fg_pop_multiplier": slider_value("fg_pop_slider", 100.0, getattr(state, "fg_pop_multiplier", 1.08)),
            "bg_push_multiplier": slider_value("bg_push_slider", 100.0, getattr(state, "bg_push_multiplier", 1.04)),

            "subject_lock_strength": slider_value("subject_lock_slider", 100.0, getattr(state, "subject_lock_strength", 0.34)),
            "subject_plane_lock_strength": slider_value("subject_plane_lock_slider", 100.0, getattr(state, "subject_plane_lock_strength", 0.0)),
            "subject_plane_lock_width": slider_value("subject_plane_width_slider", 100.0, getattr(state, "subject_plane_lock_width", 0.08)),
            "subject_screen_plane": getattr(state, "subject_screen_plane", 0.0),

            "foreground_curvature_strength": slider_value("foreground_curvature_slider", 100.0, getattr(state, "foreground_curvature_strength", 0.06)),

            "use_subject_tracking": check_value("subject_tracking_check", getattr(state, "use_subject_tracking", False)),
            "enable_dynamic_convergence": check_value("dynamic_convergence_check", getattr(state, "enable_dynamic_convergence", True)),
            "use_floating_window": check_value("floating_window_check", getattr(state, "use_floating_window", False)),
            "enable_edge_masking": check_value("edge_masking_check", getattr(state, "enable_edge_masking", True)),
            "enable_feathering": check_value("feathering_check", getattr(state, "enable_feathering", True)),
            "disable_shift_ema": check_value("disable_shift_ema_check", getattr(state, "disable_shift_ema", False)),
            "edge_repair_quality": combo_key("edge_repair_quality_combo", getattr(state, "edge_repair_quality", "Fast")),
        }


    def _mark_advanced_3d_base_dirty(self):
        """
        Marks the advanced setup as manually changed.

        The next beginner-slider movement will capture the current advanced controls
        as its new base instead of going back to the old style preset.
        """
        if getattr(self, "_applying_3d_assistant", False):
            return

        if getattr(self, "_applying_simple_3d_adjustment", False):
            return

        self._advanced_3d_base_dirty = True


    def _lerp(self, a, b, t):
        t = self._clamp(float(t), 0.0, 1.0)
        return a + ((b - a) * t)


    def _set_advanced_3d_visible(self, visible: bool):
        visible = bool(visible)

        if hasattr(self, "tuning_tabs"):
            self.tuning_tabs.setVisible(visible)

        if hasattr(self, "advanced_3d_btn"):
            self.advanced_3d_btn.setChecked(visible)
            self.advanced_3d_btn.setText(
                self._qt_text(
                    self._t("Hide Advanced 3D Controls")
                    if visible
                    else self._t("Advanced 3D Controls")
                )
            )

    def _rebuild_simple_3d_style_combo(self, selected_key=None):
        """
        Rebuilds the 3D Assistant preset combo with translated display text,
        while keeping the original English preset key in itemData.

        This avoids breaking SIMPLE_3D_PRESETS lookups when the UI language changes.
        """
        if not hasattr(self, "simple_3d_style_combo"):
            return

        combo = self.simple_3d_style_combo

        current_key = selected_key
        if current_key is None:
            current_key = combo.currentData()

        if not current_key:
            current_text = combo.currentText()
            current_key = current_text if current_text in SIMPLE_3D_PRESETS else "Custom"

        combo.blockSignals(True)
        combo.clear()

        for preset_key in SIMPLE_3D_PRESETS.keys():
            combo.addItem(self._qt_text(self._t(preset_key)), preset_key)

        index = combo.findData(current_key)
        if index < 0:
            index = combo.findData("Custom")

        if index >= 0:
            combo.setCurrentIndex(index)

        combo.blockSignals(False)

    def _update_simple_3d_labels(self):
        if hasattr(self, "simple_strength_value"):
            self.simple_strength_value.setText(str(int(self.simple_strength_slider.value())))

        if hasattr(self, "simple_pop_value"):
            self.simple_pop_value.setText(str(int(self.simple_pop_slider.value())))

        if hasattr(self, "simple_comfort_value"):
            self.simple_comfort_value.setText(str(int(self.simple_comfort_slider.value())))

        if hasattr(self, "simple_stability_value"):
            self.simple_stability_value.setText(str(int(self.simple_stability_slider.value())))

        if hasattr(self, "simple_screen_depth_value") and hasattr(self, "simple_screen_depth_slider"):
            value = int(self.simple_screen_depth_slider.value())

            if value < 45:
                label = f"{value}  {self._t('Closer')}"
            elif value > 55:
                label = f"{value}  {self._t('Deeper')}"
            else:
                label = f"{value}  {self._t('Neutral')}"

            self.simple_screen_depth_value.setText(label)

        if hasattr(self, "simple_subject_plane_value") and hasattr(self, "simple_subject_plane_slider"):
            value = int(self.simple_subject_plane_slider.value())

            if value <= 0:
                label = f"{value}%  {self._t('Off')}"
            elif value < 35:
                label = f"{value}%  {self._t('Light')}"
            elif value < 70:
                label = f"{value}%  {self._t('Medium')}"
            else:
                label = f"{value}%  {self._t('Strong')}"

            self.simple_subject_plane_value.setText(label)

        if hasattr(self, "simple_3d_desc"):
            preset_key = self.simple_3d_style_combo.currentData() or "Custom"
            preset = SIMPLE_3D_PRESETS.get(preset_key, SIMPLE_3D_PRESETS["Custom"])

            description = preset.get("description", "")
            self.simple_3d_desc.setText(self._qt_text(self._t(description)))
                
    def _set_simple_3d_preset_custom(self):
        """
        Keep the selected style name visible while the user tweaks sliders.

        Old behavior changed the combo to Custom immediately.
        That made the UI feel like the chosen style disappeared.
        """
        if getattr(self, "_applying_3d_assistant", False):
            return

        if not hasattr(self, "simple_3d_style_combo"):
            return

        # Do not switch to Custom here.
        # The sliders now modify the active style base instead.
        self._update_simple_3d_labels()


    def _on_simple_3d_style_changed(self, *_args):
        preset_key = self.simple_3d_style_combo.currentData() or "Custom"
        preset = SIMPLE_3D_PRESETS.get(preset_key)

        if not preset:
            return

        self._active_3d_style_key = preset_key

        self._update_simple_3d_labels()

        if preset_key == "Custom":
            self._active_3d_style_base = self._capture_current_advanced_3d_settings()
            self._advanced_3d_base_dirty = False
            return
            
        advanced_preset = SIMPLE_3D_ADVANCED_PRESETS.get(preset_key, {})

        self._applying_3d_assistant = True

        try:
            for slider, value in [
                (self.simple_strength_slider, int(preset.get("strength", 50))),
                (self.simple_pop_slider, int(preset.get("pop", 50))),
                (self.simple_comfort_slider, int(preset.get("comfort", 70))),
                (self.simple_stability_slider, int(preset.get("stability", 60))),
            ]:
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)

            if hasattr(self, "simple_screen_depth_slider"):
                self.simple_screen_depth_slider.blockSignals(True)
                self.simple_screen_depth_slider.setValue(int(preset.get("screen_depth", 50)))
                self.simple_screen_depth_slider.blockSignals(False)

            if hasattr(self, "simple_subject_plane_slider"):
                self.simple_subject_plane_slider.blockSignals(True)
                self.simple_subject_plane_slider.setValue(int(preset.get("subject_plane", 0)))
                self.simple_subject_plane_slider.blockSignals(False)

            self._update_simple_3d_labels()

            if advanced_preset:
                self._active_3d_style_base = dict(advanced_preset)
                self._advanced_3d_base_dirty = False
                self._apply_advanced_3d_preset_values(advanced_preset)
            else:
                self._active_3d_style_base = self._capture_current_advanced_3d_settings()
                self._advanced_3d_base_dirty = False
                self._apply_simple_3d_controls()

        finally:
            self._applying_3d_assistant = False

    def _apply_advanced_3d_preset_values(self, preset: dict):
        """
        Applies known-good advanced 3D preset values from tested JSON presets.

        This is used by the 3D Assistant dropdown.
        Manual movement of the simple sliders can still use _apply_simple_3d_controls().
        """

        def set_slider(slider_name: str, scale: float, key: str):
            if key not in preset:
                return

            slider = getattr(self, slider_name, None)
            if slider is None:
                return

            try:
                slider.setValue(int(round(float(preset[key]) * scale)))
            except Exception as e:
                print(f"[3D ASSISTANT] Could not set {key}: {e}")

        def set_check(check_name: str, key: str):
            if key not in preset:
                return

            check = getattr(self, check_name, None)
            if check is None:
                return

            try:
                check.setChecked(bool(preset[key]))
            except Exception as e:
                print(f"[3D ASSISTANT] Could not set {key}: {e}")

        def set_line_edit(edit_name: str, key: str):
            if key not in preset:
                return

            edit = getattr(self, edit_name, None)
            if edit is None:
                return

            try:
                edit.setText(str(preset[key]))
            except Exception as e:
                print(f"[3D ASSISTANT] Could not set {key}: {e}")

        def set_state(key: str):
            if key not in preset:
                return

            try:
                self.controller.set_state(key, preset[key])
            except Exception:
                pass

        # Shift sliders.
        set_slider("fg_shift_slider", 10, "fg_shift")
        set_slider("mg_shift_slider", 10, "mg_shift")
        set_slider("bg_shift_slider", 10, "bg_shift")

        # Depth and parallax sliders.
        set_slider("max_pixel_shift_slider", 1000, "max_pixel_shift")
        set_slider("zero_parallax_slider", 1000, "zero_parallax_strength")
        set_slider("parallax_balance_slider", 100, "parallax_balance")
        set_slider("convergence_slider", 1000, "convergence_strength")
        set_slider("sharpness_slider", 100, "sharpness_factor")
        set_slider("dof_slider", 10, "dof_strength")

        # Pop and subject sliders.
        set_slider("depth_pop_gamma_slider", 100, "depth_pop_gamma")
        set_slider("fg_pop_slider", 100, "fg_pop_multiplier")
        set_slider("bg_push_slider", 100, "bg_push_multiplier")
        set_slider("foreground_curvature_slider", 100, "foreground_curvature_strength")

        set_slider("subject_lock_slider", 100, "subject_lock_strength")
        set_slider("subject_plane_lock_slider", 100, "subject_plane_lock_strength")
        set_slider("subject_plane_width_slider", 100, "subject_plane_lock_width")

        # Subject Zero Lock lives in the simple assistant as 0..100,
        # while backend state uses 0.0..1.0.
        if "subject_screen_plane" in preset and hasattr(self, "simple_subject_plane_slider"):
            try:
                value = float(preset.get("subject_screen_plane", 0.0))
                ui_value = int(round(max(0.0, min(1.0, value)) * 100.0))
                self.simple_subject_plane_slider.blockSignals(True)
                self.simple_subject_plane_slider.setValue(max(0, min(100, ui_value)))
                self.simple_subject_plane_slider.blockSignals(False)
            except Exception as e:
                print(f"[3D ASSISTANT] Could not set subject zero lock: {e}")

        # Text entries.
        set_line_edit("pop_mid_edit", "depth_pop_mid")
        set_line_edit("stretch_lo_edit", "depth_stretch_lo")
        set_line_edit("stretch_hi_edit", "depth_stretch_hi")

        # Processing toggles.
        set_check("subject_tracking_check", "use_subject_tracking")
        set_check("dynamic_convergence_check", "enable_dynamic_convergence")
        set_check("floating_window_check", "use_floating_window")
        set_check("edge_masking_check", "enable_edge_masking")
        set_check("feathering_check", "enable_feathering")
        set_check("disable_shift_ema_check", "disable_shift_ema")

        # Edge repair preset.
        if "edge_repair_quality" in preset and hasattr(self, "edge_repair_quality_combo"):
            self.edge_repair_quality_combo.setCurrentText(str(preset["edge_repair_quality"]))

        # Make sure line-edit-only values are pushed to state too.
        for key in [
            "depth_pop_mid",
            "depth_stretch_lo",
            "depth_stretch_hi",
            "fg_shift",
            "mg_shift",
            "bg_shift",
            "max_pixel_shift",
            "zero_parallax_strength",
            "parallax_balance",
            "convergence_strength",
            "sharpness_factor",
            "dof_strength",
            "depth_pop_gamma",
            "fg_pop_multiplier",
            "bg_push_multiplier",
            "foreground_curvature_strength",
            "subject_lock_strength",
            "subject_plane_lock_strength",
            "subject_plane_lock_width",
            "subject_screen_plane",
            "use_subject_tracking",
            "enable_dynamic_convergence",
            "use_floating_window",
            "enable_edge_masking",
            "enable_feathering",
            "disable_shift_ema",
            "edge_repair_quality",
        ]:
            set_state(key)

        self._update_simple_3d_labels()

        if hasattr(self, "_preview_debounce"):
            self._preview_debounce.start()

    def _on_simple_3d_slider_changed(self, *_args):
        if getattr(self, "_applying_3d_assistant", False):
            return

        # If the user manually adjusted advanced controls, use those current
        # advanced values as the new base before applying beginner trims.
        if getattr(self, "_advanced_3d_base_dirty", False):
            self._active_3d_style_key = "Custom"
            self._active_3d_style_base = self._capture_current_advanced_3d_settings()
            self._advanced_3d_base_dirty = False

            if hasattr(self, "simple_3d_style_combo"):
                self.simple_3d_style_combo.blockSignals(True)
                self._set_combo_by_key(self.simple_3d_style_combo, "Custom")
                self.simple_3d_style_combo.blockSignals(False)

        self._update_simple_3d_labels()
        self._apply_simple_3d_controls_from_active_style()


    def _apply_simple_3d_controls_from_active_style(self):
        """
        Adjusts the currently selected 3D style instead of replacing it.

        This fixes the issue where selecting a strong style, then touching a simple
        slider, caused the 3D effect to collapse back to a generic weaker formula.
        """
        preset_key = getattr(self, "_active_3d_style_key", "Custom")
        base = dict(getattr(self, "_active_3d_style_base", {}) or {})

        if not base and preset_key in SIMPLE_3D_ADVANCED_PRESETS:
            base = dict(SIMPLE_3D_ADVANCED_PRESETS[preset_key])
            self._active_3d_style_base = dict(base)

        if not base:
            base = self._capture_current_advanced_3d_settings()
            self._active_3d_style_base = dict(base)

        strength = self.simple_strength_slider.value() / 100.0
        pop = self.simple_pop_slider.value() / 100.0
        comfort = self.simple_comfort_slider.value() / 100.0
        stability = self.simple_stability_slider.value() / 100.0
        screen_depth = (
            self.simple_screen_depth_slider.value() / 100.0
            if hasattr(self, "simple_screen_depth_slider")
            else 0.50
        )
        subject_zero_lock = (
            self.simple_subject_plane_slider.value() / 100.0
            if hasattr(self, "simple_subject_plane_slider")
            else 0.0
        )

        # Sliders become relative trim controls around the selected style.
        # 50 means close to the style base.
        strength_gain = self._lerp(0.75, 1.25, strength)
        pop_gain = self._lerp(0.85, 1.20, pop)
        comfort_gain = self._lerp(1.08, 0.88, comfort)
        stability_gain = self._lerp(0.75, 1.20, stability)

        # Screen depth shifts the whole stereo plane closer or deeper.
        # 50 is neutral. Lower = closer / more pop. Higher = deeper / more inside screen.
        screen_offset = self._lerp(-0.012, 0.012, screen_depth)

        # Subject Zero Lock is a local subject disparity cancel strength.
        # 0.0 = off, 1.0 = strong local subject anchoring.
        subject_screen_plane = self._clamp(subject_zero_lock, 0.0, 1.0)

        adjusted = dict(base)

        def get_float(key, default):
            try:
                return float(base.get(key, default))
            except Exception:
                return float(default)

        # Keep the style's personality, then scale around it.
        adjusted["fg_shift"] = get_float("fg_shift", -8.0) * strength_gain * pop_gain * comfort_gain
        adjusted["mg_shift"] = get_float("mg_shift", -2.5) * strength_gain * comfort_gain
        adjusted["bg_shift"] = get_float("bg_shift", 2.5) * strength_gain * comfort_gain

        adjusted["max_pixel_shift"] = self._clamp(
            get_float("max_pixel_shift", 0.070) * self._lerp(0.85, 1.12, strength) * self._lerp(1.06, 0.92, comfort),
            0.020,
            0.100,
        )

        adjusted["zero_parallax_strength"] = self._clamp(
            get_float("zero_parallax_strength", 0.0) + screen_offset,
            -0.050,
            0.050,
        )

        adjusted["parallax_balance"] = self._clamp(
            get_float("parallax_balance", 0.40) * self._lerp(0.88, 1.12, strength),
            0.05,
            1.00,
        )

        adjusted["convergence_strength"] = self._clamp(
            get_float("convergence_strength", 0.05) * self._lerp(0.85, 1.20, pop) * self._lerp(1.05, 0.85, comfort),
            -1.0,
            1.0,
        )

        adjusted["depth_pop_gamma"] = self._clamp(
            get_float("depth_pop_gamma", 0.90) * self._lerp(1.05, 0.92, pop),
            0.70,
            1.20,
        )

        adjusted["fg_pop_multiplier"] = self._clamp(
            get_float("fg_pop_multiplier", 1.05) * self._lerp(0.95, 1.12, pop),
            1.00,
            1.60,
        )

        adjusted["bg_push_multiplier"] = self._clamp(
            get_float("bg_push_multiplier", 1.05) * self._lerp(0.95, 1.12, screen_depth),
            1.00,
            1.40,
        )

        adjusted["subject_lock_strength"] = self._clamp(
            get_float("subject_lock_strength", 0.8) * stability_gain,
            0.0,
            2.0,
        )

        adjusted["subject_plane_lock_strength"] = self._clamp(
            get_float("subject_plane_lock_strength", 0.25) * self._lerp(0.75, 1.35, stability),
            0.0,
            1.0,
        )

        adjusted["subject_plane_lock_width"] = self._clamp(
            get_float("subject_plane_lock_width", 0.12),
            0.01,
            0.30,
        )

        adjusted["subject_screen_plane"] = self._clamp(
            subject_screen_plane,
            0.0,
            1.0,
        )

        adjusted["foreground_curvature_strength"] = self._clamp(
            get_float("foreground_curvature_strength", 0.05) * self._lerp(0.90, 1.15, pop),
            0.0,
            0.20,
        )

        adjusted["dof_strength"] = self._clamp(
            get_float("dof_strength", 0.30),
            0.0,
            5.0,
        )

        # Keep important style toggles from the base preset.
        for key in [
            "use_subject_tracking",
            "enable_dynamic_convergence",
            "use_floating_window",
            "enable_edge_masking",
            "enable_feathering",
            "disable_shift_ema",
            "edge_repair_quality",
            "depth_pop_mid",
            "depth_stretch_lo",
            "depth_stretch_hi",
            "sharpness_factor",
        ]:
            if key in base:
                adjusted[key] = base[key]

        self._applying_simple_3d_adjustment = True
        try:
            self._apply_advanced_3d_preset_values(adjusted)
        finally:
            self._applying_simple_3d_adjustment = False

    def _apply_simple_3d_controls(self):
        """
        Converts simple user-facing controls into the existing advanced VD3D sliders.
        This keeps the backend unchanged.
        """
        if not hasattr(self, "simple_strength_slider"):
            return

        strength = self.simple_strength_slider.value() / 100.0
        pop = self.simple_pop_slider.value() / 100.0
        comfort = self.simple_comfort_slider.value() / 100.0
        stability = self.simple_stability_slider.value() / 100.0
        screen_depth = self.simple_screen_depth_slider.value() / 100.0 if hasattr(self, "simple_screen_depth_slider") else 0.50
        subject_zero_lock = self.simple_subject_plane_slider.value() / 100.0 if hasattr(self, "simple_subject_plane_slider") else 0.0
        # 3D Strength should directly control visible stereo separation.
        # Comfort should limit strain, not erase the shift.
        strength_curve = strength ** 0.85
        pop_curve = pop ** 0.90
        comfort_limit = 1.0 - (0.10 * comfort)

        # Shift behavior.
        # These values stay inside your tested no-tear range:
        # FG about -2.8 to -9.3, MG about -0.8 to -2.8, BG about 0.7 to 3.0.
        fg_shift = -self._lerp(2.8, 9.3, strength_curve)
        mg_shift = -self._lerp(0.8, 2.8, strength_curve)
        bg_shift = self._lerp(0.7, 3.0, strength_curve)

        # Pop-Out should add a small extra foreground push, not replace 3D Strength.
        fg_shift *= self._lerp(0.95, 1.08, pop_curve)

        # Comfort only trims the very top end.
        fg_shift *= comfort_limit
        mg_shift *= comfort_limit
        bg_shift *= comfort_limit

        fg_shift = self._clamp(fg_shift, -9.3, -2.0)
        mg_shift = self._clamp(mg_shift, -2.8, -0.3)
        bg_shift = self._clamp(bg_shift, 0.2, 3.0)

        # Max Pixel Shift should follow 3D Strength clearly.
        # This is the global shift ceiling, so keep it visible but still safe.
        max_pixel_shift = self._lerp(0.030, 0.071, strength_curve)
        max_pixel_shift *= 1.0 - (0.08 * comfort)
        max_pixel_shift = self._clamp(max_pixel_shift, 0.025, 0.071)

        # Keep parallax balance in a useful visible range.
        # Too low feels flat. Too high can get aggressive.
        parallax_balance = self._lerp(0.42, 0.68, strength_curve)
        parallax_balance *= 1.0 - (0.04 * comfort)
        parallax_balance = self._clamp(parallax_balance, 0.35, 0.70)

        # Pop-Out should control convergence more than 3D Strength.
        convergence = self._lerp(0.015, 0.12, pop_curve)
        convergence *= 1.0 - (0.25 * comfort)
        convergence = self._clamp(convergence, -0.20, 0.20)

        # Screen Depth controls where the screen plane sits.
        # 0.00 = closer / more pop-out feeling
        # 0.50 = neutral
        # 1.00 = deeper behind the screen
        screen_depth_centered = (screen_depth - 0.50) * 2.0

        # Keep the simple range safe. Advanced controls can still go wider.
        screen_plane_offset = self._lerp(-0.014, 0.014, screen_depth)
        screen_plane_offset *= 1.0 - (0.10 * comfort)
        screen_plane_offset = self._clamp(screen_plane_offset, -0.018, 0.018)
        
        # Pop and shape behavior
        depth_pop_gamma = self._lerp(0.95, 0.76, pop)
        # Keep shape enhancement more conservative.
        # Too much foreground curvature + pop multiplier can exaggerate object borders.
        fg_pop = self._lerp(1.00, 1.32, pop) * (1.0 - (0.08 * comfort))
        bg_push = self._lerp(1.00, 1.18, strength) * (1.0 - (0.12 * comfort))
        foreground_curvature = self._lerp(0.01, 0.075, pop) * (1.0 - (0.35 * comfort))

        # Subject stability behavior.
        # Use a soft curve so stability does not suddenly jump around 45 percent.
        stability_curve = stability * stability

        subject_lock = self._lerp(0.25, 1.05, stability_curve)
        subject_plane_lock = self._lerp(0.00, 0.40, stability_curve) * self._lerp(0.70, 1.00, comfort)
        subject_width = self._lerp(0.06, 0.14, stability_curve)
        subject_screen_plane = self._clamp(subject_zero_lock, 0.0, 1.0)

        # Smaller supporting controls
        sharpness = self._lerp(0.00, 0.25, strength) * (1.0 - (0.35 * comfort))
        dof = self._lerp(0.0, 1.0, strength) * (1.0 - (0.25 * comfort))

        # Apply to existing sliders.
        self.fg_shift_slider.setValue(int(round(fg_shift * 10)))
        self.mg_shift_slider.setValue(int(round(mg_shift * 10)))
        self.bg_shift_slider.setValue(int(round(bg_shift * 10)))

        self.max_pixel_shift_slider.setValue(int(round(max_pixel_shift * 1000)))
        self.parallax_balance_slider.setValue(int(round(parallax_balance * 100)))
        self.convergence_slider.setValue(int(round(convergence * 1000)))
        self.zero_parallax_slider.setValue(int(round(screen_plane_offset * 1000)))

        self.depth_pop_gamma_slider.setValue(int(round(depth_pop_gamma * 100)))
        self.fg_pop_slider.setValue(int(round(fg_pop * 100)))
        self.bg_push_slider.setValue(int(round(bg_push * 100)))
        self.foreground_curvature_slider.setValue(int(round(foreground_curvature * 100)))

        self.subject_lock_slider.setValue(int(round(subject_lock * 100)))
        self.subject_plane_lock_slider.setValue(int(round(subject_plane_lock * 100)))
        self.subject_plane_width_slider.setValue(int(round(subject_width * 100)))
        self.controller.set_state("subject_screen_plane", subject_screen_plane)

        self.sharpness_slider.setValue(int(round(sharpness * 100)))
        self.dof_slider.setValue(int(round(dof * 10)))

        # Safety toggles.
        # Smart safety toggles.
        # Do NOT hard-toggle subject tracking at 45 percent.
        # That causes the preview/render to jump from no subject lock to aggressive subject lock.
        self.subject_tracking_check.setChecked(True)

        # Dynamic convergence can still be comfort-based, but keep the threshold gentle.
        self.dynamic_convergence_check.setChecked(comfort >= 0.45)

        # Floating window should only turn on for strong pop-out scenes.
        # Do not enable it just because comfort is high.
        self.floating_window_check.setChecked(pop >= 0.70 and strength >= 0.70 and comfort < 0.80)

        # Basic 3D Assistant should always keep edge protection on.
        self.edge_masking_check.setChecked(True)
        self.feathering_check.setChecked(True)

        if hasattr(self, "edge_repair_quality_combo"):
            if pop >= 0.70 or strength >= 0.75:
                self.edge_repair_quality_combo.setCurrentText("High")
            else:
                self.edge_repair_quality_combo.setCurrentText("Balanced")

        # Basic modes should always keep edge protection on.
        self.edge_masking_check.setChecked(True)
        self.feathering_check.setChecked(True)

        if hasattr(self, "edge_repair_quality_combo"):
            if pop >= 0.70 or strength >= 0.75:
                self.edge_repair_quality_combo.setCurrentText("High")
            else:
                self.edge_repair_quality_combo.setCurrentText("Balanced")

        self._preview_debounce.start()

    def _bind_events(self):
        self.input_row.browse_clicked.connect(self._browse_input_video)
        self.depth_row.browse_clicked.connect(self._browse_depth_map)
        self.output_row.browse_clicked.connect(self._browse_output_path)
        
        self.render_mode_combo.currentIndexChanged.connect(self._on_render_mode_changed)
        
        self.input_row.text_edited.connect(
            lambda text: self.controller.set_state("input_video_path", text)
        )
        self.depth_row.text_edited.connect(
            lambda text: self.controller.set_state("depth_map_path", text)
        )
        self.output_row.text_edited.connect(
            lambda text: self.controller.set_state("output_path", text)
        )        

        self.output_format_combo.currentTextChanged.connect(
            lambda value: self.controller.set_state("output_format", value)
        )

        self.preview_mode_combo.currentTextChanged.connect(self._on_preview_mode_changed)

        self.fg_shift_slider.valueChanged.connect(self._on_fg_shift_changed)
        self.mg_shift_slider.valueChanged.connect(self._on_mg_shift_changed)
        self.bg_shift_slider.valueChanged.connect(self._on_bg_shift_changed)

        self.convergence_slider.valueChanged.connect(self._on_convergence_changed)
        self.sharpness_slider.valueChanged.connect(self._on_sharpness_changed)
        self.parallax_balance_slider.valueChanged.connect(self._on_parallax_balance_changed)
        self.zero_parallax_slider.valueChanged.connect(self._on_zero_parallax_changed)
        self.max_pixel_shift_slider.valueChanged.connect(self._on_max_pixel_shift_changed)
        self.dof_slider.valueChanged.connect(self._on_dof_changed)
        
        self.depth_pop_gamma_slider.valueChanged.connect(self._on_depth_pop_gamma_changed)
        self.fg_pop_slider.valueChanged.connect(self._on_fg_pop_changed)
        self.bg_push_slider.valueChanged.connect(self._on_bg_push_changed)
        self.subject_lock_slider.valueChanged.connect(self._on_subject_lock_changed)
        self.subject_plane_lock_slider.valueChanged.connect(self._on_subject_plane_lock_changed)
        self.subject_plane_width_slider.valueChanged.connect(self._on_subject_plane_width_changed)
        self.foreground_curvature_slider.valueChanged.connect(self._on_foreground_curvature_changed)

        self.apply_pop_entries_btn.clicked.connect(self._apply_pop_entries)
        self.pop_mid_edit.returnPressed.connect(self._apply_pop_entries)
        self.stretch_lo_edit.returnPressed.connect(self._apply_pop_entries)
        self.stretch_hi_edit.returnPressed.connect(self._apply_pop_entries)
        
        self.stereo_out_combo.currentTextChanged.connect(
            lambda value: self.controller.set_state("stereo_mode", value)
        )
        
        self.use_ffmpeg_check.toggled.connect(self._on_use_ffmpeg_toggled)
        
        self.keep_original_audio_check.toggled.connect(
            lambda checked: (
                self.controller.set_state("keep_original_audio", checked),
                self._set_encoding_preset_custom(),
            )
        )

        self.preserve_hdr10_check.toggled.connect(
            lambda checked: (
                self.controller.set_state("preserve_hdr10", checked),
                self._set_encoding_preset_custom(),
            )
        )
        self.ffmpeg_codec_combo.currentTextChanged.connect(
            lambda value: (
                self.controller.set_state("selected_ffmpeg_codec", value),
                self._update_active_encoder_label(),
                self._set_encoding_preset_custom(),
            )
        )

        self.basic_codec_combo.currentTextChanged.connect(
            lambda value: (
                self.controller.set_state("selected_codec", value),
                self._update_active_encoder_label(),
                self._set_encoding_preset_custom(),
            )
        )

        self.crf_spin.valueChanged.connect(
            lambda value: (
                self.controller.set_state("crf_value", value),
                self._set_encoding_preset_custom(),
            )
        )

        self.nvenc_cq_spin.valueChanged.connect(
            lambda value: (
                self.controller.set_state("nvenc_cq_value", value),
                self._set_encoding_preset_custom(),
            )
        )

        self.preserve_aspect_check.toggled.connect(
            lambda checked: self.controller.set_state("preserve_original_aspect", checked)
        )
        self.auto_crop_check.toggled.connect(
            lambda checked: self.controller.set_state("auto_crop_black_bars", checked)
        )
        self.subject_tracking_check.toggled.connect(
            lambda checked: self.controller.set_state("use_subject_tracking", checked)
        )
        self.skip_blank_check.toggled.connect(
            lambda checked: self.controller.set_state("skip_blank_frames", checked)
        )
        self.edge_masking_check.toggled.connect(
            lambda checked: self.controller.set_state("enable_edge_masking", checked)
        )
        self.feathering_check.toggled.connect(
            lambda checked: self.controller.set_state("enable_feathering", checked)
        )
        self.edge_repair_quality_combo.currentTextChanged.connect(
            lambda value: self.controller.set_state("edge_repair_quality", value)
        )
        self.dynamic_convergence_check.toggled.connect(
            lambda checked: self.controller.set_state("enable_dynamic_convergence", checked)
        )       
        self.floating_window_check.toggled.connect(
            lambda checked: self.controller.set_state("use_floating_window", checked)
        )
        self.disable_shift_ema_check.toggled.connect(
            lambda checked: self.controller.set_state("disable_shift_ema", checked)
        )
        self.clip_start_edit.editingFinished.connect(
            lambda: self.controller.set_state("clip_start", self.clip_start_edit.text().strip())
        )
        self.clip_end_edit.editingFinished.connect(
            lambda: self.controller.set_state("clip_end", self.clip_end_edit.text().strip())
        )
        self.clear_clip_btn.clicked.connect(self._clear_clip_range)

        self.encoding_preset_combo.currentTextChanged.connect(self._on_encoding_preset_changed)
        self.encoding_advanced_btn.toggled.connect(self._set_encoding_advanced_visible)

        self.vr180_render_preset_combo.currentTextChanged.connect(self._on_vr180_render_preset_changed)
        self.vr180_advanced_btn.toggled.connect(self._set_vr180_advanced_visible)

        self.ipd_enabled_check.toggled.connect(self._on_ipd_enabled_changed)
        self.ipd_scale.valueChanged.connect(self._on_ipd_scale_changed)
        self.show_guides_check.toggled.connect(self._on_show_guides_changed)

        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.frame_slider.sliderReleased.connect(self._on_frame_slider_released)
                
        self.saturation_slider.valueChanged.connect(self._on_saturation_changed)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        self.color_reset_btn.clicked.connect(self._reset_color_grading)

        self.preview_btn.clicked.connect(self._load_preview_sources)
        self.refresh_preview_btn.clicked.connect(self.controller.update_preview)
        self.prev_frame_btn.clicked.connect(lambda: self._step_preview_frame(-1))
        self.next_frame_btn.clicked.connect(lambda: self._step_preview_frame(1))
        self.prev_frame_shortcut = QShortcut(QKeySequence("Left"), self)
        self.next_frame_shortcut = QShortcut(QKeySequence("Right"), self)

        self.prev_frame_shortcut.activated.connect(lambda: self._step_preview_frame(-1))
        self.next_frame_shortcut.activated.connect(lambda: self._step_preview_frame(1))
        self.save_preview_btn.clicked.connect(self._save_preview_image)
        self.fullscreen_preview_btn.clicked.connect(self._open_fullscreen_preview)
        self.render_btn.clicked.connect(self._start_render_clicked)

        self.simple_3d_style_combo.currentTextChanged.connect(self._on_simple_3d_style_changed)
        self.simple_strength_slider.valueChanged.connect(self._on_simple_3d_slider_changed)
        self.simple_pop_slider.valueChanged.connect(self._on_simple_3d_slider_changed)
        self.simple_comfort_slider.valueChanged.connect(self._on_simple_3d_slider_changed)
        self.simple_stability_slider.valueChanged.connect(self._on_simple_3d_slider_changed)
        self.simple_screen_depth_slider.valueChanged.connect(self._on_simple_3d_slider_changed)
        self.simple_subject_plane_slider.valueChanged.connect(self._on_simple_3d_slider_changed)
        self.advanced_3d_btn.toggled.connect(self._set_advanced_3d_visible)

        self.aspect_ratio_combo.currentTextChanged.connect(
            lambda value: self.controller.set_state("selected_aspect_ratio", value)
        )

        self.vr180_hfov_slider.valueChanged.connect(self._on_vr180_hfov_changed)

        self.vr180_equi_w_spin.valueChanged.connect(
            lambda value: self.controller.set_state("vr180_equi_w", value)
        )
        self.vr180_equi_h_spin.valueChanged.connect(
            lambda value: self.controller.set_state("vr180_equi_h", value)
        )
        self.vr180_flat_w_spin.valueChanged.connect(
            lambda value: self.controller.set_state("vr180_flat_w", value)
        )
        self.vr180_flat_h_spin.valueChanged.connect(
            lambda value: self.controller.set_state("vr180_flat_h", value)
        )
        
        self.vr180_equi_w_spin.valueChanged.connect(lambda _: self._sync_vr180_equi_preset_combo())
        self.vr180_equi_h_spin.valueChanged.connect(lambda _: self._sync_vr180_equi_preset_combo())
        self.vr180_flat_w_spin.valueChanged.connect(lambda _: self._sync_vr180_flat_preset_combo())
        self.vr180_flat_h_spin.valueChanged.connect(lambda _: self._sync_vr180_flat_preset_combo())
                        

        self.vr180_equi_preset_combo.currentTextChanged.connect(self._on_vr180_equi_preset_changed)
        self.vr180_flat_preset_combo.currentTextChanged.connect(self._on_vr180_flat_preset_changed)

        self.keyframes_enabled_check.toggled.connect(
            lambda checked: self.controller.set_state("keyframes_enabled", checked)
        )

        self.create_keyframes_btn.clicked.connect(self._create_keyframes_file)
        self.load_keyframes_btn.clicked.connect(self._load_keyframes_file_dialog)
        self.save_keyframes_btn.clicked.connect(self._save_keyframes_file)
        self.keyframe_file_btn.clicked.connect(lambda: self._show_dialog(self.keyframe_file_dialog))
        self.preview_settings_btn.clicked.connect(lambda: self._show_dialog(self.preview_settings_dialog))
        self.add_keyframe_btn.clicked.connect(self._add_or_update_keyframe_at_current_frame)
        self.delete_keyframe_btn.clicked.connect(self._delete_selected_keyframe)
        self.apply_keyframe_btn.clicked.connect(self._apply_selected_keyframe_to_sliders)
        self.keyframe_combo.currentIndexChanged.connect(self._on_selected_keyframe_changed)

        self.controller.settings_loaded.connect(self._load_initial_state)
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.preview_updated.connect(self._on_preview_updated)
        self.controller.preview_failed.connect(self._on_preview_failed)
        
        self.suspend_btn.clicked.connect(self.controller.suspend_render)
        self.resume_btn.clicked.connect(self.controller.resume_render)
        self.cancel_btn.clicked.connect(self.controller.cancel_render)

        self.controller.render_started.connect(self._on_render_started)
        self.controller.render_finished.connect(self._on_render_finished)
        self.controller.render_failed.connect(self._on_render_failed_state)
        self.controller.render_cancelled.connect(self._on_render_cancelled_state)
        self.controller.render_suspended.connect(self._on_render_suspended_state)
        self.controller.render_resumed.connect(self._on_render_resumed_state)
        
        self.load_preset_btn.clicked.connect(self._load_preset_dialog)
        self.save_preset_btn.clicked.connect(self._save_preset_dialog)
        self.preset_combo.currentTextChanged.connect(self._on_preset_combo_changed)

        self.encoding_settings_btn.clicked.connect(
            lambda: self._show_dialog(self.encoding_dialog)
        )
        self.processing_settings_btn.clicked.connect(
            lambda: self._show_dialog(self.processing_dialog)
        )

    def _load_initial_state(self):
        self.input_row.set_text(self.controller.state.input_video_path)
        self.depth_row.set_text(self.controller.state.depth_map_path)
        self.output_row.set_text(self.controller.state.output_path)
        
        self._set_combo_by_data(
            self.render_mode_combo,
            getattr(self.controller.state, "render_mode", "video"),
        )
        self._apply_render_mode_ui()

        self.output_format_combo.setCurrentText(self.controller.state.output_format)
        self._set_combo_by_key(self.preview_mode_combo, self.controller.state.preview_mode)

        self.fg_shift_slider.setValue(int(self.controller.state.fg_shift * 10))
        self.mg_shift_slider.setValue(int(self.controller.state.mg_shift * 10))
        self.bg_shift_slider.setValue(int(self.controller.state.bg_shift * 10))

        self.fg_shift_value.setText(f"{self.controller.state.fg_shift:.2f}")
        self.mg_shift_value.setText(f"{self.controller.state.mg_shift:.2f}")
        self.bg_shift_value.setText(f"{self.controller.state.bg_shift:.2f}")

        self.convergence_slider.setValue(int(self.controller.state.convergence_strength * 1000))
        self.sharpness_slider.setValue(int(self.controller.state.sharpness_factor * 100))
        self.parallax_balance_slider.setValue(int(self.controller.state.parallax_balance * 100))
        self.zero_parallax_slider.setValue(int(self.controller.state.zero_parallax_strength * 1000))
        self.max_pixel_shift_slider.setValue(int(self.controller.state.max_pixel_shift * 1000))
        self.dof_slider.setValue(int(self.controller.state.dof_strength * 10))
        
        self.depth_pop_gamma_slider.setValue(int(self.controller.state.depth_pop_gamma * 100))
        self.depth_pop_gamma_value.setText(f"{self.controller.state.depth_pop_gamma:.2f}")

        self.fg_pop_slider.setValue(int(self.controller.state.fg_pop_multiplier * 100))
        self.fg_pop_value.setText(f"{self.controller.state.fg_pop_multiplier:.2f}")

        self.bg_push_slider.setValue(int(self.controller.state.bg_push_multiplier * 100))
        self.bg_push_value.setText(f"{self.controller.state.bg_push_multiplier:.2f}")

        self.subject_lock_slider.setValue(
            int(getattr(self.controller.state, "subject_lock_strength", 1.00) * 100)
        )
        self.subject_lock_value.setText(
            f"{getattr(self.controller.state, 'subject_lock_strength', 1.00):.2f}"
        )

        self.subject_plane_lock_slider.setValue(
            int(getattr(self.controller.state, "subject_plane_lock_strength", 0.00) * 100)
        )
        self.subject_plane_lock_value.setText(
            f"{getattr(self.controller.state, 'subject_plane_lock_strength', 0.00):.2f}"
        )

        self.subject_plane_width_slider.setValue(
            int(getattr(self.controller.state, "subject_plane_lock_width", 0.08) * 100)
        )
        self.subject_plane_width_value.setText(
            f"{getattr(self.controller.state, 'subject_plane_lock_width', 0.08):.2f}"
        )

        if hasattr(self, "simple_subject_plane_slider"):
            subject_zero_lock_state = float(getattr(self.controller.state, "subject_screen_plane", 0.0))
            subject_zero_lock_ui = int(round(max(0.0, min(1.0, subject_zero_lock_state)) * 100.0))
            self.simple_subject_plane_slider.setValue(max(0, min(100, subject_zero_lock_ui)))
            
        self.foreground_curvature_slider.setValue(
            int(getattr(self.controller.state, "foreground_curvature_strength", 0.06) * 100)
        )
        self.foreground_curvature_value.setText(
            f"{getattr(self.controller.state, 'foreground_curvature_strength', 0.06):.2f}"
        )

        self.pop_mid_edit.setText(f"{self.controller.state.depth_pop_mid:.2f}")
        self.stretch_lo_edit.setText(f"{self.controller.state.depth_stretch_lo:.2f}")
        self.stretch_hi_edit.setText(f"{self.controller.state.depth_stretch_hi:.2f}")

        self.convergence_value.setText(f"{self.controller.state.convergence_strength:.3f}")
        self.sharpness_value.setText(f"{self.controller.state.sharpness_factor:.1f}")
        self.parallax_balance_value.setText(f"{self.controller.state.parallax_balance:.2f}")
        self.zero_parallax_value.setText(f"{self.controller.state.zero_parallax_strength:.3f}")
        self.max_pixel_shift_value.setText(f"{self.controller.state.max_pixel_shift:.3f}")
        self.dof_value.setText(f"{self.controller.state.dof_strength:.1f}")
        
        self.aspect_ratio_combo.setCurrentText(
            getattr(self.controller.state, "selected_aspect_ratio", "Default (16:9)")
        )

        self.vr180_hfov_slider.setValue(int(getattr(self.controller.state, "vr180_hfov_deg", 110.0)))
        self.vr180_hfov_value.setText(str(int(getattr(self.controller.state, "vr180_hfov_deg", 110.0))))

        self.vr180_equi_w_spin.setValue(getattr(self.controller.state, "vr180_equi_w", 3840))
        self.vr180_equi_h_spin.setValue(getattr(self.controller.state, "vr180_equi_h", 1920))
        self.vr180_flat_w_spin.setValue(getattr(self.controller.state, "vr180_flat_w", 1920))
        self.vr180_flat_h_spin.setValue(getattr(self.controller.state, "vr180_flat_h", 1080))
        
        self._set_combo_by_key(self.stereo_out_combo, getattr(self.controller.state, "stereo_mode", "sbs"))
        self.use_ffmpeg_check.setChecked(getattr(self.controller.state, "use_ffmpeg", False))
        self.keep_original_audio_check.setChecked(getattr(self.controller.state, "keep_original_audio", True))
        self.preserve_hdr10_check.setChecked(getattr(self.controller.state, "preserve_hdr10", False))

        self._set_combo_by_key(
            self.ffmpeg_codec_combo,
            getattr(self.controller.state, "selected_ffmpeg_codec", "H.264 / AVC (libx264 - CPU)")
        )
        self._set_combo_by_key(
            self.basic_codec_combo,
            getattr(self.controller.state, "selected_codec", "XVID")
        )
        self.crf_spin.setValue(getattr(self.controller.state, "crf_value", 23))
        self.nvenc_cq_spin.setValue(getattr(self.controller.state, "nvenc_cq_value", 23))

        self._sync_encoding_preset_from_state()
        self._set_encoding_advanced_visible(False)
        self._update_active_encoder_label()
        self._update_vr180_preset_description()
        self._set_vr180_advanced_visible(False)

        self.preserve_aspect_check.setChecked(getattr(self.controller.state, "preserve_original_aspect", False))
        self.auto_crop_check.setChecked(getattr(self.controller.state, "auto_crop_black_bars", False))
        self.subject_tracking_check.setChecked(getattr(self.controller.state, "use_subject_tracking", False))
        self.skip_blank_check.setChecked(getattr(self.controller.state, "skip_blank_frames", False))
        self.edge_masking_check.setChecked(getattr(self.controller.state, "enable_edge_masking", True))
        self.feathering_check.setChecked(getattr(self.controller.state, "enable_feathering", True))

        self._set_combo_by_key(
            self.edge_repair_quality_combo,
            getattr(self.controller.state, "edge_repair_quality", "Balanced")
        )

        self.dynamic_convergence_check.setChecked(getattr(self.controller.state, "enable_dynamic_convergence", True))
        self.floating_window_check.setChecked(getattr(self.controller.state, "use_floating_window", False))
        self.disable_shift_ema_check.setChecked(getattr(self.controller.state, "disable_shift_ema", False))

        self.clip_start_edit.setText(getattr(self.controller.state, "clip_start", ""))
        self.clip_end_edit.setText(getattr(self.controller.state, "clip_end", ""))

        self.ipd_enabled_check.setChecked(self.controller.state.ipd_enabled)
        self.ipd_scale.setValue(self.controller.state.ipd_scale)
        self.show_guides_check.setChecked(self.controller.state.show_convergence_guides)
        
        self.saturation_slider.setValue(int(self.controller.state.saturation * 100))
        self.contrast_slider.setValue(int(self.controller.state.contrast * 100))
        self.brightness_slider.setValue(int(self.controller.state.brightness * 100))

        self.saturation_value.setText(f"{self.controller.state.saturation:.2f}")
        self.contrast_value.setText(f"{self.controller.state.contrast:.2f}")
        self.brightness_value.setText(f"{self.controller.state.brightness:.2f}")

        if hasattr(self, "simple_3d_style_combo"):
            self.simple_3d_style_combo.blockSignals(True)
            self.simple_3d_style_combo.setCurrentText("Custom")
            self.simple_3d_style_combo.blockSignals(False)

        if hasattr(self, "simple_strength_slider"):
            self.simple_strength_slider.blockSignals(True)
            self.simple_pop_slider.blockSignals(True)
            self.simple_comfort_slider.blockSignals(True)
            self.simple_stability_slider.blockSignals(True)
            self.simple_screen_depth_slider.blockSignals(True)
            self.simple_subject_plane_slider.blockSignals(True)

            self.simple_strength_slider.setValue(50)
            self.simple_pop_slider.setValue(50)
            self.simple_comfort_slider.setValue(70)
            self.simple_stability_slider.setValue(60)
            self.simple_screen_depth_slider.setValue(50)
            self.simple_subject_plane_slider.setValue(0)

            self.simple_strength_slider.blockSignals(False)
            self.simple_pop_slider.blockSignals(False)
            self.simple_comfort_slider.blockSignals(False)
            self.simple_stability_slider.blockSignals(False)
            self.simple_screen_depth_slider.blockSignals(False)
            self.simple_subject_plane_slider.blockSignals(False)

            self._update_simple_3d_labels()
            self._set_advanced_3d_visible(False)

        self.frame_slider.setValue(self.controller.state.preview_frame_index)
        self._sync_vr180_equi_preset_combo()
        self._sync_vr180_flat_preset_combo()
        self.keyframes_enabled_check.setChecked(
            getattr(self.controller.state, "keyframes_enabled", False)
        )

        self.keyframes_path_edit.setText(
            getattr(self.controller.state, "keyframes_path", "")
        )

        self._load_keyframe_service_from_state(silent=True)
        self._refresh_keyframe_combo()
        
        self._refresh_preview_meta()

    def _create_card_dialog(self, title, card, min_width=500):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(False)
        dialog.setMinimumWidth(min_width)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)
        layout.addWidget(card)

        return dialog

    def _show_dialog(self, dialog):
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _build_keyframe_file_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(self._t("Keyframe File"))
        dialog.setModal(False)
        dialog.resize(700, 220)
        dialog.setMinimumWidth(620)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        file_group = QGroupBox()
        self._register_title(file_group, "Keyframe File")
        file_layout = QGridLayout(file_group)
        file_layout.setHorizontalSpacing(10)
        file_layout.setVerticalSpacing(8)

        file_layout.addWidget(self._label("Path"), 0, 0)
        file_layout.addWidget(self.keyframes_path_edit, 0, 1, 1, 2)
        file_layout.addWidget(self.create_keyframes_btn, 1, 0)
        file_layout.addWidget(self.load_keyframes_btn, 1, 1)
        file_layout.addWidget(self.save_keyframes_btn, 1, 2)

        layout.addWidget(file_group)
        layout.addStretch()

        return dialog

    def _build_preview_settings_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(self._t("Preview Settings"))
        dialog.setModal(False)
        dialog.resize(520, 260)
        dialog.setMinimumWidth(480)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        preview_group = QGroupBox()
        self._register_title(preview_group, "Preview Settings")
        preview_layout = QGridLayout(preview_group)
        preview_layout.setHorizontalSpacing(10)
        preview_layout.setVerticalSpacing(8)

        preview_layout.addWidget(self._label("Preview Mode"), 0, 0)
        preview_layout.addWidget(self.preview_mode_combo, 0, 1)
        preview_layout.addWidget(self.ipd_enabled_check, 1, 0, 1, 2)
        preview_layout.addWidget(self._label("Stereo Scaling (IPD)"), 2, 0)
        preview_layout.addWidget(self.ipd_scale, 2, 1)
        preview_layout.addWidget(self.show_guides_check, 3, 0, 1, 2)

        layout.addWidget(preview_group)
        layout.addStretch()

        return dialog

    def _set_widget_text_safely(self, widget, text: str):
        if widget is not None and hasattr(widget, "setText"):
            widget.setText(self._qt_text(self._t(text)))


    def _set_encoding_advanced_visible(self, visible: bool):
        visible = bool(visible)

        if hasattr(self, "encoding_advanced_group"):
            self.encoding_advanced_group.setVisible(visible)

        if hasattr(self, "encoding_advanced_btn"):
            self.encoding_advanced_btn.setChecked(visible)
            self.encoding_advanced_btn.setText(
                self._qt_text(
                    self._t("Hide Advanced Encoding Settings")
                    if visible
                    else self._t("Advanced Encoding Settings")
                )
            )


    def _set_vr180_advanced_visible(self, visible: bool):
        visible = bool(visible)

        if hasattr(self, "vr180_advanced_group"):
            self.vr180_advanced_group.setVisible(visible)

        if hasattr(self, "vr180_advanced_btn"):
            self.vr180_advanced_btn.setChecked(visible)
            self.vr180_advanced_btn.setText(
                self._qt_text(
                    self._t("Hide Advanced VR180 Settings")
                    if visible
                    else self._t("Advanced VR180 Settings")
                )
            )


    def _update_encoding_preset_description(self):
        preset_key = self._combo_key(
            self.encoding_preset_combo,
            "Custom",
        ) if hasattr(self, "encoding_preset_combo") else "Custom"

        preset = ENCODING_PRESETS.get(preset_key, ENCODING_PRESETS["Custom"])

        desc = preset.get("description", "")
        warning = preset.get("warning", "")

        container = preset.get("container", "manual")
        extension = preset.get("extension", "")
        pixel_format = preset.get("pixel_format", "manual")
        encoder_preset = preset.get("encoder_preset", "manual")
        audio_mode = preset.get("audio_mode", "manual")

        details = (
            f"{self._t('Container')}: {str(container).upper()}  |  "
            f"{self._t('Extension')}: {extension or self._t('manual')}  |  "
            f"{self._t('Pixel Format')}: {pixel_format}  |  "
            f"{self._t('Speed Preset')}: {encoder_preset}  |  "
            f"{self._t('Audio')}: {audio_mode}"
        )

        if hasattr(self, "encoding_preset_desc"):
            self.encoding_preset_desc.setText(self._qt_text(self._t(desc)))

        if hasattr(self, "encoding_preset_details"):
            if preset_key != "Custom":
                self.encoding_preset_details.setText(self._qt_text(details))
            else:
                self.encoding_preset_details.setText(
                    self._qt_text(self._t("Manual encoder settings."))
                )

        if hasattr(self, "encoding_preset_warning"):
            self.encoding_preset_warning.setText(self._qt_text(self._t(warning)))
            self.encoding_preset_warning.setVisible(bool(warning))
            
    def _update_vr180_preset_description(self):
        preset_key = self._combo_key(
            self.vr180_render_preset_combo,
            "Custom",
        ) if hasattr(self, "vr180_render_preset_combo") else "Custom"

        preset = VR180_RENDER_PRESETS.get(preset_key, VR180_RENDER_PRESETS["Custom"])
        desc = preset.get("description", "")

        if hasattr(self, "vr180_preset_desc"):
            self.vr180_preset_desc.setText(self._qt_text(self._t(desc)))
            
    def _update_active_encoder_label(self):
        if not hasattr(self, "active_encoder_label"):
            return

        use_ffmpeg = bool(self.use_ffmpeg_check.isChecked())

        preset_key = self._combo_key(
            self.encoding_preset_combo,
            "Custom",
        ) if hasattr(self, "encoding_preset_combo") else "Custom"

        preset = ENCODING_PRESETS.get(preset_key, {})

        container = preset.get("container", "manual")
        pixel_format = preset.get("pixel_format", "manual")
        encoder_preset = preset.get("encoder_preset", "manual")

        if use_ffmpeg:
            codec = self._combo_key(self.ffmpeg_codec_combo, self.ffmpeg_codec_combo.currentText())
            text = (
                f"{self._t('FFmpeg renderer enabled. Active codec')}: {codec}\n"
                f"{self._t('Container')}: {str(container).upper()}  |  "
                f"{self._t('Pixel Format')}: {pixel_format}  |  "
                f"{self._t('Speed Preset')}: {encoder_preset}"
            )
        else:
            codec = self._combo_key(self.basic_codec_combo, self.basic_codec_combo.currentText())
            text = (
                f"{self._t('Basic writer enabled. Active codec')}: {codec}. "
                f"{self._t('FFmpeg codec dropdown is not active.')}"
            )

        self.active_encoder_label.setText(self._qt_text(text))

        ffmpeg_controls_enabled = use_ffmpeg
        basic_controls_enabled = not use_ffmpeg

        if hasattr(self, "ffmpeg_codec_combo"):
            self.ffmpeg_codec_combo.setEnabled(ffmpeg_controls_enabled)

        if hasattr(self, "crf_spin"):
            self.crf_spin.setEnabled(ffmpeg_controls_enabled)

        if hasattr(self, "nvenc_cq_spin"):
            self.nvenc_cq_spin.setEnabled(ffmpeg_controls_enabled)

        if hasattr(self, "basic_codec_combo"):
            self.basic_codec_combo.setEnabled(basic_controls_enabled)


    def _set_encoding_preset_custom(self):
        if getattr(self, "_applying_encoding_preset", False):
            return

        if not hasattr(self, "encoding_preset_combo"):
            return

        if self._combo_key(self.encoding_preset_combo, "Custom") == "Custom":
            return

        self.encoding_preset_combo.blockSignals(True)
        self._set_combo_by_key(self.encoding_preset_combo, "Custom")
        self.encoding_preset_combo.blockSignals(False)

        self._update_encoding_preset_description()
        self._update_active_encoder_label()

    def _on_encoding_preset_changed(self, *_args):
        preset_name = self._combo_key(self.encoding_preset_combo, "Custom")
        preset = ENCODING_PRESETS.get(preset_name)

        if not preset:
            return

        self._update_encoding_preset_description()

        if preset_name == "Custom":
            self._set_encoding_advanced_visible(True)
            self._update_active_encoder_label()
            return

        self._applying_encoding_preset = True

        try:
            widgets_to_block = [
                self.use_ffmpeg_check,
                self.keep_original_audio_check,
                self.preserve_hdr10_check,
                self.ffmpeg_codec_combo,
                self.basic_codec_combo,
                self.crf_spin,
                self.nvenc_cq_spin,
            ]

            for widget in widgets_to_block:
                try:
                    widget.blockSignals(True)
                except Exception:
                    pass

            self.use_ffmpeg_check.setChecked(bool(preset.get("use_ffmpeg", True)))
            self.keep_original_audio_check.setChecked(bool(preset.get("keep_audio", True)))
            self.preserve_hdr10_check.setChecked(bool(preset.get("preserve_hdr10", False)))

            ffmpeg_codec = preset.get("ffmpeg_codec")
            if ffmpeg_codec:
                self._set_combo_by_key(self.ffmpeg_codec_combo, ffmpeg_codec)

            basic_codec = preset.get("basic_codec")
            if basic_codec:
                self._set_combo_by_key(self.basic_codec_combo, basic_codec)

            self.crf_spin.setValue(int(preset.get("crf", self.crf_spin.value())))
            self.nvenc_cq_spin.setValue(int(preset.get("nvenc_cq", self.nvenc_cq_spin.value())))

            for widget in widgets_to_block:
                try:
                    widget.blockSignals(False)
                except Exception:
                    pass

            self.controller.set_state("use_ffmpeg", bool(self.use_ffmpeg_check.isChecked()))
            self.controller.set_state("keep_original_audio", bool(self.keep_original_audio_check.isChecked()))
            self.controller.set_state("preserve_hdr10", bool(self.preserve_hdr10_check.isChecked()))
            self.controller.set_state("selected_ffmpeg_codec", self._combo_key(self.ffmpeg_codec_combo, self.ffmpeg_codec_combo.currentText()))
            self.controller.set_state("selected_codec", self._combo_key(self.basic_codec_combo, self.basic_codec_combo.currentText()))
            self.controller.set_state("crf_value", int(self.crf_spin.value()))
            self.controller.set_state("nvenc_cq_value", int(self.nvenc_cq_spin.value()))

            self.controller.set_state("encoding_container", preset.get("container", ""))
            self.controller.set_state("encoding_extension", preset.get("extension", ""))
            self.controller.set_state("encoding_pixel_format", preset.get("pixel_format", ""))
            self.controller.set_state("encoding_encoder_preset", preset.get("encoder_preset", ""))
            self.controller.set_state("encoding_audio_mode", preset.get("audio_mode", ""))
            self.controller.set_state("encoding_warning", preset.get("warning", ""))

            self._apply_output_extension_from_encoding_preset(preset)

        finally:
            self._applying_encoding_preset = False

            # Force the dropdown to stay on the preset the user picked.
            self.encoding_preset_combo.blockSignals(True)
            self._set_combo_by_key(self.encoding_preset_combo, preset_name)
            self.encoding_preset_combo.blockSignals(False)

            self._update_encoding_preset_description()
            self._update_active_encoder_label()
            self._set_encoding_advanced_visible(False)


    def _apply_output_extension_from_encoding_preset(self, preset: dict):
        """
        Updates the output path extension to match the selected preset.
        This only changes the filename extension, not the folder or base name.
        """
        extension = preset.get("extension")
        if not extension:
            return

        if not extension.startswith("."):
            extension = "." + extension

        output_path = ""
        if hasattr(self, "output_row"):
            output_path = self._get_output_row_text()

        if not output_path:
            output_path = getattr(self.controller.state, "output_path", "")

        if not output_path:
            return

        try:
            path = Path(output_path)
            new_path = str(path.with_suffix(extension))

            if new_path == output_path:
                return

            self._set_output_row_text(new_path)
            self.controller.set_state("output_path", new_path)

        except Exception as e:
            print(f"Could not update output extension from encoding preset: {e}")


    def _on_vr180_render_preset_changed(self, *_args):
        preset_name = self._combo_key(self.vr180_render_preset_combo, "Custom")
        preset = VR180_RENDER_PRESETS.get(preset_name)

        if not preset:
            return

        self._update_vr180_preset_description()

        if preset_name == "Custom":
            self._set_vr180_advanced_visible(True)
            return

        output_format = preset.get("output_format")
        if output_format:
            self._set_combo_by_key(self.output_format_combo, output_format)
            self.controller.set_state("output_format", output_format)

        hfov = int(preset.get("hfov", self.vr180_hfov_slider.value()))
        self.vr180_hfov_slider.setValue(hfov)
        self.controller.set_state("vr180_hfov_deg", float(hfov))

        equi = preset.get("equi")
        if equi:
            w, h = equi
            self.vr180_equi_w_spin.setValue(int(w))
            self.vr180_equi_h_spin.setValue(int(h))
            self.controller.set_state("vr180_equi_w", int(w))
            self.controller.set_state("vr180_equi_h", int(h))
            self._sync_vr180_equi_preset_combo()

        flat = preset.get("flat")
        if flat:
            w, h = flat
            self.vr180_flat_w_spin.setValue(int(w))
            self.vr180_flat_h_spin.setValue(int(h))
            self.controller.set_state("vr180_flat_w", int(w))
            self.controller.set_state("vr180_flat_h", int(h))
            self._sync_vr180_flat_preset_combo()

        self._set_vr180_advanced_visible(False)


    def _on_use_ffmpeg_toggled(self, checked: bool):
        if getattr(self, "_applying_encoding_preset", False):
            return

        self.controller.set_state("use_ffmpeg", bool(checked))
        self._update_active_encoder_label()


    def _sync_encoding_preset_from_state(self):
        """
        Best effort. If current state matches one of the known presets, select it.
        Otherwise select Custom.
        """
        
        def _sync_encoding_preset_from_state(self):
            """
            Best effort. If current state matches one of the known presets, select it.
            Otherwise select Custom.
            """
            if getattr(self, "_applying_encoding_preset", False):
                return
        if not hasattr(self, "encoding_preset_combo"):
            return

        current = {
            "use_ffmpeg": bool(self.use_ffmpeg_check.isChecked()),
            "ffmpeg_codec": self.ffmpeg_codec_combo.currentText(),
            "basic_codec": self.basic_codec_combo.currentText(),
            "crf": int(self.crf_spin.value()),
            "nvenc_cq": int(self.nvenc_cq_spin.value()),
            "keep_audio": bool(self.keep_original_audio_check.isChecked()),
            "preserve_hdr10": bool(self.preserve_hdr10_check.isChecked()),
            "container": getattr(self.controller.state, "encoding_container", ""),
            "pixel_format": getattr(self.controller.state, "encoding_pixel_format", ""),
            "encoder_preset": getattr(self.controller.state, "encoding_encoder_preset", ""),
            "audio_mode": getattr(self.controller.state, "encoding_audio_mode", ""),
        }

        matched = "Custom"

        for name, preset in ENCODING_PRESETS.items():
            if name == "Custom":
                continue

            test = {
                "use_ffmpeg": bool(preset.get("use_ffmpeg", True)),
                "ffmpeg_codec": preset.get("ffmpeg_codec", ""),
                "basic_codec": preset.get("basic_codec", "mp4v"),
                "crf": int(preset.get("crf", 23)),
                "nvenc_cq": int(preset.get("nvenc_cq", 23)),
                "keep_audio": bool(preset.get("keep_audio", True)),
                "preserve_hdr10": bool(preset.get("preserve_hdr10", False)),
                "container": preset.get("container", ""),
                "pixel_format": preset.get("pixel_format", ""),
                "encoder_preset": preset.get("encoder_preset", ""),
                "audio_mode": preset.get("audio_mode", ""),
            }

            if current == test:
                matched = name
                break

        self.encoding_preset_combo.blockSignals(True)
        self.encoding_preset_combo.setCurrentText(matched)
        self.encoding_preset_combo.blockSignals(False)

        self._update_encoding_preset_description()
        self._update_active_encoder_label()

    def _build_encoding_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(self._t("Output & Encoding"))
        dialog.setModal(False)
        dialog.resize(720, 760)
        dialog.setMinimumSize(660, 620)

        outer = QVBoxLayout(dialog)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setSpacing(14)

        # ------------------------------------------------------------
        # Group 1: Simple Encoding Preset
        # ------------------------------------------------------------
        preset_group = QGroupBox()
        self._register_title(preset_group, "Encoding Preset")
        preset_layout = QGridLayout(preset_group)
        preset_layout.setHorizontalSpacing(14)
        preset_layout.setVerticalSpacing(10)

        preset_layout.addWidget(self._label("Preset"), 0, 0)
        preset_layout.addWidget(self.encoding_preset_combo, 0, 1)

        preset_layout.addWidget(self._label("Summary"), 1, 0)
        preset_layout.addWidget(self.encoding_preset_desc, 1, 1)

        preset_layout.addWidget(self._label("Active Encoder"), 2, 0)
        preset_layout.addWidget(self.active_encoder_label, 2, 1)

        preset_layout.addWidget(self._label("Preset Details"), 3, 0)
        preset_layout.addWidget(self.encoding_preset_details, 3, 1)

        preset_layout.addWidget(self._label("Warning"), 4, 0)
        preset_layout.addWidget(self.encoding_preset_warning, 4, 1)

        preset_layout.addWidget(self.encoding_advanced_btn, 5, 0, 1, 2)

        content_layout.addWidget(preset_group)

        # ------------------------------------------------------------
        # Group 2: Format
        # ------------------------------------------------------------
        format_group = QGroupBox()
        self._register_title(format_group, "Format")
        format_layout = QGridLayout(format_group)
        format_layout.setHorizontalSpacing(14)
        format_layout.setVerticalSpacing(10)

        format_layout.addWidget(self._label("Output Format"), 0, 0)
        format_layout.addWidget(self.output_format_combo, 0, 1)

        format_layout.addWidget(self._label("Stereo Output"), 1, 0)
        format_layout.addWidget(self.stereo_out_combo, 1, 1)

        format_layout.addWidget(self._label("Aspect Ratio"), 2, 0)
        format_layout.addWidget(self.aspect_ratio_combo, 2, 1)

        content_layout.addWidget(format_group)

        # ------------------------------------------------------------
        # Group 3: Advanced Encoding
        # ------------------------------------------------------------
        self.encoding_advanced_group = QGroupBox()
        self._register_title(self.encoding_advanced_group, "Advanced Encoding")
        advanced_layout = QVBoxLayout(self.encoding_advanced_group)
        advanced_layout.setSpacing(12)

        render_group = QGroupBox()
        self._register_title(render_group, "Render Options")
        render_layout = QVBoxLayout(render_group)
        render_layout.setSpacing(8)

        render_layout.addWidget(self.use_ffmpeg_check)
        render_layout.addWidget(self.keep_original_audio_check)
        render_layout.addWidget(self.preserve_hdr10_check)

        advanced_layout.addWidget(render_group)

        codec_group = QGroupBox()
        self._register_title(codec_group, "Codecs & Quality")
        codec_layout = QGridLayout(codec_group)
        codec_layout.setHorizontalSpacing(14)
        codec_layout.setVerticalSpacing(10)

        self.ffmpeg_codec_label = self._label("FFmpeg Codec")
        self.basic_codec_label = self._label("Basic Codec")
        self.crf_label = self._label("CRF")
        self.nvenc_cq_label = self._label("NVENC CQ")

        codec_layout.addWidget(self.ffmpeg_codec_label, 0, 0)
        codec_layout.addWidget(self.ffmpeg_codec_combo, 0, 1)

        codec_layout.addWidget(self.basic_codec_label, 1, 0)
        codec_layout.addWidget(self.basic_codec_combo, 1, 1)

        codec_layout.addWidget(self.crf_label, 2, 0)
        codec_layout.addWidget(self.crf_spin, 2, 1)

        codec_layout.addWidget(self.nvenc_cq_label, 3, 0)
        codec_layout.addWidget(self.nvenc_cq_spin, 3, 1)

        advanced_layout.addWidget(codec_group)

        content_layout.addWidget(self.encoding_advanced_group)

        # ------------------------------------------------------------
        # Group 4: VR180 Presets
        # ------------------------------------------------------------
        vr_preset_group = QGroupBox()
        self._register_title(vr_preset_group, "VR180 Preset")
        vr_preset_layout = QGridLayout(vr_preset_group)
        vr_preset_layout.setHorizontalSpacing(14)
        vr_preset_layout.setVerticalSpacing(10)

        vr_preset_layout.addWidget(self._label("VR180 Preset"), 0, 0)
        vr_preset_layout.addWidget(self.vr180_render_preset_combo, 0, 1)

        vr_preset_layout.addWidget(self._label("Summary"), 1, 0)
        vr_preset_layout.addWidget(self.vr180_preset_desc, 1, 1)

        vr_preset_layout.addWidget(self.vr180_advanced_btn, 2, 0, 1, 2)

        content_layout.addWidget(vr_preset_group)

        # ------------------------------------------------------------
        # Group 5: Advanced VR180
        # ------------------------------------------------------------
        self.vr180_advanced_group = QGroupBox()
        self._register_title(self.vr180_advanced_group, "Advanced VR180 Settings")
        vr_layout = QGridLayout(self.vr180_advanced_group)
        vr_layout.setHorizontalSpacing(14)
        vr_layout.setVerticalSpacing(10)

        vr_layout.addWidget(self._label("HFOV"), 0, 0)
        vr_layout.addWidget(self.vr180_hfov_slider, 0, 1)
        vr_layout.addWidget(self.vr180_hfov_value, 1, 1)

        vr_layout.addWidget(self._label("Equirect Preset"), 2, 0)
        vr_layout.addWidget(self.vr180_equi_preset_combo, 2, 1)

        vr_layout.addWidget(self._label("VR180 Equirect Width"), 3, 0)
        vr_layout.addWidget(self.vr180_equi_w_spin, 3, 1)

        vr_layout.addWidget(self._label("VR180 Equirect Height"), 4, 0)
        vr_layout.addWidget(self.vr180_equi_h_spin, 4, 1)

        vr_layout.addWidget(self._label("Flat Preset"), 5, 0)
        vr_layout.addWidget(self.vr180_flat_preset_combo, 5, 1)

        vr_layout.addWidget(self._label("VR180 Flat Width"), 6, 0)
        vr_layout.addWidget(self.vr180_flat_w_spin, 6, 1)

        vr_layout.addWidget(self._label("VR180 Flat Height"), 7, 0)
        vr_layout.addWidget(self.vr180_flat_h_spin, 7, 1)

        content_layout.addWidget(self.vr180_advanced_group)

        content_layout.addStretch()

        self._update_encoding_preset_description()
        self._update_vr180_preset_description()
        self._update_active_encoder_label()
        self._set_encoding_advanced_visible(False)
        self._set_vr180_advanced_visible(False)

        return dialog
    
    def _on_state_changed(self, key, value):
        if key in {"input_video_path", "depth_map_path", "output_path"}:
            self._refresh_preview_meta()

        if key == "keyframes_path" and hasattr(self, "keyframes_path_edit"):
            self.keyframes_path_edit.setText(str(value or ""))
            self._load_keyframe_service_from_state(silent=True)
            self._refresh_keyframe_combo()

        if key == "keyframes_enabled" and hasattr(self, "keyframes_enabled_check"):
            self.keyframes_enabled_check.blockSignals(True)
            self.keyframes_enabled_check.setChecked(bool(value))
            self.keyframes_enabled_check.blockSignals(False)
            
    def _get_video_resolution(self, path: str):
        """
        Returns (width, height) for a video path.
        Uses OpenCV so users can see original source size before preview loads.
        """
        if not path or not os.path.exists(path):
            return None

        try:
            import cv2

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                return None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if width > 0 and height > 0:
                return width, height

        except Exception as exc:
            print(f"[StereoGeneratorPage] Could not read video resolution: {exc}")

        return None

    def _refresh_preview_meta(self):
        state = self.controller.state
        input_name = os.path.basename(state.input_video_path) or "None"
        depth_name = os.path.basename(state.depth_map_path) or "None"
        output_name = os.path.basename(state.output_path) or "None"

        # Prefer actual opened preview frame size when available.
        # Otherwise read the original source video directly.
        source_size = None

        mode = self._current_render_mode()

        if (
            mode == "video"
            and self._last_preview_result is not None
            and self._last_preview_result.input_frame_bgr is not None
        ):
            h, w = self._last_preview_result.input_frame_bgr.shape[:2]
            source_size = (w, h)

        elif mode == "video":
            source_size = self._get_video_resolution(state.input_video_path)

        elif mode == "image":
            pixmap = QPixmap(state.input_video_path)
            if not pixmap.isNull():
                source_size = (pixmap.width(), pixmap.height())
        source_text = ""
        if source_size:
            w, h = source_size
            ratio = w / h if h > 0 else 0
            source_text = f"    {self._t('Original')}: {w}×{h} ({ratio:.2f}:1)"

        text = (
            f"{self._t('Input')}: {input_name}    "
            f"{self._t('Depth Map')}: {depth_name}    "
            f"{self._t('Output')}: {output_name}"
            f"{source_text}    "
            f"{self._t('Mode')}: {state.preview_mode}"
        )

        self.preview_meta_label.setText(text)

        if hasattr(self.preview_panel, "set_meta"):
            self.preview_panel.set_meta("")

    def _browse_input_video(self):
        mode = self._current_render_mode()

        if mode == "image":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Input Image",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*.*)",
            )

        elif mode in ("video_folder", "image_folder"):
            title = "Select Input Video Folder" if mode == "video_folder" else "Select Input Image Folder"
            path = QFileDialog.getExistingDirectory(self, title)

        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Input Video",
                "",
                "Video Files (*.mp4 *.mkv *.avi *.mov *.webm);;All Files (*.*)",
            )

        if path:
            self.input_row.set_text(path)
            self.controller.set_state("input_video_path", path)

            if mode == "video":
                self._try_auto_open_preview_sources()

    def _browse_depth_map(self):
        mode = self._current_render_mode()

        if mode == "image":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Depth Map Image",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*.*)",
            )

        elif mode in ("video_folder", "image_folder"):
            title = "Select Depth Video Folder" if mode == "video_folder" else "Select Depth Image Folder"
            path = QFileDialog.getExistingDirectory(self, title)

        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Depth Map Video",
                "",
                "Video Files (*.mp4 *.mkv *.avi *.mov *.webm);;All Files (*.*)",
            )

        if path:
            self.depth_row.set_text(path)
            self.controller.set_state("depth_map_path", path)

            if mode == "video":
                self._try_auto_open_preview_sources()

    def _browse_output_path(self):
        mode = self._current_render_mode()

        if mode == "image":
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Choose Output Image",
                "",
                "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*.*)",
            )

        elif mode in ("video_folder", "image_folder"):
            path = QFileDialog.getExistingDirectory(
                self,
                "Select Output Folder",
            )

        else:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Choose Output File",
                "",
                "MP4 Video (*.mp4);;MKV Video (*.mkv);;All Files (*.*)",
            )

        if path:
            self.output_row.set_text(path)
            self.controller.set_state("output_path", path)

    def _try_auto_open_preview_sources(self):
        if self._current_render_mode() != "video":
            return

        if self.controller.state.input_video_path and self.controller.state.depth_map_path:
            self._load_preview_sources()

    def _load_preview_sources(self):
        total = self.controller.open_preview_sources()
        if total and total > 0:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, max(0, total - 1))
            self.frame_slider.setValue(min(self.controller.state.preview_frame_index, total - 1))
            self.frame_slider.blockSignals(False)
            self._update_frame_label(total)
            self._preview_debounce.start()

    def _update_frame_label(self, total_frames=None):
        if total_frames is None and self._last_preview_result is not None:
            total_frames = self._last_preview_result.total_frames
        if total_frames is None:
            total_frames = 0

        current = self.controller.state.preview_frame_index
        if total_frames <= 0:
            self.frame_label.setText(f"{self._t('Frame')}: 0 / 0")

        else:
            self.frame_label.setText(f"{self._t('Frame')}: {current} / {max(0, total_frames - 1)}")

    def _on_preview_mode_changed(self, value):
        self.controller.set_state("preview_mode", value)
        self._refresh_preview_meta()
        self._preview_debounce.start()

    def _on_fg_shift_changed(self, value):
        real_value = value / 10.0
        self.fg_shift_value.setText(f"{real_value:.2f}")
        self.controller.set_state("fg_shift", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_mg_shift_changed(self, value):
        real_value = value / 10.0
        self.mg_shift_value.setText(f"{real_value:.2f}")
        self.controller.set_state("mg_shift", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_bg_shift_changed(self, value):
        real_value = value / 10.0
        self.bg_shift_value.setText(f"{real_value:.2f}")
        self.controller.set_state("bg_shift", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_convergence_changed(self, value):
        real_value = value / 1000.0
        self.convergence_value.setText(f"{real_value:.3f}")
        self.controller.set_state("convergence_strength", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_sharpness_changed(self, value):
        real_value = value / 100.0
        self.sharpness_value.setText(f"{real_value:.1f}")
        self.controller.set_state("sharpness_factor", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_parallax_balance_changed(self, value):
        real_value = value / 100.0
        self.parallax_balance_value.setText(f"{real_value:.2f}")
        self.controller.set_state("parallax_balance", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()
        
    def _on_zero_parallax_changed(self, value):
        real_value = value / 1000.0
        self.zero_parallax_value.setText(f"{real_value:.3f}")
        self.controller.set_state("zero_parallax_strength", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_max_pixel_shift_changed(self, value):
        real_value = value / 1000.0
        self.max_pixel_shift_value.setText(f"{real_value:.3f}")
        self.controller.set_state("max_pixel_shift", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_dof_changed(self, value):
        real_value = value / 10.0
        self.dof_value.setText(f"{real_value:.1f}")
        self.controller.set_state("dof_strength", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()
        
    def _on_depth_pop_gamma_changed(self, value):
        real_value = value / 100.0
        self.depth_pop_gamma_value.setText(f"{real_value:.2f}")
        self.controller.set_state("depth_pop_gamma", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_fg_pop_changed(self, value):
        real_value = value / 100.0
        self.fg_pop_value.setText(f"{real_value:.2f}")
        self.controller.set_state("fg_pop_multiplier", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_bg_push_changed(self, value):
        real_value = value / 100.0
        self.bg_push_value.setText(f"{real_value:.2f}")
        self.controller.set_state("bg_push_multiplier", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_subject_lock_changed(self, value):
        real_value = value / 100.0
        self.subject_lock_value.setText(f"{real_value:.2f}")
        self.controller.set_state("subject_lock_strength", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_subject_plane_lock_changed(self, value):
        real_value = value / 100.0
        self.subject_plane_lock_value.setText(f"{real_value:.2f}")
        self.controller.set_state("subject_plane_lock_strength", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _on_subject_plane_width_changed(self, value):
        real_value = value / 100.0
        self.subject_plane_width_value.setText(f"{real_value:.2f}")
        self.controller.set_state("subject_plane_lock_width", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()
        
    def _on_foreground_curvature_changed(self, value):
        real_value = value / 100.0
        self.foreground_curvature_value.setText(f"{real_value:.2f}")
        self.controller.set_state("foreground_curvature_strength", real_value)
        self._mark_advanced_3d_base_dirty()
        self._preview_debounce.start()

    def _apply_pop_entries(self):
        try:
            pop_mid = float(self.pop_mid_edit.text())
            stretch_lo = float(self.stretch_lo_edit.text())
            stretch_hi = float(self.stretch_hi_edit.text())

            pop_mid = max(0.0, min(1.0, pop_mid))
            stretch_lo = max(0.0, min(1.0, stretch_lo))
            stretch_hi = max(0.0, min(1.0, stretch_hi))

            if stretch_hi <= stretch_lo:
                QMessageBox.warning(self, "Invalid Input", "Stretch Hi must be greater than Stretch Lo.")
                return

            self.controller.set_state("depth_pop_mid", pop_mid)
            self.controller.set_state("depth_stretch_lo", stretch_lo)
            self.controller.set_state("depth_stretch_hi", stretch_hi)
            self._mark_advanced_3d_base_dirty()

            self.pop_mid_edit.setText(f"{pop_mid:.2f}")
            self.stretch_lo_edit.setText(f"{stretch_lo:.2f}")
            self.stretch_hi_edit.setText(f"{stretch_hi:.2f}")

            self._preview_debounce.start()

        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Use numeric values for Mid / Lo / Hi in the 0..1 range.")

    def _on_ipd_enabled_changed(self, checked):
        self.controller.set_state("ipd_enabled", checked)
        self._preview_debounce.start()

    def _on_ipd_scale_changed(self, value):
        self.controller.set_state("ipd_scale", value)
        self._preview_debounce.start()

    def _on_saturation_changed(self, value):
        real_value = value / 100.0
        self.saturation_value.setText(f"{real_value:.2f}")
        self.controller.set_state("saturation", real_value)
        self._preview_debounce.start()

    def _on_contrast_changed(self, value):
        real_value = value / 100.0
        self.contrast_value.setText(f"{real_value:.2f}")
        self.controller.set_state("contrast", real_value)
        self._preview_debounce.start()

    def _on_brightness_changed(self, value):
        real_value = value / 100.0
        self.brightness_value.setText(f"{real_value:.2f}")
        self.controller.set_state("brightness", real_value)
        self._preview_debounce.start()

    def _reset_color_grading(self):
        self.controller.set_state("saturation", 1.00)
        self.controller.set_state("contrast", 1.00)
        self.controller.set_state("brightness", 0.00)

        self.saturation_slider.setValue(100)
        self.contrast_slider.setValue(100)
        self.brightness_slider.setValue(0)

        self.saturation_value.setText("1.00")
        self.contrast_value.setText("1.00")
        self.brightness_value.setText("0.00")

        self._preview_debounce.start()

    def _on_show_guides_changed(self, checked):
        self.controller.set_state("show_convergence_guides", checked)
        self._preview_debounce.start()

    def _on_frame_slider_changed(self, value):
        self.controller.set_state("preview_frame_index", value)
        self._update_frame_label()

    def _on_frame_slider_released(self):
        self._preview_debounce.start()
        
    def _clear_clip_range(self):
        self.clip_start_edit.clear()
        self.clip_end_edit.clear()
        self.controller.set_state("clip_start", "")
        self.controller.set_state("clip_end", "")

    def _load_keyframe_service_from_state(self, silent=False):
        path = str(getattr(self.controller.state, "keyframes_path", "") or "").strip()

        if not path:
            self._keyframe_service = KeyframeService()
            return self._keyframe_service

        try:
            if Path(path).exists():
                self._keyframe_service = KeyframeService.load(path)
            else:
                self._keyframe_service = KeyframeService()

            return self._keyframe_service

        except Exception as e:
            self._keyframe_service = KeyframeService()
            if not silent:
                QMessageBox.warning(
                    self,
                    self._t("Keyframes"),
                    self._t("Could not load keyframe file:") + f"\n{e}",
                )
            return self._keyframe_service


    def _ensure_keyframe_file(self):
        path = str(getattr(self.controller.state, "keyframes_path", "") or "").strip()

        if not path:
            path = str(create_default_keyframe_path(self.controller.state.input_video_path))
            self.controller.set_state("keyframes_path", path)
            self.controller.set_state("keyframes_enabled", True)
            self.keyframes_path_edit.setText(path)

        if self._keyframe_service is None:
            self._load_keyframe_service_from_state(silent=True)

        if self._keyframe_service is None:
            self._keyframe_service = KeyframeService()

        return path


    def _create_keyframes_file(self):
        path = str(create_default_keyframe_path(self.controller.state.input_video_path))

        if Path(path).exists():
            answer = QMessageBox.question(
                self,
                self._t("Keyframes"),
                self._t("A keyframe file already exists for this source. Use it?"),
            )
            if answer != QMessageBox.Yes:
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    self._t("Create Keyframes"),
                    path,
                    "VisionDepth3D Keyframes (*.json);;JSON Files (*.json);;All Files (*.*)",
                )
                if not path:
                    return

        self._keyframe_service = KeyframeService()
        self._keyframe_service.save(path)

        self.controller.set_state("keyframes_path", path)
        self.controller.set_state("keyframes_enabled", True)

        self.keyframes_enabled_check.setChecked(True)
        self.keyframes_path_edit.setText(path)
        self._refresh_keyframe_combo()

        QMessageBox.information(
            self,
            self._t("Keyframes"),
            self._t("Created keyframe file:") + f"\n{path}",
        )


    def _load_keyframes_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._t("Load Keyframes"),
            "",
            "VisionDepth3D Keyframes (*.json);;JSON Files (*.json);;All Files (*.*)",
        )

        if not path:
            return

        try:
            self._keyframe_service = KeyframeService.load(path)

            self.controller.set_state("keyframes_path", path)
            self.controller.set_state("keyframes_enabled", True)

            self.keyframes_enabled_check.setChecked(True)
            self.keyframes_path_edit.setText(path)
            self._refresh_keyframe_combo()

        except Exception as e:
            QMessageBox.critical(
                self,
                self._t("Keyframes"),
                self._t("Failed to load keyframe file:") + f"\n{e}",
            )


    def _save_keyframes_file(self):
        path = self._ensure_keyframe_file()

        try:
            self._keyframe_service.save(path)
            self.controller.set_state("keyframes_path", path)
            self.controller.set_state("keyframes_enabled", True)

            QMessageBox.information(
                self,
                self._t("Keyframes"),
                self._t("Saved keyframes to:") + f"\n{path}",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                self._t("Keyframes"),
                self._t("Failed to save keyframes:") + f"\n{e}",
            )


    def _capture_current_keyframe_settings(self) -> dict:
        """
        Captures the current UI/AppState 3D settings into the keyframe JSON format.
        This is what makes the feature editor-like instead of manual JSON editing.
        """
        state = self.controller.state

        return {
            "fg_shift": float(getattr(state, "fg_shift", 0.0)),
            "mg_shift": float(getattr(state, "mg_shift", 0.0)),
            "bg_shift": float(getattr(state, "bg_shift", 0.0)),

            "max_pixel_shift_percent": float(getattr(state, "max_pixel_shift", 0.045)),
            "parallax_balance": float(getattr(state, "parallax_balance", 0.5)),
            "zero_parallax_strength": float(getattr(state, "zero_parallax_strength", 0.0)),
            "convergence_strength": float(getattr(state, "convergence_strength", 0.0)),

            "dof_strength": float(getattr(state, "dof_strength", 0.0)),

            "depth_pop_gamma": float(getattr(state, "depth_pop_gamma", 0.85)),
            "depth_pop_mid": float(getattr(state, "depth_pop_mid", 0.50)),
            "depth_stretch_lo": float(getattr(state, "depth_stretch_lo", 0.05)),
            "depth_stretch_hi": float(getattr(state, "depth_stretch_hi", 0.95)),
            "fg_pop_multiplier": float(getattr(state, "fg_pop_multiplier", 1.0)),
            "bg_push_multiplier": float(getattr(state, "bg_push_multiplier", 1.0)),

            "subject_lock_strength": float(getattr(state, "subject_lock_strength", 0.0)),
            "subject_plane_lock_strength": float(getattr(state, "subject_plane_lock_strength", 0.0)),
            "subject_plane_lock_width": float(getattr(state, "subject_plane_lock_width", 0.08)),
            "subject_screen_plane": float(getattr(state, "subject_screen_plane", 0.0)),
            "foreground_curvature_strength": float(getattr(state, "foreground_curvature_strength", 0.0)),

            "color_saturation": float(getattr(state, "saturation", 1.0)),
            "color_contrast": float(getattr(state, "contrast", 1.0)),
            "color_brightness": float(getattr(state, "brightness", 0.0)),
        }


    def _add_or_update_keyframe_at_current_frame(self):
        path = self._ensure_keyframe_file()

        frame = int(getattr(self.controller.state, "preview_frame_index", self.frame_slider.value()))
        label = self.keyframe_label_edit.text().strip() or f"Frame {frame}"

        transition_frames = int(self.keyframe_transition_spin.value())
        transition_type = self._combo_key(self.keyframe_transition_combo, "smoothstep")

        settings = self._capture_current_keyframe_settings()

        self._keyframe_service.upsert_keyframe(
            frame=frame,
            label=label,
            transition_frames=transition_frames,
            transition_type=transition_type,
            settings=settings,
        )

        try:
            self._keyframe_service.save(path)
        except Exception as e:
            QMessageBox.critical(
                self,
                self._t("Keyframes"),
                self._t("Failed to save keyframe:") + f"\n{e}",
            )
            return

        self.controller.set_state("keyframes_enabled", True)
        self.controller.set_state("keyframes_path", path)
        self.keyframes_enabled_check.setChecked(True)

        self._refresh_keyframe_combo(select_frame=frame)

        QMessageBox.information(
            self,
            self._t("Keyframes"),
            self._t("Saved keyframe at frame") + f" {frame}.",
        )


    def _delete_selected_keyframe(self):
        if self._keyframe_service is None:
            self._load_keyframe_service_from_state(silent=True)

        frame = self.keyframe_combo.currentData()

        if frame is None:
            QMessageBox.information(
                self,
                self._t("Keyframes"),
                self._t("No keyframe selected."),
            )
            return

        frame = int(frame)

        answer = QMessageBox.question(
            self,
            self._t("Delete Keyframe"),
            self._t("Delete selected keyframe?") + f"\nFrame {frame}",
        )

        if answer != QMessageBox.Yes:
            return

        removed = self._keyframe_service.remove_keyframe_at_frame(frame)

        if not removed:
            return

        path = self._ensure_keyframe_file()
        self._keyframe_service.save(path)
        self._refresh_keyframe_combo()


    def _refresh_keyframe_combo(self, select_frame=None):
        if not hasattr(self, "keyframe_combo"):
            return

        if self._keyframe_service is None:
            self._load_keyframe_service_from_state(silent=True)

        self.keyframe_combo.blockSignals(True)
        self.keyframe_combo.clear()

        if self._keyframe_service is not None:
            keyframes = sorted(
                self._keyframe_service.keyframes,
                key=lambda kf: int(kf.frame)
            )

            for i, kf in enumerate(keyframes):
                start_frame = int(kf.frame)

                if i + 1 < len(keyframes):
                    end_frame = int(keyframes[i + 1].frame) - 1
                    range_text = f"{self._t('Frame')} {start_frame} - {end_frame}"
                else:
                    range_text = f"{self._t('Frame')} {start_frame} - {self._t('End')}"

                label = kf.label or self._t("Keyframe")
                transition = int(kf.transition_frames or 0)
                transition_type_key = str(kf.transition_type or "cut")
                transition_type = self._t(transition_type_key)

                text = f"{range_text} | {label} ({transition}f {transition_type})"
                self.keyframe_combo.addItem(self._qt_text(text), start_frame)

        if select_frame is not None:
            idx = self.keyframe_combo.findData(int(select_frame))
            if idx >= 0:
                self.keyframe_combo.setCurrentIndex(idx)

        self.keyframe_combo.blockSignals(False)


    def _on_selected_keyframe_changed(self, *_args):
        if self._keyframe_service is None:
            return

        frame = self.keyframe_combo.currentData()
        if frame is None:
            return

        kf = self._keyframe_service.get_keyframe_at_frame(int(frame))
        if kf is None:
            return

        self.keyframe_label_edit.setText(kf.label or "")
        self.keyframe_transition_spin.setValue(int(kf.transition_frames or 0))

        self._set_combo_by_key(
            self.keyframe_transition_combo,
            str(kf.transition_type or "smoothstep")
        )


    def _apply_selected_keyframe_to_sliders(self):
        """
        Loads selected keyframe settings back into the UI sliders.
        Useful when you want to inspect or revise an existing keyframe.
        """
        if self._keyframe_service is None:
            self._load_keyframe_service_from_state(silent=True)

        frame = self.keyframe_combo.currentData()
        if frame is None:
            QMessageBox.information(
                self,
                self._t("Keyframes"),
                self._t("No keyframe selected."),
            )
            return

        kf = self._keyframe_service.get_keyframe_at_frame(int(frame))
        if kf is None:
            return

        settings = kf.settings or {}

        self.frame_slider.setValue(int(kf.frame))
        self.controller.set_state("preview_frame_index", int(kf.frame))

        def set_slider_from_setting(key, slider, scale):
            if key not in settings:
                return
            slider.setValue(int(float(settings[key]) * scale))

        set_slider_from_setting("fg_shift", self.fg_shift_slider, 10)
        set_slider_from_setting("mg_shift", self.mg_shift_slider, 10)
        set_slider_from_setting("bg_shift", self.bg_shift_slider, 10)

        set_slider_from_setting("convergence_strength", self.convergence_slider, 1000)
        set_slider_from_setting("parallax_balance", self.parallax_balance_slider, 100)
        set_slider_from_setting("zero_parallax_strength", self.zero_parallax_slider, 1000)
        set_slider_from_setting("max_pixel_shift_percent", self.max_pixel_shift_slider, 1000)
        set_slider_from_setting("dof_strength", self.dof_slider, 10)

        set_slider_from_setting("depth_pop_gamma", self.depth_pop_gamma_slider, 100)
        set_slider_from_setting("fg_pop_multiplier", self.fg_pop_slider, 100)
        set_slider_from_setting("bg_push_multiplier", self.bg_push_slider, 100)
        set_slider_from_setting("subject_lock_strength", self.subject_lock_slider, 100)
        set_slider_from_setting("subject_plane_lock_strength", self.subject_plane_lock_slider, 100)
        set_slider_from_setting("subject_plane_lock_width", self.subject_plane_width_slider, 100)
        if "subject_screen_plane" in settings and hasattr(self, "simple_subject_plane_slider"):
            subject_zero_lock = max(0.0, min(1.0, float(settings["subject_screen_plane"])))
            subject_zero_lock_ui = int(round(subject_zero_lock * 100.0))
            self.simple_subject_plane_slider.blockSignals(True)
            self.simple_subject_plane_slider.setValue(max(0, min(100, subject_zero_lock_ui)))
            self.simple_subject_plane_slider.blockSignals(False)
            self.controller.set_state("subject_screen_plane", subject_zero_lock)
        set_slider_from_setting("foreground_curvature_strength", self.foreground_curvature_slider, 100)

        if "depth_pop_mid" in settings:
            self.pop_mid_edit.setText(f"{float(settings['depth_pop_mid']):.2f}")
            self.controller.set_state("depth_pop_mid", float(settings["depth_pop_mid"]))

        if "depth_stretch_lo" in settings:
            self.stretch_lo_edit.setText(f"{float(settings['depth_stretch_lo']):.2f}")
            self.controller.set_state("depth_stretch_lo", float(settings["depth_stretch_lo"]))

        if "depth_stretch_hi" in settings:
            self.stretch_hi_edit.setText(f"{float(settings['depth_stretch_hi']):.2f}")
            self.controller.set_state("depth_stretch_hi", float(settings["depth_stretch_hi"]))

        set_slider_from_setting("color_saturation", self.saturation_slider, 100)
        set_slider_from_setting("color_contrast", self.contrast_slider, 100)
        set_slider_from_setting("color_brightness", self.brightness_slider, 100)

        self._preview_debounce.start()

    def _on_preview_updated(self, result):
        self._last_preview_result = result
        self._update_frame_label(result.total_frames)

        if result.image_bgr is None:
            return

        image_bgr = result.image_bgr
        image_rgb = image_bgr[:, :, ::-1].copy()

        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._last_preview_pixmap = pixmap

        if hasattr(self.preview_panel, "placeholder"):
            target_size = self.preview_panel.frame.contentsRect().size()

            self.preview_panel.placeholder.setPixmap(
                pixmap.scaled(
                    target_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            self.preview_panel.placeholder.setAlignment(Qt.AlignCenter)

        if (
            self._fullscreen_dialog is not None
            and self._fullscreen_dialog.isVisible()
            and self._fullscreen_label is not None
        ):
            self._update_fullscreen_preview_pixmap()

        self._refresh_preview_meta()

    def _on_preview_failed(self, message):
        QMessageBox.warning(
            self,
            self._t("Preview Error"),
            str(message),
        )

    def _save_preview_image(self):
        if self._last_preview_result is None or self._last_preview_result.image_bgr is None:
            QMessageBox.information(
                self,
                self._t("Save Preview"),
                self._t("No preview image available yet."),
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            self._t("Save Preview Image"),
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*.*)",
        )

        if not path:
            return

        try:
            self.controller.preview_service.save_preview(
                path,
                self._last_preview_result.image_bgr,
            )

            QMessageBox.information(
                self,
                self._t("Save Preview"),
                self._t("Saved preview image to:") + f"\n{path}",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                self._t("Save Preview Failed"),
                str(e),
            )

    def _update_fullscreen_preview_pixmap(self):
        if self._last_preview_pixmap is None:
            return

        if self._fullscreen_label is None:
            return

        target_size = self._fullscreen_label.contentsRect().size()

        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        self._fullscreen_label.setPixmap(
            self._last_preview_pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        self._fullscreen_label.setAlignment(Qt.AlignCenter)


    def _open_fullscreen_preview(self):
        if self._last_preview_result is None or self._last_preview_result.image_bgr is None:
            QMessageBox.information(
                self,
                self._t("Fullscreen Preview"),
                self._t("No preview image available yet. Load preview sources first."),
            )
            return

        if self._last_preview_pixmap is None:
            QMessageBox.information(
                self,
                self._t("Fullscreen Preview"),
                self._t("No preview image available yet. Refresh the preview first."),
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(self._t("Fullscreen Preview"))
        dialog.setModal(False)
        dialog.setStyleSheet("""
            QDialog {
                background-color: black;
            }

            QLabel {
                background-color: black;
                color: white;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(640, 360)

        layout.addWidget(label, 1)

        self._fullscreen_dialog = dialog
        self._fullscreen_label = label

        def _cleanup():
            self._fullscreen_dialog = None
            self._fullscreen_label = None

        dialog.finished.connect(_cleanup)

        dialog.showFullScreen()

        # Let Qt finish sizing the fullscreen dialog before scaling the pixmap.
        QTimer.singleShot(50, self._update_fullscreen_preview_pixmap)

    def _on_vr180_hfov_changed(self, value):
        self.vr180_hfov_value.setText(str(int(value)))
        self.controller.set_state("vr180_hfov_deg", float(value))

    def _on_vr180_equi_preset_changed(self, *_args):
        preset_name = self._combo_key(self.vr180_equi_preset_combo, "")

        if preset_name not in VR180_EQUI_PRESETS:
            return

        w, h = VR180_EQUI_PRESETS[preset_name]

        self.vr180_equi_w_spin.blockSignals(True)
        self.vr180_equi_h_spin.blockSignals(True)

        self.vr180_equi_w_spin.setValue(int(w))
        self.vr180_equi_h_spin.setValue(int(h))

        self.vr180_equi_w_spin.blockSignals(False)
        self.vr180_equi_h_spin.blockSignals(False)

        self.controller.set_state("vr180_equi_w", int(w))
        self.controller.set_state("vr180_equi_h", int(h))


    def _on_vr180_flat_preset_changed(self, *_args):
        preset_name = self._combo_key(self.vr180_flat_preset_combo, "")

        if preset_name not in VR180_FLAT_PRESETS:
            return

        w, h = VR180_FLAT_PRESETS[preset_name]

        self.vr180_flat_w_spin.blockSignals(True)
        self.vr180_flat_h_spin.blockSignals(True)

        self.vr180_flat_w_spin.setValue(int(w))
        self.vr180_flat_h_spin.setValue(int(h))

        self.vr180_flat_w_spin.blockSignals(False)
        self.vr180_flat_h_spin.blockSignals(False)

        self.controller.set_state("vr180_flat_w", int(w))
        self.controller.set_state("vr180_flat_h", int(h))
        
    def _sync_vr180_equi_preset_combo(self):
        current = (
            int(self.vr180_equi_w_spin.value()),
            int(self.vr180_equi_h_spin.value()),
        )
        for name, size in VR180_EQUI_PRESETS.items():
            if size == current:
                self.vr180_equi_preset_combo.blockSignals(True)
                self._set_combo_by_key(self.vr180_equi_preset_combo, name)
                self.vr180_equi_preset_combo.blockSignals(False)
                return

    def _sync_vr180_flat_preset_combo(self):
        current = (
            int(self.vr180_flat_w_spin.value()),
            int(self.vr180_flat_h_spin.value()),
        )
        for name, size in VR180_FLAT_PRESETS.items():
            if size == current:
                self.vr180_flat_preset_combo.blockSignals(True)
                self._set_combo_by_key(self.vr180_flat_preset_combo, name)
                self.vr180_flat_preset_combo.blockSignals(False)
                return        

    def _refresh_preset_list(self, selected_name: str | None = None):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Select Preset")

        presets = self.controller.list_presets()
        for name in presets:
            self.preset_combo.addItem(name)

        if selected_name and selected_name in presets:
            self.preset_combo.setCurrentText(selected_name)

        self.preset_combo.blockSignals(False)

    def _load_preset_dialog(self):
        preset_dir = os.path.join(os.getcwd(), "presets")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Preset",
            preset_dir,
            "Preset JSON (*.json);;All Files (*.*)",
        )
        if not path:
            return

        try:
            ignored = self.controller.load_preset(path)

            base = os.path.splitext(os.path.basename(path))[0]
            self._refresh_preset_list(selected_name=base)
            self._load_initial_state()
            self._preview_debounce.start()

            if ignored:
                QMessageBox.information(
                    self,
                    "Preset Loaded",
                    f"Preset loaded successfully.\nIgnored keys: {ignored}",
                )
        except Exception as e:
            QMessageBox.critical(self, "Load Preset", f"Failed to load preset:\n{e}")

    def _save_preset_dialog(self):
        preset_dir = os.path.join(os.getcwd(), "presets")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Preset As",
            os.path.join(preset_dir, "custom_preset.json"),
            "Preset JSON (*.json)",
        )
        if not path:
            return

        try:
            filename = os.path.basename(path)
            saved_path = self.controller.save_preset(filename)

            base = os.path.splitext(os.path.basename(saved_path))[0]
            self._refresh_preset_list(selected_name=base)

            QMessageBox.information(
                self,
                "Preset Saved",
                f"Preset saved successfully:\n{saved_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Preset", f"Failed to save preset:\n{e}")

    def _on_preset_combo_changed(self, preset_name):
        if not preset_name or preset_name == "Select Preset":
            return

        path = os.path.join(os.getcwd(), "presets", f"{preset_name}.json")
        if not os.path.exists(path):
            return

        try:
            ignored = self.controller.load_preset(path)
            self._load_initial_state()
            self._preview_debounce.start()

            if ignored:
                print(f"Preset keys ignored: {ignored}")
        except Exception as e:
            QMessageBox.critical(self, "Load Preset", f"Failed to load preset:\n{e}")
 
    def _set_render_idle_state(self):
        self.render_btn.setEnabled(True)

        # Only video mode uses Load Preview Sources.
        # Image modes should not try to open image files as videos.
        self.preview_btn.setEnabled(self._current_render_mode() == "video")

        self.suspend_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

    def _set_render_running_state(self):
        self.render_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.suspend_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def _set_render_suspended_state(self):
        self.render_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.suspend_btn.setEnabled(False)
        self.resume_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)

    def _on_render_started(self):
        self._set_render_running_state()

    def _on_render_finished(self, outputs):
        self._set_render_idle_state()

        # For 3D Image Render, show the rendered output image in the preview panel.
        if self._current_render_mode() == "image":
            output_path = self._resolve_render_output_path(outputs)

            if output_path:
                self._show_rendered_image_in_preview(output_path)

    def _resolve_render_output_path(self, outputs):
        """
        Accepts whatever the controller emits and tries to find the actual image path.
        Supports:
        - plain string path
        - list/tuple of paths
        - dict with output_path/path/file
        - fallback to self.controller.state.output_path
        """
        candidates = []

        if isinstance(outputs, str):
            candidates.append(outputs)

        elif isinstance(outputs, dict):
            for key in ("output_path", "path", "file", "filename"):
                value = outputs.get(key)
                if isinstance(value, str):
                    candidates.append(value)

        elif isinstance(outputs, (list, tuple)):
            for item in outputs:
                if isinstance(item, str):
                    candidates.append(item)
                elif isinstance(item, dict):
                    for key in ("output_path", "path", "file", "filename"):
                        value = item.get(key)
                        if isinstance(value, str):
                            candidates.append(value)

        state_output = getattr(self.controller.state, "output_path", "")
        if state_output:
            candidates.append(state_output)

        for path in candidates:
            if path and os.path.exists(path):
                return path

        return None


    def _show_rendered_image_in_preview(self, image_path):
        """
        Loads a completed still-image render and pushes it through the same preview
        display path used by video previews.
        """
        try:
            import cv2
            from types import SimpleNamespace

            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if image_bgr is None:
                QMessageBox.warning(
                    self,
                    self._t("Preview Error"),
                    self._t("Could not load rendered image preview:") + f"\n{image_path}",
                )
                return

            result = SimpleNamespace(
                image_bgr=image_bgr,
                input_frame_bgr=image_bgr,
                total_frames=1,
            )

            self._last_preview_result = result

            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, 0)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)

            self._on_preview_updated(result)

            if hasattr(self.preview_panel, "set_meta"):
                self.preview_panel.set_meta(
                    self._t("Rendered image preview:") + f" {os.path.basename(image_path)}"
                )

        except Exception as e:
            QMessageBox.warning(
                self,
                self._t("Preview Error"),
                str(e),
            )

    def _on_render_failed_state(self, error):
        self._set_render_idle_state()

    def _on_render_cancelled_state(self):
        self._set_render_idle_state()

    def _on_render_suspended_state(self):
        self._set_render_suspended_state()

    def _on_render_resumed_state(self):
        self._set_render_running_state()
