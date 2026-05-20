from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
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
)
from ui.widgets.file_picker_row import FilePickerRow
from ui.widgets.parameter_card import ParameterCard
from ui.widgets.preview_panel import PreviewPanel
from ui.styles.page_theme import apply_unified_page_theme

import os


ASPECT_RATIO_OPTIONS = [
    "Default (16:9)",
    "Classic (4:3)",
    "Square (1:1)",
    "Vertical 9:16",
    "Instagram 4:5",
    "CinemaScope (2.39:1)",
    "Anamorphic (2.35:1)",
    "Modern Cinema (2.40:1)",
    "Ultra Panavision (2.76:1)",
    "Academy Flat (1.85:1)",
    "European Flat (1.66:1)",
    "21:9 UltraWide",
    "32:9 SuperWide",
    "2:1 (Modern Hybrid)",
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

class StereoGeneratorPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._translation_map = []
        self._last_preview_result = None
        self._last_preview_pixmap = None
        self._fullscreen_dialog = None
        self._fullscreen_label = None
        
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
        self.subject_tracking_check = self._checkbox("Stabilize Zero-Parallax")
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

        # Build frame_card BEFORE the splitter so it exists when referenced
        frame_card = ParameterCard("Frame Preview")
        self._register_title(frame_card, "Frame Preview")
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_label = QLabel(f"{self._t('Frame')}: 0 / 0")
        self.refresh_preview_btn = self._button("Refresh Preview")
        self.save_preview_btn = self._button("Save Preview Image")
        self.fullscreen_preview_btn = self._button("Fullscreen Preview")

        frame_card.inner_layout.addWidget(self.frame_label)
        frame_card.inner_layout.addWidget(self.frame_slider)
        frame_card.inner_layout.addWidget(self.refresh_preview_btn)
        frame_card.inner_layout.addWidget(self.save_preview_btn)
        frame_card.inner_layout.addWidget(self.fullscreen_preview_btn)

        frame_card.inner_layout.addSpacing(12)

        preview_settings_label = self._label("Preview Settings")
        frame_card.inner_layout.addWidget(preview_settings_label)

        preview_controls = QGridLayout()
        preview_controls.setHorizontalSpacing(10)
        preview_controls.setVerticalSpacing(8)

        preview_controls.addWidget(self._label("Preview Mode"), 0, 0)
        preview_controls.addWidget(self.preview_mode_combo, 0, 1)

        preview_controls.addWidget(self.ipd_enabled_check, 1, 0, 1, 2)

        preview_controls.addWidget(self._label("Stereo Scaling (IPD)"), 2, 0)
        preview_controls.addWidget(self.ipd_scale, 2, 1)

        preview_controls.addWidget(self.show_guides_check, 3, 0, 1, 2)

        frame_card.inner_layout.addLayout(preview_controls)

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

        depth_layout.addWidget(self._label("Zero Parallax Strength"), 6, 0)
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

        pop_layout.addWidget(self._label("Foreground Curvature"), 11, 0)
        pop_layout.addWidget(self.foreground_curvature_slider, 12, 0)
        pop_layout.addWidget(self.foreground_curvature_value, 12, 1)

        pop_layout.addWidget(self.apply_pop_entries_btn, 13, 1)

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

        # add your cards to right_col here
        right_col.addWidget(shift_card)
        right_col.addWidget(depth_card)
        right_col.addWidget(pop_card)
        right_col.addWidget(color_card)
        right_col.addStretch()

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
        self._apply_render_mode_ui()
        self._update_frame_label()
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
        self.foreground_curvature_slider.valueChanged.connect(self._on_foreground_curvature_changed)

        self.apply_pop_entries_btn.clicked.connect(self._apply_pop_entries)
        self.pop_mid_edit.returnPressed.connect(self._apply_pop_entries)
        self.stretch_lo_edit.returnPressed.connect(self._apply_pop_entries)
        self.stretch_hi_edit.returnPressed.connect(self._apply_pop_entries)
        
        self.stereo_out_combo.currentTextChanged.connect(
            lambda value: self.controller.set_state("stereo_mode", value)
        )
        self.use_ffmpeg_check.toggled.connect(
            lambda checked: self.controller.set_state("use_ffmpeg", checked)
        )
        self.keep_original_audio_check.toggled.connect(
            lambda checked: self.controller.set_state("keep_original_audio", checked)
        )
        self.preserve_hdr10_check.toggled.connect(
            lambda checked: self.controller.set_state("preserve_hdr10", checked)
        )
        self.ffmpeg_codec_combo.currentTextChanged.connect(
            lambda value: self.controller.set_state("selected_ffmpeg_codec", value)
        )
        self.basic_codec_combo.currentTextChanged.connect(
            lambda value: self.controller.set_state("selected_codec", value)
        )
        self.crf_spin.valueChanged.connect(
            lambda value: self.controller.set_state("crf_value", value)
        )
        self.nvenc_cq_spin.valueChanged.connect(
            lambda value: self.controller.set_state("nvenc_cq_value", value)
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
        self.save_preview_btn.clicked.connect(self._save_preview_image)
        self.fullscreen_preview_btn.clicked.connect(self._open_fullscreen_preview)
        self.render_btn.clicked.connect(self._start_render_clicked)

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
        self.preview_mode_combo.setCurrentText(self.controller.state.preview_mode)

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

        self.subject_lock_slider.setValue(int(self.controller.state.subject_lock_strength * 100))
        self.subject_lock_value.setText(f"{self.controller.state.subject_lock_strength:.2f}")

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
        
        self.stereo_out_combo.setCurrentText(getattr(self.controller.state, "stereo_mode", "sbs"))
        self.use_ffmpeg_check.setChecked(getattr(self.controller.state, "use_ffmpeg", False))
        self.keep_original_audio_check.setChecked(getattr(self.controller.state, "keep_original_audio", True))
        self.preserve_hdr10_check.setChecked(getattr(self.controller.state, "preserve_hdr10", False))

        self.ffmpeg_codec_combo.setCurrentText(
            getattr(self.controller.state, "selected_ffmpeg_codec", "H.264 / AVC (libx264 - CPU)")
        )
        self.basic_codec_combo.setCurrentText(getattr(self.controller.state, "selected_codec", "XVID"))
        self.crf_spin.setValue(getattr(self.controller.state, "crf_value", 23))
        self.nvenc_cq_spin.setValue(getattr(self.controller.state, "nvenc_cq_value", 23))

        self.preserve_aspect_check.setChecked(getattr(self.controller.state, "preserve_original_aspect", False))
        self.auto_crop_check.setChecked(getattr(self.controller.state, "auto_crop_black_bars", False))
        self.subject_tracking_check.setChecked(getattr(self.controller.state, "use_subject_tracking", False))
        self.skip_blank_check.setChecked(getattr(self.controller.state, "skip_blank_frames", False))
        self.edge_masking_check.setChecked(getattr(self.controller.state, "enable_edge_masking", True))
        self.feathering_check.setChecked(getattr(self.controller.state, "enable_feathering", True))

        self.edge_repair_quality_combo.setCurrentText(
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

        self.frame_slider.setValue(self.controller.state.preview_frame_index)
        self._sync_vr180_equi_preset_combo()
        self._sync_vr180_flat_preset_combo()
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

    def _build_encoding_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(self._t("Output & Encoding"))
        dialog.setModal(False)
        dialog.resize(700, 820)
        dialog.setMinimumSize(640, 700)

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

        # --- Group 1: Format ---
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

        # --- Group 2: Render Options ---
        render_group = QGroupBox()
        self._register_title(render_group, "Render Options")
        render_layout = QVBoxLayout(render_group)
        render_layout.setSpacing(8)

        render_layout.addWidget(self.use_ffmpeg_check)
        render_layout.addWidget(self.keep_original_audio_check)
        render_layout.addWidget(self.preserve_hdr10_check)

        content_layout.addWidget(render_group)

        # --- Group 3: Codecs & Quality ---
        codec_group = QGroupBox()
        self._register_title(codec_group, "Codecs & Quality")
        codec_layout = QGridLayout(codec_group)
        codec_layout.setHorizontalSpacing(14)
        codec_layout.setVerticalSpacing(10)

        codec_layout.addWidget(self._label("FFmpeg Codec"), 0, 0)
        codec_layout.addWidget(self.ffmpeg_codec_combo, 0, 1)

        codec_layout.addWidget(self._label("Basic Codec"), 1, 0)
        codec_layout.addWidget(self.basic_codec_combo, 1, 1)

        codec_layout.addWidget(self._label("CRF"), 2, 0)
        codec_layout.addWidget(self.crf_spin, 2, 1)

        codec_layout.addWidget(self._label("NVENC CQ"), 3, 0)
        codec_layout.addWidget(self.nvenc_cq_spin, 3, 1)

        content_layout.addWidget(codec_group)

        # --- Group 4: VR180 ---
        vr_group = QGroupBox()
        self._register_title(vr_group, "VR180 Settings")
        vr_layout = QGridLayout(vr_group)
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

        content_layout.addWidget(vr_group)

        content_layout.addStretch()

        return dialog
    
    def _on_state_changed(self, key, value):
        if key in {"input_video_path", "depth_map_path", "output_path"}:
            self._refresh_preview_meta()

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
        self._preview_debounce.start()

    def _on_mg_shift_changed(self, value):
        real_value = value / 10.0
        self.mg_shift_value.setText(f"{real_value:.2f}")
        self.controller.set_state("mg_shift", real_value)
        self._preview_debounce.start()

    def _on_bg_shift_changed(self, value):
        real_value = value / 10.0
        self.bg_shift_value.setText(f"{real_value:.2f}")
        self.controller.set_state("bg_shift", real_value)
        self._preview_debounce.start()

    def _on_convergence_changed(self, value):
        real_value = value / 1000.0
        self.convergence_value.setText(f"{real_value:.3f}")
        self.controller.set_state("convergence_strength", real_value)
        self._preview_debounce.start()

    def _on_sharpness_changed(self, value):
        real_value = value / 100.0
        self.sharpness_value.setText(f"{real_value:.1f}")
        self.controller.set_state("sharpness_factor", real_value)
        self._preview_debounce.start()

    def _on_parallax_balance_changed(self, value):
        real_value = value / 100.0
        self.parallax_balance_value.setText(f"{real_value:.2f}")
        self.controller.set_state("parallax_balance", real_value)
        self._preview_debounce.start()
        
    def _on_zero_parallax_changed(self, value):
        real_value = value / 1000.0
        self.zero_parallax_value.setText(f"{real_value:.3f}")
        self.controller.set_state("zero_parallax_strength", real_value)
        self._preview_debounce.start()

    def _on_max_pixel_shift_changed(self, value):
        real_value = value / 1000.0
        self.max_pixel_shift_value.setText(f"{real_value:.3f}")
        self.controller.set_state("max_pixel_shift", real_value)
        self._preview_debounce.start()

    def _on_dof_changed(self, value):
        real_value = value / 10.0
        self.dof_value.setText(f"{real_value:.1f}")
        self.controller.set_state("dof_strength", real_value)
        self._preview_debounce.start()
        
        
    def _on_depth_pop_gamma_changed(self, value):
        real_value = value / 100.0
        self.depth_pop_gamma_value.setText(f"{real_value:.2f}")
        self.controller.set_state("depth_pop_gamma", real_value)
        self._preview_debounce.start()

    def _on_fg_pop_changed(self, value):
        real_value = value / 100.0
        self.fg_pop_value.setText(f"{real_value:.2f}")
        self.controller.set_state("fg_pop_multiplier", real_value)
        self._preview_debounce.start()

    def _on_bg_push_changed(self, value):
        real_value = value / 100.0
        self.bg_push_value.setText(f"{real_value:.2f}")
        self.controller.set_state("bg_push_multiplier", real_value)
        self._preview_debounce.start()

    def _on_subject_lock_changed(self, value):
        real_value = value / 100.0
        self.subject_lock_value.setText(f"{real_value:.2f}")
        self.controller.set_state("subject_lock_strength", real_value)
        self._preview_debounce.start()
        
    def _on_foreground_curvature_changed(self, value):
        real_value = value / 100.0
        self.foreground_curvature_value.setText(f"{real_value:.2f}")
        self.controller.set_state("foreground_curvature_strength", real_value)
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

    def _save_preview_image(self):
        if self._last_preview_result is None or self._last_preview_result.image_bgr is None:
            QMessageBox.information(self, "Save Preview", "No preview image available yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Preview Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*.*)",
        )
        if not path:
            return

        try:
            self.controller.preview_service.save_preview(path, self._last_preview_result.image_bgr)
            QMessageBox.information(self, "Save Preview", f"Saved preview image to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Preview Failed", str(e))

    def _on_vr180_hfov_changed(self, value):
        self.vr180_hfov_value.setText(str(int(value)))
        self.controller.set_state("vr180_hfov_deg", float(value))

    def _on_vr180_equi_preset_changed(self, preset_name):
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

    def _on_vr180_flat_preset_changed(self, preset_name):
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
                self.vr180_equi_preset_combo.setCurrentText(name)
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
                self.vr180_flat_preset_combo.setCurrentText(name)
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
        self.preview_btn.setEnabled(True)
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

    def _on_render_failed_state(self, error):
        self._set_render_idle_state()

    def _on_render_cancelled_state(self):
        self._set_render_idle_state()

    def _on_render_suspended_state(self):
        self._set_render_suspended_state()

    def _on_render_resumed_state(self):
        self._set_render_running_state()
