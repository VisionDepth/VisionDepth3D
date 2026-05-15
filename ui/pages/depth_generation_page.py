import os
import threading
import tempfile
import shutil

from PySide6.QtCore import Qt, QTimer
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QCheckBox, QSpinBox, QMessageBox,
    QFrame, QGroupBox, QLineEdit, QScrollArea, QSplitter,
)
from ui.widgets.parameter_card import ParameterCard
from ui.styles.page_theme import apply_unified_page_theme


CODEC_OPTIONS = [
    "H.264 / AVC (libx264 - CPU)", "H.265 / HEVC (libx265 - CPU)",
    "AV1 (libaom - CPU)", "AV1 (SVT - CPU, faster)",
    "MPEG-4 (mp4v - CPU)", "XviD (AVI - CPU)", "DivX (AVI - CPU)",
    "H.264 / AVC (NVENC - NVIDIA GPU)", "H.265 / HEVC (NVENC - NVIDIA GPU)",
    "AV1 (NVENC - NVIDIA RTX 40+ GPU)", "H.264 / AVC (AMF - AMD GPU)",
    "H.265 / HEVC (AMF - AMD GPU)", "AV1 (AMF - AMD RDNA3+)",
    "H.264 / AVC (QSV - Intel GPU)", "H.265 / HEVC (QSV - Intel GPU)",
    "VP9 (QSV - Intel GPU)", "AV1 (QSV - Intel ARC / Gen11+)",
]

INFERENCE_RESOLUTIONS = {
    "Original": None,
    "256x256": (256, 256),
    "384x384 (DPT Large / MiDaS v3.0 Default)": (384, 384),
    "504x504 (DA3 Native)": (504, 504),
    "512x512 (BEiT / MiDaS v3.1 Native)": (512, 512),
    "518x518 (Depth Anything / Video Depth Anything Default)": (518, 518),
    "560x560 (Distill-Any-Depth Train Size)": (560, 560),
    "640x640": (640, 640),
    "700x700 (Distill-Any-Depth Repo Example)": (700, 700),
    "768x768 (Marigold Depth v1.1 Diffusion Default)": (768, 768),
    "896x896": (896, 896),
    "1536x1536 (Depth Pro Native)": (1536, 1536),
    "512x288": (512, 288), "640x352": (640, 352),
    "768x432": (768, 432), "896x512": (896, 512),
    "1024x576": (1024, 576), "1152x640": (1152, 640),
    "1280x720": (1280, 720), "1280x768 (LBM Depth Widescreen)": (1280, 768),
    "1344x768": (1344, 768), "1536x864": (1536, 864),
    "1600x896": (1600, 896), "1792x1008": (1792, 1008),
    "1920x1088": (1920, 1088), "1920x512 (LBM Depth Cinematic Wide)": (1920, 512),
    "512x256 (Fastest)": (512, 256), "704x384 (Balanced)": (704, 384),
    "910x518 (Depth Anything Widescreen)": (910, 518),
    "960x540 (Good Quality)": (960, 540),
    "1024x576 (Max Quality)": (1024, 576),
    "1280x720 (720p HD)": (1280, 720),
    "1920x1080 (1080p HD)": (1920, 1080),
}

OFFLOAD_MODES = ["none", "sequential", "full"]


class SectionHeader(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("SectionHeader")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        label = QLabel(title)
        label.setObjectName("SectionHeaderLabel")
        layout.addWidget(label)
        layout.addStretch()


class PathRow(QWidget):
    browse_clicked = QtSignal()

    def __init__(self, label_text: str, placeholder: str = ""):
        super().__init__()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.label = QLabel(label_text)
        self.label.setMinimumWidth(90)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setReadOnly(True)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setMinimumWidth(72)
        self.browse_btn.clicked.connect(self.browse_clicked.emit)

        layout.addWidget(self.label)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.browse_btn)

    def set_label_text(self, text: str):
        self.label.setText(text)

    def set_placeholder_text(self, text: str):
        self.edit.setPlaceholderText(text)

    def set_browse_text(self, text: str):
        self.browse_btn.setText(text)

    def set_text(self, text: str):
        self.edit.setText(text)

    def text(self) -> str:
        return self.edit.text()

class DepthGenerationPage(QWidget):
    preview_samples_ready = QtSignal(list)
    preview_samples_failed = QtSignal(str)
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._state = controller.depth_state
        self._translation_map = []
        
        self._preview_pairs = []
        self._preview_index = 0
        self._preview_running = False
        self.preview_samples_ready.connect(self._on_depth_preview_samples_ready)
        self.preview_samples_failed.connect(self._on_depth_preview_samples_failed)

        self._warmup_timer = QTimer()
        self._warmup_timer.timeout.connect(self._pulse_warmup)
        self._warmup_dots = 0

        # ═══════════ ROOT: Vertical split ═══════════
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── TOP: Controls (left) + Previews (right) ──
        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        # Left: Controls
        left_widget = QWidget()
        left_widget.setFixedWidth(300)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # --- Sources ---
        sources_card = ParameterCard("Sources")
        self._register_title(sources_card, "Sources")

        self.input_row = PathRow("Input", "Select source...")
        self._register_path_row(self.input_row, "Input Image", "Select source...")
        self.input_row.browse_clicked.connect(self._browse_input_video)

        self.output_dir_row = PathRow("Output Dir", "Select folder...")
        self._register_path_row(self.output_dir_row, "Output Dir", "Select folder...")
        self.output_dir_row.browse_clicked.connect(self._browse_output_dir)

        sources_card.inner_layout.addWidget(self.input_row)
        sources_card.inner_layout.addWidget(self.output_dir_row)

        left_layout.addWidget(sources_card)

        # --- Model ---
        model_card = ParameterCard("Model")
        self._register_title(model_card, "Model")

        self.model_combo = QComboBox()
        self.model_combo.addItem("  -- Select Model -- ")
        self.model_combo.wheelEvent = lambda event: None

        model_card.inner_layout.addWidget(self.model_combo)

        left_layout.addWidget(model_card)

        # --- Inference ---
        inference_card = ParameterCard("Inference")
        self._register_title(inference_card, "Inference")

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(list(INFERENCE_RESOLUTIONS.keys()))
        self.resolution_combo.setCurrentText("Original")
        self.resolution_combo.wheelEvent = lambda event: None

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        self.batch_spin.setPrefix(f"{self._t('Batch:')} ")

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 50)
        self.steps_spin.setValue(5)
        self.steps_spin.setPrefix(f"{self._t('Steps:')} ")

        inference_card.inner_layout.addWidget(self.resolution_combo)
        inference_card.inner_layout.addWidget(self.batch_spin)
        inference_card.inner_layout.addWidget(self.steps_spin)

        left_layout.addWidget(inference_card)

        # --- Output ---
        output_card = ParameterCard("Output")
        self._register_title(output_card, "Output")

        self.codec_combo = QComboBox()
        self.codec_combo.addItems(CODEC_OPTIONS)
        self.codec_combo.wheelEvent = lambda event: None

        output_card.inner_layout.addWidget(self.codec_combo)

        left_layout.addWidget(output_card)

        # --- Options ---
        options_card = ParameterCard("Options")
        self._register_title(options_card, "Options")

        self.invert_check = QCheckBox()
        self.save_frames_check = QCheckBox()
        self.fp16_check = QCheckBox()

        self._register_text(self.invert_check, "Invert Depth")
        self._register_text(self.save_frames_check, "Save Frames")
        self._register_text(self.fp16_check, "Use FP16")
        self.offload_combo = QComboBox()
        self.offload_combo.addItems(OFFLOAD_MODES)
        self.offload_combo.setCurrentText("none")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["Default", "Magma", "Viridis", "Inferno", "Plasma", "Gray"])
        self.colormap_combo.wheelEvent = lambda event: None 
        options_card.inner_layout.addWidget(self.invert_check)
        options_card.inner_layout.addWidget(self.save_frames_check)
        options_card.inner_layout.addWidget(self.fp16_check)
        self.offload_label = QLabel()
        self._register_text(self.offload_label, "CPU Offload Mode")

        self.colormap_label = QLabel()
        self._register_text(self.colormap_label, "Colormap")

        options_card.inner_layout.addWidget(self.offload_label)
        options_card.inner_layout.addWidget(self.offload_combo)
        options_card.inner_layout.addWidget(self.colormap_label)
        options_card.inner_layout.addWidget(self.colormap_combo)
        left_layout.addWidget(options_card)
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setWidget(left_widget)
        left_scroll.setMinimumWidth(300)
        left_scroll.setMaximumWidth(340)

        # Right: Previews side-by-side
        preview_widget = QWidget()
        preview_outer = QVBoxLayout(preview_widget)
        preview_outer.setContentsMargins(0, 0, 0, 0)
        preview_outer.setSpacing(10)

        preview_row = QWidget()
        preview_layout = QHBoxLayout(preview_row)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(12)

        input_group = QGroupBox()
        self._register_title(input_group, "Source")
        input_inner = QVBoxLayout(input_group)
        self.input_preview = QLabel()
        self._register_text(self.input_preview, "Input Preview")
        self.input_preview.setAlignment(Qt.AlignCenter)
        self.input_preview.setMinimumSize(240, 160)
        self.input_preview.setObjectName("PreviewPanel")
        input_inner.addWidget(self.input_preview)

        depth_group = QGroupBox()
        self._register_title(depth_group, "Depth Map")
        depth_inner = QVBoxLayout(depth_group)
        self.depth_preview = QLabel()
        self._register_text(self.depth_preview, "Depth Output")
        self.depth_preview.setAlignment(Qt.AlignCenter)
        self.depth_preview.setMinimumSize(240, 160)
        depth_inner.addWidget(self.depth_preview)

        preview_layout.addWidget(input_group, 1)
        preview_layout.addWidget(depth_group, 1)
        preview_controls = QWidget()
        preview_controls_layout = QHBoxLayout(preview_controls)
        preview_controls_layout.setContentsMargins(0, 0, 0, 0)
        preview_controls_layout.setSpacing(10)

        self.generate_preview_btn = QPushButton()
        self._register_text(self.generate_preview_btn, "Generate Preview")

        self.prev_preview_btn = QPushButton()
        self._register_text(self.prev_preview_btn, "Previous")

        self.next_preview_btn = QPushButton()
        self._register_text(self.next_preview_btn, "Next")

        self.preview_index_label = QLabel("Preview 0 / 0")
        self.preview_index_label.setAlignment(Qt.AlignCenter)

        preview_controls_layout.addWidget(self.generate_preview_btn, 2)
        preview_controls_layout.addWidget(self.prev_preview_btn, 1)
        preview_controls_layout.addWidget(self.preview_index_label, 1)
        preview_controls_layout.addWidget(self.next_preview_btn, 1)

        preview_outer.addWidget(preview_row, 1)
        preview_outer.addWidget(preview_controls, 0)
        
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        self.main_splitter.addWidget(left_scroll)
        self.main_splitter.addWidget(preview_widget)

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)

        self.main_splitter.setSizes([330, 1200])

        top_row.addWidget(self.main_splitter, 1)

        root.addLayout(top_row, 1)

        # ── BOTTOM: Mode + Actions + Status ──
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(12)

        mode_group = QGroupBox("Processing Mode")
        mode_inner = QHBoxLayout(mode_group)
        self.model_combo.wheelEvent = lambda event: None 
        self.mode_combo = QComboBox()
        self._refresh_mode_combo()

        self.mode_label = QLabel()
        self._register_text(self.mode_label, "Mode:")
        mode_inner.addWidget(self.mode_label)
        mode_inner.addWidget(self.mode_combo, 1)
        bottom_layout.addWidget(mode_group)

        process_group = QGroupBox()
        self._register_title(process_group, "Actions")
        process_inner = QHBoxLayout(process_group)
        self.process_btn = QPushButton()
        self._register_text(self.process_btn, "▶ Start Processing")
        self.process_btn.setMinimumHeight(36)

        self.suspend_btn = QPushButton()
        self.resume_btn = QPushButton()
        self.cancel_btn = QPushButton()

        self._register_text(self.suspend_btn, "Suspend")
        self._register_text(self.resume_btn, "Resume")
        self._register_text(self.cancel_btn, "Cancel")
        process_inner.addWidget(self.process_btn)
        process_inner.addWidget(self.suspend_btn)
        process_inner.addWidget(self.resume_btn)
        process_inner.addWidget(self.cancel_btn)
        bottom_layout.addWidget(process_group, 1)

        self.status_label = QLabel(self._t("Ready"))
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setMinimumWidth(180)
        bottom_layout.addWidget(self.status_label)

        root.addWidget(bottom_widget, 0)

        self._bind_events()
        self._load_initial_state()
        self._set_idle_state()

    def apply_theme(self, theme: dict):
        self._active_theme = theme or {}
        apply_unified_page_theme(self, self._active_theme)

    def _t(self, key: str) -> str:
        """
        Translation helper for Depth Engine.
        Supports current PySide6 keys and your older original JSON keys.
        """
        translator = getattr(self.controller, "t", None)
        if not callable(translator):
            return key

        translations = getattr(self.controller, "translations", None)
        if translations is None:
            language_service = getattr(self.controller, "language_service", None)
            translations = getattr(language_service, "translations", None)

        aliases = {
            # Modern UI labels to your original JSON keys
            "Sources": "Input / Output",
            "Inference": "Inference Resolution:",
            "Options": "Processing Options",
            "Use FP16": "Use float16",
            "Colormap": "Colormap:",
            "Batch:": "Batch Size (Frames):",
            "Steps:": "Inference Steps:",
            "Codec": "Video Codec:",
            "Output Dir": "Output Dir: None",
            "Select source...": "Select Input Video",
            "Select folder...": "Choose Directory",

            # Preview area
            "Source": "Input Image",
            "Input Preview": "Input Image",
            "Depth Output": "Depth Map",

            # Bottom area
            "Processing Mode": "Mode",
            "Start Processing": "▶ Start Processing",
            "Loading model...": "Loading model...",
            "Model ready.": "Ready",
            "Cancelled.": "Cancel",
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

    def _set_title_text(self, widget, text: str):
        """
        Works for QGroupBox and most custom card widgets.
        ParameterCard usually stores a QLabel internally, so this also tries child labels.
        """
        if hasattr(widget, "setTitle"):
            widget.setTitle(text)
            return

        for attr in ("title_label", "header_label", "label"):
            title_label = getattr(widget, attr, None)
            if title_label is not None and hasattr(title_label, "setText"):
                title_label.setText(text)
                return

        labels = widget.findChildren(QLabel)
        if labels:
            labels[0].setText(text)

    def _register_text(self, widget, key: str):
        self._translation_map.append((widget, key, "text"))
        widget.setText(self._t(key))

    def _register_title(self, widget, key: str):
        self._translation_map.append((widget, key, "title"))
        self._set_title_text(widget, self._t(key))

    def _register_path_row(self, row, label_key: str, placeholder_key: str):
        self._translation_map.append((row, label_key, "path_label"))
        self._translation_map.append((row, placeholder_key, "path_placeholder"))
        self._translation_map.append((row, "Browse", "path_browse"))

        row.set_label_text(self._t(label_key))
        row.set_placeholder_text(self._t(placeholder_key))
        row.set_browse_text(self._t("Browse"))

    def _set_spin_prefixes(self):
        if hasattr(self, "batch_spin"):
            self.batch_spin.setPrefix(f"{self._t('Batch:')} ")

        if hasattr(self, "steps_spin"):
            self.steps_spin.setPrefix(f"{self._t('Steps:')} ")

    def _refresh_mode_combo(self):
        if not hasattr(self, "mode_combo"):
            return

        current_data = self.mode_combo.currentData()
        if not current_data:
            current_data = "Video"

        items = [
            ("Process Video", "Video"),
            ("Process Video Folder", "Video Folder"),
            ("Process Image", "Image"),
            ("Process Image Folder", "Image Folder"),
        ]

        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()

        restore_index = 0
        for i, (label_key, data_key) in enumerate(items):
            self.mode_combo.addItem(self._t(label_key), data_key)
            if data_key == current_data:
                restore_index = i

        self.mode_combo.setCurrentIndex(restore_index)
        self.mode_combo.blockSignals(False)

    def refresh_labels(self):
        """
        Called by MainWindow when language changes.
        """
        for widget, key, widget_type in self._translation_map:
            try:
                if widget_type == "text":
                    widget.setText(self._t(key))

                elif widget_type == "title":
                    self._set_title_text(widget, self._t(key))

                elif widget_type == "path_label":
                    widget.set_label_text(self._t(key))

                elif widget_type == "path_placeholder":
                    widget.set_placeholder_text(self._t(key))

                elif widget_type == "path_browse":
                    widget.set_browse_text(self._t(key))

            except RuntimeError:
                pass

        self._set_spin_prefixes()
        self._refresh_mode_combo()
        self._apply_processing_mode_ui()

    def _current_processing_mode(self):
        if hasattr(self, "mode_combo"):
            return self.mode_combo.currentData() or self.mode_combo.currentText()
        return "Video"

    def _on_processing_mode_changed(self, *_args):
        mode = self._current_processing_mode()
        setattr(self._state, "processing_mode", mode)
        self._apply_processing_mode_ui()

    def _apply_processing_mode_ui(self):
        mode = self._current_processing_mode()

        if mode == "Video":
            self.input_row.set_label_text(self._t("Input Video"))
            self.input_row.set_placeholder_text(self._t("Select source video..."))
            self.output_dir_row.set_label_text(self._t("Output Dir"))
            self.output_dir_row.set_placeholder_text(self._t("Select folder..."))

        elif mode == "Video Folder":
            self.input_row.set_label_text(self._t("Input Video Folder"))
            self.input_row.set_placeholder_text(self._t("Select video folder..."))
            self.output_dir_row.set_label_text(self._t("Output Dir"))
            self.output_dir_row.set_placeholder_text(self._t("Select folder..."))

        elif mode == "Image":
            self.input_row.set_label_text(self._t("Input Image"))
            self.input_row.set_placeholder_text(self._t("Select source image..."))
            self.output_dir_row.set_label_text(self._t("Output Dir"))
            self.output_dir_row.set_placeholder_text(self._t("Select folder..."))

        elif mode == "Image Folder":
            self.input_row.set_label_text(self._t("Input Image Folder"))
            self.input_row.set_placeholder_text(self._t("Select image folder..."))
            self.output_dir_row.set_label_text(self._t("Output Dir"))
            self.output_dir_row.set_placeholder_text(self._t("Select folder..."))

        self.input_row.set_browse_text(self._t("Browse"))
        self.output_dir_row.set_browse_text(self._t("Browse"))
        
        is_video_mode = (mode == "Video")

        if hasattr(self, "generate_preview_btn"):
            self.generate_preview_btn.setEnabled(is_video_mode and not self._preview_running)

        if hasattr(self, "prev_preview_btn"):
            self.prev_preview_btn.setEnabled(is_video_mode and len(self._preview_pairs) > 1)

        if hasattr(self, "next_preview_btn"):
            self.next_preview_btn.setEnabled(is_video_mode and len(self._preview_pairs) > 1)

    # ── All the same methods from your existing file ──
    def _bind_events(self):
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_processing_mode_changed)
        self.resolution_combo.currentTextChanged.connect(lambda v: setattr(self._state, "inference_resolution", v))
        self.batch_spin.valueChanged.connect(lambda v: setattr(self._state, "batch_size", v))
        self.steps_spin.valueChanged.connect(lambda v: setattr(self._state, "inference_steps", v))
        self.codec_combo.currentTextChanged.connect(lambda v: setattr(self._state, "codec", v))
        self.invert_check.toggled.connect(lambda v: setattr(self._state, "invert_depth", v))
        self.colormap_combo.currentTextChanged.connect(lambda v: setattr(self._state, "colormap", v))
        self.save_frames_check.toggled.connect(lambda v: setattr(self._state, "save_frames", v))
        self.fp16_check.toggled.connect(lambda v: setattr(self._state, "use_fp16", v))
        self.offload_combo.currentTextChanged.connect(lambda v: setattr(self._state, "offload_mode", v))
        self.process_btn.clicked.connect(self._start_processing)
        self.generate_preview_btn.clicked.connect(self._generate_depth_preview_samples)
        self.prev_preview_btn.clicked.connect(self._show_previous_preview_pair)
        self.next_preview_btn.clicked.connect(self._show_next_preview_pair)
        self.suspend_btn.clicked.connect(self._suspend)
        self.resume_btn.clicked.connect(self._resume)
        self.cancel_btn.clicked.connect(self._cancel)
        self.controller.depth_started.connect(self._on_depth_started)
        self.controller.depth_finished.connect(self._on_depth_finished)
        self.controller.depth_finished.connect(self._on_depth_image_done)
        self.controller.depth_failed.connect(self._on_depth_failed)
        self.controller.depth_cancelled.connect(self._on_depth_cancelled)
        self.controller.depth_suspended.connect(self._on_depth_suspended)
        self.controller.depth_resumed.connect(self._on_depth_resumed)

    def _load_initial_state(self):
        try:
            from core.render_depth import load_supported_models
            supported = load_supported_models()
            self.model_combo.clear()
            self.model_combo.addItems(list(supported.keys()))
        except Exception as e:
            print(f"Failed to load model list: {e}")
        self.input_row.set_text(self._state.input_video_path)
        self.output_dir_row.set_text(self._state.output_dir)
        self.model_combo.setCurrentText(self._state.selected_model)
        self._update_diffusion_options(self._state.selected_model)
        self.resolution_combo.setCurrentText(self._state.inference_resolution)
        self.batch_spin.setValue(self._state.batch_size)
        self.steps_spin.setValue(self._state.inference_steps)
        self.codec_combo.setCurrentText(self._state.codec)
        self.invert_check.setChecked(self._state.invert_depth)
        self.save_frames_check.setChecked(self._state.save_frames)
        self.fp16_check.setChecked(self._state.use_fp16)
        self.offload_combo.setCurrentText(self._state.offload_mode)
        self.colormap_combo.setCurrentText(self._state.colormap)
        self._apply_processing_mode_ui()

    def _browse_input_video(self):
        mode = self._current_processing_mode()

        if mode == "Image":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Input Image",
                "",
                "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;All Files (*.*)"
            )

        elif mode == "Image Folder":
            path = QFileDialog.getExistingDirectory(
                self,
                "Select Input Image Folder"
            )

        elif mode == "Video Folder":
            path = QFileDialog.getExistingDirectory(
                self,
                "Select Input Video Folder"
            )

        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Input Video",
                "",
                "Video Files (*.mp4 *.mkv *.avi *.mov *.webm);;All Files (*.*)"
            )

        if path:
            self.input_row.set_text(path)
            self._state.input_video_path = path

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_row.set_text(path)
            self._state.output_dir = path

    def _on_model_changed(self, value):
        self._state.selected_model = value
        self._update_diffusion_options(value)
        if value and value != "  -- Select Model -- ":
            self._warmup_timer.start(500)
            self.process_btn.setEnabled(False)
            self.process_btn.setText(f"⏳ {self._t('Loading model...')}")
            self.controller.load_depth_model(value)
            self._ready_check = QTimer()
            self._ready_check.timeout.connect(self._check_pipe_ready)
            self._ready_check.start(2000)
            self._warmup_fallback = QTimer()
            self._warmup_fallback.setSingleShot(True)
            self._warmup_fallback.timeout.connect(self._on_model_ready)
            self._warmup_fallback.start(300000)

    def _update_diffusion_options(self, model_name):
        from core.render_depth import supported_models
        checkpoint = supported_models.get(model_name, "")
        is_diffusion = isinstance(checkpoint, str) and checkpoint.startswith("diffusers:")
        self.steps_spin.setVisible(is_diffusion)
        self.offload_combo.setVisible(is_diffusion)

    def _start_processing(self):
        mode = self.mode_combo.currentData() or self.mode_combo.currentText()
        setattr(self._state, "processing_mode", mode)

        if not self._state.input_video_path:
            QMessageBox.warning(self, "Missing Input", "Please select an input source.")
            return
        if not self._state.output_dir:
            QMessageBox.warning(self, "Missing Output", "Please select an output directory.")
            return
        if mode in ("Video", "Video Folder"):
            self.controller.start_depth_processing()
        elif mode == "Image":
            self.controller.start_depth_image()
        elif mode == "Image Folder":
            self.controller.start_depth_image_folder()

    def _suspend(self): self.controller.suspend_depth()
    def _resume(self): self.controller.resume_depth()
    def _cancel(self): self.controller.cancel_depth()

    def _on_depth_started(self):
        self._set_running_state()
        self.status_label.setText(self._t("Processing..."))

    def _on_depth_suspended(self):
        self._set_suspended_state()
        self.status_label.setText(self._t("Paused."))

    def _on_depth_resumed(self):
        self._set_running_state()
        self.status_label.setText(self._t("Resuming..."))

    def _on_depth_finished(self, output_path: str):
        self._set_idle_state()
        self.status_label.setText(f"Done: {output_path}")

    def _on_depth_failed(self, error: str):
        self._set_idle_state()
        self.status_label.setText(f"Failed: {error}")
        QMessageBox.critical(self, "Depth Failed", error)

    def _on_depth_cancelled(self):
        self._set_idle_state()
        self.status_label.setText(self._t("Cancelled."))

    def _on_depth_image_done(self, output_path):
        if self._state.input_video_path:
            self._show_preview(self.input_preview, self._state.input_video_path)
        depth_file = os.path.join(output_path,
            os.path.splitext(os.path.basename(self._state.input_video_path))[0] + "_depth.png")
        if os.path.exists(depth_file):
            self._show_preview(self.depth_preview, depth_file)

    def _show_preview(self, label, image_path):
        if not image_path or not os.path.exists(image_path):
            return
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _get_preview_sample_indices(self, total_frames: int):
        if total_frames <= 0:
            return []

        positions = [0.0, 0.25, 0.5, 0.75, 0.95]
        indices = []

        for p in positions:
            idx = int(round((total_frames - 1) * p))
            indices.append(max(0, min(total_frames - 1, idx)))

        # remove duplicates while preserving order
        unique = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)

        return unique
        
    def _generate_depth_preview_samples(self):
        mode = self._current_processing_mode()

        if mode != "Video":
            QMessageBox.information(self, "Preview", "Depth preview samples are only available in Video mode.")
            return

        if not self._state.input_video_path:
            QMessageBox.warning(self, "Missing Input", "Please select an input video first.")
            return

        if not self._state.selected_model or self._state.selected_model == "  -- Select Model -- ":
            QMessageBox.warning(self, "Missing Model", "Please select a depth model first.")
            return

        self._preview_running = True
        self._preview_pairs = []
        self._preview_index = 0

        self.status_label.setText("Generating preview samples...")
        self.generate_preview_btn.setEnabled(False)
        self.prev_preview_btn.setEnabled(False)
        self.next_preview_btn.setEnabled(False)
        self.preview_index_label.setText("Preview 0 / 0")

        # run worker here
        self._run_depth_preview_worker()
    
    def _preview_var(self, value):
        class PreviewVar:
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

            def configure(self, *args, **kwargs):
                pass

            def winfo_toplevel(self):
                return self

        return PreviewVar(value)


    def _cv_to_pixmap(self, frame_bgr):
        if frame_bgr is None:
            return None

        try:
            import cv2

            if len(frame_bgr.shape) == 2:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w

            qimg = QImage(
                frame_rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888,
            )

            return QPixmap.fromImage(qimg.copy())

        except Exception as exc:
            print(f"[Depth Preview] Failed to convert frame to pixmap: {exc}")
            return None


    def _set_preview_pair(self, pair_index: int):
        if not self._preview_pairs:
            self.preview_index_label.setText("Preview 0 / 0")
            return

        pair_index = max(0, min(pair_index, len(self._preview_pairs) - 1))
        self._preview_index = pair_index

        original_pixmap, depth_pixmap = self._preview_pairs[pair_index]

        if original_pixmap is not None:
            self.input_preview.setPixmap(
                original_pixmap.scaled(
                    self.input_preview.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        if depth_pixmap is not None:
            self.depth_preview.setPixmap(
                depth_pixmap.scaled(
                    self.depth_preview.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        self.preview_index_label.setText(
            f"Preview {pair_index + 1} / {len(self._preview_pairs)}"
        )

        self.prev_preview_btn.setEnabled(pair_index > 0)
        self.next_preview_btn.setEnabled(pair_index < len(self._preview_pairs) - 1)


    def _show_previous_preview_pair(self):
        if not self._preview_pairs:
            return

        self._set_preview_pair(self._preview_index - 1)


    def _show_next_preview_pair(self):
        if not self._preview_pairs:
            return

        self._set_preview_pair(self._preview_index + 1)


    def _on_depth_preview_samples_ready(self, pairs):
        self._preview_running = False
        self._preview_pairs = pairs or []
        self._preview_index = 0

        self.generate_preview_btn.setEnabled(True)

        if not self._preview_pairs:
            self.status_label.setText("No preview samples were generated.")
            self.preview_index_label.setText("Preview 0 / 0")
            return

        self.status_label.setText("Depth preview samples ready.")
        self._set_preview_pair(0)
        self._apply_processing_mode_ui()


    def _on_depth_preview_samples_failed(self, message):
        self._preview_running = False
        self.generate_preview_btn.setEnabled(True)
        self.prev_preview_btn.setEnabled(False)
        self.next_preview_btn.setEnabled(False)
        self.status_label.setText(f"Preview failed: {message}")
        self._apply_processing_mode_ui()

        QMessageBox.critical(
            self,
            "Depth Preview Failed",
            message,
        )


    def _run_depth_preview_worker(self):
        def _run():
            temp_root = None

            try:
                import cv2
                from core.render_depth import process_image

                video_path = self._state.input_video_path

                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    raise RuntimeError("Could not open input video for preview.")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                sample_indices = self._get_preview_sample_indices(total_frames)

                if not sample_indices:
                    raise RuntimeError("Could not find preview sample frames.")

                temp_root = tempfile.mkdtemp(prefix="vd3d_depth_preview_")
                frame_dir = os.path.join(temp_root, "frames")
                depth_dir = os.path.join(temp_root, "depth")
                os.makedirs(frame_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)

                preview_pairs = []

                for preview_num, frame_index in enumerate(sample_indices, start=1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame_bgr = cap.read()

                    if not ok or frame_bgr is None:
                        continue

                    frame_path = os.path.join(frame_dir, f"preview_{preview_num:02d}.png")
                    cv2.imwrite(frame_path, frame_bgr)

                    process_image(
                        file_path=frame_path,
                        colormap_var=self._preview_var(self._state.colormap),
                        invert_var=self._preview_var(self._state.invert_depth),
                        output_dir_var=self._preview_var(depth_dir),
                        inference_res_var=self._preview_var(self._state.inference_resolution),
                        input_label=None,
                        output_label=None,
                        status_label=None,
                        progress_bar=None,
                        folder=True,
                    )

                    base = os.path.splitext(os.path.basename(frame_path))[0]
                    expected_depth_path = os.path.join(depth_dir, f"{base}_depth.png")

                    if not os.path.exists(expected_depth_path):
                        # Fallback: find the newest image in the depth output folder.
                        depth_candidates = [
                            os.path.join(depth_dir, name)
                            for name in os.listdir(depth_dir)
                            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
                        ]

                        if depth_candidates:
                            expected_depth_path = max(depth_candidates, key=os.path.getmtime)

                    if not os.path.exists(expected_depth_path):
                        continue

                    depth_bgr = cv2.imread(expected_depth_path, cv2.IMREAD_COLOR)

                    original_pixmap = self._cv_to_pixmap(frame_bgr)
                    depth_pixmap = self._cv_to_pixmap(depth_bgr)

                    if original_pixmap is not None and depth_pixmap is not None:
                        preview_pairs.append((original_pixmap, depth_pixmap))

                cap.release()

                if not preview_pairs:
                    raise RuntimeError("No preview depth maps were created.")

                self.preview_samples_ready.emit(preview_pairs)

            except Exception as exc:
                self.preview_samples_failed.emit(str(exc))

            finally:
                if temp_root and os.path.isdir(temp_root):
                    try:
                        shutil.rmtree(temp_root, ignore_errors=True)
                    except Exception:
                        pass

        threading.Thread(target=_run, daemon=True).start()
    
    def _set_idle_state(self):
        self.process_btn.setEnabled(True)
        self.suspend_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

    def _set_running_state(self):
        self.process_btn.setEnabled(False)
        self.suspend_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def _set_suspended_state(self):
        self.process_btn.setEnabled(False)
        self.suspend_btn.setEnabled(False)
        self.resume_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)

    def _pulse_warmup(self):
        self._warmup_dots = (self._warmup_dots + 1) % 4
        dots = "." * (self._warmup_dots + 1)
        self.status_label.setText(f"{self._t('Loading model...')}{dots}")

    def _on_model_ready(self):
        self._warmup_timer.stop()
        if hasattr(self, '_ready_check'): self._ready_check.stop()
        if hasattr(self, '_warmup_fallback'): self._warmup_fallback.stop()
        self.process_btn.setEnabled(True)
        self.process_btn.setText(self._t("▶ Start Processing"))
        self.status_label.setText(self._t("Model ready."))

    def _check_pipe_ready(self):
        try:
            from core import render_depth
            if render_depth.pipe is not None:
                if hasattr(self, '_ready_check'): self._ready_check.stop()
                self._on_model_ready()
        except Exception:
            pass
