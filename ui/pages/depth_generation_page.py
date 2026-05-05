import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QCheckBox, QSpinBox, QMessageBox,
    QFrame, QGroupBox, QLineEdit, QScrollArea,
)
from ui.widgets.parameter_card import ParameterCard


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
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._state = controller.depth_state
        self._translation_map = []

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
        top_row.addWidget(left_scroll)

        # Right: Previews side-by-side
        preview_widget = QWidget()
        preview_layout = QHBoxLayout(preview_widget)
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
        mode_group = QGroupBox()
        self._register_title(mode_group, "Processing Mode")
        depth_inner.addWidget(self.depth_preview)

        preview_layout.addWidget(input_group, 1)
        preview_layout.addWidget(depth_group, 1)
        top_row.addWidget(preview_widget, 1)

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

    # ── All the same methods from your existing file ──
    def _bind_events(self):
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
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
        self.suspend_btn.clicked.connect(self._suspend)
        self.resume_btn.clicked.connect(self._resume)
        self.cancel_btn.clicked.connect(self._cancel)
        self.controller.depth_started.connect(self._on_depth_started)
        self.controller.depth_finished.connect(self._on_depth_finished)
        self.controller.depth_finished.connect(self._on_depth_image_done)
        self.controller.depth_failed.connect(self._on_depth_failed)
        self.controller.depth_cancelled.connect(self._on_depth_cancelled)

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

    def _browse_input_video(self):
        mode = self.mode_combo.currentData() or self.mode_combo.currentText()
        if mode in ("Image", "Image Folder"):
            path, _ = QFileDialog.getOpenFileName(self, "Select Input Image", "",
                "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*.*)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Input Video", "",
                "Video Files (*.mp4 *.mkv *.avi *.mov);;All Files (*.*)")
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

    def _on_depth_started(self): self._set_running_state()

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