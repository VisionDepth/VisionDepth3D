# ui/pages/live_3d_page.py

from argparse import Namespace
try:
    from core.render_depth import load_supported_models
except Exception:
    load_supported_models = None

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QGridLayout,
    QMessageBox,
    QScrollArea,
    QFrame,
    QSizePolicy,
)


class Live3DPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._translation_map = []

        self._build_ui()
        self._bind_events()

    def _t(self, key: str) -> str:
        translator = getattr(self.controller, "t", None)
        if callable(translator):
            return translator(key)
        return key

    def _register_text(self, widget, key: str):
        self._translation_map.append((widget, key, "text"))
        widget.setText(self._t(key))

    def _register_title(self, widget, key: str):
        self._translation_map.append((widget, key, "title"))
        widget.setTitle(self._t(key))

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

    def _double_spin(
        self,
        value: float,
        min_val: float,
        max_val: float,
        decimals: int = 2,
        step: float = 0.10,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setFixedWidth(150)
        spin.setMinimumHeight(34)
        spin.setButtonSymbols(QDoubleSpinBox.UpDownArrows)
        return spin

    def _load_depth_models_into_combo(self):
        self.model_combo.clear()

        fallback_models = {
            "Depth Anything v2 Small": "depth-anything/Depth-Anything-V2-Small-hf",
            "Depth Anything v2 Base": "depth-anything/Depth-Anything-V2-Base-hf",
            "Depth Anything v2 Large": "depth-anything/Depth-Anything-V2-Large-hf",
        }

        try:
            models = load_supported_models() if callable(load_supported_models) else fallback_models
        except Exception as exc:
            print(f"[Live3DPage] Failed to load depth model list: {exc}")
            models = fallback_models

        preferred = "Depth Anything v2 Small"
        preferred_index = 0

        for i, (display_name, checkpoint) in enumerate(models.items()):
            if display_name.strip() == "-- Select Model --":
                continue

            self.model_combo.addItem(display_name, checkpoint)

            if display_name == preferred:
                preferred_index = self.model_combo.count() - 1

        self.model_combo.setCurrentIndex(preferred_index)

    def _group(self, key: str) -> QGroupBox:
        group = QGroupBox()
        self._register_title(group, key)
        return group

    def refresh_labels(self):
        for widget, key, widget_type in self._translation_map:
            try:
                if widget_type == "text":
                    widget.setText(self._t(key))
                elif widget_type == "title":
                    widget.setTitle(self._t(key))
            except RuntimeError:
                pass

    def _build_ui(self):
        self.setObjectName("Live3DPage")

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        title = QLabel()
        self._register_text(title, "Live 3D")
        title.setObjectName("PageTitle")

        subtitle = QLabel()
        self._register_text(
            subtitle,
            "Real-time capture to depth estimation to VD3D stereo preview."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)

        root.addWidget(title)
        root.addWidget(subtitle)

        body_scroll = QScrollArea()
        body_scroll.setWidgetResizable(True)
        body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        body_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body_scroll.setFrameShape(QFrame.NoFrame)

        body_widget = QWidget()
        body_widget.setMinimumWidth(1160)

        layout = QHBoxLayout(body_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        # Left controls
        left_panel = QWidget()
        left_panel.setMinimumWidth(360)
        left_panel.setMaximumWidth(430)
        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)

        layout.addWidget(left_panel)

        capture_group = self._group("Capture Source")
        capture_grid = QGridLayout(capture_group)
        capture_grid.setSpacing(8)

        self.source_combo = QComboBox()
        self.source_combo.addItem("Camera / Capture Card", "device")
        self.source_combo.addItem("Screen 1", "screen:1")
        self.source_combo.addItem("Screen 2", "screen:2")

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["dshow", "msmf", "any"])

        self.device_spin = QSpinBox()
        self.device_spin.setRange(0, 16)
        self.device_spin.setValue(0)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 7680)
        self.width_spin.setValue(1280)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 4320)
        self.height_spin.setValue(720)

        self.capture_fps_spin = QSpinBox()
        self.capture_fps_spin.setRange(1, 240)
        self.capture_fps_spin.setValue(30)

        capture_grid.addWidget(self._label("Source:"), 0, 0)
        capture_grid.addWidget(self.source_combo, 0, 1)

        capture_grid.addWidget(self._label("Backend:"), 1, 0)
        capture_grid.addWidget(self.backend_combo, 1, 1)

        capture_grid.addWidget(self._label("Device Index:"), 2, 0)
        capture_grid.addWidget(self.device_spin, 2, 1)

        capture_grid.addWidget(self._label("Width:"), 3, 0)
        capture_grid.addWidget(self.width_spin, 3, 1)

        capture_grid.addWidget(self._label("Height:"), 4, 0)
        capture_grid.addWidget(self.height_spin, 4, 1)

        capture_grid.addWidget(self._label("Capture FPS:"), 5, 0)
        capture_grid.addWidget(self.capture_fps_spin, 5, 1)

        left.addWidget(capture_group)

        depth_group = self._group("Depth Model")
        depth_grid = QGridLayout(depth_group)
        depth_grid.setSpacing(8)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(280)
        self._load_depth_models_into_combo()

        self.infer_w_spin = QSpinBox()
        self.infer_w_spin.setRange(256, 1024)
        self.infer_w_spin.setValue(384)

        self.infer_h_spin = QSpinBox()
        self.infer_h_spin.setRange(256, 1024)
        self.infer_h_spin.setValue(384)

        self.depth_fps_spin = QSpinBox()
        self.depth_fps_spin.setRange(1, 60)
        self.depth_fps_spin.setValue(6)

        self.fp16_check = self._checkbox("Use FP16")
        self.fp16_check.setChecked(True)

        self.smooth_check = self._checkbox("Smooth Depth")
        self.smooth_check.setChecked(True)

        depth_grid.addWidget(self._label("Model:"), 0, 0)
        depth_grid.addWidget(self.model_combo, 0, 1)

        depth_grid.addWidget(self._label("Inference W:"), 1, 0)
        depth_grid.addWidget(self.infer_w_spin, 1, 1)

        depth_grid.addWidget(self._label("Inference H:"), 2, 0)
        depth_grid.addWidget(self.infer_h_spin, 2, 1)

        depth_grid.addWidget(self._label("Depth FPS:"), 3, 0)
        depth_grid.addWidget(self.depth_fps_spin, 3, 1)

        depth_grid.addWidget(self.fp16_check, 4, 0, 1, 2)
        depth_grid.addWidget(self.smooth_check, 5, 0, 1, 2)

        left.addWidget(depth_group)

        # Center controls
        center_panel = QWidget()
        center_panel.setMinimumWidth(430)
        center_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        center = QVBoxLayout(center_panel)
        center.setContentsMargins(0, 0, 0, 0)
        center.setSpacing(12)

        layout.addWidget(center_panel, 1)

        stereo_group = self._group("VD3D Live Stereo")
        stereo_grid = QGridLayout(stereo_group)
        stereo_grid.setSpacing(8)

        self.capture_fps_spin.setValue(30)
        self.infer_w_spin.setValue(384)
        self.infer_h_spin.setValue(384)
        self.depth_fps_spin.setValue(6)
        self.fg_shift = self._double_spin(-12.0, -50.0, 50.0, decimals=2, step=0.10)
        self.mg_shift = self._double_spin(-2.0, -50.0, 50.0, decimals=2, step=0.10)
        self.bg_shift = self._double_spin(4.0, -50.0, 50.0, decimals=2, step=0.10)
        self.max_shift = self._double_spin(0.035, 0.001, 0.20, decimals=3, step=0.001)
        self.parallax_balance = self._double_spin(1.00, 0.0, 2.0, decimals=2, step=0.05)
        self.depth_pop_gamma = self._double_spin(0.85, 0.10, 3.0, decimals=2, step=0.05)

        self.subject_tracking_check = self._checkbox("Use Subject Tracking")
        self.subject_tracking_check.setChecked(True)

        self.edge_masking_check = self._checkbox("Enable Edge Masking")
        self.edge_masking_check.setChecked(True)

        self.feathering_check = self._checkbox("Enable Feathering")
        self.feathering_check.setChecked(True)

        self.dynamic_convergence_check = self._checkbox("Enable Dynamic Convergence")
        self.dynamic_convergence_check.setChecked(True)

        self.floating_window_check = self._checkbox("Enable Floating Window (DFW)")
        self.floating_window_check.setChecked(True)

        stereo_grid.addWidget(self._label("Foreground Shift"), 0, 0)
        stereo_grid.addWidget(self.fg_shift, 0, 1)

        stereo_grid.addWidget(self._label("Midground Shift"), 1, 0)
        stereo_grid.addWidget(self.mg_shift, 1, 1)

        stereo_grid.addWidget(self._label("Background Shift"), 2, 0)
        stereo_grid.addWidget(self.bg_shift, 2, 1)

        stereo_grid.addWidget(self._label("Max Pixel Shift"), 3, 0)
        stereo_grid.addWidget(self.max_shift, 3, 1)

        stereo_grid.addWidget(self._label("Parallax Balance"), 4, 0)
        stereo_grid.addWidget(self.parallax_balance, 4, 1)

        stereo_grid.addWidget(self._label("Depth Pop Gamma"), 5, 0)
        stereo_grid.addWidget(self.depth_pop_gamma, 5, 1)

        stereo_grid.addWidget(self.subject_tracking_check, 6, 0, 1, 2)
        stereo_grid.addWidget(self.edge_masking_check, 7, 0, 1, 2)
        stereo_grid.addWidget(self.feathering_check, 8, 0, 1, 2)
        stereo_grid.addWidget(self.dynamic_convergence_check, 9, 0, 1, 2)
        stereo_grid.addWidget(self.floating_window_check, 10, 0, 1, 2)

        center.addWidget(stereo_group)

        preview_group = self._group("Preview / Output")
        preview_grid = QGridLayout(preview_group)
        preview_grid.setSpacing(8)

        self.preview_w_spin = QSpinBox()
        self.preview_w_spin.setRange(320, 3840)
        self.preview_w_spin.setValue(960)

        self.preview_h_spin = QSpinBox()
        self.preview_h_spin.setRange(240, 2160)
        self.preview_h_spin.setValue(540)

        self.no_preview_check = self._checkbox("Disable Preview Window")
        self.sbs_check = self._checkbox("Start in SBS Mode")
        self.sbs_check.setChecked(True)

        self.http_stream_edit = QLineEdit()
        self.http_stream_edit.setPlaceholderText("Optional, example: 127.0.0.1:8080")

        preview_grid.addWidget(self._label("Preview Width:"), 0, 0)
        preview_grid.addWidget(self.preview_w_spin, 0, 1)

        preview_grid.addWidget(self._label("Preview Height:"), 1, 0)
        preview_grid.addWidget(self.preview_h_spin, 1, 1)

        preview_grid.addWidget(self.sbs_check, 2, 0, 1, 2)
        preview_grid.addWidget(self.no_preview_check, 3, 0, 1, 2)

        preview_grid.addWidget(self._label("HTTP Stream:"), 4, 0)
        preview_grid.addWidget(self.http_stream_edit, 4, 1)

        center.addWidget(preview_group)
        center.addStretch()

        # Right status/actions
        right_panel = QWidget()
        right_panel.setMinimumWidth(260)
        right_panel.setMaximumWidth(330)
        right_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)

        layout.addWidget(right_panel)

        status_group = self._group("Live Status")
        status_layout = QVBoxLayout(status_group)

        self.status_label = QLabel(self._t("Ready"))
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        self.start_btn = self._button("Start Live 3D")
        self.start_btn.setMinimumHeight(42)

        self.stop_btn = self._button("Stop Live 3D")
        self.stop_btn.setMinimumHeight(42)
        self.stop_btn.setEnabled(False)

        status_layout.addWidget(self.start_btn)
        status_layout.addWidget(self.stop_btn)

        right.addWidget(status_group)
        right.addStretch()
        
        body_scroll.setWidget(body_widget)
        root.addWidget(body_scroll, 1)

        self.setStyleSheet("""
            QLabel#PageTitle {
                font-size: 22px;
                font-weight: 800;
            }
            QLabel#PageSubtitle {
                color: #9aa8b8;
            }
            QGroupBox {
                font-weight: 800;
            }
        """)
        


    def _bind_events(self):
        self.start_btn.clicked.connect(self._start_live)
        self.stop_btn.clicked.connect(self._stop_live)

        if hasattr(self.controller, "live_started"):
            self.controller.live_started.connect(self._on_live_started)
        if hasattr(self.controller, "live_finished"):
            self.controller.live_finished.connect(self._on_live_finished)
        if hasattr(self.controller, "live_failed"):
            self.controller.live_failed.connect(self._on_live_failed)
        if hasattr(self.controller, "live_status"):
            self.controller.live_status.connect(self._on_live_status)

    def _build_args(self) -> Namespace:
        return Namespace(
            source=self.source_combo.currentData(),
            device_index=self.device_spin.value(),
            backend=self.backend_combo.currentText(),
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            fps=30,
            fourcc="",
            force_bgr_swap=False,
            no_capture_swap=False,
            crop=None,
            capture_fps=self.capture_fps_spin.value(),

            model=self.model_combo.currentText(),
            model_checkpoint=self.model_combo.currentData(),
            fp16=self.fp16_check.isChecked(),
            invert_depth=False,
            depth_bit_depth=16,
            infer_w=self.infer_w_spin.value(),
            infer_h=self.infer_h_spin.value(),
            depth_fps=self.depth_fps_spin.value(),
            smooth=self.smooth_check.isChecked(),
            ema=0.35,

            sbs=self.sbs_check.isChecked(),
            fg_shift=self.fg_shift.value(),
            mg_shift=self.mg_shift.value(),
            bg_shift=self.bg_shift.value(),
            pixelshift_rgb=True,

            use_subject_tracking=self.subject_tracking_check.isChecked(),
            enable_floating_window=self.floating_window_check.isChecked(),
            enable_edge_masking=self.edge_masking_check.isChecked(),
            enable_feathering=self.feathering_check.isChecked(),
            enable_dynamic_convergence=self.dynamic_convergence_check.isChecked(),

            max_pixel_shift_percent=self.max_shift.value(),
            parallax_balance=self.parallax_balance.value(),
            zero_parallax_strength=0.0,
            convergence_strength=0.0,
            dof_strength=0.0,

            depth_pop_gamma=self.depth_pop_gamma.value(),
            depth_pop_mid=0.50,
            depth_stretch_lo=0.05,
            depth_stretch_hi=0.95,
            fg_pop_multiplier=1.08,
            bg_push_multiplier=1.06,
            subject_lock_strength=0.25,
            disable_shift_ema=False,

            no_preview=self.no_preview_check.isChecked(),
            force_preview=True,
            mask_preview=False,
            preview_x=60,
            preview_y=60,
            preview_w=self.preview_w_spin.value(),
            preview_h=self.preview_h_spin.value(),

            http_stream=self.http_stream_edit.text().strip(),
            audio_device="",
            audio_delay_ms=0,
            virtualcam=False,
            vcam_fps=30,

            diag=False,
        )

    def _start_live(self):
        try:
            args = self._build_args()
            self.controller.start_live_3d(args)
        except Exception as exc:
            QMessageBox.critical(self, self._t("Live 3D Failed"), str(exc))

    def _stop_live(self):
        self.controller.stop_live_3d()

    def _on_live_started(self):
        self.status_label.setText(self._t("Live 3D running."))
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _on_live_finished(self):
        self.status_label.setText(self._t("Live 3D stopped."))
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_live_failed(self, error: str):
        self.status_label.setText(f"{self._t('Live 3D failed:')} {error}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_live_status(self, text: str):
        self.status_label.setText(self._t(text))