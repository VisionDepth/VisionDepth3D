# ui/pages/depth_blender_page.py
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QScrollArea, QProgressBar, QGroupBox, QLineEdit,
    QRadioButton, QButtonGroup, QSlider, QMessageBox, QSplitter,
)

from ui.styles.page_theme import apply_unified_page_theme

import cv2
import numpy as np
import threading
import queue
import os


class DepthBlenderPage(QWidget):
    progress_updated = Signal(dict)
    preview_ready = Signal(object, int)
    preview_failed = Signal(str)

    PRESET_ITEMS = [
        ("Select Preset", None),
        ("Default (Balanced)", "Default (Balanced)"),
        ("Sharp Edges", "Sharp Edges"),
        ("Smooth Blend", "Smooth Blend"),
        ("Metric + Mono", "Metric + Mono"),
        ("High Contrast", "High Contrast"),
    ]

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._translation_map = []

        # State
        self.mode = "frames"
        self.overwrite_v2 = True
        self.v1_path = ""
        self.v2_path = ""
        self.out_path = ""
        self.out_w = ""
        self.out_h = ""
        self.use_gpu = True

        # Blend params
        self.white_strength = 1.0
        self.blur_k = 35
        self.clip_limit = 2.0
        self.tile_grid = 8
        self.bf_d = 12
        self.bf_sigmaColor = 75
        self.bf_sigmaSpace = 75

        # Preview
        self._preview_lock = threading.Lock()
        self._preview_thread = None
        self._preview_debounce = QTimer()
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(120)
        self._preview_debounce.timeout.connect(self._preview_now)
        self._preview_pixmap = None
        self.preview_index = 0
        self.preview_max = 0

        # Worker
        self._last_progress = 0
        self.qlog = queue.Queue()
        self.qprog = queue.Queue()
        self.stop_evt = threading.Event()
        self.worker = None

        self._build_ui()

        self.preview_ready.connect(self._apply_preview_image)
        self.preview_failed.connect(lambda msg: self._log(f"Preview error: {msg}"))

        self._start_poller()

    def _t(self, key: str) -> str:
        """
        Translation helper.
        Supports both new PySide6 label keys and older JSON keys.
        Avoids false missing-key warnings for English where value == key.
        """
        translator = getattr(self.controller, "t", None)
        if not callable(translator):
            return key

        # Try to inspect the loaded translation dictionary if available.
        translations = getattr(self.controller, "translations", None)

        # Some controllers store the language service inside another attribute.
        if translations is None:
            language_service = getattr(self.controller, "language_service", None)
            translations = getattr(language_service, "translations", None)

        aliases = {
            "Preview (scrubbable)": "Preview (scrubbable):",
            "Use GPU": "Use GPU (PyTorch CUDA)",
            "Output:": "Output path/file:",
            "W:": "Width:",
            "H:": "Height:",
            "(blank = keep source)": "(Leave blank to keep source)",

            "Blend Parameters": "Blend Parameters (preview live)",
            "Feather Blur": "Feather Blur (kernel)",
            "CLAHE Clip": "CLAHE Clip Limit",
            "CLAHE Tiles": "CLAHE Tile Grid",

            "Prev": "< Prev",
            "Next": "Next >",
            "Refresh Preview": "Preview Now",
        }

        # Best path: check actual dictionary membership.
        if isinstance(translations, dict):
            if key in translations:
                return translations[key]

            old_key = aliases.get(key)
            if old_key and old_key in translations:
                return translations[old_key]

            return key

        # Fallback path if we cannot access the dictionary directly.
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
        """
        Register widgets that use setText().
        QLabel, QPushButton, QCheckBox, QRadioButton.
        """
        self._translation_map.append((widget, key, "text"))
        widget.setText(self._t(key))

    def _register_title(self, widget, key: str):
        """
        Register widgets that use setTitle().
        QGroupBox.
        """
        self._translation_map.append((widget, key, "title"))
        widget.setTitle(self._t(key))

    def _refresh_preset_combo(self):
        current_data = self.preset_combo.currentData() if hasattr(self, "preset_combo") else None

        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()

        restore_index = 0
        for i, (label_key, preset_key) in enumerate(self.PRESET_ITEMS):
            self.preset_combo.addItem(self._t(label_key), preset_key)
            if preset_key == current_data:
                restore_index = i

        self.preset_combo.setCurrentIndex(restore_index)
        self.preset_combo.blockSignals(False)

    def refresh_labels(self):
        """
        Called by MainWindow when controller.language_changed fires.
        """
        for widget, key, widget_type in self._translation_map:
            try:
                if widget_type == "text":
                    widget.setText(self._t(key))
                elif widget_type == "title":
                    widget.setTitle(self._t(key))
            except RuntimeError:
                pass

        if hasattr(self, "preset_combo"):
            self._refresh_preset_combo()

        if hasattr(self, "frame_slider"):
            self.frame_label.setText(
                f"{self._t('Frame')}: {self.frame_slider.value()} / {self.frame_slider.maximum()}"
            )

    def apply_theme(self, theme: dict):
        self._active_theme = theme or {}
        apply_unified_page_theme(self, self._active_theme)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ── Right panel ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.preview_label = QLabel()
        self._register_text(self.preview_label, "Preview (scrubbable)")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setObjectName("PreviewPanel")

        self.preview_controls_host = QWidget()
        self.preview_controls_layout = QVBoxLayout(self.preview_controls_host)
        self.preview_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_controls_layout.setSpacing(10)

        right_layout.addWidget(self.preview_label, 1)
        right_layout.addWidget(self.preview_controls_host, 0)

        # ── Left panel ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(320)

        # Resizable left controls + right preview
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        self.main_splitter.addWidget(left_scroll)
        self.main_splitter.addWidget(right)

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)

        # Starting sizes: left panel, right preview area
        self.main_splitter.setSizes([440, 1200])

        root.addWidget(self.main_splitter, 1)

        # Mode
        mode_group = QGroupBox()
        self._register_title(mode_group, "Mode")
        mode_layout = QHBoxLayout(mode_group)

        self.rb_frames = QRadioButton()
        self.rb_videos = QRadioButton()
        self.rb_image = QRadioButton()

        self._register_text(self.rb_frames, "Folders (frames)")
        self._register_text(self.rb_videos, "Videos")
        self._register_text(self.rb_image, "Image")
        self.rb_frames.setChecked(True)
        mode_layout.addWidget(self.rb_frames)
        mode_layout.addWidget(self.rb_videos)
        mode_layout.addWidget(self.rb_image)
        left_layout.addWidget(mode_group)
        
        self.rb_frames.toggled.connect(lambda: self._set_mode("frames"))
        self.rb_videos.toggled.connect(lambda: self._set_mode("videos"))
        self.rb_image.toggled.connect(lambda: self._set_mode("image"))
        
        # Presets
        preset_group = QGroupBox()
        self._register_title(preset_group, "Presets")
        preset_layout = QHBoxLayout(preset_group)

        self.preset_combo = QComboBox()
        self._refresh_preset_combo()
        self.preset_combo.currentIndexChanged.connect(self._on_preset_combo_changed)
        preset_layout.addWidget(self.preset_combo)
        left_layout.addWidget(preset_group)

        # GPU
        self.gpu_check = QCheckBox()
        self._register_text(self.gpu_check, "Use GPU")
        self.gpu_check.setChecked(True)
        left_layout.addWidget(self.gpu_check)

        # Inputs
        inputs_group = QGroupBox()
        self._register_title(inputs_group, "Inputs")
        inputs_layout = QVBoxLayout(inputs_group)

        v1_row = QHBoxLayout()
        self.v1_label = QLabel()
        self._register_text(self.v1_label, "V1:")
        v1_row.addWidget(self.v1_label)
        self.v1_edit = QLineEdit()
        self.v1_edit.setReadOnly(True)
        v1_row.addWidget(self.v1_edit)
        v1_browse = QPushButton()
        self._register_text(v1_browse, "Browse")
        v1_browse.clicked.connect(self._browse_v1)
        v1_row.addWidget(v1_browse)
        inputs_layout.addLayout(v1_row)

        v2_row = QHBoxLayout()
        self.v2_label = QLabel()
        self._register_text(self.v2_label, "V2:")
        v2_row.addWidget(self.v2_label)
        self.v2_edit = QLineEdit()
        self.v2_edit.setReadOnly(True)
        v2_row.addWidget(self.v2_edit)
        v2_browse = QPushButton()
        self._register_text(v2_browse, "Browse")
        v2_browse.clicked.connect(self._browse_v2)
        v2_row.addWidget(v2_browse)
        inputs_layout.addLayout(v2_row)

        left_layout.addWidget(inputs_group)

        # Output
        out_group = QGroupBox()
        self._register_title(out_group, "Output")
        out_layout = QVBoxLayout(out_group)
        self.overwrite_check = QCheckBox()
        self._register_text(self.overwrite_check, "Overwrite V2 (frames mode only)")
        self.overwrite_check.setChecked(True)
        out_layout.addWidget(self.overwrite_check)

        out_row = QHBoxLayout()
        self.output_label = QLabel()
        self._register_text(self.output_label, "Output:")
        out_row.addWidget(self.output_label)
        self.out_edit = QLineEdit()
        self.out_edit.setReadOnly(True)
        out_row.addWidget(self.out_edit)
        out_browse = QPushButton()
        self._register_text(out_browse, "Browse")
        out_browse.clicked.connect(self._browse_out)
        out_row.addWidget(out_browse)
        out_layout.addLayout(out_row)

        left_layout.addWidget(out_group)

        # Final size
        size_group = QGroupBox()
        self._register_title(size_group, "Final Size (optional)")
        size_layout = QHBoxLayout(size_group)
        self.width_label = QLabel()
        self._register_text(self.width_label, "W:")
        size_layout.addWidget(self.width_label)
        self.w_edit = QLineEdit()
        self.w_edit.setFixedWidth(60)
        size_layout.addWidget(self.w_edit)
        self.height_label = QLabel()
        self._register_text(self.height_label, "H:")
        size_layout.addWidget(self.height_label)
        self.h_edit = QLineEdit()
        self.h_edit.setFixedWidth(60)
        size_layout.addWidget(self.h_edit)
        self.keep_source_label = QLabel()
        self.keep_source_label.setWordWrap(True)
        self._register_text(self.keep_source_label, "(blank = keep source)")
        size_layout.addWidget(self.keep_source_label, 1)
        left_layout.addWidget(size_group)

        # Blend parameters
        params_group = QGroupBox()
        self._register_title(params_group, "Blend Parameters")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(8)

        self.white_slider = self._add_slider_row(params_layout, "White Strength", 0.0, 2.0, self.white_strength, 0.01)
        self.blur_slider = self._add_slider_row(params_layout, "Feather Blur", 1, 99, self.blur_k, 1)
        self.clahe_slider = self._add_slider_row(params_layout, "CLAHE Clip", 0.5, 4.0, self.clip_limit, 0.1)
        self.tiles_slider = self._add_slider_row(params_layout, "CLAHE Tiles", 2, 32, self.tile_grid, 1)
        self.bfd_slider = self._add_slider_row(params_layout, "Bilateral d", 1, 25, self.bf_d, 1)
        self.sigmaC_slider = self._add_slider_row(params_layout, "Bilateral sigmaColor", 1, 200, self.bf_sigmaColor, 1)
        self.sigmaS_slider = self._add_slider_row(params_layout, "Bilateral sigmaSpace", 1, 200, self.bf_sigmaSpace, 1)

        left_layout.addWidget(params_group)

        # Scrubber
        self.scrub_group = QGroupBox()
        self._register_title(self.scrub_group, "Preview Frame")
        scrub_layout = QVBoxLayout(self.scrub_group)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_preview_frame_changed)
        scrub_layout.addWidget(self.frame_slider)
        self.frame_label = QLabel(f"{self._t('Frame')}: 0 / 0")
        scrub_layout.addWidget(self.frame_label)

        nav_row = QHBoxLayout()

        prev_btn = QPushButton()
        next_btn = QPushButton()
        self.preview_btn = QPushButton()

        self._register_text(prev_btn, "Prev")
        self._register_text(next_btn, "Next")
        self._register_text(self.preview_btn, "Refresh Preview")

        prev_btn.clicked.connect(lambda: self._nudge_preview(-1))
        next_btn.clicked.connect(lambda: self._nudge_preview(1))
        self.preview_btn.clicked.connect(self._preview_now)

        nav_row.addWidget(prev_btn)
        nav_row.addWidget(next_btn)
        nav_row.addWidget(self.preview_btn)

        scrub_layout.addLayout(nav_row)

        self.preview_controls_layout.addWidget(self.scrub_group)

        # Action buttons
        actions_group = QGroupBox()
        self._register_title(actions_group, "Actions")
        action_row = QHBoxLayout(actions_group)

        self.start_btn = QPushButton()
        self.stop_btn = QPushButton()

        self._register_text(self.start_btn, "Start Batch")
        self._register_text(self.stop_btn, "Stop")
        self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)

        action_row.addWidget(self.start_btn)
        action_row.addWidget(self.stop_btn)

        left_layout.addWidget(actions_group)

        left_layout.addStretch()
        self.apply_theme(getattr(self, "_active_theme", {}))


    def _add_slider_row(self, parent, label, mn, mx, default, step):
        row = QHBoxLayout()
        lbl = QLabel()
        self._register_text(lbl, label)
        lbl.setMinimumWidth(140)
        slider = QSlider(Qt.Horizontal)
        if isinstance(step, float):
            slider.setRange(int(mn * 100), int(mx * 100))
            slider.setValue(int(default * 100))
        else:
            slider.setRange(int(mn), int(mx))
            slider.setValue(int(default))
        slider.setMinimumHeight(28) 
        val_label = QLabel(str(default))
        val_label.setMinimumWidth(45)
        row.addWidget(lbl)
        row.addWidget(slider, 1)
        row.addWidget(val_label)
        parent.addLayout(row)

        # Update label on change, trigger preview debounce
        def _on_change(v):
            if isinstance(step, float):
                real = v / 100.0
            else:
                real = v
            val_label.setText(f"{real:.1f}" if isinstance(step, float) else str(int(real)))
            self._schedule_preview(120)

        slider.valueChanged.connect(_on_change)
        return slider

    def _schedule_preview(self, delay_ms=200):
        self._preview_debounce.start(delay_ms)

    def _preview_now(self):
        with self._preview_lock:
            if self._preview_thread and self._preview_thread.is_alive():
                return

            idx = self.frame_slider.value()
            self.preview_index = idx

            self._preview_thread = threading.Thread(
                target=self._compute_preview,
                args=(idx,),
                daemon=True
            )
            self._preview_thread.start()

    def _compute_preview(self, idx):
        try:
            v1p, v2p = self.v1_path, self.v2_path
            if not v1p or not v2p:
                return
            idx = max(0, min(int(idx), self.preview_max))

            if self.mode == "image":
                v1 = cv2.imread(v1p, cv2.IMREAD_GRAYSCALE)
                v2 = cv2.imread(v2p, cv2.IMREAD_GRAYSCALE)
                if v1 is None or v2 is None:
                    return
                if v1.shape != v2.shape:
                    v2 = cv2.resize(v2, (v1.shape[1], v1.shape[0]), interpolation=cv2.INTER_AREA)
                    

                    
            elif self.mode == "frames":
                v1_files = sorted([f for f in os.listdir(v1p) if f.lower().endswith(".png")])
                v2_files = sorted([f for f in os.listdir(v2p) if f.lower().endswith(".png")])
                if not v1_files or not v2_files:
                    return
                idx = max(0, min(idx, min(len(v1_files), len(v2_files)) - 1))
                v1 = cv2.imread(os.path.join(v1p, v1_files[idx]), cv2.IMREAD_GRAYSCALE)
                v2 = cv2.imread(os.path.join(v2p, v2_files[idx]), cv2.IMREAD_GRAYSCALE)
                if v1 is None or v2 is None:
                    return
                if v1.shape != v2.shape:
                    v2 = cv2.resize(v2, (v1.shape[1], v1.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                cap1 = cv2.VideoCapture(v1p)
                cap2 = cv2.VideoCapture(v2p)
                if not cap1.isOpened() or not cap2.isOpened():
                    return
                cap1.set(cv2.CAP_PROP_POS_FRAMES, idx)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok1, fr1 = cap1.read()
                ok2, fr2 = cap2.read()
                cap1.release()
                cap2.release()
                if not ok1 or not ok2:
                    return
                v1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
                v2 = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY)
                if v1.shape != v2.shape:
                    v2 = cv2.resize(v2, (v1.shape[1], v1.shape[0]), interpolation=cv2.INTER_AREA)

            from core.DB import lighten_beta, _put_label, _resize_max

            params = {
                "white_strength": self.white_strength,
                "blur_k": self.blur_k,
                "clip_limit": self.clip_limit,
                "tile_grid": (self.tile_grid, self.tile_grid),
                "d": self.bf_d,
                "sC": self.bf_sigmaColor,
                "sS": self.bf_sigmaSpace,
            }
            out = lighten_beta(v1, v2, use_gpu=self.gpu_check.isChecked(), **params)

            vis_v2 = cv2.cvtColor(v2, cv2.COLOR_GRAY2BGR)
            vis_out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            panel = np.hstack([_put_label(vis_v2, "V2 Base"), _put_label(vis_out, f"Blended Preview (idx {idx})")])
            panel = _resize_max(panel, max_w=840, max_h=520)

            rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
            self.preview_ready.emit(rgb.copy(), idx)
            
        except Exception as e:
            self.preview_failed.emit(str(e))

    def _apply_preview_image(self, rgb, idx):
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)

        self._preview_pixmap = pixmap
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        self.frame_label.setText(f"{self._t('Frame')}: {idx} / {self.frame_slider.maximum()}")

    def _set_mode(self, mode):
        if mode == self.mode:
            return
        self.mode = mode
        # Image mode: hide overwrite checkbox, scrubber, frame controls
        is_image = (mode == "image")
        self.overwrite_check.setVisible(not is_image)
        self.scrub_group.setVisible(not is_image)
        self.frame_slider.setVisible(not is_image)
        self.frame_label.setVisible(not is_image)
        self._update_preview_bounds()
        self._schedule_preview(0)

    def _browse_v1(self):
        if self.mode == "frames":
            p = QFileDialog.getExistingDirectory(self, "Select V1 frames folder")
        elif self.mode == "image":
            p, _ = QFileDialog.getOpenFileName(self, "Select V1 depth image", "",
                                                "Image (*.png *.jpg *.jpeg);;All (*.*)")
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Select V1 video", "",
                                                "Video (*.mp4 *.mov *.mkv *.avi);;All (*.*)")
        if p:
            self.v1_path = p
            self.v1_edit.setText(p)
            self._update_preview_bounds()
            self._schedule_preview(0)

    def _browse_v2(self):
        if self.mode == "frames":
            p = QFileDialog.getExistingDirectory(self, "Select V2 frames folder")
            
        elif self.mode == "image":
            p, _ = QFileDialog.getOpenFileName(self, "Select V1 depth image", "",
                                                "Image (*.png *.jpg *.jpeg);;All (*.*)")
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Select V2 video", "",
                                                "Video (*.mp4 *.mov *.mkv *.avi);;All (*.*)")
        if p:
            self.v2_path = p
            self.v2_edit.setText(p)
            self._update_preview_bounds()
            self._schedule_preview(0)

    def _browse_out(self):
        if self.mode == "frames" and not self.overwrite_check.isChecked():
            p = QFileDialog.getExistingDirectory(self, "Select output folder")
        elif self.mode == "image":
            p, _ = QFileDialog.getSaveFileName(self, "Save blended depth image", "",
                                                "PNG (*.png);;JPEG (*.jpg);;All (*.*)")
        else:
            p, _ = QFileDialog.getSaveFileName(self, "Save output video as", "", "MP4 (*.mp4);;All (*.*)")
        if p:
            self.out_path = p
            self.out_edit.setText(p)

    def _nudge_preview(self, delta):
        cur = self.frame_slider.value()
        mx = self.frame_slider.maximum()
        new = max(0, min(mx, cur + delta))
        if new != cur:
            self.frame_slider.setValue(new)
            self._schedule_preview(50)

    def _on_preview_frame_changed(self, value):
        self.preview_index = value
        self.frame_label.setText(f"{self._t('Frame')}: {value} / {self.frame_slider.maximum()}")
        self._schedule_preview(50)

    def _update_preview_bounds(self):
        mx = 0
        if self.mode == "frames":
            v1p, v2p = self.v1_path, self.v2_path
            if os.path.isdir(v1p) and os.path.isdir(v2p):
                n1 = len([f for f in os.listdir(v1p) if f.lower().endswith(".png")])
                n2 = len([f for f in os.listdir(v2p) if f.lower().endswith(".png")])
                mx = max(0, min(n1, n2) - 1)
        else:
            if os.path.isfile(self.v2_path):
                cap = cv2.VideoCapture(self.v2_path)
                if cap.isOpened():
                    mx = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) - 1)
                cap.release()
        self.preview_max = mx
        self.frame_slider.setMaximum(mx)
        self.frame_slider.setValue(min(self.frame_slider.value(), mx))
        self.frame_label.setText(f"{self._t('Frame')}: {self.frame_slider.value()} / {mx}")

    def _start(self):
        if not self.v1_path or not self.v2_path:
            QMessageBox.warning(self, "Missing paths", "Select both V1 and V2 paths.")
            return

        from core.DB import FramesWorker, VideosWorker

        params = {
            "white_strength": self.white_strength,
            "blur_k": self.blur_k,
            "clip_limit": self.clip_limit,
            "tile_grid": self.tile_grid,
            "bf_d": self.bf_d,
            "bf_sigmaColor": self.bf_sigmaColor,
            "bf_sigmaSpace": self.bf_sigmaSpace,
        }

        out_w = int(self.w_edit.text()) if self.w_edit.text().isdigit() else None
        out_h = int(self.h_edit.text()) if self.h_edit.text().isdigit() else None

        self.stop_evt.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)


        if self.mode == "image":
            v1 = cv2.imread(self.v1_path, cv2.IMREAD_GRAYSCALE)
            v2 = cv2.imread(self.v2_path, cv2.IMREAD_GRAYSCALE)
            if v1 is None or v2 is None:
                QMessageBox.warning(self, "Read error", "Could not read one of the images.")
                return
            if v1.shape != v2.shape:
                v2 = cv2.resize(v2, (v1.shape[1], v1.shape[0]), interpolation=cv2.INTER_AREA)

            from core.DB import lighten_beta
            blended = lighten_beta(v1, v2, use_gpu=self.gpu_check.isChecked(),
                                   white_strength=self.white_strength, blur_k=self.blur_k,
                                   clip_limit=self.clip_limit, tile_grid=(self.tile_grid, self.tile_grid),
                                   d=self.bf_d, sC=self.bf_sigmaColor, sS=self.bf_sigmaSpace)
            if out_w and out_h:
                blended = cv2.resize(blended, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            out_path = self.out_path or os.path.join(os.path.dirname(self.v2_path), "blended_depth.png")
            cv2.imwrite(out_path, blended)
            self._log(f"Saved blended image: {out_path}")
            return

        if self.mode == "frames":
            ow = self.overwrite_check.isChecked()
            out_mode = "overwrite_v2" if ow else "output_folder"
            out = self.out_path
            if not ow and not out:
                QMessageBox.warning(self, "Missing output", "Pick an output folder or enable Overwrite V2.")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return
            self.worker = FramesWorker(
                self.v1_path, self.v2_path, out_mode, out, out_w, out_h,
                self.qlog, self.qprog, self.stop_evt,
                use_gpu=self.gpu_check.isChecked(), params=params
            )
        else:
            out_file = self.out_path
            if not out_file:
                QMessageBox.warning(self, "Missing output", "Choose where to save the output video.")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return
            self.worker = VideosWorker(
                self.v1_path, self.v2_path, out_file, out_w, out_h,
                self.qlog, self.qprog, self.stop_evt,
                use_gpu=self.gpu_check.isChecked(), params=params
            )

        self.worker.start()
        

    def _on_preset_combo_changed(self, index):
        preset_key = self.preset_combo.itemData(index)
        if preset_key:
            self._apply_preset(preset_key)

    def _apply_preset(self, name):
        presets = {
            "Default (Balanced)": {
                "white_strength": 1.0, "blur_k": 35, "clip_limit": 2.0,
                "tile_grid": 8, "bf_d": 12, "bf_sigmaColor": 75, "bf_sigmaSpace": 75,
            },
            "Sharp Edges": {
                "white_strength": 1.2, "blur_k": 20, "clip_limit": 3.0,
                "tile_grid": 6, "bf_d": 8, "bf_sigmaColor": 50, "bf_sigmaSpace": 50,
            },
            "Smooth Blend": {
                "white_strength": 0.8, "blur_k": 55, "clip_limit": 1.5,
                "tile_grid": 12, "bf_d": 16, "bf_sigmaColor": 100, "bf_sigmaSpace": 100,
            },
            "Metric + Mono": {
                "white_strength": 0.7, "blur_k": 50, "clip_limit": 2.0,
                "tile_grid": 8, "bf_d": 14, "bf_sigmaColor": 80, "bf_sigmaSpace": 80,
            },
            "High Contrast": {
                "white_strength": 1.5, "blur_k": 25, "clip_limit": 4.0,
                "tile_grid": 4, "bf_d": 6, "bf_sigmaColor": 40, "bf_sigmaSpace": 40,
            },
        }
        p = presets.get(name)
        if not p:
            return

        # Apply to state
        for k, v in p.items():
            setattr(self, k, v)

        # Update sliders (suppress signals so they don't trigger preview spam)
        slider_map = {
            "white_strength": self.white_slider,
            "blur_k": self.blur_slider,
            "clip_limit": self.clahe_slider,
            "tile_grid": self.tiles_slider,
            "bf_d": self.bfd_slider,
            "bf_sigmaColor": self.sigmaC_slider,
            "bf_sigmaSpace": self.sigmaS_slider,
        }
        for k, slider in slider_map.items():
            slider.blockSignals(True)
            v = p[k]
            # Convert float sliders that use int scaling
            if k in ("white_strength", "clip_limit"):
                slider.setValue(int(v * 100))
            else:
                slider.setValue(int(v))
            slider.blockSignals(False)

        self._schedule_preview(0)

    def _stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_evt.set()
            self._log("Stopping requested...")

    def _log(self, msg):
        print(f"[Depth Blender] {msg}")
        self.progress_updated.emit({
            "progress": self._last_progress,
            "status_text": msg,
        })
        
    def _set_prog(self, done, total):
        self._last_progress = (done / max(total, 1)) * 100
        self.progress_updated.emit({
            "progress": self._last_progress,
            "status_text": f"Blending: {done}/{total}",
        })

    def _start_poller(self):
        self._poller = QTimer()
        self._poller.timeout.connect(self._poll)
        self._poller.start(100)

    def _poll(self):
        try:
            while True:
                msg = self.qlog.get_nowait()
                self._log(msg)
        except queue.Empty:
            pass
        try:
            while True:
                d, t = self.qprog.get_nowait()
                self._last_progress = (d / max(t, 1)) * 100
                self._log(f"Blending: {d}/{t}")
        except queue.Empty:
            pass
        if self.worker and not self.worker.is_alive():
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.start_btn.setText(self._t("Start Batch"))
        elif self.worker and self.worker.is_alive():
            self.start_btn.setText(self._t("Processing..."))
