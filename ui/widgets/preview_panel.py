from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QSizePolicy

import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import cv2

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    try:
        cv2.setLogLevel(2)
    except Exception:
        pass


class PreviewPanel(QWidget):
    def __init__(self):
        super().__init__()

        self._translator = None
        self._placeholder_key = "Video / Depth preview will appear here"
        self._meta_key = "No media loaded"
        self._meta_is_custom = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title = QLabel("Preview")

        self.frame = QFrame()
        self.frame.setMinimumHeight(420)
        self.frame.setObjectName("PreviewFrame")

        self.placeholder = QLabel("Video / Depth preview will appear here")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setScaledContents(False)
        self.placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.placeholder.setMinimumSize(320, 180)

        inner = QVBoxLayout(self.frame)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.addWidget(self.placeholder)

        self.meta_label = QLabel("No media loaded")
        self.meta_label.setWordWrap(True)

        layout.addWidget(self.title)
        layout.addWidget(self.frame, 1)
        layout.addWidget(self.meta_label)

    def _t(self, key: str) -> str:
        if callable(self._translator):
            try:
                return self._translator(key)
            except Exception:
                return key

        return key

    def set_translator(self, translator):
        """
        Gives PreviewPanel access to the page translation helper.
        Example:
            self.preview_panel.set_translator(self._t)
        """
        self._translator = translator
        self.refresh_labels()

    def _has_preview_image(self) -> bool:
        pixmap = self.placeholder.pixmap()
        return pixmap is not None and not pixmap.isNull()

    def refresh_labels(self):
        self.title.setText(self._t("Preview"))

        # Do not call setText() when the preview QLabel is showing an image.
        # QLabel.setText() clears the pixmap, which blanks the loaded preview
        # during language changes.
        if not self._has_preview_image():
            self.placeholder.setText(self._t(self._placeholder_key))

        if not self._meta_is_custom:
            self.meta_label.setText(self._t(self._meta_key))

    def set_placeholder(self, key: str):
        self._placeholder_key = str(key)

        # Only update placeholder text when no preview image is currently loaded.
        if not self._has_preview_image():
            self.placeholder.setText(self._t(self._placeholder_key))

    def set_meta(self, text: str, translate: bool = False):
        if translate:
            self._meta_key = str(text)
            self._meta_is_custom = False
            self.meta_label.setText(self._t(self._meta_key))
        else:
            self._meta_is_custom = True
            self.meta_label.setText(str(text))

    def reset_meta(self):
        self._meta_key = "No media loaded"
        self._meta_is_custom = False
        self.meta_label.setText(self._t(self._meta_key))
