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

    def set_meta(self, text: str):
        self.meta_label.setText(text)