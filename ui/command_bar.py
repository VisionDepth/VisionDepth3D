from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton


class CommandBar(QWidget):
    render_clicked = Signal()
    preview_clicked = Signal()

    def __init__(self):
        super().__init__()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("VisionDepth3D")
        title.setObjectName("AppTitle")

        self.preview_btn = QPushButton("Preview")
        self.render_btn = QPushButton("Render")

        self.preview_btn.clicked.connect(self.preview_clicked.emit)
        self.render_btn.clicked.connect(self.render_clicked.emit)

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.preview_btn)
        layout.addWidget(self.render_btn)