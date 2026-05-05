from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSizePolicy


class NavigationRail(QWidget):
    page_selected = Signal(str)

    def __init__(self):
        super().__init__()

        self.setObjectName("NavigationRail")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.buttons = {}
        pages = [
            ("stereo", "3D Generator"),
            ("depth", "Depth Engine"),
            ("blend", "Depth Blender"),
            ("frame", "FPS/Upscale"),
        ]

        for key, label in pages:
            btn = QPushButton(label)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda checked=False, k=key: self.page_selected.emit(k))
            layout.addWidget(btn)
            self.buttons[key] = btn

        layout.addStretch()