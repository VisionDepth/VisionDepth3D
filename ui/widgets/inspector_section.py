from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class InspectorSection(QWidget):
    def __init__(self, title: str):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.title = QLabel(title)
        self.title.setObjectName("InspectorTitle")

        layout.addWidget(self.title)