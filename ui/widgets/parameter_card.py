from PySide6.QtWidgets import QGroupBox, QVBoxLayout


class ParameterCard(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        self.inner_layout = QVBoxLayout(self)
        self.inner_layout.setContentsMargins(12, 12, 12, 12)
        self.inner_layout.setSpacing(10)