from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QPushButton


class FilePickerRow(QWidget):
    browse_clicked = Signal()
    text_edited = Signal(str)

    def __init__(self, label_text: str, placeholder: str = ""):
        super().__init__()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.label = QLabel(label_text)
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        self.browse_button = QPushButton("Browse")

        layout.addWidget(self.label)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_button)

        self.browse_button.clicked.connect(self.browse_clicked.emit)
        self.line_edit.editingFinished.connect(self._emit_text)

    def _emit_text(self):
        self.text_edited.emit(self.line_edit.text())

    def text(self) -> str:
        return self.line_edit.text()

    def set_text(self, text: str):
        self.line_edit.setText(text)