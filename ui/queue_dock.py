from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QSizePolicy,
)


class JobQueueDock(QWidget):
    def __init__(self):
        super().__init__()

        self._translator = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        self.title = QLabel("Job Queue")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(20)

        self.status_label = QLabel("Idle")

        self.telemetry_label = QLabel("")
        self.telemetry_label.setWordWrap(True)

        self.log_list = QListWidget()
        self.log_list.setVisible(False)
        self.log_list.setMaximumHeight(72)
        self.log_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setMaximumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addWidget(self.title)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.telemetry_label)
        layout.addWidget(self.log_list)

    def set_log_visible(self, visible: bool):
        self.log_list.setVisible(visible)

    def set_translator(self, translator):
        self._translator = translator
        self.refresh_labels()

    def _t(self, key: str) -> str:
        if callable(self._translator):
            return self._translator(key)
        return key

    def refresh_labels(self):
        self.title.setText(self._t("Job Queue"))

        idle_words = {
            "Idle",
            "Inactif",
            "Inactivo",
            "Bereit",
            "待機中",
            "アイドル",
            "空闲",
            "空閒",
            "閒置",
        }

        if self.status_label.text() in idle_words:
            self.status_label.setText(self._t("Idle"))

    def add_message(self, text: str):
        self.log_list.addItem(text)
        self.log_list.scrollToBottom()

    def set_progress(self, value: float):
        self.progress_bar.setValue(max(0, min(100, int(value))))

    def set_status(self, text: str):
        self.status_label.setText(text)

    def set_telemetry(self, text: str):
        self.telemetry_label.setText(text)

    def reset_progress(self):
        self.progress_bar.setValue(0)
        self.status_label.setText(self._t("Idle"))
        self.telemetry_label.setText("")
