from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QSizePolicy,
    QPushButton,
    QApplication,
)


class JobQueueDock(QWidget):
    def __init__(self):
        super().__init__()

        self._translator = None
        self.max_log_lines = 500

        self._status_key = "Idle"
        self._status_is_translatable = True

        self._telemetry_key = ""
        self._telemetry_is_translatable = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        header_row = QHBoxLayout()

        self.title = QLabel("Job Queue")

        self.copy_btn = QPushButton("Copy Log")
        self.clear_btn = QPushButton("Clear Log")
        self.copy_btn.setFixedHeight(26)
        self.clear_btn.setFixedHeight(26)

        self.copy_btn.clicked.connect(self.copy_log)
        self.clear_btn.clicked.connect(self.clear_log)

        header_row.addWidget(self.title)
        header_row.addStretch()
        header_row.addWidget(self.copy_btn)
        header_row.addWidget(self.clear_btn)

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

        # Bigger terminal area when debug is visible.
        self.log_list.setMinimumHeight(150)
        self.log_list.setMaximumHeight(260)
        self.log_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Let the dock grow instead of locking it tiny.
        self.setMinimumHeight(150)
        self.setMaximumHeight(360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout.addLayout(header_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.telemetry_label)
        layout.addWidget(self.log_list, 1)

    def set_log_visible(self, visible: bool):
        self.log_list.setVisible(visible)
        self.copy_btn.setVisible(visible)
        self.clear_btn.setVisible(visible)

    def set_translator(self, translator):
        self._translator = translator
        self.refresh_labels()

    def _t(self, key: str) -> str:
        if callable(self._translator):
            return self._translator(key)
        return key

    def refresh_labels(self):
        self.title.setText(self._t("Job Queue"))
        self.copy_btn.setText(self._t("Copy Log"))
        self.clear_btn.setText(self._t("Clear Log"))

        if self._status_is_translatable:
            self.status_label.setText(self._t(self._status_key))

        if self._telemetry_is_translatable:
            self.telemetry_label.setText(self._t(self._telemetry_key))
            
    def add_message(self, text: str):
        self.log_list.addItem(str(text))

        # Keep only the newest log lines so per-frame debug output
        # does not make the queue dock or Copy Log massive.
        while self.log_list.count() > self.max_log_lines:
            item = self.log_list.takeItem(0)
            del item

        self.log_list.scrollToBottom()
        
    def copy_log(self):
        total = self.log_list.count()
        start = max(0, total - self.max_log_lines)

        lines = []
        if total > self.max_log_lines:
            lines.append(
                f"[Copied last {self.max_log_lines} log lines out of {total}. "
                f"Older lines were not included.]"
            )

        for i in range(start, total):
            item = self.log_list.item(i)
            if item:
                lines.append(item.text())

        QApplication.clipboard().setText("\n".join(lines))

    def clear_log(self):
        self.log_list.clear()

    def set_progress(self, value: float):
        self.progress_bar.setValue(max(0, min(100, int(value))))

    def set_status(self, text: str, translate: bool = False):
        """
        Sets the status label.

        translate=False:
            Use for dynamic strings like progress, FPS, ETA, or errors.

        translate=True:
            Use for fixed language keys like Idle, Render started..., etc.
        """
        self._status_key = str(text)
        self._status_is_translatable = bool(translate)

        if translate:
            self.status_label.setText(self._t(self._status_key))
        else:
            self.status_label.setText(self._status_key)

    def set_status_key(self, key: str):
        self.set_status(key, translate=True)

    def set_telemetry(self, text: str, translate: bool = False):
        self._telemetry_key = str(text)
        self._telemetry_is_translatable = bool(translate)

        if translate:
            self.telemetry_label.setText(self._t(self._telemetry_key))
        else:
            self.telemetry_label.setText(self._telemetry_key)
            
    def reset_progress(self):
        self.progress_bar.setValue(0)
        self.set_status_key("Idle")
        self.set_telemetry("")
