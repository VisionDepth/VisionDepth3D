# services/live_service.py

import threading
import traceback
from argparse import Namespace

from PySide6.QtCore import QObject, Signal
from core.vd3d_live import run_live

class LiveWorker(QObject):
    started = Signal()
    finished = Signal()
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, args: Namespace, stop_event: threading.Event):
        super().__init__()
        self.args = args
        self.stop_event = stop_event

    def run(self):
        try:
            from core.vd3d_live import run_live

            self.started.emit()
            self.status.emit("Starting Live 3D...")
            run_live(self.args, external_stop=self.stop_event)
            self.status.emit("Live 3D stopped.")
            self.finished.emit()

        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))


class LiveService(QObject):
    started = Signal()
    finished = Signal()
    failed = Signal(str)
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.worker = None
        self.stop_event = None

    def is_running(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def start(self, args: Namespace):
        if self.is_running():
            self.status.emit("Live 3D is already running.")
            return

        self.stop_event = threading.Event()
        self.worker = LiveWorker(args, self.stop_event)

        self.worker.started.connect(self.started.emit)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.status.connect(self.status.emit)

        self.worker_thread = threading.Thread(
            target=self.worker.run,
            daemon=True,
        )
        self.worker_thread.start()

    def stop(self):
        if self.stop_event is not None:
            self.status.emit("Stopping Live 3D...")
            self.stop_event.set()

    def _on_finished(self):
        self.worker_thread = None
        self.worker = None
        self.stop_event = None
        self.finished.emit()

    def _on_failed(self, error: str):
        self.worker_thread = None
        self.worker = None
        self.stop_event = None
        self.failed.emit(error)