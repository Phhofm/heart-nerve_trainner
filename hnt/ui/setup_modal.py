"""Training dependency gate: install into the isolated venv with visible progress."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
)


class SetupWorker(QThread):
    line = Signal(str)
    done = Signal(bool, str)

    def __init__(self, paths: Path):
        super().__init__()
        self.paths = paths

    def run(self) -> None:
        try:
            from ..bootstrap import bootstrap
            bootstrap(self.paths, on_log=self.line.emit)
            self.done.emit(True, "")
        except Exception as e:  # noqa: BLE001
            self.done.emit(False, repr(e))


class SetupModal(QDialog):
    def __init__(self, paths: Path) -> None:
        super().__init__()
        self.paths = paths
        self.setWindowTitle("Training dependencies — SISR BODY Studio")
        self.resize(640, 420)

        root = QVBoxLayout(self)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        row = QHBoxLayout()
        self.back = QPushButton("Back")
        self.back.clicked.connect(self.reject)
        self.install = QPushButton("Install & test dependencies")
        self.install.setObjectName("primary")
        self.install.clicked.connect(self._install)
        row.addWidget(self.back)
        row.addWidget(self.install)
        root.addLayout(row)

        self.worker: SetupWorker | None = None

    def _install(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.log.clear()
        self.progress.setRange(0, 0)
        self.install.setEnabled(False)
        self.worker = SetupWorker(self.paths)
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _on_done(self, ok: bool, detail: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if ok else 0)
        self.install.setEnabled(True)
        if not ok:
            self.log.appendPlainText(detail)
