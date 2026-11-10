"""Training dependency gate modal."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_setup_modal_starts_install_and_reports(qapp, monkeypatch, tmp_path: Path):
    from hnt.ui.setup_modal import SetupModal

    modal = SetupModal(tmp_path)
    assert modal.back.text() == "Back"
    assert modal.install.text() == "Install & test dependencies"

    class FakeWorker(QThread):
        line = Signal(str)
        done = Signal(bool, str)

        def __init__(self, paths):
            super().__init__()
            self.paths = paths

        def run(self) -> None:
            self.line.emit("fake install")
            self.done.emit(True, "")

    monkeypatch.setattr("hnt.ui.setup_modal.SetupWorker", FakeWorker)
    modal.install.click()
    modal.worker.wait(2000)
    QApplication.processEvents()
    assert modal.worker is not None
    assert "fake install" in modal.log.toPlainText()
    assert modal.install.isEnabled()
    modal.close()
