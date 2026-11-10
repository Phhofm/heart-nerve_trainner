"""A/B viewer window."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_viewer_window_shows_input_and_output(qapp, tmp_path: Path):
    from hnt.ui.viewer import ViewerWindow

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(a)
    Image.new("RGB", (64, 64), (0, 0, 255)).save(b)

    win = ViewerWindow(a, b)
    assert win.slider.value() == 50
    assert not win.left.pixmap().isNull()
    assert not win.right.pixmap().isNull()
    win.slider.setValue(25)
    assert win._split == 25
    win.close()
