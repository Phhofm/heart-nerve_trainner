"""A/B viewer window — side-by-side panes with zoom/pan."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _img(tmp_path, name, color, size=(64, 64)):
    p = tmp_path / name
    Image.new("RGB", size, color).save(p)
    return p


def test_viewer_window_shows_input_and_output(qapp, tmp_path: Path):
    from hnt.ui.viewer import ViewerWindow

    a = _img(tmp_path, "a.png", (255, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))

    win = ViewerWindow(a, b)
    assert win.slider.value() == 50
    assert not win.left.item.pixmap().isNull()
    assert not win.right.item.pixmap().isNull()
    win.slider.setValue(25)
    assert win._split == 25
    win.close()


def test_viewer_set_pair_updates_images(qapp, tmp_path: Path):
    from hnt.ui.viewer import ViewerWindow

    a = _img(tmp_path, "a.png", (255, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))
    c = _img(tmp_path, "c.png", (0, 255, 0))
    d = _img(tmp_path, "d.png", (255, 255, 0))

    win = ViewerWindow(a, b)
    win.set_pair(c, d)
    assert win.input_path == c and win.output_path == d
    assert not win.left.item.pixmap().isNull() and not win.right.item.pixmap().isNull()
    win.close()


def test_viewer_swap_sides(qapp, tmp_path: Path):
    from hnt.ui.viewer import ViewerWindow

    a = _img(tmp_path, "a.png", (255, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))

    win = ViewerWindow(a, b)
    left_key = win.left.item.pixmap().cacheKey()
    win._swap()
    assert win.input_path == b and win.output_path == a
    assert win.left.item.pixmap().cacheKey() != left_key  # image actually changed
    win._swap()
    assert win.input_path == a and win.output_path == b
    win.close()


def test_viewer_accepts_image_drops(qapp, tmp_path: Path):
    from hnt.ui.viewer import ViewerWindow

    a = _img(tmp_path, "a.png", (9, 9, 9), (32, 32))
    win = ViewerWindow(a, a)
    assert win.acceptDrops()

    one = _img(tmp_path, "one.png", (1, 2, 3), (32, 32))
    win._add_dropped([one])
    assert win.input_path == one  # single image replaces the input side

    two = _img(tmp_path, "two.png", (4, 5, 6), (32, 32))
    three = _img(tmp_path, "three.png", (7, 8, 9), (32, 32))
    win._add_dropped([two, three])
    assert win.input_path == two and win.output_path == three

    junk = tmp_path / "note.txt"
    junk.write_text("x")
    win._add_dropped([junk])
    assert win.input_path == two  # non-image ignored
    win.close()


def test_viewer_zoom_clamps_and_syncs(qapp, tmp_path: Path):
    from hnt.ui.viewer import MAX_ZOOM, MIN_ZOOM, ViewerWindow

    a = _img(tmp_path, "a.png", (255, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))
    win = ViewerWindow(a, b)

    win._apply_zoom(2.0)
    assert win.left.transform() == win.right.transform()
    assert win._zoom_factor == 2.0

    win._apply_zoom(100.0)
    assert win._zoom_factor == MAX_ZOOM
    win._apply_zoom(0.0001)
    assert win._zoom_factor == MIN_ZOOM
    assert win.left.transform() == win.right.transform()
    win.close()


def test_viewer_reset_view_refits(qapp, tmp_path: Path):
    from hnt.ui.viewer import ViewerWindow

    a = _img(tmp_path, "a.png", (255, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))
    win = ViewerWindow(a, b)

    win._apply_zoom(4.0)
    assert win._zoom_factor == 4.0
    win.reset_view()
    assert win._zoom_factor == 1.0
    assert win.left.transform() == win.right.transform()
    win.close()
