"""First-run quick tour — shown once, never without user-visible trigger in tests."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_should_show_and_mark_seen(tmp_path: Path):
    from hnt.ui.tour import mark_seen, should_show

    marker = tmp_path / "tour_seen.json"
    assert should_show(marker) is True
    mark_seen(marker)
    assert marker.is_file()
    assert should_show(marker) is False


def test_show_tour_mentions_steps(qapp, monkeypatch, tmp_path: Path):
    from hnt.ui import tour

    captured = {}
    monkeypatch.setattr(tour.QMessageBox, "information",
                        lambda *a, **k: captured.update(text=k.get("text") or a[-1]))
    tour.show_tour(parent=None, marker=tmp_path / "m.json")
    text = captured["text"].lower()
    assert "set up" in text and "hr folder" in text and "train" in text
    assert (tmp_path / "m.json").is_file()  # viewing the tour marks it seen


def test_main_window_has_tour_button(qapp):
    from hnt.ui.main_window import MainWindow

    win = MainWindow()
    assert win.tour_btn.text() == "Quick tour"
    win.close()
