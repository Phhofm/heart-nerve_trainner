"""Offscreen GUI smoke tests. Skipped when PySide6 / pyqtgraph are not installed.

Run with the app venv:  QT_QPA_PLATFORM=offscreen ../venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_metrics_chart_collects_points(qapp):
    from hnt.ui.chart import MetricsChart

    chart = MetricsChart()
    chart.add_point("psnr", 2500, 28.0)
    chart.add_point("psnr", 5000, 29.5)
    chart.add_point("lpips", 2500, 0.2)
    assert chart._xs["psnr"] == [2500, 5000]
    assert chart._ys["psnr"] == [28.0, 29.5]
    assert chart._ys["lpips"] == [0.2]
    chart.clear()
    assert chart._xs == {}


def test_main_window_constructs_and_plots(qapp):
    from hnt.paths import find_repo
    from hnt.ui.main_window import MainWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = MainWindow()
    win.chart.add_point("ssim", 100, 0.85)
    assert win.chart._ys["ssim"] == [0.85]
    win.close()


def test_defaults_nerve_2x_fidelity_are_recommended(qapp):
    from hnt.paths import find_repo
    from hnt.ui.main_window import MainWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = MainWindow()
    assert win.model.currentData() == "NERVE"
    assert win.scale.currentData() == 2
    assert win.profile.currentData() == "FIDELITY"
    assert win.data_mode.currentData() == "OTF"
    assert win.model.property("recommended") is True
    assert win.scale.property("recommended") is True
    assert win.profile.property("recommended") is True
    req = win._request()
    assert (req.model, req.scale, req.profile, req.data_mode) == ("NERVE", 2, "FIDELITY", "OTF")
    assert req.name == "NERVE_2x_run"
    # Choosing HEART drops the recommended marking.
    win.model.setCurrentIndex(1)
    assert win.model.property("recommended") is False
    win.close()


def test_inference_window_constructs(qapp):
    from hnt.paths import find_repo
    from hnt.ui.inference import InferenceWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = InferenceWindow()
    assert win.device.count() == 3  # Auto / CUDA / CPU
    assert win.arch.currentText() == "NERVE"
    win.close()


def test_train_is_disabled_until_the_configuration_is_valid(qapp, tmp_path):
    from hnt.paths import find_repo
    from hnt.ui.main_window import MainWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = MainWindow()
    assert not win.train_btn.isEnabled()  # no HR folder yet

    hr = tmp_path / "hr"
    hr.mkdir()
    win.hr._edit.setText(str(hr))
    assert win.train_btn.isEnabled()

    win.data_mode.setCurrentIndex(win.data_mode.findData("PAIRED"))
    assert not win.train_btn.isEnabled()  # paired mode needs LR

    lr = tmp_path / "lr"
    lr.mkdir()
    win.lr._edit.setText(str(lr))
    assert win.train_btn.isEnabled()

    win.lr._edit.setText(str(tmp_path / "missing"))
    assert not win.train_btn.isEnabled()
    win.close()


def test_recent_paths_drop_missing_folders(qapp, tmp_path):
    import json
    from types import SimpleNamespace

    from hnt.paths import find_repo
    from hnt.ui.main_window import MainWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = MainWindow()
    win.paths = SimpleNamespace(data=tmp_path)
    good = tmp_path / "good"
    good.mkdir()
    win._remember_path("hr", str(good))
    win._recent["hr"].append(str(tmp_path / "gone"))
    win._recent_file().write_text(json.dumps(win._recent))

    loaded = win._load_recent()
    assert str(good) in loaded["hr"]
    assert str(tmp_path / "gone") not in loaded["hr"]
    win.close()
