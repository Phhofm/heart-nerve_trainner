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
    assert win.arch.currentText() == "nerve"
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


def test_inference_window_lists_catalog_models_and_sets_arch(qapp):
    from hnt.paths import find_repo
    from hnt.ui.inference import InferenceWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = InferenceWindow()
    assert win.catalog.count() >= 8
    ids = {win.catalog.itemData(i) for i in range(win.catalog.count())}
    assert {"pulse_fast_2x", "pulse_fast_4x", "pulse_2x", "pulse_4x",
            "nerve_2x", "nerve_4x", "heart_2x", "heart_4x"} <= ids
    assert win.catalog.currentData() == "nerve_2x"
    assert "1,801,920 parameters" in win.info.text()

    idx = win.catalog.findData("pulse_fast_4x")
    win.catalog.setCurrentIndex(idx)
    assert win.arch.currentText() == "pulse_fast"
    assert win.scale.currentText() == "4"
    win.close()


def test_use_catalog_downloads_and_sets_model_path(qapp, monkeypatch, tmp_path):
    from hnt.paths import find_repo
    from hnt.ui.inference import InferenceWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = InferenceWindow()
    fake = tmp_path / "resolved.safetensors"
    monkeypatch.setattr(
        "hnt.ui.inference.resolve_model",
        lambda entry, cache: fake,
    )
    win._use_catalog()
    win._catalog_worker.wait(2000)
    QApplication.processEvents()
    assert win.model._edit.text() == str(fake)
    assert win.use_catalog.isEnabled()
    win.close()


def test_use_catalog_reports_failure(qapp, monkeypatch, tmp_path):
    from hnt.paths import find_repo
    from hnt.ui.inference import InferenceWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = InferenceWindow()
    monkeypatch.setattr(
        "hnt.ui.inference.resolve_model",
        lambda entry, cache: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(
        "hnt.ui.inference.QMessageBox.warning", lambda *args, **kwargs: None
    )
    win._use_catalog()
    win._catalog_worker.wait(2000)
    QApplication.processEvents()
    assert win.model._edit.text() == ""
    assert "failed" in win.info.text().lower()
    assert win.use_catalog.isEnabled()
    win.close()


def test_inference_completion_lists_outputs_and_enables_actions(qapp, tmp_path):
    from hnt.paths import find_repo
    from hnt.ui.inference import InferenceWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = InferenceWindow()
    out = tmp_path / "out.png"
    out.write_bytes(b"x")
    assert not win.open_out.isEnabled()
    assert not win.view_ab.isEnabled()
    win._on_done(1, 0, [str(out)])
    assert win.outputs.count() == 1
    assert win.open_out.isEnabled()
    assert win.view_ab.isEnabled()
    win._on_done(0, 1, [])
    assert win.outputs.count() == 0
    assert not win.open_out.isEnabled()
    win.close()


def test_train_is_gated_until_setup_complete(qapp, monkeypatch, tmp_path):
    from hnt.paths import find_repo
    from hnt.ui import main_window as mw

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = mw.MainWindow()
    monkeypatch.setattr(mw, "is_setup_complete", lambda paths: False)
    hr = tmp_path / "hr"
    hr.mkdir()
    win.hr._edit.setText(str(hr))  # type: ignore[attr-defined]
    win.train_btn.click()
    assert win.worker is None
    win.close()


def test_pulse_training_uses_paired_mode(qapp):
    from hnt.paths import find_repo
    from hnt.ui.main_window import MainWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = MainWindow()
    idx = win.model.findData("PULSE_fast")
    win.model.setCurrentIndex(idx)
    assert win.data_mode.currentData() == "PAIRED"
    req = win._request()
    assert req.model == "PULSE_fast"
    assert req.data_mode == "PAIRED"

    win.model.setCurrentIndex(win.model.findData("NERVE"))
    otf_idx = win.data_mode.findData("OTF")
    assert win.data_mode.model().item(otf_idx).isEnabled()
    win.close()


def test_close_is_gated_while_training(qapp, monkeypatch):
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QMessageBox

    from hnt.paths import find_repo
    from hnt.ui import main_window as mw

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = mw.MainWindow()
    win._training = True
    stopped = {"called": False}

    class FakeRunner:
        def stop(self):
            stopped["called"] = True

    win.runner = FakeRunner()
    event = QCloseEvent()
    monkeypatch.setattr(
        mw.QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No
    )
    win.closeEvent(event)
    assert not event.isAccepted()
    assert not stopped["called"]

    monkeypatch.setattr(
        mw.QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted()
    assert stopped["called"]
    win.close()
