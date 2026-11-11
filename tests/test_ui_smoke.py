"""Offscreen GUI smoke tests. Skipped when PySide6 / pyqtgraph are not installed.

Run with the app venv:  QT_QPA_PLATFORM=offscreen ../venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from pathlib import Path

pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication


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


def test_video_window_constructs(qapp):
    from hnt.paths import find_repo
    from hnt.ui.video import VideoWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = VideoWindow()
    assert win.device.count() == 3  # Auto / CUDA / CPU
    assert win.arch.currentText() == "nerve"
    assert win.codec.currentText() == "libx264"
    assert win.crf.value() == 18
    assert not win.overwrite.isChecked()
    assert not win.cancel_btn.isEnabled()
    win.close()


def test_video_window_progress_and_done(qapp, tmp_path):
    from hnt.paths import find_repo
    from hnt.ui.video import VideoWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = VideoWindow()
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")
    win._on_line("frame 12/100")
    assert win.prog.value() == 12
    assert not win.open_out.isEnabled()
    win._on_done(True, str(out))
    assert win.open_out.isEnabled()
    assert "Done" in win.log.toPlainText()
    win.close()


def test_inference_window_accepts_dropped_images_and_folders(qapp, tmp_path):
    from hnt.paths import find_repo
    from hnt.ui.inference import InferenceWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = InferenceWindow()
    image = tmp_path / "img.png"
    image.write_bytes(b"stub")
    folder = tmp_path / "folder"
    folder.mkdir()
    win._add_dropped([image, folder])
    assert win.inputs.count() == 2
    assert win.inputs.item(0).text() == str(image)
    assert win.inputs.item(1).text() == str(folder)
    win.close()


def test_video_window_accepts_dropped_video(qapp, tmp_path):
    from hnt.paths import find_repo
    from hnt.ui.video import VideoWindow

    try:
        find_repo()
    except FileNotFoundError:
        pytest.skip("traiNNer checkout not found")

    win = VideoWindow()
    video = tmp_path / "in.mp4"
    video.write_bytes(b"stub")
    win._add_dropped([video])
    assert win.input._edit.text() == str(video)  # type: ignore[attr-defined]
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


def test_runs_dialog_empty_repo_disables_export(qapp, monkeypatch):
    from hnt.ui import main_window as mw

    monkeypatch.setattr(mw, "list_runs", lambda repo: [])
    dlg = mw.RunsDialog(repo=Path("/exp"), out_dir=Path("/out"))
    assert dlg.runs_list.count() == 0
    assert not dlg.export_btn.isEnabled()
    dlg.close()



def test_infer_output_line_parsers():
    from hnt.ui.inference import parse_output, parse_progress, parse_wrote

    assert parse_progress("progress 3/10 a.png") == (3, 10)
    assert parse_progress("progress x/10 a.png") is None
    assert parse_progress("output a.png") is None
    assert parse_output("output /tmp/a b.png") == "/tmp/a b.png"
    assert parse_output("progress 1/2 x") is None
    assert parse_wrote("wrote 7 file(s) to /out") == 7
    assert parse_wrote("wrote many") is None
    assert parse_wrote("output x") is None


def test_inference_window_progress_bar_determinate(qapp, monkeypatch):
    from hnt.ui.inference import InferenceWindow

    win = InferenceWindow()
    win._on_progress(2, 5)
    assert (win.prog.minimum(), win.prog.maximum(), win.prog.value()) == (0, 5, 2)
    win._on_progress(9, 5)  # clamp
    assert win.prog.value() == 5
    win._on_done(2, 0, [])
    assert (win.prog.minimum(), win.prog.maximum()) == (0, 1)
    win.close()


def test_warmstart_combo_populates_and_requests_pretrain(qapp, monkeypatch):
    from types import SimpleNamespace

    from hnt.ui import main_window as mw
    from hnt.ui.main_window import MainWindow

    entry = SimpleNamespace(id="nerve_2x", name="NERVE pretrain", params=1800000)
    monkeypatch.setattr(mw, "load_catalog", lambda: [entry])
    monkeypatch.setattr(mw, "warmstart_entry",
                        lambda entries, model, scale: entry if model == "NERVE" and scale == 2 else None)
    monkeypatch.setattr(mw.MainWindow, "_cached_pretrain", lambda self, eid: None)

    win = MainWindow()
    assert win.pretrain_cb.count() == 2  # scratch + matching pretrain
    assert win.pretrain_cb.itemData(0) is None
    assert win.pretrain_cb.itemData(1) == "nerve_2x"
    assert win._request().pretrain is None  # scratch is the default

    win.pretrain_cb.setCurrentIndex(1)
    assert "download" in win.pretrain_lbl.text().lower()
    assert win._request().pretrain is None  # not downloaded yet → still scratch

    win._warmstart_path = win.paths.data / "models" / "nerve_2x" / "pre.safetensors"
    assert win._request().pretrain == str(win._warmstart_path)
    win.close()


def test_warmstart_download_gates_train_then_starts(qapp, monkeypatch):
    from types import SimpleNamespace

    from hnt.ui import main_window as mw
    from hnt.ui.main_window import MainWindow

    entry = SimpleNamespace(id="nerve_2x", name="NERVE pretrain", params=1800000)
    started = {}

    from PySide6.QtCore import QThread
    from PySide6.QtCore import Signal as QtSignal

    class FakeWorker(QThread):
        done = QtSignal(str, str)

        def __init__(self, entry, cache):
            super().__init__()
            started["entry"] = entry
            started["cache"] = cache

        def isRunning(self):
            return False

        def start(self):
            started["started"] = True

    monkeypatch.setattr(mw, "load_catalog", lambda: [entry])
    monkeypatch.setattr(mw, "warmstart_entry",
                        lambda entries, model, scale: entry if model == "NERVE" and scale == 2 else None)
    monkeypatch.setattr(mw.MainWindow, "_cached_pretrain", lambda self, eid: None)
    monkeypatch.setattr(mw, "CatalogWorker", FakeWorker)

    win = MainWindow()
    win.hr._edit.setText(str(win.paths.data))  # any existing dir, avoids dataset block
    # paired off; use OTF default so no LR needed
    win.pretrain_cb.setCurrentIndex(1)
    win._warmstart_path = None

    # Simulate the _train preflight path: download requested, training blocked.
    assert win._start_pretrain_download() is True
    assert started.get("started") is True
    assert not win.train_btn.isEnabled()
    win.close()


def _fake_run(name="NERVE_4x_run", iters=(5000, 10000, 20000), resumable=True):
    from types import SimpleNamespace

    ckpts = [SimpleNamespace(iter=i, path=Path(f"/exp/{name}/models/net_g_ema_{i}.safetensors"),
                             ema=True) for i in iters]
    return SimpleNamespace(name=name, path=Path("/exp") / name,
                           config=Path("/exp") / name / f"{name}.yml",
                           checkpoints=ckpts, states=list(iters) if resumable else [],
                           latest_checkpoint=ckpts[-1] if ckpts else None,
                           latest_state=Path("/exp/x.state") if resumable else None,
                           resumable=resumable)


def test_runs_dialog_lists_runs_and_picks_checkpoints(qapp, monkeypatch):
    from hnt.ui import main_window as mw

    runs = [_fake_run(), _fake_run("HEART_2x_run", iters=(), resumable=False)]
    monkeypatch.setattr(mw, "list_runs", lambda repo: runs)
    dlg = mw.RunsDialog(repo=Path("/exp"), out_dir=Path("/out"))
    assert dlg.runs_list.count() == 2
    assert "NERVE_4x_run" in dlg.runs_list.item(0).text()
    assert "no checkpoints" in dlg.runs_list.item(1).text().lower()

    dlg.runs_list.setCurrentRow(0)
    assert dlg.iter_spin.minimum() == 5000 and dlg.iter_spin.maximum() == 20000
    assert dlg.iter_spin.value() == 20000  # defaults to the latest checkpoint

    got = []
    dlg.export_clicked.connect(
        lambda ckpt, out, model, scale: got.append((ckpt, out, model, scale)))
    dlg._export()
    assert got and got[0][0].name == "net_g_ema_20000.safetensors"
    assert got[0][1] == Path("/out") and got[0][2] == "NERVE" and got[0][3] == 4
    dlg.close()


def test_runs_dialog_export_specific_iteration(qapp, monkeypatch):
    from hnt.ui import main_window as mw

    monkeypatch.setattr(mw, "list_runs", lambda repo: [_fake_run()])
    dlg = mw.RunsDialog(repo=Path("/exp"), out_dir=Path("/out"))
    dlg.runs_list.setCurrentRow(0)
    dlg.iter_spin.setValue(5000)
    got = []
    dlg.export_clicked.connect(lambda ckpt, out, model, scale: got.append(ckpt))
    dlg._export()
    assert got[0].name == "net_g_ema_5000.safetensors"
    dlg.close()


def test_show_runs_opens_dialog(qapp, monkeypatch):
    from PySide6.QtCore import QObject, Signal

    from hnt.ui import main_window as mw
    from hnt.ui.main_window import MainWindow

    opened = {}
    fake = _fake_run()

    class FakeDialog(QObject):
        export_clicked = Signal(object, object, str, int)

        def __init__(self, repo, out_dir, parent=None):
            super().__init__()
            opened["repo"] = repo

        def exec(self):
            opened["shown"] = True
            return 0

    monkeypatch.setattr(mw, "list_runs", lambda repo: [fake])
    monkeypatch.setattr(mw, "RunsDialog", FakeDialog)
    win = MainWindow()
    win._show_runs()
    assert opened.get("shown") is True
    win.close()


def test_outputs_show_thumbnails_and_pair_viewer(qapp, tmp_path, monkeypatch):
    from PIL import Image

    from hnt.ui.inference import InferenceWindow

    win = InferenceWindow()
    # real inputs and outputs so thumbnails can decode
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(in_dir / "photo1.png")
    Image.new("RGB", (32, 32), (0, 255, 0)).save(in_dir / "photo2.png")
    (tmp_path / "out").mkdir()
    out1 = tmp_path / "out" / "photo1_model.png"
    out2 = tmp_path / "out" / "photo2_model.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(out1)
    Image.new("RGB", (64, 64), (30, 20, 10)).save(out2)

    win._add_dropped([in_dir])
    win._on_done(2, 0, [str(out1), str(out2)])
    assert win.outputs.count() == 2
    assert not win.outputs.item(0).icon().isNull()
    assert win.open_out.isEnabled() and win.view_ab.isEnabled()

    # selecting output 2 pairs it with input photo2, not the first input
    win.outputs.setCurrentRow(1)
    captured = {}

    class FakeViewer:
        def __init__(self, src, out):
            captured["src"], captured["out"] = src, out

        def show(self):
            pass

    import hnt.ui.viewer as viewer_mod
    monkeypatch.setattr(viewer_mod, "ViewerWindow", FakeViewer)
    win._open_viewer()
    assert captured["src"] == in_dir / "photo2.png"
    assert captured["out"] == out2
    win.close()


def test_runs_dialog_archive_and_delete_with_confirm(qapp, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    from hnt import runs as runs_mod
    from hnt.ui import main_window as mw

    models = tmp_path / "experiments" / "myrun" / "models"
    models.mkdir(parents=True)
    (models / "net_g_ema_1000.safetensors").write_bytes(b"x")

    real_list = runs_mod.list_runs
    monkeypatch.setattr(mw, "list_runs", lambda repo: real_list(repo))
    answers = {"n": 0}

    def fake_question(*a, **k):
        answers["n"] += 1
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(mw.QMessageBox, "question", fake_question)
    dlg = mw.RunsDialog(repo=tmp_path, out_dir=tmp_path)
    dlg.runs_list.setCurrentRow(0)

    dlg._archive()
    assert answers["n"] == 1
    assert dlg.runs_list.count() == 1
    assert "_archived_" in dlg.runs_list.item(0).text()

    before = dlg.runs_list.count()
    dlg._delete()
    assert answers["n"] == 2
    assert dlg.runs_list.count() == before - 1
    dlg.close()


def test_runs_dialog_actions_cancelled_by_user(qapp, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    from hnt import runs as runs_mod
    from hnt.ui import main_window as mw

    models = tmp_path / "experiments" / "myrun" / "models"
    models.mkdir(parents=True)
    (models / "net_g_ema_1000.safetensors").write_bytes(b"x")
    monkeypatch.setattr(mw, "list_runs", lambda repo: runs_mod.list_runs(repo))
    monkeypatch.setattr(mw.QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.No)
    dlg = mw.RunsDialog(repo=tmp_path, out_dir=tmp_path)
    dlg.runs_list.setCurrentRow(0)
    dlg._archive()
    dlg._delete()
    assert dlg.runs_list.count() == 1
    assert "myrun" in dlg.runs_list.item(0).text()
    dlg.close()


def test_export_window_constructs_and_gates(qapp, tmp_path):
    from hnt.ui.export import ExportWindow

    win = ExportWindow()
    assert win.ncnn.isEnabled() is True  # NERVE is the default arch
    win.arch.setCurrentText("HEART")
    assert not win.ncnn.isEnabled()
    win.arch.setCurrentText("NERVE")
    assert win.ncnn.isEnabled()
    assert not win.export_btn.isEnabled()  # no model chosen yet
    ck = tmp_path / "m.safetensors"
    ck.write_bytes(b"x")
    win.model_edit.setText(str(ck))
    assert win.export_btn.isEnabled()
    win.close()


def test_export_window_runs_worker(qapp, tmp_path, monkeypatch):
    from hnt.ui import export as exp_mod

    started = {}
    from PySide6.QtCore import QThread
    from PySide6.QtCore import Signal as QtSignal

    class FakeWorker(QThread):
        line = QtSignal(str)
        done = QtSignal(str, bool)

        def __init__(self, repo, checkpoint, out_dir, model, scale, **kw):
            super().__init__()
            started.update(repo=repo, ckpt=checkpoint, out=out_dir,
                           model=model, scale=scale)

        def isRunning(self):
            return False

        def start(self):
            started["started"] = True

    monkeypatch.setattr(exp_mod, "ExportWorker", FakeWorker)
    win = exp_mod.ExportWindow()
    ck = tmp_path / "m.safetensors"
    ck.write_bytes(b"x")
    win.model_edit.setText(str(ck))
    win.out_edit.setText(str(tmp_path / "out"))
    win.arch.setCurrentText("HEART")
    win.scale.setCurrentText("4")
    win._export()
    assert started["started"] is True
    assert started["model"] == "HEART" and started["scale"] == 4
    assert started["ckpt"] == ck
    assert not win.export_btn.isEnabled()  # busy while running
    win._on_done(str(tmp_path / "out"), True)
    assert win.export_btn.isEnabled()
    win.close()


def test_output_gallery_toggle(qapp):
    from PySide6.QtWidgets import QListView

    from hnt.ui.inference import InferenceWindow

    win = InferenceWindow()
    assert win.outputs.viewMode() == QListView.ViewMode.ListMode
    win._toggle_gallery(True)
    assert win.outputs.viewMode() == QListView.ViewMode.IconMode
    win._toggle_gallery(False)
    assert win.outputs.viewMode() == QListView.ViewMode.ListMode
    win.close()
