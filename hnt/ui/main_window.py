"""PySide6 main window — the beginner-first flow + dashboard.

Kept deliberately linear: Settings (left) → Review → Train (right shows live
progress + telemetry). Expert mode reveals advanced fields. Any failure routes to
the "Get help" dialog (§9.5).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dataclasses import asdict

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .. import __version__
from ..config_gen import TrainRequest, build_config, write_config
from ..degradation import DEFAULT_FIDELITY_PRESET, DEFAULT_OTF_PRESET, PRESETS
from ..diagnostics import help_prompt, system_report
from ..downloader import download_lucid, lucid_hr_dir, lucid_home
from ..hardware import detect
from ..paths import get_paths
from ..presets import ARCH, PROFILES, SCALES, auto_params, visible_profiles
from ..train_runner import TrainRunner, read_telemetry
from . import theme
from .chart import MetricsChart


class TrainWorker(QThread):
    line = Signal(str)
    progress = Signal(object)  # Progress
    done = Signal(bool, str)

    def __init__(self, runner: TrainRunner, cfg: dict, name: str, iters: int):
        super().__init__()
        self.runner, self.cfg, self.name, self.iters = runner, cfg, name, iters

    def run(self) -> None:
        try:
            log = self.runner.start(self.cfg, self.name)
            self.line.emit(f"launched → {log}")
            for p in self.runner.watch(log, self.iters):
                self.line.emit(
                    f"iter {p.iter}/{p.iters}  {p.it_s:.2f} it/s  loss={p.loss}  psnr={p.psnr}"
                )
                self.progress.emit(p)
            ok = (self.runner.proc.returncode == 0) if self.runner.proc else False
            self.done.emit(ok, str(log))
        except Exception as e:  # noqa: BLE001
            self.done.emit(False, repr(e))


class DownloadWorker(QThread):
    line = Signal(str)
    done = Signal(str, str)  # (hr_dir, error)

    def __init__(self, dest):
        super().__init__()
        self.dest = dest

    def run(self) -> None:
        try:
            download_lucid(self.dest, on_log=self.line.emit)
            self.done.emit(str(lucid_hr_dir(self.dest)), "")
        except Exception as e:  # noqa: BLE001
            self.done.emit("", repr(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.paths = get_paths()
        self.hw = detect()
        self.setWindowTitle(f"HEART / NERVE Trainer  v{__version__}")
        self.resize(1120, 760)
        self.runner = TrainRunner(self.paths.repo, self.paths.logs)
        self.worker: TrainWorker | None = None
        self._build_ui()
        self._refresh_autoparams()
        QTimer.singleShot(1500, self._startup_update_check)  # non-blocking update notice

    # ---------- UI ----------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        header = QVBoxLayout()
        header.setSpacing(6)
        title = QLabel("HEART / NERVE Trainer"); title.setObjectName("h1")
        subtitle = QLabel("Train your own super-resolution model — no code.")
        subtitle.setObjectName("subtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        chips = QHBoxLayout(); chips.setSpacing(8)
        for text in self._hw_chips():
            chip = QLabel(text); chip.setObjectName("chip"); chips.addWidget(chip)
        chips.addStretch(1)
        header.addLayout(chips)
        root.addLayout(header)

        cols = QHBoxLayout()
        cols.setSpacing(14)
        root.addLayout(cols, 1)

        # --- settings column ---
        settings = QGroupBox("Setup")
        form = QFormLayout(settings)
        self.model = QComboBox(); self.model.addItems(["NERVE", "HEART"])
        self.model.currentTextChanged.connect(self._refresh_autoparams)
        self.scale = QComboBox(); self.scale.addItems([str(s) for s in SCALES]); self.scale.setCurrentText("2")
        self.scale.currentTextChanged.connect(self._refresh_autoparams)
        self.profile = QComboBox(); self.profile.addItems(visible_profiles()); self.profile.setCurrentText("FIDELITY")
        self.profile.currentTextChanged.connect(self._refresh_autoparams)

        self.data_mode = QComboBox(); self.data_mode.addItems(["OTF", "PAIRED"])
        self.data_mode.currentTextChanged.connect(self._on_data_mode)
        self.hr = self._path_row("Select HR image folder…")
        self.lr = self._path_row("(optional) matching LR folder…")
        self.lucid_btn = QPushButton("Download Lucid dataset (CC0, large)…")
        self.lucid_btn.clicked.connect(self._download_lucid)
        self.degrad = QComboBox(); self.degrad.addItems(list(PRESETS.keys()))
        self.degrad.setCurrentText(DEFAULT_FIDELITY_PRESET)
        self.val_hr = self._path_row("(optional) validation HR…")
        self.val_lr = self._path_row("(optional) validation LR…")
        self.out = self._path_row("final-model output folder…", dir_default=True)

        form.addRow("Model", self.model)
        form.addRow("Scale", self.scale)
        form.addRow("Profile", self.profile)
        form.addRow("LR source", self.data_mode)
        form.addRow("HR folder", self.hr)
        form.addRow("", self.lucid_btn)
        form.addRow("LR folder", self.lr)
        form.addRow("Degradation", self.degrad)
        form.addRow("Val HR", self.val_hr)
        form.addRow("Val LR", self.val_lr)
        form.addRow("Output folder", self.out)
        self._on_data_mode()

        self.expert = QCheckBox("Expert mode")
        self.expert.toggled.connect(self._toggle_expert)
        form.addRow("", self.expert)
        self._adv = QGroupBox("Advanced")
        advf = QFormLayout(self._adv)
        self.batch = QSpinBox(); self.batch.setRange(1, 64); self.batch.setValue(4)
        self.patch = QSpinBox(); self.patch.setRange(32, 512); self.patch.setSingleStep(32); self.patch.setValue(96)
        self.iters = QSpinBox(); self.iters.setRange(500, 2_000_000); self.iters.setSingleStep(500); self.iters.setValue(100_000)
        self.lr_edit = QDoubleSpinBox(); self.lr_edit.setDecimals(6); self.lr_edit.setRange(1e-6, 1e-2); self.lr_edit.setValue(2e-4)
        self.compile_cb = QCheckBox("torch.compile (faster; auto-disabled if it fails)"); self.compile_cb.setChecked(True)
        self.ckpt_cb = QCheckBox("gradient checkpointing (less VRAM)"); self.ckpt_cb.setChecked(False)
        for w, lbl in ((self.batch, "Batch"), (self.patch, "Patch (LR)"), (self.iters, "Iterations"),
                       (self.lr_edit, "Learning rate")):
            advf.addRow(lbl, w)
        advf.addRow("", self.compile_cb); advf.addRow("", self.ckpt_cb)
        self._adv.setVisible(False)
        form.addRow(self._adv)

        exp_opts = QGroupBox("Export")
        ef = QFormLayout(exp_opts)
        self.e_onnx32 = QCheckBox("ONNX fp32"); self.e_onnx32.setChecked(True)
        self.e_onnx16 = QCheckBox("ONNX fp16"); self.e_onnx16.setChecked(True)
        self.e_ncnn = QCheckBox("ncnn (NERVE only)"); self.e_ncnn.setChecked(True)
        ef.addRow(self.e_onnx32); ef.addRow(self.e_onnx16); ef.addRow(self.e_ncnn)
        form.addRow(exp_opts)

        btnrow = QHBoxLayout()
        self.review_btn = QPushButton("Review settings")
        self.review_btn.clicked.connect(self._review)
        self.train_btn = QPushButton("Train"); self.train_btn.setObjectName("primary")
        self.train_btn.clicked.connect(self._train)
        btnrow.addWidget(self.review_btn); btnrow.addWidget(self.train_btn)
        form.addRow(btnrow)

        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(settings); ll.addStretch(1)
        cols.addWidget(left, 3)

        # --- dashboard column ---
        dash = QGroupBox("Training")
        dv = QVBoxLayout(dash)
        dv.setSpacing(10)
        self.prog = QProgressBar(); dv.addWidget(self.prog)
        self.status = QLabel("idle"); self.status.setObjectName("muted"); dv.addWidget(self.status)
        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        self.m_iter = self._metric("Iter", "0")
        self.m_psnr = self._metric("PSNR", "—")
        self.m_gpu = self._metric("GPU", "—")
        self.m_vram = self._metric("VRAM", "—")
        self.m_cpu = self._metric("CPU", "—")
        self.m_ram = self._metric("RAM", "—")
        for m in (self.m_iter, self.m_psnr, self.m_gpu, self.m_vram, self.m_cpu, self.m_ram):
            metrics.addWidget(m)
        dv.addLayout(metrics)
        self.chart = MetricsChart()
        dv.addWidget(self.chart, 3)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        dv.addWidget(self.log, 1)
        stop = QPushButton("Stop"); stop.clicked.connect(self._stop)
        helpbtn = QPushButton("Get help / report"); helpbtn.clicked.connect(lambda: self._help("manual"))
        updbtn = QPushButton("Check updates"); updbtn.clicked.connect(self._check_updates)
        clnbtn = QPushButton("Clean up disk"); clnbtn.clicked.connect(self._cleanup)
        inferbtn = QPushButton("Inference…"); inferbtn.clicked.connect(self._open_inference)
        row = QHBoxLayout(); row.addWidget(stop); row.addWidget(helpbtn); row.addWidget(updbtn); row.addWidget(clnbtn); row.addWidget(inferbtn); dv.addLayout(row)
        cols.addWidget(dash, 4)

        self.timer = QTimer(self); self.timer.setInterval(1000); self.timer.timeout.connect(self._tick)

        link = 'style="color:#7c9cff"'
        footer = QLabel(
            'Made by <b>Philip Hofmann</b>  ·  '
            f'<a {link} href="https://github.com/Phhofm">GitHub</a>  ·  '
            f'<a {link} href="https://huggingface.co/Phips">Hugging Face</a>  ·  '
            f'<a {link} href="https://huggingface.co/spaces/Phips/heart">HEART demo</a>  ·  '
            f'<a {link} href="https://huggingface.co/spaces/Phips/nerve">NERVE demo</a>  ·  '
            f'<a {link} href="https://huggingface.co/spaces/Phips/nerve-convert">Converter</a>'
        )
        footer.setObjectName("muted")
        footer.setOpenExternalLinks(True)
        root.addWidget(footer)

    def _hw_chips(self) -> list[str]:
        hw = self.hw
        gpu = hw.gpus[0] if hw.gpus else None
        return [
            f"{hw.os} {hw.os_version}",
            hw.cpu,
            f"{hw.ram_gb:g} GB RAM",
            f"{gpu.name} · {gpu.vram_gb:g} GB" if gpu else "CPU only",
            f"Python {hw.python}",
        ]

    def _metric(self, label: str, val: str) -> QWidget:
        w = QWidget(); w.setObjectName("stat")
        w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QVBoxLayout(w); v.setContentsMargins(12, 8, 12, 8); v.setSpacing(1)
        l = QLabel(label.upper()); l.setObjectName("statlabel")
        m = QLabel(val); m.setObjectName("metric")
        v.addWidget(l); v.addWidget(m)
        w._val = m  # type: ignore[attr-defined]
        return w

    def _path_row(self, placeholder: str, dir_default: bool = True) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        e = QLineEdit(); e.setPlaceholderText(placeholder)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._pick(e, dir_default))
        h.addWidget(e); h.addWidget(b)
        w._edit = e  # type: ignore[attr-defined]
        return w

    def _pick(self, edit: QLineEdit, directory: bool) -> None:
        if directory:
            p = QFileDialog.getExistingDirectory(self, "Select folder")
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Select file")
        if p:
            edit.setText(p)

    # ---------- behaviour ----------
    def _on_data_mode(self) -> None:
        otf = self.data_mode.currentText() == "OTF"
        self.degrad.setEnabled(otf)
        self.lr.setEnabled(not otf)
        if otf and self.degrad.currentText() == DEFAULT_FIDELITY_PRESET:
            self.degrad.setCurrentText(DEFAULT_OTF_PRESET)

    def _toggle_expert(self, on: bool) -> None:
        self._adv.setVisible(on)

    def _refresh_autoparams(self) -> None:
        ap = auto_params(self.model.currentText(), int(self.scale.currentText()),
                         self.profile.currentText(), self.hw.best_vram_gb)
        if not self.expert.isChecked():
            self.batch.setValue(ap.batch)
            self.patch.setValue(ap.lq_size)
        self.compile_cb.setChecked(ap.use_compile)
        self.status.setText(ap.note)

    def _request(self) -> TrainRequest:
        otf = self.data_mode.currentText() == "OTF"
        return TrainRequest(
            model=self.model.currentText(), scale=int(self.scale.currentText()),
            profile=self.profile.currentText(), data_mode="OTF" if otf else "PAIRED",
            hr_dir=self.hr._edit.text(), lr_dir=self.lr._edit.text(),
            degradation=self.degrad.currentText() if otf else "",
            val_hr_dir=self.val_hr._edit.text(), val_lr_dir=self.val_lr._edit.text(),
            batch=self.batch.value(), lq_size=self.patch.value(), iters=self.iters.value(),
            lr=self.lr_edit.value() if self.expert.isChecked() else None,
            use_compile=self.compile_cb.isChecked(), use_checkpoint=self.ckpt_cb.isChecked(),
            name=f"{self.model.currentText()}_{self.scale.currentText()}x_run",
        )

    def _review(self) -> None:
        req = self._request()
        try:
            cfg = build_config(req, self.paths.repo)
        except Exception as e:  # noqa: BLE001
            return self._help("config generation", str(e))
        keys = ("scale", "use_compile", "use_channels_last")
        tds = cfg["datasets"]["train"]
        msg = (f"Model: {req.model}  Scale: {req.scale}×\nProfile: {req.profile}  "
               f"Data: {req.data_mode}\nHR: {req.hr_dir}\nLR: {req.lr_dir or '(generated on-the-fly)'}\n"
               f"Degradation: {req.degradation or '-'}\nBatch: {req.batch}  Patch: {req.lq_size}  "
               f"Iters: {req.iters}\nPretrain: {req.pretrain or 'from scratch'}\n"
               f"Config data.type: {tds.get('type')}")
        QMessageBox.information(self, "Review settings", msg)

    def _train(self) -> None:
        req = self._request()
        if not req.hr_dir:
            return self._help("setup", "No HR folder selected.")
        if req.data_mode == "PAIRED" and not req.lr_dir:
            return self._help("setup", "Paired mode needs an LR folder (or switch to on-the-fly).")
        try:
            cfg = build_config(req, self.paths.repo)
        except Exception as e:  # noqa: BLE001
            return self._help("config generation", str(e))
        self.prog.setRange(0, req.iters)
        self.chart.clear()
        self.train_btn.setEnabled(False)
        self.worker = TrainWorker(self.runner, cfg, req.name, req.iters)
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.start()
        self.timer.start()

    def _on_progress(self, p) -> None:
        self.prog.setValue(p.iter)
        self.m_iter._val.setText(str(p.iter))  # type: ignore[attr-defined]
        if p.psnr is not None:
            self.m_psnr._val.setText(f"{p.psnr:.2f}")  # type: ignore[attr-defined]
        for name, value in p.metrics.items():
            self.chart.add_point(name, p.iter, value)

    def _download_lucid(self) -> None:
        dest = lucid_home(self.paths.data)
        self.lucid_btn.setEnabled(False)
        self.log.appendPlainText(f"Lucid → {dest}")
        self._dl = DownloadWorker(dest)
        self._dl.line.connect(self.log.appendPlainText)
        self._dl.done.connect(self._on_lucid_done)
        self._dl.start()

    def _on_lucid_done(self, hr_dir: str, error: str) -> None:
        self.lucid_btn.setEnabled(True)
        if error:
            self._help("lucid download", error)
            return
        self.hr._edit.setText(hr_dir)  # type: ignore[attr-defined]
        self.status.setText(f"Lucid ready → HR folder set to {hr_dir}")

    def _stop(self) -> None:
        self.runner.stop()
        self.status.setText("stopping…")

    def _on_done(self, ok: bool, info: str) -> None:
        self.timer.stop()
        self.train_btn.setEnabled(True)
        self.status.setText("done" if ok else "failed")
        if not ok:
            self._help("training", info)

    def _tick(self) -> None:
        t = read_telemetry()
        self.m_gpu._val.setText(f"{t.gpu_util:.0f}%")        # type: ignore[attr-defined]
        self.m_vram._val.setText(f"{t.vram_used_gb:.1f}/{t.vram_total_gb:.0f}G")  # type: ignore[attr-defined]
        self.m_cpu._val.setText(f"{t.cpu:.0f}%")             # type: ignore[attr-defined]
        self.m_ram._val.setText(f"{t.ram:.0f}%")             # type: ignore[attr-defined]
        self.m_gpu.setToolTip(f"GPU temp {t.gpu_temp:.0f}°C")

    def _check_updates(self) -> None:
        from ..updater import apply_update, check_for_update, download_installer

        info = check_for_update()
        if info is None:
            QMessageBox.information(self, "Updates", "You're up to date.")
            return
        if QMessageBox.question(
            self, "Update available",
            f"Version {info.version} is available.\n\n{info.notes}\n\nDownload and install now?",
        ) == QMessageBox.StandardButton.Yes:
            try:
                inst = download_installer(info, self.paths.data / "updates")
                apply_update(inst)
                QMessageBox.information(self, "Update", "Installer launched — close the app to finish.")
            except Exception as e:  # noqa: BLE001
                self._help("update", str(e))

    def _cleanup(self) -> None:
        from ..cleanup import cleanup_runs

        outs = [Path(self.out._edit.text())] if self.out._edit.text() else []
        rep = cleanup_runs(self.paths, outs, dry_run=True)
        if not rep.removed:
            QMessageBox.information(self, "Clean up", "Nothing to clean.")
            return
        if QMessageBox.question(
            self, "Clean up",
            f"Free ~{rep.freed_gb} GB by deleting {len(rep.removed)} experiment folder(s)?\n"
            "Your final output folder is kept.",
        ) == QMessageBox.StandardButton.Yes:
            done = cleanup_runs(self.paths, outs, dry_run=False)
            QMessageBox.information(self, "Clean up", f"Freed ~{done.freed_gb} GB.")

    def _startup_update_check(self) -> None:
        from ..updater import check_for_update

        try:
            info = check_for_update()
        except Exception:  # noqa: BLE001
            return
        if info is not None:
            self.status.setText(f"Update {info.version} available (button: 'Check updates').")

    def _open_inference(self) -> None:
        from .inference import InferenceWindow

        if getattr(self, "_inferwin", None) is None:
            self._inferwin = InferenceWindow()
        self._inferwin.show()
        self._inferwin.raise_()

    def _help(self, step: str, error: str = "") -> None:
        text = help_prompt(step, error, log_tail=self.log.toPlainText()[-2000:], hw=self.hw)
        QApplication.clipboard().setText(text)
        QMessageBox.information(
            self, "Get help",
            "A diagnostic message was copied to your clipboard.\n\n"
            "Paste it into any AI assistant (ChatGPT / Gemini / DeepSeek / GLM …) "
            "or a forum, and ask for step-by-step help.",
        )


def run() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)
    w = MainWindow()
    w.show()
    return app.exec()
