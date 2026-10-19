"""PySide6 main window — the beginner-first flow + dashboard.

Kept deliberately linear: Settings (left) → Review → Train (right shows live
progress + telemetry). Expert mode reveals advanced fields. Any failure routes to
the "Get help" dialog (§9.5).
"""

from __future__ import annotations

import subprocess
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
from ..bootstrap import runtime_python
from ..config_gen import TrainRequest, build_config, write_config
from ..degradation import DEFAULT_FIDELITY_PRESET, DEFAULT_OTF_PRESET, PRESETS
from ..diagnostics import help_prompt, system_report
from ..downloader import download_lucid, lucid_hr_dir, lucid_home
from ..hardware import detect
from ..paths import get_paths
from ..presets import ARCH, DEFAULT_PROFILE, PROFILES, auto_params, visible_profiles
from ..train_runner import TrainRunner, looks_like_oom, read_telemetry
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


class ExportWorker(QThread):
    line = Signal(str)
    done = Signal(str, bool)  # (out_dir, ok)

    def __init__(self, repo, checkpoint, out_dir, model, scale, *, fp32, fp16, ncnn,
                 python_exe=None):
        super().__init__()
        self.repo, self.checkpoint, self.out_dir = repo, checkpoint, out_dir
        self.model, self.scale = model, scale
        self.fp32, self.fp16, self.ncnn = fp32, fp16, ncnn
        self.python_exe = python_exe

    def run(self) -> None:
        from ..exporter import export_checkpoint

        try:
            res = export_checkpoint(
                self.repo, self.checkpoint, self.out_dir, self.model, self.scale,
                fp32=self.fp32, fp16=self.fp16, ncnn=self.ncnn,
                python_exe=self.python_exe,
            )
            for err in res.errors:
                self.line.emit(f"export: {err}")
            made = [p.name for p in (res.safetensors, res.onnx_fp32, res.onnx_fp16) if p]
            if made:
                self.line.emit("exported: " + ", ".join(made))
            self.done.emit(str(self.out_dir), res.safetensors is not None)
        except Exception as e:  # noqa: BLE001
            self.line.emit(f"export failed: {e!r}")
            self.done.emit("", False)


class SetupWorker(QThread):
    """First-run dependency install into the isolated venv (never the system Python)."""

    line = Signal(str)
    done = Signal(bool, str)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def run(self) -> None:
        try:
            from ..bootstrap import bootstrap

            bootstrap(self.paths, on_log=self.line.emit)
            self.done.emit(True, "")
        except Exception as e:  # noqa: BLE001
            self.done.emit(False, repr(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.paths = get_paths()
        self.hw = detect()
        self.py = runtime_python(self.paths)
        self.setWindowTitle(f"HEART / NERVE Trainer  v{__version__}")
        self.resize(1120, 760)
        self.runner = TrainRunner(self.paths.repo, self.paths.logs, python_exe=self.py)
        self.worker: TrainWorker | None = None
        self._stopped = False
        self._oom_retries = 0
        self._compile_retries = 0
        self._build_ui()
        self._refresh_autoparams()
        QTimer.singleShot(1500, self._startup_update_check)  # non-blocking update notice
        QTimer.singleShot(400, self._first_run_check)

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
        self.form = form

        # Model — NERVE is the recommended default (much faster to train).
        self.model = QComboBox()
        self.model.addItem("NERVE  ·  recommended (faster)", "NERVE")
        self.model.addItem("HEART (higher quality, slower)", "HEART")
        # Scale — 2× is recommended and ships with pretrained weights.
        self.scale = QComboBox()
        self.scale.addItem("2×  ·  recommended", 2)
        self.scale.addItem("4×", 4)
        # Profile — Fidelity is the recommended default.
        self.profile = QComboBox()
        for key in visible_profiles():
            tag = "  ·  recommended" if key == DEFAULT_PROFILE else ""
            self.profile.addItem(f"{PROFILES[key].label}{tag}", key)
        # Image source — beginners never need to meet the word "OTF".
        self.data_mode = QComboBox()
        self.data_mode.addItem("Automatic — I only have HR images  ·  recommended", "OTF")
        self.data_mode.addItem("I already have matching LR + HR folders", "PAIRED")
        for combo in (self.model, self.scale, self.profile):
            combo.currentIndexChanged.connect(self._on_choice_changed)
        self.data_mode.currentIndexChanged.connect(self._on_data_mode)

        self.hr = self._path_row("Select your HR image folder…")
        self.lr = self._path_row("(optional) matching LR folder…")
        self.lucid_btn = QPushButton("Download Lucid dataset (CC0, large)…")
        self.lucid_btn.clicked.connect(self._download_lucid)
        self.degrad = QComboBox()
        for key, preset in PRESETS.items():
            self.degrad.addItem(preset.label, key)
        self.degrad.setCurrentIndex(self.degrad.findData(DEFAULT_FIDELITY_PRESET))
        self.val_hr = self._path_row("(optional) validation HR…")
        self.val_lr = self._path_row("(optional) validation LR…")
        self.out = self._path_row("final-model output folder…", dir_default=True)
        self.out._edit.setText(str(self.paths.data / "outputs"))  # type: ignore[attr-defined]
        self.perceptual = QCheckBox("Also track DISTS + LPIPS (slower validation)")

        form.addRow("Model", self.model)
        form.addRow("Scale", self.scale)
        form.addRow("Profile", self.profile)
        hint = QLabel("Fields outlined in green are the recommended choices for most users.")
        hint.setObjectName("muted"); hint.setWordWrap(True)
        form.addRow(hint)
        self.guide = QLabel(); self.guide.setObjectName("subtitle"); self.guide.setWordWrap(True)
        form.addRow(self.guide)
        form.addRow("Image source", self.data_mode)
        form.addRow("HR folder (your images)", self.hr)
        form.addRow("", self.lucid_btn)
        form.addRow("LR folder", self.lr)
        form.addRow("Degradation", self.degrad)
        form.addRow("Validation HR", self.val_hr)
        form.addRow("Validation LR", self.val_lr)
        form.addRow("Output folder", self.out)
        form.addRow("", self.perceptual)

        self.expert = QCheckBox("Expert mode")
        self.expert.toggled.connect(self._toggle_expert)
        form.addRow("", self.expert)
        self._adv = QGroupBox("Advanced")
        advf = QFormLayout(self._adv)
        self.batch = QSpinBox(); self.batch.setRange(1, 64); self.batch.setValue(4)
        self.patch = QSpinBox(); self.patch.setRange(32, 512); self.patch.setSingleStep(32); self.patch.setValue(96)
        self.iters = QSpinBox(); self.iters.setRange(500, 2_000_000); self.iters.setSingleStep(500); self.iters.setValue(100_000)
        self.lr_edit = QDoubleSpinBox(); self.lr_edit.setDecimals(6); self.lr_edit.setRange(1e-6, 1e-2); self.lr_edit.setValue(2e-4)
        self.val_tile = QSpinBox(); self.val_tile.setRange(0, 2048)
        self.val_tile.setSingleStep(64); self.val_tile.setValue(256)
        self.val_tile.setToolTip("Validation is tiled to avoid out-of-memory on large images. 0 = whole image.")
        self.compile_cb = QCheckBox("torch.compile (faster; auto-disabled if it fails)"); self.compile_cb.setChecked(True)
        self.ckpt_cb = QCheckBox("gradient checkpointing (less VRAM)"); self.ckpt_cb.setChecked(False)
        for w, lbl in ((self.batch, "Batch"), (self.patch, "Patch (LR)"), (self.iters, "Iterations"),
                       (self.lr_edit, "Learning rate"), (self.val_tile, "Validation tile")):
            advf.addRow(lbl, w)
        advf.addRow("", self.compile_cb); advf.addRow("", self.ckpt_cb)
        self._adv.setVisible(False)
        form.addRow(self._adv)
        self._on_data_mode()
        self._refresh_recommended()
        self._update_field_visibility()

        exp_opts = QGroupBox("Export")
        ef = QFormLayout(exp_opts)
        self.e_onnx32 = QCheckBox("ONNX fp32"); self.e_onnx32.setChecked(True)
        self.e_onnx16 = QCheckBox("ONNX fp16"); self.e_onnx16.setChecked(True)
        self.e_ncnn = QCheckBox("ncnn (NERVE)")
        ef.addRow(self.e_onnx32); ef.addRow(self.e_onnx16); ef.addRow(self.e_ncnn)
        openbtn = QPushButton("Open output folder"); openbtn.clicked.connect(self._open_output)
        ef.addRow(openbtn)
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
        setupbtn = QPushButton("Set up / repair"); setupbtn.clicked.connect(self._setup_env)
        inferbtn = QPushButton("Inference…"); inferbtn.clicked.connect(self._open_inference)
        row = QHBoxLayout(); row.addWidget(stop); row.addWidget(helpbtn); row.addWidget(updbtn)
        row.addWidget(clnbtn); row.addWidget(setupbtn); row.addWidget(inferbtn); dv.addLayout(row)
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

        self.hr._edit.textChanged.connect(self._refresh_guide)  # type: ignore[attr-defined]
        self._refresh_guide()

    def _hw_chips(self) -> list[str]:
        hw = self.hw
        gpu = hw.gpus[0] if hw.gpus else None
        amp = (f"AMP: {'bf16' if hw.supports_bf16 else 'fp16'}" if gpu
               else "AMP: off (CPU)")
        return [
            f"{hw.os} {hw.os_version}",
            hw.cpu,
            f"{hw.ram_gb:g} GB RAM",
            f"{gpu.name} · {gpu.vram_gb:g} GB" if gpu else "CPU only",
            amp,
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

    def _refresh_guide(self) -> None:
        """One clear next step, always visible."""
        if self.worker is not None and self.worker.isRunning():
            self.guide.setText("Training — watch progress and the graphs on the right.")
        elif not self.hr._edit.text().strip():  # type: ignore[attr-defined]
            self.guide.setText("Next: choose a folder of high-res images (or download Lucid).")
        else:
            self.guide.setText("Ready — press Train. NERVE · 2× · Fidelity are already set.")

    def _open_output(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        p = Path(self.out._edit.text() or (self.paths.data / "outputs"))  # type: ignore[attr-defined]
        p.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    # ---------- behaviour ----------
    def _on_choice_changed(self, *_: object) -> None:
        self._refresh_autoparams()
        self._refresh_recommended()

    def _on_data_mode(self, *_: object) -> None:
        otf = self.data_mode.currentData() == "OTF"
        self.degrad.setEnabled(otf)
        self.lr.setEnabled(not otf)
        if otf and self.degrad.currentData() == DEFAULT_FIDELITY_PRESET:
            self.degrad.setCurrentIndex(self.degrad.findData(DEFAULT_OTF_PRESET))
        self._update_field_visibility()

    def _toggle_expert(self, on: bool) -> None:
        self._adv.setVisible(on)
        self._update_field_visibility()

    def _set_row_visible(self, widget: QWidget, visible: bool) -> None:
        if hasattr(self.form, "setRowVisible"):
            self.form.setRowVisible(widget, visible)
        else:  # pragma: no cover - very old PySide6
            widget.setVisible(visible)
            label = self.form.labelForField(widget)
            if label is not None:
                label.setVisible(visible)

    def _update_field_visibility(self) -> None:
        """Hide the fiddly rows unless the user is an expert (or they're required)."""
        expert = self.expert.isChecked()
        paired = self.data_mode.currentData() == "PAIRED"
        self._set_row_visible(self.degrad, expert)
        self._set_row_visible(self.val_hr, expert)
        self._set_row_visible(self.val_lr, expert)
        self._set_row_visible(self.lr, paired)

    def _mark(self, widget: QWidget, recommended: bool) -> None:
        widget.setProperty("recommended", recommended)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _refresh_recommended(self) -> None:
        self._mark(self.model, self.model.currentData() == "NERVE")
        self._mark(self.scale, self.scale.currentData() == 2)
        self._mark(self.profile, self.profile.currentData() == DEFAULT_PROFILE)

    def _refresh_autoparams(self) -> None:
        ap = auto_params(self.model.currentData(), int(self.scale.currentData()),
                         self.profile.currentData(), self.hw.best_vram_gb)
        if not self.expert.isChecked():
            self.batch.setValue(ap.batch)
            self.patch.setValue(ap.lq_size)
        self.compile_cb.setChecked(ap.use_compile)
        self.status.setText(ap.note)

    def _request(self) -> TrainRequest:
        otf = self.data_mode.currentData() == "OTF"
        return TrainRequest(
            model=self.model.currentData(), scale=int(self.scale.currentData()),
            profile=self.profile.currentData(), data_mode="OTF" if otf else "PAIRED",
            hr_dir=self.hr._edit.text(), lr_dir=self.lr._edit.text(),
            degradation=self.degrad.currentData() if otf else "",
            val_hr_dir=self.val_hr._edit.text(), val_lr_dir=self.val_lr._edit.text(),
            batch=self.batch.value(), lq_size=self.patch.value(), iters=self.iters.value(),
            lr=self.lr_edit.value() if self.expert.isChecked() else None,
            use_compile=self.compile_cb.isChecked(), use_checkpoint=self.ckpt_cb.isChecked(),
            amp_bf16=self.hw.supports_bf16,
            val_perceptual_metrics=self.perceptual.isChecked(),
            val_tile_size=self.val_tile.value(),
            name=f"{self.model.currentData()}_{self.scale.currentData()}x_run",
        )

    def _review(self) -> None:
        req = self._request()
        try:
            cfg = build_config(req, self.paths.repo)
        except Exception as e:  # noqa: BLE001
            return self._help("config generation", str(e))
        tds = cfg["datasets"]["train"]
        mode = "Automatic (LR generated)" if req.data_mode == "OTF" else "Paired LR + HR"
        msg = (f"Model: {req.model}  Scale: {req.scale}×\nProfile: {PROFILES[req.profile].label}  "
               f"Data: {mode}\nHR: {req.hr_dir}\nLR: {req.lr_dir or '(generated on-the-fly)'}\n"
               f"Degradation: {req.degradation or '-'}\nBatch: {req.batch}  Patch: {req.lq_size}  "
               f"Iters: {req.iters}\nPretrain: {req.pretrain or 'from scratch'}\n"
               f"Config data.type: {tds.get('type')}")
        QMessageBox.information(self, "Review settings", msg)

    def _train(self, _retry: bool = False) -> None:
        if not self.py:
            self.status.setText("set up the environment first")
            self.guide.setText("First-time setup is needed before training.")
            return self._setup_env()
        if not _retry:
            self._oom_retries = 0
            self._compile_retries = 0
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
        self.prog.setValue(0)
        self.status.setText("training…")
        self.guide.setText("Training — watch progress and the graphs on the right.")
        self.chart.clear()
        self._last_req = req
        self._stopped = False
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
        self.prog.setRange(0, 0)  # busy
        self.guide.setText("Downloading Lucid (large, resumable)…")
        self.log.appendPlainText(f"Lucid → {dest}")
        self._dl = DownloadWorker(dest)
        self._dl.line.connect(self.log.appendPlainText)
        self._dl.done.connect(self._on_lucid_done)
        self._dl.start()

    def _on_lucid_done(self, hr_dir: str, error: str) -> None:
        self.lucid_btn.setEnabled(True)
        self.prog.setRange(0, 1)
        self.prog.setValue(0)
        if error:
            self.guide.setText("Lucid download failed — press Get help if it keeps failing.")
            self._help("lucid download", error)
            return
        self.hr._edit.setText(hr_dir)  # type: ignore[attr-defined]
        self.status.setText(f"Lucid ready → HR folder set to {hr_dir}")
        self.guide.setText("Lucid is ready — press Train.")

    def _stop(self) -> None:
        self._stopped = True
        self.runner.stop()
        self.status.setText("stopping…")
        self.guide.setText("Stopping…")

    def _on_done(self, ok: bool, info: str) -> None:
        self.timer.stop()
        self.train_btn.setEnabled(True)
        if not ok:
            if self._stopped:
                self.status.setText("stopped")
                self.guide.setText("Stopped — press Train to start again.")
                return
            if self._maybe_auto_retry(info):
                return
            self.status.setText("failed")
            self.guide.setText("Something went wrong — press Get help for a copyable report.")
            return self._help("training", info)
        self.status.setText("training done — exporting model…")
        self.guide.setText("Training finished — exporting your model…")
        self._export_model()

    def _maybe_auto_retry(self, info: str) -> bool:
        """Beginner-friendly recovery: shrink on OOM, drop compile on compile errors."""
        log = Path(info)
        if not log.is_file():
            return False
        if looks_like_oom(log.read_text(errors="ignore")) and self._oom_retries < 2:
            if self.patch.value() > 64:
                self.patch.setValue(max(64, self.patch.value() // 2))
            elif self.batch.value() > 1:
                self.batch.setValue(max(1, self.batch.value() // 2))
            else:
                return False
            self._oom_retries += 1
            self.log.appendPlainText(
                f"Out of memory — retrying with patch {self.patch.value()}, "
                f"batch {self.batch.value()} …"
            )
            self.status.setText("out of memory — retrying with smaller settings")
            self.guide.setText("Ran out of memory — retrying with smaller settings…")
            QTimer.singleShot(0, lambda: self._train(_retry=True))
            return True
        if (self.compile_cb.isChecked() and self._compile_retries < 1
                and self.runner.compile_failed(log)):
            self.compile_cb.setChecked(False)
            self._compile_retries += 1
            self.log.appendPlainText("torch.compile failed — retrying without compile …")
            self.status.setText("compile failed — retrying without compile")
            self.guide.setText("Compile failed — retrying without it…")
            QTimer.singleShot(0, lambda: self._train(_retry=True))
            return True
        return False

    def _export_model(self) -> None:
        req = getattr(self, "_last_req", None)
        if req is None:
            return
        run_models = self.paths.repo / "experiments" / req.name / "models"
        ckpts = sorted(run_models.glob("net_g_ema_*.safetensors")) or sorted(
            run_models.glob("net_g_*.safetensors"))
        if not ckpts:
            self.prog.setRange(0, 1)
            self.prog.setValue(1)
            self.status.setText("done — no checkpoint found to export")
            self.guide.setText("Done — no checkpoint was written; check the log.")
            return
        out_dir = Path(self.out._edit.text()) if self.out._edit.text() else self.paths.data / "outputs"  # type: ignore[attr-defined]
        self.prog.setRange(0, 0)  # busy while converting/exporting
        self.status.setText(f"exporting model → {out_dir} …")
        self.guide.setText("Exporting safetensors / ONNX / ncnn — this can take a minute…")
        self.log.appendPlainText(f"exporting → {out_dir}")
        self.export_worker = ExportWorker(
            self.paths.repo, ckpts[-1], out_dir, req.model, req.scale,
            fp32=self.e_onnx32.isChecked(), fp16=self.e_onnx16.isChecked(),
            ncnn=self.e_ncnn.isChecked(), python_exe=self.py,
        )
        self.export_worker.line.connect(self.log.appendPlainText)
        self.export_worker.done.connect(self._on_export_done)
        self.export_worker.start()

    def _on_export_done(self, out_dir: str, ok: bool) -> None:
        self.prog.setRange(0, 1)
        self.prog.setValue(1)
        if ok:
            self.status.setText(f"model saved → {out_dir}")
            self.guide.setText("Done — open the output folder (left) to get your model.")
        else:
            self.status.setText("trained; export had problems (see log)")
            self.guide.setText("Trained — export had problems; check the log.")

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

    def _first_run_check(self) -> None:
        """Offer setup once, so a freshly installed app can actually train."""
        from ..bootstrap import is_setup_complete

        if is_setup_complete(self.paths):
            return
        if self.py and Path(self.py) == Path(sys.executable) and self._has_torch(self.py):
            return  # dev checkout using an already-working interpreter
        if QMessageBox.question(
            self, "First-time setup",
            "Set up the training environment now?\n\n"
            "This downloads PyTorch and friends into the app's own folder "
            "(~2–3 GB, one time). Your system Python is never modified.",
        ) == QMessageBox.StandardButton.Yes:
            self._setup_env()

    @staticmethod
    def _has_torch(py: str) -> bool:
        if not py:
            return False  # environment not set up yet — never launch the app binary
        try:
            return subprocess.run(
                [py, "-c", "import torch"], capture_output=True, timeout=20
            ).returncode == 0
        except Exception:  # noqa: BLE001
            return False

    def _setup_env(self) -> None:
        self.train_btn.setEnabled(False)
        self.prog.setRange(0, 0)  # busy
        self.status.setText("setting up environment (one time)…")
        self.guide.setText("Setting up the training environment (one-time, ~2–3 GB)…")
        self.log.appendPlainText("Setting up the isolated training environment …")
        self._setup_worker = SetupWorker(self.paths)
        self._setup_worker.line.connect(self.log.appendPlainText)
        self._setup_worker.done.connect(self._on_setup_done)
        self._setup_worker.start()

    def _on_setup_done(self, ok: bool, error: str) -> None:
        self.train_btn.setEnabled(True)
        self.prog.setRange(0, 1)
        self.prog.setValue(0)
        if ok:
            self.status.setText("setup complete — ready to train")
            self.guide.setText("Setup complete — choose your images, then press Train.")
            self.log.appendPlainText("Setup complete.")
        else:
            self.status.setText("setup failed")
            self.guide.setText("Setup failed — press Get help for a copyable report.")
            self._help("setup", error)

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
    from PySide6.QtCore import QLockFile

    paths = get_paths()
    lock = None
    try:
        paths.data.mkdir(parents=True, exist_ok=True)
        lock = QLockFile(str(paths.data / "app.lock"))
        lock.setStaleLockTime(30_000)
        if not lock.tryLock(300):
            return 0  # another instance is already running — don't stack windows
    except OSError:
        lock = None  # unwritable data dir: still allow the app to start

    app = QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)
    w = MainWindow()
    w.show()
    rc = app.exec()
    if lock is not None:
        lock.unlock()
    return rc
