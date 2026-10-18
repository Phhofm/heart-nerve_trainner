"""Standalone inference window — run a trained or downloaded HEART/NERVE model
on single images or whole folders. Outputs land in a chosen folder, suffixed with
the model name.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from ..bootstrap import app_source_dir, runtime_python
from ..hardware import detect
from ..paths import get_paths
from ..presets import SCALES


class InferWorker(QThread):
    """Run inference in the isolated venv (works in frozen builds, never freezes the UI)."""

    line = Signal(str)
    done = Signal(int, int)  # (wrote, errors)

    def __init__(self, model, inputs, out, arch, scale, device, tile):
        super().__init__()
        self.model, self.inputs, self.out = model, inputs, out
        self.arch, self.scale, self.device, self.tile = arch, scale, device, tile

    def run(self) -> None:
        src = app_source_dir()
        cmd = [
            runtime_python(get_paths()), "-m", "hnt.cli", "infer",
            "--model", self.model, "--arch", self.arch, "--scale", str(self.scale),
            "--input", *self.inputs, "--out", self.out,
            "--device", self.device, "--tile", str(self.tile),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            p = subprocess.run(cmd, cwd=str(src), env=env, capture_output=True, text=True)
        except Exception as e:  # noqa: BLE001
            self.line.emit(f"ERROR: {e}")
            self.done.emit(0, 1)
            return
        for ln in (p.stdout + p.stderr).splitlines():
            self.line.emit(ln)
        wrote = 0
        for ln in p.stdout.splitlines():
            if ln.startswith("wrote "):
                try:
                    wrote = int(ln.split()[1])
                except (IndexError, ValueError):
                    pass
        self.done.emit(wrote, 0 if p.returncode == 0 else 1)


class InferenceWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.paths = get_paths()
        self.setWindowTitle("Inference — HEART / NERVE Trainer")
        self.resize(800, 620)

        root = QVBoxLayout()
        central = QWidget(); central.setLayout(root); self.setCentralWidget(central)

        box = QGroupBox("Run inference")
        form = QFormLayout(box)
        self.model = self._file_row("trained or downloaded .safetensors/.pth …")
        self.arch = QComboBox(); self.arch.addItems(["NERVE", "HEART"])
        self.scale = QComboBox(); self.scale.addItems([str(s) for s in SCALES]); self.scale.setCurrentText("2")
        self.device = QComboBox(); self.device.addItems(["Auto", "CUDA", "CPU"])
        self.tile = QComboBox(); self.tile.addItems(["0 (whole image)", "256", "512"])

        self.inputs = QListWidget()
        irow = QHBoxLayout()
        for label, fn in (("Add images…", self._add_images),
                          ("Add folder…", self._add_folder),
                          ("Clear", self.inputs.clear)):
            b = QPushButton(label); b.clicked.connect(fn); irow.addWidget(b)
        inwrap = QWidget(); inwrap.setLayout(irow)

        self.out = self._dir_row("output folder …")

        form.addRow("Model", self.model)
        form.addRow("Architecture", self.arch)
        form.addRow("Scale", self.scale)
        form.addRow("Device", self.device)
        form.addRow("Big image tiling", self.tile)
        form.addRow("Input", self.inputs)
        form.addRow("", inwrap)
        form.addRow("Output folder", self.out)
        run = QPushButton("Run"); run.setObjectName("primary"); run.clicked.connect(self._run)
        form.addRow(run)
        self.run_btn = run
        root.addWidget(box)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

    # ---- helpers ----
    def _file_row(self, ph: str) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        e = QLineEdit(); e.setPlaceholderText(ph)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._set(e, QFileDialog.getOpenFileName(self, "Model", filter="Models (*.safetensors *.pth)")[0]))
        h.addWidget(e); h.addWidget(b); w._edit = e  # type: ignore[attr-defined]
        return w

    def _dir_row(self, ph: str) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        e = QLineEdit(); e.setPlaceholderText(ph)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._set(e, QFileDialog.getExistingDirectory(self, "Folder")))
        h.addWidget(e); h.addWidget(b); w._edit = e  # type: ignore[attr-defined]
        return w

    def _set(self, edit: QLineEdit, text: str) -> None:
        if text:
            edit.setText(text)

    def _add_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Images", filter="Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)")
        self.inputs.addItems(files)

    def _add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder of images")
        if d:
            self.inputs.addItem(d)

    def _resolve_device(self) -> str:
        choice = self.device.currentText()
        if choice == "CPU":
            return "cpu"
        if choice == "CUDA":
            return "cuda"
        return "cuda" if detect().best_vram_gb > 0 else "cpu"

    def _run(self) -> None:
        model = self.model._edit.text()  # type: ignore[attr-defined]
        out = self.out._edit.text()  # type: ignore[attr-defined]
        inputs = [self.inputs.item(i).text() for i in range(self.inputs.count())]
        if not model or not out or not inputs:
            QMessageBox.warning(self, "Inference", "Pick a model, an output folder, and at least one input.")
            return
        tile = int(self.tile.currentText().split()[0])
        device = self._resolve_device()
        self.log.appendPlainText(
            f"Running {self.arch.currentText()} x{self.scale.currentText()} on "
            f"{len(inputs)} input(s) [{device}]…"
        )
        self.run_btn.setEnabled(False)
        self._worker = InferWorker(
            model, inputs, out, self.arch.currentText(),
            int(self.scale.currentText()), device, tile,
        )
        self._worker.line.connect(self.log.appendPlainText)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, wrote: int, errors: int) -> None:
        self.run_btn.setEnabled(True)
        self.log.appendPlainText(f"Done — {wrote} file(s), {errors} error(s).")
