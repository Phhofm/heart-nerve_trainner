"""Standalone inference window — run a trained or downloaded HEART/NERVE model
on single images or whole folders. Outputs land in a chosen folder, suffixed with
the model name.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from ..infer import IMG_EXTS, infer_paths
from ..paths import get_paths
from ..presets import SCALES


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
        form.addRow("Big image tiling", self.tile)
        form.addRow("Input", self.inputs)
        form.addRow("", inwrap)
        form.addRow("Output folder", self.out)
        run = QPushButton("Run"); run.setObjectName("primary"); run.clicked.connect(self._run)
        form.addRow(run)
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

    def _run(self) -> None:
        model = self.model._edit.text()  # type: ignore[attr-defined]
        out = self.out._edit.text()  # type: ignore[attr-defined]
        inputs = [self.inputs.item(i).text() for i in range(self.inputs.count())]
        if not model or not out or not inputs:
            QMessageBox.warning(self, "Inference", "Pick a model, an output folder, and at least one input.")
            return
        tile = int(self.tile.currentText().split()[0])
        self.log.appendPlainText(f"Running {self.arch.currentText()} x{self.scale.currentText()} on {len(inputs)} input(s)…")
        try:
            res = infer_paths(model, inputs, out, arch=self.arch.currentText(),
                              scale=int(self.scale.currentText()), tile=tile)
        except Exception as e:  # noqa: BLE001
            self.log.appendPlainText(f"ERROR: {e}")
            return
        for p in res.outputs:
            self.log.appendPlainText(f"  wrote {p}")
        for e in res.errors:
            self.log.appendPlainText(f"  ERR {e}")
        self.log.appendPlainText(f"Done — {len(res.outputs)} file(s).")
