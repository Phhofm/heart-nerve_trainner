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
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

from ..bootstrap import app_source_dir, runtime_python
from ..catalog import CatalogEntry, load_catalog, resolve_model
from ..hardware import detect
from ..infer import IMG_EXTS
from ..paths import get_paths
from ..presets import SCALES


class InferWorker(QThread):
    """Run inference in the isolated venv (works in frozen builds, never freezes the UI)."""

    line = Signal(str)
    done = Signal(int, int, list)  # (wrote, errors, output paths)

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
            self.done.emit(0, 1, [])
            return
        for ln in (p.stdout + p.stderr).splitlines():
            self.line.emit(ln)
        wrote = 0
        outputs: list[str] = []
        for ln in p.stdout.splitlines():
            if ln.startswith("output "):
                outputs.append(ln[len("output "):].strip())
            elif ln.startswith("wrote "):
                try:
                    wrote = int(ln.split()[1])
                except (IndexError, ValueError):
                    pass
        self.done.emit(wrote, 0 if p.returncode == 0 else 1, outputs)


class CatalogWorker(QThread):
    """Fetch a catalog model without freezing the interface."""

    done = Signal(str, str)  # (model path, error)

    def __init__(self, entry: CatalogEntry, cache: Path):
        super().__init__()
        self.entry, self.cache = entry, cache

    def run(self) -> None:
        try:
            self.done.emit(str(resolve_model(self.entry, self.cache)), "")
        except Exception as e:  # noqa: BLE001
            self.done.emit("", repr(e))


class InferenceWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.paths = get_paths()
        self.setWindowTitle("Inference — SISR BODY Studio")
        self.resize(800, 620)

        root = QVBoxLayout()
        central = QWidget(); central.setLayout(root); self.setCentralWidget(central)

        box = QGroupBox("Run inference")
        form = QFormLayout(box)
        self.catalog = QComboBox()
        self._catalog_entries = {e.id: e for e in load_catalog()}
        for eid, e in self._catalog_entries.items():
            self.catalog.addItem(e.name, eid)
        preferred = "nerve_2x"
        idx = self.catalog.findData(preferred)
        if idx >= 0:
            self.catalog.setCurrentIndex(idx)
        else:
            self.catalog.setCurrentIndex(0)
        self.catalog.currentIndexChanged.connect(self._on_catalog)
        self.info = QLabel("")
        self.info.setWordWrap(True)

        self.model = self._file_row("(optional) custom .safetensors/.pth …")
        self.arch = QComboBox(); self.arch.addItems(["nerve", "heart", "pulse", "pulse_fast"])
        self.scale = QComboBox(); self.scale.addItems([str(s) for s in SCALES]); self.scale.setCurrentText("2")
        self.device = QComboBox(); self.device.addItems(["Auto", "CUDA", "CPU"])
        self.tile = QComboBox()
        for label, value in (("Auto (recommended)", 0), ("256", 256), ("512", 512),
                             ("Whole image (may OOM)", -1)):
            self.tile.addItem(label, value)

        self.inputs = QListWidget()
        irow = QHBoxLayout()
        for label, fn in (("Add images…", self._add_images),
                          ("Add folder…", self._add_folder),
                          ("Clear", self.inputs.clear)):
            b = QPushButton(label); b.clicked.connect(fn); irow.addWidget(b)
        inwrap = QWidget(); inwrap.setLayout(irow)

        self.out = self._dir_row("output folder …")

        form.addRow("Catalog model", self.catalog)
        use = QPushButton("Download / use catalog model")
        use.setObjectName("primary")
        use.clicked.connect(self._use_catalog)
        form.addRow("", use)
        self.use_catalog = use
        form.addRow("", self.info)
        form.addRow("Custom model", self.model)
        form.addRow("Architecture", self.arch)
        form.addRow("Scale", self.scale)
        form.addRow("Device", self.device)
        form.addRow("Big image tiling", self.tile)
        form.addRow("Input", self.inputs)
        form.addRow("", inwrap)
        form.addRow("Output folder", self.out)
        orow = QHBoxLayout()
        self.open_out = QPushButton("Open output folder")
        self.open_out.clicked.connect(self._open_output)
        self.view_ab = QPushButton("A/B viewer")
        self.view_ab.clicked.connect(self._open_viewer)
        orow.addWidget(self.open_out); orow.addWidget(self.view_ab)
        form.addRow("", orow)
        self.open_out.setEnabled(False); self.view_ab.setEnabled(False)
        run = QPushButton("Run"); run.setObjectName("primary"); run.clicked.connect(self._run)
        form.addRow(run)
        self.run_btn = run
        root.addWidget(box)

        self.outputs = QListWidget(); self.outputs.setObjectName("outputlist")
        self.outputs.setToolTip("Upscaled images produced by this run")
        root.addWidget(self.outputs, 1)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        root.addWidget(self.log, 1)
        self.prog = QProgressBar(); self.prog.setRange(0, 1); self.prog.setValue(0)
        root.addWidget(self.prog)
        self._outputs: list[Path] = []
        self._on_catalog()

    # ---- catalog ----
    def _use_catalog(self) -> None:
        eid = self.catalog.currentData()
        e = self._catalog_entries.get(eid)
        if not e:
            QMessageBox.warning(self, "Catalog", "No catalog model selected.")
            return
        if getattr(self, "_catalog_worker", None) is not None and self._catalog_worker.isRunning():
            return
        cache = self.paths.data / "models" / e.id
        self.use_catalog.setEnabled(False)
        self.use_catalog.setText("Downloading model…")
        self.info.setText(f"Downloading {e.name}…")
        self.log.appendPlainText(f"Downloading catalog model → {cache}")
        self._catalog_worker = CatalogWorker(e, cache)
        self._catalog_worker.done.connect(self._on_catalog_done)
        self._catalog_worker.start()

    def _on_catalog_done(self, model_path: str, error: str) -> None:
        self.use_catalog.setEnabled(True)
        self.use_catalog.setText("Download / use catalog model")
        if error:
            self.info.setText("Catalog download failed. See the log for details.")
            self.log.appendPlainText(f"ERROR: {error}")
            QMessageBox.warning(self, "Catalog", "Could not download the selected model.")
            return
        self.model._edit.setText(model_path)  # type: ignore[attr-defined]
        self.info.setText(f"Model ready → {model_path}")
        self.log.appendPlainText(f"Model ready → {model_path}")

    def _on_catalog(self) -> None:
        eid = self.catalog.currentData()
        e = self._catalog_entries.get(eid)
        if e:
            self.arch.setCurrentText(e.arch)
            self.scale.setCurrentText(str(e.scale))
            self.info.setText(
                f"{e.params:,} parameters · x{e.scale} · {e.license} · {e.vram_hint} VRAM\n{e.description}"
            )

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
        tile = int(self.tile.currentData())
        device = self._resolve_device()
        if not runtime_python(get_paths()):
            self.log.appendPlainText(
                "The training environment is not set up yet — use 'Set up / repair' "
                "in the main window, then try again."
            )
            return
        self.log.appendPlainText(
            f"Running {self.arch.currentText()} x{self.scale.currentText()} on "
            f"{len(inputs)} input(s) [{device}]…"
        )
        self.run_btn.setEnabled(False)
        self.prog.setRange(0, 0)  # busy
        self._worker = InferWorker(
            model, inputs, out, self.arch.currentText(),
            int(self.scale.currentText()), device, tile,
        )
        self._worker.line.connect(self.log.appendPlainText)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, wrote: int, errors: int, outputs: list[str]) -> None:
        self.run_btn.setEnabled(True)
        self.prog.setRange(0, 1)
        self.prog.setValue(1 if wrote else 0)
        self._outputs = [Path(p) for p in outputs]
        self.outputs.clear()
        self.outputs.addItems([str(p) for p in self._outputs])
        has_output = bool(self._outputs)
        self.open_out.setEnabled(has_output)
        self.view_ab.setEnabled(has_output)
        self.log.appendPlainText(f"Done — {wrote} file(s), {errors} error(s).")

    def _open_output(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        out = self.out._edit.text().strip()  # type: ignore[attr-defined]
        if not out:
            return
        Path(out).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(out))

    def _first_input(self) -> Path | None:
        for i in range(self.inputs.count()):
            p = Path(self.inputs.item(i).text())
            if p.is_file():
                return p
            if p.is_dir():
                found = sorted((f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS),
                               key=lambda f: f.name.lower())
                if found:
                    return found[0]
        return None

    def _open_viewer(self) -> None:
        if not self._outputs:
            QMessageBox.warning(self, "A/B viewer", "Run inference first.")
            return
        source = self._first_input()
        if source is None:
            QMessageBox.warning(self, "A/B viewer", "No input image is available to compare.")
            return
        from .viewer import ViewerWindow

        output = self._outputs[0] if len(self._outputs) == 1 else self.outputs.currentItem()
        output_path = Path(output.text()) if output is not None else self._outputs[0]
        self._viewer = ViewerWindow(source, output_path)
        self._viewer.show()
