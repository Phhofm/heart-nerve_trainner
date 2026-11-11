"""Video inference window for BODY Studio.

This window reuses the headless ``hnt.cli video`` command so video jobs run in
the isolated training environment and stream progress back to the UI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..bootstrap import app_source_dir, runtime_python
from ..catalog import load_catalog
from ..hardware import detect
from ..paths import get_paths
from ..presets import SCALES
from .inference import CatalogWorker


class VideoWorker(QThread):
    """Run a video enhancement job in the isolated venv."""

    line = Signal(str)
    done = Signal(bool, str)

    def __init__(
        self,
        model: str,
        input_path: str,
        output_path: str,
        arch: str,
        scale: int,
        device: str,
        tile: int,
        codec: str,
        crf: int,
        overwrite: bool,
        deinterlace: bool,
    ):
        super().__init__()
        self.model = model
        self.input_path = input_path
        self.output_path = output_path
        self.arch = arch
        self.scale = scale
        self.device = device
        self.tile = tile
        self.codec = codec
        self.crf = crf
        self.overwrite = overwrite
        self.deinterlace = deinterlace
        self.proc: subprocess.Popen | None = None

    def run(self) -> None:
        src = app_source_dir()
        cmd = [
            runtime_python(get_paths()),
            "-m",
            "hnt.cli",
            "video",
            "--model",
            self.model,
            "--arch",
            self.arch,
            "--scale",
            str(self.scale),
            "--input",
            self.input_path,
            "--out",
            self.output_path,
            "--device",
            self.device,
            "--tile",
            str(self.tile),
            "--codec",
            self.codec,
            "--crf",
            str(self.crf),
        ]
        if self.overwrite:
            cmd.append("--overwrite")
        if self.deinterlace:
            cmd.append("--deinterlace")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(src),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:  # noqa: BLE001
            self.line.emit(f"ERROR: {e}")
            self.done.emit(False, "")
            return

        output = ""
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("output "):
                output = line[len("output "):].strip()
            self.line.emit(line)
        if self.proc:
            self.proc.wait()
        self.done.emit(self.proc.returncode == 0 if self.proc else False, output)

    def stop(self) -> None:
        proc = self.proc
        if proc and proc.poll() is None:
            proc.terminate()


class VideoWindow(QMainWindow):
    """Offline video enhancement window for BODY models."""

    def __init__(self) -> None:
        super().__init__()
        self.paths = get_paths()
        self.setWindowTitle("Video inference — BODY Studio")
        self.resize(860, 680)

        root = QVBoxLayout()
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        box = QGroupBox("Enhance video")
        form = QFormLayout(box)
        self.catalog = QComboBox()
        self._catalog_entries = {e.id: e for e in load_catalog()}
        for eid, entry in self._catalog_entries.items():
            self.catalog.addItem(entry.name, eid)
        preferred = "nerve_2x"
        idx = self.catalog.findData(preferred)
        self.catalog.setCurrentIndex(max(idx, 0))
        self.catalog.currentIndexChanged.connect(self._on_catalog)
        self.info = QLabel("")
        self.info.setWordWrap(True)

        self.model = self._file_row("(optional) custom .safetensors/.pth …")
        self.arch = QComboBox()
        self.arch.addItems(["nerve", "heart", "pulse", "pulse_fast"])
        self.scale = QComboBox()
        self.scale.addItems([str(s) for s in SCALES])
        self.scale.setCurrentText("2")
        self.device = QComboBox()
        self.device.addItems(["Auto", "CUDA", "CPU"])
        self.tile = QComboBox()
        for label, value in (
            ("Auto (recommended)", 0),
            ("256", 256),
            ("512", 512),
            ("Whole frame (may OOM)", -1),
        ):
            self.tile.addItem(label, value)
        self.input = self._input_row()
        self.out = self._output_row()
        self.codec = QComboBox()
        self.codec.addItems(["libx264", "libx265"])
        self.crf = QSpinBox()
        self.crf.setRange(0, 51)
        self.crf.setValue(18)
        self.overwrite = QCheckBox("Overwrite output file if it exists")
        self.deinterlace = QCheckBox("Deinterlace input")

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
        form.addRow("Frame tiling", self.tile)
        form.addRow("Input video", self.input)
        form.addRow("Output video", self.out)
        form.addRow("Codec", self.codec)
        form.addRow("CRF", self.crf)
        form.addRow("", self.deinterlace)
        form.addRow("", self.overwrite)
        run = QPushButton("Run")
        run.setObjectName("primary")
        run.clicked.connect(self._run)
        form.addRow(run)
        self.run_btn = run
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self._cancel)
        form.addRow(cancel)
        self.cancel_btn = cancel
        self.cancel_btn.setEnabled(False)
        root.addWidget(box)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)
        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        root.addWidget(self.prog)

        orow = QHBoxLayout()
        self.open_out = QPushButton("Open output folder")
        self.open_out.setEnabled(False)
        self.open_out.clicked.connect(self._open_output)
        orow.addWidget(self.open_out)
        root.addLayout(orow)

        self._on_catalog()
        if not self._ffmpeg_available():
            self.info.setText(
                self.info.text()
                + "\nffmpeg/ffprobe not found. Install ffmpeg to enable video inference."
            )
            self.run_btn.setEnabled(False)
        self.setAcceptDrops(True)

    # ---- drag and drop ----
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self._add_dropped(paths)

    def _add_dropped(self, paths: list[Path]) -> None:
        video_exts = {
            ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
            ".mpg", ".mpeg", ".ts", ".mts",
        }
        for path in paths:
            if path.is_file() and path.suffix.lower() in video_exts:
                self.input._edit.setText(str(path))  # type: ignore[attr-defined]
                return

    # ---- helpers ----
    @staticmethod
    def _ffmpeg_available() -> bool:
        return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

    def _file_row(self, placeholder: str) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        browse = QPushButton("Browse…")
        browse.clicked.connect(
            lambda: self._set(
                edit,
                QFileDialog.getOpenFileName(
                    self, "Model", filter="Models (*.safetensors *.pth)"
                )[0],
            )
        )
        h.addWidget(edit)
        h.addWidget(browse)
        w._edit = edit  # type: ignore[attr-defined]
        return w

    def _input_row(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText("Choose an input video …")
        browse = QPushButton("Browse…")
        browse.clicked.connect(
            lambda: self._set(
                edit,
                QFileDialog.getOpenFileName(
                    self,
                    "Input video",
                    filter="Videos (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mpg *.mpeg *.ts *.mts)",
                )[0],
            )
        )
        h.addWidget(edit)
        h.addWidget(browse)
        w._edit = edit  # type: ignore[attr-defined]
        return w

    def _output_row(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText("Choose an output video …")
        browse = QPushButton("Browse…")
        browse.clicked.connect(
            lambda: self._set(
                edit,
                QFileDialog.getSaveFileName(
                    self,
                    "Output video",
                    filter="Videos (*.mp4 *.mkv *.mov *.avi *.webm *.m4v)",
                )[0],
            )
        )
        h.addWidget(edit)
        h.addWidget(browse)
        w._edit = edit  # type: ignore[attr-defined]
        return w

    def _set(self, edit: QLineEdit, text: str) -> None:
        if text:
            edit.setText(text)

    def _resolve_device(self) -> str:
        choice = self.device.currentText()
        if choice == "CPU":
            return "cpu"
        if choice == "CUDA":
            return "cuda"
        return "cuda" if detect().best_vram_gb > 0 else "cpu"

    # ---- catalog ----
    def _use_catalog(self) -> None:
        eid = self.catalog.currentData()
        entry = self._catalog_entries.get(eid)
        if not entry:
            QMessageBox.warning(self, "Catalog", "No catalog model selected.")
            return
        if getattr(self, "_catalog_worker", None) is not None and self._catalog_worker.isRunning():
            return
        cache = self.paths.data / "models" / entry.id
        self.use_catalog.setEnabled(False)
        self.use_catalog.setText("Downloading model…")
        self.info.setText(f"Downloading {entry.name}…")
        self.log.appendPlainText(f"Downloading catalog model → {cache}")
        self._catalog_worker = CatalogWorker(entry, cache)
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
        entry = self._catalog_entries.get(eid)
        if entry:
            self.arch.setCurrentText(entry.arch)
            self.scale.setCurrentText(str(entry.scale))
            self.info.setText(
                f"{entry.params:,} parameters · x{entry.scale} · {entry.license} · "
                f"{entry.vram_hint} VRAM\n{entry.description}"
            )

    # ---- job handling ----
    def _run(self) -> None:
        model = self.model._edit.text().strip()  # type: ignore[attr-defined]
        input_path = self.input._edit.text().strip()  # type: ignore[attr-defined]
        output_path = self.out._edit.text().strip()  # type: ignore[attr-defined]
        if not model or not input_path or not output_path:
            QMessageBox.warning(
                self,
                "Video inference",
                "Pick a model, an input video, and an output video.",
            )
            return
        if not self._ffmpeg_available():
            QMessageBox.warning(
                self,
                "Video inference",
                "ffmpeg/ffprobe is not installed. Install ffmpeg and try again.",
            )
            return
        if not runtime_python(get_paths()):
            self.log.appendPlainText(
                "The training environment is not set up yet — use 'Set up / repair' "
                "in the main window, then try again."
            )
            return
        if Path(input_path) == Path(output_path):
            QMessageBox.warning(
                self,
                "Video inference",
                "Input and output videos must be different files.",
            )
            return
        if not self.overwrite.isChecked() and Path(output_path).exists():
            QMessageBox.warning(
                self,
                "Video inference",
                "The output file already exists. Enable overwrite or choose another name.",
            )
            return

        self.log.appendPlainText(
            f"Running {self.arch.currentText()} x{self.scale.currentText()} on {input_path}…"
        )
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.prog.setRange(0, 0)
        self.open_out.setEnabled(False)
        self._worker = VideoWorker(
            model,
            input_path,
            output_path,
            self.arch.currentText(),
            int(self.scale.currentText()),
            self._resolve_device(),
            int(self.tile.currentData()),
            self.codec.currentText(),
            int(self.crf.value()),
            self.overwrite.isChecked(),
            self.deinterlace.isChecked(),
        )
        self._worker.line.connect(self._on_line)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _cancel(self) -> None:
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.isRunning():
            self.log.appendPlainText("Cancelling video job…")
            worker.stop()

    def _on_line(self, line: str) -> None:
        self.log.appendPlainText(line)
        if line.startswith("frame "):
            parts = line.split()
            if len(parts) >= 2 and "/" in parts[1]:
                done_s, total_s = parts[1].split("/", 1)
                try:
                    done = int(done_s)
                    total = int(total_s) if total_s.isdigit() else None
                except ValueError:
                    return
                if total:
                    self.prog.setRange(0, total)
                    self.prog.setValue(done)

    def _on_done(self, ok: bool, output: str) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.prog.setRange(0, 100)
        self.prog.setValue(100 if ok else 0)
        if ok and output:
            self.open_out.setEnabled(True)
            self.log.appendPlainText(f"Done — {output}")
        else:
            self.log.appendPlainText("Video inference failed. See the log for details.")
            QMessageBox.warning(
                self,
                "Video inference",
                "Could not enhance the video. Check the log for details.",
            )

    def _open_output(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        out = self.out._edit.text().strip()  # type: ignore[attr-defined]
        if not out:
            return
        path = Path(out)
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
