"""Convert/export window — turn any BODY checkpoint into deployable formats.

Standalone counterpart of the post-training export: pick a checkpoint,
choose the architecture and scale, and get safetensors + ONNX (fp32/fp16)
and optionally ncnn (NERVE) written to a folder you choose.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..paths import get_paths


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


class ExportWindow(QMainWindow):
    """Convert a trained or downloaded checkpoint without touching the trainer."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Export / Convert — BODY Studio")
        self.resize(640, 380)
        self.paths = get_paths()
        self._worker: ExportWorker | None = None

        central = QWidget()
        root = QVBoxLayout(central)
        form = QFormLayout()

        mrow = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("checkpoint .safetensors / .pth …")
        self.model_edit.textChanged.connect(lambda _t: self._refresh_validity())
        mb = QPushButton("Browse…")
        mb.clicked.connect(self._browse_model)
        mrow.addWidget(self.model_edit); mrow.addWidget(mb)
        form.addRow("Checkpoint", mrow)

        self.arch = QComboBox()
        for a in ("NERVE", "HEART", "PULSE", "PULSE_fast"):
            self.arch.addItem(a)
        self.arch.currentTextChanged.connect(lambda _t: self._refresh_validity())
        form.addRow("Architecture", self.arch)

        self.scale = QComboBox()
        self.scale.addItems(["2", "4"])
        form.addRow("Scale", self.scale)

        orow = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setText(str(self.paths.data / "outputs"))
        ob = QPushButton("Browse…")
        ob.clicked.connect(self._browse_out)
        orow.addWidget(self.out_edit); orow.addWidget(ob)
        form.addRow("Output folder", orow)

        self.fp32 = QCheckBox("ONNX fp32")
        self.fp16 = QCheckBox("ONNX fp16")
        self.ncnn = QCheckBox("ncnn (NERVE only)")
        frow = QHBoxLayout()
        frow.addWidget(self.fp32); frow.addWidget(self.fp16); frow.addWidget(self.ncnn)
        form.addRow("Formats", frow)

        root.addLayout(form)
        self.export_btn = QPushButton("Export")
        self.export_btn.setObjectName("primary")
        self.export_btn.clicked.connect(self._export)
        root.addWidget(self.export_btn)
        self.prog = QProgressBar(); self.prog.setRange(0, 1); self.prog.setValue(0)
        root.addWidget(self.prog)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        root.addWidget(self.log, 1)
        self.setCentralWidget(central)
        self._refresh_validity()

    # ---- validity and actions ----
    def _refresh_validity(self) -> None:
        ck = Path(self.model_edit.text().strip())
        self.export_btn.setEnabled(ck.is_file()
                                   and getattr(self, "_worker", None) is None)
        self.ncnn.setEnabled(self.arch.currentText() == "NERVE")

    def _browse_model(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Checkpoint", filter="Models (*.safetensors *.pth)")
        if p:
            self.model_edit.setText(p)
            self._refresh_validity()

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder")
        if d:
            self.out_edit.setText(d)

    def _export(self) -> None:
        from ..bootstrap import runtime_python

        ck = Path(self.model_edit.text().strip())
        out = Path(self.out_edit.text().strip() or (self.paths.data / "outputs"))
        if not ck.is_file():
            return
        self._worker = ExportWorker(
            self.paths.repo, ck, out, self.arch.currentText(),
            int(self.scale.currentText()),
            fp32=self.fp32.isChecked(), fp16=self.fp16.isChecked(),
            ncnn=self.ncnn.isEnabled() and self.ncnn.isChecked(),
            python_exe=runtime_python(self.paths),
        )
        self._worker.line.connect(self.log.appendPlainText)
        self._worker.done.connect(self._on_done)
        self.prog.setRange(0, 0)  # busy
        self.export_btn.setEnabled(False)
        self.log.appendPlainText(f"exporting {ck.name} → {out}")
        self._worker.start()

    def _on_done(self, out_dir: str, ok: bool) -> None:
        self._worker = None
        self.prog.setRange(0, 1)
        self.prog.setValue(1 if ok else 0)
        if ok:
            self.log.appendPlainText(f"Done — model saved to {out_dir}")
        else:
            self.log.appendPlainText("Export had problems — see the messages above.")
        self._refresh_validity()
