"""Minimal A/B viewer: input vs output with a draggable split."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow, QSlider, QSplitter, QVBoxLayout, QWidget


class ViewerWindow(QMainWindow):
    def __init__(self, input_path: str | Path, output_path: str | Path) -> None:
        super().__init__()
        self.setWindowTitle("A/B viewer — SISR BODY Studio")
        self.resize(1100, 720)

        self.splitter = QSplitter(Qt.Horizontal)
        self.left = QLabel()
        self.right = QLabel()
        for w in (self.left, self.right):
            w.setAlignment(Qt.AlignCenter)
            self.splitter.addWidget(w)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self._on_slider)
        self._split = 50
        self._input_pm = QPixmap(str(input_path))
        self._output_pm = QPixmap(str(output_path))

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.splitter, 1)
        layout.addWidget(self.slider)
        self.setCentralWidget(central)

        self.left.setPixmap(self._input_pm.scaled(
            self.left.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self.right.setPixmap(self._output_pm.scaled(
            self.right.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self._on_slider()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._set_images()

    def _set_images(self) -> None:
        """Keep both images fitted when the window changes size."""
        if self._input_pm.isNull() or self._output_pm.isNull():
            return
        self.left.setPixmap(self._input_pm.scaled(
            self.left.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self.right.setPixmap(self._output_pm.scaled(
            self.right.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _on_slider(self) -> None:
        v = self.slider.value()
        self._split = v
        self.splitter.setSizes([v, 100 - v])
