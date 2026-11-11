"""A/B viewer: input vs output with a draggable split, synced zoom and pan.

BEGINNER-FIRST: mouse wheel zooms both sides together (anchored at the
cursor), dragging pans, double-click or the Reset button returns to fit.
Images can also be dropped directly onto the window.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".avif"}
MIN_ZOOM = 0.1
MAX_ZOOM = 8.0


class _Pane(QGraphicsView):
    """One comparison side; zoom/pan events are forwarded to the owner window."""

    def __init__(self, win: ViewerWindow) -> None:
        super().__init__()
        self._win = win
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def set_pixmap(self, pm: QPixmap) -> None:
        self.item.setPixmap(pm)
        self.setSceneRect(self.item.sceneBoundingRect())
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self._win._apply_zoom(factor)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self._win.reset_view()
        event.accept()


class ViewerWindow(QMainWindow):
    def __init__(self, input_path: str | Path, output_path: str | Path) -> None:
        super().__init__()
        self.setWindowTitle("A/B viewer — BODY Studio")
        self.resize(1100, 720)

        self._zoom_factor = 1.0

        self.splitter = QSplitter(Qt.Horizontal)
        self.left = _Pane(self)
        self.right = _Pane(self)
        self.splitter.addWidget(self.left)
        self.splitter.addWidget(self.right)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self._on_slider)
        self._split = 50

        self._input_pm = QPixmap()
        self._output_pm = QPixmap()
        self.input_path: Path | None = None
        self.output_path: Path | None = None
        self.set_pair(input_path, output_path)
        self.setAcceptDrops(True)

        self.swap_btn = QPushButton("⇄ Swap sides")
        self.swap_btn.clicked.connect(self._swap)
        self.reset_btn = QPushButton("Reset zoom")
        self.reset_btn.clicked.connect(self.reset_view)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.splitter, 1)
        layout.addWidget(self.slider)
        row = QVBoxLayout()
        row.addWidget(self.swap_btn)
        row.addWidget(self.reset_btn)
        layout.addLayout(row)
        self.setCentralWidget(central)
        self._on_slider()

    def set_pair(self, input_path: str | Path, output_path: str | Path) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self._input_pm = QPixmap(str(self.input_path))
        self._output_pm = QPixmap(str(self.output_path))
        self.left.set_pixmap(self._input_pm)
        self.right.set_pixmap(self._output_pm)
        self.reset_view()

    def _swap(self) -> None:
        if self.input_path is None or self.output_path is None:
            return
        self.set_pair(self.output_path, self.input_path)
        self._split = 100 - self._split
        self.slider.blockSignals(True)
        self.slider.setValue(self._split)
        self.slider.blockSignals(False)
        self._on_slider()

    def _apply_zoom(self, factor: float) -> None:
        self._zoom_factor = float(min(MAX_ZOOM, max(MIN_ZOOM, self._zoom_factor * factor)))
        for pane in (self.left, self.right):
            pane.resetTransform()
            pane.scale(self._zoom_factor, self._zoom_factor)
            pane.centerOn(pane.sceneRect().center())

    def reset_view(self) -> None:
        self._zoom_factor = 1.0
        for pane in (self.left, self.right):
            pane.resetTransform()
            pane.fitInView(pane.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ---- drag and drop ----
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self._add_dropped(paths)

    def _add_dropped(self, paths: list[Path]) -> None:
        imgs = [p for p in paths if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not imgs:
            return
        if len(imgs) == 1:
            self.set_pair(imgs[0], self.output_path or imgs[0])
        else:
            self.set_pair(imgs[0], imgs[1])

    def _on_slider(self) -> None:
        v = self.slider.value()
        self._split = v
        self.splitter.setSizes([v, 100 - v])
