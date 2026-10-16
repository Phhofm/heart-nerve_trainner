"""Live validation-metric chart (pyqtgraph).

Two stacked panels because the metrics live on different scales and directions:
fidelity metrics (PSNR/SSIM, higher is better) and perceptual metrics
(DISTS/LPIPS/TOPIQ, lower is better). Squeezing all of them onto one y-axis
makes everything but PSNR flat, so they get their own panels.

Each panel starts in an *empty state* (no axes/grid, a friendly hint) so the app
never shows the meaningless default 0–1 grid before the first validation runs.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from . import theme

# metric name → line colour (falls back to muted for anything unknown)
_COLORS = {
    "psnr": theme.ACCENT,
    "ssim": theme.OK,
    "dists": theme.WARN,
    "lpips": theme.ERR,
    "topiq": theme.ACCENT2,
}

# panel title → metrics shown in it
_GROUPS: dict[str, tuple[str, ...]] = {
    "Fidelity  (higher is better)": ("psnr", "ssim"),
    "Perceptual  (lower is better)": ("dists", "lpips", "topiq"),
}


class MetricsChart(QWidget):
    """Plots PSNR/SSIM/DISTS/LPIPS as training validates."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption("foreground", theme.MUTED)

        self._plots: dict[str, pg.PlotWidget] = {}
        self._sub_plots: dict[str, pg.PlotItem] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._xs: dict[str, list[int]] = {}
        self._ys: dict[str, list[float]] = {}
        self._panel: dict[str, str] = {}  # metric name → panel title
        self._hint: dict[str, QLabel] = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        for title, names in _GROUPS.items():
            holder = QFrame()
            holder.setObjectName("chartpanel")
            grid = QGridLayout(holder)
            grid.setContentsMargins(12, 10, 12, 12)

            plot = pg.PlotWidget()
            plot.setBackground(None)  # transparent → the framed panel shows through
            plot.setMinimumHeight(150)
            plot.setTitle(title, color=theme.MUTED, size="10pt")
            plot.setLabel("bottom", "iteration")
            plot.setMenuEnabled(False)
            plot.addLegend(offset=(10, 10))
            grid.addWidget(plot, 0, 0)

            hint = QLabel("Validation metrics will appear during training")
            hint.setObjectName("muted")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(hint, 0, 0)

            self._set_empty(plot, True)
            lay.addWidget(holder, 1)

            self._plots[title] = plot
            self._sub_plots[title] = plot.getPlotItem()
            self._hint[title] = hint
            for name in names:
                self._panel[name] = title

    @staticmethod
    def _set_empty(plot: pg.PlotWidget, empty: bool) -> None:
        if empty:
            plot.hideAxis("bottom")
            plot.hideAxis("left")
            plot.showGrid(x=False, y=False)
        else:
            plot.showAxis("bottom")
            plot.showAxis("left")
            plot.showGrid(x=True, y=True, alpha=0.12)

    def add_point(self, name: str, x: int, value: float) -> None:
        """Append one validation sample and redraw that metric's line."""
        name = name.lower()
        panel = self._panel.get(name, next(iter(_GROUPS)))
        if name not in self._curves:
            self._curves[name] = self._plots[panel].plot(
                pen=pg.mkPen(_COLORS.get(name, theme.MUTED), width=2), name=name.upper()
            )
            self._xs[name], self._ys[name] = [], []
            self._hint[panel].hide()
            self._set_empty(self._plots[panel], False)
        self._xs[name].append(x)
        self._ys[name].append(value)
        self._curves[name].setData(self._xs[name], self._ys[name])

    def clear(self) -> None:
        for name, curve in self._curves.items():
            self._sub_plots[self._panel.get(name, next(iter(_GROUPS)))].removeItem(curve)
        self._curves.clear()
        self._xs.clear()
        self._ys.clear()
        for title, plot in self._plots.items():
            legend = self._sub_plots[title].legend
            if legend is not None:
                legend.clear()
            self._set_empty(plot, True)
            self._hint[title].show()
