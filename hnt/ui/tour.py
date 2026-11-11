"""First-run quick tour — a one-screen "what do I do" for absolute beginners.

Shown automatically once (marker file in the app data dir) when the training
environment isn't set up yet, and always available via the "Quick tour" button.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtWidgets import QMessageBox, QWidget

TOUR_STEPS = (
    ("1. Set up", ("Press <b>Set up / repair</b> once — it installs PyTorch in the "
                   "app's own folder (~2–3&nbsp;GB). Your system Python is never touched.")),
    ("2. Point at your images", ("Pick an <b>HR folder</b> of good-quality images, or "
                                 "download the free CC0 Lucid dataset.")),
    ("3. Press Train", ("Keep the recommended defaults (<b>NERVE · 2× · Fidelity</b>). "
                        "Watch the graphs; the finished model is exported automatically.")),
    ("4. Use your model", ("Upscale images or videos with the model you just trained "
                           "(<b>Inference…</b> / <b>Video…</b>), or take the exported "
                           "ONNX/safetensors anywhere.")),
)

def marker_path() -> Path:
    from ..paths import get_paths

    return get_paths().data / "ui" / "tour_seen.json"


def should_show(marker: Path) -> bool:
    return not Path(marker).is_file()


def mark_seen(marker: Path) -> None:
    marker = Path(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"seen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}))


def show_tour(parent: QWidget | None, marker: Path | None = None) -> None:
    body = "<br><br>".join(f"<b>{t}.</b> {d}" for t, d in TOUR_STEPS)
    QMessageBox.information(
        parent, "BODY Studio — quick tour",
        "Welcome! Four steps, in order:<br><br>" + body
        + "<br><br>Press <b>Train</b> when you're ready — the app always shows "
          "the next step in the hint bar above the log.")
    mark_seen(marker or marker_path())
