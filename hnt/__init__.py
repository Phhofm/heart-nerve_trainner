"""HEART / NERVE Trainer — a beginner-first GUI for training SISR models.

The app is a thin shell over the `traiNNer-redux` training harness: it detects
hardware, bootstraps an isolated venv, generates a training config, runs
`train.py`, streams progress, and exports deployable models.

This package is import-safe without a GPU and without PySide6 (the UI imports
PySide6 lazily), so the core is unit-testable in CI/headless.
"""

from .version import __version__

__all__ = ["__version__"]
