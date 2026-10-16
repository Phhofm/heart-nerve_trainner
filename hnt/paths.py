"""Locate the traiNNer-redux repo and the app's data/output directories."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .version import APP_SLUG

# Environment override wins (used by tests + power users).
ENV_REPO = "HNT_TRAINER_REPO"
ENV_DATA = "HNT_DATA_DIR"


def _looks_like_repo(p: Path) -> bool:
    return (p / "train.py").is_file() and (p / "options").is_dir()


def find_repo() -> Path:
    """Find the traiNNer-redux checkout that contains train.py.

    Order: $HNT_TRAINER_REPO → walk up from this file → bundled ``trainer/``
    next to the app (for packaged builds) → cwd walk.
    """
    env = os.environ.get(ENV_REPO)
    if env and _looks_like_repo(Path(env)):
        return Path(env).resolve()

    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if _looks_like_repo(parent):
            return parent
        bundled = parent / "trainer"
        if _looks_like_repo(bundled):
            return bundled.resolve()

    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if _looks_like_repo(parent):
            return parent

    raise FileNotFoundError(
        "Could not find the traiNNer-redux repo (train.py + options/). "
        f"Set ${ENV_REPO} to the checkout path."
    )


def default_data_dir() -> Path:
    """Per-user app data dir (runs, datasets, downloads, logs)."""
    env = os.environ.get(ENV_DATA)
    if env:
        base = Path(env)
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / APP_SLUG
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support" / APP_SLUG
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / APP_SLUG
    base.mkdir(parents=True, exist_ok=True)
    return base


@dataclass(frozen=True)
class Paths:
    repo: Path
    data: Path

    @property
    def runs(self) -> Path:
        return self.data / "runs"

    @property
    def datasets(self) -> Path:
        return self.data / "datasets"

    @property
    def logs(self) -> Path:
        return self.data / "logs"

    @property
    def venv(self) -> Path:
        return self.data / "venv"

    def ensure(self) -> "Paths":
        for p in (self.runs, self.datasets, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    return Paths(repo=find_repo(), data=default_data_dir()).ensure()
