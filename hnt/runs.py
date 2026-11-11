"""Run discovery — list training runs, checkpoints, and resumable state.

BEGINNER-FIRST: "what did I train, what can I continue?" is answered from the
traiNNer-redux experiment layout without parsing any YAML:

    experiments/<name>/models/net_g[_ema]_<iter>.safetensors
    experiments/<name>/training_states/<iter>.state
    experiments/<name>/<name>.yml            (config copy made by the trainer)
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

_CKPT = re.compile(r"^net_g(?:_ema)?_(\d+)\.(?:safetensors|pth)$")
_STATE = re.compile(r"^(\d+)\.state$")
_RUN_NAME = re.compile(r"^(HEART|NERVE|PULSE|PULSE_fast)_(\d+)x", re.IGNORECASE)


@dataclass
class Checkpoint:
    iter: int
    path: Path
    ema: bool


@dataclass
class RunInfo:
    name: str
    path: Path
    config: Path | None = None
    checkpoints: list[Checkpoint] = field(default_factory=list)
    states: list[int] = field(default_factory=list)

    @property
    def latest_checkpoint(self) -> Checkpoint | None:
        return self.checkpoints[-1] if self.checkpoints else None

    @property
    def latest_state(self) -> Path | None:
        return self.path / "training_states" / f"{self.states[-1]}.state" if self.states else None

    @property
    def resumable(self) -> bool:
        return bool(self.states)


def _checkpoints(models: Path) -> list[Checkpoint]:
    found: dict[tuple[int, bool], Checkpoint] = {}
    for p in models.glob("net_g_*"):
        m = _CKPT.match(p.name)
        if m:
            found[(int(m.group(1)), p.name.startswith("net_g_ema_"))] = Checkpoint(
                int(m.group(1)), p, p.name.startswith("net_g_ema_"))
    out = [found[key] for key in sorted(found, key=lambda k: (k[0], not k[1]))]
    # one entry per iter: EMA preferred over plain
    by_iter: dict[int, Checkpoint] = {}
    for c in out:
        prev = by_iter.get(c.iter)
        if prev is None or (c.ema and not prev.ema):
            by_iter[c.iter] = c
    return [by_iter[i] for i in sorted(by_iter)]


def _run_info(d: Path) -> RunInfo:
    info = RunInfo(name=d.name, path=d)
    ymls = sorted(d.glob("*.yml"))
    info.config = ymls[0] if ymls else None
    models = d / "models"
    if models.is_dir():
        info.checkpoints = _checkpoints(models)
    tstates = d / "training_states"
    if tstates.is_dir():
        info.states = sorted(
            int(m.group(1)) for p in tstates.glob("*.state")
            if (m := _STATE.match(p.name)))
    return info


def list_runs(repo: Path) -> list[RunInfo]:
    exp = Path(repo) / "experiments"
    if not exp.is_dir():
        return []
    return sorted((_run_info(d) for d in exp.iterdir() if d.is_dir()),
                  key=lambda r: r.name)


def find_run(repo: Path, name: str) -> RunInfo | None:
    for r in list_runs(repo):
        if r.name == name:
            return r
    return None


def parse_run_name(name: str) -> tuple[str, int] | None:
    """``NERVE_4x_run`` → ``("NERVE", 4)``; case-insensitive, PULSE_fast-aware."""
    m = _RUN_NAME.match(name or "")
    if not m:
        return None
    model = m.group(1)
    model = "PULSE_fast" if model.lower() == "pulse_fast" else model.upper()
    return model, int(m.group(2))


def pick_checkpoint(run: RunInfo | None, iters: int | None = None) -> Checkpoint | None:
    """Latest checkpoint, or the one matching ``iters``; None if absent."""
    if run is None:
        return None
    if iters is None:
        return run.latest_checkpoint
    return next((c for c in run.checkpoints if c.iter == iters), None)


def _safe_run_dir(repo: Path, name: str) -> Path | None:
    """experiments/<name> only when ``name`` is a plain directory name."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return None
    d = Path(repo) / "experiments" / name
    return d if d.is_dir() else None


def archive_run(repo: Path, name: str) -> Path | None:
    """Rename a run to ``<name>_archived_<timestamp>`` (safe, reversible)."""
    d = _safe_run_dir(repo, name)
    if d is None:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = d.with_name(f"{name}_archived_{stamp}")
    while dst.exists():  # two archives in the same second
        dst = dst.with_name(dst.name + "_1")
    d.rename(dst)
    return dst


def delete_run(repo: Path, name: str) -> bool:
    """Permanently remove a run's experiment folder. Use archive_run instead."""
    d = _safe_run_dir(repo, name)
    if d is None:
        return False
    shutil.rmtree(d, ignore_errors=True)
    return not d.exists()
