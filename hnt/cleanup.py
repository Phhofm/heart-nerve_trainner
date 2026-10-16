"""Disk cleanup — delete intermediate run dirs, never the user's outputs.

Experiments save checkpoints + validation images every N iters → big. This frees
that space while leaving the user's final-model folders (where the .safetensors /
.onnx / .ncnn land) completely untouched.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .paths import Paths


@dataclass
class CleanupReport:
    freed_bytes: int
    removed: list[str]
    kept: list[str]

    @property
    def freed_gb(self) -> float:
        return round(self.freed_bytes / 1024**3, 2)


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _is_protected(path: Path, protected: list[Path]) -> bool:
    rp = path.resolve()
    for prot in protected:
        try:
            pr = prot.resolve()
        except OSError:
            continue
        if rp == pr or pr in rp.parents or rp in pr.parents:
            return True
    return False


def cleanup_runs(paths: Paths, protected_outputs: list[Path] | None = None,
                 dry_run: bool = True) -> CleanupReport:
    """Remove trainer run dirs (experiments/*) + app run metadata.

    ``protected_outputs`` = the user's chosen final-model folders; anything
    overlapping them is skipped.
    """
    protected = [Path(p) for p in (protected_outputs or [])]
    targets: list[Path] = []

    experiments = paths.repo / "experiments"
    if experiments.is_dir():
        targets += [d for d in experiments.iterdir() if d.is_dir()]
    if paths.runs.is_dir():
        targets += [d for d in paths.runs.iterdir()]

    freed = 0
    removed: list[str] = []
    kept: list[str] = []
    for t in targets:
        if _is_protected(t, protected):
            kept.append(str(t))
            continue
        sz = _size(t)
        if not dry_run:
            shutil.rmtree(t, ignore_errors=True)
        freed += sz
        removed.append(str(t))
    return CleanupReport(freed, removed, kept)
