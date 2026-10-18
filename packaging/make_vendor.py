"""Build ``packaging/vendor/`` (trainer source + uv) for a release bundle.

Usage (from the app directory):

    python packaging/make_vendor.py --trainer /path/to/traiNNer-redux

Copies a pruned trainer (see packaging/distribution_manifest.md), a small validation
subset so live metrics work out of the box, the self-test demo images, and the `uv`
binary. The result is gitignored and consumed by the build scripts / CI.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
TRAINER = VENDOR / "trainer"

INCLUDE = [
    "train.py",
    "pyproject.toml",
    "requirements.txt",
    "traiNNer",
    "options/train/RELEASE",
    "schemas",
    "scripts/heart/export_minimal.py",
    "scripts/nerve/export_dynamic.py",
    "scripts/options",
    "LICENSE.txt",
]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}
SKIP_FILES = {"line_distiller_weights.pth"}  # 41 MB dev-only loss weights
SPLIT_COUNT = 10  # trimmed Urban100 validation subset (keeps the bundle small)


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in SKIP_DIRS or n in SKIP_FILES or n.endswith(".pyc")}


def _copy_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_subset(src: Path, dst: Path, count: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.is_file())[:count]
    for f in files:
        shutil.copy2(f, dst / f.name)


def _revision(trainer: Path) -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=trainer,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=trainer,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}" if rev else "unknown"
    except Exception:
        return "unknown"


def build(trainer: Path) -> None:
    if not (trainer / "train.py").is_file():
        raise SystemExit(f"not a trainer checkout: {trainer}")

    if TRAINER.exists():
        shutil.rmtree(TRAINER)
    TRAINER.mkdir(parents=True)

    for rel in INCLUDE:
        src = trainer / rel
        if src.exists():
            _copy_tree(src, TRAINER / rel)
        else:
            print(f"  skip (missing): {rel}")

    # Small out-of-the-box validation + self-test data.
    # The demo set has hr/lr subfolders, so copy it whole (4 MB).
    _copy_tree(trainer / "datasets/train/dataset1", TRAINER / "datasets/train/dataset1")
    _copy_subset(trainer / "datasets/val/urban100/hr", TRAINER / "datasets/val/urban100/hr", SPLIT_COUNT)
    _copy_subset(
        trainer / "datasets/val/urban100_otf_x2",
        TRAINER / "datasets/val/urban100_otf_x2",
        SPLIT_COUNT,
    )

    rev = _revision(trainer)
    (TRAINER / "REVISION").write_text(rev + "\n")
    print(f"trainer vendored: {TRAINER}  (revision {rev})")

    uv = shutil.which("uv")
    if uv:
        dst = VENDOR / "uv"
        shutil.copy2(uv, dst)
        dst.chmod(0o755)
        print(f"uv vendored: {dst}")
    else:
        print("warning: uv not found on PATH — bundle it manually or set HNT_UV")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", default="../", help="path to the traiNNer-redux checkout")
    a = ap.parse_args()
    build(Path(a.trainer).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())