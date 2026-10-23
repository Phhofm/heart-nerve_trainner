"""Pin the trainer repo for tests: vendored copy if present, else discovery."""

import os
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent.parent / "packaging" / "vendor" / "trainer"

if _VENDOR.is_dir() and (_VENDOR / "train.py").is_file():
    os.environ.setdefault("HNT_TRAINER_REPO", str(_VENDOR))
