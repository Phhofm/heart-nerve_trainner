"""Pre-flight dataset validation — turn cryptic training crashes into clear advice.

BEGINNER-FIRST: the most common failure when starting a train run is a dataset
problem (wrong folder, HR/LQ mismatch, broken image). These checks run in
seconds before any GPU work and report plain-language issues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".avif"}
_SCALE_SUFFIX = re.compile(r"x\d+$", re.IGNORECASE)
_SAMPLE_CHECK = 5  # images deep-scanned per folder; keeps pre-flight fast


@dataclass
class Issue:
    level: str  # "error" | "warning"
    message: str


@dataclass
class DatasetReport:
    hr_count: int = 0
    lq_count: int = -1  # -1 = not applicable (OTF mode)
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def error_text(self) -> str:
        return "\n".join(f"{i.level}: {i.message}" for i in self.issues if i.level == "error")


def _images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def _verify_sample(paths: list[Path]) -> list[str]:
    """Return names of unreadable/corrupt images from a small sample."""
    bad = []
    for p in paths[:_SAMPLE_CHECK]:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:  # noqa: BLE001
            bad.append(p.name)
    return bad


def _lq_key(stem: str) -> str:
    return _SCALE_SUFFIX.sub("", stem)


def check_dataset(hr_dir: str, lr_dir: str = "", scale: int = 4) -> DatasetReport:
    """Validate the training folders chosen in the UI/CLI.

    ``lr_dir`` empty means OTF (LR generated on the fly) — only HR is checked.
    Pair matching accepts both ``img.png`` and ``imgx<scale>.png`` LQ names.
    """
    rep = DatasetReport()
    hr = Path(hr_dir) if hr_dir else None
    if hr is None or not hr.is_dir():
        rep.issues.append(Issue("error", f"HR folder not found: {hr_dir}"))
        return rep
    hr_imgs = _images(hr)
    rep.hr_count = len(hr_imgs)
    if not hr_imgs:
        rep.issues.append(Issue("error", f"HR folder has no images: {hr_dir}"))
        return rep

    if not lr_dir:
        bad = _verify_sample(hr_imgs)
        if bad:
            rep.issues.append(Issue("error", "Unreadable HR image(s): " + ", ".join(bad)))
        return rep

    lr = Path(lr_dir)
    if not lr.is_dir():
        rep.issues.append(Issue("error", f"LR folder not found: {lr_dir}"))
        return rep
    lr_imgs = _images(lr)
    rep.lq_count = len(lr_imgs)
    if not lr_imgs:
        rep.issues.append(Issue("error", f"LR folder has no images: {lr_dir}"))
        return rep

    if len(hr_imgs) != len(lr_imgs):
        rep.issues.append(Issue(
            "error",
            f"HR has {len(hr_imgs)} image(s) but LR has {len(lr_imgs)} — every HR image "
            f"needs its {scale}× downscaled LR counterpart in the LR folder."))

    hr_keys = {p.stem: p for p in hr_imgs}
    lq_keys: dict[str, Path] = {}
    for p in lr_imgs:
        lq_keys.setdefault(_lq_key(p.stem), p)

    missing = sorted(k for k in hr_keys if k not in lq_keys)
    extra = sorted(k for k in lq_keys if k not in hr_keys)
    if missing:
        shown = ", ".join(missing[:3]) + ("…" if len(missing) > 3 else "")
        rep.issues.append(Issue(
            "error",
            f"{len(missing)} HR image(s) have no LR pair (e.g. {shown}). LR files must be "
            f"named like the HR file, optionally with an x{scale} suffix "
            f"(img.png ↔ imgx{scale}.png)."))
    if extra:
        shown = ", ".join(extra[:3]) + ("…" if len(extra) > 3 else "")
        rep.issues.append(Issue(
            "error", f"{len(extra)} LR image(s) have no HR pair (e.g. {shown})."))

    bad = _verify_sample(hr_imgs) + _verify_sample(lr_imgs)
    if bad:
        rep.issues.append(Issue("error", "Unreadable image(s): " + ", ".join(sorted(set(bad)))))
    return rep
