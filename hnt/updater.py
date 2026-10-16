"""Simple, user-friendly updates.

Flow: the app fetches a tiny ``latest.json`` (from GitHub Releases / a raw URL),
compares versions, asks the user, then downloads the OS installer and launches it.
**User data is preserved**: the installer only replaces the app files; runs,
datasets, the venv and the user's output folders live in the app data dir and are
left untouched. The venv is reused unless the manifest bumps ``deps_rev``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .version import __version__

# Where to look for latest.json (override with HNT_UPDATE_URL).
DEFAULT_MANIFEST_URL = "https://raw.githubusercontent.com/Phhofm/heart-nerve_trainner/main/latest.json"


@dataclass
class UpdateInfo:
    version: str
    notes: str
    url: str  # installer for this OS
    deps_rev: str = ""  # bump to force a venv rebuild


def _os_key() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    return "macos"


def _parse_version(v: str) -> tuple[int, ...]:
    out = []
    for part in v.lstrip("v").split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def check_for_update(manifest_url: str | None = None, timeout: float = 5.0) -> UpdateInfo | None:
    """Return UpdateInfo if a newer version exists, else None (never raises)."""
    url = manifest_url or DEFAULT_MANIFEST_URL
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
            data = json.loads(r.read().decode())
    except Exception:
        return None
    latest = data.get("version", "0")
    if _parse_version(latest) <= _parse_version(__version__):
        return None
    asset = (data.get("assets") or {}).get(_os_key())
    if not asset:
        return None
    return UpdateInfo(latest, data.get("notes", ""), asset, data.get("deps_rev", ""))


def download_installer(info: UpdateInfo, dest_dir: Path, on_progress=lambda p: None) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = info.url.split("/")[-1] or "update-installer"
    out = dest_dir / name
    with urllib.request.urlopen(info.url, timeout=60) as r:  # noqa: S310
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with out.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                on_progress(done / total if total else 0.0)
    return out


def apply_update(installer: Path) -> None:
    """Launch the installer, then the user closes the app (installer replaces files)."""
    if sys.platform.startswith("win"):
        subprocess.Popen([str(installer)], close_fds=True)
    elif sys.platform.startswith("linux"):
        installer.chmod(0o755)
        subprocess.Popen([str(installer)], close_fds=True)
    else:  # pragma: no cover
        raise RuntimeError("auto-apply unsupported on this OS; open the file manually")


# manifest shape (latest.json):
# {
#   "version": "0.2.0",
#   "deps_rev": "2026-10-16",
#   "notes": "Adds WebP/AVIF OTF degradations",
#   "assets": {
#     "windows": "https://github.com/.../HeartNerveTrainer-Setup.exe",
#     "linux":   "https://github.com/.../HeartNerveTrainer.AppImage"
#   }
# }
