"""One fact, four homes: the app version must agree everywhere.

`hnt/version.py` is the source of truth; `pyproject.toml`, the Inno
Setup installer, and the auto-updater manifest (`latest.json`, fetched
by `hnt/updater.py`) must match it. Drift found during onboarding:
latest.json and installer.iss still carried 0.1.4 while the app was
0.2.0, so the updater offered a bogus downgrade and the installer
stamped the wrong version (onboarding finding, 2026-11-27).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).parent.parent


def _app_version() -> str:
    from hnt.version import __version__

    return __version__


def test_pyproject_version_matches_app():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml has no version line"
    assert match.group(1) == _app_version()


def test_installer_version_matches_app():
    text = (REPO / "packaging" / "windows" / "installer.iss").read_text(
        encoding="utf-8"
    )
    match = re.search(r'#define AppVersion "([^"]+)"', text)
    assert match, "installer.iss has no AppVersion define"
    assert match.group(1) == _app_version()


def test_updater_manifest_version_and_assets_match():
    manifest = json.loads((REPO / "latest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == _app_version()

    assets = manifest["assets"]
    # The updater fetches exactly these URLs; they must name the real
    # release artifacts (README + release workflow + installer agree).
    assert assets["windows"].endswith("/BodyStudio-Setup.exe")
    assert assets["linux"].endswith("/BodyStudio-linux-x86_64.tar.gz")
