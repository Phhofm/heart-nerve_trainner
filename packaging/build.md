# Packaging notes (HEART / NERVE Trainer)

The app is a thin PySide6 shell + a bundled copy of `traiNNer-redux` + a bundled
`uv` binary. On first run it creates an isolated venv and installs pinned deps —
the user's system is never modified.

## What to vendor
- `packaging/vendor/uv` (or `uv.exe`) — the bootstrap installer.
- `packaging/vendor/trainer/` — a pinned checkout of `traiNNer-redux` (train.py,
  options/, traiNNer/, scripts/). Pin an exact commit.
- `packaging/icon.png` (256px).

## Build (local)
```bash
pip install pyinstaller -e .
pyinstaller --noconfirm --windowed --name HeartNerveTrainer app.py   # Windows: drop --windowed for console debug
# Windows:
iscc packaging/windows/installer.iss
# Linux:
bash packaging/linux/build_appimage.sh
```

## First-run flow (what the website can promise)
install → launch → dependency check → **create venv (uv) + install** → **self-test**
→ ready. Internet required for first-run setup.

## Roadmap
- Code-sign the Windows installer (reduces SmartScreen warnings).
- Publish `latest.json` for in-app update checks.
- CI (`.github/workflows/build-release.yml`) builds both installers on tag.
