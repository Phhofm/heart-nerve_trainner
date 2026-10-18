#!/usr/bin/env bash
# Build a single-file AppImage from the staged PyInstaller bundle (Linux).
# The bundle already contains the app, `trainer/`, `uv` and the `hnt` sources.
# Requires linuxdeploy on PATH (CI downloads it); APPIMAGE_EXTRACT_AND_RUN avoids FUSE.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$APP_DIR"

APPDIR=AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/256x256/apps"

cp -r dist/HeartNerveTrainer/* "$APPDIR/usr/bin/"

cat > "$APPDIR/usr/share/applications/heart-nerve-trainer.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=HEART / NERVE Trainer
Exec=HeartNerveTrainer
Icon=heart-nerve-trainer
Categories=Graphics;Education;
Terminal=false
EOF

if [ -f packaging/icon.png ]; then
  cp packaging/icon.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/heart-nerve-trainer.png"
fi

export APPIMAGE_EXTRACT_AND_RUN=1
LINUXDEPLOY="${LINUXDEPLOY:-linuxdeploy}"
"$LINUXDEPLOY" --appdir "$APPDIR" --output appimage
mv ./*.AppImage dist/ 2>/dev/null || true
echo "AppImage built in dist/"