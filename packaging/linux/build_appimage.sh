#!/usr/bin/env bash
# Build a single-file AppImage for the HEART / NERVE Trainer (Linux).
# Expects PyInstaller output at dist/HeartNerveTrainer/ and a vendored `uv` +
# a copy of traiNNer-redux under packaging/vendor/.
set -euo pipefail

APPDIR=AppDir
rm -rf "$APPDIR" && mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"

cp -r dist/HeartNerveTrainer/* "$APPDIR/usr/bin/"
cp packaging/vendor/uv "$APPDIR/usr/bin/uv"
cp -r packaging/vendor/trainer "$APPDIR/usr/bin/trainer"

cat > "$APPDIR/usr/share/applications/heart-nerve-trainer.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=HEART / NERVE Trainer
Exec=HeartNerveTrainer
Icon=heart-nerve-trainer
Categories=Graphics;Education;
Terminal=false
EOF
cp packaging/icon.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/heart-nerve-trainer.png"

# linuxdeploy + appimagetool are expected on PATH (or downloaded by CI).
linuxdeploy --appdir "$APPDIR" --output appimage
mv ./*.AppImage dist/ 2>/dev/null || true
echo "AppImage built in dist/"
