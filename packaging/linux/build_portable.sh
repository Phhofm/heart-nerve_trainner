#!/usr/bin/env bash
# Build a portable Linux bundle (frozen app + vendored trainer + uv).
# A native AppImage additionally needs linuxdeploy/appimagetool on PATH (CI does this).
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$APP_DIR"

PY="${PYTHON:-../venv/bin/python}"

if [ ! -d packaging/vendor/trainer ]; then
  "$PY" packaging/make_vendor.py --trainer "${TRAINER_REPO:-../}"
fi

"$PY" -m PyInstaller --noconfirm --clean --onedir --windowed \
  --name HeartNerveTrainer \
  --collect-submodules pyqtgraph \
  --exclude-module torch --exclude-module torchvision \
  --exclude-module onnx --exclude-module onnxruntime \
  --exclude-module onnxconverter_common --exclude-module onnxsim \
  --exclude-module pnnx --exclude-module ncnn \
  --exclude-module pyqtgraph.opengl \
  app.py

DIST="dist/HeartNerveTrainer"
rm -rf "$DIST/hnt" "$DIST/trainer" "$DIST/uv"
cp -r hnt "$DIST/hnt"
cp -r packaging/vendor/trainer "$DIST/trainer"

# `packaging/vendor/uv` is gitignored; CI installs uv and puts it on PATH.
UV_BIN="${UV_BIN:-packaging/vendor/uv}"
if [ ! -f "$UV_BIN" ]; then
  UV_BIN="$(command -v uv || true)"
fi
if [ -z "$UV_BIN" ] || [ ! -f "$UV_BIN" ]; then
  echo "error: uv binary not found (set UV_BIN or install uv)" >&2
  exit 1
fi
cp "$UV_BIN" "$DIST/uv"
chmod +x "$DIST/uv"

tar -czf "dist/HeartNerveTrainer-linux-x86_64.tar.gz" -C dist HeartNerveTrainer
echo "built dist/HeartNerveTrainer-linux-x86_64.tar.gz"