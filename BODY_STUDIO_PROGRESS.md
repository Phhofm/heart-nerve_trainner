# SISR BODY Studio — progress

Updated: 2026-11-10

## Current position

The app is a PySide6 desktop front-end for the BODY SISR suite. Core setup,
training, inference, export, catalog download, A/B viewing, and the PULSE_fast
end-to-end smoke path work. The next phase is run management, richer batch
output handling, and packaging/release polish.

Repository: `/home/phips/Documents/heart-nerve_trainner`
Remote: `https://github.com/Phhofm/heart-nerve_trainner`

## Done

### Branding and shell

- App rebranded to **SISR BODY Studio**.
- `APP_NAME` is `SISR BODY Studio`, `APP_SLUG` is `sisr-body-studio`.
- Window titles, CLI description, and `pyproject.toml` use the new name.
- Main window has a modern dark theme, hardware chips, live telemetry tiles,
  validation metric charts, and in-app guidance.

### Model catalog

- `hnt/catalog.py` provides a model catalog.
- Bundled catalog is `hnt/models.json`.
- Catalog order: `PULSE_fast → PULSE → NERVE → HEART`, with 2x/4x variants.
- Catalog loading supports a remote JSON source and falls back to the bundled
  copy on network/validation failure.
- Entries are strictly validated: HTTPS URLs, MIT-only models, supported
  scales, positive parameter counts, and non-empty formats.
- `verify_file()` computes SHA-256 when a catalog entry records a hash.
- `resolve_model()` downloads Hugging Face catalog weights into the app cache
  and verifies checksums when present.

### Inference

- `hnt/infer.py` supports `nerve`, `heart`, `pulse`, and `pulse_fast`.
- `hnt/cli.py` accepts all four architectures.
- Inference supports image files and folders, tiled auto mode, CPU/CUDA device
  selection, and model-suffixed PNG output.
- The inference window has:
  - catalog dropdown with model information;
  - default `nerve_2x`;
  - selecting a catalog model sets architecture and scale;
  - background catalog download with busy/error state;
  - custom model path field;
  - input image/folder selection;
  - output listing;
  - `Open output folder`;
  - `A/B viewer`.

### A/B viewer

- Side-by-side input/output viewer with split slider.
- Images are rescaled on window resize so both panes stay fitted.

### Setup and training

- `hnt/bootstrap.py` installs training dependencies into an isolated venv.
- `hnt/ui/setup_modal.py` shows real install/test progress and logs.
- Main window gates `Train` until setup is complete.
- After setup, the runtime Python path is re-resolved, so training can start
  without restarting the app.
- Main window training model picker now includes:
  - `PULSE_fast`
  - `PULSE`
  - `NERVE` — default/recommended
  - `HEART`
- PULSE models automatically use paired LR/HR mode; automatic on-the-fly
  degradation is unavailable for them and is disabled in the UI.
- Settings panel scrolls on smaller screens.
- Closing the window during an active training/export/download job asks before
  stopping the job.
- Auto-parameters use model window size, profile, and detected VRAM.
- OOM recovery can shrink patch/batch; compile failures retry without compile.

### Config generation

- `hnt/config_gen.py` turns GUI choices into traiNNer-redux YAML configs.
- HEART/NERVE use proven OTF/paired/perceptual templates.
- PULSE uses the synthesized paired fidelity path.
- Validation defaults to bundled CC0 tiles when the user supplies none.
- Validation is tiled by default to avoid OOM.
- DISTS/LPIPS validation metrics are opt-in.

### Export

- `hnt/exporter.py` exports trained checkpoints to:
  - safetensors copy;
  - ONNX fp32;
  - ONNX fp16.
- HEART, NERVE, and PULSE/PULSE_fast each use the appropriate vendored export
  script.
- ONNX weights are embedded where possible to avoid `.data` sidecars.
- ncnn export is available for NERVE via TorchScript + pnnx with parity check.
- PULSE export uses `packaging/vendor/trainer/scripts/pulse/export_minimal.py`.

### Vendored trainer

- `packaging/vendor/trainer/` contains the traiNNer-redux fork used by the app.
- PULSE architecture is `traiNNer/archs/pulse_arch.py`.
- ICNR utility is `traiNNer/utils/icnr.py`.
- PULSE ONNX exporter is `scripts/pulse/export_minimal.py`.

## Tests

- Catalog parsing/validation tests.
- Catalog checksum/resolution tests.
- PULSE inference tests with a real PULSE_fast checkpoint.
- PULSE export wrapper regression test.
- PULSE_fast end-to-end smoke test:
  bundled 10-image paired stub → train → validate → checkpoint → ONNX export.
- Setup modal tests.
- Viewer tests.
- UI smoke tests for model selection, paired-only PULSE behavior, catalog
  download success/failure, inference output actions, and close-job guard.

Latest full suite result: **70 passed**.

CI fix: `Pillow>=10` was added to runtime dependencies because inference and
viewer tests use Pillow; it had previously been present locally but not declared
to CI.

## Key implementation files

- `hnt/catalog.py` — catalog and downloads.
- `hnt/models.json` — bundled catalog.
- `hnt/infer.py` — PyTorch inference.
- `hnt/cli.py` — headless CLI.
- `hnt/exporter.py` — checkpoint export.
- `hnt/config_gen.py` — training config generation.
- `hnt/selftest.py` — end-to-end smoke test.
- `hnt/ui/main_window.py` — training dashboard.
- `hnt/ui/inference.py` — inference window.
- `hnt/ui/viewer.py` — A/B viewer.
- `hnt/ui/setup_modal.py` — dependency install gate.
- `hnt/ui/theme.py` — dark UI stylesheet.
- `hnt/ui/chart.py` — validation metric charts.
- `packaging/vendor/trainer/` — vendored trainer.

## Next priorities

1. Add catalog/pretrain download assistance to the main training window.
2. Improve run management:
   - list existing runs;
   - resume/restart safely;
   - show checkpoint selection.
3. Improve inference batch UX:
   - per-file progress;
   - output thumbnails or a richer output browser;
   - selected output ↔ selected input pairing.
4. Improve the A/B viewer:
   - zoom/pan;
   - direct image drag-and-drop;
   - pick arbitrary input/output pairs.
5. Update README/docs and packaging names/URLs from the old
   HEART/NERVE Trainer branding to SISR BODY Studio.
6. Add error reporting polish and a first-run quick tour.
7. Consider a dedicated Convert/Export page rather than only post-training
   export.

## Non-goals for now

- TensorRT/ONNX runtime selection inside the GUI.
- Multi-GPU orchestration.
- Fine-grained custom YAML editing.
- Automated model publishing.
