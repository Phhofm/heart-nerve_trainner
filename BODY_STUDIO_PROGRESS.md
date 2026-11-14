# BODY Studio — progress

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

- App rebranded to **BODY Studio**.
- `APP_NAME` is `BODY Studio`, `APP_SLUG` is `body-studio`.
- Window titles, CLI description, and `pyproject.toml` use the new name.
- README, docs website, packaging scripts, Windows installer, release
  workflow, and updater example URLs now use BODY Studio / `BodyStudio-*`
  artifact names (artifacts themselves change with the next release build).
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

### Video inference groundwork

- New `hnt/video.py` provides a headless video pipeline:
  - ffprobe metadata probing;
  - ffmpeg raw RGB frame streaming;
  - frame-by-frame BODY model inference;
  - optional ffmpeg deinterlacing;
  - audio-preserving re-encode;
  - metadata mapping from the source video.
- New `hnt/temporal.py` + `hnt.video.infer_video_temporal` add cut-safe
  temporal inference for T-NERVE (2026-11-12):
  - two-pass decode (downscaled scene-cut scan, then full-res windows);
  - windows never span scene cuts; scene-boundary frames fall back to
    single-frame processing (same semantics as the research repo's
    cut-safe evaluation);
  - `out_scale` supports 2x SR output (encoded video is 2x the input);
  - frame-synced by construction: output frame order matches input;
  - verified on the extreme-motion stress clip with the released T-NERVE
    weights (1041 frames -> 1920x1412, model windows + fallbacks).
- `hnt.cli` gained a `video` subcommand with model/arch/scale/device/tiling,
  codec/CRF, deinterlace, overwrite, and progress reporting.
- New `hnt/ui/video.py` adds a dedicated video inference window:
  - catalog/custom model selection;
  - device/tiling/codec/CRF controls;
  - optional deinterlace;
  - ffmpeg availability check;
  - progress and output-folder actions.
- Main window now exposes a `Video…` action.
- Image batch inference now preserves folder structure inside the output
  directory, and the CLI/UI support skipping outputs that already exist.
- Image and video inference windows accept drag-and-drop files/folders.
- App branding is now BODY Studio (package/script names updated too).

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
- Dataset pre-flight check runs before training starts and blocks with a clear
  message instead of a cryptic crash: missing/empty HR or LR folders, HR/LR
  count mismatch, missing/extra pairs (both `img.png` and `imgx<scale>.png`
  naming), and a quick corrupt-image sample scan. OTF mode only needs HR.
  Also available headless as `python -m hnt check-dataset --hr … [--lr …]`.
- Warm-start assistance in the training window: a "Warm start" picker offers
  the catalog pretrained weights matching the chosen model and scale
  (`catalog.warmstart_entry` picks the smallest safetensors candidate).
  Selecting it downloads once on Train (non-blocking worker), then training
  continues automatically; cached weights are reused; falling back to
  from-scratch is always available and is the default.
- "Existing runs…" is now a real dialog: lists every run with its latest
  checkpoint, shows resume state, lets the user pick a checkpoint iteration,
  and exports it through the same export pipeline as post-training export
  (model/scale derived from the run name).
- Dedicated Export/Convert window (`hnt/ui/export.py`): pick any checkpoint,
  choose architecture and scale, and export safetensors + ONNX fp32/fp16
  (+ ncnn for NERVE) without training. `ExportWorker` moved here so the
  training window and this window share one export pipeline; the main window
  gained an "Export / Convert…" launcher.
- First-run quick tour (`hnt/ui/tour.py`): one screen with the four steps
  (Set up → point at images → Train → use/export the model). Shown
  automatically once on first run (marker file `ui/tour_seen.json` in the app
  data dir, only when the training environment isn't set up yet), and always
  available via a "Quick tour" button in the main window.
- Output browser upgrades: the inference output list shows 64px thumbnails,
  double-clicking an output opens the A/B viewer, a "Gallery view" toggle
  switches to large-thumbnail icon mode, and the viewer's input side is
  chosen by filename-stem matching (`infer.match_input_for_output`) so the
  displayed pair actually corresponds instead of always using the first input.
- Existing-runs discovery: `hnt/runs.py` lists traiNNer-redux experiment runs
  with their config copy, EMA/plain checkpoints (sorted, EMA preferred), and
  resumable training states. Surfaced in the training window via
  "Existing runs…" (shows latest checkpoint + resume point per run, and notes
  that Train auto-resumes matching runs).
- Headless export: `python -m hnt export --run NAME [--iter N] --out DIR`
  picks the run's latest (or requested) checkpoint, derives model/scale from
  the run name, and runs the same export pipeline as the GUI (safetensors,
  ONNX fp32/fp16, optional ncnn).
- Per-file batch inference progress: the CLI emits `progress d/t path` after
  every file (skipped files included), the inference window streams CLI output
  and drives its progress bar determinately, and `skip_existing` now decides
  before running the model instead of after (existing outputs cost nothing).
- A/B viewer upgrades: rewritten on QGraphicsView — mouse-wheel zoom synced
  across both sides (anchored at the cursor, clamped 0.1×–8×), drag-to-pan,
  double-click or "Reset zoom" to refit, plus `set_pair()` for arbitrary
  input/output pairs, a "Swap sides" button (mirrors the split and exchanges
  input/output), and image drag-and-drop (one image → input side, two →
  input+output; non-images ignored).

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
- Batch output-path structure and skip-existing tests.
- Scene-cut helper tests.- Dataset pre-flight tests (empty/missing folders, count mismatch, missing/extra
  pairs with the `x<scale>` naming template, corrupt image samples, OTF mode,
  subfolder search).
- Run discovery tests (checkpoint parsing/sorting, EMA preference, resume
  states, config discovery, empty/missing repos).
- Headless export CLI tests (latest/`--iter` checkpoint pick, model+scale from
  run name, error paths).
- Per-file inference progress tests (callback order/totals, skip-existing
  avoids the model but still reports, CLI progress lines, output-line parsers,
  progress-bar determinate/clamped behavior).
- PULSE_fast end-to-end smoke test:
  bundled 10-image paired stub → train → validate → checkpoint → ONNX export.
- Setup modal tests.
- Video pipeline tests, including real ffmpeg frame streaming.
- Viewer tests (pair display, `set_pair`, swap sides, drag-and-drop rules).
- UI smoke tests for model selection, paired-only PULSE behavior, catalog
  download success/failure, inference output actions, video window construction,
  video progress handling, drag-and-drop input, and close-job guard.

Latest full suite result: **154 passed**.

Version: **0.2.0** — the 0.1.x series covered the training dashboard and
setup flow; 0.2.0 adds video inference, dataset/run/export ergonomics, the
A/B viewer rewrite, warm start, and the quick tour.


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
- `hnt/dataset_check.py` — pre-flight HR/LR dataset validation.
- `hnt/runs.py` — training run discovery (checkpoints, resume states).
- `hnt/selftest.py` — end-to-end smoke test.
- `hnt/ui/main_window.py` — training dashboard.
- `hnt/ui/inference.py` — inference window.
- `hnt/ui/video.py` — video inference window.
- `hnt/video.py` — headless video pipeline.
- `hnt/scenes.py` — scene-cut helpers for temporal training/inference.
- `hnt/ui/viewer.py` — A/B viewer.
- `hnt/ui/setup_modal.py` — dependency install gate.
- `hnt/ui/theme.py` — dark UI stylesheet.
- `hnt/ui/chart.py` — validation metric charts.
- `packaging/vendor/trainer/` — vendored trainer.

## Next priorities

1. ~~Improve run management~~ (done: Existing runs dialog with checkpoint
   export, warm start, headless export, plus safe Archive (rename to
   `<name>_archived_<ts>`, reversible) and Delete (recursive, confirmed)
   actions with path-traversal guards — `runs.archive_run`/`runs.delete_run`).
2. ~~Improve inference batch UX~~ (done: per-file progress, thumbnails,
   double-click A/B with stem-matched input pairing; a richer output browser
   with a grid/gallery view remains optional polish).
3. ~~Improve the A/B viewer~~ (done: zoom/pan on synced QGraphicsView panes,
   drag-and-drop, arbitrary pairs, swap sides).
4. ~~Add catalog/pretrain download assistance to the main training window~~
   (done: Warm start picker with automatic one-time download).
5. ~~Update README/docs and packaging names/URLs from the old
   HEART/NERVE Trainer branding to BODY Studio~~ (done: README, docs site,
   packaging scripts, installer, release workflow, updater example URLs now
   use BODY Studio / BodyStudio artifacts).
6. ~~Add error reporting polish and a first-run quick tour~~ (tour done;
   error reporting already produces a copyable plain-language report via
   "Get help / report" — further polish optional).
7. ~~Consider a dedicated Convert/Export page rather than only post-training
   export~~ (done: Export / Convert window).

## Non-goals for now

- TensorRT/ONNX runtime selection inside the GUI.
- Multi-GPU orchestration.
- Fine-grained custom YAML editing.
- Automated model publishing.


## Roadmap: beginner-friendly auto-routing (vision, 2026-11-14)

User vision: BODY Studio as the beginner-friendly front-end for the whole
BODY family — select an mp4/image, the app analyzes input and picks the
network + variant automatically (defaults), overridable before running.

- **Auto-routing v1 (heuristics)**: image vs video -> NERVE/HEART vs
  T-NERVE; damage estimate (blockiness/noise heuristics) -> fidelity tier
  vs D-NERVE. Severity-mixed models keep this simple (one model covers
  light->heavy, so routing is family-level, not severity-level).
- **Transparency**: show the detection result ("720p, heavy compression
  -> D-NERVE perceptual (recommended)") with one-click override.
- **Install**: portable packaging (PyInstaller or similar) before any
  public release.
- **Train UX**: prepare-dataset-from-video button (design exists),
  recipe presets instead of YAML editing, guided first-training flow.
- **Model zoo prerequisite**: routing lands after the zoo is complete
  (T-NERVE natural + severity-mixed, D-NERVE natural) — do not build
  against a changing lineup.
- **Inference toggle**: wavelet colorfix ("keep colors faithful to
  input") as a per-run option; default on for T-NERVE, off for D-NERVE
  heavy damage.
