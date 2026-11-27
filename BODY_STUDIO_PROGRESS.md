# BODY Studio — progress

Updated: 2026-11-27

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
8. 2026-11-19: T-NERVE in the video window — the `tnerve` arch plus the
   motion-adaptive anti-ghosting blend ("Off / Auto (recommended)",
   auto-enabled with Auto when T-NERVE is selected) are wired through
   `hnt.ui.video.video_cmd` and the worker; catalog license validator now
   accepts Apache-2.0 lineage next to MIT.
9. STAGED (blocked on the private repos going public — user decision): five
   T-NERVE catalog entries (otf/perceptual/natural/anime/anime-light,
   arch `tnerve`, 1,838,784 params, `Phips/T-NERVE/resolve/main/<ckpt>`)
   and three D-NERVE entries. Private URLs probe as 404, so shipping them
   in the bundled catalog would create dead links until then.
10. 2026-11-19: dark AND light themes. `hnt/ui/theme.py` now carries both
   palettes behind one design language (`stylesheet(mode)`, `set_mode`),
   the training dashboard has a Light/Dark toggle that persists to
   `ui/theme.json` and applies at startup, and the metrics chart recolors
   live (`MetricsChart.refresh_palette` — plot backgrounds follow the
   palette, which also fixed the not-actually-transparent plot bg).
   Verified visually offscreen (dark + light screenshots attached to the
   session log, both readable and coherent).
11. 2026-11-19: **D-NERVE inference integrated.** Vendored registry-neutral
   `hnt/dnerve_arch.py` (decoder only) + `hnt/dnerve_infer.py` (frozen
   DINOv2-L substrate with one-time download into the app data dir,
   `HNT_DINOV2` override for dev); `hnt.cli infer --arch dnerve` restores
   at input resolution (whole-frame up to ~2.5 MP, seamless 448 px tiles
   beyond/OOM); the inference window gets the `dnerve` arch with scale
   auto-pinned to 1x and disabled. Substrate source:
   `Phips/D-NERVE/substrate/dinov2_vitl14_pretrain.pth` (upload pending).
   Verified end-to-end with the released v4 perceptual weights on an
   AI-gen frame — output matches the research eval tool. Regression
   lesson: the first real input was 960x704 (not /14) and failed until
   the runner path was routed through the same pad logic as the eval
   tool — non-/14 sizes are now covered by tests.

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
- **UX TESTING POLICY (user directive 2026-11-14)**: every UI change gets
  exercised like a user would — run the app, click through flows, take
  screenshots, judge intuitiveness/cleanliness/modern look — screenshots
  attached to progress notes, not assumed from code.
- **Train UX**: prepare-dataset-from-video button (design exists),
  recipe presets instead of YAML editing, guided first-training flow.
- **Model zoo prerequisite**: routing lands after the zoo is complete
  (T-NERVE natural + severity-mixed, D-NERVE natural) — do not build
  against a changing lineup.
- **Inference toggle**: wavelet colorfix ("keep colors faithful to
  input") as a per-run option; default on for T-NERVE, off for D-NERVE
  heavy damage.


## Onboarding & static-analysis baseline (2026-11-27)

Log policy decision: this file stays the single progress/decision log —
consolidating into a separate `DEVELOPMENT.md` was considered and declined
(user decision, 2026-11-27).

Tooling baseline: polycheck installed globally (pipx; refreshed to
0.3.14 during this onboarding), plus scb-check and tach analyzers;
deep scan run and baseline saved
(`polycheck-reports/polycheck-baseline.json`) so future runs judge only
new findings. First deep run: 1740 findings — 43 HIGH / 1017 MED /
662 LOW / 18 INFO; gitleaks and pip-audit clean. Full report:
`polycheck-reports/polycheck-report.md`. Still not installed
(system-level, commands shown in session report): shellcheck,
hadolint, actionlint; depcheck/tsc skipped (npm global EACCES,
JS-only, not needed for this repo).

Complete deep re-run (same day, after polycheck 0.3.14 fixed its
sonarless PATH detection — sonarless had been installed at
`~/.sonarless` since June but is on no PATH, so every earlier scan
silently lacked the SonarQube pass; Docker was healthy throughout):
2154 findings, of which sonarless/SonarQube contributes 414 — 375 in
`packaging/vendor/trainer` (vendored upstream, 1 CRITICAL among them)
and 39 first-party (15 HIGH / 16 MED / 8 LOW; top rules: S3776
cognitive complexity ×10, S1244 float-equality ×7, S3358 nested
ternary ×4). The baseline was NOT re-taken: the 414 untriaged
SonarQube findings must be triaged and the vendor tree excluded from
the SonarQube scan first, so `--diff` runs keep surfacing them as new
until then.

## Improvement batch (2026-11-27, findings → fixes)

All findings triaged from the onboarding sweep + deep scans; suite now
**182 passed / 7 skipped** in ~5 s (before: 12 collection errors in a
torchless env, 1 failure, and a suite-hanging UI test).

- **CI parity restored**: `tests/test_dnerve_infer.py` and
  `tests/test_colorfix.py` now guard torch via
  `pytest.importorskip`; new `tests/test_ci_parity.py` blocks torch in
  a subprocess collection run (meta_path via sitecustomize — pytest 9
  skips on module *absence* only) so the next unguarded import fails
  locally instead of in CI. `pyproject.toml` pytest config gains
  `pythonpath = ["."]` so the suite runs without an editable install.
- **GPU telemetry bug fixed**: `hnt/train_runner.py` called the
  unimported `quiet()` (helper is `_quiet`); the NameError was
  swallowed by the bare except, so GPU tiles never populated.
  Regression test in `tests/test_telemetry.py`. `hnt/hardware.py`'s
  `quiet()` now uses `getattr(subprocess, "CREATE_NO_WINDOW", 0)`
  (POSIX-defined behavior, clears the mypy HIGH).
- **UI test hangs root-caused**: `MainWindow._first_run_check` shows a
  modal setup question whenever the interpreter lacks torch — its
  QTimer defers the modal into a LATER test's `processEvents()`
  (faulthandler stack proved it). `test_ui_smoke.py` gains an autouse
  fixture answering the modal with "No"; the catalog-worker tests
  drain the event loop with a bounded loop instead of a single
  `processEvents()` (same race fixed in `tests/test_setup_modal.py`).
  36 ui tests now pass in ~3 s.
- **Version consistency**: `latest.json` updated to 0.2.0 with real
  `BodyStudio-*` artifact URLs (was 0.1.4 with dead `HeartNerveTrainer-*`
  links — the updater offered a bogus downgrade); `installer.iss`
  AppVersion → 0.2.0. New `tests/test_version_consistency.py` pins
  pyproject == version.py == installer.iss == latest.json.
- **Docs/README**: `docs/index.html` header rebranded to BODY Studio
  and the lede now names the full family (NERVE/HEART/PULSE/T-NERVE/
  D-NERVE); README test count refreshed. Remaining docs gap: full
  feature documentation for PULSE/video/T-NERVE/D-NERVE/viewer/export.
- **SonarQube vendor exclusion**: `sonar-project.properties` excludes
  `packaging/vendor/**` (+ build dirs) durably. Verified with a
  sonarless-only re-scan: 72 first-party findings, 0 vendor (was 414
  with 375 vendor). `polycheck-reports/` currently holds that
  sonarless-only report; the next full deep run refreshes the rest.
- **Hygiene**: `.gitignore` now covers `.ruff_cache/`.

Still open (triage queue): first-party SonarQube findings (26 HIGH /
28 MED / 18 LOW — S3776 complexity refactors ×~10 need design care,
S1244 float-equality ×7 need per-site judgment), CI lint job (ruff
currently reports ~320 findings repo-wide — config first, then a
sweep), `main_window.py` (1302 lines) refactor when UI work continues,
AppImage `continue-on-error` visibility.

Confirmed defects (deep run + repo sweep, each evidence-checked):

- `hnt/train_runner.py:94` — `**quiet()` calls an unimported name
  (the helper is imported as `_quiet`, line 24); the surrounding
  `except Exception: pass` swallows the NameError, so GPU telemetry
  silently never updates. One-line fix.
- `tests/test_dnerve_infer.py:8` — bare `import torch` without
  `importorskip`; CI installs only `.[dev]` (no torch), so
  `pytest -q` fails at collection — CI broken since the D-NERVE tests
  landed. Every other torch-touching test file guards correctly.
- `hnt/infer.py:39` — `torch.load(..., weights_only=False)`: arbitrary
  code execution if a non-catalog/malicious `.pth` is opened
  (`dnerve_arch.py:57`, `dnerve_infer.py:53` already use
  `weights_only=True`). Decision pending: are user-supplied local
  weights an intended input, or can this path enforce True?
- Release-consistency cluster: `latest.json` still 0.1.4 with old
  `HeartNerveTrainer-*` asset URLs (the updater fetches this file and
  would offer a bogus downgrade), `installer.iss` AppVersion hardcoded
  "0.1.4" vs app 0.2.0, `docs/index.html:14` header still branded
  "HEART / NERVE Trainer".

Ist/soll sweep summary:

- Run management (log item 1) and batch output UX (item 2) are done in
  code — the "Current position" header above was stale; the real
  backlog is the auto-routing roadmap below plus packaging/release
  polish.
- Tests: 175 test functions across 19 files (pytest, runs in CI
  offscreen); tiers present: unit, integration (real ffmpeg, tiny-model
  TNerve roundtrip, real-checkpoint PULSE infer), slow-gated e2e
  (`test_pulse_smoke`), behavioral offscreen UI tests (`test_ui_smoke`,
  viewer, tour). Zero first-party coverage: `hnt/cleanup.py`
  (destructive "Clean up disk" path), `hnt/hardware.py`, the
  `downloader.py` download path, `_fp16.py` (slow-only), `cli.py`
  (direct tests only). Regular CI never exercises real
  fp16/export/download paths (torch/weights skips).
- Tooling config: no `[tool.ruff]`/mypy config in `pyproject.toml`
  despite caches existing; no lint job in CI; `.ruff_cache/` missing
  from `.gitignore`. README "130+ tests" stale (175 collected); docs
  site documents only NERVE/HEART (PULSE family, video, T-NERVE,
  D-NERVE, viewer, export window undocumented). mypy 30 HIGH-tier type
  findings; deptry 545 (mostly vendored trainer); duplication 383.
- `hnt/ui/main_window.py` at 1302 lines is the only oversized module —
  main refactor candidate when UI work continues.
