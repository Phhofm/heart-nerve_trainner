# SISR BODY Studio — progress

Updated: 2026-11-10

## Done

- App rebranded to **SISR BODY Studio** (`APP_NAME` / `APP_SLUG` / window titles).
- `hnt/catalog.py`:
  - bundled `hnt/models.json` with BODY models ordered `PULSE_fast → PULSE → NERVE → HEART`
  - optional remote catalog fetch with fallback on failure
  - strict entry validation (https, MIT, formats, fields)
  - `verify_file()` SHA-256 helper
  - `resolve_model()` downloads into the app model cache and verifies the hash when present
- `hnt/ui/inference.py`:
  - catalog dropdown with model info
  - default `nerve_2x`
  - selecting a catalog model sets arch/scale
  - “Download / use catalog model” resolves the entry and fills the custom path
- `hnt/infer.py` + `hnt/cli.py`:
  - `pulse` / `pulse_fast` inference supported (CPU tested with real checkpoint)
  - CLI `--arch` accepts all four archs
- `hnt/ui/viewer.py`:
  - minimal A/B viewer with slider and side-by-side splitter
- `hnt/ui/setup_modal.py`:
  - modal with Back / Install & test dependencies
  - real bootstrap progress/log in the modal
  - main window gates Train until setup is complete
- `hnt/exporter.py`:
  - PULSE/PULSE_fast ONNX export via the vendored
    `scripts/pulse/export_minimal.py`
  - fp16 conversion shares the existing ONNX path
- PULSE_fast smoke test: bundled 10-image paired stub → train → validate →
  checkpoint → ONNX export

## Tests

- `tests/test_catalog.py`
- `tests/test_catalog_resolve.py`
- `tests/test_pulse_infer.py`
- `tests/test_pulse_smoke.py`
- `tests/test_setup_modal.py`
- `tests/test_viewer.py`
- updated UI smoke tests and PULSE export wrapper test
- full suite: 65 passed

## Next

- Catalog/download button in the main training window
- Folder batch progress and output listing
- Monitor bar
