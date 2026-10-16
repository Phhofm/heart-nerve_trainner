# Distribution manifest — what to bundle (curated, minimal)

Ship a **pinned, pruned** copy of the trainer — not the dev fork. This keeps the
install small and avoids shipping untested code (BOLT/DUAL/DRFT, ablation configs,
dev scripts). Pin an exact commit/tag; record it in `packaging/vendor/trainer/REVISION`.

## Include
```
train.py
traiNNer/                       (the whole package)
  archs/
    heart_arch.py               # HEART
    nerve_arch.py               # NERVE
    r3gan_disc_arch.py          # (keep code; not enabled in UI yet)
    __init__.py + shared helpers the above import
  data/      (base_dataset, paired_image_dataset, realesrgan_dataset,
              realesrgan_paired_dataset, single_gt_dataset, degradations, degradations_extra, transforms)
  losses/    (basic, gan, perceptual, dists, adist, mssim, fft/focal, convnext_perceptual, feature_distillation?, ...)
  metrics/   (psnr_ssim, lpips, dists, topiq)
  models/    (models/__init__, sr_model, realesrgan_model, base_model, lr_sr_model?)
  utils/     (options, redux_options, registry, img_process_util, RNG, DiffJPEG, ...)
  __init__.py
options/train/RELEASE/          # config templates the app patches
schemas/redux-config.schema.json
scripts/heart/export_minimal.py # ONNX export
scripts/nerve/export_dynamic.py # ONNX + ncnn export
requirements / pinned env list
LICENSE, attribution
```

## Exclude (dev-only / untested)
- `traiNNer/archs/bolt_arch.py`, `dual_arch.py`, `drft_arch.py`, and other unused archs.
- `options/train/ABLATION/` and any non-RELEASE configs.
- Dev scripts under `scripts/` not needed at runtime (ablations, diagnostics, download helpers used only in dev).
- `experiments/`, `tests/` (optional to keep), `__pycache__`.

## How to produce it
- Start from the pinned trainer commit; copy only the include-set into `packaging/vendor/trainer/`.
- Keep a `REVISION` file with the commit hash.
- The app's `bootstrap.py` installs *Python deps*; this manifest governs the *trainer source*.
- Pruning is safe because the app only ever builds `heart`/`nerve` from RELEASE templates.

## Why prune
Smaller download, faster install, and **no untested code paths** shipped to beginners.
New features (e.g., modern-perceptual, new OTF ops) ride in updates that bump both the
app version and the pinned trainer revision.
