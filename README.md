# HEART / NERVE Trainer

A **beginner-first desktop app** for training SISR models (HEART / NERVE) — no
terminal, no code. Pick a model + scale, point at a folder of HR images, hit
**Train**, get deployable `.safetensors` + `.onnx` (+ `.ncnn` for NERVE).

> Thin shell over [traiNNer-redux](https://github.com/the-database/traiNNer-redux):
> it detects your hardware, bootstraps an isolated Python env, generates a training
> config, runs training, streams progress, and exports the model.

- 🌐 **Website:** https://phhofm.github.io/heart-nerve_trainner/ (source in [`docs/`](docs/))
- 📦 **Downloads:** [Releases](https://github.com/Phhofm/heart-nerve_trainner/releases)
- 🤗 **Models & demos:** [HEART demo](https://huggingface.co/spaces/Phips/heart) ·
  [NERVE demo](https://huggingface.co/spaces/Phips/nerve) ·
  [NERVE / HEART converter](https://huggingface.co/spaces/Phips/nerve-convert)

## For users
1. Download the installer for your OS from **Releases**.
2. Double-click, follow first-time setup (it installs everything into its own folder).
3. Pick **NERVE** (recommended) → **2×** → **Fidelity** → your HR folder → **Train**.
4. Your model appears in the output folder as `.safetensors`, `.onnx` (fp32 + fp16)
   and, for NERVE, `.ncnn`.

No GPU? The app warns that CPU training is very slow (possible for NERVE).

## Features
- **Live validation graphs** — PSNR / SSIM (fidelity) and DISTS / LPIPS (perceptual).
- **One-click Lucid download** — 100k CC0 training tiles, no account needed.
- **Auto-everything** — hardware detection, safe auto-params, self-test, repair, export.
- **In-app updates** — checks `latest.json`, then asks before downloading a new build.
- **AI-assisted help** — on failure it copies a friendly diagnostic report to your clipboard.

## For developers
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

pytest -q                                    # tests (GUI tests need PySide6)
python app.py                                # launch the GUI
hnt detect                                   # print detected hardware
hnt config --model NERVE --scale 2 --profile FIDELITY \
    --data-mode OTF --hr /path/hr --out /tmp/run.yml
hnt selftest                                 # end-to-end smoke (needs GPU/torch)
```

### Layout
```
hnt/
  paths.py         locate the trainer repo + app data dirs
  hardware.py      GPU/VRAM/CPU/RAM detection
  presets.py       profiles, arch defaults, VRAM→auto-params
  manifest.py      (model, scale) → pretrain
  degradation.py   OTF presets (bicubic/JPEG/Real-ESRGAN/…)
  config_gen.py    choices → valid traiNNer .yml  (template + patch)
  train_runner.py  run train.py, parse progress + metrics, sample telemetry
  exporter.py      safetensors + ONNX fp32/fp16 (+ ncnn)
  bootstrap.py     uv-based isolated venv + pinned deps
  selftest.py      NERVE 2x fidelity smoke → train/val/save/export → cleanup
  downloader.py    Lucid (CC0) dataset + pretrain downloads (login optional)
  updater.py       check latest.json → download + launch installer (keeps data/venv)
  cleanup.py       delete experiment dirs, never the final output folder
  infer.py         run a trained/downloaded HEART|NERVE model on images or folders
  diagnostics.py   "ask your AI" help text (copy-to-clipboard)
  ui/              PySide6 app (theme, main window, live charts, inference window)
tests/             pytest suite
docs/              the GitHub Pages website (static, no build step)
packaging/         Inno Setup (Windows) + AppImage (Linux) + vendoring notes
latest.json        version manifest the in-app updater reads
```

### Design rules
- **Beginner-first:** Fidelity + NERVE + sane auto-params by default; GAN is opt-in.
- **Never touch the system Python:** everything lives in the app's venv.
- **Auto-everything:** install, configure, test, repair, export.
- **AI-help, not AI-integration:** on failure we only *generate a text prompt* to copy.

## Status / roadmap
- ✅ Training GUI (flow, review, dashboard, live metric charts, inference window).
- ✅ Lucid (CC0) one-click download; pretrain manifest + resolver.
- ⏳ **Vendor + pin the trainer** under `packaging/vendor/trainer/` and the `uv`
  binary, then produce real installers (CI already stubbed in
  `.github/workflows/build-release.yml`).
- ⏳ Publish `latest.json` + a first Release so the website Download and in-app
  updates work (auto-update needs the repo public).
- ⏳ Validate the experimental Modern-perceptual profile with short runs.

## Credits
Made by **Philip Hofmann** — [GitHub](https://github.com/Phhofm) ·
[Hugging Face](https://huggingface.co/Phips). Built on
[traiNNer-redux](https://github.com/the-database/traiNNer-redux). MIT licensed.
