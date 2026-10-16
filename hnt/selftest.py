"""End-to-end self-test: prove the whole pipeline works before a real run.

Runs a tiny NERVE 2x *fidelity* training on the repo's bundled demo data, checks
that it trains → validates → saves a checkpoint → exports, then deletes the temp
run. Fast (a couple of minutes) and dependency-light (no discriminator).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config_gen import TrainRequest, build_config, write_config
from .exporter import export_checkpoint
from .paths import Paths


@dataclass
class SelfTestResult:
    ok: bool
    stage: str            # last stage reached
    log_tail: str = ""
    detail: str = ""


# Small, safe smoke settings.
_SMOKE_ITERS = 100


def _demo_dirs(repo: Path) -> tuple[Path, Path] | None:
    hr = repo / "datasets/train/dataset1"
    val = repo / "datasets/val/dataset1"
    if hr.is_dir() and val.is_dir() and any(hr.iterdir()):
        return hr, val
    return None


def run_selftest(paths: Paths, python_exe: str | None = None, on_log=lambda m: None) -> SelfTestResult:
    python_exe = python_exe or sys.executable
    demo = _demo_dirs(paths.repo)
    if demo is None:
        return SelfTestResult(False, "find-demo-data", detail="bundled demo dataset missing")

    hr, val = demo
    run_name = "hnt_selftest"
    run_dir = paths.repo / "experiments" / run_name

    req = TrainRequest(
        model="NERVE", scale=2, profile="FIDELITY", data_mode="PAIRED",
        hr_dir=str(hr), lr_dir=str(hr),          # placeholders patched below
        val_hr_dir=str(val), val_lr_dir=str(val),
        batch=2, lq_size=64, iters=_SMOKE_ITERS, name=run_name, use_compile=False,
    )
    cfg = build_config(req, paths.repo)
    # For the smoke test we lack paired LR, so degrade on the fly (clean bicubic).
    ds = cfg["datasets"]["train"]
    ds["type"] = "realesrgandataset"
    ds.pop("dataroot_lq", None)
    ds.update({"blur_prob": 0.0, "resize_prob": [0.0, 1.0, 0.0],
               "resize_mode_list": ["bicubic"], "resize_mode_prob": [1.0],
               "resize_range": [1.0, 1.0], "gaussian_noise_prob": 0.0, "jpeg_prob": 0.0,
               "final_sinc_prob": 0.0, "lq_size": 64, "batch_size_per_gpu": 2})
    cfg["datasets"]["val"]["filename_tmpl"] = "{}x2"
    cfg["logger"]["save_checkpoint_freq"] = 50
    cfg["logger"]["print_freq"] = 25
    cfg["val"]["val_freq"] = 50

    cfg_path = write_config(cfg, paths.repo / f"options/train/ABLATION/{run_name}.yml")
    log_path = paths.logs / f"{run_name}.log"
    on_log("Running self-test: NERVE 2x, ~100 iters on bundled demo data …")

    with log_path.open("w") as logf:
        rc = subprocess.run(
            [python_exe, "train.py", "-opt", str(cfg_path), "--auto_resume"],
            cwd=paths.repo, stdout=logf, stderr=subprocess.STDOUT, text=True, timeout=3600,
        ).returncode
    tail = log_path.read_text(errors="ignore")[-1500:]
    if rc != 0:
        return SelfTestResult(False, "train", tail, f"training exited {rc}")

    ckpts = sorted((run_dir / "models").glob("net_g_ema_*.safetensors"))
    if not ckpts:
        return SelfTestResult(False, "save-checkpoint", tail, "no checkpoint written")

    on_log("Exporting self-test checkpoint …")
    res = export_checkpoint(paths.repo, ckpts[-1], paths.data / "selftest_out", "NERVE", 2,
                            fp32=True, fp16=False, ncnn=False)
    if res.onnx_fp32 is None:
        return SelfTestResult(False, "export-onnx", tail, "; ".join(res.errors))

    # Clean up the temp run + config so the user's machine is left tidy.
    shutil.rmtree(run_dir, ignore_errors=True)
    cfg_path.unlink(missing_ok=True)
    return SelfTestResult(True, "done", tail, "train→validate→save→export OK")
