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
from .hardware import detect
from .paths import Paths


@dataclass
class SelfTestResult:
    ok: bool
    stage: str            # last stage reached
    log_tail: str = ""
    detail: str = ""


# Small, safe smoke settings.
_SMOKE_ITERS = 100


def _demo_dir(repo: Path) -> Path | None:
    for rel in ("datasets/train/demo/hr", "datasets/train/dataset1"):
        hr = repo / rel
        if hr.is_dir() and any(hr.iterdir()):
            return hr
    return None


def run_selftest(paths: Paths, python_exe: str | None = None, on_log=lambda m: None) -> SelfTestResult:
    python_exe = python_exe or sys.executable
    hr = _demo_dir(paths.repo)
    if hr is None:
        return SelfTestResult(False, "find-demo-data", detail="bundled demo dataset missing")

    run_name = "hnt_selftest"
    run_dir = paths.repo / "experiments" / run_name

    # OTF + clean bicubic on the bundled demo images. build_config applies the
    # degradation knobs at the config root, and the RELEASE template already brings
    # a valid validation set (Urban100), so nothing else needs patching.
    req = TrainRequest(
        model="NERVE", scale=2, profile="FIDELITY", data_mode="OTF",
        hr_dir=str(hr), degradation="bicubic",
        batch=2, lq_size=64, iters=_SMOKE_ITERS, name=run_name, use_compile=False,
        amp_bf16=detect().supports_bf16,
    )
    cfg = build_config(req, paths.repo)
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

    ckpts = sorted((run_dir / "models").glob("net_g_ema_*.safetensors")) or sorted(
        (run_dir / "models").glob("net_g_*.safetensors"))
    if not ckpts:
        return SelfTestResult(False, "save-checkpoint", tail, "no checkpoint written")

    on_log("Exporting self-test checkpoint …")
    res = export_checkpoint(paths.repo, ckpts[-1], paths.data / "selftest_out", "NERVE", 2,
                            fp32=True, fp16=True, ncnn=True)
    detail = "; ".join(res.errors)
    if res.safetensors is None:
        return SelfTestResult(False, "export-safetensors", tail, detail or "no safetensors")
    if res.onnx_fp32 is None:
        return SelfTestResult(False, "export-onnx", tail, detail)
    if res.onnx_fp16 is None:
        return SelfTestResult(False, "export-onnx-fp16", tail, detail)
    if res.ncnn_dir is None:
        return SelfTestResult(False, "export-ncnn", tail, detail)

    # Clean up the temp run + config so the user's machine is left tidy.
    shutil.rmtree(run_dir, ignore_errors=True)
    cfg_path.unlink(missing_ok=True)
    return SelfTestResult(True, "done", tail, "train→validate→save→export OK")
