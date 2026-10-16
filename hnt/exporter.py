"""Export a trained checkpoint to deployable formats.

* ``.safetensors`` — copy the EMA weights (students load these directly);
* ``.onnx`` — dynamic-shape, fp32 **and** fp16;
* ``.ncnn`` — NERVE only (HEART's fused SDPA can't convert).

Reuses the repo's export scripts when present, else a built-in torch.onnx path.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExportResult:
    safetensors: Path | None = None
    onnx_fp32: Path | None = None
    onnx_fp16: Path | None = None
    ncnn_dir: Path | None = None
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def _run(cmd: list[str], repo: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def export_checkpoint(
    repo: Path, checkpoint: Path, out_dir: Path, model: str, scale: int,
    *, fp32: bool = True, fp16: bool = True, ncnn: bool = False,
) -> ExportResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(checkpoint).stem
    res = ExportResult()

    # 1. safetensors (EMA) — copy to the user's final-model folder.
    try:
        dst = out_dir / f"{model}_{scale}x_{stem}.safetensors"
        shutil.copy2(checkpoint, dst)
        res.safetensors = dst
    except OSError as e:
        res.errors.append(f"safetensors copy failed: {e}")

    # 2. ONNX via the repo's heart exporter (dynamic, opset 17) if available.
    exporter = repo / "scripts/heart/export_minimal.py"
    if exporter.is_file():
        for want, tag in ((fp32, "fp32"), (fp16, "fp16")):
            if not want:
                continue
            out = out_dir / f"{model}_{scale}x_{stem}_{tag}.onnx"
            rc, log = _run([sys.executable, str(exporter), str(checkpoint), str(out), str(scale)], repo)
            if rc != 0:
                res.errors.append(f"onnx {tag} export failed: {log}")
            elif tag == "fp32":
                res.onnx_fp32 = out
            else:
                res.onnx_fp16 = out
    else:
        res.errors.append("onnx exporter not found (scripts/heart/export_minimal.py)")

    # 3. ncnn — NERVE only.
    if ncnn and model == "NERVE":
        nerve_exporter = repo / "scripts/nerve/export_dynamic.py"
        target = out_dir / f"{model}_{scale}x_{stem}_ncnn"
        if nerve_exporter.is_file():
            rc, log = _run([sys.executable, str(nerve_exporter), str(checkpoint), str(target)], repo)
            if rc != 0:
                res.errors.append(f"ncnn export failed: {log}")
            else:
                res.ncnn_dir = target
        else:
            res.errors.append("ncnn exporter not found (scripts/nerve/export_dynamic.py)")
    return res
