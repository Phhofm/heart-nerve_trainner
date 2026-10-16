"""Export a trained checkpoint to deployable formats.

* ``.safetensors`` — copy the EMA weights (students load these directly);
* ``.onnx`` — fp32 **and** fp16, via the model-specific export script
  (``scripts/heart/export_minimal.py`` for HEART, ``scripts/nerve/export_dynamic.py``
  for NERVE);
* ``.ncnn`` — NERVE only, when a converter is available.
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


def _nerve_arch_from_checkpoint(checkpoint: Path) -> tuple[int, int]:
    """Infer (dim, n_blocks) from a NERVE safetensors state_dict; fall back to defaults."""
    try:
        from safetensors import safe_open

        with safe_open(str(checkpoint), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            dim = int(f.get_slice("blocks.0.conv1.weight").get_shape()[0])
            blocks = {int(k.split(".")[1]) for k in keys if k.startswith("blocks.")}
            n_blocks = max(blocks) + 1 if blocks else 16
        return dim, n_blocks
    except Exception:  # noqa: BLE001
        return 48, 16


def _embed_onnx(src: Path, dst: Path) -> str | None:
    """Re-save an ONNX model with weights embedded (no external .data sidecar)."""
    try:
        import onnx

        onnx.save_model(onnx.load(str(src)), str(dst), save_as_external_data=False)
        return None
    except Exception as e:  # noqa: BLE001
        try:
            if src != dst:
                shutil.copy2(src, dst)
            for side in Path(src).parent.glob(Path(src).name + ".*"):
                shutil.copy2(side, Path(dst).parent / side.name)
        except OSError:
            pass
        return f"could not embed onnx weights ({e}); kept external-data files"


def _export_onnx_nerve(
    repo: Path, checkpoint: Path, out_dir: Path, stem: str, scale: int
) -> tuple[Path | None, list[str]]:
    script = repo / "scripts/nerve/export_dynamic.py"
    if not script.is_file():
        return None, ["nerve onnx exporter not found (scripts/nerve/export_dynamic.py)"]
    dim, n_blocks = _nerve_arch_from_checkpoint(checkpoint)
    tmp = out_dir / "_nerve_onnx"
    rc, log = _run(
        [sys.executable, str(script), "--checkpoint", str(checkpoint),
         "--scale", str(scale), "--dim", str(dim), "--n-blocks", str(n_blocks),
         "--out-root", str(tmp)],
        repo,
    )
    produced = sorted(tmp.glob("**/*.onnx"))
    if not produced:
        return None, [f"nerve onnx export failed: {log}"]
    dst = out_dir / f"NERVE_{scale}x_{stem}_fp32.onnx"
    warn = _embed_onnx(produced[-1], dst)
    shutil.rmtree(tmp, ignore_errors=True)
    return dst, [warn] if warn else []


def _export_onnx_heart(
    repo: Path, checkpoint: Path, out_dir: Path, stem: str, scale: int
) -> tuple[Path | None, list[str]]:
    script = repo / "scripts/heart/export_minimal.py"
    if not script.is_file():
        return None, ["heart onnx exporter not found (scripts/heart/export_minimal.py)"]
    dst = out_dir / f"HEART_{scale}x_{stem}_fp32.onnx"
    rc, log = _run([sys.executable, str(script), str(checkpoint), str(dst), str(scale)], repo)
    if rc != 0 or not dst.is_file():
        return None, [f"onnx fp32 export failed: {log}"]
    warn = _embed_onnx(dst, dst)
    return dst, [warn] if warn else []


def _to_fp16(src: Path, dst: Path) -> str | None:
    """Convert an fp32 ONNX file to fp16. Returns an error string, or None on success."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as e:
        return f"onnx fp16 skipped (needs onnxconverter-common): {e}"
    try:
        model16 = float16.convert_float_to_float16(onnx.load(str(src)))
        onnx.save_model(model16, str(dst), save_as_external_data=False)
        return None
    except Exception as e:  # noqa: BLE001
        return f"onnx fp16 conversion failed: {e}"


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

    # 2. ONNX — via the model's own exporter (fp32 first, then optional fp16).
    onnx_fp32 = None
    if fp32 or fp16:
        if model == "NERVE":
            onnx_fp32, errs = _export_onnx_nerve(repo, Path(checkpoint), out_dir, stem, scale)
        else:
            onnx_fp32, errs = _export_onnx_heart(repo, Path(checkpoint), out_dir, stem, scale)
        res.errors.extend(errs)
    if onnx_fp32 is not None:
        if fp32:
            res.onnx_fp32 = onnx_fp32
        if fp16:
            dst16 = out_dir / f"{model}_{scale}x_{stem}_fp16.onnx"
            err = _to_fp16(onnx_fp32, dst16)
            if err:
                res.errors.append(err)
            else:
                res.onnx_fp16 = dst16

    # 3. ncnn — NERVE only, and only if a converter happens to be present.
    if ncnn:
        res.errors.append(
            "ncnn export unavailable (add an ncnn converter to enable it)" if model == "NERVE"
            else "ncnn is NERVE-only"
        )
    return res
