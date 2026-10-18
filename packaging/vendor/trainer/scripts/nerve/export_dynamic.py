"""Export NERVE (canonical, configurable size) or SPAN to dynamic-shape ONNX and verify.

- fp32 export with dynamic H/W on the lq/sr tensors
- onnx.checker
- ONNX Runtime CPU inference vs PyTorch CPU (max abs diff) at several shapes:
  48x48, 323x711, 720x1280.

Random weights prove the graph is export-clean; pass a trained EMA checkpoint
(--checkpoint) to validate an actual trained model. Canonical export is
opset 20 (torch 2.14's dynamo exporter has an opset floor of 18; an opset-17
request still emits 18).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.export import Dim
from traiNNer.archs.nerve_arch import nerve

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "experiments" / "nerve" / "export"

SHAPES: list[tuple[int, int]] = [(48, 48), (323, 711), (720, 1280)]


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file

            sd = load_file(str(path), device="cpu")
        except Exception:  # noqa: BLE001 - legacy pickle with .safetensors name
            import io

            sd = torch.load(
                io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True
            )
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
        )


def build(arch: str, scale: int, dim: int, n_blocks: int) -> torch.nn.Module:
    if arch == "span":
        from traiNNer.archs.span_arch import span

        return span(scale=scale)
    return nerve(scale=scale, dim=dim, n_blocks=n_blocks)


def export_onnx(
    model: torch.nn.Module,
    scale: int,
    opset: int,
    out_path: Path,
) -> None:
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    dynamic_shapes = {"x": {2: Dim("H"), 3: Dim("W")}}
    torch.onnx.export(
        model,
        (x,),
        str(out_path),
        opset_version=opset,
        dynamic_shapes=dynamic_shapes,
    )


def verify_ort(model: torch.nn.Module, onnx_path: Path, scale: int, tol: float) -> dict:
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    model.eval()
    results: dict[str, float] = {}
    with torch.no_grad():
        for h, w in SHAPES:
            x = torch.randn(1, 3, h, w)
            ref = model(x)
            ort_out = sess.run(None, {iname: x.numpy()})[0]
            diff = float(np.abs(ref.numpy() - ort_out).max())
            results[f"{h}x{w}"] = round(diff, 6)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="nerve", choices=["nerve", "span"])
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--n-blocks", type=int, default=16)
    parser.add_argument(
        "--checkpoint", default=None, help="trained EMA safetensors to load"
    )
    parser.add_argument(
        "--scale", type=int, default=4, help="scale when a checkpoint is given"
    )
    parser.add_argument("--scales", nargs="+", type=int, default=[4, 2])
    parser.add_argument("--opsets", nargs="+", type=int, default=[20])
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    args = parser.parse_args()

    scales = [args.scale] if args.checkpoint else args.scales
    overall_rc = 0
    summary: dict = {}

    for scale in scales:
        model = build(args.arch, scale, args.dim, args.n_blocks)
        model.to("cpu")
        model.eval()
        if args.checkpoint:
            load_checkpoint(model, Path(args.checkpoint))
        params = sum(p.numel() for p in model.parameters())
        print(f"arch {args.arch} scale {scale} params={params:,}", flush=True)

        scale_rc = 0
        for opset in args.opsets:
            tag = f"{args.arch}_d{args.dim}_b{args.n_blocks}_s{scale}"
            out_dir = Path(args.out_root) / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"nerve_dynamic_op{opset}.onnx"
            export_onnx(model, scale, opset, out_path)
            print(f"exported {out_path.name}", flush=True)
            try:
                diffs = verify_ort(model, out_path, scale, args.tol)
            except Exception as e:  # noqa: BLE001
                print(f"opset {opset} ORT verify FAILED: {e}", flush=True)
                scale_rc = 1
                continue
            worst = max(diffs.values()) if diffs else float("inf")
            ok = worst <= args.tol
            print(
                f"opset {opset} max|diff| per shape: {diffs} -> {'OK' if ok else 'FAIL'}",
                flush=True,
            )
            scale_rc |= 0 if ok else 1
            summary[f"arch{args.arch}_s{scale}_op{opset}"] = {
                "max_diff": diffs,
                "ok": ok,
            }
        overall_rc |= scale_rc

    (Path(args.out_root) / "export_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
