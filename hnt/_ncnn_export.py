"""Convert a NERVE checkpoint to ncnn (TorchScript → pnnx).

Run with ``cwd`` set to the trainer repo so ``traiNNer`` imports resolve:

    python hnt/_ncnn_export.py --checkpoint ckpt.safetensors --scale 2 \
        --out-dir out --name NERVE_2x_ckpt

Prints a single JSON object describing the result.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.getcwd())  # trainer repo


def _load_state(ckpt: Path) -> dict:
    from safetensors.torch import load_file

    sd = load_file(str(ckpt), device="cpu")
    for key in ("params_ema", "params"):
        if isinstance(sd, dict) and key in sd:
            return sd[key]
    return sd


def _infer_arch(sd: dict) -> tuple[int, int]:
    dim = int(sd["blocks.0.conv1.weight"].shape[0])
    blocks = {int(k.split(".")[1]) for k in sd if k.startswith("blocks.")}
    return dim, (max(blocks) + 1 if blocks else 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dim", type=int, default=0)
    ap.add_argument("--n-blocks", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    import torch
    from traiNNer.archs.nerve_arch import nerve

    ckpt = Path(a.checkpoint)
    sd = _load_state(ckpt)
    dim, n_blocks = _infer_arch(sd)
    dim = a.dim or dim
    n_blocks = a.n_blocks or n_blocks

    net = nerve(scale=a.scale, dim=dim, n_blocks=n_blocks).eval()
    net.load_state_dict(sd, strict=True)

    pnnx = Path(sys.executable).with_name("pnnx")
    if not pnnx.is_file():
        pnnx = Path(shutil.which("pnnx") or "pnnx")

    with tempfile.TemporaryDirectory() as td_name:
        td = Path(td_name)
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            traced = torch.jit.trace(net, x)
        traced.save(str(td / "model.pt"))

        proc = subprocess.run(
            [str(pnnx), "model.pt", "inputshape=[1,3,64,64]"],
            cwd=td, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(json.dumps({"ok": False, "stage": "pnnx",
                              "detail": (proc.stdout + proc.stderr)[-3000:]}))
            return 1
        param = next(td.glob("*.ncnn.param"), None)
        binf = next(td.glob("*.ncnn.bin"), None)
        if param is None or binf is None:
            print(json.dumps({"ok": False, "stage": "pnnx-output",
                              "detail": [p.name for p in td.iterdir()]}))
            return 1

        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dst_param = out / f"{a.name}.ncnn.param"
        dst_bin = out / f"{a.name}.ncnn.bin"
        shutil.copy2(param, dst_param)
        shutil.copy2(binf, dst_bin)

    result: dict = {"ok": True, "param": str(dst_param), "bin": str(dst_bin),
                    "dim": dim, "n_blocks": n_blocks}
    if a.verify:
        result["verify"] = _verify(dst_param, dst_bin, net, a.scale, dim)
    print(json.dumps(result))
    return 0


def _verify(param: Path, binf: Path, net, scale: int, dim: int) -> dict:
    """Compare ncnn output with PyTorch on one random 64x64 input."""
    try:
        import numpy as np
        import ncnn
        import torch
    except ImportError as e:
        return {"skipped": f"ncnn python unavailable: {e}"}

    text = param.read_text()
    in_name = "in0"
    out_name = "out0"
    for line in text.splitlines()[2:]:  # skip magic + counts
        parts = line.split()
        if len(parts) < 5:
            continue
        ltype = parts[0]
        n_in = int(parts[2])
        n_out = int(parts[3])
        outs = parts[4 + n_in:4 + n_in + n_out]
        if ltype == "Input" and outs:
            in_name = outs[0]
        if outs:
            out_name = outs[0]

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        ref = net(x).numpy()[0]

    arr = np.ascontiguousarray(x.numpy()[0])
    mat_in = ncnn.Mat(arr).clone()  # clone: the Mat must own its memory
    net_ncnn = ncnn.Net()
    net_ncnn.opt.use_vulkan_compute = False
    net_ncnn.load_param(str(param))
    net_ncnn.load_model(str(binf))
    with net_ncnn.create_extractor() as ex:
        ret = ex.input(in_name, mat_in)
        ret2, out = ex.extract(out_name)
    if ret != 0 or ret2 != 0:
        return {"error": f"input={ret} extract={ret2} names={in_name}/{out_name}"}

    got = np.array(out)
    if got.shape != ref.shape:
        return {"error": f"shape {got.shape} != {ref.shape}"}
    return {"max_abs_diff": float(np.abs(got - ref).max()), "input": in_name,
            "output": out_name}


if __name__ == "__main__":
    sys.exit(main())