#!/usr/bin/env python3
"""Minimal dynamic ONNX export for PULSE / PULSE_fast checkpoints."""
import gc, sys
import torch
from safetensors.torch import load_file
from traiNNer.archs.pulse_arch import pulse, pulse_fast

ckpt, out, scale = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.set_grad_enabled(False)
sd = load_file(ckpt, device="cpu")
dim = int(sd["stem.weight"].shape[0])
arch = pulse_fast if dim == 32 else pulse
m = arch(scale=scale).eval()
m.load_state_dict(sd, strict=True)
del sd; gc.collect()
x = torch.randn(1, 3, 64, 64)
torch.onnx.export(
    m, x, out,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0:"batch",2:"height",3:"width"}, "output": {0:"batch",2:"height_out",3:"width_out"}},
    opset_version=17, do_constant_folding=True, dynamo=False,
)
print(f"OK -> {out}")
