#!/usr/bin/env python3
"""Minimal memory ONNX export. No validation, no onnxsim, tiny dummy shape.
If this fits in RAM, great. Otherwise the export must run on another machine."""
import gc, os, sys
import torch
from safetensors.torch import load_file
from traiNNer.archs.heart_arch import heart

ckpt, out, scale = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.set_num_threads(2)
torch.set_grad_enabled(False)

sd = load_file(ckpt, device="cpu")
m = heart(scale=scale, ape=False, use_checkpoint=False).eval()
m.load_state_dict(sd, strict=True)
del sd; gc.collect()

# tiny dummy to minimize activation memory during trace
x = torch.randn(1, 3, 64, 64)
torch.onnx.export(
    m, x, out,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0:"batch",2:"height",3:"width"}, "output": {0:"batch",2:"height_out",3:"width_out"}},
    opset_version=17, do_constant_folding=True, dynamo=False,
)
del m, x; gc.collect()
print(f"OK -> {out}")
