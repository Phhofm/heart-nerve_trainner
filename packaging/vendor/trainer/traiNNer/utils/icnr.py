"""ICNR initialization for sub-pixel (PixelShuffle) upsamplers.

ICNR (Aitken et al., "Checkerboard artifact free sub-pixel convolution")
initializes the conv that feeds a PixelShuffle so every sub-pixel phase sees
the same base kernel. The output therefore starts as a nearest-neighbour
upsampling of a single convolution, which removes the checkerboard bias that
plain random init leaves in the upsampling head.
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import nn


def icnr_init(conv: nn.Conv2d, upscale: int) -> None:
    """Re-initialize ``conv`` in-place with ICNR for a PixelShuffle(upscale)."""
    out_c = conv.out_channels
    sub = upscale * upscale
    assert out_c % sub == 0, f"out_c={out_c} not divisible by {sub}"
    base_c = out_c // sub
    base = nn.init.kaiming_uniform_(conv.weight[:base_c].clone(), a=math.sqrt(5))
    with torch.no_grad():
        conv.weight.data.zero_()
        for i in range(sub):
            conv.weight.data[i::sub] = base
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def icnr_reinit(model: nn.Module) -> int:
    """Apply ICNR to every Conv2d directly followed by a PixelShuffle.

    Walks the module tree; for each ``nn.Sequential`` with an adjacent
    (Conv2d, PixelShuffle) pair, re-initializes the conv. Returns the number
    of convs re-initialized (0 means nothing was found).
    """
    count = 0
    for _name, m in model.named_modules():
        if not isinstance(m, nn.Sequential):
            continue
        for prev, nxt in itertools.pairwise(list(m)):
            if isinstance(prev, nn.Conv2d) and isinstance(nxt, nn.PixelShuffle):
                icnr_init(prev, nxt.upscale_factor)
                count += 1
    return count
