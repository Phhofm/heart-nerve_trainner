"""Wavelet color match: keep the model's detail, take the input's colors.

Standard multiscale trick (FreeDom/DiffBIR lineage): split the model output
into low/high frequency bands with a coarse average-pool pyramid, then
recombine the output's HIGH band with the INPUT's LOW band. Useful as a
guard against color drift (e.g. restoration models trained on photo stats
slowly re-toning stylized content) — off by default, opt-in per run.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812


def _low_freq(x: torch.Tensor, levels: int = 4) -> torch.Tensor:
    """Dyadic average pyramid: downsample 2x per level, upsample back."""
    h, w = x.shape[-2:]
    step = 2 ** levels
    ph = (step - h % step) % step
    pw = (step - w % step) % step
    cur = F.pad(x, (0, pw, 0, ph), mode="replicate")
    for _ in range(levels):
        cur = F.avg_pool2d(cur, 2)
    low = F.interpolate(cur, size=(h + ph, w + pw), mode="bilinear",
                        align_corners=False)
    return low[..., :h, :w]


def wavelet_color_match(
    source: np.ndarray, target: np.ndarray, levels: int = 5
) -> np.ndarray:
    """Recolor ``source`` (model output, HxWx3 uint8) with ``target``'s
    (input image) low-frequency colors. Shapes must match; both RGB uint8.
    """
    if source.shape != target.shape:
        raise ValueError(
            f"shape mismatch: source {source.shape} vs target {target.shape}")

    def to_tensor(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)[None]

    def to_array(t: torch.Tensor) -> np.ndarray:
        out = t[0].permute(1, 2, 0).clamp(0, 1).numpy()
        return (out * 255.0).round().astype(np.uint8)

    with torch.no_grad():
        s = to_tensor(source)
        t = to_tensor(target)
        s_low = _low_freq(s, levels)
        t_low = _low_freq(t, levels)
        matched = t_low + (s - s_low)
    return to_array(matched)
