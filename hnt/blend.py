"""Motion-adaptive blending for temporal inference.

Ports the research repo's blend semantics (``scripts/nerve/blend_motion.py``):
on high inter-frame motion the temporal output is blended toward a
single-frame result, which removes ghosting/line-boil while keeping the
temporal output on calm frames. All math is float-based (uint8 inputs are
normalized first) so identical inputs score/blend exactly.
"""

from __future__ import annotations

import numpy as np


def luma_diff(prev: np.ndarray, cur: np.ndarray) -> float:
    """Mean absolute luma difference between two RGB frames in [0, 1]."""
    if prev.shape != cur.shape:
        raise ValueError(
            f"frame shapes differ: {prev.shape} vs {cur.shape}"
        )
    a = prev.astype(np.float64)
    b = cur.astype(np.float64)
    if a.max() > 1.5:  # uint8 -> [0, 1]
        a = a / 255.0
        b = b / 255.0
    ya = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    yb = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    return float(np.mean(np.abs(ya - yb)))


def motion_weight(diff: float, m0: float, m1: float) -> float:
    """Ramp 0->1 as the luma difference crosses [m0, m1]."""
    if m1 <= m0:
        raise ValueError("m1 must be greater than m0")
    w = (diff - m0) / (m1 - m0)
    return float(min(1.0, max(0.0, w)))


def blend_frames(temporal: np.ndarray, static: np.ndarray,
                 w: float) -> np.ndarray:
    """Weighted mix of two same-shape frames; w=1 returns static."""
    if temporal.shape != static.shape:
        raise ValueError(
            f"frame shapes differ: {temporal.shape} vs {static.shape}"
        )
    if w <= 0.0:
        return temporal
    if w >= 1.0:
        return static
    a = temporal.astype(np.float64)
    b = static.astype(np.float64)
    if a.max() > 1.5:
        a = a / 255.0
        b = b / 255.0
        out = a * (1.0 - w) + b * w
        return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return a * (1.0 - w) + b * w
