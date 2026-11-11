"""Scene-cut helpers for temporal video models and inference.

Keeping these helpers separate from the ffmpeg pipeline makes them reusable by
both training datasets and BODY Studio's video workflow.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def frame_difference(prev: np.ndarray, cur: np.ndarray) -> float:
    """Return normalized mean absolute pixel difference between two frames."""
    if prev.shape != cur.shape:
        raise ValueError("frames must have the same shape")
    if prev.size == 0:
        return 0.0
    return float(np.mean(np.abs(prev.astype(np.float32) - cur.astype(np.float32))) / 255.0)


def detect_scene_cuts(frames: Sequence[np.ndarray], threshold: float = 0.35) -> list[int]:
    """Return indices where a hard scene cut begins.

    The returned indices are the first frames of new shots. Index 0 is never
    considered a cut because there is no preceding frame.
    """
    cuts: list[int] = []
    for i in range(1, len(frames)):
        if frame_difference(frames[i - 1], frames[i]) >= threshold:
            cuts.append(i)
    return cuts


def scene_bounds(cuts: Sequence[int], index: int, total: int) -> tuple[int, int]:
    """Return the inclusive/exclusive frame range of the shot containing ``index``."""
    if total <= 0:
        return (0, 0)
    if not 0 <= index < total:
        raise IndexError(index)
    cuts = sorted(cuts)
    start = 0
    for cut in cuts:
        if cut <= index:
            start = cut
        else:
            break
    end = total
    for cut in cuts:
        if cut > index:
            end = cut
            break
    return (start, end)


def window_indices(
    index: int,
    total: int,
    cuts: Sequence[int],
    num_frames: int,
) -> list[int]:
    """Return frame indices for a centered temporal window.

    The window is clipped to the scene containing ``index``. If one side has
    fewer available frames, the window expands toward the other side. Short
    scenes can yield fewer than ``num_frames`` indices.
    """
    if num_frames <= 0:
        return []
    start, end = scene_bounds(cuts, index, total)
    half = num_frames // 2
    lo = max(start, index - half)
    hi = min(end, index + half + 1)

    # Expand the smaller side when a scene boundary prevents a centered window.
    while hi - lo < num_frames:
        if lo > start and hi < end:
            if (index - lo) <= (hi - index):
                hi += 1
            else:
                lo -= 1
        elif lo > start:
            lo -= 1
        elif hi < end:
            hi += 1
        else:
            break
    return list(range(lo, hi))
