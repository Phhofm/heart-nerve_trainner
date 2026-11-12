"""Cut-safe temporal window planning for BODY Studio inference.

Semantics mirror the research repo's cut-safe inference
(``scripts/nerve/eval_tnerve_clip.py``): a frame receives a full centered
window only when the entire window fits inside its scene; otherwise the
temporal model must not be used and the caller falls back to single-frame
processing. Windows therefore never span scene cuts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .scenes import detect_scene_cuts


def plan_windows(
    total: int,
    cuts: Sequence[int],
    clip_size: int,
) -> list[list[int] | None]:
    """Per-frame temporal window plan.

    Entry ``i`` is the ordered list of frame indices for frame ``i``'s
    window (``clip_size`` long, centered when possible), or ``None`` when
    no full window fits inside frame ``i``'s scene (single-frame fallback).
    """
    if clip_size < 1:
        raise ValueError("clip_size must be >= 1")
    if clip_size % 2 == 0:
        raise ValueError("clip_size must be odd")
    plan: list[list[int] | None] = []
    half = clip_size // 2
    ordered = sorted(cuts)
    for i in range(total):
        lo = i - half
        hi = i + half + 1
        if lo < 0 or hi > total:
            plan.append(None)
            continue
        scene_start = 0
        for cut in ordered:
            if cut <= i:
                scene_start = cut
            else:
                break
        scene_end = total
        for cut in ordered:
            if cut > i:
                scene_end = cut
                break
        if lo >= scene_start and hi <= scene_end:
            plan.append(list(range(lo, hi)))
        else:
            plan.append(None)
    return plan


def detect_cuts_scaled(
    frames: Sequence[np.ndarray],
    threshold: float = 0.35,
    long_side: int = 320,
) -> list[int]:
    """Scene-cut indices from (possibly downscaled) preview frames.

    Frames are RGB uint8/float arrays; they are downscaled so the scan
    stays cheap on long videos. Cut indices refer to the original sequence.
    """
    if long_side <= 0 or not frames:
        return list(detect_scene_cuts(frames, threshold))
    scaled: list[np.ndarray] = []
    h, w = frames[0].shape[:2]
    if max(h, w) <= long_side:
        return list(detect_scene_cuts(frames, threshold))
    ratio = long_side / max(h, w)
    size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
    for f in frames:
        scaled.append(
            cv2_resize(f, size)
        )
    return list(detect_scene_cuts(scaled, threshold))


def cv2_resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
