"""Tests for the motion-adaptive blend helpers."""

from __future__ import annotations

import numpy as np
import pytest

from hnt.blend import blend_frames, luma_diff, motion_weight


def test_luma_diff_zero_for_identical_and_scale_safe() -> None:
    x = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    assert luma_diff(x, x) == 0.0
    assert luma_diff(x / 255.0, x / 255.0) == 0.0
    # a bright shift moves luma measurably
    y = x.copy()
    y[:, :, :] = np.clip(y.astype(int) + 60, 0, 255).astype(np.uint8)
    assert 0.05 < luma_diff(x, y) < 0.5


def test_motion_weight_ramp() -> None:
    assert motion_weight(0.0, 0.1, 0.3) == 0.0
    assert motion_weight(0.5, 0.1, 0.3) == 1.0
    assert abs(motion_weight(0.2, 0.1, 0.3) - 0.5) < 1e-9
    with pytest.raises(ValueError):
        motion_weight(0.2, 0.3, 0.3)


def test_blend_endpoints_and_mix() -> None:
    t = np.full((4, 4, 3), 40, np.uint8)
    s = np.full((4, 4, 3), 200, np.uint8)
    assert np.array_equal(blend_frames(t, s, 0.0), t)
    assert np.array_equal(blend_frames(t, s, 1.0), s)
    mid = blend_frames(t, s, 0.5)
    assert 115 <= int(mid[0, 0, 0]) <= 125  # uint8 mix lands near (40+200)/2
    with pytest.raises(ValueError):
        blend_frames(t, np.zeros((2, 2, 3), np.uint8), 0.5)
