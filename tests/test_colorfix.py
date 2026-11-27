"""Wavelet color match: detail from the output, colors from the input."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hnt.colorfix import wavelet_color_match  # noqa: E402


def _flat(color: tuple[int, int, int], h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_identity_when_source_and_target_match():
    src = _flat((100, 120, 140))
    out = wavelet_color_match(src, src.copy())
    assert np.array_equal(out, src)


def test_flat_source_takes_target_color_exactly():
    src = _flat((10, 10, 10))       # model output: dark
    target = _flat((200, 150, 50))  # input: warm orange
    out = wavelet_color_match(src, target)
    # a flat source has no high-frequency band, so it should equal the
    # target's low-frequency color
    assert abs(int(out[32, 32, 0]) - 200) <= 2
    assert abs(int(out[32, 32, 1]) - 150) <= 2
    assert abs(int(out[32, 32, 2]) - 50) <= 2


def test_detail_from_source_survives():
    h, w = 64, 64
    target = _flat((120, 120, 120), h, w)
    src = _flat((120, 120, 120), h, w)
    # sparse isolated bright dots: unambiguously high-frequency detail
    for y, x in ((8, 12), (40, 30), (20, 50), (55, 5)):
        src[y, x] = (250, 250, 250)
    out = wavelet_color_match(src, target)
    # detail pixels stay bright (source's high band kept)
    assert out[8, 12, 0] > 200
    # background stays near the target's gray
    assert abs(int(out[2, 2, 0]) - 120) <= 3


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        wavelet_color_match(_flat((1, 2, 3)), _flat((1, 2, 3), 32, 32))


def test_odd_sizes_work():
    src = _flat((5, 5, 5), 33, 47)
    target = _flat((90, 90, 90), 33, 47)
    out = wavelet_color_match(src, target, levels=3)
    assert out.shape == (33, 47, 3)
