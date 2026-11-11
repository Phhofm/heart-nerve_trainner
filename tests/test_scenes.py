import numpy as np
import pytest

from hnt.scenes import detect_scene_cuts, scene_bounds, window_indices


def _frames(values):
    return [np.full((1, 1, 3), v, dtype=np.uint8) for v in values]


def test_detect_scene_cuts_flags_strong_changes_only():
    frames = _frames([0, 0, 255, 255, 0, 255])
    assert detect_scene_cuts(frames, threshold=0.5) == [2, 4, 5]


def test_detect_scene_cuts_ignores_tiny_motion():
    frames = _frames([0, 4, 8, 12])
    assert detect_scene_cuts(frames, threshold=0.5) == []


def test_scene_bounds_respects_cuts():
    assert scene_bounds([3, 7], 0, 10) == (0, 3)
    assert scene_bounds([3, 7], 3, 10) == (3, 7)
    assert scene_bounds([3, 7], 9, 10) == (7, 10)
    assert scene_bounds([], 4, 10) == (0, 10)


def test_window_indices_fills_around_scene_edges():
    assert window_indices(0, 10, [], 3) == [0, 1, 2]
    assert window_indices(9, 10, [], 3) == [7, 8, 9]
    assert window_indices(4, 10, [3, 7], 3) == [3, 4, 5]
    assert window_indices(4, 10, [], 5) == [2, 3, 4, 5, 6]


def test_window_indices_can_shrink_in_short_scenes():
    assert window_indices(1, 3, [1, 2], 3) == [1]
    assert window_indices(0, 1, [], 3) == [0]


def test_window_indices_rejects_out_of_range():
    with pytest.raises(IndexError):
        window_indices(10, 10, [], 3)
