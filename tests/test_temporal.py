"""Tests for cut-safe temporal window planning (hnt/temporal.py).

Semantics mirror the research repo's cut-safe inference
(scripts/nerve/eval_tnerve_clip.py): a frame gets a full centered window
only when the whole window fits inside its scene; otherwise the temporal
model must not be used and the caller falls back to single-frame
processing.
"""

from hnt.temporal import detect_cuts_scaled, plan_windows


def test_plan_windows_never_cross_cuts() -> None:
    plan = plan_windows(total=45, cuts=[15, 30], clip_size=13)
    assert len(plan) == 45
    for i, window in enumerate(plan):
        if window is None:
            continue
        assert len(window) == 13
        scene_start = 0 if i < 15 else (15 if i < 30 else 30)
        scene_end = 15 if i < 15 else (30 if i < 30 else 45)
        assert window[0] >= scene_start
        assert window[-1] < scene_end
        assert window[window.index(i)] == i  # center frame is in the window
    # with 15-frame scenes, full-window centers are 3 per scene
    full = [i for i, w in enumerate(plan) if w is not None]
    assert full == [6, 7, 8, 21, 22, 23, 36, 37, 38]


def test_plan_windows_single_scene_all_full() -> None:
    plan = plan_windows(total=20, cuts=[], clip_size=5)
    assert plan[0] is None  # no left neighbors
    assert plan[19] is None  # no right neighbors
    assert plan[2] == [0, 1, 2, 3, 4]
    assert plan[10] == [8, 9, 10, 11, 12]


def test_plan_windows_clip_size_one_is_identity() -> None:
    plan = plan_windows(total=4, cuts=[2], clip_size=1)
    assert plan == [[0], [1], [2], [3]]


def test_detect_cuts_scaled_finds_hard_cut() -> None:
    import numpy as np

    a = np.full((48, 64, 3), 0, dtype=np.uint8)
    b = np.full((48, 64, 3), 255, dtype=np.uint8)
    cuts = detect_cuts_scaled([a, a, b, b, a], threshold=0.35)
    assert cuts == [2, 4]  # b->a is a cut too
