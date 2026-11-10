"""PULSE_fast smoke test: bundled paired stub → train → val → checkpoint → export."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

from hnt.paths import Paths, find_repo  # noqa: E402
from hnt.selftest import run_selftest  # noqa: E402


@pytest.mark.slow
def test_pulse_fast_smoke_test(tmp_path: Path):
    paths = Paths(repo=find_repo(), data=tmp_path)
    res = run_selftest(paths, model="PULSE_fast")
    assert res.ok, f"{res.stage}: {res.detail}\n{res.log_tail}"
