"""Regression: GPU telemetry silently read zeros.

`read_telemetry` called `**quiet()` while the helper is imported as
`_quiet` — the NameError was swallowed by the bare `except`, so the
dashboard's GPU tiles never populated (onboarding finding, 2026-11-27).
"""
from __future__ import annotations

import subprocess

from hnt.train_runner import read_telemetry


def test_read_telemetry_parses_nvidia_smi(monkeypatch):
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="31, 4096, 8192, 55\n", stderr="",
    )
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return fake

    monkeypatch.setattr("hnt.train_runner.subprocess.run", fake_run)

    t = read_telemetry()

    assert calls, "nvidia-smi was not invoked"
    assert (t.gpu_util, t.vram_used_gb, t.vram_total_gb, t.gpu_temp) == (
        31.0, 4.0, 8.0, 55.0,
    )
