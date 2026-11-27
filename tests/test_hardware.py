"""First-party coverage for hnt.hardware's pure decision logic.

Zero-coverage module found during onboarding — these pin the VRAM /
bf16 heuristics the presets rely on, plus the cross-platform quiet()
flags (mypy flagged CREATE_NO_WINDOW as POSIX-undefined; the getattr
form is runtime-honest on every platform).
"""
from __future__ import annotations

import subprocess

from hnt.hardware import GPU, Hardware, quiet


def _hw(*caps: float | None) -> Hardware:
    gpus = [GPU(name=f"gpu{i}", vram_gb=24.0, compute_cap=c)
            for i, c in enumerate(caps)]
    return Hardware(os="linux", os_version="x", python="3.13", cpu="x",
                    ram_gb=32.0, gpus=gpus)


def test_best_vram_gb_empty_gpus_is_zero():
    assert _hw().best_vram_gb == 0.0
    assert _hw(8.9).best_vram_gb == 24.0


def test_supports_bf16_defaults_true_without_capability():
    assert _hw().supports_bf16 is True


def test_supports_bf16_pre_ampere_is_false():
    assert _hw(7.5).supports_bf16 is False
    # any Ampere+ card satisfies the heuristic (max over caps)
    assert _hw(8.6, 7.5).supports_bf16 is True


def test_supports_bf16_ampere_and_newer_is_true():
    assert _hw(8.9).supports_bf16 is True
    assert _hw(7.5, 8.6).supports_bf16 is True  # any Ampere+ card


def test_quiet_is_empty_on_posix(monkeypatch):
    monkeypatch.setattr("hnt.hardware.os.name", "posix")
    assert quiet() == {}


def test_quiet_carries_no_window_flag_on_windows(monkeypatch):
    monkeypatch.setattr("hnt.hardware.os.name", "nt")
    flags = quiet()
    assert "creationflags" in flags
    assert flags["creationflags"] == getattr(
        subprocess, "CREATE_NO_WINDOW", flags["creationflags"]
    )
