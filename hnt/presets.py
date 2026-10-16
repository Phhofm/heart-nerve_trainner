"""Profiles, model arch defaults, and VRAM-aware auto-parameters.

Design rule for beginners: FIDELITY is the default (stable, no GAN). PERCEPTUAL
(GAN) is opt-in and warned about. All numbers here mirror the release configs in
``options/train/RELEASE`` and are intentionally conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Model = Literal["HEART", "NERVE"]
Profile = Literal["FIDELITY", "PERCEPTUAL", "MODERN_PERCEPTUAL", "REALWORLD_FIDELITY"]
DataMode = Literal["PAIRED", "OTF"]

SCALES = (1, 2, 4)
DEFAULT_MODEL: Model = "NERVE"
DEFAULT_SCALE = 2
DEFAULT_PROFILE: Profile = "FIDELITY"


@dataclass(frozen=True)
class ProfileInfo:
    label: str
    description: str
    gan: bool
    otf: bool
    # conservative relative cost multiplier for VRAM sizing (GAN = heavier)
    cost: float
    experimental: bool = False  # hidden in the UI until validated; code kept for later


PROFILES: dict[Profile, ProfileInfo] = {
    "FIDELITY": ProfileInfo(
        "Fidelity", "Cleanest result, fast and stable. Recommended for beginners.", False, False, 1.0
    ),
    "PERCEPTUAL": ProfileInfo(
        "Perceptual (GAN)", "Sharper/more natural textures. Advanced: GAN can be unstable.", True, True, 1.6
    ),
    "MODERN_PERCEPTUAL": ProfileInfo(
        "Modern perceptual (experimental)",
        "Two-phase GAN fine-tune with ConvNeXt perceptual + R3GAN-style discriminator + "
        "frequency loss. Best-looking results, but experimental — needs a fidelity model first.",
        True, True, 1.9, experimental=True,
    ),
    "REALWORLD_FIDELITY": ProfileInfo(
        "Real-world fidelity", "Robust to real degradations (blur/noise/JPEG), no GAN.", False, True, 1.15
    ),
}

# Arch defaults (mirror the release configs). `window` is the tiling/pad granularity;
# lq_size must be a multiple of it.
ARCH = {
    "HEART": {"type": "heart", "embed_dim": 180, "depths": [6, 6, 6, 6, 6, 6],
              "rank": 8, "attention_freq": 2, "window": 32},
    "NERVE": {"type": "nerve", "dim": 48, "n_blocks": 16, "window": 8},
}


def visible_profiles() -> list[Profile]:
    """Profiles shown in the UI (experimental ones are hidden but still supported)."""
    return [p for p, info in PROFILES.items() if not info.experimental]


@dataclass(frozen=True)
class AutoParams:
    batch: int
    lq_size: int
    num_workers: int
    use_compile: bool
    cpu_fallback: bool
    note: str


def _round_to(x: int, step: int) -> int:
    return max(step, (x // step) * step)


# (vram_gb_floor, batch, lq_size) coarse tiers — safe defaults, profile-scaled below.
_VRAM_TIERS = [
    (0.0, 1, 64),
    (4.0, 2, 64),
    (6.0, 4, 96),
    (9.0, 4, 128),
    (12.0, 8, 128),
    (16.0, 8, 128),
    (24.0, 16, 128),
]


def auto_params(
    model: Model,
    scale: int,
    profile: Profile,
    vram_gb: float,
    *,
    want_lq_size: int | None = None,
    want_batch: int | None = None,
) -> AutoParams:
    """Pick safe batch/patch defaults from VRAM (or CPU), respecting the arch pad step."""
    step = int(ARCH[model]["window"])
    cost = PROFILES[profile].cost

    if vram_gb <= 0:  # no GPU → CPU fallback
        lq = _round_to(want_lq_size or 64, step)
        return AutoParams(
            batch=want_batch or 1, lq_size=lq, num_workers=2, use_compile=False,
            cpu_fallback=True,
            note="No GPU detected — CPU training is possible for NERVE but VERY slow.",
        )

    eff = vram_gb / cost
    batch, lq = 1, 64
    for floor, b, s in _VRAM_TIERS:
        if eff >= floor:
            batch, lq = b, s

    if want_lq_size is not None:
        lq = want_lq_size
    if want_batch is not None:
        batch = want_batch
    lq = _round_to(lq, step)

    return AutoParams(
        batch=batch, lq_size=lq, num_workers=min(8, max(2, batch)),
        use_compile=True, cpu_fallback=False,
        note=f"{vram_gb:.0f} GB GPU → batch {batch}, patch {lq}.",
    )
