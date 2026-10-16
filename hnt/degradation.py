"""LR-generation presets.

Note: in traiNNer-redux the OTF *degradation knobs* (blur/resize/noise/JPEG/sinc,
``high_order_degradation``) are **top-level config fields**, while the dataset
section holds only the blur-kernel details + paths + crop. So these presets are
overrides applied to the config ROOT (on top of an OTF template), not to the
dataset node.

PAIRED → no degradation (user supplies an LR folder).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DegradationPreset:
    label: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)  # applied at config root


_CLEAN = {
    "high_order_degradation": False,
    "lq_usm": False,
    "blur_prob": 0.0,
    "resize_prob": [0.0, 1.0, 0.0],
    "resize_mode_list": ["bicubic"],
    "resize_mode_prob": [1.0],
    "resize_range": [1.0, 1.0],
    "gaussian_noise_prob": 0.0,
    "jpeg_prob": 0.0,
}

_REALESRGAN = {
    "high_order_degradation": True,
    "blur_prob": 0.6,
    "resize_prob": [0.2, 0.7, 0.1],
    "resize_mode_list": ["bilinear", "bicubic", "nearest-exact", "lanczos"],
    "resize_mode_prob": [0.25, 0.25, 0.25, 0.25],
    "resize_range": [0.4, 1.5],
    "gaussian_noise_prob": 0.4,
    "noise_range": [1, 20],
    "poisson_scale_range": [0.05, 2],
    "gray_noise_prob": 0.1,
    "jpeg_prob": 0.4,
    "jpeg_range": [40, 95],
    "blur_prob2": 0.3,
    "resize_range2": [0.6, 1.2],
    "gaussian_noise_prob2": 0.3,
    "noise_range2": [1, 15],
    "jpeg_prob2": 0.3,
    "jpeg_range2": [40, 95],
}

PRESETS: dict[str, DegradationPreset] = {
    "bicubic": DegradationPreset(
        "Bicubic (clean)", "Pure bicubic downscale — best for max fidelity/PSNR.", dict(_CLEAN)),
    "multi_algo": DegradationPreset(
        "Multi-algorithm", "Random resize kernel per sample.",
        {**_CLEAN, "resize_mode_list": ["bilinear", "bicubic", "nearest-exact", "lanczos"],
         "resize_mode_prob": [0.25, 0.25, 0.25, 0.25]}),
    "down_up_down": DegradationPreset(
        "Down → up → down", "Random resample down/up/down with random kernels.",
        {**_CLEAN, "resize_range": [0.5, 1.0],
         "resize_mode_list": ["bilinear", "bicubic", "lanczos"],
         "resize_mode_prob": [0.34, 0.33, 0.33]}),
    "jpeg": DegradationPreset(
        "JPEG", "Downscale + random-quality JPEG.",
        {**_CLEAN, "jpeg_prob": 1.0, "jpeg_range": [30, 95]}),
    "double_jpeg": DegradationPreset(
        "Double JPEG", "Two JPEG passes at random quality.",
        {**_CLEAN, "jpeg_prob": 1.0, "jpeg_range": [30, 95],
         "jpeg_prob2": 1.0, "jpeg_range2": [30, 95]}),
    "real_esrgan": DegradationPreset(
        "Real-ESRGAN-style", "Blur + noise + JPEG + sinc resize (random).", dict(_REALESRGAN)),
    "random_mix": DegradationPreset(
        "Random mix", "Random combination of blur/noise/JPEG/resize.", dict(_REALESRGAN)),
    # --- extra OTF ops (WebP/AVIF/banding/chroma) ---
    "webp": DegradationPreset(
        "WebP", "Downscale + WebP compression (web images).", {**_CLEAN, "webp_prob": 1.0, "webp_range": [50, 90]}),
    "avif": DegradationPreset(
        "AVIF", "Downscale + AVIF compression (needs Pillow AVIF).", {**_CLEAN, "avif_prob": 1.0, "avif_range": [30, 55]}),
    "banding": DegradationPreset(
        "Banding", "Gradient banding (low-bit quantization).", {**_CLEAN, "banding_prob": 1.0, "banding_bits": 6, "banding_dither_prob": 0.3}),
    "chroma": DegradationPreset(
        "Chroma subsampling", "4:2:0 color subsampling (color bleeding).", {**_CLEAN, "chroma_prob": 1.0, "chroma_jitter": 0.3}),
}

DEFAULT_OTF_PRESET = "real_esrgan"
DEFAULT_FIDELITY_PRESET = "bicubic"


def get(name: str) -> DegradationPreset:
    return PRESETS.get(name, PRESETS[DEFAULT_OTF_PRESET])
