"""Pretrain manifest: (model, scale) → weights to fine-tune from.

The app ships a default manifest; users/releases can override it via a JSON file
in the app data dir (``pretrains.json``). Missing entries are fine — the config
generator then trains from scratch (scale 1 stays hidden until a 1x pretrain ships).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .presets import Model


@dataclass(frozen=True)
class Pretrain:
    path_or_url: str
    strict_load: bool = True
    note: str = ""


# Keys are "MODEL:scale". `path_or_url` may be a repo-relative path (dev) or an
# https URL (release). 1x entries are intentionally absent until pretrains ship.
DEFAULT: dict[str, Pretrain] = {
    "HEART:2": Pretrain("huggingface://Phips/HEART/2x_HEART.safetensors", False,
                        "finetuned from 4x (upsampler re-init)"),
    "HEART:4": Pretrain("huggingface://Phips/HEART/4x_HEART.safetensors", True),
    "NERVE:2": Pretrain("huggingface://Phips/HEART/2x_NERVE.safetensors", True),
    "NERVE:4": Pretrain("huggingface://Phips/HEART/4x_NERVE.safetensors", True),
}


def load_manifest(data_dir: Path) -> dict[str, Pretrain]:
    """Default manifest, overlaid by a user/CI ``pretrains.json`` if present."""
    merged = dict(DEFAULT)
    f = Path(data_dir) / "pretrains.json"
    if f.is_file():
        try:
            for k, v in json.loads(f.read_text()).items():
                merged[k] = Pretrain(v["path_or_url"], v.get("strict_load", True), v.get("note", ""))
        except (ValueError, KeyError):
            pass
    return merged


def pretrain_for(model: Model, scale: int, manifest: dict[str, Pretrain]) -> Pretrain | None:
    return manifest.get(f"{model}:{scale}")


def is_1x_available(manifest: dict[str, Pretrain]) -> dict[str, bool]:
    return {m: f"{m}:1" in manifest for m in ("HEART", "NERVE")}
