"""Config generator: turn GUI choices into a valid traiNNer-redux training config.

Strategy: **load a proven release config as a template and patch it** — keeps us
correct even as the networks evolve. OTF degradation presets are applied at the
config ROOT (see ``degradation.py``). A synthesized fallback covers PAIRED
fidelity when no template is found.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .degradation import DEFAULT_FIDELITY_PRESET, DEFAULT_OTF_PRESET, get as get_deg
from .paths import find_repo
from .presets import ARCH, DataMode, Model, Profile

R = "options/train/RELEASE"
# templates selected by (model, profile, data_mode)
_TEMPLATES: dict[tuple[str, str, str], list[str]] = {
    ("HEART", "FIDELITY", "PAIRED"): [f"{R}/4x_HEART_release.yml"],
    ("HEART", "FIDELITY", "OTF"): [f"{R}/4x_HEART_OTF_v2.yml", f"{R}/4x_HEART_OTF.yml"],
    ("HEART", "PERCEPTUAL", "OTF"): [f"{R}/4x_HEART_OTF_gan.yml"],
    ("HEART", "REALWORLD_FIDELITY", "OTF"): [f"{R}/4x_HEART_OTF_v2.yml", f"{R}/4x_HEART_OTF.yml"],
    ("NERVE", "FIDELITY", "PAIRED"): [f"{R}/2x_NERVE_release.yml"],
    ("NERVE", "FIDELITY", "OTF"): [f"{R}/2x_NERVE_OTF_fidelity.yml"],
    ("NERVE", "PERCEPTUAL", "OTF"): [f"{R}/2x_NERVE_OTF_gan.yml", f"{R}/2x_NERVE_OTF_gan_ext.yml"],
    ("NERVE", "REALWORLD_FIDELITY", "OTF"): [f"{R}/2x_NERVE_OTF_fidelity.yml"],
    # Modern-perceptual reuses the GAN template, then swaps D/losses (see _apply_modern).
    ("HEART", "MODERN_PERCEPTUAL", "OTF"): [f"{R}/4x_HEART_OTF_gan.yml"],
    ("NERVE", "MODERN_PERCEPTUAL", "OTF"): [f"{R}/2x_NERVE_OTF_gan.yml",
                                            f"{R}/2x_NERVE_OTF_gan_ext.yml"],
}


# Modern-perceptual loss stack: pixel/structure + ConvNeXt perceptual + frequency + hinge GAN
# with an R3GAN-style spectral-norm discriminator. Weights are STARTING POINTS (tune).
def _apply_modern(cfg: dict[str, Any], req: "TrainRequest") -> None:
    cfg["network_d"] = {"type": "r3gandisc", "num_feat": 48}
    cfg["train"]["losses"] = [
        {"type": "mssimloss", "loss_weight": 0.5},
        {"type": "convnextperceptualloss", "criterion": "charbonnier", "loss_weight": 0.05},
        {"type": "fftloss", "loss_weight": 0.5},
        {"type": "ganloss", "gan_type": "hinge", "loss_weight": 0.1},
    ]
    cfg["train"].setdefault("optim_d", {"type": "AdamW", "lr": 1e-4, "weight_decay": 0,
                                        "betas": [0.9, 0.99]})


class ConfigError(RuntimeError):
    pass


def _num_workers() -> int:
    """Enough loader processes to keep the GPU fed without starving the desktop."""
    return min(8, max(2, (os.cpu_count() or 4) // 2))


@dataclass
class TrainRequest:
    model: Model
    scale: int
    profile: Profile
    data_mode: DataMode
    hr_dir: str
    val_hr_dir: str = ""
    val_lr_dir: str = ""
    lr_dir: str = ""
    degradation: str = ""
    batch: int = 4
    lq_size: int = 96
    iters: int = 100_000
    lr: float | None = None
    pretrain: str | None = None
    strict_load: bool = True
    use_compile: bool = True
    use_checkpoint: bool = False
    # None = keep the template's proven value. bf16 is right for Ampere+; older
    # cards should pass False to use fp16.
    amp_bf16: bool | None = None
    compile_mode: str | None = None
    val_perceptual_metrics: bool = False  # DISTS/LPIPS — accurate but much slower
    val_tile_size: int = 256  # 0 = whole image (can OOM on huge val images)
    name: str = "hnt_run"
    extra: dict[str, Any] = field(default_factory=dict)


def _find_template(repo: Path, model: Model, profile: Profile, data_mode: DataMode) -> Path | None:
    for rel in _TEMPLATES.get((model, profile, data_mode), []):
        p = repo / rel
        if p.is_file():
            return p
    return None


def _synth_paired(req: TrainRequest) -> dict[str, Any]:
    arch = ARCH[req.model]
    return {
        "name": req.name, "scale": req.scale,
        "use_amp": True,
        "amp_bf16": True if req.amp_bf16 is None else req.amp_bf16,
        "use_channels_last": True,
        "fast_matmul": False, "use_compile": req.use_compile,
        "compile_mode": req.compile_mode or "max-autotune-no-cudagraphs",
        "num_gpu": "auto", "manual_seed": 1024,
        "datasets": {
            "train": {"name": "Train", "type": "pairedimagedataset",
                      "dataroot_gt": [req.hr_dir], "dataroot_lq": [req.lr_dir],
                      "lq_size": req.lq_size, "use_hflip": True, "use_rot": True,
                      "num_worker_per_gpu": _num_workers(), "persistent_workers": True,
                      "prefetch_factor": 2,
                      "batch_size_per_gpu": req.batch, "accum_iter": 1},
            "val": {"name": "Val", "type": "pairedimagedataset",
                    "dataroot_gt": [req.val_hr_dir], "dataroot_lq": [req.val_lr_dir],
                    "filename_tmpl": f"{{}}x{req.scale}"},
        },
        "network_g": {"type": arch["type"], "use_checkpoint": req.use_checkpoint}
        if req.model == "HEART"
        else {"type": arch["type"]},
        "path": {"pretrain_network_g": req.pretrain, "param_key_g": None,
                 "strict_load_g": req.strict_load, "resume_state": None},
        "train": {
            "ema_decay": 0.999, "ema_power": 0.75, "grad_clip": False, "grad_clip_max_norm": 100,
            "optim_g": {"type": "AdamW", "lr": req.lr or 2e-4, "weight_decay": 0,
                        "betas": [0.9, 0.99]},
            "scheduler": {"type": "MultiStepLR",
                          "milestones": [int(req.iters * 0.5), int(req.iters * 0.75)], "gamma": 0.5},
            "total_iter": req.iters, "warmup_iter": -1,
            "losses": [{"type": "charbonnierloss", "loss_weight": 1.0}],
        },
        "val": {"val_enabled": True, "val_freq": max(2500, req.iters // 40), "save_img": False,
                "tile_size": req.val_tile_size, "tile_overlap": 16, "metrics_enabled": True,
                "metrics": {"psnr": {"type": "calculate_psnr", "crop_border": req.scale * 2,
                                     "test_y_channel": True},
                            "ssim": {"type": "calculate_ssim", "crop_border": req.scale * 2,
                                     "test_y_channel": True}}},
        "logger": {"print_freq": 200, "save_checkpoint_freq": max(5000, req.iters // 20),
                   "save_checkpoint_format": "safetensors", "use_tb_logger": True},
    }


def _patch(cfg: dict[str, Any], req: TrainRequest) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg["name"] = req.name
    cfg["scale"] = req.scale
    cfg.setdefault("use_amp", True)
    cfg["amp_bf16"] = True if req.amp_bf16 is None else req.amp_bf16
    cfg["use_compile"] = req.use_compile
    if req.compile_mode:  # else keep the template's proven mode (reduce-overhead is not universal)
        cfg["compile_mode"] = req.compile_mode
    cfg["use_channels_last"] = True

    ds = cfg.setdefault("datasets", {})
    train = ds.setdefault("train", {})
    train["dataroot_gt"] = [req.hr_dir]
    if req.data_mode == "PAIRED":
        train["type"] = "pairedimagedataset"
        train["dataroot_lq"] = [req.lr_dir]
    else:
        train["type"] = "realesrgandataset"
        train.pop("dataroot_lq", None)
        # OTF degradation knobs are top-level fields. All preset keys are valid schema
        # fields (blur/resize/noise/JPEG/sinc + our extra WebP/AVIF/banding/chroma).
        for k, v in get_deg(req.degradation or DEFAULT_OTF_PRESET).params.items():
            cfg[k] = v
    train["lq_size"] = req.lq_size
    train["batch_size_per_gpu"] = req.batch
    train["num_worker_per_gpu"] = _num_workers()
    train["persistent_workers"] = True
    train["prefetch_factor"] = 2
    train["use_hflip"] = True
    train["use_rot"] = True

    val = ds.setdefault("val", {})
    if req.val_hr_dir:
        val["dataroot_gt"] = [req.val_hr_dir]
    if req.val_lr_dir:
        val["dataroot_lq"] = [req.val_lr_dir]

    ng = cfg.setdefault("network_g", {})
    if "use_checkpoint" in ng:  # HEART supports it; NERVE's arch does not
        ng["use_checkpoint"] = req.use_checkpoint

    path = cfg.setdefault("path", {})
    path["pretrain_network_g"] = req.pretrain
    path["param_key_g"] = None
    path["strict_load_g"] = req.strict_load
    path["resume_state"] = None

    tr = cfg.setdefault("train", {})
    tr["total_iter"] = req.iters
    if req.lr is not None:
        tr.setdefault("optim_g", {})["lr"] = req.lr

    cfg.setdefault("val", {})["val_freq"] = max(2500, req.iters // 40)
    # Tile validation by default: full-resolution val images are a classic OOM.
    val = cfg["val"]
    val["tile_size"] = req.val_tile_size
    val["tile_overlap"] = 16
    # PSNR/SSIM are always on. DISTS/LPIPS are opt-in: the trainer rebuilds the
    # perceptual net for every validation image, so they are much slower.
    vmetrics = cfg["val"].setdefault("metrics", {})
    if req.val_perceptual_metrics:
        vmetrics.setdefault("dists", {"type": "calculate_dists"})
        vmetrics.setdefault("lpips", {"type": "calculate_lpips"})
    cfg.setdefault("logger", {})["save_checkpoint_format"] = "safetensors"
    return cfg


def _bundled_val(repo: Path, scale: int) -> tuple[Path, Path] | None:
    """CC0 Lucid validation tiles shipped with the app (used when the user gives none)."""
    base = repo / "datasets/val/hnt_val"
    hr, lr = base / "hr", base / f"lr_x{scale}"
    if hr.is_dir() and lr.is_dir() and any(hr.iterdir()):
        return hr, lr
    return None


def build_config(req: TrainRequest, repo: Path | None = None) -> dict[str, Any]:
    repo = repo or find_repo()
    tmpl = _find_template(repo, req.model, req.profile, req.data_mode)
    if tmpl is not None:
        cfg = _patch(yaml.safe_load(tmpl.read_text()), req)
        if req.profile == "MODERN_PERCEPTUAL":
            _apply_modern(cfg, req)
    elif req.data_mode == "PAIRED":
        cfg = _synth_paired(req)
        cfg.update(req.extra)
    else:
        raise ConfigError(
            f"No OTF template for {req.model}/{req.profile}; cannot generate an on-the-fly "
            "degradation config (use paired LR instead)."
        )

    # No user validation folder → validate on the bundled CC0 tiles (never a full-res
    # external benchmark set, and tiled by default).
    if not req.val_hr_dir:
        bundled = _bundled_val(repo, req.scale)
        if bundled is not None:
            hr, lr = bundled
            v = cfg.setdefault("datasets", {}).setdefault("val", {})
            v["dataroot_gt"] = [str(hr)]
            v["dataroot_lq"] = [str(lr)]
            v["filename_tmpl"] = "{}"
    return cfg


def write_config(cfg: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path
