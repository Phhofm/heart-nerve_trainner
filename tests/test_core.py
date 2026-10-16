"""Core tests — run from the ``heart_nerve_trainer/`` folder:

    ../venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import pytest

from hnt.config_gen import ConfigError, TrainRequest, build_config
from hnt.degradation import PRESETS, get as get_deg
from hnt.manifest import DEFAULT, pretrain_for
from hnt.presets import ARCH, PROFILES, SCALES, auto_params
from hnt.paths import find_repo

REPO = None
try:
    REPO = find_repo()
except FileNotFoundError:
    pass  # standalone app repo without the trainer checkout → repo tests skip

requires_repo = pytest.mark.skipif(REPO is None, reason="traiNNer-redux checkout not found")


def _req(**kw):
    base = dict(model="NERVE", scale=2, profile="FIDELITY", data_mode="PAIRED",
                hr_dir="/d/hr", lr_dir="/d/lr", val_hr_dir="/d/vhr", val_lr_dir="/d/vlr")
    base.update(kw)
    return TrainRequest(**base)


@requires_repo
@pytest.mark.parametrize("model", ["NERVE", "HEART"])
@pytest.mark.parametrize("scale", [2, 4])
@pytest.mark.parametrize("profile", ["FIDELITY", "PERCEPTUAL", "REALWORLD_FIDELITY"])
def test_build_config_all_combos(model, scale, profile):
    mode = "PAIRED" if profile == "FIDELITY" else "OTF"
    lr = "/d/lr" if mode == "PAIRED" else ""
    cfg = build_config(_req(model=model, scale=scale, profile=profile, data_mode=mode, lr_dir=lr), REPO)
    assert cfg["scale"] == scale
    assert cfg["network_g"]["type"] == ARCH[model]["type"]
    assert cfg["datasets"]["train"]["type"] == ("pairedimagedataset" if mode == "PAIRED" else "realesrgandataset")
    assert cfg["datasets"]["train"]["dataroot_gt"] == ["/d/hr"]
    assert cfg["path"]["pretrain_network_g"] is None  # none set in the request


@requires_repo
def test_perceptual_metrics_are_opt_in():
    off = build_config(_req(), REPO)
    assert "dists" not in off["val"].get("metrics", {})
    assert "lpips" not in off["val"].get("metrics", {})
    on = build_config(_req(val_perceptual_metrics=True), REPO)
    assert on["val"]["metrics"]["dists"]["type"] == "calculate_dists"
    assert on["val"]["metrics"]["lpips"]["type"] == "calculate_lpips"


def test_lq_size_is_multiple_of_window():
    for model in ("HEART", "NERVE"):
        step = ARCH[model]["window"]
        ap = auto_params(model, 4, "FIDELITY", 12.0)
        assert ap.lq_size % step == 0


@requires_repo
def test_otf_requires_template():
    # NERVE/HEART all have OTF templates, so this should succeed, not raise.
    cfg = build_config(_req(profile="REALWORLD_FIDELITY", data_mode="OTF", lr_dir=""), REPO)
    assert cfg["datasets"]["train"]["type"] == "realesrgandataset"


def test_auto_params_cpu_fallback():
    ap = auto_params("NERVE", 2, "FIDELITY", 0.0)
    assert ap.cpu_fallback is True
    assert ap.batch >= 1


def test_auto_params_vram_scaling():
    small = auto_params("NERVE", 2, "FIDELITY", 4.0)
    big = auto_params("NERVE", 2, "FIDELITY", 16.0)
    assert big.batch >= small.batch


def test_perceptual_costs_more_vram():
    # GAN profile should not give a *larger* batch than fidelity at same VRAM.
    f = auto_params("HEART", 4, "FIDELITY", 12.0)
    g = auto_params("HEART", 4, "PERCEPTUAL", 12.0)
    assert g.batch <= f.batch


def test_degradation_presets_exist():
    for name in ("bicubic", "multi_algo", "down_up_down", "jpeg", "double_jpeg",
                 "real_esrgan", "random_mix"):
        assert name in PRESETS
    assert get_deg("bicubic").params["blur_prob"] == 0.0


def test_manifest_lookup():
    assert pretrain_for("HEART", 2, DEFAULT) is not None
    assert pretrain_for("HEART", 1, DEFAULT) is None  # 1x not shipped yet
    assert pretrain_for("NERVE", 4, DEFAULT) is not None


def test_profiles_and_scales():
    assert set(PROFILES) == {"FIDELITY", "PERCEPTUAL", "MODERN_PERCEPTUAL", "REALWORLD_FIDELITY"}
    assert SCALES == (1, 2, 4)


def test_short_helpers():
    # diagnostics builds without a GPU
    from hnt.diagnostics import help_prompt

    txt = help_prompt("train", "boom", log_tail="tail", hw=None)
    assert "step-by-step" in txt and "boom" in txt


def test_parse_metrics_reads_validation_line():
    from hnt.train_runner import parse_metrics

    line = "\t # psnr : 30.1234\tBest: 30.2000 @    25,000 iter\n\t # ssim :  0.8901"
    m = parse_metrics(line)
    assert m["psnr"] == pytest.approx(30.1234)
    assert m["ssim"] == pytest.approx(0.8901)


def test_parse_metrics_reads_perceptual_metrics():
    from hnt.train_runner import parse_metrics

    m = parse_metrics("\t # dists:  0.1234\n\t # lpips:  0.2000\n\t # topiq:  0.5000")
    assert m == {"dists": 0.1234, "lpips": 0.2, "topiq": 0.5}


def test_parse_metrics_ignores_unknown_names():
    from hnt.train_runner import parse_metrics

    assert parse_metrics("# note : 5") == {}
    assert parse_metrics("Validation val") == {}


def test_lucid_hr_dir_prefers_hr_subfolder(tmp_path):
    from hnt.downloader import lucid_hr_dir

    assert lucid_hr_dir(tmp_path) == tmp_path
    (tmp_path / "hr").mkdir()
    assert lucid_hr_dir(tmp_path) == tmp_path / "hr"

