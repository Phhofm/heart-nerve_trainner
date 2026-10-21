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


@requires_repo
def test_compile_mode_and_amp_are_template_aware():
    cfg = build_config(_req(), REPO)
    assert cfg["compile_mode"] == "default"  # from 2x_NERVE_release.yml
    assert cfg["amp_bf16"] is True

    forced = build_config(_req(compile_mode="max-autotune-no-cudagraphs", amp_bf16=False), REPO)
    assert forced["compile_mode"] == "max-autotune-no-cudagraphs"
    assert forced["amp_bf16"] is False


@requires_repo
def test_loader_pipeline_defaults():
    train = build_config(_req(), REPO)["datasets"]["train"]
    assert train["persistent_workers"] is True
    assert train["num_worker_per_gpu"] >= 2
    assert train["prefetch_factor"] == 2


def test_oom_detection():
    from hnt.train_runner import looks_like_oom

    assert looks_like_oom("RuntimeError: CUDA out of memory. Tried to allocate 2 GiB")
    assert looks_like_oom("torch.OutOfMemoryError: CUDA out of memory")
    assert not looks_like_oom("FileNotFoundError: dataset missing")


@requires_repo
def test_validation_is_tiled_by_default():
    val = build_config(_req(), REPO)["val"]
    assert val["tile_size"] == 256
    assert val["tile_overlap"] == 16
    assert build_config(_req(val_tile_size=0), REPO)["val"]["tile_size"] == 0


def test_choose_tile_auto():
    from hnt.infer import _choose_tile

    assert _choose_tile(512, 512, 0) == 0          # modest image → whole
    assert _choose_tile(3000, 4000, 0) == 512      # huge → tiled
    assert _choose_tile(512, 512, -1) == 0         # explicit whole
    assert _choose_tile(512, 512, 256) == 256      # explicit tile


def test_bundled_val_requires_hr_and_lr(tmp_path):
    from hnt.config_gen import _bundled_val

    assert _bundled_val(tmp_path, 2) is None
    base = tmp_path / "datasets/val/hnt_val"
    (base / "hr").mkdir(parents=True)
    (base / "lr_x2").mkdir()
    (base / "hr" / "a.png").write_bytes(b"x")
    assert _bundled_val(tmp_path, 2) == (base / "hr", base / "lr_x2")
    assert _bundled_val(tmp_path, 4) is None


def test_python_exe_never_returns_frozen_app_binary(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from hnt import bootstrap

    fake = SimpleNamespace(venv=tmp_path / "venv")
    monkeypatch.setattr(bootstrap, "is_frozen", lambda: True)
    # Clean install: no venv yet. Must be "" (never the app binary, which would
    # relaunch the GUI and cause a launch loop).
    assert bootstrap.python_exe(fake) == ""

    py = fake.venv / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    assert bootstrap.python_exe(fake) == str(py)

    monkeypatch.setattr(bootstrap, "is_frozen", lambda: False)
    assert bootstrap.python_exe(SimpleNamespace(venv=tmp_path / "missing")) == bootstrap.sys.executable


def test_create_venv_reports_a_useful_error(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hnt import bootstrap

    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\nexit 2\n")
    fake_uv.chmod(0o755)
    monkeypatch.setattr(bootstrap, "_system_pythons", lambda: [])

    with pytest.raises(RuntimeError) as exc:
        bootstrap.create_venv(str(fake_uv), SimpleNamespace(venv=tmp_path / "venv"))
    assert "Could not create the training environment" in str(exc.value)


@requires_repo
def test_trainer_deps_include_required_packages():
    from hnt.bootstrap import trainer_deps

    deps = trainer_deps(REPO)
    assert "antialiased_cnns" in deps
    assert any(d.startswith("torch>=") for d in deps)


def test_train_error_extracts_the_traceback(tmp_path):
    from hnt.ui.main_window import MainWindow

    log = tmp_path / "run.log"
    log.write_text(
        "noise\nTraceback (most recent call last):\n  File x\n"
        "RuntimeError: antialiased_cnns is not installed\n"
    )
    assert "RuntimeError: antialiased_cnns" in MainWindow._train_error(str(log))


def test_nerve_arch_inferred_from_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    from safetensors.torch import save_file

    from hnt.exporter import _nerve_arch_from_checkpoint

    sd = {
        "blocks.0.conv1.weight": torch.zeros(48, 48, 3, 3),
        "blocks.1.conv1.weight": torch.zeros(48, 48, 3, 3),
    }
    p = tmp_path / "m.safetensors"
    save_file(sd, str(p))
    assert _nerve_arch_from_checkpoint(p) == (48, 2)


def test_ncnn_infer_arch_from_state_dict():
    torch = pytest.importorskip("torch")
    from hnt._ncnn_export import _infer_arch

    sd = {
        "blocks.0.conv1.weight": torch.zeros(64, 64, 3, 3),
        "blocks.5.conv1.weight": torch.zeros(64, 64, 3, 3),
    }
    assert _infer_arch(sd) == (64, 6)


def test_updater_checks_and_downloads_from_manifest(tmp_path):
    import json
    import threading
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    from hnt import updater

    srv = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        (tmp_path / "installer.bin").write_bytes(b"installer-bytes")
        manifest = {
            "version": "99.0.0",
            "notes": "test release",
            "assets": {
                "windows": f"http://127.0.0.1:{port}/installer.bin",
                "linux": f"http://127.0.0.1:{port}/installer.bin",
            },
        }
        (tmp_path / "latest.json").write_text(json.dumps(manifest))
        url = f"http://127.0.0.1:{port}/latest.json"

        info = updater.check_for_update(url)
        assert info is not None and info.version == "99.0.0"
        assert info.notes == "test release"

        got = updater.download_installer(info, tmp_path / "downloads")
        assert got.read_bytes() == b"installer-bytes"

        manifest["version"] = "0.0.1"  # older than the installed build
        (tmp_path / "latest.json").write_text(json.dumps(manifest))
        assert updater.check_for_update(url) is None
    finally:
        srv.shutdown()


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

