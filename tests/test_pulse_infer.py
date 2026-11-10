"""PULSE / PULSE_fast inference in the app's isolated runtime."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from hnt.infer import infer_paths, load_model  # noqa: E402

REPO = None
try:
    from hnt.paths import find_repo
    REPO = find_repo()
except FileNotFoundError:
    pass

requires_repo = pytest.mark.skipif(REPO is None, reason="traiNNer-redux checkout not found")

PULSE_FAST_4X = Path(
    "/home/phips/Documents/trainner-redux-playground/release/pulse/models/pulse_fast/4x_pretrain.safetensors"
)


@requires_repo
@pytest.mark.skipif(not PULSE_FAST_4X.is_file(), reason="PULSE_fast 4x checkpoint missing")
def test_pulse_fast_model_loads():
    net = load_model(PULSE_FAST_4X, arch="pulse_fast", scale=4, device="cpu")
    assert net.upscale == 4


@requires_repo
@pytest.mark.skipif(not PULSE_FAST_4X.is_file(), reason="PULSE_fast 4x checkpoint missing")
def test_pulse_fast_infer_paths(tmp_path: Path):
    from PIL import Image

    inp = tmp_path / "in.png"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(inp)
    out = tmp_path / "out"
    res = infer_paths(PULSE_FAST_4X, [inp], out, arch="pulse_fast", scale=4,
                      device="cpu", tile=-1)
    assert not res.errors
    assert len(res.outputs) == 1
    assert res.outputs[0].is_file()


def test_cli_accepts_pulse_arches():
    from hnt.cli import build_parser

    p = build_parser()
    args = p.parse_args(["infer", "--model", "m.safetensors", "--arch", "pulse_fast",
                         "--input", "x.png", "--out", "o"])
    assert args.arch == "pulse_fast"
