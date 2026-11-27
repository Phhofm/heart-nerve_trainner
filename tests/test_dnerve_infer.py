"""D-NERVE Studio integration: vendored arch parity, runner plumbing, routing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hnt.dnerve_arch import DnervDecoder, load_dnerve_decoder  # noqa: E402
from hnt.dnerve_infer import ensure_substrate, run_dnerve  # noqa: E402


class StubDino(torch.nn.Module):
    """Stands in for DINOv2-L: only the feature contract run_dnerve needs."""

    def forward_features(self, x: torch.Tensor) -> dict:
        b, _, h, w = x.shape
        gh, gw = h // 14, w // 14
        return {"x_norm_patchtokens": torch.rand(b, gh * gw, 1024)}


def _image(h: int = 112, w: int = 112) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_vendored_decoder_matches_research_arch():
    from traiNNer.archs.dnerve_arch import DnervDecoder as ResearchDecoder

    ours, theirs = DnervDecoder(), ResearchDecoder()
    assert ours.load_state_dict(theirs.state_dict(), strict=True) is not None
    assert theirs.load_state_dict(ours.state_dict(), strict=True) is not None
    ours_params = sum(p.numel() for p in ours.parameters())
    theirs_params = sum(p.numel() for p in theirs.parameters())
    assert ours_params == theirs_params


def test_load_dnerve_decoder_roundtrip(tmp_path: Path):
    from safetensors.torch import save_file

    net = DnervDecoder()
    p = tmp_path / "dec.safetensors"
    save_file(net.state_dict(), str(p))
    loaded = load_dnerve_decoder(str(p))
    assert not loaded.training
    for k in net.state_dict():
        assert torch.equal(net.state_dict()[k], loaded.state_dict()[k])


def test_run_dnerve_restores_input_resolution_with_stub_dino():
    decoder = DnervDecoder().eval()
    out = run_dnerve(decoder, StubDino(), _image(), device="cpu")
    assert out.shape == (112, 112, 3)
    assert out.dtype == np.uint8

    # non-/14 sizes get reflect-padded and cropped back
    out = run_dnerve(decoder, StubDino(), _image(100, 130), device="cpu")
    assert out.shape == (100, 130, 3)


def test_ensure_substrate_prefers_override_then_cache(tmp_path: Path, monkeypatch):
    paths = SimpleNamespace(data=tmp_path)

    existing = tmp_path / "local.pth"
    existing.write_bytes(b"x")
    monkeypatch.setenv("HNT_DINOV2", str(existing))
    assert ensure_substrate(paths) == str(existing)

    monkeypatch.setenv("HNT_DINOV2", str(tmp_path / "missing.pth"))
    with pytest.raises(FileNotFoundError):
        ensure_substrate(paths)

    monkeypatch.delenv("HNT_DINOV2")
    downloaded = tmp_path / "substrates" / "dinov2_vitl14_pretrain.pth"

    def fake_download(repo_id, filename, local_dir):
        assert repo_id == "Phips/D-NERVE"
        assert filename == "substrate/dinov2_vitl14_pretrain.pth"
        downloaded.write_bytes(b"weights")
        return str(downloaded)

    monkeypatch.setattr("hnt.dnerve_infer.hf_hub_download", fake_download)
    assert ensure_substrate(paths) == str(downloaded)
    # second call: cached, no download
    monkeypatch.setattr("hnt.dnerve_infer.hf_hub_download",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no re-download")))
    assert ensure_substrate(paths) == str(downloaded)


def test_infer_paths_dnerve_route(tmp_path: Path, monkeypatch):
    from hnt import dnerve_infer
    from hnt.infer import infer_paths

    src = tmp_path / "in"
    src.mkdir()
    arr = _image(112, 110)  # NOT /14-divisible: the padding path must engage
    from PIL import Image

    Image.fromarray(arr).save(src / "photo.png")

    monkeypatch.setattr(
        dnerve_infer, "load_dnerve",
        lambda decoder_path, paths, device: (DnervDecoder().eval(), StubDino()))

    out = tmp_path / "out"
    res = infer_paths(tmp_path / "fake.safetensors", [src], out,
                      arch="dnerve", scale=2, device="cpu")
    assert res.errors == []
    assert len(res.outputs) == 1
    saved = np.asarray(Image.open(res.outputs[0]))
    assert saved.shape == (112, 110, 3)  # scale forced to 1; size preserved
    assert res.outputs[0].name == "photo_fake.png"
