"""T-NERVE runner tests: load, temporal pipeline integration, blending.

Uses the smallest viable TNerve (dim 8, 2 blocks, 3-frame windows) with
seeded random weights — the contract under test is plumbing (shapes,
frame counts, blend semantics), not restoration quality.
"""

from __future__ import annotations

import io
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")

from hnt.tnerve_runner import load_tnerve, make_fns  # noqa: E402
from hnt.video import infer_video_temporal  # noqa: E402
from hnt.tnerve_arch import TNerve  # noqa: E402


def _flat(value: int) -> np.ndarray:
    return np.full((16, 16, 3), value, dtype=np.uint8)


def _seeded_net() -> TNerve:
    torch.manual_seed(7)
    return TNerve(dim=8, n_blocks=2, upscale=2, clip_size=3)


@pytest.fixture(scope="module")
def ckpt_path(tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("ckpt") / "tiny_tnerve.safetensors"
    safetensors.torch.save_file(_seeded_net().state_dict(), str(p))
    return str(p)


class _NonClosingBytesIO(io.BytesIO):
    def close(self) -> None:
        return  # keep the buffer readable after the pipeline closes it


class _FakeProc:
    def __init__(self, data: bytes = b""):
        self.stdout = _NonClosingBytesIO(data)
        self.stdin = _NonClosingBytesIO()
        self.waited = False

    def wait(self):
        self.waited = True
        return 0

    def kill(self):
        self.waited = True


def _factory(*datas: bytes):
    fakes = [_FakeProc(d) for d in datas]
    calls = iter(fakes)
    return lambda cmd: next(calls), fakes


def _info(total: int):
    from hnt.video import VideoInfo

    return VideoInfo(width=16, height=16, frame_rate=Fraction(2, 1),
                     has_audio=False, total_frames=total)


def test_load_tnerve_roundtrip(ckpt_path) -> None:
    net = load_tnerve(ckpt_path, device="cpu", clip_size=3, dim=8,
                      n_blocks=2)
    ref = _seeded_net()
    for k in net.state_dict():
        assert torch.equal(net.state_dict()[k], ref.state_dict()[k])


def test_temporal_pipeline_with_blend(tmp_path: Path, ckpt_path) -> None:
    net = load_tnerve(ckpt_path, device="cpu", clip_size=3, dim=8,
                      n_blocks=2)
    window_fn, static_fn = make_fns(net, device="cpu", clip_size=3)

    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"stub")

    frames = [_flat(10), _flat(10), _flat(10), _flat(200), _flat(200),
              _flat(200)]
    raw = b"".join(f.tobytes() for f in frames)
    factory, fakes = _factory(raw, raw)  # scan pass + process pass
    encoder = _FakeProc()

    result = infer_video_temporal(
        input_path,
        output_path,
        window_fn=window_fn,
        frame_fn=static_fn,  # boundary frames: no temporal context
        static_fn=static_fn,
        blend=(0.0, 1.0),  # w == luma diff directly
        clip_size=3,
        out_scale=2,
        info=_info(6),
        decoder_factory=factory,
        encoder_factory=lambda cmd: encoder,
    )

    assert result.frames == 6
    written = encoder.stdin.getvalue()
    assert len(written) == 6 * 32 * 32 * 3  # 2x upscale, 16 -> 32


def test_static_fn_deterministic_per_frame(ckpt_path) -> None:
    net = load_tnerve(ckpt_path, device="cpu", clip_size=3, dim=8,
                      n_blocks=2)
    window_fn, static_fn = make_fns(net, device="cpu", clip_size=3)
    frame = _flat(123)
    assert np.array_equal(static_fn(frame, 0), static_fn(frame, 1))
    assert static_fn(frame, 0).shape == (32, 32, 3)
