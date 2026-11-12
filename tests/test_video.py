import io
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from hnt.video import (
    VideoInfo,
    VideoResult,
    build_decode_args,
    build_encode_args,
    infer_video,
    infer_video_temporal,
    parse_video_info,
)


def _info(
    width: int = 2,
    height: int = 2,
    frame_rate: Fraction = Fraction(2, 1),
    has_audio: bool = True,
    total_frames: int | None = 2,
) -> VideoInfo:
    return VideoInfo(
        width=width,
        height=height,
        frame_rate=frame_rate,
        has_audio=has_audio,
        total_frames=total_frames,
    )


def _sample_probe_payload() -> dict:
    return {
        "streams": [
            {
                "codec_type": "video",
                "width": 320,
                "height": 240,
                "r_frame_rate": "30000/1001",
                "avg_frame_rate": "30000/1001",
                "nb_frames": "120",
                "duration": "4.004000",
                "side_data_list": [{"rotation": 90}],
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ]
    }


def test_parse_video_info_reads_fps_dimensions_audio_and_rotation():
    info = parse_video_info(_sample_probe_payload())

    assert info.width == 320
    assert info.height == 240
    assert info.frame_rate == Fraction(30000, 1001)
    assert info.has_audio is True
    assert info.rotation == 90
    assert info.total_frames == 120


def test_build_decode_args_uses_raw_rgb_pipe():
    args = build_decode_args(_info(), Path("in.mp4"))

    assert args[0] == "ffmpeg"
    assert "-i" in args
    assert "in.mp4" in args
    assert "-f" in args
    assert "rawvideo" in args
    assert "-pix_fmt" in args
    assert "rgb24" in args
    assert "pipe:1" in args


def test_build_decode_args_can_request_deinterlacing():
    args = build_decode_args(_info(), Path("in.mp4"), deinterlace=True)
    assert "-vf" in args
    assert "bwdif" in args[args.index("-vf") + 1]


def test_build_encode_args_maps_audio_when_present():
    args = build_encode_args(
        _info(has_audio=True),
        Path("out.mp4"),
        source=Path("in.mp4"),
        overwrite=True,
    )

    assert args[0] == "ffmpeg"
    assert "-y" in args
    assert "-f" in args
    assert "rawvideo" in args
    assert "-s" in args
    assert "2x2" in args
    assert "-r" in args
    assert "2" in args
    assert "-map" in args
    assert "1:a?" in args
    assert "-c:v" in args
    assert "libx264" in args
    assert "-crf" in args
    assert "out.mp4" in args


def test_build_encode_args_omits_audio_mapping_without_audio():
    args = build_encode_args(
        _info(has_audio=False),
        Path("out.mp4"),
        source=Path("in.mp4"),
    )

    assert "1:a?" not in args


def test_infer_video_processes_each_frame_and_writes_output(tmp_path: Path):
    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"stub")

    frames = [
        np.full((2, 2, 3), 10, dtype=np.uint8),
        np.full((2, 2, 3), 20, dtype=np.uint8),
    ]
    decoder = FakeProcess(b"".join(frame.tobytes() for frame in frames))
    encoder = FakeProcess()
    progress: list[tuple[int, int | None]] = []

    def frame_fn(frame: np.ndarray, index: int) -> np.ndarray:
        out = frame.copy()
        out += index
        return out

    result = infer_video(
        None,
        input_path,
        output_path,
        info=_info(),
        frame_fn=frame_fn,
        decoder_factory=lambda cmd: decoder,
        encoder_factory=lambda cmd: encoder,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert result.output == output_path
    assert result.frames == 2
    assert result.info.frame_rate == Fraction(2, 1)
    assert progress == [(1, 2), (2, 2)]
    expected = b"".join((frame + i).tobytes() for i, frame in enumerate(frames))
    assert encoder.stdin.getvalue() == expected
    assert decoder.waited is True
    assert encoder.waited is True


def test_infer_video_refuses_to_overwrite_by_default(tmp_path: Path):
    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"stub")
    output_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        infer_video(None, input_path, output_path, info=_info())


def test_infer_video_rejects_same_input_and_output(tmp_path: Path):
    same = tmp_path / "same.mp4"

    with pytest.raises(ValueError):
        infer_video(None, same, same, info=_info())


def test_infer_video_requires_model_or_frame_fn(tmp_path: Path):
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"stub")
    with pytest.raises(ValueError):
        infer_video(None, input_path, tmp_path / "out.mp4", info=_info())


class _NonClosingBytesIO(io.BytesIO):
    def close(self) -> None:
        return

class FakeProcess:
    """Minimal process-like object for pipeline tests."""

    def __init__(self, stdout: bytes = b"", returncode: int = 0):
        self.stdin = _NonClosingBytesIO()
        self.stdout = _NonClosingBytesIO(stdout)
        self.stderr = _NonClosingBytesIO()
        self.returncode = returncode
        self.waited = False

    def wait(self) -> int:
        self.waited = True
        return self.returncode


def test_infer_video_streams_with_real_ffmpeg(tmp_path: Path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed")

    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=black:size=2x2:rate=2",
            "-frames:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(input_path),
        ],
        check=True,
    )

    result = infer_video(
        None,
        input_path,
        output_path,
        frame_fn=lambda frame, _index: frame,
    )

    assert result.frames == 2
    assert output_path.exists()


def test_cli_video_command_passes_args(monkeypatch, tmp_path, capsys):
    from hnt.cli import main

    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"stub")
    captured: dict = {}

    def fake_infer_video(model_path, input, out, **kwargs):
        captured["model"] = model_path
        captured["input"] = input
        captured["out"] = out
        captured.update(kwargs)
        return VideoResult(output=out, frames=2, info=_info())

    monkeypatch.setattr("hnt.cli.infer_video", fake_infer_video)
    rc = main(
        [
            "video",
            "--model",
            "model.safetensors",
            "--input",
            str(input_path),
            "--out",
            str(output_path),
            "--arch",
            "nerve",
            "--scale",
            "2",
            "--device",
            "cpu",
            "--tile",
            "256",
            "--codec",
            "libx265",
            "--crf",
            "20",
            "--deinterlace",
            "--overwrite",
        ]
    )

    assert rc == 0
    assert captured["model"] == "model.safetensors"
    assert captured["input"] == str(input_path)
    assert captured["out"] == str(output_path)
    assert captured["arch"] == "nerve"
    assert captured["scale"] == 2
    assert captured["device"] == "cpu"
    assert captured["tile"] == 256
    assert captured["codec"] == "libx265"
    assert captured["crf"] == 20
    assert captured["deinterlace"] is True
    assert captured["overwrite"] is True
    assert "output" in capsys.readouterr().out


def _flat(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def _fake_decoders(*streams: bytes):
    """decoder_factory yielding one FakeProcess per decode pass."""
    fakes = [FakeProcess(data) for data in streams]
    calls = iter(fakes)
    return lambda cmd: next(calls), fakes


def test_infer_video_temporal_windows_never_cross_cuts(tmp_path: Path):
    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"stub")

    # hard cut between frames 2 and 3 (flat colors, diff = 1.0)
    frames = [_flat(10), _flat(10), _flat(10), _flat(200), _flat(200), _flat(200)]
    raw = b"".join(f.tobytes() for f in frames)
    factory, fakes = _fake_decoders(raw, raw)  # two passes: scan + process
    encoder = FakeProcess()

    windows_seen: list[tuple[int, int]] = []
    fallbacks: list[int] = []

    def window_fn(window: list[np.ndarray], center: int) -> np.ndarray:
        windows_seen.append((len(window), center))
        return window[1] + 1  # center frame, marked

    def frame_fn(frame: np.ndarray, index: int) -> np.ndarray:
        fallbacks.append(index)
        return frame + 2  # fallback, marked differently

    result = infer_video_temporal(
        input_path,
        output_path,
        window_fn=window_fn,
        frame_fn=frame_fn,
        clip_size=3,
        info=_info(total_frames=6),
        decoder_factory=factory,
        encoder_factory=lambda cmd: encoder,
    )

    assert result.frames == 6
    # full windows only for centers fully inside a scene (half=1): 1 and 4
    assert windows_seen == [(3, 1), (3, 4)]
    # scene-boundary frames fall back to single-frame processing
    assert fallbacks == [0, 2, 3, 5]
    # output order is frame order: fallback(0), win(1), fallback(2),
    # fallback(3), win(4), fallback(5)
    written = encoder.stdin.getvalue()
    expected = b"".join(
        [
            (_flat(10) + 2).tobytes(),
            (_flat(10) + 1).tobytes(),
            (_flat(10) + 2).tobytes(),
            (_flat(200) + 2).tobytes(),
            (_flat(200) + 1).tobytes(),
            (_flat(200) + 2).tobytes(),
        ]
    )
    assert written == expected
    assert fakes[0].waited is True
    assert fakes[1].waited is True
    assert encoder.waited is True


def test_infer_video_temporal_requires_window_fn(tmp_path: Path):
    input_path = tmp_path / "in.mp4"
    input_path.write_bytes(b"stub")
    with pytest.raises(ValueError):
        infer_video_temporal(input_path, tmp_path / "out.mp4", info=_info())


def test_infer_video_temporal_rejects_bad_output_shape(tmp_path: Path):
    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"stub")
    frames = [_flat(10)] * 4
    raw = b"".join(f.tobytes() for f in frames)
    factory, _ = _fake_decoders(raw, raw)

    def window_fn(window: list[np.ndarray], center: int) -> np.ndarray:
        return np.zeros((3, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        infer_video_temporal(
            input_path,
            output_path,
            window_fn=window_fn,
            frame_fn=lambda frame, _i: frame,
            clip_size=3,
            info=_info(total_frames=4),
            decoder_factory=factory,
            encoder_factory=lambda cmd: FakeProcess(),
        )


def test_infer_video_temporal_out_scale_encodes_scaled(tmp_path: Path):
    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"stub")

    # 1x1 frames -> 2x2 output at out_scale=2; single scene
    frames = [np.full((1, 1, 3), 10, dtype=np.uint8)] * 4
    raw = b"".join(f.tobytes() for f in frames)
    factory, _ = _fake_decoders(raw, raw)
    encoder = FakeProcess()

    def window_fn(window: list[np.ndarray], center: int) -> np.ndarray:
        return np.full((2, 2, 3), 7, dtype=np.uint8)

    result = infer_video_temporal(
        input_path,
        output_path,
        window_fn=window_fn,
        frame_fn=lambda frame, _i: np.full((2, 2, 3), 9, dtype=np.uint8),
        clip_size=3,
        out_scale=2,
        info=_info(width=1, height=1, has_audio=False, total_frames=4),
        decoder_factory=factory,
        encoder_factory=lambda cmd: encoder,
    )
    assert result.frames == 4
    assert result.info.width == 2 and result.info.height == 2
    # decode stays 1x1; the encode args carry the scaled size
    # order: fallback(0), win(1), win(2), fallback(3)
    assert encoder.stdin.getvalue() == b"".join(
        [np.full((2, 2, 3), 9, np.uint8).tobytes(),
         np.full((2, 2, 3), 7, np.uint8).tobytes(),
         np.full((2, 2, 3), 7, np.uint8).tobytes(),
         np.full((2, 2, 3), 9, np.uint8).tobytes()]
    )


def test_infer_video_temporal_out_scale_requires_fallback_fn(tmp_path: Path):
    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"stub")
    # frames with a cut: fallback frames exist but frame_fn is None
    frames = [_flat(10), _flat(10), _flat(10), _flat(200), _flat(200), _flat(200)]
    raw = b"".join(f.tobytes() for f in frames)
    factory, _ = _fake_decoders(raw, raw)
    with pytest.raises(ValueError):
        infer_video_temporal(
            input_path,
            output_path,
            window_fn=lambda w, c: np.zeros((2, 2, 3), dtype=np.uint8),
            clip_size=3,
            out_scale=2,
            info=_info(total_frames=6),
            decoder_factory=factory,
            encoder_factory=lambda cmd: FakeProcess(),
        )
