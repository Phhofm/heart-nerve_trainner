"""Video inference helpers for BODY Studio.

This module keeps video I/O separate from UI code so it can be tested
headlessly. The initial implementation streams RGB frames through ffmpeg,
runs one model call per frame, and encodes the result back to a container
while preserving audio when present.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .infer import _choose_tile, _run_full, _run_tiled, load_model


@dataclass(frozen=True)
class VideoInfo:
    """Metadata needed to drive a stateless frame pipeline."""

    width: int
    height: int
    frame_rate: Fraction
    has_audio: bool
    rotation: int = 0
    duration: float | None = None
    total_frames: int | None = None


@dataclass(frozen=True)
class VideoResult:
    """Result of a completed video inference job."""

    output: Path
    frames: int
    info: VideoInfo


FrameFn = Callable[[np.ndarray, int], np.ndarray]
ProgressFn = Callable[[int, int | None], None]
ProcessFactory = Callable[[list[str]], object]


def parse_video_info(payload: dict) -> VideoInfo:
    """Parse ``ffprobe -show_streams -show_format -of json`` output."""
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError("no video streams found")

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("no video streams found")

    rate = _parse_rate(video.get("r_frame_rate") or video.get("avg_frame_rate"))
    if rate is None:
        rate = _parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    if rate is None:
        rate = Fraction(25, 1)

    duration = None
    raw_duration = video.get("duration") or payload.get("format", {}).get("duration")
    if raw_duration:
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            duration = None

    total_frames = None
    raw_frames = video.get("nb_frames")
    if raw_frames:
        try:
            total_frames = int(raw_frames)
        except (TypeError, ValueError):
            total_frames = None
    if total_frames is None and duration is not None and rate:
        total_frames = max(0, int(duration * float(rate)))

    rotation = 0
    side_data = video.get("side_data_list") or []
    for item in side_data:
        if "rotation" in item:
            try:
                rotation = int(item["rotation"])
                break
            except (TypeError, ValueError):
                pass
    tags = video.get("tags") or {}
    if rotation == 0 and "rotate" in tags:
        try:
            rotation = int(tags["rotate"])
        except (TypeError, ValueError):
            rotation = 0

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        frame_rate=rate,
        has_audio=has_audio,
        rotation=rotation,
        duration=duration,
        total_frames=total_frames,
    )


def probe_video(path: str | Path, ffprobe: str = "ffprobe") -> VideoInfo:
    """Probe a video file with ffprobe."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffprobe was not found. Please install ffmpeg/ffprobe.") from e
    except subprocess.CalledProcessError as e:
        detail = e.stderr.strip() if e.stderr else "unknown ffprobe failure"
        raise RuntimeError(f"ffprobe failed for {path}: {detail}") from e
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe returned invalid JSON for {path}") from e
    return parse_video_info(payload)


def build_decode_args(
    info: VideoInfo,
    input_path: str | Path,
    *,
    deinterlace: bool = False,
) -> list[str]:
    """Build the ffmpeg decoder command used to read raw RGB frames."""
    args = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
    ]
    if deinterlace:
        args.extend(["-vf", "bwdif=mode=send_frame:parity=auto:deint=all"])
    args.extend(
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    return args


def build_encode_args(
    info: VideoInfo,
    output_path: str | Path,
    source: str | Path,
    *,
    overwrite: bool = False,
    codec: str = "libx264",
    crf: int = 18,
) -> list[str]:
    """Build the ffmpeg encoder command used to write the enhanced video."""
    args = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{info.width}x{info.height}",
        "-r",
        str(info.frame_rate),
        "-i",
        "pipe:0",
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    if info.has_audio:
        args.extend(["-map", "1:a?"])
    args.extend(
        [
            "-map_metadata",
            "1",
            "-c:v",
            codec,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(output_path),
        ]
    )
    return args


def _parse_rate(value) -> Fraction | None:
    if not value:
        return None
    try:
        if "/" in str(value):
            num, den = str(value).split("/", 1)
            return Fraction(int(num), int(den))
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _start_process(cmd: list[str]):
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _frame_fn_from_model(
    model_path: str | Path,
    arch: str,
    scale: int,
    device: str,
    tile: int,
    overlap: int,
) -> FrameFn:
    net = load_model(model_path, arch, scale, device)

    def run(frame: np.ndarray, _index: int) -> np.ndarray:
        tile_px = _choose_tile(frame.shape[0], frame.shape[1], tile)
        if tile_px:
            return _run_tiled(net, frame, scale, device, tile_px, overlap)
        return _run_full(net, frame, scale, device)

    return run


def _finish_process(name: str, proc) -> None:
    try:
        proc.wait()
    except Exception:  # noqa: BLE001
        proc.kill()
        proc.wait()
    if getattr(proc, "returncode", 0) != 0:
        stderr = b""
        try:
            if proc.stderr:
                stderr = proc.stderr.read()
        except Exception:  # noqa: BLE001
            stderr = b""
        detail = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"{name} failed with code {proc.returncode}: {detail}")


def _stop_process(proc) -> None:
    try:
        proc.kill()
        proc.wait()
    except Exception:  # noqa: BLE001
        return


def infer_video(
    model_path: str | Path | None,
    input_path: str | Path,
    output_path: str | Path,
    *,
    arch: str = "NERVE",
    scale: int = 2,
    device: str = "cuda",
    tile: int = 0,
    overlap: int = 16,
    codec: str = "libx264",
    crf: int = 18,
    overwrite: bool = False,
    deinterlace: bool = False,
    frame_fn: FrameFn | None = None,
    info: VideoInfo | None = None,
    decoder_factory: ProcessFactory | None = None,
    encoder_factory: ProcessFactory | None = None,
    on_progress: ProgressFn | None = None,
) -> VideoResult:
    """Enhance a video frame-by-frame using a stateless BODY model.

    ``frame_fn`` is primarily useful for tests and custom runners. When it is
    omitted, ``model_path`` is loaded and used directly.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path == output_path:
        raise ValueError("input and output paths must differ")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not overwrite and output_path.exists():
        raise FileExistsError(output_path)
    if frame_fn is None and model_path is None:
        raise ValueError("either model_path or frame_fn must be provided")

    info = info or probe_video(input_path)
    if frame_fn is None:
        frame_fn = _frame_fn_from_model(model_path, arch, scale, device, tile, overlap)

    decode_cmd = build_decode_args(info, input_path, deinterlace=deinterlace)
    encode_cmd = build_encode_args(
        info,
        output_path,
        input_path,
        overwrite=overwrite,
        codec=codec,
        crf=crf,
    )
    decoder = (decoder_factory or _start_process)(decode_cmd)
    encoder = (encoder_factory or _start_process)(encode_cmd)
    frame_bytes = info.width * info.height * 3
    frames = 0

    try:
        while True:
            raw = _read_exact(decoder.stdout, frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                info.height, info.width, 3
            )
            out = frame_fn(frame.copy(), frames)
            if out.shape != frame.shape:
                raise ValueError(
                    f"model returned shape {out.shape}, expected {frame.shape}"
                )
            encoder.stdin.write(np.ascontiguousarray(out, dtype=np.uint8).tobytes())
            frames += 1
            if on_progress:
                on_progress(frames, info.total_frames)
        encoder.stdin.close()
        _finish_process("encoder", encoder)
        decoder.stdout.close()
        _finish_process("decoder", decoder)
    except Exception:
        _stop_process(decoder)
        _stop_process(encoder)
        raise

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return VideoResult(output=output_path, frames=frames, info=info)
