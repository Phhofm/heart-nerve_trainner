"""Inference for trained / downloaded HEART & NERVE models.

PyTorch path (loads ``.safetensors`` / ``.pth`` into the arch, runs image or whole
folders, writes outputs suffixed with the model name). Tiled for big images.
ONNX / TensorRT / ncnn runners are a later addition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import find_repo

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class InferResult:
    outputs: list[Path]
    errors: list[str]


def load_model(path: str | Path, arch: str = "NERVE", scale: int = 2, device: str = "cuda"):
    """Build the arch and load weights (handles params_ema/params wrappers)."""
    import torch

    from traiNNer.archs import build_network

    arch = arch.lower()
    net = build_network({"type": arch, "scale": int(scale)}).eval().to(device)
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(p))
    else:
        sd = torch.load(str(p), map_location="cpu", weights_only=False)
    for key in ("params_ema", "params"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    net.load_state_dict(sd, strict=True)
    return net


def _to_tensor(img: np.ndarray, device: str):
    import torch

    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def _to_uint8(t) -> np.ndarray:
    a = t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (a * 255.0).round().astype(np.uint8)


def _run_full(net, img: np.ndarray, scale: int, device: str) -> np.ndarray:
    import torch

    t = _to_tensor(img, device)
    with torch.no_grad():
        out = net(t)
    return _to_uint8(out)


def _run_tiled(net, img: np.ndarray, scale: int, device: str, tile: int, overlap: int) -> np.ndarray:
    """Simple tiled inference with linear overlap blending (seam-reduced)."""
    import torch

    h, w, c = img.shape
    out = np.zeros((h * scale, w * scale, c), dtype=np.float32)
    weight = np.zeros((h * scale, w * scale, 1), dtype=np.float32)
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, h - overlap), step))
    xs = list(range(0, max(1, w - overlap), step))
    for y in ys:
        for x in xs:
            y2, x2 = min(h, y + tile), min(w, x + tile)
            patch = img[y:y2, x:x2]
            with torch.no_grad():
                o = _to_uint8(net(_to_tensor(patch, device)))
            ry, rx = y * scale, x * scale
            hh, ww = o.shape[0], o.shape[1]
            out[ry : ry + hh, rx : rx + ww] += o.astype(np.float32)
            weight[ry : ry + hh, rx : rx + ww] += 1.0
    return (out / np.clip(weight, 1, None)).round().astype(np.uint8)


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _save_image(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def infer_paths(
    model_path: str | Path,
    inputs: list[str | Path],
    out_dir: str | Path,
    arch: str = "NERVE",
    scale: int = 2,
    device: str = "cuda",
    tile: int = 0,
    overlap: int = 16,
    suffix: str | None = None,
) -> InferResult:
    """Run inference on files and/or folders; outputs go to ``out_dir``."""
    net = load_model(model_path, arch, scale, device)
    model_stem = Path(model_path).stem
    suffix = suffix or model_stem
    out_dir = Path(out_dir)
    outputs: list[Path] = []
    errors: list[str] = []

    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in IMG_EXTS]
        elif p.suffix.lower() in IMG_EXTS:
            files.append(p)

    for f in files:
        try:
            img = _load_image(f)
            res = _run_tiled(net, img, scale, device, tile, overlap) if tile > 0 else _run_full(net, img, scale, device)
            dst = out_dir / f"{f.stem}_{suffix}.png"
            _save_image(res, dst)
            outputs.append(dst)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{f.name}: {e}")
    return InferResult(outputs, errors)
