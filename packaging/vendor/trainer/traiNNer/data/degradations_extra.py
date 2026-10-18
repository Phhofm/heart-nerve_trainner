"""Extra OTF degradations beyond Real-ESRGAN's blur/resize/noise/JPEG.

Adds WebP + AVIF compression, gradient banding, and chroma subsampling as
GPU/CPU tensor ops. All are **opt-in** (probability 0 by default) so existing
pipelines are unaffected.

Tensors are (B, 3, H, W) in [0, 1]. WebP/AVIF run on CPU (PIL); banding/chroma
are pure torch (GPU-friendly).
"""

from __future__ import annotations

import io

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def _to_pil(t: Tensor):  # t: (3,H,W) in [0,1]
    from PIL import Image

    arr = (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def _from_pil(im) -> Tensor:
    arr = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _pil_compress(img: Tensor, quality: Tensor | float, fmt: str) -> Tensor:
    out = []
    for i in range(img.size(0)):
        q = int(quality[i].item() if torch.is_tensor(quality) else quality)
        buf = io.BytesIO()
        _to_pil(img[i]).save(buf, format=fmt, quality=q)
        buf.seek(0)
        from PIL import Image

        out.append(_from_pil(Image.open(buf)))
    return torch.stack(out).to(img.device)


def add_webp_compression(img: Tensor, quality: Tensor | float = 90) -> Tensor:
    """WebP lossy compression (blocky/soft artifacts distinct from JPEG)."""
    try:
        return _pil_compress(img, quality, "WEBP")
    except Exception:  # pragma: no cover - WebP is always in Pillow; safety net
        return img


def add_avif_compression(img: Tensor, quality: Tensor | float = 40) -> Tensor:
    """AVIF compression (needs Pillow AVIF support; otherwise a no-op)."""
    try:
        return _pil_compress(img, quality, "AVIF")
    except Exception:
        return img


def add_banding(img: Tensor, bits: int = 6, dither_prob: float = 0.0) -> Tensor:
    """Quantize to `bits` levels (gradient banding), optional ordered dither."""
    levels = float(2**bits - 1)
    q = torch.round(img * levels) / levels
    if dither_prob > 0:
        noise = (torch.rand_like(q) - 0.5) / levels
        q = q + noise * (torch.rand_like(q) < dither_prob).float()
    return q.clamp(0, 1)


def chroma_subsample(img: Tensor, factor: int = 2, jitter: float = 0.0) -> Tensor:
    """4:2:0-style chroma subsampling (color bleeding). factor 2 → half-res chroma."""
    # RGB -> YCbCr (BT.601)
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    chroma = torch.cat([cb, cr], dim=1)
    h, w = chroma.shape[-2:]
    if jitter > 0 and torch.rand(()) < jitter:
        factor = int(torch.randint(2, 5, ()).item())
    ds = F.interpolate(chroma, size=(max(1, h // factor), max(1, w // factor)), mode="bilinear",
                       align_corners=False)
    chroma = F.interpolate(ds, size=(h, w), mode="bilinear", align_corners=False)
    cb, cr = chroma[:, 0:1], chroma[:, 1:2]
    r2 = y + 1.402 * (cr - 0.5)
    g2 = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b2 = y + 1.772 * (cb - 0.5)
    return torch.cat([r2, g2, b2], dim=1).clamp(0, 1)
