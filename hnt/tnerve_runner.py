"""T-NERVE temporal video runner for BODY Studio.

Loads a T-NERVE checkpoint (2x, odd-size frame windows) and adapts it
to :func:`hnt.video.infer_video_temporal`:

- ``window_fn`` runs the full temporal window (the model's native mode),
- ``static_fn`` runs the same checkpoint on a static window (all slots
  = the center frame), i.e. the single-frame-equivalent output used by
  the motion-adaptive blend as the anti-ghosting fallback.

The vendored ``tnerve_arch.py`` is registry-neutral (it does not
register into traiNNer's ARCH_REGISTRY to avoid clashing with a
research install that already has it).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .tnerve_arch import TNerve


def load_tnerve(path: str | Path, device: str = "cuda",
                clip_size: int = 13, dim: int = 64,
                n_blocks: int = 24) -> torch.nn.Module:
    """Build TNerve and load a safetensors checkpoint (strict)."""
    net = TNerve(dim=dim, n_blocks=n_blocks, upscale=2, clip_size=clip_size)
    from safetensors.torch import load_file

    sd = load_file(str(path))
    for key in ("params_ema", "params"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def _run_window(net, window: list[np.ndarray], device: str) -> np.ndarray:
    t = torch.stack([
        torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
        for f in window
    ]).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(t)[0].float().clamp(0, 1)
    return (out.permute(1, 2, 0).cpu().numpy() * 255.0).round() \
        .astype(np.uint8)


def make_fns(net, device: str = "cuda", clip_size: int = 13) -> tuple[
        Callable[[list[np.ndarray], int], np.ndarray],
        Callable[[np.ndarray, int], np.ndarray]]:
    """(window_fn, static_fn) pair for infer_video_temporal.

    ``static_fn`` feeds ``clip_size`` copies of the single frame —
    no temporal context — so the blend can fall back to the
    single-frame-equivalent rendering of the same checkpoint.
    """

    def window_fn(window: list[np.ndarray], center: int) -> np.ndarray:
        return _run_window(net, window, device)

    def static_fn(frame: np.ndarray, index: int) -> np.ndarray:
        return _run_window(net, [frame] * clip_size, device)

    return window_fn, static_fn
