# T-NERVE — temporal NERVE, stateless bidirectional stacked window.
#
# A temporal video super-resolution architecture for the BODY suite: the
# NERVE backbone consumes a stacked temporal window instead of a single
# frame. Input is (B, T, C, H, W) with an odd T (13 recommended); the output
# is the HR center frame. Only standard conv/elementwise ops — the checkpoint
# exports to dynamic-shape ONNX exactly like NERVE.
#
# Center-frame initialization: all NERVE parameters carry over unchanged
# except the stem, whose center-frame slice receives the NERVE kernel and
# whose neighbor slices start at zero. Step 0 therefore behaves exactly like
# single-image NERVE, and the network learns to use its neighbors from a
# stable starting point. See load_nerve_into_tnerve().
#
# Design was ablated on real degraded video (Apache-2.0 sources): channel-
# stacked windows beat flow alignment (learned tiny flow and frozen NeuFlow
# v2 both tied or trailed), beat split-fuse fusion, and beat window-local
# recurrence — while the window size curve rose monotonically through T13.
# Windows are sampled as sliding windows inside a scene (filename prefix =
# scene), so they never cross scene cuts. Like TPULSE, T-NERVE is registered
# only in ARCH_REGISTRY (rank-5 input is not spandrel-friendly).

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from traiNNer.archs.nerve_arch import NerveBlock
from traiNNer.utils.icnr import icnr_reinit


class TNerve(nn.Module):
    """5x5 stem over T stacked frames -> NERVE blocks -> PixelShuffle head
    -> + bicubic(center frame). Input (B, T, C, H, W), output HR center."""

    hyperparameters = {}  # noqa: RUF012

    def __init__(
        self,
        *,
        out_ch: int = 3,
        dim: int = 64,
        n_blocks: int = 24,
        upscale: int = 4,
        clip_size: int = 13,
    ) -> None:
        super().__init__()
        if clip_size < 1 or clip_size % 2 == 0:
            raise ValueError("clip_size must be an odd integer >= 1.")

        self.upscale = upscale
        self.clip_size = clip_size
        self.center_idx = clip_size // 2

        self.stem = nn.Conv2d(3 * clip_size, dim, 5, padding=2, bias=False)
        self.blocks = nn.Sequential(*[NerveBlock(dim) for _ in range(n_blocks)])
        self.upsampler = nn.Sequential(
            nn.Conv2d(dim, out_ch * upscale**2, 3, padding=1, bias=False),
            nn.PixelShuffle(upscale),
        )
        icnr_reinit(self.upsampler)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c, h, w = x.shape
        if t != self.clip_size:
            raise ValueError(f"Expected clip_size={self.clip_size}, got {t} frames.")
        if c != 3:
            raise ValueError(f"Expected RGB frames (C=3), got {c} channels.")

        feat = self.stem(x.reshape(b, t * c, h, w))
        out = feat + self.blocks(feat)
        out = self.upsampler(out)

        base = F.interpolate(
            x[:, self.center_idx], scale_factor=self.upscale, mode="bicubic",
            align_corners=False,
        )
        return out + base


def load_nerve_into_tnerve(
    t_nerve: TNerve, nerve_state: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Adapt a NERVE state_dict for T-NERVE (center-frame initialization).

    Every NERVE parameter is copied unchanged except the stem: its
    center-frame slice receives the NERVE kernel and all neighbor slices
    start at zero, so a freshly converted T-NERVE behaves exactly like
    single-image NERVE. The returned dict loads with ``strict=True``.
    """
    center = t_nerve.center_idx
    src = dict(nerve_state)
    stem = t_nerve.stem.weight.new_zeros(t_nerve.stem.weight.shape)
    w = src.get("stem.weight")
    if w is None or tuple(w.shape) != (stem.shape[0], 3, *stem.shape[2:]):
        raise ValueError(
            f"NERVE stem weight {tuple(w.shape) if w is not None else ()} "
            f"does not match T-NERVE stem {tuple(stem.shape)} "
            f"(dim={stem.shape[0]}, kernel={tuple(stem.shape[2:])})."
        )
    stem[:, center * 3 : (center + 1) * 3] = w
    src["stem.weight"] = stem
    # modules that NERVE does not have keep their own initialized weights
    src.update({
        k: v.detach().clone()
        for k, v in t_nerve.state_dict().items() if k not in src
    })
    return src



def tnerve(
    scale: int = 4,
    *,
    out_ch: int = 3,
    dim: int = 64,
    n_blocks: int = 24,
    clip_size: int = 13,
) -> TNerve:
    return TNerve(
        out_ch=out_ch, dim=dim, n_blocks=n_blocks, upscale=scale,
        clip_size=clip_size,
    )
