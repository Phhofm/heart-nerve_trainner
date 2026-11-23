# D-NERVE decoder — generative restoration architecture (the beautifier tier).
#
# One-pass restoration on a frozen foundation-model substrate: DINOv2 patch
# features are bilinearly upsampled to full resolution, concatenated with the
# degraded image, and processed by a small residual decoder (~1.9M params).
# Trained severity-mixed so ONE model modulates: gentle on mild damage, bold
# on destroyed input. This vendored copy is registry-neutral and training-free
# (decoder only — the frozen DINOv2-L substrate loads in hnt.dnerve_infer).

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class DnervDecoder(nn.Module):
    """One-pass decoder: frozen-substrate features + degraded image -> RGB.

    Features are bilinearly upsampled from the patch grid to full resolution,
    concatenated with the degraded image, and processed by residual blocks.
    """

    def __init__(self, dim: int = 1024, width: int = 128, n_blocks: int = 6,
                 patch: int = 14) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(dim + 3, width, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1),
            ) for _ in range(n_blocks)
        ])
        self.out = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, feats: Tensor, degraded: Tensor) -> Tensor:
        _b, _, _gh, _gw = feats.shape
        h, w = degraded.shape[-2:]
        up = F.interpolate(feats, size=(h, w), mode="bilinear",
                           align_corners=False)
        x = self.proj(torch.cat([up, degraded], dim=1))
        for blk in self.blocks:
            x = x + blk(x)
        return self.out(x)


def load_dnerve_decoder(path: str) -> DnervDecoder:
    """Load a D-NERVE decoder checkpoint (strict)."""
    p = str(path)
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(p)
    else:
        sd = torch.load(p, map_location="cpu", weights_only=True)
    net = DnervDecoder()
    net.load_state_dict(sd, strict=True)
    return net.eval()
