"""R3GAN-style discriminator: residual conv blocks + spectral norm, no batch-norm.

A modern, artifact-resistant discriminator (in the spirit of R3GAN / projected
GANs) as an alternative to the classic UNet/PatchGAN discriminators. Returns a
logit map (B, 1, H, W), matching the ``DUnet`` interface used by SRModel.

Train it with a stable objective, e.g. hinge GAN + R1 (see configs).
"""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm

from traiNNer.utils.registry import ARCH_REGISTRY


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = spectral_norm(nn.Conv2d(ch, ch, 3, 1, 1))
        self.c2 = spectral_norm(nn.Conv2d(ch, ch, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        h = F.leaky_relu(self.c1(F.leaky_relu(x, 0.2)), 0.2)
        return x + self.c2(h)


@ARCH_REGISTRY.register()
class R3GANDisc(nn.Module):
    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, num_blocks: int = 6) -> None:
        super().__init__()
        self.inp = spectral_norm(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.down = spectral_norm(nn.Conv2d(num_feat, num_feat, 4, 2, 1))
        self.body = nn.Sequential(*[_ResBlock(num_feat) for _ in range(num_blocks)])
        self.out = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, 1, 3, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.inp(x), 0.2)
        x = F.leaky_relu(self.down(x), 0.2)
        x = self.body(x)
        return self.out(x)
