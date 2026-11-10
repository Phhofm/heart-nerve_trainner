"""PULSE — fast, deployable single-image super-resolution.

A ~162K parameter SISR network (354K at dim=48). The body is a plain conv
residual stack; the base image is a fixed Magic Kernel Sharp 2013 resampler
implemented as a phase-kernel grouped convolution plus PixelShuffle. No
normalization, no attention, no custom ops: the exported graph is
Pad / Conv / Add / Relu / DepthToSpace, which is friendly to ONNX Runtime,
TensorRT, and NCNN.
"""

from __future__ import annotations

import torch
from spandrel.util import store_hyperparameters
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.icnr import icnr_init
from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY


def _mks_1d(x: Tensor) -> Tensor:
    """Magic Kernel Sharp 2013, one dimension, support 2.5."""
    x = x.abs()
    return torch.where(
        x <= 0.5,
        17 / 16 - (7 / 4) * x**2,
        torch.where(
            x <= 1.5,
            0.25 * (4 * x**2 - 11 * x + 7),
            torch.where(x <= 2.5, -0.125 * (x - 2.5) ** 2, torch.zeros_like(x)),
        ),
    )


class MKS2013Base(nn.Module):
    """Fixed Magic Kernel Sharp 2013 integer upsampler.

    For an integer scale, each output phase has a fixed 5x5 separable kernel.
    Storing the kernels as a grouped convolution and using PixelShuffle gives
    the exact integer-scale kernel with standard Conv / DepthToSpace ops and no
    Resize. Borders use replicate padding rather than Costella's weight
    renormalization, and the kernel is applied in the input color space (sRGB
    for images), not the linear-light space used by the reference.
    """

    def __init__(self, channels: int, upscale: int) -> None:
        super().__init__()
        self.channels = channels
        self.upscale = upscale
        if upscale == 1:
            return

        # One-time constant kernels; no gradients and no state-dict payload.
        offsets = torch.arange(-2, 3, dtype=torch.float32)
        weight = torch.empty(channels * upscale**2, 1, 5, 5)
        for c in range(channels):
            for py in range(upscale):
                for px in range(upscale):
                    fy = (py + 0.5) / upscale - 0.5
                    fx = (px + 0.5) / upscale - 0.5
                    wy = _mks_1d(fy - offsets)
                    wx = _mks_1d(fx - offsets)
                    weight[c * upscale**2 + py * upscale + px, 0] = torch.outer(wy, wx)
        self.register_buffer("weight", weight, persistent=False)
        self.shuffle = nn.PixelShuffle(upscale)

    def forward(self, x: Tensor) -> Tensor:
        if self.upscale == 1:
            return x
        x = F.pad(x, (2, 2, 2, 2), mode="replicate")
        y = F.conv2d(x, self.get_buffer("weight"), groups=self.channels)
        return self.shuffle(y)


class PulseBlock(nn.Module):
    """Conv-ReLU-Conv residual block. No normalization, no gating."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


@store_hyperparameters()
class PULSE(nn.Module):
    """MKS2013 base + residual convs + PixelShuffle correction."""

    hyperparameters = {}  # noqa: RUF012

    def __init__(
        self,
        *,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 48,
        n_blocks: int = 8,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        # The MKS base is added directly to the output, so channels must match.
        if in_ch != out_ch:
            raise ValueError("PULSE requires in_ch == out_ch for its base residual.")

        self.upscale = upscale
        self.base = MKS2013Base(in_ch, upscale)
        self.stem = nn.Conv2d(in_ch, dim, 3, padding=1, bias=False)
        self.body = nn.Sequential(*[PulseBlock(dim) for _ in range(n_blocks)])
        # Head bias is intentional: body convs are bias-free and the head
        # learns any residual color shift.
        self.head = nn.Sequential(
            nn.Conv2d(dim, out_ch * upscale**2, 3, padding=1, bias=True),
            nn.PixelShuffle(upscale),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # ICNR removes phase bias from the PixelShuffle head; the tiny scale
        # keeps the initial output close to the MKS base.
        head = self.head[0]
        assert isinstance(head, nn.Conv2d)
        icnr_init(head, self.upscale)
        with torch.no_grad():
            head.weight.mul_(1e-3)
            if head.bias is not None:
                head.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        base = self.base(x)
        feat = self.stem(x)
        # Direct stem-to-head path; the blocks are already residual.
        feat = feat + self.body(feat)
        return base + self.head(feat)


@SPANDREL_REGISTRY.register()
@ARCH_REGISTRY.register()
def pulse(
    scale: int = 4,
    *,
    in_ch: int = 3,
    out_ch: int = 3,
) -> PULSE:
    return PULSE(in_ch=in_ch, out_ch=out_ch, dim=48, n_blocks=8, upscale=scale)


@SPANDREL_REGISTRY.register()
@ARCH_REGISTRY.register()
def pulse_fast(
    scale: int = 4,
    *,
    in_ch: int = 3,
    out_ch: int = 3,
) -> PULSE:
    return PULSE(in_ch=in_ch, out_ch=out_ch, dim=32, n_blocks=8, upscale=scale)
