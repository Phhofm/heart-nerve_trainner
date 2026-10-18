# NERVE — lightweight super-resolution for mobile/edge (single config).
#
# Conv-ReLU-Conv residual blocks, no normalization, no attention, no gating.
# A 5x5 stem opens the receptive field for real-world degradations; a global
# residual and a bicubic input residual let the net learn only what bicubic
# upscaling misses. dim=64, n_blocks=24, ~1.8M params at 4x.
#
# Norm-free is decided by ablation (NERVE_ABLATION_LOG.md §4): removing
# LayerNorm won +1.15 dB at ~2x the training speed. dim=64/n_blocks=24 is the
# capacity-sweep sweet spot (§15): 0.69M -> 1.80M bought a large GAN quality
# gain (lpips -0.015), while 1.80M -> 2.39M bought ~4x less gain per parameter.
# Every op in the graph is standard (Conv/Add/ReLU/PixelShuffle/bicubic
# Resize), so the checkpoint exports to dynamic-shape ONNX and converts to
# NCNN/CoreML without fused/unfused pairs.

from spandrel.util import store_hyperparameters
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.icnr import icnr_reinit
from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY


class NerveBlock(nn.Module):
    """Conv-ReLU-Conv residual block. No normalization, no shortcuts."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(F.relu(self.conv1(x)))
        return x + out


@store_hyperparameters()
class NERVE(nn.Module):
    """5x5 stem -> n_blocks residual blocks -> PixelShuffle head -> + bicubic(x)."""

    hyperparameters = {}  # noqa: RUF012

    def __init__(
        self,
        *,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        n_blocks: int = 24,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        self.upscale = upscale

        self.stem = nn.Conv2d(in_ch, dim, 5, padding=2, bias=False)
        self.blocks = nn.Sequential(*[NerveBlock(dim) for _ in range(n_blocks)])
        self.upsampler = nn.Sequential(
            nn.Conv2d(dim, out_ch * upscale**2, 3, padding=1, bias=False),
            nn.PixelShuffle(upscale),
        )
        # ICNR: every sub-pixel phase starts from the same base kernel, so the
        # head cannot train up a checkerboard. Init-only, zero graph/runtime cost.
        icnr_reinit(self.upsampler)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.stem(x)
        out = feat + self.blocks(feat)
        out = self.upsampler(out)

        base = F.interpolate(
            x, scale_factor=self.upscale, mode="bicubic", align_corners=False
        )
        return out + base


@SPANDREL_REGISTRY.register()
@ARCH_REGISTRY.register()
def nerve(
    scale: int = 4,
    *,
    in_ch: int = 3,
    out_ch: int = 3,
    dim: int = 64,
    n_blocks: int = 24,
) -> NERVE:
    return NERVE(in_ch=in_ch, out_ch=out_ch, dim=dim, n_blocks=n_blocks, upscale=scale)
