"""ConvNeXt-feature perceptual loss — a modern alternative to VGG perceptual.

Registered as ``convnextperceptualloss`` for traiNNer's LOSS_REGISTRY. Uses
timm's ConvNeXt (`features_only`) at a few stages and matches features (L1 or
Charbonnier). Discarded at inference (only the generator ships).

Falls back to a clear error if timm / pretrained weights are unavailable, so the
app can gracefully use the existing VGG ``perceptualloss`` instead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@LOSS_REGISTRY.register()
class ConvNeXtPerceptualLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
        criterion: str = "charbonnier",
        model_name: str = "convnext_nano",
        layer_weights: list[float] | None = None,
        pretrained: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = criterion
        self.eps = eps
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ConvNeXtPerceptualLoss needs `timm`. Install it or use the VGG `perceptualloss`."
            ) from e

        self.net = timm.create_model(model_name, features_only=True, pretrained=pretrained)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

        n = len(self.net.feature_info)
        self.layer_weights = layer_weights or [1.0 / n] * n
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _crit(self, a: Tensor, b: Tensor) -> Tensor:
        if self.criterion == "l1":
            return F.l1_loss(a, b)
        return torch.sqrt((a - b) ** 2 + self.eps**2).mean()

    def _features(self, x: Tensor) -> list[Tensor]:
        x = (x.clamp(0.0, 1.0) - self.mean) / self.std
        return self.net(x)

    def forward(self, x: Tensor, gt: Tensor) -> Tensor:
        # Detach the teacher side; only the generator gets gradients.
        with torch.no_grad():
            fg = self._features(gt)
        fx = self._features(x)
        loss = x.new_zeros(())
        for w, a, b in zip(self.layer_weights, fx, fg):
            loss = loss + w * self._crit(a, b)
        return loss
