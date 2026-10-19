import numpy as np
import torch
from torch import Tensor

from traiNNer.losses.dists_loss import DISTSLoss
from traiNNer.utils.img_util import img2batchedtensor
from traiNNer.utils.registry import METRIC_REGISTRY

# Built once per device: validation calls this per image, and rebuilding the net
# (plus reloading weights) every call was the main cost of perceptual validation.
_CACHE: dict[str, DISTSLoss] = {}


def _get_dists(device: torch.device) -> DISTSLoss:
    key = str(device)
    model = _CACHE.get(key)
    if model is None:
        model = DISTSLoss(loss_weight=1.0, as_loss=False).to(device)
        model.eval()
        _CACHE[key] = model
    return model


@METRIC_REGISTRY.register()
def calculate_dists(
    img: np.ndarray, img2: np.ndarray, device: torch.device, **kwargs
) -> Tensor:
    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    loss = _get_dists(device)
    with torch.inference_mode():
        return loss(
            img2batchedtensor(img, device, from_bgr=False),
            img2batchedtensor(img2, device, from_bgr=False),
        )
