from typing import Literal

import numpy as np
import torch
from torch import Tensor

from traiNNer.archs.lpips_arch import LPIPS
from traiNNer.utils.img_util import img2batchedtensor
from traiNNer.utils.registry import METRIC_REGISTRY



# Built once per (backbone, device). Validation calls this per image; rebuilding the
# network (and reloading its weights) every time made perceptual validation crawl.
_CACHE: dict[tuple[str, str], LPIPS] = {}


def _get_lpips(net: str, device: torch.device) -> LPIPS:
    key = (net, str(device))
    model = _CACHE.get(key)
    if model is None:
        model = LPIPS(net=net).to(device)
        model.eval()
        _CACHE[key] = model
    return model


@METRIC_REGISTRY.register()
def calculate_lpips(
    img: np.ndarray,
    img2: np.ndarray,
    device: torch.device,
    net: Literal["alex", "vgg", "squeeze"] = "alex",
    **kwargs,
) -> Tensor:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )

    loss = _get_lpips(net, device)

    with torch.inference_mode():
        return loss(
            img2batchedtensor(img, device, from_bgr=False),
            img2batchedtensor(img2, device, from_bgr=False),
            normalize=True,
        ).view(())
