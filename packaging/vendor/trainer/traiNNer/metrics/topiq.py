import numpy as np
import torch
from torch import Tensor

from traiNNer.archs.topiq_arch import CFANet
from traiNNer.utils.img_util import img2batchedtensor
from traiNNer.utils.registry import METRIC_REGISTRY



# Cache per (variant, device): validation calls these per image.
_CACHE: dict[tuple[str, str], CFANet] = {}


def _get_topiq(name: str, use_ref: bool, device: torch.device) -> CFANet:
    key = (f"{name}:{int(use_ref)}", str(device))
    model = _CACHE.get(key)
    if model is None:
        model = CFANet(model_name=name, use_ref=use_ref).to(device)
        model.eval()
        _CACHE[key] = model
    return model


@METRIC_REGISTRY.register()
def calculate_topiq(
    img: np.ndarray, img2: np.ndarray, device: torch.device, **kwargs
) -> Tensor:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )

    topiq = _get_topiq("cfanet_fr_kadid_res50", True, device)
    with torch.inference_mode():
        return topiq(
            img2batchedtensor(img, device, from_bgr=False),
            img2batchedtensor(img2, device, from_bgr=False),
        )


@METRIC_REGISTRY.register()
def calculate_topiq_nr(img: np.ndarray, device: torch.device, **kwargs) -> Tensor:
    topiq = _get_topiq("cfanet_nr_koniq_res50", False, device)
    with torch.inference_mode():
        return topiq(
            img2batchedtensor(img, device, from_bgr=False),
        )
