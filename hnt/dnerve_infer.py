"""D-NERVE inference for BODY Studio: substrate + decoder -> restored image.

The decoder is tiny; the frozen DINOv2-L substrate (~1.2 GB, Apache-2.0)
is the heavyweight piece. It is downloaded once into the app data dir and
cached there. Inference is one pass at input resolution (restoration, not
upscaling): DINOv2 patch features are bilinearly upsampled to full size,
concatenated with the degraded image, and decoded.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.nn import functional as F  # noqa: N812

from .dnerve_arch import DnervDecoder, load_dnerve_decoder

SUBSTRATE_REPO = "Phips/D-NERVE"
SUBSTRATE_FILE = "substrate/dinov2_vitl14_pretrain.pth"


def substrate_file(paths) -> Path:
    """Local substrate path: HNT_DINOV2 override, else the app data dir."""
    override = os.environ.get("HNT_DINOV2")
    if override:
        return Path(override)
    return paths.data / "substrates" / "dinov2_vitl14_pretrain.pth"


def ensure_substrate(paths, on_log=lambda _m: None) -> str:
    """Return the substrate path, downloading it once if needed."""
    p = substrate_file(paths)
    if p.is_file():
        return str(p)
    if os.environ.get("HNT_DINOV2"):
        raise FileNotFoundError(
            f"HNT_DINOV2 points at a missing file: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    on_log(f"Downloading the D-NERVE vision substrate ({SUBSTRATE_FILE})…")
    got = hf_hub_download(SUBSTRATE_REPO, SUBSTRATE_FILE, local_dir=str(p.parent))
    return str(got)


def load_dino(weights_path: str, device: str = "cuda"):
    """Frozen DINOv2-L feature extractor from a local weights file."""
    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14",
                         pretrained=False)
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    net.load_state_dict(sd, strict=True)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def extract_features(dino, x: torch.Tensor) -> torch.Tensor:
    """Patch-token features reshaped to a (B, 1024, h/14, w/14) grid."""
    b, _, h, w = x.shape
    out = dino.forward_features(x)
    tokens = out["x_norm_patchtokens"]  # (B, gh*gw, 1024)
    gh, gw = h // 14, w // 14
    return tokens.transpose(1, 2).reshape(b, -1, gh, gw)


def load_dnerve(decoder_path: str, paths, device: str = "cuda",
                on_log=lambda _m: None):
    """(decoder, dino) pair ready for run_dnerve; substrate is cached."""
    decoder = load_dnerve_decoder(decoder_path).to(device).eval()
    dino = load_dino(ensure_substrate(paths, on_log), device)
    return decoder, dino


@torch.no_grad()
def run_dnerve(decoder, dino, img: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Restore one RGB uint8 image at its input resolution."""
    arr = img.astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    h, w = x.shape[-2:]
    ph = (14 - h % 14) % 14
    pw = (14 - w % 14) % 14
    if ph or pw:
        x = F.pad(x[None], (0, pw, 0, ph), mode="reflect")[0]
    xb = x[None].to(device)
    feats = extract_features(dino, xb)
    out = decoder(feats, xb)[0, :, :h, :w].clamp(0, 1).cpu()
    return (out.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
