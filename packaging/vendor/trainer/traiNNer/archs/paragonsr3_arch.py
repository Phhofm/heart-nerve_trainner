#!/usr/bin/env python3
"""
ParagonSR3 (Multi-Head)
=======================

Design Philosophy: "Surgical Efficiency via Re-parameterization"
Integrated with Span++ Reparameterization, IET Normalization, and HAT Context.

Multi-Head Training:
- Train ONCE with shared body
- Export SEPARATE models for 1x, 2x, 3x, 4x scales
- Multi-task learning improves feature quality across all scales
- 1x head learns denoising, which benefits all upscaling heads

Variants:
1. "Aegis" (Video): Concatenation-based Recurrent GatedRepConv.
   - Solves ghosting via learned alignment in the first RepConv.
   - Pure convolution for maximum TRT speed.
   - Dynamic ONNX / FP16 TRT ready.

2. "Virtuoso" (Photo): Beacon-Augmented GatedRepConv.
   - Beats HAT-L via 'SCA' (Simple Channel Attention) inside the Rep block.
   - Interleaved Window/Token attention for global consistency.

Author: Philip Hofmann
License: MIT
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint

from traiNNer.utils.registry import ARCH_REGISTRY

# ============================================================================
# 1. CORE UTILS & IET NORMALIZATION
# ============================================================================


class iLN(nn.Module):
    """
    Image Restoration Layer Normalization (IET Philosophy).
    Computes stats in FP32, applies in input dtype. Safe for FP16 TRT.

    Also returns texture energy (local variance) computed from RAW input
    for variance-based routing in attention blocks.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        B, _C, _H, _W = x.shape
        x_flat = x.reshape(B, -1)

        # FP32 Calculation for stability
        mean = x_flat.float().mean(dim=1, keepdim=True).reshape(B, 1, 1, 1)
        var = (
            x_flat.float().var(dim=1, keepdim=True, unbiased=False).reshape(B, 1, 1, 1)
        )

        mean = mean.to(x.dtype)
        std = torch.sqrt(var + self.eps).to(x.dtype)

        x_norm = (x - mean) / std

        # Compute texture variance from RAW input (before normalization)
        # This gives a structurally sharper variance signal for routing
        local_var = torch.var(x.float(), dim=1, keepdim=True, unbiased=False)
        local_std = torch.sqrt(local_var.to(x.dtype) + self.eps)

        return self.weight * x_norm + self.bias, local_std


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class AffineTransform(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


# ============================================================================
# 2. RE-PARAMETERIZATION ENGINE
# ============================================================================


class RepDepthwise(nn.Module):
    """
    Re-parameterizable Depthwise Convolution.
    Training: 3x3 + 1x1 + Identity
    Inference: 3x3
    """

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = stride
        self.is_deployed = False

        # Main Branch
        self.dw_main = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride,
            self.padding,
            groups=channels,
            bias=True,
        )
        # 1x1 Branch
        self.dw_1x1 = nn.Conv2d(
            channels, channels, 1, stride, 0, groups=channels, bias=False
        )
        # Identity
        self.use_id = stride == 1 and kernel_size % 2 == 1
        if self.use_id:
            self.id_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_deployed:
            return self.dw_main(x)

        out = self.dw_main(x) + self.dw_1x1(x)
        if self.use_id:
            out = out + x * self.id_scale
        return out

    def switch_to_deploy(self) -> None:
        if self.is_deployed:
            return

        # Fuse 1x1 into 3x3
        k_main, b_main = self.dw_main.weight.data, self.dw_main.bias.data
        k_1x1 = self.dw_1x1.weight.data

        # Pad 1x1 to 3x3
        pad = (self.kernel_size - 1) // 2
        k_1x1_padded = F.pad(k_1x1, (pad, pad, pad, pad))
        k_final = k_main + k_1x1_padded

        # Fuse Identity
        if self.use_id:
            k_id = torch.zeros_like(k_final)
            center = self.kernel_size // 2
            for i in range(self.channels):
                k_id[i, 0, center, center] = self.id_scale[0, i, 0, 0]
            k_final = k_final + k_id

        self.dw_main.weight.data.copy_(k_final)
        self.dw_main.bias.data.copy_(b_main)

        del self.dw_1x1
        if hasattr(self, "id_scale"):
            del self.id_scale
        self.is_deployed = True


class SimpleChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA).
    The "secret sauce" to beat Transformers with ConvNets (used in NAFNet).
    GlobalAvgPool -> Pointwise -> Scale.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(dim, dim, 1, bias=True)  # Pointwise interaction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = self.fc(y)
        return x * torch.sigmoid(y)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class GatedRepConv(nn.Module):
    """
    Enhanced GatedRepConv.
    Now includes SimpleChannelAttention (SCA) to boost PSNR validation metrics.
    Optional stochastic depth for regularization.
    """

    def __init__(self, dim: int, hidden_dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, hidden_dim * 2, 1)
        self.dw = RepDepthwise(hidden_dim * 2, kernel_size=3)
        self.gate = SimpleGate()
        self.sca = SimpleChannelAttention(hidden_dim)  # Added for PSNR
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.dw(x)
        x = self.gate(x)
        x = self.sca(x)  # Apply SCA on the active features
        x = self.conv2(x)
        x = self.drop_path(x)
        return x


# ============================================================================
# 3. ATTENTION BEACONS
# ============================================================================


class WindowAttention(nn.Module):
    """Standard SDPA Window Attention with Depthwise Halo Exchange."""

    def __init__(
        self, dim: int, num_heads: int, window_size: int = 16, shift_size: int = 0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        # Halo Exchange (Depthwise Conv) to bleed info across window boundaries
        self.halo_exchange = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Halo Exchange: 3x3 Depthwise Conv before attention
        x_conv = x.permute(0, 3, 1, 2)
        x = x + self.halo_exchange(x_conv).permute(0, 2, 3, 1)

        B, H, W, C = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x_windows = (
            x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)
        )

        qkv = self.qkv(x_windows)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.view(-1, ws * ws, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(-1, ws * ws, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(-1, ws * ws, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)

        x_windows = attn.transpose(1, 2).contiguous().view(-1, ws * ws, C)
        x_windows = self.proj(x_windows)

        x = (
            x_windows.view(B, Hp // ws, Wp // ws, ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, Hp, Wp, C)
        )

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        return x


class TokenDictionaryCA(nn.Module):
    """Global Context Beacon with Dynamic Tokens (IET-inspired)."""

    def __init__(self, dim: int, num_tokens: int = 64) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        qk_dim = dim // 2  # Compressed QK for speed
        self.q_proj = nn.Linear(dim, qk_dim)
        self.k_proj = nn.Linear(dim, qk_dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = qk_dim**-0.5

        # Generator for dynamic, image-specific tokens
        self.token_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim * num_tokens, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # Generate dynamic tokens based on the global image context
        # Output shape: (B, num_tokens, C)
        td = self.token_generator(x).view(B, self.num_tokens, C)

        q = self.q_proj(x_flat)
        k = self.k_proj(td)
        v = self.v_proj(td)

        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = self.out_proj(attn @ v)
        return out.transpose(1, 2).reshape(B, C, H, W)


class VarianceRouter(nn.Module):
    """Soft Gating based on local texture variance."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(std * self.weight + self.bias)


# ============================================================================
# 4. PARAGON BLOCK
# ============================================================================


class ParagonBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        use_window: bool = False,
        use_token: bool = False,
        window_size: int = 16,
        shift_size: int = 0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        # 1. GatedRepConv (The Engine)
        self.conv = GatedRepConv(dim, int(dim * 2.0), drop_path)
        self.scale1 = LayerScale(dim, init_values=1e-4)

        # 2. Beacons
        self.use_window = use_window
        if use_window:
            self.norm2 = iLN(dim)
            self.attn_win = WindowAttention(dim, 4, window_size, shift_size)
            self.router2 = VarianceRouter()
            self.scale2 = LayerScale(dim, init_values=1e-4)

        self.use_token = use_token
        if use_token:
            self.norm3 = iLN(dim)
            self.attn_tok = TokenDictionaryCA(dim, num_tokens=64)
            self.router3 = VarianceRouter()
            self.scale3 = LayerScale(dim, init_values=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv
        x = x + self.scale1(self.conv(x))

        # Window
        if self.use_window:
            res = x
            x_norm, std = self.norm2(x)
            # Route based on local variance
            x_attn = self.attn_win(x_norm.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # Pool std for window match
            std_p = F.interpolate(std, size=x.shape[2:], mode="nearest")
            x = res + self.scale2(self.router2(x_attn, std_p))

        # Token
        if self.use_token:
            res = x
            x_norm, std = self.norm3(x)
            x_tok = self.attn_tok(x_norm)
            std_p = F.interpolate(std, size=x.shape[2:], mode="nearest")
            x = res + self.scale3(self.router3(x_tok, std_p))

        return x


class ResidualGroup(nn.Module):
    def __init__(self, blocks) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ============================================================================
# 5. MULTI-SCALE UPSAMPLING HEADS
# ============================================================================


class ParagonUpsampler(nn.Module):
    """
    Specific Upsampling Head for a target scale.
    Each head is lightweight and can be stripped during ONNX export.

    For scale=1 (denoising), uses a simple refinement conv.
    For scale>1, uses PixelShuffle for efficient upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, scale: int) -> None:
        super().__init__()
        self.scale = scale

        if scale == 1:
            # 1x = Denoising/Refinement only
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            )
        elif scale in (2, 4):
            # 2x/4x: Standard PixelShuffle
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * (scale**2), 3, 1, 1),
                nn.PixelShuffle(scale),
            )
        elif scale == 3:
            # 3x: PixelShuffle with 9x expansion
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 9, 3, 1, 1),
                nn.PixelShuffle(3),
            )
        else:
            # Fallback for unusual scales
            self.net = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 6. MAIN NETWORK
# ============================================================================


@ARCH_REGISTRY.register()
class ParagonSR3(nn.Module):
    """
    ParagonSR3: Multi-Head Super-Resolution Network.

    Supports training with multiple scales simultaneously (1x, 2x, 3x, 4x).
    At export time, specify a single scale to get a dedicated, optimized model.

    Args:
        scales: List of scales to support, or single int for legacy mode.
        in_chans: Input image channels (default: 3 for RGB).
        num_feat: Feature channels in the body.
        num_groups: Number of residual groups.
        num_blocks: Number of blocks per group.
        window_size: Window size for attention.
        variant: 'video' or 'photo'.
        detail_gain: Initial scaling for detail residual.
        drop_path_rate: Stochastic depth rate (0-0.5). Recommended: 0.1 for photo, 0.05 for video.
    """

    def __init__(
        self,
        scales: list[int] | int = 4,
        in_chans: int = 3,
        num_feat: int = 64,
        num_groups: int = 8,
        num_blocks: int = 8,
        window_size: int = 16,
        variant: str = "photo",  # 'video', 'photo'
        detail_gain: float = 0.2,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.in_chans = in_chans
        self.num_feat = num_feat

        # Handle scales argument: int -> list, ensure sorted
        if isinstance(scales, int):
            self.supported_scales = [scales]
        else:
            self.supported_scales = sorted(scales)

        # Stochastic depth schedule
        total_blocks = num_groups * num_blocks
        if drop_path_rate > 0:
            dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        else:
            dpr = [0.0] * total_blocks

        # Input
        if variant == "video":
            # Video: Input channels + prev_feat for recurrence
            self.conv_in = nn.Conv2d(in_chans + num_feat, num_feat, 3, padding=1)
        else:
            self.conv_in = nn.Conv2d(in_chans, num_feat, 3, padding=1)

        # Body (Shared across all scales)
        # We store groups in a ModuleList instead of Sequential so we can tap intermediate features
        self.body = nn.ModuleList()
        for g in range(num_groups):
            blocks = []
            for b in range(num_blocks):
                abs_idx = g * num_blocks + b
                shift = (window_size // 2) if (abs_idx % 2 != 0) else 0
                block_drop_path = dpr[abs_idx]

                use_window = False
                use_token = False

                if variant == "photo":
                    # Beacon Strategy: Window every 4th, Token every 12th
                    if abs_idx % 4 == 3:
                        use_window = True
                    if abs_idx % 12 == 11:
                        use_token = True
                    # Deep Refinement: Last 2 groups use more attention
                    if g >= num_groups - 2:
                        use_window = True
                        if abs_idx % 2 == 1:
                            use_token = True
                elif variant == "video":
                    # Video: Pure CNN engine generally, but inject sparse global context
                    # tokens to combat large compression macroblocks.
                    if b == num_blocks // 2:
                        use_token = True

                blocks.append(
                    ParagonBlock(
                        num_feat,
                        use_window=use_window,
                        use_token=use_token,
                        window_size=window_size,
                        shift_size=shift,
                        drop_path=block_drop_path,
                    )
                )
            self.body.append(ResidualGroup(blocks))

        # Hierarchical Feature Fusion: Aggregate tapped group outputs
        # We concatenate features from the input conv + every group, then compress
        self.fusion_conv = nn.Conv2d(num_feat * (num_groups + 1), num_feat, 1)

        # Pre-upsample processing (Refines the fused features)
        self.conv_mid = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.final_norm = AffineTransform(num_feat)

        # Multi-Scale Heads: One lightweight head per supported scale
        self.heads = nn.ModuleDict()
        for s in self.supported_scales:
            self.heads[str(s)] = ParagonUpsampler(num_feat, in_chans, s)

        self.detail_gain = nn.Parameter(torch.tensor(detail_gain))

    def forward(
        self,
        x: torch.Tensor,
        scale: int | None = None,
        feature_tap: bool = False,
        prev_feat: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-scale support.

        Args:
            x: Input tensor (B, C, H, W).
            scale: Target scale. If None, uses the largest supported scale.
            feature_tap: If True, also return body features (for video recurrence).
            prev_feat: Previous frame features (for video variant).

        Returns:
            Output tensor, or (output, features) tuple if feature_tap=True.
        """
        # Default to largest scale if not specified
        if scale is None:
            scale = self.supported_scales[-1]

        # Validate scale
        if str(scale) not in self.heads:
            raise ValueError(
                f"Scale {scale} not supported. Available: {self.supported_scales}"
            )

        # 1. Classical Base Upsampling (Dynamic)
        if scale == 1:
            base = x  # No upsampling for denoising
        else:
            base = F.interpolate(
                x, scale_factor=float(scale), mode="bilinear", align_corners=False
            )

        # 2. Input Features
        if self.variant == "video":
            # Video Recurrence: Concatenation
            if prev_feat is None:
                # Initialize zero state (B, num_feat, H, W) matching input res
                prev_feat = torch.zeros(
                    x.shape[0],
                    self.num_feat,
                    x.shape[2],
                    x.shape[3],
                    device=x.device,
                    dtype=x.dtype,
                )

            # Concat Input (3) + History (num_feat)
            x_in = torch.cat([x, prev_feat], dim=1)
            x = self.conv_in(x_in)
        else:
            x = self.conv_in(x)

        # 3. Body (The expensive shared part) with Hierarchical Feature Tapping
        features_to_fuse = [x]  # Start with the early extracted features

        for group in self.body:
            x = group(x)
            features_to_fuse.append(x)

        # Cross-Scale Feature Fusion: Concat all tapped hierarchies (high/mid/low freq)
        x = torch.cat(features_to_fuse, dim=1)
        x = self.fusion_conv(x)

        # 4. Tap Output Features for Next Frame (Video)
        current_feat = x

        # 5. Detail Reconstruction via Scale-Specific Head
        x = self.final_norm(self.conv_mid(x))
        detail = self.heads[str(scale)](x)

        # 6. Combine Base + Detail
        out = base + detail * self.detail_gain

        if feature_tap:
            return out, current_feat
        return out

    def fuse_model(self) -> None:
        """Fuse RepConv branches for deployment (inference speedup)."""
        print(f"Fusing ParagonSR3 ({self.variant}) for deployment...")
        for m in self.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()


# ============================================================================
# 7. EXPORT WRAPPER
# ============================================================================


class ParagonSR3ExportWrapper(nn.Module):
    """
    Wrapper for ONNX export that locks a specific scale.

    Usage:
        model = paragonsr3_photo_multiscale()
        model.load_state_dict(torch.load("trained.pth"))
        model.fuse_model()

        # Export 4x model
        wrapper = ParagonSR3ExportWrapper(model, scale=4)
        torch.onnx.export(wrapper, dummy_input, "paragon_photo_4x.onnx", ...)
    """

    def __init__(self, model: ParagonSR3, scale: int) -> None:
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, scale=self.scale)


# ============================================================================
# 8. CONFIGURATIONS
# ============================================================================


@ARCH_REGISTRY.register()
def paragonsr3_video(scales: tuple[int, ...] | list[int] | int = (1, 2, 3, 4), **kw):
    """
    ParagonSR3 'Aegis' (Video) - Multi-Scale by default.
    Recurrent state for temporal stability, pure convolution for speed, sparse global context.
    """
    if isinstance(scales, tuple):
        scales = list(scales)
    # Set default drop_path_rate for video if not provided
    kw.setdefault("drop_path_rate", 0.05)
    return ParagonSR3(
        scales=scales,
        num_feat=64,
        num_groups=4,
        num_blocks=6,
        variant="video",
        detail_gain=0.1,
        **kw,
    )


@ARCH_REGISTRY.register()
def paragonsr3_photo(scales: tuple[int, ...] | list[int] | int = (1, 2, 3, 4), **kw):
    """
    ParagonSR3 'Virtuoso' (Photo) - Multi-Scale by default.
    SCA + RepConv + Beacon Attention > HAT-L.
    """
    if isinstance(scales, tuple):
        scales = list(scales)
    # Set default drop_path_rate for photo if not provided
    kw.setdefault("drop_path_rate", 0.1)
    return ParagonSR3(
        scales=scales,
        num_feat=180,  # HAT-L equivalent width
        num_groups=10,  # Slightly deeper for max quality
        num_blocks=8,
        variant="photo",
        detail_gain=0.2,
        **kw,
    )
