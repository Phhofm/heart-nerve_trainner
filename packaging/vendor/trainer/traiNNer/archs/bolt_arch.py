"""BOLT — GRL-B's Global/Regional/Local hierarchy with a swappable attention bias.

BOLT keeps GRL-B's proven structure (anchored stripe attention for Global, window
attention for Regional, a channel-attention conv for Local) and separates the two
concerns that are easy to tangle together: the *tiling geometry* and the *score
function*. The geometry — tile size, shift, partition, head split, merge — is written
once in :class:`WindowAttention` / :class:`AnchorStripeAttention` and shared. The
score function is a plugin (:class:`ScoreFunction`) with two implementations:

``attn_backend="grl"`` (:class:`GRLCosine`)
    GRL's own: cosine similarity, a CPB-MLP-generated relative position bias added
    to the logits, an explicit softmax, and a cyclic ``torch.roll`` shift with
    additive attention masks. The bias table and the masks force the score matrix to
    be materialized, so this path cannot use a fused kernel. It is the ablation
    control, and :func:`scripts/bolt/verify_grl_equivalence.py` checks it is
    numerically identical to spandrel's GRL.

``attn_backend="rib"`` (:class:`RIBRank`)
    The same two attention patterns with the bias table and the masks removed, so
    the whole thing runs on fused FlashAttention/SDPA kernels:

    * the continuous CPB bias becomes rank-factorized position features
      concatenated onto Q/K (RIB-style), which turns bias addition into a dot
      product;
    * cyclic roll + additive mask becomes non-wrapping pad-and-partition, which
      needs no mask. This works for the anchored stripe attention too: the anchor
      tile is exactly ``stripe / df``, so padding the stripe and the anchor by half
      of *their own* tile size yields the same tile count and keeps the two grids
      co-aligned, which is what makes a mask-free two-resolution cross-attention
      possible;
    * GRL's learnable ``logit_scale`` is preserved exactly. Because the logits are
      now a plain dot product, it is applied as a half power to Q and K so their
      product reproduces ``logit_scale * cos(q, k)``. No default ``1/sqrt(d)``
      scaling is added anywhere.

The body is tokens-first ``(B, L, C)``, matching GRL and the repo's ``iLN``.

:class:`remap_grl_state_dict` maps a ``grl_b`` checkpoint across, so the RIB arm can
warm-start from the GRL arm's weights and the measured difference is the attention
mechanism rather than a fresh draw of the shared weights.
"""

from __future__ import annotations

import functools
import inspect
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal, NamedTuple

import torch
from spandrel.util import store_hyperparameters
from spandrel.util.timm import trunc_normal_
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.utils import checkpoint

from traiNNer.archs.arch_util import iLN
from traiNNer.archs.hat_iln_arch import DropPath, Mlp, Upsample
from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY

NUM_OUT_FEATS = 64

# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def bchw_to_blc(x: Tensor) -> Tensor:
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x: Tensor, x_size: tuple[int, int]) -> Tensor:
    b, _, c = x.shape
    return x.transpose(1, 2).view(b, c, *x_size)


def blc_to_bhwc(x: Tensor, x_size: tuple[int, int]) -> Tensor:
    b, _, c = x.shape
    return x.view(b, *x_size, c)


# ---------------------------------------------------------------------------
# tiling
#
# Both backends need the same two primitives: cut a (B, H, W, C) map into tiles and
# put it back together. They differ only in *how the tile grid is offset* when a
# block shifts its windows -- a cyclic roll (GRL) or a pad-then-partition (RIB) --
# so each of those is isolated in one function and chosen by
# :attr:`ScoreFunction.shift_mode`.
# ---------------------------------------------------------------------------


def shift_cyclic(x: Tensor, shift: tuple[int, int]) -> Tensor:
    """Roll a (B, H, W, C) map so the tile grid is offset by ``shift``.

    Cyclic, so content wraps across the border. That is what forces the additive
    window mask: the wrapped seam joins two tiles that should not attend to each
    other, and the mask is the only record of it.
    """
    return torch.roll(x, shifts=(-shift[0], -shift[1]), dims=(1, 2))


def unshift_cyclic(x: Tensor, shift: tuple[int, int]) -> Tensor:
    """Exact inverse of :func:`shift_cyclic`."""
    return torch.roll(x, shifts=(shift[0], shift[1]), dims=(1, 2))


def pad_replicate(x: Tensor, shift: tuple[int, int]) -> Tensor:
    """Replicate-pad the H/W axes of a (B, H, W, C) map by ``shift``.

    Non-cyclic, so the grid is offset by a real pad instead of a wrap and no mask is
    needed: the extra border tiles are dropped again by :func:`merge_tiles`.

    ``F.pad`` acts on the *last* two axes, so it cannot be called directly on a
    (B, H, W, C) tensor -- that would pad W and C instead. Slicing the border row
    and column keeps the intent obvious and stays in the channels-last layout, and
    ``cat`` yields contiguous output so the reshape in :func:`partition` is a view.

    Replicate rather than reflect: reflection only carries meaning for images, and
    replicate is what ONNX and TensorRT both implement natively.
    """
    sh, sw = shift
    if sh:
        x = torch.cat(
            (x[:, :1].expand(-1, sh, -1, -1), x, x[:, -1:].expand(-1, sh, -1, -1)),
            dim=1,
        )
    if sw:
        x = torch.cat(
            (
                x[:, :, :1].expand(-1, -1, sw, -1),
                x,
                x[:, :, -1:].expand(-1, -1, sw, -1),
            ),
            dim=2,
        )
    return x


def partition(x: Tensor, tile: tuple[int, int]) -> Tensor:
    """(B, H, W, C) -> (B * num_tiles, th, tw, C). H and W must be tile multiples."""
    b, h, w, c = x.shape
    th, tw = tile
    x = x.view(b, h // th, th, w // tw, tw, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, th, tw, c)


def merge_tiles(
    tiles: Tensor, tile: tuple[int, int], pad: tuple[int, int], x_size: tuple[int, int]
) -> Tensor:
    """Inverse of :func:`partition`, cropping the ``pad`` that was added."""
    th, tw = tile
    sh, sw = pad
    h, w = x_size
    nh, nw = (h + 2 * sh) // th, (w + 2 * sw) // tw
    b = tiles.shape[0] // (nh * nw)
    x = tiles.reshape(b, nh, nw, th, tw, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h + 2 * sh, w + 2 * sw, -1)
    return x[:, sh : sh + h, sw : sw + w]


def split_qkv_heads(tiles: Tensor, num_heads: int) -> tuple[Tensor, Tensor, Tensor]:
    """(n_tiles, N, 3C) -> three (n_tiles, heads, N, head_dim)."""
    n_tiles, n, _ = tiles.shape
    qkv = tiles.reshape(n_tiles, n, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def project_to_heads(tokens: Tensor, num_heads: int) -> Tensor:
    """(n_tiles, N, C) -> (n_tiles, heads, N, head_dim), no QKV split."""
    return tokens.reshape(tokens.shape[0], tokens.shape[1], num_heads, -1).permute(
        0, 2, 1, 3
    )


def merge_heads(out: Tensor) -> Tensor:
    """(n_tiles, heads, N, head_dim) -> (n_tiles, N, heads * head_dim)."""
    return out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)


def get_stripe_info(
    stripe_size_in: Sequence[int],
    stripe_groups_in: Sequence[int | None],
    stripe_shift: bool,
    input_resolution: tuple[int, int],
) -> tuple[list[int], list[int]]:
    """Resolve a stripe config into the tile and shift actually used at runtime."""
    stripe_size, shift_size = [], []
    for s, g, d in zip(stripe_size_in, stripe_groups_in, input_resolution):
        if g is None:
            stripe_size.append(s)
            shift_size.append(s // 2 if stripe_shift else 0)
        else:
            stripe_size.append(d // g)
            shift_size.append(0 if g == 1 else d // (g * 2))
    return stripe_size, shift_size


# ---------------------------------------------------------------------------
# GRL geometry tables
#
# The bias table, the pair-wise index into it, and the additive mask that hides the
# seam a cyclic shift opens. All three are pure functions of the geometry, so they
# are built once and shared by every block.
# ---------------------------------------------------------------------------


def _relative_coords_table(
    window_size: tuple[int, int],
    pretrained_window_size: Sequence[int] = (0, 0),
    down_factor: int = 1,
) -> Tensor:
    """Continuous relative-coordinate table: 1, Wh + AWh - 1, Ww + AWw - 1, 2."""
    ws = window_size
    aws = [w // down_factor for w in window_size]
    paws = [w // down_factor for w in pretrained_window_size]

    ts_p = [w1 - 1 - (w1 - w2) // 2 for w1, w2 in zip(ws, aws)]
    ts_n = [-(w2 - 1) - (w1 - w2) // 2 for w1, w2 in zip(ws, aws)]
    pts = [w1 - 1 - (w1 - w2) // 2 for w1, w2 in zip(pretrained_window_size, paws)]

    coord_h = torch.arange(ts_n[0], ts_p[0] + 1, dtype=torch.float32)
    coord_w = torch.arange(ts_n[1], ts_p[1] + 1, dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(
        1, 2, 0
    )
    table = table.contiguous().unsqueeze(0)
    # GRL normalizes against the pre-training window size when the checkpoint has
    # one, otherwise against the current one, so coordinates stay comparable.
    divisor = pts if pts[0] > 0 else ts_p
    table = torch.stack(
        [table[:, :, :, 0] / divisor[0], table[:, :, :, 1] / divisor[1]], dim=-1
    )
    table *= 8
    return torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / math.log2(8)


def _meshgrid_coords(
    start_coords: Sequence[int], end_coords: Sequence[int]
) -> Tensor:
    coord_h = torch.arange(start_coords[0], end_coords[0])
    coord_w = torch.arange(start_coords[1], end_coords[1])
    coords = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij"))
    return torch.flatten(coords, 1)


def _coords_diff_odd(
    coords1: Tensor, coords2: Tensor, start_coord: Sequence[int], max_diff: int
) -> Tensor:
    coords = coords1[:, :, None] - coords2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()
    coords[:, :, 0] += start_coord[0]
    coords[:, :, 1] += start_coord[1]
    coords[:, :, 0] *= max_diff
    return coords.sum(-1)


def _relative_index(
    window_size: tuple[int, int],
    down_factor: int = 1,
    window_to_anchor: bool = True,
) -> Tensor:
    """Pair-wise index into :func:`_relative_coords_table`. Both grids start at (0, 0)."""
    ws = window_size
    aws = [w // down_factor for w in window_size]

    coords = _meshgrid_coords((0, 0), window_size)
    coords_anchor = _meshgrid_coords((0, 0), aws)

    max_horizontal_diff = aws[1] + ws[1] - 1
    if window_to_anchor:
        return _coords_diff_odd(
            coords, coords_anchor, [w2 - 1 for w2 in aws], max_horizontal_diff
        )
    return _coords_diff_odd(
        coords_anchor, coords, [w1 - 1 for w1 in ws], max_horizontal_diff
    )


def _region_labels(
    input_resolution: tuple[int, int],
    window_size: tuple[int, int],
    shift_size: Sequence[int],
) -> Tensor:
    """Label each token with which of the 9 shifted regions it came from."""
    img_mask = torch.zeros((1, *input_resolution, 1))
    h_slices = (
        slice(0, -window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    )
    w_slices = (
        slice(0, -window_size[1]),
        slice(-window_size[1], -shift_size[1]),
        slice(-shift_size[1], None),
    )
    cnt = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, h_slice, w_slice, :] = cnt
            cnt += 1

    return partition(img_mask, window_size).view(-1, math.prod(window_size))


def _attn_mask(
    input_resolution: tuple[int, int],
    window_size: tuple[int, int],
    shift_size: Sequence[int],
) -> Tensor:
    """Additive SW-MSA mask: (num_tiles, window_size**2, window_size**2)."""
    labels = _region_labels(input_resolution, window_size, shift_size)
    mask = labels.unsqueeze(1) - labels.unsqueeze(2)
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


def _cross_mask(
    input_resolution: tuple[int, int],
    window_size: tuple[int, int],
    shift_size: Sequence[int],
    down_factor: int,
    window_to_anchor: bool,
) -> Tensor:
    """Additive mask between a stripe tile and the anchor tile covering it."""
    anchor_resolution = [s // down_factor for s in input_resolution]
    anchor_tile = [s // down_factor for s in window_size]
    anchor_shift = [s // down_factor for s in shift_size]

    labels = _region_labels(input_resolution, window_size, shift_size)
    labels_anchor = _region_labels(anchor_resolution, anchor_tile, anchor_shift)

    if window_to_anchor:
        mask = labels.unsqueeze(2) - labels_anchor.unsqueeze(1)
    else:
        mask = labels_anchor.unsqueeze(2) - labels.unsqueeze(1)
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


class CrossTables(NamedTuple):
    """Stripe cross-attention tables for one orientation."""

    table: Tensor
    #: ``(anchor->stripe, stripe->anchor)``.
    index: tuple[Tensor, Tensor]
    #: ``None`` in the unshifted direction: nothing to mask.
    mask: tuple[Tensor, Tensor]


class GRLTables(nn.Module):
    """GRL's shared bias table, index, and mask, held as buffers.

    One instance per network, read by every block. They are buffers rather than
    plain tensors so ``.to(device)`` actually reaches them, and a block resolves them
    per forward so it always reads the current, moved tensor. Registering the same
    tensors on every block instead would multiply the multi-megabyte stripe indices by
    the block count for no benefit.
    """

    def __init__(
        self,
        input_resolution: tuple[int, int],
        window_tile: tuple[int, int],
        window_shift: tuple[int, int],
        stripe_tiles: dict[Literal["h", "v"], tuple[int, int]],
        stripe_shifts: dict[Literal["h", "v"], tuple[int, int]],
        down_factor: int,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "window_table",
            _relative_coords_table(window_tile),
            persistent=False,
        )
        self.register_buffer(
            "window_index", _relative_index(window_tile), persistent=False
        )
        self.register_buffer(
            "window_mask",
            _attn_mask(input_resolution, window_tile, window_shift),
            persistent=False,
        )
        for orientation, tile in stripe_tiles.items():
            table = _relative_coords_table(tile, (0, 0), down_factor)
            self.register_buffer(f"stripe_{orientation}_table", table, persistent=False)
            for i, w2a in ((0, False), (1, True)):
                self.register_buffer(
                    f"stripe_{orientation}_index_{i}",
                    _relative_index(tile, down_factor, w2a),
                    persistent=False,
                )
                self.register_buffer(
                    f"stripe_{orientation}_mask_{i}",
                    _cross_mask(
                        input_resolution, tile, stripe_shifts[orientation], down_factor,
                        w2a,
                    ),
                    persistent=False,
                )

    def window(self, shifted: bool) -> tuple[Tensor, Tensor, Tensor | None]:
        """(table, index, mask). ``mask`` is ``None`` when the block does not shift."""
        return self.window_table, self.window_index, self.window_mask if shifted else None

    def stripe(
        self, orientation: Literal["h", "v"], shifted: bool
    ) -> tuple[Tensor, tuple[Tensor, Tensor], tuple[Tensor | None, Tensor | None]]:
        prefix = f"stripe_{orientation}"
        index = (getattr(self, f"{prefix}_index_0"), getattr(self, f"{prefix}_index_1"))
        if not shifted:
            return getattr(self, f"{prefix}_table"), index, (None, None)
        mask = (getattr(self, f"{prefix}_mask_0"), getattr(self, f"{prefix}_mask_1"))
        return getattr(self, f"{prefix}_table"), index, mask


# ---------------------------------------------------------------------------
# what a score function is told about its geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowBias:
    """Geometry of one window-attention site."""

    tile: tuple[int, int]
    #: Populated only for the GRL backend, which is the one that needs a bias table.
    table: Tensor | None = None
    index: Tensor | None = None
    mask: Tensor | None = None


@dataclass(frozen=True)
class CrossBias:
    """Geometry of one anchored stripe-attention site."""

    stripe_tile: tuple[int, int]
    anchor_tile: tuple[int, int]
    table: Tensor | None = None
    index: tuple[Tensor, Tensor] | None = None
    mask: tuple[Tensor | None, Tensor | None] | None = None


# ---------------------------------------------------------------------------
# score functions
# ---------------------------------------------------------------------------


def _cosine_attn(q: Tensor, k: Tensor) -> Tensor:
    return F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)


class CPBMLP(nn.Sequential):  # noqa: N801
    """Generates the continuous relative position bias (GRL / SwinV2).

    A ``Sequential`` subclass rather than a wrapped ``body`` so the parameter names
    match GRL's exactly, which is what lets a ``grl_b`` checkpoint transfer.
    """

    def __init__(self, in_channels: int, out_channels: int, channels: int = 512) -> None:
        super().__init__(
            nn.Linear(in_channels, channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels, out_channels, bias=False),
        )


class AffineTransform(nn.Module):
    """GRL's ``logit_scale`` + CPB-generated relative bias + additive mask."""

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )
        self.cpb_mlp = CPBMLP(2, num_heads)

    def forward(
        self, attn: Tensor, table: Tensor, index: Tensor, mask: Tensor | None
    ) -> Tensor:
        b_, h, n1, n2 = attn.shape
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()

        bias_table = self.cpb_mlp(table).view(-1, h)
        bias = bias_table[index.reshape(-1)]
        bias = bias.view(n1, n2, -1).permute(2, 0, 1).contiguous()
        attn = attn + (16 * torch.sigmoid(bias)).unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(b_ // nw, nw, h, n1, n2) + mask
            attn = attn.view(-1, h, n1, n2)

        return attn


class ScoreFunction(nn.Module):
    """Turns Q/K/V into attention output. One interface, two implementations.

    Both score with ``logit_scale * cos(q, k)`` plus a positional term, and both
    keep GRL's learnable, clamped ``logit_scale``. They differ in how the positional
    term enters the logits, which is also what decides the SDPA kernel: a bias table
    plus a mask has to be materialized (:class:`GRLCosine`), a rank-factorized
    position feature concatenated onto Q/K does not (:class:`RIBRank`).

    Subclasses own their learnable parameters, so there is one instance per attention
    site. :attr:`shift_mode` is the other half of the backend choice and says how the
    pattern should offset its tile grid.
    """

    #: "roll" = cyclic shift plus additive mask, "pad" = pad and partition, no mask.
    shift_mode: ClassVar[Literal["roll", "pad"]]

    def window(self, q: Tensor, k: Tensor, v: Tensor, bias: WindowBias) -> Tensor:
        raise NotImplementedError

    def cross(
        self, q: Tensor, k: Tensor, v: Tensor, bias: CrossBias, direction: int
    ) -> Tensor:
        """``direction`` 0 is anchor->stripe, 1 is stripe->anchor."""
        raise NotImplementedError


class GRLCosine(ScoreFunction):
    """GRL's own score function. The ablation control; never a fused kernel."""

    shift_mode: ClassVar[Literal["roll", "pad"]] = "roll"

    def __init__(self, num_heads: int, num_directions: int, attn_drop: float) -> None:
        super().__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        # GRL names these attn_transform (window) and attn_transform1/2 (the two
        # stripe directions); remap_grl_state_dict() maps them onto this list.
        self.transforms = nn.ModuleList(
            AffineTransform(num_heads) for _ in range(num_directions)
        )

    def _attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        transform: AffineTransform,
        table: Tensor | None,
        index: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        if table is None or index is None:
            raise ValueError("the grl backend needs its bias tables")
        attn = transform(_cosine_attn(q, k), table, index, mask)
        return self.attn_drop(self.softmax(attn)) @ v

    def window(self, q: Tensor, k: Tensor, v: Tensor, bias: WindowBias) -> Tensor:
        return self._attend(
            q, k, v, self.transforms[0], bias.table, bias.index, bias.mask
        )

    def cross(
        self, q: Tensor, k: Tensor, v: Tensor, bias: CrossBias, direction: int
    ) -> Tensor:
        if bias.index is None or bias.mask is None:
            raise ValueError("the grl backend needs its bias tables")
        return self._attend(
            q,
            k,
            v,
            self.transforms[direction],
            bias.table,
            bias.index[direction],
            bias.mask[direction],
        )


@functools.lru_cache(maxsize=32)
def _tile_coords(tile: tuple[int, int], n_freqs: int, device: torch.device) -> Tensor:
    """Tile-local coordinates with sinusoidal features: (N, 2 + 4 * n_freqs).

    Normalizing over the tile's own extent is what lets the stripe grid and the
    anchor grid (exactly ``tile / df``) share one coordinate space. The device is
    part of the cache key, so the host-to-device copy happens once per geometry rather
    than once per attention site per forward.
    """
    th, tw = tile
    yy, xx = torch.meshgrid(
        torch.arange(th, device=device),
        torch.arange(tw, device=device),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2).float()
    coords[:, 0] = (2.0 * (coords[:, 0] + 0.5) / tw) - 1.0
    coords[:, 1] = (2.0 * (coords[:, 1] + 0.5) / th) - 1.0

    for i in range(n_freqs):
        freq = 2.0**i
        coords = torch.cat(
            [coords, torch.sin(coords[:, :2] * freq), torch.cos(coords[:, :2] * freq)],
            dim=-1,
        )
    return coords


class PositionFeatures(nn.Module):
    """Rank-factorized implicit position features, one set per direction.

    ``num_directions`` lets one hidden layer serve several attention directions: the
    anchored stripe attention needs two (anchor->stripe and stripe->anchor), window
    attention needs one.
    """

    def __init__(
        self,
        num_heads: int,
        rank: int = 8,
        hidden_dim: int = 32,
        n_freqs: int = 10,
        num_directions: int = 1,
    ) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        self.n_input = 2 + 4 * n_freqs

        self.to_hidden = nn.Parameter(torch.empty(self.n_input, hidden_dim))
        self.hidden_b = nn.Parameter(torch.zeros(1, hidden_dim))
        self.to_q = nn.Parameter(
            torch.empty(num_directions, num_heads, hidden_dim, rank)
        )
        self.to_k = nn.Parameter(
            torch.empty(num_directions, num_heads, hidden_dim, rank)
        )

        for p in (self.to_hidden, self.to_q, self.to_k):
            nn.init.normal_(p, mean=0.0, std=0.05)
        nn.init.zeros_(self.hidden_b)

    def forward(self, tile: tuple[int, int]) -> list[tuple[Tensor, Tensor]]:
        """Per direction, the ``(query, key)`` position features: (heads, N, rank).

        Query and key get their own read-out of the shared hidden layer, so the
        positional logit ``<P_q(c_i), P_k(c_j)>`` is a free function of the coordinate
        pair rather than a self-similarity.
        """
        coords = _tile_coords(tile, self.n_freqs, self.to_hidden.device)
        hidden = F.relu(coords @ self.to_hidden.float() + self.hidden_b.float())
        return [
            (
                torch.einsum("nd,hdr->hnr", hidden, self.to_q[d].float()),
                torch.einsum("nd,hdr->hnr", hidden, self.to_k[d].float()),
            )
            for d in range(self.to_q.shape[0])
        ]


def sdpa_with_position(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pos_q: Tensor,
    pos_k: Tensor,
    logit_scale: Tensor,
    pos_logit_scale: Tensor,
    dropout_p: float,
) -> Tensor:
    """Attention with logits ``logit_scale * cos(q, k) + pos_scale * <pos_q, pos_k>``.

    ``logit_scale`` goes on as a *half power* to Q and K so their plain dot product
    reproduces it exactly. No default ``1/sqrt(d)`` scaling is added, because the
    logits already are cosine-based and temperature-scaled.

    Q and K are L2-normalized *before* the position features are concatenated, so the
    content half reproduces GRL's cosine similarity while the positional half stays a
    separate, independently scaled bilinear term.

    The head dim is padded up to a multiple of 8 because the fused kernels require
    it; the padding is sliced back off the output and contributes nothing to V.
    """
    d = q.shape[-1]
    r = pos_q.shape[-1]

    ls = logit_scale.clamp(max=math.log(1.0 / 0.01)).exp().sqrt()
    pl = pos_logit_scale.exp().sqrt()

    q_cat = torch.cat([F.normalize(q, dim=-1) * ls, pos_q * pl], dim=-1)
    k_cat = torch.cat([F.normalize(k, dim=-1) * ls, pos_k * pl], dim=-1)

    pad = (8 - (q_cat.shape[-1] % 8)) % 8
    if pad:
        q_cat = F.pad(q_cat, (0, pad))
        k_cat = F.pad(k_cat, (0, pad))
    v_cat = F.pad(v, (0, r + pad))

    out = F.scaled_dot_product_attention(
        q_cat, k_cat, v_cat, scale=1.0, dropout_p=dropout_p
    )
    return out[..., :d]


class RIBRank(ScoreFunction):
    """Rank-factorized position features on fused SDPA. BOLT's default backend."""

    shift_mode: ClassVar[Literal["roll", "pad"]] = "pad"

    def __init__(
        self,
        num_heads: int,
        num_directions: int,
        attn_drop: float,
        rank: int,
        hidden_dim: int,
        n_freqs: int,
    ) -> None:
        super().__init__()
        self.pos = PositionFeatures(
            num_heads, rank, hidden_dim, n_freqs, num_directions=num_directions
        )
        # One parameter per direction, so a GRL logit_scale maps across as a plain
        # rename instead of needing a stack-and-slice step in the remap.
        self.logit_scale = nn.ParameterList(
            nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
            for _ in range(num_directions)
        )
        self.pos_logit_scale = nn.ParameterList(
            nn.Parameter(torch.zeros((num_heads, 1, 1)))
            for _ in range(num_directions)
        )
        self.attn_drop = attn_drop

    def _attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pos_q: Tensor,
        pos_k: Tensor,
        direction: int,
    ) -> Tensor:
        n_tiles = q.shape[0]
        pos_q = pos_q.unsqueeze(0).expand(n_tiles, -1, -1, -1).to(q.dtype)
        pos_k = pos_k.unsqueeze(0).expand(n_tiles, -1, -1, -1).to(k.dtype)
        return sdpa_with_position(
            q,
            k,
            v,
            pos_q,
            pos_k,
            self.logit_scale[direction],
            self.pos_logit_scale[direction],
            self.attn_drop if self.training else 0.0,
        )
    def window(self, q: Tensor, k: Tensor, v: Tensor, bias: WindowBias) -> Tensor:
        # Self-attention inside one tile; the offset is what carries position.
        pos_q, pos_k = self.pos(bias.tile)[0]
        return self._attend(q, k, v, pos_q, pos_k, 0)

    def cross(
        self, q: Tensor, k: Tensor, v: Tensor, bias: CrossBias, direction: int
    ) -> Tensor:
        # The stripe grid and the anchor grid (exactly stripe / df) share one
        # normalized coordinate space, so a stripe token and the anchor token
        # covering it get the same coordinate.
        q_tile, k_tile = (
            (bias.anchor_tile, bias.stripe_tile)
            if direction == 0
            else (bias.stripe_tile, bias.anchor_tile)
        )
        pos_q, _ = self.pos(q_tile)[direction]
        _, pos_k = self.pos(k_tile)[direction]
        return self._attend(q, k, v, pos_q, pos_k, direction)


# ---------------------------------------------------------------------------
# attention patterns
# ---------------------------------------------------------------------------


def _offset(x: Tensor, shift: tuple[int, int], mode: str) -> Tensor:
    if shift == (0, 0):
        return x
    return shift_cyclic(x, shift) if mode == "roll" else pad_replicate(x, shift)


class WindowAttention(nn.Module):
    """Regional attention inside a ``tile`` x ``tile`` window."""

    def __init__(
        self, tile: tuple[int, int], num_heads: int, shifted: bool, score: ScoreFunction
    ) -> None:
        super().__init__()
        self.tile = tile
        self.num_heads = num_heads
        self.shifted = shifted
        self.score = score

    def forward(
        self, qkv: Tensor, x_size: tuple[int, int], tables: GRLTables | None
    ) -> Tensor:
        h, w = x_size
        b, _, c = qkv.shape
        out_dim = c // 3
        # A shift is half a tile, whichever way the backend realizes it.
        shift = (self.tile[0] // 2, self.tile[1] // 2) if self.shifted else (0, 0)

        x = _offset(qkv.view(b, h, w, c), shift, self.score.shift_mode)
        tiles = partition(x, self.tile)

        q, k, v = split_qkv_heads(
            tiles.reshape(-1, math.prod(self.tile), c), self.num_heads
        )
        bias = (
            WindowBias(self.tile)
            if tables is None
            else WindowBias(self.tile, *tables.window(self.shifted))
        )
        out = self.score.window(q, k, v, bias)
        out = merge_heads(out).reshape(-1, *self.tile, out_dim)
        # Only the pad backend changed the spatial size; a roll did not.
        out = merge_tiles(
            out, self.tile, shift if self.score.shift_mode == "pad" else (0, 0), x_size
        )

        if self.score.shift_mode == "roll" and self.shifted:
            out = unshift_cyclic(out, shift)
        return out.reshape(b, h * w, out_dim)


class AnchorStripeAttention(nn.Module):
    """Global attention between stripe tiles and the anchor tiles covering them.

    The anchor queries the stripe, then the stripe queries the anchor, whose values
    are the first pass's output. That two-resolution round trip is what lets a token
    gather information from far outside its own window.

    No mask is needed on the pad backend: the stripe and the anchor are padded by half
    of *their own* tile and partitioned from the padded origin, and because the anchor
    tile is exactly ``stripe / df`` both grids end up with the same tile count, so
    every stripe tile stays paired with the anchor tile that covers it. A count
    mismatch would mean silent misalignment, so it is asserted.
    """

    def __init__(
        self,
        stripe_tile: tuple[int, int],
        num_heads: int,
        shifted: bool,
        score: ScoreFunction,
        down_factor: int,
        orientation: Literal["h", "v"],
    ) -> None:
        super().__init__()
        self.stripe_tile = stripe_tile
        self.anchor_tile = (
            stripe_tile[0] // down_factor,
            stripe_tile[1] // down_factor,
        )
        self.num_heads = num_heads
        self.shifted = shifted
        self.score = score
        self.down_factor = down_factor
        self.orientation = orientation

    def forward(
        self,
        qkv: Tensor,
        anchor: Tensor,
        x_size: tuple[int, int],
        tables: GRLTables | None,
    ) -> Tensor:
        h, w = x_size
        b, _, c = qkv.shape
        out_dim = c // 3
        th, tw = self.stripe_tile
        shifted = self.shifted and self.score.shift_mode == "roll"

        stripe_shift = (th // 2, tw // 2) if self.shifted else (0, 0)
        anchor_shift = (
            stripe_shift[0] // self.down_factor,
            stripe_shift[1] // self.down_factor,
        )
        mode = self.score.shift_mode

        x = _offset(qkv.view(b, h, w, c), stripe_shift, mode)
        anchor = _offset(anchor, anchor_shift, mode)
        stripe_tiles = partition(x, self.stripe_tile)
        anchor_tiles = partition(anchor, self.anchor_tile)
        if mode == "pad" and self.shifted:
            # Both grids grew by exactly one tile, which is what keeps them paired.
            assert stripe_tiles.shape[0] == anchor_tiles.shape[0], (
                "stripe and anchor partitions must yield the same number of tiles: "
                f"{stripe_tiles.shape[0]} vs {anchor_tiles.shape[0]}"
            )

        q, k, v = split_qkv_heads(
            stripe_tiles.reshape(-1, math.prod(self.stripe_tile), c), self.num_heads
        )
        a = project_to_heads(
            anchor_tiles.reshape(-1, math.prod(self.anchor_tile), out_dim),
            self.num_heads,
        )

        bias = CrossBias(self.stripe_tile, self.anchor_tile)
        if tables is not None:
            table, index, mask = tables.stripe(self.orientation, shifted)
            bias = CrossBias(self.stripe_tile, self.anchor_tile, table, index, mask)

        # a2w: the anchor queries the stripe.
        x_anchor = self.score.cross(a, k, v, bias, 0)
        # w2a: the stripe queries the anchor, whose values are the a2w output.
        out = self.score.cross(q, a, x_anchor, bias, 1)
        out = merge_heads(out).reshape(-1, *self.stripe_tile, out_dim)
        out = merge_tiles(
            out,
            self.stripe_tile,
            stripe_shift if mode == "pad" else (0, 0),
            x_size,
        )

        if shifted:
            out = unshift_cyclic(out, stripe_shift)
        return out.reshape(b, h * w, out_dim)


# ---------------------------------------------------------------------------
# shared small pieces
# ---------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    def __init__(self, num_feat: int, reduction: int = 16) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    """GRL's Local branch. Takes tokens-first input plus ``x_size``."""

    def __init__(
        self, num_feat: int, compress_ratio: int = 4, reduction: int = 18
    ) -> None:
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, reduction),
        )

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        return bchw_to_blc(self.cab(blc_to_bchw(x, x_size).contiguous()))


class _PoolAnchor(nn.Module):
    """GRL's anchor projection: average-pool down, then project channels."""

    def __init__(self, in_channels: int, down_factor: int) -> None:
        super().__init__()
        self.pooling = nn.AvgPool2d(down_factor, down_factor)
        self.reduction = nn.Linear(in_channels, in_channels // 2)

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        x = self.pooling(blc_to_bchw(x, x_size))
        return self.reduction(bchw_to_blc(x))


class _ConvAnchor(nn.Module):
    """Learned alternative to the pool: depthwise 3x3 stride ``df``, then 1x1.

    Lets the anchor aggregate a 3x3 neighbourhood instead of a single averaged
    sample, at the cost of depthwise weights.
    """

    def __init__(self, in_channels: int, down_factor: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=down_factor,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, in_channels // 2, 1)

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        return bchw_to_blc(self.pointwise(self.depthwise(blc_to_bchw(x, x_size))))


class AnchorProjection(nn.Module):
    """Downsamples to the anchor grid.

    ``pool`` is GRL-B's avg-pool + linear and keeps GRL's ``body.0.reduction``
    naming, so a ``grl_b`` checkpoint loads without key surgery. ``conv`` is the
    learned variant, whose parameter names deliberately differ -- a checkpoint from
    one is not loadable into the other.
    """

    def __init__(
        self,
        dim: int,
        down_factor: int,
        proj_type: Literal["pool", "conv"] = "pool",
    ) -> None:
        super().__init__()
        self.down_factor = down_factor
        body: nn.Module
        if proj_type == "pool":
            body = _PoolAnchor(dim, down_factor)
        elif proj_type == "conv":
            body = _ConvAnchor(dim, down_factor)
        else:
            raise ValueError(f"Unsupported anchor projection type: {proj_type}")
        self.body = nn.ModuleList([body])

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        x = self.body[0](x, x_size)
        return blc_to_bhwc(x, [s // self.down_factor for s in x_size])


class MixedAttention(nn.Module):
    """One qkv projection feeding both attention patterns, plus the anchor."""

    def __init__(
        self,
        dim: int,
        num_heads_window: int,
        num_heads_stripe: int,
        window_tile: tuple[int, int],
        stripe_tile: tuple[int, int],
        window_shifted: bool,
        stripe_shifted: bool,
        stripe_orientation: Literal["h", "v"],
        anchor_down_factor: int,
        anchor_proj: Literal["pool", "conv"],
        qkv_bias: bool,
        attn_drop: float,
        backend: Literal["grl", "rib"],
        rank: int,
        rib_hidden_dim: int,
        rib_n_freqs: int,
    ) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.anchor = AnchorProjection(dim, anchor_down_factor, anchor_proj)

        def score(num_heads: int, num_directions: int) -> ScoreFunction:
            if backend == "rib":
                return RIBRank(
                    num_heads,
                    num_directions,
                    attn_drop,
                    rank,
                    rib_hidden_dim,
                    rib_n_freqs,
                )
            return GRLCosine(num_heads, num_directions, attn_drop)

        self.window_attn = WindowAttention(
            window_tile, num_heads_window, window_shifted, score(num_heads_window, 1)
        )
        self.stripe_attn = AnchorStripeAttention(
            stripe_tile,
            num_heads_stripe,
            stripe_shifted,
            score(num_heads_stripe, 2),
            anchor_down_factor,
            stripe_orientation,
        )
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: Tensor, x_size: tuple[int, int], tables: GRLTables | None
    ) -> Tensor:
        _, _, c = x.shape
        qkv = self.qkv(x)
        qkv_window, qkv_stripe = torch.split(qkv, c * 3 // 2, dim=-1)
        anchor = self.anchor(x, x_size)

        x_window = self.window_attn(qkv_window, x_size, tables)
        x_stripe = self.stripe_attn(qkv_stripe, anchor, x_size, tables)
        return self.proj(torch.cat([x_window, x_stripe], dim=-1))


def build_last_conv(conv_type: str, dim: int) -> nn.Module:
    if conv_type == "1conv":
        return nn.Conv2d(dim, dim, 3, 1, 1)
    if conv_type == "3conv":
        return nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim // 4, dim, 3, 1, 1),
        )
    if conv_type == "1conv1x1":
        return nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim // 4, dim, 1, 1, 0),
        )
    if conv_type == "linear":
        return nn.Conv2d(dim, dim, 1, 1, 0)
    raise ValueError(f"Unknown conv_type: {conv_type}")


def build_tail(
    upsampler: str, upscale: int, embed_dim: int, out_channels: int
) -> nn.Module:
    return ReconstructionTail(upsampler, upscale, embed_dim, out_channels)


class ReconstructionTail(nn.Module):
    """Upsample the deep features and project to the output channels.

    Assembled as a module so :meth:`BOLT.forward` has exactly one tail path, and with
    GRL's submodule names so a ``grl_b`` checkpoint maps across untouched.
    """

    def __init__(
        self, upsampler: str, upscale: int, embed_dim: int, out_channels: int
    ) -> None:
        super().__init__()
        num_feat = embed_dim
        if upsampler == "pixelshuffle":
            self.conv_before_upsample: nn.Module = nn.Sequential(
                nn.Conv2d(embed_dim, NUM_OUT_FEATS, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            num_feat = NUM_OUT_FEATS
            self.upsample: nn.Module = Upsample(upscale, num_feat)
        elif upsampler == "pixelshuffledirect":
            self.conv_before_upsample = nn.Identity()
            self.upsample = Upsample(upscale, num_feat)
        elif upsampler == "nearest":
            self.conv_before_upsample = nn.Identity()
            self.upsample = nn.Upsample(scale_factor=upscale, mode="nearest")
        else:
            raise ValueError(f"Unknown upsampler: {upsampler}")
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_last(self.upsample(self.conv_before_upsample(x)))


# ---------------------------------------------------------------------------
# block and stage
# ---------------------------------------------------------------------------


class BOLTBlock(nn.Module):
    """One transformer block with an attention branch, a local conv, and an MLP.

    Two residual layouts, because they are genuinely different math: pre-norm i-LN
    rescaled by the input's std (BOLT's default) and GRL's original post-norm LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads_window: int,
        num_heads_stripe: int,
        mlp_ratio: float,
        qkv_bias: bool,
        local_connection: bool,
        res_scale: float,
        drop_path: float,
        drop: float,
        norm_type: Literal["ln", "iln"],
        window_tile: tuple[int, int],
        stripe_tile: tuple[int, int],
        block_index: int,
        stripe_shift_enabled: bool,
        anchor_down_factor: int,
        anchor_proj: Literal["pool", "conv"],
        attn_drop: float,
        backend: Literal["grl", "rib"],
        rank: int,
        rib_hidden_dim: int,
        rib_n_freqs: int,
        fused_conv: bool = False,
        fused_expansion: int = 2,
        conv_ffn: bool = False,
        spatial_gate: bool = False,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.local_connection = local_connection
        self.use_iln = norm_type == "iln"
        self.use_fused_conv = fused_conv
        self.use_conv_ffn = conv_ffn
        self.use_spatial_gate = spatial_gate

        norm_cls: type[nn.Module] = iLN if self.use_iln else nn.LayerNorm
        self.norm1: nn.Module = norm_cls(dim)
        self.norm2: nn.Module = norm_cls(dim)

        # GRL's shift schedule: alternate the window grid every block, and shift the
        # stripes for two blocks out of every four. Half the blocks therefore use
        # horizontal stripes and half vertical, so the two axes both get long-range
        # mixing over the stack.
        self.use_attention = use_attention
        self.attn: MixedAttention | None = None
        if use_attention:
            self.attn = MixedAttention(
                dim=dim,
                num_heads_window=num_heads_window,
                num_heads_stripe=num_heads_stripe,
                window_tile=window_tile,
                stripe_tile=tuple(reversed(stripe_tile))
                if block_index % 2 == 1
                else tuple(stripe_tile),
                window_shifted=block_index % 2 == 0,
                stripe_shifted=stripe_shift_enabled and block_index % 4 in (2, 3),
                stripe_orientation="h" if block_index % 2 == 0 else "v",
                anchor_down_factor=anchor_down_factor,
                anchor_proj=anchor_proj,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                backend=backend,
                rank=rank,
                rib_hidden_dim=rib_hidden_dim,
                rib_n_freqs=rib_n_freqs,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp: nn.Module = (
            ConvSwiGLUFFN(dim, mlp_ratio, drop)
            if conv_ffn
            else Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        )
        if local_connection:
            self.conv = CAB(dim)
        if fused_conv:
            self.fused_conv = FusedConv(dim, fused_expansion)
        if spatial_gate:
            self.spatial_gate = SpatialGate(dim)

    def _residual(self, x: Tensor, sublayer: Callable[[Tensor], Tensor], norm: nn.Module) -> Tensor:
        """One residual branch, in whichever of the two placements is configured.

        The residual itself is the same either way; only which side of the branch the
        norm sits on, and whether the branch is rescaled, differ.
        """
        if self.use_iln:
            x_norm, std = norm(x)
            return x + self.res_scale * std * self.drop_path(sublayer(x_norm))
        return x + self.res_scale * self.drop_path(norm(sublayer(x)))

    def forward(
        self, x: Tensor, x_size: tuple[int, int], tables: GRLTables | None
    ) -> Tensor:
        # GRL taps the local conv and the fused conv off the block *input*,
        # not off the post-attention x, so both are read before the
        # residual rewrites x.
        conv = self.conv(x, x_size) if self.local_connection else None
        fused = (
            self.fused_conv(x, x_size) if self.use_fused_conv else None
        )

        if self.attn is not None:

            def attn_branch(t: Tensor) -> Tensor:
                attended = self.attn(t, x_size, tables)  # pyright: ignore[reportOptionalMemberAccess]
                if self.use_spatial_gate:
                    attended = attended * self.spatial_gate(t, x_size)
                return attended

            x = self._residual(x, attn_branch, self.norm1)
        if conv is not None:
            x = x + conv
        if fused is not None:
            x = x + fused
        mlp_fn: Callable[[Tensor], Tensor] = (
            (lambda t: self.mlp(t, x_size))
            if self.use_conv_ffn
            else self.mlp
        )
        return self._residual(x, mlp_fn, self.norm2)


# ---------------------------------------------------------------------------
# dynamic convolution (premixed expert kernels, input-routed)
# ---------------------------------------------------------------------------


class DynamicRouter(nn.Module):
    """Per-image GAP -> LayerNorm -> MLP routing, centred around 1/K.

    The bias is zero-initialised so the softmax starts uniform (1/K each)
    and the centred weights start at zero, preserving the base conv's
    initial function exactly.
    """

    def __init__(self, channels: int, experts: int) -> None:
        super().__init__()
        if channels < 4 or experts < 2:
            raise ValueError("dynamic conv needs channels >= 4 and experts >= 2")
        self.channels, self.num_experts = channels, experts
        self.fc1 = nn.Linear(channels, channels // 4)
        self.fc2 = nn.Linear(channels // 4, experts)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean((2, 3))
        normalized = F.layer_norm(pooled, (self.channels,))
        hidden = F.gelu(F.linear(normalized, self.fc1.weight, self.fc1.bias))
        logits = F.linear(hidden, self.fc2.weight, self.fc2.bias)
        return (logits.softmax(-1) - 1.0 / self.num_experts)[:, :, None, None]


class DynamicConv3x3(nn.Module):
    """Base 3x3 conv plus K learnable, input-routed residual kernels.

    Kernels are mixed per image BEFORE convolution (one group conv, not
    K convs), so the cost is one conv plus a tiny router. At export the
    router stays input-dependent; the base and expert kernels compose into
    a single effective kernel per sample.
    """

    def __init__(self, dim: int, experts: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.dim, self.experts = dim, experts
        self.expert_kernels = nn.Parameter(torch.zeros(experts, dim, dim * 9))
        self.router = DynamicRouter(dim, experts)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        coeff = self.router(x).reshape(batch, self.experts)
        residual = coeff @ self.expert_kernels.reshape(self.experts, -1)
        weight = (
            self.conv.weight.reshape(1, -1) + residual
        ).reshape(batch * channels, channels, 3, 3)
        bias = (
            self.conv.bias.repeat(batch)
            if self.conv.bias is not None
            else None
        )
        return F.conv2d(
            x.reshape(1, batch * channels, height, width),
            weight,
            bias,
            padding=1,
            groups=batch,
        ).reshape(batch, channels, height, width)


class ConvSwiGLUFFN(nn.Module):
    """Convolutional SwiGLU feed-forward network (DRFT-style).

    Fused gate+value projection, SwiGLU gating, a depthwise conv for
    locality, then an output projection. Replaces the plain MLP.
    Operates on BLC tokens.
    """

    def __init__(
        self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.0
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        hidden = ((hidden + 7) // 8) * 8  # multiple of 8 for tensor cores
        self.fc_gate_value = nn.Linear(dim, hidden * 2)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.fc_out = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        gate, value = self.fc_gate_value(x).chunk(2, dim=-1)
        h = F.silu(gate) * value
        batch, tokens, channels = h.shape
        height, width = x_size
        h = h.transpose(1, 2).reshape(batch, channels, height, width)
        h = self.dwconv(h)
        h = h.flatten(2).transpose(1, 2)
        return self.drop(self.fc_out(h))


class FusedConv(nn.Module):
    """Inverted-bottleneck conv with squeeze-excitation (HMA's Fused Conv).

    expand (1x1) -> spatial (3x3) -> SE -> reduce (1x1), placed before
    attention and tapped off the block input. Operates on BLC tokens.
    """

    def __init__(
        self, dim: int, expansion: int = 2, shrink: float = 0.5
    ) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.expand = nn.Conv2d(dim, hidden, 1)
        self.spatial = nn.Conv2d(hidden, hidden, 3, 1, 1)
        reduced = max(1, int(hidden * shrink))
        self.se_reduce = nn.Conv2d(hidden, reduced, 1)
        self.se_expand = nn.Conv2d(reduced, hidden, 1)
        self.reduce = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        batch, tokens, channels = x.shape
        height, width = x_size
        h = x.transpose(1, 2).reshape(batch, channels, height, width)
        h = F.gelu(self.spatial(self.expand(h)))
        se = F.adaptive_avg_pool2d(h, 1)
        se = self.se_expand(F.gelu(self.se_reduce(se))).sigmoid()
        h = self.reduce(h * se)
        return h.flatten(2).transpose(1, 2)


class SpatialGate(nn.Module):
    """Spatial gate: depthwise + pointwise + sigmoid (DUAL-style).

    Returns a per-location gate in [0, 1] shaped like the input tokens,
    computed from the attention input; the attention output is multiplied
    by it. Operates on BLC tokens.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        batch, tokens, channels = x.shape
        height, width = x_size
        h = x.transpose(1, 2).reshape(batch, channels, height, width)
        gate = self.pointwise(self.depthwise(h)).sigmoid()
        return gate.flatten(2).transpose(1, 2)


class TransformerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads_window: int,
        num_heads_stripe: int,
        window_tile: tuple[int, int],
        stripe_tile: tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        anchor_down_factor: int,
        anchor_proj: Literal["pool", "conv"],
        attn_drop: float,
        drop: float,
        drop_path: Sequence[float],
        local_connection: bool,
        res_scale: float,
        norm_type: Literal["ln", "iln"],
        backend: Literal["grl", "rib"],
        rank: int,
        rib_hidden_dim: int,
        rib_n_freqs: int,
        conv_type: str,
        stripe_shift: bool,
        use_checkpoint: bool,
        dynamic_conv: bool = False,
        dynamic_experts: int = 4,
        fused_conv: bool = False,
        fused_expansion: int = 2,
        conv_ffn: bool = False,
        spatial_gate: bool = False,
        attn_freq: int = 1,
        block_offset: int = 0,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            BOLTBlock(
                dim=dim,
                num_heads_window=num_heads_window,
                num_heads_stripe=num_heads_stripe,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                local_connection=local_connection,
                res_scale=res_scale,
                drop_path=drop_path[i],
                drop=drop,
                norm_type=norm_type,
                window_tile=window_tile,
                stripe_tile=stripe_tile,
                block_index=i,
                stripe_shift_enabled=stripe_shift,
                anchor_down_factor=anchor_down_factor,
                anchor_proj=anchor_proj,
                attn_drop=attn_drop,
                backend=backend,
                rank=rank,
                rib_hidden_dim=rib_hidden_dim,
                rib_n_freqs=rib_n_freqs,
                fused_conv=fused_conv,
                fused_expansion=fused_expansion,
                conv_ffn=conv_ffn,
                spatial_gate=spatial_gate,
                use_attention=(block_offset + i) % attn_freq == 0,
            )
            for i in range(depth)
        )
        if dynamic_conv:
            self.conv = DynamicConv3x3(dim, dynamic_experts)
        else:
            self.conv = build_last_conv(conv_type, dim)

    def forward(
        self, x: Tensor, x_size: tuple[int, int], tables: GRLTables | None
    ) -> Tensor:
        res = x
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                res = checkpoint.checkpoint(
                    blk, res, x_size, tables, use_reentrant=False
                )
            else:
                res = blk(res, x_size, tables)
        return bchw_to_blc(self.conv(blc_to_bchw(res, x_size))) + x


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------


def _apply_norm(norm: nn.Module, x: Tensor) -> Tensor:
    """i-LN returns the normalized tokens and their std; LayerNorm returns tokens."""
    if isinstance(norm, iLN):
        x, _ = norm(x)
        return x
    return norm(x)


@store_hyperparameters()
class BOLT(nn.Module):
    """BOLT: GRL-B's hierarchy with a selectable attention backend.

    Requires ``img_size`` to be a multiple of ``pad_size``, which is the max of the
    window size, the largest stripe size, and ``max(stripe_groups) *
    anchor_window_down_factor``. At the GRL-B settings this is 64, so ``img_size``
    must be 64, 128, ... — a crop of 96 would be silently padded at every batch and
    teach the model false border statistics.
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        upscale: int = 4,
        img_size: int = 64,
        in_channels: int = 3,
        out_channels: int | None = None,
        embed_dim: int = 180,
        img_range: float = 1.0,
        upsampler: str = "pixelshuffle",
        depths: Sequence[int] = (4, 4, 8, 8, 8, 4, 4),
        num_heads_window: Sequence[int] = (3, 3, 3, 3, 3, 3, 3),
        num_heads_stripe: Sequence[int] = (3, 3, 3, 3, 3, 3, 3),
        window_size: int = 32,
        stripe_size: Sequence[int] = (64, 64),
        stripe_groups: Sequence[int | None] = (None, None),
        stripe_shift: bool = True,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        anchor_window_down_factor: int = 4,
        anchor_proj: Literal["pool", "conv"] = "pool",
        local_connection: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        conv_type: str = "1conv",
        attn_backend: Literal["grl", "rib"] = "rib",
        norm_type: Literal["ln", "iln"] = "iln",
        res_scale: float = 1.0,
        rank: int = 8,
        rib_hidden_dim: int = 32,
        rib_n_freqs: int = 10,
        use_checkpoint: bool = False,
        dynamic_conv: bool = False,
        dynamic_experts: int = 4,
        fused_conv: bool = False,
        fused_expansion: int = 2,
        conv_ffn: bool = False,
        spatial_gate: bool = False,
        attn_freq: int = 1,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale = upscale
        self.img_range = img_range
        self.use_checkpoint = use_checkpoint

        if len(num_heads_window) != len(depths) or len(num_heads_stripe) != len(
            depths
        ):
            raise ValueError("num_heads_window/num_heads_stripe must match depths")
        if embed_dim % 2 != 0:
            # The window and stripe branches each get qkv width 3*dim//2, i.e.
            # dim//2 per projection, and the anchor has dim//2 channels. An odd
            # dim would break that correspondence.
            raise ValueError("embed_dim must be even")
        # Each attention branch gets embed_dim // 2 channels, so that must divide by
        # the head count or the head split is not exact.
        for name, heads in (
            ("num_heads_window", num_heads_window),
            ("num_heads_stripe", num_heads_stripe),
        ):
            if any((embed_dim // 2) % h != 0 for h in heads):
                raise ValueError(f"embed_dim // 2 must be divisible by {name}")

        window_tile = (window_size, window_size)
        self.window_size = window_tile
        self.img_size = img_size
        input_resolution = (img_size, img_size)

        # pad_size is what the tiling can actually resolve. A grouped stripe is
        # derived from the input size, so it never constrains it.
        grouped = all(g is not None for g in stripe_groups)
        max_stripe_size = 0 if grouped else max(s for s in stripe_size if s is not None)
        max_stripe_groups = max(
            (0 if g is None else g) * anchor_window_down_factor for g in stripe_groups
        )
        self.pad_size = max(window_size, max_stripe_size, max_stripe_groups)
        if img_size % self.pad_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be a multiple of pad_size"
                f" ({self.pad_size})"
            )

        rgb_mean = (0.4488, 0.4371, 0.4040) if in_channels == 3 else (0.0,) * in_channels
        self.register_buffer(
            "mean",
            torch.tensor(rgb_mean, dtype=torch.float32).view(1, in_channels, 1, 1),
            persistent=False,
        )

        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        norm_cls: type[nn.Module] = iLN if norm_type == "iln" else nn.LayerNorm
        self.norm_start: nn.Module = norm_cls(embed_dim)
        self.norm_end: nn.Module = norm_cls(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Resolve the stripe geometry once; the vertical orientation is the transpose
        # of the horizontal one, and the GRL tables need both.
        stripe_t, stripe_shift_size = get_stripe_info(
            stripe_size, stripe_groups, stripe_shift, input_resolution
        )
        stripe_tile = (int(stripe_t[0]), int(stripe_t[1]))
        shift = (int(stripe_shift_size[0]), int(stripe_shift_size[1]))
        self.stripe_tile = stripe_tile

        # The tables are only needed by the GRL backend, and the stripe index
        # buffers are megabytes, so the RIB backend never builds them.
        self.grl_tables: GRLTables | None = None
        if attn_backend == "grl":
            self.grl_tables = GRLTables(
                input_resolution=input_resolution,
                window_tile=window_tile,
                window_shift=(window_size // 2, window_size // 2),
                stripe_tiles={"h": stripe_tile, "v": tuple(reversed(stripe_tile))},
                stripe_shifts={"h": shift, "v": tuple(reversed(shift))},
                down_factor=anchor_window_down_factor,
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList(
            TransformerStage(
                dim=embed_dim,
                depth=depth,
                num_heads_window=num_heads_window[i],
                num_heads_stripe=num_heads_stripe[i],
                window_tile=window_tile,
                stripe_tile=stripe_tile,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                anchor_down_factor=anchor_window_down_factor,
                anchor_proj=anchor_proj,
                attn_drop=attn_drop_rate,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                local_connection=local_connection,
                res_scale=res_scale,
                norm_type=norm_type,
                backend=attn_backend,
                rank=rank,
                rib_hidden_dim=rib_hidden_dim,
                rib_n_freqs=rib_n_freqs,
                conv_type=conv_type,
                stripe_shift=stripe_shift,
                use_checkpoint=use_checkpoint,
                dynamic_conv=dynamic_conv,
                dynamic_experts=dynamic_experts,
                fused_conv=fused_conv,
                fused_expansion=fused_expansion,
                conv_ffn=conv_ffn,
                spatial_gate=spatial_gate,
                attn_freq=attn_freq,
                block_offset=sum(depths[:i]),
            )
            for i, depth in enumerate(depths)
        )

        self.conv_after_body = build_last_conv(conv_type, embed_dim)
        self.tail = build_tail(upsampler, upscale, embed_dim, out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        pad_h = (self.pad_size - h % self.pad_size) % self.pad_size
        pad_w = (self.pad_size - w % self.pad_size) % self.pad_size
        if pad_h == 0 and pad_w == 0:
            return x
        can_reflect = (pad_h == 0 or pad_h < h) and (pad_w == 0 or pad_w < w)
        return F.pad(
            x, (0, pad_w, 0, pad_h), mode="reflect" if can_reflect else "replicate"
        )

    def forward_features(
        self, x: Tensor, return_features: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        x_size = (x.shape[2], x.shape[3])
        x = bchw_to_blc(x)
        x = _apply_norm(self.norm_start, x)
        x = self.pos_drop(x)

        tables = self.grl_tables
        feats: list[Tensor] = []
        for layer in self.layers:
            x = layer(x, x_size, tables)
            if return_features:
                # one LR-res feature map per stage, embed_dim channels
                feats.append(blc_to_bchw(x, x_size))

        x = _apply_norm(self.norm_end, x)
        out = blc_to_bchw(x, x_size)
        if return_features:
            return out, feats
        return out

    def forward(
        self, x: Tensor, return_features: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        h, w = x.shape[2:]
        x = self.check_image_size(x)
        x = (x - self.mean) * self.img_range

        feat = self.conv_first(x)
        if return_features:
            body, stage_feats = self.forward_features(feat, return_features=True)
            feat = self.conv_after_body(body) + feat
            x = self.tail(feat)
            x = x / self.img_range + self.mean
            return (
                x[:, :, : h * self.upscale, : w * self.upscale],
                stage_feats,
            )
        feat = self.conv_after_body(self.forward_features(feat)) + feat
        x = self.tail(feat)

        x = x / self.img_range + self.mean
        return x[:, :, : h * self.upscale, : w * self.upscale]


# ---------------------------------------------------------------------------
# grl_b -> BOLT weight remapping
# ---------------------------------------------------------------------------

_GRB_ONLY_MARKERS = ("cpb_mlp",)
_RIB_ONLY_MARKERS = (
    "to_hidden",
    "hidden_b",
    "pos.to_q",
    "pos.to_k",
    "pos_logit_scale",
)

# GRL nests the pixel-shuffle convolutions under ``upsample.up``; BOLT reuses HAT's
# ``Upsample``, which is a flat Sequential. Same math, different prefix.
_UPSAMPLE_PREFIX = "upsample.up."

# GRL scatters the reconstruction head over the top level; BOLT keeps it in one
# ``tail`` module so the forward pass has a single path. Same parameters, one level
# deeper.
_TAIL_MODULES = ("upsample.", "conv_before_upsample.", "conv_last.")

# GRL's QKV projection carries a ``body`` wrapper that holds no parameters; BOLT's
# is the Linear itself.
_QKV_RENAMES = (("attn.qkv.body.", "attn.qkv."),)

# GRL's per-direction AffineTransform: one for the window, two for the stripes. BOLT
# holds them in one ModuleList, and direction 0 is anchor->stripe (GRL's
# attn_transform1) while direction 1 is stripe->anchor (attn_transform2).
_ATTN_TRANSFORM_RE = re.compile(r"attn_transform(1|2)?\.")
_LOGIT_SCALE_RE = re.compile(r"score\.transforms\.[01]\.logit_scale$")


def _remap_attn_transform(key: str) -> tuple[str, int | None]:
    match = _ATTN_TRANSFORM_RE.search(key)
    if match is None:
        return key, None
    direction = 0 if match.group(1) in (None, "1") else 1
    start, end = match.span()
    return f"{key[:start]}score.transforms.{direction}.{key[end:]}", direction


def remap_grl_state_dict(
    state_dict: dict[str, Tensor], attn_backend: Literal["grl", "rib"] = "rib"
) -> dict[str, Tensor]:
    """Map a ``grl_b`` state dict onto BOLT's parameter names.

    BOLT mirrors GRL's module tree closely, so almost nothing needs renaming. What
    does is handled here:

    * the CPB MLP exists only in the ``grl`` backend, and RIB's position
      projections only in the ``rib`` backend, so those tensors are dropped;
    * ``attn_transform``/``attn_transform1``/``attn_transform2`` become
      ``score.transforms.0``/``.0``/``.1``;
    * in the ``rib`` backend ``logit_scale`` leaves the transform and joins the
      stacked per-direction parameter;
    * the shared table/index/mask buffers are dropped, because BOLT rebuilds them
      from its own configuration;
    * ``upsample.up.*`` becomes ``upsample.*``.

    Everything else — qkv, anchor projection, CAB, MLP, projections, and the
    convolutional tail — maps across unchanged, which is what makes a remapped
    ``grl_b`` checkpoint a controlled starting point for the ``rib`` ablation.
    """
    drop_markers = _GRB_ONLY_MARKERS if attn_backend == "rib" else _RIB_ONLY_MARKERS

    out: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if any(
            m in key
            for m in ("table_w", "table_sh", "table_sv", "index_", "mask_", "std_ema")
        ):
            continue
        if any(marker in key for marker in drop_markers):
            continue

        if key.startswith(_UPSAMPLE_PREFIX):
            key = "upsample." + key[len(_UPSAMPLE_PREFIX) :]
        for old, new in _QKV_RENAMES:
            key = key.replace(old, new)
        for module in _TAIL_MODULES:
            if key.startswith(module):
                key = f"tail.{key}"
                break

        key, direction = _remap_attn_transform(key)
        if (
            attn_backend == "rib"
            and direction is not None
            and _LOGIT_SCALE_RE.search(key)
        ):
            # RIB's stacked parameter lives on the score function, not per transform.
            key = _LOGIT_SCALE_RE.sub(f"score.logit_scale.{direction}", key)

        out[key] = value
    return out


# ---------------------------------------------------------------------------
# registered entry points
# ---------------------------------------------------------------------------


@ARCH_REGISTRY.register()
def bolt(
    *,
    scale: int = 4,
    img_size: int = 64,
    in_channels: int = 3,
    out_channels: int = 3,
    embed_dim: int = 180,
    img_range: float = 1.0,
    upsampler: str = "pixelshuffle",
    depths: Sequence[int] = (4, 4, 8, 8, 8, 4, 4),
    num_heads_window: Sequence[int] = (3, 3, 3, 3, 3, 3, 3),
    num_heads_stripe: Sequence[int] = (3, 3, 3, 3, 3, 3, 3),
    window_size: int = 32,
    stripe_size: Sequence[int] = (64, 64),
    stripe_groups: Sequence[int | None] = (None, None),
    stripe_shift: bool = True,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    anchor_window_down_factor: int = 4,
    anchor_proj: Literal["pool", "conv"] = "pool",
    local_connection: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    conv_type: str = "1conv",
    attn_backend: Literal["grl", "rib"] = "rib",
    norm_type: Literal["ln", "iln"] = "iln",
    res_scale: float = 1.0,
    rank: int = 8,
    rib_hidden_dim: int = 32,
    rib_n_freqs: int = 10,
    use_checkpoint: bool = False,
    dynamic_conv: bool = False,
    dynamic_experts: int = 4,
    fused_conv: bool = False,
    fused_expansion: int = 2,
    conv_ffn: bool = False,
    spatial_gate: bool = False,
    attn_freq: int = 1,
) -> BOLT:
    return BOLT(
        upscale=scale,
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        img_range=img_range,
        upsampler=upsampler,
        depths=depths,
        num_heads_window=num_heads_window,
        num_heads_stripe=num_heads_stripe,
        window_size=window_size,
        stripe_size=stripe_size,
        stripe_groups=stripe_groups,
        stripe_shift=stripe_shift,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        anchor_window_down_factor=anchor_window_down_factor,
        anchor_proj=anchor_proj,
        local_connection=local_connection,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        conv_type=conv_type,
        attn_backend=attn_backend,
        norm_type=norm_type,
        res_scale=res_scale,
        rank=rank,
        rib_hidden_dim=rib_hidden_dim,
        rib_n_freqs=rib_n_freqs,
        use_checkpoint=use_checkpoint,
        dynamic_conv=dynamic_conv,
        dynamic_experts=dynamic_experts,
        fused_conv=fused_conv,
        fused_expansion=fused_expansion,
        conv_ffn=conv_ffn,
        spatial_gate=spatial_gate,
        attn_freq=attn_freq,
    )


def _register_variant(name: str, **overrides: object) -> None:
    """Register a BOLT variant that differs from :func:`bolt` only in ``overrides``.

    The variant reuses ``bolt``'s signature so introspection -- config schema
    generation, docs, editor completion -- still sees every parameter, with the
    variant's own defaults substituted. That is what keeps 30-odd parameters from
    being copy-pasted per size, which is how they silently drift apart.
    """
    signature = inspect.signature(bolt)
    params = [
        p.replace(default=overrides[p.name]) if p.name in overrides else p
        for p in signature.parameters.values()
    ]

    def build(**kwargs: object) -> BOLT:
        return BOLT(**{**kwargs, **overrides})  # pyright: ignore[reportArgumentType]

    build.__name__ = name
    build.__qualname__ = name
    build.__doc__ = f"BOLT variant: {name}. Overrides: {sorted(overrides)}."
    build.__signature__ = signature.replace(parameters=params)  # pyright: ignore[reportFunctionMemberAccess]
    ARCH_REGISTRY.register(build)
    SPANDREL_REGISTRY.register(build)


_register_variant("bolt_l", depths=(4, 4, 8, 8, 12, 8, 4))
