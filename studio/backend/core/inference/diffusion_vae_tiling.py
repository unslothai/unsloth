# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seam-free tiled decode for the Qwen-Image-2.1 VAE.

diffusers gives AutoencoderKLQwenImage21 the Wan VAE's tile geometry in PIXELS (256 px tiles, 192 px stride),
but this VAE compresses 16x, not 8x: a tile is 16 latents with a 4-latent (64 px) blend, and the last row /
column is a 4-latent sliver. The decoder sees further than 4 latents past a tile edge, so every seam and the
sliver at the right / bottom edge decode differently from their neighbours and the short linear blend leaves
thin vertical / horizontal lines, one tile long. A 1024x1024 render is 6x6 such tiles, so every low-VRAM tier
that tiles the decode (streaming / whole-model offload) shows them; resident loads decode untiled.

Here a tile is 32 latents with at least a 16-latent overlap (ComfyUI's decode overlap), and the tiles are
spread evenly so the last one ends at the image edge at full size (no sliver). A tile gets no weight within 4
latents of an edge it shares with another tile and ramps to full weight over the next 8, normalised where more
than two tiles meet (any stride under 16 latents, e.g. 1344 or 2400 px, makes three tiles overlap). A canvas that fits one tile decodes
untiled. Only the decode changes; the stock tiled encode is left alone. Kill switch
``UNSLOTH_DIFFUSION_VAE_WIDE_TILES=0``: at load it skips the install (the fused batched tile decode, if any,
installs as before); set later, each decode takes the stock serial tiled decode.
"""

from __future__ import annotations

import os
import types
from typing import Any, Optional

WIDE_TILES_ENV = "UNSLOTH_DIFFUSION_VAE_WIDE_TILES"

TILE_LATENTS = 32
OVERLAP_LATENTS = 16
# Blend: a tile gets no weight within MARGIN latents of an edge it shares, then ramps to full over RAMP latents.
MARGIN_LATENTS = 4
RAMP_LATENTS = 8

_VAE_CLASSES = frozenset({"AutoencoderKLQwenImage21"})


def wide_tiles_disabled() -> bool:
    return (os.environ.get(WIDE_TILES_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def tile_starts(
    length: int,
    tile: int = TILE_LATENTS,
    overlap: int = OVERLAP_LATENTS,
) -> list[int]:
    """Start offsets of the fewest full-size tiles covering ``length`` with every neighbour overlap >= ``overlap``.

    The first tile starts at 0 and the last ends at ``length``; the rest are spread evenly between them."""
    length, tile, overlap = int(length), int(tile), int(overlap)
    if tile <= overlap or overlap < 0:
        raise ValueError(f"tile {tile} must exceed overlap {overlap}")
    if length <= tile:
        return [0]
    step = tile - overlap
    count = -(-(length - overlap) // step)
    span = length - tile
    return [(i * span) // (count - 1) for i in range(count)]


def axis_weights(
    starts: list[int],
    tile: int,
    length: int,
    scale: int,
    torch: Any,
    device: Any,
    margin: int = MARGIN_LATENTS,
    ramp: int = RAMP_LATENTS,
) -> list:
    """Per-tile blend weights (fp32, ``min(tile, length) * scale`` long) along one axis, summing to 1 at every pixel.

    A tile's weight is 0 within ``margin`` latents of an edge it shares with another tile (where its decode lacks
    context), rises linearly over the next ``ramp`` latents, and is 1 deeper in; image borders are not shared edges.
    When three tiles overlap (a stride under 16 latents) the outer tile's edge region still gets no weight. With every
    overlap >= 16 latents, each pixel lies >= 8 latents inside some tile, so a margin under 8 keeps the sum positive;
    with margin 4 and ramp 8 a two-tile overlap of exactly 16 is a linear cross-fade over its middle 8 latents."""
    size = min(tile, length) * scale
    pos = (
        torch.arange(size, dtype = torch.float64, device = device) + 0.5
    ) / scale  # pixel centres, in latents
    total = torch.zeros(length * scale, dtype = torch.float64, device = device)
    weights = []
    for s in starts:
        w = torch.ones(size, dtype = torch.float64, device = device)
        if s > 0:
            w = torch.minimum(w, ((pos - margin) / ramp).clamp(0, 1))
        if s + tile < length:
            w = torch.minimum(w, ((min(tile, length) - pos - margin) / ramp).clamp(0, 1))
        total[s * scale : s * scale + size] += w
        weights.append(w)
    if float(total.min()) <= 0.0:
        raise ValueError(f"tiles {starts} leave pixels without weight on a {length}-latent axis")
    return [(w / total[s * scale : s * scale + size]).float() for w, s in zip(weights, starts)]


def _decode_tile(vae: Any, z: Any) -> Any:
    """The VAE's own untiled decode of one tile (fused / compiled module forwards still apply)."""
    prev = vae.use_tiling
    vae.use_tiling = False
    try:
        return vae._decode(z, return_dict = False)[0]
    finally:
        vae.use_tiling = prev


def tiled_decode(
    vae: Any,
    z: Any,
    return_dict: bool = True,
) -> Any:
    """Decode ``z`` (B, C, T, H, W) in evenly spread 32-latent tiles with 16-latent overlaps."""
    import torch
    from diffusers.models.autoencoders.vae import DecoderOutput

    _, _, _, height, width = z.shape
    ratio = int(vae.spatial_compression_ratio)
    hs = tile_starts(height)
    ws = tile_starts(width)
    th, tw = min(TILE_LATENTS, height), min(TILE_LATENTS, width)
    if len(hs) == 1 and len(ws) == 1:
        dec = _decode_tile(vae, z)
    else:
        wy = axis_weights(hs, TILE_LATENTS, height, ratio, torch, z.device)
        wx = axis_weights(ws, TILE_LATENTS, width, ratio, torch, z.device)
        out, dtype = None, None
        for i, y in enumerate(hs):
            for j, x in enumerate(ws):
                tile = _decode_tile(vae, z[:, :, :, y : y + th, x : x + tw])
                if out is None:
                    dtype = tile.dtype
                    out = torch.zeros(
                        (*tile.shape[:3], height * ratio, width * ratio),
                        dtype = torch.float32,
                        device = tile.device,
                    )
                w = wy[i].view(-1, 1) * wx[j].view(1, -1)
                out[..., y * ratio : (y + th) * ratio, x * ratio : (x + tw) * ratio].add_(
                    tile.float() * w
                )
                del tile
        dec = out.to(dtype)
        del out
    if not return_dict:
        return (dec,)
    return DecoderOutput(sample = dec)


def _wants_wide_tiles(vae: Any) -> bool:
    if vae is None or type(vae).__name__ not in _VAE_CLASSES:
        return False
    config = getattr(vae, "config", None)
    if getattr(config, "patch_size", None) is not None:
        return False
    try:
        ratio = int(vae.spatial_compression_ratio)
    except (AttributeError, TypeError, ValueError):
        return False
    return ratio >= 16 and callable(getattr(vae, "_decode", None)) and hasattr(vae, "use_tiling")


def install(vae: Any, logger: Any = None) -> bool:
    """Route ``vae``'s tiled decode through the wide, edge-aligned tiles. Idempotent; False when not covered.

    Only the decode changes: the stock tiled encode and the VAE's tile attributes stay as diffusers set them.
    ``_unsloth_decode_tile_side`` carries the real decode tile side for Studio's memory estimates."""
    if not _wants_wide_tiles(vae) or wide_tiles_disabled():
        return False
    if getattr(vae, "_unsloth_wide_tiles", False):
        return True
    own = vae.__dict__.get("tiled_decode")
    stock = vae.tiled_decode

    def _tiled_decode(
        self: Any,
        z: Any,
        return_dict: bool = True,
    ) -> Any:
        if wide_tiles_disabled():
            return stock(z, return_dict = return_dict)
        return tiled_decode(self, z, return_dict = return_dict)

    vae._unsloth_wide_tiles_own = own
    vae.tiled_decode = types.MethodType(_tiled_decode, vae)
    vae._unsloth_decode_tile_side = TILE_LATENTS * int(vae.spatial_compression_ratio)
    vae._unsloth_wide_tiles = True
    if logger is not None:
        logger.info(
            "diffusion.vae_tiling: %s decodes in %d-latent tiles with %d-latent overlaps, the last tile edge-aligned",
            type(vae).__name__,
            TILE_LATENTS,
            OVERLAP_LATENTS,
        )
    return True


def uninstall(vae: Any) -> None:
    if not getattr(vae, "_unsloth_wide_tiles", False):
        return
    own: Optional[Any] = vae.__dict__.get("_unsloth_wide_tiles_own")
    if own is None:
        vae.__dict__.pop("tiled_decode", None)
    else:
        vae.tiled_decode = own
    for name in ("_unsloth_wide_tiles_own", "_unsloth_decode_tile_side"):
        vae.__dict__.pop(name, None)
    vae._unsloth_wide_tiles = False
