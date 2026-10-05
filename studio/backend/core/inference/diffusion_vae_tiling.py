# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seam-free tiled decode / encode for the Qwen-Image-2.1 VAE.

diffusers gives AutoencoderKLQwenImage21 the Wan VAE's pixel tile geometry, but this VAE compresses 16x: tiles are 16
latents with 4-latent blends and a sliver last tile, too little context, so tiled tiers draw thin seam lines. Here
tiles are >= 32 latents (larger when free VRAM allows), overlap >= 16, edge-aligned, with a normalised blend that
gives no weight near shared edges. Encode tiles are 64 latents: the encoder attends over the whole tile, so small
tiles shift the condition latent everywhere. Kill switch ``UNSLOTH_DIFFUSION_VAE_WIDE_TILES=0``.
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
# Encode: a 1024 px reference image (Studio's default) is one tile, bit-identical to the untiled encode.
ENCODE_TILE_LATENTS = 64
ENCODE_OVERLAP_LATENTS = 32
ENCODE_MARGIN_LATENTS = 8
ENCODE_RAMP_LATENTS = 16

_VAE_CLASSES = frozenset({"AutoencoderKLQwenImage21"})


def wide_tiles_disabled() -> bool:
    return (os.environ.get(WIDE_TILES_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def tile_starts(
    length: int,
    tile: int = TILE_LATENTS,
    overlap: int = OVERLAP_LATENTS,
) -> list[int]:
    """Starts of the fewest full-size tiles covering ``length`` with overlaps >= ``overlap``, first at 0, last at the end."""
    length, tile, overlap = int(length), int(tile), int(overlap)
    if length <= tile:
        return [0]
    if tile <= overlap or overlap < 0:
        raise ValueError(f"tile {tile} must exceed overlap {overlap}")
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
    """Per-tile fp32 blend weights along one axis, summing to 1 at every pixel: 0 within ``margin`` latents of a
    shared edge, linear over the next ``ramp``. Overlaps >= 16 put every pixel >= 8 latents inside some tile, so a
    margin under 8 keeps the sum positive."""
    size = min(tile, length) * scale
    pos = (torch.arange(size, dtype = torch.float64, device = device) + 0.5) / scale
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
    """The VAE's own untiled decode of one tile (fused / compiled forwards still apply)."""
    prev = vae.use_tiling
    vae.use_tiling = False
    try:
        return vae._decode(z, return_dict = False)[0]
    finally:
        vae.use_tiling = prev


# Unfused bf16 decode peak per latent of tile area (measured 1,697 MiB at 32x32, 6,717 MiB at 64x64).
DECODE_MIB_PER_LATENT = 1.7
FREE_FRACTION = 0.75
MAX_TILE_ENV = "UNSLOTH_DIFFUSION_VAE_MAX_TILE"


def _axis_tiles(length: int, count: int) -> Optional[int]:
    """Smallest tile side (>= 32 latents) that covers ``length`` in ``count`` tiles with 16-latent overlaps."""
    if count == 1:
        return length
    side = max(TILE_LATENTS, -(-(length + (count - 1) * OVERLAP_LATENTS) // count))
    return side if side < length else None


def choose_tiles(height: int, width: int, max_area: Optional[int]) -> tuple[int, int]:
    """Tile (height, width) in latents decoding the fewest latents (overlaps counted) within ``max_area``; 32x32 floor."""
    floor = (min(TILE_LATENTS, height), min(TILE_LATENTS, width))
    if max_area is None or max_area <= floor[0] * floor[1]:
        return floor
    best, best_cost = floor, None
    sides_h = [(n, _axis_tiles(height, n)) for n in range(1, len(tile_starts(height)) + 1)]
    sides_w = [(n, _axis_tiles(width, n)) for n in range(1, len(tile_starts(width)) + 1)]
    for _, th in sides_h:
        for _, tw in sides_w:
            if th is None or tw is None or th * tw > max_area:
                continue
            cost = (len(tile_starts(height, th)) * th * len(tile_starts(width, tw)) * tw, th * tw)
            if best_cost is None or cost < best_cost:
                best, best_cost = (th, tw), cost
    return best


def decode_tile_budget(vae: Any, z: Any) -> Optional[int]:
    """Largest decode tile area (latents) that fits ``FREE_FRACTION`` of the free VRAM now; None off CUDA."""
    raw = (os.environ.get(MAX_TILE_ENV) or "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw) ** 2
    try:
        import torch

        if getattr(z, "device", None) is None or z.device.type != "cuda":
            return None
        free, _ = torch.cuda.mem_get_info(z.device)
        free += torch.cuda.memory_reserved(z.device) - torch.cuda.memory_allocated(z.device)
        ratio = int(vae.spatial_compression_ratio)
        out_bytes = 4 * 4 * z.shape[0] * z.shape[2] * z.shape[-2] * z.shape[-1] * ratio * ratio
        elem = next(vae.decoder.parameters()).element_size()
        mib = FREE_FRACTION * (free - out_bytes) / 2**20
        return max(0, int(mib / (DECODE_MIB_PER_LATENT * elem / 2)))
    except Exception:  # noqa: BLE001 - unknown budget: the 32-latent tiles the planner budgeted
        return None


def tiled_decode(
    vae: Any,
    z: Any,
    return_dict: bool = True,
    max_area: Any = "auto",
) -> Any:
    """Decode ``z`` (B, C, T, H, W) in evenly spread tiles, 32x32 latents or larger when free VRAM allows."""
    import torch
    from diffusers.models.autoencoders.vae import DecoderOutput

    _, _, _, height, width = z.shape
    ratio = int(vae.spatial_compression_ratio)
    if max_area == "auto":
        max_area = decode_tile_budget(vae, z)
    th, tw = choose_tiles(height, width, max_area)
    vae._unsloth_last_decode_tile = (th, tw, max_area)
    hs = tile_starts(height, th)
    ws = tile_starts(width, tw)
    if len(hs) == 1 and len(ws) == 1:
        dec = _decode_tile(vae, z)
    else:
        wy = axis_weights(hs, th, height, ratio, torch, z.device)
        wx = axis_weights(ws, tw, width, ratio, torch, z.device)
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


def _encode_tile(vae: Any, x: Any) -> Any:
    """The VAE's own untiled ``_encode`` of one pixel tile: the (mean, logvar) moments, not a sample."""
    prev = vae.use_tiling
    vae.use_tiling = False
    try:
        return vae._encode(x)
    finally:
        vae.use_tiling = prev


def tiled_encode(vae: Any, x: Any) -> Any:
    """Encode ``x`` (B, C, T, H, W pixels) to moments in 64-latent tiles, blended in latent space."""
    import torch

    _, _, _, height_px, width_px = x.shape
    ratio = int(vae.spatial_compression_ratio)
    height, width = height_px // ratio, width_px // ratio
    tile, overlap = ENCODE_TILE_LATENTS, ENCODE_OVERLAP_LATENTS
    hs = tile_starts(height, tile, overlap)
    ws = tile_starts(width, tile, overlap)
    th, tw = min(tile, height), min(tile, width)
    if len(hs) == 1 and len(ws) == 1:
        return _encode_tile(vae, x)
    margin, ramp = ENCODE_MARGIN_LATENTS, ENCODE_RAMP_LATENTS
    wy = axis_weights(hs, tile, height, 1, torch, x.device, margin, ramp)
    wx = axis_weights(ws, tile, width, 1, torch, x.device, margin, ramp)
    out, dtype = None, None
    for i, y in enumerate(hs):
        for j, xs in enumerate(ws):
            tile_out = _encode_tile(
                vae, x[:, :, :, y * ratio : (y + th) * ratio, xs * ratio : (xs + tw) * ratio]
            )
            if out is None:
                dtype = tile_out.dtype
                out = torch.zeros(
                    (*tile_out.shape[:3], height, width),
                    dtype = torch.float32,
                    device = tile_out.device,
                )
            out[..., y : y + th, xs : xs + tw].add_(
                tile_out.float() * (wy[i].view(-1, 1) * wx[j].view(1, -1))
            )
            del tile_out
    return out.to(dtype)


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
    """Route ``vae``'s tiled decode / encode through the wide tiles. Idempotent; False when not covered.
    ``_unsloth_decode_tile_side`` carries the real decode tile side for the memory estimates."""
    if not _wants_wide_tiles(vae) or wide_tiles_disabled():
        return False
    if getattr(vae, "_unsloth_wide_tiles", False):
        return True
    own = vae.__dict__.get("tiled_decode")
    own_encode = vae.__dict__.get("tiled_encode")
    stock = vae.tiled_decode
    stock_encode = getattr(vae, "tiled_encode", None)

    def _tiled_decode(
        self: Any,
        z: Any,
        return_dict: bool = True,
    ) -> Any:
        if wide_tiles_disabled():
            return stock(z, return_dict = return_dict)
        return tiled_decode(self, z, return_dict = return_dict)

    def _tiled_encode(self: Any, x: Any) -> Any:
        if wide_tiles_disabled():
            return stock_encode(x)
        return tiled_encode(self, x)

    vae._unsloth_wide_tiles_own = own
    vae.tiled_decode = types.MethodType(_tiled_decode, vae)
    if callable(stock_encode) and callable(getattr(vae, "_encode", None)):
        vae._unsloth_wide_tiles_own_encode = own_encode
        vae.tiled_encode = types.MethodType(_tiled_encode, vae)
    vae._unsloth_decode_tile_side = TILE_LATENTS * int(vae.spatial_compression_ratio)
    vae._unsloth_wide_tiles = True
    if logger is not None:
        logger.info(
            "diffusion.vae_tiling: %s decodes in %d-latent tiles with %d-latent overlaps and encodes in %d-latent "
            "tiles with %d-latent overlaps, the last tile edge-aligned",
            type(vae).__name__,
            TILE_LATENTS,
            OVERLAP_LATENTS,
            ENCODE_TILE_LATENTS,
            ENCODE_OVERLAP_LATENTS,
        )
    return True


def uninstall(vae: Any) -> None:
    if not getattr(vae, "_unsloth_wide_tiles", False):
        return
    for attr, saved in (
        ("tiled_decode", "_unsloth_wide_tiles_own"),
        ("tiled_encode", "_unsloth_wide_tiles_own_encode"),
    ):
        if saved not in vae.__dict__ and attr == "tiled_encode":
            continue
        own: Optional[Any] = vae.__dict__.get(saved)
        if own is None:
            vae.__dict__.pop(attr, None)
        else:
            setattr(vae, attr, own)
    for name in (
        "_unsloth_wide_tiles_own",
        "_unsloth_wide_tiles_own_encode",
        "_unsloth_decode_tile_side",
    ):
        vae.__dict__.pop(name, None)
    vae._unsloth_wide_tiles = False
