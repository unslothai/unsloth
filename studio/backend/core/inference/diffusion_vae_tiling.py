# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seam-free tiled decode and encode for every image VAE whose stock tiles fall under a floor.

Studio's low-VRAM tiers (whole-model / sequential offload, streaming, the per-call guard) turn on diffusers' tiled VAE
decode with the stock tile geometry, which is too small for several VAEs:

- Qwen-Image-2.1 (AutoencoderKLQwenImage21, 16x) gets the Wan VAE's pixel geometry (256 px tiles, 192 px stride): a
  tile is 16 latents with a 4-latent (64 px) blend, and the last row / column is a 4-latent sliver. The decoder sees
  further than 4 latents past a tile edge, so every seam and the sliver decode differently from their neighbours and
  leave thin vertical / horizontal lines, one tile long.
- HunyuanImage-2.1 (AutoencoderKLHunyuanImage, 32x) tiles at 384 px: 12-latent tiles with 3-latent blends, which
  draws a visible grid and over-sharpened texture (21 to 23 dB against the untiled decode of the same latent).
- Qwen-Image, Qwen-Image-Edit and Krea-2 (AutoencoderKLQwenImage, 8x) tile at 32 latents with 8-latent blends.
- AutoencoderKL (FLUX.1, Kontext, Z-Image, HiDream, Lumina 2, SDXL) and AutoencoderKLFlux2 (FLUX.2, Ideogram 4) tile at
  128 latents with 32-latent blends, which is fine, except that the last tile is whatever is left: an 8-latent sliver
  on a 1600 px side.

The rule is read off the VAE, not a class list: ``stock_tiles`` takes diffusers' tile and overlap (in latents) from
the VAE's own tile attributes and ``compression_ratio`` its pixels per latent. Each tiled decode keeps the stock tiles
(bit-identical to diffusers) when every stock tile is at least 32 latents and every overlap at least 16 (ComfyUI's
decode overlap), and otherwise takes the wide tiles: at least 32 latents (or the stock tile, when larger) with at least
a 16-latent (or the stock) overlap. ``KEEP_STOCK`` names VAEs that must keep the stock tiles, with the measured reason.

A wide decode uses larger tiles, down to one untiled decode, when three quarters of the free VRAM hold them: fewer
tiles decode less overlap, so they are faster and leave fewer seams. Otherwise it uses the floor tiles, which the
planner budgets. Tiles are spread evenly so the last one ends at the image edge at full size (no sliver). A tile gets no
weight within 4 latents of an edge it shares with another tile and ramps to full weight over the next 8, normalised
where more than two tiles meet (a stride under 16 latents, e.g. 1344 or 2400 px on Qwen-Image-2.1). A canvas that fits
one tile is decoded / encoded untiled. The Wan-family encode (edit and img2img inputs) uses 64-latent tiles with
32-latent overlaps, blended in latent space: the encoder attends over the whole tile, so the stock tiles shift the
condition latent everywhere and derail an edit, not only at the seams. On Qwen-Image-2.1 a 1024 px input is one tile,
identical to the untiled encode; the guard charges edit inputs as condition pixels, which covers one encode tile at
every reference resolution. AutoencoderKL-style VAEs keep their private 1024 px encode tiles.
Kill switch ``UNSLOTH_DIFFUSION_VAE_WIDE_TILES=0``: at load it skips the install (the fused batched tile decode, if
any, installs as before); set later, each decode / encode takes the stock tiled path.
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
# Encode: twice the decode tile (stable-diffusion.cpp's ratio), so a 1024 px condition image, Studio's default
# reference resolution, encodes as one tile, bit-identical to the untiled encode. The encoder's middle block attends
# over the whole tile, so smaller tiles shift the latent everywhere, not only at the seams.
ENCODE_TILE_LATENTS = 64
ENCODE_OVERLAP_LATENTS = 32
ENCODE_MARGIN_LATENTS = 8
ENCODE_RAMP_LATENTS = 16

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
    """Per-tile blend weights (fp32, ``min(tile, length) * scale`` long) along one axis, summing to 1 at every pixel.

    A tile's weight is 0 within ``margin`` latents of an edge it shares with another tile (where its decode lacks
    context), rises linearly over the next ``ramp`` latents, and is 1 deeper in; image borders are not shared edges.
    When three tiles overlap (a stride under 16 latents) the outer tile's edge region still gets no weight. With every
    overlap >= 16 latents, each pixel lies >= 8 latents inside some tile, so a margin under 8 keeps the sum positive;
    with margin 4 and ramp 8 a two-tile overlap of exactly 16 is a linear cross-fade over its middle 8 latents."""
    size = min(tile, length) * scale
    # pixel centres, in latents
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
    """The VAE's own untiled decode of one tile (fused / compiled module forwards still apply)."""
    prev = vae.use_tiling
    vae.use_tiling = False
    try:
        return vae._decode(z, return_dict = False)[0]
    finally:
        vae.use_tiling = prev


# Unfused bf16 decode peak per latent of tile area at 16x (measured on Qwen-Image-2.1: 1,697 MiB for 32x32, 6,717 MiB
# for 64x64 untiled). Other ratios scale it by the tile's pixel count; it bounds every covered VAE's measured peak.
DECODE_MIB_PER_LATENT = 1.7
DECODE_MIB_RATIO = 16
# Share of the free VRAM (after the output accumulator) a decode tile may take; the per-area figure above is an
# upper bound on the measured peaks, so this keeps at least a quarter of the free memory unused.
FREE_FRACTION = 0.75
MAX_TILE_ENV = "UNSLOTH_DIFFUSION_VAE_MAX_TILE"


def _axis_tiles(
    length: int,
    count: int,
    tile: int = TILE_LATENTS,
    overlap: int = OVERLAP_LATENTS,
) -> Optional[int]:
    """Smallest tile side (>= ``tile`` latents) that covers ``length`` in ``count`` tiles with ``overlap`` overlaps."""
    if count == 1:
        return length
    side = max(tile, -(-(length + (count - 1) * overlap) // count))
    return side if side < length else None


def choose_tiles(
    height: int,
    width: int,
    max_area: Optional[int],
    tile: int = TILE_LATENTS,
    overlap: int = OVERLAP_LATENTS,
) -> tuple[int, int]:
    """Tile (height, width) in latents: the fewest decoded latents (overlaps counted) whose tile area fits
    ``max_area``. A canvas that fits decodes untiled; with no room for more, ``tile``-sided tiles (32x32 unless the
    VAE's own tiles are larger) as the planner budgets."""
    floor = (min(tile, height), min(tile, width))
    if max_area is None or max_area <= floor[0] * floor[1]:
        return floor
    best, best_cost = floor, None
    sides_h = [
        _axis_tiles(height, n, tile, overlap)
        for n in range(1, len(tile_starts(height, tile, overlap)) + 1)
    ]
    sides_w = [
        _axis_tiles(width, n, tile, overlap)
        for n in range(1, len(tile_starts(width, tile, overlap)) + 1)
    ]
    for th in sides_h:
        for tw in sides_w:
            if th is None or tw is None or th * tw > max_area:
                continue
            cost = (
                len(tile_starts(height, th, overlap)) * th * len(tile_starts(width, tw, overlap)) * tw,
                th * tw,
            )
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
        ratio = compression_ratio(vae)
        frames = z.shape[2] if z.dim() == 5 else 1
        # fp32 accumulator
        out_bytes = 4 * 4 * z.shape[0] * frames * z.shape[-2] * z.shape[-1] * ratio * ratio
        elem = next(vae.decoder.parameters()).element_size()
        mib = FREE_FRACTION * (free - out_bytes) / 2**20
        per_latent = DECODE_MIB_PER_LATENT * (ratio / DECODE_MIB_RATIO) ** 2
        return max(0, int(mib / (per_latent * elem / 2)))
    except Exception:  # noqa: BLE001 - unknown budget: the 32-latent tiles the planner budgeted
        return None


def tiled_decode(
    vae: Any,
    z: Any,
    return_dict: bool = True,
    max_area: Any = "auto",
) -> Any:
    """Decode ``z`` ((B, C, T, H, W) or (B, C, H, W)) in evenly spread tiles with 16-latent overlaps: 32x32 latents at
    least (or the VAE's own tile and overlap, when larger), larger (down to one untiled decode) when the free VRAM
    allows, since fewer tiles decode less overlap."""
    import torch
    from diffusers.models.autoencoders.vae import DecoderOutput

    height, width = z.shape[-2], z.shape[-1]
    ratio = compression_ratio(vae)
    tile, overlap = floor_tiles(vae)
    if max_area == "auto":
        max_area = decode_tile_budget(vae, z)
    th, tw = choose_tiles(height, width, max_area, tile, overlap)
    vae._unsloth_last_decode_tile = (th, tw, max_area)
    hs = tile_starts(height, th, overlap)
    ws = tile_starts(width, tw, overlap)
    if len(hs) == 1 and len(ws) == 1:
        dec = _decode_tile(vae, z)
    else:
        wy = axis_weights(hs, th, height, ratio, torch, z.device)
        wx = axis_weights(ws, tw, width, ratio, torch, z.device)
        out, dtype = None, None
        for i, y in enumerate(hs):
            for j, x in enumerate(ws):
                tile = _decode_tile(vae, z[..., y : y + th, x : x + tw])
                if out is None:
                    dtype = tile.dtype
                    out = torch.zeros(
                        (*tile.shape[:-2], height * ratio, width * ratio),
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
    """Encode ``x`` (B, C, T, H, W pixels) to moments in evenly spread 64-latent tiles with 32-latent overlaps,
    blended in latent space with the decode's normalised ramp (no weight within 8 latents of a shared edge, full
    weight 16 latents further in). Stock ``tiled_encode`` uses 16-latent tiles, 4-latent blends and sliver tiles."""
    import torch

    height_px, width_px = x.shape[-2], x.shape[-1]
    ratio = compression_ratio(vae)
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


def compression_ratio(vae: Any) -> int:
    """Pixels per latent along a side: the VAE's own ``spatial_compression_ratio`` (an attribute on the Wan-family
    VAEs, a config value on HunyuanImage), else its pixel / latent tile sides (AutoencoderKL, FLUX.2)."""
    if "spatial_compression_ratio" in getattr(vae, "__dict__", {}) or hasattr(
        type(vae), "spatial_compression_ratio"
    ):
        return int(vae.spatial_compression_ratio)
    ratio = getattr(getattr(vae, "config", None), "spatial_compression_ratio", None)
    if ratio is not None:
        return int(ratio)
    sample = getattr(vae, "tile_sample_min_size", None)
    latent = getattr(vae, "tile_latent_min_size", None)
    if isinstance(sample, int) and isinstance(latent, int) and latent > 0 and sample % latent == 0:
        return sample // latent
    raise ValueError(f"{type(vae).__name__} has no readable spatial compression ratio")


def stock_tiles(vae: Any) -> Optional[tuple[int, int]]:
    """The (tile, overlap) in latents that diffusers' own tiled decode / encode uses on ``vae``, read off the VAE's
    tile attributes; None when they are not readable. Wan-family VAEs (Qwen-Image, Qwen-Image-2.1) carry pixel tiles
    and strides, AutoencoderKL-style VAEs (FLUX.1, FLUX.2, SDXL, HunyuanImage) a latent tile and an overlap factor.
    The overlap is the blend width: both stock loops cross-fade over all of it."""
    try:
        ratio = compression_ratio(vae)
        if hasattr(vae, "tile_sample_stride_height"):
            sides = (
                (int(vae.tile_sample_min_height), int(vae.tile_sample_stride_height)),
                (int(vae.tile_sample_min_width), int(vae.tile_sample_stride_width)),
            )
            tile = min(t // ratio for t, _ in sides)
            overlap = min((t - st) // ratio for t, st in sides)
        elif hasattr(vae, "tile_latent_min_size") and hasattr(vae, "tile_overlap_factor"):
            tile = int(vae.tile_latent_min_size)
            overlap = tile - int(tile * (1 - float(vae.tile_overlap_factor)))
        else:
            return None
    except (AttributeError, TypeError, ValueError):
        return None
    if tile <= 0 or overlap < 0 or overlap >= tile:
        return None
    return tile, overlap


def stock_layout_ok(stock: tuple[int, int], height: int, width: int) -> bool:
    """Whether the stock tiles of ``stock`` (tile, overlap) on a ``height`` x ``width`` latent canvas meet the wide
    tiles' floor: every tile at least 32 latents (no edge sliver) and every overlap / blend at least 16 latents.

    diffusers starts a tile every ``tile - overlap`` latents until the canvas ends, so the last tile is whatever is
    left (an 8-latent sliver on a 200-latent side of a 128 / 96 VAE). A canvas within one tile is not tiled."""
    tile, overlap = stock
    if tile < TILE_LATENTS or overlap < OVERLAP_LATENTS:
        return False
    stride = tile - overlap
    for length in (int(height), int(width)):
        if length <= tile:
            continue
        if any(length - s < TILE_LATENTS for s in range(stride, length, stride)):
            return False
    return True


def floor_tiles(vae: Any) -> tuple[int, int]:
    """(tile, overlap) in latents the wide path never goes under: 32 / 16, or the VAE's own when larger."""
    stock = stock_tiles(vae) or (0, 0)
    return max(TILE_LATENTS, stock[0]), max(OVERLAP_LATENTS, stock[1])


# VAEs that keep diffusers' tiles whatever their geometry, each with the measured reason. Empty: every image VAE
# Studio tiles measured better (or identical) on the wide tiles.
KEEP_STOCK: dict[str, str] = {}


def _wants_wide_tiles(vae: Any) -> bool:
    """Covered when the stock tile geometry is readable off the VAE (so each call can be checked against the floor),
    the Wan-family latents are not patchified, and the class is not in ``KEEP_STOCK``."""
    if vae is None or type(vae).__name__ in KEEP_STOCK:
        return False
    config = getattr(vae, "config", None)
    # A patchified Wan-family VAE (patch_size set) blends its stock tiles in patch space; FLUX.2's patch_size is the
    # pipeline's latent packing, which its decode never sees.
    if getattr(config, "patch_size", None) is not None and hasattr(vae, "tile_sample_stride_height"):
        return False
    if stock_tiles(vae) is None:
        return False
    return (
        callable(getattr(vae, "_decode", None))
        and callable(getattr(vae, "tiled_decode", None))
        and hasattr(vae, "use_tiling")
    )


def _wide_encode(vae: Any) -> bool:
    """The tiled encode is replaced only where ``_encode`` routes through ``tiled_encode`` and gets moments back (the
    Wan family); AutoencoderKL-style VAEs tile their encode in a private ``_tiled_encode`` with 1024 px tiles."""
    return (
        hasattr(vae, "tile_sample_stride_height")
        and callable(getattr(vae, "tiled_encode", None))
        and callable(getattr(vae, "_encode", None))
    )


def install(vae: Any, logger: Any = None) -> bool:
    """Route ``vae``'s tiled decode and encode through the wide, edge-aligned tiles. Idempotent; False when not covered.

    Each tiled decode keeps diffusers' own tiles when they already meet the floor (``stock_layout_ok``: FLUX.1 /
    FLUX.2 / SDXL at most sizes), so it is bit-identical to the stock decode there, and takes the wide tiles when
    they do not (an edge sliver, or tiles / blends under 32 / 16 latents). The tiled encode (img2img, edits) of the
    Wan-family VAEs gets 64-latent tiles. The VAE's tile attributes stay as diffusers set them;
    ``_unsloth_decode_tile_side`` carries the real decode tile side for Studio's memory estimates."""
    if not _wants_wide_tiles(vae) or wide_tiles_disabled():
        return False
    if getattr(vae, "_unsloth_wide_tiles", False):
        return True
    own = vae.__dict__.get("tiled_decode")
    own_encode = vae.__dict__.get("tiled_encode")
    stock = vae.tiled_decode
    stock_encode = getattr(vae, "tiled_encode", None)
    geometry = stock_tiles(vae)
    ratio = compression_ratio(vae)

    def _tiled_decode(
        self: Any,
        z: Any,
        return_dict: bool = True,
    ) -> Any:
        if wide_tiles_disabled() or stock_layout_ok(geometry, z.shape[-2], z.shape[-1]):
            return stock(z, return_dict = return_dict)
        return tiled_decode(self, z, return_dict = return_dict)

    def _tiled_encode(self: Any, x: Any) -> Any:
        if wide_tiles_disabled() or stock_layout_ok(
            geometry, x.shape[-2] // ratio, x.shape[-1] // ratio
        ):
            return stock_encode(x)
        return tiled_encode(self, x)

    vae._unsloth_wide_tiles_own = own
    vae.tiled_decode = types.MethodType(_tiled_decode, vae)
    if _wide_encode(vae):
        vae._unsloth_wide_tiles_own_encode = own_encode
        vae.tiled_encode = types.MethodType(_tiled_encode, vae)
    tile, overlap = floor_tiles(vae)
    vae._unsloth_decode_tile_side = tile * ratio
    vae._unsloth_wide_tiles = True
    if logger is not None:
        logger.info(
            "diffusion.vae_tiling: %s (%dx, stock %d-latent tiles with %d-latent overlaps) decodes in %d-latent "
            "tiles with %d-latent overlaps, the last tile edge-aligned, whenever the stock tiles fall under that",
            type(vae).__name__,
            ratio,
            geometry[0],
            geometry[1],
            tile,
            overlap,
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
