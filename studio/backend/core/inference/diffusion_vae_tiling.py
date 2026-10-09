# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seam-free tiled VAE decode / encode for every image VAE whose stock tiles fall under a floor.

Low-VRAM tiers tile the VAE with diffusers' stock geometry, which draws seams where tiles are under 32 latents or
blends under 16 (Qwen-Image-2.1 16x, HunyuanImage-2.1 32x, Qwen-Image 8x) or the last tile is a sliver (AutoencoderKL,
FLUX.2 at e.g. 1600 px). The rule is read off the VAE (``stock_tiles``, ``compression_ratio``): stock tiles that meet
the floor stay bit-identical to diffusers; otherwise tiles are >= 32 latents (larger when free VRAM allows), overlap
>= 16, edge-aligned, with a normalised blend giving no weight near shared edges. Wan-family encode tiles are 64
latents: the encoder attends over the whole tile, so small tiles shift the condition latent everywhere.
Kill switch ``UNSLOTH_DIFFUSION_VAE_WIDE_TILES=0``.
"""

from __future__ import annotations

import os
import types
from typing import Any, Optional

from .diffusion_device import float64_device

WIDE_TILES_ENV = "UNSLOTH_DIFFUSION_VAE_WIDE_TILES"

TILE_LATENTS = 32
OVERLAP_LATENTS = 16
# Blend: a tile gets no weight within MARGIN latents of an edge it shares, then ramps to full over RAMP latents.
MARGIN_LATENTS = 4
RAMP_LATENTS = 8
# Encode: a 1024 px reference is one tile (bit-identical); the encoder attends over the whole tile.
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
    build = float64_device(device)
    pos = (torch.arange(size, dtype = torch.float64, device = build) + 0.5) / scale
    total = torch.zeros(length * scale, dtype = torch.float64, device = build)
    weights = []
    for s in starts:
        w = torch.ones(size, dtype = torch.float64, device = build)
        if s > 0:
            w = torch.minimum(w, ((pos - margin) / ramp).clamp(0, 1))
        if s + tile < length:
            w = torch.minimum(w, ((min(tile, length) - pos - margin) / ramp).clamp(0, 1))
        total[s * scale : s * scale + size] += w
        weights.append(w)
    if float(total.min()) <= 0.0:
        raise ValueError(f"tiles {starts} leave pixels without weight on a {length}-latent axis")
    return [
        (w / total[s * scale : s * scale + size]).float().to(device)
        for w, s in zip(weights, starts)
    ]


def _decode_tile(vae: Any, z: Any) -> Any:
    """The VAE's own untiled decode of one tile (fused / compiled forwards still apply)."""
    prev = vae.use_tiling
    vae.use_tiling = False
    try:
        return vae._decode(z, return_dict = False)[0]
    finally:
        vae.use_tiling = prev


# bf16 decode peak MiB per latent of tile area, (unfused, fused): worst measured tile side 8 to 128, real weights.
# Per class (VAEs of one ratio differ 2x); an unmeasured class takes its ratio's worst.
DECODE_MIB_PER_LATENT_BY_CLASS = {
    "AutoencoderKLQwenImage": (0.27, 0.18),
    "AutoencoderKLQwenImage21": (1.7, 0.43),
    "AutoencoderKLHunyuanImage": (1.3, 1.3),
    "AutoencoderKL": (0.17, 0.09),
    "AutoencoderKLFlux2": (0.17, 0.09),
}
DECODE_MIB_PER_LATENT_BY_RATIO = {8: (0.27, 0.18), 16: (1.7, 0.43), 32: (1.3, 1.3)}
DECODE_MIB_PER_LATENT = 1.7
DECODE_MIB_RATIO = 16
FREE_FRACTION = 0.75
MAX_TILE_ENV = "UNSLOTH_DIFFUSION_VAE_MAX_TILE"


def decode_mib_per_latent(
    ratio: int,
    fused: bool = False,
    cls: Optional[str] = None,
) -> float:
    """bf16 decode peak MiB per latent: by class, else ratio, else the 16x figure scaled by tile pixels."""
    known = DECODE_MIB_PER_LATENT_BY_CLASS.get(cls or "") or DECODE_MIB_PER_LATENT_BY_RATIO.get(
        int(ratio)
    )
    if known is not None:
        return known[1] if fused else known[0]
    return DECODE_MIB_PER_LATENT * (int(ratio) / DECODE_MIB_RATIO) ** 2


def _geometry(vae: Any) -> tuple[int, int, int]:
    """(ratio, floor tile, floor overlap), cached on the VAE at install."""
    cached = getattr(vae, "__dict__", {}).get("_unsloth_wide_geometry")
    if cached is not None:
        return cached
    ratio = compression_ratio(vae)
    tile, overlap = floor_tiles(vae)
    return ratio, tile, overlap


def _cached_axis_weights(
    vae: Any, starts: list[int], tile: int, length: int, ratio: int, torch: Any, device: Any
):
    """``axis_weights`` memoised per VAE (a repeat decode syncs no device)."""
    cache = vae.__dict__.setdefault("_unsloth_wide_weights", {}) if hasattr(vae, "__dict__") else {}
    key = (tuple(starts), int(tile), int(length), int(ratio), str(device))
    hit = cache.get(key)
    if hit is None:
        if len(cache) >= 64:
            cache.clear()
        hit = cache[key] = axis_weights(starts, tile, length, ratio, torch, device)
    return hit


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


def _stock_span(length: int, tile: int, overlap: int) -> int:
    """Latents diffusers' stock loop decodes along one side: a tile every ``tile - overlap``, the last cut short."""
    if length <= tile:
        return length
    return sum(min(tile, length - s) for s in range(0, length, tile - overlap))


def _large_stock_sides(length: int, tile: int, overlap: int) -> list[int]:
    """Per tile count, the smallest covering side and the widest decoding no more latents than the stock loop."""
    if length <= tile:
        return [length]
    span = _stock_span(length, tile, overlap)
    out = {length}
    for n in range(2, len(tile_starts(length, tile, overlap)) + 1):
        low = _axis_tiles(length, n, TILE_LATENTS, overlap)
        if low is None:
            continue
        out.add(low)
        for side in range(min(span // n, length - 1), low, -1):
            if len(tile_starts(length, side, overlap)) * side <= span:
                out.add(side)
                break
    return sorted(out)


def choose_tiles(
    height: int,
    width: int,
    max_area: Optional[int],
    tile: int = TILE_LATENTS,
    overlap: int = OVERLAP_LATENTS,
    stock_tile: int = 0,
) -> tuple[int, int]:
    """Tile (height, width) in latents within ``max_area`` (at least the floor tile). Stock tiles under the floor:
    fewest decoded latents. Stock tile == floor (AutoencoderKL, FLUX.2 edge sliver): fewest calls, then largest
    tile, never decoding more latents per side than stock (1600 px: two 120-latent tiles, not 128 + 104 + 8)."""
    floor = (min(tile, height), min(tile, width))
    if max_area is None or max_area <= floor[0] * floor[1]:
        max_area = floor[0] * floor[1]
    if stock_tile >= tile > TILE_LATENTS:
        best, best_key = None, None
        for th in _large_stock_sides(height, tile, overlap):
            for tw in _large_stock_sides(width, tile, overlap):
                if th * tw > max_area:
                    continue
                key = (
                    len(tile_starts(height, th, overlap)) * len(tile_starts(width, tw, overlap)),
                    -th * tw,
                )
                if best_key is None or key < best_key:
                    best, best_key = (th, tw), key
        if best is not None:
            return best
    side_min = min(tile, TILE_LATENTS)
    best, best_cost = floor, None
    sides_h = [
        _axis_tiles(height, n, side_min, overlap)
        for n in range(1, len(tile_starts(height, tile, overlap)) + 1)
    ]
    sides_w = [
        _axis_tiles(width, n, side_min, overlap)
        for n in range(1, len(tile_starts(width, tile, overlap)) + 1)
    ]
    for th in sides_h:
        for tw in sides_w:
            if th is None or tw is None or th * tw > max_area:
                continue
            cost = (
                len(tile_starts(height, th, overlap))
                * th
                * len(tile_starts(width, tw, overlap))
                * tw,
                th * tw,
            )
            if best_cost is None or cost < best_cost:
                best, best_cost = (th, tw), cost
    return best


def _free_mib(
    vae: Any,
    z: Any,
    cached: bool = True,
) -> Optional[tuple[float, float]]:
    """(free MiB after the fp32 output accumulator, decoder bytes per param / 2); None off CUDA."""
    import torch

    if getattr(z, "device", None) is None or z.device.type != "cuda":
        return None
    free, _ = torch.cuda.mem_get_info(z.device)
    if cached:
        free += torch.cuda.memory_reserved(z.device) - torch.cuda.memory_allocated(z.device)
    ratio = _geometry(vae)[0]
    frames = z.shape[2] if z.dim() == 5 else 1
    # fp32 accumulator
    out_bytes = 4 * 4 * z.shape[0] * frames * z.shape[-2] * z.shape[-1] * ratio * ratio
    elem = next(vae.decoder.parameters()).element_size()
    return (free - out_bytes) / 2**20, elem / 2


def _fused(vae: Any) -> bool:
    return bool(getattr(vae, "_unsloth_vae_fused_installed", 0)) and not getattr(
        vae, "_unsloth_vae_fused_failed", False
    )


def decode_tile_budget(vae: Any, z: Any) -> Optional[int]:
    """Largest decode tile area (latents) that fits ``FREE_FRACTION`` of the free VRAM now; None off CUDA."""
    raw = (os.environ.get(MAX_TILE_ENV) or "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw) ** 2
    # The larger-tile coefficients below were measured on CUDA. Qwen-Image-2.1's unfused
    # MIOpen decoder needs substantially more workspace for larger tiles: at 17/19 GiB
    # limits, 40x40, 64x32 and 96x56 attempts OOM before recovering with 32x32 tiles.
    # Start at that tested floor on ROCm rather than retrying the same oversized tile
    # on every image. An explicit MAX_TILE_ENV above still overrides this policy.
    import torch

    if (
        z.device.type == "cuda"
        and getattr(torch.version, "hip", None)
        and type(vae).__name__ == "AutoencoderKLQwenImage21"
        and not _fused(vae)
    ):
        return TILE_LATENTS**2
    try:
        _, tile, _ = _geometry(vae)
        # Stock tile == floor: grow only into device-free memory; decoding out of the allocator cache stalled on
        # flushes (FLUX.1 1600 px, 8 GB tier: 0.21 s vs stock 0.08).
        stock_tile = vae.__dict__.get("_unsloth_wide_stock_tile") or (stock_tiles(vae) or (0, 0))[0]
        large_stock = stock_tile >= tile > TILE_LATENTS
        free = _free_mib(vae, z, cached = not large_stock)
        if free is None:
            return None
        mib, scale = free
        per_latent = (
            decode_mib_per_latent(_geometry(vae)[0], _fused(vae), type(vae).__name__) * scale
        )
        return max(0, int(FREE_FRACTION * mib / per_latent))
    except Exception:  # noqa: BLE001 - unknown budget: the 32-latent tiles the planner budgeted
        return None


def floor_shortfall(vae: Any, z: Any) -> Optional[str]:
    """Why even the floor tile cannot fit the free VRAM now, else None (also for unknown memory or an explicit
    max tile). Only VAEs whose stock tile is under the floor can fall short."""
    if (os.environ.get(MAX_TILE_ENV) or "").strip():
        return None
    try:
        free = _free_mib(vae, z)
        if free is None:
            return None
        mib, scale = free
        ratio, tile, _ = _geometry(vae)
        stock = vae.__dict__.get("_unsloth_wide_stock_tile") or (stock_tiles(vae) or (0, 0))[0]
        if stock >= tile:
            return None
        area = min(tile, z.shape[-2]) * min(tile, z.shape[-1])
        need = area * decode_mib_per_latent(ratio, _fused(vae), type(vae).__name__) * scale
    except Exception:  # noqa: BLE001
        return None
    if need <= mib:
        return None
    return (
        f"a {tile}-latent floor tile needs about {need:.0f} MiB and {max(0.0, mib):.0f} MiB is free"
    )


def tiled_decode(
    vae: Any,
    z: Any,
    return_dict: bool = True,
    max_area: Any = "auto",
) -> Any:
    """Decode ``z`` (5-D or 4-D) in evenly spread floor-sized tiles, or larger when free VRAM allows."""
    import torch
    from diffusers.models.autoencoders.vae import DecoderOutput

    height, width = z.shape[-2], z.shape[-1]
    ratio, tile, overlap = _geometry(vae)
    if max_area == "auto":
        max_area = decode_tile_budget(vae, z)
    stock_tile = vae.__dict__.get("_unsloth_wide_stock_tile") or (stock_tiles(vae) or (0, 0))[0]
    th, tw = choose_tiles(height, width, max_area, tile, overlap, stock_tile)
    vae._unsloth_last_decode_tile = (th, tw, max_area)
    hs = tile_starts(height, th, overlap)
    ws = tile_starts(width, tw, overlap)
    if len(hs) == 1 and len(ws) == 1:
        dec = _decode_tile(vae, z)
    else:
        wy = _cached_axis_weights(vae, hs, th, height, ratio, torch, z.device)
        wx = _cached_axis_weights(vae, ws, tw, width, ratio, torch, z.device)
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
    """Encode ``x`` (B, C, T, H, W pixels) to moments in 64-latent tiles, blended in latent space."""
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
    """Pixels per latent: ``spatial_compression_ratio`` (attribute or config), else pixel / latent tile sides."""
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
    """diffusers' stock (tile, overlap) in latents for ``vae``, or None when unreadable. The overlap is the blend width."""
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
    """Whether every stock tile is >= 32 latents and every overlap >= 16. diffusers starts a tile every ``tile -
    overlap`` latents until the canvas ends, so the last tile can be a sliver (8 latents on a 200-latent side)."""
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
    """Wide-path minimum (tile, overlap): 32 / 16, or the stock values when larger."""
    stock = stock_tiles(vae) or (0, 0)
    return max(TILE_LATENTS, stock[0]), max(OVERLAP_LATENTS, stock[1])


# Class name -> measured reason to keep diffusers' tiles. Empty: every covered VAE measured better or identical.
KEEP_STOCK: dict[str, str] = {}


def _wants_wide_tiles(vae: Any) -> bool:
    if vae is None or type(vae).__name__ in KEEP_STOCK:
        return False
    config = getattr(vae, "config", None)
    # Patchified Wan VAEs blend in patch space; FLUX.2's patch_size is pipeline packing its decode never sees.
    if getattr(config, "patch_size", None) is not None and hasattr(
        vae, "tile_sample_stride_height"
    ):
        return False
    if stock_tiles(vae) is None:
        return False
    return (
        callable(getattr(vae, "_decode", None))
        and callable(getattr(vae, "tiled_decode", None))
        and hasattr(vae, "use_tiling")
    )


def _wide_encode(vae: Any) -> bool:
    """Wan family only: AutoencoderKL-style VAEs tile their encode in a private ``_tiled_encode``."""
    return (
        hasattr(vae, "tile_sample_stride_height")
        and callable(getattr(vae, "tiled_encode", None))
        and callable(getattr(vae, "_encode", None))
    )


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch
        oom = getattr(torch, "OutOfMemoryError", None) or getattr(
            torch.cuda, "OutOfMemoryError", None
        )
        if oom is not None and isinstance(exc, oom):
            return True
    except Exception:  # noqa: BLE001
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _release_cache(z: Any) -> None:
    try:
        import torch
        if getattr(z, "device", None) is not None and z.device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _log_stock_fallback(vae: Any, reason: str) -> None:
    logger = vae.__dict__.get("_unsloth_wide_logger")
    seen = vae.__dict__.setdefault("_unsloth_wide_fallback_logged", set())
    kind = reason.split(" ", 3)[:3]
    key = " ".join(kind)
    msg = "diffusion.vae_tiling: %s decodes in its stock tiles (seams possible): %s"
    if logger is None:
        return
    if key in seen:
        logger.debug(msg, type(vae).__name__, reason)
    else:
        seen.add(key)
        logger.warning(msg, type(vae).__name__, reason)


def install(vae: Any, logger: Any = None) -> bool:
    """Route ``vae``'s tiled decode / encode through the wide tiles where stock falls under the floor. Idempotent;
    False when not covered. ``_unsloth_decode_tile_side`` carries the real decode tile side for the memory estimates."""
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

    def _stock(z: Any, return_dict: bool) -> Any:
        fast = vae.__dict__.get("_unsloth_wide_stock_decode")
        return (fast or stock)(z, return_dict = return_dict)

    def _tiled_decode(
        self: Any,
        z: Any,
        return_dict: bool = True,
    ) -> Any:
        if wide_tiles_disabled() or stock_layout_ok(geometry, z.shape[-2], z.shape[-1]):
            return _stock(z, return_dict)
        # Never trade an OOM for quality: floor tile cannot fit (or OOMs) -> the smaller stock tiles.
        reason = floor_shortfall(self, z)
        if reason is None:
            for max_area in ("auto", 0):
                try:
                    return tiled_decode(self, z, return_dict = return_dict, max_area = max_area)
                except Exception as exc:  # noqa: BLE001
                    if not _is_oom(exc):
                        raise
                    reason = f"the wide tiles ran out of memory ({type(exc).__name__})"
                    tried = (self.__dict__.get("_unsloth_last_decode_tile") or (0, 0))[:2]
                # outside the except block, so the failed attempt's tensors are freed
                _release_cache(z)
                ratio, tile, _ = _geometry(self)
                if tuple(tried) == (min(tile, z.shape[-2]), min(tile, z.shape[-1])):
                    break
        _log_stock_fallback(self, reason)
        return _stock(z, return_dict)

    def _tiled_encode(self: Any, x: Any) -> Any:
        if wide_tiles_disabled() or stock_layout_ok(
            geometry, x.shape[-2] // ratio, x.shape[-1] // ratio
        ):
            return stock_encode(x)
        return tiled_encode(self, x)

    vae._unsloth_wide_tiles_own = own
    vae._unsloth_wide_geometry = (ratio, *floor_tiles(vae))
    vae._unsloth_wide_stock_tile = geometry[0]
    vae._unsloth_wide_logger = logger
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
    fast = vae.__dict__.pop("_unsloth_wide_stock_decode", None)
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
        "_unsloth_wide_geometry",
        "_unsloth_wide_weights",
        "_unsloth_wide_logger",
        "_unsloth_wide_fallback_logged",
        "_unsloth_wide_stock_tile",
    ):
        vae.__dict__.pop(name, None)
    if fast is not None and "tiled_decode" not in vae.__dict__:
        vae.tiled_decode = fast
    vae._unsloth_wide_tiles = False
