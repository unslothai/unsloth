# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seam-free LTX-2 / 2.3 VAE decode: the fewest, largest tiles that fit free VRAM (untiled when it fits).

Stock tiles (16 latents, 2-latent blend) are narrower than the decoder's receptive field, so every seam shows a line.
Kill switch ``UNSLOTH_VIDEO_VAE_WIDE_TILES=0``; ``UNSLOTH_VIDEO_VAE_UNTILED=0`` keeps >= 2 tiles. Tiles span every
frame: Studio never decodes LTX framewise.
"""

from __future__ import annotations

import os
import types
from typing import Any, Optional

WIDE_TILES_ENV = "UNSLOTH_VIDEO_VAE_WIDE_TILES"

# Stock tile side: the smallest tile, so the tightest tier never needs more per tile than stock.
MIN_TILE_LATENTS = 16
OVERLAP_LATENTS = 8
MARGIN_LATENTS = 3
RAMP_LATENTS = 2

# Decode peak MiB per output frame x latent pixel (bf16, B200, 2.3 VAE): measured 0.102-0.105 stock, ~0.071 fused.
DECODE_MIB_PER_FRAME_LATENT = 0.11
DECODE_MIB_PER_FRAME_LATENT_FUSED = 0.075
# Coefficients already sit ~5% over measured peaks; an OOM still retries in stock-size tiles.
_MARGIN = 1.05
_MARGIN_BYTES = 512 * 2**20

_VAE_CLASSES = frozenset({"AutoencoderKLLTX2Video"})


def wide_tiles_disabled() -> bool:
    return (os.environ.get(WIDE_TILES_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _untiled_allowed() -> bool:
    from .video_vae_untiled import UNTILED_ENV
    return (os.environ.get(UNTILED_ENV) or "").strip().lower() not in ("0", "false", "no", "off")


def tile_starts(
    length: int,
    tile: int,
    overlap: Optional[int] = None,
) -> list[int]:
    """Evenly spread starts of the fewest tiles covering ``length`` with overlaps >= ``overlap``; last ends at length."""
    overlap = OVERLAP_LATENTS if overlap is None else overlap
    length, tile, overlap = int(length), int(tile), int(overlap)
    if length <= tile:
        return [0]
    if tile <= overlap or overlap < 0:
        raise ValueError(f"tile {tile} must exceed overlap {overlap}")
    count = -(-(length - overlap) // (tile - overlap))
    span = length - tile
    return [(i * span) // (count - 1) for i in range(count)]


def axis_weights(
    starts: list[int],
    tile: int,
    length: int,
    scale: int,
    torch: Any,
    device: Any,
    margin: Optional[int] = None,
    ramp: Optional[int] = None,
) -> list:
    """Per-tile fp32 weights along one axis summing to 1: 0 within ``margin`` of a shared edge, then a linear ``ramp``."""
    margin = MARGIN_LATENTS if margin is None else margin
    ramp = RAMP_LATENTS if ramp is None else ramp
    size = min(tile, length) * scale
    pos = (torch.arange(size, dtype = torch.float64, device = "cpu") + 0.5) / scale
    total = torch.zeros(length * scale, dtype = torch.float64, device = "cpu")
    weights = []
    for s in starts:
        w = torch.ones(size, dtype = torch.float64, device = "cpu")
        if s > 0:
            w = torch.minimum(w, ((pos - margin) / ramp).clamp(0, 1))
        if s + tile < length:
            w = torch.minimum(w, ((min(tile, length) - pos - margin) / ramp).clamp(0, 1))
        total[s * scale : s * scale + size] += w
        weights.append(w)
    if float(total.min()) <= 0.0:
        raise ValueError(f"tiles {starts} leave pixels without weight on a {length}-latent axis")
    # float64 on CPU: MPS has no float64.
    return [
        (w / total[s * scale : s * scale + size]).float().to(device)
        for w, s in zip(weights, starts)
    ]


def output_frames(vae: Any, latent_frames: int) -> int:
    ratio = int(getattr(vae, "temporal_compression_ratio", 8) or 8)
    return (max(1, int(latent_frames)) - 1) * ratio + 1


def tile_bytes(
    frames: int,
    th: int,
    tw: int,
    batch: int = 1,
    itemsize: int = 2,
    fused: bool = False,
) -> int:
    """Estimated extra bytes of one decode of a (th, tw)-latent tile over ``frames`` output frames."""
    coef = DECODE_MIB_PER_FRAME_LATENT_FUSED if fused else DECODE_MIB_PER_FRAME_LATENT
    return int(coef * 2**20 * max(2, itemsize) / 2 * max(1, batch) * frames * th * tw)


def _axis_side(length: int, count: int) -> Optional[int]:
    """Smallest tile side (>= MIN_TILE_LATENTS) covering ``length`` in ``count`` tiles with the overlap."""
    if count == 1:
        return length
    side = max(MIN_TILE_LATENTS, -(-(length + (count - 1) * OVERLAP_LATENTS) // count))
    return side if side < length else None


def choose_tiles(
    height: int,
    width: int,
    fits: Any,
    allow_single: bool = True,
) -> tuple[int, int]:
    """Tile (height, width) in latents: the fewest decoded latents (overlaps counted) among tiles ``fits(th, tw)``
    accepts. Nothing fits: the stock 16-latent tile."""
    floor = (min(MIN_TILE_LATENTS, height), min(MIN_TILE_LATENTS, width))
    best, best_cost = floor, None
    for nh in range(1, len(tile_starts(height, floor[0])) + 1):
        th = _axis_side(height, nh)
        if th is None:
            continue
        for nw in range(1, len(tile_starts(width, floor[1])) + 1):
            tw = _axis_side(width, nw)
            if (
                tw is None
                or (not allow_single and th == height and tw == width)
                or not fits(th, tw)
            ):
                continue
            n = len(tile_starts(height, th)) * len(tile_starts(width, tw))
            cost = (n * th * tw, n)
            if best_cost is None or cost < best_cost:
                best, best_cost = (th, tw), cost
    return best


def _free_bytes(device: Any) -> Optional[int]:
    from .video_vae_untiled import _free_bytes as free_bytes
    return free_bytes(device)


def _itemsize(vae: Any) -> int:
    from .video_vae_untiled import _decoder_itemsize
    return _decoder_itemsize(vae)


def plan_tiles(
    vae: Any,
    z: Any,
    free: Optional[int] = None,
) -> tuple[int, int]:
    """The decode tile for ``z`` (B, C, T, H, W) given ``free`` bytes (read from the device when None)."""
    return _plan(vae, z, free)[0]


def _plan(
    vae: Any,
    z: Any,
    free: Optional[int] = None,
) -> tuple[tuple[int, int], bool]:
    """``(tile, budgeted)``; not budgeted (nothing fit) means no room was priced in for the fp32 accumulator."""
    batch, _, latent_frames, height, width = (int(x) for x in z.shape)
    frames = output_frames(vae, latent_frames)
    ratio = int(vae.spatial_compression_ratio)
    if free is None:
        free = _free_bytes(getattr(z, "device", None))
    allow_single = _untiled_allowed()
    if free is None:
        return choose_tiles(height, width, lambda th, tw: False, allow_single), False
    itemsize = _itemsize(vae)
    fused = bool(getattr(vae, "_unsloth_vae_fused_installed", 0))
    accum = 4 * batch * 3 * frames * height * ratio * width * ratio

    def fits(th: int, tw: int) -> bool:
        need = tile_bytes(frames, th, tw, batch, itemsize, fused) * _MARGIN + _MARGIN_BYTES
        return need + (0 if (th == height and tw == width) else accum) <= free

    tile = choose_tiles(height, width, fits, allow_single)
    return tile, fits(*tile)


def _decode_tiles(
    vae: Any,
    z: Any,
    temb: Any,
    causal: Any,
    th: int,
    tw: int,
    fp32_accum: bool = True,
) -> Any:
    import torch

    _, _, _, height, width = z.shape
    ratio = int(vae.spatial_compression_ratio)
    hs, ws = tile_starts(height, th), tile_starts(width, tw)
    if len(hs) == 1 and len(ws) == 1:
        return vae.decoder(z, temb, causal = causal)
    wy = axis_weights(hs, th, height, ratio, torch, z.device)
    wx = axis_weights(ws, tw, width, ratio, torch, z.device)
    out, dtype = None, None
    for i, y in enumerate(hs):
        for j, x in enumerate(ws):
            tile = vae.decoder(z[:, :, :, y : y + th, x : x + tw], temb, causal = causal)
            if out is None:
                dtype = tile.dtype
                # Unbudgeted (nothing fit): accumulate in the tile dtype, else the fp32 buffer exceeds stock's peak.
                out = torch.zeros(
                    (*tile.shape[:3], height * ratio, width * ratio),
                    dtype = torch.float32 if fp32_accum else dtype,
                    device = tile.device,
                )
            w = wy[i].view(-1, 1) * wx[j].view(1, -1)
            out[..., y * ratio : (y + th) * ratio, x * ratio : (x + tw) * ratio].addcmul_(
                tile.float() if fp32_accum else tile, w
            )
            del tile
    dec = out.to(dtype)
    del out
    return dec


def tiled_decode(
    vae: Any,
    z: Any,
    temb: Any = None,
    causal: Any = None,
    return_dict: bool = True,
) -> Any:
    """Decode ``z`` in the tiles ``plan_tiles`` picks; on OOM, once more in the stock-size 16-latent tiles."""
    from diffusers.models.autoencoders.vae import DecoderOutput

    from .diffusion_batched import is_oom_error

    (th, tw), budgeted = _plan(vae, z)
    floor = (min(MIN_TILE_LATENTS, int(z.shape[-2])), min(MIN_TILE_LATENTS, int(z.shape[-1])))
    stats = vae.__dict__.setdefault(
        "_unsloth_wide_tiles_stats", {"untiled": 0, "tiled": 0, "oom_fallback": 0}
    )
    vae._unsloth_last_decode_tile = (th, tw)
    failed = False
    try:
        dec = _decode_tiles(vae, z, temb, causal, th, tw, budgeted)
    except Exception as exc:  # noqa: BLE001
        if (th, tw) == floor or not is_oom_error(exc):
            raise
        failed = True
    if failed:
        # Retried outside the handler so the failed attempt's tensors (held by the traceback) are gone first.
        import torch

        stats["oom_fallback"] += 1
        torch.cuda.empty_cache()
        th, tw = floor
        vae._unsloth_last_decode_tile = floor
        dec = _decode_tiles(vae, z, temb, causal, th, tw, False)
    stats["untiled" if (th, tw) == tuple(int(x) for x in z.shape[-2:]) else "tiled"] += 1
    if not return_dict:
        return (dec,)
    return DecoderOutput(sample = dec)


def _covered(vae: Any) -> bool:
    if vae is None or type(vae).__name__ not in _VAE_CLASSES:
        return False
    try:
        ratio = int(vae.spatial_compression_ratio)
    except (AttributeError, TypeError, ValueError):
        return False
    return ratio == 32 and callable(getattr(vae, "decoder", None)) and hasattr(vae, "use_tiling")


def install(vae: Any, logger: Any = None) -> bool:
    """Route ``vae``'s tiled decode through tiles sized to the free VRAM. Idempotent; False when not covered."""
    if not _covered(vae) or wide_tiles_disabled():
        return False
    if getattr(vae, "_unsloth_wide_tiles", False):
        return True
    stock = vae.tiled_decode

    def _tiled_decode(
        self: Any,
        z: Any,
        temb: Any = None,
        causal: Any = None,
        return_dict: bool = True,
    ) -> Any:
        if wide_tiles_disabled():
            return stock(z, temb, causal = causal, return_dict = return_dict)
        return tiled_decode(self, z, temb, causal = causal, return_dict = return_dict)

    vae._unsloth_wide_tiles_own = vae.__dict__.get("tiled_decode")
    vae.tiled_decode = types.MethodType(_tiled_decode, vae)
    vae._unsloth_wide_tiles = True
    if logger is not None:
        logger.info(
            "video.vae_wide_tiles: %s decodes in tiles sized to free VRAM (>= %d latents, %d-latent overlaps)",
            type(vae).__name__,
            MIN_TILE_LATENTS,
            OVERLAP_LATENTS,
        )
    return True


def uninstall(vae: Any) -> None:
    if not getattr(vae, "_unsloth_wide_tiles", False):
        return
    own = vae.__dict__.pop("_unsloth_wide_tiles_own", None)
    if own is None:
        vae.__dict__.pop("tiled_decode", None)
    else:
        vae.tiled_decode = own
    vae.__dict__.pop("_unsloth_last_decode_tile", None)
    vae._unsloth_wide_tiles = False
