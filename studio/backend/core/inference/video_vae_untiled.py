# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decode a resident video VAE untiled per call when its estimated peak fits in free memory; tiled otherwise or on OOM."""

from __future__ import annotations

import os
from typing import Any, Optional

from .family_profile import profile_table as _profile_table

UNTILED_ENV = "UNSLOTH_VIDEO_VAE_UNTILED"

# Untiled decode peak models per family (``FamilyProfile.untiled_vae`` on the video registry, measured on a B200; the
# reasoning sits next to the data). Per latent pixel (h x w) + the (spatial, temporal) RGB output for a causal decoder
# (Wan), or per output frame x latent pixel + the temporal scale for a decoder holding every frame (LTX-2 / 2.3 / 2.5).
_UNTILED = _profile_table("untiled_vae")
_BYTES_PER_LATENT_PIXEL = {
    name: m.bytes_per_latent_pixel
    for name, m in _UNTILED.items()
    if m.bytes_per_latent_pixel is not None
}
_OUTPUT_SCALE = {name: m.output_scale for name, m in _UNTILED.items() if m.output_scale is not None}
_BYTES_PER_FRAME_LATENT_PIXEL = {
    name: m.bytes_per_frame_latent_pixel
    for name, m in _UNTILED.items()
    if m.bytes_per_frame_latent_pixel is not None
}
_TEMPORAL_SCALE = {
    name: m.temporal_scale for name, m in _UNTILED.items() if m.temporal_scale is not None
}
_MARGIN = 1.25
_MARGIN_BYTES = 2 * 2**30


def _enabled() -> bool:
    return os.environ.get(UNTILED_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def untiled_decode_bytes(
    family: str,
    latent_shape: tuple,
    itemsize: int = 2,
) -> Optional[int]:
    """Extra bytes of an untiled decode of ``latent_shape`` (B, C, T, h, w), or None if unmeasured."""
    if len(latent_shape) != 5:
        return None
    batch, _, latent_frames, height, width = (int(x) for x in latent_shape)
    batch, itemsize = max(1, batch), max(2, itemsize)
    per_frame = _BYTES_PER_FRAME_LATENT_PIXEL.get(family)
    if per_frame is not None:
        frames = (max(1, latent_frames) - 1) * _TEMPORAL_SCALE[family] + 1
        return int(per_frame * itemsize / 2 * batch * frames * height * width)
    coef = _BYTES_PER_LATENT_PIXEL.get(family)
    if coef is None:
        return None
    spatial, temporal = _OUTPUT_SCALE[family]
    frames = (max(1, latent_frames) - 1) * temporal + 1
    output = 2 * batch * 3 * frames * height * spatial * width * spatial * itemsize
    return int(coef * itemsize / 2 * batch * height * width + output)


# When untiled does not fit, decode the fewest tiles that fit at the stock overlap (stock Wan: 28 tiles at 1280x704).
WIDE_TILES_ENV = "UNSLOTH_VIDEO_VAE_WAN_WIDE_TILES"
_STOCK_TILE = (256, 192)  # diffusers AutoencoderKLWan tile_sample_min / tile_sample_stride


def _wide_tiles_enabled() -> bool:
    return os.environ.get(WIDE_TILES_ENV, "").strip().lower() not in ("0", "false", "no", "off")


# (longest tile, edge tile) latents a family's wide tiles keep; Cosmos3: the Wan@16x seam audit floor.
_WIDE_TILE_FLOOR = {"cosmos3-nano": (16, 8), "cosmos3-edge": (16, 8)}


def _wide_axis(
    length: int, n: int, overlap: int, floor: Optional[tuple]
) -> Optional[tuple[int, int]]:
    """(tile, stride) latents for ``n`` tiles over ``length``, or None when ``floor`` rules the count out."""
    if n == 1:
        return length, length
    stride = -(-length // n)
    if floor is None:
        return min(length, stride + overlap), stride
    tile_floor, edge_floor = floor
    if -(-length // stride) != n or length - (n - 1) * stride < edge_floor:
        return None
    return min(length, max(stride + overlap, tile_floor)), stride


def wide_tile_plan(
    family: str,
    latent_shape: tuple,
    ratio: int,
    free: int,
    itemsize: int = 2,
    max_tiles: int = 16,
) -> Optional[tuple[int, int, int, int]]:
    """(min_h, stride_h, min_w, stride_w) sample px, or None. A single-tile axis gets stride >= size: no blend there."""
    coef = _BYTES_PER_LATENT_PIXEL.get(family)
    if coef is None or len(latent_shape) != 5 or ratio <= 0:
        return None
    batch, _, latent_frames, height, width = (int(x) for x in latent_shape)
    spatial, temporal = _OUTPUT_SCALE[family]
    overlap = max(1, (_STOCK_TILE[0] - _STOCK_TILE[1]) // ratio)
    frames = (max(1, latent_frames) - 1) * temporal + 1
    itemsize = max(2, itemsize)
    out_bytes = 2 * max(1, batch) * 3 * frames * height * spatial * width * spatial * itemsize
    best = None
    for n_h in range(1, height + 1):
        for n_w in range(1, width + 1):
            if n_h * n_w > max_tiles or (n_h == 1 and n_w == 1):
                continue
            axis_h = _wide_axis(height, n_h, overlap, _WIDE_TILE_FLOOR.get(family))
            axis_w = _wide_axis(width, n_w, overlap, _WIDE_TILE_FLOOR.get(family))
            if axis_h is None or axis_w is None:
                continue
            (tile_h, stride_h), (tile_w, stride_w) = axis_h, axis_w
            need = coef * itemsize / 2 * max(1, batch) * tile_h * tile_w + out_bytes
            if need * _MARGIN + _MARGIN_BYTES > free:
                continue
            area = (n_h * tile_h) * (n_w * tile_w)
            key = (n_h * n_w, area)
            if best is None or key < best[0]:
                best = (
                    key,
                    (
                        tile_h,
                        stride_h if n_h > 1 else height,
                        tile_w,
                        stride_w if n_w > 1 else width,
                    ),
                )
    if best is None:
        return None
    tile_h, stride_h, tile_w, stride_w = best[1]
    stock_tiles = len(range(0, height, _STOCK_TILE[1] // ratio)) * len(
        range(0, width, _STOCK_TILE[1] // ratio)
    )
    if best[0][0] >= stock_tiles:
        return None
    return tile_h * ratio, stride_h * ratio, tile_w * ratio, stride_w * ratio


def _decoder_itemsize(vae: Any) -> int:
    """Decoder weight element size; 4 if unreadable."""
    try:
        for param in vae.decoder.parameters():
            if param.is_floating_point():
                return int(param.element_size())
    except Exception:  # noqa: BLE001
        pass
    return 4


def _free_bytes(device: Any) -> Optional[int]:
    """Device free memory plus torch's reusable cache (not CUDA graph pools), less the user's VRAM Budget reserve (raw at the default)."""
    try:
        import torch

        from .diffusion_memory import cuda_reusable_cache_bytes, user_budget_reserve_bytes

        free, total = torch.cuda.mem_get_info(device)
        cached = cuda_reusable_cache_bytes(device)
        return max(0, int(free) + cached - user_budget_reserve_bytes(device, total))
    except Exception:  # noqa: BLE001
        return None


def _in_chain(fn: Any, target: Any) -> bool:
    for _ in range(16):
        if fn is None:
            return False
        if fn is target:
            return True
        from .decode_layers import next_layer  # noqa: PLC0415

        fn = next_layer(fn)
    return False


def _decode_device(vae: Any, z: Any) -> Any:
    """Where the decode allocates: the latent's card, unless a multi-GPU split put the VAE on another CUDA card (its
    boundary moves the latent there inside this wrapper). An offloaded VAE (weights on the host) keeps the latent's."""
    device = getattr(z, "device", None)
    try:
        vae_device = next(vae.parameters()).device
    except Exception:  # noqa: BLE001
        return device
    if (
        getattr(vae_device, "type", None) == "cuda"
        and getattr(device, "type", None) == "cuda"
        and vae_device != device
    ):
        return vae_device
    return device


def install_untiled_decode(
    pipe: Any,
    family: str,
    *,
    logger: Any = None,
) -> bool:
    """Wrap ``pipe.vae.decode`` to decode untiled when it fits; the caller gates on placement and speed tier."""
    vae = getattr(pipe, "vae", None)
    decode = getattr(vae, "decode", None)
    if not _enabled() or vae is None or not callable(decode):
        return False
    if (
        family not in _BYTES_PER_LATENT_PIXEL and family not in _BYTES_PER_FRAME_LATENT_PIXEL
    ) or not getattr(vae, "use_tiling", False):
        return False
    if getattr(decode, "_unsloth_untiled_decode", False):
        return True
    stats = {"untiled": 0, "tiled": 0, "wide": 0, "oom_fallback": 0}
    # Smallest estimates that OOMed, never retried (untiled / wide).
    oom_need: list = []
    oom_wide: list = []

    def gated(z: Any, *args: Any, **kwargs: Any) -> Any:
        if not getattr(vae, "use_tiling", False):
            return decode(z, *args, **kwargs)
        need = untiled_decode_bytes(
            family, tuple(getattr(z, "shape", ())), itemsize = _decoder_itemsize(vae)
        )
        free = _free_bytes(_decode_device(vae, z))
        fits = (
            need is not None
            and free is not None
            and need * _MARGIN + _MARGIN_BYTES <= free
            and not (oom_need and need >= oom_need[0])
        )
        if logger is not None:
            logger.info(
                "video.vae_untiled: decode %s (untiled needs ~%s MiB with margin, %s MiB free)",
                "untiled" if fits else "tiled",
                "?" if need is None else int((need * _MARGIN + _MARGIN_BYTES) / 2**20),
                "?" if free is None else int(free / 2**20),
            )
        if not fits:
            stats["tiled"] += 1
            wide = None
            ratio = int(getattr(vae, "spatial_compression_ratio", 0) or 0)
            if (
                _wide_tiles_enabled()
                and free is not None
                and hasattr(vae, "tile_sample_min_height")
            ):
                wide = wide_tile_plan(
                    family,
                    tuple(getattr(z, "shape", ())),
                    ratio,
                    free,
                    itemsize = _decoder_itemsize(vae),
                )
            if wide is None or (oom_wide and need is not None and need >= oom_wide[0]):
                return decode(z, *args, **kwargs)
            stock = (
                vae.tile_sample_min_height,
                vae.tile_sample_stride_height,
                vae.tile_sample_min_width,
                vae.tile_sample_stride_width,
            )
            (
                vae.tile_sample_min_height,
                vae.tile_sample_stride_height,
                vae.tile_sample_min_width,
                vae.tile_sample_stride_width,
            ) = wide
            if logger is not None:
                logger.info(
                    "video.vae_untiled: wide tiles %dx%d (stride %dx%d) instead of %dx%d",
                    wide[0],
                    wide[2],
                    wide[1],
                    wide[3],
                    stock[0],
                    stock[2],
                )
            import torch

            from .diffusion_batched import is_oom_error

            try:
                out = decode(z, *args, **kwargs)
                stats["wide"] += 1
                return out
            except Exception as exc:  # noqa: BLE001
                if not is_oom_error(exc):
                    raise
            finally:
                (
                    vae.tile_sample_min_height,
                    vae.tile_sample_stride_height,
                    vae.tile_sample_min_width,
                    vae.tile_sample_stride_width,
                ) = stock
            stats["oom_fallback"] += 1
            oom_wide[:] = [min([need, *oom_wide])] if need is not None else oom_wide
            from .diffusion_vae_oom import clear_partial_decode  # noqa: PLC0415

            clear_partial_decode(vae, z)
            torch.cuda.empty_cache()
            if logger is not None:
                logger.warning(
                    "video.vae_untiled: wide tiles ran out of memory; decoding at the stock tile size"
                )
            return decode(z, *args, **kwargs)
        import torch

        from .diffusion_batched import is_oom_error

        vae.use_tiling = False
        try:
            out = decode(z, *args, **kwargs)
            stats["untiled"] += 1
            return out
        except Exception as exc:  # noqa: BLE001
            if not is_oom_error(exc):
                raise
        finally:
            vae.use_tiling = True
        # Retried outside the handler so the failed attempt's tensors (held by the traceback) are gone first.
        stats["oom_fallback"] += 1
        oom_need[:] = [min([need, *oom_need])]
        from .diffusion_vae_oom import clear_partial_decode  # noqa: PLC0415

        clear_partial_decode(
            vae, z
        )  # the untiled attempt's causal state goes before the tiled decode
        torch.cuda.empty_cache()
        if logger is not None:
            logger.warning("video.vae_untiled: untiled decode ran out of memory; decoding tiled")
        return decode(z, *args, **kwargs)

    def untiled_when_fits(*args: Any, **kwargs: Any) -> Any:
        try:
            return gated(*args, **kwargs)
        finally:
            # A failed VAE compile restores its eager decode into this slot; the guarded inner call already runs eager.
            # (a layer installed over this one, e.g. the OOM retry's decode scope, keeps it in its __wrapped__ chain)
            if not _in_chain(vae.__dict__.get("decode"), untiled_when_fits):
                vae.decode = untiled_when_fits

    untiled_when_fits._unsloth_untiled_decode = True
    untiled_when_fits._unsloth_untiled_stats = stats
    vae.decode = untiled_when_fits
    if logger is not None:
        logger.info("video.vae_untiled: resident %s VAE decodes untiled when it fits", family)
    return True
