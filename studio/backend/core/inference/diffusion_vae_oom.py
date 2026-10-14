# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Retry a VAE decode that ran out of memory in smaller tiles instead of failing the render.

Wraps the instance ``_decode`` every diffusers VAE routes ``decode`` through, so it sits under the decode-level
wrappers (fp16 non-finite rerun, compiled decode guard, untiled-when-fits) and over the tiled decode. On an OOM:
an untiled decode is retried tiled (ComfyUI's one retry); a tiled decode is retried with every spatial tile
attribute the VAE exposes halved (``tile_sample_min_*``, ``tile_latent_min_*``, ``tile_sample_stride_*``,
``tile_sample_min_overlap_*``), so the overlap fraction and the stock blended seams are kept, down to a 4-latent
floor. The attributes are restored after the call; the level that worked is remembered per latent shape so later
renders of that size do not pay the failed attempts again. Kill switch ``UNSLOTH_VAE_OOM_RETRY=0``.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Optional

OOM_RETRY_ENV = "UNSLOTH_VAE_OOM_RETRY"
MIN_TILE_LATENTS = 4
MIN_TILE_PIXELS = 32
MAX_HALVINGS = 6

# Spatial tile attributes diffusers VAEs expose; temporal ones are left alone (causal decoders hold few frames).
_SAMPLE_MIN = ("tile_sample_min_size", "tile_sample_min_height", "tile_sample_min_width")
_LATENT_MIN = ("tile_latent_min_size", "tile_latent_min_height", "tile_latent_min_width")
_SCALED_WITH_MIN = (
    "tile_sample_stride_height",
    "tile_sample_stride_width",
    "tile_sample_min_overlap_height",
    "tile_sample_min_overlap_width",
    "tile_latent_stride_height",
    "tile_latent_stride_width",
)


def oom_retry_disabled() -> bool:
    return os.environ.get(OOM_RETRY_ENV, "").strip().lower() in ("0", "false", "no", "off")


def is_oom(exc: BaseException) -> bool:
    if any(cls.__name__ in ("OutOfMemoryError", "OutOfMemoryError_") for cls in type(exc).__mro__):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in text or "tried to allocate" in text
    )


def _int_attr(vae: Any, name: str) -> Optional[int]:
    value = vae.__dict__.get(name, getattr(type(vae), name, None))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _spatial_ratio(vae: Any) -> int:
    for name in ("spatial_compression_ratio",):
        try:
            r = int(getattr(vae, name))
            if r > 0:
                return r
        except Exception:  # noqa: BLE001
            pass
    try:
        cfg = vae.config
        r = getattr(cfg, "spatial_compression_ratio", None)
        if r:
            return int(r)
        boc = getattr(cfg, "block_out_channels", None)
        if boc:
            return 2 ** (len(boc) - 1)
    except Exception:  # noqa: BLE001
        pass
    return 8


def tile_attrs(vae: Any) -> dict:
    """The spatial tile attributes this VAE exposes, name -> int."""
    out = {}
    for name in (*_SAMPLE_MIN, *_LATENT_MIN, *_SCALED_WITH_MIN):
        value = _int_attr(vae, name)
        if value is not None:
            out[name] = value
    return out


def halved(attrs: dict, ratio: int, level: int) -> Optional[dict]:
    """``attrs`` with every spatial tile size divided by 2**level, or None under the floor / nothing to halve."""
    if level <= 0:
        return dict(attrs)
    scale = 2**level
    sample = [attrs[n] for n in _SAMPLE_MIN if n in attrs]
    latent = [attrs[n] for n in _LATENT_MIN if n in attrs]
    if not sample and not latent:
        return None
    min_latent = min([*(s // ratio for s in sample), *latent])
    if min_latent // scale < MIN_TILE_LATENTS:
        return None
    if sample and min(sample) // scale < MIN_TILE_PIXELS:
        return None
    new = {}
    for name, value in attrs.items():
        new[name] = max(1, value // scale)
    # Keep stride < tile (overlap > 0) so the stock blend still crossfades every seam.
    for axis in ("height", "width"):
        tile = new.get(f"tile_sample_min_{axis}")
        stride = new.get(f"tile_sample_stride_{axis}")
        if tile is not None and stride is not None and stride >= tile:
            new[f"tile_sample_stride_{axis}"] = max(1, tile - max(1, tile // 4))
    return new


def _set_attrs(vae: Any, values: dict) -> None:
    for name, value in values.items():
        setattr(vae, name, value)


def _release_cache(z: Any) -> None:
    try:
        import torch

        if getattr(z, "device", None) is not None and z.device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _compiling() -> bool:
    try:
        import torch

        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


def _outer_gate_owns_untiled(vae: Any) -> bool:
    """video_vae_untiled's gate turns tiling off for its own attempt and retries tiled itself."""
    return bool(getattr(vae.__dict__.get("decode"), "_unsloth_untiled_decode", False))


def install(vae: Any, logger: Any = None) -> bool:
    """Wrap ``vae._decode`` with the OOM retry ladder. Idempotent; False when the VAE has no ``_decode``."""
    if vae is None or oom_retry_disabled():
        return False
    inner = getattr(vae, "_decode", None)
    if not callable(inner):
        return False
    if getattr(inner, "_unsloth_vae_oom_retry", False):
        return True
    can_tile = callable(getattr(vae, "enable_tiling", None)) and hasattr(vae, "use_tiling")
    stock_attrs = tile_attrs(vae)
    if not can_tile or not stock_attrs:
        return False
    ratio = _spatial_ratio(vae)
    stats = {"calls": 0, "oom": 0, "tiled_retry": 0, "halvings": 0, "recovered": 0, "failed": 0}
    memo: dict = {}
    depth = [0]

    def _key(z: Any) -> tuple:
        return tuple(int(x) for x in getattr(z, "shape", ()))

    def _attempt(level: int, tiled: bool, z: Any, args: tuple, kwargs: dict) -> Any:
        prev_tiling = bool(getattr(vae, "use_tiling", False))
        current = tile_attrs(vae)
        values = halved(stock_attrs, ratio, level) if level else None
        vae._unsloth_vae_oom_halvings = level
        try:
            if tiled and not prev_tiling:
                vae.use_tiling = True
            if values:
                _set_attrs(vae, values)
            return inner(z, *args, **kwargs)
        finally:
            vae.use_tiling = prev_tiling
            _set_attrs(vae, current)
            vae._unsloth_vae_oom_halvings = 0

    @functools.wraps(inner)
    def _decode(z: Any, *args: Any, **kwargs: Any) -> Any:
        # Traced inside a compiled decode: a pass-through (no try/except in the graph; the OOM surfaces as before).
        # Nested (a wide-tile override decoding one tile through _decode): the outermost call owns the retry.
        if _compiling() or oom_retry_disabled() or depth[0]:
            return inner(z, *args, **kwargs)
        depth[0] += 1
        try:
            return _retrying(z, args, kwargs)
        finally:
            depth[0] -= 1

    def _retrying(z: Any, args: tuple, kwargs: dict) -> Any:
        stats["calls"] += 1
        tiled_now = bool(getattr(vae, "use_tiling", False))
        level = memo.get(_key(z), 0) if tiled_now else 0
        if level == 0:
            # The fast path: no wrapper state changes, exactly the stock call.
            try:
                return inner(z, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if not is_oom(exc) or (not tiled_now and _outer_gate_owns_untiled(vae)):
                    raise
                stats["oom"] += 1
            level = 0 if not tiled_now else 1
            tiled = True
            if not tiled_now:
                stats["tiled_retry"] += 1
        else:
            tiled = True
        while True:
            _release_cache(z)  # outside the handler: the failed attempt's tensors are gone
            if level and halved(stock_attrs, ratio, level) is None:
                stats["failed"] += 1
                raise RuntimeError(
                    "VAE decode ran out of memory even at the smallest tile "
                    f"({MIN_TILE_LATENTS} latents); free GPU memory or lower the resolution"
                )
            if level:
                stats["halvings"] = max(stats["halvings"], level)
            if logger is not None:
                logger.warning(
                    "diffusion.vae_oom: %s decode ran out of memory; retrying %s",
                    type(vae).__name__,
                    "tiled" if level == 0 else f"with tiles halved {level}x",
                )
            failed = False
            try:
                out = _attempt(level, tiled, z, args, kwargs)
            except Exception as exc:  # noqa: BLE001
                if not is_oom(exc):
                    raise
                failed = True
            if not failed:
                stats["recovered"] += 1
                if level:
                    memo[_key(z)] = level
                return out
            stats["oom"] += 1
            level += 1
            if level > MAX_HALVINGS:
                stats["failed"] += 1
                raise RuntimeError("VAE decode ran out of memory at every tile size")

    _decode._unsloth_vae_oom_retry = True
    _decode._unsloth_vae_oom_stats = stats
    vae._unsloth_vae_oom_inner = inner
    vae._unsloth_vae_oom_stats = stats
    vae._decode = _decode
    if logger is not None:
        logger.info(
            "diffusion.vae_oom: %s decode retries tiled, then with halved tiles, on out-of-memory",
            type(vae).__name__,
        )
    return True


def uninstall(vae: Any) -> None:
    inner = vae.__dict__.pop("_unsloth_vae_oom_inner", None)
    if inner is None:
        return
    if getattr(inner, "__self__", None) is vae and getattr(inner, "__func__", None) is getattr(
        type(vae), "_decode", None
    ):
        vae.__dict__.pop("_decode", None)
    else:
        vae._decode = inner
    vae.__dict__.pop("_unsloth_vae_oom_stats", None)


def oom_halving_active(vae: Any) -> bool:
    """True while a halved retry runs; wide-tile overrides then defer to the stock (halved) tiled decode."""
    return bool(getattr(vae, "__dict__", {}).get("_unsloth_vae_oom_halvings", 0))


__all__ = [
    "OOM_RETRY_ENV",
    "halved",
    "install",
    "is_oom",
    "oom_halving_active",
    "oom_retry_disabled",
    "tile_attrs",
    "uninstall",
]
