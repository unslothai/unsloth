# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Retry an OOM VAE decode in smaller tiles instead of failing the render.

Wraps the instance ``_decode`` (under the decode-level wrappers, over the tiled decode). Untiled OOM -> retry tiled;
tiled OOM -> halve the exposed spatial tile geometry (whole latents, overlap fraction kept so stock seams blend)
down to an 8-latent floor. Attributes are restored after the call; the working level is remembered per latent shape.
Kill switch ``UNSLOTH_VAE_OOM_RETRY=0``.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Any, Optional

OOM_RETRY_ENV = "UNSLOTH_VAE_OOM_RETRY"
# Under 8 latents a tile's decode is mostly edge: HunyuanImage-2.1 (32x) at 4-latent tiles measured LPIPS 0.46 against
# the untiled decode, 8-latent LTX-2 (32x) tiles 0.019. A render that cannot fit 8-latent tiles fails as before.
MIN_TILE_LATENTS = 8
MAX_HALVINGS = 6
# Halved tiles must decode a corner crop this close to the untiled crop (FLUX.1 16-latent tiles measured 36.8 dB on a
# 2048 px image, LTX-2.5 8-latent tiles 33.5, HunyuanImage-2.1 8-latent tiles 26.7 at LPIPS 0.19).
MIN_PSNR_DB = 30.0
# Tiles this large were never seen to seam measurably; smaller ones need the probe to have run.
UNPROBED_MIN_LATENTS = 16

_AXES = ("size", "height", "width")


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
    cfg = getattr(vae, "config", None)
    for source in (vae.__dict__, cfg if isinstance(cfg, dict) else getattr(cfg, "__dict__", {})):
        r = source.get("spatial_compression_ratio") if hasattr(source, "get") else None
        if isinstance(r, int) and r > 0:
            return r
    try:
        r = getattr(type(vae), "spatial_compression_ratio", None)
        if isinstance(r, int) and r > 0:
            return r
        boc = (
            cfg.get("block_out_channels")
            if hasattr(cfg, "get")
            else getattr(cfg, "block_out_channels", None)
        )
        if boc:
            return 2 ** (len(boc) - 1)
    except Exception:  # noqa: BLE001
        pass
    for axis in _AXES:
        sample, latent = (
            _int_attr(vae, f"tile_sample_min_{axis}"),
            _int_attr(vae, f"tile_latent_min_{axis}"),
        )
        if sample and latent and sample % latent == 0:
            return sample // latent
    return 8


def tile_attrs(vae: Any) -> dict:
    out = {}
    for axis in _AXES:
        for kind in (
            "tile_sample_min",
            "tile_latent_min",
            "tile_sample_stride",
            "tile_sample_min_overlap",
        ):
            name = f"{kind}_{axis}"
            value = _int_attr(vae, name)
            if value is not None:
                out[name] = value
    return out


def _overlap_ok(latent: int, sample: int, factor: Optional[float]) -> bool:
    """``tile_overlap_factor`` VAEs (AutoencoderKL, HunyuanImage) step ``int(L * (1 - f))`` latents and keep
    ``S - int(S * f)`` pixels per tile; the two only tile the image exactly when both products are whole."""
    if factor is None:
        return True
    step = latent * (1 - factor)
    blend = sample * factor
    return abs(step - round(step)) < 1e-9 and abs(blend - round(blend)) < 1e-9 and round(step) > 0


def _axis_latents(
    base_latent: int, ratio: int, level: int, overlap_factor: Optional[float]
) -> Optional[int]:
    """One axis's tile size in latents at ``level``: the largest valid size at or under the halved target, else the
    smallest valid one above it, always under ``level - 1``'s size and at least MIN_TILE_LATENTS. None when there is
    none (an overlap factor allows only some sizes to tile exactly)."""
    if level <= 0:
        return base_latent
    prev = _axis_latents(base_latent, ratio, level - 1, overlap_factor)
    if prev is None:
        return None
    target = max(MIN_TILE_LATENTS, base_latent // 2**level)  # the last halving lands on the floor
    for lat in (*range(min(target, prev - 1), MIN_TILE_LATENTS - 1, -1), *range(target + 1, prev)):
        if _overlap_ok(lat, lat * ratio, overlap_factor):
            return lat
    return None


def halved(
    attrs: dict,
    ratio: int,
    level: int,
    overlap_factor: Optional[float] = None,
) -> Optional[dict]:
    """``attrs`` with the tile geometry halved ``level`` times, every size a whole number of latents. An axis already
    at the floor keeps its last size while the others keep halving; None when no axis gets smaller at ``level``."""
    if level <= 0:
        return dict(attrs)
    new = dict(attrs)
    touched = False
    for axis in _AXES:
        s_name, l_name = f"tile_sample_min_{axis}", f"tile_latent_min_{axis}"
        sample, latent = attrs.get(s_name), attrs.get(l_name)
        if sample is None and latent is None:
            continue
        base_latent = latent if latent is not None else sample // ratio
        axis_level, lat = level, None
        while axis_level > 0:
            lat = _axis_latents(base_latent, ratio, axis_level, overlap_factor)
            if lat is not None:
                break
            axis_level -= 1
        if lat is None:
            continue  # this axis cannot shrink at all: stock geometry
        touched = touched or axis_level == level
        scale = 2**axis_level
        if sample is not None:
            new[s_name] = lat * ratio
        if latent is not None:
            new[l_name] = lat

        stride_name = f"tile_sample_stride_{axis}"
        if stride_name in attrs:
            stride_lat = max(1, attrs[stride_name] // scale // ratio)
            if stride_lat >= lat:
                stride_lat = max(1, lat - max(1, lat // 4))
            new[stride_name] = stride_lat * ratio
        overlap_name = f"tile_sample_min_overlap_{axis}"
        if overlap_name in attrs:
            ov_lat = max(1, attrs[overlap_name] // scale // ratio)
            new[overlap_name] = min(ov_lat, max(1, lat // 2)) * ratio
    return new if touched else None


def _set_attrs(vae: Any, values: dict) -> None:
    for name, value in values.items():
        setattr(vae, name, value)


def _release_cache(z: Any, vae: Any = None) -> None:
    try:
        # A causal decoder's frame cache (Wan / Qwen-Image) still holds the failed attempt's tensors.
        clear = getattr(vae, "clear_cache", None)
        if callable(clear):
            clear()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        if getattr(z, "device", None) is not None and z.device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _free_bytes(z: Any) -> Optional[int]:
    try:
        import torch

        if z.device.type != "cuda":
            return None
        free, _ = torch.cuda.mem_get_info(z.device)
        return int(
            free + torch.cuda.memory_reserved(z.device) - torch.cuda.memory_allocated(z.device)
        )
    except Exception:  # noqa: BLE001
        return None


def _compiling() -> bool:
    try:
        import torch
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


def _outer_gate_owns_untiled(vae: Any) -> bool:
    """video_vae_untiled's gate turns tiling off for its own attempt and retries tiled itself. Outer wrappers (the ROCm
    bf16 decode) expose it only through ``__wrapped__``."""
    fn = vae.__dict__.get("decode")
    for _ in range(8):
        if fn is None:
            return False
        if getattr(fn, "_unsloth_untiled_decode", False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def _decoded(out: Any) -> Any:
    if isinstance(out, (tuple, list)):
        return out[0] if out else None
    return getattr(out, "sample", out)


def install(vae: Any, logger: Any = None) -> bool:
    """Wrap ``vae._decode`` with the OOM retry ladder. Idempotent; False when the VAE cannot tile."""
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
    factor = vae.__dict__.get("tile_overlap_factor")
    factor = (
        float(factor) if isinstance(factor, (int, float)) and not isinstance(factor, bool) else None
    )
    stats = {"calls": 0, "oom": 0, "tiled_retry": 0, "halvings": 0, "recovered": 0, "failed": 0,
             "quality_refused": 0, "probe_psnr": {}}  # fmt: skip
    memo: dict = {}
    quality: dict = {}
    depth = [0]

    def _key(z: Any) -> tuple:
        return tuple(int(x) for x in getattr(z, "shape", ()))

    def _attempt(level: int, z: Any, args: tuple, kwargs: dict) -> Any:
        prev_tiling = bool(getattr(vae, "use_tiling", False))
        current = tile_attrs(vae)
        vae._unsloth_vae_oom_halvings = level
        try:
            vae.use_tiling = True
            if level:
                _set_attrs(vae, halved(stock_attrs, ratio, level, factor) or {})
            return inner(z, *args, **kwargs)
        finally:
            vae.use_tiling = prev_tiling
            _set_attrs(vae, current)
            vae._unsloth_vae_oom_halvings = 0

    def _level_quality_ok(level: int, z: Any, args: tuple, kwargs: dict) -> Optional[bool]:
        """Halved tiles decode a corner crop (2 tiles a side, so one seam each way; 2 frames) within MIN_PSNR_DB of the
        untiled crop. Measured once per level. When the untiled crop itself runs out of memory, only tiles of at least
        UNPROBED_MIN_LATENTS are allowed. None: the halved crop came back the wrong size (level unusable).
        The verdict is kept per (latent shape, level): the probe measures how that shape's tile geometry blends."""
        qkey = (_key(z), level)
        if qkey in quality:
            return quality[qkey]
        values = halved(stock_attrs, ratio, level, factor) or {}

        def axis_latents(axis: str) -> int:
            # ``*_size`` attributes tile both axes
            found = [
                values[f"tile_sample_min_{a}"] // ratio
                for a in (axis, "size")
                if f"tile_sample_min_{a}" in values
            ] + [
                values[f"tile_latent_min_{a}"]
                for a in (axis, "size")
                if f"tile_latent_min_{a}" in values
            ]
            return min(found) if found else MIN_TILE_LATENTS

        lat_h, lat_w = axis_latents("height"), axis_latents("width")
        # The unprobed floor only looks at axes this input actually tiles (an axis it fits in has no seam).
        seamed = [n for n, size in ((lat_h, int(z.shape[-2])), (lat_w, int(z.shape[-1]))) if size > n]
        lat = min(seamed) if seamed else UNPROBED_MIN_LATENTS
        sl = [slice(None)] * z.dim()
        sl[0] = slice(0, 1)  # one sample: seam quality is per sample, and the untiled reference must not scale with batch
        # two tiles along each axis, so each axis gets a seam
        sl[-1] = slice(0, min(int(z.shape[-1]), 2 * lat_w))
        sl[-2] = slice(0, min(int(z.shape[-2]), 2 * lat_h))
        if z.dim() == 5:
            sl[2] = slice(0, min(int(z.shape[2]), 2))
        crop = z[tuple(sl)]
        prev_tiling = bool(getattr(vae, "use_tiling", False))
        try:
            import torch
            with torch.no_grad():
                vae.use_tiling = False
                try:
                    ref = _decoded(inner(crop, *args, **kwargs)).float()
                finally:
                    vae.use_tiling = prev_tiling
                tiled = _decoded(_attempt(level, crop, args, kwargs)).float()
                if tuple(tiled.shape) != tuple(ref.shape):
                    mse = None  # the halved tiles do not cover the crop: this level is unusable
                else:
                    mse = float((tiled - ref).pow(2).mean())
            del ref, tiled
        except Exception as exc:  # noqa: BLE001
            if not is_oom(exc):
                raise
            _release_cache(z, vae)
            stats["probe_psnr"][level] = None
            if lat < UNPROBED_MIN_LATENTS:
                return False  # not cached: with more memory free the probe may run next time
            quality[qkey] = (
                True  # a remembered working level must not rerun a probe known to run out of memory
            )
            return True
        if mse is None:
            quality[qkey] = None
            stats["probe_psnr"][level] = "shape"
            _release_cache(z, vae)
            return None
        psnr = 99.0 if mse == 0 else 10.0 * math.log10(4.0 / mse)
        quality[qkey] = psnr >= MIN_PSNR_DB
        stats["probe_psnr"][level] = round(psnr, 1)
        _release_cache(z, vae)
        return quality[qkey]

    def _shape_ok(out: Any, z: Any) -> bool:
        dec = _decoded(out)
        shape = getattr(dec, "shape", None)
        if shape is None or len(shape) < 2:
            return True
        return tuple(shape[-2:]) == (int(z.shape[-2]) * ratio, int(z.shape[-1]) * ratio)

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
        # Tiled and untiled callers of one shape fail at different sizes: keyed apart.
        mkey = (_key(z), tiled_now)
        level_free = memo.get(mkey)
        start: Optional[int] = (
            None  # None: no remembered level; 0 is a remembered stock-tile recovery
        )
        if level_free is not None:
            free = _free_bytes(z)
            # More memory than when the caller's own decode failed: try it again rather than keep the fallback.
            if free is not None and level_free[1] is not None and free > 1.25 * level_free[1]:
                memo.pop(mkey, None)
            else:
                start = level_free[0]
        if start is None:
            try:
                return inner(z, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if not is_oom(exc) or (not tiled_now and _outer_gate_owns_untiled(vae)):
                    raise
                stats["oom"] += 1
            free_at_fail = _free_bytes(z)
            # Untiled failed: the stock tiles first (level 0, tiling on). Tiled failed: halve.
            levels = [0] if not tiled_now else []
            levels += list(range(1, MAX_HALVINGS + 1))
            if not tiled_now:
                stats["tiled_retry"] += 1
        elif start == 0:
            free_at_fail = level_free[1]
            levels = list(range(0, MAX_HALVINGS + 1))
        else:
            free_at_fail = level_free[1]
            levels = list(range(start, MAX_HALVINGS + 1)) + [0]
        lowest = start or 0
        error = "VAE decode ran out of memory at every tile size; free GPU memory or lower the resolution"
        for level in levels:
            if level and halved(stock_attrs, ratio, level, factor) is None:
                if level > lowest:
                    break
                continue
            # Outside the handler, and before the probe: the failed attempt's tensors (and a causal decoder's frame
            # cache) are gone, so the probe does not run out of memory on them.
            _release_cache(z, vae)
            if level:
                usable = _level_quality_ok(level, z, args, kwargs)
                if usable is None:
                    continue  # the probe crop came back the wrong size: try the next level
                if not usable:
                    # Smaller tiles only get worse: fail as the stock decode would have.
                    stats["quality_refused"] += 1
                    error = (
                        f"VAE decode ran out of memory, and tiles small enough to fit decode {type(vae).__name__} "
                        "with visible seams; free GPU memory or lower the resolution"
                    )
                    break
            if logger is not None:
                logger.warning(
                    "diffusion.vae_oom: %s decode ran out of memory; retrying %s",
                    type(vae).__name__,
                    "tiled" if level == 0 else f"with tiles halved {level}x",
                )
            failed = False
            try:
                out = _attempt(level, z, args, kwargs)
            except Exception as exc:  # noqa: BLE001
                if not is_oom(exc):
                    raise
                failed = True
            if not failed and _shape_ok(out, z):
                stats["recovered"] += 1
                stats["halvings"] = max(stats["halvings"], level)
                if level or not tiled_now:
                    # an untiled caller that recovered through the stock tiles goes straight to them next time
                    memo[mkey] = (level, free_at_fail)
                else:
                    memo.pop(mkey, None)
                return out
            if not failed:
                del out  # this VAE's stock tiling does not cover the latent at that size
                continue
            stats["oom"] += 1
        stats["failed"] += 1
        _release_cache(z, vae)
        raise RuntimeError(error)

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
