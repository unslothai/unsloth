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
    """Device free memory plus torch's unused cache."""
    try:
        import torch

        free, _ = torch.cuda.mem_get_info(device)
        cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return int(free) + max(0, int(cached))
    except Exception:  # noqa: BLE001
        return None


def _in_chain(fn: Any, target: Any) -> bool:
    for _ in range(16):
        if fn is None:
            return False
        if fn is target:
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


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
    stats = {"untiled": 0, "tiled": 0, "oom_fallback": 0}
    # Smallest estimate that ran out of memory; never retried untiled.
    oom_need: list = []

    def gated(z: Any, *args: Any, **kwargs: Any) -> Any:
        if not getattr(vae, "use_tiling", False):
            return decode(z, *args, **kwargs)
        need = untiled_decode_bytes(
            family, tuple(getattr(z, "shape", ())), itemsize = _decoder_itemsize(vae)
        )
        free = _free_bytes(getattr(z, "device", None))
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
