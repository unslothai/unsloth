# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decode a resident video VAE untiled per call when its estimated peak fits in free memory; tiled otherwise or on OOM."""

from __future__ import annotations

import os
from typing import Any, Optional

UNTILED_ENV = "UNSLOTH_VIDEO_VAE_UNTILED"

# Untiled fp16 decode peak per latent pixel (h x w), measured on a B200; frame count does not enter (Wan's causal cache
# decodes one latent frame at a time). Scaled by the decoder's element size: fp32 peaked at 21.1 GiB vs fp16 9.6 GiB.
# A14B: Wan2.1 VAE (8x spatial), measured <= 0.58 MiB per latent pixel at 480p / 720p.
_BYTES_PER_LATENT_PIXEL = {
    "wan2.2-ti2v-5b": 2.5 * 2**20,
    "wan2.2-t2v-a14b": 0.6 * 2**20,
}
# (spatial, temporal) upsampling to the RGB output, which Wan's decode holds about twice (per-frame torch.cat, then
# clamp) and which grows with frames: 1021 frames at 1280x704 is 5.2 GiB per fp16 copy.
_OUTPUT_SCALE = {
    "wan2.2-ti2v-5b": (16, 4),
    "wan2.2-t2v-a14b": (8, 4),
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
    coef = _BYTES_PER_LATENT_PIXEL.get(family)
    if coef is None or len(latent_shape) != 5:
        return None
    batch, _, latent_frames, height, width = (int(x) for x in latent_shape)
    batch, itemsize = max(1, batch), max(2, itemsize)
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
    if family not in _BYTES_PER_LATENT_PIXEL or not getattr(vae, "use_tiling", False):
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
            if vae.__dict__.get("decode") is not untiled_when_fits:
                vae.decode = untiled_when_fits

    untiled_when_fits._unsloth_untiled_decode = True
    untiled_when_fits._unsloth_untiled_stats = stats
    vae.decode = untiled_when_fits
    if logger is not None:
        logger.info("video.vae_untiled: resident %s VAE decodes untiled when it fits", family)
    return True
