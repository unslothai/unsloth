# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decode a resident video VAE in one piece when the untiled decode fits.

Every conventional video load turns VAE tiling on, whatever the card, so a B200 with 150 GB free decodes a
1280x704 Wan clip in overlapping 256 px tiles: the overlaps are decoded twice and the seams are blended. With the
whole pipeline resident there is usually room for the plain decode, which is faster and is the decoder's own
output rather than a blend of tiles.

Per call, from the latent shape: the untiled decode's extra memory is estimated from a measured per-family
coefficient, and the call runs untiled only when that estimate (plus a margin) fits in the memory free right
now. Otherwise, or if the untiled decode still runs out of memory, the call decodes tiled exactly as before.
Families without a measured coefficient keep tiling.
"""

from __future__ import annotations

import os
from typing import Any, Optional

UNTILED_ENV = "UNSLOTH_VIDEO_VAE_UNTILED"

# Extra device memory of an untiled decode, in bytes per LATENT pixel (h x w), measured on a B200 at each family's
# default shape with Studio's own fp16 decode path, then rounded up; scaled by the decoder's element size, since an
# fp32 decode (ROCm, no fp16 path, or after its non-finite fallback) measured 21.1 GiB against fp16's 9.6 GiB at
# 1280x704. Wan's decoder runs one latent frame at a time (causal cache), so its peak does not grow with the clip length.
_BYTES_PER_LATENT_PIXEL = {
    "wan2.2-ti2v-5b": 2.5 * 2**20,
}
# Headroom on top of the estimate: allocator fragmentation and the output tensor itself.
_MARGIN = 1.25
_MARGIN_BYTES = 2 * 2**30


def _enabled() -> bool:
    return os.environ.get(UNTILED_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def untiled_decode_bytes(
    family: str,
    latent_shape: tuple,
    itemsize: int = 2,
) -> Optional[int]:
    """Estimated extra memory of an untiled decode of ``latent_shape`` (B, C, T, h, w) by a decoder whose weights
    are ``itemsize`` bytes wide, or None if unmeasured."""
    coef = _BYTES_PER_LATENT_PIXEL.get(family)
    if coef is None or len(latent_shape) != 5:
        return None
    batch, _, _, height, width = (int(x) for x in latent_shape)
    return int(coef * max(2, itemsize) / 2 * max(1, batch) * height * width)


def _decoder_itemsize(vae: Any) -> int:
    """Element size of the decoder's weights, read per call; 4 (fp32, the larger estimate) if unreadable."""
    try:
        for param in vae.decoder.parameters():
            if param.is_floating_point():
                return int(param.element_size())
    except Exception:  # noqa: BLE001 -- unreadable means assume the wider dtype
        pass
    return 4


def _free_bytes(device: Any) -> Optional[int]:
    """Memory an allocation can use now: free on the device plus what torch's cache holds unused."""
    try:
        import torch

        free, _ = torch.cuda.mem_get_info(device)
        cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return int(free) + max(0, int(cached))
    except Exception:  # noqa: BLE001 -- no reading means no untiled decode
        return None


def install_untiled_decode(
    pipe: Any,
    family: str,
    *,
    logger: Any = None,
) -> bool:
    """Wrap ``pipe.vae.decode`` so each call decodes untiled when it fits. Returns whether it was installed.

    Only for a CUDA-resident pipeline whose VAE is tiled and whose family has a measured coefficient; the caller
    gates on placement and speed tier.
    """
    vae = getattr(pipe, "vae", None)
    decode = getattr(vae, "decode", None)
    if not _enabled() or vae is None or not callable(decode):
        return False
    if family not in _BYTES_PER_LATENT_PIXEL or not getattr(vae, "use_tiling", False):
        return False
    if getattr(decode, "_unsloth_untiled_decode", False):
        return True
    stats = {"untiled": 0, "tiled": 0, "oom_fallback": 0}

    def untiled_when_fits(z: Any, *args: Any, **kwargs: Any) -> Any:
        if not getattr(vae, "use_tiling", False):
            return decode(z, *args, **kwargs)
        need = untiled_decode_bytes(
            family, tuple(getattr(z, "shape", ())), itemsize = _decoder_itemsize(vae)
        )
        free = _free_bytes(getattr(z, "device", None))
        fits = need is not None and free is not None and need * _MARGIN + _MARGIN_BYTES <= free
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

        vae.use_tiling = False
        try:
            out = decode(z, *args, **kwargs)
            stats["untiled"] += 1
            return out
        except torch.cuda.OutOfMemoryError:  # torch.OutOfMemoryError only exists from torch 2.5
            pass
        finally:
            vae.use_tiling = True
        # Retried outside the handler so the failed attempt's tensors (held by the traceback) are gone first.
        stats["oom_fallback"] += 1
        torch.cuda.empty_cache()
        if logger is not None:
            logger.warning("video.vae_untiled: untiled decode ran out of memory; decoding tiled")
        return decode(z, *args, **kwargs)

    untiled_when_fits._unsloth_untiled_decode = True
    untiled_when_fits._unsloth_untiled_stats = stats
    vae.decode = untiled_when_fits
    if logger is not None:
        logger.info("video.vae_untiled: resident %s VAE decodes untiled when it fits", family)
    return True
