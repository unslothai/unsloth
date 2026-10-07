# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HunyuanVideo-1.5 VAE: build the causal attention mask with one masked_fill instead of a Python loop.

The stock per-row loop unrolls under compile into a graph inductor cannot schedule (RecursionError in
``reorder_for_peak_memory``). Same values, bit-identical decode.
"""

from __future__ import annotations

import inspect
from typing import Any

_ATTN_BLOCK = "HunyuanVideo15AttnBlock"
_VAE_CLASS = "AutoencoderKLHunyuanVideo15"
_STOCK_NEEDLES = ("for i in range(seq_len)", "mask[i, : (i_frame + 1) * n_hw] = 0")


from .diffusion_vae_fused import _hv_causal_mask as causal_attention_mask  # noqa: E402


def _stock_loop(cls: type) -> bool:
    fn = cls.__dict__.get("prepare_causal_attention_mask")
    fn = getattr(fn, "__func__", fn)
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return all(n in src for n in _STOCK_NEEDLES)


def install_vectorised_causal_mask(pipe: Any, logger: Any = None) -> int:
    """Patch each HV1.5 VAE attention block instance of ``pipe``; returns the count patched. Never raises."""
    vae = getattr(pipe, "vae", None)
    if vae is None or type(vae).__name__ != _VAE_CLASS:
        return 0
    patched = 0
    try:
        for module in vae.modules():
            cls = type(module)
            if cls.__name__ != _ATTN_BLOCK or not _stock_loop(cls):
                continue
            # A plain function on the instance is unbound, so it gets the staticmethod's arguments.
            module.prepare_causal_attention_mask = causal_attention_mask
            patched += 1
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.warning("video.hv15_vae_mask: install failed, keeping the stock mask: %s", exc)
        return 0
    if patched and logger is not None:
        logger.info(
            "video.hv15_vae_mask: vectorised causal mask on %d VAE attention block(s)", patched
        )
    return patched
