# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HunyuanVideo-1.5 VAE: build the causal attention mask with one masked_fill instead of a Python loop.

Diffusers' ``HunyuanVideo15AttnBlock.prepare_causal_attention_mask`` writes the mask row by row, ``seq_len`` slice
assignments per call (7,936 per 832x480 tile, 48,360 untiled), so an eager decode spends 0.46 of 5.13 s launching
them and a compiled decoder unrolls them into one graph that inductor cannot schedule (RecursionError in
``reorder_for_peak_memory``). Row ``i`` may attend to column ``j`` iff frame(j) <= frame(i): the same values,
bit-identical decode.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

_ATTN_BLOCK = "HunyuanVideo15AttnBlock"
_VAE_CLASS = "AutoencoderKLHunyuanVideo15"
# The stock body this replaces; a diffusers release that rewrites it keeps its own version.
_STOCK_NEEDLES = ("for i in range(seq_len)", "mask[i, : (i_frame + 1) * n_hw] = 0")


def causal_attention_mask(
    n_frame: int,
    n_hw: int,
    dtype: Any,
    device: Any,
    batch_size: Optional[int] = None,
) -> Any:
    """``prepare_causal_attention_mask`` without the per-row loop: 0 where frame(col) <= frame(row), else -inf.

    The predicate is built per FRAME pair (n_frame x n_frame) and broadcast over the (n_hw, n_hw) blocks of a 4-D view
    of the output, so the only seq_len x seq_len allocation is the mask itself, the same peak as the stock loop."""
    import torch

    seq_len = n_frame * n_hw
    mask = torch.zeros((seq_len, seq_len), dtype = dtype, device = device)
    later = torch.ones((n_frame, n_frame), dtype = torch.bool, device = device).triu_(1)
    mask.view(n_frame, n_hw, n_frame, n_hw).masked_fill_(later[:, None, :, None], float("-inf"))
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


def _stock_loop(cls: type) -> bool:
    fn = cls.__dict__.get("prepare_causal_attention_mask")
    fn = getattr(fn, "__func__", fn)
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return all(n in src for n in _STOCK_NEEDLES)


def install_vectorised_causal_mask(pipe: Any, logger: Any = None) -> int:
    """Point every HV1.5 VAE attention block of ``pipe.vae`` at the vectorised mask. Per instance, so nothing
    outside this pipe changes. Returns the number of blocks patched (0 for other VAEs). Never raises."""
    vae = getattr(pipe, "vae", None)
    if vae is None or type(vae).__name__ != _VAE_CLASS:
        return 0
    patched = 0
    try:
        for module in vae.modules():
            cls = type(module)
            if cls.__name__ != _ATTN_BLOCK or not _stock_loop(cls):
                continue
            # A plain function on the instance is not bound, so ``self.prepare_causal_attention_mask(...)`` calls it
            # with the staticmethod's own arguments.
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
