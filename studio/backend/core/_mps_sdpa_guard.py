# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Apple Silicon: split large SDPA calls that do not take the MPS prefill kernel into query-row chunks.

Below torch 2.13, MPS SDPA always runs the MPSGraph path, which builds the full score matrix and returns wrong values
(cosine 0.000 against fp64) from about 0.9 * 2^29 score elements per call on macOS 15 (pytorch#179352). New Apple
Silicon installs take torch 2.13, whose prefill kernel never builds the scores, but existing homes keep their torch.
Each query row of an attention call is independent, so running at most MPS_SCORE_ELEMENTS scores per call and
concatenating the rows is the same computation; it also caps the score memory. Calls the prefill kernel serves, calls
under the cap and every non-MPS call go to SDPA unchanged. ``UNSLOTH_MPS_ATTENTION_GUARD=0`` turns it off."""

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Any, Optional

# Same bound as Qwen-Image-2.1's bounded attention (diffusion_qwenimage21_math.MPS_SCORE_ELEMENTS): half the size
# where macOS 15 / torch 2.11 SDPA starts returning wrong values.
MPS_SCORE_ELEMENTS = 2**28
GUARD_ENV = "UNSLOTH_MPS_ATTENTION_GUARD"
_GUARD_ATTR = "_unsloth_mps_large_attention_guard"
_ARG_NAMES = ("attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")

logger = logging.getLogger(__name__)

# Calls split into chunks since the guard was installed (a test / status hook).
stats = {"chunked_calls": 0}


def _is_mps(t: Any) -> bool:
    return getattr(getattr(t, "device", None), "type", None) == "mps"


def _prefill_serves(query: Any, key: Any, value: Any, attn_mask: Any) -> bool:
    try:
        from core.inference.diffusion_attention import mps_prefill_attention
        return bool(mps_prefill_attention(query, key, value, attn_mask, seq_dim = -2))
    except Exception:  # noqa: BLE001 -- cannot tell: chunking is exact, so split
        return False


def _compiling() -> bool:
    try:
        import torch
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001 -- torch without torch.compiler: never tracing
        return False


def _broadcast_lead(a: tuple, b: tuple) -> Any:
    """Product of Q's and K's broadcast leading (batch, head) dims: SDPA builds that many score matrices (a mask must
    broadcast to them). Plain loops, no ``int()`` or generator, so Dynamo traces it with sizes kept symbolic."""
    if len(a) < len(b):
        a, b = b, a
    b = (1,) * (len(a) - len(b)) + tuple(b)
    lead = 1
    for x, y in zip(a, b):
        lead = lead * (x if x >= y else y)  # broadcast (one side 1), or GQA (Q has the more heads)
    return lead


def chunk_rows(query: Any, key: Any) -> Optional[int]:
    """Query rows per call when ``query`` x ``key`` exceeds the score cap, else None (one call)."""
    q_shape, k_shape = tuple(query.shape), tuple(key.shape)
    if len(q_shape) < 2 or len(k_shape) < 2:
        return None
    sq, sk = q_shape[-2], k_shape[-2]
    lead = _broadcast_lead(q_shape[:-2], k_shape[:-2])
    if sq <= 1 or lead * sq * sk <= MPS_SCORE_ELEMENTS:
        return None
    return max(1, MPS_SCORE_ELEMENTS // max(lead * sk, 1))


def _split(sdpa: Any, query: Any, key: Any, value: Any, rows: int, kwargs: dict) -> Any:
    import torch

    sq, sk = query.shape[-2], key.shape[-2]
    mask = kwargs.pop("attn_mask", None)
    causal = bool(kwargs.pop("is_causal", False))
    sliced_mask = mask is not None and mask.dim() >= 2 and mask.shape[-2] != 1
    outputs = []
    for start in range(0, sq, rows):
        end = min(start + rows, sq)
        part_mask = mask[..., start:end, :] if sliced_mask else mask
        if causal:
            # SDPA's is_causal is top-left aligned: query row i sees keys 0..i.
            part_mask = (
                torch.arange(sk, device = query.device)[None, :]
                <= torch.arange(start, end, device = query.device)[:, None]
            )
        outputs.append(sdpa(query[..., start:end, :], key, value, attn_mask = part_mask, **kwargs))
    return torch.cat(outputs, dim = -2)


def guard_sdpa(sdpa: Any) -> Any:
    """``sdpa`` with every MPS call past the score cap that the prefill kernel does not take split by query rows."""
    if getattr(sdpa, _GUARD_ATTR, False):
        return sdpa

    @functools.wraps(sdpa)
    def guarded(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_mps(query) or getattr(query, "is_nested", False) or len(args) > len(_ARG_NAMES):
            return sdpa(query, key, value, *args, **kwargs)
        bound = dict(zip(_ARG_NAMES, args))
        for name in bound:
            if name in kwargs:
                return sdpa(query, key, value, *args, **kwargs)
        bound.update(kwargs)
        mask = bound.get("attn_mask")
        rows = chunk_rows(query, key)
        if rows is None:
            return sdpa(query, key, value, *args, **kwargs)
        if mask is not None and (
            bool(bound.get("is_causal", False))
            or mask.dim() < 2
            or (mask.shape[-2] != 1 and mask.shape[-2] != query.shape[-2])
        ):
            return sdpa(query, key, value, *args, **kwargs)
        if _prefill_serves(query, key, value, mask):
            return sdpa(query, key, value, *args, **kwargs)
        if (
            not _compiling()
        ):  # a traced counter bump becomes a guard on its value: one recompile per call
            stats["chunked_calls"] += 1
        return _split(sdpa, query, key, value, rows, bound)

    setattr(guarded, _GUARD_ATTR, True)
    guarded.__wrapped__ = sdpa  # type: ignore[attr-defined]
    return guarded


def _disabled() -> bool:
    return os.environ.get(GUARD_ENV, "").strip().lower() in ("0", "false", "no", "off")


def install_mps_large_attention_guard(log: Any = None) -> bool:
    """Wrap ``torch.nn.functional.scaled_dot_product_attention`` on an Apple Silicon host with MPS; idempotent; never
    raises. Callers look the function up per call (diffusers' native backend, its SDPA processors, transformers), so
    models loaded later are covered. True when the guard is in place."""
    if sys.platform != "darwin" or _disabled():
        return False
    try:
        import torch
        import torch.nn.functional as F

        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            return False
        if getattr(F.scaled_dot_product_attention, _GUARD_ATTR, False):
            return True
        F.scaled_dot_product_attention = guard_sdpa(F.scaled_dot_product_attention)
        (log or logger).info(
            "Apple Silicon: MPS attention calls past %d score elements without the prefill kernel run in "
            "query chunks (torch %s)",
            MPS_SCORE_ELEMENTS,
            torch.__version__,
        )
        return True
    except Exception:  # noqa: BLE001 -- stock SDPA stays rather than failing the load
        logger.debug("MPS large-attention guard not installed", exc_info = True)
        return False
