# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Give FLUX.1's first single block the slice layout of blocks 1..37 so all 38 share one compiled graph.

Block 0 gets two separate tensors; the rest get two slices of one ``[B, text + image, D]`` buffer. Dynamo guards on
strides, so block 0 otherwise compiles a second graph ("stride mismatch at index 0"). The block concatenates its
inputs first, so copying them into one buffer is value-identical. ``UNSLOTH_DIFFUSION_BLOCK_RESTRIDE=0`` disables it.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

_ENV = "UNSLOTH_DIFFUSION_BLOCK_RESTRIDE"
_FALSE = ("0", "false", "no", "off")
_TARGETS = {
    "FluxTransformer2DModel": ("single_transformer_blocks", "FluxSingleTransformerBlock"),
}
_MARK = "_unsloth_block_restride"


def enabled() -> bool:
    return (os.environ.get(_ENV) or "").strip().lower() not in _FALSE


def _takes_both_streams(block: Any) -> bool:
    try:
        params = inspect.signature(type(block).forward).parameters
    except (TypeError, ValueError):
        return False
    return "hidden_states" in params and "encoder_hidden_states" in params


def _restrided(hidden: Any, encoder: Any) -> Any:
    """``(hidden, encoder)`` as slices of one buffer, or None when the pair is not the shape this targets."""
    import torch

    if not (isinstance(hidden, torch.Tensor) and isinstance(encoder, torch.Tensor)):
        return None
    if hidden.dim() != 3 or encoder.dim() != 3:
        return None
    if hidden.shape[0] != encoder.shape[0] or hidden.shape[2] != encoder.shape[2]:
        return None
    if hidden.dtype != encoder.dtype or hidden.device != encoder.device:
        return None
    if hidden.requires_grad or encoder.requires_grad:
        return None
    batch, text, dim = encoder.shape
    image = hidden.shape[1]
    if hidden.stride() == ((text + image) * dim, dim, 1) and encoder.stride() == hidden.stride():
        return None
    buf = torch.empty((batch, text + image, dim), dtype = hidden.dtype, device = hidden.device)
    buf[:, :text].copy_(encoder)
    buf[:, text:].copy_(hidden)
    return buf[:, text:], buf[:, :text]


def wrap(compiled: Any) -> Any:
    """``compiled`` behind the restride: its block-0 inputs arrive as slices of one buffer."""

    def restride(*args: Any, **kwargs: Any) -> Any:
        if not args and "hidden_states" in kwargs and "encoder_hidden_states" in kwargs:
            pair = _restrided(kwargs["hidden_states"], kwargs["encoder_hidden_states"])
            if pair is not None:
                kwargs = dict(kwargs)
                kwargs["hidden_states"], kwargs["encoder_hidden_states"] = pair
        return compiled(*args, **kwargs)

    setattr(restride, _MARK, True)
    return restride


def is_wrapped(fn: Any) -> bool:
    return bool(getattr(fn, _MARK, False))


def install(transformer: Any, logger: Any = None) -> bool:
    """Wrap the first single block's compiled call. Idempotent; True when installed. Never raises."""
    if not enabled():
        return False
    try:
        target = _TARGETS.get(type(transformer).__name__)
        if target is None:
            return False
        attr, block_cls = target
        blocks = getattr(transformer, attr, None)
        if blocks is None or len(blocks) < 2:
            return False
        first = blocks[0]
        if type(first).__name__ != block_cls or not _takes_both_streams(first):
            return False
        compiled = getattr(first, "_compiled_call_impl", None)
        if compiled is None:
            return False
        if getattr(compiled, _MARK, False):
            return True
        restride = wrap(compiled)
        # Keep the guard's identity visible: guard_compiled_blocks skips modules already wrapped.
        guard = getattr(compiled, "_unsloth_compile_guard", None)
        if guard is not None:
            restride._unsloth_compile_guard = guard  # type: ignore[attr-defined]
        first._compiled_call_impl = restride
        if logger is not None:
            logger.info(
                "diffusion.block_restride: %s[0] gets the slice layout of blocks 1..%d (one compiled graph)",
                attr,
                len(blocks) - 1,
            )
        return True
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.warning("diffusion.block_restride: skipped (%s: %s)", type(exc).__name__, exc)
        return False
