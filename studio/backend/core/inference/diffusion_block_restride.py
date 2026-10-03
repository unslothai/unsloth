# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hand FLUX.1's first single-stream block the same input layout as the other 37, so it reuses their compiled graph.

Diffusers' ``FluxSingleTransformerBlock`` returns ``hidden_states`` and ``encoder_hidden_states`` as two slices of one
``[B, text + image, D]`` tensor, so blocks 2..38 see a batch stride of ``(text + image) * D``. Block 1 gets the double
stream's two separate tensors instead, whose batch stride is ``image * D`` and ``text * D``. Dynamo guards on every
stride, so the regional compile traces, lowers and autotunes the single block TWICE: a whole extra graph on the first
render after every start (FLUX ``1/1`` recompile: "stride mismatch at index 0").

Block 1's two inputs are copied into one ``[B, text + image, D]`` buffer here, text first, the order the block
concatenates them in, and it is handed the two slices: the layout every later block already sees. The block reads
the same values (it concatenates the two inputs before touching them), one graph serves all 38 blocks, and the copy
is a few microseconds per step. ``UNSLOTH_DIFFUSION_BLOCK_RESTRIDE=0`` disables it.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

_ENV = "UNSLOTH_DIFFUSION_BLOCK_RESTRIDE"
_FALSE = ("0", "false", "no", "off")
# Transformer class -> (ModuleList attribute, block class) whose first entry gets the restride.
_TARGETS = {
    "FluxTransformer2DModel": ("single_transformer_blocks", "FluxSingleTransformerBlock"),
}
_MARK = "_unsloth_block_restride"


def enabled() -> bool:
    return (os.environ.get(_ENV) or "").strip().lower() not in _FALSE


def _takes_both_streams(block: Any) -> bool:
    """The block concatenates ``encoder_hidden_states`` + ``hidden_states`` itself (diffusers >= 0.32 layout)."""
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
    # Already the slice layout (a caller that concatenated itself): nothing to do.
    if hidden.stride() == ((text + image) * dim, dim, 1) and encoder.stride() == hidden.stride():
        return None
    buf = torch.empty((batch, text + image, dim), dtype = hidden.dtype, device = hidden.device)
    buf[:, :text].copy_(encoder)
    buf[:, text:].copy_(hidden)
    return buf[:, text:], buf[:, :text]


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

        def restride(*args: Any, **kwargs: Any) -> Any:
            # Keyword call is what diffusers' FluxTransformer2DModel.forward makes; anything else passes through.
            if not args and "hidden_states" in kwargs and "encoder_hidden_states" in kwargs:
                pair = _restrided(kwargs["hidden_states"], kwargs["encoder_hidden_states"])
                if pair is not None:
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"], kwargs["encoder_hidden_states"] = pair
            return compiled(*args, **kwargs)

        setattr(restride, _MARK, True)
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
