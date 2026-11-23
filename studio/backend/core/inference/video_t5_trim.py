# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan's umT5 encodes only the real prompt tokens when the encoder runs on the CPU or MPS.

diffusers' Wan pipelines tokenize to ``max_sequence_length`` (512), run the encoder over every padded position, then
keep ``hidden[:seq_len]`` per prompt and zero the rest. The padded keys are masked, so the kept rows do not depend on
them: encoding ``ids[:, :longest]`` and zero-padding the output back gives the same embeddings (fp32 reduction order
only) for a fraction of the work. On a GPU the encode is about a second and stays as it is.
"""

from __future__ import annotations

import os
import threading
from typing import Any

ENV = "UNSLOTH_VIDEO_T5_TRIM"
_DISABLE_TOKENS = frozenset({"0", "off", "false", "no"})
_TRIM_DEVICES = frozenset({"cpu", "mps"})
_METHOD = "_get_t5_prompt_embeds"
_MARK = "_unsloth_t5_trim"

STATS = {"trimmed": 0, "passed": 0}
_LOCK = threading.Lock()


def enabled() -> bool:
    return (os.environ.get(ENV) or "").strip().lower() not in _DISABLE_TOKENS


def _prefix_length(attention_mask: Any) -> int:
    """Longest real length when every row is right-padded (ones then zeros); -1 for any other mask."""
    import torch

    real = attention_mask.gt(0)
    lengths = real.sum(dim = 1)
    positions = torch.arange(real.shape[1], device = real.device)
    if not bool((real == (positions.unsqueeze(0) < lengths.unsqueeze(1))).all()):
        return -1
    return int(lengths.max())


def _trimmed_forward(encoder: Any, forward: Any) -> Any:
    def run(
        input_ids: Any = None,
        attention_mask: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        import torch

        plain = (
            not args
            and not kwargs
            and input_ids is not None
            and attention_mask is not None
            and input_ids.dim() == 2
            and attention_mask.shape == input_ids.shape
            and input_ids.device.type in _TRIM_DEVICES
        )
        keep = _prefix_length(attention_mask) if plain else -1
        if keep < 1 or keep >= input_ids.shape[1]:
            STATS["passed"] += 1
            return forward(input_ids, attention_mask, *args, **kwargs)
        out = forward(input_ids[:, :keep], attention_mask[:, :keep])
        hidden = out.last_hidden_state
        pad = hidden.new_zeros(hidden.shape[0], input_ids.shape[1] - keep, hidden.shape[2])
        out.last_hidden_state = torch.cat([hidden, pad], dim = 1)
        STATS["trimmed"] += 1
        return out

    return run


def install(pipe: Any, logger: Any = None) -> bool:
    """Wrap a Wan pipeline's T5 prompt encode; False when the pipeline is not one or the switch is off."""
    if not enabled() or not type(pipe).__name__.startswith("Wan"):
        return False
    encode = getattr(pipe, _METHOD, None)
    if not callable(encode) or getattr(encode, _MARK, False):
        return False

    def trimmed_encode(*args: Any, **kwargs: Any) -> Any:
        encoder = getattr(pipe, "text_encoder", None)
        if encoder is None:
            return encode(*args, **kwargs)
        with _LOCK:
            # An offload hook may already own ``forward`` as an instance attribute: put back exactly what was there.
            owned = "forward" in encoder.__dict__
            previous = encoder.__dict__.get("forward")
            encoder.forward = _trimmed_forward(encoder, encoder.forward)
            try:
                return encode(*args, **kwargs)
            finally:
                if owned:
                    encoder.forward = previous
                else:
                    del encoder.forward

    setattr(trimmed_encode, _MARK, True)
    trimmed_encode.__wrapped__ = encode
    setattr(pipe, _METHOD, trimmed_encode)
    if logger is not None:
        logger.info("video.t5_trim: on for %s (CPU / MPS text encoder only)", type(pipe).__name__)
    return True
