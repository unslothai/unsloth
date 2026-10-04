# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drop the language-model head of a Qwen3-VL text encoder that is only read for hidden states.

Qwen-Image-2.1 conditions on ``hidden_states[-1]`` of a ``Qwen3VLForConditionalGeneration``
(diffusers ``QwenImage21Pipeline._get_qwen_prompt_embeds``). The encoder's ``lm_head`` is a
151936 x 4096 projection (622M parameters, 1.16 GiB in bfloat16, 0.58 GiB in the hosted fp8 file)
that sits AFTER every hidden state: ``Qwen3VLForConditionalGeneration.forward`` computes
``logits = self.lm_head(hidden_states[:, slice_indices, :])`` on the decoder output and nothing
upstream reads it. Its untied weight (``tie_word_embeddings`` is false for this checkpoint) is still
read from disk, cast, moved to the GPU, and multiplied against every prompt token on each encode,
because ``logits_to_keep`` defaults to 0, which keeps all positions.

Replacing it with :class:`NoLogitsHead` keeps the module tree and every hidden state bit-identical
(the head is the last op), returns an empty ``[..., 0]`` logits view for zero FLOPs, and frees the
weight. On a pipeline loaded dense (mmap) the trim runs before the first device move, so the head's
pages are never read; on the pre-cast fp8 path the key is filtered out of the state dict before
``load_state_dict``.

The vision tower is NOT trimmed: the same loaded pipeline serves image-conditioned (edit) requests,
which run ``model.visual``.

Applies only to text encoders whose pipeline is known to read hidden states and never logits; any
other class is left alone. Kill switch: ``UNSLOTH_TE_KEEP_LM_HEAD=1``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

KEEP_LM_HEAD_ENV = "UNSLOTH_TE_KEEP_LM_HEAD"

# Text-encoder classes whose diffusion pipeline reads ``hidden_states`` only. Keyed by class name so this module never
# imports transformers.
_HIDDEN_STATE_ONLY_CLASSES = frozenset({"Qwen3VLForConditionalGeneration"})

# Families whose pipeline reads the encoder's hidden states only (never logits or generate()).
_HIDDEN_STATE_ONLY_FAMILIES = frozenset({"qwen-image-2.1"})

LM_HEAD_KEY = "lm_head.weight"


def trim_enabled() -> bool:
    return os.environ.get(KEEP_LM_HEAD_ENV, "").strip().lower() not in ("1", "true", "yes", "on")


def family_trims_lm_head(family: Optional[str]) -> bool:
    return trim_enabled() and (family or "").strip().lower() in _HIDDEN_STATE_ONLY_FAMILIES


def class_trims_lm_head(class_name: Optional[str]) -> bool:
    return (class_name or "") in _HIDDEN_STATE_ONLY_CLASSES


def config_ties_lm_head(config: Any) -> bool:
    """True when ``config`` (or its ``text_config``) ties ``lm_head`` to the input embedding."""
    if config is None:
        return False
    text_config = getattr(config, "text_config", None) or config
    return bool(getattr(text_config, "tie_word_embeddings", False)) or bool(
        getattr(config, "tie_word_embeddings", False)
    )


@lru_cache(maxsize = None)
def _no_logits_head_class() -> Any:
    """Built once: a fresh class per load would be a fresh dynamo type guard."""
    from torch import nn

    class NoLogitsHead(nn.Module):
        """Stand-in for an unused ``lm_head``: no weight, returns an empty logits view."""

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

        def forward(self, hidden_states: Any) -> Any:  # noqa: D102
            return hidden_states[..., :0]

        def extra_repr(self) -> str:  # noqa: D102
            return f"in_features={self.in_features}, out_features={self.out_features}, dropped"

    return NoLogitsHead


def is_trimmed(text_encoder: Any) -> bool:
    head = getattr(text_encoder, "lm_head", None)
    return head is not None and type(head).__name__ == "NoLogitsHead"


def trim_text_encoder(text_encoder: Any, *, family: Optional[str] = None) -> dict:
    """Replace ``text_encoder.lm_head`` with :class:`NoLogitsHead` in place.

    Returns a small record (``{"lm_head": "dropped", "params": N}`` or the reason it was kept).
    Never raises for an unsupported encoder: it is left unchanged.
    """
    if text_encoder is None:
        return {"lm_head": "kept", "reason": "no encoder"}
    if not trim_enabled():
        return {"lm_head": "kept", "reason": KEEP_LM_HEAD_ENV}
    if family is not None and not family_trims_lm_head(family):
        return {"lm_head": "kept", "reason": f"family {family}"}
    if not class_trims_lm_head(type(text_encoder).__name__):
        return {"lm_head": "kept", "reason": f"class {type(text_encoder).__name__}"}
    if is_trimmed(text_encoder):
        return {"lm_head": "dropped", "params": 0, "already": True}
    head = getattr(text_encoder, "lm_head", None)
    weight = getattr(head, "weight", None)
    if head is None or weight is None or getattr(weight, "ndim", 0) != 2:
        return {"lm_head": "kept", "reason": "no 2D lm_head weight"}
    if config_ties_lm_head(getattr(text_encoder, "config", None)):
        # A tied head shares storage with embed_tokens: dropping it frees nothing.
        return {"lm_head": "kept", "reason": "tied to embed_tokens"}
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    params = int(weight.numel())
    text_encoder.lm_head = _no_logits_head_class()(in_features, out_features)
    return {"lm_head": "dropped", "params": params}
