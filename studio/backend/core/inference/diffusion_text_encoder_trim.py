# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drop the unused ``lm_head`` of a Qwen3-VL text encoder whose pipeline reads only hidden states.

Qwen-Image-2.1 reads ``hidden_states[-1]`` (diffusers ``QwenImage21Pipeline._get_qwen_prompt_embeds``); the untied
151936 x 4096 head runs after every hidden state, so replacing it with :class:`NoLogitsHead` is bit-identical. The
vision tower stays: edit requests on the same pipeline run ``model.visual``. Kill switch: ``UNSLOTH_TE_KEEP_LM_HEAD=1``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

KEEP_LM_HEAD_ENV = "UNSLOTH_TE_KEEP_LM_HEAD"

# By class name so this module never imports transformers.
_HIDDEN_STATE_ONLY_CLASSES = frozenset({"Qwen3VLForConditionalGeneration"})

_HIDDEN_STATE_ONLY_FAMILIES = frozenset({"qwen-image-2.1"})

LM_HEAD_KEY = "lm_head.weight"


def trim_enabled() -> bool:
    return os.environ.get(KEEP_LM_HEAD_ENV, "").strip().lower() not in ("1", "true", "yes", "on")


def family_trims_lm_head(family: Optional[str]) -> bool:
    return trim_enabled() and (family or "").strip().lower() in _HIDDEN_STATE_ONLY_FAMILIES


def class_trims_lm_head(class_name: Optional[str]) -> bool:
    return (class_name or "") in _HIDDEN_STATE_ONLY_CLASSES


def config_ties_lm_head(config: Any) -> bool:
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
    """Replace ``text_encoder.lm_head`` in place; returns ``{"lm_head": "dropped", "params": N}`` or why it was kept."""
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
        return {"lm_head": "kept", "reason": "tied to embed_tokens"}
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    params = int(weight.numel())
    text_encoder.lm_head = _no_logits_head_class()(in_features, out_features)
    return {"lm_head": "dropped", "params": params}
