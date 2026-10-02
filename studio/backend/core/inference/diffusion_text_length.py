# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-conditioning defaults that follow ComfyUI. No torch/diffusers imports.

FLUX.1 T5 sequence length:

diffusers pads (and truncates) the FLUX.1 T5 prompt to ``max_sequence_length=512`` and runs T5
without an attention mask, so every pad token takes part in the encoder's self-attention. ComfyUI,
our baseline, pads the same T5 prompt only up to 256 tokens and otherwise uses its real length
(again with no mask). Both the embeddings and the text tokens the DiT attends over therefore differ,
and 512 costs 256 extra joint-attention tokens per step for a short prompt.

``flux_t5_sequence_length`` reproduces the ComfyUI length: the T5 token count of the longest
prompt in the call (EOS included), floored at 256 and capped at the pipeline's 512, so a prompt past
512 tokens truncates exactly as before.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

# ComfyUI's Ideogram 4 scheduler defaults (its official template): logit-normal mean and spread before the resolution
# term, which both implementations add the same way.
IDEOGRAM4_COMFY_MU = 0.5
IDEOGRAM4_COMFY_STD = 1.75

# FLUX.1 families on a FluxPipeline-style ``max_sequence_length`` whose ComfyUI tokenizer is the FLUX T5 one.
FLUX_T5_FAMILIES = frozenset({"flux.1", "flux.1-kontext"})
FLUX_T5_MIN_TOKENS = 256
FLUX_T5_MAX_TOKENS = 512


def _texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def t5_token_count(tokenizer: Any, text: str) -> int:
    """T5 tokens for ``text`` including the EOS the tokenizer appends."""
    ids = tokenizer(text, add_special_tokens = True)["input_ids"]
    return len(ids)


def flux_t5_sequence_length(
    tokenizer: Any,
    prompts: Iterable[Any],
    *,
    floor: int = FLUX_T5_MIN_TOKENS,
    cap: int = FLUX_T5_MAX_TOKENS,
) -> Optional[int]:
    """``max_sequence_length`` for one FLUX.1 call, or None when it cannot be measured (keep the
    pipeline default). ``prompts`` holds each str / list-of-str the call encodes with T5."""
    if tokenizer is None or not callable(tokenizer):
        return None
    texts: list[str] = []
    for value in prompts:
        texts.extend(_texts(value))
    if not texts:
        return None
    try:
        longest = max(t5_token_count(tokenizer, t) for t in texts)
    except Exception:  # noqa: BLE001 - an odd tokenizer only keeps the pipeline default
        return None
    return max(int(floor), min(int(cap), int(longest)))


def flux_t5_kwarg(
    family_name: str,
    pipe: Any,
    call_params: Any,
    chunk_kwargs: dict,
) -> Optional[int]:
    """The ``max_sequence_length`` to pass for this chunk, or None to leave the kwarg alone.

    Only FLUX.1 families, only when the pipeline accepts the kwarg, and never over a value the
    caller already set. The negative prompt counts only when true CFG actually encodes it."""
    if family_name not in FLUX_T5_FAMILIES:
        return None
    if "max_sequence_length" not in call_params or "max_sequence_length" in chunk_kwargs:
        return None
    prompts: list[Any] = [chunk_kwargs.get("prompt_2") or chunk_kwargs.get("prompt")]
    negative = chunk_kwargs.get("negative_prompt_2") or chunk_kwargs.get("negative_prompt")
    try:
        true_cfg = float(chunk_kwargs.get("true_cfg_scale", 1.0) or 1.0)
    except (TypeError, ValueError):
        true_cfg = 1.0
    if negative and true_cfg > 1.0:
        prompts.append(negative)
    return flux_t5_sequence_length(getattr(pipe, "tokenizer_2", None), prompts)


def true_cfg_needs_empty_negative(cfg_kwarg: str, guidance: Any) -> bool:
    """True when a ``true_cfg_scale`` pipeline (Qwen-Image family) would skip CFG for want of a
    negative prompt. diffusers enables true CFG only when ``true_cfg_scale > 1`` AND a negative is
    given; ComfyUI always encodes the (possibly empty) negative and applies CFG above 1."""
    if cfg_kwarg != "true_cfg_scale":
        return False
    try:
        return float(guidance) > 1.0
    except (TypeError, ValueError):
        return False
