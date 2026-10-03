# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-conditioning defaults that follow ComfyUI. No torch/diffusers imports.

FLUX.1 T5 sequence length:

diffusers pads (and truncates) the FLUX.1 T5 prompt to ``max_sequence_length=512`` and runs T5
without an attention mask, so every pad token takes part in the encoder's self-attention. ComfyUI,
our baseline, pads the same T5 prompt only up to 256 tokens and otherwise uses its real length
(again with no mask). Both the embeddings and the text tokens the DiT attends over therefore differ,
and 512 costs 256 extra joint-attention tokens per step for a short prompt.

``flux_t5_sequence_length`` reproduces the ComfyUI length (256) for every prompt of at most 256 T5
tokens (EOS included). Longer prompts pad to 512 as before (bucketed, see the function), and a prompt
past 512 tokens truncates exactly as before.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

# ComfyUI's Ideogram 4 template ("Default" preset): 20 steps, logit-normal mean 0.0 and spread 1.75 before the
# resolution term (both implementations add it the same way), and guidance 7 overridden to 3 over the last 30% of
# sampling (CFG override, start 0.7 / end 1.0). Ideogram 4 samples a plain flow model (shift 1), so that range is
# sigma <= 0.3.
IDEOGRAM4_COMFY_STEPS = 20
IDEOGRAM4_COMFY_MU = 0.0
IDEOGRAM4_COMFY_STD = 1.75
IDEOGRAM4_COMFY_GUIDANCE = 7.0
IDEOGRAM4_COMFY_TAIL_GUIDANCE = 3.0
IDEOGRAM4_COMFY_TAIL_SIGMA = 0.3
_IDEOGRAM4_LOGSNR_MIN = -15.0
_IDEOGRAM4_LOGSNR_MAX = 18.0


def ideogram4_sigmas(steps: int, width: int, height: int, mu: float, std: float) -> list[float]:
    """The ``steps`` sigmas the Ideogram 4 logit-normal schedule visits (terminal 0 excluded), highest first."""
    from statistics import NormalDist

    mean = float(mu) + 0.5 * math.log((int(width) * int(height)) / (512 * 512))
    t_min = 1.0 / (1.0 + math.exp(0.5 * _IDEOGRAM4_LOGSNR_MAX))
    t_max = 1.0 / (1.0 + math.exp(0.5 * _IDEOGRAM4_LOGSNR_MIN))
    out = []
    for i in range(int(steps)):
        u = 1.0 - i / int(steps)  # the schedule is built on linspace(0, 1) and flipped
        if u >= 1.0:
            t = 0.0
        else:
            y = mean + float(std) * NormalDist().inv_cdf(u)
            t = 1.0 - 1.0 / (1.0 + math.exp(-y))
        out.append(1.0 - min(max(t, t_min), t_max))
    return out


def ideogram4_comfy_guidance_schedule(steps: int, width: int, height: int) -> list[float]:
    """Per-step guidance ComfyUI's template applies: 7, then 3 on every step whose sigma is <= 0.3."""
    return [
        IDEOGRAM4_COMFY_TAIL_GUIDANCE
        if sigma <= IDEOGRAM4_COMFY_TAIL_SIGMA
        else IDEOGRAM4_COMFY_GUIDANCE
        for sigma in ideogram4_sigmas(steps, width, height, IDEOGRAM4_COMFY_MU, IDEOGRAM4_COMFY_STD)
    ]


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
    # Bucketed, a deliberate deviation from ComfyUI's exact length: every distinct text length is a new denoiser shape,
    # and on the compiled default path the first one past 256 measured a 14 s recompile, each further one a CUDA graph
    # recapture (0.4 to 1 s) until the 4-graph cap made later shapes run eager. So a prompt past 256 tokens pads to the
    # cap (exactly the old behaviour); T5 runs unmasked, so that padding shifts the embeddings of those prompts only.
    return int(floor) if longest <= int(floor) else int(cap)


def flux_t5_kwarg(
    family_name: str, pipe: Any, call_params: Any, chunk_kwargs: dict
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
