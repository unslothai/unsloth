# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Presence-penalty logits helpers for the safetensors/MLX inference paths.

Kept in a dependency-light leaf module (torch + transformers only, no unsloth /
peft) so the pure logic can be imported and unit-tested without pulling in the
full inference backend. ``core.inference.inference`` re-exports these for the
runtime generate paths.
"""

import torch


def apply_presence_penalty(input_ids, scores, penalty: float, prompt_len: int):
    """OpenAI/llama.cpp presence penalty: subtract ``penalty`` once per distinct
    completion token (positions >= prompt_len; prompt excluded, multiplicity
    ignored, negatives raise). In place; zero is a no-op."""
    if not penalty:
        return scores
    vocab_size = scores.shape[-1]
    for b in range(input_ids.shape[0]):
        generated = input_ids[b, prompt_len:]
        if generated.numel() == 0:
            continue
        seen = torch.unique(generated)
        # Bound generated ids to the valid range [0, vocab_size). Real completion
        # tokens are always in range, so this is a zero-regression safety net that
        # drops any stray out-of-range or negative id before indexing (mirrors the
        # MLX path's bound). Filtering both ends avoids indexing scores with a
        # negative id (which would silently wrap to the wrong row).
        seen = seen[(seen >= 0) & (seen < vocab_size)]
        if seen.numel():
            scores[b, seen] = scores[b, seen] - penalty
    return scores


def _make_presence_penalty_processor(penalty: float, prompt_len: int):
    """``LogitsProcessorList`` for ``apply_presence_penalty``; ``None`` at zero penalty (generate call stays byte-identical)."""
    if not penalty:
        return None
    from transformers import LogitsProcessor, LogitsProcessorList

    class _PresencePenaltyLogitsProcessor(LogitsProcessor):
        @torch.no_grad()
        def __call__(self, input_ids, scores):
            return apply_presence_penalty(input_ids, scores, penalty, prompt_len)

    return LogitsProcessorList([_PresencePenaltyLogitsProcessor()])
