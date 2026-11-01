# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Presence-penalty logits helpers; torch and transformers only, so they import without the backend."""

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
        # Drop out-of-range ids: a negative index would silently wrap (mirrors the MLX path).
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
