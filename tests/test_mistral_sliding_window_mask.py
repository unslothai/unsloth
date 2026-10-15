# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A padded prefill on a sliding-window model must window like transformers does.

transformers keeps k > q - sliding_window (the current key plus window - 1 before it). The converter
behind _prepare_4d_causal_attention_mask_for_sdpa counts only the keys before the current one, so
LlamaModel_fast_forward built a mask one key wider than the window.
"""

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.models import llama as llama_module


class _Captured(Exception):
    pass


@pytest.mark.parametrize("window", [2, 4, 6])
def test_padded_prefill_mask_matches_transformers_window(window, monkeypatch):
    from transformers import MistralConfig, MistralModel

    n = 10
    config = MistralConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        sliding_window = window,
    )
    model = MistralModel(config).eval()
    attention_mask = torch.ones(2, n, dtype = torch.long)
    attention_mask[1, :3] = 0

    real = llama_module._prepare_4d_causal_attention_mask_for_sdpa
    captured = {}

    def capture(*args, **kwargs):
        captured["mask"] = real(*args, **kwargs)
        raise _Captured

    monkeypatch.setattr(llama_module, "_prepare_4d_causal_attention_mask_for_sdpa", capture)
    with pytest.raises(_Captured):
        llama_module.LlamaModel_fast_forward(
            model,
            input_ids = torch.randint(1, 64, (2, n)),
            attention_mask = attention_mask,
            use_cache = False,
        )

    visible = captured["mask"] == 0  # (bsz, 1, q, k)
    q = torch.arange(n)[:, None]
    k = torch.arange(n)[None, :]
    expected = (k <= q) & (k > q - window) & attention_mask.bool()[:, None, None, :]
    real_queries = attention_mask.bool()[:, None, :, None]
    assert torch.equal(visible & real_queries, expected & real_queries)
