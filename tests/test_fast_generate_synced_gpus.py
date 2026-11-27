# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastModel generate() with synced_gpus = True (expert parallel sets it) decodes over a growing cache: a rank that
finishes first keeps stepping until every rank is done, past a static cache sized for its own max length (Kaggle
2x T4, OLMoE ep=2: 'expanded size of the tensor (25) must match the existing size (24)' on the shorter rank)."""

import types

import pytest
import torch

import unsloth.models.vision as vision


def _model():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        num_hidden_layers = 2,
        hidden_size = 32,
        intermediate_size = 64,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        vocab_size = 64,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    model._old_generate = model.generate
    model.generate = types.MethodType(vision.unsloth_base_fast_generate, model)
    return model


@pytest.mark.parametrize("synced", [True, False])
def test_synced_gpus_decodes_over_a_growing_cache(synced):
    model = _model()
    seen = {}

    def spy(*args, **kw):
        seen.update(kw)
        return kw[
            "input_ids"
        ]  # synced_gpus needs a process group: only the cache decision is checked here

    model._old_generate = spy
    ids = torch.randint(3, 64, (1, 8), generator = torch.Generator().manual_seed(0))
    kwargs = dict(input_ids = ids, max_new_tokens = 2, do_sample = False)
    if synced:
        kwargs["synced_gpus"] = True
    model.generate(**kwargs)
    assert seen["cache_implementation"] == ("dynamic" if synced else "static"), seen.get(
        "cache_implementation"
    )
