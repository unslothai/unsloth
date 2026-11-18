# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastModel generate() with the caller's past_key_values or assisted decoding decodes like transformers,
instead of forcing a static cache transformers refuses next to them."""

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


@pytest.mark.parametrize(
    "case", ["past_key_values", "prompt_lookup_num_tokens", "generation_config"]
)
def test_caller_cache_and_assisted_decoding_match_transformers(case):
    model = _model()
    history = torch.randint(3, 64, (1, 8), generator = torch.Generator().manual_seed(0))
    ids = torch.cat([history, history[:, :4]], 1)
    kwargs = dict(
        attention_mask = torch.ones_like(ids), max_new_tokens = 6, do_sample = False, pad_token_id = 0
    )

    def run(generate):
        extra = {"prompt_lookup_num_tokens": 3}
        if case == "past_key_values":
            with torch.no_grad():
                extra = {
                    "past_key_values": model(input_ids = history, use_cache = True).past_key_values
                }
        elif case == "generation_config":
            from transformers import GenerationConfig
            extra = {"generation_config": GenerationConfig(prompt_lookup_num_tokens = 3)}
        return generate(input_ids = ids, **kwargs, **extra)

    torch.testing.assert_close(run(model.generate), run(model._old_generate), rtol = 0, atol = 0)


def test_kwarg_none_overrides_config_lookup_and_keeps_static_cache():
    model = _model()
    model.generation_config.prompt_lookup_num_tokens = 3
    seen = {}
    old = model._old_generate

    def spy(*args, **kw):
        seen.update(kw)
        return old(*args, **kw)

    model._old_generate = spy
    ids = torch.randint(3, 64, (1, 8), generator = torch.Generator().manual_seed(0))
    model.generate(input_ids = ids, max_new_tokens = 2, do_sample = False, prompt_lookup_num_tokens = None)
    assert seen["cache_implementation"] == "static"
