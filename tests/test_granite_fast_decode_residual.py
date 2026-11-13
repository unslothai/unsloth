# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GraniteModel_fast_forward_inference must match transformers' Granite decode."""

import torch
import unsloth  # noqa: F401

from transformers import GraniteConfig, GraniteForCausalLM
from unsloth.models import granite as granite_module


def test_granite_fast_decode_matches_transformers(monkeypatch):
    monkeypatch.setattr(
        granite_module, "per_layer_device", lambda module, *args, **kwargs: (torch.device("cpu"), 0)
    )
    torch.manual_seed(0)
    config = GraniteConfig(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 64,
        attention_multiplier = 0.25,
        embedding_multiplier = 1.0,
        residual_multiplier = 1.0,
        logits_scaling = 1.0,
    )
    model = GraniteForCausalLM(config).eval()
    model.max_seq_length = 64
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(torch.zeros(1), torch.arange(64)[None])
    model.model.rotary_emb.get_cached = lambda seq_len, device_index = None: (cos[0], sin[0])

    prompt = torch.randint(1, 64, (1, 6))
    new_tokens = torch.randint(1, 64, (1, 3))
    with torch.no_grad():
        cache = model.model(input_ids = prompt, use_cache = True).past_key_values
        past_key_values = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
        for step in range(new_tokens.shape[1]):
            token = new_tokens[:, step : step + 1]
            position_ids = torch.tensor([[prompt.shape[1] + step]])
            expected = model.model(
                input_ids = token,
                position_ids = position_ids,
                past_key_values = cache,
                use_cache = True,
            ).last_hidden_state
            out = granite_module.GraniteModel_fast_forward_inference(
                model, token, past_key_values, position_ids, next_sequence_length = 1
            )
            past_key_values = out.past_key_values
            torch.testing.assert_close(out.last_hidden_state, expected, atol = 1e-5, rtol = 1e-5)
