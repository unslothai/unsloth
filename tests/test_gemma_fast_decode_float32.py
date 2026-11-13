# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gemma / Gemma2 fast decode must match transformers in float32.

fast_rms_layernorm_inference_gemma normalized a float32 input in place (.to(torch.float32) returns
the same tensor), and the decode loops keep that input as the residual.
"""

import types

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.models import gemma as gemma_module
from unsloth.models import gemma2 as gemma2_module
from unsloth_zoo.hf_utils import set_dtype_in_config


@pytest.mark.parametrize("arch", ["gemma", "gemma2"])
def test_gemma_fast_decode_matches_transformers_in_float32(arch, monkeypatch):
    from transformers import Gemma2Config, Gemma2ForCausalLM, GemmaConfig, GemmaForCausalLM

    module = gemma_module if arch == "gemma" else gemma2_module
    norm = module.fast_rms_layernorm_inference_gemma
    # No per-device float32 buffers on CPU: run the real norm without one.
    monkeypatch.setattr(module, "DEVICE_COUNT", 0)
    monkeypatch.setattr(
        module, "per_layer_device", lambda layer: (torch.device("cpu"), slice(None))
    )
    monkeypatch.setattr(module, "fast_rms_layernorm_inference_gemma", lambda ln, x, w: norm(ln, x))

    sizes = dict(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        max_position_embeddings = 64,
    )
    torch.manual_seed(0)
    if arch == "gemma":
        model = GemmaForCausalLM(GemmaConfig(**sizes))
        fast_forward = gemma_module.GemmaModel_fast_forward_inference
    else:
        # Eager: transformers' SDPA path drops Gemma2's attention softcapping.
        config = Gemma2Config(
            **sizes,
            sliding_window = 4,
            query_pre_attn_scalar = 16,
            attn_implementation = "eager",
        )
        model = Gemma2ForCausalLM(config)
        fast_forward = gemma2_module.Gemma2Model_fast_forward_inference
    model = model.float().eval()
    set_dtype_in_config(model.config, torch.float32)
    model.max_seq_length = 64
    for module_ in model.modules():
        if hasattr(module_, "eps") and not hasattr(module_, "variance_epsilon"):
            module_.variance_epsilon = module_.eps
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(torch.zeros(1), torch.arange(64)[None])
    rotary = types.SimpleNamespace(
        get_cached = lambda seq_len, device_index = None: (cos[0], sin[0]),
        extend_rope_embedding = lambda *args: None,
    )
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary

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
            out = fast_forward(model, token, past_key_values, position_ids)
            past_key_values = out.past_key_values
            torch.testing.assert_close(out.last_hidden_state, expected, atol = 1e-5, rtol = 1e-5)
