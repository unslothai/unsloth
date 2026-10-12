# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The GRPO hidden-state wrapper must hand back exactly what the output head consumes."""

from __future__ import annotations

import os
import types

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl as rl


def _hidden_then_head(model, input_ids):
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        with torch.no_grad():
            out = model(input_ids = input_ids)
    finally:
        os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
    return out.logits


def test_minicpm3_hidden_states_carry_the_pre_head_scaling():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "MiniCPM3ForCausalLM"):
        pytest.skip("this transformers has no MiniCPM3ForCausalLM")
    config = transformers.MiniCPM3Config(
        vocab_size = 97,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        q_lora_rank = 16,
        kv_lora_rank = 8,
        qk_nope_head_dim = 8,
        qk_rope_head_dim = 8,
        v_head_dim = 8,
        dim_model_base = 8,
        tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    assert config.logits_scaling == 4.0
    model = transformers.MiniCPM3ForCausalLM(config).eval()
    input_ids = torch.randint(0, 97, (2, 9))
    with torch.no_grad():
        want = model(input_ids = input_ids).logits
    assert rl._install_grpo_hidden_states_forward_wrapper(model)
    hidden = _hidden_then_head(model, input_ids)
    assert hidden.shape[-1] == 32
    torch.testing.assert_close(model.lm_head(hidden), want, rtol = 1e-5, atol = 1e-5)
    assert getattr(model, rl._UNSLOTH_GRPO_HIDDEN_STATES_VERIFIED_ATTR) is True


class _PreHeadScaledLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(97, 16)
        self.lm_head = torch.nn.Linear(16, 97, bias = False)
        self.config = types.SimpleNamespace(model_type = "tiny_prehead")

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        output_hidden_states = False,
        return_dict = True,
        logits_to_keep = 0,
    ):
        hidden = torch.tanh(self.embed(input_ids))
        logits = self.lm_head(
            (hidden * 3.0)[:, -logits_to_keep:, :] if logits_to_keep else hidden * 3.0
        )
        return types.SimpleNamespace(
            logits = logits, hidden_states = (hidden,) if output_hidden_states else None
        )


def test_an_unknown_pre_head_transform_falls_back_to_real_logits():
    model = _PreHeadScaledLM()
    input_ids = torch.randint(0, 97, (2, 7))
    assert rl._install_grpo_hidden_states_forward_wrapper(model)
    first = _hidden_then_head(model, input_ids)
    assert first.shape[-1] == 97  # real logits, not hidden states
    assert getattr(model, rl._UNSLOTH_GRPO_HIDDEN_STATES_UNSAFE_ATTR) is True
    assert getattr(model, rl._UNSLOTH_GRPO_HIDDEN_STATES_DEGRADED_ATTR) is True
    assert _hidden_then_head(model, input_ids).shape[-1] == 97  # sticky
