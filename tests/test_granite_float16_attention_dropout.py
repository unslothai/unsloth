# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Granite 3.0-3.2 attention dropout NaNs float16 training, so a float16 load turns it off (#3002)."""

from __future__ import annotations

import pytest
import torch
from transformers import GraniteConfig, GraniteForCausalLM, LlamaConfig, LlamaForCausalLM

from unsloth.models._utils import _drop_float16_overflowing_attention_dropout


def _tiny(config_class, model_class, attention_dropout):
    config = config_class(
        vocab_size = 16,
        hidden_size = 8,
        intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        attention_dropout = attention_dropout,
    )
    return model_class(config)


def _dropouts(model):
    return [
        m.attention_dropout
        for m in model.modules()
        if hasattr(m, "self_attn") is False and hasattr(m, "attention_dropout")
    ]


def test_float16_granite_attention_dropout_is_turned_off(capsys):
    model = _tiny(GraniteConfig, GraniteForCausalLM, 0.1)
    assert _dropouts(model) == [0.1, 0.1]
    _drop_float16_overflowing_attention_dropout(model, torch.float16)
    assert _dropouts(model) == [0.0, 0.0]
    # The config is left alone so a saved or merged model keeps the checkpoint's value.
    assert model.config.attention_dropout == 0.1
    assert "attention dropout" in capsys.readouterr().out
    _drop_float16_overflowing_attention_dropout(model, torch.float16)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_other_dtypes_keep_granite_attention_dropout(dtype):
    model = _tiny(GraniteConfig, GraniteForCausalLM, 0.1)
    _drop_float16_overflowing_attention_dropout(model, dtype)
    assert _dropouts(model) == [0.1, 0.1]


def test_float16_other_architectures_keep_attention_dropout():
    model = _tiny(LlamaConfig, LlamaForCausalLM, 0.1)
    _drop_float16_overflowing_attention_dropout(model, torch.float16)
    assert _dropouts(model) == [0.1, 0.1]
