# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Without layers_pattern PEFT takes the first `<name>.<int>.` as the layer index, so other numbered stacks match too."""

import pytest
import torch

import unsloth  # noqa: F401
import unsloth.models.llama as llama_module
from peft import get_peft_model
from unsloth.models.llama import FastLlamaModel


def _tiny_llama_with_aux_stack():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 2,
    )
    model = LlamaForCausalLM(config)
    model.max_seq_length = 64
    model._saved_temp_tokenizer = None
    # A second numbered stack that owns a q_proj, like a vision tower or a per-expert list.
    model.model.aux = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"q_proj": torch.nn.Linear(16, 16)}) for _ in range(2)]
    )
    return model


class _Captured(Exception):
    pass


def _lora_config(monkeypatch, **kwargs):
    def capture(model, lora_config, *args, **kw):
        raise _Captured(lora_config)

    monkeypatch.setattr(llama_module, "_get_peft_model", capture)
    with pytest.raises(_Captured) as info:
        FastLlamaModel.get_peft_model(
            _tiny_llama_with_aux_stack(),
            r = 4,
            target_modules = ["q_proj"],
            use_gradient_checkpointing = False,
            **kwargs,
        )
    return info.value.args[0]


def _lora_modules(lora_config):
    wrapped = get_peft_model(_tiny_llama_with_aux_stack(), lora_config)
    return sorted(
        name.removeprefix("base_model.model.").removesuffix(".lora_A")
        for name, _ in wrapped.named_modules()
        if name.endswith(".lora_A")
    )


def test_layers_pattern_reaches_lora_config(monkeypatch):
    config = _lora_config(monkeypatch, layers_to_transform = [1], layers_pattern = "layers")
    assert config.layers_pattern == "layers"


def test_layers_pattern_keeps_the_aux_stack_out(monkeypatch):
    config = _lora_config(monkeypatch, layers_to_transform = [1], layers_pattern = "layers")
    assert _lora_modules(config) == ["model.layers.1.self_attn.q_proj"]
