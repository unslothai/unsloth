# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""PEFT rejects layers_to_transform with a str target_modules; FastModel passes a regex by default."""

import pytest

import unsloth  # noqa: F401
import unsloth.models.llama as llama_module
import unsloth.models.vision as vision_module
from peft import get_peft_model
from unsloth.models.llama import FastLlamaModel
from unsloth.models.vision import FastBaseModel


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 4,
        num_attention_heads = 2,
        num_key_value_heads = 2,
    )
    model = LlamaForCausalLM(config)
    model.max_seq_length = 64
    model._saved_temp_tokenizer = None
    return model


class _Captured(Exception):
    pass


def _lora_config(monkeypatch, module, loader, **kwargs):
    def capture(model, lora_config, *args, **kw):
        raise _Captured(lora_config)

    monkeypatch.setattr(module, "_get_peft_model", capture)
    monkeypatch.setenv("UNSLOTH_USE_NEW_MODEL", "0")
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "0")
    with pytest.raises(_Captured) as info:
        loader.get_peft_model(_tiny_llama(), r = 4, use_gradient_checkpointing = False, **kwargs)
    return info.value.args[0]


def _layers(lora_config):
    wrapped = get_peft_model(_tiny_llama(), lora_config)
    return sorted(
        {
            int(name.split(".layers.")[1].split(".")[0])
            for name, _ in wrapped.named_modules()
            if name.endswith(".lora_A")
        }
    )


def test_default_targets_with_finetune_last_n_layers(monkeypatch):
    config = _lora_config(monkeypatch, vision_module, FastBaseModel, finetune_last_n_layers = 2)
    assert _layers(config) == [2, 3]


def test_default_targets_with_layers_to_transform(monkeypatch):
    config = _lora_config(monkeypatch, vision_module, FastBaseModel, layers_to_transform = [1])
    assert _layers(config) == [1]


def test_llama_path_regex_targets_with_finetune_last_n_layers(monkeypatch):
    config = _lora_config(
        monkeypatch,
        llama_module,
        FastLlamaModel,
        target_modules = r".*\.(q_proj|v_proj)",
        finetune_last_n_layers = 1,
    )
    assert _layers(config) == [3]
