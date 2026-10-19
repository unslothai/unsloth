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


def test_llama_path_all_linear_with_finetune_last_n_layers(monkeypatch):
    config = _lora_config(
        monkeypatch,
        llama_module,
        FastLlamaModel,
        target_modules = "all-linear",
        finetune_last_n_layers = 2,
    )
    assert _layers(config) == [2, 3]
    assert not any("lm_head" in name for name in config.target_modules)


def _tiny_qwen3_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        moe_intermediate_size = 8,
        num_hidden_layers = 3,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        num_experts = 4,
        num_experts_per_tok = 2,
    )
    model = Qwen3MoeForCausalLM(config)
    model.max_seq_length = 64
    model._saved_temp_tokenizer = None
    return model


def test_moe_expert_parameters_follow_finetune_last_n_layers(monkeypatch):
    def capture(model, lora_config, *args, **kw):
        raise _Captured(lora_config)

    monkeypatch.setattr(vision_module, "_get_peft_model", capture)
    monkeypatch.setenv("UNSLOTH_USE_NEW_MODEL", "0")
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "0")
    with pytest.raises(_Captured) as info:
        FastBaseModel.get_peft_model(
            _tiny_qwen3_moe(), r = 4, use_gradient_checkpointing = False, finetune_last_n_layers = 1
        )
    config = info.value.args[0]
    if not config.target_parameters:
        pytest.skip("this transformers version has no fused expert parameters")
    layers = {int(name.split(".layers.")[1].split(".")[0]) for name in config.target_parameters}
    assert layers == {2}


def test_nested_expert_index_is_not_the_layer_index():
    import torch.nn as nn

    def block():
        experts = nn.ModuleList([nn.ModuleDict({"gate_proj": nn.Linear(4, 4)}) for _ in range(4)])
        return nn.ModuleDict({"mlp": nn.ModuleDict({"experts": experts})})

    model = nn.ModuleDict(
        {"model": nn.ModuleDict({"layers": nn.ModuleList([block() for _ in range(4)])})}
    )
    names = vision_module._regex_targets_as_names(model, r".*\.gate_proj", [3])
    assert names == [f"model.layers.3.mlp.experts.{i}.gate_proj" for i in range(4)]


def test_expanded_targets_drop_the_layer_filter(monkeypatch):
    config = _lora_config(monkeypatch, vision_module, FastBaseModel, finetune_last_n_layers = 2)
    assert config.layers_to_transform is None and config.layers_pattern is None
    config = _lora_config(
        monkeypatch,
        llama_module,
        FastLlamaModel,
        target_modules = r".*\.(q_proj|v_proj)",
        layers_to_transform = [0, 1, 2, 3],
    )
    assert config.layers_to_transform is None
    assert _layers(config) == [0, 1, 2, 3]


def test_every_selected_expert_is_wrapped_when_peft_compacts_targets():
    import torch.nn as nn
    from peft import LoraConfig

    def block():
        experts = nn.ModuleList([nn.ModuleDict({"gate_proj": nn.Linear(2, 2)}) for _ in range(32)])
        return nn.ModuleDict({"mlp": nn.ModuleDict({"experts": experts})})

    def model():
        return nn.ModuleDict(
            {"model": nn.ModuleDict({"layers": nn.ModuleList([block() for _ in range(4)])})}
        )

    names = vision_module._regex_targets_as_names(model(), r".*\.gate_proj", [0, 1, 2, 3])
    # 128 names compact to ["gate_proj"]; PEFT < 0.20 then reads experts.N as the layer unless no filter is left.
    wrapped = get_peft_model(model(), LoraConfig(r = 1, target_modules = names))
    assert sum(name.endswith("lora_A.default") for name, _ in wrapped.named_modules()) == 128


def test_layered_reuse_compares_against_the_wrapped_modules():
    from peft import LoraConfig

    match = llama_module._layered_request_matches_adapter
    names = ["model.layers.3.self_attn.q_proj", "model.layers.3.self_attn.v_proj"]
    wrapped = get_peft_model(_tiny_llama(), LoraConfig(r = 4, target_modules = names))
    assert match(wrapped, r".*\.(q_proj|v_proj)", [3])
    assert not match(wrapped, r".*\.(q_proj|v_proj)", [2])
    assert not match(wrapped, r".*\.(q_proj|v_proj)", [3], exclude_modules = ["v_proj"])
    only_q = get_peft_model(_tiny_llama(), LoraConfig(r = 4, target_modules = names[:1]))
    assert match(only_q, r".*\.(q_proj|v_proj)", [3], exclude_modules = ["v_proj"])


def test_layered_reuse_compares_expert_tensors():
    from peft import LoraConfig

    match = llama_module._layered_request_matches_adapter
    if not hasattr(_tiny_qwen3_moe().model.layers[2].mlp.experts, "gate_up_proj"):
        pytest.skip("this transformers version has no fused expert parameters")
    wrapped = get_peft_model(
        _tiny_qwen3_moe(),
        LoraConfig(
            r = 4,
            target_modules = ["model.layers.2.self_attn.q_proj"],
            target_parameters = ["model.layers.2.mlp.experts.gate_up_proj"],
        ),
    )
    assert match(wrapped, r".*\.(self_attn\.q_proj|mlp\.gate_proj)", [2])
    assert not match(wrapped, r".*\.(self_attn\.q_proj|mlp\.down_proj)", [2])
    experts_only = get_peft_model(
        _tiny_qwen3_moe(),
        LoraConfig(
            r = 4, target_modules = [], target_parameters = ["model.layers.2.mlp.experts.gate_up_proj"]
        ),
    )
    assert match(experts_only, r".*\.mlp\.experts\.gate_proj", [2])
