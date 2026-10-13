# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import re

import pytest

import unsloth  # noqa: F401
import unsloth.models.llama as llama_module
from peft import get_peft_model
from unsloth.models.llama import FastLlamaModel


def _tiny_llama(layers = 2):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = layers,
        num_attention_heads = 2,
        num_key_value_heads = 2,
    )
    model = LlamaForCausalLM(config)
    model.max_seq_length = 64
    model._saved_temp_tokenizer = None
    return model


class _Captured(Exception):
    pass


def _lora_config_for(target_modules, monkeypatch):
    def capture(model, lora_config, *args, **kwargs):
        raise _Captured(lora_config)

    monkeypatch.setattr(llama_module, "_get_peft_model", capture)
    with pytest.raises(_Captured) as info:
        FastLlamaModel.get_peft_model(
            _tiny_llama(),
            r = 4,
            target_modules = target_modules,
            use_gradient_checkpointing = False,
        )
    return info.value.args[0]


def _lora_modules(lora_config):
    wrapped = get_peft_model(_tiny_llama(), lora_config)
    return sorted(
        name.removeprefix("base_model.model.").removesuffix(".lora_A")
        for name, _ in wrapped.named_modules()
        if name.endswith(".lora_A")
    )


def test_regex_string_targets_exactly_the_matched_modules(monkeypatch):
    # Spectrum-style selection: different projections in different layers.
    wanted = ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"]
    regex = r"(?:.*\.)?(?:" + "|".join(re.escape(name) for name in wanted) + r")"
    config = _lora_config_for(regex, monkeypatch)
    assert config.target_modules == regex
    assert _lora_modules(config) == wanted


def test_all_linear_string_is_passed_to_peft(monkeypatch):
    config = _lora_config_for("all-linear", monkeypatch)
    assert config.target_modules == "all-linear"
    assert len(_lora_modules(config)) == 2 * 7


def test_list_targets_unchanged(monkeypatch):
    config = _lora_config_for(["q_proj", "v_proj"], monkeypatch)
    assert config.target_modules == {"q_proj", "v_proj"}
    assert len(_lora_modules(config)) == 2 * 2


def test_add_moe_target_modules_keeps_each_form():
    from unsloth.models._utils import add_moe_target_modules

    experts = ["gate_up_projs.0", "gate_up_projs.1", "down_projs.0"]
    assert add_moe_target_modules(["q_proj"], experts) == ["q_proj"] + experts
    assert add_moe_target_modules(["q_proj", "down_projs.0"], experts) == [
        "q_proj",
        "down_projs.0",
        "gate_up_projs.0",
        "gate_up_projs.1",
    ]
    assert add_moe_target_modules("all-linear", experts) == "all-linear"
    assert add_moe_target_modules(".*q_proj", []) == ".*q_proj"
    combined = add_moe_target_modules(".*q_proj", experts)
    assert re.fullmatch(combined, "model.layers.0.self_attn.q_proj")
    assert re.fullmatch(combined, "model.layers.3.mlp.experts.down_projs.7")
    assert not re.fullmatch(combined, "model.layers.0.mlp.experts.router")


def _prewrapped(
    target_modules,
    layers = 2,
    **extra,
):
    from peft import LoraConfig
    config = LoraConfig(
        r = 16, lora_alpha = 16, target_modules = target_modules, task_type = "CAUSAL_LM", **extra
    )
    return get_peft_model(_tiny_llama(layers), config)


REGEX = r".*\.layers\.0\.self_attn\.q_proj"


@pytest.mark.parametrize(
    "target_modules, layers",
    [("all-linear", 2), ("all-linear", 4), (REGEX, 2)],
    # 4 layers = 28 Linears: past PEFT's 20-module threshold, the stored list is compacted to suffixes.
    ids = ["all_linear", "all_linear_compacted", "regex"],
)
def test_repeat_call_with_same_str_keeps_the_adapter(target_modules, layers):
    model = _prewrapped(target_modules, layers)
    again = FastLlamaModel.get_peft_model(
        model, r = 16, target_modules = target_modules, use_gradient_checkpointing = False
    )
    assert again is model


@pytest.mark.parametrize(
    "stored, requested",
    [
        (["q_proj", "v_proj"], "all-linear"),
        (REGEX, r".*\.layers\.1\.self_attn\.q_proj"),
    ],
    ids = ["narrower_then_all_linear", "other_regex"],
)
def test_repeat_call_with_different_str_is_refused(stored, requested):
    with pytest.raises(TypeError, match = "new parameters are different"):
        FastLlamaModel.get_peft_model(
            _prewrapped(stored), r = 16, target_modules = requested, use_gradient_checkpointing = False
        )


def test_all_linear_matches_an_adapter_on_every_linear():
    # The adapter wraps exactly what "all-linear" resolves to, however its targets were spelled.
    model = _prewrapped(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 4
    )
    again = FastLlamaModel.get_peft_model(
        model, r = 16, target_modules = "all-linear", use_gradient_checkpointing = False
    )
    assert again is model


def test_all_linear_repeat_honours_exclude_modules():
    model = _prewrapped("all-linear", 4, exclude_modules = ["down_proj"])
    again = FastLlamaModel.get_peft_model(
        model,
        r = 16,
        target_modules = "all-linear",
        exclude_modules = ["down_proj"],
        use_gradient_checkpointing = False,
    )
    assert again is model
    with pytest.raises(TypeError, match = "new parameters are different"):
        FastLlamaModel.get_peft_model(
            model, r = 16, target_modules = "all-linear", use_gradient_checkpointing = False
        )
