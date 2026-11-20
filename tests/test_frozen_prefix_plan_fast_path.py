# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Frozen-prefix planning runs on every training forward. With adapters on every layer (the common case)
it must answer from the first layer of each stack, not by walking every parameter of the model (about
4-7 ms per forward on a 36-layer PEFT model on CPU), and the plan must be unchanged. CPU, tiny Llama."""

import importlib.util
import pathlib

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
pytest.importorskip("transformers.modeling_layers")

_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_frozen_prefix.py"
_spec = importlib.util.spec_from_file_location("_unsloth_frozen_prefix_plan_under_test", _PATH)
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)

LAYERS = 4


def _model(layers_to_transform):
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = LAYERS,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 64,
    )
    model = transformers.LlamaForCausalLM(config)
    lora = peft.LoraConfig(
        r = 4,
        target_modules = ["q_proj", "v_proj"],
        layers_to_transform = layers_to_transform,
    )
    model = peft.get_peft_model(model, lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.train()
    return model


def _plan_counting(model):
    stacks = fp._decoder_stacks(model)
    heads = fp._head_parameters(model)
    walks = [0]
    original = model.parameters

    def parameters(*args, **kwargs):
        walks[0] += 1
        return original(*args, **kwargs)

    model.parameters = parameters
    try:
        plan = fp._plan(model, stacks, heads)
    finally:
        del model.parameters
    return plan, walks[0]


def test_plan_module_is_this_tree():
    assert pathlib.Path(fp.__file__).resolve() == _PATH


def test_every_layer_trained_skips_the_full_parameter_walk():
    plan, walks = _plan_counting(_model(None))
    assert plan is None
    assert walks == 0


def test_upper_layers_only_still_plans_the_cut():
    model = _model([2, 3])
    plan, walks = _plan_counting(model)
    assert plan is not None
    stack, k = plan
    assert stack is model.base_model.model.model.layers and k == 2
    assert walks == 1


def test_trainable_outside_the_stack_still_refuses():
    model = _model([2, 3])
    model.base_model.model.model.norm.weight.requires_grad_(True)
    plan, _ = _plan_counting(model)
    assert plan is None


def test_frozen_vision_tower_with_text_adapters_everywhere_skips_the_walk():
    config_cls = getattr(transformers, "Qwen2VLConfig", None)
    model_cls = getattr(transformers, "Qwen2VLForConditionalGeneration", None)
    if model_cls is None:
        pytest.skip("this transformers has no Qwen2-VL")
    torch.manual_seed(0)
    config = config_cls(
        vision_config = dict(depth = 3, embed_dim = 32, hidden_size = 32, num_heads = 4, mlp_ratio = 2),
        text_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 3,
            vocab_size = 64,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            rope_scaling = {"type": "mrope", "mrope_section": [1, 1, 2]},
        ),
    )
    model = model_cls(config).train()
    model.requires_grad_(False)
    for (
        layer
    ) in model.model.language_model.layers:  # adapters on every text layer, vision tower frozen
        layer.self_attn.q_proj.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    stacks = fp._decoder_stacks(model)
    assert len(stacks) == 2
    plan, walks = _plan_counting(model)
    assert plan is None
    assert walks == 0
