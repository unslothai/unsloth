# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""for_training() on a model loaded with use_gradient_checkpointing=False.

The default used to be True, which flipped every transformers GradientCheckpointingLayer on even
though gradient_checkpointing_enable() never gave it a `_gradient_checkpointing_func`, so the next
training forward raised AttributeError.
"""

import pytest
import torch


def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config)


def _decoder_layers(model):
    return [m for m in model.modules() if type(m).__name__.endswith("DecoderLayer")]


def _fast_classes():
    from unsloth.models.llama import FastLlamaModel
    from unsloth.models.vision import FastBaseModel
    return [FastLlamaModel, FastBaseModel]


@pytest.mark.parametrize("which", [0, 1])
def test_default_keeps_checkpointing_off_when_it_was_off_at_load(which):
    fast = _fast_classes()[which]
    model = _tiny_qwen3()
    model._unsloth_gradient_checkpointing = False
    fast.for_training(model)
    assert all(not layer.gradient_checkpointing for layer in _decoder_layers(model))
    ids = torch.randint(0, 64, (1, 8))
    model(input_ids = ids, labels = ids).loss.backward()


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("arg", [True, "unsloth"])
def test_explicit_enable_arms_a_model_loaded_without_checkpointing(which, arg):
    fast = _fast_classes()[which]
    model = _tiny_qwen3()
    model._unsloth_gradient_checkpointing = False
    fast.for_training(model, use_gradient_checkpointing = arg)
    layers = _decoder_layers(model)
    assert layers and all(layer.gradient_checkpointing == arg for layer in layers)
    assert all(callable(layer._gradient_checkpointing_func) for layer in layers)
    ids = torch.randint(0, 64, (1, 8))
    model(input_ids = ids, labels = ids).loss.backward()
    assert all(p.grad is not None for p in model.model.layers.parameters())


@pytest.mark.parametrize("which", [0, 1])
def test_for_training_default_restores_the_recorded_mode(which):
    fast = _fast_classes()[which]
    model = _tiny_qwen3()
    model.gradient_checkpointing_enable()
    model._unsloth_gradient_checkpointing = "unsloth"
    fast.for_training(model)
    layers = _decoder_layers(model)
    assert layers and all(layer.gradient_checkpointing == "unsloth" for layer in layers)
    fast.for_training(model, use_gradient_checkpointing = False)
    assert all(not layer.gradient_checkpointing for layer in layers)


def test_unrecorded_model_keeps_the_old_default_of_on():
    from unsloth.models._utils import resolve_training_gradient_checkpointing
    assert resolve_training_gradient_checkpointing(object(), None) is True
    assert resolve_training_gradient_checkpointing(object(), False) is False
