# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""for_training() on a model loaded with use_gradient_checkpointing=False used to crash on the next forward."""

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
    assert resolve_training_gradient_checkpointing(torch.nn.Linear(2, 2), None) is True
    assert resolve_training_gradient_checkpointing(torch.nn.Linear(2, 2), False) is False


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("armed", [True, False])
def test_unrecorded_model_keeps_its_load_time_choice(which, armed):
    fast = _fast_classes()[which]
    model = _tiny_qwen3()
    if armed:
        model.gradient_checkpointing_enable()
    fast.for_training(model)
    assert all(bool(layer.gradient_checkpointing) == armed for layer in _decoder_layers(model))


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, hidden):
        return torch.tanh(self.linear(hidden))


class _RemoteBackbone(torch.nn.Module):
    """Older / remote-code protocol: the backbone, not the block, calls the checkpoint function."""

    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = torch.nn.ModuleList([_Block(), _Block()])

    def forward(self, hidden):
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = self._gradient_checkpointing_func(layer.__call__, hidden)
            else:
                hidden = layer(hidden)
        return hidden


class _RemoteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _RemoteBackbone()
        self.enabled_by = []

    def gradient_checkpointing_enable(self):
        from torch.utils.checkpoint import checkpoint
        import functools

        self.enabled_by.append("model")
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = functools.partial(
                    checkpoint, use_reentrant = False
                )
                module.gradient_checkpointing = True

    def forward(self, hidden):
        return self.model(hidden)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("arg", [True, None, False])
def test_backbone_that_calls_the_checkpoint_function_itself(which, arg):
    fast = _fast_classes()[which]
    model = _RemoteModel()
    fast.for_training(model, use_gradient_checkpointing = arg)
    assert model.model.gradient_checkpointing == bool(arg)
    assert model.enabled_by == (["model"] if arg else [])
    model(torch.randn(2, 8, requires_grad = True)).sum().backward()


@pytest.mark.parametrize("which", [0, 1])
def test_explicit_enable_goes_through_the_outer_override(which):
    fast = _fast_classes()[which]
    model = _tiny_qwen3()
    model._unsloth_gradient_checkpointing = False
    calls = []
    original = model.gradient_checkpointing_enable

    def override(**kwargs):
        calls.append(kwargs)
        return original(gradient_checkpointing_kwargs = {"use_reentrant": True})

    model.gradient_checkpointing_enable = override
    fast.for_training(model, use_gradient_checkpointing = True)
    assert calls == [{}]
    assert all(
        layer._gradient_checkpointing_func.keywords == {"use_reentrant": True}
        for layer in _decoder_layers(model)
    )


class _SlowDiffusion(torch.nn.Module):
    _unsloth_slow_diffusion = True

    def __init__(self, recorded):
        super().__init__()
        self._unsloth_gradient_checkpointing = recorded
        self.enabled = 0

    def gradient_checkpointing_enable(self):
        self.enabled += 1


@pytest.mark.parametrize("recorded", [True, False])
def test_fastmodel_default_keeps_the_diffusion_adapter_choice(recorded):
    from unsloth.models.loader import FastModel

    model = _SlowDiffusion(recorded)
    FastModel.for_training(model)
    assert model.enabled == int(recorded)
    FastModel.for_training(model, use_gradient_checkpointing = True)
    assert model.enabled == int(recorded) + 1


@pytest.mark.parametrize("which", [0, 1])
def test_late_enable_turns_the_cache_off_and_inference_restores_it(which):
    fast = _fast_classes()[which]
    model = _tiny_qwen3()
    model._unsloth_gradient_checkpointing = False
    assert model.config.use_cache
    fast.for_training(model, use_gradient_checkpointing = True)
    assert model.config.use_cache is False
    fast.for_inference(model)
    assert model.config.use_cache is True


def test_unreadable_forward_counts_as_checkpointing():
    from unsloth.models._utils import _forward_reads_checkpoint_function

    class _NoSource(torch.nn.Module):
        gradient_checkpointing = False

    _NoSource.forward = eval("lambda self, x: x")
    assert _forward_reads_checkpoint_function(_NoSource) is True
