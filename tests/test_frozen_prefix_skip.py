# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Frozen-prefix skip: with adapters only on the upper layers, reentrant gradient checkpointing must not
replay the frozen layers below, and loss / gradients must stay bitwise equal. CPU, tiny random Llama."""

import importlib.util
import os
import pathlib

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
pytest.importorskip("transformers.modeling_layers")

_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_frozen_prefix.py"
_spec = importlib.util.spec_from_file_location("_unsloth_frozen_prefix_under_test", _PATH)
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)

LAYERS = 4


def _model(
    layers_to_transform,
    extra_trainable = None,
    install = True,
):
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
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config)
    lora = peft.LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = ["q_proj", "v_proj", "down_proj"],
        layers_to_transform = layers_to_transform,
        init_lora_weights = False,
    )
    model = peft.get_peft_model(model, lora)
    if extra_trainable is not None:
        for name, param in model.named_parameters():
            if extra_trainable in name:
                param.requires_grad_(True)
    # Unsloth's setup: reentrant checkpointing plus an embedding output that requires grad.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.enable_input_require_grads()
    model.train()
    if install:
        fp.install_frozen_prefix_skip(model)
    return model


def _step(model):
    calls = [0] * LAYERS
    layers = model.base_model.model.model.layers
    handles = [
        # A submodule: transformers >= 5.23 checkpoints inside the layer's forward, not around its hooks.
        layer.self_attn.register_forward_hook(
            lambda m, a, o, i = i: calls.__setitem__(i, calls[i] + 1)
        )
        for i, layer in enumerate(layers)
    ]
    ids = torch.randint(0, 128, (2, 16), generator = torch.Generator().manual_seed(1))
    model.zero_grad(set_to_none = True)
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    for handle in handles:
        handle.remove()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    flags = [layer.gradient_checkpointing for layer in layers]
    return loss.detach(), grads, calls, flags


def _assert_same(a, b):
    assert torch.equal(a[0], b[0])
    assert a[1].keys() == b[1].keys()
    for name in a[1]:
        assert a[1][name] is not None and b[1][name] is not None, name
        assert torch.equal(a[1][name], b[1][name]), name


@pytest.mark.parametrize("layers_to_transform", [[2, 3], [3], [1, 3]])
def test_frozen_layers_run_once_and_grads_match(monkeypatch, layers_to_transform):
    k = layers_to_transform[0]
    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "0")
    base = _step(_model(layers_to_transform))
    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "1")
    before = dict(fp.FROZEN_PREFIX_STATS)
    head = _step(_model(layers_to_transform))
    _assert_same(base, head)
    # Checkpointed layers run twice (forward + replay); the frozen prefix only once with the skip.
    assert base[2] == [2] * LAYERS
    assert head[2] == [1] * k + [2] * (LAYERS - k)
    assert fp.FROZEN_PREFIX_STATS["forwards"] == before["forwards"] + 1
    assert fp.FROZEN_PREFIX_STATS["layers"] == before["layers"] + k
    # The checkpoint flags come back after the forward.
    assert head[3] == [True] * LAYERS


@pytest.mark.parametrize("extra", ["embed_tokens", "layers.0.input_layernorm"])
def test_trainable_below_the_cut_keeps_full_backward(extra):
    model = _model([2, 3], extra_trainable = extra)
    before = dict(fp.FROZEN_PREFIX_STATS)
    loss, grads, calls, _ = _step(model)
    assert calls == [2] * LAYERS
    assert fp.FROZEN_PREFIX_STATS == before
    assert any(extra in name and grad is not None for name, grad in grads.items())


def test_adapter_on_layer_zero_and_no_grad_forward_are_untouched():
    model = _model([0, 3])
    before = dict(fp.FROZEN_PREFIX_STATS)
    assert _step(model)[2] == [2] * LAYERS
    model = _model([2, 3])
    with torch.no_grad():
        model(input_ids = torch.zeros(1, 4, dtype = torch.long))
    model.eval()
    model(input_ids = torch.zeros(1, 4, dtype = torch.long))
    assert fp.FROZEN_PREFIX_STATS == before


def test_inputs_embeds_that_require_grad_keep_their_gradient():
    model = _model([2, 3])
    embeds = model.get_input_embeddings()(torch.zeros(1, 8, dtype = torch.long)).detach()
    embeds.requires_grad_(True)
    before = dict(fp.FROZEN_PREFIX_STATS)
    model(inputs_embeds = embeds).logits.sum().backward()
    assert embeds.grad is not None and embeds.grad.abs().sum() > 0
    assert fp.FROZEN_PREFIX_STATS == before


def test_first_trainable_layer_outside_checkpointing_still_matches(monkeypatch):
    # checkpoint_skip_layers can leave the first trainable layer unchecked while the rest replay.
    def build():
        model = _model([2, 3])
        model.base_model.model.model.layers[2].gradient_checkpointing = False
        return model

    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "0")
    base = _step(build())
    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "1")
    head = _step(build())
    _assert_same(base, head)
    assert head[2][:2] == [1, 1] and base[2][:2] == [2, 2]


def test_no_checkpointing_or_block_swap_is_untouched():
    model = _model([2, 3])
    model.gradient_checkpointing_disable()
    before = dict(fp.FROZEN_PREFIX_STATS)
    _step(model)
    assert fp.FROZEN_PREFIX_STATS == before
    model = _model([2, 3])
    model._unsloth_block_swap = object()
    assert _step(model)[2] == [2] * LAYERS
    assert fp.FROZEN_PREFIX_STATS == before
