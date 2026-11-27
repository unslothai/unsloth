# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Frozen-prefix skip with a gradient wanted at a nested input: transformers takes `attention_mask` as a
{layer type: mask} mapping, and a mask that requires grad inside it must get the same gradient as with
the skip off (the skip used to check only top-level inputs and detached it below the cut). CPU, Qwen3."""

import importlib.util
import pathlib

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
pytest.importorskip("transformers.modeling_layers")

_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_frozen_prefix.py"
_spec = importlib.util.spec_from_file_location("_unsloth_frozen_prefix_nested_under_test", _PATH)
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)


def _run(install, nested):
    config = transformers.Qwen3Config(
        vocab_size = 32,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    model = peft.get_peft_model(
        transformers.Qwen3ForCausalLM(config),
        peft.LoraConfig(r = 2, target_modules = ["q_proj"], layers_to_transform = [2]),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    if install:
        fp.install_frozen_prefix_skip(model)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    bias = torch.zeros(1, 1, 6, 6, requires_grad = nested)
    mask = {"full_attention": bias}
    loss = model(input_ids = ids, labels = ids, attention_mask = mask, use_cache = False).loss
    loss.backward()
    lora = [p.grad.clone() for p in model.parameters() if p.requires_grad]
    return loss.detach(), bias.grad, lora


def test_frozen_prefix_module_is_this_tree():
    assert pathlib.Path(fp.__file__).resolve() == _PATH


def test_mask_mapping_without_grad_still_engages(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "1")
    before = fp.FROZEN_PREFIX_STATS["forwards"]
    _run(install = True, nested = False)
    assert fp.FROZEN_PREFIX_STATS["forwards"] == before + 1


def test_mask_mapping_requiring_grad_keeps_its_gradient(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "1")
    ref_loss, ref_bias, ref_lora = _run(install = False, nested = True)
    before = fp.FROZEN_PREFIX_STATS["forwards"]
    loss, bias, lora = _run(install = True, nested = True)
    assert fp.FROZEN_PREFIX_STATS["forwards"] == before  # stood aside
    assert torch.equal(loss, ref_loss)
    torch.testing.assert_close(bias, ref_bias, rtol = 0, atol = 0)
    for grad, ref in zip(lora, ref_lora):
        torch.testing.assert_close(grad, ref, rtol = 0, atol = 0)
