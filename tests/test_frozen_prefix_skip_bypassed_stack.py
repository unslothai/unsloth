# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Frozen-prefix skip planned on a stack the forward does not run: training only the top vision block
(and an untied head) of a VLM, a text-only batch never enters the vision tower. That forward skipped
nothing, so it must train exactly as with the skip off instead of raising. CPU, tiny random Qwen2-VL."""

import importlib.util
import pathlib

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.modeling_layers")
Qwen2VLConfig = getattr(transformers, "Qwen2VLConfig", None)
Qwen2VLForConditionalGeneration = getattr(transformers, "Qwen2VLForConditionalGeneration", None)
if Qwen2VLForConditionalGeneration is None:
    pytest.skip("this transformers has no Qwen2-VL", allow_module_level = True)

_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_frozen_prefix.py"
_spec = importlib.util.spec_from_file_location("_unsloth_frozen_prefix_bypass_under_test", _PATH)
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)


def _model(install):
    torch.manual_seed(0)
    config = Qwen2VLConfig(
        vision_config = dict(depth = 2, embed_dim = 32, hidden_size = 32, num_heads = 4, mlp_ratio = 2),
        text_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 2,
            vocab_size = 64,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            rope_scaling = {"type": "mrope", "mrope_section": [1, 1, 2]},
            tie_word_embeddings = False,
        ),
        tie_word_embeddings = False,
    )
    model = Qwen2VLForConditionalGeneration(config).train()
    model.requires_grad_(False)
    model.model.visual.blocks[-1].requires_grad_(True)
    model.lm_head.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.enable_input_require_grads()
    if install:
        fp.install_frozen_prefix_skip(model)
    return model


def _step(model):
    ids = torch.arange(4).unsqueeze(0)
    model.zero_grad(set_to_none = True)
    loss = model(input_ids = ids, labels = ids, use_cache = False).loss
    loss.backward()
    return loss.detach(), model.lm_head.weight.grad.clone()


def test_frozen_prefix_module_is_this_tree():
    assert pathlib.Path(fp.__file__).resolve() == _PATH


def test_text_only_batch_with_vision_stack_planned(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "1")
    ref_loss, ref_grad = _step(_model(install = False))
    model = _model(install = True)
    stacks = fp._decoder_stacks(model)
    plan = fp._plan(model, stacks, fp._head_parameters(model))
    assert plan is not None and plan[0] is model.model.visual.blocks  # the skip would engage
    before = dict(fp.FROZEN_PREFIX_STATS)
    for _ in range(2):  # also the step after: a raise used to disable the skip for the run
        loss, grad = _step(model)
        assert torch.equal(loss, ref_loss)
        assert torch.equal(grad, ref_grad)
    assert fp.FROZEN_PREFIX_STATS == before  # nothing was skipped
    blocks = model.model.visual.blocks
    assert all(block.gradient_checkpointing for block in blocks)
    assert not any(block._forward_pre_hooks or block._forward_hooks for block in blocks)
