# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Training on inputs_embeds under reentrant gradient checkpointing (#2178)."""

import pytest

transformers = pytest.importorskip("transformers")

import unsloth  # noqa: E402,F401  applies patch_enable_input_require_grads
import torch  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

HOOK_NAME = "make_inputs_embeds_require_grads"


def _model():
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        vocab_size = 64,
    )
    model = LlamaForCausalLM(config).float()
    model.requires_grad_(False)
    # Stand-in for a LoRA weight: trainable, inside a checkpointed layer.
    trainable = model.model.layers[0].self_attn.q_proj.weight
    trainable.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.enable_input_require_grads()
    model.train()
    return model, trainable


def _grad(model, trainable, **inputs):
    trainable.grad = None
    ids = torch.arange(12).reshape(2, 6)
    loss = model(labels = ids, **inputs).loss
    loss.backward()
    return loss.detach(), trainable.grad.clone()


def test_precomputed_inputs_embeds_match_input_ids():
    model, trainable = _model()
    ids = torch.arange(12).reshape(2, 6)
    loss_ids, grad_ids = _grad(model, trainable, input_ids = ids)
    # F.embedding skips the embedding's own hook, as a precomputed tensor would.
    embeds = torch.nn.functional.embedding(ids, model.get_input_embeddings().weight)
    loss_embeds, grad_embeds = _grad(model, trainable, inputs_embeds = embeds)
    assert torch.equal(loss_ids, loss_embeds)
    assert grad_ids.abs().sum() > 0
    torch.testing.assert_close(grad_embeds, grad_ids)
    # The caller's tensor is aliased, never flipped in place.
    assert not embeds.requires_grad and embeds.grad is None


def test_upstream_projection_still_gets_gradients():
    model, trainable = _model()
    proj = torch.nn.Linear(5, model.config.hidden_size)
    embeds = proj(torch.randn(2, 6, 5))
    _grad(model, trainable, inputs_embeds = embeds)
    assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0


def test_hook_is_registered_once_and_removed_by_disable():
    model, _ = _model()
    model.enable_input_require_grads()

    def count():
        return sum(
            getattr(hook, "__name__", None) == HOOK_NAME
            for module in model.modules()
            for hook in module._forward_pre_hooks.values()
        )

    assert count() == 2  # LlamaForCausalLM and LlamaModel, once each
    model.disable_input_require_grads()
    assert count() == 0


def test_no_grad_forward_is_untouched():
    model, _ = _model()
    model.eval()
    ids = torch.arange(12).reshape(2, 6)
    embeds = torch.nn.functional.embedding(ids, model.get_input_embeddings().weight)
    with torch.no_grad():
        out = model(inputs_embeds = embeds)
    assert not out.logits.requires_grad
