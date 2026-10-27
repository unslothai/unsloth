# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""External inputs_embeds from a trainable module must keep their graph through LlamaModel_fast_forward.

With a frozen embedding table, the label-free (DPO) attention-mask branch and the Gemma normalizer
used to detach a non-leaf inputs_embeds and scale it in place, so the projector that produced them
got no gradient and the caller's tensor was overwritten.
"""

import math

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.models import llama as llama_module


class _Captured(Exception):
    pass


class _CaptureHidden(torch.nn.Module):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def extend_rope_embedding(self, hidden_states, *args, **kwargs):
        self.store["hidden"] = hidden_states
        raise _Captured


def _model(model_type):
    if model_type == "gemma":
        from transformers import GemmaConfig as Config, GemmaModel as Model
    else:
        from transformers import LlamaConfig as Config, LlamaModel as Model
    config = Config(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        head_dim = 8,
    )
    model = Model(config).train()
    model.embed_tokens.weight.requires_grad_(False)
    model.max_seq_length = 64
    model._has_no_labels = True
    return model


def _forward(model, inputs_embeds, attention_mask):
    store = {}
    model.rotary_emb = _CaptureHidden(store)
    with pytest.raises(_Captured):
        llama_module.LlamaModel_fast_forward(
            model,
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            use_cache = False,
        )
    return store["hidden"]


def _scale(model):
    if model.config.model_type.startswith("gemma") and not llama_module.embedding_applies_scale(
        model.embed_tokens
    ):
        return math.sqrt(model.config.hidden_size)
    return 1.0


@pytest.mark.parametrize("model_type", ["llama", "gemma"])
def test_projector_receives_gradient_through_masked_inputs_embeds(model_type):
    torch.manual_seed(0)
    model = _model(model_type)
    projector = torch.nn.Linear(8, 16)
    features = torch.randn(2, 5, 8)
    attention_mask = torch.ones(2, 5, dtype = torch.long)
    attention_mask[1, :2] = 0

    inputs_embeds = projector(features)
    original = inputs_embeds.detach().clone()
    hidden = _forward(model, inputs_embeds, attention_mask)
    weights = torch.randn_like(hidden)
    (hidden * weights).sum().backward()

    reference = torch.nn.Linear(8, 16)
    reference.load_state_dict(projector.state_dict())
    expected = reference(features) * _scale(model) * attention_mask.unsqueeze(-1)
    (expected * weights).sum().backward()

    assert projector.weight.grad is not None
    assert torch.allclose(projector.weight.grad, reference.weight.grad, atol = 1e-5)
    assert torch.allclose(hidden, expected, atol = 1e-5)
    assert torch.equal(inputs_embeds.detach(), original)


def test_float_mask_keeps_embedding_dtype():
    model = _model("llama").to(torch.bfloat16)
    inputs_embeds = torch.randn(2, 5, 16, dtype = torch.bfloat16)
    attention_mask = torch.ones(2, 5, dtype = torch.float32)
    attention_mask[0, :1] = 0
    hidden = _forward(model, inputs_embeds, attention_mask)
    assert hidden.dtype == torch.bfloat16
    assert torch.equal(hidden[0, 0], torch.zeros(16, dtype = torch.bfloat16))


@pytest.mark.parametrize("model_type", ["llama", "gemma"])
def test_caller_leaf_inputs_embeds_get_gradient(model_type):
    model = _model(model_type)
    inputs_embeds = torch.randn(2, 5, 16, requires_grad = True)
    original = inputs_embeds.detach().clone()
    attention_mask = torch.ones(2, 5, dtype = torch.long)
    attention_mask[0, :2] = 0
    _forward(model, inputs_embeds, attention_mask).sum().backward()
    expected = _scale(model) * attention_mask.unsqueeze(-1).expand(2, 5, 16).float()
    assert torch.allclose(inputs_embeds.grad, expected)
    assert torch.equal(inputs_embeds.detach(), original)


@pytest.mark.parametrize("model_type", ["llama", "gemma"])
def test_own_embedding_output_is_not_kept_for_backward(model_type):
    import weakref

    model = _model(model_type)
    produced = []

    def require_grad(module, args, output):
        # What enable_input_require_grads does for frozen embeddings under LoRA.
        output.requires_grad_(True)
        produced.append(weakref.ref(output))

    model.embed_tokens.register_forward_hook(require_grad)
    store = {}
    model.rotary_emb = _CaptureHidden(store)
    with pytest.raises(_Captured):
        llama_module.LlamaModel_fast_forward(
            model,
            input_ids = torch.randint(1, 64, (2, 5)),
            attention_mask = torch.ones(2, 5, dtype = torch.long),
            use_cache = False,
        )
    hidden = store["hidden"]
    assert hidden.is_leaf and hidden.requires_grad
    # Nothing beyond hidden itself may stay alive (base scaled it in place, so both are one tensor).
    assert produced[0]() is None or produced[0]() is hidden
