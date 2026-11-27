# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Llama / Mistral CausalLM head routing: logits_to_keep with labels, shift-only labels on the fused
loss, and a biased lm_head on the single-token matvec decode."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast

import unsloth.models.llama as llama_module
import unsloth.models.mistral as mistral_module

VOCAB, HIDDEN, SEQ = 16, 8, 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _Backbone:
    def __init__(self, hidden):
        self.hidden = hidden

    def __call__(self, *args, **kwargs):
        return BaseModelOutputWithPast(last_hidden_state = self.hidden)


def _stub(seq = SEQ, bias = False):
    torch.manual_seed(0)
    hidden = torch.randn(1, seq, HIDDEN, device = DEVICE, requires_grad = True)
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias = bias, device = DEVICE)
    config = SimpleNamespace(
        output_attentions = False,
        output_hidden_states = False,
        use_return_dict = True,
        model_type = "llama",
        final_logit_softcapping = 0,
        logit_scale = 0,
        torch_dtype = torch.float32,
    )
    stub = SimpleNamespace(model = _Backbone(hidden), lm_head = lm_head, training = True, config = config)
    return stub, hidden


def _torch_ce(
    logits,
    labels,
    logit_softcapping = 0,
    logit_scaling = 0,
    n_items = None,
    **_,
):
    # The Triton kernel asserts the same: one row of logits per target.
    assert logits.shape[:-1] == labels.shape, (logits.shape, labels.shape)
    flat = labels.reshape(-1)
    total = F.cross_entropy(logits.float().reshape(flat.numel(), -1), flat, reduction = "sum")
    return total / (n_items if n_items is not None else flat.ne(-100).sum())


def _forward(module_name):
    if module_name == "llama":
        return llama_module.CausalLM_fast_forward(lambda *a, **k: None)
    return mistral_module.MistralForCausalLM_fast_forward


@pytest.fixture(autouse = True)
def _cpu_ce(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    if DEVICE == "cpu":
        # The Triton CE kernel needs a GPU; the routing under test is the same.
        for module in (llama_module, mistral_module):
            monkeypatch.setattr(module, "fast_cross_entropy_loss", _torch_ce)


@pytest.mark.parametrize("return_logits", ["0", "1"], ids = ["fused", "logits"])
@pytest.mark.parametrize("module_name", ["llama", "mistral"])
def test_logits_to_keep_ignored_with_labels(monkeypatch, module_name, return_logits):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", return_logits)
    labels = torch.randint(0, VOCAB, (1, SEQ), device = DEVICE)
    ids = torch.zeros(1, SEQ, dtype = torch.long, device = DEVICE)
    stub, _ = _stub()
    full = _forward(module_name)(stub, input_ids = ids, labels = labels).loss
    stub, _ = _stub()
    kept = _forward(module_name)(stub, input_ids = ids, labels = labels, logits_to_keep = 2).loss
    assert torch.isfinite(kept)
    torch.testing.assert_close(kept, full)


@pytest.mark.parametrize("module_name", ["llama", "mistral"])
def test_shift_only_labels_take_fused_loss(monkeypatch, module_name):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    shift_labels = torch.randint(0, VOCAB, (1, SEQ), device = DEVICE)
    shift_labels[0, -1] = -100
    ids = torch.zeros(1, SEQ, dtype = torch.long, device = DEVICE)
    stub, hidden = _stub()
    out = _forward(module_name)(stub, input_ids = ids, shift_labels = shift_labels)
    # Fused CE never builds the logits.
    assert out.logits is llama_module.EMPTY_LOGITS
    assert stub.model._has_no_labels is False
    logits = stub.lm_head(hidden.detach()).double()
    expected = F.cross_entropy(logits.reshape(-1, VOCAB), shift_labels.reshape(-1))
    assert out.loss.item() == pytest.approx(expected.item(), rel = 1e-4)


@pytest.mark.parametrize("module_name", ["llama", "mistral"])
def test_biased_head_single_token_logits(module_name):
    stub, hidden = _stub(seq = 1, bias = True)
    with torch.no_grad():
        stub.lm_head.bias.fill_(0.5)
    stub.training = False
    ids = torch.zeros(1, 1, dtype = torch.long, device = DEVICE)
    logits = _forward(module_name)(stub, input_ids = ids).logits
    torch.testing.assert_close(logits, stub.lm_head(hidden))


def test_decode_logits_keeps_head_bias():
    stub, hidden = _stub(seq = 1, bias = True)
    with torch.no_grad():
        stub.lm_head.bias.fill_(0.5)
    logits = llama_module._decode_logits(stub, hidden)
    torch.testing.assert_close(logits, stub.lm_head(hidden))
