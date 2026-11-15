# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallel hands the forward pre-shifted, globally boundary-masked shift_labels."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast

import unsloth.models.llama as llama_module
import unsloth.models.mistral as mistral_module

VOCAB, HIDDEN, SEQ = 16, 8, 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _Backbone:
    def __init__(self, hidden):
        self.hidden = hidden

    def __call__(self, *args, **kwargs):
        return BaseModelOutputWithPast(last_hidden_state = self.hidden)


def _stub():
    torch.manual_seed(0)
    hidden = torch.randn(1, SEQ, HIDDEN, device = DEVICE, requires_grad = True)
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias = False, device = DEVICE)
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
    flat = labels.reshape(-1)
    total = F.cross_entropy(logits.float().reshape(flat.numel(), -1), flat, reduction = "sum")
    return total / (n_items if n_items is not None else flat.ne(-100).sum())


def _forward(module_name):
    if module_name == "llama":
        return llama_module.CausalLM_fast_forward(lambda *a, **k: None)
    return mistral_module.MistralForCausalLM_fast_forward


@pytest.mark.parametrize("return_logits", ["0", "1"], ids = ["fused", "logits"])
@pytest.mark.parametrize("with_labels", [True, False], ids = ["labels", "shift_only"])
@pytest.mark.parametrize("module_name", ["llama", "mistral"])
def test_cp_shard_keeps_its_boundary_target(monkeypatch, module_name, with_labels, return_logits):
    # Rank 0 of CP2 over 8 tokens where only labels[4] is a target: its last slot predicts token 4.
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", return_logits)
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    if DEVICE == "cpu":
        # The Triton CE kernel needs a GPU; the routing under test is the same.
        for module in (llama_module, mistral_module):
            monkeypatch.setattr(module, "fast_cross_entropy_loss", _torch_ce)
    stub, hidden = _stub()
    shift_labels = torch.tensor([[-100, -100, -100, 4]], device = DEVICE)
    kwargs = dict(
        input_ids = torch.zeros(1, SEQ, dtype = torch.long, device = DEVICE), shift_labels = shift_labels
    )
    if with_labels:
        kwargs["labels"] = torch.full((1, SEQ), -100, device = DEVICE)
    kwargs["packed_seq_lengths"] = torch.tensor([3, 1], dtype = torch.int32, device = DEVICE)
    loss = _forward(module_name)(stub, **kwargs).loss

    logits = stub.lm_head(hidden.detach()).double()
    expected = F.cross_entropy(logits.reshape(-1, VOCAB), shift_labels.reshape(-1))
    assert loss is not None
    assert loss.item() == pytest.approx(expected.item(), rel = 1e-4)
    loss.backward()
    assert hidden.grad[0, -1].abs().sum() > 0
    assert shift_labels.tolist() == [[-100, -100, -100, 4]]
