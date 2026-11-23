# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The labelled forward must train an unmerged LoRA adapter on lm_head and keep a head bias in the loss.

The fused CE projects with lm_head.weight and no bias. On a PEFT LoRA layer .weight is the base weight, so the adapter
got no gradient and the loss ignored it; a head bias dropped out the same way.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast

peft_lora = pytest.importorskip("peft.tuners.lora.layer")
from peft import LoraConfig

import unsloth.models.llama as llama_module
import unsloth.models.mistral as mistral_module
from unsloth.models._utils import EMPTY_LOGITS

VOCAB, HIDDEN, SEQ = 16, 8, 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _Backbone:
    def __init__(self, hidden):
        self.hidden = hidden

    def __call__(self, *args, **kwargs):
        return BaseModelOutputWithPast(last_hidden_state = self.hidden)


def _head(kind):
    if kind != "lora":
        head = torch.nn.Linear(HIDDEN, VOCAB, bias = kind == "bias", device = DEVICE)
        if kind == "bias":
            with torch.no_grad():
                head.bias.copy_(torch.linspace(-1, 1, VOCAB))
        return head
    base = torch.nn.Linear(HIDDEN, VOCAB, bias = False, device = DEVICE)
    base.weight.requires_grad_(False)
    head = peft_lora.Linear(
        base, "default", config = LoraConfig(r = 2, lora_alpha = 2), r = 2, lora_alpha = 2
    ).to(DEVICE)
    with torch.no_grad():
        head.lora_B["default"].weight.copy_(
            torch.linspace(-0.5, 0.5, VOCAB * 2, device = DEVICE).reshape(VOCAB, 2)
        )
    return head


def _stub(kind):
    torch.manual_seed(0)
    hidden = torch.randn(1, SEQ, HIDDEN, device = DEVICE, requires_grad = True)
    config = SimpleNamespace(
        output_attentions = False,
        output_hidden_states = False,
        use_return_dict = True,
        model_type = "llama",
        final_logit_softcapping = 0,
        logit_scale = 0,
        torch_dtype = torch.float32,
    )
    stub = SimpleNamespace(
        model = _Backbone(hidden), lm_head = _head(kind), training = True, config = config
    )
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


def _loss(monkeypatch, module_name, stub):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    if DEVICE == "cpu":
        # The Triton CE kernel needs a GPU; the routing under test is the same.
        for module in (llama_module, mistral_module):
            monkeypatch.setattr(module, "fast_cross_entropy_loss", _torch_ce)
    labels = torch.tensor([[3, 5, 1, 7, 2, 9]], device = DEVICE)
    output = _forward(module_name)(
        stub, input_ids = torch.zeros(1, SEQ, dtype = torch.long, device = DEVICE), labels = labels
    )
    return output, labels


@pytest.mark.parametrize("kind", ["lora", "bias"])
@pytest.mark.parametrize("module_name", ["llama", "mistral"])
def test_labelled_forward_trains_the_head_adapter_and_bias(monkeypatch, module_name, kind):
    stub, hidden = _stub(kind)
    output, labels = _loss(monkeypatch, module_name, stub)
    logits = stub.lm_head(hidden.detach())[:, :-1]
    expected = F.cross_entropy(logits.reshape(-1, VOCAB).double(), labels[:, 1:].reshape(-1))
    assert output.loss.item() == pytest.approx(expected.item(), rel = 1e-4)
    output.loss.backward()
    trained = {n: p.grad for n, p in stub.lm_head.named_parameters() if p.requires_grad}
    assert trained
    for name, grad in trained.items():
        assert grad is not None and grad.abs().sum() > 0, name


@pytest.mark.parametrize("module_name", ["llama", "mistral"])
def test_plain_head_and_merged_adapter_stay_fused(monkeypatch, module_name):
    stub, _ = _stub("plain")
    output, _ = _loss(monkeypatch, module_name, stub)
    assert output.logits is EMPTY_LOGITS
    stub, hidden = _stub("lora")
    stub.lm_head.merge()
    output, labels = _loss(monkeypatch, module_name, stub)
    assert output.logits is EMPTY_LOGITS
    logits = stub.lm_head(hidden.detach())[:, :-1]
    expected = F.cross_entropy(logits.reshape(-1, VOCAB).double(), labels[:, 1:].reshape(-1))
    assert output.loss.item() == pytest.approx(expected.item(), rel = 1e-4)


@pytest.mark.parametrize("module_name", ["llama", "mistral"])
def test_modules_to_save_head_stays_fused_and_trains_its_copy(monkeypatch, module_name):
    # modules_to_save resolves .weight to the trainable copy, so the fused CE is already exact there.
    from peft.utils.other import ModulesToSaveWrapper

    stub, hidden = _stub("plain")
    stub.lm_head.weight.requires_grad_(False)
    stub.lm_head = ModulesToSaveWrapper(stub.lm_head, "default")
    stub.lm_head.modules_to_save["default"].weight.requires_grad_(True)
    output, labels = _loss(monkeypatch, module_name, stub)
    assert output.logits is EMPTY_LOGITS
    logits = stub.lm_head(hidden.detach())[:, :-1]
    expected = F.cross_entropy(logits.reshape(-1, VOCAB).double(), labels[:, 1:].reshape(-1))
    assert output.loss.item() == pytest.approx(expected.item(), rel = 1e-4)
    output.loss.backward()
    grad = stub.lm_head.modules_to_save["default"].weight.grad
    assert grad is not None and grad.abs().sum() > 0
