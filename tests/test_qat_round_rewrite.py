# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A rounded QAT base weight that is rewritten after prepare (_qat_round.py) must be fake quantized again.

load_state_dict of a different base weight must give exactly what UNSLOTH_QAT_ROUND_INPLACE=0 gives after the
same load, and round_frozen_qat_weights must round an in place rewrite again. Reloading the rounded weight
itself, adapter only loads and copies must leave the rounded weight alone (int8 fake quant is not
idempotent, so a second rounding would show).
"""

import copy
import os

import pytest
import torch

pytest.importorskip("torchao.quantization.qat")
peft = pytest.importorskip("peft")

import unsloth  # noqa: F401
from unsloth.models import _qat_round as qr
from unsloth.models._utils import _prepare_model_for_qat

DIM = 128


def test_imports_this_tree():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(qr.__file__).startswith(here), qr.__file__


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(DIM, 2 * DIM, bias = False)
        self.down_proj = torch.nn.Linear(2 * DIM, DIM, bias = False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


def _model(scheme, monkeypatch, mode):
    monkeypatch.setenv("UNSLOTH_QAT_ROUND_INPLACE", mode)
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
    torch.manual_seed(0)
    model = peft.get_peft_model(
        _Block(),
        peft.LoraConfig(r = 8, lora_alpha = 8, target_modules = ["up_proj", "down_proj"]),
    )
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std = 0.02)
    model = _prepare_model_for_qat(model, scheme)
    monkeypatch.setenv("UNSLOTH_QAT_ROUND_INPLACE", "1")
    return model


def _base(model, name = "up_proj"):
    return getattr(model.base_model.model, name).base_layer


def _out(model, x):
    with torch.no_grad():
        return model(x)


def _replacement():
    return torch.randn(2 * DIM, DIM, generator = torch.Generator().manual_seed(9))


@pytest.mark.parametrize("scheme", ["int4", "int8"])
@pytest.mark.parametrize("whole_model", [False, True])
def test_load_state_dict_of_a_new_weight(scheme, whole_model, monkeypatch):
    x = torch.randn(4, DIM)
    ref = _model(scheme, monkeypatch, "0")
    model = _model(scheme, monkeypatch, "1")
    assert _base(model).weight_fake_quantizer.enabled is False  # rounded
    assert torch.equal(_out(model, x), _out(ref, x))
    for m in (ref, model):
        if whole_model:
            state = m.state_dict()
            key = next(k for k in state if k.endswith("up_proj.base_layer.weight"))
            state[key] = _replacement()
            m.load_state_dict(state)
        else:
            _base(m).load_state_dict({"weight": _replacement()})
    assert torch.equal(_out(model, x), _out(ref, x))
    assert _base(model).weight_fake_quantizer.enabled is False  # rounded again, in place


def test_round_again_after_in_place_rewrite(monkeypatch):
    x = torch.randn(4, DIM)
    ref = _model("int8", monkeypatch, "0")
    model = _model("int8", monkeypatch, "1")
    with torch.no_grad():
        for m in (ref, model):
            _base(m).weight.copy_(_replacement())
    assert qr.round_frozen_qat_weights(model) == 1
    assert _base(model).weight_fake_quantizer.enabled is False
    assert torch.equal(_out(model, x), _out(ref, x))


def test_unchanged_weights_stay_rounded(monkeypatch):
    x = torch.randn(4, DIM)
    model = _model("int8", monkeypatch, "1")
    before = _out(model, x)
    model.load_state_dict(model.state_dict())  # its own rounded weights
    assert torch.equal(_out(model, x), before)
    adapters = peft.get_peft_model_state_dict(model)
    peft.set_peft_model_state_dict(model, adapters)  # adapter only (Trainer resume)
    assert torch.equal(_out(model, x), before)
    assert qr.round_frozen_qat_weights(model) == 0
    clone = copy.deepcopy(model)
    assert qr.round_frozen_qat_weights(clone) == 0
    clone.load_state_dict(model.state_dict())
    assert torch.equal(_out(clone, x), before)
    assert _base(clone).weight_fake_quantizer.enabled is False
