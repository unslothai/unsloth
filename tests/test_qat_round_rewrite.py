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


def _model(
    scheme,
    monkeypatch,
    mode,
    seed = 0,
):
    monkeypatch.setenv("UNSLOTH_QAT_ROUND_INPLACE", mode)
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
    torch.manual_seed(seed)
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


@pytest.mark.parametrize("scheme", ["int4", "int8"])
@pytest.mark.parametrize("through_disk", [False, True])
def test_load_state_dict_of_another_rounded_model(scheme, through_disk, monkeypatch):
    # The source's base weights are already rounded and differ from the destination's: rounding them again
    # would move int8 weights (not idempotent), so the destination must reproduce the source exactly.
    import io

    x = torch.randn(4, DIM)
    source = _model(scheme, monkeypatch, "1", seed = 1)
    destination = _model(scheme, monkeypatch, "1", seed = 2)
    expected = _out(source, x)
    assert not torch.equal(_out(destination, x), expected)
    state = source.state_dict()
    if through_disk:
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only = True)
    destination.load_state_dict(state)
    for name in ("up_proj", "down_proj"):
        assert torch.equal(_base(destination, name).weight, _base(source, name).weight)
        assert _base(destination, name).weight_fake_quantizer.enabled is False
    assert torch.equal(_out(destination, x), expected)
    assert qr.round_frozen_qat_weights(destination) == 0
    assert torch.equal(_out(destination, x), expected)


@pytest.mark.parametrize("scheme", ["int4", "int8"])
def test_rounded_provenance_survives_a_device_move(scheme, monkeypatch):
    # model.cpu() / .to() give the weights new storage with the same rounded values: still marked, so a
    # differently initialised prepared model loads them unchanged. An edit after the move is not marked.
    x = torch.randn(4, DIM)
    source = _model(scheme, monkeypatch, "1", seed = 1)
    destination = _model(scheme, monkeypatch, "1", seed = 2)
    expected = _out(source, x)
    source._apply(lambda t: t.clone())  # what a device move does to every parameter
    destination.load_state_dict(source.state_dict())
    assert torch.equal(_out(destination, x), expected)
    with torch.no_grad():
        _base(source, "up_proj").weight.add_(1e-3)
    marked = [k for k, v in source.state_dict()._metadata.items() if qr._META in v]
    assert not any(k.endswith("up_proj.base_layer") for k in marked)
    assert any(k.endswith("down_proj.base_layer") for k in marked)


@pytest.mark.parametrize("scheme", ["int4", "int8"])
def test_rounding_again_after_a_device_move_is_a_no_op(scheme, monkeypatch):
    # round_frozen_qat_weights after model.cpu() / .to(): same rounded values in new storage, so nothing is
    # rounded a second time (int8 fake quant is not idempotent); an edited weight is still re-rounded.
    x = torch.randn(4, DIM)
    model = _model(scheme, monkeypatch, "1")
    expected = _out(model, x)
    model._apply(lambda t: t.clone())
    assert qr.round_frozen_qat_weights(model) == 0
    assert torch.equal(_out(model, x), expected)
    with torch.no_grad():
        _base(model, "up_proj").weight.add_(1e-3)
    assert qr.round_frozen_qat_weights(model) == 1


def test_marked_rounded_state_dict_keeps_tensors_unchanged(monkeypatch):
    # The marker rides in the state dict's per module metadata: no new keys, so exports read the same tensors.
    model = _model("int8", monkeypatch, "1")
    ref = _model("int8", monkeypatch, "0")
    state = model.state_dict()
    assert list(state) == list(ref.state_dict())
    marked = [k for k, v in state._metadata.items() if qr._META in v]
    assert sorted(marked) == sorted(
        k[: -len(".weight")] for k in state if k.endswith("base_layer.weight")
    )
    assert not any(qr._META in v for v in ref.state_dict()._metadata.values())
