# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A rounded QAT base weight that is an inference tensor (built under torch.inference_mode, as weights shared with
vLLM are) has no version counter: a reload into the same storage under inference mode kept its (pointer, None)
state, so the load hook and round_frozen_qat_weights both took it as unchanged and left the raw incoming weight
in service with fake quantization off. CPU only."""

import os

import pytest
import torch

pytest.importorskip("torchao.quantization.qat")
peft = pytest.importorskip("peft")

from tests.test_qat_round_rewrite import _base, _model, _out, _replacement, qr  # noqa: E402


def test_imports_this_tree():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(qr.__file__).startswith(here), qr.__file__


@pytest.mark.parametrize("via", ["load_state_dict", "copy_then_round"])
def test_inference_weight_reloaded_in_place_is_rounded_again(via, monkeypatch):
    x = torch.randn(4, 128)
    with torch.inference_mode():
        ref = _model("int8", monkeypatch, "0")
        model = _model("int8", monkeypatch, "1")
        layer = _base(model)
        assert layer.weight.is_inference() and layer.weight_fake_quantizer.enabled is False
        for m in (ref, model):
            if via == "load_state_dict":
                _base(m).load_state_dict({"weight": _replacement()})
            else:
                _base(m).weight.copy_(_replacement())
        qr.round_frozen_qat_weights(model)
        # Same as per call fake quantization of the reloaded weight (UNSLOTH_QAT_ROUND_INPLACE=0).
        assert torch.equal(_out(model, x), _out(ref, x))
        assert layer.weight_fake_quantizer.enabled is False  # rounded again, in place


def test_unchanged_inference_weight_stays_rounded(monkeypatch):
    x = torch.randn(4, 128)
    with torch.inference_mode():
        model = _model("int8", monkeypatch, "1")
        before = _out(model, x)
        assert (
            qr.round_frozen_qat_weights(model) == 0
        )  # nothing rewritten: int8 rounding again would move it
        _base(model).load_state_dict({"weight": _base(model).weight.clone()})
        assert torch.equal(_out(model, x), before)
