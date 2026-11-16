# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import pytest
import torch
from torch import nn
import unsloth  # noqa: F401

from unsloth.kernels import multi_lora as ml


def _model(proj: nn.Module) -> nn.Module:
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.proj = proj
    model.layer.dense = nn.Linear(16, 12, bias = False)
    return model


def _assert_refused_untouched(model: nn.Module, proj: nn.Module) -> None:
    with pytest.raises(NotImplementedError, match = r"layer\.proj.*4-bit or in 16-bit"):
        ml.inject_multi_lora(model, ["proj", "dense"], num_adapters = 2, r = 2)
    assert model.layer.proj is proj
    assert isinstance(model.layer.dense, nn.Linear)
    assert model.layer.dense.weight.requires_grad


def test_bitsandbytes_linear8bit_base_is_refused():
    bnb = pytest.importorskip("bitsandbytes")
    proj = bnb.nn.Linear8bitLt(16, 12, bias = False, has_fp16_weights = False)
    _assert_refused_untouched(_model(proj), proj)
    with pytest.raises(NotImplementedError, match = "Linear8bitLt"):
        ml.MultiLoraLinear(proj, 2, r = 2)


def test_a_weight_carrying_int8_scales_is_refused():
    proj = nn.Linear(16, 12, bias = False)
    proj.weight.SCB = torch.ones(12)
    _assert_refused_untouched(_model(proj), proj)


def test_a_dense_base_is_still_wrapped():
    model = _model(nn.Linear(16, 12, bias = False))
    assert ml.inject_multi_lora(model, ["proj", "dense"], num_adapters = 2, r = 2) == [
        "layer.proj",
        "layer.dense",
    ]
    assert isinstance(model.layer.proj, ml.MultiLoraLinear)
