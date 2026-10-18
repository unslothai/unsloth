# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fused CE reads lm_head.weight directly, so it must apply the same TorchAO QAT fake quantization as
lm_head.forward(): the activation fake quantizer on the hidden states and the weight fake quantizer on
the weight. NVFP4 / MX QAT linears fake-quantize inside forward() and must take the unfused path."""

import inspect

import pytest
import torch

pytest.importorskip("torchao.quantization.qat")

import unsloth  # noqa: F401
import unsloth.models.llama as llama
import unsloth.models.mistral as mistral
from unsloth.models._utils import TorchAOConfig, _prepare_model_for_qat

DIM = 128
VOCAB = 96


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(DIM, VOCAB, bias = False)


def _base_config(scheme):
    from torchao.quantization import (
        Int4WeightOnlyConfig,
        Int8DynamicActivationIntxWeightConfig,
        IntxWeightOnlyConfig,
    )
    from torchao.quantization.granularity import PerAxis, PerGroup

    if scheme == "int8":
        return IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0))
    if scheme == "int4":
        return Int4WeightOnlyConfig(group_size = DIM)
    if scheme == "int8-int4":
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype = torch.int4, weight_granularity = PerGroup(32)
        )
    mx = pytest.importorskip("torchao.prototype.mx_formats")
    if scheme == "nvfp4":
        return mx.NVFP4DynamicActivationNVFP4WeightConfig()
    return mx.MXDynamicActivationMXWeightConfig(
        activation_dtype = torch.float8_e4m3fn, weight_dtype = torch.float8_e4m3fn
    )


def _qat_head(scheme):
    torch.manual_seed(0)
    model = _Head()
    _prepare_model_for_qat(
        model,
        TorchAOConfig(qat_scheme = None, base_config_and_filter_fns = [(_base_config(scheme), None)]),
    )
    return model.lm_head


@pytest.mark.parametrize("scheme", ["int8", "int4", "int8-int4"])
def test_fused_inputs_match_lm_head_forward(scheme):
    lm_head = _qat_head(scheme)
    x = torch.randn(2, 5, DIM)
    hidden_states, weight = llama.fake_quantize_lm_head_inputs(lm_head, x, lm_head.weight)
    expected = lm_head(x)
    # Without the fake quantizers the plain weight gives different logits, so the check can fail.
    assert not torch.equal(torch.nn.functional.linear(x, lm_head.weight), expected)
    torch.testing.assert_close(
        torch.nn.functional.linear(hidden_states, weight), expected, rtol = 0, atol = 0
    )
    assert not llama.lm_head_needs_forward(lm_head)


def test_plain_lm_head_is_untouched():
    lm_head = torch.nn.Linear(DIM, VOCAB, bias = False)
    x = torch.randn(2, 5, DIM)
    hidden_states, weight = llama.fake_quantize_lm_head_inputs(lm_head, x, lm_head.weight)
    assert hidden_states is x and weight is lm_head.weight
    assert not llama.lm_head_needs_forward(lm_head)


@pytest.mark.parametrize("scheme", ["nvfp4", "mx"])
def test_forward_fake_quant_lm_head_takes_unfused_path(scheme):
    lm_head = _qat_head(scheme)
    assert llama.lm_head_needs_forward(lm_head)


@pytest.mark.parametrize(
    "forward",
    [llama.CausalLM_fast_forward, mistral.MistralForCausalLM_fast_forward],
    ids = ["llama", "mistral"],
)
def test_fused_ce_branches_fake_quantize_lm_head(forward):
    source = "".join(inspect.getsource(forward).split())
    assert "fake_quantize_lm_head_inputs(self.lm_head," in source
    assert "lm_head_needs_forward(self.lm_head)" in source
