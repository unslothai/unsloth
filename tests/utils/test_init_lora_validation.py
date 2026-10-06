# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import pytest
import torch

import unsloth  # noqa: F401
from unsloth.models._utils import _has_quantized_linears, validate_init_target_parameters


class _UnslothNVFP4Linear(torch.nn.Linear):
    pass


def _model(*layers):
    return torch.nn.Sequential(torch.nn.Embedding(8, 4), *layers)


def test_dense_model_is_not_quantized():
    assert not _has_quantized_linears(_model(torch.nn.Linear(4, 4)), routed_ok = True)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.float8_e4m3fn])
def test_non_float_linear_is_quantized(dtype):
    layer = torch.nn.Linear(4, 4)
    layer.weight = torch.nn.Parameter(layer.weight.data.to(dtype), requires_grad = False)
    assert _has_quantized_linears(_model(layer), routed_ok = True)


def test_packed_params4bit_in_float_storage_is_quantized():
    layer = torch.nn.Linear(4, 4)
    Params4bit = type("Params4bit", (torch.nn.Parameter,), {})
    layer.weight = Params4bit(layer.weight.data.to(torch.bfloat16), requires_grad = False)
    assert _has_quantized_linears(_model(layer), routed_ok = True)


@pytest.mark.parametrize("packed", ["qweight", "W_q", "weight_packed"])
def test_packed_non_linear_projection_is_quantized(packed):
    # GPTQ / AWQ (qweight), HQQ (W_q), packed MXFP4 / INT4 (weight_packed): no dense .weight.
    layer = torch.nn.Module()
    layer.register_buffer(packed, torch.zeros(4, 1, dtype = torch.int32))
    assert _has_quantized_linears(_model(layer), routed_ok = True)


def test_packed_mxfp4_linear_is_quantized():
    from unsloth.models.mxfp4_compressed_linear import Mxfp4PackedLinear

    layer = Mxfp4PackedLinear(4, 4, bias = False)
    del layer.weight
    layer.register_buffer("weight_packed", torch.zeros(4, 2, dtype = torch.uint8))
    assert _has_quantized_linears(_model(layer), routed_ok = True)


def test_routed_compressed_linears_pass_except_for_mica():
    nvfp4 = _UnslothNVFP4Linear(4, 4)
    nvfp4.register_buffer("weight_packed", torch.zeros(4, 2, dtype = torch.uint8))
    fp8 = torch.nn.Linear(4, 4)
    fp8.weight = torch.nn.Parameter(fp8.weight.data.to(torch.float8_e4m3fn), requires_grad = False)
    fp8._unsloth_compressed_tensors_fp8 = True
    model = _model(nvfp4, fp8)
    assert not _has_quantized_linears(model, routed_ok = True)
    assert _has_quantized_linears(model, routed_ok = False)


@pytest.mark.parametrize(
    "init", ["pissa", "pissa_niter_4", "olora", "orthogonal", "corda", "loftq", "lora_ga", "mica"]
)
def test_weight_reading_inits_reject_moe_expert_parameters(init):
    with pytest.raises(ValueError, match = "target_parameters = \\[\\]"):
        validate_init_target_parameters(init, ["gate_up_proj", "down_proj"])


@pytest.mark.parametrize("init", [True, False, "gaussian", "eva"])
def test_other_inits_allow_moe_expert_parameters(init):
    validate_init_target_parameters(init, ["gate_up_proj"])


def test_no_expert_parameters_allows_pissa():
    validate_init_target_parameters("pissa", None)
    validate_init_target_parameters("pissa", [])
