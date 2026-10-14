# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""float32 base weights skip the fused LoRA kernels, and fp32 NF4 dequantizes with the fp32 kernel."""

from __future__ import annotations

import types

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda
from unsloth.models.llama import _base_weight_dtype, _fused_lora_skip_reason


def _proj(weight):
    return types.SimpleNamespace(base_layer = types.SimpleNamespace(weight = weight))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_base_weight_dtype_plain(dtype):
    assert _base_weight_dtype(_proj(torch.zeros(2, 2, dtype = dtype))) == dtype


def test_base_weight_dtype_reads_quant_state():
    weight = torch.zeros(2, 2, dtype = torch.uint8)
    weight.quant_state = types.SimpleNamespace(dtype = torch.float32)
    assert _base_weight_dtype(_proj(weight)) == torch.float32


def test_float32_skip_reason():
    assert _fused_lora_skip_reason(0, "none") == ""
    assert "float32" in _fused_lora_skip_reason(0, "none", float32_base = True)


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "needs a real CUDA device")
def test_fp32_nf4_dequant_matches_bitsandbytes():
    import bitsandbytes as bnb
    from unsloth.kernels.utils import fast_dequantize

    W = torch.randn(256, 128) * 0.02
    p = bnb.nn.Params4bit(W, requires_grad = False, quant_type = "nf4").cuda()
    assert p.quant_state.dtype == torch.float32
    ref = bnb.functional.dequantize_4bit(p.data, p.quant_state)
    torch.testing.assert_close(fast_dequantize(p, p.quant_state), ref, rtol = 0, atol = 0)


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "needs a real CUDA device")
def test_fp32_nf4_single_token_decode():
    import bitsandbytes as bnb
    from unsloth.kernels.utils import fast_linear_forward

    layer = bnb.nn.Linear4bit(128, 256, bias = False, compute_dtype = torch.float32, quant_type = "nf4")
    layer.weight = bnb.nn.Params4bit(
        torch.randn(256, 128) * 0.02, requires_grad = False, quant_type = "nf4"
    )
    layer = layer.cuda()
    X = torch.randn(1, 1, 128, device = "cuda")
    ref = X @ bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state).t()
    torch.testing.assert_close(fast_linear_forward(layer, X), ref, rtol = 1e-4, atol = 1e-4)
