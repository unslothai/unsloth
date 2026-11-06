# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""An FP8Linear holding a bf16 weight (GLM-5.3-Flash forget gate) must run a plain linear."""

import pytest
import torch
import torch.nn.functional as F

cuda_available = torch.cuda.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
dev = "cuda" if cuda_available else "xpu" if xpu_available else "cpu"

pytestmark = pytest.mark.skipif(not (cuda_available or xpu_available), reason = "needs CUDA or XPU")


@pytest.mark.parametrize(
    "bias", [None, torch.bfloat16, torch.float32], ids = ["no_bias", "bf16_bias", "fp32_bias"]
)
def test_bf16_weight_in_fp8linear_runs_plain_linear(bias):
    from unsloth.kernels import fp8  # installs the patched forward

    FP8Linear = fp8.FP8Linear
    if FP8Linear is None:
        pytest.skip("this transformers has no FP8Linear")
    layer = FP8Linear(256, 128, block_size = (128, 128)).to(dev)
    torch.manual_seed(0)
    layer.weight = torch.nn.Parameter(
        torch.randn(128, 256, device = dev, dtype = torch.bfloat16) * 0.05, requires_grad = False
    )
    layer.bias = (
        torch.nn.Parameter(torch.randn(128, device = dev, dtype = bias)) if bias is not None else None
    )
    x = torch.randn(4, 7, 256, device = dev, dtype = torch.bfloat16)
    out = layer(x)
    assert out.dtype == torch.bfloat16
    ref = F.linear(x, layer.weight, None if layer.bias is None else layer.bias.to(x.dtype))
    torch.testing.assert_close(out, ref, rtol = 0, atol = 0)
