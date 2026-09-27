# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""An FP8Linear whose checkpoint weight stayed bf16 must run a plain linear.

GLM-5.3-Flash's FP8 checkpoint keeps the linear-attention forget gate (f_a_proj, f_b_proj)
in bf16 and lists them in modules_to_not_convert under their checkpoint names. transformers
renames them to forget_gate.f_a_proj, the exclusion no longer matches, so they are built as
FP8Linear holding a bf16 weight. transformers' own FP8Linear.forward runs F.linear for that;
Unsloth's replacement forward handed the bf16 weight to the fp8 block kernel, which Triton
rejects ("Unsupported lhs dtype fp8e4nv") on the first forward."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


@pytest.mark.parametrize("bias", [None, torch.bfloat16, torch.float32], ids = ["no_bias", "bf16_bias", "fp32_bias"])
def test_bf16_weight_in_fp8linear_runs_plain_linear(bias):
    from unsloth.kernels import fp8  # installs the patched forward
    FP8Linear = fp8.FP8Linear
    if FP8Linear is None:
        pytest.skip("this transformers has no FP8Linear")
    layer = FP8Linear(256, 128).cuda()
    torch.manual_seed(0)
    layer.weight = torch.nn.Parameter(torch.randn(128, 256, device = "cuda", dtype = torch.bfloat16) * 0.05,
                                      requires_grad = False)
    # An fp32 bias (fbgemm keeps it that way) must not change the activation dtype.
    layer.bias = torch.nn.Parameter(torch.randn(128, device = "cuda", dtype = bias)) if bias is not None else None
    x = torch.randn(4, 7, 256, device = "cuda", dtype = torch.bfloat16)
    out = layer(x)
    assert out.dtype == torch.bfloat16
    ref = F.linear(x, layer.weight, None if layer.bias is None else layer.bias.to(x.dtype))
    torch.testing.assert_close(out, ref, rtol = 0, atol = 0)
