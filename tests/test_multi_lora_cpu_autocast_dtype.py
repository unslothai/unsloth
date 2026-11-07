# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import pytest
import torch
from torch import nn
import unsloth  # noqa: F401

from unsloth.kernels import multi_lora as ml


@pytest.mark.parametrize("bias", [False, True])
def test_cpu_autocast_output_dtype_matches_the_base_layer(bias):
    torch.manual_seed(3)
    lin = nn.Linear(8, 4, bias = bias)
    wrapped = ml.MultiLoraLinear(lin, 2, r = 2)
    with torch.no_grad():
        for B in wrapped.lora_B:
            B.normal_()
    ids = torch.tensor([0, 1, -1])
    ml.set_multi_lora_batch(wrapped, ids)
    x = torch.randn(3, 8)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        base_out = lin(x)
        out = wrapped(x)
    assert base_out.dtype == torch.bfloat16
    assert out.dtype == base_out.dtype
    A, B = wrapped.stacked()
    delta = ml.multi_lora_delta(x, A, B, wrapped.scales, ml.MultiLoraBatch.from_adapter_ids(ids, 2))
    torch.testing.assert_close(out.float(), base_out.float() + delta, rtol = 2e-2, atol = 2e-2)
    out.float().sum().backward()
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in wrapped.lora_B)
