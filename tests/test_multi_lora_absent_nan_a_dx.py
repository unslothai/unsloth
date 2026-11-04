# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Reference path: an absent adapter's non-finite lora_A must not reach dX, or every earlier layer's
present adapters get NaN gradients (isolation across layers)."""

import pytest
import torch
import unsloth  # noqa: F401
from unsloth.kernels import multi_lora as ml


def _two_layer(poison, adapter = 2):
    torch.manual_seed(0)
    m = torch.nn.Sequential(
        torch.nn.Linear(16, 24), torch.nn.GELU(), torch.nn.Linear(24, 8)
    ).double()
    ml.inject_multi_lora(m, ["0", "2"], 3, r = 4)
    for _, layer in ml._layers(m):
        layer.double()
        for B in layer.lora_B:
            torch.nn.init.normal_(B, std = 0.1)
    if poison:
        with torch.no_grad():
            m[2].lora_A[adapter][0, 0] = float("nan")
    return m


@pytest.mark.parametrize("negative_control", [False, True])
def test_absent_nan_lora_a_does_not_reach_earlier_layers(negative_control):
    ml.set_multi_lora_triton(False)
    try:
        grads = {}
        for poison in (False, True):
            # Negative control: poison present adapter 0 so the check can fail.
            m = _two_layer(poison, adapter = 0 if negative_control else 2)
            x = torch.randn(40, 16, dtype = torch.float64, generator = torch.Generator().manual_seed(1))
            ids = torch.randint(0, 2, (40,), generator = torch.Generator().manual_seed(2))
            ml.set_multi_lora_batch(m, ids)
            m(x).pow(2).sum().backward()
            grads[poison] = [p.grad for p in m[0].lora_A[:2]] + [p.grad for p in m[0].lora_B[:2]]
        finite = all(torch.isfinite(g).all() for g in grads[True])
        if negative_control:
            assert not finite
        else:
            assert finite
            for a, b in zip(grads[False], grads[True]):
                torch.testing.assert_close(a, b, rtol = 0, atol = 0)
    finally:
        ml.set_multi_lora_triton(True)
