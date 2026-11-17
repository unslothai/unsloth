# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The mHC projection and the router logits are fp32 in the reference (model.py:819, 953-956); an fp16 autocast
region must not run them as fp16 matmuls (the unnormalised mHC projection overflows fp16 before the RMS scale)."""

import types

import pytest

torch = pytest.importorskip("torch")

from unsloth.models.deepseek_v41 import modeling as dsv41  # noqa: E402


def _hc(width = 64, hc = 4):
    config = types.SimpleNamespace(
        hc_mult = hc, hc_sinkhorn_iters = 3, hc_eps = 1e-6, rms_norm_eps = 1e-6, hidden_size = width
    )
    module = dsv41.DeepseekV41HyperConnection(config)
    with torch.no_grad():
        module.fn.fill_(1.0)
    return module


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hyper_connection_mixes_are_fp32_under_autocast(dtype):
    module = _hc()
    # flattened width 256, entries 300: the unnormalised projection is 76800 (> fp16 max), normalised 256.
    x = torch.full((1, 2, 4, 64), 300.0, requires_grad = True)
    reference = module(x)
    with torch.autocast("cpu", dtype = dtype):
        mixed = module(x)
    for got, want in zip(mixed, reference):
        assert got.dtype == torch.float32
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, rtol = 0, atol = 0)
    sum(t.sum() for t in mixed).backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(module.fn.grad).all()


def test_router_logits_are_fp32_under_autocast():
    config = types.SimpleNamespace(
        scoring_func = "sqrtsoftplus",
        gate_temperature = 1.0,
        norm_topk_prob = True,
        routed_scaling_factor = 1.0,
        hidden_size = 64,
    )
    router = dsv41.DeepseekV41TopKRouter(config, n_experts = 8, top_k = 2)
    torch.manual_seed(0)
    with torch.no_grad():
        router.weight.normal_()
    x = torch.randn(16, 64)
    weights, indices = router(x)
    with torch.autocast("cpu", dtype = torch.float16):
        weights16, indices16 = router(x)
    assert torch.equal(indices16, indices)
    torch.testing.assert_close(weights16, weights, rtol = 0, atol = 0)
