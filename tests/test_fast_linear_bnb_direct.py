# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""fast_linear_forward's bitsandbytes fused-GEMM branch goes through unsloth_zoo's direct call when it
applies, and through bitsandbytes.matmul_4bit otherwise or with an unsloth_zoo that lacks it."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from real_accelerator import has_real_cuda

from unsloth.kernels import utils


def _fake_bnb(calls):
    def matmul_4bit(
        X,
        Wt,
        quant_state = None,
        out = None,
    ):
        calls.append(("bnb", Wt, quant_state, out))
        return "bnb_result"

    return SimpleNamespace(matmul_4bit = matmul_4bit)


def test_direct_result_is_used(monkeypatch):
    calls = []
    monkeypatch.setattr(utils, "bnb", _fake_bnb(calls))
    seen = []
    monkeypatch.setattr(
        utils,
        "_bnb_matmul_4bit_direct",
        lambda X, W, qs, out = None: seen.append((X, W, qs, out)) or "direct_result",
    )
    X, W, qs, out = (
        torch.zeros(2, 1, 4),
        torch.zeros(4, 1, dtype = torch.uint8),
        object(),
        torch.zeros(2, 1, 8),
    )
    assert utils._bnb_matmul_4bit(X, W, qs, out) == "direct_result"
    assert seen == [(X, W, qs, out)] and not calls


def test_falls_back_to_bitsandbytes(monkeypatch):
    calls = []
    monkeypatch.setattr(utils, "bnb", _fake_bnb(calls))
    X, W, qs = torch.zeros(2, 1, 4), torch.zeros(4, 1, dtype = torch.uint8), object()
    # The direct call declined.
    monkeypatch.setattr(utils, "_bnb_matmul_4bit_direct", lambda *a, **k: None)
    assert utils._bnb_matmul_4bit(X, W, qs, None) == "bnb_result"
    # An unsloth_zoo without it.
    monkeypatch.setattr(utils, "_bnb_matmul_4bit_direct", None)
    assert utils._bnb_matmul_4bit(X, W, qs, None) == "bnb_result"
    assert len(calls) == 2 and all(c[2] is qs for c in calls)
    assert torch.equal(calls[0][1], W.t())


def test_fast_linear_forward_routes_its_bnb_branch_through_it():
    src = inspect.getsource(utils.fast_linear_forward)
    assert "_bnb_matmul_4bit(X, W, W_quant, out)" in src and "bnb.matmul_4bit(" not in src


@pytest.mark.skipif(not has_real_cuda() or torch.version.hip is not None, reason = "NVIDIA CUDA")
@pytest.mark.parametrize("bsz", [5, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_bnb_branch_bitwise_and_direct(bsz, dtype, monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    if utils._bnb_matmul_4bit_direct is None:
        pytest.skip("unsloth_zoo without bnb_matmul_4bit_direct")
    from unsloth_zoo.temporary_patches import bitsandbytes as Z

    if Z._bnb_direct_ops() is False:
        pytest.skip("bitsandbytes without the fused 4-bit GEMM")
    torch.manual_seed(0)
    proj = bnb.nn.Linear4bit(4096, 1024, bias = False, compute_dtype = dtype, quant_type = "nf4").cuda()
    X = torch.randn(bsz, 1, 4096, dtype = dtype, device = "cuda")
    monkeypatch.setattr(utils, "_gemv_rows_cap", lambda device: 4)
    monkeypatch.setattr(utils, "_BNB_GEMM_4BIT", True)
    monkeypatch.setattr(utils, "_BNB_GEMM_4BIT_DTYPES", (torch.float16, torch.bfloat16))
    monkeypatch.setattr(utils, "_bnb_gemm_4bit_device_ok", lambda device: True)
    from bitsandbytes.backends.cuda import ops

    # bitsandbytes' fused-vs-dequantize choice, forced for both the reference and the branch, so the
    # direct call has to engage on every GPU.
    monkeypatch.setattr(ops, "_gemm_4bit_use_custom_fn", lambda *a: True)
    with torch.no_grad():
        ref = bnb.matmul_4bit(X, proj.weight.t(), quant_state = proj.weight.quant_state)
        before = Z.BNB_4BIT_DIRECT_CALLS["direct"]
        out = torch.empty(bsz, 1, 1024, dtype = dtype, device = "cuda")
        for _ in range(3):
            assert torch.equal(utils.fast_linear_forward(proj, X), ref)
            assert utils.fast_linear_forward(proj, X, out = out) is out and torch.equal(out, ref)
    assert Z.BNB_4BIT_DIRECT_CALLS["direct"] - before == 6
