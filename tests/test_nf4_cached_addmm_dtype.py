# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""NF4 base-weight gradient sites cast the dense weight to the gradient's dtype: the cached in-place addmm (as the
cached matmul casts to X.dtype) and the eager accumulate / out= matmul, which autocast skips as well."""

import pytest
import torch

import unsloth  # noqa: F401
from unsloth.kernels import dequant_cache as dq

M, K, N = 3, 8, 5


def _seeded(*shape, dtype, seed):
    return torch.randn(*shape, generator = torch.Generator().manual_seed(seed)).to(dtype)


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_addmm_impl_casts_dense_weight(monkeypatch, weight_dtype):
    dense = _seeded(N, K, dtype = weight_dtype, seed = 0)
    dense_before = dense.clone()
    monkeypatch.setattr(dq, "_dense_from_parts", lambda *args, **kwargs: dense)
    X = _seeded(M, K, dtype = torch.bfloat16, seed = 1)
    out = _seeded(M, N, dtype = torch.bfloat16, seed = 2)
    expected = out.clone().addmm_(X, dense.to(torch.bfloat16).t())
    # Positional parts after X: W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype.
    dq._addmm_impl(out, X, None, None, None, None, None, None, 64, 256, [N, K], weight_dtype)
    torch.testing.assert_close(out, expected, rtol = 0, atol = 0)
    # The cached dense weight is read, never written.
    assert torch.equal(dense, dense_before)


def _nf4_state(dtype):
    bnb = pytest.importorskip("bitsandbytes.functional")
    return bnb.QuantState(absmax = torch.ones(1), quant_type = "nf4", dtype = dtype)


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eager_nf4_dgrad_casts_dense_weight(monkeypatch, weight_dtype):
    # Cache off: the eager decode returns the quant state's dtype, whatever dtype = dY.dtype asks for.
    from unsloth.kernels import fast_lora, utils

    quant_state = _nf4_state(weight_dtype)
    packed = torch.empty((N * K // 2, 1), dtype = torch.uint8)
    decoded = _seeded(K, N, dtype = weight_dtype, seed = 0)  # fast_dequantize(W.t()): [in, out]
    monkeypatch.setattr(dq, "_MODE", None)
    monkeypatch.setattr(utils, "fast_dequantize", lambda *args, **kwargs: decoded)
    dY = _seeded(M, N, dtype = torch.bfloat16, seed = 1)
    weight = decoded.to(dY.dtype)

    dX = _seeded(M, K, dtype = dY.dtype, seed = 2)
    expected = dX.clone().addmm_(dY, weight.t())
    with torch.autocast("cpu", dtype = dY.dtype):
        result = fast_lora._base_dgrad_accum(dX, dY, packed, quant_state)
    assert result is dX
    torch.testing.assert_close(dX, expected, rtol = 0, atol = 0)

    out = torch.empty((M, K), dtype = dY.dtype)
    with torch.autocast("cpu", dtype = dY.dtype):
        result = fast_lora._base_dgrad(dY, packed, quant_state, out = out)
    assert result is out
    torch.testing.assert_close(out, dY @ weight.t(), rtol = 0, atol = 0)
