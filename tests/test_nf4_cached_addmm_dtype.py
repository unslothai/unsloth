# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The cached NF4 in-place addmm casts the dense weight to out.dtype, as the cached matmul casts to X.dtype."""

import pytest
import torch

import unsloth  # noqa: F401
from unsloth.kernels import dequant_cache as dq


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
def test_addmm_impl_casts_dense_weight(monkeypatch, weight_dtype):
    M, K, N = 3, 8, 5
    dense = torch.randn(N, K, generator = torch.Generator().manual_seed(0))
    monkeypatch.setattr(dq, "_dense_from_parts", lambda *args, **kwargs: dense.to(weight_dtype))
    X = torch.randn(M, K, dtype = torch.bfloat16)
    out = torch.randn(M, N, dtype = torch.bfloat16)
    expected = out.float() + X.float() @ dense.to(weight_dtype).to(torch.bfloat16).float().t()
    # Positional parts after X: W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype.
    dq._addmm_impl(out, X, None, None, None, None, None, None, 64, 256, [N, K], weight_dtype)
    torch.testing.assert_close(out.float(), expected, atol = 2e-2, rtol = 2e-2)
