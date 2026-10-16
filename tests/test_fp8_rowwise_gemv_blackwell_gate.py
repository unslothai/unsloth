# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Triton 3.3 builds a wrong FP8 rowwise GEMV on Blackwell, so those cards must not take it."""

import os
import sys
from pathlib import Path

import pytest
from real_accelerator import has_real_cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


@pytest.fixture(scope = "module")
def F():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import fp8
    return fp8


@pytest.mark.parametrize(
    "capability, version, hip, expected",
    [
        ((10, 0), "3.3.1", None, True),
        ((12, 0), "3.3.0", None, True),
        ((10, 0), "3.4.0", None, False),
        ((9, 0), "3.3.1", None, False),
        ((8, 9), "3.2.0", None, False),
        ((10, 0), "3.3.1", "6.3", False),
    ],
)
def test_gate(F, monkeypatch, capability, version, hip, expected):
    monkeypatch.setattr(F.torch.cuda, "get_device_capability", lambda device = None: capability)
    monkeypatch.setattr(F.triton, "__version__", version)
    monkeypatch.setattr(F.torch.version, "hip", hip)
    F._fp8_gemv_miscompiles.cache_clear()
    try:
        assert F._fp8_gemv_miscompiles(0) is expected
    finally:
        F._fp8_gemv_miscompiles.cache_clear()


@pytest.mark.parametrize("version, expected", [("3.2.0", False), ("3.3.1", True), ("3.4.0", True)])
def test_block_n_1_exception_needs_a_triton_that_builds_blackwell(
    F, monkeypatch, version, expected
):
    monkeypatch.setattr(F.triton, "__version__", version)
    F._triton_builds_blackwell.cache_clear()
    try:
        assert F._triton_builds_blackwell() is expected
    finally:
        F._triton_builds_blackwell.cache_clear()


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a GPU")
@pytest.mark.parametrize("N, K", [(5000, 1536), (40000, 1024), (4096, 4096), (4095, 4096), (33, 7)])
def test_chosen_gemv_is_exact(F, N, K):
    if F._fp8_kernel_unsupported(torch.empty(1, device = "cuda", dtype = torch.float8_e4m3fn)):
        pytest.skip("no fp8e4nv Triton kernel on this device")
    g = torch.Generator(device = "cuda").manual_seed(0)
    X = torch.randn(1, K, device = "cuda", dtype = torch.bfloat16, generator = g)
    W = (torch.randn(N, K, device = "cuda", generator = g) * 0.5).to(torch.float8_e4m3fn)
    S = torch.rand(N, 1, device = "cuda", generator = g) + 0.5
    if not F.can_use_fp8_rowwise_gemv(X, W, S):
        assert F._fp8_gemv_miscompiles(W.device) and N > 4096
        return
    out = F.fp8_rowwise_gemv(X, W, S).float().reshape(-1)
    ref = (X.float() @ (W.float() * S).T).reshape(-1)
    assert (out - ref).abs().max() <= 1e-2 * ref.abs().max()
