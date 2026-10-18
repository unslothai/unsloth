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


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a GPU")
@pytest.mark.parametrize(
    "N, K", [(5000, 1536), (40000, 1024), (4096, 4096), (256, 16384), (2, 4096)]
)
def test_chosen_gemv_is_exact(F, N, K):
    if F._fp8_kernel_unsupported(torch.empty(1, device = "cuda", dtype = torch.float8_e4m3fn)):
        pytest.skip("no fp8e4nv Triton kernel on this device")
    g = torch.Generator(device = "cuda").manual_seed(0)
    X = torch.randn(1, K, device = "cuda", dtype = torch.bfloat16, generator = g)
    W = (torch.randn(N, K, device = "cuda", generator = g) * 0.5).to(torch.float8_e4m3fn)
    S = torch.rand(N, 1, device = "cuda", generator = g) + 0.5
    if not F.can_use_fp8_rowwise_gemv(X, W, S):
        assert F._fp8_gemv_miscompiles(W.device)
        return
    out = F.fp8_rowwise_gemv(X, W, S).float().reshape(-1)
    ref = (X.float() @ (W.float() * S).T).reshape(-1)
    assert (out - ref).abs().max() <= 1e-2 * ref.abs().max()


@pytest.mark.parametrize(
    "version, release",
    [("3.3.1", (3, 3)), ("3.4.0rc1", (3, 4)), ("3.10.0", (3, 10)), ("3.3.1+git9a8b7c6", (3, 3))],
)
def test_triton_release_parse(F, monkeypatch, version, release):
    monkeypatch.setattr(F.triton, "__version__", version)
    assert F._triton_release() == release


@pytest.mark.parametrize("version", ["3.3.1", "3.4.0"])
def test_gate_traces_without_graph_break(F, monkeypatch, version):
    # Compiled decode steps call the gate; torch 2.7 dynamo broke inside packaging's Version.
    monkeypatch.setattr(F.torch.cuda, "get_device_capability", lambda device = None: (10, 0))
    monkeypatch.setattr(F.triton, "__version__", version)
    monkeypatch.setattr(F.torch.version, "hip", None)

    def fn(x):
        return x + int(F._fp8_gemv_miscompiles(0))

    F._fp8_gemv_miscompiles.cache_clear()
    torch._dynamo.reset()
    try:
        out = torch.compile(fn, backend = "eager", fullgraph = True)(torch.zeros(2))
        assert out.tolist() == [float(version.startswith("3.3"))] * 2
    finally:
        torch._dynamo.reset()
        F._fp8_gemv_miscompiles.cache_clear()


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a GPU")
@pytest.mark.parametrize(
    "capability, version, N, K, expected",
    [
        # Blackwell + Triton 3.3: BLOCK_N 1 only, and not few rows over a long K.
        ((10, 0), "3.3.1", 256, 65536, True),
        ((12, 0), "3.3.1", 4096, 8192, True),
        ((10, 0), "3.3.1", 255, 4096, True),
        ((10, 0), "3.3.1", 2, 4096, True),
        ((10, 0), "3.3.1", 255, 4097, False),
        ((12, 0), "3.3.1", 2, 65536, False),
        ((10, 0), "3.3.1", 4097, 1024, False),
        ((10, 0), "3.2.0", 256, 4096, False),
        ((10, 0), "3.4.0", 4097, 8192, True),
        ((10, 0), "3.4.0", 2, 65536, True),
        ((9, 0), "3.3.1", 2, 65536, True),
    ],
)
def test_gate_boundary(F, monkeypatch, capability, version, N, K, expected):
    monkeypatch.setattr(F.torch.cuda, "get_device_capability", lambda device = None: capability)
    monkeypatch.setattr(F.triton, "__version__", version)
    monkeypatch.setattr(F.torch.version, "hip", None)
    caches = (F._fp8_gemv_miscompiles, F._triton_builds_blackwell, F._fp8_device_lacks_kernel)
    for cache in caches:
        cache.cache_clear()
    try:
        X = torch.zeros(1, K, device = "cuda", dtype = torch.bfloat16)
        W = torch.zeros(N, K, device = "cuda", dtype = torch.float8_e4m3fn)
        S = torch.ones(N, 1, device = "cuda")
        with torch.no_grad():
            assert F.can_use_fp8_rowwise_gemv(X, W, S) is expected
    finally:
        for cache in caches:
            cache.cache_clear()


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a GPU")
@pytest.mark.parametrize("N, K", [(1024, 512), (2, 65536)])
def test_gemv_route_compiles_fullgraph(F, N, K):
    if F._fp8_kernel_unsupported(torch.empty(1, device = "cuda", dtype = torch.float8_e4m3fn)):
        pytest.skip("no fp8e4nv Triton kernel on this device")
    X = torch.randn(1, 1, K, device = "cuda", dtype = torch.bfloat16)
    W = (torch.randn(N, K, device = "cuda") * 0.1).to(torch.float8_e4m3fn)
    S = torch.rand(N, 1, device = "cuda") * 0.01 + 0.001

    def fn(X):
        return X.sum() * 2 if F.can_use_fp8_rowwise_gemv(X, W, S) else X.sum() * 3

    F._fp8_gemv_miscompiles.cache_clear()
    torch._dynamo.reset()
    try:
        with torch.no_grad():
            assert torch.equal(torch.compile(fn, backend = "eager", fullgraph = True)(X), fn(X))
    finally:
        torch._dynamo.reset()
        F._fp8_gemv_miscompiles.cache_clear()


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_a_trainable_scale_keeps_the_differentiable_path(F):
    # The GEMV output has no autograd history, so any grad-requiring operand must take fp8_linear.
    X = torch.randn(1, 1, 512, device = "cuda", dtype = torch.bfloat16)
    W = (torch.randn(256, 512, device = "cuda") * 0.1).to(torch.float8_e4m3fn)
    S = (torch.rand(256, 1, device = "cuda") * 0.01 + 0.001).requires_grad_()
    assert not F.can_use_fp8_rowwise_gemv(X, W, S)
    with torch.no_grad():
        assert F.can_use_fp8_rowwise_gemv(X, W, S) == F.can_use_fp8_rowwise_gemv(X, W, S.detach())
