# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Decode-sized block FP8: act_quant fused into a narrow-N w8a8 GEMM (fp8_block_small_m_linear)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)
if torch.cuda.get_device_capability() < (8, 9):
    pytest.skip("fp8e4nv Triton kernels need sm89+", allow_module_level = True)

import unsloth  # noqa: F401
import unsloth.kernels.fp8 as f

pytestmark = pytest.mark.skipif(
    f.fp8_block_matmul is not f.w8a8_block_fp8_matmul_triton,
    reason = "only the Triton block matmul route is replaced",
)


@pytest.fixture(autouse = True)
def _no_block_gemv(monkeypatch):
    # The GEMV, which takes some decode shapes first, is tested in test_fp8_block_gemv.py.
    monkeypatch.setattr(f, "_FP8_BLOCK_GEMV", False, raising = False)


def _block_weight(
    N,
    K,
    scale_dtype = torch.float32,
):
    W = torch.randn(N, K, device = "cuda") * 0.02
    s = W.abs().view(N // 128, 128, K // 128, 128).amax((1, 3)) / 448.0
    q = (
        (W.view(N // 128, 128, K // 128, 128) / s[:, None, :, None])
        .view(N, K)
        .to(torch.float8_e4m3fn)
    )
    return q, s.contiguous().to(scale_dtype)


def _main_path(X, q, s, monkeypatch):
    monkeypatch.setattr(f, "_FP8_BLOCK_SMALL_M", False)
    out = f.fp8_linear(X, q, s)
    monkeypatch.setattr(f, "_FP8_BLOCK_SMALL_M", True)
    return out


def _oracle(X, q, s):
    M, K = X.numel() // X.shape[-1], X.shape[-1]
    N = q.shape[0]
    xq, xs = f.act_quant(X, 128)
    x64 = (xq.double().view(M, K // 128, 128) * xs.double().view(M, K // 128, 1)).view(M, K)
    w64 = (q.double().view(N // 128, 128, K // 128, 128) * s.double()[:, None, :, None]).view(N, K)
    return (x64 @ w64.t()).view(*X.shape[:-1], N)


def _oracle_abs(X, q, s):
    M, K = X.numel() // X.shape[-1], X.shape[-1]
    N = q.shape[0]
    xq, xs = f.act_quant(X, 128)
    x64 = (xq.double().view(M, K // 128, 128) * xs.double().view(M, K // 128, 1)).view(M, K)
    w64 = (q.double().view(N // 128, 128, K // 128, 128) * s.double()[:, None, :, None]).view(N, K)
    return (x64.abs() @ w64.abs().t()).view(*X.shape[:-1], N)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N,K", [(4096, 4096), (1024, 4096), (6144, 2048), (2048, 6144)])
@pytest.mark.parametrize("M", [1, 3, 8, 16])
def test_small_m_matches_main(monkeypatch, dtype, scale_dtype, N, K, M):
    torch.manual_seed(0)
    q, s = _block_weight(N, K, scale_dtype)
    X = torch.randn(1, M, K, device = "cuda", dtype = dtype)
    X[0, 0, :128] = 0  # an all-zero block keeps scale 1, as act_quant does
    assert f.can_use_fp8_block_small_m(X, q, s, [128, 128])
    main = _main_path(X, q, s, monkeypatch)
    calls = []
    orig = f.fp8_block_small_m_linear
    monkeypatch.setattr(f, "fp8_block_small_m_linear", lambda *a: calls.append(1) or orig(*a))
    new = f.fp8_linear(X, q, s)
    assert calls == [1]
    ref = _oracle(X, q, s)
    split_k = f._small_m_config(M, N, K, X.device)[1]
    if split_k == 1:
        # Same quantized operands and the same accumulation order as main.
        assert torch.equal(new, main)
    else:
        # Only the fp32 sum order over K splits differs: bound is one output rounding plus fp32 sum error.
        mag = torch.maximum(new.double().abs(), main.double().abs())
        k_blocks = K // 128
        bound = torch.finfo(dtype).eps * mag + 2 * k_blocks * torch.finfo(
            torch.float32
        ).eps * _oracle_abs(X, q, s)
        assert ((new.double() - main.double()).abs() <= bound + 1e-30).all()
        err_new = (new.double() - ref).abs().mean()
        err_main = (main.double() - ref).abs().mean()
        assert err_new <= err_main * 1.05
    assert torch.isfinite(new).all()


def test_split_k_is_deterministic(monkeypatch):
    torch.manual_seed(0)
    q, s = _block_weight(1024, 4096)
    X = torch.randn(1, 4, 4096, device = "cuda", dtype = torch.bfloat16)
    monkeypatch.setattr(f, "_small_m_config", lambda *a: (32, 8, 4, 3))
    first = f.fp8_block_small_m_linear(X, q, s)
    for _ in range(20):
        assert torch.equal(f.fp8_block_small_m_linear(X, q, s), first)


def test_larger_m_and_kill_switch_keep_main_path(monkeypatch):
    q, s = _block_weight(4096, 4096)
    for M in (17, 64):
        X = torch.randn(1, M, 4096, device = "cuda", dtype = torch.bfloat16)
        assert not f.can_use_fp8_block_small_m(X, q, s, [128, 128])
    X = torch.randn(1, 1, 4096, device = "cuda", dtype = torch.bfloat16)
    monkeypatch.setattr(f, "_FP8_BLOCK_SMALL_M", False)
    assert not f.can_use_fp8_block_small_m(X, q, s, [128, 128])


def test_unsupported_layouts_fall_back():
    q, s = _block_weight(4096, 4096)
    X = torch.randn(1, 1, 4096, device = "cuda", dtype = torch.bfloat16)
    assert not f.can_use_fp8_block_small_m(X.float(), q, s, [128, 128])
    assert not f.can_use_fp8_block_small_m(X, q, s, [64, 128])
    assert not f.can_use_fp8_block_small_m(X, q, s[:, :-1], [128, 128])
    assert not f.can_use_fp8_block_small_m(X, q.to(torch.float8_e5m2), s, [128, 128])


def test_pre_sm89_keeps_dequant_route(monkeypatch):
    # Below sm89 FP8BlockQuantLinear returns the dequant + matmul result before the small-M branch.
    q, s = _block_weight(1024, 4096)
    X = torch.randn(1, 2, 4096, device = "cuda", dtype = torch.bfloat16)
    monkeypatch.setattr(f, "_fp8_device_lacks_kernel", lambda device: True)
    monkeypatch.setattr(
        f, "fp8_block_small_m_linear", lambda *a: pytest.fail("small-M kernel used below sm89")
    )
    out = f.fp8_linear(X, q, s)
    ref = X @ f._blockwise_weight_dequant_any_shape(q, s, [128, 128], X.dtype).t()
    assert torch.equal(out, ref)


def test_grad_path_still_uses_autograd_function():
    q, s = _block_weight(1024, 4096)
    X = torch.randn(1, 2, 4096, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    out = f.fp8_linear(X, q, s)
    assert out.grad_fn is not None
    out.float().sum().backward()
    ref = f._blockwise_weight_dequant_any_shape(q, s, [128, 128], X.dtype).float().sum(0)
    torch.testing.assert_close(X.grad[0, 0].float(), ref, rtol = 1e-2, atol = 1e-2)


def test_compiled_path_adds_no_graph_or_break():
    # Compiled regions keep main's route: no break, and no recompile when M crosses the small-M bound.
    from torch._dynamo.utils import counters

    q, s = _block_weight(1024, 4096)
    torch._dynamo.reset()
    counters.clear()
    fn = torch.compile(lambda x: f.fp8_linear(x, q, s), dynamic = True)
    with torch.no_grad():
        for M in (1, 2, 16, 17, 64, 1):
            X = torch.randn(1, M, 4096, device = "cuda", dtype = torch.bfloat16)
            torch.testing.assert_close(fn(X), _main_path_eager(X, q, s), rtol = 0, atol = 0)
    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] <= 2


def _main_path_eager(X, q, s):
    old = f._FP8_BLOCK_SMALL_M
    f._FP8_BLOCK_SMALL_M = False
    try:
        return f.fp8_linear(X, q, s)
    finally:
        f._FP8_BLOCK_SMALL_M = old
