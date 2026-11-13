# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)
pytest.importorskip("triton")

import triton  # noqa: E402
import unsloth  # noqa: E402,F401
from unsloth.kernels import fp8 as F8  # noqa: E402

if F8._fp8_device_lacks_kernel(torch.device("cuda")):
    pytest.skip("e4m3 Triton kernels need sm89+", allow_module_level = True)


@pytest.fixture(autouse = True)
def _gemv_only(monkeypatch):
    monkeypatch.setattr(F8, "_FP8_BLOCK_SMALL_M", False)


def _quantize(
    N,
    K,
    seed = 0,
):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    W = torch.randn(N, K, device = "cuda", generator = g)
    nb, kb = triton.cdiv(N, 128), triton.cdiv(K, 128)
    Wp = torch.zeros(nb * 128, kb * 128, device = "cuda")
    Wp[:N, :K] = W
    blocks = Wp.view(nb, 128, kb, 128)
    s = blocks.abs().amax(dim = (1, 3)).clamp_min(1e-12) / 448.0
    q = (blocks / s[:, None, :, None]).view(nb * 128, kb * 128)[:N, :K].to(torch.float8_e4m3fn)
    return q.contiguous(), s.float().contiguous()


def _uncached(q):
    # Drop cached eligibility: gate tests flip device gates on the same weight.
    q.__dict__.pop("_unsloth_fp8_block_gemv", None)
    return q


def _oracle(X, q, s):
    S = s.double().repeat_interleave(128, 0).repeat_interleave(128, 1)[: q.shape[0], : q.shape[1]]
    return X.double().reshape(-1, q.shape[1]) @ (q.double() * S).T


def _rel_err(y, ref):
    return ((y.double().reshape(ref.shape) - ref).abs().mean() / ref.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", [1, 3, 16])
@pytest.mark.parametrize("NK", [(2048, 4096), (1024, 4096), (500, 1152), (320, 1000)])
def test_block_gemv_matches_oracle_and_beats_act_quant(dtype, M, NK):
    N, K = NK
    q, s = _quantize(N, K, seed = M)
    X = torch.randn(M, 1, K, device = "cuda", dtype = torch.float32).to(dtype)
    assert M > 2 or F8.can_use_fp8_block_gemv(X, q, s)
    with torch.no_grad():
        y = F8.fp8_block_gemv(X, q, s)
        y_main = F8.fp8_torch_block_quant_forward(X, q, s)
    assert y.shape == (M, 1, N) and y.dtype == dtype
    ref = _oracle(X, q, s)
    err, err_main = _rel_err(y, ref), _rel_err(y_main, ref)
    # Only the output rounding is left: about half an ulp of the activation dtype.
    assert err <= torch.finfo(dtype).eps
    assert err <= err_main


def test_split_k_matches_single_pass():
    q, s = _quantize(256, 8192)
    X = torch.randn(4, 1, 8192, device = "cuda").to(torch.bfloat16)
    with torch.no_grad():
        split = F8.fp8_block_gemv(X, q, s)
        sm = F8._sm_count
        try:
            F8._sm_count = lambda device: 1  # 2 tiles >= 2 * SMs: no split
            single = F8.fp8_block_gemv(X, q, s)
        finally:
            F8._sm_count = sm
    ref = _oracle(X, q, s)
    assert _rel_err(split, ref) <= torch.finfo(torch.bfloat16).eps
    assert _rel_err(single, ref) <= torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(split, single, atol = 0, rtol = 2 * torch.finfo(torch.bfloat16).eps)


def test_routing_decode_only(monkeypatch):
    q, s = _quantize(512, 1024)
    calls = []
    real = F8.fp8_block_gemv
    monkeypatch.setattr(F8, "fp8_block_gemv", lambda *a: calls.append(1) or real(*a))
    with torch.no_grad():
        F8.fp8_block_quant_linear(
            torch.randn(8, 1, 1024, device = "cuda", dtype = torch.bfloat16), q, s
        )
        assert len(calls) == 1
        F8.fp8_block_quant_linear(
            torch.randn(2, 4, 1024, device = "cuda", dtype = torch.bfloat16), q, s
        )
        F8.fp8_block_quant_linear(
            torch.randn(17, 1, 1024, device = "cuda", dtype = torch.bfloat16), q, s
        )
    X = torch.randn(2, 1, 1024, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    F8.fp8_block_quant_linear(X, q, s).sum().backward()  # training keeps the autograd path
    assert len(calls) == 1 and X.grad is not None
    assert not F8.can_use_fp8_block_gemv(X.detach(), q, s.t().contiguous())
    assert not F8.can_use_fp8_block_gemv(X.detach().float(), q, s)
    assert not F8.can_use_fp8_block_gemv(X.detach(), q, s.to(torch.float64))


def test_kill_switch_and_device_gates(monkeypatch):
    q, s = _quantize(256, 512)
    X = torch.randn(1, 1, 512, device = "cuda", dtype = torch.bfloat16)
    assert F8.can_use_fp8_block_gemv(X, q, s)
    monkeypatch.setattr(F8, "_FP8_BLOCK_GEMV", False)
    assert not F8.can_use_fp8_block_gemv(X, q, s)
    monkeypatch.setattr(F8, "_FP8_BLOCK_GEMV", True)
    # pre-sm89 (no e4m3 Triton) and Blackwell with Triton < 3.4 keep main's routing
    monkeypatch.setattr(F8, "_fp8_device_lacks_kernel", lambda device: True)
    assert not F8.can_use_fp8_block_gemv(X, _uncached(q), s)
    monkeypatch.undo()
    monkeypatch.setattr(F8, "_fp8_gemv_miscompiles", lambda device: True)
    assert not F8.can_use_fp8_block_gemv(X, _uncached(q), s)
    monkeypatch.undo()
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert not F8.can_use_fp8_block_gemv(X, _uncached(q), s)


def test_weight_decision_cached_per_scale():
    q, s = _quantize(256, 512)
    X = torch.randn(1, 1, 512, device = "cuda", dtype = torch.bfloat16)
    assert F8.can_use_fp8_block_gemv(X, q, s)
    assert q._unsloth_fp8_block_gemv[0] is s and q._unsloth_fp8_block_gemv[1]
    assert not F8.can_use_fp8_block_gemv(X, q, s.t().contiguous())
    assert F8.can_use_fp8_block_gemv(X, q, s)


def test_compiled_call_is_one_opaque_op():
    q, s = _quantize(512, 1024)
    X = torch.randn(4, 1, 1024, device = "cuda", dtype = torch.bfloat16)

    def f(x):
        return F8.fp8_block_quant_linear(x, q, s) * 2

    torch._dynamo.reset()
    with torch.no_grad():
        eager = f(X)
        compiled = torch.compile(f, fullgraph = True)(X)
    torch.testing.assert_close(compiled, eager, atol = 0, rtol = 0)


@pytest.mark.parametrize("block", [(64, 64), (128, 64), (64, 256)])
def test_non_128_blocks(block):
    bn, bk = block
    N, K = 384, 1024
    g = torch.Generator(device = "cuda").manual_seed(1)
    W = torch.randn(N, K, device = "cuda", generator = g)
    blocks = W.view(N // bn, bn, K // bk, bk)
    s = (blocks.abs().amax(dim = (1, 3)) / 448.0).float().contiguous()
    q = (blocks / s[:, None, :, None]).view(N, K).to(torch.float8_e4m3fn).contiguous()
    q.block_size = [bn, bk]
    X = torch.randn(4, 1, K, device = "cuda", generator = g).to(torch.bfloat16)
    assert F8.can_use_fp8_block_gemv(X, q, s)
    with torch.no_grad():
        y = F8.fp8_block_gemv(X, q, s)
    S = s.double().repeat_interleave(bn, 0).repeat_interleave(bk, 1)
    ref = X.double().reshape(4, K) @ (q.double() * S).T
    assert _rel_err(y, ref) <= torch.finfo(torch.bfloat16).eps


def test_routes_only_where_never_slower(monkeypatch):
    # Wide outputs (q, gate / up) keep the block GEMM everywhere; on < 100-SM parts big weights do too.
    X = torch.randn(2, 1, 4096, device = "cuda", dtype = torch.bfloat16)
    q, s = _quantize(4096, 4096)
    assert not F8.can_use_fp8_block_gemv(X, q, s)
    q, s = _quantize(2048, 4096)
    assert F8.can_use_fp8_block_gemv(X, q, s)
    Xd = torch.randn(2, 1, 12288, device = "cuda", dtype = torch.bfloat16)
    qd, sd = _quantize(4096, 12288)
    monkeypatch.setattr(F8, "_sm_count", lambda device: 188)
    assert F8.can_use_fp8_block_gemv(Xd, _uncached(qd), sd)
    monkeypatch.setattr(F8, "_sm_count", lambda device: 58)
    assert not F8.can_use_fp8_block_gemv(Xd, _uncached(qd), sd)
    assert F8.can_use_fp8_block_gemv(X, *_quantize(1024, 4096))


@pytest.mark.parametrize("route, expect", [("small_m_first", "small_m"), ("gemv_first", "gemv")])
def test_router_order(monkeypatch, route, expect):
    # 100+ SM parts take the fused kernel here, smaller ones the GEMV.
    if F8.fp8_block_matmul is not F8.w8a8_block_fp8_matmul_triton:
        pytest.skip("the fused small-M kernel replaces the Triton block matmul route only")
    monkeypatch.setattr(F8, "_FP8_BLOCK_SMALL_M", True)
    fn = (
        F8._fp8_block_small_m_first
        if route == "small_m_first"
        else F8._with_fp8_block_gemv(F8.fp8_torch_block_quant_forward)
    )
    q, s = _quantize(1024, 4096)
    X = torch.randn(4, 1, 4096, device = "cuda", dtype = torch.bfloat16)
    calls = []
    real_gemv, real_small = F8.fp8_block_gemv, F8.fp8_block_small_m_linear
    monkeypatch.setattr(F8, "fp8_block_gemv", lambda *a: calls.append("gemv") or real_gemv(*a))
    monkeypatch.setattr(
        F8, "fp8_block_small_m_linear", lambda *a: calls.append("small_m") or real_small(*a)
    )
    with torch.no_grad():
        y = fn(X, q, s)
    assert calls == [expect]
    # act_quant rounds X to e4m3 on the fused path: a few percent, against the GEMV's bf16 eps.
    assert _rel_err(y, _oracle(X, q, s)) <= (
        5e-2 if expect == "small_m" else torch.finfo(torch.bfloat16).eps
    )
    Xg = X.clone().requires_grad_()
    calls.clear()
    fn(Xg, q, s).sum().backward()
    assert "gemv" not in calls and Xg.grad is not None


def test_router_small_m_first_falls_back(monkeypatch):
    # Compiled regions (the fused kernel is eager only) and pre-sm89 parts do not take the fused kernel.
    monkeypatch.setattr(F8, "_FP8_BLOCK_SMALL_M", True)
    q, s = _quantize(1024, 4096)
    X = torch.randn(2, 1, 4096, device = "cuda", dtype = torch.bfloat16)
    torch._dynamo.reset()
    with torch.no_grad():
        compiled = torch.compile(lambda x: F8._fp8_block_small_m_first(x, q, s), fullgraph = True)(X)
        gemv = F8.fp8_block_gemv(X, q, s)
    torch.testing.assert_close(compiled, gemv, atol = 0, rtol = 0)
    monkeypatch.setattr(F8, "_fp8_device_lacks_kernel", lambda device: True)
    monkeypatch.setattr(
        F8, "fp8_block_small_m_linear", lambda *a: pytest.fail("fused kernel used below sm89")
    )
    with torch.no_grad():
        F8._fp8_block_small_m_first(X, _uncached(q), s)


def test_router_binding_follows_sm_count():
    bound = F8.fp8_block_quant_linear
    if (
        F8.fp8_block_matmul is not F8.w8a8_block_fp8_matmul_triton
        or bound.__wrapped__ is not F8.fp8_torch_block_quant_forward
    ):
        pytest.skip("FBGEMM / torchao block route: the GEMV wraps it")
    big = torch.version.hip is None and min(F8._SM_COUNTS) >= 100
    assert (bound is F8._fp8_block_small_m_first) == big


def test_small_gpu_row_cap(monkeypatch):
    # Below 100 SMs: up to 8 rows, 2 over 8M elements; 100+ SM parts take all 16.
    def rows_ok(sms, N, K, rows):
        monkeypatch.setattr(F8, "_sm_count", lambda device: sms)
        q, s = _quantize(N, K)
        X = torch.randn(rows, 1, K, device = "cuda", dtype = torch.bfloat16)
        return F8.can_use_fp8_block_gemv(X, _uncached(q), s)

    assert rows_ok(58, 1024, 4096, 8) and not rows_ok(58, 1024, 4096, 9)
    assert rows_ok(58, 2048, 6144, 2) and not rows_ok(58, 2048, 6144, 3)
    assert rows_ok(188, 2048, 6144, 16) and rows_ok(188, 1024, 4096, 16)
