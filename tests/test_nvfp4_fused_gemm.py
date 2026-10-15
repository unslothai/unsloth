# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
# NVFP4 W4A16 matmuls decode the weight tile by tile inside the GEMM: same per-element weight rounding as
# dequantize + torch.matmul, no full 16 bit weight, and UNSLOTH_NVFP4_FUSED=0 restores the dense path.
import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA"),
]


@pytest.fixture(scope = "module")
def N():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import nvfp4
    return nvfp4


@pytest.fixture
def fused(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_FUSED", "1")


def _dtypes():
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
        return [torch.bfloat16, torch.float16]
    return [torch.float16]


def _random_nvfp4(
    rows,
    cols,
    global_scale = 7.3,
    seed = 0,
):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    packed = torch.randint(0, 256, (rows, cols // 2), dtype = torch.uint8, generator = g)
    scale = (torch.rand(rows, -(-cols // 16), generator = g) * 3 + 0.01).to(torch.float8_e4m3fn)
    return packed.cuda(), scale.cuda(), torch.tensor([global_scale], device = "cuda")


def _calls(N):
    return dict(N._FUSED_CALLS)


def _dense(N, X, packed, scale, gs, monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_FUSED", "0")
    before = _calls(N)
    Y = N._nvfp4_matmul_fwd(X, packed, scale, gs)
    assert _calls(N) == before
    monkeypatch.setenv("UNSLOTH_NVFP4_FUSED", "1")
    return Y


def _fwd(N, X, packed, scale, gs):
    before = N._FUSED_CALLS["fwd"]
    Y = N._nvfp4_matmul_fwd(X, packed, scale, gs)
    assert N._FUSED_CALLS["fwd"] == before + 1, "fused forward did not run"
    return Y


def _dx(N, G, packed, scale, gs):
    before = N._FUSED_CALLS["bwd"]
    dX = N._nvfp4_matmul_bwd(G, packed, scale, gs)
    assert N._FUSED_CALLS["bwd"] == before + 1, "fused dX did not run"
    return dX


def _same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.isnan(), b.isnan())
    assert torch.equal(a.nan_to_num(), b.nan_to_num())


@pytest.mark.parametrize("dtype", _dtypes())
def test_every_e2m1_code_decodes_exactly(N, fused, dtype):
    # Byte b holds code b & 15 in the even column and b >> 4 in the odd one; identity activations read the decoded
    # weight back exactly through both kernels (forward gives W^T, dX gives W).
    packed = torch.arange(256, dtype = torch.uint8, device = "cuda").reshape(16, 16)
    scale = torch.ones(16, 2, device = "cuda").to(torch.float8_e4m3fn)
    gs = torch.ones(1, device = "cuda")
    W = N.nvfp4_dequantize(packed, scale, gs, dtype)
    lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    expect = torch.tensor(
        [[(-1 if c & 8 else 1) * lut[c & 7] for c in (b & 15, b >> 4)] for b in range(256)]
    ).reshape(16, 32)
    assert torch.equal(W.float().cpu(), expect)
    _same(_fwd(N, torch.eye(32, device = "cuda", dtype = dtype), packed, scale, gs), W.t().contiguous())
    _same(_dx(N, torch.eye(16, device = "cuda", dtype = dtype), packed, scale, gs), W)


@pytest.mark.parametrize("dtype", _dtypes())
@pytest.mark.parametrize("global_scale", [1.0, 1e-3, 3.7e4, 0.37])
def test_scale_extremes_match_the_dense_weight(N, fused, monkeypatch, dtype, global_scale):
    # Every finite E4M3 block scale (subnormals, -0.0, 448) against tiny and huge global scales: the in-loop weight
    # must be the dense weight bit for bit. Where fp16 overflows to inf, 0 * inf makes the GEMM NaN on both paths.
    codes = torch.tensor([c for c in range(256) if c & 127 != 127], dtype = torch.uint8)
    rows = 32
    scale = codes[torch.arange(rows * 16) % len(codes)].reshape(rows, 16).cuda().view(torch.float8_e4m3fn)
    packed = torch.full((rows, 128), 0x7F, dtype = torch.uint8, device = "cuda")  # -6.0, 6.0
    packed[:, 1::2] = 0x1A  # -1.0, 0.5
    gs = torch.tensor([global_scale], device = "cuda")
    W = N.nvfp4_dequantize(packed, scale, gs, dtype)
    finite = bool(W.isfinite().all())
    # One nonzero activation reads one weight column (forward) or row (dX) back.
    for col in (0, 1, 2, 3, 17, 255):
        X = torch.zeros(1, 256, device = "cuda", dtype = dtype)
        X[0, col] = 1
        Y = _fwd(N, X, packed, scale, gs)
        _same(Y, W[:, col][None] if finite else _dense(N, X, packed, scale, gs, monkeypatch))
    for row in (0, 5, 31):
        G = torch.zeros(1, rows, device = "cuda", dtype = dtype)
        G[0, row] = 1
        dX = _dx(N, G, packed, scale, gs)
        _same(dX, W[row][None] if finite else G @ W)


def test_nan_block_scale_propagates_like_dense(N, fused, monkeypatch):
    dtype = _dtypes()[0]
    packed, scale, gs = _random_nvfp4(64, 128)
    raw = scale.view(torch.uint8).clone()
    raw[3, 2] = 0x7F
    raw[9, 0] = 0xFF
    scale = raw.view(torch.float8_e4m3fn)
    X = torch.randn(8, 128, device = "cuda", dtype = dtype)
    Y = _fwd(N, X, packed, scale, gs)
    ref = _dense(N, X, packed, scale, gs, monkeypatch)
    assert torch.equal(Y.isnan(), ref.isnan()) and Y.isnan().any()


def _close_to_dense(a, ref):
    # Accumulation order differs from cuBLAS, so cancellation-sized outputs can move by an output ulp.
    torch.testing.assert_close(a, ref, atol = 2**-7 * ref.abs().max().item(), rtol = 1.6e-2)


def _check(a, ref, exact):
    # Against the fp64 product of the same rounded weight, the fused result is no worse than the dense one.
    err_a, err_ref = _oracle_errors(a, ref, exact)
    assert err_a <= 1.25 * err_ref + 1e-6, (err_a, err_ref)
    _close_to_dense(a, ref)


def _oracle_errors(Y, ref, exact):
    exact = exact.double()
    return (Y.double() - exact).norm().item() / exact.norm().item(), (
        ref.double() - exact
    ).norm().item() / exact.norm().item()


@pytest.mark.parametrize("dtype", _dtypes())
@pytest.mark.parametrize(
    "M, N_out, K",
    [(1, 96, 48), (5, 80, 40), (16, 64, 256), (37, 130, 400), (129, 257, 272), (300, 64, 1040)],
)
def test_forward_and_dx_match_dense_and_oracle(N, fused, monkeypatch, dtype, M, N_out, K):
    # Tails everywhere: K not a multiple of the 16 group or of the K tile, odd M and N.
    packed, scale, gs = _random_nvfp4(N_out, K, global_scale = 0.37, seed = M)
    W = N.nvfp4_dequantize(packed, scale, gs, dtype)
    X = torch.randn(M, K, device = "cuda", dtype = dtype)
    G = torch.randn(M, N_out, device = "cuda", dtype = dtype)

    Y = _fwd(N, X, packed, scale, gs)
    ref = _dense(N, X, packed, scale, gs, monkeypatch)
    # fp64 oracle on the rounded weight: both paths differ from it only by accumulation order + output rounding.
    err_fused, err_dense = _oracle_errors(Y, ref, X.double() @ W.double().t())
    assert err_fused <= 1.25 * err_dense + 1e-6, (err_fused, err_dense)
    _close_to_dense(Y, ref)

    dX = _dx(N, G, packed, scale, gs)
    monkeypatch.setenv("UNSLOTH_NVFP4_FUSED", "0")
    dref = N._nvfp4_matmul_bwd(G, packed, scale, gs)
    err_fused, err_dense = _oracle_errors(dX, dref, G.double() @ W.double())
    assert err_fused <= 1.25 * err_dense + 1e-6, (err_fused, err_dense)
    _close_to_dense(dX, dref)


@pytest.mark.parametrize("dtype", _dtypes())
@pytest.mark.parametrize("M, N_out, K", [(4, 128, 4096), (70, 96, 3000), (3, 4096, 200)])
def test_split_reduction_matches_unsplit(N, fused, monkeypatch, dtype, M, N_out, K):
    # Few output tiles split the reduction (forward over K, dX over N) into fp32 partials; uneven last split.
    packed, scale, gs = _random_nvfp4(N_out, K, seed = 7)
    X = torch.randn(M, K, device = "cuda", dtype = dtype)
    G = torch.randn(M, N_out, device = "cuda", dtype = dtype)
    splits = (N._fused_config(M, N_out, K, X.device, False)[-1], N._fused_config(M, K, N_out, X.device, True)[-1])
    assert max(splits) > 1, splits
    Y, dX = _fwd(N, X, packed, scale, gs), _dx(N, G, packed, scale, gs)
    monkeypatch.setattr(N, "_CONFIG_HOOK", lambda *a: N._fused_config(*a)[:-1] + (1,))
    _close_to_dense(Y, _fwd(N, X, packed, scale, gs))
    _close_to_dense(dX, _dx(N, G, packed, scale, gs))
    W = N.nvfp4_dequantize(packed, scale, gs, dtype)
    _close_to_dense(Y, X @ W.t())
    _close_to_dense(dX, G @ W)
    # Deterministic: the partials are summed in a fixed order.
    monkeypatch.undo()
    monkeypatch.setenv("UNSLOTH_NVFP4_FUSED", "1")
    assert torch.equal(Y, _fwd(N, X, packed, scale, gs)) and torch.equal(dX, _dx(N, G, packed, scale, gs))


@pytest.mark.parametrize("dtype", _dtypes())
def test_strided_operands(N, fused, monkeypatch, dtype):
    # Packed weight / scales as column slices of wider buffers, a transposed activation and a 3D activation.
    big_packed, big_scale, gs = _random_nvfp4(72, 512, seed = 3)
    packed, scale = big_packed[:, :96], big_scale[:, :12]
    assert not packed.is_contiguous()
    X = torch.randn(192, 33, device = "cuda", dtype = dtype).t()
    assert X.stride(-1) != 1
    W = N.nvfp4_dequantize(packed, scale, gs, dtype).double()
    Y = _fwd(N, X, packed, scale, gs)
    _check(Y, _dense(N, X, packed, scale, gs, monkeypatch), X.double() @ W.t())
    X3 = torch.randn(3, 7, 192, device = "cuda", dtype = dtype)
    Y3 = _fwd(N, X3, packed, scale, gs)
    assert Y3.shape == (3, 7, 72)
    _check(Y3, _dense(N, X3, packed, scale, gs, monkeypatch), X3.double() @ W.t())
    G3 = torch.randn(3, 7, 72, device = "cuda", dtype = dtype)
    dX3 = _dx(N, G3, packed, scale, gs)
    assert dX3.shape == (3, 7, 192)
    _check(dX3, G3 @ W.to(dtype), G3.double() @ W)


def test_float32_block_scales(N, fused, monkeypatch):
    dtype = _dtypes()[0]
    packed, scale, gs = _random_nvfp4(48, 160, seed = 4)
    scale = scale.float()
    X = torch.randn(9, 160, device = "cuda", dtype = dtype)
    W = N.nvfp4_dequantize(packed, scale, gs, dtype).double()
    _check(_fwd(N, X, packed, scale, gs), _dense(N, X, packed, scale, gs, monkeypatch), X.double() @ W.t())


def test_kill_switch_and_fallbacks(N, monkeypatch):
    dtype = _dtypes()[0]
    packed, scale, gs = _random_nvfp4(64, 128)
    X = torch.randn(4, 128, device = "cuda", dtype = dtype)
    for env in ("0", "false"):
        monkeypatch.setenv("UNSLOTH_NVFP4_FUSED", env)
        before = _calls(N)
        N.nvfp4_linear(X, packed, scale, gs)
        assert _calls(N) == before
    # Default: the per-GPU row gate (4 rows engage wherever a gate is measured; forced here via the override).
    monkeypatch.delenv("UNSLOTH_NVFP4_FUSED")
    monkeypatch.setenv("UNSLOTH_NVFP4_FUSED_MAX_ROWS", "4")
    before = N._FUSED_CALLS["fwd"]
    N.nvfp4_linear(X, packed, scale, gs)
    assert N._FUSED_CALLS["fwd"] == before + 1
    before = N._FUSED_CALLS["fwd"]
    N.nvfp4_linear(torch.cat([X, X]), packed, scale, gs)
    assert N._FUSED_CALLS["fwd"] == before  # 8 rows: past the gate, dense
    # fp32 activations and too few block scales keep the dense path (and its error).
    before = _calls(N)
    N.nvfp4_linear(X.float(), packed, scale, gs)
    assert _calls(N) == before
    with pytest.raises(ValueError):
        N.nvfp4_linear(X, packed, scale[:, :2], gs)


@pytest.mark.parametrize("with_bias", [False, True])
def test_autograd_uses_both_fused_kernels(N, fused, monkeypatch, with_bias):
    dtype = _dtypes()[0]
    packed, scale, gs = _random_nvfp4(192, 128)
    X = torch.randn(2, 5, 128, device = "cuda", dtype = dtype, requires_grad = True)
    dY = torch.randn(2, 5, 192, device = "cuda", dtype = dtype)
    bias = torch.randn(192, device = "cuda", dtype = dtype) if with_bias else None
    before = _calls(N)
    out = N.nvfp4_linear(X, packed, scale, gs, bias)
    out.backward(dY)
    assert _calls(N) == {"fwd": before["fwd"] + 1, "bwd": before["bwd"] + 1}
    W = N.nvfp4_dequantize(packed, scale, gs, dtype)
    Xr = X.detach().clone().requires_grad_(True)
    ref = Xr @ W.t() + (bias if with_bias else 0)
    ref.backward(dY)
    Wd = W.double()
    _check(out, ref, Xr.double() @ Wd.t() + (bias.double() if with_bias else 0))
    _check(X.grad, Xr.grad, dY.double() @ Wd)


def test_no_full_weight_temporary(N, fused, monkeypatch):
    dtype = _dtypes()[0]
    rows, cols = 4096, 4096
    packed, scale, gs = _random_nvfp4(rows, cols, seed = 5)
    dense_bytes = rows * cols * 2
    X = torch.randn(16, cols, device = "cuda", dtype = dtype)
    G = torch.randn(16, rows, device = "cuda", dtype = dtype)

    def peak(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    fwd = peak(lambda: N._nvfp4_matmul_fwd(X, packed, scale, gs))
    bwd = peak(lambda: N._nvfp4_matmul_bwd(G, packed, scale, gs))
    monkeypatch.setenv("UNSLOTH_NVFP4_FUSED", "0")
    dense = peak(lambda: N._nvfp4_matmul_fwd(X, packed, scale, gs))
    assert dense >= dense_bytes  # the measurement does see a full 16 bit weight
    # Output + split fp32 partials only (partials capped at 1/8 of the 16 bit weight).
    assert fwd < dense_bytes // 4 and bwd < dense_bytes // 4, (fwd, bwd, dense)


def test_lora_training_steps_match_dense(N, monkeypatch):
    # A frozen NVFP4 base + LoRA, 4 AdamW steps: fused and dense runs track each other.
    dtype = _dtypes()[0]
    packed, scale, gs = _random_nvfp4(256, 192, global_scale = 37.5, seed = 6)
    data = [torch.randn(4, 24, 192, device = "cuda", dtype = dtype) for _ in range(4)]

    def train(mode):
        monkeypatch.setenv("UNSLOTH_NVFP4_FUSED", mode)
        torch.manual_seed(0)
        A = torch.nn.Parameter(torch.randn(8, 192, device = "cuda") * 0.02)
        B = torch.nn.Parameter(torch.randn(256, 8, device = "cuda") * 0.02)
        opt = torch.optim.AdamW([A, B], lr = 1e-3)
        before = _calls(N)
        losses = []
        for X in data:
            X = X.clone().requires_grad_(True)
            h = N.nvfp4_linear(X, packed, scale, gs) + (X @ A.t().to(dtype)) @ B.t().to(dtype) * 2.0
            loss = h.float().square().mean() + X.float().mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        engaged = _calls(N)["fwd"] - before["fwd"], _calls(N)["bwd"] - before["bwd"]
        return losses, A.detach().float(), B.detach().float(), engaged

    lf, Af, Bf, ef = train("1")
    ld, Ad, Bd, ed = train("0")
    assert ef == (4, 4) and ed == (0, 0)
    for a, b in zip(lf, ld):
        assert abs(a - b) <= 2e-3 * abs(b)
    assert (Af - Ad).norm() / Ad.norm() < 2e-2
    assert (Bf - Bd).norm() / Bd.norm() < 2e-2


def test_fullgraph_compile_runs_the_fused_kernels(N, fused):
    import torch._dynamo

    dtype = _dtypes()[0]
    packed, scale, gs = _random_nvfp4(384, 256)
    step = lambda X: N.nvfp4_linear(X, packed, scale, gs).float().square().mean()
    X = torch.randn(64, 256, device = "cuda", dtype = dtype, requires_grad = True)
    step(X).backward()
    eager = X.grad.clone()
    X.grad = None
    torch._dynamo.reset()
    assert torch._dynamo.explain(step)(X).graph_break_count == 0
    torch._dynamo.reset()
    compiled = torch.compile(step, fullgraph = True)
    before = _calls(N)
    for rows in (16, 48):
        compiled(torch.randn(rows, 256, device = "cuda", dtype = dtype, requires_grad = True)).backward()
    compiled(X).backward()
    assert _calls(N)["fwd"] >= before["fwd"] + 3 and _calls(N)["bwd"] >= before["bwd"] + 3
    torch.testing.assert_close(X.grad, eager)
