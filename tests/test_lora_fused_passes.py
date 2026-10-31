# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_LORA_FUSED_PASSES: the fused LoRA kernels write the rank-r term first and let the base GEMM
accumulate onto it, share X @ A across projections reading the same X, and read X once for every
projection's LoRA gradients. Outputs and gradients match the base-then-LoRA order and an fp64 PEFT
oracle; the shared paths are taken only when they apply."""

from __future__ import annotations

import copy

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda
from unsloth.kernels import fast_lora
from unsloth.kernels import utils as kernel_utils

H, I = 64, 96
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEVICES = ["cpu"] + (["cuda"] if has_real_cuda() else [])


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(H, H, bias = False)
        self.k_proj = torch.nn.Linear(H, H // 2, bias = False)
        self.v_proj = torch.nn.Linear(H, H // 2, bias = False)
        self.o_proj = torch.nn.Linear(H, H, bias = False)
        self.gate_proj = torch.nn.Linear(H, I, bias = False)
        self.up_proj = torch.nn.Linear(H, I, bias = False)
        self.down_proj = torch.nn.Linear(I, H, bias = False)
        self.act_fn = torch.nn.SiLU()


def _model(
    device,
    dtype,
    alpha_pattern = None,
    dropout = 0.0,
    lora_bias = False,
):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(3407)
    cfg = LoraConfig(
        r = 4,
        lora_alpha = 8,
        lora_dropout = dropout,
        target_modules = TARGETS,
        init_lora_weights = False,
        alpha_pattern = alpha_pattern or {},
        lora_bias = lora_bias,
    )
    model = get_peft_model(_Block(), cfg).to(device, dtype)
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
    if lora_bias:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name and name.endswith("bias"):
                    p.normal_()
    model.train()
    return model, model.base_model.model


PATHS = {
    "qkv": (
        lambda b: (lambda X: fast_lora.apply_lora_qkv(b, X)),
        lambda b: (lambda X: (b.q_proj(X), b.k_proj(X), b.v_proj(X))),
    ),
    "o": (lambda b: (lambda X: fast_lora.apply_lora_o(b, X)), lambda b: b.o_proj),
    "mlp": (
        lambda b: (lambda X: fast_lora.apply_lora_mlp_swiglu(b, X)),
        lambda b: (lambda X: b.down_proj(b.act_fn(b.gate_proj(X)) * b.up_proj(X))),
    ),
}


def _run(fn, model, X0):
    X = X0.detach().clone().requires_grad_()
    out = fn(X * 1)  # a non-leaf: the fused backward may write dX into its input buffer
    outs = out if isinstance(out, tuple) else (out,)
    gen = torch.Generator(device = "cpu").manual_seed(7)
    loss = sum(
        (o.double() * torch.randn(o.shape, generator = gen, dtype = torch.float64).to(o.device)).sum()
        for o in outs
    )
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)
    return [o.detach() for o in outs], X.grad.clone(), grads


def _skip_without_native_bf16(device, dtype = torch.bfloat16):
    # Same rule as test_lora_gradient_layout: Triton's bf16 kernels need sm_80+, and is_bf16_supported() also
    # reports emulated bf16 on older GPUs.
    if (
        device == "cuda"
        and dtype is torch.bfloat16
        and (
            not torch.cuda.is_bf16_supported()
            or (not torch.version.hip and torch.cuda.get_device_capability()[0] < 8)
        )
    ):
        pytest.skip("bf16 kernels need native bf16 (sm_80+)")


def _set(monkeypatch, on):
    monkeypatch.setattr(kernel_utils, "LORA_FUSED_PASSES", on)


def _err(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("path", ["qkv", "o", "mlp"])
@pytest.mark.parametrize("alphas", ["equal", "mixed"])
def test_fused_passes_match_the_unfused_order_and_an_fp64_oracle(
    device, dtype, path, alphas, monkeypatch
):
    if device == "cpu" and path == "mlp":
        pytest.skip("the SwiGLU kernels are Triton")
    if device == "cpu" and dtype is torch.bfloat16:
        pytest.skip("bf16 GEMMs on CPU are slow and not what runs in training")
    _skip_without_native_bf16(device, dtype)
    pattern = {"k_proj": 16, "up_proj": 16} if alphas == "mixed" else None
    model, block = _model(device, dtype, pattern)
    X0 = torch.randn(3, 5, H, device = device, dtype = dtype)
    _set(monkeypatch, False)
    ref_out, ref_dX, ref_g = _run(PATHS[path][0](block), model, X0)
    _set(monkeypatch, True)
    out, dX, g = _run(PATHS[path][0](block), model, X0)
    oracle_model = copy.deepcopy(model).double()
    o_out, o_dX, o_g = _run(
        PATHS[path][1](oracle_model.base_model.model), oracle_model, X0.double()
    )
    tol = 1e-5 if dtype is torch.float32 else 2e-2
    for a, b, o in zip(out, ref_out, o_out):
        assert _err(a, o) <= max(tol, 1.5 * _err(b, o)), (_err(a, o), _err(b, o))
    assert _err(dX, o_dX) <= max(tol, 1.5 * _err(ref_dX, o_dX)), (
        _err(dX, o_dX),
        _err(ref_dX, o_dX),
    )
    assert set(g) == set(ref_g) == set(o_g)
    for n in g:
        assert g[n].shape == ref_g[n].shape
        assert _err(g[n], o_g[n]) <= max(tol, 1.5 * _err(ref_g[n], o_g[n])), (
            n,
            _err(g[n], o_g[n]),
            _err(ref_g[n], o_g[n]),
        )


def test_shared_paths_engage_only_when_they_apply(monkeypatch):
    calls = {"xa": 0, "bwd": 0}
    real_xa, real_bwd = fast_lora._shared_XA, fast_lora._shared_lora_backward

    def xa(*a, **k):
        calls["xa"] += 1
        return real_xa(*a, **k)

    def bwd(*a, **k):
        calls["bwd"] += 1
        return real_bwd(*a, **k)

    monkeypatch.setattr(fast_lora, "_shared_XA", xa)
    monkeypatch.setattr(fast_lora, "_shared_lora_backward", bwd)
    X0 = torch.randn(2, 3, H)
    for alphas, on, want in (
        ("equal", True, (1, 1)),
        ("mixed", True, (1, 0)),
        ("equal", False, (0, 0)),
    ):
        _set(monkeypatch, on)
        calls.update(xa = 0, bwd = 0)
        model, block = _model("cpu", torch.float32, {"k_proj": 16} if alphas == "mixed" else None)
        _run(PATHS["qkv"][0](block), model, X0)
        assert (calls["xa"], calls["bwd"]) == want, (alphas, on, calls)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kill_switch_is_the_base_then_lora_order(device, dtype, monkeypatch):
    # UNSLOTH_LORA_FUSED_PASSES=0: base GEMM, then the rank-r term added in place, bit for bit.
    if device == "cpu" and dtype is torch.bfloat16:
        pytest.skip("bf16 GEMMs on CPU are slow and not what runs in training")
    _skip_without_native_bf16(device, dtype)
    _set(monkeypatch, False)
    g = torch.Generator(device = "cpu").manual_seed(0)
    X, W = torch.randn(7, H, generator = g), torch.randn(I, H, generator = g)
    A, B = torch.randn(4, H, generator = g), torch.randn(I, 4, generator = g)
    X, W, A, B = (t.to(device, dtype) for t in (X, W, A, B))
    want = X @ W.t()
    kernel_utils.addmm_(want, X @ A.t(), B.t(), alpha = 2.0)
    assert torch.equal(kernel_utils.matmul_lora(X, W, None, A, B, 2.0), want)


def _drop_lora_bias(block, keep):
    # PEFT puts lora_bias on every target; keep it only on `keep` (a down-only bias, for example).
    for name in TARGETS:
        if name not in keep:
            getattr(block, name).lora_B["default"].bias = None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("path", ["qkv", "o", "mlp"])
@pytest.mark.parametrize("config", ["plain", "dropout", "lora_bias", "down_bias"])
def test_exclusions_keep_the_base_then_lora_order(device, dtype, path, config, monkeypatch):
    # Every configuration the fused order excludes (fp16 at all, dropout, lora_bias anywhere in the
    # module) gives the kill switch's outputs and gradients bit for bit, forward and backward.
    if device == "cpu" and path == "mlp":
        pytest.skip("the SwiGLU kernels are Triton")
    if config == "down_bias" and path != "mlp":
        pytest.skip("a down-only bias is an MLP case")
    if config == "plain" and dtype is not torch.float16:
        pytest.skip("plain bf16 / fp32 take the fused order")
    _skip_without_native_bf16(device, dtype)
    kwargs = {"dropout": 0.1} if config == "dropout" else {}
    kwargs["lora_bias"] = config in ("lora_bias", "down_bias")
    model, block = _model(device, dtype, **kwargs)
    if config == "down_bias":
        _drop_lora_bias(block, keep = ("down_proj",))
    X0 = torch.randn(3, 5, H, device = device, dtype = dtype)
    runs = []
    for on in (False, True):
        _set(monkeypatch, on)
        torch.manual_seed(11)
        runs.append(_run(PATHS[path][0](block), model, X0))
    (out0, dX0, g0), (out1, dX1, g1) = runs
    assert all(torch.equal(a, b) for a, b in zip(out0, out1))
    assert torch.equal(dX0, dX1)
    assert set(g0) == set(g1) and all(torch.equal(g0[n], g1[n]) for n in g0)


@pytest.mark.parametrize("device", DEVICES)
def test_fp16_rank_r_term_cannot_overflow_before_the_base_cancels_it(device, monkeypatch):
    # The rank-r term alone is ~70000 (past fp16's 65504) but the output is ~10000. The base-then-LoRA
    # order adds it in the GEMM's fp32 epilogue; writing it first would store inf.
    X = torch.ones(1, H, dtype = torch.float16, device = device)
    W = torch.zeros(I, H, dtype = torch.float16, device = device)
    W[0] = -60000 / H
    A = torch.zeros(4, H, dtype = torch.float16, device = device)
    A[0, 0] = 1
    B = torch.zeros(I, 4, dtype = torch.float16, device = device)
    B[0, 0] = 35000
    outs = {}
    for on in (True, False):
        _set(monkeypatch, on)
        outs[on] = kernel_utils.matmul_lora(X, W, None, A, B, 2.0)
    assert torch.isfinite(outs[True]).all() and abs(float(outs[True][0, 0]) - 10000) <= 32
    assert torch.equal(outs[True], outs[False])


def test_fused_eligibility_is_one_rule(monkeypatch):
    _set(monkeypatch, True)
    from types import SimpleNamespace

    from unsloth.kernels.int4_packed import Int4QuantState

    ok = kernel_utils._lora_fused_ok
    dense = lambda dtype: torch.empty(4, 4, dtype = dtype)  # noqa: E731
    u8 = torch.empty(4, 4, dtype = torch.uint8)
    assert ok(dense(torch.bfloat16), None, torch.bfloat16) and ok(
        dense(torch.float32), None, torch.float32
    )
    assert not ok(dense(torch.float16), None, torch.float16)  # fp16 keeps the old order
    assert not ok(dense(torch.float32), None, torch.bfloat16)  # fp32 weight under bf16 autocast
    # fast_dequantize returns a 4-bit state's own dtype, not the one asked for.
    assert ok(u8, SimpleNamespace(dtype = torch.bfloat16), torch.bfloat16)
    assert not ok(u8, SimpleNamespace(dtype = torch.float16), torch.bfloat16)
    assert not ok(u8, [None, None, torch.float16], torch.bfloat16)
    assert not ok(dense(torch.float8_e4m3fn), torch.ones(1), torch.bfloat16)
    int4 = object.__new__(Int4QuantState)
    w = torch.empty(8, 4, dtype = torch.int8)
    assert not ok(w, int4, torch.bfloat16) and not ok(
        w.t(), int4, torch.bfloat16
    )  # either orientation
    packed = type("Packed", (), {"_unsloth_packed_weight_state": True, "dtype": torch.bfloat16})()
    assert not ok(u8, packed, torch.bfloat16)


def test_a_caller_buffer_that_is_not_the_output_keeps_main_behavior(monkeypatch):
    # torch.matmul(..., out = empty) resizes; the fused order writes the rank-r term first, so it is
    # taken only when the buffer is already the output.
    g = torch.Generator(device = "cpu").manual_seed(0)
    X, W = torch.randn(7, H, generator = g), torch.randn(I, H, generator = g)
    A, B = torch.randn(4, H, generator = g), torch.randn(I, 4, generator = g)
    outs = {}
    for on in (True, False):
        _set(monkeypatch, on)
        out = torch.empty(0)
        assert kernel_utils.matmul_lora(X, W, None, A, B, 2.0, out = out) is not None
        outs[on] = out
    assert outs[True].shape == (7, I) and torch.equal(outs[True], outs[False])


@pytest.mark.parametrize("device", DEVICES)
def test_fp32_base_under_autocast_keeps_the_base_then_lora_order(device, monkeypatch):
    # addmm_ is not autocast: an fp32 base weight with bf16 activations takes the old order.
    _skip_without_native_bf16(device)
    half = torch.bfloat16
    g = torch.Generator(device = "cpu").manual_seed(0)
    X, W = torch.randn(7, H, generator = g), torch.randn(I, H, generator = g)
    A, B = torch.randn(4, H, generator = g), torch.randn(I, 4, generator = g)
    X, W, A, B = X.to(device, half), W.to(device), A.to(device), B.to(device)
    model, block = _model(device, torch.float32)
    X0 = torch.randn(2, 5, H, generator = g).to(device, half)
    outs, runs = {}, {}
    for on in (True, False):
        _set(monkeypatch, on)
        with torch.autocast(device, dtype = half):
            outs[on] = kernel_utils.matmul_lora(X, W, None, A, B, 2.0)
            runs[on] = _run(PATHS["o"][0](block), model, X0)
    assert torch.equal(outs[True], outs[False])
    (out_on, dX_on, g_on), (out_off, dX_off, g_off) = runs[True], runs[False]
    assert torch.equal(out_on[0], out_off[0]) and torch.equal(dX_on, dX_off)
    assert g_on.keys() == g_off.keys() and all(torch.equal(g_on[k], g_off[k]) for k in g_on)


@pytest.mark.skipif(not has_real_cuda(), reason = "needs bitsandbytes 4-bit on a real GPU")
def test_fp16_nf4_state_under_bf16_autocast_keeps_the_base_then_lora_order(monkeypatch):
    # fast_dequantize returns the state's fp16, so the in-place base accumulate would mix dtypes.
    bnb = pytest.importorskip("bitsandbytes")
    _skip_without_native_bf16("cuda")
    lin = bnb.nn.Linear4bit(H, I, bias = False, compute_dtype = torch.float16, quant_type = "nf4")
    lin.weight = bnb.nn.Params4bit(
        torch.randn(I, H, dtype = torch.float16), requires_grad = False, quant_type = "nf4"
    )
    lin = lin.cuda()
    W, Wq = lin.weight, lin.weight.quant_state
    assert Wq.dtype == torch.float16
    g = torch.Generator(device = "cpu").manual_seed(0)
    X = torch.randn(7, H, generator = g).cuda().bfloat16()
    A, B = torch.randn(4, H, generator = g).cuda(), torch.randn(I, 4, generator = g).cuda()
    outs = {}
    for on in (True, False):
        _set(monkeypatch, on)
        with torch.autocast("cuda", dtype = torch.bfloat16):
            outs[on] = kernel_utils.matmul_lora(X, W, Wq, A, B, 2.0)
    assert torch.equal(outs[True], outs[False])


def test_small_update_survives_the_single_rounding(monkeypatch):
    # bf16 base output 1.0 plus a rank-r term of 1/128 rounds once to 1.0078125; the old order rounds the
    # base output first and the separate add then loses everything below half a bf16 step.
    x = torch.tensor([[1.0, 1.0]], dtype = torch.bfloat16)
    w = torch.tensor([[1.0, 1 / 256]], dtype = torch.bfloat16)
    a = torch.tensor([[1.0, 0.0]], dtype = torch.bfloat16)
    b = torch.tensor([[1 / 512]], dtype = torch.bfloat16)
    exact = (x.double() @ w.double().t() + (x.double() @ a.double().t()) @ b.double().t()).to(
        torch.bfloat16
    )
    _set(monkeypatch, True)
    got = kernel_utils.matmul_lora(x, w, None, a, b, 1.0)
    assert torch.equal(got, exact) and got.item() == 1.0078125
    _set(monkeypatch, False)
    assert kernel_utils.matmul_lora(x, w, None, a, b, 1.0).item() == 1.0


def test_a_caller_buffer_sharing_an_operand_keeps_main_behavior(monkeypatch):
    # Writing the rank-r term first would overwrite a base operand the GEMM has not read yet.
    _set(monkeypatch, True)
    g = torch.Generator(device = "cpu").manual_seed(0)
    X, W = torch.randn(H, H, generator = g), torch.randn(H, H, generator = g)
    A, B = torch.randn(4, H, generator = g), torch.randn(H, 4, generator = g)
    assert kernel_utils._shares_storage(W, X, W)
    assert not kernel_utils._shares_storage(torch.empty(H, H), X, W, A, B, None)
    # The branch taken is checked, not the values: matmul with out= overlapping an input is undefined on
    # main too (on CPU a 64 x 64 result differs from call to call).
    betas = []
    real = kernel_utils.addmm_
    monkeypatch.setattr(
        kernel_utils, "addmm_", lambda *a, **k: betas.append(k.get("beta", 1)) or real(*a, **k)
    )
    out = W.clone()
    kernel_utils.matmul_lora(X, out, None, A, B, 2.0, out = out)
    assert betas == [1], "a buffer sharing W must take the base-then-LoRA order"
    betas.clear()
    kernel_utils.matmul_lora(X, W, None, A, B, 2.0, out = torch.empty(H, H))
    assert betas == [0], "a separate buffer takes the fused order"


def test_shared_lora_a_grads_have_the_parameter_layout(monkeypatch):
    # The shared backward hands back d_A whose .t() is contiguous like the [r, K] parameter, so autograd
    # does not copy it into a new layout.
    _set(monkeypatch, True)
    X = torch.randn(10, H)
    As = (torch.randn(4, H).t(), torch.randn(4, H).t())
    Bs = (torch.randn(H, 4).t(), torch.randn(I, 4).t())
    dYs = (torch.randn(10, H), torch.randn(10, I))
    d_As, _, _ = fast_lora._shared_lora_backward(X, dYs, As, Bs, 2.0, None)
    for dA in d_As:
        assert dA.t().is_contiguous(), dA.stride()


def test_shared_lora_a_grads_are_views_of_one_buffer(monkeypatch):
    # Each d_A is a row block of one [sum r, K] buffer: no per-projection copy into the parameter layout.
    _set(monkeypatch, True)
    X = torch.randn(10, H)
    As = (torch.randn(4, H).t(), torch.randn(3, H).t())
    Bs = (torch.randn(H, 4).t(), torch.randn(I, 3).t())
    dYs = (torch.randn(10, H), torch.randn(10, I))
    d_As, d_Bs, dX = fast_lora._shared_lora_backward(X, dYs, As, Bs, 2.0, None)
    ptrs = {dA.untyped_storage().data_ptr() for dA in d_As}
    assert len(ptrs) == 1, "d_A blocks were copied out of the shared buffer"
    X64 = X.double()
    for dA, dB_, A, B, dY in zip(d_As, d_Bs, As, Bs, dYs):
        y_dB = dY.double() @ B.double().t()
        torch.testing.assert_close(dA.double(), 2.0 * X64.t() @ y_dB, rtol = 1e-4, atol = 1e-4)
        torch.testing.assert_close(
            dB_.double(), 2.0 * (A.double().t() @ X64.t()) @ dY.double(), rtol = 1e-4, atol = 1e-4
        )


def _small_update(dtype = torch.bfloat16):
    # Base output 1 + 1/256 plus a rank-r term 1/512: the fused order rounds once to 1.0078125, the
    # base-then-LoRA order rounds the base to 1.0 first and then loses the update.
    x = torch.tensor([[1.0, 1.0]], dtype = dtype)
    w = torch.tensor([[1.0, 1 / 256]], dtype = dtype)
    a = torch.tensor([[1.0, 0.0]], dtype = dtype)
    b = torch.tensor([[1 / 512]], dtype = dtype)
    return x, w, a, b


def test_compiled_caller_buffer_keeps_main_behavior(monkeypatch):
    # Traced, a caller's buffer cannot be checked for aliasing, so it keeps the base-then-LoRA order.
    _set(monkeypatch, True)
    torch._dynamo.reset()
    x, w, a, b = _small_update()

    def f(x, w, a, b):
        out = torch.empty(1, 1, dtype = x.dtype)
        return kernel_utils.matmul_lora(x, w, None, a, b, 1.0, out = out)

    got = torch.compile(f, backend = "aot_eager", fullgraph = True)(x, w, a, b)
    assert got.item() == 1.0
    # A buffer that aliases W: the old order reads W before the output is written.
    g = torch.Generator(device = "cpu").manual_seed(1)
    X, W = torch.randn(4, 4, generator = g), torch.randn(4, 4, generator = g)
    A, B = torch.randn(4, 4, generator = g), torch.randn(4, 4, generator = g)
    ref = X @ W.t() + 2.0 * (X @ A.t()) @ B.t()

    def g_(X, W, A, B):
        out = W.clone()
        return kernel_utils.matmul_lora(X, out, None, A, B, 2.0, out = out)

    torch._dynamo.reset()
    got = torch.compile(g_, backend = "aot_eager", fullgraph = True)(X, W, A, B)
    torch.testing.assert_close(got, ref, rtol = 1e-5, atol = 1e-5)


def test_fake_tracing_with_a_caller_buffer(monkeypatch):
    # make_fx fake tensors have no data pointer: the buffer is treated as shared, not inspected.
    from torch.fx.experimental.proxy_tensor import make_fx

    _set(monkeypatch, True)
    X, W, A, B = torch.randn(3, H), torch.randn(H, H), torch.randn(4, H), torch.randn(H, 4)

    def f(X, W, A, B, out):
        return kernel_utils.matmul_lora(X, W, None, A, B, 2.0, out = out)

    gm = make_fx(f, tracing_mode = "fake")(X, W, A, B, torch.empty(3, H))
    out = torch.empty(3, H)
    torch.testing.assert_close(
        gm(X, W, A, B, out), X @ W.t() + 2.0 * (X @ A.t()) @ B.t(), rtol = 1e-5, atol = 1e-5
    )


@pytest.mark.parametrize("fused", [True, False])
def test_a_precomputed_xa_sharing_the_caller_buffer_is_not_reused(fused, monkeypatch):
    # The base GEMM writes the buffer before the rank-r term reads XA: XA is recomputed instead.
    _set(monkeypatch, True)
    g = torch.Generator(device = "cpu").manual_seed(3)
    X, W = torch.randn(3, 4, generator = g), torch.randn(2, 4, generator = g)
    A, B = torch.randn(2, 4, generator = g), torch.randn(2, 2, generator = g)
    ref = X @ W.t() + 2.0 * (X @ A.t()) @ B.t()
    XA = X @ A.t()
    got = kernel_utils.matmul_lora(X, W, None, A, B, 2.0, XA = XA, out = XA, fused = fused)
    torch.testing.assert_close(got, ref, rtol = 1e-5, atol = 1e-5)


def test_sparse_base_weight_keeps_the_base_then_lora_order(monkeypatch):
    # addmm_ cannot accumulate a dense X @ sparse W.t() in place; main's matmul handles it.
    _set(monkeypatch, True)
    X, A, B = torch.ones(1, 2), torch.ones(1, 2), torch.ones(2, 1)
    W = torch.eye(2).to_sparse_coo()
    assert not kernel_utils._lora_fused_ok(W, None, torch.float32)
    torch.testing.assert_close(
        kernel_utils.matmul_lora(X, W, None, A, B, 1.0), torch.full((1, 2), 3.0)
    )


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_multi_token_inference_follows_the_training_order(dropout, monkeypatch):
    # fast_linear_forward with q_len > 1 takes the order the fused training kernels use: a configured
    # lora_dropout (even in eval) keeps base-then-LoRA, a plain adapter takes the fused order.
    _skip_without_native_bf16("cpu")
    from peft import LoraConfig, get_peft_model

    _set(monkeypatch, True)
    lin = torch.nn.Sequential(torch.nn.Linear(2, 1, bias = False))
    model = get_peft_model(
        lin, LoraConfig(r = 1, lora_alpha = 1, lora_dropout = dropout, target_modules = ["0"])
    )
    model = model.to(torch.bfloat16).eval()
    proj = model.base_model.model[0]
    x, w, a, b = _small_update()
    with torch.no_grad():
        proj.base_layer.weight.copy_(w)
        proj.lora_A["default"].weight.copy_(a)
        proj.lora_B["default"].weight.copy_(b)
    X = x.view(1, 1, 2).expand(1, 3, 2).contiguous()
    got = kernel_utils.fast_linear_forward(proj, X)
    assert got[0, 0, 0].item() == (1.0 if dropout else 1.0078125)


_DYNAMIC_RANK_CASE = """
import sys
import torch
import unsloth  # noqa: F401
from unsloth.kernels import fast_lora

torch.manual_seed(0)
weights = []
for n, r in zip((16, 12, 12), (2, 3, 4)):
    W = torch.randn(n, 16) * 0.1
    A, B = torch.randn(r, 16) * 0.1, torch.randn(n, r) * 0.1
    if sys.argv[1] == "param":
        A, B = torch.nn.Parameter(A), torch.nn.Parameter(B)
    else:
        A, B = A.requires_grad_(), B.requires_grad_()
    weights.append((W, A, B))
(QW, QA, QB), (KW, KA, KB), (VW, VA, VB) = weights


def f(X, QA, QB, KA, KB, VA, VB):
    q, k, v = fast_lora.LoRA_QKV.apply(
        X, QW, None, QA, QB, 2.0, KW, None, KA, KB, 2.0, VW, None, VA, VB, 2.0, False
    )
    return q.sum() + k.sum() + v.sum()


g = torch.compile(f, backend = "aot_eager", fullgraph = True, dynamic = True)
for seq in (3, 5, 7):
    X = torch.randn(2, seq, 16, requires_grad = True)
    g(X, QA, QB, KA, KB, VA, VB).backward()
    assert X.grad.shape == X.shape
print("OK")
"""


@pytest.mark.parametrize("param", [False, True])
def test_dynamic_compile_with_a_rank_equal_to_the_first_length(param):
    # torch 2.14: dynamic-shape plain-tensor LoRA weights once recompiled to a width-0 dQ; they keep the
    # base-then-LoRA order when traced. Fresh interpreter: depends on Dynamo state.
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH = os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]),
        UNSLOTH_DISABLE_CPU_FALLBACK = "0",
    )
    p = subprocess.run(
        [sys.executable, "-c", _DYNAMIC_RANK_CASE, "param" if param else "plain"],
        cwd = root,
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert p.returncode == 0 and "OK" in p.stdout, (p.stdout + p.stderr)[-3000:]


@pytest.mark.parametrize("param", [False, True])
def test_traced_order_depends_on_parameter_lora_weights(param, monkeypatch):
    _skip_without_native_bf16("cpu")
    _set(monkeypatch, True)
    torch._dynamo.reset()
    x, w, a, b = _small_update()
    if param:
        a, b = torch.nn.Parameter(a), torch.nn.Parameter(b)
    else:
        a, b = a.requires_grad_(), b.requires_grad_()

    def f(x, a, b):
        return fast_lora.LoRA_W.apply(x.view(1, 1, 2), w, None, a, b, 1.0)

    got = torch.compile(f, backend = "aot_eager", fullgraph = True)(x, a, b)
    assert got.item() == (1.0078125 if param else 1.0)


class _FakeF8(torch.Tensor):
    pass


@pytest.mark.parametrize("backend", ["packed", "int4", "float8"])
def test_a_precomputed_xa_sharing_the_buffer_is_not_reused_by_any_dispatch(backend, monkeypatch):
    # Packed, INT4 and Float8Tensor bases write the caller's buffer too before the rank-r term reads XA.
    _set(monkeypatch, True)
    g = torch.Generator(device = "cpu").manual_seed(3)
    X, W = torch.randn(3, 4, generator = g), torch.randn(2, 4, generator = g)
    A, B = torch.randn(2, 4, generator = g), torch.randn(2, 2, generator = g)
    ref = X @ W.t() + 2.0 * (X @ A.t()) @ B.t()
    W_quant = None
    if backend == "packed":
        W_quant = type(
            "Packed",
            (),
            {
                "_unsloth_packed_weight_state": True,
                "dtype": torch.float32,
                "matmul": lambda self, X, W, out = None: torch.matmul(X, W.t(), out = out),
            },
        )()
        monkeypatch.setattr(kernel_utils, "_is_packed_state", lambda q: q is W_quant)
    elif backend == "int4":
        Int4 = type("Int4QuantState", (), {})
        monkeypatch.setattr(kernel_utils, "Int4QuantState", Int4)
        monkeypatch.setattr(
            kernel_utils, "_int4_matmul", lambda X, W, q, out = None: torch.matmul(X, W.t(), out = out)
        )
        W_quant = Int4()
    else:
        monkeypatch.setattr(kernel_utils, "Float8Tensor", _FakeF8)
        W = W.as_subclass(_FakeF8)
        W.block_size = (1, W.shape[1])  # not rowwise-after-transpose: the forward GEMM
    XA = X @ A.t()
    got = kernel_utils.matmul_lora(X, W, W_quant, A, B, 2.0, XA = XA, out = XA)
    torch.testing.assert_close(got.as_subclass(torch.Tensor), ref, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("on", [True, False])
def test_a_3d_precomputed_xa_is_flattened_with_x(on, monkeypatch):
    _set(monkeypatch, on)
    g = torch.Generator(device = "cpu").manual_seed(5)
    X, W = torch.randn(2, 3, H, generator = g), torch.randn(I, H, generator = g)
    A, B = torch.randn(4, H, generator = g), torch.randn(I, 4, generator = g)
    ref = kernel_utils.matmul_lora(X, W, None, A, B, 2.0)
    got = kernel_utils.matmul_lora(X, W, None, A, B, 2.0, XA = X @ A.t())
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol = 1e-5, atol = 1e-5)


def test_sparse_base_weight_keeps_the_base_then_lora_order_under_export(monkeypatch):
    # torch.export traces with is_compiling() True: the layout check must still run there.
    _set(monkeypatch, True)
    if not hasattr(torch, "export"):
        pytest.skip("torch.export is not available")

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.W = torch.eye(2).to_sparse_coo()
            self.A, self.B = torch.ones(1, 2), torch.ones(2, 1)

        def forward(self, x):
            return kernel_utils.matmul_lora(x, self.W, None, self.A, self.B, 1.0)

    x = torch.ones(1, 2)
    ep = torch.export.export(M(), (x,))
    torch.testing.assert_close(ep.module()(x), torch.full((1, 2), 3.0))


@pytest.fixture
def one_rank_mesh(tmp_path):
    # One-rank gloo group on a file store (no TCP / libuv rendezvous, no environment variables); only a
    # group this fixture created is destroyed.
    if not isinstance(kernel_utils._LORA_DTENSOR, type):
        pytest.skip("torch.distributed.tensor is not available")
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo is not available")
    from torch.distributed.device_mesh import init_device_mesh

    created = not dist.is_initialized()
    if created:
        store = dist.FileStore(str(tmp_path / "store"), 1)
        dist.init_process_group("gloo", store = store, rank = 0, world_size = 1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        if created:
            dist.destroy_process_group()


def test_dtensor_base_weight_keeps_the_base_then_lora_order(one_rank_mesh, monkeypatch):
    # A DTensor carries placements the plain accumulator would drop: keep main's order.
    from torch.distributed.tensor import Replicate, distribute_tensor

    _set(monkeypatch, True)
    W = distribute_tensor(torch.randn(4, 3), one_rank_mesh, [Replicate()])
    assert not kernel_utils._lora_fused_ok(W, None, torch.float32)


@pytest.mark.parametrize("placement", ["replicate", "shard0", "shard1"])
def test_dtensor_activation_keeps_the_base_then_lora_order(placement, one_rank_mesh, monkeypatch):
    # A DTensor X with plain W / A / B (implicit replication) keeps its value and placement.
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.experimental import implicit_replication

    g = torch.Generator(device = "cpu").manual_seed(7)
    X, W = torch.randn(4, 6, generator = g), torch.randn(5, 6, generator = g)
    A, B = torch.randn(2, 6, generator = g), torch.randn(5, 2, generator = g)
    pl = {"replicate": Replicate(), "shard0": Shard(0), "shard1": Shard(1)}[placement]
    Xd = distribute_tensor(X, one_rank_mesh, [pl])
    out = {}
    for on in (False, True):
        _set(monkeypatch, on)
        with implicit_replication():
            try:
                out[on] = kernel_utils.matmul_lora(Xd, W, None, A, B, 2.0)
            except NotImplementedError as e:
                if on:
                    raise
                pytest.skip(f"main's order needs DTensor ops this torch lacks: {e}"[:200])
    assert out[True].placements == out[False].placements
    torch.testing.assert_close(out[True].full_tensor(), out[False].full_tensor())


def _dtensor_lora_inputs(mesh, which, placement):
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    g = torch.Generator().manual_seed(0)
    X, W = torch.randn(2, 4, 4, generator = g), torch.randn(4, 4, generator = g)
    A, B = torch.randn(2, 4, generator = g), torch.randn(4, 2, generator = g)
    pl = {"replicate": Replicate(), "shard0": Shard(0), "shard1": Shard(1)}[placement]
    # Mixed plain / DTensor LoRA weights fail on main as well (in-place gradient writes), so the adapters
    # are either both plain or both DTensor.
    X = distribute_tensor(X, mesh, [pl])
    if which == "XAB":
        A, B = (distribute_tensor(t, mesh, [Replicate()]) for t in (A, B))
    return X.requires_grad_(), W, A.requires_grad_(), B.requires_grad_()


def _apply_lora_function(name, X, W, A, B):
    from unsloth.kernels.swiglu import _DWf_DW_dfg_traced

    if name == "W":
        return (fast_lora.LoRA_W.apply(X, W, None, A, B, 1.0),)
    if name == "QKV":
        return fast_lora.LoRA_QKV.apply(
            X, W, None, A, B, 1.0, W, None, A, B, 1.0, W, None, A, B, 1.0
        )
    silu_mul = lambda e, g: torch.nn.functional.silu(e) * g
    return (
        fast_lora.LoRA_MLP.apply(
            X,
            W,
            None,
            A,
            B,
            1.0,
            W,
            None,
            A,
            B,
            1.0,
            W,
            None,
            A,
            B,
            1.0,
            silu_mul,
            _DWf_DW_dfg_traced,
            False,
        ),
    )


def _run_dtensor_lora(name, which, placement, mesh, monkeypatch, on):
    # Main's order is the reference: where this torch's DTensor cannot run it (older releases lack the
    # sharding rules for views of a sharded dim / in-place addmm), there is nothing to compare against.
    from torch.distributed.tensor.experimental import implicit_replication

    _set(monkeypatch, on)
    X, W, A, B = _dtensor_lora_inputs(mesh, which, placement)
    try:
        with implicit_replication():
            out = _apply_lora_function(name, X, W, A, B)
            sum(o.sum() for o in out).backward()
    except Exception as e:  # noqa: BLE001
        if on:
            raise
        pytest.skip(f"main's order needs DTensor ops this torch lacks: {e}"[:200])
    return out, X.grad


@pytest.mark.parametrize("name", ["W", "QKV", "MLP"])
@pytest.mark.parametrize("placement", ["replicate", "shard0", "shard1"])
def test_dtensor_operands_keep_the_base_then_lora_order(
    name, placement, one_rank_mesh, monkeypatch
):
    # matmul_lora falls back for a DTensor X / A / B, so the decision the backward reads must agree: the
    # shared and fused backward paths would otherwise redistribute the input gradient.
    _, ref = _run_dtensor_lora(name, "XAB", placement, one_rank_mesh, monkeypatch, False)
    out, got = _run_dtensor_lora(name, "XAB", placement, one_rank_mesh, monkeypatch, True)
    assert out[0].grad_fn.fused is False
    assert got.placements == ref.placements
    torch.testing.assert_close(got.full_tensor(), ref.full_tensor())


@pytest.mark.parametrize("name", ["W", "QKV", "MLP"])
def test_a_dtensor_activation_records_the_base_then_lora_order(name, one_rank_mesh, monkeypatch):
    # With plain adapters the backward cannot run under DTensor on main either, so only the recorded
    # forward decision is checked here.
    from torch.distributed.tensor.experimental import implicit_replication

    X, W, A, B = _dtensor_lora_inputs(one_rank_mesh, "X", "replicate")
    for on in (False, True):
        _set(monkeypatch, on)
        try:
            with implicit_replication():
                out = _apply_lora_function(name, X, W, A, B)
        except Exception as e:  # noqa: BLE001
            if on:
                raise
            pytest.skip(f"main's order needs DTensor ops this torch lacks: {e}"[:200])
    assert out[0].grad_fn.fused is False


def test_a_wrapper_subclass_base_keeps_the_base_then_lora_order(monkeypatch):
    # TorchAO's NF4Tensor reports bf16 over packed storage and implements mm but not addmm_: with no quant
    # state it must not qualify for the in-place accumulate.
    nf4 = pytest.importorskip("torchao.quantization.quantize_.workflows.nf4.nf4_tensor")
    _set(monkeypatch, True)
    g = torch.Generator().manual_seed(0)
    W = nf4.to_nf4(torch.randn(128, 128, generator = g).to(torch.bfloat16))
    X = torch.randn(8, 128, generator = g).to(torch.bfloat16)
    A = (torch.randn(4, 128, generator = g) * 0.1).to(torch.bfloat16)
    B = (torch.randn(128, 4, generator = g) * 0.1).to(torch.bfloat16)
    assert not kernel_utils._lora_fused_ok(W, None, torch.bfloat16)
    got = kernel_utils.matmul_lora(X, W, None, A, B, 0.5)
    _set(monkeypatch, False)
    torch.testing.assert_close(got, kernel_utils.matmul_lora(X, W, None, A, B, 0.5), rtol = 0, atol = 0)


@pytest.mark.parametrize("layout", ["csr", "csc", "bsr"])
def test_compressed_sparse_lora_a_takes_one_product_per_projection(layout):
    # A compressed sparse A cannot be concatenated for the shared X @ [A_1; A_2].t(): one product each.
    if layout == "bsr" and not torch.backends.mkl.is_available():
        # CPU dense @ block-sparse needs MKL (the ROCm wheels ship without it); main's product fails the same way.
        pytest.skip("CPU dense @ block-sparse needs MKL")
    g = torch.Generator().manual_seed(0)
    X = torch.randn(4, 6, generator = g)
    dense = [torch.randn(3, 6, generator = g) for _ in range(2)]
    to = {
        "csr": lambda t: t.to_sparse_csr(),
        "csc": lambda t: t.to_sparse_csc(),
        "bsr": lambda t: t.to_sparse_bsr((3, 3)),
    }[layout]
    views = fast_lora._shared_XA(X, [to(t) for t in dense], torch.float32)
    for v, t in zip(views, dense):
        v = v.to_dense() if v.layout != torch.strided else v
        torch.testing.assert_close(v, X @ t.t(), rtol = 1e-5, atol = 1e-5)


def test_a_caller_out_sharing_the_quant_state_is_written_after_dequantizing(monkeypatch):
    # The caller's buffer may share storage with the quant state (absmax): the weight is dequantized from
    # it before the rank-r term is written into the buffer.
    _set(monkeypatch, True)

    class State:
        pass

    def dequant(
        W,
        W_quant,
        use_global_buffer = True,
        dtype = None,
    ):
        return (W * W_quant.absmax.sum()).to(dtype)

    monkeypatch.setattr(kernel_utils, "fast_dequantize", dequant)
    monkeypatch.setattr(kernel_utils, "_quant_state_dtype", lambda q: torch.float32)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2, 8, generator = g)
    W = torch.randn(4, 8, generator = g)
    A, B = torch.randn(2, 8, generator = g), torch.randn(4, 2, generator = g)

    def run(shared):
        state = State()
        buf = torch.empty(2, 4)
        state.absmax = buf.view(-1)[:3] if shared else torch.empty(3)
        state.absmax.copy_(torch.tensor([0.5, 0.25, 0.25]))
        return kernel_utils.matmul_lora(X, W, state, A, B, 0.5, out = buf).clone()

    torch.testing.assert_close(run(True), run(False), rtol = 0, atol = 0)
