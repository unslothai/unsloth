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
    # In-place accumulation is not autocast: an fp32 base weight with bf16 activations (fp32 weights under
    # autocast, as in fast_linear_forward's multi-token path and LoRA_W) takes the old order instead of
    # failing on the dtype mismatch. (main's LoRA_QKV / LoRA_MLP backward already needs matching dtypes.)
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
