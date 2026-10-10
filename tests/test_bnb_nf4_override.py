# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes NF4 Linear4bit.forward on Unsloth's NF4 kernels (unsloth/kernels/bnb_override.py).

Training (dequantize + F.linear, and dX) must equal bitsandbytes bit for bit, eager and compiled;
single-row decode uses the GEMV, judged against fp64 like bitsandbytes' own GEMV; every unsupported
case must run the original forward.
"""

import os
import subprocess
import sys

import pytest
import torch

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

pytestmark = pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")

if torch.cuda.is_available():
    import unsloth  # noqa: F401  (applies zoo's Linear4bit patch first, like the product)
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_f
    from unsloth.kernels import bnb_override as O
    from unsloth.kernels import utils as U

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# _eligible rejects HIP: tests of the override's own numerics or routing would assert it ran.
needs_override = pytest.mark.skipif(
    torch.version.hip is not None, reason = "the override never runs on ROCm"
)


@pytest.fixture(autouse = True)
def _installed(monkeypatch):
    monkeypatch.setattr(
        O, "_LINEAR", "1"
    )  # numerics hold on every GPU; the default gate is tested apart
    O.uninstall_bnb_nf4_override()
    assert O.install_bnb_nf4_override()
    monkeypatch.setattr(O, "_BNB_FUSED", False)
    yield
    O.uninstall_bnb_nf4_override()


def _layer(
    N,
    K,
    dtype,
    nested = True,
    bias = False,
    quant_type = "nf4",
    blocksize = 64,
    seed = 0,
):
    torch.manual_seed(seed)
    ref = torch.nn.Linear(K, N, bias = bias, dtype = dtype)
    with torch.no_grad():
        ref.weight.normal_()  # unit scale exposes rounding-order drift
        if bias:
            ref.bias.normal_()
    lin = bnb.nn.Linear4bit(K, N, bias = bias, compute_dtype = dtype, compress_statistics = nested,
                            quant_type = quant_type, quant_storage = torch.uint8)  # fmt: skip
    lin.weight = bnb.nn.Params4bit(ref.weight.data.clone(), requires_grad = False, compress_statistics = nested,
                                   quant_type = quant_type, blocksize = blocksize)  # fmt: skip
    if bias:
        lin.bias = torch.nn.Parameter(ref.bias.data.clone(), requires_grad = False)
    lin = lin.cuda()
    assert lin.weight.quant_state.dtype is dtype
    return lin


def _call(lin, x, ours):
    if ours:
        return O._nf4_forward(lin, x)
    return O._FALLBACK["forward"](lin, x)


def _bits(a, b):
    return torch.equal(a.view(torch.uint8) if a.dtype != torch.float32 else a.view(torch.int32),
                       b.view(torch.uint8) if b.dtype != torch.float32 else b.view(torch.int32))  # fmt: skip


def _count_fallback(monkeypatch):
    calls = []
    original = O._FALLBACK["forward"]

    def counted(self, x):
        calls.append(1)
        return original(self, x)

    monkeypatch.setitem(O._FALLBACK, "forward", counted)
    return calls, original


@needs_override
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("bias", [False, True])
def test_train_forward_backward_bit_exact(dtype, nested, bias, monkeypatch):
    lin = _layer(1000, 576, dtype, nested = nested, bias = bias)
    # M = 2 * 1025 > 1536 is past every bitsandbytes fused-kernel range: bnb is dequantize + F.linear.
    x0 = torch.randn(2, 1025, 576, device = "cuda", dtype = dtype)
    dy = torch.randn(2, 1025, 1000, device = "cuda", dtype = dtype)
    _call(lin, x0[:, :8], False)  # bitsandbytes sets the compute dtype on the first call
    xs = []
    outs = []
    for ours in (False, True):
        x = x0.clone().requires_grad_()
        calls, _ = _count_fallback(monkeypatch) if ours else ([], None)
        y = _call(lin, x, ours)
        y.backward(dy)
        outs.append(y.detach())
        xs.append(x.grad)
        if ours:
            assert not calls, "the override fell back to bitsandbytes"
    assert outs[1].dtype is outs[0].dtype
    assert _bits(outs[1], outs[0]), "forward differs from bitsandbytes"
    assert _bits(xs[1], xs[0]), "dX differs from bitsandbytes"
    # And equal to bitsandbytes' own dequantize + F.linear at shapes its fused kernel may take.
    x = x0[:1, :9]
    W = bnb_f.dequantize_4bit(lin.weight.data, lin.weight.quant_state).to(dtype)
    b = None if lin.bias is None else lin.bias.to(dtype)
    assert _bits(_call(lin, x, True), torch.nn.functional.linear(x, W, b))


@needs_override
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_gemv_accuracy(dtype):
    lin = _layer(2048, 1024, dtype)
    _call(lin, torch.randn(1, 4, 1024, device = "cuda", dtype = dtype), False)
    W = bnb_f.dequantize_4bit(lin.weight.data, lin.weight.quant_state).double()
    worst = 0.0
    for seed in range(16):
        torch.manual_seed(seed)
        x = torch.randn(1, 1, 1024, device = "cuda", dtype = dtype)
        ref = x.double() @ W.t()
        with torch.no_grad():
            ours = _call(lin, x, True)
            lib = _call(lin, x, False)
        assert ours.shape == lib.shape == (1, 1, 2048) and ours.dtype is dtype
        e_ours = (ours.double() - ref).abs().max().item()
        e_lib = (lib.double() - ref).abs().max().item()
        ulp = 2 * torch.finfo(dtype).eps * ref.abs().max().item()
        assert e_ours <= e_lib + ulp, (seed, e_ours, e_lib)
        worst = max(worst, e_ours)
    assert worst > 0


def test_mixed_dtype_input_casts_like_bitsandbytes():
    # fp32 activations: bnb switches compute dtype to fp32, which no longer matches the quant state.
    lin = _layer(256, 128, torch.bfloat16)
    x = torch.randn(4, 9, 128, device = "cuda", dtype = torch.float32)
    a = _call(lin, x, False)
    b = _call(lin, x, True)
    assert b.dtype is torch.float32 and _bits(a, b)
    # bf16 layer fed fp16 activations: compute dtype stays bf16, output comes back as fp16.
    lin = _layer(256, 128, torch.bfloat16, seed = 1)
    x = torch.randn(4, 400, 128, device = "cuda", dtype = torch.float16)
    a = _call(lin, x, False)
    b = _call(lin, x, True)
    assert b.dtype is torch.float16 and _bits(a, b)


@pytest.mark.parametrize(
    "case",
    [
        "fp4",
        "k_not_multiple",
        "compute_dtype_mismatch",
        "cpu",
        "no_quant_state",
        "nested_blocksize",
        "empty",
    ],
)
def test_unsupported_cases_fall_back(case, monkeypatch):
    dtype = torch.bfloat16
    if case == "fp4":
        lin = _layer(256, 128, dtype, quant_type = "fp4")
    elif case == "k_not_multiple":
        lin = _layer(256, 96, dtype)
    else:
        lin = _layer(256, 128, dtype)
    K = lin.in_features
    x = torch.randn(3, 50, K, device = "cuda", dtype = dtype)
    _call(lin, x, False)
    if case == "compute_dtype_mismatch":
        lin.weight.quant_state.dtype = torch.float16
    elif case == "no_quant_state":
        assert not O._eligible(lin, x, lin.weight, None)
        return
    elif case == "cpu":
        assert not O._eligible(lin, x.cpu(), lin.weight, lin.weight.quant_state)
        return
    elif case == "nested_blocksize":
        lin.weight.quant_state.state2.blocksize = 128
        assert not O._eligible(lin, x, lin.weight, lin.weight.quant_state)
        return
    elif case == "empty":
        x = x[:, :0]
    calls, _ = _count_fallback(monkeypatch)
    _call(lin, x, True)
    assert calls, f"{case} must run the original forward"


def test_bias_with_grad_falls_back(monkeypatch):
    lin = _layer(256, 128, torch.bfloat16, bias = True)
    lin.bias.requires_grad_(True)
    x = torch.randn(2, 40, 128, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    _call(lin, x, False)
    calls, _ = _count_fallback(monkeypatch)
    _call(lin, x, True).sum().backward()
    assert calls and lin.bias.grad is not None


@needs_override
@pytest.mark.skipif(not has_real_cuda() or not O._TRACE, reason = "traced from torch 2.11")
@pytest.mark.parametrize("dtype", DTYPES)
def test_compiled_no_breaks_bit_exact(dtype):
    import torch._dynamo.utils as du

    if dtype is torch.bfloat16 and not O._TRACE_BF16:
        pytest.skip("bf16 traced only when every GPU is sm80+")
    lin = _layer(512, 256, dtype, bias = True)
    x0 = torch.randn(2, 70, 256, device = "cuda", dtype = dtype)
    dy = torch.randn(2, 70, 512, device = "cuda", dtype = dtype)
    _call(lin, x0, False)

    def f(x):
        return _call(lin, x, True) * 2

    torch._dynamo.reset()
    du.counters.clear()
    cf = torch.compile(f)
    res = []
    for fn in (f, cf):
        x = x0.clone().requires_grad_()
        y = fn(x)
        y.backward(dy)
        res.append((y.detach(), x.grad))
    with torch.no_grad():
        res.append((cf(x0), None))
        res.append((cf(x0[:1, :1]), None))
        ref_dec = f(x0[:1, :1])
    assert int(sum(du.counters["graph_break"].values())) == 0, dict(du.counters["graph_break"])
    ref = res[0]
    if dtype is torch.float32:
        # Inductor splits fp32 addmm with pointwise users into mm + bias add (unfuse_bias_add_to_pointwise),
        # which rounds unlike cuBLAS' bias epilogue on some GPUs: compare to compiled plain F.linear.
        W = bnb_f.dequantize_4bit(lin.weight.data, lin.weight.quant_state).to(dtype)

        def stock(x):
            return torch.nn.functional.linear(x, W, lin.bias.to(dtype)) * 2

        torch._dynamo.reset()
        cs = torch.compile(stock)
        x = x0.clone().requires_grad_()
        y = cs(x)
        y.backward(dy)
        ref = (y.detach(), x.grad)
        with torch.no_grad():
            ref_ng = cs(x0)
    else:
        ref_ng = ref[0]
    assert _bits(res[1][0], ref[0]) and _bits(res[1][1], ref[1])
    assert _bits(res[2][0], ref_ng)
    if dtype is not torch.float32:
        # Compiled GEMV is the same Triton kernel; Inductor may fuse its epilogue, so compare by error.
        assert (res[3][0].float() - ref_dec.float()).abs().max().item() <= 4 * torch.finfo(
            dtype
        ).eps * ref_dec.abs().max().item()


def test_install_is_idempotent_and_chains():
    before = O._FALLBACK["forward"]
    assert O.install_bnb_nf4_override()
    assert O._FALLBACK["forward"] is before
    assert bnb.nn.Linear4bit.forward is O._nf4_forward


def test_default_gate():
    # bitsandbytes < 0.50 (no fused 4-bit GEMM): on for every CUDA GPU sm75+.
    for caps in (
        [(7, 5)],
        [(8, 0)],
        [(8, 9)],
        [(9, 0)],
        [(10, 0), (10, 0)],
        [(12, 0)],
        [(8, 0), (7, 5)],
    ):
        assert O._default_on(caps, False, False), caps
    # bitsandbytes >= 0.50: only where every GPU is sm100 / sm120 (measured without a regressed shape).
    for caps in ([(10, 0)], [(10, 3)], [(12, 0)], [(12, 1)], [(10, 0), (12, 0)]):
        assert O._default_on(caps, True, False), caps
    for caps in ([(7, 5)], [(8, 0)], [(8, 6)], [(8, 9)], [(9, 0)], [(10, 0), (8, 0)]):
        assert not O._default_on(caps, True, False), caps
    # Never on ROCm, without a GPU, or below sm75.
    assert not O._default_on([(9, 4)], False, True)
    assert not O._default_on([(10, 0)], True, True)
    assert not O._default_on([], False, False)
    assert not O._default_on([(7, 0)], False, False)
    assert not O._default_on([(8, 0), (6, 1)], False, False)


@needs_override
@pytest.mark.parametrize(
    "rows,pick,to_bnb",
    [
        (1, True, False),
        (2, True, True),
        (8, True, True),
        (8, False, False),
        (1536, True, True),
        (1537, True, False),
    ],
)
def test_fused_bnb_keeps_its_fused_shapes(rows, pick, to_bnb, monkeypatch):
    # bitsandbytes >= 0.50 keeps the 2..1536-row shapes its fused 4-bit GEMM takes.
    monkeypatch.setattr(O, "_BNB_FUSED", True)
    monkeypatch.setattr(O, "_BNB_FUSED_MAX_ROWS", 1536)
    asked = []
    monkeypatch.setattr(O, "_BNB_PICK", lambda dev, dtype, M, N, K: asked.append((M, N, K)) or pick)
    lin = _layer(256, 128, torch.bfloat16)
    x = torch.randn(1, rows, 128, device = "cuda", dtype = torch.bfloat16)
    ref = _call(lin, x, False)
    calls, _ = _count_fallback(monkeypatch)
    with torch.no_grad():
        y = _call(lin, x, True)
    assert bool(calls) is to_bnb
    assert bool(asked) is (1 < rows <= 1536)
    if asked:
        assert asked[-1] == (rows, 256, 128)
    if to_bnb or rows > 1536:
        assert _bits(
            y, ref
        )  # bitsandbytes itself, or past its fused range the same dequantize + F.linear
    x = x.clone().requires_grad_()
    calls.clear()
    _call(lin, x, True).sum().backward()
    assert bool(calls) is to_bnb
    assert len(asked) == (1 < rows <= 1536)  # asked once per layer and row count, then cached


def test_fused_bnb_heuristic_is_found():
    if not O._bnb_has_fused_gemm():
        pytest.skip("bitsandbytes < 0.50 has no fused 4-bit GEMM")
    pick, max_rows = O._bnb_fused_heuristic()
    assert callable(pick) and max_rows >= 1
    assert pick(0, torch.bfloat16, 2, 4096, 4096) is True  # its floor: always fused at <= 4 rows
    assert O._BNB_PICK is not None or not O._BNB_FUSED


@pytest.mark.parametrize("value,expect", [("0", False), ("1", True)])
def test_linear_env_override(value, expect):
    code = (
        "import unsloth, bitsandbytes as bnb\n"
        "from unsloth.kernels import bnb_override as O\n"
        "O.uninstall_bnb_nf4_override()\n"
        f"assert O.install_bnb_nf4_override() is {expect}\n"
        f"assert getattr(bnb.nn.Linear4bit.forward, '_unsloth_nf4_override', False) is {expect}\n"
        "print('OK')\n"
    )
    env = dict(os.environ, UNSLOTH_BNB_NF4_LINEAR = value)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, cwd = root
    )
    assert "OK" in out.stdout, out.stderr[-2000:]


@pytest.mark.skipif(torch.version.hip is None, reason = "ROCm only")
def test_rocm_always_runs_bitsandbytes(monkeypatch):
    lin = _layer(256, 128, torch.bfloat16)
    x = torch.randn(2, 1025, 128, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    ref = _call(lin, x, False)
    calls, _ = _count_fallback(monkeypatch)
    y = _call(lin, x, True)
    y.sum().backward()
    assert calls and _bits(y, ref)
    assert not O._default_on([(11, 5)], True, True)


def test_kill_switch():
    code = (
        "import os, unsloth, bitsandbytes as bnb\n"
        "from unsloth.kernels import bnb_override as O\n"
        "f = bnb.nn.Linear4bit.forward\n"
        "assert O.install_bnb_nf4_override() is False\n"
        "assert bnb.nn.Linear4bit.forward is f\n"
        "print('OK')\n"
    )
    env = dict(os.environ, UNSLOTH_BNB_TRITON = "0")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, cwd = root
    )
    assert "OK" in out.stdout, out.stderr[-2000:]


@pytest.mark.parametrize("rows", [1, 3, 7])
def test_fp32_few_rows_falls_back(rows, monkeypatch):
    lin = _layer(512, 256, torch.float32)
    x = torch.randn(1, rows, 256, device = "cuda", dtype = torch.float32)
    a = _call(lin, x, False)
    calls, _ = _count_fallback(monkeypatch)
    b = _call(lin, x, True)
    assert calls and _bits(a, b)


def _breaks(fn, *args):
    import torch._dynamo.utils as du

    torch._dynamo.reset()
    du.counters.clear()
    out = fn(*args)
    return out, int(sum(du.counters["graph_break"].values()))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_untraced_torch_defers_to_bitsandbytes_when_compiled(dtype, monkeypatch):
    # Before torch 2.11 compiled code keeps bitsandbytes' traceable forward; eager uses the kernels.
    if not O._BNB_OPS:
        pytest.skip("bitsandbytes without registered ops (0.45.5) runs one opaque call instead")
    if dtype is torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        # Inductor's _check_triton_bf16_support raises SkipFrame on sm75: the frame runs eager.
        pytest.skip("Inductor skips bf16 frames below sm80, so nothing is compiled")
    monkeypatch.setattr(O, "_TRACE", False)
    lin = _layer(512, 256, dtype)
    x = torch.randn(2, 33, 256, device = "cuda", dtype = dtype)
    ref = _call(lin, x, False)
    calls, _ = _count_fallback(monkeypatch)
    with torch.no_grad():
        y, n = _breaks(torch.compile(lambda t: lin(t)), x)
    assert n == 0 and calls and _bits(y, ref)


@needs_override
@pytest.mark.skipif(not has_real_cuda() or not O._TRACE, reason = "traced from torch 2.11")
def test_compiled_dynamic_decode_through_module_no_breaks():
    # zoo compiles with dynamic=True: the GEMV sees weight sizes as SymInts.
    dtype = torch.bfloat16 if O._TRACE_BF16 else torch.float16
    mlp = torch.nn.Sequential(_layer(1536, 256, dtype), _layer(256, 1536, dtype, seed = 1))
    for m in mlp:
        _call(m, torch.randn(1, 8, m.in_features, device = "cuda", dtype = dtype), False)

    def f(module, x):
        return module(x)

    cf = torch.compile(f, dynamic = True)
    with torch.no_grad():
        for K in (1, 1):
            x = torch.randn(1, K, 256, device = "cuda", dtype = dtype)
            y, n = _breaks(cf, mlp, x)
            assert n == 0
            ref = f(mlp, x)
            assert (y.float() - ref.float()).abs().max().item() <= 4 * torch.finfo(
                dtype
            ).eps * ref.abs().max().item()
