# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Rotated INT8 frozen-base LoRA GEMMs (kernels/int8_lora.py) against fp64 references, on a CUDA GPU."""

import math

import pytest
import torch

from real_accelerator import has_real_cuda

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not has_real_cuda()
        or torch.version.hip is not None
        or torch.cuda.get_device_capability() < (8, 0),
        reason = "needs an NVIDIA GPU with int8 tensor cores",
    ),
]

IL = pytest.importorskip("unsloth.kernels.int8_lora")
if IL.triton is None:
    pytest.skip("needs Triton", allow_module_level = True)


def _h(device):
    return IL._hadamard_pm1(device, torch.float64) / math.sqrt(IL.GROUP)


def _rot(t, H):
    n, k = t.shape
    return (t.double().view(n, k // IL.GROUP, IL.GROUP) @ H).view(n, k)


def _rowq(t):
    amax = t.abs().amax(1).clamp_min(1e-30)
    return torch.round(t * (127.0 / amax)[:, None]).clamp(-127, 127), amax / 127.0


def _inputs(
    M,
    N,
    K,
    dtype,
    seed = 0,
):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    x = torch.randn((M, K), device = "cuda", generator = g)
    cols = torch.randperm(K, device = "cuda", generator = g)[: max(1, K // 64)]
    x[:, cols] *= 20.0  # outlier channels, as in post-norm LLM activations
    w = torch.randn((N, K), device = "cuda", generator = g) * 0.02
    dy = (
        torch.randn((M, N), device = "cuda", generator = g)
        * torch.rand((M, 1), device = "cuda", generator = g)
        * 1e-3
    )
    return x.to(dtype), w.to(dtype), dy.to(dtype)


def _emulate_fwd(x, wq, ws, H):
    # The algorithm in fp64: rotated per-token codes of x, the weight's codes and scales, one fp64 GEMM.
    xq, xs = _rowq(_rot(x, H))
    return (xq @ wq.double().t()) * xs[:, None] * ws.double()[None, :]


def _emulate_dgrad(dy, wq, ws, H):
    dq, ds = _rowq(dy.double() * ws.double()[None, :])
    rot = (dq @ wq.double()) * ds[:, None]
    n, k = rot.shape
    return (rot.view(n, k // IL.GROUP, IL.GROUP) @ H.t()).view(n, k)


def _rel(a, b):
    return float((a.double() - b).norm() / b.norm())


SHAPES = [
    (1024, 512, 2048),
    (2048, 1024, 4096),
    (1000, 768, 1024),
    (4096, 4096, 4096),
    (256, 2048, 768 + 128 * 2),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_forward_and_dgrad_match_fp64(M, N, K, dtype):
    x, w, dy = _inputs(M, N, K, dtype)
    H = _h(x.device)
    wq, ws = IL.quantize_weight(w)
    IL.clear_int8_lora_cache()
    y = IL.int8_lora_fwd(x, wq, ws)
    dx = IL.int8_lora_dgrad(dy, wq, ws, None, None)
    dx_cached = IL.int8_lora_dgrad(dy, wq, ws, wq.t().contiguous(), None)
    assert torch.equal(dx, dx_cached)
    assert y.dtype == dtype and dx.dtype == dtype and y.shape == (M, N) and dx.shape == (M, K)
    # Same algorithm in fp64: only code ties and the fp32 epilogue differ.
    emu_y = _emulate_fwd(x, wq, ws, H)
    emu_dx = _emulate_dgrad(dy, wq, ws, H)
    assert _rel(y, emu_y) < 4e-3, _rel(y, emu_y)
    assert _rel(dx, emu_dx) < 4e-3, _rel(dx, emu_dx)
    # Against the exact 16-bit linear: the quantization error itself, bounded.
    ref_y = x.double() @ w.double().t()
    ref_dx = dy.double() @ w.double()
    assert _rel(y, ref_y) < 0.03, _rel(y, ref_y)
    assert _rel(dx, ref_dx) < 0.03, _rel(dx, ref_dx)
    # The dequantized weight is the matrix both GEMMs approximate.
    wd = IL.dequantize_weight(wq, ws, torch.float32)
    assert _rel(wd, w.double()) < 0.01


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_epilogue_delta_is_added_once(dtype):
    M, N, K = 1024, 1024, 2048
    x, w, dy = _inputs(M, N, K, dtype, seed = 1)
    wq, ws = IL.quantize_weight(w)
    delta = torch.randn((M, N), device = "cuda", dtype = dtype)
    IL.clear_int8_lora_cache()
    plain = IL.int8_lora_fwd(x, wq, ws)
    fused = IL.int8_lora_fwd(x, wq, ws, delta)
    assert _rel(fused, plain.double() + delta.double()) < 5e-3
    ddelta = torch.randn((M, K), device = "cuda", dtype = dtype) * 1e-3
    plain = IL.int8_lora_dgrad(dy, wq, ws, None, None)
    fused = IL.int8_lora_dgrad(dy, wq, ws, None, ddelta)
    assert _rel(fused, plain.double() + ddelta.double()) < 5e-3


def test_negative_controls_fail():
    """Flip checks: an unrotated activation, or a dgrad without the epilogue un-rotation, must be caught."""
    M, N, K = 1024, 1024, 2048
    x, w, dy = _inputs(M, N, K, torch.bfloat16, seed = 2)
    H = _h(x.device)
    wq, ws = IL.quantize_weight(w)
    ref_y = x.double() @ w.double().t()
    xq, xs = _rowq(x.double())  # no rotation of x
    wrong = (xq @ wq.double().t()) * xs[:, None] * ws.double()[None, :]
    assert _rel(wrong, ref_y) > 0.5
    dq, ds = _rowq(dy.double() * ws.double()[None, :])
    wrong_dx = (dq @ wq.double()) * ds[:, None]  # rotated space, never un-rotated
    assert _rel(wrong_dx, dy.double() @ w.double()) > 0.5
    # and the kernel output differs from those wrong answers by the same margin
    IL.clear_int8_lora_cache()
    assert _rel(IL.int8_lora_fwd(x, wq, ws), wrong) > 0.5


def test_activation_quant_is_shared_and_invalidated():
    M, K = 2048, 4096
    x = torch.randn((2, M // 2, K), device = "cuda", dtype = torch.bfloat16)
    IL.clear_int8_lora_cache()
    IL.reset_int8_lora_stats()
    a = IL.quantize_x(x.view(-1, K))
    b = IL.quantize_x(x.view(-1, K))  # q / k / v: new views of the same input
    assert IL.INT8_LORA_STATS["x_quant"] == 2 and IL.INT8_LORA_STATS["x_quant_shared"] == 1
    assert a[0] is b[0]
    x.mul_(2.0)  # an in-place write bumps the version: must requantize
    c = IL.quantize_x(x.view(-1, K))
    assert IL.INT8_LORA_STATS["x_quant_shared"] == 1
    assert torch.allclose(c[1], a[1] * 2.0, rtol = 1e-6)
    y = x.view(-1, K).clone()  # equal values, different tensor: requantized, not shared
    IL.quantize_x(y)
    assert IL.INT8_LORA_STATS["x_quant_shared"] == 1
    with torch.inference_mode():
        z = torch.randn((M, K), device = "cuda", dtype = torch.bfloat16)
        IL.quantize_x(z)
        IL.quantize_x(z)
    assert IL.INT8_LORA_STATS["x_quant_shared"] == 1
    IL.clear_int8_lora_cache()


def test_state_routes_and_row_gate(monkeypatch):
    lin = torch.nn.Linear(
        2048, 1024, bias = False, device = "cuda", dtype = torch.bfloat16
    ).requires_grad_(False)
    state = IL.attach_int8_lora_base(lin, offload = False)
    assert isinstance(state, IL.Int8LoRABaseState) and state.wq.dtype == torch.int8
    IL.reset_int8_lora_stats()
    x = torch.randn((4, 512, 2048), device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    y = lin(x)
    y.float().square().mean().backward()
    assert IL.INT8_LORA_STATS["fwd"] == 1 and IL.INT8_LORA_STATS["dgrad"] == 1
    ref = x.detach().double() @ lin.weight.double().t()
    assert _rel(y.detach(), ref.view(4, 512, -1)) < 0.03
    # Below the row floor the 16-bit GEMM runs (kill-switch arm: 0 int8 calls).
    monkeypatch.setattr(IL, "_MIN_ROWS", 10**9)
    IL.reset_int8_lora_stats()
    y2 = lin(x)
    assert IL.INT8_LORA_STATS["fwd"] == 0 and IL.INT8_LORA_STATS["fwd_16bit"] == 1
    assert torch.equal(y2, torch.nn.functional.linear(x, lin.weight))
    IL.clear_int8_lora_cache()


def test_offloaded_master_dense_matches_dequant():
    lin = torch.nn.Linear(
        1024, 512, bias = False, device = "cuda", dtype = torch.bfloat16
    ).requires_grad_(False)
    w = lin.weight.detach().clone()
    state = IL.attach_int8_lora_base(lin, offload = True)
    assert state.offloaded and lin.weight.device.type == "cpu"
    dense = state._dense(None)
    assert dense.device.type == "cuda" and _rel(dense.float(), w.double()) < 0.01
    state.restore(lin)
    assert torch.equal(lin.weight, w)


def test_compiled_fullgraph_no_breaks():
    from torch._dynamo.utils import counters

    lin = torch.nn.Linear(
        2048, 1024, bias = False, device = "cuda", dtype = torch.bfloat16
    ).requires_grad_(False)
    state = IL.attach_int8_lora_base(lin, offload = False)

    def f(x):
        y = state.matmul(x, lin.weight)
        return state.dgrad(y.to(x.dtype), lin.weight)

    x = torch.randn((2048, 2048), device = "cuda", dtype = torch.bfloat16)
    IL.clear_int8_lora_cache()
    eager = f(x)
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(f, fullgraph = True)
    out = compiled(x)
    out = compiled(x)
    assert sum(counters["graph_break"].values()) == 0
    assert torch.equal(out, eager)


_ARCHS = {"own": None, "sm80": (8, 0), "sm89": (8, 9)}


def _use_arch(monkeypatch, name):
    """Run the launch configuration of another GPU class on this GPU (its kernels and tiles run anywhere)."""
    cap = _ARCHS[name]
    if cap is not None:
        monkeypatch.setitem(IL._ARCH, torch.cuda.current_device(), IL._arch_for(cap))


@pytest.mark.parametrize("arch", list(_ARCHS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nan_rows_propagate(monkeypatch, dtype, arch):
    """A NaN in one row of x / dY makes that output row NaN (as the 16-bit GEMM does) and leaves the others finite."""
    _use_arch(monkeypatch, arch)
    M, N, K = 512, 256, 512
    x, w, dy = _inputs(M, N, K, dtype, seed = 3)
    wq, ws = IL.quantize_weight(w)
    x[7, 11] = float("nan")
    dy[9, 3] = float("nan")
    IL.clear_int8_lora_cache()
    y = IL.int8_lora_fwd(x, wq, ws)
    dx = IL.int8_lora_dgrad(dy, wq, ws, None, None)
    assert torch.isnan(y[7]).all() and torch.isfinite(torch.cat([y[:7], y[8:]])).all()
    assert torch.isnan(dx[9]).all() and torch.isfinite(torch.cat([dx[:9], dx[10:]])).all()


def _same(a, b):
    return torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(
        a.nan_to_num(), b.nan_to_num()
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "M,N,K", [(2048, 3072, 4096), (1000, 1000, 1024), (512, 20480, 512), (300, 512, 20480)]
)
def test_arch_configurations_are_bit_identical(monkeypatch, M, N, K, dtype):
    """sm_80 / sm_89 configurations match this GPU's bits (N or K = 20480 keeps the two-pass kernels)."""
    x, w, dy = _inputs(M, N, K, dtype, seed = 5)
    x[5, 9] = float("nan")
    dy[3, 7] = float("nan")
    wq, ws = IL.quantize_weight(w)
    delta = torch.randn((M, K), device = "cuda", dtype = dtype) * 1e-3
    outs = {}
    for arch in _ARCHS:
        monkeypatch.delitem(IL._ARCH, torch.cuda.current_device(), raising = False)
        _use_arch(monkeypatch, arch)
        IL.clear_int8_lora_cache()
        xq, xs = IL.quantize_rotated_rows(x)
        dq, ds = IL.quantize_rows_scaled(dy, ws)
        outs[arch] = (
            xq,
            xs,
            dq,
            ds,
            IL.int8_lora_fwd(x, wq, ws),
            IL.int8_lora_dgrad(dy, wq, ws, None, None),
            IL.int8_lora_dgrad(dy, wq, ws, None, delta),
        )
    for arch in ("sm80", "sm89"):
        for a, b in zip(outs["own"], outs[arch]):
            assert _same(a, b), arch


def test_one_pass_kernels_stay_within_two_variants(monkeypatch):
    """Every Qwen3 / Llama projection width compiles at most 2 variants of each one-pass quantization kernel."""
    if not hasattr(IL._int8_rowq1_kernel, "device_caches"):
        pytest.skip("this Triton keeps no per-device compile cache")
    monkeypatch.setitem(IL._ARCH, torch.cuda.current_device(), IL._arch_for((8, 9)))
    widths = [1024, 2048, 2560, 3072, 4096, 5120, 6144, 8192, 9728, 12288, 14336]
    before = {
        k: len(k.device_caches[torch.cuda.current_device()][0])
        for k in (IL._int8_rowq1_kernel, IL._int8_rotq1_kernel)
    }
    for n in widths:
        x = torch.randn((64, n), device = "cuda", dtype = torch.bfloat16)
        IL.quantize_rows_scaled(x, torch.ones(n, device = "cuda"))
        IL.quantize_rotated_rows(x)
    for k, n0 in before.items():
        assert len(k.device_caches[torch.cuda.current_device()][0]) - n0 <= 2, k.__name__


def test_cuda_graph_capture_requantizes_the_static_input():
    """A warm-up call must not let capture reuse its cached quantization: replay with new data in the static
    input has to give that data's result."""
    M, N, K = 2048, 1024, 2048
    x, w, _ = _inputs(M, N, K, torch.bfloat16, seed = 7)
    wq, ws = IL.quantize_weight(w)
    static = x.clone()
    IL.clear_int8_lora_cache()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        IL.int8_lora_fwd(static, wq, ws)  # warm-up fills the sharing cache
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = IL.int8_lora_fwd(static, wq, ws)
    new = _inputs(M, N, K, torch.bfloat16, seed = 8)[0]
    static.copy_(new)
    graph.replay()
    torch.cuda.synchronize()
    IL.clear_int8_lora_cache()
    assert torch.equal(out, IL.int8_lora_fwd(new, wq, ws))
    IL.clear_int8_lora_cache()


def test_offloaded_small_rows_run_int8_without_rebuilding_the_weight(monkeypatch):
    lin = torch.nn.Linear(
        2048, 1024, bias = False, device = "cuda", dtype = torch.bfloat16
    ).requires_grad_(False)
    state = IL.attach_int8_lora_base(lin, offload = True)
    assert state.offloaded

    def no_rebuild(*a, **k):
        raise AssertionError("dense weight rebuilt for a small-row call")

    monkeypatch.setattr(IL, "dequantize_weight", no_rebuild)
    IL.reset_int8_lora_stats()
    x = torch.randn((3, 16, 2048), device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    y = lin(x)  # 48 rows, far below _MIN_ROWS
    y.float().sum().backward()
    assert IL.INT8_LORA_STATS["fwd"] == 1 and IL.INT8_LORA_STATS["dgrad"] == 1
    assert IL.INT8_LORA_STATS["fwd_16bit"] == 0
    IL.clear_int8_lora_cache()


@pytest.mark.parametrize(
    "x_dtype,amp_dtype", [(torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16)]
)
def test_large_rows_follow_the_autocast_dtype(x_dtype, amp_dtype):
    """Under autocast the converted Linear returns what F.linear would: the autocast dtype."""
    lin = torch.nn.Linear(2048, 1024, bias = True, device = "cuda", dtype = x_dtype).requires_grad_(False)
    ref_lin = torch.nn.Linear(2048, 1024, bias = True, device = "cuda", dtype = x_dtype).requires_grad_(
        False
    )
    ref_lin.load_state_dict(lin.state_dict())
    IL.attach_int8_lora_base(lin, offload = False)
    x = torch.randn((2048, 2048), device = "cuda", dtype = x_dtype)
    IL.reset_int8_lora_stats()
    with torch.autocast("cuda", dtype = amp_dtype):
        y = lin(x)
        ref = ref_lin(x)
    assert IL.INT8_LORA_STATS["fwd"] == 1
    assert y.dtype == ref.dtype == amp_dtype
    assert _rel(y, ref.double()) < 0.03
    IL.clear_int8_lora_cache()


def test_small_row_fused_forward_is_counted(monkeypatch):
    lin = torch.nn.Linear(256, 128, bias = False, device = "cuda", dtype = torch.bfloat16).requires_grad_(
        False
    )
    state = IL.attach_int8_lora_base(lin, offload = False)
    X = torch.randn((16, 256), device = "cuda", dtype = torch.bfloat16)
    A = torch.randn((8, 256), device = "cuda", dtype = torch.bfloat16)
    B = torch.randn((128, 8), device = "cuda", dtype = torch.bfloat16)
    IL.reset_int8_lora_stats()
    state.lora_first(X, lin.weight, X @ A.t(), B.t(), 2.0)
    assert IL.INT8_LORA_STATS["fwd"] == 0 and IL.INT8_LORA_STATS["fwd_16bit"] == 1


def test_each_layer_device_is_gated(monkeypatch):
    """The recipe gate is applied to the layer's own GPU, not only to the current one."""
    import dataclasses
    from unsloth.precision import recipes as R, gpu_class

    lin = torch.nn.Linear(256, 128, bias = False, device = "cuda", dtype = torch.bfloat16).requires_grad_(
        False
    )
    assert IL.eligible_base(lin) is None
    cls = gpu_class(torch.cuda.get_device_capability(lin.weight.device))
    r = R.RECIPES["int8_lora_rotated"]
    gates = dict(r.gates)
    gates[cls] = R.Gate("refused", "measured slower here")
    monkeypatch.setitem(R.RECIPES, "int8_lora_rotated", dataclasses.replace(r, gates = gates))
    assert "measured slower here" in IL.eligible_base(lin)
