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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nan_rows_propagate(dtype):
    """A NaN in one row of x / dY makes that output row NaN (as the 16-bit GEMM does) and leaves the others finite."""
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
