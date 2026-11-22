# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""The NVFP4 QAT custom op backward skips the dequant + GEMM of unrequested grads, CPU only."""

from __future__ import annotations

import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None, reason = "require torchao"
)

import unsloth.models._utils as qat_utils


@pytest.fixture(scope = "module")
def op():
    # Skip only on the optional dependency; a broken op factory or kernel registration fails setup.
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            NVFP4Tensor,
            per_tensor_amax_to_scale,
        )
    except ImportError as e:
        pytest.skip(f"torchao NVFP4 internals unavailable: {e!r}")
    op = qat_utils._nvfp4_qat_linear_op()

    # No FP4 GEMM on CPU: the forward runs the same quantizer with a dequantized matmul. The real op
    # only ever sees CUDA inputs, so this CPU kernel does not touch it.
    def cpu_forward(x, weight, bias, act_per_tensor, w_per_tensor, w_swizzled, w_triton, mm_dtype):
        def q(t, per_tensor, swizzled):
            scale = None
            if per_tensor:
                scale = per_tensor_amax_to_scale(torch.max(torch.abs(t)))
            return NVFP4Tensor.to_nvfp4(t, per_tensor_scale = scale, is_swizzled_scales = swizzled)

        x_q, w_q = q(x, act_per_tensor, False), q(weight, w_per_tensor, w_swizzled)
        out = x_q.dequantize(torch.float32) @ w_q.dequantize(torch.float32).t()
        if bias is not None:
            out = out + bias.float()

        # A fresh scalar per missing scale, as the real forward does: custom op outputs must not alias.
        def pt(t):
            if t.per_tensor_scale is None:
                return x.new_ones((), dtype = torch.float32)
            return t.per_tensor_scale.float()

        return (out.to(x.dtype), x_q.qdata, x_q.scale, pt(x_q), w_q.qdata, w_q.scale, pt(w_q))

    op.register_kernel("cpu", cpu_forward)  # bound to the op object, not a hard-coded name
    return op


def _run(
    op,
    need_x,
    need_w,
    bias_grad = False,
    compiled = False,
    calls = None,
    per_tensor = True,
):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 128, generator = g, dtype = torch.bfloat16).requires_grad_(need_x)
    w = torch.randn(96, 128, generator = g, dtype = torch.bfloat16).requires_grad_(need_w)
    b = torch.randn(96, generator = g, dtype = torch.bfloat16).requires_grad_(bias_grad)
    saved = []

    def pack(t):
        saved.append(t.numel() * t.element_size())
        return t

    def f(x, w, b):
        return op(x, w, b, per_tensor, per_tensor, False, False, None)[0]

    if compiled:
        torch._dynamo.reset()
        f = torch.compile(f, fullgraph = True, backend = "aot_eager")
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = f(x, w, b)
    if calls is not None:  # count the backward only
        calls.update(mm = 0, dequantize = 0)
        calls["order"] = []
    out.float().square().mean().backward()
    return out.detach(), x.grad, w.grad, b.grad, sum(saved)


def _count(monkeypatch):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    calls = {"mm": 0, "dequantize": 0}
    mm, dequantize = torch.mm, NVFP4Tensor.dequantize

    def counting_mm(*a, **k):
        calls["mm"] += 1
        calls.setdefault("order", []).append("mm")
        return mm(*a, **k)

    def counting_dequantize(self, *a, **k):
        calls["dequantize"] += 1
        calls.setdefault("order", []).append("dequantize")
        return dequantize(self, *a, **k)

    monkeypatch.setattr(torch, "mm", counting_mm)
    monkeypatch.setattr(NVFP4Tensor, "dequantize", counting_dequantize)
    return calls


@pytest.mark.parametrize("need_x, need_w", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("bias_grad", [False, True])
def test_backward_runs_only_requested_gemms(op, monkeypatch, need_x, need_w, bias_grad):
    full = _run(op, True, True, bias_grad)
    calls = _count(monkeypatch)
    out, gx, gw, gb, saved = _run(op, need_x, need_w, bias_grad, calls = calls)
    assert (calls["mm"], calls["dequantize"]) == (need_x + need_w, need_x + need_w)
    assert torch.equal(out, full[0])
    for got, want, need in ((gx, full[1], need_x), (gw, full[2], need_w), (gb, full[3], bias_grad)):
        assert (got is not None) == need
        if need:
            assert torch.equal(got, want)
    # The frozen-weight (LoRA) case drops the saved activation payload, the no-input-grad case the weight's.
    assert (saved < full[4]) == (not (need_x and need_w))


def test_backward_skips_op_for_bias_only_grad(op, monkeypatch):
    full = _run(op, True, True, bias_grad = True)
    calls = _count(monkeypatch)
    out, gx, gw, gb, saved = _run(op, False, False, bias_grad = True, calls = calls)
    assert (calls["mm"], calls["dequantize"]) == (0, 0)
    assert torch.equal(out, full[0])
    assert gx is None and gw is None and torch.equal(gb, full[3])
    assert saved == 0


@pytest.mark.parametrize("need_x, need_w", [(True, False), (True, True)])
def test_backward_without_per_tensor_scales(op, need_x, need_w):
    # Both per-tensor scales absent: the forward returns two placeholder scalars, which must not alias.
    full = _run(op, True, True, per_tensor = False)
    _, gx, gw, _, _ = _run(op, need_x, need_w, per_tensor = False)
    assert torch.equal(gx, full[1])
    assert (gw is not None) == need_w and (not need_w or torch.equal(gw, full[2]))


def test_backward_dequantizes_before_any_gemm(op, monkeypatch):
    # Full fine-tuning: both operands dequantize before the first GEMM (the pre-skip order), so neither dX
    # nor the dense weight is live while the activation dequantizes; failing before: dX ran first.
    calls = _count(monkeypatch)
    _run(op, True, True, calls = calls)
    assert calls["order"] == ["dequantize", "dequantize", "mm", "mm"]


def test_backward_materializes_no_payload_grads(op):
    # The six payload outputs are non-differentiable: autograd must not build zero grads for them.
    from torch.utils._python_dispatch import TorchDispatchMode

    zeros = []

    class Zeros(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            if func._schema.name in ("aten::zeros", "aten::zeros_like", "aten::new_zeros"):
                zeros.append(func)
            return func(*args, **(kwargs or {}))

    x = torch.randn(64, 128, dtype = torch.bfloat16, requires_grad = True)
    w = torch.randn(96, 128, dtype = torch.bfloat16)
    out = op(x, w, None, True, True, False, False, None)[0]
    with Zeros():
        out.float().sum().backward()
    assert zeros == [] and x.grad is not None


def test_backward_eager_accepts_no_requested_grads(op):
    # Direct dispatch with both grads omitted: two distinct empty outputs (aliased outputs raise).
    backward = torch.ops.unsloth.nvfp4_qat_linear_v2_backward.default
    grad = torch.empty(64, 96, dtype = torch.bfloat16)
    args = (
        grad,
        *(grad.new_empty(0) for _ in range(6)),
        grad.dtype,
        grad.dtype,
        True,
        True,
        False,
        None,
        False,
        False,
    )
    torch.library.opcheck(backward, args, test_utils = ("test_schema",))
    gx, gw = backward(*args)
    assert gx.shape == gw.shape == (0,) and gx.dtype == gw.dtype == grad.dtype


@pytest.mark.parametrize(
    "need_x, need_w, bias_grad",
    [(True, False, False), (False, True, False), (True, True, False), (False, False, True)],
)
def test_backward_skip_under_compile(op, need_x, need_w, bias_grad):
    eager = _run(op, need_x, need_w, bias_grad = bias_grad)
    compiled = _run(op, need_x, need_w, bias_grad = bias_grad, compiled = True)
    for a, b in zip(eager[:4], compiled[:4]):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a, b)


def test_backward_fake_matches_requested_grads(op):
    from torch._subclasses.fake_tensor import FakeTensorMode
    backward = torch.ops.unsloth.nvfp4_qat_linear_v2_backward
    with FakeTensorMode():
        grad = torch.empty(64, 96, dtype = torch.bfloat16)
        x_q = (
            torch.empty(64, 64, dtype = torch.uint8),
            torch.empty(64, 8, dtype = torch.float8_e4m3fn),
        )
        w_q = (
            torch.empty(96, 64, dtype = torch.uint8),
            torch.empty(96, 8, dtype = torch.float8_e4m3fn),
        )
        one, empty = torch.ones(()), torch.empty(0)
        for need_x, need_w in ((True, False), (False, True), (True, True), (False, False)):
            xs = (*x_q, one) if need_w else (empty, empty, empty)
            ws = (*w_q, one) if need_x else (empty, empty, empty)
            dtypes = (torch.bfloat16, torch.bfloat16)
            gx, gw = backward(grad, *xs, *ws, *dtypes, True, True, False, None, need_x, need_w)
            assert gx.shape == ((64, 128) if need_x else (0,))
            assert gw.shape == ((96, 128) if need_w else (0,))
