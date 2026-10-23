# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""fp16 LoRA on block-FP8: dequant follows the activation dtype (was bf16 -> `Half != BFloat16`)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")

BLOCK = 128
HIDDEN, INTER, RANK, TOKENS = 256, 384, 8, 16


def _fp8_weight(out_f, in_f, seed):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    W = torch.randn(out_f, in_f, generator = g, device = "cuda") * 0.05
    Wb = W.view(out_f // BLOCK, BLOCK, in_f // BLOCK, BLOCK)
    s = Wb.abs().amax(dim = (1, 3)) / 448.0
    q = (Wb / s[:, None, :, None]).view(out_f, in_f).to(torch.float8_e4m3fn)
    return q.contiguous(), s.float().contiguous()


def _exact(W, s):
    return (W.float().view(W.shape[0] // BLOCK, BLOCK, -1, BLOCK) * s[:, None, :, None]).view(
        W.shape
    )


def _lora(out_f, in_f, seed, dtype):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    A = (torch.randn(RANK, in_f, generator = g, device = "cuda") * 0.05).to(dtype)
    B = (torch.randn(out_f, RANK, generator = g, device = "cuda") * 0.05).to(dtype)
    return A.requires_grad_(), B.requires_grad_()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fast_dequantize_fp8_follows_dtype(dtype):
    from unsloth.kernels.utils import fast_dequantize

    W, s = _fp8_weight(INTER, HIDDEN, 0)
    out = fast_dequantize(W, s, dtype = dtype)
    assert out.dtype == dtype
    assert torch.equal(out, _exact(W, s).to(dtype))
    out_t = fast_dequantize(W.t(), s, dtype = dtype)
    assert out_t.dtype == dtype and torch.equal(out_t, out.t())


def test_fast_dequantize_fp8_default_stays_bf16():
    from unsloth.kernels.utils import fast_dequantize

    W, s = _fp8_weight(INTER, HIDDEN, 0)
    a = fast_dequantize(W, s)
    assert a.dtype == torch.bfloat16
    assert torch.equal(a, fast_dequantize(W, s, dtype = torch.bfloat16))


def _x(dtype):
    g = torch.Generator(device = "cuda").manual_seed(99)
    return (
        (torch.randn(1, TOKENS, HIDDEN, generator = g, device = "cuda") * 0.5)
        .to(dtype)
        .requires_grad_()
    )


def _run_lora_w(dtype):
    from unsloth.kernels import fast_lora

    W, s = _fp8_weight(INTER, HIDDEN, 1)
    A, B = _lora(INTER, HIDDEN, 2, dtype)
    X = _x(dtype)
    Y = fast_lora.LoRA_W.apply(X, W, s, A, B, 2.0)
    Y.float().square().sum().backward()
    return Y, X.grad, A.grad, B.grad, W, s, A, B, X


def _run_mlp(dtype):
    from unsloth.kernels import fast_lora
    from unsloth.kernels.swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel

    gW, gs = _fp8_weight(INTER, HIDDEN, 3)
    uW, us = _fp8_weight(INTER, HIDDEN, 4)
    dW, ds = _fp8_weight(HIDDEN, INTER, 5)
    gA, gB = _lora(INTER, HIDDEN, 6, dtype)
    uA, uB = _lora(INTER, HIDDEN, 7, dtype)
    dA, dB = _lora(HIDDEN, INTER, 8, dtype)
    X = _x(dtype)
    Y = fast_lora.LoRA_MLP.apply(
        X, gW, gs, gA, gB, 2.0, uW, us, uA, uB, 2.0, dW, ds, dA, dB, 2.0,
        swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel, False,
    )  # fmt: skip
    Y.float().square().sum().backward()
    return Y, X.grad, gA.grad, uB.grad, dA.grad


def _run_qkv(dtype):
    from unsloth.kernels import fast_lora

    Ws = [_fp8_weight(n, HIDDEN, 10 + i) for i, n in enumerate((HIDDEN, BLOCK, BLOCK))]
    AB = [_lora(n, HIDDEN, 20 + i, dtype) for i, n in enumerate((HIDDEN, BLOCK, BLOCK))]
    X = _x(dtype)
    args = []
    for (W, s), (A, B) in zip(Ws, AB):
        args += [W, s, A, B, 2.0]
    Q, K, V = fast_lora.LoRA_QKV.apply(X, *args, False)
    (Q.float().square().sum() + K.float().square().sum() + V.float().square().sum()).backward()
    return Q, K, V, X.grad, AB[0][0].grad, AB[2][1].grad


def test_lora_w_fp16_fp8_backward():
    Y, dX, dA, dB, W, s, A, B, _ = _run_lora_w(torch.float16)
    assert Y.dtype == dX.dtype == torch.float16
    Weff = _exact(W, s) + 2.0 * B.detach().float() @ A.detach().float()
    # Forward quantizes X to FP8, so dY = 2 * Y from the real output.
    torch.testing.assert_close(dX.float(), (2 * Y.detach().float()) @ Weff, rtol = 2e-2, atol = 2e-2)
    assert torch.isfinite(dA).all() and torch.isfinite(dB).all()


def test_lora_mlp_fp16_fp8_backward():
    outs = _run_mlp(torch.float16)
    assert outs[1].dtype == torch.float16
    assert all(torch.isfinite(t).all() for t in outs)


def test_lora_qkv_fp16_fp8_backward():
    outs = _run_qkv(torch.float16)
    assert outs[3].dtype == torch.float16
    assert all(torch.isfinite(t).all() for t in outs)


@pytest.mark.parametrize("runner", [_run_lora_w, _run_mlp, _run_qkv])
def test_bf16_fp8_backward_matches_bf16_dequant(runner, monkeypatch):
    # bf16 must stay bit-identical to the old no-dtype (bf16 default) path.
    from unsloth.kernels import fast_lora, fp8

    new = [
        t.detach().clone()
        for t in runner(torch.bfloat16)
        if t.dtype in (torch.bfloat16, torch.float32)
    ]
    real = fast_lora.fast_dequantize

    def old_fast_dequantize(
        W,
        q = None,
        *a,
        dtype = None,
        **k,
    ):
        if W.dtype == torch.float8_e4m3fn:
            return fp8.weight_dequant(W, q)
        return real(W, q, *a, **k)

    monkeypatch.setattr(fast_lora, "fast_dequantize", old_fast_dequantize)
    old = [
        t.detach().clone()
        for t in runner(torch.bfloat16)
        if t.dtype in (torch.bfloat16, torch.float32)
    ]
    assert len(new) == len(old)
    for a, b in zip(new, old):
        assert a.dtype == b.dtype and torch.equal(a, b)
