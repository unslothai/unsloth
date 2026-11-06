# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fused LoRA backward skips a frozen A / B's GEMMs; dX and the other gradients stay bit for bit."""

import itertools

import pytest
import torch
import unsloth  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode

from unsloth.kernels.fast_lora import LoRA_QKV, LoRA_W, _shared_lora_backward

_GEMMS = {"mm", "addmm", "addmm_", "bmm", "matmul"}


class _CountGemms(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        if func.overloadpacket.__name__ in _GEMMS:
            self.count += 1
        return func(*args, **(kwargs or {}))


def _tensors(
    n_proj,
    dtype,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(2, 5, 48, generator = g).to(dtype)
    weights, factors = [], []
    for _ in range(n_proj):
        weights.append((torch.randn(40, 48, generator = g) * 0.05).to(dtype))
        factors.append((torch.randn(8, 48, generator = g) * 0.1).to(dtype))
        factors.append((torch.randn(40, 8, generator = g) * 0.1).to(dtype))
    grads = [torch.randn(2, 5, 40, generator = g).to(dtype) for _ in range(n_proj)]
    return X, weights, factors, grads


def _run(
    cls,
    X,
    weights,
    factors,
    grads,
    trainable,
    dropout_p = None,
):
    X = X.clone().requires_grad_(True)
    factors = [f.clone().requires_grad_(t) for f, t in zip(factors, trainable)]
    scales = [0.5, 0.25, 0.75]
    if cls is LoRA_W:
        args = (X, weights[0], None, factors[0], factors[1], scales[0])
        if dropout_p is not None:
            args = args + (dropout_p,)
    else:
        args = [X]
        for i in range(3):
            args += [weights[i], None, factors[2 * i], factors[2 * i + 1], scales[0]]
        args += [False]
        if dropout_p is not None:
            args += [(dropout_p,) * 3]
    torch.manual_seed(1)  # same dropout mask in every run
    outs = cls.apply(*args)
    outs = outs if isinstance(outs, tuple) else (outs,)
    with _CountGemms() as counter:
        torch.autograd.backward(outs, grads)
    return X.grad, [f.grad for f in factors], counter.count


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dropout_p", [None, 0.25])
def test_lora_w_frozen_factor_skips_its_gemm_and_keeps_the_rest(dtype, dropout_p):
    X, weights, factors, grads = _tensors(1, dtype)
    dX_ref, (dA_ref, dB_ref), n_ref = _run(
        LoRA_W, X, weights, factors, grads, (True, True), dropout_p
    )
    for trainable in [(False, True), (True, False), (False, False)]:
        dX, (dA, dB), n = _run(LoRA_W, X, weights, factors, grads, trainable, dropout_p)
        assert torch.equal(dX, dX_ref)
        assert (dA is None) != trainable[0] and (dB is None) != trainable[1]
        if trainable[0]:
            assert torch.equal(dA, dA_ref)
        if trainable[1]:
            assert torch.equal(dB, dB_ref)
        assert n == n_ref - (0 if trainable[0] else 1) - (0 if trainable[1] else 2), (
            trainable,
            n,
            n_ref,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lora_qkv_frozen_factors_keep_the_rest_bit_for_bit(dtype):
    X, weights, factors, grads = _tensors(3, dtype)
    dX_ref, f_ref, n_ref = _run(LoRA_QKV, X, weights, factors, grads, (True,) * 6)
    for trainable in [
        (False, True) * 3,
        (True, False) * 3,
        (False, True, True, True, True, False),
        (False,) * 6,
    ]:
        dX, f, n = _run(LoRA_QKV, X, weights, factors, grads, trainable)
        assert torch.equal(dX, dX_ref)
        for got, ref, t in zip(f, f_ref, trainable):
            assert (got is None) != t
            if t:
                assert torch.equal(got, ref)
        assert n < n_ref, (trainable, n, n_ref)


@pytest.mark.parametrize(
    "need_A, need_B",
    list(
        itertools.product(
            [(True, True, True), (False, False, False), (True, False, True)],
            [(True, True, True), (False, False, False), (False, True, False)],
        )
    ),
)
def test_shared_backward_skips_frozen_factor_gemms(need_A, need_B):
    g = torch.Generator().manual_seed(3)
    X = torch.randn(10, 48, generator = g)
    As = [torch.randn(48, 8, generator = g) for _ in range(3)]
    Bs = [torch.randn(8, 40, generator = g) for _ in range(3)]
    dYs = [torch.randn(10, 40, generator = g) for _ in range(3)]
    ref_As, ref_Bs, ref_dX = _shared_lora_backward(X, dYs, As, Bs, 0.5, None)
    with _CountGemms() as counter:
        d_As, d_Bs, dX = _shared_lora_backward(
            X, dYs, As, Bs, 0.5, None, need_A = need_A, need_B = need_B
        )
    assert torch.equal(dX, ref_dX)
    for got, ref, need in zip(d_As + d_Bs, ref_As + ref_Bs, need_A + need_B):
        assert (got is None) != need
        if need:
            assert torch.equal(got, ref)
    expected = 3 + 1 + (1 if any(need_A) else 0) + ((1 + sum(need_B)) if any(need_B) else 0)
    assert counter.count == expected, (counter.count, expected)
