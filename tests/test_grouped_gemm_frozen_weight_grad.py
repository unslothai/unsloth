# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Backward runs only the kernels autograd asks for: a frozen W (LoRA, dequantized experts) gets no
dW kernel or (E, N, K) buffer, a constant X no dX, and trainable operands keep both."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

pytest.importorskip("triton", reason = "the grouped GEMM is a Triton kernel")
if not torch.cuda.is_available():
    pytest.skip("grouped GEMM needs a CUDA device", allow_module_level = True)

try:
    from unsloth.kernels.moe.grouped_gemm import interface
except Exception as exc:  # pragma: no cover - depends on the installed stack
    pytest.skip(f"grouped_gemm is unimportable here: {exc}", allow_module_level = True)

E, T, N, K = 2, 8, 256, 256


def _run(x_grad, w_grad, monkeypatch):
    calls = []
    for name in ("grouped_gemm_dX", "grouped_gemm_dW"):
        real = getattr(interface, name)
        monkeypatch.setattr(
            interface,
            name,
            lambda *a, real = real, name = name, **k: calls.append(name) or real(*a, **k),
        )
    torch.manual_seed(0)
    X = torch.randn(T, K, device = "cuda", dtype = torch.float16, requires_grad = x_grad)
    W = torch.randn(E, N, K, device = "cuda", dtype = torch.float16, requires_grad = w_grad)
    m_sizes = torch.full((E,), T // E, device = "cuda", dtype = torch.int32)
    bias = torch.zeros(
        N, device = "cuda", dtype = torch.float16, requires_grad = True
    )  # something always trains
    out = interface.grouped_gemm(
        X = X, W = W, m_sizes = m_sizes, topk = 1, permute_x = False, permute_y = False, autotune = True
    )
    (out + bias).float().square().sum().backward()
    ref_X, ref_W = X.detach().float().requires_grad_(), W.detach().float().requires_grad_()
    ref = torch.cat([ref_X[e * (T // E) : (e + 1) * (T // E)] @ ref_W[e].T for e in range(E)])
    ref.square().sum().backward()
    return calls, X, W, ref_X, ref_W


@pytest.mark.parametrize(
    "x_grad, w_grad, want",
    [
        (True, False, ["grouped_gemm_dX"]),
        (False, True, ["grouped_gemm_dW"]),
        (True, True, ["grouped_gemm_dW", "grouped_gemm_dX"]),
        (False, False, []),
    ],
)
def test_backward_launches_only_the_needed_kernels(x_grad, w_grad, want, monkeypatch):
    calls, X, W, ref_X, ref_W = _run(x_grad, w_grad, monkeypatch)
    assert sorted(calls) == sorted(want)
    for t, ref, needed in ((X, ref_X, x_grad), (W, ref_W, w_grad)):
        if needed:
            torch.testing.assert_close(
                t.grad.float(), ref.grad, rtol = 2e-2, atol = 2e-2 * ref.grad.abs().max().item()
            )
        else:
            assert t.grad is None
