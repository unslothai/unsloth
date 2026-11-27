# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""float32 grouped GEMM (forward, dX, both dW kernels) vs an fp64 reference: Triton's default dot precision for
float32 is TF32 (about 3e-4 relative error, 10 mantissa bits), which a float32 run must not get."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

pytest.importorskip("triton", reason = "the grouped GEMM is a Triton kernel")
if not torch.cuda.is_available():
    pytest.skip("grouped GEMM needs a CUDA device", allow_module_level = True)

try:
    from unsloth.kernels.moe.grouped_gemm import interface as gg
    from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
        KernelConfigBackward_dW,
        KernelConfigBackward_dX,
        KernelConfigForward,
    )
except Exception as exc:  # pragma: no cover - depends on the installed stack
    pytest.skip(f"grouped_gemm is unimportable here: {exc}", allow_module_level = True)

DEV = "cuda"
# IEEE float32 lands near 1e-7 here; TF32 near 3e-4.
TOL = 2e-5


def _rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


@pytest.mark.parametrize("mode", ["permute_x", "permute_y", "none"])
@pytest.mark.parametrize("dw_kernel", ["", "legacy"])
def test_float32_grouped_gemm_is_ieee(mode, dw_kernel, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_DW_KERNEL", dw_kernel)
    E, N, K, T, topk = 4, 128, 256, 256, 2
    torch.manual_seed(0)
    idx = torch.topk(torch.randn(T, E, device = DEV), topk).indices.view(-1)
    m_sizes = torch.bincount(idx, minlength = E).to(torch.int32)
    gather = torch.argsort(idx, stable = True).to(torch.int32)
    rows = T if mode == "permute_x" else T * topk
    X = torch.randn(rows, K, device = DEV, requires_grad = True)
    W = torch.randn(E, N, K, device = DEV, requires_grad = True)
    G = torch.randn(T * topk, N, device = DEV)
    cfg = dict(BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 64)
    Y = gg.grouped_gemm(
        X = X, W = W, m_sizes = m_sizes, topk = topk, gather_indices = gather,
        permute_x = mode == "permute_x", permute_y = mode == "permute_y",
        kernel_config_fwd = KernelConfigForward(**cfg),
        kernel_config_bwd_dX = KernelConfigBackward_dX(**cfg),
        kernel_config_bwd_dW = KernelConfigBackward_dW(**cfg),
        is_first_gemm = mode != "permute_y",
    )  # fmt: skip
    Y.backward(G)

    # fp64 reference in the kernel's expert-grouped row order.
    Xd = X.detach().double().requires_grad_()
    Wd = W.detach().double().requires_grad_()
    g = gather.long()
    xs = Xd[g // topk] if mode == "permute_x" else Xd
    outs, start = [], 0
    for e, m in enumerate(m_sizes.tolist()):
        outs.append(xs[start : start + m] @ Wd[e].T)
        start += m
    grouped = torch.cat(outs)
    if mode == "permute_y":
        Yd = torch.empty_like(grouped)
        Yd[g] = grouped
    else:
        Yd = grouped
    Yd.backward(G.double())

    assert Y.dtype == torch.float32
    errors = {
        "Y": _rel(Y.detach(), Yd.detach()),
        "dX": _rel(X.grad, Xd.grad),
        "dW": _rel(W.grad, Wd.grad),
    }
    assert all(v < TOL for v in errors.values()), errors
