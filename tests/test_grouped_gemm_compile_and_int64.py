# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Triton MoE grouped GEMM must run under torch.compile and index weights past 2^31 elements.

1. The kernels were marked `allow_in_graph`, so AOT traced their Python bodies on fake
   tensors: the graph kept the `torch.empty` output and dropped the Triton launch, and the
   compiled forward / backward returned uninitialised memory. They are now custom ops.
2. Weight offsets (`expert_idx * N * K`) were int32, so an (E, N, K) weight holding more
   than 2^31 elements read and wrote the wrong experts (Step-3.7 at 288 experts).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

pytest.importorskip("triton", reason = "the grouped GEMM is a Triton kernel")

try:
    from unsloth.kernels.moe.grouped_gemm import interface as gg
    from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
        KernelConfigBackward_dW,
        KernelConfigBackward_dX,
        KernelConfigForward,
    )
except Exception as exc:  # pragma: no cover - depends on the installed stack
    pytest.skip(reason = f"grouped_gemm is unimportable here: {exc}", allow_module_level = True)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "grouped GEMM needs a real CUDA device"
)

E, TOPK, T, H, I = 4, 2, 64, 256, 128
FWD = KernelConfigForward(
    BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 64, num_warps = 4, num_stages = 2
)
DX = KernelConfigBackward_dX(
    BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 64, num_warps = 4, num_stages = 2
)
DW = KernelConfigBackward_dW(
    BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 64, num_warps = 4, num_stages = 2
)


def _moe_mlp(x, w1, w2, m_sizes, gather_indices):
    h = gg.grouped_gemm(
        X = x, W = w1, m_sizes = m_sizes, topk = TOPK, gather_indices = gather_indices,
        permute_x = True, kernel_config_fwd = FWD, kernel_config_bwd_dX = DX,
        kernel_config_bwd_dW = DW, is_first_gemm = True,
    )  # fmt: skip
    gate, up = h.chunk(2, dim = -1)
    return gg.grouped_gemm(
        X = torch.nn.functional.silu(gate) * up, W = w2, m_sizes = m_sizes, topk = TOPK,
        gather_indices = gather_indices, permute_y = True, kernel_config_fwd = FWD,
        kernel_config_bwd_dX = DX, kernel_config_bwd_dW = DW, is_first_gemm = False,
    )  # fmt: skip


def _routing(device):
    idx = torch.topk(torch.randn(T, E, device = device), TOPK).indices.view(-1)
    m_sizes = torch.bincount(idx, minlength = E).to(torch.int32)
    return m_sizes, torch.argsort(idx, stable = True).to(torch.int32)


def test_kernels_are_opaque_ops_for_the_compiler():
    """allow_in_graph is what let the compiler trace through the launch."""
    if not hasattr(torch.library, "custom_op"):
        pytest.skip(
            reason = "torch < 2.4 has no torch.library.custom_op; the kernels fall back to dynamo.disable"
        )
    for name in ("grouped_gemm_forward", "grouped_gemm_dX", "grouped_gemm_dW"):
        assert hasattr(torch.ops.unsloth, name)


@requires_cuda
def test_compiled_matches_eager_forward_and_backward():
    torch.manual_seed(0)
    dev = "cuda"
    w1 = (torch.randn(E, 2 * I, H, device = dev) * 0.05).bfloat16().requires_grad_(True)
    w2 = (torch.randn(E, H, I, device = dev) * 0.05).bfloat16().requires_grad_(True)
    compiled = torch.compile(_moe_mlp, fullgraph = True)
    # Several calls: the stale buffer only shows up once the allocator reuses memory.
    for step in range(3):
        m_sizes, gather_indices = _routing(dev)
        x = torch.randn(T, H, device = dev).bfloat16() * (step + 1)
        grad = torch.randn(T * TOPK, H, device = dev)
        outs = []
        for fn in (_moe_mlp, compiled):
            xi = x.clone().requires_grad_(True)
            w1.grad = w2.grad = None
            y = fn(xi, w1, w2, m_sizes, gather_indices)
            (y.float() * grad).sum().backward()
            outs.append([t.detach().float().clone() for t in (y, xi.grad, w1.grad, w2.grad)])
        for name, e, c in zip(("y", "dx", "dw1", "dw2"), *outs):
            rel = ((c - e).norm() / e.norm()).item()
            assert rel < 2e-2, f"call {step}: compiled {name} differs from eager, rel {rel:.3g}"


@requires_cuda
def test_weight_past_2_pow_31_elements_reads_the_right_expert():
    n_experts, N, K, tokens = 129, 4096, 4096, 256  # 129 * 4096 * 4096 > 2^31
    need = n_experts * N * K * 2 + (1 << 30)
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(reason = f"needs {need / 2**30:.1f} GiB free for a > 2^31 element bf16 weight")
    torch.manual_seed(0)
    W = torch.empty(n_experts, N, K, device = "cuda", dtype = torch.bfloat16)
    W[:-1].zero_()
    W[-1].normal_(0, 0.02)
    # Every token goes to the last expert, whose weights start past element 2^31.
    m_sizes = torch.zeros(n_experts, device = "cuda", dtype = torch.int32)
    m_sizes[-1] = tokens
    X = torch.randn(tokens, K, device = "cuda", dtype = torch.bfloat16)
    y = gg.grouped_gemm_forward(
        X, W, 1, m_sizes, BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 128, BLOCK_SIZE_K = 64
    )
    ref = X.float() @ W[-1].float().T
    assert (y.float() - ref).abs().max().item() < 0.05 * ref.abs().max().item()
    del W
    torch.cuda.empty_cache()
