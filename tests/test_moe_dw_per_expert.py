# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-expert grouped GEMM dW kernel vs the legacy kernel (UNSLOTH_MOE_DW_KERNEL=legacy) and fp64."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def _gg():
    pytest.importorskip("triton", reason = "the grouped GEMM is a Triton kernel")
    if not torch.cuda.is_available():
        pytest.skip("grouped GEMM needs a CUDA device")
    try:
        from unsloth.kernels.moe.grouped_gemm import interface as gg
    except Exception as exc:  # pragma: no cover - depends on the installed stack
        pytest.skip(f"grouped_gemm is unimportable here: {exc}")
    return gg


def _routing(T, E, topk, empty, seed):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    logits = torch.randn(T, E, device = "cuda", generator = g)
    for e in empty:
        logits[:, e] = -float("inf")
    if empty:
        logits[:, (max(empty) + 1) % E] += 4.0  # one hot expert next to the empty ones
    idx = torch.topk(logits, topk).indices.view(-1)
    m_sizes = torch.bincount(idx, minlength = E).to(torch.int32)
    gather = torch.argsort(idx, stable = True).to(torch.int32)
    return m_sizes, gather


def _ref(X, dY, m_sizes, gather, topk, mode, E, N, K):
    X, dY, gather = X.double(), dY.double(), gather.long()
    if mode == "permute_x":
        xs, dys = X[gather // topk], dY
    elif mode == "permute_y":
        xs, dys = X, dY[gather]
    else:
        xs, dys = X, dY
    out = torch.zeros(E, N, K, dtype = torch.float64, device = X.device)
    start = 0
    for e, m in enumerate(m_sizes.tolist()):
        out[e] = dys[start : start + m].T @ xs[start : start + m]
        start += m
    return out


# (E, N, K, T, topk): Qwen3-30B-A3B down, Qwen1.5-MoE gate_up (ragged N tail), Mixtral-like few experts.
SHAPES = [(128, 2048, 768, 256, 8), (60, 2816, 2048, 128, 4), (8, 704, 320, 300, 2)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", ["permute_x", "permute_y", "none"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("accumulate", [False, True])
def test_per_expert_matches_legacy_and_fp64(monkeypatch, shape, mode, dtype, accumulate):
    gg = _gg()
    E, N, K, T, topk = shape
    empty = (0, 3, E - 1)
    m_sizes, gather = _routing(T, E, topk, empty, seed = E)
    g = torch.Generator(device = "cuda").manual_seed(1)
    X = torch.randn(T if mode == "permute_x" else T * topk, K, device = "cuda", generator = g).to(dtype)
    dY = torch.randn(T * topk, N, device = "cuda", generator = g).to(dtype)
    kw = dict(
        BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 64, num_warps = 4, num_stages = 3,
        permute_x = mode == "permute_x", permute_y = mode == "permute_y",
    )  # fmt: skip
    prev = torch.randn(E, N, K, device = "cuda", generator = g).to(dtype)
    outs = {}
    for kernel in ("auto", "legacy"):
        monkeypatch.setenv("UNSLOTH_MOE_DW_KERNEL", kernel)
        if accumulate:
            out = prev.clone()
            gg._grouped_gemm_dW_accumulate_impl(X, dY, m_sizes, gather, out, topk, **kw)
        else:
            out = gg._grouped_gemm_dW_impl(X, dY, m_sizes, gather, topk, **kw)
        outs[kernel] = out
    # Same reduction order per (expert, tile): bit-identical to the legacy kernel.
    assert torch.equal(outs["auto"], outs["legacy"])
    ref = _ref(X, dY, m_sizes, gather, topk, mode, E, N, K) + (prev.double() if accumulate else 0)
    rel = (outs["auto"].double() - ref).norm() / ref.norm()
    assert rel < (3e-3 if dtype == torch.bfloat16 else 5e-4), rel
    for e in empty:  # an empty expert is exactly zero, or exactly the previous gradient
        assert torch.equal(outs["auto"][e], prev[e] if accumulate else torch.zeros_like(prev[e]))


def test_per_expert_fp32_dw_dtype_and_poisoned_output(monkeypatch):
    gg = _gg()
    E, N, K, T, topk = 16, 256, 192, 64, 2
    m_sizes, gather = _routing(T, E, topk, (2, 5), seed = 3)
    X = torch.randn(T, K, device = "cuda").bfloat16()
    dY = torch.randn(T * topk, N, device = "cuda").bfloat16()
    # The per-expert path allocates dW with torch.empty: poison the allocator's next block first.
    torch.full((E * N * K,), float("nan"), device = "cuda", dtype = torch.float32)
    monkeypatch.setenv("UNSLOTH_MOE_DW_KERNEL", "auto")
    out = gg._grouped_gemm_dW_impl(
        X, dY, m_sizes, gather, topk, dW_dtype = torch.float32, permute_x = True,
        BLOCK_SIZE_M = 32, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 64,
    )  # fmt: skip
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    ref = _ref(X, dY, m_sizes, gather, topk, "permute_x", E, N, K)
    assert ((out.double() - ref).norm() / ref.norm()) < 1e-5
    assert (out[2] == 0).all() and (out[5] == 0).all()


@pytest.mark.parametrize("accumulate", [False, True])
def test_per_expert_fp32_activations_bf16_weight(monkeypatch, accumulate):
    # bf16 full finetune: fp32 RMSNorm output into a bf16 expert weight, dW stored in bf16.
    gg = _gg()
    E, N, K, T, topk = 32, 512, 384, 256, 4
    m_sizes, gather = _routing(T, E, topk, (1, 7), seed = 11)
    g = torch.Generator(device = "cuda").manual_seed(2)
    X = torch.randn(T, K, device = "cuda", generator = g)
    dY = torch.randn(T * topk, N, device = "cuda", generator = g)
    prev = torch.randn(E, N, K, device = "cuda", generator = g).bfloat16()
    kw = dict(
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        num_warps = 8,
        num_stages = 3,
        permute_x = True,
    )
    outs = {}
    for kernel in ("auto", "legacy"):
        monkeypatch.setenv("UNSLOTH_MOE_DW_KERNEL", kernel)
        if accumulate:
            outs[kernel] = prev.clone()
            gg._grouped_gemm_dW_accumulate_impl(X, dY, m_sizes, gather, outs[kernel], topk, **kw)
        else:
            outs[kernel] = gg._grouped_gemm_dW_impl(
                X, dY, m_sizes, gather, topk, dW_dtype = torch.bfloat16, **kw
            )
    assert outs["auto"].dtype == torch.bfloat16
    ref = _ref(X, dY, m_sizes, gather, topk, "permute_x", E, N, K) + (
        prev.double() if accumulate else 0
    )
    for kernel, out in outs.items():
        assert ((out.double() - ref).norm() / ref.norm()) < 5e-3, kernel
    assert torch.equal(outs["auto"], outs["legacy"])


def test_kill_switch_selects_legacy(monkeypatch):
    gg = _gg()
    monkeypatch.setenv("UNSLOTH_MOE_DW_KERNEL", "legacy")
    assert not gg._use_per_expert_dW_kernel()
    monkeypatch.setenv("UNSLOTH_MOE_DW_KERNEL", "per_expert")
    assert gg._use_per_expert_dW_kernel()
    monkeypatch.delenv("UNSLOTH_MOE_DW_KERNEL")
    assert gg._use_per_expert_dW_kernel()
