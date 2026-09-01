# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`grouped_gemm(gather_indices = None)` must survive when nothing permutes.

The signature defaults `gather_indices` to None and the wrapper only asserts it
is present when `permute_x` or `permute_y` is set, but it then normalised it
with an unconditional `gather_indices.view(-1)`, so the documented default died
with `AttributeError: 'NoneType' object has no attribute 'view'` (#8627). The
same unconditional dereference sat in `grouped_gemm_dX`, which reads
`gather_indices.shape[0]` to size dX, so the backward pass failed identically
once the forward was fixed.

Calling the kernel on activations that are already in expert-contiguous order is
a documented use of `permute_x = False`, and none of the three Triton kernels
touch `gather_indices_ptr` outside their `PERMUTE_X or PERMUTE_Y` branches, so
the caller should not have to pass a `torch.arange` the kernel never reads.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

pytest.importorskip("triton", reason = "the grouped GEMM is a Triton kernel")

try:
    from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm
except Exception as exc:  # pragma: no cover - depends on the installed stack
    pytest.skip(f"grouped_gemm is unimportable here: {exc}", allow_module_level = True)


CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason = "grouped GEMM needs a real CUDA device")

NUM_EXPERTS = 2
TOKENS_PER_EXPERT = 4
TOTAL_TOKENS = NUM_EXPERTS * TOKENS_PER_EXPERT
# The dX and dW kernels static_assert that N and K divide the autotuned block sizes, and those go up to 256.
N = K = 256


def _operands(device, requires_grad = False):
    X = torch.randn(TOTAL_TOKENS, K, device = device, dtype = torch.bfloat16)
    W = torch.randn(NUM_EXPERTS, N, K, device = device, dtype = torch.bfloat16)
    m_sizes = torch.full((NUM_EXPERTS,), TOKENS_PER_EXPERT, device = device, dtype = torch.int32)
    return X.requires_grad_(requires_grad), W.requires_grad_(requires_grad), m_sizes


# the contract, without a GPU


# the contract, without a GPU ----------------------------------------
def test_the_default_survives_the_wrapper_when_nothing_permutes():
    """On CPU the call has to die inside `grouped_gemm_forward` on its device
    assert. An AttributeError instead means the wrapper dereferenced None."""
    X, W, m_sizes = _operands("cpu")
    with pytest.raises(AssertionError, match = "must be on CUDA"):
        grouped_gemm(
            X = X,
            W = W,
            m_sizes = m_sizes,
            topk = 1,
            permute_x = False,
            permute_y = False,
            autotune = True,
        )


@pytest.mark.parametrize("permute_x, permute_y", [(True, False), (False, True)])
def test_permuting_without_indices_still_fails_with_the_explicit_message(permute_x, permute_y):
    """The guard is the whole reason the parameter can be optional, so it must
    keep firing ahead of anything that would dereference None."""
    X, W, m_sizes = _operands("cpu")
    with pytest.raises(AssertionError, match = "gather_indices is required"):
        grouped_gemm(
            X = X,
            W = W,
            m_sizes = m_sizes,
            topk = 1,
            permute_x = permute_x,
            permute_y = permute_y,
            autotune = True,
        )


# the numerics, on a real device


@requires_cuda
def test_forward_matches_the_dummy_index_workaround():
    """`torch.arange(total_tokens)` is what callers pass today to get past the
    crash, and the kernel never reads it, so both paths must agree exactly."""
    X, W, m_sizes = _operands("cuda")
    dummy = torch.arange(TOTAL_TOKENS, device = "cuda", dtype = torch.int32)

    without = grouped_gemm(
        X = X,
        W = W,
        m_sizes = m_sizes,
        topk = 1,
        permute_x = False,
        permute_y = False,
        autotune = True,
    )
    with_dummy = grouped_gemm(
        X = X,
        W = W,
        m_sizes = m_sizes,
        topk = 1,
        gather_indices = dummy,
        permute_x = False,
        permute_y = False,
        autotune = True,
    )

    assert without.shape == (TOTAL_TOKENS, N)
    assert torch.equal(without, with_dummy)

    reference = torch.cat(
        [
            X[e * TOKENS_PER_EXPERT : (e + 1) * TOKENS_PER_EXPERT] @ W[e].T
            for e in range(NUM_EXPERTS)
        ]
    )
    torch.testing.assert_close(without, reference)


@requires_cuda
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_backward_matches_the_dummy_index_workaround(topk):
    """`grouped_gemm_dX` sized its output off `gather_indices.shape[0]`, so the
    backward pass has to be exercised separately from the forward.

    Parametrised on topk because at topk = 1 the replacement (`M_total`) and the
    thing it replaces coincide, so that case alone cannot tell a correct fix from
    one that only holds when dX's `[NUM_TOKENS * TOPK, K]` output is `M_total`.
    """
    grads = {}
    for name, gather_indices in (
        ("none", None),
        ("dummy", torch.arange(TOTAL_TOKENS, device = "cuda", dtype = torch.int32)),
    ):
        torch.manual_seed(0)
        X, W, m_sizes = _operands("cuda", requires_grad = True)
        grouped_gemm(
            X = X,
            W = W,
            m_sizes = m_sizes,
            topk = topk,
            gather_indices = gather_indices,
            permute_x = False,
            permute_y = False,
            autotune = True,
        ).sum().backward()
        grads[name] = (X.grad, W.grad)

    assert grads["none"][0].shape == grads["dummy"][0].shape
    assert torch.equal(grads["none"][0], grads["dummy"][0])
    assert torch.equal(grads["none"][1], grads["dummy"][1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
