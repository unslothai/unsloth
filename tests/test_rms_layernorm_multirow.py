# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Narrow-row RMSNorm (several rows per program) is byte-identical to the one-row kernels."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level = True)

pytest.importorskip("triton")

import unsloth  # noqa: F401  (patches first)
import torch._dynamo.utils as dynamo_utils
from unsloth.kernels import rms_layernorm
from unsloth.kernels.rms_layernorm import fast_rms_layernorm

if torch.version.hip:
    pytest.skip("the multi-row kernels are off on ROCm", allow_module_level = True)

BF16_OK = torch.cuda.get_device_capability()[0] >= 8
DTYPES = [torch.float16, torch.float32] + ([torch.bfloat16] if BF16_OK else [])
NARROW = [33, 64, 96, 100, 127, 128]
_eager = lambda kernel: kernel


@pytest.fixture
def multirow(monkeypatch):
    monkeypatch.setattr(rms_layernorm, "_MULTIROW", True)
    return monkeypatch


def _bits(t):
    return t.contiguous().view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _inputs(n_rows, n_cols, dtype, kind, seed = 0):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    X = torch.randn(n_rows, n_cols, device = "cuda", generator = g)
    dY = torch.randn(n_rows, n_cols, device = "cuda", generator = g)
    if kind == "wide":
        # Magnitudes spread over 2^+-12 make every change in the order of adds visible.
        span = lambda: torch.randint(-12, 13, (n_rows, n_cols), device = "cuda", generator = g)
        X = X * torch.exp2(span().float())
        dY = dY * torch.exp2(span().float())
    elif kind == "offset":
        X = X + 30.0
    X, dY = X.to(dtype), dY.to(dtype)
    if kind == "zero_rows":
        # All-zero rows with zero upstream grads: dX must keep the one-row kernel's signed zeros.
        X[::3] = 0
        dY[::3, ::2] = 0
    W = torch.randn(n_cols, device = "cuda", generator = g).to(dtype)
    return X, W, dY


def _run(X, W, dY, gemma):
    Y, r = rms_layernorm._rms_forward(X, W, 1e-6, gemma, _eager)
    dY = dY.clone()
    dX = torch.empty_like(dY) if gemma else dY
    rms_layernorm._rms_backward(dY, dX, X, W, r, 1e-6, gemma, _eager)
    return Y, r, dX


def _both(monkeypatch, X, W, dY, gemma):
    monkeypatch.setattr(rms_layernorm, "_MULTIROW", False)
    one_row = _run(X, W, dY, gemma)
    monkeypatch.setattr(rms_layernorm, "_MULTIROW", True)
    many_rows = _run(X, W, dY, gemma)
    return one_row, many_rows


@pytest.mark.parametrize("kind", ["normal", "wide", "offset", "zero_rows"])
@pytest.mark.parametrize("n_rows", [1, 3, 129, 4099])
@pytest.mark.parametrize("n_cols", NARROW)
@pytest.mark.parametrize("gemma", [False, True], ids = ["llama", "gemma"])
@pytest.mark.parametrize("dtype", DTYPES, ids = str)
def test_multirow_matches_one_row_kernel(multirow, dtype, gemma, n_cols, n_rows, kind):
    """Y, the saved rstd and dX are bit-identical to one row per program."""
    assert rms_layernorm._multirow_settings(n_cols) is not None
    X, W, dY = _inputs(n_rows, n_cols, dtype, kind)
    one_row, many_rows = _both(multirow, X, W, dY, gemma)
    for a, b, name in zip(one_row, many_rows, ("Y", "r", "dX")):
        assert torch.equal(_bits(a), _bits(b)), name


@pytest.mark.parametrize("elements,num_warps", [(512, 1), (1024, 1), (4096, 4), (8192, 8)])
def test_multirow_bits_do_not_depend_on_tile_shape(multirow, elements, num_warps):
    """The row sum spells out every add, so other tile shapes give the same bytes."""
    multirow.setattr(rms_layernorm, "_MULTIROW_ELEMENTS", elements)
    multirow.setattr(rms_layernorm, "_MULTIROW_NUM_WARPS", num_warps)
    dtype = torch.bfloat16 if BF16_OK else torch.float16
    for gemma in (False, True):
        X, W, dY = _inputs(2051, 128, dtype, "wide", seed = 1)
        one_row, many_rows = _both(multirow, X, W, dY, gemma)
        for a, b, name in zip(one_row, many_rows, ("Y", "r", "dX")):
            assert torch.equal(_bits(a), _bits(b)), name


def test_multirow_selection(multirow):
    """Only rows of 33 to 128 columns (one element per lane in the one-row kernels) qualify."""
    for n_cols in (1, 16, 32, 129, 256, 1024, 4096):
        assert rms_layernorm._multirow_settings(n_cols) is None, n_cols
    for n_cols in (33, 64, 128):
        assert rms_layernorm._multirow_settings(n_cols) is not None, n_cols
    multirow.setattr(rms_layernorm, "_MULTIROW", False)
    assert rms_layernorm._multirow_settings(128) is None


class _Norm(torch.nn.Module):
    def __init__(self, dim, dtype):
        super().__init__()
        g = torch.Generator(device = "cuda").manual_seed(1)
        self.weight = torch.nn.Parameter((torch.randn(dim, device = "cuda", generator = g)).to(dtype))
        self.variance_epsilon = 1e-6


def _qk_norm_case(fn, dtype, gemma):
    # Qwen3's q norm: (batch, seq, heads, head_dim), normalised over head_dim.
    g = torch.Generator(device = "cuda").manual_seed(0)
    norm = _Norm(128, dtype)
    X = torch.randn(2, 37, 16, 128, device = "cuda", generator = g).to(dtype).requires_grad_(True)
    dY = torch.randn(2, 37, 16, 128, device = "cuda", generator = g).to(dtype)
    Y = fn(norm, X, gemma)
    Y.backward(dY)
    return Y.detach(), X.grad


@pytest.mark.parametrize("gemma", [False, True], ids = ["llama", "gemma"])
def test_multirow_compiled_matches_one_row_eager(multirow, gemma):
    """Under torch.compile (fullgraph, no graph break) the bytes match the one-row eager kernel."""
    if not getattr(torch.library, "triton_op", None):
        pytest.skip("needs torch.library.triton_op")
    dtype = torch.bfloat16 if rms_layernorm._BF16_TRACEABLE else torch.float16
    multirow.setattr(rms_layernorm, "_MULTIROW", False)
    ref = _qk_norm_case(fast_rms_layernorm, dtype, gemma)
    multirow.setattr(rms_layernorm, "_MULTIROW", True)
    torch._dynamo.reset()
    dynamo_utils.counters.clear()
    compiled = torch.compile(lambda n, x, gm: fast_rms_layernorm(n, x, gm), fullgraph = True)
    got = _qk_norm_case(compiled, dtype, gemma)
    torch._dynamo.reset()
    assert sum(dynamo_utils.counters["graph_break"].values()) == 0
    for a, b, name in zip(ref, got, ("Y", "dX")):
        assert torch.equal(_bits(a), _bits(b)), name
