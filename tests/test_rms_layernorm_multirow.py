# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Narrow-row RMSNorm (several rows per program) is byte-identical to the one-row kernels."""

import pytest
import torch

from real_accelerator import has_real_cuda

# The CUDA spoof patches torch.cuda probes to True process-wide; ask the pre-spoof answer.
if not has_real_cuda():
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
_eager = rms_layernorm._eager_kernel


@pytest.fixture
def multirow(monkeypatch):
    monkeypatch.setattr(rms_layernorm, "_MULTIROW", True)
    monkeypatch.setattr(rms_layernorm, "_MULTIROW_CHECKED", {})
    return monkeypatch


def _bits(t):
    return t.contiguous().view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _inputs(
    n_rows,
    n_cols,
    dtype,
    kind,
    seed = 0,
):
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


def _run(
    X,
    W,
    dY,
    gemma,
    rows = None,
):
    Y, r = rms_layernorm._rms_forward(X, W, 1e-6, gemma, _eager, rows)
    dY = dY.clone()
    dX = torch.empty_like(dY) if gemma else dY
    rms_layernorm._rms_backward(dY, dX, X, W, r, 1e-6, gemma, _eager, rows)
    return Y, r, dX


def _both(monkeypatch, X, W, dY, gemma):
    # Forced past the self-check: a mismatch there would fall back and compare one row to itself.
    settings = rms_layernorm._multirow_settings(X.shape[1])
    assert settings is not None
    return _run(X, W, dY, gemma, False), _run(X, W, dY, gemma, settings)


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
    assert list(rms_layernorm._MULTIROW_CHECKED.values()) == [True]
    assert rms_layernorm._MULTIROW
    for a, b, name in zip(ref, got, ("Y", "dX")):
        assert torch.equal(_bits(a), _bits(b)), name


class _Raising:
    def __init__(self, error):
        self.error = error

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            raise self.error

        return launch


class _OffByOneUlp:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        def launch(Y, *args, **kwargs):
            self.kernel[grid](Y, *args, **kwargs)
            _bits(Y).view(-1)[0] ^= 1

        return launch


def _narrow_case(monkeypatch, gemma = False):
    dtype = torch.bfloat16 if BF16_OK else torch.float16
    X, W, dY = _inputs(129, 128, dtype, "wide", seed = 2)
    monkeypatch.setattr(rms_layernorm, "_MULTIROW", False)
    one_row = _run(X, W, dY, gemma)
    monkeypatch.setattr(rms_layernorm, "_MULTIROW", True)
    return X, W, dY, one_row


def _assert_bits(one_row, got):
    for a, b, name in zip(one_row, got, ("Y", "r", "dX")):
        assert torch.equal(_bits(a), _bits(b)), name


@pytest.mark.parametrize("kernel", ["_rms_layernorm_forward_rows", "_rms_layernorm_backward_rows"])
def test_multirow_launch_failure_falls_back(multirow, kernel):
    """A failed multi-row launch runs the one-row kernel and turns the lever off process-wide."""
    X, W, dY, one_row = _narrow_case(multirow)
    multirow.setitem(
        rms_layernorm._MULTIROW_CHECKED, (X.device, X.dtype, W.dtype, 128, 1e-6, False), True
    )
    multirow.setattr(rms_layernorm, kernel, _Raising(RuntimeError("PTX JIT compilation failed")))
    with pytest.warns(UserWarning, match = "launch failed"):
        got = _run(X, W, dY, False)
    assert not rms_layernorm._MULTIROW
    _assert_bits(one_row, got)


def test_multirow_self_check_runs_per_eps(multirow):
    """eps is a constexpr, so every distinct eps gets its own self-check."""
    X, W, dY, _ = _narrow_case(multirow)
    for eps in (1e-6, 1e-5, 1e-6):
        rms_layernorm._rms_forward(X, W, eps, False, _eager)
    assert sorted(key[4] for key in rms_layernorm._MULTIROW_CHECKED) == [1e-6, 1e-5]
    assert all(rms_layernorm._MULTIROW_CHECKED.values())


def test_multirow_self_check_failure_under_compile_falls_back(multirow):
    """The first call's self-check launches re-raise while tracing; it must not escape."""
    X, W, dY, one_row = _narrow_case(multirow)
    multirow.setattr(
        rms_layernorm,
        "_rms_layernorm_forward_rows",
        _Raising(RuntimeError("PTX JIT compilation failed")),
    )
    multirow.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.warns(UserWarning, match = "self-check failed"):
        Y, r = rms_layernorm._rms_forward(X, W, 1e-6, False, _eager)
    assert not rms_layernorm._MULTIROW
    assert torch.equal(_bits(one_row[0]), _bits(Y))


@pytest.mark.parametrize(
    "error",
    [
        torch.cuda.OutOfMemoryError("CUDA out of memory"),
        torch._dynamo.exc.TorchRuntimeError("traced"),
    ],
    ids = ["oom", "dynamo"],
)
def test_multirow_launch_failure_reraises(multirow, error):
    """OOM and dynamo errors are not launch failures: they propagate and keep the lever on."""
    X, W, dY, _ = _narrow_case(multirow)
    multirow.setitem(
        rms_layernorm._MULTIROW_CHECKED, (X.device, X.dtype, W.dtype, 128, 1e-6, False), True
    )
    multirow.setattr(rms_layernorm, "_rms_layernorm_forward_rows", _Raising(error))
    with pytest.raises(type(error)):
        _run(X, W, dY, False)
    with pytest.raises(RuntimeError):
        rms_layernorm._rms_forward(X, W, 1e-6, False, lambda kernel: _Raising(RuntimeError("x")))
    assert rms_layernorm._MULTIROW


@pytest.mark.parametrize("gemma", [False, True], ids = ["llama", "gemma"])
def test_multirow_self_check_disables_on_mismatch(multirow, gemma):
    """A multi-row kernel that differs by one ulp is caught before use: one-row bytes, lever off."""
    X, W, dY, one_row = _narrow_case(multirow, gemma)
    multirow.setattr(
        rms_layernorm,
        "_rms_layernorm_forward_rows",
        _OffByOneUlp(rms_layernorm._rms_layernorm_forward_rows),
    )
    with pytest.warns(UserWarning, match = "self-check mismatch"):
        got = _run(X, W, dY, gemma)
    assert not rms_layernorm._MULTIROW
    assert list(rms_layernorm._MULTIROW_CHECKED.values()) == [False]
    _assert_bits(one_row, got)


def test_multirow_self_check_runs_once(multirow):
    """One check per (device, dtypes, width, variant); later calls only read the verdict."""
    X, W, dY, one_row = _narrow_case(multirow)
    calls = []
    check = rms_layernorm._multirow_self_check
    multirow.setattr(rms_layernorm, "_multirow_self_check", lambda *a: calls.append(a) or check(*a))
    launches = []
    rows_kernel = rms_layernorm._rms_layernorm_forward_rows

    class _Counting:
        def __getitem__(self, grid):
            launches.append(grid)
            return rows_kernel[grid]

    multirow.setattr(rms_layernorm, "_rms_layernorm_forward_rows", _Counting())
    for _ in range(3):
        _assert_bits(one_row, _run(X, W, dY, False))
    assert len(calls) == 1 and rms_layernorm._MULTIROW
    assert len(launches) == 1 + 3


@pytest.mark.skipif(not has_real_cuda() or torch.cuda.device_count() < 2, reason = "needs two GPUs")
@pytest.mark.parametrize("path", ["eager", "compiled", "op"])
def test_multirow_launches_on_the_tensors_device(multirow, path):
    """With cuda:0 current, cuda:1 inputs give the one-row bytes (launches use the tensor's device)."""
    if path != "eager" and not getattr(torch.library, "triton_op", None):
        pytest.skip("needs torch.library.triton_op")
    dtype = torch.bfloat16 if BF16_OK else torch.float16
    g = torch.Generator(device = "cuda:1").manual_seed(0)
    X = torch.randn(1184, 128, device = "cuda:1", generator = g).to(dtype)
    W = torch.randn(128, device = "cuda:1", generator = g).to(dtype)
    multirow.setattr(rms_layernorm, "_MULTIROW", False)
    with torch.cuda.device(1):
        ref, _ = rms_layernorm._rms_forward(X, W, 1e-6, False, _eager)
    multirow.setattr(rms_layernorm, "_MULTIROW", True)
    norm = torch.nn.Module()
    norm.weight, norm.variance_epsilon = torch.nn.Parameter(W), 1e-6
    with torch.cuda.device(0):
        if path == "op":
            got, _ = torch.ops.unsloth.rms_layernorm(X, W, 1e-6, False)
        else:
            fn = lambda n, x: fast_rms_layernorm(n, x)
            got = (torch.compile(fn, fullgraph = True) if path == "compiled" else fn)(
                norm, X
            ).detach()
            torch._dynamo.reset()
    assert torch.equal(_bits(ref), _bits(got))
    assert any(rms_layernorm._MULTIROW_CHECKED.values()) and rms_layernorm._MULTIROW
