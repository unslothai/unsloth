# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The FP8 rowwise GEMV must address weights holding more than 2^31 elements."""

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("triton", reason = "the FP8 GEMV is a Triton kernel")

CUDA = torch.cuda.is_available()


@pytest.fixture(scope = "module")
def F():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import fp8

    return fp8


class _Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def _launch_kwargs(F, monkeypatch, N, K):
    rec = _Recorder()
    monkeypatch.setattr(F, "_fp8_rowwise_gemv_kernel", rec)
    monkeypatch.setattr(F, "_fp8_triton_device_context", lambda *a, **k: nullcontext())
    X = torch.empty(1, K, device = "meta", dtype = torch.bfloat16)
    W = torch.empty(N, K, device = "meta", dtype = torch.float8_e4m3fn)
    S = torch.empty(N, 1, device = "meta", dtype = torch.float32)
    F.fp8_rowwise_gemv(X, W, S)
    assert len(rec.calls) == 1
    return rec.calls[0][2]


@pytest.mark.parametrize(
    "N, K, expected",
    [
        (4096, 4096, False),
        (152064, 8192, False),
        (2**31 // 4096, 4096, False),  # exactly 2^31 elements: last offset is 2^31 - 1
        (2**31 // 4096 + 1, 4096, True),  # one row past: last row starts at 2^31
        (2**31 // 7168 + 64, 7168, True),  # K not a power of two
        (262144, 16384, True),  # 2^32 elements
    ],
)
def test_launch_widens_exactly_when_offsets_pass_int32(F, monkeypatch, N, K, expected):
    kwargs = _launch_kwargs(F, monkeypatch, N, K)
    assert kwargs.get("LONG_INDEXING", False) is expected, (N, K, kwargs)


def _compile_ir(F, long_indexing, arch = 90):
    from triton.compiler import ASTSource

    fn = F._fp8_rowwise_gemv_kernel
    sig = dict(x_ptr = "*bf16", w_ptr = "*fp8e4nv", s_ptr = "*fp32", y_ptr = "*bf16")
    sig |= dict(N = "i32", K = "i32", stride_wn = "i32")
    sig |= dict(BLOCK_N = "constexpr", BLOCK_K = "constexpr", LONG_INDEXING = "constexpr")
    cx = dict(BLOCK_N = 8, BLOCK_K = 512, LONG_INDEXING = long_indexing)
    try:
        src = ASTSource(fn, sig, constexprs = cx)
    except TypeError:  # pragma: no cover - Triton 3.2 uses `constants`
        sig = {k: v for k, v in sig.items() if v != "constexpr"}
        src = ASTSource(fn, sig, constants = cx)
    try:
        return triton_compile(src, arch)
    except Exception as exc:  # pragma: no cover
        if "driver" in str(exc).lower():
            pytest.skip(f"no Triton CUDA driver to compile against: {exc}")
        raise


def triton_compile(src, arch):
    import triton
    from triton.backends.compiler import GPUTarget

    return triton.compile(src, target = GPUTarget("cuda", arch, 32)).asm["ttir"]


@pytest.mark.parametrize("long_indexing", [False, True])
def test_row_offset_multiply_width(F, long_indexing):
    """The weight-row multiply is i64 exactly on the LONG_INDEXING path (compile-only)."""
    ttir = _compile_ir(F, long_indexing)
    muls = [line for line in ttir.splitlines() if "arith.muli" in line]
    wide = [line for line in muls if "i64>" in line or line.rstrip().endswith(": i64")]
    assert muls, ttir
    if long_indexing:
        assert wide, "\n".join(muls)
    else:
        assert not wide, "\n".join(wide)


def _operands(N, K, device = "cuda", seed = 0):
    g = torch.Generator(device = device).manual_seed(seed)
    X = torch.randn(1, K, device = device, dtype = torch.bfloat16, generator = g)
    W = (torch.randn(N, K, device = device, generator = g) * 0.05).to(torch.float8_e4m3fn)
    S = torch.rand(N, 1, device = device, generator = g) * 0.01 + 0.001
    return X, W, S


def _launch(F, X, W, S, long_indexing):
    import triton

    N, K = W.shape
    out = torch.empty(1, N, device = X.device, dtype = X.dtype)
    block_n = 1 if N <= 4096 else (4 if N <= 32768 else 8)
    F._fp8_rowwise_gemv_kernel[(triton.cdiv(N, block_n),)](
        X, W, S, out, N, K, W.stride(0),
        BLOCK_N = block_n, BLOCK_K = 512, LONG_INDEXING = long_indexing, num_warps = 2,
    )  # fmt: skip
    return out


def _skip_without_kernel(F):
    W = torch.empty(1, device = "cuda", dtype = torch.float8_e4m3fn)
    if F._fp8_kernel_unsupported(W):
        pytest.skip("no fp8e4nv Triton kernel on this device")
    if F._fp8_gemv_miscompiles(W.device):
        pytest.skip("Triton < 3.4 miscompiles this GEMV on Blackwell; production skips it")


@pytest.mark.skipif(not CUDA, reason = "needs a CUDA device")
@pytest.mark.parametrize("N, K", [(4096, 4096), (5000, 3000), (40000, 1536)])
def test_long_path_matches_int32_path_on_ordinary_shapes(F, N, K):
    _skip_without_kernel(F)
    X, W, S = _operands(N, K)
    short = _launch(F, X, W, S, False)
    wide = _launch(F, X, W, S, True)
    torch.cuda.synchronize()
    assert torch.equal(short, wide)
    ref = (X.float() @ (W.float() * S).T).to(X.dtype)
    torch.testing.assert_close(short.float(), ref.float(), atol = 2e-2, rtol = 2e-2)


@pytest.mark.skipif(not CUDA, reason = "needs a CUDA device")
def test_weight_past_2_31_elements(F):
    """A real (N, K) FP8 weight one block past 2^31 elements (2 GiB): the tail rows must match."""
    _skip_without_kernel(F)
    K = 4096
    N = 2**31 // K + 256
    if torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        pytest.skip("needs 4 GB free GPU memory")
    X = torch.randn(1, K, device = "cuda", dtype = torch.bfloat16)
    W = torch.empty(N, K, device = "cuda", dtype = torch.float8_e4m3fn)
    tail = slice(N - 512, N)
    W[tail] = (torch.randn(512, K, device = "cuda") * 0.05).to(torch.float8_e4m3fn)
    S = torch.rand(N, 1, device = "cuda") * 0.01 + 0.001
    assert F.can_use_fp8_rowwise_gemv(X, W, S)
    out = F.fp8_rowwise_gemv(X, W, S)
    ref = F.fp8_rowwise_gemv(X, W[tail].contiguous(), S[tail].contiguous())
    torch.cuda.synchronize()
    assert torch.equal(out[..., tail], ref)
