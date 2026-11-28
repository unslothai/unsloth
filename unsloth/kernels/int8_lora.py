# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Rotated INT8 frozen-base GEMMs for LoRA (opt-in: UNSLOTH_LORA_BASE_INT8=1, recipe int8_lora_rotated).

The frozen weight is stored once as int8 codes of W @ H (H: block-diagonal Walsh-Hadamard, group 128, along the
input dim) with one fp32 scale per output row: 1 byte per weight plus the 16-bit master, which moves to host RAM
as in fp8 LoRA. Forward: per-token int8 codes of X @ H (rotation and quantization in one kernel), an int8 GEMM
with int32 accumulation and the dequant (plus the LoRA term) in its epilogue. Input gradient: dY @ W =
(dY * s) @ Q @ H, so the weight row scale folds into dY before its per-token quantization, the same int8 codes
are the second operand, and the GEMM epilogue undoes the rotation on each 128-column block. The rotation spreads
activation outliers across the block, which is what makes per-token int8 accurate enough for training.

Same quant-state interface as fp8_lora.Fp8LoRABaseState (Unsloth's fused LoRA kernels call matmul / lora_first
/ dgrad / dgrad_accum), so attach, offload, merge and save reuse the fp8 LoRA plumbing. Consecutive projections
reading the same input (q / k / v, gate / up) share one activation quantization.
"""

import contextlib
import math
import os
import threading
from typing import Optional

import torch

from . import fp8_lora as _fp8
from .fp8_lora import (
    Fp8LoRABaseState,
    _compiling,
    _fp8_base_apply,
    _fp8_requantize_after_load,
    _is_transposed,
    _launch,
    _opaque_under_compile,
    offload_supported,
)

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

__all__ = [
    "INT8_LORA_STATS",
    "Int8LoRABaseState",
    "attach_int8_lora_base",
    "eligible_base",
    "int8_lora_fwd",
    "int8_lora_dgrad",
    "int8_lora_supported",
    "reset_int8_lora_stats",
    "GROUP",
]

GROUP = 128  # Hadamard block along the reduction dim of the forward GEMM
INT8_LORA_STATS = {
    "fwd": 0,
    "dgrad": 0,
    "fwd_16bit": 0,
    "dgrad_16bit": 0,
    "quantize": 0,
    "x_quant": 0,  # eager activation-quant requests
    "x_quant_shared": 0,  # of those, served by the previous projection's quantization
    "x_quant_kernel": 0,  # activation quantizations actually run (eager and compiled)
}
# Below this many rows (tokens in the call) the activation quantization costs more than the int8 GEMM saves
# (RTX PRO 6000, Qwen3 / Llama layers): those calls run the 16-bit GEMM on the 16-bit (or dequantized) weight.
_MIN_ROWS = int(os.environ.get("UNSLOTH_INT8_LORA_MIN_ROWS", "2048"))
# Models narrower than this convert nothing: at hidden size 1024 (Qwen3-0.6B) every layer measured slower than
# bf16 even at 16k tokens, since the quantization pass is not amortized over enough output columns.
MIN_HIDDEN = int(os.environ.get("UNSLOTH_INT8_LORA_MIN_HIDDEN", "2048"))
_SHARE_XQ = os.environ.get("UNSLOTH_INT8_LORA_SHARE_XQ", "1") != "0"
# Keep the transposed int8 weight for the input gradient (+1 byte per weight) instead of transposing per call.
_CACHE_T = os.environ.get("UNSLOTH_INT8_LORA_CACHE_T", "0") == "1"
_MULTI_GPU = torch.cuda.is_available() and torch.cuda.device_count() > 1
# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, warps, stages): fastest on 5090 / PRO 6000, and A100 / L4 without prefetch.
_TILE = (128, 128, 64, 8, 4, 4)
# Input gradient (BLOCK_M, BLOCK_R, GROUP_M, warps, stages): fastest of 8 on PRO 6000 and L4.
_DGRAD_TILE = (128, 128, 8, 8, 3)
# A100: 1.10x the tile above (geomean, Qwen3 1.7B-8B shapes).
_DGRAD_TILE_SM80 = (64, 128, 8, 4, 3)
_ROTQ_ROWS = 16
# One-pass quantization up to this row length (measured): rotated x on L4 (1.1-1.7x), dY on sm_8x (2.2x on A100);
# A100 keeps two-pass rotated x (one-pass was up to 1.7x slower there).
_ONE_PASS_MAX = 16384
_ROTQ1_BUDGET = 128  # rows x 128-blocks held per program by the one-pass rotated quantization


def _one_pass_block(n) -> Optional[int]:
    # Two row widths only (<= 2 compiled variants per call site across a model's projections), None = two-pass.
    return 4096 if n <= 4096 else (_ONE_PASS_MAX if n <= _ONE_PASS_MAX else None)


# Triton >= 3.8's sm_8x prefetch pass splits 8-bit dots and the tiles spill (0.2x bf16 on A100); skipping it for
# these kernels gives 1.5x bf16 on A100 / 1.7x on L4, bit-identical. Older Triton keeps it (3.2 is faster with it).
_SKIP_PREFETCH = os.environ.get("UNSLOTH_INT8_LORA_SKIP_PREFETCH", "1") != "0"


def _triton_version():
    import re
    m = re.match(r"(\d+)\.(\d+)", getattr(triton, "__version__", "") if triton is not None else "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


_TRITON_SPLITS_8BIT_PREFETCH = _triton_version() >= (3, 8)


def reset_int8_lora_stats():
    for key in INT8_LORA_STATS:
        INT8_LORA_STATS[key] = 0


def _count(key):
    if not _compiling():
        INT8_LORA_STATS[key] += 1


class _Arch(tuple):
    """Per-device launch configuration: (prefetch_off, dgrad_tile, rowq_one_pass, rotq_one_pass)."""

    __slots__ = ()

    prefetch_off = property(lambda self: self[0])
    dgrad_tile = property(lambda self: self[1])
    rowq_one_pass = property(lambda self: self[2])
    rotq_one_pass = property(lambda self: self[3])


def _arch_for(cap) -> _Arch:
    """The launch configuration for compute capability cap (major, minor)."""
    cap = (int(cap[0]), int(cap[1]))
    sm8x = cap[0] == 8
    prefetch_off = bool(
        sm8x and _SKIP_PREFETCH and _TRITON_SPLITS_8BIT_PREFETCH and _install_prefetch_skip()
    )
    dgrad_tile = _DGRAD_TILE_SM80 if prefetch_off and cap == (8, 0) else _DGRAD_TILE
    return _Arch((prefetch_off, dgrad_tile, sm8x, cap == (8, 9)))


_ARCH = {}


def _arch(device) -> _Arch:
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = _ARCH.get(index)
    if arch is None:
        arch = _ARCH[index] = _arch_for(torch.cuda.get_device_capability(index))
    return arch


_PREFETCH = threading.local()
_PREFETCH_SENTINEL = "_unsloth_int8_lora_prefetch_skip"
_PREFETCH_INSTALLED = []


def _install_prefetch_skip() -> bool:
    """Wrap Triton's prefetch pass so it is skipped while _no_prefetch() is active (this thread only)."""
    if _PREFETCH_INSTALLED:
        return _PREFETCH_INSTALLED[0]
    ok = False
    try:
        from triton._C.libtriton import passes

        ttgpuir = passes.ttgpuir
        original = ttgpuir.add_prefetch
        if getattr(original, _PREFETCH_SENTINEL, False):
            ok = True
        elif callable(original):

            def add_prefetch(*args, **kwargs):
                if getattr(_PREFETCH, "skip", False):
                    return None
                return original(*args, **kwargs)

            setattr(add_prefetch, _PREFETCH_SENTINEL, True)
            add_prefetch.__wrapped__ = original
            ttgpuir.add_prefetch = add_prefetch
            ok = True
    except Exception:
        ok = False
    _PREFETCH_INSTALLED.append(ok)
    return ok


@contextlib.contextmanager
def _no_prefetch(active):
    # PREFETCH_OFF gives these compiles their own Triton cache key, so stock and no-prefetch never mix.
    if not active:
        yield
        return
    previous = getattr(_PREFETCH, "skip", False)
    _PREFETCH.skip = True
    try:
        yield
    finally:
        _PREFETCH.skip = previous


def int8_lora_supported(device) -> Optional[str]:
    """None when this device can run the int8 base, else why not."""
    device = torch.device(device)
    if triton is None:
        return "Triton is not installed (on Windows: pip install triton-windows)"
    if device.type != "cuda" or torch.version.hip is not None:
        return f"needs an NVIDIA GPU (got {device})"
    if torch.cuda.get_device_capability(device) < (8, 0):
        return "needs int8 tensor cores with Triton int8 dot support (compute capability 8.0+)"
    return None


_HADAMARD = {}


def _hadamard_pm1(device, dtype):
    """The +-1 Sylvester Hadamard of order GROUP (exact in bf16 / fp32; normalized by 1 / sqrt(GROUP) elsewhere)."""
    key = (device, dtype)
    h = _HADAMARD.get(key)
    if h is None:
        m = torch.ones((1, 1), dtype = torch.float32)
        while m.shape[0] < GROUP:
            m = torch.cat([torch.cat([m, m], 1), torch.cat([m, -m], 1)], 0)
        h = _HADAMARD[key] = m.to(device = device, dtype = dtype).contiguous()
    return h


if triton is not None:

    @triton.jit(do_not_specialize = ["M"])
    def _int8_rotq_kernel(
        x_ptr,
        h_ptr,
        q_ptr,
        s_ptr,
        M,
        K,
        stride_x,
        G: tl.constexpr,  # Hadamard block, fixed 128
        INV_SCALE: tl.constexpr,  # 1 / (127 * sqrt(G))
        BLOCK_M: tl.constexpr,  # rows per program, fixed 16
    ):
        # q = rint(127 * (x @ H) / rowmax|x @ H|), s = rowmax / (127 * sqrt(G)): two passes over the row, the
        # second one mostly from L2.
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < M
        g = tl.arange(0, G)
        H = tl.load(h_ptr + g[:, None] * G + g[None, :])
        amax = tl.zeros((BLOCK_M,), dtype = tl.float32)
        nans = tl.zeros((BLOCK_M,), dtype = tl.int32)
        for k0 in range(0, K, G):
            x = tl.load(
                x_ptr + rows[:, None] * stride_x + (k0 + g)[None, :], mask = rmask[:, None], other = 0.0
            )
            xr = tl.dot(x, H)
            amax = tl.maximum(amax, tl.max(tl.abs(xr), axis = 1))
            nans += tl.sum((xr != xr).to(tl.int32), axis = 1)
        amax = tl.maximum(amax, 1e-30)
        mul = 127.0 / amax
        for k0 in range(0, K, G):
            x = tl.load(
                x_ptr + rows[:, None] * stride_x + (k0 + g)[None, :], mask = rmask[:, None], other = 0.0
            )
            xr = tl.dot(x, H)
            y = tl.extra.cuda.libdevice.rint(xr * mul[:, None])
            y = tl.minimum(tl.maximum(y, -127.0), 127.0)
            tl.store(
                q_ptr + rows[:, None] * K + (k0 + g)[None, :], y.to(tl.int8), mask = rmask[:, None]
            )
        # tl.max drops NaN: a NaN row scale makes the whole output row NaN, as the 16-bit GEMM would.
        tl.store(s_ptr + rows, tl.where(nans > 0, float("nan"), amax * INV_SCALE), mask = rmask)

    @triton.jit(do_not_specialize = ["M"])
    def _int8_rotq1_kernel(
        x_ptr,
        h_ptr,
        q_ptr,
        s_ptr,
        M,
        K,
        stride_x,
        G: tl.constexpr,  # Hadamard block, fixed 128
        INV_SCALE: tl.constexpr,  # 1 / (127 * sqrt(G))
        ROWS: tl.constexpr,  # rows per program
        NB: tl.constexpr,  # next power of two >= K / G (>= 16): the row is held as NB blocks of G
    ):
        # Same codes and scales as _int8_rotq_kernel, one read of x.
        r = tl.arange(0, ROWS)
        b = tl.arange(0, NB)
        g = tl.arange(0, G)
        rows = tl.program_id(0).to(tl.int64) * ROWS + r
        mask = (rows < M)[:, None, None] & (b < K // G)[None, :, None]
        cols = b[None, :, None] * G + g[None, None, :]
        x = tl.load(x_ptr + rows[:, None, None] * stride_x + cols, mask = mask, other = 0.0)
        H = tl.load(h_ptr + g[:, None] * G + g[None, :])
        xr = tl.reshape(tl.dot(tl.reshape(x, (ROWS * NB, G)), H), (ROWS, NB * G))
        amax = tl.maximum(tl.max(tl.abs(xr), axis = 1), 1e-30)
        nans = tl.sum((xr != xr).to(tl.int32), axis = 1)
        y = tl.extra.cuda.libdevice.rint(xr * (127.0 / amax)[:, None])
        y = tl.reshape(tl.minimum(tl.maximum(y, -127.0), 127.0).to(tl.int8), (ROWS, NB, G))
        tl.store(q_ptr + rows[:, None, None] * K + cols, y, mask = mask)
        # tl.max drops NaN: a NaN row scale makes the whole output row NaN, as the 16-bit GEMM would.
        tl.store(s_ptr + rows, tl.where(nans > 0, float("nan"), amax * INV_SCALE), mask = rows < M)

    @triton.jit(do_not_specialize = ["N"])
    def _int8_rowq_kernel(
        x_ptr,
        c_ptr,
        q_ptr,
        s_ptr,
        N,
        stride_x,
        BLOCK: tl.constexpr,  # columns per chunk, fixed 2048
    ):
        # Per-row int8 of x * c (c: the weight's per-row scales, folded into dY for the input gradient).
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        amax = tl.zeros((BLOCK,), dtype = tl.float32)
        nans = tl.zeros((BLOCK,), dtype = tl.int32)
        for n0 in range(0, N, BLOCK):
            mask = n0 + offs < N
            x = tl.load(x_ptr + row * stride_x + n0 + offs, mask = mask, other = 0.0).to(tl.float32)
            c = tl.load(c_ptr + n0 + offs, mask = mask, other = 0.0)
            xc = x * c
            amax = tl.maximum(amax, tl.abs(xc))
            nans += (xc != xc).to(tl.int32)
        rmax = tl.maximum(tl.max(amax, axis = 0), 1e-30)
        mul = 127.0 / rmax
        for n0 in range(0, N, BLOCK):
            mask = n0 + offs < N
            x = tl.load(x_ptr + row * stride_x + n0 + offs, mask = mask, other = 0.0).to(tl.float32)
            c = tl.load(c_ptr + n0 + offs, mask = mask, other = 0.0)
            y = tl.extra.cuda.libdevice.rint(x * c * mul)
            y = tl.minimum(tl.maximum(y, -127.0), 127.0)
            tl.store(q_ptr + row * N + n0 + offs, y.to(tl.int8), mask = mask)
        tl.store(s_ptr + row, tl.where(tl.sum(nans, axis = 0) > 0, float("nan"), rmax / 127.0))

    @triton.jit(do_not_specialize = ["M"])
    def _int8_rowq1_kernel(
        x_ptr,
        c_ptr,
        q_ptr,
        s_ptr,
        M,
        N,
        stride_x,
        ROWS: tl.constexpr,  # rows per program
        BLOCK: tl.constexpr,  # next power of two >= N: the whole row is read once
    ):
        # Same codes and scales as _int8_rowq_kernel, one read of x.
        rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        offs = tl.arange(0, BLOCK)
        mask = (rows < M)[:, None] & (offs < N)[None, :]
        x = tl.load(x_ptr + rows[:, None] * stride_x + offs[None, :], mask = mask, other = 0.0).to(
            tl.float32
        )
        xc = x * tl.load(c_ptr + offs, mask = offs < N, other = 0.0)[None, :]
        rmax = tl.maximum(tl.max(tl.abs(xc), axis = 1), 1e-30)
        nans = tl.sum((xc != xc).to(tl.int32), axis = 1)
        y = tl.extra.cuda.libdevice.rint(xc * (127.0 / rmax)[:, None])
        y = tl.minimum(tl.maximum(y, -127.0), 127.0)
        tl.store(q_ptr + rows[:, None] * N + offs[None, :], y.to(tl.int8), mask = mask)
        tl.store(s_ptr + rows, tl.where(nans > 0, float("nan"), rmax / 127.0), mask = rows < M)

    @triton.jit(do_not_specialize = ["M"])
    def _int8_mm_kernel(
        a_ptr,
        b_ptr,
        as_ptr,
        bs_ptr,
        d_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_d,
        HAS_D: tl.constexpr,  # add the LoRA term / running input gradient in the epilogue
        BLOCK_M: tl.constexpr,  # rows of c per program
        BLOCK_N: tl.constexpr,  # columns of c per program
        BLOCK_K: tl.constexpr,  # reduction step; K % BLOCK_K == 0, so the K loop loads unmasked
        GROUP_M: tl.constexpr,  # row tiles per swizzle group, for L2 reuse of the b tiles
        PREFETCH_OFF: tl.constexpr,  # compiled without the sm_8x prefetch pass (cache key only)
    ):
        # c[m, n] = as[m] * bs[n] * sum_k a[m, k] b[n, k] (+ d[m, n]); a, b int8 with K contiguous, K % BLOCK_K == 0.
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rm[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + rn[None, :] * K + offs_k[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.int32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype = tl.int32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        xs = tl.load(as_ptr + rm)
        ws = tl.load(bs_ptr + rn)
        y = acc.to(tl.float32) * xs[:, None] * ws[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if HAS_D:
            y += tl.load(
                d_ptr + offs_m[:, None] * stride_d + offs_n[None, :], mask = mask, other = 0.0
            ).to(tl.float32)
        tl.store(
            c_ptr + offs_m[:, None] * N + offs_n[None, :], y.to(c_ptr.dtype.element_ty), mask = mask
        )

    @triton.jit(do_not_specialize = ["M"])
    def _int8_dgrad_kernel(
        a_ptr,
        b_ptr,
        as_ptr,
        h_ptr,
        d_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_d,
        HAS_D: tl.constexpr,  # add d (the running input gradient) in the epilogue
        INV_SQRT_G: tl.constexpr,  # 1 / sqrt(G), the Hadamard normalisation
        G: tl.constexpr,  # output columns per program = the Hadamard block, fixed 128
        BLOCK_M: tl.constexpr,  # rows of c per program
        BLOCK_R: tl.constexpr,  # reduction (N) step
        GROUP_M: tl.constexpr,  # row tiles per swizzle group, for L2 reuse of the bt tiles
        PREFETCH_OFF: tl.constexpr,  # compiled without the sm_8x prefetch pass (cache key only)
    ):
        # c[m, kblock] = (as[m] * sum_n a[m, n] bt[kblock, n]) @ H / sqrt(G) (+ d): a [M, N] int8 codes of dY * s,
        # bt [K, N] the transposed forward int8 codes of W @ H (int8 MMA operands are K-major); the epilogue undoes
        # the rotation of each 128-column block, split into two bf16 halves so the rotation keeps fp32 accuracy.
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(K, G)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_n.to(tl.int64) * G + tl.arange(0, G)
        rm = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
        offs_r = tl.arange(0, BLOCK_R)
        acc = tl.zeros((BLOCK_M, G), dtype = tl.int32)
        for r0 in range(0, N, BLOCK_R):
            rmask = r0 + offs_r < N
            a = tl.load(
                a_ptr + rm[:, None] * N + (r0 + offs_r)[None, :], mask = rmask[None, :], other = 0
            )
            b = tl.load(
                b_ptr + offs_k[None, :] * N + (r0 + offs_r)[:, None], mask = rmask[:, None], other = 0
            )
            acc = tl.dot(a, b, acc, out_dtype = tl.int32)
        y = acc.to(tl.float32) * tl.load(as_ptr + rm)[:, None]
        g = tl.arange(0, G)
        H = tl.load(h_ptr + g[:, None] * G + g[None, :])
        hi = y.to(tl.bfloat16)
        lo = (y - hi.to(tl.float32)).to(tl.bfloat16)
        z = (tl.dot(hi, H) + tl.dot(lo, H)) * INV_SQRT_G
        mask = offs_m[:, None] < M
        if HAS_D:
            z += tl.load(
                d_ptr + offs_m[:, None] * stride_d + offs_k[None, :], mask = mask, other = 0.0
            ).to(tl.float32)
        tl.store(
            c_ptr + offs_m[:, None] * K + offs_k[None, :], z.to(c_ptr.dtype.element_ty), mask = mask
        )


def _device_ctx(t):
    return torch.cuda.device(t.device) if _MULTI_GPU else contextlib.nullcontext()


def quantize_rotated_rows(x: torch.Tensor):
    """x [M, K] 16-bit (K % 128 == 0) -> (int8 codes of x @ H / sqrt(128) [M, K], fp32 per-row scales [M])."""
    M, K = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    q = torch.empty((M, K), device = x.device, dtype = torch.int8)
    s = torch.empty((M,), device = x.device, dtype = torch.float32)
    h = _hadamard_pm1(x.device, x.dtype)
    width = _one_pass_block(K)
    if width is not None and _arch(x.device).rotq_one_pass:
        nb = width // GROUP
        rows = _ROTQ1_BUDGET // nb
        with _device_ctx(x):
            _launch(
                _int8_rotq1_kernel,
                ((M + rows - 1) // rows,),
                (x, h, q, s, M, K, x.stride(0)),
                4,
                {
                    "G": GROUP,
                    "INV_SCALE": 1.0 / (127.0 * math.sqrt(GROUP)),
                    "ROWS": rows,
                    "NB": nb,
                },
                x.device.index,
                num_warps = 8,
            )
        return q, s
    with _device_ctx(x):
        _launch(
            _int8_rotq_kernel,
            ((M + _ROTQ_ROWS - 1) // _ROTQ_ROWS,),
            (x, h, q, s, M, K, x.stride(0)),
            4,
            {"G": GROUP, "INV_SCALE": 1.0 / (127.0 * math.sqrt(GROUP)), "BLOCK_M": _ROTQ_ROWS},
            x.device.index,
            num_warps = 4,
        )
    return q, s


def quantize_rows_scaled(x: torch.Tensor, col_scale: torch.Tensor):
    """x [M, N] 16-bit, col_scale fp32 [N] -> (int8 codes of x * col_scale per row [M, N], fp32 scales [M])."""
    M, N = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    q = torch.empty((M, N), device = x.device, dtype = torch.int8)
    s = torch.empty((M,), device = x.device, dtype = torch.float32)
    block = _one_pass_block(N)
    if block is not None and _arch(x.device).rowq_one_pass:
        rows = _ONE_PASS_MAX // block
        with _device_ctx(x):
            _launch(
                _int8_rowq1_kernel,
                ((M + rows - 1) // rows,),
                (x, col_scale, q, s, M, N, x.stride(0)),
                4,
                {"ROWS": rows, "BLOCK": block},
                x.device.index,
                num_warps = 8,
            )
        return q, s
    with _device_ctx(x):
        _launch(
            _int8_rowq_kernel,
            (M,),
            (x, col_scale, q, s, N, x.stride(0)),
            4,
            {"BLOCK": 2048},
            x.device.index,
            num_warps = 8,
        )
    return q, s


def _mm(xq, xs, wq, ws, delta, out_dtype):
    M, K = xq.shape
    N = wq.shape[0]
    out = torch.empty((M, N), device = xq.device, dtype = out_dtype)
    bm, bn, bk, gm, warps, stages = _TILE
    if delta is not None and delta.stride(-1) != 1:
        delta = delta.contiguous()  # the epilogue reads rows with a unit column stride
    d = delta if delta is not None else out
    off = _arch(xq.device).prefetch_off
    with _device_ctx(xq), _no_prefetch(off):
        _launch(
            _int8_mm_kernel,
            (((M + bm - 1) // bm) * ((N + bn - 1) // bn),),
            (xq, wq, xs, ws, d, out, M, N, K, d.stride(0)),
            6,
            {
                "HAS_D": delta is not None,
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "BLOCK_K": bk,
                "GROUP_M": gm,
                "PREFETCH_OFF": off,
            },
            xq.device.index,
            num_warps = warps,
            num_stages = stages,
        )
    return out


def _dgrad_mm(dyq, dys, wqt, delta, out_dtype):
    M, N = dyq.shape
    K = wqt.shape[0]
    out = torch.empty((M, K), device = dyq.device, dtype = out_dtype)
    arch = _arch(dyq.device)
    bm, br, gm, warps, stages = arch.dgrad_tile
    if delta is not None and delta.stride(-1) != 1:
        delta = delta.contiguous()
    d = delta if delta is not None else out
    h = _hadamard_pm1(dyq.device, torch.bfloat16)
    off = arch.prefetch_off
    with _device_ctx(dyq), _no_prefetch(off):
        _launch(
            _int8_dgrad_kernel,
            (((M + bm - 1) // bm) * (K // GROUP),),
            (dyq, wqt, dys, h, d, out, M, N, K, d.stride(0)),
            6,
            {
                "HAS_D": delta is not None,
                "INV_SQRT_G": 1.0 / math.sqrt(GROUP),
                "G": GROUP,
                "BLOCK_M": bm,
                "BLOCK_R": br,
                "GROUP_M": gm,
                "PREFETCH_OFF": off,
            },
            dyq.device.index,
            num_warps = warps,
            num_stages = stages,
        )
    return out


# One shared activation quantization: q / k / v (and gate / up) read the same X one after the other. Eager only:
# the key is the identity of the root tensor X is a view of (held, so the id cannot be reused while the entry
# lives) plus its version counter (bumped by every in-place write through any view). Traced code always
# quantizes: buffer reuse inside a compiled graph would not bump a version counter. One entry, replaced by the
# next different input.
_XQ = [None]


def _xq_key(x):
    root = x._base if x._base is not None else x
    try:
        return root, (
            id(root),
            x.data_ptr(),
            tuple(x.shape),
            tuple(x.stride()),
            x.dtype,
            root._version,
        )
    except Exception:  # inference tensors have no version counter
        return root, None


def quantize_x(x):
    """Rotated per-token int8 codes of x [M, K] (shared with the previous call on the same eager input)."""
    _count("x_quant")
    # A CUDA graph replays the kernels it captured: a cache hit while capturing would bake in stale codes.
    if not _SHARE_XQ or _compiling() or torch.cuda.is_current_stream_capturing():
        return int8_lora_quant_x(x)
    root, key = _xq_key(x)
    entry = _XQ[0]
    if key is not None and entry is not None and entry[0] == key:
        _count("x_quant_shared")
        return entry[2], entry[3]
    q, s = int8_lora_quant_x(x)
    _XQ[0] = (key, root, q, s) if key is not None else None
    return q, s


def clear_int8_lora_cache():
    _XQ[0] = None


def _quant_x_fake(x):
    return x.new_empty(x.shape, dtype = torch.int8), x.new_empty((x.shape[0],), dtype = torch.float32)


@_opaque_under_compile("int8_lora_quant_x", _quant_x_fake)
def int8_lora_quant_x(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Pure (no mutation): inside a compiled region identical calls on one input are CSE'd into one.
    _count("x_quant_kernel")
    return quantize_rotated_rows(x)


def _mm_fake(xq, xs, wq, ws, delta, out_dtype_of):
    return out_dtype_of.new_empty((xq.shape[0], wq.shape[0]))


@_opaque_under_compile("int8_lora_mm", _mm_fake)
def int8_lora_mm(
    xq: torch.Tensor,
    xs: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
    delta: Optional[torch.Tensor],
    out_dtype_of: torch.Tensor,
) -> torch.Tensor:
    """Forward GEMM on quantized codes: xs * ws * (xq @ wq.T) (+ delta), in out_dtype_of.dtype."""
    _count("fwd")
    return _mm(xq, xs, wq, ws, delta, out_dtype_of.dtype)


def int8_lora_fwd(
    x,
    wq,
    ws,
    delta = None,
):
    """x [M, K] (16-bit) @ W.T (+ delta [M, N]) -> [M, N] in x.dtype, W = (wq * ws[:, None]) @ H.T."""
    xq, xs = quantize_x(x)
    return int8_lora_mm(xq, xs, wq, ws, delta, x)


def _dgrad_fake(dy, wq, ws, wqt, delta):
    return dy.new_empty((dy.shape[0], wq.shape[1]))


@_opaque_under_compile("int8_lora_dgrad", _dgrad_fake)
def int8_lora_dgrad(
    dy: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
    wqt: Optional[torch.Tensor],
    delta: Optional[torch.Tensor],
) -> torch.Tensor:
    """dy [M, N] (16-bit) @ W (+ delta [M, K]) -> [M, K] in dy.dtype. wqt: the cached wq.T (contiguous), else
    transposed here (1 byte per weight, a few percent of the GEMM)."""
    _count("dgrad")
    if wqt is None:
        wqt = _fp8.fp8_transpose(wq)
    dyq, dys = quantize_rows_scaled(dy, ws)
    return _dgrad_mm(dyq, dys, wqt, delta, dy.dtype)


def quantize_weight(weight: torch.Tensor, rows_per_chunk: int = 4096):
    """W [N, K] -> (int8 codes of W @ H / sqrt(128) per row [N, K], fp32 row scales [N]), in row chunks."""
    N, K = weight.shape
    h = _hadamard_pm1(weight.device, torch.float32) / math.sqrt(GROUP)
    q = torch.empty((N, K), device = weight.device, dtype = torch.int8)
    s = torch.empty((N,), device = weight.device, dtype = torch.float32)
    for r0 in range(0, N, rows_per_chunk):
        w = weight[r0 : r0 + rows_per_chunk].float()
        wr = (w.view(-1, K // GROUP, GROUP) @ h).view(-1, K)
        amax = wr.abs().amax(1).clamp_min(1e-30)
        q[r0 : r0 + rows_per_chunk] = (
            torch.round(wr * (127.0 / amax)[:, None]).clamp_(-127, 127).to(torch.int8)
        )
        s[r0 : r0 + rows_per_chunk] = amax / 127.0
    return q, s


def dequantize_weight(wq: torch.Tensor, ws: torch.Tensor, dtype):
    """The weight the int8 GEMMs use: (wq * ws) @ H.T, in `dtype`."""
    N, K = wq.shape
    h = _hadamard_pm1(wq.device, torch.float32) / math.sqrt(GROUP)
    w = (wq.float() * ws[:, None]).view(N, K // GROUP, GROUP) @ h.t()
    return w.view(N, K).to(dtype)


class Int8LoRABaseState(Fp8LoRABaseState):
    """Quant state of a frozen LoRA base on the rotated int8 GEMMs (same interface as Fp8LoRABaseState)."""

    _unsloth_packed_weight_state = True
    _unsloth_fp8_dgrad = (
        True  # its own input-gradient GEMM: the fused LoRA kernels call dgrad / dgrad_accum
    )
    _unsloth_int8_lora = True

    def __init__(self, weight: torch.Tensor):
        self.wq = self.ws = self.wqt = None
        super().__init__(weight)

    def refresh(self, weight: torch.Tensor):
        if _compiling():
            return self
        key = self._weight_key(weight)
        if key == self._key:
            return self
        with torch.inference_mode(False), torch.no_grad():
            w = weight.detach()
            if self.offloaded:
                w = w.to(self.device)
            else:
                self.device = w.device
            self.wq, self.ws = quantize_weight(w)
            self.wqt = _fp8.fp8_transpose(self.wq) if _CACHE_T else None
        self.scaling = "int8_rotated"
        self.dtype = weight.dtype
        self._key = key
        self.quantizations += 1
        _count("quantize")
        return self

    def _dense(self, W):
        if W is not None and not self.offloaded and W.device == self.device:
            return W
        dense = dequantize_weight(self.wq, self.ws, self.dtype if W is None else W.dtype)
        return dense.t() if W is not None and _is_transposed(W) else dense

    def resident_bytes(self, module = None):
        nbytes = sum(
            t.numel() * t.element_size() for t in (self.wq, self.ws, self.wqt) if t is not None
        )
        out = {"fp8": nbytes, "int8": nbytes, "master_device": 0, "master_host": 0}
        if module is not None:
            w = module.weight
            size = w.numel() * w.element_size()
            if self.offloaded:
                out["master_host"] = size
            elif w.device == self.device:
                out["master_device"] = size
        return out

    def _use_fp8(self, X):
        # Offloaded master: the 16-bit path would rebuild the dense weight every call, so any row count runs int8.
        return X.dtype in (torch.bfloat16, torch.float16) and (
            self.offloaded or X.numel() // max(X.shape[-1], 1) >= _MIN_ROWS
        )

    def matmul(
        self,
        X,
        W,
        out = None,
    ):
        shape = X.shape
        X2 = X.reshape(-1, shape[-1])
        if not self._use_fp8(X2):
            if _is_transposed(W):
                _count("dgrad_16bit")
            else:
                _count("fwd_16bit")
            result = torch.matmul(X2, self._dense(W).t().to(X2.dtype))
        elif _is_transposed(W):
            result = int8_lora_dgrad(X2, self.wq, self.ws, self.wqt, None)
        else:
            result = int8_lora_fwd(X2, self.wq, self.ws)
        result = result.view(*shape[:-1], result.shape[-1])
        return result if out is None else out.copy_(result)

    def lora_first(
        self,
        X,
        W,
        XA,
        Bt,
        alpha,
        out = None,
    ):
        delta = X.new_empty((X.shape[0], Bt.shape[1]))
        delta.addmm_(XA, Bt, alpha = alpha, beta = 0)
        if not self._use_fp8(X):
            _count("dgrad_16bit" if _is_transposed(W) else "fwd_16bit")
            result = delta.addmm_(X, self._dense(W).t().to(X.dtype))
        elif _is_transposed(W):
            result = int8_lora_dgrad(X, self.wq, self.ws, self.wqt, delta)
        else:
            result = int8_lora_fwd(X, self.wq, self.ws, delta)
        return result if out is None else out.copy_(result)

    def dgrad_accum(self, dY, W, delta):
        if not self._use_fp8(dY):
            _count("dgrad_16bit")
            return delta.addmm_(dY, self._dense(W).to(dY.dtype))
        return int8_lora_dgrad(dY, self.wq, self.ws, self.wqt, delta)

    def dgrad(
        self,
        dY,
        W = None,
    ):
        shape = dY.shape
        dY2 = dY.reshape(-1, shape[-1])
        if not self._use_fp8(dY2):
            _count("dgrad_16bit")
            if W is None and not self.offloaded:
                raise RuntimeError("Unsloth: 16-bit dgrad needs the base weight")
            result = torch.matmul(dY2, self._dense(W).to(dY2.dtype))
        else:
            result = int8_lora_dgrad(dY2, self.wq, self.ws, self.wqt, None)
        return result.view(*shape[:-1], result.shape[-1])


class _Int8FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, wq, ws, wqt):
        ctx.has_t = wqt is not None
        ctx.save_for_backward(wq, ws, wqt if wqt is not None else ws)
        shape = x.shape
        y = int8_lora_fwd(x.reshape(-1, shape[-1]), wq, ws)
        return y.view(*shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, dy):
        wq, ws, wqt = ctx.saved_tensors
        shape = dy.shape
        dx = int8_lora_dgrad(dy.reshape(-1, shape[-1]), wq, ws, wqt if ctx.has_t else None, None)
        return dx.view(*shape[:-1], dx.shape[-1]), None, None, None, None


def _int8_base_forward(self, x):
    """nn.Linear.forward of a converted base when Unsloth's fused LoRA kernels do not run it."""
    state = self.__dict__.get("_unsloth_fp8_lora")
    weight = self.weight
    if x.is_floating_point() and torch.is_autocast_enabled(x.device.type):
        dtype = torch.get_autocast_dtype(x.device.type)  # what F.linear would compute in
        x = x if x.dtype == dtype else x.to(dtype)
    if state is None or weight.requires_grad or not state._use_fp8(x.reshape(-1, x.shape[-1])):
        _count("fwd_16bit")
        if state is not None and state.offloaded:
            if weight.requires_grad:
                state.restore(self)
            else:
                weight = state._dense(weight)
        return torch.nn.functional.linear(x, weight.to(x.dtype), self.bias)
    state.refresh(weight)
    y = _Int8FrozenLinear.apply(x, weight, state.wq, state.ws, state.wqt)
    if self.bias is not None:
        y = y + self.bias.to(y.dtype)
    return y


def _device_refusal(device) -> Optional[str]:
    """Why the layer's own GPU cannot run the int8 base (kernel support, then its recipe gate), else None."""
    why = int8_lora_supported(device)
    if why is not None:
        return why
    from ..precision import get_recipe, gpu_class

    cls = gpu_class(torch.cuda.get_device_capability(device))
    gate = get_recipe("int8_lora_rotated").gate(cls)
    return f"{cls} ({device}): {gate.reason}" if gate.status == "refused" else None


def eligible_base(module) -> Optional[str]:
    why = _fp8.eligible_base(module)
    if why is not None:
        return why
    weight = getattr(module, "weight", None)
    if weight is not None and weight.is_cuda:  # each layer's GPU, not only the current one
        why = _device_refusal(weight.device)
        if why is not None:
            return why
    if module.in_features % GROUP:
        return f"in_features {module.in_features} is not a multiple of {GROUP}"
    return None


def attach_int8_lora_base(module, offload: Optional[bool] = None) -> Int8LoRABaseState:
    """Quantize the frozen base `module` (nn.Linear) once and route its GEMMs through the rotated int8 kernels."""
    import types

    state = module.__dict__.get("_unsloth_fp8_lora")
    if state is None:
        state = Int8LoRABaseState(module.weight)
        module.__dict__["_unsloth_fp8_lora"] = state
        module.forward = types.MethodType(_int8_base_forward, module)
        module._apply = types.MethodType(_fp8_base_apply, module)
        module.__dict__["_unsloth_fp8_lora_load_hook"] = module.register_load_state_dict_post_hook(
            _fp8_requantize_after_load
        )
    if offload is None:
        offload = offload_supported()
    if offload:
        state.offload(module)
    return state
