# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Triton blockwise NF4 / FP4 4-bit and 8-bit dynamic quantization for Unsloth's bitsandbytes
stand-in (unsloth/bnb_shim), byte-identical to bitsandbytes' CUDA kernels on NVIDIA.

bitsandbytes builds with --use_fast_math: absmax = max(|x|) with subnormals flushed, the scale is
rcp.approx.ftz(absmax) and the product x * scale is mul.ftz; NF4 / FP4 then code the product with
`>` comparisons on the thresholds as compiled. These kernels issue the same PTX through inline asm
(so fp fusion cannot change them either way; launches still pass enable_fp_fusion=True like every
bnb-exact kernel). ROCm and TRITON_INTERPRET=1 run the same algorithm with IEEE ops.
Imported lazily on first quantization (never at `import unsloth`); no GPU query at import."""

import os

import torch
import triton
import triton.language as tl

from .triton_launch import tag_compile_cache

__all__ = ["quantize_4bit_triton", "quantize_8bit_triton"]

_EXACT = torch.version.hip is None and os.environ.get("TRITON_INTERPRET", "0") != "1"


@triton.jit
def _t_rcp(x, EXACT: tl.constexpr):  # EXACT: per process (see _quant4_kernel)
    if EXACT:
        return tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype = tl.float32, is_pure = True, pack = 1
        )
    else:
        return 1.0 / x


@triton.jit
def _t_mul(a, b, EXACT: tl.constexpr):  # EXACT: per process (see _quant4_kernel)
    if EXACT:
        return tl.inline_asm_elementwise(
            "mul.ftz.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype = tl.float32, is_pure = True, pack = 1
        )
    else:
        return a * b


@triton.jit
def _t_ftz(x):
    return tl.where(tl.abs(x) < 1.1754943508222875e-38, 0.0, x)


@triton.jit
def _t_absmax_input(x):
    # |x| with subnormals flushed; NaN -> -FLT_MAX, which fmaxf from -FLT_MAX ignores.
    a = _t_ftz(tl.abs(x))
    return tl.where(a != a, -3.4028234663852886e38, a)


@triton.jit
def _t_nf4(x, T_ptr):
    q = tl.zeros(x.shape, dtype = tl.int32)
    for k in tl.static_range(15):
        q += (x > tl.load(T_ptr + k)).to(tl.int32)
    return q


@triton.jit
def _t_fp4(x, T_ptr):
    a = tl.abs(x)
    b = tl.zeros(x.shape, dtype = tl.int32)
    for k in tl.static_range(7):
        b += (a > tl.load(T_ptr + k)).to(tl.int32)
    # bucket -> code: 0,1,6,7,4,5,2,3
    code = tl.where(b < 2, b, tl.where(b < 4, b + 4, tl.where(b < 6, b, b - 4)))
    sign = tl.where(x < 0, 8, 0)
    return code + sign


# n, n_bytes and n_blocks only bound masks: one compiled variant per (dtype, blocksize, type).
@triton.jit(do_not_specialize = ["n", "n_bytes", "n_blocks"])
def _quant4_kernel(
    A,
    OUT,
    ABSMAX,
    T_ptr,
    n,
    n_bytes,
    n_blocks,
    BLOCK: tl.constexpr,  # quantization blocksize (64 for every bnb-4bit checkpoint): tile width
    ROWS: tl.constexpr,  # blocks per program, derived from BLOCK
    NF4: tl.constexpr,  # code table (nf4 / fp4): fixed per weight
    EXACT: tl.constexpr,  # PTX rcp.approx / mul.ftz (NVIDIA) vs IEEE (ROCm, interpreter): per process
):
    HALF: tl.constexpr = BLOCK // 2
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, HALF)
    even = rows[:, None] * BLOCK + 2 * cols[None, :]
    x0 = tl.load(A + even, mask = even < n, other = 0.0).to(tl.float32)
    x1 = tl.load(A + even + 1, mask = (even + 1) < n, other = 0.0).to(tl.float32)
    a0 = _t_absmax_input(x0)
    a1 = _t_absmax_input(x1)
    absmax = tl.maximum(tl.max(a0, axis = 1), tl.max(a1, axis = 1))
    tl.store(ABSMAX + rows, absmax, mask = rows < n_blocks)
    r = _t_rcp(absmax, EXACT)[:, None]
    s0 = _t_mul(x0, r, EXACT)
    s1 = _t_mul(x1, r, EXACT)
    if not EXACT:
        s0 = _t_ftz(s0)
        s1 = _t_ftz(s1)
    if NF4:
        q0 = _t_nf4(s0, T_ptr)
        q1 = _t_nf4(s1, T_ptr)
    else:
        q0 = _t_fp4(s0, T_ptr)
        q1 = _t_fp4(s1, T_ptr)
    byte = ((q0 << 4) | q1).to(tl.uint8)
    offs = rows[:, None] * HALF + cols[None, :]
    tl.store(OUT + offs, byte, mask = offs < n_bytes)


@triton.jit(do_not_specialize = ["n", "n_blocks"])
def _quant8_kernel(
    A,
    OUT,
    ABSMAX,
    CODE,
    n,
    n_blocks,
    BLOCK: tl.constexpr,  # double-quant blocksize (256 in bitsandbytes): tile width
    ROWS: tl.constexpr,  # blocks per program, derived from BLOCK
    EXACT: tl.constexpr,  # PTX rcp.approx / mul.ftz (NVIDIA) vs IEEE (ROCm, interpreter): per process
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    offs = rows[:, None] * BLOCK + cols[None, :]
    mask = offs < n
    x = tl.load(A + offs, mask = mask, other = 0.0).to(tl.float32)
    absmax = tl.max(_t_absmax_input(x), axis = 1)
    tl.store(ABSMAX + rows, absmax, mask = rows < n_blocks)
    x = _t_mul(x, _t_rcp(absmax, EXACT)[:, None], EXACT)
    if not EXACT:
        x = _t_ftz(x)
    pivot = tl.full(x.shape, 127, tl.int32)
    upper_pivot = tl.full(x.shape, 255, tl.int32)
    lower_pivot = tl.zeros(x.shape, tl.int32)
    lower = tl.full(x.shape, -1.0, tl.float32)
    upper = tl.full(x.shape, 1.0, tl.float32)
    val = tl.load(CODE + pivot)
    step = 64
    for _ in tl.static_range(7):
        gt = x > val
        lower_pivot = tl.where(gt, pivot, lower_pivot)
        lower = tl.where(gt, val, lower)
        upper_pivot = tl.where(gt, upper_pivot, pivot)
        upper = tl.where(gt, upper, val)
        pivot = tl.where(gt, pivot + step, pivot - step)
        val = tl.load(CODE + pivot)
        step = step // 2
    upper = tl.where(upper_pivot == 255, tl.load(CODE + 255), upper)
    lower = tl.where(lower_pivot == 0, tl.load(CODE), lower)
    gt = x > val
    up = (upper + val) * 0.5
    down = (lower + val) * 0.5
    q = tl.where(gt, tl.where(x > up, upper_pivot, pivot), tl.where(x < down, lower_pivot, pivot))
    tl.store(OUT + offs, q.to(tl.uint8), mask = mask)


tag_compile_cache(__file__)


def quantize_4bit_triton(A, thresholds, blocksize, nf4):
    """A contiguous on a CUDA / ROCm device: (packed uint8 [(n + 1) // 2], fp32 absmax [blocks])."""
    n = A.numel()
    blocks = -(n // -blocksize)
    n_bytes = (n + 1) // 2
    out = torch.empty(n_bytes, dtype = torch.uint8, device = A.device)
    absmax = torch.empty(blocks, dtype = torch.float32, device = A.device)
    if n == 0:
        return out, absmax
    rows = max(1, min(64, 4096 // blocksize))
    # Launch on the operand's device, not the current one.
    with torch.cuda.device(A.device):
        _quant4_kernel[(-(blocks // -rows),)](
            A,
            out,
            absmax,
            thresholds,
            n,
            n_bytes,
            blocks,
            BLOCK = blocksize,
            ROWS = rows,
            NF4 = nf4,
            EXACT = _EXACT,
            enable_fp_fusion = True,
        )
    return out, absmax


def quantize_8bit_triton(A, code, blocksize):
    """A contiguous, code the fp32 [256] dynamic map: (uint8 codes like A, fp32 absmax [blocks])."""
    n = A.numel()
    blocks = -(n // -blocksize)
    out = torch.empty(A.shape, dtype = torch.uint8, device = A.device)
    absmax = torch.empty(blocks, dtype = torch.float32, device = A.device)
    if n == 0:
        return out, absmax
    rows = max(1, min(16, 4096 // blocksize))
    with torch.cuda.device(A.device):
        _quant8_kernel[(-(blocks // -rows),)](
            A,
            out,
            absmax,
            code,
            n,
            blocks,
            BLOCK = blocksize,
            ROWS = rows,
            EXACT = _EXACT,
            enable_fp_fusion = True,
        )
    return out, absmax
