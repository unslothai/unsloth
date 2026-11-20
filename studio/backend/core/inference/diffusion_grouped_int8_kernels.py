# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton kernel behind ``diffusion_grouped_int8_triton``; helpers are module level because Triton resolves them via
globals. Imported only once Triton is known to import."""

import triton
import triton.language as tl


@triton.jit
def _e4m3(s):
    # s: int32 byte of a float8_e4m3fn -> its exact float32 (NaN for 0x7f / 0xff), sign as a bit
    e = (s >> 3) & 15
    m = s & 7
    normal = ((e + 120) << 23) | (m << 20)
    sub = (m.to(tl.float32) * 0.001953125).to(tl.int32, bitcast = True)
    bits = tl.where(e == 0, sub, normal)
    bits = tl.where((e == 15) & (m == 7), 0x7FF00000, bits)
    return (bits | ((s & 128) << 24)).to(tl.float32, bitcast = True)


@triton.jit
def _rint_clamp(x):
    # round half to even, then clamp to [-127, 127]; clamping to +-128 first changes no result and keeps
    # the truncating int conversion in range
    x = tl.minimum(tl.maximum(x, -128.0), 128.0)
    t = x.to(tl.int32)
    frac = x - t.to(tl.float32)
    a = tl.abs(frac)
    up = (a > 0.5) | ((a == 0.5) & ((t & 1) != 0))
    q = t + tl.where(up, tl.where(x < 0, -1, 1), 0)
    return tl.minimum(tl.maximum(q, -127), 127)


@triton.jit
def grouped_decode_kernel(
    q_ptr,
    sf_ptr,
    cb_ptr,
    out_ptr,
    rows,
    k,
    cols,
    groups,
    GROUP: tl.constexpr,
    BITS: tl.constexpr,
    HAS_CB: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (r < rows)[:, None] & (c < k)[None, :]
    r64 = r.to(tl.int64)
    c64 = c.to(tl.int64)
    row_q = r64[:, None] * cols
    b = tl.load(q_ptr + row_q + (c64 // 2)[None, :], mask = mask, other = 0).to(tl.int32)
    code = (b >> ((c & 1) * 4)[None, :]) & 15
    if BITS == 6:
        hb = tl.load(q_ptr + row_q + (k // 2 + c64 // 4)[None, :], mask = mask, other = 0)
        code = code | (((hb.to(tl.int32) >> ((c & 3) * 2)[None, :]) & 3) << 4)
    if HAS_CB:
        level = tl.load(cb_ptr + code, mask = mask, other = 0.0)
    else:
        level = (code - (1 << (BITS - 1))).to(tl.float32)
    s = tl.load(sf_ptr + r64[:, None] * groups + (c64 // GROUP)[None, :], mask = mask, other = 0)
    q = _rint_clamp(level * _e4m3(s.to(tl.int32)))
    tl.store(out_ptr + r64[:, None] * k + c64[None, :], q.to(tl.int8), mask = mask)


@triton.jit
def int8_epilogue_kernel(
    acc_ptr,
    xs_ptr,
    ws_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # torchao's epilogue on a torch._int_mm result, every rounding kept (as Studio's int8 GEMM epilogue with fp32
    # scales): fp32(acc) * xs -> bf16, * ws in fp32, + bias in fp32, -> bf16
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = rm < M
    mn = rn < N
    mask = mm[:, None] & mn[None, :]
    offs = rm.to(tl.int64)[:, None] * N + rn[None, :]
    acc = tl.load(acc_ptr + offs, mask = mask, other = 0)
    xs = tl.load(xs_ptr + rm, mask = mm, other = 0.0)
    ws = tl.load(ws_ptr + rn, mask = mn, other = 0.0)
    y = tl.extra.cuda.libdevice.mul_rn(acc.to(tl.float32), xs[:, None])
    y = y.to(tl.bfloat16).to(tl.float32)
    y = tl.extra.cuda.libdevice.mul_rn(y, ws[None, :])
    if HAS_BIAS:
        y = tl.extra.cuda.libdevice.add_rn(
            y, tl.load(b_ptr + rn, mask = mn, other = 0.0).to(tl.float32)[None, :]
        )
    tl.store(out_ptr + offs, y.to(tl.bfloat16), mask = mask)
