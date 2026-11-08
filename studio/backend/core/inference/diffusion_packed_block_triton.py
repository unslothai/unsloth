# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton widen for packed nvfp4; helpers are module level because Triton resolves them via globals."""

import triton
import triton.language as tl


@triton.jit
def _e2m1(c):
    # c: int32 nibble -> +-0, 0.5, 1, 1.5, 2, 3, 4, 6, built bit by bit (a float negation would lose -0.0)
    k = c & 7
    e = k >> 1
    m = k & 1
    bits = tl.where(e == 0, m * 0x3F000000, ((e + 126) << 23) | (m << 22))
    return (bits | ((c & 8) << 28)).to(tl.float32, bitcast = True)


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
def _fp16_bits(value):
    """float32 -> float16 bits, round to nearest even by integer ops (overflow to inf, subnormals kept), as torch
    casts on every device; ROCm's own conversion differs."""
    u = value.to(tl.uint32, bitcast = True)
    sign = (u >> 16) & 0x8000
    a = u & 0x7FFFFFFF
    r = a - (112 << 23)
    normal = (r + 0x0FFF + ((r >> 13) & 1)) >> 13
    # below 2**-14: the subnormal count, |x| * 2**24 rounded to the nearest even integer (exact scaling, one add)
    scaled = (a.to(tl.float32, bitcast = True) * 16777216.0 + 8388608.0) - 8388608.0
    sub = scaled.to(tl.uint32)
    bits = tl.where(a >= 0x38800000, normal, sub)
    bits = tl.where(a >= 0x477FF000, 0x7C00, bits)
    # NaN: quieted, top payload bits kept, as the hardware conversion torch uses
    bits = tl.where(a > 0x7F800000, 0x7E00 | ((a >> 13) & 0x3FF), bits)
    return (bits | sign).to(tl.uint16)


@triton.jit
def _store(dst, value, mask):
    """``value`` in the output dtype; bf16 rounds to nearest even by integer ops, as torch does on every device
    (the interpreter, pre-sm_80 and ROCm conversions are not guaranteed to); fp16 the same way."""
    ty = dst.dtype.element_ty
    if ty == tl.bfloat16:
        u = value.to(tl.uint32, bitcast = True)
        rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        rounded = tl.where(value != value, 0x7FC0, rounded)
        tl.store(dst, rounded.to(tl.uint16).to(tl.bfloat16, bitcast = True), mask = mask)
    elif ty == tl.float16:
        tl.store(dst, _fp16_bits(value).to(tl.float16, bitcast = True), mask = mask)
    else:
        tl.store(dst, value.to(ty), mask = mask)


@triton.jit
def nvfp4_widen_kernel(
    q_ptr,
    sf_ptr,
    rs_ptr,
    pq_ptr,
    out_ptr,
    rows,
    half_k,
    blocks,
    HAS_RS: tl.constexpr,
    HAS_PQ: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    j = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    rmask = r < rows
    jmask = j < half_k
    mask = rmask[:, None] & jmask[None, :]
    r64 = r.to(tl.int64)
    j64 = j.to(tl.int64)
    b = tl.load(q_ptr + r64[:, None] * half_k + j64[None, :], mask = mask, other = 0)
    b = b.to(tl.int32)
    s = tl.load(sf_ptr + r64[:, None] * blocks + (j64 // 8)[None, :], mask = mask, other = 0)
    # same multiply order as dequant_block (bit-identical)
    step = _e4m3(s.to(tl.int32))
    if HAS_RS:
        step = step * tl.load(rs_ptr + r64, mask = rmask, other = 1.0)[:, None]
    lo = _e2m1(b & 15) * step
    hi = _e2m1(b >> 4) * step
    if HAS_PQ:
        lo = lo * tl.load(pq_ptr + 2 * j64, mask = jmask, other = 1.0)[None, :]
        hi = hi * tl.load(pq_ptr + 2 * j64 + 1, mask = jmask, other = 1.0)[None, :]
    # even column = low nibble
    w = tl.reshape(tl.join(lo, hi), (BLOCK_R, 2 * BLOCK_C))
    c = tl.program_id(1).to(tl.int64) * (2 * BLOCK_C) + tl.arange(0, 2 * BLOCK_C)
    dst = out_ptr + r64[:, None] * (2 * half_k) + c[None, :]
    _store(dst, w, rmask[:, None] & (c < 2 * half_k)[None, :])
