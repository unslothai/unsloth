# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton kernel for ``diffusion_fp8_dequant`` (module level: a compiled kernel resolves ``tl`` via globals)."""

import triton
import triton.language as tl


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
def dequant_rows_kernel(
    codes_ptr,
    scale_ptr,
    out_ptr,
    cols,
    FMT: tl.constexpr,
    BF16: tl.constexpr,
    BF16_NAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = col < cols
    offs = row * cols + col
    s = tl.load(scale_ptr + row)
    c = tl.load(codes_ptr + offs, mask = mask, other = 0)
    if FMT == 0:
        wide = c.to(tl.float32)
    else:
        # fp8_bits_to_float32 by integer ops (no fp8 type: sm75, ROCm); NaN bits as torch's own cast.
        b = c.to(tl.int32)
        if FMT == 1:
            e = (b >> 3) & 15
            m = b & 7
            normal = ((e + 120) << 23) | (m << 20)
            sub = (m.to(tl.float32) * 0.001953125).to(tl.int32, bitcast = True)
            special = (e == 15) & (m == 7)
            special_bits = tl.full(b.shape, 0x7FF00000, tl.int32)
        else:
            e = (b >> 2) & 31
            m = b & 3
            normal = ((e + 112) << 23) | (m << 21)
            sub = (m.to(tl.float32) * 0.0000152587890625).to(tl.int32, bitcast = True)
            special = e == 31
            special_bits = 0x7F800000 | (m << 21) | tl.where(m != 0, 0x00400000, 0)
        mag = tl.where(e == 0, sub, normal)
        mag = tl.where(special, special_bits, mag)
        wide = (mag | ((b & 0x80) << 24)).to(tl.float32, bitcast = True)
    prod = wide * s
    if BF16:
        # nearest even by integer ops (the interpreter truncates); NaN as torch's cast on this device
        bits = prod.to(tl.int32, bitcast = True)
        rne = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        out = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, BF16_NAN, rne).to(tl.int16)
        out = out.to(tl.bfloat16, bitcast = True)
    else:
        if out_ptr.dtype.element_ty == tl.float16:
            # nearest even by integer ops: ROCm's own fp32 -> fp16 conversion differs from torch's
            out = _fp16_bits(prod).to(tl.float16, bitcast = True)
        else:
            out = prod.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, out, mask = mask)
