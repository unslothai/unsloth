# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton kernel behind ``diffusion_grouped_int8_gemm``: int8 activations times packed W4A8 / W6A8 weights, the
weight tile decoded to int8 in registers inside the K loop. Imported only once Triton is known to import.

Layout: see ``diffusion_grouped_int8``. The sum over K is order free, so the phase loop splits each K block into the
four column phases ``c = 4j + t``: no unpacked tile is interleaved or gathered."""

import triton
import triton.language as tl

from .diffusion_grouped_int8_kernels import _e4m3, _rint_clamp


@triton.jit
def _rbf16(x, RN: tl.constexpr):
    if RN:
        return x.to(tl.bfloat16).to(tl.float32)
    # round to nearest even by integer ops (finite values): the interpreter's bf16 cast truncates
    b = x.to(tl.int32, bitcast = True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & -65536
    return b.to(tl.float32, bitcast = True)


@triton.jit
def _step_f32(sb, HW: tl.constexpr):
    """e4m3 step bytes -> their exact float32 (hardware convert on sm89+, integer ops elsewhere)."""
    if HW:
        return sb.to(tl.float8e4nv, bitcast = True).to(tl.float32)
    return _e4m3(sb.to(tl.int32))


@triton.jit
def _to_int8(code, step, cb_ptr, BITS: tl.constexpr, HAS_CB: tl.constexpr, HW: tl.constexpr):
    """int32 codes and their float32 steps -> the int8 grid: clamp(rint(level * step), -127, 127)."""
    if HAS_CB:
        level = tl.load(cb_ptr + code)
    else:
        level = (code - (1 << (BITS - 1))).to(tl.float32)
    if HW:
        # cvt.rni (half to even) straight to int32; clamping after rounding to the integer bounds is the same
        q = tl.extra.cuda.libdevice.float2int_rn(
            tl.minimum(tl.maximum(level * step, -128.0), 128.0)
        )
        return tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    return _rint_clamp(level * step).to(tl.int8)


@triton.jit
def _levels_to_int8(code, sb, cb_ptr, BITS: tl.constexpr, HAS_CB: tl.constexpr, HW: tl.constexpr):
    return _to_int8(code, _step_f32(sb, HW), cb_ptr, BITS, HAS_CB, HW)


@triton.jit
def _tile_loop(
    a_ptr,
    q_ptr,
    s_ptr,
    cb_ptr,
    rm,
    rn,
    K,
    stride_am,
    stride_qn,
    groups,
    BITS: tl.constexpr,
    GROUP: tl.constexpr,
    HAS_CB: tl.constexpr,
    HW: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * stride_am + offs_k[None, :]
    row_q = q_ptr + rn[:, None] * stride_qn
    s_ptrs = s_ptr + rn[:, None] * groups + (offs_k // GROUP)[None, :]
    if GATHER:
        lo_ptrs = row_q + (offs_k // 2)[None, :]
        hi_ptrs = row_q + K // 2 + (offs_k // 4)[None, :]
    else:
        lo_ptrs = row_q + tl.arange(0, BLOCK_K // 2)[None, :]
        hi_ptrs = row_q + K // 2 + tl.arange(0, BLOCK_K // 4)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.int32)
    for _k in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs)
        if GATHER:
            code = (tl.load(lo_ptrs).to(tl.int32) >> ((offs_k & 1) * 4)[None, :]) & 15
            if BITS == 6:
                code = code | (
                    ((tl.load(hi_ptrs).to(tl.int32) >> ((offs_k & 3) * 2)[None, :]) & 3) << 4
                )
        else:
            lb = tl.load(lo_ptrs)
            code = tl.reshape(tl.join(lb & 15, lb >> 4), (BLOCK_N, BLOCK_K)).to(tl.int32)
            if BITS == 6:
                hb = tl.load(hi_ptrs)
                top = tl.reshape(
                    tl.join(tl.join(hb & 3, (hb >> 4) & 3), tl.join((hb >> 2) & 3, (hb >> 6) & 3)),
                    (BLOCK_N, BLOCK_K),
                )
                code = code | (top.to(tl.int32) << 4)
        w = _levels_to_int8(code, tl.load(s_ptrs), cb_ptr, BITS, HAS_CB, HW)
        acc = tl.dot(a, tl.trans(w), acc, out_dtype = tl.int32)
        a_ptrs += BLOCK_K
        lo_ptrs += BLOCK_K // 2
        hi_ptrs += BLOCK_K // 4
        s_ptrs += BLOCK_K // GROUP
    return acc


@triton.jit
def _phase_code(nib, hi, T: tl.constexpr, BITS: tl.constexpr):
    code = (nib >> (4 * (T % 2))) & 15
    if BITS == 6:
        code = code | (((hi >> (2 * T)) & 3) << 4)
    return code


@triton.jit
def _phase_dot(
    acc,
    a_ptrs,
    a_mask,
    nib,
    hi,
    step,
    cb_ptr,
    T: tl.constexpr,
    BITS: tl.constexpr,
    HAS_CB: tl.constexpr,
    HW: tl.constexpr,
):
    w = _to_int8(_phase_code(nib, hi, T, BITS), step, cb_ptr, BITS, HAS_CB, HW)
    a = tl.load(a_ptrs + T, mask = a_mask, other = 0)
    return tl.dot(a, tl.trans(w), acc, out_dtype = tl.int32)


@triton.jit
def _phase_loop(
    a_ptr,
    q_ptr,
    s_ptr,
    cb_ptr,
    pid_m,
    pid_n,
    M,
    N,
    K,
    stride_am,
    stride_qn,
    groups,
    rn,
    k_first,
    k_steps,
    BITS: tl.constexpr,
    GROUP: tl.constexpr,
    HAS_CB: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    Q: tl.constexpr = BLOCK_K // 4
    # Plain pointers, not block pointers: Triton 3.9 removed tl.make_block_ptr / tl.advance (triton-lang/triton#10833).
    # Same bytes as the block pointers loaded: rows past M / N read as zeros (never stored), columns stay inside K.
    rows_m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_n = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = (rows_m < M)[:, None]
    mask_n = (rows_n < N)[:, None]
    jj = tl.arange(0, Q)
    col = (k_first * Q + jj).to(tl.int64)
    # activations at columns 4j + t: _phase_dot adds t
    a_ptrs = a_ptr + rows_m[:, None] * stride_am + (4 * col)[None, :]
    # nibble bytes 2j (phases 0, 1) and 2j + 1 (phases 2, 3), top-plane byte j
    q_rows = q_ptr + rows_n[:, None] * stride_qn
    ne_ptrs = q_rows + (2 * col)[None, :]
    hp_ptrs = q_rows + K // 2 + col[None, :]
    s_ptrs = s_ptr + rn[:, None] * groups + ((k_first * BLOCK_K + 4 * jj) // GROUP)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.int32)
    for _k in range(0, k_steps):
        step = _step_f32(tl.load(s_ptrs), HW)  # one convert serves the four phases
        nib_e = tl.load(ne_ptrs, mask = mask_n, other = 0).to(tl.int32)
        nib_o = tl.load(ne_ptrs + 1, mask = mask_n, other = 0).to(tl.int32)
        if BITS == 6:
            hi = tl.load(hp_ptrs, mask = mask_n, other = 0).to(tl.int32)
        else:
            hi = nib_e
        acc = _phase_dot(acc, a_ptrs, mask_m, nib_e, hi, step, cb_ptr, 0, BITS, HAS_CB, HW)
        acc = _phase_dot(acc, a_ptrs, mask_m, nib_e, hi, step, cb_ptr, 1, BITS, HAS_CB, HW)
        acc = _phase_dot(acc, a_ptrs, mask_m, nib_o, hi, step, cb_ptr, 2, BITS, HAS_CB, HW)
        acc = _phase_dot(acc, a_ptrs, mask_m, nib_o, hi, step, cb_ptr, 3, BITS, HAS_CB, HW)
        a_ptrs += BLOCK_K
        ne_ptrs += BLOCK_K // 2
        hp_ptrs += Q
        s_ptrs += BLOCK_K // GROUP
    return acc


@triton.jit(do_not_specialize = ["M"])
def grouped_mm_dq(
    a_ptr,
    q_ptr,
    s_ptr,
    cb_ptr,
    xs_ptr,
    ws_ptr,
    b_ptr,
    c_ptr,
    p_ptr,
    M,
    N,
    K,
    stride_am,
    stride_qn,
    groups,
    stride_cm,
    BITS: tl.constexpr,  # 4 or 6
    GROUP: tl.constexpr,  # weights per s_rel step (a multiple of 4)
    HAS_CB: tl.constexpr,  # 4-bit codebook levels
    HAS_BIAS: tl.constexpr,
    XS_FP32: tl.constexpr,
    WS_FP32: tl.constexpr,
    RN: tl.constexpr,  # libdevice *_rn epilogue (off only under the interpreter, which lacks them)
    HW: tl.constexpr,  # hardware e4m3 -> fp32 convert and rint (sm89+, never under the interpreter)
    GATHER: tl.constexpr,  # 2 = column phases, 1 = per-weight byte gathers, 0 = byte tiles + join
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # divides K, a multiple of GROUP and of 8 (of 64 for the phases)
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,  # > 1: K blocks split over grid axis 1 into the int32 buffer p (phase loop only)
):
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
    rm = offs_m % M
    rn = offs_n % N
    if GATHER == 2:
        steps = K // BLOCK_K
        per = tl.cdiv(steps, SPLIT_K)
        k_first = tl.program_id(1) * per
        acc = _phase_loop(
            a_ptr, q_ptr, s_ptr, cb_ptr, pid_m, pid_n, M, N, K, stride_am, stride_qn, groups, rn,
            k_first, tl.minimum(per, steps - k_first),
            BITS, GROUP, HAS_CB, HW, BLOCK_M, BLOCK_N, BLOCK_K,
        )  # fmt: skip
    else:
        acc = _tile_loop(
            a_ptr, q_ptr, s_ptr, cb_ptr, rm, rn, K, stride_am, stride_qn, groups,
            BITS, GROUP, HAS_CB, HW, GATHER, BLOCK_M, BLOCK_N, BLOCK_K,
        )  # fmt: skip

    if SPLIT_K == 1:
        _epilogue(acc, xs_ptr, ws_ptr, b_ptr, c_ptr, offs_m, offs_n, rm, rn, M, N, stride_cm,
                  HAS_BIAS, XS_FP32, WS_FP32, RN)  # fmt: skip
    else:
        # int32 partial sums add exactly in any order; grouped_epilogue applies the epilogue afterwards
        p_ptrs = p_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.atomic_add(p_ptrs, acc, mask = mask, sem = "relaxed")


@triton.jit
def _epilogue(
    acc, xs_ptr, ws_ptr, b_ptr, c_ptr, offs_m, offs_n, rm, rn, M, N, stride_cm,
    HAS_BIAS: tl.constexpr, XS_FP32: tl.constexpr, WS_FP32: tl.constexpr, RN: tl.constexpr,
):  # fmt: skip
    """Studio's int8 GEMM epilogue (torchao's eager order), every rounding kept."""
    xs = tl.load(xs_ptr + rm).to(tl.float32)
    ws = tl.load(ws_ptr + rn).to(tl.float32)
    if XS_FP32:
        y = acc.to(tl.float32)
    else:
        if RN:
            y = _rbf16(tl.extra.cuda.libdevice.int2float_rn(acc), RN)
        else:
            y = _rbf16(acc.to(tl.float32), RN)
    y = _rbf16(y * xs[:, None], RN)
    if RN:
        y = tl.extra.cuda.libdevice.mul_rn(y, ws[None, :])
    else:
        y = y * ws[None, :]
    if not WS_FP32:
        y = _rbf16(y, RN)
    if HAS_BIAS:
        bias = tl.load(b_ptr + rn).to(tl.float32)[None, :]
        if RN:
            y = tl.extra.cuda.libdevice.add_rn(y, bias)
        else:
            y = y + bias
    if not RN:
        y = _rbf16(y, RN)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, y.to(tl.bfloat16), mask = mask)


@triton.jit
def grouped_epilogue(
    p_ptr, xs_ptr, ws_ptr, b_ptr, c_ptr, M, N, stride_cm,
    HAS_BIAS: tl.constexpr, XS_FP32: tl.constexpr, WS_FP32: tl.constexpr, RN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):  # fmt: skip
    offs_m = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = offs_m % M
    rn = offs_n % N
    acc = tl.load(p_ptr + rm[:, None] * N + rn[None, :])
    _epilogue(acc, xs_ptr, ws_ptr, b_ptr, c_ptr, offs_m, offs_n, rm, rn, M, N, stride_cm,
              HAS_BIAS, XS_FP32, WS_FP32, RN)  # fmt: skip
