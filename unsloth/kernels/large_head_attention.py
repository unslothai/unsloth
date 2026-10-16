# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Flash attention for head_dim > 256 (Gemma-4 global layers), where FA2 and SDPA's
# flash / cuDNN backends stop. Tiling follows axolotl's flash_attn_d512.py (Apache-2.0,
# https://github.com/axolotl-ai-cloud/axolotl); this one reads GQA kv heads in place and
# takes any mask whose allowed keys per row are one interval [lo, hi], lo and hi
# non-decreasing down the rows (causal, packed, right padding, sliding window).
# UNSLOTH_HD512_ATTENTION=0 leaves SDPA untouched.

import functools
import os
import sys

import torch
import triton
import triton.language as tl

__all__ = [
    "large_head_attention",
    "patch_large_head_attention",
    "intervals_from_mask",
    "LARGE_HEAD_ENGAGED",
]

# calls: kernel ran; fallback: a qualifying call went back to SDPA after an error.
LARGE_HEAD_ENGAGED = {"calls": 0, "fallback": 0}
_SUPPORTED_HEAD_DIMS = (512,)
_INT32_MAX = 2**31 - 1


def _fwd_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps = w, num_stages = s)
        for bm, bn, w, s in (
            (32, 32, 4, 1),
            (32, 32, 4, 2),
            (32, 16, 4, 1),
            (16, 16, 4, 1),
        )
    ]


def _bwd_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps = w, num_stages = 1)
        for bm, bn, w in (
            (16, 32, 4),
            (16, 32, 8),
            (32, 32, 8),
            (16, 16, 4),
        )
    ]


@triton.autotune(configs = _fwd_configs(), key = ["HEAD_DIM", "G"])
@triton.jit
def _lh_fwd(
    Q,
    K,
    V,
    O,
    LSE,
    LO,
    HI,
    qk_scale,
    sqb,
    sqh,
    sqm,
    skb,
    skh,
    skn,
    svb,
    svh,
    svn,
    sob,
    soh,
    som,
    H,
    G,
    S,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = (off_bh // H).to(tl.int64)
    off_h = (off_bh % H).to(tl.int64)
    off_hk = off_h // G
    Q += off_b * sqb + off_h * sqh
    K += off_b * skb + off_hk * skh
    V += off_b * svb + off_hk * svh
    O += off_b * sob + off_h * soh
    LO += off_b * S
    HI += off_b * S

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < S
    lo = tl.load(LO + offs_m, mask = row_ok, other = S)
    hi = tl.load(HI + offs_m, mask = row_ok, other = -1)
    # lo / hi are non-decreasing, so the block's key range is [lo[first], hi[last]].
    n_start = (tl.load(LO + start_m * BLOCK_M) // BLOCK_N) * BLOCK_N
    n_end = tl.load(HI + tl.minimum(start_m * BLOCK_M + BLOCK_M, S) - 1) + 1

    q = tl.load(Q + offs_m[:, None] * sqm + offs_d[None, :], mask = row_ok[:, None], other = 0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    for start_n in range(n_start, n_end, BLOCK_N):
        cur = start_n + offs_n
        col_ok = cur < S
        k = tl.load(K + cur[None, :] * skn + offs_d[:, None], mask = col_ok[None, :], other = 0.0)
        s = tl.dot(q, k) * qk_scale
        keep = (cur[None, :] >= lo[:, None]) & (cur[None, :] <= hi[:, None])
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # A row with no allowed key yet keeps m = -inf; exp2(-inf - -inf) would be NaN.
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_use[:, None])
        alpha = tl.exp2(m_i - m_use)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + cur[:, None] * svn + offs_d[None, :], mask = col_ok[:, None], other = 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        O + offs_m[:, None] * som + offs_d[None, :],
        acc.to(O.dtype.element_ty),
        mask = row_ok[:, None],
    )
    tl.store(LSE + off_bh.to(tl.int64) * S + offs_m, m_i + tl.log2(l_safe), mask = row_ok)


@triton.jit
def _lh_bwd_pre(
    O,
    DO,
    DELTA,
    sob,
    soh,
    som,
    sdob,
    sdoh,
    sdom,
    H,
    S,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = (off_bh // H).to(tl.int64)
    off_h = (off_bh % H).to(tl.int64)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < S
    o = tl.load(
        O + off_b * sob + off_h * soh + offs_m[:, None] * som + offs_d[None, :],
        mask = row_ok[:, None],
        other = 0.0,
    )
    do = tl.load(
        DO + off_b * sdob + off_h * sdoh + offs_m[:, None] * sdom + offs_d[None, :],
        mask = row_ok[:, None],
        other = 0.0,
    )
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    tl.store(DELTA + off_bh.to(tl.int64) * S + offs_m, delta, mask = row_ok)


@triton.autotune(configs = _bwd_configs(), key = ["HEAD_DIM", "G"])
@triton.jit
def _lh_bwd_dq(
    Q,
    K,
    V,
    DO,
    DQ,
    LSE,
    DELTA,
    LO,
    HI,
    qk_scale,
    sm_scale,
    sqb,
    sqh,
    sqm,
    skb,
    skh,
    skn,
    svb,
    svh,
    svn,
    sdob,
    sdoh,
    sdom,
    sdqb,
    sdqh,
    sdqm,
    H,
    G,
    S,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = (off_bh // H).to(tl.int64)
    off_h = (off_bh % H).to(tl.int64)
    off_hk = off_h // G
    Q += off_b * sqb + off_h * sqh
    K += off_b * skb + off_hk * skh
    V += off_b * svb + off_hk * svh
    DO += off_b * sdob + off_h * sdoh
    DQ += off_b * sdqb + off_h * sdqh
    LO += off_b * S
    HI += off_b * S

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < S
    lo = tl.load(LO + offs_m, mask = row_ok, other = S)
    hi = tl.load(HI + offs_m, mask = row_ok, other = -1)
    n_start = (tl.load(LO + start_m * BLOCK_M) // BLOCK_N) * BLOCK_N
    n_end = tl.load(HI + tl.minimum(start_m * BLOCK_M + BLOCK_M, S) - 1) + 1

    q = tl.load(Q + offs_m[:, None] * sqm + offs_d[None, :], mask = row_ok[:, None], other = 0.0)
    do = tl.load(DO + offs_m[:, None] * sdom + offs_d[None, :], mask = row_ok[:, None], other = 0.0)
    lse = tl.load(LSE + off_bh.to(tl.int64) * S + offs_m, mask = row_ok, other = 0.0)
    delta = tl.load(DELTA + off_bh.to(tl.int64) * S + offs_m, mask = row_ok, other = 0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    for start_n in range(n_start, n_end, BLOCK_N):
        cur = start_n + offs_n
        col_ok = cur < S
        k = tl.load(K + cur[:, None] * skn + offs_d[None, :], mask = col_ok[:, None], other = 0.0)
        v = tl.load(V + cur[:, None] * svn + offs_d[None, :], mask = col_ok[:, None], other = 0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        keep = (cur[None, :] >= lo[:, None]) & (cur[None, :] <= hi[:, None])
        p = tl.where(keep, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    dq = dq * sm_scale
    tl.store(
        DQ + offs_m[:, None] * sdqm + offs_d[None, :],
        dq.to(DQ.dtype.element_ty),
        mask = row_ok[:, None],
    )


@triton.autotune(configs = _bwd_configs(), key = ["HEAD_DIM", "G"])
@triton.jit
def _lh_bwd_dkdv(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    LSE,
    DELTA,
    LO,
    HI,
    QLO,
    QHI,
    qk_scale,
    sm_scale,
    sqb,
    sqh,
    sqm,
    skb,
    skh,
    skn,
    svb,
    svh,
    svn,
    sdob,
    sdoh,
    sdom,
    sdkb,
    sdkh,
    sdkn,
    sdvb,
    sdvh,
    sdvn,
    HKV,
    G,
    S,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program owns a kv block of one kv head and loops over its G query heads,
    # so GQA needs no atomics and no repeated K / V.
    start_n = tl.program_id(0)
    off_bkh = tl.program_id(1)
    off_b = (off_bkh // HKV).to(tl.int64)
    off_hk = (off_bkh % HKV).to(tl.int64)
    H = HKV * G
    K += off_b * skb + off_hk * skh
    V += off_b * svb + off_hk * svh
    DK += off_b * sdkb + off_hk * sdkh
    DV += off_b * sdvb + off_hk * sdvh
    LO += off_b * S
    HI += off_b * S
    QLO += off_b * S
    QHI += off_b * S

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    col_ok = offs_n < S
    k = tl.load(K + offs_n[:, None] * skn + offs_d[None, :], mask = col_ok[:, None], other = 0.0)
    v = tl.load(V + offs_n[:, None] * svn + offs_d[None, :], mask = col_ok[:, None], other = 0.0)
    # Query rows attending any key of this block: [QLO[first key], QHI[last key]].
    m_start = tl.load(QLO + start_n * BLOCK_N)
    m_end = tl.load(QHI + tl.minimum(start_n * BLOCK_N + BLOCK_N, S) - 1) + 1
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    for g in range(G):
        off_h = off_hk * G + g
        Qh = Q + off_b * sqb + off_h * sqh
        DOh = DO + off_b * sdob + off_h * sdoh
        row_base = (off_b * H + off_h) * S
        for start_m in range(m_start, m_end, BLOCK_M):
            cur_m = start_m + offs_m
            row_ok = cur_m < S
            q = tl.load(
                Qh + cur_m[:, None] * sqm + offs_d[None, :], mask = row_ok[:, None], other = 0.0
            )
            do = tl.load(
                DOh + cur_m[:, None] * sdom + offs_d[None, :], mask = row_ok[:, None], other = 0.0
            )
            lse = tl.load(LSE + row_base + cur_m, mask = row_ok, other = 0.0)
            delta = tl.load(DELTA + row_base + cur_m, mask = row_ok, other = 0.0)
            lo = tl.load(LO + cur_m, mask = row_ok, other = S)
            hi = tl.load(HI + cur_m, mask = row_ok, other = -1)
            s = tl.dot(q, tl.trans(k)) * qk_scale
            keep = (offs_n[None, :] >= lo[:, None]) & (offs_n[None, :] <= hi[:, None])
            p = tl.where(keep, tl.exp2(s - lse[:, None]), 0.0)
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    dk = dk * sm_scale
    tl.store(
        DK + offs_n[:, None] * sdkn + offs_d[None, :],
        dk.to(DK.dtype.element_ty),
        mask = col_ok[:, None],
    )
    tl.store(
        DV + offs_n[:, None] * sdvn + offs_d[None, :],
        dv.to(DV.dtype.element_ty),
        mask = col_ok[:, None],
    )


_LOG2E = 1.4426950408889634


def _fwd_impl(q, k, v, lo, hi, scale):
    B, H, S, D = q.shape
    HKV = k.shape[1]
    # (B, S, H, D) storage so the caller's transpose(1, 2).contiguous() is free.
    o = torch.empty((B, S, H, D), device = q.device, dtype = q.dtype).transpose(1, 2)
    lse = torch.empty((B, H, S), device = q.device, dtype = torch.float32)
    with torch.cuda.device(q.device):
        _lh_fwd[lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B * H)](
            q,
            k,
            v,
            o,
            lse,
            lo,
            hi,
            scale * _LOG2E,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            H,
            H // HKV,
            S,
            HEAD_DIM = D,
            enable_fp_fusion = True,
        )
    return o, lse


def _bwd_impl(do, q, k, v, o, lse, lo, hi, scale):
    B, H, S, D = q.shape
    HKV = k.shape[1]
    G = H // HKV
    if do.stride(-1) != 1:
        do = do.contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty((B, H, S), device = q.device, dtype = torch.float32)
    positions = (
        torch.arange(S, device = q.device, dtype = torch.int32).expand(lo.shape[0], S).contiguous()
    )
    # First query row whose interval ends at or after key j, last row starting at or before j.
    qlo = torch.searchsorted(hi, positions, out_int32 = True)
    qhi = torch.searchsorted(lo, positions, right = True, out_int32 = True) - 1
    with torch.cuda.device(q.device):
        _lh_bwd_pre[(triton.cdiv(S, 16), B * H)](
            o,
            do,
            delta,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            H,
            S,
            HEAD_DIM = D,
            BLOCK_M = 16,
            num_warps = 4,
            enable_fp_fusion = True,
        )
        _lh_bwd_dkdv[lambda meta: (triton.cdiv(S, meta["BLOCK_N"]), B * HKV)](
            q,
            k,
            v,
            do,
            dk,
            dv,
            lse,
            delta,
            lo,
            hi,
            qlo,
            qhi,
            scale * _LOG2E,
            scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            HKV,
            G,
            S,
            HEAD_DIM = D,
            enable_fp_fusion = True,
        )
        _lh_bwd_dq[lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B * H)](
            q,
            k,
            v,
            do,
            dq,
            lse,
            delta,
            lo,
            hi,
            scale * _LOG2E,
            scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            H,
            G,
            S,
            HEAD_DIM = D,
            enable_fp_fusion = True,
        )
    return dq, dk, dv


# Opaque custom ops keep torch.compile from tracing into Triton (no graph break, no
# re-emission with fp fusion). The plain autograd.Function is the fallback for torch
# builds without torch.library.custom_op.
_HAS_CUSTOM_OP = (
    hasattr(torch, "library")
    and hasattr(torch.library, "custom_op")
    and hasattr(torch.library, "register_autograd")
)

if _HAS_CUSTOM_OP:

    @torch.library.custom_op("unsloth::large_head_attention_fwd", mutates_args = ())
    def _fwd_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _fwd_impl(q, k, v, lo, hi, scale)

    @_fwd_op.register_fake
    def _fwd_op_fake(q, k, v, lo, hi, scale):
        B, H, S, D = q.shape
        o = q.new_empty((B, S, H, D)).transpose(1, 2)
        return o, q.new_empty((B, H, S), dtype = torch.float32)

    @torch.library.custom_op("unsloth::large_head_attention_bwd", mutates_args = ())
    def _bwd_op(
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _bwd_impl(do, q, k, v, o, lse, lo, hi, scale)

    @_bwd_op.register_fake
    def _bwd_op_fake(do, q, k, v, o, lse, lo, hi, scale):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _setup_context(ctx, inputs, output):
        q, k, v, lo, hi, scale = inputs
        o, lse = output
        ctx.save_for_backward(q, k, v, o, lse, lo, hi)
        ctx.scale = scale

    def _backward(ctx, do, dlse):
        q, k, v, o, lse, lo, hi = ctx.saved_tensors
        dq, dk, dv = _bwd_op(do, q, k, v, o, lse, lo, hi, ctx.scale)
        return dq, dk, dv, None, None, None

    torch.library.register_autograd(
        "unsloth::large_head_attention_fwd",
        _backward,
        setup_context = _setup_context,
    )

    def _attention(q, k, v, lo, hi, scale):
        return _fwd_op(q, k, v, lo, hi, scale)[0]

else:

    class _LargeHeadAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, lo, hi, scale):
            o, lse = _fwd_impl(q, k, v, lo, hi, scale)
            ctx.save_for_backward(q, k, v, o, lse, lo, hi)
            ctx.scale = scale
            return o

        @staticmethod
        def backward(ctx, do):
            q, k, v, o, lse, lo, hi = ctx.saved_tensors
            dq, dk, dv = _bwd_impl(do, q, k, v, o, lse, lo, hi, ctx.scale)
            return dq, dk, dv, None, None, None

    def _attention(q, k, v, lo, hi, scale):
        return _LargeHeadAttention.apply(q, k, v, lo, hi, scale)


def _dense_intervals(B, S, causal, device):
    hi = (
        torch.arange(S, device = device, dtype = torch.int32)
        if causal
        else torch.full((S,), S - 1, device = device, dtype = torch.int32)
    )
    lo = torch.zeros((B, S), device = device, dtype = torch.int32)
    return lo, hi.expand(B, S).contiguous()


def intervals_from_mask(
    mask,
    B,
    S,
    _rows = 1024,
):
    """(lo, hi) int32 [B, S] such that query i attends exactly keys lo[i]..hi[i], or None.

    None unless every row of `mask` (bool, or additive float with 0 = keep) allows one
    non-empty contiguous key range and both ends are non-decreasing down the rows. One
    host sync; the verdict is cached on the mask so every layer sharing it reuses it.
    Inference tensors have no version counter, so they are rescanned on every call.
    """
    if not torch.is_tensor(mask) or mask.dim() != 4 or mask.shape[1] != 1:
        return None
    if mask.shape[-2] != S or mask.shape[-1] != S or mask.shape[0] not in (1, B):
        return None
    if not (mask.dtype == torch.bool or mask.is_floating_point()):
        return None
    use_cache = not mask.is_inference()
    key = (B, S, mask._version) if use_cache else None
    cached = getattr(mask, "_unsloth_large_head_intervals", None) if use_cache else None
    if cached is not None and cached[0] == key:
        return cached[1]
    Bm = mask.shape[0]
    idx = torch.arange(S, device = mask.device, dtype = torch.int32)
    lo = torch.empty((Bm, S), device = mask.device, dtype = torch.int32)
    hi = torch.empty_like(lo)
    ok = torch.ones((), device = mask.device, dtype = torch.bool)
    # Row chunks keep the int32 temporaries at Bm * _rows * S elements.
    for r0 in range(0, S, _rows):
        m = mask[:, 0, r0 : r0 + _rows]
        if m.dtype == torch.bool:
            allowed = m
        else:
            allowed = m == 0
            # Additive masks must only hold 0 or a value that zeroes the softmax weight.
            ok = ok & ((allowed | (m <= torch.finfo(m.dtype).min / 2)).all())
        lo_c = torch.where(allowed, idx, S).amin(-1)
        hi_c = torch.where(allowed, idx, -1).amax(-1)
        cnt = allowed.sum(-1, dtype = torch.int32)
        ok = ok & (cnt > 0).all() & (cnt == hi_c - lo_c + 1).all()
        lo[:, r0 : r0 + _rows] = lo_c
        hi[:, r0 : r0 + _rows] = hi_c
    ok = ok & (lo[:, 1:] >= lo[:, :-1]).all() & (hi[:, 1:] >= hi[:, :-1]).all()
    result = None
    if bool(ok.item()):
        result = (lo.expand(B, S).contiguous(), hi.expand(B, S).contiguous())
    if use_cache:
        try:
            mask._unsloth_large_head_intervals = (key, result)
        except Exception:
            pass
    return result


def large_head_attention(
    q,
    k,
    v,
    lo,
    hi,
    scale = None,
):
    """Attention over q [B, Hq, S, D], k / v [B, Hkv, S, D] where query i attends keys
    lo[i]..hi[i] (int32 [B, S]). Returns [B, S, Hq, D] contiguous."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v)]
    out = _attention(
        q, k, v, lo.to(torch.int32).contiguous(), hi.to(torch.int32).contiguous(), float(scale)
    )
    return out.transpose(1, 2)


@functools.lru_cache(maxsize = 1)
def _enabled():
    return os.environ.get("UNSLOTH_HD512_ATTENTION", "1") != "0"


def _shapes_qualify(query, key, value):
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False
    B, H, S, D = query.shape
    if D not in _SUPPORTED_HEAD_DIMS or key.shape[-1] != D or value.shape[-1] != D:
        return False
    if key.shape[0] != B or value.shape[0] != B or key.shape[2] != S or value.shape[2] != S:
        return False
    HKV = key.shape[1]
    if value.shape[1] != HKV or HKV == 0 or H % HKV != 0 or S < 2:
        return False
    if not (query.is_cuda and key.device == query.device and value.device == query.device):
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    return max(query.numel(), key.numel(), value.numel()) <= _INT32_MAX


_SELF_CHECK = {}


def _self_check(device, dtype):
    """One small fwd + bwd against SDPA math per (device, dtype) before first use."""
    key = (str(device), dtype)
    if key in _SELF_CHECK:
        return _SELF_CHECK[key]
    passed = False
    try:
        # A local generator: torch.manual_seed would reseed every CUDA device's RNG.
        gen = torch.Generator(device = device)
        gen.manual_seed(0)
        # Inference tensors cannot record autograd, even under enable_grad.
        with torch.inference_mode(False), torch.enable_grad():
            q, k, v, g = [
                torch.randn(shape, device = device, dtype = dtype, generator = gen)
                for shape in ((1, 4, 80, 512), (1, 2, 80, 512), (1, 2, 80, 512), (1, 80, 4, 512))
            ]
            q, k, v = [x.requires_grad_() for x in (q, k, v)]
            lo, hi = _dense_intervals(1, 80, True, device)
            out = large_head_attention(q, k, v, lo, hi, 512**-0.5)
            grads = torch.autograd.grad(out, (q, k, v), g)
            qf, kf, vf = [x.detach().float().requires_grad_() for x in (q, k, v)]
            ref = torch.nn.functional.scaled_dot_product_attention(
                qf,
                kf.repeat_interleave(2, 1),
                vf.repeat_interleave(2, 1),
                is_causal = True,
            ).transpose(1, 2)
            ref_grads = torch.autograd.grad(ref, (qf, kf, vf), g.float())
        passed = all(
            torch.isfinite(a).all().item()
            and ((a.float() - b).abs().max() <= 0.05 * b.abs().max() + 0.02).item()
            for a, b in zip((out, *grads), (ref, *ref_grads))
        )
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:
        _warn_once(
            f"Unsloth: head_dim 512 attention kernel failed its self check ({exc}); using SDPA."
        )
        passed = False
    _SELF_CHECK[key] = passed
    return passed


_WARNED = set()


def _warn_once(message):
    if message in _WARNED:
        return
    _WARNED.add(message)
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.get_logger("unsloth").warning(message)
    except Exception:
        print(message)


def _autocast_dtype_differs(query):
    # SDPA follows autocast; the kernel computes in the input dtype.
    try:
        enabled = torch.is_autocast_enabled(query.device.type)
        dtype = torch.get_autocast_dtype(query.device.type) if enabled else None
    except (AttributeError, TypeError):
        enabled = torch.is_autocast_enabled()
        dtype = torch.get_autocast_gpu_dtype() if enabled else None
    return enabled and dtype != query.dtype


def _math_sdpa_requested():
    """A disabled memory efficient backend or a math-first sdpa_kernel priority asks for SDPA itself."""
    cuda = torch.backends.cuda
    if not cuda.mem_efficient_sdp_enabled():
        return True
    order = list(getattr(torch._C, "_get_sdp_priority_order", list)())
    from torch.nn.attention import SDPBackend

    math, efficient = int(SDPBackend.MATH), int(SDPBackend.EFFICIENT_ATTENTION)
    return math in order and efficient in order and order.index(math) < order.index(efficient)


if hasattr(torch.compiler, "assume_constant_result"):
    _math_sdpa_requested = torch.compiler.assume_constant_result(_math_sdpa_requested)


def _make_router(original):
    def _sdpa_large_head(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout = 0.0,
        scaling = None,
        is_causal = None,
        **kwargs,
    ):
        if (
            _enabled()
            and torch.is_tensor(query)
            and query.dim() == 4
            and query.shape[-1] > 256
            and not dropout
            # The kernel only reads the mask's structure, so a trainable mask would get no grad.
            and not (torch.is_tensor(attention_mask) and attention_mask.requires_grad)
            and kwargs.get("position_bias", None) is None
            and kwargs.get("softcap", None) is None
            and not kwargs.get("output_attentions", False)
            and torch.version.hip is None
            and _shapes_qualify(query, key, value)
            and not _autocast_dtype_differs(query)
            and not _math_sdpa_requested()
        ):
            B, S = query.shape[0], query.shape[2]
            intervals = None
            if attention_mask is None:
                causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
                intervals = _dense_intervals(B, S, bool(causal), query.device)
            elif not torch.compiler.is_compiling():
                # Reading a mask's structure needs a host sync; compiled graphs keep SDPA.
                intervals = intervals_from_mask(attention_mask, B, S)
            compiling = torch.compiler.is_compiling()
            if compiling:
                checked = _SELF_CHECK.get((str(query.device), query.dtype), False)
            elif intervals is not None:
                checked = _self_check(query.device, query.dtype)
            if intervals is not None and checked:
                scale = scaling if scaling is not None else query.shape[-1] ** -0.5
                if compiling:
                    return large_head_attention(
                        query, key, value, *intervals, scale
                    ).contiguous(), None
                try:
                    out = large_head_attention(query, key, value, *intervals, scale).contiguous()
                    LARGE_HEAD_ENGAGED["calls"] += 1
                    return out, None
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as exc:
                    LARGE_HEAD_ENGAGED["fallback"] += 1
                    _SELF_CHECK[(str(query.device), query.dtype)] = False
                    _warn_once(
                        f"Unsloth: head_dim 512 attention kernel failed ({exc}); using SDPA."
                    )
        return original(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout = dropout,
            scaling = scaling,
            is_causal = is_causal,
            **kwargs,
        )

    # Keep the wrapped router's sentinels (unsloth_zoo's gemma-4 sliding router checks its own), so a
    # second model load does not wrap this router again and recurse between the two.
    for name, value in vars(original).items() if hasattr(original, "__dict__") else ():
        if name.startswith(("_unsloth_", "__unsloth_")):
            setattr(_sdpa_large_head, name, value)
    _sdpa_large_head._unsloth_large_head = True
    _sdpa_large_head._unsloth_original = original
    return _sdpa_large_head


_PATCHED = [False]


def patch_large_head_attention(model = None):
    """Route transformers' `sdpa` attention through the large head kernel when a call
    qualifies; everything else reaches the previous `sdpa` function unchanged. With a
    model, only installs if one of its modules has head_dim > 256."""
    if not _enabled() or not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    if model is not None:
        head_dims = (getattr(m, "head_dim", 0) for m in model.modules())
        if not any(isinstance(d, int) and d in _SUPPORTED_HEAD_DIMS for d in head_dims):
            return False
        # Compiled graphs only read the cached verdict, so check before any tracing.
        try:
            targets = {
                (p.device, p.dtype)
                for p in model.parameters()
                if p.is_cuda and p.dtype in (torch.float16, torch.bfloat16)
            }
        except Exception:
            targets = set()
        for device, dtype in targets:
            _self_check(device, dtype)
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    except Exception:
        return False
    # Other wrappers (unsloth_zoo's gemma-4 sliding router) may sit on top of ours.
    if _PATCHED[0] or getattr(current, "_unsloth_large_head", False):
        return True
    # unsloth/utils/hf_packed_attention.py sends packed rows (packed_seq_lengths) past head dim 256 to
    # SDPA's efficient kernel over cu_seqlens. Sit beneath it whichever is installed first, so a packed
    # call takes the same route as when a packed trainer is built after the model load.
    packed = sys.modules.get("unsloth.utils.hf_packed_attention")
    packed_orig = getattr(packed, "_ORIG_SDPA", None)
    if isinstance(packed_orig, list) and packed_orig and packed_orig[0] is not None:
        packed_orig[0] = _make_router(packed_orig[0])
    else:
        # Direct assignment: register() does not update the global mapping layers read.
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _make_router(current)
    _PATCHED[0] = True
    return True
