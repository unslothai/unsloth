# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""INT8 attention in Triton, for cards the CUDA kernel cannot reach: ROCm (only gfx1151 has a config) and NVIDIA
cards where the CUDA kernel cannot be built or loaded (no NVRTC, a failed compile or module load), sm80 and newer.

Same quantisation as ``diffusion_int8_attention`` (written from scratch, ideas from QuaRot arXiv 2404.00456,
SageAttention arXiv 2410.02367 / 2411.10958, INT-FlashAttention arXiv 2409.16997):

* Q and K rotated by fixed signs and a 16-point Hadamard transform per 16 channels, K shifted by its per-channel mean
  over every 8th key, one scale per query row and one per key (the CUDA kernel groups keys by accumulator lane; a
  Triton tile can apply a per-key vector at no cost).
* ``pv = "int8"``: P rounded to 0..127 per (query row, key tile), V centred, divided by a per-channel gain and
  rounded with one scale per key tile; the softmax denominator is summed over the rounded P.
  ``pv = "half"``: P and V in the inputs' 16-bit type (V read in place, no quantiser).

Rounding is round-half-even through the fp32 1.5 * 2**23 constant, so nothing depends on a libdevice
``float2int_rn`` (ROCm's libdevice has none).

Inputs are the strided ``[B, S, H, D]`` views; the output is ``[B, S, H, D]`` in V's dtype.

On NVIDIA sm_8x the attention kernel's 8-bit dots compile through ``diffusion_triton_prefetch`` (Triton >= 3.8 would
otherwise spill them); ``PREFETCH_FIX`` keeps cubins compiled with and without that fix apart in Triton's cache.
"""

from __future__ import annotations

import contextlib
import math
import types
from typing import Any, Optional

LOG2E = 1.4426950408889634
QUANT_ROWS = 64
STATS_STEP = 8
HEAD_DIMS = (64, 128)
DEFAULT_CONFIGS: dict = {
    128: {"bm": 64, "bn": 32, "warps": 4, "stages": 1, "pv": "half_strided", "kscale": "tile"},
    64: {"bm": 128, "bn": 32, "warps": 4, "stages": 1, "pv": "half_strided", "kscale": "tile"},
}


try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 -- no Triton: the module imports, nothing here runs
    triton = None


def _rocm_build() -> bool:
    try:
        import torch

        from .rocm_bf16 import is_rocm_torch
        return bool(is_rocm_torch(torch))
    except Exception:  # noqa: BLE001 -- read as an NVIDIA build: sizes stay out of the compile key
        return False


def triton_version() -> Optional[tuple]:
    """(major, minor) of the imported Triton, or None (no Triton, or a version string that does not parse)."""
    try:
        return tuple(int(x) for x in triton.__version__.split(".")[:2])
    except Exception:  # noqa: BLE001
        return None


def join_free_rotation() -> bool:
    """True where the rotation is built without ``tl.join``: NVIDIA Triton 3.3, whose layout-conversion pass gives
    the join a result layout its own type inference rejects ("'tt.join' op failed to infer returned types"). The
    rotation's bits are the join's; the int8 Q codes can differ by one at a rounding tie (21 of 16.8M at Qwen-Image
    1024 px on Triton 3.7.1, same accuracy). Every other Triton (and ROCm 3.3, which compiles the join) keeps it."""
    return triton is not None and triton_version() == (3, 3) and not _rocm_build()


# Sequence lengths stay out of the attention kernel's compile key except on ROCm, where the % 16 facts are worth 3%
# at 75k tokens on gfx1151 and a run keeps one resolution.
_ATTN_NO_SPECIALIZE = (
    ["n_heads"] if triton is not None and _rocm_build() else ["sq", "sk", "n_heads"]
)

if triton is not None:

    @triton.jit
    def _rint(x):
        # round half to even in fp32 for |x| < 2**22: adding then removing 1.5 * 2**23 drops the fraction
        return (x + 12582912.0) - 12582912.0

    if not join_free_rotation():

        @triton.jit
        def _rotate(x, R: tl.constexpr, D: tl.constexpr):
            cols = tl.arange(0, D)
            flip = (((cols * 7919 + 13) * (cols + 3)) >> 7) & 1
            x = x * (1.0 - 2.0 * flip.to(tl.float32))[None, :]
            for _ in tl.static_range(4):
                a, b = tl.split(tl.reshape(x, (R, D // 2, 2)))
                z = tl.reshape(tl.join(a + b, a - b), (R, D // 16, 8, 2))
                x = tl.reshape(tl.permute(z, (0, 1, 3, 2)), (R, D))
            return x

    else:
        # Chosen in Python, not by a constexpr global: Triton 3.2 / 3.3 leave a global's value out of the cache key, so
        # both bodies behind one ``if`` would share compiled kernels. Its own source keeps this one apart.
        @triton.jit
        def _rotate(x, R: tl.constexpr, D: tl.constexpr):
            cols = tl.arange(0, D)
            flip = (((cols * 7919 + 13) * (cols + 3)) >> 7) & 1
            x = x * (1.0 - 2.0 * flip.to(tl.float32))[None, :]
            pair = tl.arange(0, 2)[None, None, :]
            for _ in tl.static_range(4):
                a, b = tl.split(tl.reshape(x, (R, D // 2, 2)))
                # (a + b, a - b) interleaved by a select on a size-2 axis: the join's values without tt.join
                z = tl.reshape(
                    tl.where(pair == 0, (a + b)[:, :, None], (a - b)[:, :, None]),
                    (R, D // 16, 8, 2),
                )
                x = tl.reshape(tl.permute(z, (0, 1, 3, 2)), (R, D))
            return x

    @triton.jit
    def _stats(
        x_ptr,
        c_ptr,
        g_ptr,
        bh,
        n_rows,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        STEP: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        MODE: tl.constexpr,
    ):
        # per (batch, head, channel) over every STEP-th token: MODE 0 mean -> c; MODE 1 mid-range -> c, half range -> g
        b = (bh // n_heads).to(tl.int64)
        h = (bh % n_heads).to(tl.int64)
        cols = tl.arange(0, D)
        n_sub = (n_rows + STEP - 1) // STEP
        acc = tl.zeros([D], tl.float32)
        lo = tl.full([D], 3.0e38, tl.float32)
        hi = tl.full([D], -3.0e38, tl.float32)
        for blk in range(0, tl.cdiv(n_sub, R)):
            rows = (blk * R + tl.arange(0, R)) * STEP
            live = rows < n_rows
            x = tl.load(
                x_ptr
                + b * x_stride_b
                + h * x_stride_h
                + rows[:, None].to(tl.int64) * x_stride_s
                + cols[None, :],
                mask = live[:, None],
                other = 0.0,
            ).to(tl.float32)
            if MODE == 0:
                acc += tl.sum(x, axis = 0)
            else:
                lo = tl.minimum(lo, tl.min(tl.where(live[:, None], x, 3.0e38), axis = 0))
                hi = tl.maximum(hi, tl.max(tl.where(live[:, None], x, -3.0e38), axis = 0))
        if MODE == 0:
            tl.store(c_ptr + bh.to(tl.int64) * D + cols, acc / n_sub)
        else:
            tl.store(c_ptr + bh.to(tl.int64) * D + cols, 0.5 * (hi + lo))
            tl.store(g_ptr + bh.to(tl.int64) * D + cols, tl.maximum(0.5 * (hi - lo), 1e-12))

    @triton.jit
    def _quant_rows(
        x_ptr,
        c_ptr,
        q_ptr,
        s_ptr,
        bh,
        blk,
        n_rows,
        n_pad,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        mul,
        R: tl.constexpr,
        D: tl.constexpr,
        CENTRE: tl.constexpr,
        TILE: tl.constexpr,
    ):
        # R rows of one (batch, head), optionally centred, rotated: int8 = rint(x * 127 / amax), amax per row
        # (scale [BH, n_pad]) or over the R rows (TILE, scale [BH, n_pad / R]); scale = amax / 127 * mul. Padded
        # rows are zeros (with scale 0 per row)
        b = (bh // n_heads).to(tl.int64)
        h = (bh % n_heads).to(tl.int64)
        rows = blk * R + tl.arange(0, R)
        cols = tl.arange(0, D)
        live = rows < n_rows
        x = tl.load(
            x_ptr
            + b * x_stride_b
            + h * x_stride_h
            + rows[:, None].to(tl.int64) * x_stride_s
            + cols[None, :],
            mask = live[:, None],
            other = 0.0,
        ).to(tl.float32)
        if CENTRE:
            c = tl.load(c_ptr + bh.to(tl.int64) * D + cols)
            x = tl.where(live[:, None], x - c[None, :], 0.0)
        x = _rotate(x, R, D)
        base = bh.to(tl.int64) * n_pad
        if TILE:
            amax = tl.maximum(tl.max(tl.max(tl.abs(x), axis = 1), axis = 0), 1e-20)
            tl.store(s_ptr + bh.to(tl.int64) * (n_pad // R) + blk, amax * (mul / 127.0))
            q = _rint(x * (127.0 / amax))
        else:
            amax = tl.maximum(tl.max(tl.abs(x), axis = 1), 1e-20)
            tl.store(s_ptr + base + rows, tl.where(live, amax * (mul / 127.0), 0.0))
            q = _rint(x * (127.0 / amax)[:, None])
        tl.store(q_ptr + (base + rows[:, None]) * D + cols[None, :], q.to(tl.int8))

    @triton.jit
    def _quant_v(
        x_ptr,
        c_ptr,
        g_ptr,
        q_ptr,
        s_ptr,
        bh,
        blk,
        n_rows,
        n_pad,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        T: tl.constexpr,
        D: tl.constexpr,
        TRANS: tl.constexpr,
    ):
        # one tile of T keys of one (batch, head): (x - centre) / gain per channel, one scale for the tile,
        # stored key-major [BH, n_pad, D] or (TRANS) channel-major [BH, D, n_pad]
        b = (bh // n_heads).to(tl.int64)
        h = (bh % n_heads).to(tl.int64)
        rows = blk * T + tl.arange(0, T)
        cols = tl.arange(0, D)
        live = rows < n_rows
        x = tl.load(
            x_ptr
            + b * x_stride_b
            + h * x_stride_h
            + rows[:, None].to(tl.int64) * x_stride_s
            + cols[None, :],
            mask = live[:, None],
            other = 0.0,
        ).to(tl.float32)
        c = tl.load(c_ptr + bh.to(tl.int64) * D + cols)
        g = tl.load(g_ptr + bh.to(tl.int64) * D + cols)
        x = tl.where(live[:, None], (x - c[None, :]) / g[None, :], 0.0)
        amax = tl.maximum(tl.max(tl.max(tl.abs(x), axis = 1), axis = 0), 1e-20)
        tile = bh.to(tl.int64) * (n_pad // T) + blk
        tl.store(s_ptr + tile, amax / 127.0)
        base = bh.to(tl.int64) * n_pad
        q = _rint(x * (127.0 / amax)).to(tl.int8)
        if TRANS:
            tl.store(q_ptr + base * D + cols[None, :].to(tl.int64) * n_pad + rows[:, None], q)
        else:
            tl.store(q_ptr + (base + rows[:, None]) * D + cols[None, :], q)

    @triton.jit
    def _transpose_v(
        x_ptr,
        o_ptr,
        bh,
        blk,
        n_rows,
        n_pad,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        T: tl.constexpr,
        D: tl.constexpr,
        TO_F16: tl.constexpr,
        VB: tl.constexpr = 0,
    ):
        # one tile of T keys of one (batch, head) -> channel-major [BH, D, n_pad], padded keys zero; in V's dtype,
        # or (TO_F16) fp16 saturated at +-65504 so P.V can run on fp16 operands (RDNA3 has no bf16 convert).
        # VB > 0: block-contiguous [BH, n_pad / VB, D, VB] instead, so one attention key tile of VB keys is one
        # contiguous D * VB run (comfy-kitchen #368)
        b = (bh // n_heads).to(tl.int64)
        h = (bh % n_heads).to(tl.int64)
        rows = blk * T + tl.arange(0, T)
        cols = tl.arange(0, D)
        x = tl.load(
            x_ptr
            + b * x_stride_b
            + h * x_stride_h
            + rows[:, None].to(tl.int64) * x_stride_s
            + cols[None, :],
            mask = (rows < n_rows)[:, None],
            other = 0.0,
        )
        if TO_F16:
            x = tl.minimum(tl.maximum(x.to(tl.float32), -65504.0), 65504.0).to(tl.float16)
        base = bh.to(tl.int64) * n_pad * D
        if VB:
            blk_off = ((rows // VB)[:, None] * D + cols[None, :]).to(tl.int64) * VB + (rows % VB)[
                :, None
            ]
            tl.store(o_ptr + base + blk_off, x)
        else:
            tl.store(o_ptr + base + cols[None, :].to(tl.int64) * n_pad + rows[:, None], x)

    # One launch per quantiser: kept for tests and tools.

    @triton.jit
    def stats_kernel(
        x_ptr,
        c_ptr,
        g_ptr,
        n_rows,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        STEP: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        MODE: tl.constexpr,
    ):
        _stats(
            x_ptr,
            c_ptr,
            g_ptr,
            tl.program_id(0),
            n_rows,
            n_heads,
            x_stride_b,
            x_stride_s,
            x_stride_h,
            STEP,
            R,
            D,
            MODE,
        )

    @triton.jit
    def quant_rows_kernel(
        x_ptr,
        c_ptr,
        q_ptr,
        s_ptr,
        n_rows,
        n_pad,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        mul,
        R: tl.constexpr,
        D: tl.constexpr,
        CENTRE: tl.constexpr,
        TILE: tl.constexpr,
    ):
        _quant_rows(
            x_ptr,
            c_ptr,
            q_ptr,
            s_ptr,
            tl.program_id(0),
            tl.program_id(1),
            n_rows,
            n_pad,
            n_heads,
            x_stride_b,
            x_stride_s,
            x_stride_h,
            mul,
            R,
            D,
            CENTRE,
            TILE,
        )

    @triton.jit
    def quant_v_kernel(
        x_ptr,
        c_ptr,
        g_ptr,
        q_ptr,
        s_ptr,
        n_rows,
        n_pad,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        T: tl.constexpr,
        D: tl.constexpr,
        TRANS: tl.constexpr,
    ):
        _quant_v(
            x_ptr,
            c_ptr,
            g_ptr,
            q_ptr,
            s_ptr,
            tl.program_id(0),
            tl.program_id(1),
            n_rows,
            n_pad,
            n_heads,
            x_stride_b,
            x_stride_s,
            x_stride_h,
            T,
            D,
            TRANS,
        )

    @triton.jit
    def transpose_v_kernel(
        x_ptr,
        o_ptr,
        n_rows,
        n_pad,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        T: tl.constexpr,
        D: tl.constexpr,
        TO_F16: tl.constexpr,
        VB: tl.constexpr = 0,
    ):
        _transpose_v(
            x_ptr,
            o_ptr,
            tl.program_id(0),
            tl.program_id(1),
            n_rows,
            n_pad,
            n_heads,
            x_stride_b,
            x_stride_s,
            x_stride_h,
            T,
            D,
            TO_F16,
            VB,
        )

    # Token, block and head counts stay out of the compile key (Triton would otherwise specialise them on
    # % 16 and == 1), so one compile per head dim serves every sequence length and head count.
    @triton.jit(do_not_specialize = ["sq", "sk", "n_heads", "n_qblk"])
    def prep_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        qi_ptr,
        qs_ptr,
        kc_ptr,
        vc_ptr,
        vg_ptr,
        vh_ptr,
        sq,
        sk,
        n_qpad,
        n_kpad,
        n_heads,
        q_sb,
        q_ss,
        q_sh,
        k_sb,
        k_ss,
        k_sh,
        v_sb,
        v_ss,
        v_sh,
        mul,
        n_qblk,
        R: tl.constexpr,
        STEP: tl.constexpr,
        SR: tl.constexpr,
        D: tl.constexpr,
        V_STATS: tl.constexpr,
        V_T: tl.constexpr,
        VB: tl.constexpr = 0,
    ):
        # Input-only work per (batch, head): block 0 K mean, block 1 V range, then Q tiles, then the channel-major V
        # copy. Statistics blocks go first (longest); each block is its standalone kernel's code, so bits match.
        bh = tl.program_id(0)
        blk = tl.program_id(1)
        j = blk - 1 - V_STATS
        if blk == 0:
            _stats(k_ptr, kc_ptr, kc_ptr, bh, sk, n_heads, k_sb, k_ss, k_sh, STEP, SR, D, 0)
        elif j < 0:
            if V_STATS:
                _stats(v_ptr, vc_ptr, vg_ptr, bh, sk, n_heads, v_sb, v_ss, v_sh, STEP, SR, D, 1)
        elif j < n_qblk:
            _quant_rows(
                q_ptr,
                q_ptr,
                qi_ptr,
                qs_ptr,
                bh,
                j,
                sq,
                n_qpad,
                n_heads,
                q_sb,
                q_ss,
                q_sh,
                mul,
                R,
                D,
                False,
                False,
            )
        else:
            if V_T != 0:
                _transpose_v(
                    v_ptr,
                    vh_ptr,
                    bh,
                    j - n_qblk,
                    sk,
                    n_kpad,
                    n_heads,
                    v_sb,
                    v_ss,
                    v_sh,
                    R,
                    D,
                    V_T == 2,
                    VB,
                )

    @triton.jit(do_not_specialize = ["sk", "n_heads", "n_kblk"])
    def kv_kernel(
        k_ptr,
        kc_ptr,
        ki_ptr,
        ks_ptr,
        v_ptr,
        vc_ptr,
        vg_ptr,
        vi_ptr,
        vs_ptr,
        sk,
        n_kpad,
        n_heads,
        k_sb,
        k_ss,
        k_sh,
        v_sb,
        v_ss,
        v_sh,
        kmul,
        n_kblk,
        RK: tl.constexpr,
        T: tl.constexpr,
        D: tl.constexpr,
        K_TILE: tl.constexpr,
        V_Q: tl.constexpr,
    ):
        # Needs prep_kernel's statistics: centred K tiles, then INT8 V tiles (V_Q 1 key-, 2 channel-major).
        bh = tl.program_id(0)
        blk = tl.program_id(1)
        if blk < n_kblk:
            _quant_rows(
                k_ptr,
                kc_ptr,
                ki_ptr,
                ks_ptr,
                bh,
                blk,
                sk,
                n_kpad,
                n_heads,
                k_sb,
                k_ss,
                k_sh,
                kmul,
                RK,
                D,
                True,
                K_TILE,
            )
        else:
            if V_Q != 0:
                _quant_v(
                    v_ptr,
                    vc_ptr,
                    vg_ptr,
                    vi_ptr,
                    vs_ptr,
                    bh,
                    blk - n_kblk,
                    sk,
                    n_kpad,
                    n_heads,
                    v_sb,
                    v_ss,
                    v_sh,
                    T,
                    D,
                    V_Q == 2,
                )

    @triton.jit
    def _attn_tile(
        acc,
        l_i,
        m_i,
        t,
        q,
        qs,
        q_ptr,
        k_ptr,
        ks_ptr,
        v_ptr,
        vs_ptr,
        b,
        h,
        bh,
        kbase,
        n_ktiles,
        n_kpad,
        sk,
        qk_mul,
        k_sb,
        k_ss,
        k_sh,
        v_sb,
        v_ss,
        v_sh,
        offs_n,
        cols,
        BN: tl.constexpr,
        D: tl.constexpr,
        QK_INT8: tl.constexpr,
        K_TILE: tl.constexpr,
        PV_MODE: tl.constexpr,
        ACC_IN_DOT: tl.constexpr,
        MASK: tl.constexpr,
        V_BLK: tl.constexpr = 0,
    ):
        keys = t * BN + offs_n
        if QK_INT8:
            k = tl.load(k_ptr + (kbase + keys[:, None]) * D + cols[None, :])
            dots = tl.dot(q, tl.trans(k), out_dtype = tl.int32).to(tl.float32)
            if K_TILE:
                kst = tl.load(ks_ptr + bh.to(tl.int64) * n_ktiles + t)
                s = dots * (qs * kst)[:, None]
            else:
                ks = tl.load(ks_ptr + kbase + keys)
                s = dots * qs[:, None] * ks[None, :]
        else:
            k_ptrs = k_ptr + b * k_sb + h * k_sh + keys[:, None].to(tl.int64) * k_ss + cols[None, :]
            if MASK:
                k = tl.load(k_ptrs, mask = (keys < sk)[:, None], other = 0.0)
            else:
                k = tl.load(k_ptrs)
            s = tl.dot(q, tl.trans(k)) * qk_mul
        if MASK:
            s = tl.where((keys < sk)[None, :], s, -float("inf"))
        m_tile = tl.max(s, axis = 1)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.math.exp2(m_i - m_new)
        if PV_MODE == 2 or PV_MODE == 4:
            # 127 * 2^(s - tile max): log2(127) folded into the exponent
            p8 = _rint(tl.math.exp2(s - m_tile[:, None] + 6.988684686772166))
            if PV_MODE == 4:
                v = tl.load(v_ptr + kbase * D + cols[None, :].to(tl.int64) * n_kpad + keys[:, None])
            else:
                v = tl.load(v_ptr + (kbase + keys[:, None]) * D + cols[None, :])
            vs = tl.load(vs_ptr + bh.to(tl.int64) * n_ktiles + t)
            pv = tl.dot(p8.to(tl.int8), v, out_dtype = tl.int32).to(tl.float32)
            w = tl.math.exp2(m_tile - m_new) * (1.0 / 127.0)
            acc = acc * alpha[:, None] + pv * (w * vs)[:, None]
            l_i = l_i * alpha + tl.sum(p8, axis = 1) * w
        else:
            p = tl.math.exp2(s - m_new[:, None])
            if PV_MODE == 1:
                v = tl.load(v_ptr + (kbase + keys[:, None]) * D + cols[None, :])
            elif PV_MODE == 3:
                if V_BLK:
                    # block-contiguous V^T: key tile t is the D * BN run at t * D * BN
                    v = tl.load(
                        v_ptr
                        + kbase * D
                        + (t * D + cols[None, :]).to(tl.int64) * BN
                        + offs_n[:, None]
                    )
                else:
                    v = tl.load(
                        v_ptr + kbase * D + cols[None, :].to(tl.int64) * n_kpad + keys[:, None]
                    )
            else:
                v_ptrs = (
                    v_ptr + b * v_sb + h * v_sh + keys[:, None].to(tl.int64) * v_ss + cols[None, :]
                )
                if MASK:
                    v = tl.load(v_ptrs, mask = (keys < sk)[:, None], other = 0.0)
                else:
                    v = tl.load(v_ptrs)
            if ACC_IN_DOT:
                acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
            else:
                acc = acc * alpha[:, None]
                acc += tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis = 1)
        m_i = m_new
        return acc, l_i, m_i

    @triton.jit(do_not_specialize = _ATTN_NO_SPECIALIZE)
    def attn_kernel(
        q_ptr,
        qs_ptr,
        k_ptr,
        ks_ptr,
        v_ptr,
        vs_ptr,
        vc_ptr,
        vg_ptr,
        o_ptr,
        sq,
        sk,
        n_qpad,
        n_kpad,
        n_heads,
        qk_mul,
        q_sb,
        q_ss,
        q_sh,
        k_sb,
        k_ss,
        k_sh,
        v_sb,
        v_ss,
        v_sh,
        o_sb,
        o_ss,
        o_sh,
        BM: tl.constexpr,
        BN: tl.constexpr,
        D: tl.constexpr,
        QK_INT8: tl.constexpr,
        K_TILE: tl.constexpr,
        PV_MODE: tl.constexpr,
        EVEN_K: tl.constexpr,
        ACC_IN_DOT: tl.constexpr,
        PREFETCH_FIX: tl.constexpr,  # cache key only: compiled with the sm_8x 8-bit prefetch fix
        V_BLK: tl.constexpr = 0,
    ):
        # PV_MODE 0: 16-bit P.V, V read from the strided input; 1: 16-bit P.V, V from a heads-major padded copy;
        # 2: INT8 P.V, P as 0..127 relative to each (row, key tile) maximum, V centred / gained, one scale per tile;
        # 3: as 1 and 4: as 2, with V stored channel-major [BH, D, n_kpad] so that the P.V operand reads keys
        # contiguously (a key-major V tile is gathered element by element from shared memory on RDNA3 WMMA)
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)
        b = (bh // n_heads).to(tl.int64)
        h = (bh % n_heads).to(tl.int64)
        rows = pid_m * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        cols = tl.arange(0, D)
        qbase = bh.to(tl.int64) * n_qpad
        kbase = bh.to(tl.int64) * n_kpad
        n_ktiles = n_kpad // BN
        if QK_INT8:
            q = tl.load(q_ptr + (qbase + rows[:, None]) * D + cols[None, :])
            qs = tl.load(qs_ptr + qbase + rows)
        else:
            q = tl.load(
                q_ptr + b * q_sb + h * q_sh + rows[:, None].to(tl.int64) * q_ss + cols[None, :],
                mask = (rows < sq)[:, None],
                other = 0.0,
            )
            qs = tl.zeros([BM], tl.float32)
        m_i = tl.full([BM], -float("inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, D], tl.float32)
        n_full = sk // BN
        for t in range(0, n_full):
            acc, l_i, m_i = _attn_tile(
                acc,
                l_i,
                m_i,
                t,
                q,
                qs,
                q_ptr,
                k_ptr,
                ks_ptr,
                v_ptr,
                vs_ptr,
                b,
                h,
                bh,
                kbase,
                n_ktiles,
                n_kpad,
                sk,
                qk_mul,
                k_sb,
                k_ss,
                k_sh,
                v_sb,
                v_ss,
                v_sh,
                offs_n,
                cols,
                BN,
                D,
                QK_INT8,
                K_TILE,
                PV_MODE,
                ACC_IN_DOT,
                False,
                V_BLK,
            )
        if not EVEN_K:
            acc, l_i, m_i = _attn_tile(
                acc,
                l_i,
                m_i,
                n_full,
                q,
                qs,
                q_ptr,
                k_ptr,
                ks_ptr,
                v_ptr,
                vs_ptr,
                b,
                h,
                bh,
                kbase,
                n_ktiles,
                n_kpad,
                sk,
                qk_mul,
                k_sb,
                k_ss,
                k_sh,
                v_sb,
                v_ss,
                v_sh,
                offs_n,
                cols,
                BN,
                D,
                QK_INT8,
                K_TILE,
                PV_MODE,
                ACC_IN_DOT,
                True,
                V_BLK,
            )
        out = acc / l_i[:, None]
        if PV_MODE == 2 or PV_MODE == 4:
            vg = tl.load(vg_ptr + bh.to(tl.int64) * D + cols)
            vc = tl.load(vc_ptr + bh.to(tl.int64) * D + cols)
            out = out * vg[None, :] + vc[None, :]
        tl.store(
            o_ptr + b * o_sb + h * o_sh + rows[:, None].to(tl.int64) * o_ss + cols[None, :],
            out.to(o_ptr.dtype.element_ty),
            mask = (rows < sq)[:, None],
        )


_PREFETCH_FIX_ON: Optional[bool] = None


def prefetch_fix() -> bool:
    """True once the sm_8x 8-bit prefetch fix is installed in this process (NVIDIA builds of torch only)."""
    global _PREFETCH_FIX_ON
    if _PREFETCH_FIX_ON is None:
        on = False
        try:
            import torch
            from .rocm_bf16 import is_rocm_torch

            # AMD SDK / Radeon wheels leave torch.version.hip unset, so read ROCm from the build tag too.
            if not is_rocm_torch(torch):
                from .diffusion_triton_prefetch import fix_triton_sm8x_8bit_prefetch
                on = bool(fix_triton_sm8x_8bit_prefetch())
        except Exception:  # noqa: BLE001 -- the kernels compile as stock Triton would
            on = False
        _PREFETCH_FIX_ON = on
    return _PREFETCH_FIX_ON


_KERNELS: Any = None


def kernels() -> Any:
    global _KERNELS
    if triton is None:
        raise RuntimeError("Triton is not installed")
    prefetch_fix()
    if _KERNELS is None:
        _KERNELS = types.SimpleNamespace(
            stats = stats_kernel,
            quant_rows = quant_rows_kernel,
            quant_v = quant_v_kernel,
            transpose_v = transpose_v_kernel,
            prep = prep_kernel,
            kv = kv_kernel,
            attn = attn_kernel,
        )
    return _KERNELS


def _pad(n: int, multiple: int) -> int:
    return (n + multiple - 1) // multiple * multiple


PV_MODES = {"half_strided": 0, "half": 1, "int8": 2, "half_t": 3, "int8_t": 4, "f16_t": 3}
# blocks along y in one launch; a call whose fused launch would need more runs one launch per quantiser
GRID_LIMIT = 65535
# Launch arithmetic (padded lengths, grids, non-tensor args) cached per call signature: short calls are mostly glue.
_PLAN_LIMIT = 256
_PLANS: dict = {}


def _config_key(config: dict) -> tuple:
    return tuple(
        (key, tuple(sorted(value.items())) if isinstance(value, dict) else value)
        for key, value in config.items()
    )


def _strides(x: Any) -> tuple:
    return x.stride(0), x.stride(1), x.stride(2)


def _make_plan(
    q: Any, k: Any, v: Any, config: dict, pv: Optional[str], scale: Optional[float]
) -> dict:
    config = select_config(config, int(k.shape[1]))
    pv = pv or config.get("pv", "int8")
    if pv not in PV_MODES:
        raise ValueError(f"pv {pv!r}")
    k_tile = config.get("kscale", "key") == "tile"
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    bm, bn = int(config["bm"]), int(config["bn"])
    # log2(e) turns 2^x into e^x; 16 is the norm the rotation adds to Q and to K
    mul = (1.0 / math.sqrt(D) if scale is None else float(scale)) * LOG2E / 16.0
    n_qpad = _pad(Sq, math.lcm(QUANT_ROWS, bm))
    n_kpad = _pad(Sk, math.lcm(QUANT_ROWS, bn))
    BH = B * H
    mode = PV_MODES[pv]
    int8_v = pv in ("int8", "int8_t")
    v_t = 2 if pv == "f16_t" else 1 if pv == "half_t" else 0
    # block-contiguous V^T (one key tile = one contiguous run); only the 16-bit transposed P.V modes store V^T
    v_blk = bool(config.get("vblk")) and v_t != 0
    n_qblk = n_qpad // QUANT_ROWS
    rk = bn if k_tile else QUANT_ROWS
    n_kblk = n_kpad // rk
    n_vblk = n_kpad // QUANT_ROWS if v_t else 0
    n_vqblk = n_kpad // bn if int8_v else 0
    prep_blocks = 1 + int(int8_v) + n_qblk + n_vblk
    kv_blocks = n_kblk + n_vqblk
    sq_, sk_, sv_ = _strides(q), _strides(k), _strides(v)
    return {
        "pv": pv,
        "mode": mode,
        "k_tile": k_tile,
        "int8_v": int8_v,
        "v_t": v_t,
        "shape": (B, Sq, Sk, H, D),
        "BH": BH,
        "n_qpad": n_qpad,
        "n_kpad": n_kpad,
        "bm": bm,
        "bn": bn,
        "rk": rk,
        "mul": mul,
        "strides": (sq_, sk_, sv_),
        # one launch for what reads only the inputs, one for what needs their statistics
        "fused": prep_blocks <= GRID_LIMIT and kv_blocks <= GRID_LIMIT,
        "prep_grid": (BH, prep_blocks),
        "prep_args": (Sq, Sk, n_qpad, n_kpad, H, *sq_, *sk_, *sv_, mul, n_qblk),
        "prep_meta": {
            "R": QUANT_ROWS,
            "STEP": STATS_STEP,
            "SR": 32,
            "D": D,
            "V_STATS": int(int8_v),
            "V_T": v_t,
            "VB": bn if v_blk else 0,
        },
        "kv_grid": (BH, kv_blocks),
        "kv_args": (Sk, n_kpad, H, *sk_, *sv_, 1.0, n_kblk),
        "kv_meta": {
            "RK": rk,
            "T": bn,
            "D": D,
            "K_TILE": k_tile,
            "V_Q": (2 if pv == "int8_t" else 1) if int8_v else 0,
        },
        "attn_grid": (n_qpad // bm, BH),
        "attn_args": (Sq, Sk, n_qpad, n_kpad, H, 1.0, 0, 0, 0, 0, 0, 0, *sv_, Sq * H * D, H * D, D),
        "attn_meta": {
            "BM": bm,
            "BN": bn,
            "D": D,
            "QK_INT8": True,
            "K_TILE": k_tile,
            "PV_MODE": mode,
            "EVEN_K": Sk % bn == 0,
            "ACC_IN_DOT": bool(config.get("acc_in_dot", False)),
            "PREFETCH_FIX": prefetch_fix(),
            "V_BLK": int(v_blk),
            **_launch_options(config),
        },
        "config": config,
    }


def _plan(q: Any, k: Any, v: Any, config: dict, pv: Optional[str], scale: Optional[float]) -> dict:
    key = (
        q.shape,
        k.shape,
        v.shape,
        q.dtype,
        k.dtype,
        v.dtype,
        q.stride(),
        k.stride(),
        v.stride(),
        _config_key(config),
        pv,
        scale,
    )
    plan = _PLANS.get(key)
    if plan is None:
        plan = _make_plan(q, k, v, config, pv, scale)
        if len(_PLANS) >= _PLAN_LIMIT:
            _PLANS.clear()
        _PLANS[key] = plan
    return plan


_NO_SCOPE = contextlib.nullcontext()


def _on_card(t: Any) -> Any:
    """Triton launches on the current device: run the launches of a call on another card under that card. One
    check per public call; the same card (every single-card render) never switches."""
    import torch

    d = t.device
    if d.type != "cuda" or d.index is None or d.index == torch.cuda.current_device():
        return _NO_SCOPE
    if torch.compiler.is_compiling():
        return _NO_SCOPE
    return torch.cuda.device(d.index)


def _quantise(plan: dict, q: Any, k: Any, v: Any) -> dict:
    """The operands of ``attend`` for one call: two launches (``prep_kernel``, ``kv_kernel``) on the current card."""
    import torch

    kk = kernels()
    dev = q.device
    BH, n_qpad, n_kpad, D = plan["BH"], plan["n_qpad"], plan["n_kpad"], plan["shape"][4]
    qi = torch.empty((BH, n_qpad, D), dtype = torch.int8, device = dev)
    qs = torch.empty((BH, n_qpad), dtype = torch.float32, device = dev)
    kc = torch.empty((BH, D), dtype = torch.float32, device = dev)
    ki = torch.empty((BH, n_kpad, D), dtype = torch.int8, device = dev)
    ks = torch.empty(
        (BH, n_kpad // plan["rk"] if plan["k_tile"] else n_kpad), dtype = torch.float32, device = dev
    )
    out = {"qi": qi, "qs": qs, "ki": ki, "ks": ks, "plan": plan, "v": v}
    vc = vg = vh = vi = vs = qs
    if plan["int8_v"]:
        vc = torch.empty((BH, D), dtype = torch.float32, device = dev)
        vg = torch.empty((BH, D), dtype = torch.float32, device = dev)
        vi = torch.empty((BH, n_kpad, D), dtype = torch.int8, device = dev)
        vs = torch.empty((BH, n_kpad // plan["bn"]), dtype = torch.float32, device = dev)
        out.update(vi = vi, vs = vs, vc = vc, vg = vg)
    elif plan["v_t"]:
        vh = torch.empty(
            (BH, D, n_kpad), dtype = torch.float16 if plan["v_t"] == 2 else v.dtype, device = dev
        )
        out["vh"] = vh
    elif plan["pv"] == "half":
        B, _, Sk, H, _ = plan["shape"]
        vh4 = torch.empty((B, H, n_kpad, D), dtype = v.dtype, device = dev)
        vh4[:, :, :Sk].copy_(v.permute(0, 2, 1, 3))
        if n_kpad > Sk:
            vh4[:, :, Sk:].zero_()
        out["vh"] = vh4
    if not plan["fused"]:
        _quantise_unfused(kk, plan, q, k, v, out, kc)
        return out
    _launch(kk.prep, plan, "prep", (q, k, v, qi, qs, kc, vc, vg, vh))
    _launch(kk.kv, plan, "kv", (k, kc, ki, ks, v, vc, vg, vi, vs))
    return out


def _quantise_unfused(kk: Any, plan: dict, q: Any, k: Any, v: Any, out: dict, kc: Any) -> None:
    """One launch per quantiser, for grids past ``GRID_LIMIT``."""
    BH, n_qpad, n_kpad, D = plan["BH"], plan["n_qpad"], plan["n_kpad"], plan["shape"][4]
    Sq, Sk, H = plan["shape"][1], plan["shape"][2], plan["shape"][3]
    sq_, sk_, sv_ = plan["strides"]
    kk.quant_rows[(BH, n_qpad // QUANT_ROWS)](
        q,
        q,
        out["qi"],
        out["qs"],
        Sq,
        n_qpad,
        H,
        *sq_,
        plan["mul"],
        R = QUANT_ROWS,
        D = D,
        CENTRE = False,
        TILE = False,
    )
    kk.stats[(BH,)](k, kc, kc, Sk, H, *sk_, STEP = STATS_STEP, R = 32, D = D, MODE = 0)
    rk = plan["rk"]
    kk.quant_rows[(BH, n_kpad // rk)](
        k,
        kc,
        out["ki"],
        out["ks"],
        Sk,
        n_kpad,
        H,
        *sk_,
        1.0,
        R = rk,
        D = D,
        CENTRE = True,
        TILE = plan["k_tile"],
    )
    if plan["int8_v"]:
        kk.stats[(BH,)](v, out["vc"], out["vg"], Sk, H, *sv_, STEP = STATS_STEP, R = 32, D = D, MODE = 1)
        kk.quant_v[(BH, n_kpad // plan["bn"])](
            v,
            out["vc"],
            out["vg"],
            out["vi"],
            out["vs"],
            Sk,
            n_kpad,
            H,
            *sv_,
            T = plan["bn"],
            D = D,
            TRANS = plan["pv"] == "int8_t",
        )
    elif plan["v_t"]:
        kk.transpose_v[(BH, n_kpad // QUANT_ROWS)](
            v,
            out["vh"],
            Sk,
            n_kpad,
            H,
            *sv_,
            T = QUANT_ROWS,
            D = D,
            TO_F16 = plan["v_t"] == 2,
            VB = plan["prep_meta"]["VB"],
        )


FAST_LAUNCH_ENV = "UNSLOTH_DIFFUSION_INT8_TRITON_FAST_LAUNCH"
_FAST_LAUNCH: dict = {"ok": None}


def _fast_launch_ok() -> bool:
    if triton is None:  # no Triton (or a test's stand-in kernels): only the regular launch exists
        return False
    if _FAST_LAUNCH["ok"] is None:
        import os
        _FAST_LAUNCH["ok"] = os.environ.get(FAST_LAUNCH_ENV, "1").strip() != "0"
    return bool(_FAST_LAUNCH["ok"])


def _launch_hooks() -> tuple:
    knobs = getattr(triton, "knobs", None)
    rt = getattr(knobs, "runtime", None) if knobs is not None else None
    if rt is not None and hasattr(rt, "launch_enter_hook"):
        return rt.launch_enter_hook, rt.launch_exit_hook
    from triton.compiler.compiler import CompiledKernel

    return CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook


def _launcher(
    kernel: Any, compiled: Any, tensors: tuple, scalars: tuple, meta: dict
) -> Optional[tuple]:
    """(compiled kernel, every argument in parameter order with the tensors' positions left open) for the launch
    JITFunction.run makes, or None when this Triton does not expose it (the call keeps kernel[grid](...))."""
    try:
        if compiled is None or not all(
            hasattr(compiled, x) for x in ("run", "function", "packed_metadata")
        ):
            return None
        given = iter((*tensors, *scalars))
        template, slots = [], []
        for i, param in enumerate(kernel.params):
            if param.is_constexpr:
                template.append(meta[param.name] if param.name in meta else param.default)
            else:
                value = next(given)
                if i < len(tensors):
                    # an open slot, not the tensor: the plan outlives the call and must not pin its buffers
                    slots.append(i)
                    value = None
                template.append(value)
        if next(given, None) is not None or slots != list(range(len(tensors))):
            return None
        return compiled, template
    except Exception:  # noqa: BLE001 -- an unexpected Triton layout keeps the regular launch
        return None


def _launch(kernel: Any, plan: dict, name: str, tensors: tuple) -> None:
    """``kernel[grid](*tensors, *scalars, **meta)`` of the plan. The first call per device goes through Triton's
    launch (compile or cache lookup, argument specialisation); later calls on that device hand the same arguments
    straight to the compiled kernel's launcher, as JITFunction.run does after its lookup, so a short call does not
    pay the per-argument specialisation again. Taken only while every tensor is 16-byte aligned, as at the first
    call (the alignment is part of what was compiled); any failure falls back to the regular launch for good."""
    grid, scalars, meta = plan[name + "_grid"], plan[name + "_args"], plan[name + "_meta"]
    # TRITON_INTERPRET or no Triton: no compiled JITFunction to hand arguments to
    if (
        triton is not None
        and _fast_launch_ok()
        and isinstance(kernel, triton.runtime.jit.JITFunction)
    ):
        from triton.runtime.driver import driver

        # keyed by the operands' card; operands on another card take the regular launch
        dev = driver.active.get_current_device()
        fast = plan.setdefault("fast", {}).get((name, dev))
        if fast and tensors[0].get_device() == dev and all(t.data_ptr() % 16 == 0 for t in tensors):
            compiled, template = fast
            args = list(template)
            args[: len(tensors)] = tensors
            try:
                stream = driver.active.get_current_stream(dev)
                enter, leave = _launch_hooks()
                metadata = (
                    compiled.launch_metadata(grid, stream, *args)
                    if hasattr(compiled, "launch_metadata")
                    else None
                )
                compiled.run(
                    grid[0],
                    grid[1],
                    grid[2] if len(grid) > 2 else 1,
                    stream,
                    compiled.function,
                    compiled.packed_metadata,
                    metadata,
                    enter,
                    leave,
                    *args,
                )
                return
            except Exception:  # noqa: BLE001 -- this Triton's launcher wants something else: regular launches
                _FAST_LAUNCH["ok"] = False
        compiled = kernel[grid](*tensors, *scalars, **meta)
        if (
            fast is None
            and tensors[0].get_device() == dev
            and all(t.data_ptr() % 16 == 0 for t in tensors)
        ):
            plan["fast"][(name, dev)] = _launcher(kernel, compiled, tensors, scalars, meta) or False
        return
    kernel[grid](*tensors, *scalars, **meta)


def quantise(
    q: Any,
    k: Any,
    v: Any,
    config: dict,
    pv: Optional[str] = None,
    scale: Optional[float] = None,
) -> dict:
    """Quantised operands for ``attend`` from ``[B, S, H, D]`` q / k / v (last dim stride 1).

    ``config``: ``bm`` / ``bn`` (query rows / keys per tile), ``warps``, ``stages``, ``pv`` (``int8``, ``half``:
    16-bit P.V on a heads-major copy of V, ``half_strided``: 16-bit P.V reading V in place) and ``kscale``
    (``key``: one K scale per key, ``tile``: one per key tile)."""
    with _on_card(q):
        return _quantise(_plan(q, k, v, config, pv, scale), q, k, v)


def _attend(p: dict) -> Any:
    """The attention launch on ``_quantise``'s operands, on the current card."""
    import torch

    kk = kernels()
    plan = p["plan"]
    B, Sq, _, H, D = plan["shape"]
    v = p["v"]
    out = torch.empty((B, Sq, H, D), dtype = v.dtype, device = v.device)
    mode = plan["mode"]
    vv = v if mode == 0 else p["vh"] if mode in (1, 3) else p["vi"]
    dummy = p["qs"]
    _launch(
        kk.attn,
        plan,
        "attn",
        (
            p["qi"],
            p["qs"],
            p["ki"],
            p["ks"],
            vv,
            p.get("vs", dummy),
            p.get("vc", dummy),
            p.get("vg", dummy),
            out,
        ),
    )
    return out


def attend(p: dict) -> Any:
    """``[B, Sq, H, D]`` attention on ``quantise``'s operands, launched on their card."""
    with _on_card(p["qi"]):
        return _attend(p)


def int8_attention(
    q: Any,
    k: Any,
    v: Any,
    config: Optional[dict] = None,
    pv: Optional[str] = None,
    scale: Optional[float] = None,
) -> Any:
    """``softmax(q k^T * scale) v`` with int8 Q.K^T (and int8 P.V for ``pv="int8"``) on ``[B, S, H, D]``: three
    launches, their arithmetic planned once per call signature, on the card of ``q``."""
    with _on_card(q):
        return _int8_attention(q, k, v, config, pv, scale)


def _int8_attention(
    q: Any,
    k: Any,
    v: Any,
    config: Optional[dict] = None,
    pv: Optional[str] = None,
    scale: Optional[float] = None,
) -> Any:
    """``int8_attention`` on the current card: for callers that already hold ``torch.cuda.device(q.device)``
    (Studio's dispatcher), so a call pays no device check."""
    import torch

    if q.stride(3) != 1:
        q = q.contiguous()
    if k.stride(3) != 1:
        k = k.contiguous()
    if v.stride(3) != 1:
        v = v.contiguous()
    plan = _plan(q, k, v, config or DEFAULT_CONFIGS[int(q.shape[-1])], pv, scale)
    with torch.no_grad():
        return _attend(_quantise(plan, q, k, v))


def half_attention(
    q: Any,
    k: Any,
    v: Any,
    config: Optional[dict] = None,
    scale: Optional[float] = None,
) -> Any:
    """The same kernel without quantisation (16-bit Q.K^T and P.V, inputs read in place): the Triton baseline."""
    import torch

    kk = kernels()
    config = config or DEFAULT_CONFIGS[int(q.shape[-1])]
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    bm, bn = int(config["bm"]), int(config["bn"])
    mul = (1.0 / math.sqrt(D) if scale is None else float(scale)) * LOG2E
    out = torch.empty((B, Sq, H, D), dtype = v.dtype, device = v.device)
    k = k.to(q.dtype)
    v2 = v.to(q.dtype)
    with _on_card(out):
        kk.attn[(triton_cdiv(Sq, bm), B * H)](
            q,
            q,
            k,
            k,
            v2,
            q,
            q,
            q,
            out,
            Sq,
            Sk,
            0,
            0,
            H,
            mul,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v2.stride(0),
            v2.stride(1),
            v2.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BM = bm,
            BN = bn,
            D = D,
            QK_INT8 = False,
            K_TILE = False,
            PV_MODE = 0,
            EVEN_K = Sk % bn == 0,
            ACC_IN_DOT = bool(config.get("acc_in_dot", False)),
            PREFETCH_FIX = prefetch_fix(),
            **_launch_options(config),
        )
    return out


def _launch_options(config: dict) -> dict:
    out = {"num_warps": int(config["warps"]), "num_stages": int(config["stages"])}
    if config.get("waves_per_eu"):
        # AMD backend only: occupancy hint for the register allocator
        out["waves_per_eu"] = int(config["waves_per_eu"])
    return out


def select_config(config: dict, keys: int) -> dict:
    """``config`` with its ``long`` overrides applied from ``long_keys`` keys on (more query rows per program once
    one head's K / V no longer stay in L2)."""
    long_keys = config.get("long_keys")
    if long_keys and keys >= int(long_keys):
        return {**config, **config.get("long", {})}
    return config


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
