# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""INT8 attention in Triton, for cards the CUDA kernel cannot reach (ROCm: RDNA3 / RDNA4 WMMA, CDNA MFMA).

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
"""

from __future__ import annotations

import math
import types
from typing import Any, Optional

LOG2E = 1.4426950408889634
QUANT_ROWS = 64
STATS_STEP = 8
HEAD_DIMS = (64, 128)
# (BM, BN, warps, stages) per head dim; measured on gfx1151 (see the PR)
DEFAULT_CONFIGS: dict = {
    128: {"bm": 64, "bn": 32, "warps": 4, "stages": 1, "pv": "half_strided", "kscale": "tile"},
    64: {"bm": 128, "bn": 32, "warps": 4, "stages": 1, "pv": "half_strided", "kscale": "tile"},
}


try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 -- no Triton: the module imports, nothing here runs
    triton = None

if triton is not None:

    @triton.jit
    def _rint(x):
        # round half to even in fp32 for |x| < 2**22: adding then removing 1.5 * 2**23 drops the fraction
        return (x + 12582912.0) - 12582912.0

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
        # per (batch, head, channel) over every STEP-th token: MODE 0 mean -> c; MODE 1 mid-range -> c, half range -> g
        bh = tl.program_id(0)
        b = bh // n_heads
        h = bh % n_heads
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
            tl.store(c_ptr + bh * D + cols, acc / n_sub)
        else:
            tl.store(c_ptr + bh * D + cols, 0.5 * (hi + lo))
            tl.store(g_ptr + bh * D + cols, tl.maximum(0.5 * (hi - lo), 1e-12))

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
        # R rows of one (batch, head), optionally centred, rotated: int8 = rint(x * 127 / amax), amax per row
        # (scale [BH, n_pad]) or over the R rows (TILE, scale [BH, n_pad / R]); scale = amax / 127 * mul. Padded
        # rows are zeros (with scale 0 per row)
        bh = tl.program_id(0)
        blk = tl.program_id(1)
        b = bh // n_heads
        h = bh % n_heads
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
            c = tl.load(c_ptr + bh * D + cols)
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
        # one tile of T keys of one (batch, head): (x - centre) / gain per channel, one scale for the tile,
        # stored key-major [BH, n_pad, D] or (TRANS) channel-major [BH, D, n_pad]
        bh = tl.program_id(0)
        blk = tl.program_id(1)
        b = bh // n_heads
        h = bh % n_heads
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
        c = tl.load(c_ptr + bh * D + cols)
        g = tl.load(g_ptr + bh * D + cols)
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
    ):
        # one tile of T keys of one (batch, head) -> channel-major [BH, D, n_pad], padded keys zero; in V's dtype,
        # or (TO_F16) fp16 saturated at +-65504 so P.V can run on fp16 operands (RDNA3 has no bf16 convert)
        bh = tl.program_id(0)
        blk = tl.program_id(1)
        b = bh // n_heads
        h = bh % n_heads
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
        tl.store(o_ptr + base + cols[None, :].to(tl.int64) * n_pad + rows[:, None], x)

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
    ):
        # one key tile of the online softmax; MASK only for the ragged last tile
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
                v = tl.load(v_ptr + kbase * D + cols[None, :].to(tl.int64) * n_kpad + keys[:, None])
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

    @triton.jit
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
    ):
        # PV_MODE 0: 16-bit P.V, V read from the strided input; 1: 16-bit P.V, V from a heads-major padded copy;
        # 2: INT8 P.V, P as 0..127 relative to each (row, key tile) maximum, V centred / gained, one scale per tile;
        # 3: as 1 and 4: as 2, with V stored channel-major [BH, D, n_kpad] so that the P.V operand reads keys
        # contiguously (a key-major V tile is gathered element by element from shared memory on RDNA3 WMMA)
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // n_heads
        h = bh % n_heads
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
            )
        out = acc / l_i[:, None]
        if PV_MODE == 2 or PV_MODE == 4:
            vg = tl.load(vg_ptr + bh * D + cols)
            vc = tl.load(vc_ptr + bh * D + cols)
            out = out * vg[None, :] + vc[None, :]
        tl.store(
            o_ptr + b * o_sb + h * o_sh + rows[:, None].to(tl.int64) * o_ss + cols[None, :],
            out.to(o_ptr.dtype.element_ty),
            mask = (rows < sq)[:, None],
        )


def kernels() -> Any:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    return types.SimpleNamespace(
        stats = stats_kernel,
        quant_rows = quant_rows_kernel,
        quant_v = quant_v_kernel,
        transpose_v = transpose_v_kernel,
        attn = attn_kernel,
    )


def _pad(n: int, multiple: int) -> int:
    return (n + multiple - 1) // multiple * multiple


PV_MODES = {"half_strided": 0, "half": 1, "int8": 2, "half_t": 3, "int8_t": 4, "f16_t": 3}


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
    import torch

    kk = kernels()
    pv = pv or config.get("pv", "int8")
    if pv not in PV_MODES:
        raise ValueError(f"pv {pv!r}")
    k_tile = config.get("kscale", "key") == "tile"
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    bm, bn = int(config["bm"]), int(config["bn"])
    dev = q.device
    # log2(e) turns 2^x into e^x; 16 is the norm the rotation adds to Q and to K
    mul = (1.0 / math.sqrt(D) if scale is None else float(scale)) * LOG2E / 16.0
    n_qpad = _pad(Sq, math.lcm(QUANT_ROWS, bm))
    n_kpad = _pad(Sk, math.lcm(QUANT_ROWS, bn))
    BH = B * H

    def st(x):
        return x.stride(0), x.stride(1), x.stride(2)

    qi = torch.empty((BH, n_qpad, D), dtype = torch.int8, device = dev)
    qs = torch.empty((BH, n_qpad), dtype = torch.float32, device = dev)
    kk.quant_rows[(BH, n_qpad // QUANT_ROWS)](
        q, q, qi, qs, Sq, n_qpad, H, *st(q), mul, R = QUANT_ROWS, D = D, CENTRE = False, TILE = False
    )
    kc = torch.empty((BH, D), dtype = torch.float32, device = dev)
    kk.stats[(BH,)](k, kc, kc, Sk, H, *st(k), STEP = STATS_STEP, R = 32, D = D, MODE = 0)
    ki = torch.empty((BH, n_kpad, D), dtype = torch.int8, device = dev)
    if k_tile:
        ks = torch.empty((BH, n_kpad // bn), dtype = torch.float32, device = dev)
        kk.quant_rows[(BH, n_kpad // bn)](
            k, kc, ki, ks, Sk, n_kpad, H, *st(k), 1.0, R = bn, D = D, CENTRE = True, TILE = True
        )
    else:
        ks = torch.empty((BH, n_kpad), dtype = torch.float32, device = dev)
        kk.quant_rows[(BH, n_kpad // QUANT_ROWS)](
            k, kc, ki, ks, Sk, n_kpad, H, *st(k), 1.0, R = QUANT_ROWS, D = D, CENTRE = True, TILE = False
        )
    out = {
        "qi": qi,
        "qs": qs,
        "ki": ki,
        "ks": ks,
        "n_qpad": n_qpad,
        "n_kpad": n_kpad,
        "v": v,
        "pv": pv,
        "k_tile": k_tile,
        "shape": (B, Sq, Sk, H, D),
        "config": config,
    }
    if pv in ("int8", "int8_t"):
        vc = torch.empty((BH, D), dtype = torch.float32, device = dev)
        vg = torch.empty((BH, D), dtype = torch.float32, device = dev)
        kk.stats[(BH,)](v, vc, vg, Sk, H, *st(v), STEP = STATS_STEP, R = 32, D = D, MODE = 1)
        vi = torch.empty((BH, n_kpad, D), dtype = torch.int8, device = dev)
        vs = torch.empty((BH, n_kpad // bn), dtype = torch.float32, device = dev)
        kk.quant_v[(BH, n_kpad // bn)](
            v, vc, vg, vi, vs, Sk, n_kpad, H, *st(v), T = bn, D = D, TRANS = pv == "int8_t"
        )
        out.update(vi = vi, vs = vs, vc = vc, vg = vg)
    elif pv == "half":
        vh = torch.empty((B, H, n_kpad, D), dtype = v.dtype, device = dev)
        vh[:, :, :Sk].copy_(v.permute(0, 2, 1, 3))
        if n_kpad > Sk:
            vh[:, :, Sk:].zero_()
        out["vh"] = vh
    elif pv in ("half_t", "f16_t"):
        f16 = pv == "f16_t"
        vh = torch.empty((BH, D, n_kpad), dtype = torch.float16 if f16 else v.dtype, device = dev)
        kk.transpose_v[(BH, n_kpad // QUANT_ROWS)](
            v, vh, Sk, n_kpad, H, *st(v), T = QUANT_ROWS, D = D, TO_F16 = f16
        )
        out["vh"] = vh
    return out


def attend(p: dict) -> Any:
    import torch

    kk = kernels()
    B, Sq, Sk, H, D = p["shape"]
    cfg = p["config"]
    bm, bn = int(cfg["bm"]), int(cfg["bn"])
    v = p["v"]
    out = torch.empty((B, Sq, H, D), dtype = v.dtype, device = v.device)
    mode = PV_MODES[p["pv"]]
    vv = {0: v, 1: p.get("vh"), 2: p.get("vi"), 3: p.get("vh"), 4: p.get("vi")}[mode]
    dummy = p["qs"]
    kk.attn[(p["n_qpad"] // bm, B * H)](
        p["qi"],
        p["qs"],
        p["ki"],
        p["ks"],
        vv,
        p.get("vs", dummy),
        p.get("vc", dummy),
        p.get("vg", dummy),
        out,
        Sq,
        Sk,
        p["n_qpad"],
        p["n_kpad"],
        H,
        1.0,
        0,
        0,
        0,
        0,
        0,
        0,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BM = bm,
        BN = bn,
        D = D,
        QK_INT8 = True,
        K_TILE = p["k_tile"],
        PV_MODE = mode,
        EVEN_K = Sk % bn == 0,
        ACC_IN_DOT = bool(cfg.get("acc_in_dot", False)),
        **_launch_options(cfg),
    )
    return out


def int8_attention(
    q: Any,
    k: Any,
    v: Any,
    config: Optional[dict] = None,
    pv: Optional[str] = None,
    scale: Optional[float] = None,
) -> Any:
    """``softmax(q k^T * scale) v`` with int8 Q.K^T (and int8 P.V for ``pv="int8"``) on ``[B, S, H, D]``."""
    import torch

    config = select_config(config or DEFAULT_CONFIGS[int(q.shape[-1])], int(k.shape[1]))
    q, k, v = (x if x.stride(3) == 1 else x.contiguous() for x in (q, k, v))
    with torch.no_grad():
        return attend(quantise(q, k, v, config, pv, scale))


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
