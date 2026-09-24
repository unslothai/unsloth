# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Faster MiniMax-H3 video VAE encode and decode on the Diffusers path.

Diffusers' ``AutoencoderKLMiniMaxH3`` spends most of its time outside its matmuls. Profiled on a B200 at
1344x768: the encoder's per-frame GroupNorm, SiLU, reflect pad and causal pad are six full passes over an
activation between two convolutions (plus cuDNN's own NCDHW <-> NDHWC conversions around every conv), and the
ViT decoder's fp32 RMSNorms, rope ``torch.cat``s and residual arithmetic cost more than its GEMMs and attention
together. This module replaces those passes with a few Triton kernels and leaves the matmuls, the convolutions,
the tiling and the chunking exactly as Diffusers runs them.

Levers, each reported by name. Numbers are 1344x768, 129 frames of real footage, B200:

* ``fused_encoder``: per-frame GroupNorm + SiLU + reflect/causal padding in one kernel that writes the
  channels-last layout the next convolution wants, so cuDNN runs NDHWC natively. Each conv's bias rides into the
  next fused pass instead of PyTorch's separate strided bias add. A single-frame encode (the keyframe and
  image-reference path) convolves only the last temporal tap instead of two all-zero frames. Same float32
  arithmetic: 122 dB latent SNR against the stock encoder with TF32 off on both sides.
* ``fp16_encoder``: the encoder runs in float16 (statistics and epilogues in float32) instead of float32 with TF32
  convs. TF32 already rounds every conv input to float16's 10-bit mantissa, and the encode recipe rounds its
  latent to float16 afterwards: 64 dB latent SNR against the stock TF32 encode, 59 dB PSNR / LPIPS 6e-5 after
  decoding, against a 22 dB reconstruction floor for the VAE itself on the same footage. A keyframe moves 25x
  less than the posterior noise the recipe samples on top. ``default`` and ``max``.
* ``fused_decoder``: float32 RMSNorm fused with the residual add, per-head QK RMSNorm fused with the partial rope
  (in place), and SwiGLU in one pass, keeping the reference's float16 / float32 rounding points: 77.6 dB PSNR
  against the stock decode, 98.6% of 8-bit values identical.
* ``tile_batch``: several spatial tiles of a clip go through the decoder as one batch (bit-identical).
* ``fp16_accum`` (``max`` on consumer GPUs, diffusion_speed's rule for a float16-compute workload): the decode's
  float16 GEMMs accumulate in float16, 64.5 dB PSNR against float32 accumulation. Everywhere else the flag is held
  OFF for the decode: the bf16 denoiser's speed layer sets it process-wide on consumer GPUs in every tier, and it
  used to leak into this float16-autocast decode. ``UNSLOTH_DISABLE_FP16_ACCUM=1`` turns it off here too. cuDNN has
  no float16-accumulate mode, so the encoder's convolutions never see the flag.
* ``int8_decoder`` (opt-in, ``UNSLOTH_H3_VAE_INT8=1``): the second half of the decoder's blocks as ConvRot W8A8
  through ``torch._int_mm`` (62 dB; all blocks measured 53 dB, the early blocks carry the error). Not faster on
  a B200, where the activation quantisation eats the GEMM saving, so it stays out of every tier.

``UNSLOTH_H3_VAE_FAST=0`` turns everything off. NVIDIA CUDA only: ROCm, MPS and CPU keep the stock Diffusers path,
and so does a host without Triton.
"""

from __future__ import annotations

import os
import types
from contextlib import nullcontext
from functools import lru_cache
from typing import Any, Optional

H3_VAE_FAST_ENV = "UNSLOTH_H3_VAE_FAST"
H3_VAE_INT8_ENV = "UNSLOTH_H3_VAE_INT8"
H3_VAE_TILE_BATCH_ENV = "UNSLOTH_H3_VAE_TILE_BATCH"
# The denoiser speed layer's own opt-out (diffusion_speed._enable_fp16_accumulation), honoured here too.
FP16_ACCUM_DISABLE_ENV = "UNSLOTH_DISABLE_FP16_ACCUM"

LEVER_FUSED_ENCODER = "fused_encoder"
LEVER_FP16_ENCODER = "fp16_encoder"
LEVER_FUSED_DECODER = "fused_decoder"
LEVER_TILE_BATCH = "tile_batch"
LEVER_INT8_DECODER = "int8_decoder"
LEVER_FP16_ACCUM = "fp16_accum"

# Tiles per decoder call. Each 256x256 tile over a 7-latent-frame clip is ~1.8k tokens. Measured at 1344x768 on a
# B200: 1 tile 2.64 s, 2 tiles 2.03 s, 4 tiles 2.02 s (+0.06 GiB), 8 tiles 1.96 s (+0.61 GiB), so 4. The batch also
# shrinks to what the device has free.
_TILE_BATCH_MAX = 4
# One tile's decoder activations measured ~150 MB at float16 (the 4-tile batch peaked 0.06 GiB over one tile at a
# time, the 8-tile batch 0.61 GiB); 256 MiB per tile keeps headroom in the free-memory check.
_TILE_BATCH_BYTES_PER_TILE = 256 * 2**20
# ConvRot group for the int8 decoder: 256 input channels share one rotation.
H3_VAE_INT8_ROT_GROUP = 256
# Blocks the opt-in int8 decoder leaves in float16. Quantising only blocks 0-8 measured 56 dB, only 27-35 73 dB: the
# early blocks' error propagates through the rest, so the first half stays float (62 dB for the whole decode).
H3_VAE_INT8_FLOAT_BLOCKS = 18

_SPEED_OFF = "off"
_SPEED_EAGER = "eager"
_SPEED_MAX = "max"
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _env(name: str) -> str:
    return os.environ.get(name, "").strip().lower()


def h3_vae_fast_disabled() -> bool:
    return _env(H3_VAE_FAST_ENV) in _FALSE


def plan_h3_vae_levers(
    speed_mode: Optional[str],
    *,
    workflow: Optional[str] = None,
    consumer_gpu: bool = False,
) -> tuple[str, ...]:
    """Which levers a load should engage. Pure: no torch, no device probe, so it is testable anywhere.

    ``off`` is the bit-exact tier and gets nothing. ``eager`` gets the rounding-level fusions and tile batching,
    ``default`` adds the float16 encoder, ``max`` adds the float16-accumulate decode on a consumer GPU. ``t2va``
    never encodes, so the encoder levers are skipped for it (its encoder is not even resident after
    ``trim_h3_video_vae``). The int8 decoder is opt-in only.
    """
    mode = str(speed_mode or "").strip().lower()
    if h3_vae_fast_disabled() or mode == _SPEED_OFF:
        return ()
    levers: list[str] = []
    if workflow != "t2va":
        levers.append(LEVER_FUSED_ENCODER)
        # near-lossless rather than rounding-level, so not in "eager", the lossless-only tier
        if mode != _SPEED_EAGER:
            levers.append(LEVER_FP16_ENCODER)
    levers += [LEVER_FUSED_DECODER, LEVER_TILE_BATCH]
    if _env(H3_VAE_INT8_ENV) in _TRUE:
        levers.append(LEVER_INT8_DECODER)
    # diffusion_speed's own rule for a float16-compute workload: float16 accumulation only under "max"
    if mode == _SPEED_MAX and consumer_gpu and _env(FP16_ACCUM_DISABLE_ENV) not in _TRUE:
        levers.append(LEVER_FP16_ACCUM)
    return tuple(levers)


# ── Triton kernels ─────────────────────────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize = 1)
def _kernels() -> Optional[types.SimpleNamespace]:
    """Compile-on-first-use Triton kernels, or None when Triton is unavailable."""
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
    except Exception:  # noqa: BLE001 - no Triton means the stock path
        return None

    @triton.jit
    def _gn_partials(
        x_ptr,
        bias_ptr,
        pn_ptr,
        pmean_ptr,
        pm2_ptr,
        T,
        HW,
        W,
        n_chunks,
        sb,
        sc,
        st,
        sh,
        sw,
        C: tl.constexpr,
        G: tl.constexpr,
        CPG: tl.constexpr,
        BLOCK_P: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        # Per (batch*frame, pixel chunk): each group's count, mean and M2 over BLOCK_P pixels x its channels. A chunk
        # spans every channel, so a channels-last read is one contiguous run per pixel.
        chunk = tl.program_id(0)
        bt = tl.program_id(1)
        t = bt % T
        b = bt // T
        p = chunk * BLOCK_P + tl.arange(0, BLOCK_P)
        pmask = p < HW
        hh = p // W
        ww = p - hh * W
        c = tl.arange(0, C)
        off = (
            b.to(tl.int64) * sb
            + t.to(tl.int64) * st
            + hh.to(tl.int64)[:, None] * sh
            + ww.to(tl.int64)[:, None] * sw
            + c.to(tl.int64)[None, :] * sc
        )
        v = tl.load(x_ptr + off, mask = pmask[:, None], other = 0.0).to(tl.float32)
        if HAS_BIAS:
            v = v + tl.load(bias_ptr + c).to(tl.float32)[None, :]
            v = tl.where(pmask[:, None], v, 0.0)
        n_pix = tl.sum(pmask.to(tl.float32), axis = 0)
        v3 = tl.reshape(v, (BLOCK_P, G, CPG))
        mean = tl.sum(tl.sum(v3, axis = 2), axis = 0) / (n_pix * CPG)
        d = tl.where(pmask[:, None, None], v3 - mean[None, :, None], 0.0)
        m2 = tl.sum(tl.sum(d * d, axis = 2), axis = 0)
        slot = bt.to(tl.int64) * n_chunks + chunk
        g = tl.arange(0, G)
        tl.store(pmean_ptr + slot * G + g, mean)
        tl.store(pm2_ptr + slot * G + g, m2)
        tl.store(pn_ptr + slot, n_pix * CPG)

    @triton.jit
    def _gn_combine(
        pn_ptr, pmean_ptr, pm2_ptr, mean_ptr, rstd_ptr, n_chunks, G, eps, BLOCK: tl.constexpr
    ):
        # Chan's parallel merge of the chunk partials of one (batch*frame, group), in two vectorised passes.
        row = tl.program_id(0)
        g = row % G
        bt = (row // G).to(tl.int64)
        zero = tl.sum(tl.zeros([BLOCK], dtype = tl.float32), axis = 0)
        acc_n = zero
        acc_s = zero
        for k0 in range(0, n_chunks, BLOCK):
            k = k0 + tl.arange(0, BLOCK)
            km = k < n_chunks
            n = tl.load(pn_ptr + bt * n_chunks + k, mask = km, other = 0.0)
            mu = tl.load(pmean_ptr + (bt * n_chunks + k) * G + g, mask = km, other = 0.0)
            acc_n += tl.sum(n, axis = 0)
            acc_s += tl.sum(n * mu, axis = 0)
        mean = acc_s / acc_n
        acc_m2 = zero
        for k0 in range(0, n_chunks, BLOCK):
            k = k0 + tl.arange(0, BLOCK)
            km = k < n_chunks
            n = tl.load(pn_ptr + bt * n_chunks + k, mask = km, other = 0.0)
            mu = tl.load(pmean_ptr + (bt * n_chunks + k) * G + g, mask = km, other = 0.0)
            m2 = tl.load(pm2_ptr + (bt * n_chunks + k) * G + g, mask = km, other = 0.0)
            dm = mu - mean
            acc_m2 += tl.sum(m2 + n * dm * dm, axis = 0)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, 1.0 / tl.sqrt(acc_m2 / acc_n + eps))

    @triton.jit
    def _gn_silu_pad(
        x_ptr,
        out_ptr,
        mean_ptr,
        rstd_ptr,
        w_ptr,
        b_ptr,
        ib_ptr,
        C,
        T,
        H,
        W,
        To,
        Ho,
        Wo,
        G,
        front,
        pad_t,
        pad_l,
        sb,
        sc,
        st,
        sh,
        sw,
        CPG: tl.constexpr,
        HAS_NORM: tl.constexpr,
        HAS_IN_BIAS: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        # One program per (batch, output frame, output row) x a block of output columns x a block of channels.
        # Reads any 5D layout, writes channels-last [B, To, Ho, Wo, C]: the input plus its pending conv bias, then
        # GroupNorm, SiLU, reflect in space and zero frames in front. 64-bit math only for the per-program bases.
        pid_row = tl.program_id(0)
        pid_w = tl.program_id(1)
        pid_c = tl.program_id(2)
        h_o = pid_row % Ho
        bt = pid_row // Ho
        t_o = bt % To
        b = bt // To
        t_i = t_o - front
        t_ok = t_i >= 0
        t_c = tl.maximum(t_i, 0)
        r = h_o - pad_t
        r = tl.where(r < 0, -r, r)
        r = tl.where(r >= H, 2 * (H - 1) - r, r)
        in_base = x_ptr + (b.to(tl.int64) * sb + t_c.to(tl.int64) * st + r.to(tl.int64) * sh)
        out_base = out_ptr + (((b.to(tl.int64) * To + t_o) * Ho + h_o) * Wo) * C
        wo = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
        wmask = wo < Wo
        q = wo - pad_l
        q = tl.where(q < 0, -q, q)
        q = tl.where(q >= W, 2 * (W - 1) - q, q)
        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        cmask = c < C
        m = wmask[:, None] & cmask[None, :]
        # int64: an NCDHW input's channel stride times 127 passes 2^31 from ~1 MP x 17 frames (an untiled encode)
        in_off = q.to(tl.int64)[:, None] * sw + c.to(tl.int64)[None, :] * sc
        v = tl.load(in_base + in_off, mask = m & t_ok, other = 0.0).to(tl.float32)
        if HAS_IN_BIAS:
            v = v + tl.load(ib_ptr + c, mask = cmask, other = 0.0).to(tl.float32)[None, :]
        if HAS_NORM:
            gidx = (b * T + t_c) * G + c // CPG
            mu = tl.load(mean_ptr + gidx, mask = cmask, other = 0.0)
            rs = tl.load(rstd_ptr + gidx, mask = cmask, other = 0.0)
            gw = tl.load(w_ptr + c, mask = cmask, other = 0.0).to(tl.float32)
            gb = tl.load(b_ptr + c, mask = cmask, other = 0.0).to(tl.float32)
            # PyTorch's GroupNorm folds the affine the same way: y = x * (rstd * gamma) + (beta - mean * rstd * gamma)
            a = rs * gw
            v = v * a[None, :] + (gb - mu * a)[None, :]
            v = v / (1.0 + tl.exp(-v))
        v = tl.where(t_ok, v, 0.0)
        tl.store(out_base + (wo[:, None] * C + c[None, :]), v.to(out_ptr.dtype.element_ty), mask = m)

    @triton.jit
    def _add_residual(
        o_ptr,
        ob_ptr,
        r_ptr,
        rb_ptr,
        P,
        T,
        H,
        W,
        sb,
        sc,
        st,
        sh,
        sw,
        C: tl.constexpr,
        HAS_OB: tl.constexpr,
        HAS_RB: tl.constexpr,
        R_SAME: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        # o (channels-last, contiguous) = o + o_bias + r + r_bias over BLOCK_P pixels x all C channels, r in any
        # layout: the conv bias, the shortcut's bias and the skip connection in one pass.
        p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
        pmask = p < P
        c = tl.arange(0, C)
        base = tl.program_id(0).to(tl.int64) * BLOCK_P * C
        local = (tl.arange(0, BLOCK_P) * C)[:, None] + c[None, :]
        m = pmask[:, None]
        o = tl.load(o_ptr + base + local, mask = m, other = 0.0).to(tl.float32)
        if HAS_OB:
            o = o + tl.load(ob_ptr + c).to(tl.float32)[None, :]
        if R_SAME:
            r = tl.load(r_ptr + base + local, mask = m, other = 0.0).to(tl.float32)
        else:
            w = p % W
            rest = p // W
            h = rest % H
            rest = rest // H
            t = rest % T
            b = rest // T
            poff = (
                b.to(tl.int64) * sb
                + t.to(tl.int64) * st
                + h.to(tl.int64) * sh
                + w.to(tl.int64) * sw
            )
            r = tl.load(
                r_ptr + poff[:, None] + (c.to(tl.int64) * sc)[None, :], mask = m, other = 0.0
            ).to(tl.float32)
        if HAS_RB:
            r = r + tl.load(rb_ptr + c).to(tl.float32)[None, :]
        tl.store(o_ptr + base + local, (o + r).to(o_ptr.dtype.element_ty), mask = m)

    @triton.jit
    def _add_rmsnorm(
        h_ptr, o_ptr, s_ptr, w_ptr, n_ptr, N, eps, HAS_RES: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        # h (fp32 residual stream) += o * s in place, then n = rmsnorm(h) * w in float32, stored in n's dtype.
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        h = tl.load(h_ptr + row * N + cols, mask = mask, other = 0.0).to(tl.float32)
        if HAS_RES:
            o = tl.load(o_ptr + row * N + cols, mask = mask, other = 0.0).to(tl.float32)
            s = tl.load(s_ptr + cols, mask = mask, other = 0.0).to(tl.float32)
            # the reference's separate multiply and add, each rounded (no fma contraction)
            h = libdevice.add_rn(h, libdevice.mul_rn(o, s))
            tl.store(h_ptr + row * N + cols, h.to(h_ptr.dtype.element_ty), mask = mask)
        ms = tl.sum(h * h, axis = 0) / N
        w = tl.load(w_ptr + cols, mask = mask, other = 0.0).to(tl.float32)
        y = h * (1.0 / tl.sqrt(ms + eps)) * w
        tl.store(n_ptr + row * N + cols, y.to(n_ptr.dtype.element_ty), mask = mask)

    @triton.jit
    def _qk_norm_rope(
        x_ptr,
        cos_ptr,
        sin_ptr,
        n_head_rows,
        heads,
        cs_stride,
        eps,
        HEAD_D: tl.constexpr,
        ROT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # In place over BLOCK consecutive (token, head) rows of a contiguous [tokens, heads * HEAD_D] tensor, so one
        # program reads one contiguous span: x = rope(rmsnorm_fp32(x) -> x.dtype). The rotary partner of lane d is
        # d +- ROT/2, gathered from the lines the first load already brought in.
        hr = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        rmask = hr < n_head_rows
        d = tl.arange(0, HEAD_D)
        half = ROT // 2
        partner = tl.where(d < half, d + half, tl.where(d < ROT, d - half, d))
        base = hr[:, None] * HEAD_D
        x = tl.load(x_ptr + base + d[None, :], mask = rmask[:, None], other = 0.0).to(tl.float32)
        xp = tl.load(x_ptr + base + partner[None, :], mask = rmask[:, None], other = 0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis = 1) / HEAD_D + eps)
        dt = x_ptr.dtype.element_ty
        xn = (x * rstd[:, None]).to(dt).to(tl.float32)
        xpn = (xp * rstd[:, None]).to(dt).to(tl.float32)
        rot = d < ROT
        token = hr // heads
        cs_off = token[:, None] * cs_stride + tl.where(rot, d, 0)[None, :]
        cs_m = rmask[:, None] & rot[None, :]
        cos = tl.load(cos_ptr + cs_off, mask = cs_m, other = 1.0).to(tl.float32)
        sin = tl.load(sin_ptr + cs_off, mask = cs_m, other = 0.0).to(tl.float32)
        # The rope in x.dtype arithmetic, each product and the sum rounded, as eager's separate kernels compute it.
        # mul_rn / add_rn because the compiler otherwise contracts the pair into one fma and skips a rounding.
        xps = tl.where((d < half)[None, :], -xpn, xpn)
        a = libdevice.mul_rn(xn, cos).to(dt).to(tl.float32)
        bb = libdevice.mul_rn(xps, sin).to(dt).to(tl.float32)
        y = tl.where(rot[None, :], libdevice.add_rn(a, bb), xn)
        tl.store(x_ptr + base + d[None, :], y.to(dt), mask = rmask[:, None])

    @triton.jit
    def _swiglu(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
        # out = hidden * silu(gate) for x = [hidden | gate], silu and product each rounded to out's dtype.
        row = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        hid = tl.load(x_ptr + row * 2 * N + cols, mask = mask, other = 0.0).to(tl.float32)
        gate = tl.load(x_ptr + row * 2 * N + N + cols, mask = mask, other = 0.0).to(tl.float32)
        dt = out_ptr.dtype.element_ty
        act = (gate / (1.0 + tl.exp(-gate))).to(dt).to(tl.float32)
        tl.store(out_ptr + row * N + cols, (hid * act).to(dt), mask = mask)

    @triton.jit
    def _quant_rows(x_ptr, q_ptr, s_ptr, N, BLOCK_N: tl.constexpr):
        # Symmetric per-row int8: s = absmax / 127, q = round(x / s).
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask = mask, other = 0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), axis = 0), 1e-12)
        s = amax / 127.0
        q = x / s
        q = tl.where(q >= 0, tl.floor(q + 0.5), tl.ceil(q - 0.5))
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(q_ptr + row * N + cols, q.to(tl.int8), mask = mask)
        tl.store(s_ptr + row, s)

    @triton.jit
    def _dequant_epilogue(
        acc_ptr, xs_ptr, ws_ptr, bias_ptr, out_ptr, N, HAS_BIAS: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        acc = tl.load(acc_ptr + row * N + cols, mask = mask, other = 0).to(tl.float32)
        xs = tl.load(xs_ptr + row)
        ws = tl.load(ws_ptr + cols, mask = mask, other = 0.0)
        y = acc * xs * ws
        if HAS_BIAS:
            y = y + tl.load(bias_ptr + cols, mask = mask, other = 0.0).to(tl.float32)
        tl.store(out_ptr + row * N + cols, y.to(out_ptr.dtype.element_ty), mask = mask)

    return types.SimpleNamespace(
        gn_partials = _gn_partials,
        gn_combine = _gn_combine,
        gn_silu_pad = _gn_silu_pad,
        add_residual = _add_residual,
        add_rmsnorm = _add_rmsnorm,
        qk_norm_rope = _qk_norm_rope,
        swiglu = _swiglu,
        quant_rows = _quant_rows,
        dequant_epilogue = _dequant_epilogue,
    )


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


# ── runtime fallback ───────────────────────────────────────────────────────────────────────────────────────────


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001
        pass
    return "out of memory" in str(exc).lower()


def _guarded(fast: Any, stock: Any, label: str) -> Any:
    """A forward that runs ``fast`` until it raises something the ``stock`` path does not, then logs once and runs
    ``stock`` from then on. A Triton or driver problem on some host costs the speedup, never the render; an OOM is
    the caller's to handle, exactly as it would be on the stock path."""

    def forward(self, *args, **kwargs):
        if getattr(self, "_unsloth_fast_failed", False):
            return stock(self, *args, **kwargs)
        try:
            return fast(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise
            # an input the stock path rejects too is the caller's error, not a broken kernel: raise it, keep the
            # fast path for the next call
            out = stock(self, *args, **kwargs)
            self._unsloth_fast_failed = True
            log = getattr(self, "_unsloth_logger", None)
            if log is not None:
                log.warning(
                    "video.h3_vae_fast: the fused %s failed, using the stock path: %s", label, exc
                )
            return out

    return forward


# ── encoder ────────────────────────────────────────────────────────────────────────────────────────────────────────
#
# Every activation travels as (tensor, pending_bias): cuDNN convolves without its bias, and the bias is added by
# whichever fused pass reads that output next (the GroupNorm statistics and apply, the downsample pad, or the
# residual add). PyTorch otherwise adds a conv bias as its own strided pass over a channels-last output, which
# measured ~30% of the fused encoder's time. A 1x1 conv folds its input's pending bias through its weight exactly.


def _bias_view(bias: Any) -> Any:
    return bias.view(1, -1, 1, 1, 1)


def norm_silu_pad_reference(
    x: Any,
    norm: Any,
    pad: tuple,
    front: int,
    in_bias: Any = None,
) -> Any:
    """The unfused semantics the kernel reproduces, for tests and hosts without Triton: add the pending bias,
    per-frame GroupNorm, SiLU, reflect ``(left, right, top, bottom)`` and ``front`` zero frames, channels-last out."""
    import torch
    import torch.nn.functional as F

    if norm is not None:
        # float32 from the bias add through the SiLU, one rounding to x.dtype at the end, as the kernel does
        b, c, t, h, w = x.shape
        y = x.float() if in_bias is None else x.float() + _bias_view(in_bias).float()
        y = y.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = F.group_norm(y, norm.num_groups, norm.weight.float(), norm.bias.float(), norm.eps)
        x = F.silu(y).view(b, t, c, h, w).permute(0, 2, 1, 3, 4).to(x.dtype)
    elif in_bias is not None:
        x = (x.float() + _bias_view(in_bias).float()).to(x.dtype)
    return _pad_reference(x, pad, front).contiguous(memory_format = torch.channels_last_3d)


def _pad_reference(x: Any, pad: tuple, front: int) -> Any:
    """Reflect ``(left, right, top, bottom)`` spatially, then ``front`` causal zero frames."""
    import torch.nn.functional as F

    if any(pad):
        x = F.pad(x, (*pad, 0, 0), mode = "reflect")
    return F.pad(x, (0, 0, 0, 0, front, 0)) if front else x


def _fusable(x: Any, pad: tuple, norm: Any) -> bool:
    c = x.shape[1]
    return (
        _kernels() is not None
        and x.is_cuda
        and max(pad[0], pad[1]) < x.shape[-1]
        and max(pad[2], pad[3]) < x.shape[-2]
        and (norm is None or (c & (c - 1) == 0 and c % int(norm.num_groups) == 0 and c <= 1024))
    )


def norm_silu_pad(
    x: Any,
    norm: Any,
    pad: tuple,
    front: int,
    in_bias: Any = None,
) -> Any:
    """Fused pending-bias add + per-frame GroupNorm (float32 statistics) + SiLU + reflect/causal pad,
    channels-last out."""
    import torch

    if not _fusable(x, pad, norm):
        return norm_silu_pad_reference(x, norm, pad, front, in_bias)
    k = _kernels()
    left, right, top, bottom = pad
    b, c, t, h, w = x.shape
    to, ho, wo = t + front, h + top + bottom, w + left + right
    out = torch.empty((b, to, ho, wo, c), dtype = x.dtype, device = x.device).permute(0, 4, 1, 2, 3)
    sb, sc, st, sh, sw = x.stride()
    ib = in_bias if in_bias is not None else x
    if norm is not None:
        g = int(norm.num_groups)
        cpg = c // g
        hw = h * w
        block_p = max(1, min(128, 8192 // c))
        n_chunks = (hw + block_p - 1) // block_p
        rows = b * t
        pmean, pm2 = torch.empty((2, rows * n_chunks * g), dtype = torch.float32, device = x.device)
        pn = torch.empty(rows * n_chunks, dtype = torch.float32, device = x.device)
        # chunks on grid axis 0: axis 1 stops at 65535, and a large untiled frame has more chunks than that
        k.gn_partials[(n_chunks, rows)](
            x,
            ib,
            pn,
            pmean,
            pm2,
            t,
            hw,
            w,
            n_chunks,
            sb,
            sc,
            st,
            sh,
            sw,
            C = c,
            G = g,
            CPG = cpg,
            BLOCK_P = block_p,
            HAS_BIAS = in_bias is not None,
            num_warps = 4,
        )
        mean = torch.empty(rows * g, dtype = torch.float32, device = x.device)
        rstd = torch.empty(rows * g, dtype = torch.float32, device = x.device)
        k.gn_combine[(rows * g,)](
            pn, pmean, pm2, mean, rstd, n_chunks, g, float(norm.eps), BLOCK = 1024, num_warps = 4
        )
        weight, bias = norm.weight, norm.bias
    else:
        # unused without a norm; constant, so the pad-only kernel compiles once for every channel count
        g, cpg = 1, 1
        mean = rstd = weight = bias = x
    # the whole channel run per pixel (up to 128) and 4096 elements per program measured fastest on a B200
    block_c = min(128, _next_pow2(c))
    block_w = max(16, 4096 // block_c)
    grid = (b * to * ho, (wo + block_w - 1) // block_w, (c + block_c - 1) // block_c)
    k.gn_silu_pad[grid](
        x,
        out,
        mean,
        rstd,
        weight,
        bias,
        ib,
        c,
        t,
        h,
        w,
        to,
        ho,
        wo,
        g,
        front,
        top,
        left,
        sb,
        sc,
        st,
        sh,
        sw,
        CPG = cpg,
        HAS_NORM = norm is not None,
        HAS_IN_BIAS = in_bias is not None,
        BLOCK_W = block_w,
        BLOCK_C = block_c,
        num_warps = 4,
    )
    return out


def add_residual(out: Any, out_bias: Any, res: Any, res_bias: Any) -> Any:
    """``out + out_bias + res + res_bias`` written into ``out`` (the conv's own fresh output)."""
    import torch

    k = _kernels()
    c = out.shape[1]
    if (
        k is None
        or not out.is_cuda
        or not out.is_contiguous(memory_format = torch.channels_last_3d)
        or res.shape != out.shape
        or c & (c - 1)
        or c > 1024
    ):
        if out_bias is not None:
            out.add_(_bias_view(out_bias).to(out.dtype))
        if res_bias is not None:
            res = res + _bias_view(res_bias).to(res.dtype)
        return out.add_(res)
    b, _, t, h, w = out.shape
    pixels = b * t * h * w
    block_p = max(1, 4096 // c)
    k.add_residual[((pixels + block_p - 1) // block_p,)](
        out,
        out_bias if out_bias is not None else out,
        res,
        res_bias if res_bias is not None else res,
        pixels,
        t,
        h,
        w,
        *res.stride(),
        C = c,
        HAS_OB = out_bias is not None,
        HAS_RB = res_bias is not None,
        R_SAME = res.is_contiguous(memory_format = torch.channels_last_3d),
        BLOCK_P = block_p,
        num_warps = 4,
    )
    return out


def _last_tap(weight: Any) -> Any:
    """A conv weight's last temporal tap, for a single frame whose causal front padding is all zeros. Sliced per call
    rather than cached: a cached tensor would not follow the module through the offload hooks' ``.to()``."""
    import torch
    return weight[:, :, -1:].contiguous(memory_format = torch.channels_last_3d)


def _causal_conv(
    conv: Any,
    x: Any,
    x_bias: Any = None,
    *,
    norm: Any = None,
    spatial_pad: Optional[tuple] = None,
) -> tuple:
    """``conv(pad(silu(norm(x + x_bias))))`` WITHOUT the conv's bias, which is returned as the output's pending bias."""
    import torch.nn.functional as F

    kt = conv.kernel_size[0]
    sp = int(getattr(conv, "spatial_padding", 0) or 0)
    front = int(getattr(conv, "temporal_padding", 0) or 0)
    pad = (sp, sp, sp, sp) if spatial_pad is None else tuple(spatial_pad)
    weight = conv.weight
    bias = conv.bias
    if x.shape[2] == 1 and kt > 1 and front == kt - 1:
        # zeros times the first taps add exactly nothing
        front = 0
        weight = _last_tap(weight)
    if norm is not None or front or any(pad):
        if x.shape[1] % 8 == 0 or norm is not None:
            x = norm_silu_pad(x, norm, pad, front, in_bias = x_bias)
        else:
            # conv_in's 3 RGB channels: channels-last buys nothing and cuDNN would convert it back
            if x_bias is not None:
                x = x + _bias_view(x_bias).to(x.dtype)
            x = _pad_reference(x, pad, front)
    elif x_bias is not None:
        # a 1x1x1 conv without padding is linear in its input, so the pending bias folds through the weight exactly
        folded = weight.reshape(weight.shape[0], -1).float() @ x_bias.float()
        bias = folded if bias is None else bias.float() + folded
    out = F.conv3d(x, weight, None, stride = conv.stride)
    return out, bias


def _fast_encoder_forward(self: Any, hidden_states: Any) -> Any:
    """``MiniMaxH3VideoEncoder3d.forward`` over the fused passes. Same modules, same order."""
    import torch

    # Triton launches on the current device, which on a multi-GPU host need not be the tensor's
    with torch.cuda.device(hidden_states.device) if hidden_states.is_cuda else nullcontext():
        return _fast_encoder_body(self, hidden_states)


def _fast_encoder_body(self: Any, hidden_states: Any) -> Any:
    dtype = getattr(self, "_unsloth_compute_dtype", None) or hidden_states.dtype
    out_dtype = hidden_states.dtype
    h, hb = _causal_conv(self.conv_in, hidden_states.to(dtype))
    for down_block in self.down_blocks:
        for resnet in down_block.resnets:
            x, xb = h, hb
            h, hb = _causal_conv(resnet.conv1, x, xb, norm = resnet.norm1)
            h, hb = _causal_conv(resnet.conv2, h, hb, norm = resnet.norm2)
            if resnet.conv_shortcut is None:
                r, rb = x, xb
            else:
                r, rb = _causal_conv(resnet.conv_shortcut, x, xb)
            h, hb = add_residual(h, hb, r, rb), None
        if down_block.downsamplers is not None:
            for ds in down_block.downsamplers:
                pad = (0, 1, 0, 1) if ds.spatial_stride == 2 else None
                h, hb = _causal_conv(ds.conv, h, hb, spatial_pad = pad)
    h, hb = _causal_conv(self.conv_out, h, hb, norm = self.norm_out)
    h = h.float()
    if hb is not None:
        h = h + _bias_view(hb).float()
    return h.to(out_dtype).contiguous()


def _install_encoder(vae: Any, *, fp16: bool) -> bool:
    import torch

    encoder = getattr(vae, "encoder", None)
    if encoder is None or not hasattr(encoder, "down_blocks"):
        return False
    # the kernel pads by reflection, which is what the released config uses; any other mode keeps the stock path
    modes = {getattr(m, "spatial_padding_mode", "reflect") for m in encoder.modules()}
    if modes - {"reflect"}:
        return False
    convs = [m for m in encoder.modules() if isinstance(m, torch.nn.Conv3d)]
    with torch.no_grad():
        # every new tensor first, then assign: a failure part way (an OOM on a resident VAE) must leave the stock
        # encoder exactly as it was, not half float16
        new = []
        for module in convs:
            w = module.weight.data.to(torch.float16) if fp16 else module.weight.data
            if module is not encoder.conv_in:
                w = w.contiguous(memory_format = torch.channels_last_3d)
            bias = module.bias
            b = None if bias is None else (bias.data.to(torch.float16) if fp16 else bias.data)
            new.append((module, w, b))
        for module, w, b in new:
            module.weight.data = w
            if b is not None:
                module.bias.data = b
    encoder._unsloth_compute_dtype = torch.float16 if fp16 else None
    encoder._unsloth_stock_forward = encoder.forward
    encoder.forward = types.MethodType(
        _guarded(_fast_encoder_forward, _stock_encoder_forward, "encoder"), encoder
    )
    return True


def _stock_encoder_forward(self: Any, hidden_states: Any) -> Any:
    """Diffusers' own encoder forward over the (possibly float16) weights: under float16 autocast when the weights
    were cast, which is the arithmetic the fused float16 path reproduces."""
    import torch

    if getattr(self, "_unsloth_compute_dtype", None) is not torch.float16:
        return self._unsloth_stock_forward(hidden_states)
    if hidden_states.is_cuda:
        with torch.autocast("cuda", dtype = torch.float16):
            return self._unsloth_stock_forward(hidden_states).to(hidden_states.dtype)
    return self._unsloth_stock_forward(hidden_states.to(torch.float16)).to(hidden_states.dtype)


# ── decoder ────────────────────────────────────────────────────────────────────────────────────────────────────────


def _add_rmsnorm(h: Any, o: Any, scale: Any, norm: Any, out_dtype: Any) -> Any:
    """``h += o * scale`` (skipped when ``o`` is None), then ``rmsnorm(h) * norm.weight`` in ``out_dtype``."""
    import torch

    k = _kernels()
    n_cols = h.shape[-1]
    eps = norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps
    if k is None or not h.is_cuda:
        if o is not None:
            h.add_(o * scale)
        return torch.nn.functional.rms_norm(h.float(), (n_cols,), norm.weight.float(), eps).to(
            out_dtype
        )
    h2 = h.view(-1, n_cols)
    normed = torch.empty(h2.shape, dtype = out_dtype, device = h.device)
    k.add_rmsnorm[(h2.shape[0],)](
        h2,
        o if o is not None else h2,
        scale if scale is not None else norm.weight,
        norm.weight,
        normed,
        n_cols,
        float(eps),
        HAS_RES = o is not None,
        BLOCK_N = _next_pow2(n_cols),
        num_warps = 8,
    )
    return normed.view(*h.shape[:-1], n_cols)


def qk_norm_rope_reference(x: Any, cos: Any, sin: Any, heads: int, eps: float) -> Any:
    """The reference processor's per-head float32 RMSNorm then partial split-half rope in ``x.dtype``."""
    import torch

    shape = x.shape
    q = x.reshape(-1, heads, shape[-1] // heads)
    q = torch.nn.functional.rms_norm(q.float(), (q.shape[-1],), None, eps).to(x.dtype)
    rot = cos.shape[-1]
    c = cos.reshape(-1, 1, rot).to(x.dtype)
    s = sin.reshape(-1, 1, rot).to(x.dtype)
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    first, second = q_rot.chunk(2, dim = -1)
    rotated = torch.cat([-second, first], dim = -1)
    return torch.cat([q_rot * c + rotated * s, q_pass], dim = -1).reshape(shape)


def _qk_norm_rope_(x: Any, cos: Any, sin: Any, heads: int, eps: float) -> None:
    k = _kernels()
    if k is None or not x.is_cuda or not x.is_contiguous():
        x.copy_(qk_norm_rope_reference(x, cos, sin, heads, eps))
        return
    head_d = x.shape[-1] // heads
    n = x.numel() // head_d
    # 32 head-rows x 4 warps measured 1.4x a plain copy of the same bytes on a B200; 128 was 5x
    block = 32
    k.qk_norm_rope[((n + block - 1) // block,)](
        x,
        cos,
        sin,
        n,
        heads,
        cos.stride(-2),
        float(eps),
        HEAD_D = head_d,
        ROT = cos.shape[-1],
        BLOCK = block,
        num_warps = 4,
    )


def _swiglu(x: Any) -> Any:
    import torch

    k = _kernels()
    n = x.shape[-1] // 2
    if k is None or not x.is_cuda:
        hidden, gate = x.chunk(2, dim = -1)
        return hidden * torch.nn.functional.silu(gate)
    rows = x.numel() // x.shape[-1]
    out = torch.empty((*x.shape[:-1], n), dtype = x.dtype, device = x.device)
    block_n = 2048
    k.swiglu[(rows, (n + block_n - 1) // block_n)](x, out, n, BLOCK_N = block_n, num_warps = 4)
    return out


def _fast_block_stack(decoder: Any, hidden_states: Any, cos: Any, sin: Any, gemm_dtype: Any) -> Any:
    """The transformer blocks. ``gemm_dtype`` is the autocast dtype the GEMMs run in; the kernels store the normed
    activations straight in it."""
    import torch.nn.functional as F

    blocks = decoder.transformer_blocks
    batch, seq, dim = hidden_states.shape
    h = hidden_states.contiguous()
    normed = _add_rmsnorm(h, None, None, blocks[0].norm1, gemm_dtype)
    cos2 = cos.reshape(-1, cos.shape[-1])
    sin2 = sin.reshape(-1, sin.shape[-1])
    for i, block in enumerate(blocks):
        attn = block.attn
        query = attn.to_q(normed)
        key = attn.to_k(normed)
        value = attn.to_v(normed)
        _qk_norm_rope_(query, cos2, sin2, attn.heads, attn.norm_q.eps)
        _qk_norm_rope_(key, cos2, sin2, attn.heads, attn.norm_k.eps)
        query, key, value = (
            t.view(batch, seq, attn.heads, -1).permute(0, 2, 1, 3) for t in (query, key, value)
        )
        out = F.scaled_dot_product_attention(query, key, value)
        out = out.permute(0, 2, 1, 3).flatten(2, 3)
        out = attn.to_out[0](out)
        normed = _add_rmsnorm(h, out, block.scale1, block.norm2, gemm_dtype)
        ff = block.ff.net
        proj = ff[0].proj(normed)
        out = ff[2](_swiglu(proj))
        if i + 1 < len(blocks):
            normed = _add_rmsnorm(h, out, block.scale2, blocks[i + 1].norm1, gemm_dtype)
        else:
            h.addcmul_(out, block.scale2)
    return h


def _fast_decoder_forward(self: Any, hidden_states: Any) -> Any:
    """``MiniMaxH3VideoViTDecoder3d.forward`` with the block stack over the fused kernels."""
    import torch
    with torch.cuda.device(hidden_states.device) if hidden_states.is_cuda else nullcontext():
        return _fast_decoder_body(self, hidden_states)


def _fast_decoder_body(self: Any, hidden_states: Any) -> Any:
    import torch

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
        batch_size, num_frames * height * width, num_channels
    )
    hidden_states = self.proj_in(hidden_states)
    num_patches = hidden_states.shape[1]
    register_tokens = self.register_tokens.expand(batch_size, -1, -1)
    cls_token = torch.zeros_like(hidden_states[:, :1, :])
    hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim = 1)
    grids = [
        2.0 * (torch.arange(0.5, size, dtype = torch.float32, device = hidden_states.device) / size)
        - 1.0
        for size in (num_frames, height, width)
    ]
    position_ids = torch.stack(torch.meshgrid(*grids, indexing = "ij"), dim = -1).flatten(0, 2)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
    suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
    position_ids = torch.cat([position_ids, suffix_ids], dim = 1)
    cos, sin = self.rope(position_ids)
    # the reference casts the tables to the query dtype, which is the GEMM (autocast) dtype
    q_dtype = (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled("cuda")
        else hidden_states.dtype
    )
    cos = cos.to(q_dtype).contiguous()
    sin = sin.to(q_dtype).contiguous()
    hidden_states = _fast_block_stack(self, hidden_states, cos, sin, q_dtype)
    hidden_states = self.norm_out(hidden_states)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states[:, :num_patches, :]
    p, pt = self.patch_size, self.patch_size_t
    hidden_states = hidden_states.view(
        batch_size, num_frames, height, width, self.out_channels, pt, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return hidden_states.reshape(
        batch_size, self.out_channels, num_frames * pt, height * p, width * p
    )


def _decoder_fusable(decoder: Any) -> bool:
    try:
        blocks = decoder.transformer_blocks
        attn = blocks[0].attn
        return (
            hasattr(attn, "to_q")
            and hasattr(attn, "to_k")
            and hasattr(attn, "to_v")
            and hasattr(blocks[0].ff, "net")
            and hasattr(blocks[0].ff.net[0], "proj")
            and hasattr(decoder, "rope")
            and hasattr(decoder, "register_tokens")
            # the rope kernel spans one head in a single power-of-two block
            and attn.dim_head & (attn.dim_head - 1) == 0
        )
    except Exception:  # noqa: BLE001
        return False


def _install_decoder(vae: Any) -> bool:
    decoder = getattr(vae, "decoder", None)
    if decoder is None or not _decoder_fusable(decoder):
        return False
    decoder._unsloth_stock_forward = decoder.forward
    decoder.forward = types.MethodType(
        _guarded(_fast_decoder_forward, lambda self, x: self._unsloth_stock_forward(x), "decoder"),
        decoder,
    )
    return True


# ── tile batching ─────────────────────────────────────────────────────────────────────────────────────────────────


def _tile_batch_size(z: Any) -> int:
    import torch

    env = _env(H3_VAE_TILE_BATCH_ENV)
    if env.isdigit() and int(env) > 0:
        return int(env)
    try:
        free, _ = torch.cuda.mem_get_info(z.device)
    except Exception:  # noqa: BLE001
        return 1
    return int(
        max(1, min(_TILE_BATCH_MAX, free // (_TILE_BATCH_BYTES_PER_TILE * max(1, z.shape[0]))))
    )


def _batched_decode_clip(self: Any, z: Any) -> Any:
    """``AutoencoderKLMiniMaxH3._decode_clip`` with the tiles of a clip decoded several at a time. ``_split_tiles``
    gives every tile of an axis the same length, so they stack along batch."""
    import torch

    if not self.use_tiling:
        return self.decoder(self.post_quant_conv(z))
    height = z.shape[-2] * self.spatial_compression_ratio
    width = z.shape[-1] * self.spatial_compression_ratio
    y_indices, y_lengths, y_overlaps = self._split_tiles(
        height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
    )
    x_indices, x_lengths, x_overlaps = self._split_tiles(
        width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
    )
    ratio = self.spatial_compression_ratio
    tiles = [
        z[
            ...,
            i_pos // ratio : i_pos // ratio + i_len // ratio,
            j_pos // ratio : j_pos // ratio + j_len // ratio,
        ]
        for i_pos, i_len in zip(y_indices, y_lengths)
        for j_pos, j_len in zip(x_indices, x_lengths)
    ]
    per = z.shape[0]
    batch = _tile_batch_size(z)
    decoded: list = []
    start = 0
    while start < len(tiles):
        group = tiles[start : start + batch]
        try:
            out = self.decoder(
                self.post_quant_conv(torch.cat(group) if len(group) > 1 else group[0])
            )
        except Exception as exc:  # noqa: BLE001
            # batching is what raised the peak, so an OOM with several tiles retries them one at a time
            if len(group) == 1 or not _is_oom(exc):
                raise
            torch.cuda.empty_cache()
            batch = 1
            continue
        decoded.extend(out.split(per) if len(group) > 1 else [out])
        start += len(group)
    cols = len(x_indices)
    rows = [decoded[r * cols : (r + 1) * cols] for r in range(len(y_indices))]
    return self._stitch_tiles(rows, y_overlaps, x_overlaps)


def _install_tile_batch(vae: Any) -> bool:
    if not all(
        hasattr(vae, n)
        for n in ("_decode_clip", "_split_tiles", "_stitch_tiles", "post_quant_conv")
    ):
        return False
    vae._decode_clip = types.MethodType(_batched_decode_clip, vae)
    return True


# ── int8 decoder ───────────────────────────────────────────────────────────────────────────────────────────────────


def _int8_linear(linear: Any, x: Any) -> Any:
    """W8A8: per-token int8 activations, per-output-channel int8 weights, int32 accumulate via ``torch._int_mm``."""
    import torch

    if x.is_cuda and x.device.index is not None and x.device.index != torch.cuda.current_device():
        with torch.cuda.device(x.device):
            return _int8_linear(linear, x)
    k = _kernels()
    wq, ws = linear._unsloth_int8
    out_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    rot = getattr(linear, "_unsloth_rot", None)
    if rot is not None:
        # ConvRot: x @ blockdiag(H), the weight was stored as W @ blockdiag(H).T, so the product is unchanged while
        # the Hadamard spreads the outlier channels that would otherwise pin each token's int8 scale
        g = rot.shape[0]
        x = (x.reshape(-1, x.shape[-1] // g, g).to(out_dtype) @ rot.to(out_dtype)).reshape(x.shape)
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    rows, n_in = x2.shape
    if rows <= 16:
        w = (wq.float() * ws[:, None]).to(out_dtype)
        return torch.nn.functional.linear(x2.to(out_dtype), w, linear.bias).view(*x.shape[:-1], -1)
    if k is None or not x.is_cuda:
        xf = x2.float()
        xs = xf.abs().amax(dim = 1).clamp(min = 1e-12) / 127.0
        xq = (xf / xs[:, None]).round().clamp(-127, 127).to(torch.int8)
        acc = torch._int_mm(xq, wq.t()).float() * xs[:, None] * ws[None, :]
        if linear.bias is not None:
            acc = acc + linear.bias.float()
        return acc.to(out_dtype).view(*x.shape[:-1], -1)
    xq = torch.empty((rows, n_in), dtype = torch.int8, device = x.device)
    xs = torch.empty(rows, dtype = torch.float32, device = x.device)
    k.quant_rows[(rows,)](x2, xq, xs, n_in, BLOCK_N = _next_pow2(n_in), num_warps = 8)
    acc = torch._int_mm(xq, wq.t())
    n_out = acc.shape[1]
    out = torch.empty((rows, n_out), dtype = out_dtype, device = x.device)
    block_n = 1024
    bias = linear.bias
    k.dequant_epilogue[(rows, (n_out + block_n - 1) // block_n)](
        acc,
        xs,
        ws,
        bias if bias is not None else ws,
        out,
        n_out,
        HAS_BIAS = bias is not None,
        BLOCK_N = block_n,
        num_warps = 4,
    )
    return out.view(*x.shape[:-1], n_out)


def _install_int8_decoder(vae: Any, *, keep_blocks: Optional[tuple] = None) -> int:
    """Quantize the decoder's transformer-block Linears to int8 (per output channel), ConvRot-rotated in
    ``H3_VAE_INT8_ROT_GROUP`` blocks when the input width allows, leaving ``keep_blocks`` (default: the first
    ``H3_VAE_INT8_FLOAT_BLOCKS``) in float. The float weight is dropped; the class is swapped so the stock decoder
    path runs W8A8 as well. Returns the bytes freed."""
    import torch

    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        return 0
    from .diffusion_convrot import build_convrot_hadamard

    freed = 0
    group = H3_VAE_INT8_ROT_GROUP
    hadamard = None
    with torch.no_grad():
        if keep_blocks is None:
            keep_blocks = tuple(range(H3_VAE_INT8_FLOAT_BLOCKS))
        for index, block in enumerate(decoder.transformer_blocks):
            if index in keep_blocks:
                continue
            for linear in block.modules():
                if (
                    not isinstance(linear, torch.nn.Linear)
                    or getattr(linear, "_unsloth_int8", None) is not None
                ):
                    continue
                if linear.in_features % 8 or linear.out_features % 8:
                    continue
                w = linear.weight.data.float()
                if linear.in_features % group == 0:
                    if hadamard is None:
                        hadamard = build_convrot_hadamard(
                            group, device = w.device, dtype = torch.float32
                        )
                    w = (w.reshape(w.shape[0], -1, group) @ hadamard.T).reshape(w.shape)
                    # exact in float16: the entries are +-2^-k
                    linear.register_buffer(
                        "_unsloth_rot", hadamard.to(torch.float16), persistent = False
                    )
                scale = w.abs().amax(dim = 1).clamp(min = 1e-12) / 127.0
                wq = (w / scale[:, None]).round_().clamp_(-127, 127).to(torch.int8)
                freed += linear.weight.numel() * linear.weight.element_size() - wq.numel()
                linear.register_buffer("_unsloth_wq", wq)
                linear.register_buffer("_unsloth_ws", scale.to(torch.float32).view(torch.int32))
                del linear.weight
                linear.weight = None
                linear.__class__ = _int8_linear_class()
    return freed


@lru_cache(maxsize = 1)
def _int8_linear_class():
    import torch
    class H3Int8Linear(torch.nn.Linear):
        """A Linear whose weight lives in ``_unsloth_wq`` (int8) and ``_unsloth_ws`` (fp32 scales stored as int32,
        so a module-wide ``.to(dtype)`` from an offload hook cannot round them)."""

        @property
        def _unsloth_int8(self):
            return self._unsloth_wq, self._unsloth_ws.view(torch.float32)

        def forward(self, x):
            return _int8_linear(self, x)

    return H3Int8Linear


# ── fp16 accumulation scope ────────────────────────────────────────────────────────────────────────────────────────


def _install_decode_scope(vae: Any, *, fp16_accum: bool) -> bool:
    """Pin ``allow_fp16_accumulation`` for the duration of the decode: ON only when the plan says so, otherwise
    OFF even if a bf16 denoiser's speed layer turned the process-wide flag on (the decode is float16 autocast,
    so the flag would change it)."""
    import torch

    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_fp16_accumulation"):
        return False
    stock = vae.decode

    def decode(self, *args, **kwargs):
        prev = matmul.allow_fp16_accumulation
        matmul.allow_fp16_accumulation = bool(fp16_accum)
        try:
            return stock(*args, **kwargs)
        finally:
            matmul.allow_fp16_accumulation = prev

    vae.decode = types.MethodType(decode, vae)
    return True


# ── entry point ────────────────────────────────────────────────────────────────────────────────────────────────────


def cuda_fast_path_available(vae: Any = None) -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    if vae is not None:
        try:
            if type(vae).__name__ != "AutoencoderKLMiniMaxH3":
                return False
        except Exception:  # noqa: BLE001
            return False
    return _kernels() is not None


def apply_h3_vae_speedups(
    vae: Any,
    *,
    speed_mode: Optional[str],
    workflow: Optional[str] = None,
    consumer_gpu: Optional[bool] = None,
    logger: Any = None,
) -> tuple[str, ...]:
    """Engage the planned levers on a loaded H3 video VAE. Never raises: a lever that fails is logged and skipped,
    and the stock Diffusers path stays in place for it. Returns the engaged lever names."""
    if vae is None:
        return ()
    done = getattr(vae, "_unsloth_h3_vae_levers", None)
    if done is not None:
        return tuple(done)
    if consumer_gpu is None:
        try:
            from .diffusion_transformer_quant import _is_consumer_gpu
            consumer_gpu = bool(_is_consumer_gpu())
        except Exception:  # noqa: BLE001
            consumer_gpu = False
    planned = plan_h3_vae_levers(speed_mode, workflow = workflow, consumer_gpu = consumer_gpu)
    engaged: list[str] = []
    if not planned or not cuda_fast_path_available(vae):
        try:
            vae._unsloth_h3_vae_levers = ()
        except Exception:  # noqa: BLE001
            pass
        return ()

    def _try(name: str, fn) -> None:
        try:
            if fn():
                engaged.append(name)
        except Exception as exc:  # noqa: BLE001 - optimisation only
            if logger is not None:
                logger.warning(
                    "video.h3_vae_fast: %s failed, keeping the stock path: %s", name, exc
                )

    for holder in (vae, getattr(vae, "encoder", None), getattr(vae, "decoder", None)):
        if holder is not None:
            try:
                holder._unsloth_logger = logger
            except Exception:  # noqa: BLE001
                pass
    if LEVER_FUSED_ENCODER in planned:
        _try(LEVER_FUSED_ENCODER, lambda: _install_encoder(vae, fp16 = LEVER_FP16_ENCODER in planned))
        if LEVER_FUSED_ENCODER in engaged and LEVER_FP16_ENCODER in planned:
            engaged.append(LEVER_FP16_ENCODER)
    if LEVER_INT8_DECODER in planned:
        _try(LEVER_INT8_DECODER, lambda: _install_int8_decoder(vae) > 0)
    if LEVER_FUSED_DECODER in planned:
        _try(LEVER_FUSED_DECODER, lambda: _install_decoder(vae))
    if LEVER_TILE_BATCH in planned:
        _try(LEVER_TILE_BATCH, lambda: _install_tile_batch(vae))
    fp16_accum = LEVER_FP16_ACCUM in planned
    # installed on every engaged load: it also holds the flag OFF when float16 accumulation is not planned
    try:
        if _install_decode_scope(vae, fp16_accum = fp16_accum) and fp16_accum:
            engaged.append(LEVER_FP16_ACCUM)
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.warning(
                "video.h3_vae_fast: decode_scope failed, keeping the stock decode: %s", exc
            )
    vae._unsloth_h3_vae_levers = tuple(engaged)
    if logger is not None:
        logger.info("video.h3_vae_fast: engaged %s", ", ".join(engaged) or "nothing")
    return tuple(engaged)


def settle_h3_vae_fallback(state: Any, pipe: Any) -> None:
    """After a render, drop a fused lever the runtime fell back from out of ``state.speed_optims``, the way
    ``settle_compile_fallback`` does for a compile, so the status reports what actually ran. Never raises."""
    try:
        vae = getattr(pipe, "vae", None)
        failed = [
            f"h3_vae_{lever}"
            for lever, part in ((LEVER_FUSED_ENCODER, "encoder"), (LEVER_FUSED_DECODER, "decoder"))
            if getattr(getattr(vae, part, None), "_unsloth_fast_failed", False)
        ]
        optims = tuple(getattr(state, "speed_optims", None) or ())
        updated = tuple(o for o in optims if o not in failed)
        if updated != optims:
            # the load states are frozen dataclasses
            object.__setattr__(state, "speed_optims", updated)
    except Exception:  # noqa: BLE001 - a status fix-up must never fail a render
        pass
