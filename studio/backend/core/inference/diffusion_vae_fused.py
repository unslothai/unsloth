# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton-fused normalisation for the image and video VAEs Studio decodes eagerly.

PyTorch's CUDA GroupNorm has no channels-last kernel (up to torch 2.13): every norm of a channels_last VAE copies
NHWC -> NCHW, runs a slow bf16 row reduction and copies back, 80% of an eager AutoencoderKL decode. Channel RMS
norms (Wan / Qwen-Image / HunyuanVideo-1.5) run as ~6 separate elementwise passes plus a ``torch.cat`` / ``clone``
per causal conv. The kernels here read any layout and write channels-last, with fp32 statistics:

* ``group_norm_act``: GroupNorm (+ pending bias) (+ SiLU), split-spatial fp32 statistics then one apply pass.
* ``rms_norm_act``: per-pixel channel RMS norm (+ SiLU), optionally prepending causal cache frames (the conv input
  the stock code builds with ``torch.cat`` + ``F.pad``), in one pass.

``install(vae)`` patches module instances in place (never classes), each guarded: an exception the stock path
does not also raise falls back to stock for that module for good. NVIDIA CUDA + Triton >= 3.3 only (ROCm, CPU, MPS,
Windows without the MSVC toolchain and old Triton keep the stock path). ``UNSLOTH_VAE_FUSED=0`` disables.
The GroupNorm statistics split (per-chunk mean/M2 merged with Chan's formula) follows video_minimax_h3_vae.py.
"""

from __future__ import annotations

import os
import sys
import types
from functools import lru_cache
from typing import Any, Optional

VAE_FUSED_ENV = "UNSLOTH_VAE_FUSED"
_FALSE = ("0", "false", "no", "off")
_MIN_TRITON = (3, 3)
_MAX_C = 4096


def fused_vae_disabled() -> bool:
    return os.environ.get(VAE_FUSED_ENV, "").strip().lower() in _FALSE


@lru_cache(maxsize = 1)
def _triton_version_ok() -> bool:
    try:
        import re

        import triton
        m = re.match(r"(\d+)\.(\d+)", str(triton.__version__))
        return bool(m) and (int(m.group(1)), int(m.group(2))) >= _MIN_TRITON
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize = 1)
def _toolchain_ok() -> bool:
    if sys.platform != "win32":
        return True
    try:
        from .._msvc_env import crt_headers_reachable
        return bool(crt_headers_reachable())
    except Exception:  # noqa: BLE001
        return True


@lru_cache(maxsize = 1)
def _kernels() -> Optional[types.SimpleNamespace]:
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001
        return None
    globals().update(triton = triton, tl = tl)

    @triton.jit
    def _gn_partials(
        x_ptr, bias_ptr, pn_ptr, pmean_ptr, pm2_ptr,
        T, HW, W, n_chunks, sb, sc, st, sh, sw,
        C: tl.constexpr, G: tl.constexpr, CPG: tl.constexpr, BLOCK_P: tl.constexpr, HAS_BIAS: tl.constexpr,
    ):  # fmt: skip
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
    def _gn_combine(pn_ptr, pmean_ptr, pm2_ptr, mean_ptr, rstd_ptr, n_chunks, G, eps, BLOCK: tl.constexpr):
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
    def _gn_apply(
        x_ptr, out_ptr, mean_ptr, rstd_ptr, w_ptr, b_ptr, ib_ptr,
        C, T, HW, W, G, sb, sc, st, sh, sw,
        CPG: tl.constexpr, ACT: tl.constexpr, HAS_IN_BIAS: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
    ):  # fmt: skip
        pid_p = tl.program_id(0)
        bt = tl.program_id(1)
        pid_c = tl.program_id(2)
        t = bt % T
        b = bt // T
        p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        pmask = p < HW
        hh = p // W
        ww = p - hh * W
        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        cmask = c < C
        m = pmask[:, None] & cmask[None, :]
        off = (
            b.to(tl.int64) * sb
            + t.to(tl.int64) * st
            + hh.to(tl.int64)[:, None] * sh
            + ww.to(tl.int64)[:, None] * sw
            + c.to(tl.int64)[None, :] * sc
        )
        v = tl.load(x_ptr + off, mask = m, other = 0.0).to(tl.float32)
        if HAS_IN_BIAS:
            v = v + tl.load(ib_ptr + c, mask = cmask, other = 0.0).to(tl.float32)[None, :]
        gidx = bt * G + c // CPG
        mu = tl.load(mean_ptr + gidx, mask = cmask, other = 0.0)
        rs = tl.load(rstd_ptr + gidx, mask = cmask, other = 0.0)
        gw = tl.load(w_ptr + c, mask = cmask, other = 0.0).to(tl.float32)
        gb = tl.load(b_ptr + c, mask = cmask, other = 0.0).to(tl.float32)
        a = rs * gw
        v = v * a[None, :] + (gb - mu * a)[None, :]
        if ACT:
            v = v / (1.0 + tl.exp(-v))
        out_off = (bt.to(tl.int64) * HW + p.to(tl.int64))[:, None] * C + c[None, :]
        tl.store(out_ptr + out_off, v.to(out_ptr.dtype.element_ty), mask = m)

    @triton.jit
    def _rms_act(
        x_ptr, out_ptr, w_ptr, b_ptr, ib_ptr, cache_ptr,
        C, T, H, W, Ho, Wo, ph, pw, To, front, n_cache, scale, eps,
        sb, sc, st, sh, sw, cb, cc, ct, ch, cw,
        NORM: tl.constexpr, ACT: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_IN_BIAS: tl.constexpr,
        REPLICATE: tl.constexpr, MEAN_SQ: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
    ):  # fmt: skip
        # out (B, To, Ho, Wo, C) = [front frames] + act(norm(x + ib)); REPLICATE: replicate-padded in T (front), H, W
        pid_p = tl.program_id(0)
        bt = tl.program_id(1)
        t_o = bt % To
        b = bt // To
        p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        pmask = p < Ho * Wo
        ho = p // Wo
        wo = p - ho * Wo
        if REPLICATE:
            hh = tl.minimum(tl.maximum(ho - ph, 0), H - 1)
            ww = tl.minimum(tl.maximum(wo - pw, 0), W - 1)
        else:
            hh = ho
            ww = wo
        c = tl.arange(0, BLOCK_C)
        cmask = c < C
        m = pmask[:, None] & cmask[None, :]
        out_off = ((b.to(tl.int64) * To + t_o) * (Ho * Wo) + p.to(tl.int64))[:, None] * C + c[None, :]
        if (not REPLICATE) and t_o < front:
            # causal frames: the cache's last ``n_cache`` activations, zeros before them
            tc = t_o - (front - n_cache)
            if tc >= 0:
                coff = (
                    b.to(tl.int64) * cb
                    + tc.to(tl.int64) * ct
                    + hh.to(tl.int64)[:, None] * ch
                    + ww.to(tl.int64)[:, None] * cw
                    + c.to(tl.int64)[None, :] * cc
                )
                v = tl.load(cache_ptr + coff, mask = m, other = 0.0)
                tl.store(out_ptr + out_off, v.to(out_ptr.dtype.element_ty), mask = m)
            else:
                tl.store(out_ptr + out_off, tl.zeros([BLOCK_P, BLOCK_C], out_ptr.dtype.element_ty), mask = m)
        else:
            t = tl.minimum(tl.maximum(t_o - front, 0), T - 1)
            off = (
                b.to(tl.int64) * sb
                + t.to(tl.int64) * st
                + hh.to(tl.int64)[:, None] * sh
                + ww.to(tl.int64)[:, None] * sw
                + c.to(tl.int64)[None, :] * sc
            )
            v = tl.load(x_ptr + off, mask = m, other = 0.0).to(tl.float32)
            if HAS_IN_BIAS:
                v = v + tl.load(ib_ptr + c, mask = cmask, other = 0.0).to(tl.float32)[None, :]
                v = tl.where(m, v, 0.0)
            if NORM:
                ss = tl.sum(v * v, axis = 1)
                if MEAN_SQ:
                    # LTX PerChannelRMSNorm: x / sqrt(mean(x^2) + eps), no affine
                    v = v / tl.sqrt(ss / C + eps)[:, None]
                else:
                    # Wan-lineage RMS_norm: F.normalize(x) * sqrt(C) * gamma (+ bias)
                    inv = 1.0 / tl.maximum(tl.sqrt(ss), 1e-12)
                    gw = tl.load(w_ptr + c, mask = cmask, other = 0.0).to(tl.float32) * scale
                    v = v * inv[:, None] * gw[None, :]
                    if HAS_BIAS:
                        v = v + tl.load(b_ptr + c, mask = cmask, other = 0.0).to(tl.float32)[None, :]
            if ACT:
                v = v / (1.0 + tl.exp(-v))
            tl.store(out_ptr + out_off, v.to(out_ptr.dtype.element_ty), mask = m)

    @triton.jit
    def _bias_residual(
        o_ptr, ob_ptr, r_ptr, rb_ptr, P, C, T, HW, W, sb, sc, st, sh, sw, inv_scale,
        HAS_OB: tl.constexpr, HAS_RB: tl.constexpr, SCALE: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
    ):  # fmt: skip
        # o (channels-last, contiguous: pixel-major, C fastest) += ob + r + rb, then * inv_scale
        p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        pm = p < P
        cm = c < C
        m = pm[:, None] & cm[None, :]
        o_off = p.to(tl.int64)[:, None] * C + c[None, :]
        o = tl.load(o_ptr + o_off, mask = m, other = 0.0).to(tl.float32)
        if HAS_OB:
            o = o + tl.load(ob_ptr + c, mask = cm, other = 0.0).to(tl.float32)[None, :]
        ww = p % W
        rest = p // W
        hh = rest % (HW // W)
        bt = rest // (HW // W)
        t = bt % T
        b = bt // T
        r_off = (
            b.to(tl.int64) * sb + t.to(tl.int64) * st + hh.to(tl.int64) * sh + ww.to(tl.int64) * sw
        )[:, None] + c.to(tl.int64)[None, :] * sc
        r = tl.load(r_ptr + r_off, mask = m, other = 0.0).to(tl.float32)
        if HAS_RB:
            r = r + tl.load(rb_ptr + c, mask = cm, other = 0.0).to(tl.float32)[None, :]
        o = o + r
        if SCALE:
            o = o * inv_scale
        tl.store(o_ptr + o_off, o.to(o_ptr.dtype.element_ty), mask = m)

    @triton.jit
    def _dcae_up_add(
        h_ptr, hb_ptr, x_ptr, out_ptr,
        c, Cin, F, H, W, Fo, rep,
        hsb, hsc, hst, hsh, hsw, xsb, xsc, xst, xsh, xsw,
        TEMPORAL: tl.constexpr, HAS_HB: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
    ):  # fmt: skip
        # HunyuanVideo-1.5 upsample: depth-to-space of the conv output + the repeat_interleave'd shortcut, one pass
        pid_p = tl.program_id(0)
        bf = tl.program_id(1)
        pid_c = tl.program_id(2)
        fo = bf % Fo
        b = bf // Fo
        Wo = 2 * W
        p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = p < 4 * H * W
        ho = p // Wo
        wo = p - ho * Wo
        hh = ho // 2
        ww = wo // 2
        ph = (ho - hh * 2) * 2 + (wo - ww * 2)  # r2i * 2 + r3i
        ci = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        cm = ci < c
        m = pm[:, None] & cm[None, :]
        if TEMPORAL:
            if fo == 0:
                f = fo * 0
                pk = ph[:, None] * (2 * c) + ci[None, :]
                xk = ph[:, None] * (Cin // 4) + ci[None, :] // (rep // 2)
            else:
                f = 1 + (fo - 1) // 2
                r1 = (fo - 1) % 2
                pk = (r1 * 4 + ph)[:, None] * c + ci[None, :]
                xk = pk // rep
        else:
            f = fo
            pk = ph[:, None] * c + ci[None, :]
            xk = pk // rep
        pix_h = b.to(tl.int64) * hsb + f.to(tl.int64) * hst + (hh.to(tl.int64) * hsh + ww.to(tl.int64) * hsw)[:, None]
        v = tl.load(h_ptr + pix_h + pk.to(tl.int64) * hsc, mask = m, other = 0.0).to(tl.float32)
        if HAS_HB:
            v = v + tl.load(hb_ptr + pk, mask = m, other = 0.0).to(tl.float32)
        v = v.to(out_ptr.dtype.element_ty).to(tl.float32)  # the stock conv output is rounded before the add
        pix_x = b.to(tl.int64) * xsb + f.to(tl.int64) * xst + (hh.to(tl.int64) * xsh + ww.to(tl.int64) * xsw)[:, None]
        xv = tl.load(x_ptr + pix_x + xk.to(tl.int64) * xsc, mask = m, other = 0.0).to(tl.float32)
        o_off = (bf.to(tl.int64) * (4 * H * W) + p.to(tl.int64))[:, None] * c + ci[None, :]
        tl.store(out_ptr + o_off, (v + xv).to(out_ptr.dtype.element_ty), mask = m)

    return types.SimpleNamespace(
        gn_partials = _gn_partials, gn_combine = _gn_combine, gn_apply = _gn_apply, rms_act = _rms_act,
        bias_residual = _bias_residual, dcae_up_add = _dcae_up_add,
    )


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


def runtime_ok() -> bool:
    """NVIDIA CUDA + a usable Triton; everything else keeps the stock path."""
    if fused_vae_disabled():
        return False
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    return _triton_version_ok() and _toolchain_ok() and _kernels() is not None


def _as5d(x: Any) -> tuple:
    """(tensor viewed as B,C,T,H,W, was_4d)."""
    if x.dim() == 4:
        return x.unsqueeze(2), True
    if x.dim() == 3:  # (B, C, L): attention's GroupNorm input
        return x.unsqueeze(2).unsqueeze(2), True
    return x, False


# ----------------------------------------------------------------------------------------------------------------
# GroupNorm


def group_norm_reference(x: Any, norm: Any, act: bool, in_bias: Any = None) -> Any:
    import torch.nn.functional as F

    if in_bias is not None:
        x = x + in_bias.view(1, -1, *([1] * (x.dim() - 2))).to(x.dtype)
    y = F.group_norm(x, norm.num_groups, norm.weight, norm.bias, norm.eps)
    return F.silu(y) if act else y


def _gn_fusable(x: Any, norm: Any) -> bool:
    c = x.shape[1] if x.dim() >= 2 else 0
    g = int(getattr(norm, "num_groups", 0) or 0)
    return (
        x.is_cuda
        and x.dim() in (3, 4, 5)
        and g > 0
        and c % g == 0
        and c & (c - 1) == 0
        and g & (g - 1) == 0
        and c <= 1024
        and getattr(norm, "affine", True)
        and norm.weight is not None
        and x.dtype in (_torch().float16, _torch().bfloat16, _torch().float32)
        and x.numel() > 0
    )


def _torch():
    import torch
    return torch


def group_norm_act(x: Any, norm: Any, act: bool = True, in_bias: Any = None) -> Any:
    """``silu?(group_norm(x + in_bias))`` for a 3/4/5-D ``x`` of any layout; channels-last output.

    GroupNorm over (C/G, spatial) per sample; for 5-D input the statistics span T as well (``nn.GroupNorm``)."""
    torch = _torch()
    if not _gn_fusable(x, norm):
        return group_norm_reference(x, norm, act, in_bias)
    k = _kernels()
    ndim = x.dim()
    x5, _ = _as5d(x)
    b, c, t, h, w = x5.shape
    # nn.GroupNorm pools the whole (T, H, W) extent: flatten T into H when strides allow, else fall back
    if t > 1:
        if x5.stride(2) == h * x5.stride(3):
            x5 = x5.as_strided((b, c, 1, t * h, w), (x5.stride(0), x5.stride(1), 0, x5.stride(3), x5.stride(4)))
            b, c, t, h, w = x5.shape
        else:
            return group_norm_reference(x, norm, act, in_bias)
    g = int(norm.num_groups)
    cpg = c // g
    hw = h * w
    sb, sc, st, sh, sw = x5.stride()
    ib = in_bias if in_bias is not None else x5
    budget = 4096 if x5.element_size() == 4 else 8192
    block_p = max(1, min(256, budget // c))
    n_chunks = (hw + block_p - 1) // block_p
    rows = b * t
    pmean, pm2 = torch.empty((2, rows * n_chunks * g), dtype = torch.float32, device = x.device)
    pn = torch.empty(rows * n_chunks, dtype = torch.float32, device = x.device)
    k.gn_partials[(n_chunks, rows)](
        x5, ib, pn, pmean, pm2, t, hw, w, n_chunks, sb, sc, st, sh, sw,
        C = c, G = g, CPG = cpg, BLOCK_P = block_p, HAS_BIAS = in_bias is not None, num_warps = 4,
    )  # fmt: skip
    mean = torch.empty(rows * g, dtype = torch.float32, device = x.device)
    rstd = torch.empty(rows * g, dtype = torch.float32, device = x.device)
    k.gn_combine[(rows * g,)](pn, pmean, pm2, mean, rstd, n_chunks, g, float(norm.eps), BLOCK = 1024, num_warps = 4)
    out = torch.empty((b, t, h, w, c), dtype = x.dtype, device = x.device)
    block_c = min(128, _next_pow2(c))
    block_p2 = max(1, 4096 // block_c)
    grid = ((hw + block_p2 - 1) // block_p2, rows, (c + block_c - 1) // block_c)
    k.gn_apply[grid](
        x5, out, mean, rstd, norm.weight, norm.bias, ib, c, t, hw, w, g, sb, sc, st, sh, sw,
        CPG = cpg, ACT = bool(act), HAS_IN_BIAS = in_bias is not None, BLOCK_P = block_p2, BLOCK_C = block_c,
        num_warps = 4,
    )  # fmt: skip
    out = out.permute(0, 4, 1, 2, 3)  # B, C, T', H', W'
    shape = x.shape
    if ndim == 5:
        return out.reshape(b, c, *shape[2:]) if out.shape != shape else out
    if ndim == 4:
        return out[:, :, 0]
    return out[:, :, 0].reshape(shape)  # (B, C, L) view over the (B, L, C) buffer


# ----------------------------------------------------------------------------------------------------------------
# Channel RMS norm (Wan / Qwen-Image / HunyuanVideo-1.5 ``*RMS_norm`` with channel_first=True)


def rms_norm_reference(x: Any, norm: Any, act: bool, in_bias: Any = None) -> Any:
    import torch.nn.functional as F

    if in_bias is not None:
        x = x + in_bias.view(1, -1, *([1] * (x.dim() - 2))).to(x.dtype)
    y = x if norm is None else norm(x)
    return F.silu(y) if act else y


def _is_pixel_norm(norm: Any) -> bool:
    return type(norm).__name__ == "PerChannelRMSNorm" and getattr(norm, "channel_dim", 1) == 1


def _rms_params(norm: Any) -> Optional[tuple]:
    torch = _torch()
    if _is_pixel_norm(norm):
        return None, None, False, 1.0
    gamma = norm.gamma.reshape(-1)
    bias = norm.bias
    has_bias = isinstance(bias, torch.Tensor)
    if not has_bias and float(bias) != 0.0:
        return None
    return gamma, (bias.reshape(-1) if has_bias else gamma), has_bias, float(norm.scale)


def _rms_fusable(x: Any, norm: Any, cache: Any) -> bool:
    torch = _torch()
    c = x.shape[1]
    return (
        x.is_cuda
        and x.dim() == 5
        and c <= _MAX_C
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() > 0
        and (
            cache is None
            or (
                torch.is_tensor(cache)
                and cache.dim() == 5
                and cache.shape[:2] == x.shape[:2]
                and cache.shape[3:] == x.shape[3:]
                and cache.device == x.device
            )
        )
        and (
            norm is None
            or _is_pixel_norm(norm)
            or (getattr(norm, "channel_first", False) and _rms_params(norm) is not None)
        )
    )


def rms_norm_act(
    x: Any,
    norm: Any,
    act: bool = True,
    *,
    front: int = 0,
    cache: Any = None,
    in_bias: Any = None,
    replicate_pad: Optional[tuple] = None,
    back: int = 0,
) -> Any:
    """``cat([cache or zeros] -> front frames, silu?(rms_norm(x + in_bias)))`` along T, as ONE channels-last_3d tensor.

    ``norm=None`` skips the norm (a plain causal ``cat``). ``cache`` holds already-activated frames (the stock causal
    cache); its last ``min(front, T_cache)`` frames are used and zeros fill any remaining front slots: exactly the
    stock ``torch.cat`` + causal ``F.pad`` input of a conv.

    ``replicate_pad=(ph, pw)``: HunyuanVideo-1.5's ``F.pad(mode="replicate")`` instead: ``front`` copies of the first
    frame and ``ph`` / ``pw`` replicated rows / columns each side (``cache`` must be None); ``back`` copies of the
    last frame (LTX-2's non-causal decoder)."""
    torch = _torch()
    if replicate_pad is not None and cache is not None:
        raise ValueError("replicate padding takes no cache")
    if back and replicate_pad is None:
        raise ValueError("back frames need replicate padding")
    if not _rms_fusable(x, norm, cache):
        y = rms_norm_reference(x, norm, act, in_bias)
        if replicate_pad is not None:
            import torch.nn.functional as F
            ph, pw = replicate_pad
            y = F.pad(y.float(), (pw, pw, ph, ph, front, back), mode = "replicate").to(y.dtype)
            return y.contiguous(memory_format = torch.channels_last_3d)
        if front:
            import torch.nn.functional as F
            n_cache = 0
            if cache is not None:
                cache = cache[:, :, -front:].to(y.dtype)
                n_cache = cache.shape[2]
                y = torch.cat([cache, y], dim = 2)
            y = F.pad(y, (0, 0, 0, 0, front - n_cache, 0))
        return y.contiguous(memory_format = torch.channels_last_3d)
    k = _kernels()
    mean_sq = norm is not None and _is_pixel_norm(norm)
    if norm is not None and not mean_sq:
        gamma, bias, has_bias, scale = _rms_params(norm)
    else:
        gamma, bias, has_bias, scale = x, x, False, 1.0
    eps = float(getattr(norm, "eps", 0.0) or 0.0) if mean_sq else 0.0
    b, c, t, h, w = x.shape
    ph, pw = replicate_pad if replicate_pad is not None else (0, 0)
    ho, wo = h + 2 * ph, w + 2 * pw
    to = t + front + back
    n_cache = 0 if cache is None else min(front, cache.shape[2])
    cache_t = x if cache is None else cache[:, :, cache.shape[2] - n_cache :]
    out = torch.empty((b, to, ho, wo, c), dtype = x.dtype, device = x.device)
    block_c = _next_pow2(c)
    block_p = max(1, min(128, 8192 // block_c))
    grid = ((ho * wo + block_p - 1) // block_p, b * to)
    k.rms_act[grid](
        x, out, gamma, bias, in_bias if in_bias is not None else x, cache_t,
        c, t, h, w, ho, wo, ph, pw, to, front, n_cache, scale, eps,
        *x.stride(), *cache_t.stride(),
        NORM = norm is not None, ACT = bool(act), HAS_BIAS = has_bias, HAS_IN_BIAS = in_bias is not None,
        REPLICATE = replicate_pad is not None, MEAN_SQ = mean_sq, BLOCK_P = block_p, BLOCK_C = block_c,
        num_warps = 4 if block_c * block_p <= 4096 else 8,
    )  # fmt: skip
    return out.permute(0, 4, 1, 2, 3)


def _conv_kind(conv: Any) -> Optional[str]:
    torch = _torch()
    pad = getattr(conv, "_padding", None)
    if pad is None or getattr(conv, "padding_mode", "zeros") != "zeros" or conv.groups != 1:
        return None
    if isinstance(conv, torch.nn.Conv3d) and len(pad) == 6 and pad[5] == 0 and pad[0] == pad[1] and pad[2] == pad[3]:
        return "3d"
    if isinstance(conv, torch.nn.Conv2d) and len(pad) == 4 and pad[0] == pad[1] and pad[2] == pad[3]:
        return "2d"
    return None


def causal_conv(
    conv: Any,
    x: Any,
    cache: Any = None,
    *,
    norm: Any = None,
    act: bool = False,
    in_bias: Any = None,
    with_bias: bool = True,
) -> tuple:
    """Wan-lineage causal conv of ``silu?(rms?(x + in_bias))`` with the causal cache; returns (out, new_cache).

    ``new_cache`` is what the stock block stores: the activated input's last ``CACHE_T`` (2) frames, prefixed by the
    previous cache's last frame when the chunk is one frame. It is a VIEW of the conv input (no clone)."""
    import torch.nn.functional as F

    kind = _conv_kind(conv)
    bias = conv.bias if with_bias else None
    if kind == "2d":  # Qwen-Image-2.1: the one-frame specialisation, never cached
        if cache is not None:
            raise ValueError("2D causal conv takes no cache")
        y = x if (norm is None and not act and in_bias is None) else rms_norm_act(x, norm, act, in_bias = in_bias)
        pw, _, ph, _ = conv._padding
        out = F.conv2d(y[:, :, 0], conv.weight, bias, conv.stride, (ph, pw), conv.dilation)
        return out.unsqueeze(2), None
    if kind != "3d":
        raise ValueError("unsupported causal conv " + type(conv).__name__)
    pw, _, ph, _, front, _ = conv._padding
    kt = conv.kernel_size[0]
    spatial = (0, ph, pw)
    t = x.shape[2]
    if front == 0:
        y = x if (norm is None and not act and in_bias is None) else rms_norm_act(x, norm, act, in_bias = in_bias)
        return F.conv3d(y, conv.weight, bias, conv.stride, spatial, conv.dilation), None
    if cache is None and t == 1 and front == kt - 1 and conv.stride[0] == 1 and conv.dilation[0] == 1:
        # one frame after kt-1 zero frames: only the last temporal tap meets data
        y = rms_norm_act(x, norm, act, in_bias = in_bias)
        weight = conv.weight[:, :, -1:]
        return F.conv3d(y, weight, bias, conv.stride, spatial, conv.dilation), y
    p = rms_norm_act(x, norm, act, front = front, cache = cache, in_bias = in_bias)
    out = F.conv3d(p, conv.weight, bias, conv.stride, spatial, conv.dilation)
    return out, p[:, :, -2:]


# ----------------------------------------------------------------------------------------------------------------
# Conv without its bias (the bias rides into the next fused pass) and the fused bias + residual epilogue


def conv_nobias(conv: Any, x: Any) -> Any:
    """``conv(x)`` minus the bias; the caller adds ``conv.bias`` in a later fused pass."""
    import torch.nn.functional as F

    if getattr(conv, "padding_mode", "zeros") != "zeros":
        raise ValueError("conv_nobias: padding_mode " + str(conv.padding_mode))
    fn = F.conv2d if x.dim() == 4 else F.conv3d
    return fn(x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)


def _cl_contig(t: Any) -> bool:
    torch = _torch()
    fmt = torch.channels_last if t.dim() == 4 else torch.channels_last_3d
    return t.is_contiguous(memory_format = fmt)


def add_bias_residual(out: Any, out_bias: Any, res: Any, res_bias: Any = None, scale: float = 1.0) -> Any:
    """``(out + out_bias + res + res_bias) / scale`` written into ``out`` (a fresh channels-last conv output)."""
    torch = _torch()
    k = _kernels()
    if k is None or not out.is_cuda or out.dim() not in (4, 5) or not _cl_contig(out) or res.shape != out.shape:
        if out_bias is not None:
            out.add_(out_bias.view(1, -1, *([1] * (out.dim() - 2))).to(out.dtype))
        if res_bias is not None:
            res = res + res_bias.view(1, -1, *([1] * (res.dim() - 2))).to(res.dtype)
        out.add_(res)
        return out if scale == 1.0 else out.div_(scale)
    r5, _ = _as5d(res)
    o5, _ = _as5d(out)
    b, c, t, h, w = o5.shape
    pixels = b * t * h * w
    block_c = min(128, _next_pow2(c))
    block_p = max(1, 4096 // block_c)
    grid = ((pixels + block_p - 1) // block_p, (c + block_c - 1) // block_c)
    k.bias_residual[grid](
        out, out_bias if out_bias is not None else out, res, res_bias if res_bias is not None else res,
        pixels, c, t, h * w, w, *r5.stride(), 1.0 / float(scale),
        HAS_OB = out_bias is not None, HAS_RB = res_bias is not None, SCALE = float(scale) != 1.0,
        BLOCK_P = block_p, BLOCK_C = block_c, num_warps = 4,
    )  # fmt: skip
    return out


# ----------------------------------------------------------------------------------------------------------------
# Install: instance-level patches, each guarded


def _is_oom(exc: BaseException) -> bool:
    try:
        if isinstance(exc, _torch().cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001
        pass
    return "out of memory" in str(exc).lower()


def _guard(module: Any, fast: Any, stock: Any, label: str, logger: Any) -> None:
    """``module.forward = fast`` until it raises something ``stock`` does not; then stock for good."""

    def forward(*args, **kwargs):
        if getattr(module, "_unsloth_vae_fused_failed", False):
            return stock(*args, **kwargs)
        try:
            return fast(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise
            out = stock(*args, **kwargs)
            module._unsloth_vae_fused_failed = True
            if logger is not None:
                logger.warning("diffusion.vae_fused: fused %s failed, using the stock path: %s", label, exc)
            return out

    forward._unsloth_vae_fused = True
    module.forward = forward


def _stock_forward(module: Any) -> Any:
    return type(module).forward.__get__(module)


def _resnet2d_fusable(block: Any) -> bool:
    torch = _torch()
    return (
        isinstance(getattr(block, "norm1", None), torch.nn.GroupNorm)
        and isinstance(getattr(block, "norm2", None), torch.nn.GroupNorm)
        and isinstance(getattr(block, "nonlinearity", None), torch.nn.SiLU)
        and getattr(block, "upsample", None) is None
        and getattr(block, "downsample", None) is None
        and getattr(block, "time_emb_proj", None) is None
        # "group" / "spatial" name the temb norm; without temb (a VAE) both run plain norm2 like "default"
        and getattr(block, "time_embedding_norm", "default") in ("default", "group")
        and isinstance(getattr(block, "conv1", None), torch.nn.Conv2d)
        and isinstance(getattr(block, "conv2", None), torch.nn.Conv2d)
    )


def _fast_resnet2d(block: Any) -> Any:
    def fast(input_tensor: Any, temb: Any = None, *args: Any, **kwargs: Any) -> Any:
        if temb is not None or args or kwargs or block.training:
            return _stock_forward(block)(input_tensor, temb, *args, **kwargs)
        h = group_norm_act(input_tensor, block.norm1, act = True)
        h = conv_nobias(block.conv1, h)
        h = group_norm_act(h, block.norm2, act = True, in_bias = block.conv1.bias)
        h = conv_nobias(block.conv2, h)
        shortcut = block.conv_shortcut(input_tensor) if block.conv_shortcut is not None else input_tensor
        return add_bias_residual(h, block.conv2.bias, shortcut, None, float(block.output_scale_factor))

    return fast


def _fast_norm(norm: Any, act: bool) -> Any:
    def fast(x: Any) -> Any:
        return group_norm_act(x, norm, act = act)

    return fast


def install_group_norm_vae(vae: Any, logger: Any = None) -> int:
    """Patch every eligible ResnetBlock2D / GroupNorm of a 2-D (AutoencoderKL-family) VAE. Returns patch count."""
    torch = _torch()
    if vae is None or not runtime_ok():
        return 0
    n = 0
    fused_act_norms = set()
    for part_name in ("encoder", "decoder"):
        part = getattr(vae, part_name, None)
        if part is None:
            continue
        norm_out = getattr(part, "conv_norm_out", None)
        act = getattr(part, "conv_act", None)
        if isinstance(norm_out, torch.nn.GroupNorm) and isinstance(act, torch.nn.SiLU):
            fused_act_norms.add(id(norm_out))
            _guard(
                norm_out,
                _fast_norm(norm_out, True),
                lambda x, _n = norm_out: torch.nn.functional.silu(torch.nn.GroupNorm.forward(_n, x)),
                "conv_norm_out",
                logger,
            )
            act.forward = lambda x: x
            act._unsloth_vae_fused = True
            n += 1
        for module in part.modules():
            if getattr(module.forward, "_unsloth_vae_fused", False):
                continue
            if type(module).__name__ == "ResnetBlock2D" and _resnet2d_fusable(module):
                _guard(module, _fast_resnet2d(module), _stock_forward(module), "ResnetBlock2D", logger)
                n += 1
        for module in part.modules():
            if (
                isinstance(module, torch.nn.GroupNorm)
                and id(module) not in fused_act_norms
                and not getattr(module.forward, "_unsloth_vae_fused", False)
            ):
                _guard(module, _fast_norm(module, False), _stock_forward(module), "GroupNorm", logger)
                n += 1
    return n


def uninstall(vae: Any) -> None:
    vae.__dict__.pop("_unsloth_vae_fused_installed", None)
    for attr in ("tiled_decode", "blend_v", "blend_h", "blend_t"):
        if getattr(vae.__dict__.get(attr), "_unsloth_vae_fused", False):
            delattr(vae, attr)
    for module in vae.modules():
        if getattr(getattr(module, "forward", None), "_unsloth_vae_fused", False) or getattr(
            module, "_unsloth_vae_fused", False
        ):
            try:
                del module.forward
            except AttributeError:
                pass
            module.__dict__.pop("_unsloth_vae_fused", None)
            module.__dict__.pop("_unsloth_vae_fused_failed", None)
        module.__dict__.pop("prepare_causal_attention_mask", None)


# ----------------------------------------------------------------------------------------------------------------
# Wan lineage: AutoencoderKLWan (2.1 / 2.2), AutoencoderKLQwenImage, AutoencoderKLQwenImage21

_WAN_VAES = frozenset({"AutoencoderKLWan", "AutoencoderKLQwenImage", "AutoencoderKLQwenImage21"})
_CACHE_T = 2


def _is_rms(norm: Any) -> bool:
    return type(norm).__name__.endswith("RMS_norm") and hasattr(norm, "gamma") and hasattr(norm, "scale")


def _wan_resblock_fusable(block: Any) -> bool:
    torch = _torch()
    return (
        _is_rms(getattr(block, "norm1", None))
        and _is_rms(getattr(block, "norm2", None))
        and isinstance(getattr(block, "nonlinearity", None), torch.nn.SiLU)
        and _conv_kind(getattr(block, "conv1", None)) is not None
        and _conv_kind(getattr(block, "conv2", None)) is not None
    )


def _fast_wan_resblock(block: Any) -> Any:
    import torch.nn.functional as F

    def fast(x: Any, feat_cache: Any = None, feat_idx: Any = None, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs or block.training or (block.dropout.p and block.dropout.training):
            return _stock_forward(block)(x, feat_cache, feat_idx if feat_idx is not None else [0], *args, **kwargs)
        sc = block.conv_shortcut
        if isinstance(sc, _torch().nn.Identity):
            h = x
        elif _conv_kind(sc) == "2d":
            h = F.conv2d(x[:, :, 0], sc.weight, sc.bias, sc.stride, 0).unsqueeze(2)
        else:
            h = F.conv3d(x, sc.weight, sc.bias, sc.stride)
        if feat_cache is not None:
            idx = feat_idx[0]
            out, new = causal_conv(block.conv1, x, feat_cache[idx], norm = block.norm1, act = True, with_bias = False)
            feat_cache[idx] = new
            feat_idx[0] += 1
            idx = feat_idx[0]
            out, new = causal_conv(
                block.conv2, out, feat_cache[idx], norm = block.norm2, act = True, in_bias = block.conv1.bias,
                with_bias = False,
            )  # fmt: skip
            feat_cache[idx] = new
            feat_idx[0] += 1
        else:
            out, _ = causal_conv(block.conv1, x, None, norm = block.norm1, act = True, with_bias = False)
            out, _ = causal_conv(
                block.conv2, out, None, norm = block.norm2, act = True, in_bias = block.conv1.bias, with_bias = False
            )
        return add_bias_residual(out, block.conv2.bias, h)

    return fast


def _fast_causal_conv(conv: Any) -> Any:
    def fast(x: Any, cache_x: Any = None) -> Any:
        return causal_conv(conv, x, cache_x)[0]

    return fast


def _fast_rms_act(norm: Any) -> Any:
    def fast(x: Any) -> Any:
        if x.dim() != 5:
            return _torch().nn.functional.silu(type(norm).forward(norm, x))
        return rms_norm_act(x, norm, True)

    return fast


def install_wan_vae(vae: Any, logger: Any = None) -> int:
    """Patch the residual blocks, causal convs and output heads of a Wan-lineage VAE. Returns patch count."""
    torch = _torch()
    if vae is None or not runtime_ok():
        return 0
    n = 0
    for part_name in ("encoder", "decoder"):
        part = getattr(vae, part_name, None)
        if part is None:
            continue
        norm_out = getattr(part, "norm_out", None)
        act = getattr(part, "nonlinearity", None)
        if _is_rms(norm_out) and isinstance(act, torch.nn.SiLU):
            _guard(
                norm_out,
                _fast_rms_act(norm_out),
                lambda x, _n = norm_out: torch.nn.functional.silu(type(_n).forward(_n, x)),
                "norm_out",
                logger,
            )
            act.forward = lambda x: x
            act._unsloth_vae_fused = True
            n += 1
        for module in part.modules():
            if "forward" in module.__dict__:
                continue  # already patched (by us or another speed path)
            if type(module).__name__.endswith("ResidualBlock") and _wan_resblock_fusable(module):
                _guard(module, _fast_wan_resblock(module), _stock_forward(module), "residual block", logger)
                n += 1
        for module in part.modules():
            if "forward" in module.__dict__:
                continue
            if type(module).__name__.endswith("CausalConv3d") and _conv_kind(module) is not None:
                _guard(module, _fast_causal_conv(module), _stock_forward(module), "causal conv", logger)
                n += 1
            elif (
                isinstance(module, torch.nn.Upsample)
                and type(module).__name__.endswith("Upsample")
                and module.mode in ("nearest", "nearest-exact")
            ):
                _guard(module, _fast_upsample(module), _stock_forward(module), "upsample", logger)
                n += 1
    if install_wan_tile_batch(vae, logger):
        n += 1
    return n


def _fast_upsample(mod: Any) -> Any:
    """Wan's ``nearest-exact`` upsample round-trips through fp32 (``x.float()`` ... ``type_as``); nearest only selects
    pixels, so running it in the input dtype is bit-identical and skips two full-size copies."""

    def fast(x: Any) -> Any:
        import torch.nn.functional as F

        return F.interpolate(x, mod.size, mod.scale_factor, mod.mode, mod.align_corners, recompute_scale_factor = mod.recompute_scale_factor)

    return fast


# ----------------------------------------------------------------------------------------------------------------
# Tile batching + vectorised seam blending for the Wan-lineage tiled decode

TILE_BATCH_ENV = "UNSLOTH_VAE_TILE_BATCH"
_TILE_BATCH_MAX = 4
# measured decode peak per 256 px Wan-2.1 tile (fp16, 4 frames, 96 full-res channels) ~= 1 GiB = 24x one activation
_TILE_PEAK_FACTOR = 24


def _tile_batch_cap(vae: Any, z: Any) -> int:
    """Tiles per decoder call: env override, else what half the free VRAM holds at the measured per-tile peak,
    at most 4 (Wan-2.1 832x480x81: 1 -> 1.27 s, 2 -> 0.91 s, 4 -> 0.80 s, 8 -> 0.77 s at +1 GiB per tile)."""
    raw = os.environ.get(TILE_BATCH_ENV, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    try:
        free, _ = _torch().cuda.mem_get_info(z.device)
        norm_out = getattr(getattr(vae, "decoder", None), "norm_out", None)
        c_last = int(norm_out.gamma.numel()) if norm_out is not None else 128
        px = int(vae.tile_sample_min_height) * int(vae.tile_sample_min_width)
        frames = int(getattr(vae, "temporal_compression_ratio", 4) or 4)
        elem = next(vae.decoder.parameters()).element_size()
        per_tile = _TILE_PEAK_FACTOR * px * frames * c_last * elem
        return int(max(1, min(_TILE_BATCH_MAX, (free // 2) // max(1, per_tile))))
    except Exception:  # noqa: BLE001
        return 1


def _blend_weights(extent: int, device: Any) -> tuple:
    torch = _torch()
    # the stock python scalars y / extent and 1 - y / extent (double), as the fp32 opmath scalars PyTorch uses
    wb = torch.tensor([y / extent for y in range(extent)], dtype = torch.float64).float().to(device)
    wa = torch.tensor([1 - y / extent for y in range(extent)], dtype = torch.float64).float().to(device)
    return wa, wb


def blend_seam(a: Any, b: Any, extent: int, dim: int) -> Any:
    """Vectorised, bit-identical ``blend_v`` (dim=-2) / ``blend_h`` (dim=-1): one pass instead of ``extent`` x 4."""
    torch = _torch()
    extent = min(a.shape[dim], b.shape[dim], extent)
    if extent <= 0:
        return b
    wa, wb = _blend_weights(extent, b.device)
    shape = [1] * b.dim()
    shape[dim] = extent
    wa, wb = wa.view(shape), wb.view(shape)
    sa = a.narrow(dim, a.shape[dim] - extent, extent)
    sb = b.narrow(dim, 0, extent)
    dt = b.dtype
    pa = (sa.float() * wa).to(dt)
    pb = (sb.float() * wb).to(dt)
    sb.copy_((pa.float() + pb.float()).to(dt))
    return b


def _accepts_first_chunk(decoder: Any) -> bool:
    import inspect

    try:
        return "first_chunk" in inspect.signature(decoder.forward).parameters
    except (TypeError, ValueError):
        return False


def _wan_batched_tiled_decode(self: Any, z: Any, return_dict: bool = True) -> Any:
    """``tiled_decode`` of a Wan-lineage VAE with same-shaped tiles decoded as one batch (fewer, larger launches:
    the stock loop issues ~50k kernels for an 81-frame 832x480 decode) and vectorised seam blending."""
    torch = _torch()
    from diffusers.models.autoencoders.vae import DecoderOutput

    _, _, num_frames, height, width = z.shape
    ratio = self.spatial_compression_ratio
    sample_height, sample_width = height * ratio, width * ratio
    tmh, tmw = self.tile_sample_min_height // ratio, self.tile_sample_min_width // ratio
    tsh, tsw = self.tile_sample_stride_height // ratio, self.tile_sample_stride_width // ratio
    out_sh, out_sw = self.tile_sample_stride_height, self.tile_sample_stride_width
    patch = getattr(self.config, "patch_size", None)
    if patch is not None:
        sample_height, sample_width = sample_height // patch, sample_width // patch
        out_sh, out_sw = out_sh // patch, out_sw // patch
        blend_h = self.tile_sample_min_height // patch - out_sh
        blend_w = self.tile_sample_min_width // patch - out_sw
    else:
        blend_h = self.tile_sample_min_height - out_sh
        blend_w = self.tile_sample_min_width - out_sw
    first_chunk = _accepts_first_chunk(self.decoder)
    rows_i = list(range(0, height, tsh))
    cols_j = list(range(0, width, tsw))
    groups: dict = {}
    for i in rows_i:
        for j in cols_j:
            groups.setdefault((min(tmh, height - i), min(tmw, width - j)), []).append((i, j))
    cap = _tile_batch_cap(self, z)
    tiles: dict = {}

    def decode_tiles(chunk: list) -> Any:
        zt = torch.cat([z[:, :, :, i : i + tmh, j : j + tmw] for i, j in chunk], 0)
        self.clear_cache()
        frames = []
        for k in range(num_frames):
            self._conv_idx = [0]
            tile = self.post_quant_conv(zt[:, :, k : k + 1])
            kw = {"first_chunk": k == 0} if first_chunk else {}
            frames.append(self.decoder(tile, feat_cache = self._feat_map, feat_idx = self._conv_idx, **kw))
        return torch.cat(frames, dim = 2)

    for members in groups.values():
        s = 0
        while s < len(members):
            chunk = members[s : s + cap]
            try:
                out = decode_tiles(chunk)
            except Exception as exc:  # noqa: BLE001
                # batching raised the peak: an OOM with several tiles retries them one at a time
                if len(chunk) == 1 or not _is_oom(exc):
                    raise
                self.clear_cache()
                torch.cuda.empty_cache()
                cap = 1
                continue
            for n, key in enumerate(chunk):
                tiles[key] = out[n : n + 1]
            s += len(chunk)
    self.clear_cache()
    rows = [[tiles[(i, j)] for j in cols_j] for i in rows_i]
    result_rows = []
    for ri, row in enumerate(rows):
        result_row = []
        for ci, tile in enumerate(row):
            if ri > 0:
                tile = blend_seam(rows[ri - 1][ci], tile, blend_h, -2)
            if ci > 0:
                tile = blend_seam(row[ci - 1], tile, blend_w, -1)
            result_row.append(tile[:, :, :, :out_sh, :out_sw])
        result_rows.append(torch.cat(result_row, dim = -1))
    dec = torch.cat(result_rows, dim = 3)[:, :, :, :sample_height, :sample_width]
    if patch is not None:
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

        dec = unpatchify(dec, patch_size = patch)
    dec = torch.clamp(dec, min = -1.0, max = 1.0)
    if not return_dict:
        return (dec,)
    return DecoderOutput(sample = dec)


def install_wan_tile_batch(vae: Any, logger: Any = None) -> bool:
    if vae is None or "tiled_decode" in vae.__dict__ or not callable(getattr(vae, "tiled_decode", None)):
        return False
    stock = _stock_forward_attr(vae, "tiled_decode")

    def tiled_decode(z: Any, return_dict: bool = True) -> Any:
        if z.shape[0] != 1 or getattr(vae, "_unsloth_vae_fused_failed", False):
            return stock(z, return_dict = return_dict)
        try:
            return _wan_batched_tiled_decode(vae, z, return_dict)
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise
            vae._unsloth_vae_fused_failed = True
            if logger is not None:
                logger.warning("diffusion.vae_fused: batched tiled decode failed, using the stock loop: %s", exc)
            return stock(z, return_dict = return_dict)

    tiled_decode._unsloth_vae_fused = True
    vae.tiled_decode = tiled_decode
    return True


def _stock_forward_attr(obj: Any, name: str) -> Any:
    return getattr(type(obj), name).__get__(obj)


# ----------------------------------------------------------------------------------------------------------------
# HunyuanVideo-1.5: replicate-padded causal convs, no feature cache (a tile decodes its whole clip at once)


def _hv_conv_ok(conv: Any) -> bool:
    torch = _torch()
    inner = getattr(conv, "conv", None)
    pad = getattr(conv, "time_causal_padding", None)
    return (
        isinstance(inner, torch.nn.Conv3d)
        and getattr(conv, "pad_mode", None) == "replicate"
        and pad is not None
        and len(pad) == 6
        and pad[0] == pad[1]
        and pad[2] == pad[3]
        and pad[5] == 0
        and inner.padding == (0, 0, 0)
        and inner.groups == 1
        and inner.padding_mode == "zeros"
    )


def hv_conv(conv: Any, x: Any, *, norm: Any = None, act: bool = False, in_bias: Any = None, with_bias: bool = True) -> Any:
    """``conv(pad_replicate(silu?(rms?(x + in_bias))))`` with the pad, norm and act in one channels-last pass."""
    import torch.nn.functional as F

    pw, _, ph, _, front, _ = conv.time_causal_padding
    inner = conv.conv
    y = rms_norm_act(x, norm, act, front = front, in_bias = in_bias, replicate_pad = (ph, pw))
    return F.conv3d(y, inner.weight, inner.bias if with_bias else None, inner.stride, 0, inner.dilation)


def _hv_resnet_ok(block: Any) -> bool:
    torch = _torch()
    return (
        _is_rms(getattr(block, "norm1", None))
        and _is_rms(getattr(block, "norm2", None))
        and isinstance(getattr(block, "nonlinearity", None), torch.nn.SiLU)
        and _hv_conv_ok(getattr(block, "conv1", None))
        and _hv_conv_ok(getattr(block, "conv2", None))
    )


def _fast_hv_resnet(block: Any) -> Any:
    import torch.nn.functional as F

    def fast(x: Any) -> Any:
        h = hv_conv(block.conv1, x, norm = block.norm1, act = True, with_bias = False)
        h = hv_conv(block.conv2, h, norm = block.norm2, act = True, in_bias = block.conv1.conv.bias, with_bias = False)
        sc = block.conv_shortcut
        res = x if sc is None else F.conv3d(x, sc.weight, sc.bias, sc.stride, sc.padding)
        return add_bias_residual(h, block.conv2.conv.bias, res)

    return fast


def _fast_hv_conv(conv: Any) -> Any:
    def fast(x: Any) -> Any:
        return hv_conv(conv, x)

    return fast


def _fast_hv_norm_act(norm: Any) -> Any:
    def fast(x: Any) -> Any:
        if x.dim() != 5:
            return _torch().nn.functional.silu(type(norm).forward(norm, x))
        return rms_norm_act(x, norm, True)

    return fast


def hv_upsample(up: Any, x: Any) -> Any:
    """``HunyuanVideo15Upsample.forward``: conv, then depth-to-space + repeat_interleave shortcut + add in one pass."""
    torch = _torch()
    k = _kernels()
    conv = up.conv
    h = hv_conv(conv, x, with_bias = False)
    b, cin, f, hh, ww = x.shape
    cp = h.shape[1]
    temporal = bool(up.add_temporal_upsample)
    factor = 8 if temporal else 4
    c = cp // factor
    rep = int(up.repeats)
    if cp % factor or (temporal and (f < 1 or rep % 2 or cin % 4)) or rep * cin != factor * c:
        raise ValueError("hv_upsample: unexpected shapes")
    fo = 1 + (f - 1) * 2 if temporal else f
    out = torch.empty((b, fo, 2 * hh, 2 * ww, c), dtype = x.dtype, device = x.device)
    block_c = min(128, _next_pow2(c))
    block_p = max(1, 4096 // block_c)
    grid = ((4 * hh * ww + block_p - 1) // block_p, b * fo, (c + block_c - 1) // block_c)
    bias = conv.conv.bias
    k.dcae_up_add[grid](
        h, bias if bias is not None else h, x, out, c, cin, f, hh, ww, fo, rep, *h.stride(), *x.stride(),
        TEMPORAL = temporal, HAS_HB = bias is not None, BLOCK_P = block_p, BLOCK_C = block_c, num_warps = 4,
    )  # fmt: skip
    return out.permute(0, 4, 1, 2, 3)


def _hv_causal_mask(n_frame: int, n_hw: int, dtype: Any, device: Any, batch_size: Any = None) -> Any:
    """Vectorised ``prepare_causal_attention_mask`` (the stock one launches one fill kernel per token row: 80k fills
    per 848x480x121 decode). Same values: 0 where key frame <= query frame, else -inf."""
    torch = _torch()
    frame = torch.arange(n_frame * n_hw, device = device) // n_hw
    mask = torch.where(frame[None, :] <= frame[:, None], torch.zeros((), dtype = dtype, device = device),
                       torch.full((), float("-inf"), dtype = dtype, device = device))  # fmt: skip
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


def install_hv15_vae(vae: Any, logger: Any = None) -> int:
    torch = _torch()
    if vae is None or not runtime_ok():
        return 0
    n = 0
    for part_name in ("encoder", "decoder"):
        part = getattr(vae, part_name, None)
        if part is None:
            continue
        norm_out, act = getattr(part, "norm_out", None), getattr(part, "conv_act", None)
        if _is_rms(norm_out) and isinstance(act, torch.nn.SiLU) and "forward" not in norm_out.__dict__:
            _guard(
                norm_out,
                _fast_hv_norm_act(norm_out),
                lambda x, _n = norm_out: torch.nn.functional.silu(type(_n).forward(_n, x)),
                "norm_out",
                logger,
            )
            act.forward = lambda x: x
            act._unsloth_vae_fused = True
            n += 1
        for module in part.modules():
            if "forward" in module.__dict__:
                continue
            name = type(module).__name__
            if name.endswith("ResnetBlock") and _hv_resnet_ok(module):
                _guard(module, _fast_hv_resnet(module), _stock_forward(module), "resnet", logger)
                n += 1
        for module in part.modules():
            if "forward" in module.__dict__:
                continue
            name = type(module).__name__
            if name.endswith("Upsample") and _hv_conv_ok(getattr(module, "conv", None)) and hasattr(module, "repeats"):
                _guard(module, lambda x, _m = module: hv_upsample(_m, x), _stock_forward(module), "upsample", logger)
                n += 1
            elif name.endswith("AttnBlock") and callable(getattr(module, "prepare_causal_attention_mask", None)):
                module.prepare_causal_attention_mask = _hv_causal_mask
                n += 1
        for module in part.modules():
            if "forward" in module.__dict__:
                continue
            if type(module).__name__.endswith("CausalConv3d") and _hv_conv_ok(module):
                _guard(module, _fast_hv_conv(module), _stock_forward(module), "causal conv", logger)
                n += 1
    return n


# ----------------------------------------------------------------------------------------------------------------
# LTX-2 / LTX-2.3: PerChannelRMSNorm, first/last-frame replicate temporal pad, zero spatial pad (left to cuDNN)


def _ltx_conv_ok(conv: Any) -> bool:
    torch = _torch()
    inner = getattr(conv, "conv", None)
    return (
        isinstance(inner, torch.nn.Conv3d)
        and inner.padding_mode == "zeros"
        and inner.padding[0] == 0
        and inner.stride[0] == 1
        and inner.dilation[0] == 1
        and len(getattr(conv, "kernel_size", ())) == 3
        and conv.kernel_size[0] % 2 == 1
    )


def ltx_conv(
    conv: Any, x: Any, causal: bool, *, norm: Any = None, act: bool = False, in_bias: Any = None, with_bias: bool = True
) -> Any:
    """``LTX2VideoCausalConv3d.forward`` over ``silu?(pixel_norm?(x + in_bias))``, pad + norm + act in one pass."""
    import torch.nn.functional as F

    kt = conv.kernel_size[0]
    front, back = (kt - 1, 0) if causal else ((kt - 1) // 2, (kt - 1) // 2)
    inner = conv.conv
    if kt == 1 and norm is None and not act and in_bias is None:
        y = x
    else:
        y = rms_norm_act(x, norm, act, front = front, back = back, in_bias = in_bias, replicate_pad = (0, 0))
    return F.conv3d(y, inner.weight, inner.bias if with_bias else None, inner.stride, inner.padding, inner.dilation)


def _ltx_resnet_ok(block: Any) -> bool:
    torch = _torch()
    return (
        _is_pixel_norm(getattr(block, "norm1", None))
        and _is_pixel_norm(getattr(block, "norm2", None))
        and isinstance(getattr(block, "nonlinearity", None), torch.nn.SiLU)
        and _ltx_conv_ok(getattr(block, "conv1", None))
        and _ltx_conv_ok(getattr(block, "conv2", None))
        and getattr(block, "per_channel_scale1", None) is None
        and getattr(block, "per_channel_scale2", None) is None
        and getattr(block, "scale_shift_table", None) is None
    )


def _fast_ltx_resnet(block: Any) -> Any:
    import torch.nn.functional as F

    def fast(inputs: Any, temb: Any = None, generator: Any = None, causal: bool = True) -> Any:
        if block.training and block.dropout.p:
            return _stock_forward(block)(inputs, temb, generator, causal = causal)
        h = ltx_conv(block.conv1, inputs, causal, norm = block.norm1, act = True, with_bias = False)
        h = ltx_conv(block.conv2, h, causal, norm = block.norm2, act = True, in_bias = block.conv1.conv.bias, with_bias = False)
        res = inputs
        if block.norm3 is not None:
            res = block.norm3(res.movedim(1, -1)).movedim(-1, 1)
        if block.conv_shortcut is not None:
            sc = block.conv_shortcut
            res = F.conv3d(res, sc.weight, sc.bias, sc.stride, sc.padding)
        return add_bias_residual(h, block.conv2.conv.bias, res)

    return fast


def _fast_ltx_conv(conv: Any) -> Any:
    def fast(hidden_states: Any, causal: bool = True) -> Any:
        return ltx_conv(conv, hidden_states, causal)

    return fast


def _fast_pixel_norm_act(norm: Any) -> Any:
    def fast(x: Any, channel_dim: Any = None) -> Any:
        if x.dim() != 5 or channel_dim not in (None, 1):
            return _torch().nn.functional.silu(type(norm).forward(norm, x, channel_dim))
        return rms_norm_act(x, norm, True)

    return fast


def install_ltx2_vae(vae: Any, logger: Any = None) -> int:
    torch = _torch()
    if vae is None or not runtime_ok():
        return 0
    n = 0
    # decoder only: the causal one-frame encode (i2v conditioning) measured 33 -> 35 ms fused, so it stays stock
    for part_name in ("decoder",):
        part = getattr(vae, part_name, None)
        if part is None:
            continue
        norm_out, act = getattr(part, "norm_out", None), getattr(part, "conv_act", None)
        if (
            _is_pixel_norm(norm_out)
            and isinstance(act, torch.nn.SiLU)
            and getattr(part, "time_embedder", None) is None
            and "forward" not in norm_out.__dict__
        ):
            _guard(
                norm_out,
                _fast_pixel_norm_act(norm_out),
                lambda x, channel_dim = None, _n = norm_out: torch.nn.functional.silu(type(_n).forward(_n, x, channel_dim)),
                "norm_out",
                logger,
            )
            act.forward = lambda x: x
            act._unsloth_vae_fused = True
            n += 1
        for module in part.modules():
            if "forward" in module.__dict__:
                continue
            if type(module).__name__.endswith("ResnetBlock3d") and _ltx_resnet_ok(module):
                _guard(module, _fast_ltx_resnet(module), _stock_forward(module), "resnet", logger)
                n += 1
        for module in part.modules():
            if "forward" in module.__dict__:
                continue
            if type(module).__name__ == "LTX2VideoCausalConv3d" and _ltx_conv_ok(module):
                _guard(module, _fast_ltx_conv(module), _stock_forward(module), "causal conv", logger)
                n += 1
    return n


def install_vectorised_blend(vae: Any) -> int:
    """Replace a VAE's per-row python ``blend_v`` / ``blend_h`` / ``blend_t`` loops (extent x 4 launches each) with
    the bit-identical one-pass :func:`blend_seam`."""
    n = 0
    for name, dim in (("blend_v", -2), ("blend_h", -1), ("blend_t", -3)):
        if callable(getattr(vae, name, None)) and name not in vae.__dict__:
            fn = lambda a, b, blend_extent, _d = dim: blend_seam(a, b, blend_extent, _d)  # noqa: E731
            fn._unsloth_vae_fused = True
            setattr(vae, name, fn)
            n += 1
    return n


_CL_WEIGHT_VAES = frozenset({"AutoencoderKL", "AutoencoderKLFlux2", "AutoencoderKLHunyuanVideo15", "AutoencoderKLLTX2Video"})


def channels_last_weights(vae: Any) -> int:
    """The fused passes hand cuDNN channels-last activations; contiguous weights would be re-laid-out on every call."""
    torch = _torch()
    n = 0
    for part in (getattr(vae, "encoder", None), getattr(vae, "decoder", None)):
        if part is None:
            continue
        for m in part.modules():
            w = getattr(m, "weight", None)
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)) and isinstance(w, torch.Tensor) and w.dim() in (4, 5):
                fmt = torch.channels_last if w.dim() == 4 else torch.channels_last_3d
                if not w.is_contiguous(memory_format = fmt):
                    w.data = w.data.contiguous(memory_format = fmt)
                    n += 1
    return n


SUPPORTED_VAES = frozenset(
    {
        "AutoencoderKL",
        "AutoencoderKLFlux2",
        "AutoencoderKLWan",
        "AutoencoderKLQwenImage",
        "AutoencoderKLQwenImage21",
        "AutoencoderKLHunyuanVideo15",
        "AutoencoderKLLTX2Video",
    }
)


def will_install(vae: Any) -> bool:
    """Whether :func:`install` would engage on ``vae`` here (class covered, NVIDIA CUDA, usable Triton, not disabled)."""
    return vae is not None and type(vae).__name__ in SUPPORTED_VAES and runtime_ok()


def install(vae: Any, logger: Any = None, level: str = "fused") -> int:
    """Install every fused path that applies to ``vae``'s class. Returns the number of patched modules."""
    if vae is None or not will_install(vae):
        return 0
    done = getattr(vae, "_unsloth_vae_fused_installed", 0)
    if done:
        return done  # idempotent: a dual-DiT family runs the speed layer once per expert over the same VAE
    n = _install(vae, logger)
    # measured: HV-1.5 -3%, LTX-2.3 -4%, AutoencoderKL layout-neutral (Studio already sets it); the Wan lineage keeps
    # its own (Qwen-Image's one-frame conv3d is 15% SLOWER on channels-last weights; Wan's fp16 decode sets them)
    if n and type(vae).__name__ in _CL_WEIGHT_VAES and os.environ.get("UNSLOTH_VAE_FUSED_CL_WEIGHTS", "1") != "0":
        channels_last_weights(vae)
    if n:
        vae._unsloth_vae_fused_installed = n
        if logger is not None:
            logger.info("diffusion.vae_fused: fused %d VAE modules on %s", n, type(vae).__name__)
    return n


def _install(vae: Any, logger: Any = None) -> int:
    name = type(vae).__name__
    if name in ("AutoencoderKL", "AutoencoderKLFlux2"):
        return install_group_norm_vae(vae, logger)
    if name in _WAN_VAES:
        return install_wan_vae(vae, logger)
    if name == "AutoencoderKLHunyuanVideo15":
        n = install_hv15_vae(vae, logger)
        return n + (install_vectorised_blend(vae) if n else 0)
    if name == "AutoencoderKLLTX2Video":
        n = install_ltx2_vae(vae, logger)
        return n + (install_vectorised_blend(vae) if n else 0)
    return 0
