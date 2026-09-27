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
        x_ptr, out_ptr, w_ptr, b_ptr, cache_ptr,
        C, T, HW, W, To, front, n_cache, scale,
        sb, sc, st, sh, sw, cb, cc, ct, ch, cw,
        ACT: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
    ):  # fmt: skip
        pid_p = tl.program_id(0)
        bt = tl.program_id(1)
        t_o = bt % To
        b = bt // To
        p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        pmask = p < HW
        hh = p // W
        ww = p - hh * W
        c = tl.arange(0, BLOCK_C)
        cmask = c < C
        m = pmask[:, None] & cmask[None, :]
        out_off = ((b.to(tl.int64) * To + t_o) * HW + p.to(tl.int64))[:, None] * C + c[None, :]
        if t_o < front:
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
            t = t_o - front
            off = (
                b.to(tl.int64) * sb
                + t.to(tl.int64) * st
                + hh.to(tl.int64)[:, None] * sh
                + ww.to(tl.int64)[:, None] * sw
                + c.to(tl.int64)[None, :] * sc
            )
            v = tl.load(x_ptr + off, mask = m, other = 0.0).to(tl.float32)
            ss = tl.sum(v * v, axis = 1)
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

    return types.SimpleNamespace(
        gn_partials = _gn_partials, gn_combine = _gn_combine, gn_apply = _gn_apply, rms_act = _rms_act,
        bias_residual = _bias_residual,
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


def rms_norm_reference(x: Any, norm: Any, act: bool) -> Any:
    import torch.nn.functional as F

    y = norm(x)
    return F.silu(y) if act else y


def _rms_params(norm: Any) -> tuple:
    torch = _torch()
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
        and getattr(norm, "channel_first", False)
        and c <= _MAX_C
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() > 0
        and (cache is None or (cache.dim() == 5 and cache.shape[:2] == x.shape[:2] and cache.shape[3:] == x.shape[3:]))
        and _rms_params(norm) is not None
    )


def rms_norm_act(x: Any, norm: Any, act: bool = True, *, front: int = 0, cache: Any = None) -> Any:
    """``cat([cache or zeros] -> front frames, silu?(rms_norm(x)))`` along T, as ONE channels-last_3d tensor.

    ``cache`` holds already-activated frames (the stock causal cache); its last ``min(front, T_cache)`` frames are
    used, zeros fill any remaining front slots, exactly the stock ``cat`` + causal ``F.pad`` input of a conv."""
    torch = _torch()
    if not _rms_fusable(x, norm, cache):
        y = rms_norm_reference(x, norm, act)
        if front:
            import torch.nn.functional as F
            if cache is not None:
                cache = cache[:, :, -front:].to(y.dtype)
                y = torch.cat([cache, y], dim = 2)
            y = F.pad(y, (0, 0, 0, 0, front - (cache.shape[2] if cache is not None else 0), 0))
        return y
    k = _kernels()
    gamma, bias, has_bias, scale = _rms_params(norm)
    b, c, t, h, w = x.shape
    to = t + front
    hw = h * w
    n_cache = 0 if cache is None else min(front, cache.shape[2])
    cache_t = x if cache is None else cache[:, :, cache.shape[2] - n_cache :]
    out = torch.empty((b, to, h, w, c), dtype = x.dtype, device = x.device)
    block_c = _next_pow2(c)
    block_p = max(1, min(128, 8192 // block_c))
    grid = ((hw + block_p - 1) // block_p, b * to)
    k.rms_act[grid](
        x, out, gamma, bias, cache_t, c, t, hw, w, to, front, n_cache, scale,
        *x.stride(), *cache_t.stride(),
        ACT = bool(act), HAS_BIAS = has_bias, BLOCK_P = block_p, BLOCK_C = block_c,
        num_warps = 4 if block_c * block_p <= 4096 else 8,
    )  # fmt: skip
    return out.permute(0, 4, 1, 2, 3)


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


def install(vae: Any, logger: Any = None, level: str = "fused") -> int:
    """Install every fused path that applies to ``vae``'s class. Returns the number of patched modules."""
    name = type(vae).__name__
    if name in ("AutoencoderKL", "AutoencoderKLFlux2"):
        return install_group_norm_vae(vae, logger)
    return 0
