# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 fused q/k RMSNorm + partial RoPE, one read and one write per row (``UNSLOTH_H3_QK_ROPE=0`` = off).

Rounds where the eager bf16 chain rounds; only the order of the sum of squares is free, so rstd can differ in the
last fp32 bit (bounded at one bf16 step by the tests). Opaque custom op with a fake kernel; Triton on NVIDIA CUDA
only, the stock math everywhere else.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

QK_ROPE_ENV = "UNSLOTH_H3_QK_ROPE"
_OP = "unsloth_h3::qk_norm_rope"


def qk_rope_enabled() -> bool:
    return (os.environ.get(QK_ROPE_ENV) or "1").strip().lower() not in ("0", "off", "false", "no")


def reference_qk_norm_rope(x: Any, weight: Any, cos: Any, sin: Any, eps: float) -> Any:
    """The stock math (RMSNorm then diffusers' ``_apply_rotary_emb``), contiguous."""
    import torch
    import torch.nn.functional as F

    x = F.rms_norm(x, (x.shape[-1],), weight, eps)
    rotary_dim = cos.shape[-1]
    xr = x[..., :rotary_dim]
    xp = x[..., rotary_dim:]
    cos = cos.to(x.dtype)[None, :, None, :]
    sin = sin.to(x.dtype)[None, :, None, :]
    x1, x2 = xr.chunk(2, dim = -1)
    rot = torch.cat((-x2, x1), dim = -1)
    xr = xr * cos + rot * sin
    return torch.cat((xr, xp), dim = -1).contiguous()


@lru_cache(maxsize = None)
def _kernel() -> Any:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice
    except Exception:  # noqa: BLE001 -- older Triton layout
        from triton.language.extra.cuda import libdevice

    # rsqrt as torch's norm (not 1/sqrt); *_rn because Triton otherwise FMA-contracts away eager's bf16 roundings.
    _rsqrt = libdevice.rsqrt
    _mul = libdevice.mul_rn
    _add = libdevice.add_rn

    @triton.jit
    def _qk_norm_rope_kernel(
        x_ptr,
        w_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        n_rows,
        n_heads,
        stride_s,
        stride_h,
        cos_stride,
        eps,
        D: tl.constexpr,
        HALF: tl.constexpr,
        ROWS: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        rows = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
        row_ok = rows < n_rows
        s = rows // n_heads
        h = rows % n_heads
        cols = tl.arange(0, D)
        partner = tl.where(cols < HALF, cols + HALF, tl.where(cols < 2 * HALF, cols - HALF, cols))
        base = s * stride_s + h * stride_h
        mask = row_ok[:, None]
        x = tl.load(x_ptr + base[:, None] + cols[None, :], mask = mask, other = 0.0).to(tl.float32)
        xpart = tl.load(x_ptr + base[:, None] + partner[None, :], mask = mask, other = 0.0).to(
            tl.float32
        )
        w = tl.load(w_ptr + cols).to(tl.float32)
        wpart = tl.load(w_ptr + partner).to(tl.float32)
        ms = tl.sum(x * x, axis = 1) / D
        rstd = _rsqrt(ms + eps)
        xn = _mul(_mul(x, rstd[:, None]), w[None, :]).to(out_ptr.dtype.element_ty)
        pn = _mul(_mul(xpart, rstd[:, None]), wpart[None, :]).to(out_ptr.dtype.element_ty)
        rot_cols = cols < 2 * HALF
        cmask = mask & rot_cols[None, :]
        c = tl.load(cos_ptr + s[:, None] * cos_stride + cols[None, :], mask = cmask, other = 0.0)
        sn = tl.load(sin_ptr + s[:, None] * cos_stride + cols[None, :], mask = cmask, other = 0.0)
        c = c.to(out_ptr.dtype.element_ty).to(tl.float32)
        sn = sn.to(out_ptr.dtype.element_ty).to(tl.float32)
        sign = tl.where(cols < HALF, -1.0, 1.0)
        a = _mul(xn.to(tl.float32), c).to(out_ptr.dtype.element_ty).to(tl.float32)
        b = _mul(pn.to(tl.float32) * sign[None, :], sn).to(out_ptr.dtype.element_ty).to(tl.float32)
        roped = _add(a, b).to(out_ptr.dtype.element_ty)
        y = tl.where(rot_cols[None, :], roped, xn)
        tl.store(out_ptr + rows[:, None] * D + cols[None, :], y, mask = mask)

    return _qk_norm_rope_kernel


def _triton_ok(x: Any) -> bool:
    try:
        import torch
        if not (x.is_cuda and torch.version.hip is None):
            return False
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return (
        x.dim() == 4
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.stride(-1) == 1
        and x.shape[0] == 1
        and x.shape[-1] in (64, 128, 256)
    )


_ROWS = 16
_WARPS = 4


def _launch(
    x: Any,
    weight: Any,
    cos: Any,
    sin: Any,
    eps: float,
    rows_per_program: int = 0,
    warps: int = 0,
) -> Any:
    import torch

    _, seq, heads, dim = x.shape
    half = cos.shape[-1] // 2
    out = torch.empty((1, seq, heads, dim), device = x.device, dtype = x.dtype)
    cos = cos.contiguous()
    sin = sin.contiguous()
    rows = seq * heads
    block = int(rows_per_program or _ROWS)
    grid = (triton_cdiv(rows, block),)
    with torch.cuda.device(x.device):
        _kernel()[grid](
            x,
            weight.contiguous(),
            cos,
            sin,
            out,
            rows,
            heads,
            x.stride(1),
            x.stride(2),
            cos.stride(0),
            float(eps),
            D = dim,
            HALF = half,
            ROWS = block,
            num_warps = int(warps or _WARPS),
        )
    return out


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _supported_layout(x: Any, weight: Any, cos: Any) -> bool:
    return (
        cos.dim() == 2
        and cos.shape[0] == x.shape[1]
        and cos.shape[-1] % 2 == 0
        and cos.shape[-1] <= x.shape[-1]
        and weight is not None
        and weight.shape == (x.shape[-1],)
    )


def qk_norm_rope_impl(x: Any, weight: Any, cos: Any, sin: Any, eps: float) -> Any:
    """Triton where it applies, the stock math elsewhere."""
    if _triton_ok(x) and _supported_layout(x, weight, cos):
        try:
            return _launch(x, weight, cos, sin, eps)
        except Exception:  # noqa: BLE001
            pass
    return reference_qk_norm_rope(x, weight, cos, sin, eps)


@lru_cache(maxsize = None)
def qk_norm_rope_op() -> Any:
    """Registered once per process."""
    import torch

    # explicit schema: the annotations here are strings
    @torch.library.custom_op(
        _OP,
        mutates_args = (),
        schema = "(Tensor x, Tensor weight, Tensor cos, Tensor sin, float eps) -> Tensor",
    )
    def qk_norm_rope(x, weight, cos, sin, eps):
        return qk_norm_rope_impl(x, weight, cos, sin, eps)

    @qk_norm_rope.register_fake
    def _(x, weight, cos, sin, eps):
        return x.new_empty(x.shape)

    return qk_norm_rope


def qk_norm_rope(x: Any, weight: Any, cos: Any, sin: Any, eps: float) -> Any:
    """Eager only: compiled code calls ``torch.ops.unsloth_h3.qk_norm_rope`` (dynamo graph-breaks in here)."""
    import torch

    qk_norm_rope_op()
    return torch.ops.unsloth_h3.qk_norm_rope(x, weight, cos, sin, eps)
