# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 fused q/k RMSNorm + partial RoPE: one pass over q and k per block (``UNSLOTH_H3_QK_ROPE=0`` = off).

Every H3 block normalises q and k per head (``RMSNorm(128)``) and then rotates the leading 96 of the 128 channels
(split-half RoPE, ``cat(-x2, x1)``) before attention. Inductor lowers that chain as separate reductions plus rope
kernels that re-read the input, measured at 5-6x the memory roofline at H3's shape (~19.3k rows x 56 heads): on an
RTX PRO 6000 the q/k prologue is ~4 ms of every block. This op reads each row once and writes it once, contiguous
``(B, S, H, D)``, which is the layout the strided attention processor hands to SDPA.

Numerics follow the eager chain exactly, step by step: the norm in float32 with ONE rounding to bfloat16 at the end
(torch's ``_fused_rms_norm`` decomposition: ``x * rsqrt(mean(x^2) + eps) * w``), cos / sin rounded to the activation
dtype first, then every rope product and the sum rounded to the activation dtype, as the bfloat16 eager ops do.
The only freedom left is the order of the 128-term sum of squares, so a row's rstd can differ from torch's in the
last float32 bit; the oracle bounds that at one bfloat16 rounding step and the end-to-end check is LPIPS.

Opaque ``torch.library`` custom op with a fake kernel, so the regional compile treats it as one node. The Triton
kernel is used on NVIDIA CUDA only; anything else (CPU, ROCm, a failed launch, an unexpected layout) runs the eager
reference, which is the stock math.
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
    """The stock math (RMSNorm module then diffusers' ``_apply_rotary_emb``), contiguous output."""
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

    # torch's norm uses the CUDA rsqrt intrinsic (Inductor emits libdevice.rsqrt): match it, not 1/sqrt. Every
    # product / sum whose result eager rounds to the activation dtype is pinned with *_rn: Triton otherwise contracts
    # ``(a*b).to(bf16).to(f32) + c`` into an FMA and drops the intermediate rounding (measured: 29% of elements off).
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
        # split-half partner inside the rotary part, identity beyond it
        partner = tl.where(cols < HALF, cols + HALF, tl.where(cols < 2 * HALF, cols - HALF, cols))
        base = s * stride_s + h * stride_h
        mask = row_ok[:, None]
        x = tl.load(x_ptr + base[:, None] + cols[None, :], mask = mask, other = 0.0).to(tl.float32)
        xpart = tl.load(x_ptr + base[:, None] + partner[None, :], mask = mask, other = 0.0).to(tl.float32)
        w = tl.load(w_ptr + cols).to(tl.float32)
        wpart = tl.load(w_ptr + partner).to(tl.float32)
        ms = tl.sum(x * x, axis = 1) / D
        rstd = _rsqrt(ms + eps)
        # one rounding to the activation dtype, as the eager norm does
        xn = _mul(_mul(x, rstd[:, None]), w[None, :]).to(out_ptr.dtype.element_ty)
        pn = _mul(_mul(xpart, rstd[:, None]), wpart[None, :]).to(out_ptr.dtype.element_ty)
        rot_cols = cols < 2 * HALF
        cmask = mask & rot_cols[None, :]
        c = tl.load(cos_ptr + s[:, None] * cos_stride + cols[None, :], mask = cmask, other = 0.0)
        sn = tl.load(sin_ptr + s[:, None] * cos_stride + cols[None, :], mask = cmask, other = 0.0)
        c = c.to(out_ptr.dtype.element_ty).to(tl.float32)
        sn = sn.to(out_ptr.dtype.element_ty).to(tl.float32)
        # rotate_half: first half pairs with -x2, second half with +x1
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


# rows per program / warps: tuned at H3's shape (scripts in the change's evidence); overridable for sweeps only
_ROWS = 16
_WARPS = 4


def _launch(x: Any, weight: Any, cos: Any, sin: Any, eps: float, rows_per_program: int = 0, warps: int = 0) -> Any:
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
    """Eager entry: the Triton kernel where it applies, the stock math everywhere else."""
    if _triton_ok(x) and _supported_layout(x, weight, cos):
        try:
            return _launch(x, weight, cos, sin, eps)
        except Exception:  # noqa: BLE001 -- a failed launch falls back to the stock math
            pass
    return reference_qk_norm_rope(x, weight, cos, sin, eps)


@lru_cache(maxsize = None)
def qk_norm_rope_op() -> Any:
    """The registered custom op (one registration per process)."""
    import torch

    # Explicit schema: this module's annotations are strings (``from __future__ import annotations``).
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
    return qk_norm_rope_op()(x, weight, cos, sin, eps)
