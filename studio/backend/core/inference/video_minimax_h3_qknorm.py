# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused per-head q/k RMSNorm + partial RoPE, bit-identical to the stock eager chain. Off with ``UNSLOTH_H3_QK_ROPE=0``.

Forms (NVIDIA CUDA, each engaged only after a per-device bit-exact check that also races it vs stock):
``fused`` reproduces torch's ``_fused_rms_norm`` row-kernel reduction order; ``rope`` keeps ``F.rms_norm`` and
fuses only the rotation. ``UNSLOTH_H3_QK_ROPE_MODE=fused|rope`` forces one. Unchecked shapes / alignments and a
first call during CUDA graph capture take the stock math.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any, Optional

from .rocm_bf16 import is_rocm_torch

QK_ROPE_ENV = "UNSLOTH_H3_QK_ROPE"
QK_ROPE_MODE_ENV = "UNSLOTH_H3_QK_ROPE_MODE"
_OP = "unsloth_h3::qk_norm_rope"
MODES = ("fused", "rope")

_LOCK = threading.Lock()
_VERDICTS: dict = {}
STATS = {"fused": 0, "rope": 0, "stock": 0}


def qk_rope_enabled() -> bool:
    return (os.environ.get(QK_ROPE_ENV) or "1").strip().lower() not in ("0", "off", "false", "no")


def allowed_modes() -> tuple:
    value = (os.environ.get(QK_ROPE_MODE_ENV) or "auto").strip().lower()
    return (value,) if value in MODES else MODES


def reference_qk_norm_rope(x: Any, weight: Any, cos: Any, sin: Any, eps: float) -> Any:
    """The stock math (RMSNorm then diffusers' ``_apply_rotary_emb``), contiguous."""
    import torch.nn.functional as F
    return _reference_rope(F.rms_norm(x, (x.shape[-1],), weight, eps), cos, sin)


def _reference_rope(x: Any, cos: Any, sin: Any) -> Any:
    import torch

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

    # torch's row kernel: ::rsqrtf (MUFU), an IEEE division by N, an fp32 add of eps
    _rsqrt = libdevice.rsqrt
    _div = libdevice.div_rn
    _add = libdevice.add_rn

    # Explicit-rounding PTX: plain ops get contracted into one bf16 FMA (skips eager's per-product rounding),
    # and libdevice *_rn flushes subnormals.
    @triton.jit
    def _mul_rn(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype = tl.float32, is_pure = True, pack = 1
        )

    @triton.jit
    def _add_rn(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype = tl.float32, is_pure = True, pack = 1
        )

    @triton.jit(do_not_specialize = ["n_rows", "n_heads"])
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
        D: tl.constexpr,  # head dim
        HALF: tl.constexpr,  # rotary dim // 2
        ROWS: tl.constexpr,  # (token, head) rows per program
        NORM: tl.constexpr,  # False: x is already normed (the stock norm ran), rotate only
        G: tl.constexpr,  # warps of torch's row kernel that hold data: max(1, D // 128)
    ):
        odt = out_ptr.dtype.element_ty
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
        if NORM:
            # torch's order: lane l of warp g sums elements 4(32g+l)..+3 with explicit adds (tl.sum may
            # reassociate), tl.sum folds the 32 lanes, then warps in order. The device check proves the layout.
            lane = tl.arange(0, 32)
            for g in tl.static_range(G):
                vec = 32 * g + lane
                vmask = row_ok[:, None] & (vec < D // 4)[None, :]
                vptr = x_ptr + base[:, None] + 4 * vec[None, :]
                acc = tl.zeros([ROWS, 32], dtype = tl.float32)
                for j in tl.static_range(4):
                    v = tl.load(vptr + j, mask = vmask, other = 0.0).to(tl.float32)
                    acc = (
                        acc + v * v
                    )  # a bf16 / fp16 square is exact in fp32: fma or not, the same sum
                if g == 0:
                    total = tl.sum(acc, axis = 1)
                else:
                    total = total + tl.sum(acc, axis = 1)
            rstd = _rsqrt(_add(_div(total, D * 1.0), eps))
            w = tl.load(w_ptr + cols).to(tl.float32)
            wpart = tl.load(w_ptr + partner).to(tl.float32)
            xn = (w[None, :] * (rstd[:, None] * x)).to(odt)
            pn = (wpart[None, :] * (rstd[:, None] * xpart)).to(odt)
        else:
            xn = x.to(odt)
            pn = xpart.to(odt)
        rot_cols = cols < 2 * HALF
        cmask = mask & rot_cols[None, :]
        c = tl.load(cos_ptr + s[:, None] * cos_stride + cols[None, :], mask = cmask, other = 0.0)
        sn = tl.load(sin_ptr + s[:, None] * cos_stride + cols[None, :], mask = cmask, other = 0.0)
        c = c.to(odt).to(tl.float32)
        sn = sn.to(odt).to(tl.float32)
        sign = tl.where(cols < HALF, -1.0, 1.0)
        # eager: x * cos and rot * sin each rounded to the activation dtype, then their sum rounded
        a = _mul_rn(xn.to(tl.float32), c).to(odt).to(tl.float32)
        b = _mul_rn(pn.to(tl.float32) * sign[None, :], sn).to(odt).to(tl.float32)
        roped = _add_rn(a, b).to(odt)
        y = tl.where(rot_cols[None, :], roped, xn)
        tl.store(out_ptr + rows[:, None] * D + cols[None, :], y, mask = mask)

    return _qk_norm_rope_kernel


def _triton_ok(x: Any) -> bool:
    try:
        import torch
        if not (x.is_cuda and not is_rocm_torch(torch)):
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
    norm: bool = True,
) -> Any:
    """One launch: ``norm=True`` the fused form on raw ``x``, ``norm=False`` the rotation of an already-normed ``x``."""
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
            weight.contiguous() if norm else x,
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
            NORM = bool(norm),
            G = max(1, dim // 128),
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


def torch_fused_rms_norm_on_cuda() -> bool:
    """``F.rms_norm`` on CUDA runs ``_fused_rms_norm`` (the row kernel ``fused`` reproduces) on this torch."""
    try:
        import torch
        return bool(torch._C._dispatch_has_kernel_for_dispatch_key("aten::_fused_rms_norm", "CUDA"))
    except Exception:  # noqa: BLE001
        return False


def _fused_row_path(x: Any, weight: Any) -> bool:
    """Inputs hit torch's vectorized row kernel and the 16-byte-aligned specialization the device check proved."""
    align = 4 * x.element_size()
    if weight.dtype != x.dtype or weight.data_ptr() % align or not weight.is_contiguous():
        return False
    if x.data_ptr() % 16 or x.stride(1) % 16 or x.stride(2) % 16:
        return False
    return not (x.is_contiguous() and x.data_ptr() % align)


def _probe_inputs(device: Any, dtype: Any, dim: int, rot: int, seed: int) -> list:
    import torch

    g = torch.Generator(device = "cpu").manual_seed(seed)
    out = []
    for seq, heads, strided in ((257, 6, True), (64, 3, False)):
        if strided:
            qkv = torch.randn(1, seq, 3 * heads * dim, generator = g)
            base = qkv[..., : heads * dim].unflatten(-1, (heads, dim))
        else:
            base = torch.randn(1, seq, heads, dim, generator = g)
        # per-row scales from 1e-3 to 1e3, DC offsets, a few outliers: every rounding branch of the sum
        scale = 10.0 ** torch.randint(-3, 4, (1, seq, heads, 1), generator = g).float()
        base.mul_(scale).add_(torch.randn(1, seq, heads, 1, generator = g) * scale * 0.5)
        base[..., :3] *= 40
        # Keep fp16 probes finite (row scale 1e3 x outlier 40 passes 65504): an inf input makes NaN on both sides.
        cap = torch.finfo(dtype).max / 8
        base.clamp_(-cap, cap)
        if strided:
            qkv = qkv.to(dtype).to(device)
            x = qkv[..., : heads * dim].unflatten(-1, (heads, dim))
        else:
            x = base.to(dtype).to(device)
        w = (1 + 0.2 * torch.randn(dim, generator = g)).to(dtype).to(device)
        ang = torch.rand(seq, rot // 2, generator = g) * 60
        ang = torch.cat((ang, ang), dim = -1).to(device)
        out.append((x, w, ang.cos(), ang.sin(), 1e-5 if seed % 2 else 1e-6))
    return out


def _run_form(form: str, x: Any, w: Any, cos: Any, sin: Any, eps: float) -> Any:
    import torch.nn.functional as F

    if form == "fused":
        return _launch(x, w, cos, sin, eps)
    if form == "rope":
        return _launch(F.rms_norm(x, (x.shape[-1],), w, eps), w, cos, sin, eps, norm = False)
    return reference_qk_norm_rope(x, w, cos, sin, eps)


def _race_ms(
    form: str,
    x: Any,
    w: Any,
    cos: Any,
    sin: Any,
    eps: float,
    reps: int = 7,
) -> float:
    import torch

    for _ in range(2):
        _run_form(form, x, w, cos, sin, eps)
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        a.record()
        _run_form(form, x, w, cos, sin, eps)
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def _check(device: Any, dtype: Any, dim: int, rot: int) -> dict:
    """``{form: matched the eager chain bit for bit on every probe}`` plus ``order``: the passing forms that beat
    the stock chain by more than 2% on a timed probe, fastest first (the stock chain is kept on a tie)."""
    import torch

    passed = {m: True for m in MODES}
    for seed in (11, 12, 13):
        for x, w, cos, sin, eps in _probe_inputs(device, dtype, dim, rot, seed):
            ref = reference_qk_norm_rope(x, w, cos, sin, eps)
            for form in MODES:
                if not passed[form]:
                    continue
                if form == "fused" and not (
                    _fused_row_path(x, w) and torch_fused_rms_norm_on_cuda()
                ):
                    passed[form] = False
                    continue
                try:
                    got = _run_form(form, x, w, cos, sin, eps)
                except torch.OutOfMemoryError:
                    raise
                except Exception:  # noqa: BLE001 -- a JIT / launch failure keeps the stock math
                    got = None
                passed[form] = got is not None and torch.equal(got, ref)
    ms = {}
    if any(passed.values()):
        g = torch.Generator(device = "cpu").manual_seed(29)
        x = torch.randn(1, 2048, 24, dim, generator = g).to(dtype).to(device)
        w = (1 + 0.2 * torch.randn(dim, generator = g)).to(dtype).to(device)
        ang = (torch.rand(2048, rot // 2, generator = g) * 60).to(device)
        cos, sin = torch.cat((ang, ang), -1).cos(), torch.cat((ang, ang), -1).sin()
        for form in ("stock", *[m for m in MODES if passed[m]]):
            try:
                ms[form] = round(_race_ms(form, x, w, cos, sin, 1e-5), 5)
            except torch.OutOfMemoryError:
                raise
            except Exception:  # noqa: BLE001
                ms[form] = float("inf")
    order = sorted(
        (m for m in MODES if passed[m] and ms.get(m, float("inf")) < 0.98 * ms["stock"]), key = ms.get
    )
    return {**passed, "order": order, "ms": ms}


def verdict(x: Any, rot: int) -> Optional[dict]:
    """``_check`` cached per (device, dtype, head dim, rotary dim); None while capturing a CUDA graph before it ran."""
    import torch

    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    key = (index, x.dtype, int(x.shape[-1]), int(rot))
    found = _VERDICTS.get(key)
    if found is not None:
        return found
    if torch.cuda.is_current_stream_capturing():
        return None
    with _LOCK:
        if key not in _VERDICTS:
            with torch.cuda.device(x.device):
                _VERDICTS[key] = _check(x.device, x.dtype, int(x.shape[-1]), int(rot))
    return _VERDICTS[key]


def _mode(x: Any, weight: Any, cos: Any) -> Optional[str]:
    if not (_triton_ok(x) and _supported_layout(x, weight, cos)):
        return None
    passed = verdict(x, int(cos.shape[-1]))
    if not passed:
        return None
    allowed = allowed_modes()
    # a forced form (UNSLOTH_H3_QK_ROPE_MODE) runs whenever it is exact; otherwise the race order decides
    candidates = allowed if len(allowed) == 1 else passed.get("order", ())
    for form in candidates:
        if passed.get(form) and (form != "fused" or _fused_row_path(x, weight)):
            return form
    return None


def qk_norm_rope_impl(x: Any, weight: Any, cos: Any, sin: Any, eps: float) -> Any:
    """The exact Triton form the device check passed, the stock math elsewhere."""
    import torch

    mode = _mode(x, weight, cos)
    try:
        if mode == "fused":
            out = _launch(x, weight, cos, sin, eps)
        elif mode == "rope":
            import torch.nn.functional as F
            out = _launch(
                F.rms_norm(x, (x.shape[-1],), weight, eps), weight, cos, sin, eps, norm = False
            )
        else:
            out = None
    except torch.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 -- a failed launch keeps the stock math
        mode, out = None, None
    STATS[mode or "stock"] += 1
    return out if out is not None else reference_qk_norm_rope(x, weight, cos, sin, eps)


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
