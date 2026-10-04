# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused int8 activation path between two torchao int8 GEMMs of a DiT feed-forward: one Triton kernel
reads the int32 ``_int_mm`` output, applies dequant + GELU(tanh) / SwiGLU and writes int8 + the fp32 row scale.

NUMERICS: bit-identical to EAGER torchao + ATen (every bf16 rounding kept, ATen's GELU(tanh) formula op for op (equal
on every bf16 input), fp contraction off so
``y * w_scale + bias`` rounds twice, row scale ``bf16(amax / 127.5)`` clamped at fp32 eps, quantizer multiplies by
the correctly rounded reciprocal). The compiled stock path is NOT eager-exact (Inductor keeps fp32 chains).
Activation-scale contract follows the incoming scale's dtype: fp32 on torchao >= 0.18; bf16 on <= 0.17, where
``int32 * bf16`` rounds twice (int32 -> fp32 -> bf16) and ``1 / s`` and ``x * (1 / s)`` round to bf16.

Anything but a plain dynamic symmetric per-row ``Int8Tensor`` on CUDA + Triton >= 3.2 keeps the stock path.
Kill switch: ``UNSLOTH_DIFFUSION_INT8_FUSED=0``.
"""

from __future__ import annotations

import os
import sys
import threading
import types
from functools import lru_cache
from typing import Any, Optional

INT8_FUSED_ENV = "UNSLOTH_DIFFUSION_INT8_FUSED"
_MIN_TRITON = (3, 2)
# torch._int_mm needs more than 16 rows; torchao's own safe_int_mm handles the rest, so small M keeps the stock path.
_MIN_ROWS = 17
_OP_NAMESPACE = "unsloth_studio"
_OP_NAME = "int8_dq_gelu_quant"
_OP_NAME_SWIGLU = "int8_dq_swiglu_quant"
_SWIGLU_ATTR = "_unsloth_i8_swiglu"

_LOCK = threading.Lock()
# Read by the traced forwards: dynamo must not trace into the lru_cache'd registration.
_OP_HANDLE: Any = None
# torchao's safe_int_mm home, resolved at install (outside any trace) for the same reason.
_INTMM_MODULE: Any = None
# Marker on each patched module (no global registry: it would pin an unloaded transformer).
_MARK = "_unsloth_i8_fused_prev"
_NO_PREV = object()


def int8_fused_disabled() -> bool:
    return (os.environ.get(INT8_FUSED_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _triton_version_ok(version: Optional[str] = None) -> bool:
    if version is None:
        try:
            import triton
            version = str(triton.__version__)
        except Exception:  # noqa: BLE001
            return False
    import re

    match = re.match(r"(\d+)\.(\d+)", version)
    return bool(match) and (int(match.group(1)), int(match.group(2))) >= _MIN_TRITON


@lru_cache(maxsize = 1)
def _triton_jit_toolchain_ok() -> bool:
    """Windows: Triton needs the MSVC CRT headers. Elsewhere, or if the probe fails, True."""
    if sys.platform != "win32":
        return True
    try:
        from .._msvc_env import crt_headers_reachable
        return bool(crt_headers_reachable())
    except Exception:  # noqa: BLE001
        return True


@lru_cache(maxsize = 1)
def _kernels() -> Optional[types.SimpleNamespace]:
    """Compile-on-first-use Triton kernel, or None when Triton is unavailable."""
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001 - no Triton means the stock path
        return None
    # Triton <= 3.2 resolves string annotations against module globals, not this function's locals
    globals().update(triton = triton, tl = tl)

    @triton.jit
    def _rbf16(x):
        # Round-to-nearest-even to bf16, kept in fp32 (one cvt; faster here than the integer-op form).
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _gelu_tanh(y):
        # ATen's GELU(tanh) op for op (y / (1 + exp(-2u)) is NOT bf16-equal); *_rn: ptxas FMA-fuses f32x2 on sm_100.
        cube = tl.extra.cuda.libdevice.mul_rn(tl.extra.cuda.libdevice.mul_rn(y, y), y)
        inner = 0.7978845608028654 * tl.extra.cuda.libdevice.add_rn(
            y, tl.extra.cuda.libdevice.mul_rn(0.044715, cube)
        )
        return 0.5 * y * (1.0 + tl.extra.cuda.libdevice.tanh(inner))

    @triton.jit
    def _pre(
        c_ptr,
        xs,
        ws_ptr,
        b_ptr,
        offs,
        mask,
        HAS_BIAS: tl.constexpr,
        WS_FP32: tl.constexpr,
        EVICT: tl.constexpr,
        FINAL_ROUND: tl.constexpr,
        FP32_SCALE: tl.constexpr,
        PIN_RN: tl.constexpr,
    ):
        # torchao Int8Tensor linear epilogue: (int32 * x_scale).to(bf16) * w_scale (+ bias), then .to(bf16).
        # FINAL_ROUND=False leaves the last rounding to the caller (it commutes with a row max).
        c = tl.load(c_ptr + offs, mask = mask, other = 0, eviction_policy = EVICT)
        if FP32_SCALE:
            c = c.to(tl.float32)
        else:
            # int32 -> fp32 -> bf16 rounds twice; Triton folds a plain cast chain into one rounding.
            c = _rbf16(tl.extra.cuda.libdevice.int2float_rn(c))
        y = _rbf16(c * xs)
        w = tl.load(ws_ptr + offs, mask = mask, other = 0.0, eviction_policy = "evict_last").to(
            tl.float32
        )
        # PIN_RN (sm_100+): ptxas fuses packed f32x2 mul + add into an FMA even with fp fusion off; elsewhere it costs.
        if PIN_RN:
            y = tl.extra.cuda.libdevice.mul_rn(y, w)
        else:
            y = y * w
        if not WS_FP32:
            y = _rbf16(y)
        if HAS_BIAS:
            b = tl.load(b_ptr + offs, mask = mask, other = 0.0, eviction_policy = "evict_last").to(
                tl.float32
            )
            if PIN_RN:
                y = tl.extra.cuda.libdevice.add_rn(y, b)
            else:
                y = y + b
        if FINAL_ROUND:
            y = _rbf16(y)
        return y

    @triton.jit
    def dq_gelu_quant(
        c_ptr,
        xs_ptr,
        ws_ptr,
        b_ptr,
        p_ptr,
        q_ptr,
        s_ptr,
        N,
        NA,
        S,
        stride_c,
        stride_q,
        p_sb,
        p_ss,
        p_sh,
        HAS_BIAS: tl.constexpr,
        WS_FP32: tl.constexpr,
        FP32_SCALE: tl.constexpr,
        PIN_RN: tl.constexpr,
        HAS_PREFIX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CHUNK: tl.constexpr,
        CHUNK_A: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        xs = tl.load(xs_ptr + row).to(tl.float32)
        base = tl.arange(0, CHUNK)
        crow = c_ptr + row * stride_c
        qrow = q_ptr + row * stride_q
        # pass 1a: max of the pre-activation. GELU is increasing for y > -0.75 and |gelu(y)| < 0.16997 for y < 0,
        # so max|bf16(gelu)| over the row is bf16(gelu(max y)) whenever that is >= 0.171875; otherwise recompute.
        vmax = tl.full((CHUNK,), float("-inf"), tl.float32)
        for k in range(0, N, CHUNK):
            offs = k + base
            m = offs < N
            y = _pre(
                crow,
                xs,
                ws_ptr,
                b_ptr,
                offs,
                m,
                HAS_BIAS,
                WS_FP32,
                "evict_last",
                False,
                FP32_SCALE,
                PIN_RN,
            )
            vmax = tl.maximum(vmax, tl.where(m, y, float("-inf")))
        # bf16 rounding is monotone, so max(round(v)) == round(max(v)): round once, after the reduction.
        amax = _rbf16(_gelu_tanh(_rbf16(tl.max(vmax, axis = 0))))
        if amax < 0.171875:
            acc = tl.zeros((CHUNK,), dtype = tl.float32)
            for k in range(0, N, CHUNK):
                offs = k + base
                m = offs < N
                g = _rbf16(
                    _gelu_tanh(
                        _pre(
                            crow,
                            xs,
                            ws_ptr,
                            b_ptr,
                            offs,
                            m,
                            HAS_BIAS,
                            WS_FP32,
                            "evict_last",
                            True,
                            FP32_SCALE,
                            PIN_RN,
                        )
                    )
                )
                acc = tl.maximum(acc, tl.where(m, tl.abs(g), 0.0))
            amax = tl.max(acc, axis = 0)
        if HAS_PREFIX:
            # FluxSingleTransformerBlock: the bf16 attention output sits in front of the GELU branch in the row.
            b_idx = row // S
            s_idx = row % S
            pbase = p_ptr + b_idx * p_sb + s_idx * p_ss
            abase = tl.arange(0, CHUNK_A)
            pacc = tl.zeros((CHUNK_A,), dtype = tl.float32)
            for k in range(0, NA, CHUNK_A):
                j = k + abase
                m = j < NA
                v = tl.load(pbase + (j // HEAD_DIM) * p_sh + (j % HEAD_DIM), mask = m, other = 0.0).to(
                    tl.float32
                )
                pacc = tl.maximum(pacc, tl.abs(v))
            amax = tl.maximum(amax, tl.max(pacc, axis = 0))
        scale = _rbf16(tl.math.div_rn(amax, 127.5))
        scale = tl.maximum(scale, 1.1920928955078125e-07)
        inv = tl.math.div_rn(1.0, scale)
        if not FP32_SCALE:
            inv = _rbf16(inv)
        tl.store(s_ptr + row, scale.to(s_ptr.dtype.element_ty))
        if HAS_PREFIX:
            for k in range(0, NA, CHUNK_A):
                j = k + abase
                m = j < NA
                v = tl.load(pbase + (j // HEAD_DIM) * p_sh + (j % HEAD_DIM), mask = m, other = 0.0).to(
                    tl.float32
                )
                p = v * inv
                if not FP32_SCALE:
                    p = _rbf16(p)
                qi = tl.extra.cuda.libdevice.nearbyint(p)
                qi = tl.minimum(tl.maximum(qi, -128.0), 127.0)
                tl.store(qrow + j, qi.to(tl.int8), mask = m)
        # pass 2: rebuild (the int32 row is mostly still in L2) and quantize.
        for k in range(0, N, CHUNK):
            offs = k + base
            m = offs < N
            g = _rbf16(
                _gelu_tanh(
                    _pre(
                        crow,
                        xs,
                        ws_ptr,
                        b_ptr,
                        offs,
                        m,
                        HAS_BIAS,
                        WS_FP32,
                        "evict_first",
                        True,
                        FP32_SCALE,
                        PIN_RN,
                    )
                )
            )
            p = g * inv
            if not FP32_SCALE:
                p = _rbf16(p)
            qi = tl.extra.cuda.libdevice.nearbyint(p)
            qi = tl.minimum(tl.maximum(qi, -128.0), 127.0)
            tl.store(qrow + NA + offs, qi.to(tl.int8), mask = m)

    @triton.jit
    def _silu(y):
        # ATen: x / (1 + exp(-x)) in fp32
        return y / (1.0 + tl.exp(-y))

    @triton.jit
    def _swiglu_val(
        crow,
        xs,
        ws_ptr,
        b_ptr,
        g0,
        v0,
        offs,
        m,
        HAS_BIAS: tl.constexpr,
        WS_FP32: tl.constexpr,
        EVICT: tl.constexpr,
        FP32_SCALE: tl.constexpr,
        PIN_RN: tl.constexpr,
    ):
        g = _pre(
            crow + g0,
            xs,
            ws_ptr + g0,
            b_ptr + g0,
            offs,
            m,
            HAS_BIAS,
            WS_FP32,
            EVICT,
            True,
            FP32_SCALE,
            PIN_RN,
        )
        v = _pre(
            crow + v0,
            xs,
            ws_ptr + v0,
            b_ptr + v0,
            offs,
            m,
            HAS_BIAS,
            WS_FP32,
            EVICT,
            True,
            FP32_SCALE,
            PIN_RN,
        )
        return _rbf16(_rbf16(_silu(g)) * v)

    @triton.jit
    def dq_swiglu_quant(
        c_ptr,
        xs_ptr,
        ws_ptr,
        b_ptr,
        h_ptr,
        q_ptr,
        s_ptr,
        N,
        G0,
        V0,
        stride_c,
        stride_q,
        HAS_BIAS: tl.constexpr,
        WS_FP32: tl.constexpr,
        FP32_SCALE: tl.constexpr,
        PIN_RN: tl.constexpr,
        CHUNK: tl.constexpr,
    ):
        # c row = the fused GEMM output; the gate half starts at column G0, the value half at V0 (both N wide).
        # Two int32 rows per output row make recomputing in pass 2 ALU-bound, so pass 1 parks the bf16 product in a
        # scratch row (still in L2 when pass 2 reads it back).
        row = tl.program_id(0).to(tl.int64)
        xs = tl.load(xs_ptr + row).to(tl.float32)
        base = tl.arange(0, CHUNK)
        crow = c_ptr + row * stride_c
        hrow = h_ptr + row * N
        acc = tl.zeros((CHUNK,), dtype = tl.float32)
        for k in range(0, N, CHUNK):
            offs = k + base
            m = offs < N
            h = _swiglu_val(
                crow,
                xs,
                ws_ptr,
                b_ptr,
                G0,
                V0,
                offs,
                m,
                HAS_BIAS,
                WS_FP32,
                "evict_first",
                FP32_SCALE,
                PIN_RN,
            )
            tl.store(hrow + offs, h.to(tl.bfloat16), mask = m, eviction_policy = "evict_last")
            acc = tl.maximum(acc, tl.where(m, tl.abs(h), 0.0))
        amax = tl.max(acc, axis = 0)
        scale = _rbf16(tl.math.div_rn(amax, 127.5))
        scale = tl.maximum(scale, 1.1920928955078125e-07)
        inv = tl.math.div_rn(1.0, scale)
        if not FP32_SCALE:
            inv = _rbf16(inv)
        tl.store(s_ptr + row, scale.to(s_ptr.dtype.element_ty))
        qrow = q_ptr + row * stride_q
        for k in range(0, N, CHUNK):
            offs = k + base
            m = offs < N
            h = tl.load(hrow + offs, mask = m, other = 0.0, eviction_policy = "evict_first").to(
                tl.float32
            )
            p = h * inv
            if not FP32_SCALE:
                p = _rbf16(p)
            qi = tl.extra.cuda.libdevice.nearbyint(p)
            qi = tl.minimum(tl.maximum(qi, -128.0), 127.0)
            tl.store(qrow + offs, qi.to(tl.int8), mask = m)

    @triton.jit
    def gelu_bf16(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        y = tl.load(x_ptr + offs, mask = m, other = 0.0).to(tl.float32)
        tl.store(o_ptr + offs, _gelu_tanh(y).to(tl.bfloat16), mask = m)

    @triton.jit
    def dq_bf16(
        c_ptr,
        xs_ptr,
        ws_ptr,
        b_ptr,
        o_ptr,
        N,
        HAS_BIAS: tl.constexpr,
        WS_FP32: tl.constexpr,
        FP32_SCALE: tl.constexpr,
        PIN_RN: tl.constexpr,
        CHUNK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        xs = tl.load(xs_ptr + row).to(tl.float32)
        base = tl.arange(0, CHUNK)
        for k in range(0, N, CHUNK):
            offs = k + base
            m = offs < N
            y = _pre(
                c_ptr + row * N,
                xs,
                ws_ptr,
                b_ptr,
                offs,
                m,
                HAS_BIAS,
                WS_FP32,
                "evict_first",
                True,
                FP32_SCALE,
                PIN_RN,
            )
            tl.store(o_ptr + row * N + offs, y.to(tl.bfloat16), mask = m)

    return types.SimpleNamespace(
        dq_gelu_quant = dq_gelu_quant,
        dq_swiglu_quant = dq_swiglu_quant,
        gelu_bf16 = gelu_bf16,
        dq_bf16 = dq_bf16,
        triton = triton,
    )


@lru_cache(maxsize = 8)
def _pin_rn_index(index: int) -> bool:
    import torch
    return torch.cuda.get_device_capability(index)[0] >= 10


def _pin_rn(device: Any) -> bool:
    """Packed f32x2 mul / add (which ptxas contracts) exist from sm_100 on."""
    import torch
    return _pin_rn_index(device.index if device.index is not None else torch.cuda.current_device())


def _scale_dtype(xs: Any) -> Any:
    """The output act-quant scale dtype: the incoming one (torchao's contract: fp32 on >= 0.18, bf16 on <= 0.17)."""
    import torch
    return torch.float32 if xs.dtype == torch.float32 else torch.bfloat16


def _launch(c: Any, xs: Any, ws: Any, bias: Any, prefix: Any) -> tuple:
    """Run the kernel. ``prefix`` is None or a [B, S, H, D] view (any strides, last dim contiguous)."""
    import torch

    k = _kernels()
    m_rows, n = c.shape
    if prefix is not None:
        b, s, h, d = prefix.shape
        na = h * d
        p_sb, p_ss, p_sh = prefix.stride(0), prefix.stride(1), prefix.stride(2)
    else:
        s, d, na = 1, 1, 0
        p_sb = p_ss = p_sh = 0
    q = torch.empty((m_rows, na + n), device = c.device, dtype = torch.int8)
    scale = torch.empty((m_rows,), device = c.device, dtype = _scale_dtype(xs))
    with torch.cuda.device(c.device):
        k.dq_gelu_quant[(m_rows,)](
            c,
            xs,
            ws,
            bias if bias is not None else ws,
            prefix if prefix is not None else ws,
            q,
            scale,
            n,
            na,
            s,
            c.stride(0),
            q.stride(0),
            p_sb,
            p_ss,
            p_sh,
            HAS_BIAS = bias is not None,
            WS_FP32 = ws.dtype == torch.float32,
            FP32_SCALE = xs.dtype == torch.float32,
            PIN_RN = _pin_rn(xs.device),
            HAS_PREFIX = prefix is not None,
            HEAD_DIM = d,
            CHUNK = 2048,
            CHUNK_A = 1024,
            num_warps = 8,
            enable_fp_fusion = False,
        )
    return q, scale


def _launch_swiglu(
    c: Any, xs: Any, ws: Any, bias: Any, gate_col: int, value_col: int, n: int
) -> tuple:
    """SwiGLU on the int32 output ``c`` of one fused GEMM: gate columns [gate_col, +n), value columns [value_col, +n)."""
    import torch

    k = _kernels()
    m_rows = c.shape[0]
    q = torch.empty((m_rows, n), device = c.device, dtype = torch.int8)
    scale = torch.empty((m_rows,), device = c.device, dtype = _scale_dtype(xs))
    scratch = torch.empty((m_rows, n), device = c.device, dtype = torch.bfloat16)
    with torch.cuda.device(c.device):
        k.dq_swiglu_quant[(m_rows,)](
            c,
            xs,
            ws,
            bias if bias is not None else ws,
            scratch,
            q,
            scale,
            n,
            gate_col,
            value_col,
            c.stride(0),
            q.stride(0),
            HAS_BIAS = bias is not None,
            WS_FP32 = ws.dtype == torch.float32,
            FP32_SCALE = xs.dtype == torch.float32,
            PIN_RN = _pin_rn(xs.device),
            CHUNK = 2048,
            num_warps = 8,
            enable_fp_fusion = False,
        )
    return q, scale


def reference_dq_swiglu_quant(
    c: Any, xs: Any, ws: Any, bias: Any, gate_col: int, value_col: int, n: int
) -> tuple:
    """Eager mirror: two torchao epilogues (gate, value), ATen SiLU, bf16 product, torchao per-row act quant."""
    import torch
    import torch.nn.functional as F

    def part(col):
        y = (c[:, col : col + n] * xs.reshape(-1, 1)).to(torch.bfloat16) * ws[col : col + n]
        if bias is not None:
            y = y + bias[col : col + n]
        return y.to(torch.bfloat16)

    h = F.silu(part(gate_col)) * part(value_col)
    return _reference_act_quant(h, _scale_dtype(xs))


def _reference_act_quant(h: Any, scale_dtype: Any) -> tuple:
    """torchao ``Int8Tensor.from_hp(h, PerRow())`` for a bf16 ``h``."""
    import torch

    amax = torch.maximum(-h.amin(dim = 1).clamp(max = 0), h.amax(dim = 1).clamp(min = 0))
    scale = (amax / 127.5).clamp(min = torch.finfo(torch.float32).eps).to(scale_dtype)
    q = torch.clamp(torch.round(h * (1.0 / scale).reshape(-1, 1)), -128, 127).to(torch.int8)
    return q, scale


@lru_cache(maxsize = 1)
def _op() -> Any:
    """The torch.library op (opaque to dynamo, CUDA-graph safe: no host sync, allocations only), or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    qualname = f"{_OP_NAMESPACE}::{_OP_NAME}"
    ns = getattr(torch.ops, _OP_NAMESPACE, None)
    if ns is not None and hasattr(ns, _OP_NAME) and hasattr(ns, _OP_NAME_SWIGLU):
        return types.SimpleNamespace(
            gelu = getattr(ns, _OP_NAME), swiglu = getattr(ns, _OP_NAME_SWIGLU)
        )
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:  # torch < 2.4
        return None
    try:
        # Explicit schema: this module's annotations are strings (``from __future__ import annotations``) and torch is
        # imported lazily, so schema inference from them would not resolve.
        @custom_op(
            qualname,
            mutates_args = (),
            schema = "(Tensor c, Tensor xs, Tensor ws, Tensor? bias, Tensor? prefix) -> (Tensor, Tensor)",
        )
        def _dq_gelu_quant(c, xs, ws, bias, prefix):
            return _launch(c, xs, ws, bias, prefix)

        @_dq_gelu_quant.register_fake
        def _(c, xs, ws, bias, prefix):
            na = 0 if prefix is None else prefix.shape[-2] * prefix.shape[-1]
            return (
                c.new_empty((c.shape[0], na + c.shape[1]), dtype = torch.int8),
                c.new_empty((c.shape[0],), dtype = _scale_dtype(xs)),
            )

        @custom_op(
            f"{_OP_NAMESPACE}::{_OP_NAME_SWIGLU}",
            mutates_args = (),
            schema = "(Tensor c, Tensor xs, Tensor ws, Tensor? bias, int gate_col, int value_col, int n) -> (Tensor, Tensor)",
        )
        def _dq_swiglu_quant(c, xs, ws, bias, gate_col, value_col, n):
            return _launch_swiglu(c, xs, ws, bias, gate_col, value_col, n)

        @_dq_swiglu_quant.register_fake
        def _(c, xs, ws, bias, gate_col, value_col, n):
            return (
                c.new_empty((c.shape[0], n), dtype = torch.int8),
                c.new_empty((c.shape[0],), dtype = _scale_dtype(xs)),
            )
    except Exception:  # noqa: BLE001 - a registration failure keeps the stock path
        return None
    ns = getattr(torch.ops, _OP_NAMESPACE)
    return types.SimpleNamespace(gelu = getattr(ns, _OP_NAME), swiglu = getattr(ns, _OP_NAME_SWIGLU))


@lru_cache(maxsize = 8)
def _device_ok(index: int) -> bool:
    """CUDA NVIDIA, Triton new enough, toolchain present, kernel compiles and matches the reference once."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    if (
        not (_triton_version_ok() and _triton_jit_toolchain_ok())
        or _kernels() is None
        or _op() is None
    ):
        return False
    try:
        dev = torch.device("cuda", index)
        g = torch.Generator(device = "cpu").manual_seed(0)
        c = torch.randint(-(2**20), 2**20, (33, 200), generator = g, dtype = torch.int32)
        c[1:5] = _bf16_tie_ints((4, 200), g)
        c = c.to(dev)
        xs = (torch.rand(33, generator = g) * 1e-3 + 1e-5).to(torch.bfloat16)
        xs[1:5] *= 2**-10
        ws = (torch.rand(200, generator = g) * 1e-4 + 1e-6).to(dev)
        bias = (torch.randn(200, generator = g) * 0.1).to(torch.bfloat16).to(dev)
        for x_scale in (xs.float().to(dev), xs.to(dev)):
            for w_scale in (ws, ws.to(torch.bfloat16)):
                q, s = _launch(c, x_scale, w_scale, bias, None)
                q_ref, s_ref = reference_dq_gelu_quant(c, x_scale, w_scale, bias, None)
                if not (s.dtype == s_ref.dtype and torch.equal(q, q_ref) and torch.equal(s, s_ref)):
                    return False
                q, s = _launch_swiglu(c, x_scale, w_scale, bias, 104, 0, 96)
                q_ref, s_ref = reference_dq_swiglu_quant(c, x_scale, w_scale, bias, 104, 0, 96)
                if not (s.dtype == s_ref.dtype and torch.equal(q, q_ref) and torch.equal(s, s_ref)):
                    return False
        return (
            _act_quant_contract_ok(dev)
            and _gelu_matches_aten(dev)
            and _epilogue_matches_torchao(dev)
        )
    except Exception:  # noqa: BLE001 - any build / launch failure keeps the stock path
        return False


def _bf16_tie_ints(shape: tuple, generator: Any) -> Any:
    """int32 values in [2^24, 2^30) within one fp32 ulp of a bf16 midpoint: one vs two roundings disagree there."""
    import torch

    n = 1
    for d in shape:
        n *= d
    e = torch.randint(24, 30, (n,), generator = generator, dtype = torch.int64)
    mant = torch.randint(0, 128, (n,), generator = generator, dtype = torch.int64)
    ulp = torch.bitwise_left_shift(torch.ones_like(e), e - 23)
    off = torch.randint(-1, 2, (n,), generator = generator, dtype = torch.int64) * (ulp // 2).clamp(
        min = 1
    ) + torch.randint(-1, 2, (n,), generator = generator, dtype = torch.int64)
    mid = (
        torch.bitwise_left_shift(torch.ones_like(e), e)
        + mant * torch.bitwise_left_shift(torch.ones_like(e), e - 7)
        + torch.bitwise_left_shift(torch.ones_like(e), e - 8)
    )
    sign = torch.randint(0, 2, (n,), generator = generator, dtype = torch.int64) * 2 - 1
    return ((mid + off) * sign).to(torch.int32).reshape(shape)


def _epilogue_matches_torchao(
    dev: Any,
    m: int = 1031,
    n: int = 2056,
) -> bool:
    """The epilogue equals torchao's ``bf16(bf16(c * xs) * ws + bias)``; fp32 ``ws`` + bias exposes an FMA."""
    import torch

    g = torch.Generator(device = "cpu").manual_seed(2)
    c = torch.randint(-(2**20), 2**20, (m, n), generator = g, dtype = torch.int32).to(dev)
    ws = (torch.rand(n, generator = g) * 1e-4 + 1e-6).to(dev)
    bias = (torch.randn(n, generator = g) * 0.1).to(torch.bfloat16).to(dev)
    xs = (torch.rand(m, generator = g) * 1e-3 + 1e-5).to(torch.bfloat16)
    k = _kernels()
    for x_scale in (xs.float().to(dev), xs.to(dev)):
        out = torch.empty((m, n), device = dev, dtype = torch.bfloat16)
        with torch.cuda.device(dev):
            k.dq_bf16[(m,)](
                c,
                x_scale,
                ws,
                bias,
                out,
                n,
                HAS_BIAS = True,
                WS_FP32 = True,
                FP32_SCALE = x_scale.dtype == torch.float32,
                PIN_RN = _pin_rn(x_scale.device),
                CHUNK = 2048,
                num_warps = 8,
                enable_fp_fusion = False,
            )
        ref = ((c * x_scale.reshape(-1, 1)).to(torch.bfloat16) * ws + bias).to(torch.bfloat16)
        if not torch.equal(out, ref):
            return False
    return True


def _gelu_matches_aten(dev: Any) -> bool:
    """The kernels' GELU(tanh) equals ``F.gelu(x, approximate="tanh")`` on every finite bf16 ``x``, bit for bit."""
    import torch
    import torch.nn.functional as F

    x = torch.arange(-(2**15), 2**15, dtype = torch.int32).to(torch.int16).view(torch.bfloat16)
    x = x[torch.isfinite(x)].to(dev)
    out = torch.empty_like(x)
    k = _kernels()
    with torch.cuda.device(dev):
        k.gelu_bf16[(k.triton.cdiv(x.numel(), 1024),)](
            x, out, x.numel(), BLOCK = 1024, num_warps = 4, enable_fp_fusion = False
        )
    return bool(torch.equal(out.view(torch.int16), F.gelu(x, approximate = "tanh").view(torch.int16)))


def _act_quant_contract_ok(dev: Any) -> bool:
    """``_reference_act_quant`` in this torchao's scale dtype equals ``Int8Tensor.from_hp(h, PerRow())`` bit for bit."""
    import torch
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor

    g = torch.Generator(device = "cpu").manual_seed(1)
    h = torch.randn(37, 520, generator = g) * (torch.rand(1, 520, generator = g) * 4)
    h[:, :3] *= 50
    h[2] = 0
    h = h.to(torch.bfloat16).to(dev)
    t = Int8Tensor.from_hp(h, PerRow())
    s_ref = t.scale.reshape(-1)
    q, s = _reference_act_quant(h, s_ref.dtype)
    return bool(torch.equal(q, t.qdata) and torch.equal(s, s_ref))


def reference_dq_gelu_quant(c: Any, xs: Any, ws: Any, bias: Any, prefix: Any) -> tuple:
    """Eager torch mirror of the stock chain (torchao epilogue, ATen GELU, torchao per-row act quant)."""
    import torch
    import torch.nn.functional as F

    y = (c * xs.reshape(-1, 1)).to(torch.bfloat16)
    y = y * ws.flatten()
    if bias is not None:
        y = y + bias
    g = F.gelu(y.to(torch.bfloat16), approximate = "tanh")
    if prefix is not None:
        g = torch.cat(
            [prefix.reshape(prefix.shape[0] * prefix.shape[1], -1).to(g.dtype), g], dim = -1
        )
    return _reference_act_quant(g, _scale_dtype(xs))


def _plain_int8_weight(w: Any) -> bool:
    """A torchao Int8Tensor with dynamic, symmetric, per-row activation quant and nothing else attached."""
    try:
        if type(w).__name__ != "Int8Tensor":
            return False
        kw = getattr(w, "act_quant_kwargs", None)
        if kw is None or getattr(w, "act_pre_scale", None) is not None:
            return False
        if (
            getattr(w, "act_quant_scale", None) is not None
            or getattr(w, "act_quant_zero_point", None) is not None
        ):
            return False
        if getattr(kw, "reduce_range", False) or getattr(w, "reduce_range", False):
            return False
        if "SYMMETRIC" not in str(getattr(kw, "mapping_type", "")) or "ASYMMETRIC" in str(
            getattr(kw, "mapping_type", "")
        ):
            return False
        if type(getattr(kw, "granularity", None)).__name__ != "PerRow":
            return False
        qd, sc = w.qdata, w.scale
        if qd.dim() != 2 or list(getattr(w, "block_size", [])) != [1, qd.shape[1]]:
            return False
        if sc.dim() != 2 or tuple(sc.shape) != (qd.shape[0], 1):
            return False
        return qd.shape[0] % 8 == 0 and qd.shape[1] % 8 == 0
    except Exception:  # noqa: BLE001
        return False


def _act_quant(x2d: Any, weight: Any) -> tuple:
    """torchao's own dynamic activation quant for ``weight`` (the stock path's call), as (int8 rows, fp32 scale)."""
    from torchao.quantization.quantize_.common.quantize_tensor_kwargs import (
        _choose_quant_func_and_quantize_tensor,
    )

    kwargs = {"scale": weight.act_quant_scale}
    # torchao <= 0.16 has no activation zero point (neither the attribute nor the kwarg).
    if hasattr(weight, "act_quant_zero_point"):
        kwargs["zero_point"] = weight.act_quant_zero_point
    act = _choose_quant_func_and_quantize_tensor(x2d, weight.act_quant_kwargs, **kwargs)
    return act.qdata, act.scale


def _resolve_intmm() -> Any:
    """The module holding torchao's ``safe_int_mm``, or None. torchao <= 0.18 ships it in ``torchao.kernel.intmm``;
    main after pytorch/ao#4718 moved it to the int8 workflow and deleted ``torchao.kernel``. Read off the module at
    call time, so Studio's capture-safe rebinding (diffusion_torchao_patches) is the one that runs."""
    import importlib

    from .diffusion_torchao_patches import _TORCHAO_INTMM_MODULES

    for name in _TORCHAO_INTMM_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if callable(getattr(module, "safe_int_mm", None)):
            return module
    return None


def _int_mm(a: Any, weight: Any) -> Any:
    module = _INTMM_MODULE
    if module is None:  # a direct call before any install (tests): resolve eagerly
        module = _resolve_intmm()
        if module is None:
            raise ImportError("torchao ships no safe_int_mm in any known module")
    return module.safe_int_mm(a, weight.qdata.contiguous().t())


def _linear_from_q(q: Any, xs: Any, weight: Any, bias: Any, out_dtype: Any) -> Any:
    """torchao Int8Tensor linear on an already-quantized activation: same ops, same order, as the stock epilogue."""
    import torch

    if out_dtype == torch.bfloat16:
        from .diffusion_int8_gemm import linear_from_q
        fused = linear_from_q(q, xs.reshape(-1), weight, bias)
        if fused is not None:
            return fused
    inter = torch.float32 if xs.dtype == torch.float16 else xs.dtype
    y = (_int_mm(q, weight) * xs.reshape(-1, 1).to(inter)).to(out_dtype)
    y = y * weight.scale.flatten()
    if bias is not None:
        y = y + bias
    return y.to(out_dtype)


def _ff_forward(self: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
    """``FeedForward.forward`` for GELU(tanh) -> Linear with plain int8 weights."""
    import torch

    proj = self.net[0].proj
    down = self.net[2]
    lead = hidden_states.shape[:-1]
    x2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    if (
        args
        or kwargs
        or x2d.shape[0] < _MIN_ROWS
        or not hidden_states.is_cuda
        or hidden_states.dtype != torch.bfloat16
    ):
        return type(self).forward(self, hidden_states, *args, **kwargs)
    xq, xs = _act_quant(x2d, proj.weight)
    c = _int_mm(xq.reshape(-1, xq.shape[-1]), proj.weight)
    q, s = _OP_HANDLE.gelu(c, xs.reshape(-1), proj.weight.scale.flatten(), proj.bias, None)
    y = _linear_from_q(q, s, down.weight, down.bias, hidden_states.dtype)
    for extra in self.net[3:]:
        y = extra(y)
    return y.reshape(*lead, y.shape[-1])


def _flux_single_forward(
    self: Any,
    hidden_states: Any,
    encoder_hidden_states: Any,
    temb: Any,
    image_rotary_emb: Any = None,
    joint_attention_kwargs: Any = None,
) -> Any:
    """``FluxSingleTransformerBlock.forward`` with the GELU branch, the concat and ``proj_out``'s activation
    quant fused into one kernel. Mirrors the installed class forward (stock or Studio's addcmul arch patch)."""
    import torch

    text_seq_len = encoder_hidden_states.shape[1]
    rows = hidden_states.shape[0] * (hidden_states.shape[1] + text_seq_len)
    if rows < _MIN_ROWS or hidden_states.dtype != torch.bfloat16 or not hidden_states.is_cuda:
        return type(self).forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
            joint_attention_kwargs,
        )
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim = 1)
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb = temb)
    x2d = norm_hidden_states.reshape(rows, -1)
    xq, xs = _act_quant(x2d, self.proj_mlp.weight)
    c = _int_mm(xq.reshape(rows, -1), self.proj_mlp.weight)
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states = norm_hidden_states,
        image_rotary_emb = image_rotary_emb,
        **joint_attention_kwargs,
    )
    heads = self.attn.heads
    prefix = attn_output.unflatten(-1, (heads, attn_output.shape[-1] // heads))
    if prefix.stride(-1) != 1:
        prefix = prefix.contiguous()
    q, s = _OP_HANDLE.gelu(
        c, xs.reshape(-1), self.proj_mlp.weight.scale.flatten(), self.proj_mlp.bias, prefix
    )
    out = _linear_from_q(q, s, self.proj_out.weight, self.proj_out.bias, hidden_states.dtype)
    out = out.reshape(*hidden_states.shape[:-1], out.shape[-1])
    gate = gate.unsqueeze(1)
    if getattr(self, "_unsloth_i8_addcmul", False):
        hidden_states = torch.addcmul(residual, gate, out)
    else:
        hidden_states = residual + gate * out
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]


def _swiglu_spec(module: Any) -> Optional[tuple]:
    """(fused in-proj name, None, down name, width, gate column, value column) for the SwiGLU MLPs whose two halves
    already come from ONE Linear, else None."""
    name = type(module).__name__
    try:
        if name == "FeedForward" and type(module).__module__ == "diffusers.models.attention":
            from diffusers.models.activations import SwiGLU

            net = getattr(module, "net", None)
            if net is None or len(net) < 3 or type(net[0]) is not SwiGLU:
                return None
            if any(type(extra).__name__ != "Dropout" for extra in (net[1], *net[3:])):
                return None
            n = net[0].proj.out_features // 2
            # SwiGLU.forward: hidden, gate = proj(x).chunk(2) -> value first, gate second
            return ("net.0.proj", None, "net.2", n, n, 0)
        if (
            name == "Flux2FeedForward"
            and type(getattr(module, "act_fn", None)).__name__ == "Flux2SwiGLU"
        ):
            n = module.linear_in.out_features // 2
            return ("linear_in", None, "linear_out", n, 0, n)  # gate first, value second (FLUX.2)
    except Exception:  # noqa: BLE001
        return None
    return None


def _split_spec(module: Any) -> Optional[tuple]:
    """Two separate projections (gate, value) + down: Z-Image FeedForward (w1, w3, w2), Qwen-Image-2.1 SwiGLU."""
    name = type(module).__name__
    mod = type(module).__module__
    if (
        name == "FeedForward"
        and mod.endswith("transformer_z_image")
        and all(hasattr(module, n) for n in ("w1", "w2", "w3"))
    ):
        return ("w1", "w3", "w2")
    if name == "QwenImage21SwiGLUFeedForward":
        return ("gate_layer", "proj", "out")
    return None


# SwiGLU kernel is bit-exact vs eager but moved FLUX.2-klein / Qwen-Image-2.1 compiled renders further from eager bf16
# (LPIPS) than the stock path: Z-Image only until understood.
_SWIGLU_ALL_LAYOUTS = False


def _is_zimage_ff(module: Any) -> bool:
    return _split_spec(module) == ("w1", "w3", "w2") and type(module).__module__.endswith(
        "transformer_z_image"
    )


def _swiglu_layout_allowed(module: Any) -> bool:
    return _SWIGLU_ALL_LAYOUTS or _is_zimage_ff(module)


def _get(module: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _swiglu_forward(self: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
    """SwiGLU MLP with plain int8 weights: one act quant, one GEMM over [gate | value], fused SiLU-product quant."""
    import torch

    rec = self.__dict__.get(_SWIGLU_ATTR)
    lead = hidden_states.shape[:-1]
    x2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    ok = (
        rec is not None
        and not args
        and not kwargs
        and x2d.shape[0] >= _MIN_ROWS
        and hidden_states.is_cuda
    )
    ok = ok and hidden_states.dtype == torch.bfloat16
    if ok:
        fused_in, parts, down_name, n, gate_col, value_col = rec
        ok = all(_get(self, nm) is mod for nm, mod in parts)
    if not ok:
        return type(self).forward(self, hidden_states, *args, **kwargs)
    down = _get(self, down_name)
    xq, xs = _act_quant(x2d, fused_in.weight)
    c = _int_mm(xq.reshape(-1, xq.shape[-1]), fused_in.weight)
    q, s = _OP_HANDLE.swiglu(
        c, xs.reshape(-1), fused_in.weight.scale.flatten(), fused_in.bias, gate_col, value_col, n
    )
    y = _linear_from_q(q, s, down.weight, down.bias, hidden_states.dtype)
    return y.reshape(*lead, y.shape[-1])


def _prepare_swiglu(module: Any) -> bool:
    """Attach the (off-tree) record the SwiGLU forward reads; False keeps the stock forward."""
    from torch import nn

    if not _swiglu_layout_allowed(module):
        return False
    spec = _swiglu_spec(module)
    if spec is not None:
        in_name, _unused, down_name, n, gate_col, value_col = spec
        lin, down = _get(module, in_name), _get(module, down_name)
        if type(lin) is not nn.Linear or type(down) is not nn.Linear:
            return False
        if not (_plain_int8_weight(lin.weight) and _plain_int8_weight(down.weight)):
            return False
        module.__dict__[_SWIGLU_ATTR] = (
            lin,
            ((in_name, lin), (down_name, down)),
            down_name,
            n,
            gate_col,
            value_col,
        )
        return True
    split = _split_spec(module)
    if split is None:
        return False
    gate_name, value_name, down_name = split
    gate, value, down = _get(module, gate_name), _get(module, value_name), _get(module, down_name)
    if any(type(m) is not nn.Linear for m in (gate, value, down)):
        return False
    if not all(_plain_int8_weight(m.weight) for m in (gate, value, down)):
        return False
    from .diffusion_zimage_fused import _fuse_linears, _share_storage

    fused = _fuse_linears([gate, value])
    if fused is None or not _share_storage(fused, [gate, value]):
        return False
    n = gate.weight.shape[0]
    parts = ((gate_name, gate), (value_name, value), (down_name, down))
    module.__dict__[_SWIGLU_ATTR] = (fused, parts, down_name, n, 0, n)
    return True


def _exact_linear(*modules: Any) -> bool:
    """Plain nn.Linear only: a subclass (ConvRotLinear) may transform the input the fused _int_mm would skip."""
    from torch import nn
    return all(type(m) is nn.Linear for m in modules)


def _ff_eligible(module: Any) -> bool:
    try:
        from diffusers.models.activations import GELU
        from diffusers.models.attention import FeedForward
    except Exception:  # noqa: BLE001
        return False
    if type(module) is not FeedForward:
        return False
    net = getattr(module, "net", None)
    if net is None or len(net) < 3:
        return False
    act, drop, down = net[0], net[1], net[2]
    if type(act) is not GELU or getattr(act, "approximate", None) != "tanh":
        return False
    if type(drop).__name__ != "Dropout" or not _exact_linear(act.proj, down):
        return False
    if any(type(extra).__name__ != "Dropout" for extra in net[3:]):
        return False
    return _plain_int8_weight(act.proj.weight) and _plain_int8_weight(down.weight)


def _flux_single_eligible(module: Any) -> bool:
    if type(module).__name__ != "FluxSingleTransformerBlock":
        return False
    try:
        act = module.act_mlp
        if type(act).__name__ != "GELU" or getattr(act, "approximate", None) != "tanh":
            return False
        if getattr(module.attn, "heads", None) is None or not getattr(
            module.attn, "pre_only", False
        ):
            return False
        if not _exact_linear(module.proj_mlp, module.proj_out):
            return False
        return _plain_int8_weight(module.proj_mlp.weight) and _plain_int8_weight(
            module.proj_out.weight
        )
    except Exception:  # noqa: BLE001
        return False


def _flux_single_class_is_arch_patched(module: Any) -> bool:
    try:
        from . import diffusion_arch_patches as ap
        return getattr(type(module), "forward", None) is getattr(
            ap, "_flux_single_forward", object()
        )
    except Exception:  # noqa: BLE001
        return False


def resident_cuda_device(module: Any) -> Any:
    """The one CUDA device every parameter of ``module`` lives on (NVIDIA, not ROCm), else None."""
    try:
        import torch
        if getattr(torch.version, "hip", None):
            return None
        devices = {p.device for p in module.parameters()}
    except Exception:  # noqa: BLE001
        return None
    if len(devices) != 1:
        return None
    dev = next(iter(devices))
    return dev if dev.type == "cuda" else None


def run_on_first_call(module: Any, key: str, fn: Any) -> None:
    """Run ``fn(module)`` once, eagerly, right before ``module``'s first forward (after load-time placement moved
    the weights, before the regional compile traces the blocks and before any CUDA-graph capture). Idempotent per key."""
    hooks = module.__dict__.setdefault("_unsloth_first_call_hooks", {})
    if key in hooks:
        return

    def _pre_hook(mod, args, kwargs):
        handle = hooks.pop(key, None)
        if handle is not None:
            handle.remove()
        try:
            import torch

            # Studio renders under inference_mode, where detaching / viewing a torchao weight subclass raises
            # "Cannot set version_counter for inference tensor" and the fused SwiGLU / QKV silently stayed off.
            with torch.inference_mode(False), torch.no_grad():
                fn(mod)
        except Exception:  # noqa: BLE001 - an optimisation: the stock path stays
            pass
        return None

    hooks[key] = module.register_forward_pre_hook(_pre_hook, with_kwargs = True)


def cancel_first_call(module: Any, key: str) -> None:
    """Drop a ``run_on_first_call`` hook that has not fired yet (no-op if it has, or was never registered)."""
    hooks = getattr(module, "__dict__", {}).get("_unsloth_first_call_hooks")
    handle = hooks.pop(key, None) if hooks else None
    if handle is not None:
        handle.remove()


def install(
    transformer: Any,
    logger: Any = None,
    offload_active: bool = False,
) -> int:
    """Idempotent; returns the (candidate) count. Must run before the first compiled forward, which traces ``forward``."""
    if int8_fused_disabled() or transformer is None or offload_active:
        return 0
    if resident_cuda_device(transformer) is not None:
        return _finalize(transformer, logger)
    candidates = sum(
        1
        for m in transformer.modules()
        if _ff_eligible(m) or _flux_single_eligible(m) or _swiglu_candidate(m)
    )
    if candidates:
        run_on_first_call(transformer, "int8_fused", lambda t: _finalize(t, logger))
    return candidates


def _swiglu_candidate(module: Any) -> bool:
    return (
        _swiglu_spec(module) is not None or _split_spec(module) is not None
    ) and _swiglu_layout_allowed(module)


def _swiglu_eligible(module: Any) -> bool:
    """The checks ``_prepare_swiglu`` makes, without fusing anything."""
    from torch import nn

    if not _swiglu_layout_allowed(module):
        return False
    try:
        spec = _swiglu_spec(module)
        if spec is not None:
            parts = (_get(module, spec[0]), _get(module, spec[2]))
        else:
            split = _split_spec(module)
            if split is None:
                return False
            parts = tuple(_get(module, name) for name in split)
        return all(type(m) is nn.Linear and _plain_int8_weight(m.weight) for m in parts)
    except Exception:  # noqa: BLE001
        return False


def _has_eligible(transformer: Any) -> bool:
    return any(
        _MARK in m.__dict__ or _ff_eligible(m) or _swiglu_eligible(m) or _flux_single_eligible(m)
        for m in transformer.modules()
    )


def _finalize(transformer: Any, logger: Any = None) -> int:
    global _OP_HANDLE, _INTMM_MODULE
    dev = resident_cuda_device(transformer)
    if dev is None:
        return 0
    # Before the probe: it compiles and launches both Triton kernels, pure latency for a bf16 / fp16 model.
    if not _has_eligible(transformer):
        return 0
    intmm = _resolve_intmm()
    if (
        intmm is None
    ):  # an int8 GEMM home this file does not know: keep the stock path rather than fail the render
        return 0
    _INTMM_MODULE = intmm
    import torch

    if not _device_ok(dev.index if dev.index is not None else torch.cuda.current_device()):
        return 0
    count = 0
    rearm = False
    with _LOCK:
        _OP_HANDLE = _op()
        for _name, module in transformer.named_modules():
            if _MARK in module.__dict__:
                count += 1
                continue
            if _ff_eligible(module):
                fn = _ff_forward
            elif _prepare_swiglu(module):
                fn = _swiglu_forward
            elif _flux_single_eligible(module):
                fn = _flux_single_forward
                module._unsloth_i8_addcmul = _flux_single_class_is_arch_patched(module)
            else:
                continue
            bound = types.MethodType(fn, module)
            slot = _hooked_forward_slot(module)
            if slot is not None:
                # A step cache (FBCache / MagCache) engaged before the speed layer: its hook owns the instance forward,
                # so the swap goes under it. Overwriting the wrapper would drop the skip logic and the tail residuals.
                ref, attr = slot
                stock_inner = _drop_compiled_inner(module)
                module.__dict__[_MARK] = (
                    stock_inner if stock_inner is not None else getattr(ref, attr)
                )
                setattr(ref, attr, bound)
                rearm = rearm or stock_inner is not None
            else:
                module.__dict__[_MARK] = module.__dict__.get("forward", _NO_PREV)
                module.forward = bound
            count += 1
    if rearm:
        try:
            from .diffusion_cache import _compile_hooked_block_inners
            _compile_hooked_block_inners(transformer, logger)
        except Exception:  # noqa: BLE001 - the fused inner still runs, eager
            pass
    if logger is not None and count:
        logger.info(
            "diffusion.int8_fused: %d int8 MLP region(s) run the fused dequant/activation/quant kernel",
            count,
        )
    return count


def _hooked_forward_slot(module: Any) -> Optional[tuple]:
    """(fn_ref, attribute) holding the module's own forward in the innermost ``HookFunctionReference``, else None."""
    registry = getattr(module, "_diffusers_hook", None)
    fn_refs = getattr(registry, "_fn_refs", None)
    if not fn_refs or "forward" not in getattr(module, "__dict__", {}):
        return None
    ref = fn_refs[0]
    if getattr(ref, "original_forward", None) is not None:
        return ref, "original_forward"
    if getattr(ref, "forward", None) is not None:
        return ref, "forward"
    return None


def _is_fused_forward(fn: Any, module: Any) -> bool:
    return getattr(fn, "__self__", None) is module and getattr(fn, "__func__", None) in (
        _ff_forward,
        _swiglu_forward,
        _flux_single_forward,
    )


def _class_forward(module: Any) -> Any:
    return types.MethodType(type(module).forward, module)


def _cache_hooks(module: Any) -> list:
    try:
        from .diffusion_cache import _CACHE_HOOK_NAMES
    except Exception:  # noqa: BLE001
        return []
    hooks = getattr(getattr(module, "_diffusers_hook", None), "hooks", None) or {}
    return [hooks[name] for name in _CACHE_HOOK_NAMES if name in hooks]


def _drop_compiled_inner(module: Any) -> Any:
    """Drop a cache hook's compiled wrapper of the STOCK inner forward (deferred install); return that inner, else None."""
    stock = None
    for hook in _cache_hooks(module):
        inner = getattr(hook, "_unsloth_orig_inner", None)
        if inner is not None:
            hook._unsloth_orig_inner = None
            stock = inner
    return stock


def _disarm_fused_inner(module: Any, prev: Any) -> None:
    """Uninstall under an armed cache hook: its compiled inner wraps the fused forward, point it back at stock."""
    for hook in _cache_hooks(module):
        inner = getattr(hook, "_unsloth_orig_inner", None)
        if inner is not None and _is_fused_forward(inner, module):
            stock = prev if prev is not _NO_PREV else _class_forward(module)
            hook._unsloth_orig_inner = None
            try:
                hook.fn_ref.original_forward = stock
            except Exception:  # noqa: BLE001
                pass


def uninstall(transformer: Any = None) -> None:
    """Restore the stock forwards under ``transformer`` (a dropped transformer needs nothing: the patch is per instance)."""
    if transformer is None:
        return
    # A deferred install still pending (weights not yet on the GPU) must not fire at the next forward.
    cancel_first_call(transformer, "int8_fused")
    with _LOCK:
        for module in transformer.modules():
            prev = module.__dict__.pop(_MARK, None)
            if prev is None:
                continue
            slot = _hooked_forward_slot(module)
            if slot is not None and _is_fused_forward(getattr(*slot), module):
                # Swapped under a hook, or a cache engaged after the swap: restore the hook's inner, keep its wrapper.
                setattr(*slot, prev if prev is not _NO_PREV else _class_forward(module))
            elif _is_fused_forward(module.__dict__.get("forward"), module):
                if prev is _NO_PREV:
                    module.__dict__.pop("forward", None)
                else:
                    module.forward = prev
            _disarm_fused_inner(module, prev)
            module.__dict__.pop("_unsloth_i8_addcmul", None)
            module.__dict__.pop(_SWIGLU_ATTR, None)


def is_installed(module: Any) -> bool:
    return _MARK in getattr(module, "__dict__", {})
