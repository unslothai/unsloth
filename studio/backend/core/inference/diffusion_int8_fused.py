# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused int8 activation path between two torchao int8 GEMMs of a DiT feed-forward.

With torchao's dynamic int8 (``Int8Tensor``, per-row symmetric activations), a DiT MLP runs
``_int_mm`` (int32 out) -> dequant -> GELU(tanh) -> per-row amax -> quantize -> ``_int_mm``. Inductor
fuses the middle into one looped reduction that writes the GELU output (bf16 or fp32) to HBM and
reads it back, about 2x the bytes of a single pass. This module replaces that middle with one
Triton kernel that reads the int32 GEMM output, rebuilds each value in registers (twice: once for
the row max of the pre-activation, once to quantize, the second read mostly hitting L2) and writes int8 + the fp32 row
scale that the next ``_int_mm`` consumes.

NUMERICS: bit-identical to EAGER torchao + ATen (every bf16 rounding of the stock chain is kept,
fp contraction is off so ``y * w_scale + bias`` rounds twice like the two ATen kernels do, the row
scale is ``bf16(amax / 127.5)`` clamped at fp32 eps and the quantizer multiplies by the correctly
rounded reciprocal). The compiled stock path is NOT eager-exact (Inductor keeps fp32 chains), so
this is at least as close to eager as what it replaces. GELU(tanh) is evaluated as ``y * sigmoid(2u)``
(same function, ~1e-7 relative), which agreed with ATen on every one of 5e7 test elements after
the bf16 rounding.

Covered: ``diffusers.models.attention.FeedForward`` with ``GELU(approximate="tanh")`` (FLUX.1 double
blocks, Qwen-Image, Wan) and ``FluxSingleTransformerBlock`` (attention output concatenated in
front of the GELU branch before ``proj_out``). Anything else, or any weight that is not a plain
dynamic symmetric per-row ``Int8Tensor``, keeps the stock path. CUDA + Triton only: ROCm, CPU,
MPS, Windows without the MSVC CRT headers, a Triton older than 3.2 or a failing kernel build leave
the model untouched. Kill switch: ``UNSLOTH_DIFFUSION_INT8_FUSED=0``.
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
# The resolved op, read by the traced forwards (dynamo must not trace into the lru_cache'd registration).
_OP_HANDLE: Any = None
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
        # 0.5 * y * (1 + tanh(u)) == y * sigmoid(2u) == y / (1 + exp(-2u)), u = sqrt(2/pi) * (y + 0.044715 y^3)
        u = 0.7978845608028654 * (y + 0.044715 * (y * y * y))
        return y / (1.0 + tl.exp2(u * -2.8853900817779268))

    @triton.jit
    def _pre(c_ptr, xs, ws_ptr, b_ptr, offs, mask, HAS_BIAS: tl.constexpr, WS_FP32: tl.constexpr, EVICT: tl.constexpr,
             FINAL_ROUND: tl.constexpr):
        # torchao Int8Tensor linear epilogue: (int32 * x_scale).to(bf16) * w_scale (+ bias), then .to(bf16).
        # FINAL_ROUND=False leaves the last rounding to the caller (it commutes with a row max).
        c = tl.load(c_ptr + offs, mask = mask, other = 0, eviction_policy = EVICT).to(tl.float32)
        y = _rbf16(c * xs)
        y = y * tl.load(ws_ptr + offs, mask = mask, other = 0.0, eviction_policy = "evict_last").to(tl.float32)
        if not WS_FP32:
            y = _rbf16(y)
        if HAS_BIAS:
            y = y + tl.load(b_ptr + offs, mask = mask, other = 0.0, eviction_policy = "evict_last").to(tl.float32)
        if FINAL_ROUND:
            y = _rbf16(y)
        return y

    @triton.jit
    def dq_gelu_quant(
        c_ptr, xs_ptr, ws_ptr, b_ptr, p_ptr, q_ptr, s_ptr,
        N, NA, S, stride_c, stride_q, p_sb, p_ss, p_sh,
        HAS_BIAS: tl.constexpr, WS_FP32: tl.constexpr, HAS_PREFIX: tl.constexpr, HEAD_DIM: tl.constexpr,
        CHUNK: tl.constexpr, CHUNK_A: tl.constexpr,
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
            y = _pre(crow, xs, ws_ptr, b_ptr, offs, m, HAS_BIAS, WS_FP32, "evict_last", False)
            vmax = tl.maximum(vmax, tl.where(m, y, float("-inf")))
        # bf16 rounding is monotone, so max(round(v)) == round(max(v)): round once, after the reduction.
        amax = _rbf16(_gelu_tanh(_rbf16(tl.max(vmax, axis = 0))))
        if amax < 0.171875:
            acc = tl.zeros((CHUNK,), dtype = tl.float32)
            for k in range(0, N, CHUNK):
                offs = k + base
                m = offs < N
                g = _rbf16(_gelu_tanh(_pre(crow, xs, ws_ptr, b_ptr, offs, m, HAS_BIAS, WS_FP32, "evict_last", True)))
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
                v = tl.load(pbase + (j // HEAD_DIM) * p_sh + (j % HEAD_DIM), mask = m, other = 0.0).to(tl.float32)
                pacc = tl.maximum(pacc, tl.abs(v))
            amax = tl.maximum(amax, tl.max(pacc, axis = 0))
        scale = _rbf16(tl.math.div_rn(amax, 127.5))
        scale = tl.maximum(scale, 1.1920928955078125e-07)
        inv = tl.math.div_rn(1.0, scale)
        tl.store(s_ptr + row, scale)
        if HAS_PREFIX:
            for k in range(0, NA, CHUNK_A):
                j = k + abase
                m = j < NA
                v = tl.load(pbase + (j // HEAD_DIM) * p_sh + (j % HEAD_DIM), mask = m, other = 0.0).to(tl.float32)
                qi = tl.extra.cuda.libdevice.nearbyint(v * inv)
                qi = tl.minimum(tl.maximum(qi, -128.0), 127.0)
                tl.store(qrow + j, qi.to(tl.int8), mask = m)
        # pass 2: rebuild (the int32 row is mostly still in L2) and quantize.
        for k in range(0, N, CHUNK):
            offs = k + base
            m = offs < N
            g = _rbf16(_gelu_tanh(_pre(crow, xs, ws_ptr, b_ptr, offs, m, HAS_BIAS, WS_FP32, "evict_first", True)))
            qi = tl.extra.cuda.libdevice.nearbyint(g * inv)
            qi = tl.minimum(tl.maximum(qi, -128.0), 127.0)
            tl.store(qrow + NA + offs, qi.to(tl.int8), mask = m)

    @triton.jit
    def _silu(y):
        # ATen: x / (1 + exp(-x)) in fp32
        return y / (1.0 + tl.exp(-y))

    @triton.jit
    def _swiglu_val(crow, xs, ws_ptr, b_ptr, g0, v0, offs, m, HAS_BIAS: tl.constexpr, WS_FP32: tl.constexpr,
                    EVICT: tl.constexpr):
        g = _pre(crow + g0, xs, ws_ptr + g0, b_ptr + g0, offs, m, HAS_BIAS, WS_FP32, EVICT, True)
        v = _pre(crow + v0, xs, ws_ptr + v0, b_ptr + v0, offs, m, HAS_BIAS, WS_FP32, EVICT, True)
        return _rbf16(_rbf16(_silu(g)) * v)

    @triton.jit
    def dq_swiglu_quant(
        c_ptr, xs_ptr, ws_ptr, b_ptr, h_ptr, q_ptr, s_ptr, N, G0, V0, stride_c, stride_q,
        HAS_BIAS: tl.constexpr, WS_FP32: tl.constexpr, CHUNK: tl.constexpr,
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
            h = _swiglu_val(crow, xs, ws_ptr, b_ptr, G0, V0, offs, m, HAS_BIAS, WS_FP32, "evict_first")
            tl.store(hrow + offs, h.to(tl.bfloat16), mask = m, eviction_policy = "evict_last")
            acc = tl.maximum(acc, tl.where(m, tl.abs(h), 0.0))
        amax = tl.max(acc, axis = 0)
        scale = _rbf16(tl.math.div_rn(amax, 127.5))
        scale = tl.maximum(scale, 1.1920928955078125e-07)
        inv = tl.math.div_rn(1.0, scale)
        tl.store(s_ptr + row, scale)
        qrow = q_ptr + row * stride_q
        for k in range(0, N, CHUNK):
            offs = k + base
            m = offs < N
            h = tl.load(hrow + offs, mask = m, other = 0.0, eviction_policy = "evict_first").to(tl.float32)
            qi = tl.extra.cuda.libdevice.nearbyint(h * inv)
            qi = tl.minimum(tl.maximum(qi, -128.0), 127.0)
            tl.store(qrow + offs, qi.to(tl.int8), mask = m)

    return types.SimpleNamespace(dq_gelu_quant = dq_gelu_quant, dq_swiglu_quant = dq_swiglu_quant, triton = triton)


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
    scale = torch.empty((m_rows,), device = c.device, dtype = torch.float32)
    with torch.cuda.device(c.device):
        k.dq_gelu_quant[(m_rows,)](
            c, xs, ws, bias if bias is not None else ws, prefix if prefix is not None else ws, q, scale,
            n, na, s, c.stride(0), q.stride(0), p_sb, p_ss, p_sh,
            HAS_BIAS = bias is not None,
            WS_FP32 = ws.dtype == torch.float32,
            HAS_PREFIX = prefix is not None,
            HEAD_DIM = d,
            CHUNK = 2048,
            CHUNK_A = 1024,
            num_warps = 8,
            enable_fp_fusion = False,
        )
    return q, scale


def _launch_swiglu(c: Any, xs: Any, ws: Any, bias: Any, gate_col: int, value_col: int, n: int) -> tuple:
    """SwiGLU on the int32 output ``c`` of one fused GEMM: gate columns [gate_col, +n), value columns [value_col, +n)."""
    import torch

    k = _kernels()
    m_rows = c.shape[0]
    q = torch.empty((m_rows, n), device = c.device, dtype = torch.int8)
    scale = torch.empty((m_rows,), device = c.device, dtype = torch.float32)
    scratch = torch.empty((m_rows, n), device = c.device, dtype = torch.bfloat16)
    with torch.cuda.device(c.device):
        k.dq_swiglu_quant[(m_rows,)](
            c, xs, ws, bias if bias is not None else ws, scratch, q, scale, n, gate_col, value_col, c.stride(0), q.stride(0),
            HAS_BIAS = bias is not None,
            WS_FP32 = ws.dtype == torch.float32,
            CHUNK = 2048,
            num_warps = 8,
            enable_fp_fusion = False,
        )
    return q, scale


def reference_dq_swiglu_quant(c: Any, xs: Any, ws: Any, bias: Any, gate_col: int, value_col: int, n: int) -> tuple:
    """Eager mirror: two torchao epilogues (gate, value), ATen SiLU, bf16 product, torchao per-row act quant."""
    import torch
    import torch.nn.functional as F

    def part(col):
        y = (c[:, col:col + n] * xs.reshape(-1, 1)).to(torch.bfloat16) * ws[col:col + n]
        if bias is not None:
            y = y + bias[col:col + n]
        return y.to(torch.bfloat16)

    h = F.silu(part(gate_col)) * part(value_col)
    amax = torch.maximum(-h.amin(dim = 1).clamp(max = 0), h.amax(dim = 1).clamp(min = 0))
    scale = (amax / 127.5).clamp(min = torch.finfo(torch.float32).eps).to(torch.float32)
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
        return types.SimpleNamespace(gelu = getattr(ns, _OP_NAME), swiglu = getattr(ns, _OP_NAME_SWIGLU))
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
                c.new_empty((c.shape[0],), dtype = torch.float32),
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
                c.new_empty((c.shape[0],), dtype = torch.float32),
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
    if not (_triton_version_ok() and _triton_jit_toolchain_ok()) or _kernels() is None or _op() is None:
        return False
    try:
        dev = torch.device("cuda", index)
        g = torch.Generator(device = "cpu").manual_seed(0)
        c = torch.randint(-(2**20), 2**20, (33, 200), generator = g, dtype = torch.int32).to(dev)
        xs = (torch.rand(33, generator = g) * 1e-3 + 1e-5).to(torch.bfloat16).float().to(dev)
        ws = (torch.rand(200, generator = g) * 1e-4 + 1e-6).to(dev)
        bias = (torch.randn(200, generator = g) * 0.1).to(torch.bfloat16).to(dev)
        q, s = _launch(c, xs, ws, bias, None)
        q_ref, s_ref = reference_dq_gelu_quant(c, xs, ws, bias, None)
        ok = bool(torch.equal(q, q_ref) and torch.equal(s, s_ref))
        q, s = _launch_swiglu(c, xs, ws, bias, 104, 0, 96)
        q_ref, s_ref = reference_dq_swiglu_quant(c, xs, ws, bias, 104, 0, 96)
        return ok and bool(torch.equal(q, q_ref) and torch.equal(s, s_ref))
    except Exception:  # noqa: BLE001 - any build / launch failure keeps the stock path
        return False


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
        g = torch.cat([prefix.reshape(prefix.shape[0] * prefix.shape[1], -1).to(g.dtype), g], dim = -1)
    amax = torch.maximum(-g.amin(dim = 1).clamp(max = 0), g.amax(dim = 1).clamp(min = 0))
    scale = (amax / 127.5).clamp(min = torch.finfo(torch.float32).eps).to(torch.float32)
    q = torch.clamp(torch.round(g * (1.0 / scale).reshape(-1, 1)), -128, 127).to(torch.int8)
    return q, scale


def _plain_int8_weight(w: Any) -> bool:
    """A torchao Int8Tensor with dynamic, symmetric, per-row activation quant and nothing else attached."""
    try:
        if type(w).__name__ != "Int8Tensor":
            return False
        kw = getattr(w, "act_quant_kwargs", None)
        if kw is None or getattr(w, "act_pre_scale", None) is not None:
            return False
        if getattr(w, "act_quant_scale", None) is not None or getattr(w, "act_quant_zero_point", None) is not None:
            return False
        if getattr(kw, "reduce_range", False) or getattr(w, "reduce_range", False):
            return False
        if "SYMMETRIC" not in str(getattr(kw, "mapping_type", "")) or "ASYMMETRIC" in str(getattr(kw, "mapping_type", "")):
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

    act = _choose_quant_func_and_quantize_tensor(
        x2d,
        weight.act_quant_kwargs,
        scale = weight.act_quant_scale,
        zero_point = weight.act_quant_zero_point,
    )
    return act.qdata, act.scale


def _int_mm(a: Any, weight: Any) -> Any:
    from torchao.kernel.intmm import safe_int_mm

    return safe_int_mm(a, weight.qdata.contiguous().t())


def _linear_from_q(q: Any, xs: Any, weight: Any, bias: Any, out_dtype: Any) -> Any:
    """torchao Int8Tensor linear on an already-quantized activation: same ops, same order, as the stock epilogue."""
    import torch

    y = (_int_mm(q, weight) * xs.reshape(-1, 1).to(torch.float32)).to(out_dtype)
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
    if args or kwargs or x2d.shape[0] < _MIN_ROWS or not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
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
        return type(self).forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim = 1)
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb = temb)
    x2d = norm_hidden_states.reshape(rows, -1)
    xq, xs = _act_quant(x2d, self.proj_mlp.weight)
    c = _int_mm(xq.reshape(rows, -1), self.proj_mlp.weight)
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(hidden_states = norm_hidden_states, image_rotary_emb = image_rotary_emb, **joint_attention_kwargs)
    heads = self.attn.heads
    prefix = attn_output.unflatten(-1, (heads, attn_output.shape[-1] // heads))
    if prefix.stride(-1) != 1:
        prefix = prefix.contiguous()
    q, s = _OP_HANDLE.gelu(c, xs.reshape(-1), self.proj_mlp.weight.scale.flatten(), self.proj_mlp.bias, prefix)
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
            # SwiGLU.forward: hidden, gate = proj(x).chunk(2) -> value first, gate second (MiniMax-H3)
            return ("net.0.proj", None, "net.2", n, n, 0)
        if name == "Flux2FeedForward" and type(getattr(module, "act_fn", None)).__name__ == "Flux2SwiGLU":
            n = module.linear_in.out_features // 2
            return ("linear_in", None, "linear_out", n, 0, n)  # gate first, value second (FLUX.2)
    except Exception:  # noqa: BLE001
        return None
    return None


def _split_spec(module: Any) -> Optional[tuple]:
    """Two separate projections (gate, value) + down: Z-Image FeedForward (w1, w3, w2), Qwen-Image-2.1 SwiGLU."""
    name = type(module).__name__
    mod = type(module).__module__
    if name == "FeedForward" and mod.endswith("transformer_z_image") and all(hasattr(module, n) for n in ("w1", "w2", "w3")):
        return ("w1", "w3", "w2")
    if name == "QwenImage21SwiGLUFeedForward":
        return ("gate_layer", "proj", "out")
    return None


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
    ok = rec is not None and not args and not kwargs and x2d.shape[0] >= _MIN_ROWS and hidden_states.is_cuda
    ok = ok and hidden_states.dtype == torch.bfloat16
    if ok:
        fused_in, parts, down_name, n, gate_col, value_col = rec
        ok = all(_get(self, nm) is mod for nm, mod in parts)
    if not ok:
        return type(self).forward(self, hidden_states, *args, **kwargs)
    down = _get(self, down_name)
    xq, xs = _act_quant(x2d, fused_in.weight)
    c = _int_mm(xq.reshape(-1, xq.shape[-1]), fused_in.weight)
    q, s = _OP_HANDLE.swiglu(c, xs.reshape(-1), fused_in.weight.scale.flatten(), fused_in.bias, gate_col, value_col, n)
    y = _linear_from_q(q, s, down.weight, down.bias, hidden_states.dtype)
    return y.reshape(*lead, y.shape[-1])


def _prepare_swiglu(module: Any) -> bool:
    """Attach the (off-tree) record the SwiGLU forward reads; False keeps the stock forward."""
    from torch import nn

    spec = _swiglu_spec(module)
    if spec is not None:
        in_name, _unused, down_name, n, gate_col, value_col = spec
        lin, down = _get(module, in_name), _get(module, down_name)
        if type(lin) is not nn.Linear or type(down) is not nn.Linear:
            return False
        if not (_plain_int8_weight(lin.weight) and _plain_int8_weight(down.weight)):
            return False
        module.__dict__[_SWIGLU_ATTR] = (lin, ((in_name, lin), (down_name, down)), down_name, n, gate_col, value_col)
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
    if type(drop).__name__ != "Dropout" or type(down).__name__ != "Linear":
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
        if getattr(module.attn, "heads", None) is None or not getattr(module.attn, "pre_only", False):
            return False
        return _plain_int8_weight(module.proj_mlp.weight) and _plain_int8_weight(module.proj_out.weight)
    except Exception:  # noqa: BLE001
        return False


def _flux_single_class_is_arch_patched(module: Any) -> bool:
    try:
        from . import diffusion_arch_patches as ap
        return getattr(type(module), "forward", None) is getattr(ap, "_flux_single_forward", object())
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
            fn(mod)
        except Exception:  # noqa: BLE001 - an optimisation: the stock path stays
            pass
        return None

    hooks[key] = module.register_forward_pre_hook(_pre_hook, with_kwargs = True)


def install(transformer: Any, logger: Any = None, offload_active: bool = False) -> int:
    """Point every eligible block of ``transformer`` at the fused forward. Idempotent; returns the count (or the
    count of candidates when the weights are not on the GPU yet: the swap then happens at the first forward).
    Must run before the first compiled forward (the regional compile traces whatever ``forward`` is then)."""
    if int8_fused_disabled() or transformer is None or offload_active:
        return 0
    if resident_cuda_device(transformer) is not None:
        return _finalize(transformer, logger)
    candidates = sum(1 for m in transformer.modules() if _ff_eligible(m) or _flux_single_eligible(m) or _swiglu_candidate(m))
    if candidates:
        run_on_first_call(transformer, "int8_fused", lambda t: _finalize(t, logger))
    return candidates


def _swiglu_candidate(module: Any) -> bool:
    return _swiglu_spec(module) is not None or _split_spec(module) is not None


def _finalize(transformer: Any, logger: Any = None) -> int:
    global _OP_HANDLE
    dev = resident_cuda_device(transformer)
    if dev is None:
        return 0
    import torch

    if not _device_ok(dev.index if dev.index is not None else torch.cuda.current_device()):
        return 0
    count = 0
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
            module.__dict__[_MARK] = module.__dict__.get("forward", _NO_PREV)
            module.forward = types.MethodType(fn, module)
            count += 1
    if logger is not None and count:
        logger.info("diffusion.int8_fused: %d int8 MLP region(s) run the fused dequant/activation/quant kernel", count)
    return count


def uninstall(transformer: Any = None) -> None:
    """Restore the stock forwards under ``transformer`` (a dropped transformer needs nothing: the patch is per instance)."""
    if transformer is None:
        return
    with _LOCK:
        for module in transformer.modules():
            prev = module.__dict__.pop(_MARK, None)
            if prev is None:
                continue
            if prev is _NO_PREV:
                module.__dict__.pop("forward", None)
            else:
                module.forward = prev
            module.__dict__.pop("_unsloth_i8_addcmul", None)
            module.__dict__.pop(_SWIGLU_ATTR, None)


def is_installed(module: Any) -> bool:
    return _MARK in getattr(module, "__dict__", {})
