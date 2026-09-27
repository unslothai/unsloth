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

_LOCK = threading.Lock()
# The resolved op, read by the traced forwards (dynamo must not trace into the lru_cache'd registration).
_OP_HANDLE: Any = None
_INSTALLED: dict = {}  # id(module) -> (module, previous instance forward or None)


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

    return types.SimpleNamespace(dq_gelu_quant = dq_gelu_quant, triton = triton)


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


@lru_cache(maxsize = 1)
def _op() -> Any:
    """The torch.library op (opaque to dynamo, CUDA-graph safe: no host sync, allocations only), or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    qualname = f"{_OP_NAMESPACE}::{_OP_NAME}"
    existing = getattr(getattr(torch.ops, _OP_NAMESPACE, None), _OP_NAME, None)
    if existing is not None:
        return existing
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
    except Exception:  # noqa: BLE001 - a registration failure keeps the stock path
        return None
    return getattr(getattr(torch.ops, _OP_NAMESPACE), _OP_NAME)


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
        return bool(torch.equal(q, q_ref) and torch.equal(s, s_ref))
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
    q, s = _OP_HANDLE(c, xs.reshape(-1), proj.weight.scale.flatten(), proj.bias, None)
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
    q, s = _OP_HANDLE(c, xs.reshape(-1), self.proj_mlp.weight.scale.flatten(), self.proj_mlp.bias, prefix)
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


def install(transformer: Any, logger: Any = None) -> int:
    """Point every eligible block of ``transformer`` at the fused forward. Idempotent; returns the count.
    Must run before the first compiled forward (the regional compile traces whatever ``forward`` is then)."""
    if int8_fused_disabled() or transformer is None:
        return 0
    try:
        import torch
        dev = next((p.device for p in transformer.parameters() if p.is_cuda), None)
    except Exception:  # noqa: BLE001
        return 0
    if dev is None or not _device_ok(dev.index if dev.index is not None else torch.cuda.current_device()):
        return 0
    global _OP_HANDLE
    count = 0
    with _LOCK:
        _OP_HANDLE = _op()
        for _name, module in transformer.named_modules():
            if id(module) in _INSTALLED:
                count += 1
                continue
            if _ff_eligible(module):
                fn = _ff_forward
            elif _flux_single_eligible(module):
                fn = _flux_single_forward
                module._unsloth_i8_addcmul = _flux_single_class_is_arch_patched(module)
            else:
                continue
            _INSTALLED[id(module)] = (module, module.__dict__.get("forward"))
            module.forward = types.MethodType(fn, module)
            count += 1
    if logger is not None and count:
        logger.info("diffusion.int8_fused: %d int8 GELU MLP region(s) run the fused dequant/GELU/quant kernel", count)
    return count


def uninstall(transformer: Any = None) -> None:
    with _LOCK:
        for key, (module, previous) in list(_INSTALLED.items()):
            if transformer is not None and not any(m is module for m in transformer.modules()):
                continue
            if previous is None:
                module.__dict__.pop("forward", None)
            else:
                module.forward = previous
            module.__dict__.pop("_unsloth_i8_addcmul", None)
            _INSTALLED.pop(key, None)


def is_installed(module: Any) -> bool:
    return id(module) in _INSTALLED
