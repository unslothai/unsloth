# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eager Qwen-Image-2.1 renders before the deferred profile: fused q/k RMSNorm + RoPE (fp32 sum of squares may
differ in the last bit) and bit-identical broadcast modulation. Kill switch UNSLOTH_DIFFUSION_Q21_EAGER_FAST=0"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any, Optional

from .diffusion_autoopt.patches import source_digest
from .rocm_bf16 import is_rocm_torch

EAGER_FAST_ENV = "UNSLOTH_DIFFUSION_Q21_EAGER_FAST"

_CLASS = "QwenImage21Transformer2DModel"
# diffusers 0.41.0 / 0.42.1; the prepare copy below uses 0.42.1's head split, equal for both.
_FINGERPRINTS: dict[str, frozenset] = {
    "_qwenimage21_prepare_qkv": frozenset({"574bacc8f355456d", "4c3830595aef56a0"}),
    "RMSNorm.forward": frozenset({"2633294611ed3bdf"}),
    "_select_modulation_rows": frozenset({"7e12608919321d40"}),
    "QwenImage21TransformerBlock.forward": frozenset({"f2cdd58b41ad3441"}),
    "QwenImage21TransformerBlock._modulate": frozenset({"3bf1998487634f22"}),
    "QwenImage21AttnProcessor.__call__": frozenset({"2125e0b56c1490b6"}),
}

# EAGER: every eager render of this load. BG: only renders a pending background compile keeps eager.
_ATTR = "_unsloth_q21_eager_fast"
EAGER = "eager"
BG = "bg"

_LOCK = threading.Lock()
_STOCK: dict = {}
_FAILED = [False]
_FUSION: dict = {}
STATS = {"qk_fused": 0, "mod_broadcast": 0}


def disabled() -> bool:
    return (os.environ.get(EAGER_FAST_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def active(module: Any) -> bool:
    import torch

    state = module.__dict__.get(_ATTR)
    if state is None or torch.compiler.is_compiling() or torch.is_grad_enabled():
        return False
    if state == EAGER:
        return True
    from .diffusion_bg_compile import eager_forced

    return eager_forced()


def _modules(transformer: Any) -> list:
    out = [transformer]
    for block in getattr(transformer, "transformer_blocks", ()) or ():
        attn = getattr(block, "attn", None)
        if attn is not None:
            out.append(attn)
    return out


def set_state(transformer: Any, state: Optional[str]) -> None:
    for module in _modules(transformer):
        if state is None:
            module.__dict__.pop(_ATTR, None)
        else:
            module.__dict__[_ATTR] = state


@functools.lru_cache(maxsize = 1)
def _kernel() -> Any:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice
    except Exception:  # noqa: BLE001 - older Triton layout
        from triton.language.extra.cuda import libdevice

    # Module globals, which the JIT resolves. *_rn: Triton would otherwise FMA-contract the products eager rounds.
    global _rsqrt, _mul, _add
    _rsqrt, _mul, _add = libdevice.rsqrt, libdevice.mul_rn, libdevice.add_rn

    @triton.jit
    def _norm_rope_kernel(
        x_ptr,
        w_ptr,
        f_ptr,
        out_ptr,
        n_rows,
        n_heads,
        n_seq,
        stride_b,
        stride_s,
        stride_h,
        f_stride,
        eps,
        HALF: tl.constexpr,
        ROWS: tl.constexpr,
        REAL_FMA_X: tl.constexpr,
        IMAG_FMA_X: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        rows = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
        ok = rows < n_rows
        h = rows % n_heads
        bs = rows // n_heads
        s = bs % n_seq
        b = bs // n_seq
        base = b * stride_b + s * stride_s + h * stride_h
        i = tl.arange(0, HALF)
        m = ok[:, None]
        xe = tl.load(x_ptr + base[:, None] + 2 * i[None, :], mask = m, other = 0.0)
        xo = tl.load(x_ptr + base[:, None] + 2 * i[None, :] + 1, mask = m, other = 0.0)
        dt = xe.dtype
        xe = xe.to(tl.float32)
        xo = xo.to(tl.float32)
        # diffusers RMSNorm: x * rsqrt in fp32, round, then * weight, round
        ms = (tl.sum(_mul(xe, xe), axis = 1) + tl.sum(_mul(xo, xo), axis = 1)) * (0.5 / HALF)
        r = _rsqrt(_add(ms, eps))[:, None]
        we = tl.load(w_ptr + 2 * i).to(tl.float32)[None, :]
        wo = tl.load(w_ptr + 2 * i + 1).to(tl.float32)[None, :]
        a = _mul(_mul(xe, r).to(dt).to(tl.float32), we).to(dt).to(tl.float32)
        bb = _mul(_mul(xo, r).to(dt).to(tl.float32), wo).to(dt).to(tl.float32)
        # the fma pairing this card's complex multiply uses
        fo = s[:, None] * f_stride + 2 * i[None, :]
        c = tl.load(f_ptr + fo, mask = m, other = 0.0)
        d = tl.load(f_ptr + fo + 1, mask = m, other = 0.0)
        if REAL_FMA_X:
            real = tl.fma(a, c, -_mul(bb, d))
        else:
            real = tl.fma(-bb, d, _mul(a, c))
        if IMAG_FMA_X:
            imag = tl.fma(bb, c, _mul(a, d))
        else:
            imag = tl.fma(a, d, _mul(bb, c))
        o = rows[:, None] * (2 * HALF) + 2 * i[None, :]
        tl.store(out_ptr + o, real.to(dt), mask = m)
        tl.store(out_ptr + o + 1, imag.to(dt), mask = m)

    return _norm_rope_kernel


_ROWS = 16
_WARPS = 4


def _fusion(device: Any) -> Optional[tuple]:
    index = device.index
    if index not in _FUSION:
        from .diffusion_qwenimage21_rope import probe_fusion
        try:
            _FUSION[index] = probe_fusion(device)
        except Exception:  # noqa: BLE001 - unknown pairing: keep the stock RoPE
            _FUSION[index] = None
    return _FUSION[index]


def _eligible(x: Any, norm: Any, freqs: Any) -> bool:
    import torch

    if (
        _FAILED[0]
        or not x.is_cuda
        or x.dim() != 4
        or x.dtype not in (torch.bfloat16, torch.float16)
    ):
        return False
    dim = x.shape[-1]
    weight = getattr(norm, "weight", None)
    if (
        dim not in (64, 128, 256)
        or x.stride(-1) != 1
        or x.stride(-2) != dim
        or not _stock_norm(norm)
        or getattr(norm, "bias", None) is not None
        or weight is None
        or weight.dtype != x.dtype
        or weight.device != x.device
        or tuple(weight.shape) != (dim,)
    ):
        return False
    return (
        torch.is_tensor(freqs)
        and freqs.dtype == torch.complex64
        and freqs.device == x.device
        and freqs.dim() == 2
        and freqs.shape[0] == x.shape[1]
        and freqs.shape[1] * 2 == dim
        and freqs.stride(1) == 1
    )


def _stock_norm(norm: Any) -> bool:
    """Stock forward or the eager tier's RMSNorm (installed on render 3); they differ only in rounding x * rsqrt."""
    import inspect

    forward = getattr(type(norm), "forward", None)
    want = _STOCK.get("rmsnorm_forward")
    if want is None or forward is None:
        return False
    if forward is want:
        return True
    from .diffusion_eager_patches import _rmsnorm_forward

    return getattr(inspect.unwrap(forward), "__code__", None) is _rmsnorm_forward.__code__


def _norm_rope(x: Any, norm: Any, freqs: Any) -> Optional[Any]:
    """``apply_rotary_emb_qwen(norm(x), freqs, use_real=False)`` in one kernel, or None to run the stock pair."""
    import torch

    if not _eligible(x, norm, freqs):
        return None
    fusion = _fusion(x.device)
    if fusion is None:
        return None
    batch, seq, heads, dim = x.shape
    out = torch.empty((batch, seq, heads, dim), device = x.device, dtype = x.dtype)
    rows = batch * seq * heads
    f = torch.view_as_real(freqs)
    try:
        with torch.cuda.device(x.device):
            _kernel()[((rows + _ROWS - 1) // _ROWS,)](
                x,
                norm.weight,
                f,
                out,
                rows,
                heads,
                seq,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                f.stride(0),
                float(norm.eps),
                HALF = dim // 2,
                ROWS = _ROWS,
                REAL_FMA_X = fusion[0] == "x",
                IMAG_FMA_X = fusion[1] == "x",
                num_warps = _WARPS,
            )
    except Exception:  # noqa: BLE001 - a JIT / driver failure: the stock math for the rest of the process
        _FAILED[0] = True
        return None
    STATS["qk_fused"] += 1
    return out


def _make_prepare(mod: Any, stock: Any) -> Any:
    @functools.wraps(stock)
    def _qwenimage21_prepare_qkv(
        attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
    ):
        if rotary_emb is None or not active(attn):
            return stock(
                attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
            )
        import torch

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        head_dim = attn.inner_dim // attn.heads
        query = query.unflatten(-1, (-1, head_dim))
        key = key.unflatten(-1, (-1, head_dim))
        value = value.unflatten(-1, (-1, head_dim))
        fused_q = _norm_rope(query, attn.norm_q, rotary_emb) if query.dtype == value.dtype else None
        fused_k = _norm_rope(key, attn.norm_k, rotary_emb) if fused_q is not None else None
        if fused_k is None:
            rope = mod.apply_rotary_emb_qwen
            query = rope(attn.norm_q(query).to(value.dtype), rotary_emb, use_real = False)
            key = rope(attn.norm_k(key).to(value.dtype), rotary_emb, use_real = False)
        else:
            query, key = fused_q, fused_k

        if layer_cache is not None:
            if kv_cache_mode == "extract" and cache_write_slice is not None:
                layer_cache.store(
                    key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone()
                )
            elif kv_cache_mode == "cached":
                cached_k, cached_v = layer_cache.get()
                key = torch.cat([cached_k, key], dim = 1)
                value = torch.cat([cached_v, value], dim = 1)
        return query, key, value, query.shape[1]

    _qwenimage21_prepare_qkv.__unsloth_q21_eager__ = True
    return _qwenimage21_prepare_qkv


def why_unsupported(mod: Any) -> Optional[str]:
    for name, want in _FINGERPRINTS.items():
        obj = mod
        for part in name.split("."):
            obj = getattr(obj, part, None)
        got = source_digest(obj)
        if got not in want:
            return f"{name} differs from the version this was written against ({got})"
    return None


def install(logger: Any = None) -> bool:
    try:
        import torch
        if is_rocm_torch(torch) or not torch.cuda.is_available():
            return False
        from diffusers.models.transformers import transformer_qwenimage21 as mod
    except Exception:  # noqa: BLE001 - no CUDA, or diffusers without Qwen-Image 2.1
        return False
    with _LOCK:
        current = getattr(mod, "_qwenimage21_prepare_qkv", None)
        if getattr(current, "__unsloth_q21_eager__", False):
            return True
        why = why_unsupported(mod)
        if why is not None:
            if logger is not None:
                logger.info("diffusion.q21_eager: stock eager step kept: %s", why)
            return False
        _STOCK["prepare"] = current
        _STOCK["rmsnorm_forward"] = mod.RMSNorm.forward
        mod._qwenimage21_prepare_qkv = _make_prepare(mod, current)
    return True


def uninstall() -> None:
    try:
        from diffusers.models.transformers import transformer_qwenimage21 as mod
    except Exception:  # noqa: BLE001
        return
    with _LOCK:
        stock = _STOCK.pop("prepare", None)
        if stock is not None and getattr(
            getattr(mod, "_qwenimage21_prepare_qkv", None), "__unsloth_q21_eager__", False
        ):
            mod._qwenimage21_prepare_qkv = stock


def install_for_pipe(pipe: Any, logger: Any = None) -> bool:
    transformer = getattr(pipe, "transformer", None)
    if disabled() or type(transformer).__name__ != _CLASS:
        return False
    if not install(logger):
        return False
    set_state(transformer, EAGER)
    if logger is not None:
        logger.info(
            "diffusion.q21_eager: renders before the deferred compile run the fused q/k RMSNorm + RoPE "
            "and broadcast modulation"
        )
    return True


def demote(pipe: Any) -> None:
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None and getattr(transformer, "__dict__", {}).get(_ATTR) is not None:
        set_state(transformer, BG)
