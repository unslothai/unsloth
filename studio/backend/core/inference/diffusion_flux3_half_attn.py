# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 attention in float16 on fp16-only GPUs; off via UNSLOTH_DIFFUSION_FLUX3_FP16_ATTN=0.

GPUs without bf16 (T4 and the other Turing / Volta cards) run FLUX.3 in float32, because its residual stream
overflows float16. Attention itself cannot overflow there, so this runs it on float16 tensor cores instead of the float32
SIMT kernel (memory-efficient SDPA, 42 s of a 210 s T4 render):

- Q and K come out of an RMSNorm (then RoPE, which keeps each rotated pair's norm), so every entry is at most
  ``sqrt(head_dim) * max|norm weight|``. That bound is read from the loaded weights once at install; the patch engages
  only when it sits below ``QK_BOUND`` (float16 max is 65504).
- V is unbounded, so each (batch, head) slice is divided by a power of two that puts its largest entry in [2^13, 2^14]
  (exact in float32), and the output, a convex combination of V rows, is multiplied back in float32.

Float16 keeps 10 mantissa bits, more than the 7 bf16 GPUs run this same attention at. An explicit speed ``off``
(exact) keeps stock float32 attention, as does every 16-bit load and every GPU with bf16.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any

FLUX3_FP16_ATTN_ENV = "UNSLOTH_DIFFUSION_FLUX3_FP16_ATTN"
_MODULE = "diffusers.models.transformers.transformer_flux3"
_ATTR = "dispatch_attention_fn"
# Q / K entries must stay well inside float16 (65504): the scores are accumulated in float32 by the kernel.
QK_BOUND = 16384.0
# V's largest entry per (batch, head) after its power-of-two scale: in [V_TARGET / 2, V_TARGET].
V_TARGET = 16384.0
_LOCK = threading.Lock()
_STOCK: dict = {}
_ORIGINAL: list = [None]
_CALLS = [0]


def disabled() -> bool:
    return (os.environ.get(FLUX3_FP16_ATTN_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def fp16_only_tensor_core_gpu(device: Any = "cuda") -> bool:
    """NVIDIA sm70 / sm75 (Volta, Turing): float16 tensor cores, no bf16."""
    try:
        import torch

        from .rocm_bf16 import is_rocm_torch

        if (
            not str(device).startswith("cuda")
            or not torch.cuda.is_available()
            or is_rocm_torch(torch)
        ):
            return False
        major, minor = torch.cuda.get_device_capability(torch.device(device))
        return (7, 0) <= (int(major), int(minor)) < (8, 0)
    except Exception:  # noqa: BLE001 - an unanswered probe keeps stock attention
        return False


def qk_bound(transformer: Any) -> float:
    """Largest possible |Q| / |K| entry after the attention RMSNorms (inf when any is missing or not affine)."""
    import torch

    worst = 0.0
    seen = 0
    for module in transformer.modules():
        if type(module).__name__ != "Flux3Attention":
            continue
        head_dim = int(getattr(module, "head_dim", 0) or 0)
        for name in ("norm_q", "norm_k"):
            norm = getattr(module, name, None)
            weight = getattr(norm, "weight", None)
            if not isinstance(norm, torch.nn.RMSNorm) or weight is None or head_dim <= 0:
                return math.inf
            try:
                wmax = float(weight.detach().abs().max().float().item())
            except Exception:  # noqa: BLE001 - an unreadable weight (meta, foreign subclass) proves nothing
                return math.inf
            if not math.isfinite(wmax):
                return math.inf
            worst = max(worst, math.sqrt(head_dim) * wmax)
        seen += 1
    return worst if seen else math.inf


def _eligible(query: Any, key: Any, value: Any, attn_mask: Any) -> bool:
    import torch

    for t in (query, key, value):
        if not torch.is_tensor(t) or t.dtype != torch.float32 or not t.is_cuda or t.dim() != 4:
            return False
        if t.requires_grad and torch.is_grad_enabled():
            return False
    if attn_mask is not None and (not torch.is_tensor(attn_mask) or attn_mask.dtype != torch.bool):
        return False
    return True


def _half_dispatch(query, key, value, *args, **kwargs):
    import torch

    stock = _STOCK.get(_ATTR) or _ORIGINAL[0]
    attn_mask = kwargs.get("attn_mask", args[0] if args else None)
    if not _eligible(query, key, value, attn_mask):
        return stock(query, key, value, *args, **kwargs)
    # (batch, seq, heads, head_dim): one power-of-two scale per (batch, head)
    amax = value.abs().amax(dim = (1, 3), keepdim = True).clamp_min(2.0**-100)
    scale = torch.exp2(torch.ceil(torch.log2(amax / V_TARGET)))
    out = stock(query.half(), key.half(), (value / scale).half(), *args, **kwargs)
    if not torch.compiler.is_compiling():
        _CALLS[0] += 1
    return out.float() * scale


def install_for_pipe(
    pipe: Any,
    device: Any = "cuda",
    speed_mode: Any = None,
    logger: Any = None,
    explicit_off: Any = None,
) -> bool:
    """Idempotent. False (stock) unless: a FLUX.3 transformer on an sm70 / sm75 GPU (where FLUX.3 computes in float32;
    each call still checks float32 inputs), no explicit speed ``off`` (``explicit_off``; None reads ``speed_mode``: a
    dense load's unset request also resolves to off and keeps this), the diffusers symbol is the expected one, and the
    RMSNorm bound proves Q / K fit."""
    import sys

    transformer = getattr(pipe, "transformer", None)
    if (
        disabled()
        or transformer is None
        or type(transformer).__module__ != _MODULE
        or (
            str(speed_mode or "").strip().lower() == "off"
            if explicit_off is None
            else bool(explicit_off)
        )
        or not fp16_only_tensor_core_gpu(device)
    ):
        uninstall()
        return False
    bound = qk_bound(transformer)
    if not bound < QK_BOUND:
        uninstall()
        if logger is not None:
            logger.info(
                "diffusion.flux3_fp16_attn: stock float32 attention (Q / K bound %.0f)", bound
            )
        return False
    with _LOCK:
        if _ATTR in _STOCK:
            return True
        mod = sys.modules.get(_MODULE)
        stock = getattr(mod, _ATTR, None) if mod is not None else None
        if stock is None or getattr(stock, "__name__", "") != _ATTR:
            return False
        _STOCK[_ATTR] = stock
        _ORIGINAL[0] = stock
        setattr(mod, _ATTR, _half_dispatch)
    if logger is not None:
        logger.info(
            "diffusion.flux3_fp16_attn: float16 attention on this fp16-only GPU (float32 compute, Q / K bound %.0f)",
            bound,
        )
    return True


def uninstall() -> None:
    import sys
    with _LOCK:
        stock = _STOCK.pop(_ATTR, None)
        if stock is None:
            return
        mod = sys.modules.get(_MODULE)
        if mod is not None and getattr(mod, _ATTR, None) is _half_dispatch:
            setattr(mod, _ATTR, stock)


def is_installed() -> bool:
    return _ATTR in _STOCK


def call_count() -> int:
    return _CALLS[0]
