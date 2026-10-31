# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused ConvRot rotation and act quant per shared input, bit-identical to separate ops; fused QKV
Linear only where diffusion_int8_gemm leaves rotated Linears on stock path, resident only; kill
UNSLOTH_DIFFUSION_Q21_CONVROT_FUSED=0"""

from __future__ import annotations

import functools
import os
import threading
import types
from typing import Any, Optional
from .rocm_bf16 import is_rocm_torch

Q21_CONVROT_FUSED_ENV = "UNSLOTH_DIFFUSION_Q21_CONVROT_FUSED"
_PREPARE = "_qwenimage21_prepare_qkv"
_QKV_ATTR = "_unsloth_q21_fused_qkv"
_OUT_MARK = "_unsloth_q21_out_prev"
_NO_PREV = object()

_LOCK = threading.Lock()
_STATE: dict = {}


def q21_convrot_fused_disabled() -> bool:
    return (os.environ.get(Q21_CONVROT_FUSED_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def arch_ok(device: Any = None) -> bool:
    """True where ``diffusion_int8_gemm`` has neither its fused GEMM nor ``rotq_i8`` for this arch."""
    try:
        import torch

        if not torch.cuda.is_available() or is_rocm_torch(torch):
            return False
        dev = torch.device(device) if device is not None else None
        if dev is None or dev.type != "cuda":
            index = torch.cuda.current_device()
        else:
            index = dev.index if dev.index is not None else torch.cuda.current_device()
        from .diffusion_int8_gemm import _gates, int8_gemm_mode, rotated_swapped

        if int8_gemm_mode() == "off":
            return True
        swapped = rotated_swapped(index)
        if swapped is not None:
            return not swapped
        gemm, rotq = _gates(torch.cuda.get_device_capability(index))
        return not (gemm or rotq)
    except Exception:  # noqa: BLE001
        return False


def _rotated_group(linears: list) -> int:
    """The shared ConvRot group of ``linears``, else 0 (plain or mixed)."""
    from .diffusion_zimage_fused import _rotation_group

    group = _rotation_group(linears)
    return group if isinstance(group, int) and group > 0 else 0


def _plain(lin: Any) -> bool:
    from .diffusion_int8_fused import _I8_GEMM_MARK, _plain_int8_weight
    d = getattr(lin, "__dict__", {})
    return "forward" not in d and _I8_GEMM_MARK not in d and _plain_int8_weight(lin.weight)


def rotated_ff(module: Any) -> bool:
    """A Qwen-Image-2.1 SwiGLU with rotated gate/value and down projections, on a covered arch."""
    if type(module).__name__ != "QwenImage21SwiGLUFeedForward" or q21_convrot_fused_disabled():
        return False
    try:
        parts = [module.gate_layer, module.proj]
        if not _rotated_group(parts) or not _rotated_group([module.out]):
            return False
        return arch_ok(module.gate_layer.weight.device)
    except Exception:  # noqa: BLE001
        return False


def _qkv_intact(attn: Any) -> Any:
    rec = attn.__dict__.get(_QKV_ATTR)
    if rec is None:
        return None
    lin, q, k, v = rec
    if attn.to_q is not q or attn.to_k is not k or attn.to_v is not v:
        return None
    if lin.weight.device != q.weight.device:
        return None  # a layer split moved the parts after the fusion; the stale fused copy is on another card
    return lin


def _make_prepare(stock: Any, mod: Any) -> Any:
    import torch

    from .diffusion_int8_fused import ensure_intmm_module, int8_linear
    from .diffusion_int8_gemm import shared_rotq_linears

    ensure_intmm_module()  # resolved here, eagerly: an import under the trace is a graph break per Linear

    def _qwenimage21_prepare_qkv(
        attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
    ):
        fused = _qkv_intact(attn) if torch.compiler.is_compiling() else None
        if fused is not None:
            query, key, value = int8_linear(fused, hidden_states).chunk(3, dim = -1)
        else:
            qkv = shared_rotq_linears(hidden_states, (attn.to_q, attn.to_k, attn.to_v))
            if qkv is None:
                return stock(
                    attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
                )
            query, key, value = qkv
        # stock body below the projections, op for op (fingerprinted at install)

        # main's split (by head_dim); equal to 0.41.0's ``(attn.heads, -1)`` when every head is local
        head_dim = attn.inner_dim // attn.heads
        query = query.unflatten(-1, (-1, head_dim))
        key = key.unflatten(-1, (-1, head_dim))
        value = value.unflatten(-1, (-1, head_dim))

        query = attn.norm_q(query).to(value.dtype)
        key = attn.norm_k(key).to(value.dtype)

        if rotary_emb is not None:
            # read per call: diffusion_qwenimage21_rope may have replaced it
            query = mod.apply_rotary_emb_qwen(query, rotary_emb, use_real = False)
            key = mod.apply_rotary_emb_qwen(key, rotary_emb, use_real = False)

        if layer_cache is not None:
            if kv_cache_mode == "extract" and cache_write_slice is not None:
                layer_cache.store(
                    key[:, cache_write_slice].clone(),
                    value[:, cache_write_slice].clone(),
                )
            elif kv_cache_mode == "cached":
                cached_k, cached_v = layer_cache.get()
                key = torch.cat([cached_k, key], dim = 1)
                value = torch.cat([cached_v, value], dim = 1)

        seq_len_q = query.shape[1]
        return query, key, value, seq_len_q

    functools.update_wrapper(
        _qwenimage21_prepare_qkv, stock, assigned = ("__name__", "__doc__"), updated = ()
    )
    _qwenimage21_prepare_qkv.__unsloth_q21_fused__ = True
    return _qwenimage21_prepare_qkv


def _out_forward(self: Any, x: Any) -> Any:
    """``to_out[0]`` under compile: own rotation + the probed act quant kernel."""
    import torch

    if not torch.compiler.is_compiling():
        return type(self).forward(self, x)
    from .diffusion_int8_fused import int8_linear_core

    return int8_linear_core(self, x)


def _patch_prepare(logger: Any = None) -> bool:
    try:
        from diffusers.models.transformers import transformer_qwenimage21 as mod
    except Exception:  # noqa: BLE001 - diffusers without Qwen-Image 2.1
        return False
    with _LOCK:
        current = getattr(mod, _PREPARE, None)
        if getattr(current, "__unsloth_q21_fused__", False):
            return True
        from .diffusion_qwenimage21_rope import _FINGERPRINTS, _digest

        digest = _digest(current)
        if digest not in _FINGERPRINTS[_PREPARE]:
            if logger is not None:
                logger.info(
                    "diffusion.q21_convrot_fused: stock QKV kept: %s differs (%s)", _PREPARE, digest
                )
            return False
        _STATE["stock"] = (mod, current)
        setattr(mod, _PREPARE, _make_prepare(current, mod))
    return True


def _rotated_attention(root: Any) -> list:
    """(attention, [to_q, to_k, to_v]) whose Q / K / V share one ConvRot group on one device."""
    todo = []
    for _name, module in root.named_modules():
        if type(module).__name__ != "QwenImage21Attention":
            continue
        parts = [getattr(module, n, None) for n in ("to_q", "to_k", "to_v")]
        if any(p is None for p in parts) or not _rotated_group(parts):
            continue
        if len({p.weight.device for p in parts}) != 1:
            continue
        todo.append((module, parts))
    return todo


def install_shared_qkv(transformer: Any, logger: Any = None) -> int:
    """Patch the QKV helper for the int8 GEMM path (checked per call, so install order does not matter); returns the
    rotated attention count."""
    n = len(_rotated_attention(transformer))
    if not n or not _patch_prepare(logger):
        return 0
    if logger is not None:
        logger.info(
            "diffusion.q21_convrot_fused: Q / K / V share one ConvRot rotation + act quant on the int8 GEMM path "
            "(%d attention module(s))",
            n,
        )
    return n


def _swap_card_present(transformer: Any) -> bool:
    """A card the transformer sits on (each card of a layer split; every visible card while its weights are off the
    GPU) where the int8 GEMM swap covers rotated Linears, so the shared-QKV helper can engage. Not the current card:
    on a mixed host it need not hold the transformer."""
    import torch

    from .diffusion_int8_fused import resident_cuda_cards

    cards = resident_cuda_cards(transformer) or range(torch.cuda.device_count())
    return any(_swap_covers(index) for index in cards)


def _swap_covers(index: int) -> bool:
    """True where ``diffusion_int8_gemm`` swaps rotated Linears onto ``rotq_i8`` on ``cuda:<index>``: NVIDIA CUDA only
    (ROCm reports a CUDA-like capability but the swap declines it), the swap on, and the rotation kernel gated in and
    not failed its per-device probe (``shared_rotq_linears`` declines every record without it, so the fused GEMM
    alone does not count). Any failure counts as not covered, so the shared-QKV helper is never installed on a
    guess."""
    try:
        import torch

        if not torch.cuda.is_available() or is_rocm_torch(torch):
            return False
        from .diffusion_int8_gemm import _gates, int8_gemm_mode, rotated_swapped

        if int8_gemm_mode() == "off":
            return False
        swapped = rotated_swapped(index)
        if swapped is not None:
            return swapped
        _, rotq = _gates(torch.cuda.get_device_capability(index))
        if not rotq:
            return False
        from .diffusion_convrot_quant import _ROTQ_DEVICE

        # probed already (resident swap ran first): its verdict; else the arch gate, re-checked per call
        return _ROTQ_DEVICE.get(index, True) is not None
    except Exception:  # noqa: BLE001
        return False


def install(
    transformer: Any,
    logger: Any = None,
    offload_active: bool = False,
) -> dict:
    """Before the first compile; deferred to the first forward when the weights are not on the GPU yet."""
    result = {"fused_qkv": 0, "out": 0}
    if q21_convrot_fused_disabled():
        uninstall(transformer)
        return result
    if type(transformer).__name__ != "QwenImage21Transformer2DModel":
        return result
    try:
        import torch
        if torch.cuda.is_available() and _swap_card_present(transformer):
            result["fused_qkv"] = install_shared_qkv(transformer, logger)
    except Exception:  # noqa: BLE001 - the stock helper stays
        pass
    shared = result["fused_qkv"]
    if offload_active:
        return result
    from .diffusion_int8_fused import resident_cuda_cards, run_on_first_call

    if resident_cuda_cards(transformer):
        out = install_modules(transformer, logger)
        out["fused_qkv"] = max(out["fused_qkv"], shared)
        return out
    run_on_first_call(transformer, "q21_convrot_fused", lambda t: install_modules(t, logger))
    return {"fused_qkv": shared or None, "out": None}


def _rotq_note(cards: Any, armed: list) -> str:
    """Log suffix: whether the fused rotq_h16 rotation + quant serves these cards, with its latency check."""
    try:
        from .diffusion_convrot_quant import _UNLISTED_TIMING
        timing = {i: _UNLISTED_TIMING.get(i) for i in cards if _UNLISTED_TIMING.get(i)}
    except Exception:  # noqa: BLE001
        timing = {}
    shown = "; ".join(
        f"cuda:{i} {t['h16']:.0f} vs {t['stock']:.0f} us stock"
        for i, t in timing.items()
        if "h16" in t and "stock" in t
    )
    if armed:
        return ", rotation fused into the act quant (rotq_h16" + (f": {shown})" if shown else ")")
    return f", fused rotation kept off ({shown})" if shown else ""


def install_modules(root: Any, logger: Any = None) -> dict:
    result = {"fused_qkv": 0, "out": 0}
    if q21_convrot_fused_disabled():
        return result
    import torch

    from .diffusion_int8_fused import resident_cuda_cards

    cards = resident_cuda_cards(root)  # one card, or each card of a layer split
    if not cards or not all(arch_ok(torch.device("cuda", index)) for index in cards):
        return result
    from .diffusion_zimage_fused import _fuse_linears, _share_storage

    todo = _rotated_attention(root)
    if not todo or not _patch_prepare(logger):
        return result
    from .diffusion_int8_fused import arm_rotated

    armed = [index for index in cards if arm_rotated(index)]
    note = _rotq_note(cards, armed)
    with _LOCK:
        for module, parts in todo:
            if _QKV_ATTR not in module.__dict__ and all(_plain(p) for p in parts):
                fused = _fuse_linears(parts)
                if fused is not None and _share_storage(fused, parts):
                    module.__dict__[_QKV_ATTR] = (fused, *parts)
            if _QKV_ATTR in module.__dict__:
                result["fused_qkv"] += 1
            out = module.to_out[0]
            if _OUT_MARK in out.__dict__:
                result["out"] += 1
            elif _rotated_group([out]) and _plain(out):
                out.__dict__[_OUT_MARK] = True
                out.forward = types.MethodType(_out_forward, out)
                result["out"] += 1
    if logger is not None:
        logger.info(
            "diffusion.q21_convrot_fused: one ConvRot rotation + act quant for QKV on %d attention module(s), "
            "probed act quant on %d output projection(s)%s",
            result["fused_qkv"],
            result["out"],
            note,
        )
    return result


def uninstall(transformer: Any = None) -> None:
    """Restore the stock QKV helper; drop the fused Linears and to_out forwards."""
    if transformer is not None:
        from .diffusion_int8_fused import cancel_first_call
        cancel_first_call(transformer, "q21_convrot_fused")
    with _LOCK:
        stock = _STATE.pop("stock", None)
        if stock is not None:
            mod, fn = stock
            if getattr(getattr(mod, _PREPARE, None), "__unsloth_q21_fused__", False):
                setattr(mod, _PREPARE, fn)
        if transformer is not None:
            for module in transformer.modules():
                module.__dict__.pop(_QKV_ATTR, None)
                if module.__dict__.pop(_OUT_MARK, None) is not None:
                    fwd = module.__dict__.get("forward")
                    if getattr(fwd, "__func__", None) is _out_forward:
                        module.__dict__.pop("forward", None)


def is_installed(module: Any) -> bool:
    return _QKV_ATTR in getattr(module, "__dict__", {}) or _OUT_MARK in getattr(
        module, "__dict__", {}
    )
