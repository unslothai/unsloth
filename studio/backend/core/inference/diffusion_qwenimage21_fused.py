# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 ConvRot int8: one rotation + one act quant per shared input (QKV, SwiGLU gate/value) in the compile.

Rotation is on the shared input axis and int8 scales are per row, so fused outputs equal the separate rotated ones bit
for bit. Only where ``diffusion_int8_gemm`` leaves rotated Linears on the stock path (sm90, sm100), resident only.
Kill switch: ``UNSLOTH_DIFFUSION_Q21_CONVROT_FUSED=0``.
"""

from __future__ import annotations

import functools
import os
import threading
import types
from typing import Any, Optional

Q21_CONVROT_FUSED_ENV = "UNSLOTH_DIFFUSION_Q21_CONVROT_FUSED"
_MODULE = "diffusers.models.transformers.transformer_qwenimage21"
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

        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            return False
        dev = torch.device(device) if device is not None else None
        if dev is None or dev.type != "cuda":
            index = torch.cuda.current_device()
        else:
            index = dev.index if dev.index is not None else torch.cuda.current_device()
        from .diffusion_int8_gemm import _gates, int8_gemm_mode

        if int8_gemm_mode() == "off":
            return True
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
    return lin


def _make_prepare(stock: Any, mod: Any) -> Any:
    import torch

    from .diffusion_int8_fused import int8_linear

    def _qwenimage21_prepare_qkv(
        attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
    ):
        fused = _qkv_intact(attn) if torch.compiler.is_compiling() else None
        if fused is None:
            return stock(
                attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
            )
        # stock body below the projections, op for op (fingerprinted at install)
        query, key, value = int8_linear(fused, hidden_states).chunk(3, dim = -1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

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
        import importlib
        mod = importlib.import_module(_MODULE)
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
    if offload_active or type(transformer).__name__ != "QwenImage21Transformer2DModel":
        return result
    from .diffusion_int8_fused import resident_cuda_device, run_on_first_call

    if resident_cuda_device(transformer) is not None:
        return install_modules(transformer, logger)
    run_on_first_call(transformer, "q21_convrot_fused", lambda t: install_modules(t, logger))
    return {"fused_qkv": None, "out": None}


def install_modules(root: Any, logger: Any = None) -> dict:
    result = {"fused_qkv": 0, "out": 0}
    if q21_convrot_fused_disabled():
        return result
    from .diffusion_int8_fused import resident_cuda_device

    dev = resident_cuda_device(root)
    if dev is None or not arch_ok(dev):
        return result
    from .diffusion_zimage_fused import _fuse_linears, _share_storage

    todo = []
    for _name, module in root.named_modules():
        if type(module).__name__ != "QwenImage21Attention":
            continue
        parts = [getattr(module, n, None) for n in ("to_q", "to_k", "to_v")]
        if any(p is None for p in parts) or not _rotated_group(parts):
            continue
        todo.append((module, parts))
    if not todo or not _patch_prepare(logger):
        return result
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
            "probed act quant on %d output projection(s)",
            result["fused_qkv"],
            result["out"],
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
