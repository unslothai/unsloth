# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 text_fusion / txt_in / rotary_emb run once per prompt, not once per step (none reads timestep or latents).

Entries live for one render (cleared by Studio's generate ``finally``); inputs match on identity / address and the
version counter where one exists (inference tensors have none). Streamed modules, graph capture and tracing run
stock. ``UNSLOTH_DIFFUSION_KREA2_TEXT_CACHE=0`` disables.
"""

from __future__ import annotations

import os
import weakref
from typing import Any, Optional

TEXT_CACHE_ENV = "UNSLOTH_DIFFUSION_KREA2_TEXT_CACHE"
CLASS_NAME = "Krea2Transformer2DModel"
MODULES = ("text_fusion", "txt_in", "rotary_emb")
_INSTALLED_ATTR = "_unsloth_krea2_text_cache"
# keyed weakly by module, never stored on it: a deepcopy (CFG-parallel replica) holds nothing of the source's
_MEMOS: Any = weakref.WeakKeyDictionary()
_STOCK: Any = weakref.WeakKeyDictionary()
_MEMO_MAX = 4
_OFF = ("0", "off", "false", "no")
STATS = {"hits": 0, "misses": 0, "stock": 0}


def enabled() -> bool:
    return (os.environ.get(TEXT_CACHE_ENV) or "").strip().lower() not in _OFF


def _version(t: Any) -> Any:
    return "inference" if t.is_inference() else t._version


def _sources(args: tuple, kwargs: dict) -> Optional[tuple]:
    import torch

    flat = []
    for value in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if value is not None and not torch.is_tensor(value):
            return None
        flat.append(value)
    return tuple(flat), tuple(sorted(kwargs))


def _geometry(t: Any) -> tuple:
    root = t._base if t._base is not None else t
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device, _version(root))


def _key(t: Any) -> tuple:
    # the mask is re-sliced each call (new view, no _base under inference_mode); holding the old view keeps its
    # storage alive, so equal address + geometry means the same memory
    return (t, _geometry(t))


def _same(held: tuple, flat: tuple) -> bool:
    if len(held) != len(flat):
        return False
    for key, t in zip(held, flat):
        if key is None or t is None:
            if (key is None) != (t is None):
                return False
            continue
        if key[0] is not t and key[1] != _geometry(t):
            return False
        if key[0] is t and key[1][-1] != _version(t._base if t._base is not None else t):
            return False
    return True


def _streamed(module: Any) -> bool:
    for sub in module.modules():
        if getattr(sub, "_hf_hook", None) is not None:
            return True
        hooks = getattr(getattr(sub, "_diffusers_hook", None), "hooks", None) or {}
        if any("offload" in str(k).lower() for k in hooks):
            return True
    return False


def _amp_state(flat: tuple) -> Optional[tuple]:
    """Part of the key: an fp32 entry must not answer a bf16 autocast call."""
    import torch

    device = next((t.device.type for t in flat if t is not None), "cpu")
    try:
        on = bool(torch.is_autocast_enabled(device))
        return (device, on, torch.get_autocast_dtype(device) if on else None)
    except Exception:  # noqa: BLE001 - older torch: no per-device query
        return None


def _uncacheable() -> bool:
    import torch
    if torch.is_grad_enabled():
        return True
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return True


def _cached_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    import torch

    if torch.compiler.is_compiling():
        # before touching any Python state: it would become a Dynamo guard failing every step
        return type(self).forward(self, *args, **kwargs)
    held_stock = _STOCK.get(self)
    if held_stock is None:
        stock = type(self).forward.__get__(self)
    elif held_stock[0] == "method":
        stock = held_stock[1].__get__(self)
    else:
        stock = held_stock[1]
    if not enabled() or _uncacheable() or _streamed(self):
        STATS["stock"] += 1
        return stock(*args, **kwargs)
    found = _sources(args, kwargs)
    if found is None:
        STATS["stock"] += 1
        return stock(*args, **kwargs)
    flat, names = found
    amp = _amp_state(flat)
    if amp is None:
        STATS["stock"] += 1
        return stock(*args, **kwargs)
    memo = _MEMOS.setdefault(self, [])
    for held, held_names, held_amp, out in memo:
        if held_names == names and held_amp == amp and _same(held, flat):
            STATS["hits"] += 1
            return out
    out = stock(*args, **kwargs)
    held = tuple(None if t is None else _key(t) for t in flat)
    memo.insert(0, (held, names, amp, out))
    del memo[_MEMO_MAX:]
    STATS["misses"] += 1
    return out


_cached_forward._unsloth_krea2_text_cache = True


def _wrap(module: Any, stock: Any) -> Any:
    import types
    if isinstance(stock, types.MethodType):
        if stock.__self__ is module and stock.__func__ is not type(module).forward:
            # store the function only so the table never holds the module
            _STOCK[module] = ("method", stock.__func__)
    elif stock is not None:
        _STOCK[module] = ("plain", stock)
    return types.MethodType(_cached_forward, module)


def installed(transformer: Any) -> bool:
    return bool(getattr(transformer, "__dict__", {}).get(_INSTALLED_ATTR))


def install(transformer: Any, logger: Any = None) -> bool:
    if type(transformer).__name__ != CLASS_NAME or not enabled():
        return False
    if installed(transformer):
        return True
    for name in MODULES:
        if not hasattr(getattr(transformer, name, None), "forward"):
            return False
    for name in MODULES:
        module = getattr(transformer, name)
        stock = module.forward
        if getattr(stock, "_unsloth_krea2_text_cache", False):
            continue
        module.forward = _wrap(module, stock)
    transformer.__dict__[_INSTALLED_ATTR] = True
    if logger is not None:
        logger.info(
            "diffusion.krea2: text fusion, text projection and rotary table run once per prompt"
        )
    return True


def clear(transformer: Any) -> None:
    for name in MODULES:
        module = getattr(transformer, name, None)
        if module is not None:
            _MEMOS.pop(module, None)


def end_render(pipe: Any) -> None:
    """Never raises; inference-mode prompt tensors have no version counter, so entries must not outlive a render."""
    try:
        transformer = getattr(pipe, "transformer", None)
        if installed(transformer):
            clear(transformer)
    except Exception:  # noqa: BLE001
        pass
