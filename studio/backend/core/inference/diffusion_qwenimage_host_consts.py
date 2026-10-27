# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image denoiser forward without host-to-device copies, so its CUDA graph captures.

The stock ``QwenImageTransformer2DModel.forward`` builds two device tensors from Python values on every step:
Qwen-Image-Edit-2511's ``modulate_index`` (``torch.tensor(<list of 0/1 per image token>, device = ...)``, only when
the config sets ``zero_cond_t``) and the text length fallback in ``compute_text_seq_len_from_mask``
(``torch.as_tensor(text_seq_len, device = ...)``). Each is a copy from pageable host memory, which a CUDA graph
capture refuses (``Cannot copy between CPU and CUDA tensors during CUDA graph capture``), so the edit denoiser ran
every step without its graph, and with the capture counted as failed it never recorded a learned activation peak.

This points the module's ``torch`` global at a copy of the torch namespace whose ``tensor`` / ``as_tensor`` return
a cached device tensor for the same Python value: the warm-up forward before a capture builds it, the captured
forward reads it. The RoPE tables (``pos_embed.pos_freqs`` / ``neg_freqs``, plain CPU attributes the stock code
copies to the device per call when compiled, where its ``lru_cache`` is traced through) move to the device once,
before the first forward. Values are identical, so renders are bit-identical. Every other call goes to torch unchanged.
Kill switch: ``UNSLOTH_DIFFUSION_QWEN_HOST_CONSTS=0`` (or ``UNSLOTH_DIFFUSION_PATCH_QWENIMAGE_HOST_CONSTS=0``).
"""

from __future__ import annotations

import collections
import os
import threading
import types
from typing import Any

QWEN_HOST_CONSTS_ENV = "UNSLOTH_DIFFUSION_QWEN_HOST_CONSTS"
_MODULE = "diffusers.models.transformers.transformer_qwenimage"
_CLASSES = ("QwenImageTransformer2DModel",)
# device types whose constants are cached (a CPU tensor has no copy to avoid)
_MEMO_DEVICE_TYPES = ("cuda",)
_MAX_ENTRIES = 64
_LOCK = threading.Lock()
_CACHE: "collections.OrderedDict[Any, Any]" = collections.OrderedDict()
_STOCK: dict = {}  # id(module) -> (module, the torch it held)


def disabled() -> bool:
    return (os.environ.get(QWEN_HOST_CONSTS_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def _freeze(data: Any) -> Any:
    """A hashable copy of nested lists / tuples of Python scalars, or None for anything else."""
    if isinstance(data, bool) or isinstance(data, (int, float)):
        return (type(data).__name__, data)
    if isinstance(data, (list, tuple)):
        if data and all(type(x) is int for x in data):
            return ("i", tuple(data))
        out = []
        for x in data:
            f = _freeze(x)
            if f is None:
                return None
            out.append(f)
        return ("l", tuple(out))
    return None


def _cached(make: Any, data: Any, args: tuple, kwargs: dict) -> Any:
    import torch

    device = kwargs.get("device")
    if args or set(kwargs) - {"device", "dtype"} or device is None:
        return make(data, *args, **kwargs)
    dev = torch.device(device)
    if dev.type not in _MEMO_DEVICE_TYPES:
        return make(data, *args, **kwargs)
    frozen = _freeze(data)
    if frozen is None:
        return make(data, *args, **kwargs)
    if dev.index is None and dev.type == "cuda":
        dev = torch.device("cuda", torch.cuda.current_device())
    key = (make.__name__, str(dev), kwargs.get("dtype"), frozen)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    value = make(data, device = dev, dtype = kwargs.get("dtype"))
    with _LOCK:
        _CACHE[key] = value
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last = False)
    return value


def _shim(torch_mod: Any) -> types.ModuleType:
    """The torch namespace with ``tensor`` / ``as_tensor`` cached for Python values on a CUDA device."""
    shim = types.ModuleType("unsloth_qwenimage_torch")
    shim.__dict__.update(
        {
            k: v
            for k, v in vars(torch_mod).items()
            if k not in ("__name__", "__spec__", "__loader__")
        }
    )
    stock_tensor, stock_as_tensor = torch_mod.tensor, torch_mod.as_tensor

    def tensor(data: Any, *args: Any, **kwargs: Any) -> Any:
        return _cached(stock_tensor, data, args, kwargs)

    def as_tensor(data: Any, *args: Any, **kwargs: Any) -> Any:
        return _cached(stock_as_tensor, data, args, kwargs)

    disable = getattr(getattr(torch_mod, "compiler", None), "disable", None)
    shim.tensor = disable(tensor) if callable(disable) else tensor
    shim.as_tensor = disable(as_tensor) if callable(disable) else as_tensor
    shim.__unsloth_host_consts__ = True
    return shim


def _rope_tables_to(transformer: Any, device: Any) -> int:
    """Move ``pos_embed``'s CPU tensor attributes (the RoPE frequency tables) to ``device``; returns how many moved."""
    import torch

    rope = getattr(transformer, "pos_embed", None)
    if rope is None or device is None or torch.device(device).type not in _MEMO_DEVICE_TYPES:
        return 0
    moved = 0
    for name in ("pos_freqs", "neg_freqs"):
        t = rope.__dict__.get(name)
        if isinstance(t, torch.Tensor) and t.device != torch.device(device):
            rope.__dict__[name] = t.to(device)
            moved += 1
    return moved


def _first_call(transformer: Any) -> None:
    """Before the first forward (outside any compile or capture): the RoPE tables go to the input's device."""
    hooks = transformer.__dict__.setdefault("_unsloth_first_call_hooks", {})
    if "qwenimage_host_consts" in hooks:
        return

    def _pre_hook(mod: Any, args: Any, kwargs: Any) -> None:
        handle = hooks.pop("qwenimage_host_consts", None)
        if handle is not None:
            handle.remove()
        x = kwargs.get("hidden_states", args[0] if args else None)
        try:
            _rope_tables_to(mod, getattr(x, "device", None))
        except Exception:  # noqa: BLE001 - the stock per-call copy still runs
            pass
        return None

    hooks["qwenimage_host_consts"] = transformer.register_forward_pre_hook(
        _pre_hook, with_kwargs = True
    )


def install(transformer: Any, logger: Any = None) -> bool:
    """Patch the Qwen-Image transformer module (process-global, idempotent). False when disabled or not Qwen-Image."""
    if disabled():
        uninstall()
        return False
    if type(transformer).__name__ not in _CLASSES:
        return False
    import importlib

    import torch

    try:
        mod = importlib.import_module(_MODULE)
    except Exception:  # noqa: BLE001 - no Qwen-Image in this diffusers
        return False
    with _LOCK:
        current = getattr(mod, "torch", None)
        if not getattr(current, "__unsloth_host_consts__", False):
            if current is not torch:
                return False
            _STOCK[id(mod)] = (mod, current)
            mod.torch = _shim(torch)
    _first_call(transformer)
    if logger is not None:
        logger.info(
            "diffusion.qwenimage_host_consts: modulate_index / text-length constants cached on the device "
            "(the denoiser step captures as a CUDA graph)"
        )
    return True


def uninstall(transformer: Any = None) -> None:
    if transformer is not None:
        hooks = getattr(transformer, "__dict__", {}).get("_unsloth_first_call_hooks") or {}
        handle = hooks.pop("qwenimage_host_consts", None)
        if handle is not None:
            handle.remove()
    with _LOCK:
        for mod, stock in list(_STOCK.values()):
            mod.torch = stock
        _STOCK.clear()
        _CACHE.clear()
