# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-memory prompt-conditioning cache for the diffusion and video backends.

A repeated prompt returns the text-encoder output it produced last time instead of running the
encoder again. Under an offload policy that also means the encoder never moves to the device, which
for MiniMax-H3's 32B Qwen3-VL conditioner is most of a render's prompt cost.

On by default (``UNSLOTH_DIFFUSION_PROMPT_CACHE=0`` turns it off), bounded by bytes
(``UNSLOTH_DIFFUSION_PROMPT_CACHE_MB``, default 256, scaled down on small hosts), least recently
used first out. Entries live in host memory, so the cache never takes VRAM from a render. The
cache lives on the pipeline object, so an unload or a reload starts empty;
``release`` frees it eagerly on teardown.

Keying: every bound argument of the encode call except placement (``device``) and RNG
(``generator``), plus the identity of each text encoder module and the attached LoRA set. A call
carrying anything that is not a plain value (a tensor ``prompt_embeds``, a PIL image) bypasses the
cache. A hit returns fresh tensors, never the stored ones, so a pipeline that edits its embeddings
in place cannot poison a later render. The stored tensors are exact copies of the encoder output,
so a cached render is bit-identical to an uncached one.

Hooks: ``pipe.encode_prompt`` (per instance, composes with the opt-in disk cache in
``diffusion_cond_cache``) and MiniMax-H3's modular ``get_qwen3vl_prompt_embeds``, which its
text-encoder steps call as a module global. torch is imported lazily.
"""

from __future__ import annotations

import collections
import hashlib
import inspect
import json
import os
import threading
import weakref
from typing import Any, Optional

_ENV_ENABLE = "UNSLOTH_DIFFUSION_PROMPT_CACHE"
_ENV_BUDGET_MB = "UNSLOTH_DIFFUSION_PROMPT_CACHE_MB"
_DEFAULT_BUDGET_MB = 256
_FALSE_TOKENS = ("0", "false", "no", "off")

_KEY_EXCLUDED_ARGS = frozenset({"device", "generator"})
_TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4")


def enabled() -> bool:
    return (os.environ.get(_ENV_ENABLE) or "").strip().lower() not in _FALSE_TOKENS


def budget_bytes() -> int:
    raw = (os.environ.get(_ENV_BUDGET_MB) or "").strip()
    try:
        mb = float(raw) if raw else float(_DEFAULT_BUDGET_MB)
    except ValueError:
        mb = float(_DEFAULT_BUDGET_MB)
    budget = int(max(0.0, mb) * 1024 * 1024)
    if not raw:
        total = _host_ram_bytes()
        if total:
            budget = min(budget, max(16 * 1024 * 1024, total // 64))
    return budget


def _host_ram_bytes() -> int:
    """Physical RAM, capped by an enforcing cgroup limit (pinned entries are charged to it)."""
    try:
        total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        total = 0
    try:
        from .diffusion_memory import _cgroup_memory_limit_mib
        limit_mib = _cgroup_memory_limit_mib()
    except Exception:  # noqa: BLE001 - no readable limit is the same answer as none
        limit_mib = None
    if limit_mib:
        limit = int(limit_mib) * 1024 * 1024
        total = min(total, limit) if total else limit
    return total


def _torch():
    import torch
    return torch


def _plain(value: Any) -> Any:
    """``value`` as a JSON-safe key component, or raise TypeError when it is not a plain value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    torch = _torch()
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    raise TypeError(type(value).__name__)


def _is_tensor(obj: Any) -> bool:
    return hasattr(obj, "detach") and hasattr(obj, "clone") and hasattr(obj, "element_size")


def _map_tensors(obj: Any, fn: Any) -> Any:
    """Apply ``fn`` to every tensor in a nested tuple / list / dict of tensors and plain values;
    raise TypeError on anything else (such a result is simply not cached)."""
    if _is_tensor(obj):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_map_tensors(o, fn) for o in obj)
    if isinstance(obj, list):
        return [_map_tensors(o, fn) for o in obj]
    if isinstance(obj, dict) and type(obj) is dict:
        return {k: _map_tensors(v, fn) for k, v in obj.items()}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(type(obj).__name__)


def _nbytes(obj: Any) -> int:
    total = [0]

    def count(t: Any) -> Any:
        total[0] += int(t.numel()) * int(t.element_size())
        return t

    _map_tensors(obj, count)
    return total[0]


class PromptCache:
    """A bytes-bounded LRU of encode results. Thread-safe; every method is best-effort."""

    def __init__(self, budget: Optional[int] = None) -> None:
        self.budget = budget_bytes() if budget is None else int(budget)
        self._entries: "collections.OrderedDict[str, tuple[Any, int]]" = collections.OrderedDict()
        self._lock = threading.Lock()
        self.bytes = 0
        self.stats = {"hits": 0, "misses": 0, "bypassed": 0, "evictions": 0, "stored": 0}

    def get(self, key: str, device: Any) -> Any:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            self.stats["hits"] += 1
            stored = entry[0]
        return _map_tensors(stored, lambda t: _fresh(t, device))

    def put(self, key: str, result: Any) -> bool:
        try:
            size = _nbytes(result)
        except TypeError:
            return False
        if size <= 0 or size > self.budget:
            return False
        stored = _map_tensors(result, _store_copy)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self.bytes -= old[1]
            self._entries[key] = (stored, size)
            self.bytes += size
            self.stats["stored"] += 1
            while self.bytes > self.budget and self._entries:
                _, (_, dropped) = self._entries.popitem(last = False)
                self.bytes -= dropped
                self.stats["evictions"] += 1
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.bytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    def describe(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self.bytes,
                "budget": self.budget,
                **self.stats,
            }


def _store_copy(t: Any) -> Any:
    """The stored copy of an encoder output tensor, always in host memory so the cache never holds VRAM a render
    could need (pinned when the source is on CUDA, so the copy back is asynchronous). Remembers the source device,
    which is where a hit lands unless the caller names one."""
    t = t.detach()
    device = t.device
    stored = t.to("cpu", copy = True)
    if device.type == "cuda":
        try:
            stored = stored.pin_memory()
        except Exception:  # noqa: BLE001 - pinning is optional
            pass
    stored._unsloth_src_device = device
    return stored


def _fresh(t: Any, device: Any) -> Any:
    target = device if device is not None else getattr(t, "_unsloth_src_device", t.device)
    torch = _torch()
    target = torch.device(target)
    if target.type == t.device.type and (target.index is None or target.index == t.device.index):
        return t.clone()
    non_blocking = bool(t.device.type == "cpu" and t.is_pinned())
    return t.to(target, non_blocking = non_blocking, copy = True)


def _encoder_identity(pipe: Any) -> list:
    ident = []
    for attr in _TEXT_ENCODER_ATTRS:
        module = getattr(pipe, attr, None)
        if module is not None:
            ident.append([attr, type(module).__name__, id(module)])
    return ident


def _lora_state(pipe: Any) -> Any:
    try:
        return _plain(tuple(getattr(pipe, "_unsloth_loras", ()) or ()))
    except TypeError:
        return None


def _hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys = True, separators = (",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_for(pipe: Any) -> Optional[PromptCache]:
    return getattr(pipe, "_unsloth_prompt_cache", None)


def install(
    pipe: Any,
    *,
    identity: Optional[dict] = None,
    lora_owner: Any = None,
    logger: Any = None,
) -> bool:
    """Wrap ``pipe.encode_prompt`` (when present) and register the pipe's text encoder for the
    MiniMax-H3 hook. No-op (False) when disabled or when the pipe has nothing to cache.

    ``lora_owner`` is the pipe that records the attached adapters, for a workflow pipe built with
    ``from_pipe`` around the loaded one (same text encoders, adapters tracked on the original)."""
    if not enabled() or pipe is None:
        return False
    if cache_for(pipe) is not None:
        return True
    try:
        cache = PromptCache()
    except Exception as exc:  # noqa: BLE001 - optimisation only
        _warn(logger, "setup", exc)
        return False
    if cache.budget <= 0:
        return False
    load_fp = _plain(identity or {})
    wrapped = _wrap_encode_prompt(pipe, cache, load_fp, lora_owner or pipe, logger)
    registered = _register_h3(pipe, cache, load_fp, logger)
    if not (wrapped or registered):
        return False
    try:
        pipe._unsloth_prompt_cache = cache
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "attach", exc)
        return False
    if logger is not None:
        logger.info(
            "diffusion.prompt_cache: on (%.0f MB budget; %s)",
            cache.budget / 1024**2,
            "encode_prompt" if wrapped else "modular text encoder",
        )
    return True


def release(pipe: Any) -> None:
    """Drop every cached entry now (teardown), rather than whenever the pipe is collected."""
    cache = cache_for(pipe)
    if cache is not None:
        cache.clear()


def _wrap_encode_prompt(
    pipe: Any, cache: PromptCache, load_fp: Any, lora_owner: Any, logger: Any
) -> bool:
    encode = getattr(pipe, "encode_prompt", None)
    if not callable(encode):
        return False
    try:
        signature = inspect.signature(encode)
    except (TypeError, ValueError):
        return False

    def cached_encode_prompt(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            keyed = {
                k: _plain(v) for k, v in bound.arguments.items() if k not in _KEY_EXCLUDED_ARGS
            }
            key = _hash(
                {
                    "load": load_fp,
                    "encoders": _encoder_identity(pipe),
                    "loras": _lora_state(lora_owner),
                    "args": keyed,
                }
            )
            hit = cache.get(key, bound.arguments.get("device"))
        except TypeError:
            cache.stats["bypassed"] += 1
            return encode(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - never fail a render over the cache
            _warn(logger, "lookup", exc)
            return encode(*args, **kwargs)
        if hit is not None:
            return hit
        result = encode(*args, **kwargs)
        cache.stats["misses"] += 1
        try:
            cache.put(key, result)
        except Exception as exc:  # noqa: BLE001 - a failed store only skips reuse
            _warn(logger, "store", exc)
        return result

    cached_encode_prompt.__signature__ = signature
    cached_encode_prompt.__wrapped__ = encode
    pipe.encode_prompt = cached_encode_prompt
    return True


# MiniMax-H3: the modular text-encoder steps call ``encoders.get_qwen3vl_prompt_embeds`` as a module global, and that
# call is what fires the conditioner's offload hook. One process-wide shim routes each call to the cache registered for
# its text encoder; an unregistered encoder (or a call with vision inputs) runs the original untouched.
_H3_MODULE = "diffusers.modular_pipelines.minimax_h3.encoders"
_H3_FUNC = "get_qwen3vl_prompt_embeds"
_H3_REGISTRY: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_H3_LOCK = threading.Lock()


def _register_h3(pipe: Any, cache: PromptCache, load_fp: Any, logger: Any) -> bool:
    text_encoder = _modular_text_encoder(pipe)
    if text_encoder is None:
        return False
    try:
        import importlib
        module = importlib.import_module(_H3_MODULE)
    except Exception:  # noqa: BLE001 - not an H3-capable diffusers
        return False
    with _H3_LOCK:
        original = getattr(module, _H3_FUNC, None)
        if not callable(original):
            return False
        if not getattr(original, "_unsloth_prompt_cache_shim", False):
            shim = _make_h3_shim(original, logger)
            setattr(module, _H3_FUNC, shim)
        try:
            _H3_REGISTRY[text_encoder] = (cache, load_fp)
        except TypeError:
            return False
    return True


def _modular_text_encoder(pipe: Any) -> Any:
    """The text encoder of a modular (MiniMax-H3) pipeline, or None for an ordinary pipeline."""
    if callable(getattr(pipe, "encode_prompt", None)):
        return None
    if not hasattr(pipe, "blocks") and "Modular" not in type(pipe).__name__:
        return None
    try:
        return getattr(pipe, "text_encoder", None)
    except Exception:  # noqa: BLE001 - a lazy component that is not loaded
        return None


def _make_h3_shim(original: Any, logger: Any) -> Any:
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        signature = None

    def get_qwen3vl_prompt_embeds(*args: Any, **kwargs: Any) -> Any:
        if signature is None:
            return original(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            text_encoder = arguments.get("text_encoder")
            registered = _H3_REGISTRY.get(text_encoder) if text_encoder is not None else None
        except Exception:  # noqa: BLE001
            registered = None
        if registered is None:
            return original(*args, **kwargs)
        cache, load_fp = registered
        try:
            if arguments.get("vision_inputs"):
                raise TypeError("vision inputs")
            key = _hash(
                {
                    "load": load_fp,
                    "encoder": [type(text_encoder).__name__, id(text_encoder)],
                    "token_ids": _plain(list(arguments.get("token_ids") or ())),
                    "layer": _plain(arguments.get("text_encoder_layer")),
                    "dtype": _plain(arguments.get("dtype")),
                }
            )
            hit = cache.get(key, arguments.get("device"))
        except TypeError:
            cache.stats["bypassed"] += 1
            return original(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            _warn(logger, "h3 lookup", exc)
            return original(*args, **kwargs)
        if hit is not None:
            return hit
        result = original(*args, **kwargs)
        cache.stats["misses"] += 1
        try:
            cache.put(key, result)
        except Exception as exc:  # noqa: BLE001
            _warn(logger, "h3 store", exc)
        return result

    get_qwen3vl_prompt_embeds._unsloth_prompt_cache_shim = True
    get_qwen3vl_prompt_embeds.__wrapped__ = original
    if signature is not None:
        get_qwen3vl_prompt_embeds.__signature__ = signature
    return get_qwen3vl_prompt_embeds


def _warn(logger: Any, what: str, exc: Any) -> None:
    if logger is not None:
        logger.warning("diffusion.prompt_cache: %s failed: %s", what, exc)
