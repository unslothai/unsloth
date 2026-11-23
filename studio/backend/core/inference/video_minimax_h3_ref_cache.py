# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 condition latent cache: a re-render with the same reference / keyframe pixels skips the VAE encode.

- Key: sha256 of the exact uint8 pixels the encode receives (so the file bytes, decode, trim, frame sampling, resize
  and crop are all in it), their shape and device, the pixel normalisation, the latent mode and encode seed, the VAE
  config, the encoder's parameter dtypes and every patched forward / tiling attribute on the VAE and its encoder.
- Identity: each entry holds a weakref to the VAE it came from and only that object can hit it; a model load or unload
  clears the cache, so nothing crosses a model / VAE load.
- Value: the exact float32 CPU tensor the encode returned (the caller always moves it), handed out as a copy.
- Bound: LRU, ``UNSLOTH_H3_REF_LATENT_CACHE_ENTRIES`` entries (default 12, one full Ref2VA request) and
  ``UNSLOTH_H3_REF_LATENT_CACHE_MB`` host MiB (default 1024); an entry over the byte bound is not kept.
- ``UNSLOTH_H3_REF_LATENT_CACHE=0`` disables it.
"""

from __future__ import annotations

import hashlib
import os
import threading
import weakref
from collections import OrderedDict
from typing import Any, Callable, Optional

H3_REF_LATENT_CACHE_ENV = "UNSLOTH_H3_REF_LATENT_CACHE"
H3_REF_LATENT_CACHE_ENTRIES_ENV = "UNSLOTH_H3_REF_LATENT_CACHE_ENTRIES"
H3_REF_LATENT_CACHE_MB_ENV = "UNSLOTH_H3_REF_LATENT_CACHE_MB"
_DEFAULT_ENTRIES = 12
_DEFAULT_MB = 1024
# above this many pixel bytes the digest is a sha256 over per-slice sha256s, one thread per slice
_PARALLEL_HASH_BYTES = 64 * 2**20
_HASH_THREADS = 8

_LOCK = threading.Lock()
_ENTRIES: "OrderedDict[tuple, tuple]" = OrderedDict()
_STATS = {"hits": 0, "misses": 0, "evictions": 0, "clears": 0}


def enabled() -> bool:
    raw = (os.environ.get(H3_REF_LATENT_CACHE_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int((os.environ.get(name) or "").strip() or default))
    except ValueError:
        return default


def _limits() -> tuple[int, int]:
    return (
        _int_env(H3_REF_LATENT_CACHE_ENTRIES_ENV, _DEFAULT_ENTRIES),
        _int_env(H3_REF_LATENT_CACHE_MB_ENV, _DEFAULT_MB) * 2**20,
    )


def _pixel_digest(pixels: Any) -> str:
    """sha256 of the pixel bytes, row-major; slices hash in parallel (hashlib drops the GIL)."""
    # copied to host before the row-major layout is made, so no second device copy of the clip is allocated
    host = pixels.detach().cpu().contiguous()
    data = memoryview(host.view(-1).numpy()).cast("B")
    if len(data) <= _PARALLEL_HASH_BYTES:
        return hashlib.sha256(data).hexdigest()
    from concurrent.futures import ThreadPoolExecutor

    step = -(-len(data) // _HASH_THREADS)
    parts = [data[i : i + step] for i in range(0, len(data), step)]
    with ThreadPoolExecutor(max_workers = len(parts)) as pool:
        digests = list(pool.map(lambda p: hashlib.sha256(p).digest(), parts))
    return "tree:" + hashlib.sha256(b"".join(digests)).hexdigest()


def _plain(value: Any) -> Any:
    """A hashable stand-in for one attribute; callables by identity, tensors and modules skipped (weights are keyed
    by dtype, and a reload clears the cache)."""
    import torch

    if value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device)):
        return value
    if isinstance(value, (tuple, list)):
        items = tuple(_plain(v) for v in value)
        return None if any(i is _SKIP for i in items) else items
    if isinstance(value, (torch.Tensor, torch.nn.Module)):
        return _SKIP
    if callable(value):
        return (
            "callable",
            id(getattr(value, "__func__", value)),
            getattr(value, "__qualname__", None),
        )
    return _SKIP


_SKIP = object()


# The decode side's state on the shared VAE object: a decode wrapper re-bound every render (``decode`` /
# ``_decode``), and private counters, last tile, canvas / OOM state. The encode never reads any of it, so keying on
# it would make every render miss. Everything else, encode wrappers included, is keyed.
_UNKEYED_VAE_ATTR_PARTS = ("canvas", "_calls", "_stats", "oom_active", "logger", "strips")


def _unkeyed(name: str) -> bool:
    return "decode" in name or (
        name.startswith("_") and any(p in name for p in _UNKEYED_VAE_ATTR_PARTS)
    )


def _attrs(obj: Any, *, skip_bookkeeping: bool = False) -> tuple:
    out = []
    for name, value in sorted(getattr(obj, "__dict__", {}).items()):
        if name in ("_parameters", "_buffers", "_modules"):
            continue
        if skip_bookkeeping and _unkeyed(name):
            continue
        plain = _plain(value)
        if plain is not _SKIP:
            out.append((name, plain))
    return tuple(out)


def _vae_fingerprint(vae: Any) -> tuple:
    config = getattr(vae, "config", None)
    try:
        config_items = tuple(sorted((k, repr(v)) for k, v in dict(config).items()))
    except Exception:  # noqa: BLE001 - an unreadable config never hits
        config_items = ("unreadable", id(config))
    encoder = getattr(vae, "encoder", None)
    param_dtypes = ()
    if encoder is not None and hasattr(encoder, "named_parameters"):
        param_dtypes = tuple(sorted({str(p.dtype) for _, p in encoder.named_parameters()}))
    quant = getattr(vae, "quant_conv", None)
    if quant is not None and hasattr(quant, "parameters"):
        param_dtypes = param_dtypes + tuple(sorted({str(p.dtype) for p in quant.parameters()}))
    cls = type(vae)
    methods = tuple(
        (name, id(getattr(cls, name, None))) for name in ("encode", "_encode", "_encode_clip")
    )
    return (
        cls.__module__,
        cls.__qualname__,
        methods,
        config_items,
        param_dtypes,
        _attrs(vae, skip_bookkeeping = True),
        _attrs(encoder) if encoder is not None else (),
    )


def condition_key(
    vae: Any, pixels: Any, pixel_mean: Any, pixel_std: Any, encode_seed: Any, recipe: tuple
) -> Optional[tuple]:
    """Everything the encode's output depends on, or None when it cannot be keyed (then the encode always runs)."""
    import torch

    if not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.uint8:
        return None
    try:
        return (
            id(vae),
            _vae_fingerprint(vae),
            tuple(pixels.shape),
            str(pixels.device),
            _pixel_digest(pixels),
            tuple(float(v) for v in pixel_mean),
            tuple(float(v) for v in pixel_std),
            int(encode_seed),
            recipe,
        )
    except Exception:  # noqa: BLE001 - keying is best effort, the encode is the fallback
        return None


def _bytes(value: Any) -> int:
    return value.numel() * value.element_size()


def lookup(key: tuple, vae: Any) -> Optional[Any]:
    with _LOCK:
        entry = _ENTRIES.get(key)
        if entry is None or entry[0]() is not vae:
            if entry is not None:
                del _ENTRIES[key]  # id(vae) reused by another object
            _STATS["misses"] += 1
            return None
        _ENTRIES.move_to_end(key)
        _STATS["hits"] += 1
        return entry[1].clone()


def store(key: tuple, vae: Any, latents: Any) -> bool:
    import torch

    if not isinstance(latents, torch.Tensor) or latents.device.type != "cpu":
        return False
    max_entries, max_bytes = _limits()
    size = _bytes(latents)
    if max_entries <= 0 or size > max_bytes:
        return False
    try:
        ref = weakref.ref(vae)
    except TypeError:
        return False
    with _LOCK:
        _ENTRIES[key] = (ref, latents.detach().clone())
        _ENTRIES.move_to_end(key)
        total = sum(_bytes(v) for _, v in _ENTRIES.values())
        while _ENTRIES and (len(_ENTRIES) > max_entries or total > max_bytes):
            _, (_, old) = _ENTRIES.popitem(last = False)
            total -= _bytes(old)
            _STATS["evictions"] += 1
    return True


def clear(reason: str = "", logger: Any = None) -> int:
    """Drops every entry (model load / unload, VAE change); returns how many were held."""
    with _LOCK:
        held = len(_ENTRIES)
        _ENTRIES.clear()
        _STATS["clears"] += 1
    if held and logger is not None:
        logger.info("video.h3_ref_latent_cache: cleared %d entries (%s)", held, reason or "clear")
    return held


def stats() -> dict:
    with _LOCK:
        return dict(
            _STATS,
            entries = len(_ENTRIES),
            bytes = sum(_bytes(v) for _, v in _ENTRIES.values()),
            enabled = enabled(),
        )


def cached_encode(
    encode: Callable[[], Any],
    vae: Any,
    pixels: Any,
    pixel_mean: Any,
    pixel_std: Any,
    encode_seed: Any,
    recipe: tuple,
    logger: Any = None,
) -> Any:
    """``encode()`` through the cache; with the cache off, or an unkeyable call, exactly ``encode()``."""
    if not enabled():
        return encode()
    import time

    started = time.perf_counter()
    key = condition_key(vae, pixels, pixel_mean, pixel_std, encode_seed, recipe)
    if key is None:
        return encode()
    keyed = time.perf_counter()
    hit = lookup(key, vae)
    if hit is not None:
        if logger is not None:
            logger.info(
                "video.h3_ref_latent_cache: hit %s frames=%d key %.3f s",
                tuple(hit.shape),
                int(pixels.shape[2]) if pixels.ndim == 5 else 1,
                keyed - started,
            )
        return hit
    why = _miss_reason(key) if logger is not None else ""
    latents = encode()
    kept = store(key, vae, latents)
    if logger is not None:
        logger.info(
            "video.h3_ref_latent_cache: miss frames=%d key %.3f s encode %.3f s%s%s",
            int(pixels.shape[2]) if pixels.ndim == 5 else 1,
            keyed - started,
            time.perf_counter() - keyed,
            "" if kept else " (not kept)",
            why,
        )
    return latents


_KEY_FIELDS = (
    "vae",
    "vae_state",
    "shape",
    "device",
    "pixels",
    "pixel_mean",
    "pixel_std",
    "encode_seed",
    "recipe",
)
_FINGERPRINT_FIELDS = (
    "module",
    "class",
    "methods",
    "config",
    "dtypes",
    "vae_attrs",
    "encoder_attrs",
)


def _diff_names(old: Any, new: Any) -> list:
    if (
        isinstance(old, tuple)
        and isinstance(new, tuple)
        and all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in old + new)
    ):
        a, b = dict(old), dict(new)
        return sorted(k for k in set(a) | set(b) if a.get(k, _SKIP) != b.get(k, _SKIP))
    return []


def _miss_reason(key: tuple) -> str:
    """Which part of the key changed, when the same pixels were encoded before (a diagnostic only)."""
    with _LOCK:
        same = [k for k in _ENTRIES if len(k) == len(key) and k[4] == key[4]]
    if not same:
        return ""
    old = same[-1]
    parts = []
    for i, name in enumerate(_KEY_FIELDS):
        if old[i] == key[i]:
            continue
        if name == "vae_state":
            for j, sub in enumerate(_FINGERPRINT_FIELDS):
                if old[1][j] != key[1][j]:
                    names = _diff_names(old[1][j], key[1][j])
                    parts.append(f"{sub}{names}" if names else sub)
        else:
            parts.append(name)
    return f" (same pixels, changed: {', '.join(parts)})"
