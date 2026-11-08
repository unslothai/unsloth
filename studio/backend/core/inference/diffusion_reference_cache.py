# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 / Turbo: reuse an unchanged condition image's VAE latents across renders (a posterior-mode encode,
so the generator never enters). Kill switch UNSLOTH_REFERENCE_VAE_CACHE=0."""

from __future__ import annotations

import hashlib
import inspect
import os
import threading
import weakref
from collections import OrderedDict
from typing import Any, Optional

CACHE_ENV = "UNSLOTH_REFERENCE_VAE_CACHE"
_MARK = "_unsloth_reference_vae_cache"
_CLASSES = frozenset({"QwenImage21Pipeline"})
_MAX_ENTRIES = 16
_MAX_BYTES = 256 << 20
# VAE attributes that change what ``encode`` returns for the same pixels.
_VAE_STATE = (
    "use_tiling",
    "use_slicing",
    "use_framewise_encoding",
    "tile_sample_min_height",
    "tile_sample_min_width",
    "tile_sample_stride_height",
    "tile_sample_stride_width",
)

STATS = {"hits": 0, "misses": 0, "bypassed": 0}


def disabled() -> bool:
    return (os.environ.get(CACHE_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _mode_encode(fn: Any) -> bool:
    try:
        src = inspect.getsource(inspect.unwrap(fn))
    except (OSError, TypeError):
        return False
    return 'sample_mode="argmax"' in src and '"sample"' not in src and ".sample(" not in src


def _overrides(module: Any) -> tuple:
    own = getattr(module, "__dict__", {})
    out = []
    for name, value in own.items():
        if name in ("encode", "_encode", "forward", "tiled_encode") and callable(value):
            out.append((name, id(getattr(value, "__func__", value))))
        elif name.startswith("_unsloth_") and isinstance(
            value, (bool, int, float, str, type(None))
        ):
            out.append((name, value))
    return tuple(sorted(out, key = lambda item: item[0]))


def _vae_key(vae: Any) -> Optional[tuple]:
    try:
        param = next(vae.parameters())
    except Exception:  # noqa: BLE001 - no parameters / an unreadable VAE is never cached
        return None
    # Not the device or address: offload hooks move parameters between renders.
    state = tuple(repr(getattr(vae, name, None)) for name in _VAE_STATE)
    encoder = getattr(vae, "encoder", None)
    return (
        id(vae),
        str(param.dtype),
        state,
        _overrides(vae),
        _overrides(encoder) if encoder is not None else (),
    )


def _pixels_key(image: Any) -> Optional[tuple]:
    import torch

    if not isinstance(image, torch.Tensor) or image.is_complex() or image.numel() == 0:
        return None
    raw = image.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy()
    digest = hashlib.blake2b(raw.data, digest_size = 20).hexdigest()
    return (digest, tuple(image.shape), str(image.dtype), str(image.device))


class _Cache:
    def __init__(self) -> None:
        self.entries: "OrderedDict[tuple, Any]" = OrderedDict()
        self.bytes = 0
        self.vae_ref: Any = None
        self.lock = threading.Lock()

    def clear(self) -> None:
        self.entries.clear()
        self.bytes = 0

    def get(self, key: tuple) -> Any:
        with self.lock:
            hit = self.entries.get(key)
            if hit is not None:
                self.entries.move_to_end(key)
            return hit

    def put(self, key: tuple, value: Any) -> None:
        size = value.numel() * value.element_size()
        if size > _MAX_BYTES:
            return
        with self.lock:
            old = self.entries.pop(key, None)
            if old is not None:
                self.bytes -= old.numel() * old.element_size()
            self.entries[key] = value
            self.bytes += size
            while self.entries and (len(self.entries) > _MAX_ENTRIES or self.bytes > _MAX_BYTES):
                _, dropped = self.entries.popitem(last = False)
                self.bytes -= dropped.numel() * dropped.element_size()


def install(pipe: Any, logger: Any = None) -> bool:
    if disabled() or type(pipe).__name__ not in _CLASSES:
        return False
    if getattr(pipe, _MARK, None) is not None:
        return True
    stock_fn = getattr(type(pipe), "_encode_vae_image", None)
    if not callable(stock_fn) or not _mode_encode(stock_fn):
        if logger is not None:
            logger.info(
                "diffusion.reference_cache: off (the reference encode is not a posterior mode)"
            )
        return False
    stock = getattr(pipe, "_encode_vae_image")
    cache = _Cache()
    pipe_ref = weakref.ref(pipe)

    def _encode_vae_image(image: Any, generator: Any) -> Any:
        import torch

        owner = pipe_ref()
        vae = getattr(owner, "vae", None) if owner is not None else None
        if disabled() or vae is None or torch.is_grad_enabled():
            STATS["bypassed"] += 1
            return stock(image, generator)
        vae_key = _vae_key(vae)
        pixels = _pixels_key(image)
        if vae_key is None or pixels is None:
            STATS["bypassed"] += 1
            return stock(image, generator)
        if cache.vae_ref is None or cache.vae_ref() is not vae:
            with cache.lock:
                cache.clear()
                cache.vae_ref = weakref.ref(vae)
        key = (pixels, vae_key)
        hit = cache.get(key)
        if hit is not None:
            STATS["hits"] += 1
            return hit.to(device = image.device, copy = True)
        STATS["misses"] += 1
        out = stock(image, generator)
        if isinstance(out, torch.Tensor):
            cache.put(key, out.detach().to("cpu", copy = True))
        return out

    _encode_vae_image.__wrapped__ = stock
    pipe._encode_vae_image = _encode_vae_image
    setattr(pipe, _MARK, cache)
    if logger is not None:
        logger.info("diffusion.reference_cache: condition-image VAE latents reused across renders")
    return True


def uninstall(pipe: Any) -> None:
    if getattr(pipe, _MARK, None) is None:
        return
    try:
        del pipe._encode_vae_image
    except AttributeError:
        pass
    try:
        delattr(pipe, _MARK)
    except AttributeError:
        pass
