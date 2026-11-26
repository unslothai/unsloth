# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows converting loads, and every load on a Linux ROCm integrated GPU (unified memory), read safetensors shards
with pread instead of mmap: no copy-on-write commit on Windows, no mapped shard held in the shared pool until its last
tensor moved on an APU. transformers' ``safe_open`` and diffusers' ``safetensors.torch.load_file`` are routed inside a
scope (diffusers' only on a ROCm integrated GPU). UNSLOTH_SAFETENSORS_PREAD=0 turns it off everywhere, =1 on.

On a ROCm integrated GPU diffusers' reader also drops each tensor's file pages from the page cache once it is read
(POSIX_FADV_DONTNEED; the pages are clean, the copy is already in the tensor), so a cold load does not leave the whole
checkpoint cached in the shared pool. UNSLOTH_SAFETENSORS_FADVISE=0 keeps the pages."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import struct
import sys
import threading
from typing import Any, Iterable, Iterator, Optional

PREAD_ENV = "UNSLOTH_SAFETENSORS_PREAD"
FADVISE_ENV = "UNSLOTH_SAFETENSORS_FADVISE"

_lock = threading.Lock()
_depth = 0
_original: Any = None
_original_load_file: Any = None
# per load thread, so concurrent media loads never share rules: (target dtype or None = fp32-only video rule, kept dirs,
# every shard)
_local = threading.local()
# DONTNEED calls issued by this process (diagnostic; the load path never reads it)
fadvise_calls = 0


def _active_rules() -> tuple[tuple[Optional[str], frozenset[str]], ...]:
    return getattr(_local, "rules", ())


def pread_wanted() -> bool:
    value = (os.environ.get(PREAD_ENV) or "").strip().lower()
    if value in ("0", "off", "false", "no"):
        return False
    if value in ("1", "on", "true", "yes"):
        return True
    return sys.platform == "win32" or rocm_integrated()


def rocm_integrated() -> bool:
    """Linux ROCm on an integrated GPU (unified memory: VRAM is host RAM). CUDA, discrete cards and anything
    unclassifiable answer False. Never raises."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import torch

        from .rocm_bf16 import is_rocm_torch

        if not is_rocm_torch(torch) or not torch.cuda.is_available():
            return False
        from .host_pinning import device_is_integrated

        return device_is_integrated()
    except Exception:  # noqa: BLE001 - unclassifiable: keep mmap
        return False


def _supports_backend(safe_open: Any) -> bool:
    # safetensors < 0.8 has no backend keyword (TypeError); wrappers drop __text_signature__, so read the innermost.
    fn = safe_open
    for _ in range(8):
        inner = getattr(fn, "__wrapped__", None)
        if inner is None:
            break
        fn = inner
    return "backend" in (getattr(fn, "__text_signature__", None) or "")


def pread_available() -> bool:
    """Whether a scope opened now routes shards through ``pread`` (Windows, safetensors >= 0.8, not disabled)."""
    if not pread_wanted():
        return False
    try:
        from safetensors import safe_open
        return _supports_backend(safe_open)
    except Exception:  # noqa: BLE001
        return False


def _backend_for(filename: Any) -> Optional[str]:
    """ "pread", "keep" (a component the small-host route loads at its stored dtype) or None (caller's default)."""
    from pathlib import Path

    from .diffusion_small_host import _safetensors_bytes_by_dtype

    path = Path(filename)
    rules = _active_rules()
    if any(path.parent.name in keep for _, keep, _ in rules):
        return "keep"
    stored = {k: v for k, v in _safetensors_bytes_by_dtype(path).items() if k != "kept"}
    if not stored:
        return None
    dominant = max(stored.items(), key = lambda kv: kv[1])[0]
    for to, _, every in rules:
        if every:
            return "pread"
        if to is None and stored.get("float32", 0) * 2 > sum(stored.values()):
            return "pread"
        if to is not None and dominant in ("bfloat16", "float16", "float32") and dominant != to:
            return "pread"
    return None


def _core_pread_below(safe_open: Any) -> bool:
    # unsloth's Windows pread patch (transformers < 5.17) may sit underneath
    fn = safe_open
    for _ in range(8):
        if fn is None:
            return False
        if getattr(fn, "_unsloth_windows_pread", False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def _with_pread(safe_open: Any) -> Any:
    @functools.wraps(safe_open)
    def pread_safe_open(filename: Any, *args: Any, **kwargs: Any) -> Any:
        # transformers 5.13 - 5.16 pass backend="mmap" explicitly; 5.17 already picks pread on Windows (#48341).
        if not _active_rules():
            return safe_open(filename, *args, **kwargs)
        choice = _backend_for(filename)
        if choice == "pread" and kwargs.get("backend") in (None, "mmap"):
            kwargs["backend"] = "pread"
        elif choice == "keep" and kwargs.get("backend") == "pread":
            # transformers >= 5.17 asks for pread on Windows itself; the small-host route needs this shard paged
            kwargs["backend"] = "mmap"
        if choice != "pread" and _core_pread_below(safe_open):
            # this scope decides inside a media load, over unsloth's patch too
            kwargs["_unsloth_keep_mmap"] = True
        return safe_open(filename, *args, **kwargs)

    return pread_safe_open


def _with_pread_load_file(load_file: Any, safe_open: Any) -> Any:
    """``safetensors.torch.load_file`` (diffusers' shard reader) through ``safe_open(backend="pread")`` for shards the
    active scope routes; everything else, and every call outside a scope, keeps the original."""

    @functools.wraps(load_file)
    def pread_load_file(
        filename: Any,
        device: Any = "cpu",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if args or kwargs or not _active_rules() or _backend_for(filename) != "pread":
            return load_file(filename, device, *args, **kwargs)
        out = {}
        with (
            safe_open(filename, framework = "pt", device = device, backend = "pread") as fh,
            _page_dropper(filename) as drop,
        ):
            for key in fh.keys():
                out[key] = fh.get_tensor(key)
                drop(key)
        return out

    pread_load_file._unsloth_pread = True
    return pread_load_file


def _tensor_spans(filename: Any) -> dict[str, tuple[int, int]]:
    """Absolute (offset, length) of every tensor's bytes, from the safetensors header."""
    with open(filename, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(size))
    base = 8 + size
    spans = {}
    for key, entry in header.items():
        if key == "__metadata__" or not isinstance(entry, dict):
            continue
        begin, end = entry["data_offsets"]
        spans[key] = (base + int(begin), int(end) - int(begin))
    return spans


def fadvise_wanted() -> bool:
    """Drop read tensors' pages: on a Linux ROCm integrated GPU (the page cache is the GPU's pool), unless disabled."""
    value = (os.environ.get(FADVISE_ENV) or "").strip().lower()
    if value in ("0", "off", "false", "no") or not hasattr(os, "posix_fadvise"):
        return False
    return rocm_integrated()


@contextlib.contextmanager
def _page_dropper(filename: Any) -> Iterator[Any]:
    """Yields ``drop(key)``: advise the kernel the tensor's file pages are not needed again. A no-op where unwanted or
    the header / file cannot be read; never raises."""
    fd = None
    spans: dict[str, tuple[int, int]] = {}
    if fadvise_wanted():
        try:
            spans = _tensor_spans(filename)
            fd = os.open(filename, os.O_RDONLY)
        except Exception:  # noqa: BLE001 - unreadable header: keep the pages, the load is unchanged
            fd, spans = None, {}

    def drop(key: str) -> None:
        global fadvise_calls
        span = spans.get(key)
        if fd is None or span is None or span[1] <= 0:
            return
        try:
            os.posix_fadvise(fd, span[0], span[1], os.POSIX_FADV_DONTNEED)
            fadvise_calls += 1
        except OSError:
            pass

    try:
        yield drop
    finally:
        if fd is not None:
            os.close(fd)


def _install_load_file() -> None:
    global _original_load_file
    try:
        import safetensors.torch as st_torch
        from safetensors import safe_open

        if not _supports_backend(safe_open):
            return
        current = getattr(st_torch, "load_file", None)
        if current is not None and not getattr(current, "_unsloth_pread", False):
            _original_load_file = current
            st_torch.load_file = _with_pread_load_file(current, safe_open)
    except Exception:  # noqa: BLE001 - diffusers keeps the mmap reader
        _original_load_file = None


def _restore_load_file() -> None:
    global _original_load_file
    try:
        import safetensors.torch as st_torch
        if (
            _original_load_file is not None
            and getattr(st_torch.load_file, "__wrapped__", None) is _original_load_file
        ):
            st_torch.load_file = _original_load_file
    except Exception:  # noqa: BLE001
        pass
    _original_load_file = None


@contextlib.contextmanager
def pread_safetensors(
    convert_to: Any = None,
    keep: Iterable[str] = (),
    every_shard: bool = False,
) -> Iterator[bool]:
    """Route ``transformers.modeling_utils.safe_open`` of fp32 shards, and with ``convert_to`` of every shard stored in
    another dtype outside the ``keep`` component dirs, through ``pread``; yields whether it did. ``every_shard`` reads
    same-dtype shards through ``pread`` too: on an integrated GPU the device copy is host commit, and a mapped shard
    stays charged until its last tensor has moved. Nests; a concurrent load in another thread only gets the same bytes
    through another backend."""
    global _depth, _original
    installed = False
    previous_rules = None
    if pread_wanted():
        try:
            import transformers.modeling_utils as modeling_utils
            with _lock:
                current = getattr(modeling_utils, "safe_open", None)
                if _depth:
                    installed = True
                elif current is not None and _supports_backend(current):
                    _original = current
                    modeling_utils.safe_open = _with_pread(current)
                    if rocm_integrated():
                        # diffusers' reader too, only where the shared pool is the point (CUDA / Windows untouched)
                        _install_load_file()
                    installed = True
                if installed:
                    _depth += 1
        except Exception:  # noqa: BLE001 - the mmap backend still loads wherever commit allows
            installed = False
        if installed:
            to = None if convert_to is None else str(convert_to).replace("torch.", "")
            previous_rules = _active_rules()
            _local.rules = previous_rules + ((to, frozenset(keep), bool(every_shard)),)
    try:
        yield installed
    finally:
        if installed:
            _local.rules = previous_rules
            with _lock:
                _depth -= 1
                if not _depth:
                    import transformers.modeling_utils as modeling_utils

                    if getattr(modeling_utils.safe_open, "__wrapped__", None) is _original:
                        modeling_utils.safe_open = _original
                    _original = None
                    _restore_load_file()
