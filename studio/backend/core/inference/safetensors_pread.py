# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows media loads read converted safetensors shards with safetensors' ``pread`` backend instead of mmap (#9977,
#14513).

torch maps a shard copy-on-write and Windows charges the whole view to commit while transformers holds every shard
open, so a converting load commits the stored bytes next to the converted copy: LTX-2's 48.7 GB fp32 Gemma3 loaded at
bf16 needs ~73 GB, a 16.7 GB bf16 Qwen-Image encoder loaded at fp16 ~38 GB, and torch's unchecked ``MapViewOfFile``
NULL is an access violation. ``pread`` reads each tensor into its own buffer, freed once converted (what
transformers >= 5.17 does on Windows, huggingface/transformers#48341; ComfyUI and llama.cpp map read-only instead).
A shard loaded at its stored dtype keeps mmap: its mapping is the loaded weight, and the small-host route keeps it
paged from disk. ``UNSLOTH_SAFETENSORS_PREAD=0|1`` disables / forces it on any OS.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import threading
from typing import Any, Iterable, Iterator, Optional

PREAD_ENV = "UNSLOTH_SAFETENSORS_PREAD"

_lock = threading.Lock()
_depth = 0
_original: Any = None
# per load thread, so concurrent media loads never share rules: (target dtype or None = fp32-only video rule, kept dirs)
_local = threading.local()


def _active_rules() -> tuple[tuple[Optional[str], frozenset[str]], ...]:
    return getattr(_local, "rules", ())


def pread_wanted() -> bool:
    value = (os.environ.get(PREAD_ENV) or "").strip().lower()
    if value in ("0", "off", "false", "no"):
        return False
    if value in ("1", "on", "true", "yes"):
        return True
    return sys.platform == "win32"


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
    if any(path.parent.name in keep for _, keep in rules):
        return "keep"
    stored = {k: v for k, v in _safetensors_bytes_by_dtype(path).items() if k != "kept"}
    if not stored:
        return None
    dominant = max(stored.items(), key = lambda kv: kv[1])[0]
    for to, _ in rules:
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


@contextlib.contextmanager
def pread_safetensors(convert_to: Any = None, keep: Iterable[str] = ()) -> Iterator[bool]:
    """Route ``transformers.modeling_utils.safe_open`` of fp32 shards, and with ``convert_to`` of every shard stored in
    another dtype outside the ``keep`` component dirs, through ``pread``; yields whether it did. Nests; a concurrent
    load in another thread only gets the same bytes through another backend."""
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
                    installed = True
                if installed:
                    _depth += 1
        except Exception:  # noqa: BLE001 - the mmap backend still loads wherever commit allows
            installed = False
        if installed:
            to = None if convert_to is None else str(convert_to).replace("torch.", "")
            previous_rules = _active_rules()
            _local.rules = previous_rules + ((to, frozenset(keep)),)
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
