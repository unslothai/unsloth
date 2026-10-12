# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows video loads read fp32 safetensors shards with safetensors' ``pread`` backend instead of mmap (#9977).

torch maps a shard copy-on-write and Windows charges the whole view to commit while transformers holds every shard
open: LTX-2's 48.7 GB fp32 Gemma3 loaded at bf16 needs ~73 GB, and torch's unchecked ``MapViewOfFile`` NULL is an
access violation on a 32 GB host. bf16 / fp16 shards keep mmap: their mapping is the loaded weight.
``UNSLOTH_SAFETENSORS_PREAD=0|1`` disables / forces it on any OS.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import threading
from typing import Any, Iterator

PREAD_ENV = "UNSLOTH_SAFETENSORS_PREAD"

_lock = threading.Lock()
_depth = 0
_original: Any = None


def pread_wanted() -> bool:
    value = (os.environ.get(PREAD_ENV) or "").strip().lower()
    if value in ("0", "off", "false", "no"):
        return False
    if value in ("1", "on", "true", "yes"):
        return True
    return sys.platform == "win32"


def _supports_backend(safe_open: Any) -> bool:
    # safetensors < 0.8 has no backend keyword, and an unknown keyword is a TypeError on every load.
    return "backend" in (getattr(safe_open, "__text_signature__", None) or "")


def _stored_mostly_fp32(filename: Any) -> bool:
    from pathlib import Path

    from .diffusion_small_host import _safetensors_bytes_by_dtype

    stored = _safetensors_bytes_by_dtype(Path(filename))
    return stored.get("float32", 0) * 2 > sum(stored.values())


def _with_pread(safe_open: Any) -> Any:
    @functools.wraps(safe_open)
    def pread_safe_open(filename: Any, *args: Any, **kwargs: Any) -> Any:
        # transformers 5.6 - 5.16 pass backend="mmap" explicitly; 5.17 already picks pread on Windows (#48341).
        if kwargs.get("backend") in (None, "mmap") and _stored_mostly_fp32(filename):
            kwargs["backend"] = "pread"
        return safe_open(filename, *args, **kwargs)

    return pread_safe_open


@contextlib.contextmanager
def pread_safetensors() -> Iterator[bool]:
    """Route ``transformers.modeling_utils.safe_open`` of fp32 shards through ``pread``; yields whether it did. Nests; a
    concurrent load in another thread only gets the same bytes through another backend."""
    global _depth, _original
    installed = False
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
    try:
        yield installed
    finally:
        if installed:
            with _lock:
                _depth -= 1
                if not _depth:
                    import transformers.modeling_utils as modeling_utils
                    if getattr(modeling_utils.safe_open, "__wrapped__", None) is _original:
                        modeling_utils.safe_open = _original
                    _original = None
