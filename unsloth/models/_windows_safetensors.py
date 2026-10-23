# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Windows, transformers 5.0 - 5.16: read safetensors shards with ``pread`` instead of a copy-on-write mmap.

transformers 5.x keeps every shard open for the whole load, and Windows charges each copy-on-write view to the commit
limit in full, so a dtype-converting load needs ~2x the model in commit (os error 1455 / access violation). Backports
huggingface/transformers#48341 (5.17). Opt out with ``UNSLOTH_SAFETENSORS_PREAD=0``.
"""

import functools
import os
import sys

__all__ = ["patch_windows_safetensors_pread"]

PREAD_ENV = "UNSLOTH_SAFETENSORS_PREAD"
# private keyword: keep this shard on mmap (Studio's small-host route keeps an encoder paged from disk)
KEEP_MMAP_KWARG = "_unsloth_keep_mmap"


def _wanted() -> bool:
    value = (os.environ.get(PREAD_ENV) or "").strip().lower()
    if value in ("0", "off", "false", "no"):
        return False
    if value in ("1", "on", "true", "yes"):
        return True
    return sys.platform == "win32"


def _transformers_needs_it(version: str) -> bool:
    from packaging.version import Version
    v = Version(version)
    # 4.x loads one shard at a time; 5.17+ already reads with pread on Windows
    return Version("5.0.0") <= v < Version("5.17.0")


def _supports_backend() -> bool:
    try:
        import inspect

        from safetensors import safe_open

        text = getattr(safe_open, "__text_signature__", None) or ""
        if text:
            return "backend" in text
        return "backend" in inspect.signature(safe_open).parameters
    except Exception:
        return False


def _installed(fn) -> bool:
    for _ in range(8):
        if fn is None:
            return False
        if getattr(fn, "_unsloth_windows_pread", False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def _wrap(real_safe_open):
    @functools.wraps(real_safe_open)
    def _pread_safe_open(*args, **kwargs):
        keep = kwargs.pop(KEEP_MMAP_KWARG, False)
        device = kwargs.get("device", args[2] if len(args) > 2 else "cpu")
        # transformers 5.13 - 5.16 pass backend="mmap" explicitly; 5.17 replaces it the same way
        if not keep and str(device) == "cpu" and kwargs.get("backend") in (None, "mmap"):
            kwargs["backend"] = "pread"
        return real_safe_open(*args, **kwargs)

    _pread_safe_open._unsloth_windows_pread = True
    return _pread_safe_open


def patch_windows_safetensors_pread() -> bool:
    """Default CPU shard opens to ``backend="pread"`` in ``modeling_utils`` and, where it has its own reference (the MTP
    loader, 5.16), ``modeling_layers``; idempotent, returns whether the main loader is wrapped."""
    if not _wanted():
        return False
    try:
        import transformers
        from transformers import modeling_utils as _mu
    except Exception:
        return False
    if getattr(_mu, "safe_open", None) is None:
        return False
    if _installed(_mu.safe_open):
        return True
    forced = (os.environ.get(PREAD_ENV) or "").strip().lower() in ("1", "on", "true", "yes")
    if not forced and not _transformers_needs_it(transformers.__version__):
        return False
    if not _supports_backend():
        return False
    modules = [_mu]
    try:
        from transformers import modeling_layers as _ml
        if getattr(_ml, "safe_open", None) is not None:
            modules.append(_ml)
    except Exception:
        pass
    for module in modules:
        if not _installed(module.safe_open):
            module.safe_open = _wrap(module.safe_open)
    return True
