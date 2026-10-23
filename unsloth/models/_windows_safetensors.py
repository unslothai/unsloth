# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Windows: read safetensors shards with ``pread`` instead of a copy-on-write mmap on transformers 5.0 - 5.16.

transformers 5.x opens every shard of a checkpoint before materialising any weight, and safetensors maps each one
copy-on-write. Windows charges the whole view to the commit limit (RAM + pagefile) for as long as the shard is open,
so a 16 GB bf16 checkpoint costs 16 GB of commit on top of the loaded weights: a dtype-converting load (fp16 GPUs)
needs ~2x the model, a 4-bit load the full bf16 size, and exhausting commit is "The paging file is too small for
this operation to complete" (os error 1455) or an access violation inside torch. ``pread`` reads each tensor into
its own buffer instead. transformers 5.17 made the same switch upstream (huggingface/transformers#48341); this
backports it. llama.cpp and ComfyUI avoid the charge by mapping read-only.

Windows only, transformers >= 5.0 and < 5.17, safetensors >= 0.8 (the ``backend`` keyword). Opt out with
``UNSLOTH_SAFETENSORS_PREAD=0``.
"""

import functools
import os
import sys

__all__ = ["patch_windows_safetensors_pread"]

PREAD_ENV = "UNSLOTH_SAFETENSORS_PREAD"
# private keyword a caller passes to keep one shard on mmap (Studio's small-host route keeps an encoder paged)
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


def patch_windows_safetensors_pread() -> bool:
    """Wrap ``transformers.modeling_utils.safe_open`` so CPU shard opens default to ``backend="pread"``. Idempotent;
    returns whether the wrapper is installed."""
    if not _wanted():
        return False
    try:
        import transformers
        from transformers import modeling_utils as _mu
    except Exception:
        return False
    current = getattr(_mu, "safe_open", None)
    if current is None:
        return False
    fn = current
    for _ in range(8):
        if getattr(fn, "_unsloth_windows_pread", False):
            return True
        fn = getattr(fn, "__wrapped__", None)
        if fn is None:
            break
    forced = (os.environ.get(PREAD_ENV) or "").strip().lower() in ("1", "on", "true", "yes")
    if not forced and not _transformers_needs_it(transformers.__version__):
        return False
    if not _supports_backend():
        return False

    real_safe_open = current

    @functools.wraps(real_safe_open)
    def _pread_safe_open(*args, **kwargs):
        keep = kwargs.pop(KEEP_MMAP_KWARG, False)
        device = kwargs.get("device", args[2] if len(args) > 2 else "cpu")
        # transformers 5.13 - 5.16 pass backend="mmap" explicitly; 5.17 replaces it the same way
        if not keep and str(device) == "cpu" and kwargs.get("backend") in (None, "mmap"):
            kwargs["backend"] = "pread"
        return real_safe_open(*args, **kwargs)

    _pread_safe_open._unsloth_windows_pread = True
    _mu.safe_open = _pread_safe_open
    return True
