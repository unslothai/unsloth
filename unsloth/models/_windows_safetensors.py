# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""CPU shard opens read with safetensors pread, not mmap: on Windows with transformers 5.0-5.16 (commit limit) and on
CUDA unified-memory GPUs (DGX Spark / GB10, Jetson: copying mmap views to the GPU is many times slower than pread).
UNSLOTH_SAFETENSORS_PREAD=0 off, =1 everywhere."""

import functools
import os
import platform
import sys

__all__ = ["patch_windows_safetensors_pread", "pread_backend_kwargs"]

PREAD_ENV = "UNSLOTH_SAFETENSORS_PREAD"
# private keyword: keep this shard on mmap (Studio's small-host route keeps an encoder paged from disk)
KEEP_MMAP_KWARG = "_unsloth_keep_mmap"


def _env():
    value = (os.environ.get(PREAD_ENV) or "").strip().lower()
    if value in ("0", "off", "false", "no"):
        return False
    if value in ("1", "on", "true", "yes"):
        return True
    return None


def _wanted() -> bool:
    value = _env()
    if value is not None:
        return value
    return sys.platform == "win32" or _may_be_unified_memory()


def _may_be_unified_memory() -> bool:
    # CUDA integrated GPUs (GB10, Thor, Orin, Xavier) are all ARM parts (Linux / Windows, never macOS); no CUDA call.
    return sys.platform != "darwin" and platform.machine().strip().lower() in ("aarch64", "arm64")


_UMA = None


def _unified_memory_now() -> bool:
    """Every visible CUDA device is integrated (shares host RAM). Never initialises CUDA: a process that has not
    created a context yet is not loading to the GPU, so it keeps mmap. ``UNSLOTH_FORCE_UMA=1|0`` overrides."""
    global _UMA
    forced = os.environ.get("UNSLOTH_FORCE_UMA")
    if forced in ("0", "1"):
        return forced == "1"
    if _UMA is not None:
        return _UMA
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        if getattr(torch.version, "hip", None) or not torch.cuda.is_initialized():
            return False
        count = torch.cuda.device_count()
        _UMA = count > 0 and all(
            getattr(torch.cuda.get_device_properties(i), "is_integrated", 0) for i in range(count)
        )
    except Exception:
        return False
    return _UMA


def _transformers_needs_it(version: str) -> bool:
    from packaging.version import Version
    v = Version(version)
    # 4.x loads one shard at a time; 5.17+ already reads with pread on Windows
    return Version("5.0.0") <= v < Version("5.17.0")


@functools.lru_cache(maxsize = None)
def _supports_backend() -> bool:
    # safetensors 0.8.0 added backend=; older releases raise TypeError on it. Read the extension's own class:
    # safetensors.safe_open may be wrapped by another library as (*args, **kwargs).
    try:
        from safetensors._safetensors_rust import safe_open
        text = getattr(safe_open, "__text_signature__", None) or ""
        if text:
            return "backend" in text
    except Exception:
        pass
    try:
        import safetensors
        from packaging.version import Version
        return Version(safetensors.__version__) >= Version("0.8.0rc1")
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


def pread_backend_kwargs(device = "cpu") -> dict:
    """``{"backend": "pread"}`` for a CPU open whose tensors are each read once and copied to the GPU, on a CUDA
    unified-memory GPU where pread beats mmap, when safetensors has the keyword; else ``{}`` (stock mmap). Discrete
    GPUs keep mmap (faster there, Windows included). Not for tensors that stay on the host: pread turns each into a
    private copy instead of a page-cache view."""
    value = _env()
    if value is False or str(device) != "cpu":
        return {}
    if value is None and not _unified_memory_now():
        return {}
    return {"backend": "pread"} if _supports_backend() else {}


def _wrap(real_safe_open, always):
    @functools.wraps(real_safe_open)
    def _pread_safe_open(*args, **kwargs):
        keep = kwargs.pop(KEEP_MMAP_KWARG, False)
        device = kwargs.get("device", args[2] if len(args) > 2 else "cpu")
        # transformers 5.13 - 5.16 pass backend="mmap" explicitly; 5.17 replaces it the same way
        if (
            not keep
            and str(device) == "cpu"
            and kwargs.get("backend") in (None, "mmap")
            and _env() is not False
            and (always or _unified_memory_now())
        ):
            kwargs["backend"] = "pread"
        return real_safe_open(*args, **kwargs)

    _pread_safe_open._unsloth_windows_pread = True
    return _pread_safe_open


def patch_windows_safetensors_pread() -> bool:
    """Default CPU shard opens to ``backend="pread"`` in ``modeling_utils`` and, where it has its own reference (the MTP
    loader, 5.16), ``modeling_layers``: always on Windows with transformers 5.0-5.16, and on an ARM host only while
    every CUDA device is integrated (checked per open, after CUDA is up). Idempotent, returns whether the main loader
    is wrapped."""
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
    forced = _env() is True
    # Windows' commit limit needs every open on pread; unified memory is decided per open (CUDA must be up first)
    always = forced or (
        sys.platform == "win32" and _transformers_needs_it(transformers.__version__)
    )
    if not always and not _may_be_unified_memory():
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
            module.safe_open = _wrap(module.safe_open, always)
    return True
