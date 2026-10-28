# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Early cuInit via ctypes before the torch CUDA probe, releasing the GIL; idempotent, never raises;
UNSLOTH_STUDIO_DISABLE_CUDA_PRIME=1 disables it"""

from __future__ import annotations

import os
import sys
import threading

DISABLE_ENV_VAR = "UNSLOTH_STUDIO_DISABLE_CUDA_PRIME"

_claim_lock = threading.Lock()
_claimed = False
_done = threading.Event()


def _driver_names() -> tuple[str, ...]:
    if sys.platform.startswith("linux"):
        return ("libcuda.so.1", "libcuda.so")
    if sys.platform == "win32":
        return ("nvcuda.dll",)
    return ()


def _should_prime(torch_module) -> bool:
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    # The user asked torch to answer from NVML so the process is not cuInit'ed here (fork safety).
    if os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") == "1":
        return False
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1"):
        return False
    version = getattr(torch_module, "version", None)
    if not getattr(version, "cuda", None) or getattr(version, "hip", None):
        return False
    try:
        return not torch_module.cuda.is_initialized()
    except Exception:
        return False


def _cu_init() -> bool:
    import ctypes
    for name in _driver_names():
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        try:
            fn = lib.cuInit
            fn.argtypes = [ctypes.c_uint]
            fn.restype = ctypes.c_int
            fn(0)
        except Exception:
            return False
        return True
    return False


def prime_cuda_driver(torch_module = None) -> bool:
    """Run ``cuInit`` off the GIL once per process, before torch's first CUDA probe. True iff this call ran it."""
    global _claimed
    if _done.is_set():
        return False
    with _claim_lock:
        first = not _claimed
        _claimed = True
    if not first:
        _done.wait()
        return False
    try:
        if not _driver_names():
            return False
        if torch_module is None:
            # Never imports torch: a caller still inside `import torch` would deadlock its waiters.
            torch_module = sys.modules.get("torch")
            if torch_module is None:
                return False
        if not _should_prime(torch_module):
            return False
        return _cu_init()
    except Exception:
        return False
    finally:
        _done.set()


def _after_fork_in_child() -> None:
    # The thread that would set _done exists only in the parent, and cuInit is not fork-safe: the child never primes.
    global _claim_lock, _claimed, _done
    _claim_lock = threading.Lock()
    _claimed = True
    _done = threading.Event()
    _done.set()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child = _after_fork_in_child)


def _reset_for_tests() -> None:
    global _claimed
    with _claim_lock:
        _claimed = False
        _done.clear()
