# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ask whether a real accelerator exists, answered before any test spoofs torch.cuda.is_available()."""

from __future__ import annotations

_REAL_ACCELERATOR: bool | None = None
_REAL_CUDA: bool | None = None


def _ask(probe) -> bool:
    try:
        return bool(probe())
    except Exception:
        # torch.xpu without XPU support and torch.accelerator on torch < 2.6 raise here.
        return False


def _record() -> None:
    """Fill both caches in one pass, so the pre-spoof call primes CUDA and accelerator answers together."""
    global _REAL_ACCELERATOR, _REAL_CUDA
    try:
        import torch
    except Exception:
        _REAL_ACCELERATOR, _REAL_CUDA = False, False
        return
    _REAL_CUDA = _ask(lambda: hasattr(torch, "cuda") and torch.cuda.is_available())
    _REAL_ACCELERATOR = (
        _REAL_CUDA
        or _ask(lambda: hasattr(torch, "xpu") and torch.xpu.is_available())
        or _ask(lambda: hasattr(torch, "accelerator") and torch.accelerator.is_available())
    )


def has_real_accelerator() -> bool:
    """Whether a real accelerator exists; cached on first call, which conftest makes before any spoof."""
    if _REAL_ACCELERATOR is None:
        _record()
    return _REAL_ACCELERATOR


def has_real_cuda() -> bool:
    """True only for real CUDA; has_real_accelerator() is also true on XPU or NPU hosts without CUDA."""
    if _REAL_CUDA is None:
        _record()
    return _REAL_CUDA
