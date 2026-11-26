# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ask whether a real accelerator exists, answered before any test spoofs torch.cuda.is_available()."""

from __future__ import annotations

_REAL_ACCELERATOR: bool | None = None
_REAL_CUDA: bool | None = None
_REAL_XPU: bool | None = None


def _ask(probe) -> bool:
    try:
        return bool(probe())
    except Exception:
        # torch.xpu without XPU support and torch.accelerator on torch < 2.6 raise here.
        return False


def _record() -> None:
    """Fill both caches in one pass, so the pre-spoof call primes CUDA and accelerator answers together."""
    global _REAL_ACCELERATOR, _REAL_CUDA, _REAL_XPU
    try:
        import torch
    except Exception:
        _REAL_ACCELERATOR, _REAL_CUDA, _REAL_XPU = False, False, False
        return
    _REAL_CUDA = _ask(lambda: hasattr(torch, "cuda") and torch.cuda.is_available())
    _REAL_XPU = _ask(lambda: hasattr(torch, "xpu") and torch.xpu.is_available())
    _REAL_ACCELERATOR = (
        _REAL_CUDA
        or _REAL_XPU
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


def has_real_xpu() -> bool:
    """True only for a real Intel XPU."""
    if _REAL_XPU is None:
        _record()
    return _REAL_XPU


def has_real_cuda_or_xpu() -> bool:
    """A real CUDA (or ROCm) GPU or a real Intel XPU: the devices Unsloth's Triton and torch GPU paths run on.

    Narrower than has_real_accelerator(), which is also true for MPS / NPU hosts through torch.accelerator.
    """
    return has_real_cuda() or has_real_xpu()


def accelerator_device_type() -> str:
    """'cuda' on CUDA / ROCm hosts, 'xpu' on Intel XPU hosts, else 'cpu'. CUDA wins when both exist."""
    if has_real_cuda():
        return "cuda"
    if has_real_xpu():
        return "xpu"
    return "cpu"


def accelerator_module():
    """torch.cuda or torch.xpu for the device accelerator_device_type() picks (same synchronize / memory-stats /
    Stream / Event API). Raises on a host with neither."""
    import torch

    device_type = accelerator_device_type()
    if device_type == "cpu":
        raise RuntimeError("no CUDA or XPU device")
    return getattr(torch, device_type)
