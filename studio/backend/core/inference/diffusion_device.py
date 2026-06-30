# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Device + dtype policy for the local diffusion backend.

torch is imported lazily inside each function so this module stays importable in
a no-torch runtime (mirrors ``diffusion.py`` / ``diffusion_families.py``).

Studio's hardware layer reports product backends (CUDA, XPU, MLX, CPU); diffusers
runs on PyTorch devices, so Apple Silicon maps to MPS and ROCm maps to PyTorch's
``cuda`` device type. This module centralises that mapping plus the per-backend
dtype choice and the capability flags the backend keys optimisation paths off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen = True)
class DiffusionDeviceTarget:
    """Resolved torch device + compute dtype + per-backend capability flags."""

    device: str
    dtype: Any
    backend: str
    vendor: Optional[str]
    supports_model_cpu_offload: bool
    supports_default_torch_compile: bool
    supports_pinned_transfer: bool


def _studio_device_is(studio_device: Any, device_type: Any, name: str) -> bool:
    """True if ``studio_device`` equals ``DeviceType.<name>`` (when that member exists)."""
    member = getattr(device_type, name, None)
    return member is not None and studio_device == member


def resolve_diffusion_device_target() -> DiffusionDeviceTarget:
    """Resolve the torch device + dtype + capability flags for diffusion.

    Prefers Studio's hardware layer when importable, else probes torch directly
    (CUDA -> XPU -> MPS -> CPU). On Apple Silicon Studio reports MLX/CPU when its
    product backend is gated on the ``mlx`` package, but diffusers runs on
    PyTorch's MPS backend, so those cases still fall through to the MPS probe.
    """
    import torch

    try:
        from utils.hardware import DeviceType, get_device
        from utils.hardware import hardware as hardware_mod

        studio_device = get_device()
        is_rocm = bool(getattr(hardware_mod, "IS_ROCM", False))
    except Exception:
        DeviceType = None
        studio_device = None
        is_rocm = bool(getattr(getattr(torch, "version", None), "hip", None))

    if DeviceType is not None and studio_device is not None:
        if _studio_device_is(studio_device, DeviceType, "CUDA"):
            if torch.cuda.is_available():
                return _cuda_or_rocm_target(torch, is_rocm = is_rocm)
            return _cpu_target(torch)
        if _studio_device_is(studio_device, DeviceType, "XPU"):
            return _xpu_target(torch)
        # MLX / CPU / anything else: diffusers uses MPS (not MLX), so fall
        # through to the torch probe below, which prefers MPS over CPU.

    if torch.cuda.is_available():
        return _cuda_or_rocm_target(torch, is_rocm = is_rocm)

    xpu = getattr(torch, "xpu", None)
    if xpu is not None and callable(getattr(xpu, "is_available", None)):
        try:
            if xpu.is_available():
                return _xpu_target(torch)
        except Exception:
            pass

    return _mps_or_cpu_target(torch)


def _cuda_or_rocm_target(torch: Any, *, is_rocm: bool) -> DiffusionDeviceTarget:
    if is_rocm:
        # ROCm (AMD) does not have NVIDIA's pre-Ampere bf16-emulation quirk, so
        # is_bf16_supported() is trustworthy here; bf16 only when it proves it.
        try:
            bf16_ok = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_ok = False
        dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        # NVIDIA: bf16 needs Ampere+ (capability major >= 8). Checked by
        # capability, NOT is_bf16_supported() -- pre-Ampere cards emulate bf16
        # and report it supported, but it is slow / unwanted (the #6658 fix).
        try:
            major = torch.cuda.get_device_capability()[0]
        except Exception:
            major = 0
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    return DiffusionDeviceTarget(
        device = "cuda",
        dtype = dtype,
        backend = "rocm" if is_rocm else "cuda",
        vendor = "amd" if is_rocm else "nvidia",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = not is_rocm,
        supports_pinned_transfer = True,
    )


def _xpu_target(torch: Any) -> DiffusionDeviceTarget:
    bf16_ok = False
    xpu = getattr(torch, "xpu", None)
    try:
        bf16_ok = bool(xpu.is_bf16_supported()) if xpu is not None else False
    except Exception:
        bf16_ok = False
    return DiffusionDeviceTarget(
        device = "xpu",
        dtype = torch.bfloat16 if bf16_ok else torch.float16,
        backend = "xpu",
        vendor = "intel",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = False,
        supports_pinned_transfer = False,
    )


def _mps_supports_bfloat16(torch: Any) -> bool:
    """Runtime probe for usable MPS bfloat16.

    PyTorch only supports bfloat16 on MPS on macOS 14+; on older macOS a bfloat16
    op raises. Probe with a tiny compute forced to evaluate (device->host sync)
    rather than guessing from the macOS / chip version.
    """
    try:
        x = torch.ones(2, dtype = torch.bfloat16, device = "mps")
        return bool(torch.isfinite((x + x).float()).all().item())
    except Exception:
        return False


def _mps_or_cpu_target(torch: Any) -> DiffusionDeviceTarget:
    mps_available = False
    try:
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_available = bool(
            mps_backend is not None
            and callable(getattr(mps_backend, "is_available", None))
            and mps_backend.is_available()
        )
    except Exception:
        mps_available = False

    if mps_available:
        # Prefer bfloat16; otherwise fall back to float32, NEVER silent float16.
        # Modern diffusion transformers (Z-Image, FLUX.2, ...) produce activations
        # far outside float16's finite range (~6.5e4) -- Z-Image's MLP
        # down-projections peak near 9e5, overflowing to inf -> NaN -> a black
        # image. bfloat16 (macOS 14+) shares float32's exponent range; on older
        # macOS the probe fails and float32 keeps output correct (if slower).
        dtype = torch.bfloat16 if _mps_supports_bfloat16(torch) else torch.float32
        return DiffusionDeviceTarget(
            device = "mps",
            dtype = dtype,
            backend = "mps",
            vendor = "apple",
            supports_model_cpu_offload = False,
            supports_default_torch_compile = False,
            supports_pinned_transfer = False,
        )
    return _cpu_target(torch)


def _cpu_target(torch: Any, dtype: Any = None) -> DiffusionDeviceTarget:
    if dtype is None:
        dtype = torch.float32
    return DiffusionDeviceTarget(
        device = "cpu",
        dtype = dtype,
        backend = "cpu",
        vendor = None,
        supports_model_cpu_offload = False,
        supports_default_torch_compile = False,
        supports_pinned_transfer = False,
    )
