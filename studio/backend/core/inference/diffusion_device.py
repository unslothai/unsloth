# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Device policy for Studio Diffusers inference.

Studio's global hardware layer reports product backends such as CUDA, XPU,
MLX, and CPU. Diffusers itself runs through PyTorch devices, so Apple Silicon
maps to MPS, while ROCm maps to PyTorch's CUDA device type. This module keeps
that mapping explicit and centralizes which optimization paths are valid for
each backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen = True)
class DiffusionDeviceTarget:
    torch_device: str
    backend: str
    vendor: Optional[str]
    dtype: Any
    supports_model_cpu_offload: bool
    supports_default_torch_compile: bool
    supports_gguf_cpu_resident: bool
    supports_gguf_cuda_cache: bool
    supports_pinned_transfer: bool

    @property
    def is_cuda_torch_device(self) -> bool:
        return self.torch_device == "cuda"

    def as_public_dict(self) -> dict[str, Any]:
        dtype_name = str(self.dtype).replace("torch.", "")
        return {
            "torch_device": self.torch_device,
            "backend": self.backend,
            "vendor": self.vendor,
            "dtype": dtype_name,
            "supports_model_cpu_offload": self.supports_model_cpu_offload,
            "supports_default_torch_compile": self.supports_default_torch_compile,
            "supports_gguf_cpu_resident": self.supports_gguf_cpu_resident,
            "supports_gguf_cuda_cache": self.supports_gguf_cuda_cache,
            "supports_pinned_transfer": self.supports_pinned_transfer,
        }


def resolve_diffusion_device_target() -> DiffusionDeviceTarget:
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

    if (
        studio_device is not None
        and DeviceType is not None
        and studio_device == DeviceType.CUDA
    ):
        if torch.cuda.is_available():
            return _cuda_or_rocm_target(torch, is_rocm = is_rocm)
        return _cpu_target(torch)

    if (
        studio_device is not None
        and DeviceType is not None
        and studio_device == DeviceType.XPU
    ):
        return _xpu_target(torch)

    if (
        studio_device is not None
        and DeviceType is not None
        and studio_device == DeviceType.MLX
    ):
        return _mps_or_cpu_target(torch)

    if (
        studio_device is not None
        and DeviceType is not None
        and studio_device == DeviceType.CPU
    ):
        return _cpu_target(torch)

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


def diffusion_device_target_from_torch_device(
    torch_device: str,
    dtype: Any,
) -> DiffusionDeviceTarget:
    device = str(torch_device).split(":", 1)[0]
    if device == "cuda":
        try:
            import torch

            is_rocm = bool(getattr(getattr(torch, "version", None), "hip", None))
        except Exception:
            is_rocm = False
        backend = "rocm" if is_rocm else "cuda"
        return DiffusionDeviceTarget(
            torch_device = "cuda",
            backend = backend,
            vendor = "amd" if is_rocm else "nvidia",
            dtype = dtype,
            supports_model_cpu_offload = True,
            supports_default_torch_compile = not is_rocm,
            supports_gguf_cpu_resident = True,
            supports_gguf_cuda_cache = not is_rocm,
            supports_pinned_transfer = True,
        )
    if device == "xpu":
        return DiffusionDeviceTarget(
            torch_device = "xpu",
            backend = "xpu",
            vendor = "intel",
            dtype = dtype,
            supports_model_cpu_offload = True,
            supports_default_torch_compile = False,
            supports_gguf_cpu_resident = False,
            supports_gguf_cuda_cache = False,
            supports_pinned_transfer = False,
        )
    if device == "mps":
        return DiffusionDeviceTarget(
            torch_device = "mps",
            backend = "mps",
            vendor = "apple",
            dtype = dtype,
            supports_model_cpu_offload = False,
            supports_default_torch_compile = False,
            supports_gguf_cpu_resident = False,
            supports_gguf_cuda_cache = False,
            supports_pinned_transfer = False,
        )
    return _cpu_target(torch = None, dtype = dtype)


def _cuda_or_rocm_target(torch: Any, *, is_rocm: bool) -> DiffusionDeviceTarget:
    bf16_ok = False
    try:
        bf16_ok = bool(torch.cuda.is_bf16_supported())
    except Exception:
        bf16_ok = False
    return DiffusionDeviceTarget(
        torch_device = "cuda",
        backend = "rocm" if is_rocm else "cuda",
        vendor = "amd" if is_rocm else "nvidia",
        dtype = torch.bfloat16 if bf16_ok else torch.float16,
        supports_model_cpu_offload = True,
        supports_default_torch_compile = not is_rocm,
        supports_gguf_cpu_resident = True,
        supports_gguf_cuda_cache = not is_rocm,
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
        torch_device = "xpu",
        backend = "xpu",
        vendor = "intel",
        dtype = torch.bfloat16 if bf16_ok else torch.float16,
        supports_model_cpu_offload = True,
        supports_default_torch_compile = False,
        supports_gguf_cpu_resident = False,
        supports_gguf_cuda_cache = False,
        supports_pinned_transfer = False,
    )


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
        return DiffusionDeviceTarget(
            torch_device = "mps",
            backend = "mps",
            vendor = "apple",
            dtype = torch.float16,
            supports_model_cpu_offload = False,
            supports_default_torch_compile = False,
            supports_gguf_cpu_resident = False,
            supports_gguf_cuda_cache = False,
            supports_pinned_transfer = False,
        )
    return _cpu_target(torch)


def _cpu_target(torch: Any, dtype: Any = None) -> DiffusionDeviceTarget:
    if dtype is None:
        dtype = torch.float32
    return DiffusionDeviceTarget(
        torch_device = "cpu",
        backend = "cpu",
        vendor = None,
        dtype = dtype,
        supports_model_cpu_offload = False,
        supports_default_torch_compile = False,
        supports_gguf_cpu_resident = False,
        supports_gguf_cuda_cache = False,
        supports_pinned_transfer = False,
    )
