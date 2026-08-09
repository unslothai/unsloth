# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Device + dtype policy for the local diffusion backend.

torch imported lazily so this stays importable in a no-torch runtime. Studio's hardware layer
reports product backends (CUDA, XPU, MLX, CPU); diffusers runs on PyTorch devices, so Apple
Silicon maps to MPS and ROCm to ``cuda``. Centralises that mapping plus the per-backend dtype and
the capability flags optimisation paths key off.
"""

from __future__ import annotations

import os
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
    supports_float64: bool = True

    @property
    def is_cuda_torch_device(self) -> bool:
        return self.device == "cuda"

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "backend": self.backend,
            "vendor": self.vendor,
            "supports_model_cpu_offload": self.supports_model_cpu_offload,
            "supports_default_torch_compile": self.supports_default_torch_compile,
            "supports_pinned_transfer": self.supports_pinned_transfer,
            "supports_float64": self.supports_float64,
        }


def force_float32_rope(
    pipe: Any,
    target: DiffusionDeviceTarget,
    *,
    logger: Any = None,
) -> int:
    """Drop the float64 intermediate in RoPE frequency tables on a device without float64.

    LTX-2's rotary embeddings build ``theta ** linspace(0, 1, n)`` in float64 and cast straight
    back to float32. Metal implements no float64, so torch raises and generation dies before the
    first step. The modules gate that intermediate on a ``double_precision`` attribute; clearing
    it costs at most 6 float32 ULP, measured across the theta/dim combinations these models use,
    against a value the next line truncates to float32 anyway.

    Returns the number of modules changed. A no-op wherever float64 works, so CUDA/XPU/CPU keep
    the upstream computation bit-for-bit.
    """
    if target.supports_float64:
        return 0
    changed = 0
    for component in getattr(pipe, "components", {}).values() or ():
        modules = getattr(component, "modules", None)
        if not callable(modules):
            continue
        for module in modules():
            if getattr(module, "double_precision", False):
                module.double_precision = False
                changed += 1
    if changed and logger is not None:
        logger.info("video.rope_float32: %d module(s) demoted (no float64 on this device)", changed)
    return changed


# Fraction of the device's recommended working set above which a decode starts synchronising.
DECODE_SYNC_FRACTION = 0.85


def install_decoder_sync(
    pipe: Any,
    target: DiffusionDeviceTarget,
    *,
    logger: Any = None,
) -> bool:
    """Cap the memory a video VAE decode holds on Metal, by synchronising once it is running out.

    Wan's VAE decodes one latent frame per call in a loop that never forces a commit, and Metal
    cannot reuse a buffer until the work holding it completes, so intermediates accumulate until the
    OS kills the process. Neither tiling (the growth is within one tile) nor torch's adaptive commit
    (a low watermark far below the observed peak changed nothing) bounds it.

    Fires per decoder call -- per frame on Wan, per tile on the VAEs that decode a whole tensor at
    once -- and only above the threshold, so a decode with room to spare pays only the memory read.
    Synchronising waits on work already enqueued, so what it costs is the pipelining, not the decode.
    """
    if target.device != "mps":
        return False
    decoder = getattr(getattr(pipe, "vae", None), "decoder", None)
    if not callable(getattr(decoder, "register_forward_hook", None)):
        return False
    import torch

    budget = torch.mps.recommended_max_memory() * DECODE_SYNC_FRACTION

    def _sync(_module, _args, _output) -> None:
        if torch.mps.driver_allocated_memory() >= budget:
            torch.mps.synchronize()

    decoder.register_forward_hook(_sync)
    if logger is not None:
        logger.info("video.decoder_sync: decode synchronises above %.1f GiB", budget / 1024**3)
    return True


def _studio_device_is(studio_device: Any, device_type: Any, name: str) -> bool:
    """True if ``studio_device`` equals ``DeviceType.<name>`` (when that member exists)."""
    member = getattr(device_type, name, None)
    return member is not None and studio_device == member


def resolve_diffusion_device_target() -> DiffusionDeviceTarget:
    """Resolve the torch device + dtype + capability flags for diffusion.

    Prefers Studio's hardware layer, else probes torch (CUDA -> XPU -> MPS -> CPU). On Apple
    Silicon Studio may report MLX/CPU, but diffusers uses MPS, so those fall through to the MPS
    probe. Torch is optional: without it the native sd.cpp engine still runs, so a missing torch
    reports a torch-free CPU target instead of crashing ``/images/load`` before engine selection.
    """
    try:
        import torch
    except Exception:
        return DiffusionDeviceTarget(
            device = "cpu",
            dtype = None,
            backend = "cpu",
            vendor = None,
            supports_model_cpu_offload = False,
            supports_default_torch_compile = False,
            supports_pinned_transfer = False,
        )

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
        # MLX / CPU / else: diffusers uses MPS, so fall through to the torch probe (MPS over CPU).

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
    torch_device: str, dtype: Any
) -> DiffusionDeviceTarget:
    """Reconstruct a target from a (device, dtype) pair, so a caller overriding the tuple (the
    ``_pick_device_and_dtype`` shim / monkeypatch path) can still recover the capability flags."""
    device = str(torch_device).split(":", 1)[0]
    if device == "cuda":
        try:
            import torch
            is_rocm = bool(getattr(getattr(torch, "version", None), "hip", None))
        except Exception:
            is_rocm = False
        return DiffusionDeviceTarget(
            device = "cuda",
            dtype = dtype,
            backend = "rocm" if is_rocm else "cuda",
            vendor = "amd" if is_rocm else "nvidia",
            supports_model_cpu_offload = True,
            supports_default_torch_compile = not is_rocm,
            supports_pinned_transfer = True,
        )
    if device == "xpu":
        return DiffusionDeviceTarget(
            device = "xpu",
            dtype = dtype,
            backend = "xpu",
            vendor = "intel",
            supports_model_cpu_offload = True,
            supports_default_torch_compile = False,
            supports_pinned_transfer = False,
        )
    if device == "mps":
        return DiffusionDeviceTarget(
            device = "mps",
            dtype = dtype,
            backend = "mps",
            vendor = "apple",
            supports_model_cpu_offload = False,
            supports_default_torch_compile = False,
            supports_pinned_transfer = False,
            supports_float64 = False,
        )
    return _cpu_target(torch = None, dtype = dtype)


def _cuda_or_rocm_target(torch: Any, *, is_rocm: bool) -> DiffusionDeviceTarget:
    if is_rocm:
        # ROCm lacks NVIDIA's pre-Ampere bf16-emulation quirk, so is_bf16_supported() is trustworthy.
        try:
            bf16_ok = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_ok = False
        dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        # NVIDIA: bf16 needs Ampere+ (major >= 8), by capability NOT is_bf16_supported() (pre-Ampere cards emulate bf16 slowly but report it supported).
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
    """Runtime probe for usable MPS bfloat16 (only on macOS 14+; older macOS raises). Probes with
    a tiny forced compute rather than guessing from the macOS / chip version."""
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
        # torch reads PYTORCH_MPS_HIGH_WATERMARK_RATIO once, at the first MPS allocation (the probe below), so relax it first or
        # the allocator caps at ~1.7x recommendedMaxWorkingSet and can OOM a model that would fit. setdefault respects an override.
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        # Prefer bfloat16, else float32, NEVER silent float16: modern DiTs produce activations far outside fp16's range (Z-Image
        # MLP peaks near 9e5 -> inf -> NaN -> black image). bf16 (macOS 14+) shares fp32's exponent range; older macOS uses fp32.
        dtype = torch.bfloat16 if _mps_supports_bfloat16(torch) else torch.float32
        return DiffusionDeviceTarget(
            device = "mps",
            dtype = dtype,
            backend = "mps",
            vendor = "apple",
            supports_model_cpu_offload = False,
            supports_default_torch_compile = False,
            supports_pinned_transfer = False,
            supports_float64 = False,
        )
    return _cpu_target(torch)


def _cpu_target(torch: Any, dtype: Any = None) -> DiffusionDeviceTarget:
    # torch is None on the no-torch CPU fallback; leave dtype=None rather than crash.
    if dtype is None and torch is not None:
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
