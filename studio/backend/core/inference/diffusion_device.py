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
from contextlib import contextmanager
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
    # Selected CUDA/ROCm physical index, kept OUT of ``device``: the memory, speed and attention policies compare that string against "cuda", so a "cuda:1" there disables them silently.
    ordinal: Optional[int] = None

    @property
    def is_cuda_torch_device(self) -> bool:
        return self.device == "cuda"

    @property
    def torch_device(self) -> str:
        """The device string to PLACE weights on, indexed when one card was selected."""
        return f"{self.device}:{self.ordinal}" if self.ordinal is not None else self.device

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
            "ordinal": self.ordinal,
        }


def force_float32_rope(
    pipe: Any,
    target: DiffusionDeviceTarget,
    *,
    logger: Any = None,
) -> int:
    """Drop the float64 intermediate in RoPE frequency tables on a device without float64.

    LTX-2 builds ``theta ** linspace(0, 1, n)`` in float64 and casts straight back to float32;
    Metal has no float64, so torch raises before the first step. The modules gate that
    intermediate on a ``double_precision`` attribute, and clearing it costs at most 6 float32 ULP
    against a value the next line truncates anyway.

    Returns the number of modules changed; a no-op wherever float64 works, so CUDA/XPU/CPU stay
    bit-for-bit.
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
    cannot reuse a buffer until the work holding it completes, so intermediates accumulate until
    the OS kills the process. Neither tiling (the growth is within one tile) nor torch's adaptive
    commit bounds it.

    Fires per decoder call and only above the threshold, so a decode with room to spare pays only
    the memory read; synchronising costs the pipelining, not the decode.

    ``torch.mps.recommended_max_memory()`` arrived in torch 2.5 while install.sh keeps an existing
    venv's torch as far back as 2.4, so an unreadable budget falls back to synchronising every
    call (measured to hold the same decode at 4.90 GiB for no wall-clock cost) rather than
    failing the load or dropping the bound. Every probe is best-effort for the same reason.
    """
    if target.device != "mps":
        return False
    decoder = getattr(getattr(pipe, "vae", None), "decoder", None)
    if not callable(getattr(decoder, "register_forward_hook", None)):
        return False
    import torch

    budget: Optional[float] = None
    try:
        budget = torch.mps.recommended_max_memory() * DECODE_SYNC_FRACTION
    except Exception as exc:  # noqa: BLE001 -- torch < 2.5 has no such reading
        if logger is not None:
            logger.info(
                "video.decoder_sync: no memory reading (%s); synchronising every decode", exc
            )

    def _sync(_module, _args, _output) -> None:
        if budget is not None:
            try:
                if torch.mps.driver_allocated_memory() < budget:
                    return
            except Exception:  # noqa: BLE001 -- an unreadable gauge syncs, the safe side
                pass
        try:
            torch.mps.synchronize()
        except Exception:  # noqa: BLE001 -- a decode is worth more than the bound
            pass

    decoder.register_forward_hook(_sync)
    if logger is not None and budget is not None:
        logger.info("video.decoder_sync: decode synchronises above %.1f GiB", budget / 1024**3)
    return True


def _studio_device_is(studio_device: Any, device_type: Any, name: str) -> bool:
    """True if ``studio_device`` equals ``DeviceType.<name>`` (when that member exists)."""
    member = getattr(device_type, name, None)
    return member is not None and studio_device == member


def resolve_selected_cuda_ordinal(
    gpu_ids: Optional[list[int]], *, allow_ranking: bool = True
) -> Optional[int]:
    """The torch ordinal one diffusion load should run on, or None for automatic.

    ``gpu_ids`` carries PHYSICAL ids, as chat, training and the UI use. Torch indexes only the
    parent-visible subset, so under a ``CUDA_VISIBLE_DEVICES`` mask the two differ in value and
    order (``4,5`` -> torch 0,1; ``1,0`` reverses them), hence going through the hardware layer
    that owns the mask.

    Neither engine shards a checkpoint, so several cards still resolve to one: most free VRAM
    wins, as ``auto_select_gpu_ids`` already does for training, ties to the lowest ordinal. Taking
    the FIRST id instead would land on ordinal 0 whenever everything is selected, i.e. the small
    card on the mixed boxes this exists for. Resolved ONCE per load and carried, never re-derived:
    free VRAM moves the moment the checkpoint lands.

    Raises ValueError for a selection this host cannot honour, so the load is refused with a
    reason rather than quietly running somewhere the user did not choose.

    ``allow_ranking = False`` drops only the free-VRAM probe, for a caller that must not open a
    CUDA context (the plan routes while a trainer holds the cards). Validation and translation
    still run -- they read the mask and nvidia-smi -- so the single card the UI sends resolves and
    only a multi-card pick comes back None.
    """
    wanted = sorted({int(gpu_id) for gpu_id in gpu_ids or ()})
    if not wanted:
        return None
    try:
        from utils.hardware.hardware import (
            get_parent_visible_gpu_ids,
            resolve_requested_gpu_ids,
        )
    except Exception as exc:  # noqa: BLE001 -- without the hardware layer the mask is unknowable
        raise ValueError(f"GPU selection is unavailable on this host: {exc}") from exc
    allowed = resolve_requested_gpu_ids(wanted)
    visible = get_parent_visible_gpu_ids()
    # Torch enumerates the parent-visible list in order, so its ordinal for a physical id is that
    # id's position in the mask. Unmasked, the layer reports range(physical count) and this is
    # the identity mapping.
    ordinals = [visible.index(gpu_id) for gpu_id in allowed if gpu_id in visible]
    if not ordinals:
        raise ValueError(
            f"Requested GPU {wanted} but none of them are visible to this process "
            f"(visible: {visible}). Clear the GPU selection to use the default device."
        )
    if len(ordinals) == 1:
        return ordinals[0]
    if not allow_ranking:
        return None

    def _free_vram(ordinal: int) -> int:
        try:
            import torch
            return int(torch.cuda.mem_get_info(ordinal)[0])
        except Exception:  # noqa: BLE001 -- an unreadable card sorts last rather than failing the load
            return -1

    return max(ordinals, key = lambda ordinal: (_free_vram(ordinal), -ordinal))


@contextmanager
def diffusion_device_scope(ordinal: Optional[int]):
    """Make ``ordinal`` the current CUDA device for the block, then restore the previous one.

    For probes on a POOLED thread. ``torch.cuda.set_device`` is thread-local but not scoped, so a
    permanent pin on an asyncio.to_thread executor thread outlives the request and leaves the next
    one -- perhaps an automatic load -- resolving bare "cuda" against the previous request's card.
    Worker threads are dedicated and keep the permanent pin.
    """
    if ordinal is None:
        yield
        return
    # Entering the context is what may fail on an unusable index; the BODY's exceptions have to
    # travel untouched, or a yield-after-throw replaces the caller's real refusal with
    # "generator didn't stop after throw()".
    try:
        import torch
        scope = torch.cuda.device(ordinal)
        scope.__enter__()
    except Exception:  # noqa: BLE001 -- an unreadable index still runs the probe, unpinned
        yield
        return
    try:
        yield
    finally:
        try:
            scope.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 -- restoring is best effort; never mask the body
            pass


def apply_diffusion_device_ordinal(target: DiffusionDeviceTarget) -> None:
    """Point this thread's CUDA context at ``target.ordinal``.

    Thread-local, so every worker that loads or runs a pipeline has to call it; the load thread
    setting it does nothing for the generate thread. The right lever rather than an indexed device
    string because the offload policy reads ``torch.cuda.mem_get_info()`` with no argument, i.e.
    the CURRENT device, so this steers the weights and their budget to the same card. A no-op for
    an automatic pick.
    """
    if not target.is_cuda_torch_device:
        return
    pin_cuda_ordinal(target.ordinal)


def pin_cuda_ordinal(ordinal: Optional[int]) -> None:
    """``torch.cuda.set_device``, thread-local, never fatal. A no-op for None."""
    if ordinal is None:
        return
    try:
        import torch
        torch.cuda.set_device(ordinal)
    except Exception:  # noqa: BLE001 -- placement still works off torch_device; never fail a load here
        pass


def placed_cuda_ordinal(target: DiffusionDeviceTarget) -> Optional[int]:
    """The card the weights are actually on: the selection when there was one, else the card the
    loading thread was pointing at.

    Recorded WITH the pipeline because ``/images/generate`` runs on a pooled ``asyncio.to_thread``
    worker: a pinned load leaves that worker on its card permanently, and a later automatic load
    has no ordinal to re-pin with, so its bare "cuda" Generators and allocations would land on the
    previous model's GPU while the weights sat on the default one. Kept apart from ``ordinal`` so
    the automatic path still reports a bare device and an un-indexed target, as it always did.
    """
    if not target.is_cuda_torch_device:
        return None
    if target.ordinal is not None:
        return target.ordinal
    try:
        import torch
        return int(torch.cuda.current_device())
    except Exception:  # noqa: BLE001 -- an unreadable device simply leaves the worker alone
        return None


def resolve_diffusion_device_target(*, ordinal: Optional[int] = None) -> DiffusionDeviceTarget:
    """Resolve the torch device + dtype + capability flags for diffusion.

    Prefers Studio's hardware layer, else probes torch (CUDA -> XPU -> MPS -> CPU). On Apple
    Silicon Studio may report MLX/CPU, but diffusers uses MPS, so those fall through to the MPS
    probe. Torch is optional: without it the native sd.cpp engine still runs, so a missing torch
    reports a torch-free CPU target instead of crashing ``/images/load`` before engine selection.

    ``ordinal`` is an ALREADY-RESOLVED torch index from ``resolve_selected_cuda_ordinal``, carried
    for one load rather than re-derived. Honoured only on CUDA / ROCm, where an index is what the
    runners speak; XPU has no applicator and MPS / CPU nothing to choose between.
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
                return _cuda_or_rocm_target(torch, is_rocm = is_rocm, ordinal = ordinal)
            return _cpu_target(torch)
        if _studio_device_is(studio_device, DeviceType, "XPU"):
            return _xpu_target(torch)
        # MLX / CPU / else: diffusers uses MPS, so fall through to the torch probe (MPS over CPU).

    if torch.cuda.is_available():
        return _cuda_or_rocm_target(torch, is_rocm = is_rocm, ordinal = ordinal)

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
    device, _, index = str(torch_device).partition(":")
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
            # An overriding caller's "cuda:1" is a device choice to keep, not one to drop back to ordinal 0.
            ordinal = int(index) if index.isdigit() else None,
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


def _cuda_or_rocm_target(
    torch: Any,
    *,
    is_rocm: bool,
    ordinal: Optional[int] = None,
) -> DiffusionDeviceTarget:
    if is_rocm:
        # ROCm lacks NVIDIA's pre-Ampere bf16-emulation quirk, so is_bf16_supported() is trustworthy.
        # It takes no device argument, so the selected card is asked by scoping the current device.
        try:
            with diffusion_device_scope(ordinal):
                bf16_ok = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_ok = False
        dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        # NVIDIA: bf16 needs Ampere+ (major >= 8), by capability NOT is_bf16_supported() (pre-Ampere cards emulate bf16 slowly but report it supported).
        # Asked of the SELECTED card, since the argument-less form reports the current device, a different generation on a mixed box; still argument-less without a selection.
        try:
            major = (
                torch.cuda.get_device_capability()
                if ordinal is None
                else torch.cuda.get_device_capability(ordinal)
            )[0]
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
        ordinal = ordinal,
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
