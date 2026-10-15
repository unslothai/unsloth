# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Device + dtype policy for the local diffusion backend.

torch imported lazily so this stays importable in a no-torch runtime. Unsloth's hardware layer
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
    # Kept out of device: policies compare device to "cuda", so "cuda:1" would disable them.
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
    """Metal has no float64, so RoPE tables drop their float64 intermediate; a no-op where float64 works."""
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


# Downsample shortcuts that zero-pad only the frame axis in front, to a multiple of ``factor_t``.
_FRAME_PAD_CLASSES = frozenset({"QwenImage21AvgDown3D", "AvgDown3D"})


def _prepend_zero_frames(module: Any, args: tuple) -> Optional[tuple]:
    x = args[0]
    pad_t = -x.shape[2] % module.factor_t
    if not pad_t:
        return None
    import torch

    zeros = x.new_zeros((*x.shape[:2], pad_t, *x.shape[3:]))
    return (torch.cat([zeros, x], dim = 2), *args[1:])


def install_frame_pad_fix(
    pipe: Any,
    target: DiffusionDeviceTarget,
    *,
    logger: Any = None,
) -> int:
    """MPS F.pad returns wrong data for large 5-D frame-axis pads, so shortcuts concatenate zero frames."""
    if target.device != "mps":
        return 0
    patched = 0
    for component in getattr(pipe, "components", {}).values() or ():
        modules = getattr(component, "modules", None)
        if not callable(modules):
            continue
        for module in modules():
            if type(module).__name__ not in _FRAME_PAD_CLASSES:
                continue
            if getattr(module, "_unsloth_frame_pad_fix", False):
                continue
            module.register_forward_pre_hook(_prepend_zero_frames)
            module._unsloth_frame_pad_fix = True
            patched += 1
    if patched and logger is not None:
        logger.info(
            "diffusion.vae_frame_pad: %d module(s) pad by concatenation (MPS pad defect)", patched
        )
    return patched


# Fraction of the device's recommended working set above which a decode starts synchronising.
DECODE_SYNC_FRACTION = 0.85


def install_decoder_sync(
    pipe: Any,
    target: DiffusionDeviceTarget,
    *,
    logger: Any = None,
) -> bool:
    """Syncs a Metal video VAE decode when memory runs low; buffers are held until their work completes."""
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


VAE_BF16_DECODE_ENV = "UNSLOTH_VIDEO_VAE_BF16_DECODE"
# RDNA3+ have bf16 WMMA; RDNA2 and older have no bf16 matrix path.
_ROCM_BF16_DECODE_ARCH_PREFIXES = ("gfx11", "gfx12")


def _rocm_bf16_decode_arch(torch: Any, target: DiffusionDeviceTarget) -> Optional[str]:
    try:
        index = target.ordinal if target.ordinal is not None else torch.cuda.current_device()
        arch = str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", "") or "")
    except Exception:  # noqa: BLE001 -- unreadable arch: keep fp32
        return None
    return arch if arch.startswith(_ROCM_BF16_DECODE_ARCH_PREFIXES) else None


def _as_float32(value: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_as_float32(v, torch) for v in value)
    if isinstance(value, list):
        return [_as_float32(v, torch) for v in value]
    sample = getattr(value, "sample", None)
    if isinstance(sample, torch.Tensor):
        value.sample = _as_float32(sample, torch)
    return value


_VAE_BF16_OFF = ("0", "false", "off", "no")
_VAE_BF16_FORCE = ("1", "true", "on", "yes")
VAE_BF16_DECODE_MODES = ("weights", "autocast")


def _vae_bf16_decode_request(gate: str) -> tuple[str, bool]:
    """(mode, forced) for a non-off UNSLOTH_VIDEO_VAE_BF16_DECODE; unknown values cast weights, "1" also forces."""
    if gate in VAE_BF16_DECODE_MODES:
        return gate, False
    return "weights", gate in _VAE_BF16_FORCE


def _cast_float_args(torch: Any, dtype: Any) -> Any:
    def _hook(module: Any, args: tuple) -> tuple:
        return tuple(
            a.to(dtype)
            if isinstance(a, torch.Tensor) and a.is_floating_point() and a.dtype != dtype
            else a
            for a in args
        )

    return _hook


def install_rocm_vae_bf16_decode(
    pipe: Any,
    target: DiffusionDeviceTarget,
    *,
    logger: Any = None,
) -> Optional[str]:
    """fp32 Wan VAE decode is very slow on ROCm gfx11/12; UNSLOTH_VIDEO_VAE_BF16_DECODE selects bf16."""
    gate = os.environ.get(VAE_BF16_DECODE_ENV, "auto").strip().lower()
    if gate in _VAE_BF16_OFF or target.device != "cuda":
        return None
    vae = getattr(pipe, "vae", None)
    decode = getattr(vae, "decode", None)
    if not callable(decode) or getattr(decode, "_unsloth_bf16_decode", False):
        return None
    if getattr(vae, "_unsloth_half_decode", False):
        return None
    import torch

    if getattr(vae, "dtype", None) is not torch.float32:
        return None
    mode, forced = _vae_bf16_decode_request(gate)
    if forced:
        if not torch.cuda.is_bf16_supported():
            return None
        arch = "forced"
    else:
        if target.backend != "rocm":
            return None
        arch = _rocm_bf16_decode_arch(torch, target)
        if arch is None:
            return None

    parts = [
        m
        for m in (getattr(vae, "post_quant_conv", None), getattr(vae, "decoder", None))
        if isinstance(m, torch.nn.Module)
    ]
    if mode == "weights" and not parts:
        mode = "autocast"

    if mode == "weights":
        for part in parts:
            part.to(torch.bfloat16)
            part.register_forward_pre_hook(_cast_float_args(torch, torch.bfloat16))

        def _bf16_decode(z: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(z, torch.Tensor) and z.is_floating_point():
                z = z.to(torch.bfloat16)
            return _as_float32(decode(z, *args, **kwargs), torch)

    else:

        def _bf16_decode(*args: Any, **kwargs: Any) -> Any:
            with torch.autocast(device_type = "cuda", dtype = torch.bfloat16):
                out = decode(*args, **kwargs)
            return _as_float32(out, torch)

    _bf16_decode._unsloth_bf16_decode = True  # type: ignore[attr-defined]
    _bf16_decode.__wrapped__ = decode  # type: ignore[attr-defined]
    vae.decode = _bf16_decode
    vae._unsloth_bf16_decode_mode = mode
    if logger is not None:
        logger.info("video.vae_decode: bf16 %s on %s", mode, arch)
    return mode


def _studio_device_is(studio_device: Any, device_type: Any, name: str) -> bool:
    """True if ``studio_device`` equals ``DeviceType.<name>`` (when that member exists)."""
    member = getattr(device_type, name, None)
    return member is not None and studio_device == member


def resolve_selected_cuda_ordinal(
    gpu_ids: Optional[list[int]], *, allow_ranking: bool = True
) -> Optional[int]:
    """gpu_ids are physical; a CUDA_VISIBLE_DEVICES mask changes torch ordinals, so translate them."""
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
    """Restores the prior CUDA device so a pooled worker's pin does not leak into the next request."""
    if ordinal is None:
        yield
        return
    # Only entering may fail; body exceptions must propagate untouched (yield-after-throw).
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
    """Thread-local, so each loading or running thread must call it; offload reads the current device."""
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
    """Records the card the weights actually sit on, since a pooled worker's pin would otherwise go
    stale."""
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
    """Honours ordinal only on CUDA/ROCm; a missing torch yields a torch-free CPU target, not a crash."""
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

            from .rocm_bf16 import is_rocm_torch

            # Tag-aware: AMD SDK / Radeon wheels leave torch.version.hip unset.
            is_rocm = is_rocm_torch(torch)
        except Exception:
            is_rocm = False
        return DiffusionDeviceTarget(
            device = "cuda",
            dtype = dtype,
            backend = "rocm" if is_rocm else "cuda",
            vendor = "amd" if is_rocm else "nvidia",
            supports_model_cpu_offload = True,
            supports_default_torch_compile = (not is_rocm) or _rocm_compile_allowed(),
            supports_pinned_transfer = True,
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


def float64_device(device: Any) -> Any:
    """Device to build float64 values on before moving the result to ``device``: itself, or CPU when it has no float64."""
    target = diffusion_device_target_from_torch_device(str(device), None)
    return device if target.supports_float64 else "cpu"


def _rocm_compile_allowed() -> bool:
    """Kill switch only; the install probe runs at compile time (``diffusion_rocm_compile``)."""
    try:
        from .diffusion_rocm_compile import rocm_compile_allowed
        return rocm_compile_allowed()
    except Exception:  # noqa: BLE001
        return False


def _cuda_or_rocm_target(
    torch: Any,
    *,
    is_rocm: bool,
    ordinal: Optional[int] = None,
) -> DiffusionDeviceTarget:
    if is_rocm:
        # is_bf16_supported() takes no device argument: scope the selected card current.
        from .rocm_bf16 import rocm_bf16_supported
        try:
            with diffusion_device_scope(ordinal):
                bf16_ok = rocm_bf16_supported(torch, ordinal)
        except Exception:
            bf16_ok = False
        dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        # bf16 needs Ampere+ by capability: pre-Ampere reports supported but emulates slowly.
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
        supports_default_torch_compile = (not is_rocm) or _rocm_compile_allowed(),
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


MPS_HIGH_WATERMARK_ENV = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
# torch's accepted range for the high watermark ratio (MPSAllocator default_high_watermark_upper_bound).
_MPS_RATIO_MAX = 2.0
# Room left outside the cap for macOS, the window server and this process's own CPU-side memory.
_MPS_CAP_OS_RESERVE_MIN = 1536 * 1024**2
_MPS_CAP_OS_RESERVE_FRAC = 0.10


def mps_high_watermark_ratio(
    recommended_bytes: Optional[int], total_ram_bytes: Optional[int]
) -> Optional[float]:
    """The MPS high watermark ratio (a multiple of the recommended working set) that caps Metal allocations at
    physical RAM minus an OS reserve, never below the working set Apple recommends. None when either reading is
    missing (no cap: the caller keeps the unbounded allocator).

    torch's own default is 1.7x the working set, which is above physical RAM on every Apple Silicon Mac (the working
    set is ~2/3 to 3/4 of RAM), so past RAM it swaps instead of raising; 0.0 removes the limit entirely. A finite cap
    turns that into an "MPS backend out of memory" RuntimeError that the batch backoff and VAE tile retry handle."""
    try:
        recommended = int(recommended_bytes or 0)
        total = int(total_ram_bytes or 0)
    except (TypeError, ValueError):
        return None
    if recommended <= 0 or total <= 0:
        return None
    reserve = max(_MPS_CAP_OS_RESERVE_MIN, int(total * _MPS_CAP_OS_RESERVE_FRAC))
    cap = max(recommended, total - reserve)
    return round(min(_MPS_RATIO_MAX, cap / recommended), 4)


def _total_ram_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 -- fall back to sysconf
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:  # noqa: BLE001
        return None


def _mps_cap_disabled() -> bool:
    if MPS_HIGH_WATERMARK_ENV in os.environ and os.environ.get(MPS_HIGH_WATERMARK_ENV) != "0.0":
        return True
    # The oversized-load opt-in asks to run past the unified-memory guard, so it keeps the unbounded allocator too.
    return os.environ.get("UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def apply_mps_memory_cap(torch: Any, logger: Any = None) -> Optional[float]:
    """Cap MPS allocations below physical RAM (``mps_high_watermark_ratio``). Returns the ratio set, or None when
    nothing was changed: a user-set PYTORCH_MPS_HIGH_WATERMARK_RATIO, the oversized-load opt-in, or torch without
    ``recommended_max_memory`` (< 2.5) / ``set_per_process_memory_fraction``. Set at runtime because the allocator
    may already exist (its env reading is one-shot), and best-effort: a failure keeps today's allocator."""
    if _mps_cap_disabled():
        return None
    try:
        mps = torch.mps
        ratio = mps_high_watermark_ratio(mps.recommended_max_memory(), _total_ram_bytes())
        if ratio is None:
            return None
        mps.set_per_process_memory_fraction(float(ratio))
    except Exception as exc:  # noqa: BLE001 -- old torch / no reading: keep the unbounded allocator
        if logger is not None:
            logger.info("diffusion: MPS memory cap not set (%s)", exc)
        return None
    if logger is not None:
        logger.info("diffusion: MPS allocations capped at %.3fx the recommended working set", ratio)
    return ratio


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
        user_ratio = MPS_HIGH_WATERMARK_ENV in os.environ
        # torch reads PYTORCH_MPS_HIGH_WATERMARK_RATIO once, when the allocator is built (the probe below, or an earlier
        # recommended_max_memory() reading). 0.0 here only lifts torch's 1.7x-of-working-set default, which sits above
        # physical RAM; apply_mps_memory_cap() then sets the finite cap. setdefault respects an override.
        os.environ.setdefault(MPS_HIGH_WATERMARK_ENV, "0.0")
        # Prefer bfloat16, else float32, NEVER silent float16: modern DiTs produce activations far outside fp16's
        # range (Z-Image MLP peaks near 9e5 -> inf -> NaN -> black image). bf16 (macOS 14+) shares fp32's exponent
        # range; older macOS uses fp32.
        dtype = torch.bfloat16 if _mps_supports_bfloat16(torch) else torch.float32
        if not user_ratio:
            apply_mps_memory_cap(torch)
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
