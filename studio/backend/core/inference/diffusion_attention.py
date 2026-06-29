# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Select the diffusion transformer's attention backend.

diffusers exposes a unified ``transformer.set_attention_backend(name)`` dispatcher that
swaps the scaled-dot-product-attention kernel, validating hardware/package requirements at
set time and otherwise leaving the default (``native`` = ``F.scaled_dot_product_attention``).
Attention is memory-bandwidth bound, so a better kernel is a real end-to-end win that is
orthogonal to the linear-weight quantisation (it speeds the QK/PV matmuls torchao never
touches) and composes with torch.compile.

  auto  - the best *exact* (non-quantized) backend for the device. On NVIDIA CUDA that is
          cuDNN's fused attention (``_native_cudnn``), measured ~1.18x end-to-end on a B200
          with LPIPS ~0.004 vs the default (below the compile/quant noise floor). On
          AMD/Intel/Apple/CPU it stays ``native`` (the dispatcher already routes those).
          ``auto`` only upgrades when a speed profile is active, so ``speed_mode=off`` stays
          bit-identical.
  native - force the default SDPA (bit-identical reference).
  cudnn  - cuDNN fused attention (exact; NVIDIA).
  flash / flash3 / flash4 - FlashAttention 2 / 3 (Hopper) / 4 (SM100); exact, kernel-gated.
  sage   - SageAttention (INT8 QK); quantized, a small quality cost, consumer-friendly.
  xformers / aiter - memory-efficient (NVIDIA) / AITER (AMD ROCm).

Best-effort: an unavailable backend (missing kernel / wrong arch) is caught and the load
falls back to the diffusers default rather than failing. torch/diffusers imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional

ATTN_AUTO = "auto"
ATTN_NATIVE = "native"

# User-facing alias -> the diffusers dispatcher backend name.
_ALIASES: dict[str, str] = {
    "native": "native",
    "sdpa": "native",
    "cudnn": "_native_cudnn",
    "flash": "flash",
    "flash2": "flash",
    "flash3": "_flash_3_hub",
    "flash4": "flash_4_hub",
    "sage": "sage",
    "xformers": "xformers",
    "aiter": "aiter",
}
ATTN_ALIASES = (ATTN_AUTO,) + tuple(dict.fromkeys(_ALIASES))


def normalize_attention_backend(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested attention backend; None / "" / "auto" -> "auto".

    Raises ValueError for an unsupported alias so a bad request is rejected cheaply."""
    if value is None:
        return ATTN_AUTO
    normalized = str(value).strip().lower()
    if not normalized:
        return ATTN_AUTO
    if normalized not in ATTN_ALIASES:
        raise ValueError(
            f"Unsupported attention_backend '{value}'. Use one of: {', '.join(ATTN_ALIASES)}."
        )
    return normalized


# Backends diffusers validates only by *package* at set time (``_check_attention_backend_
# requirements`` checks the ``kernels`` install, not the GPU), but whose kernels need a
# specific CUDA arch at run time -- so an explicit request on the wrong card loads/sets fine
# and then crashes mid-generation. Gate them up front: minimum (major, minor) compute
# capability per dispatcher backend name.
_MIN_CUDA_CAPABILITY: dict[str, tuple[int, int]] = {
    "_flash_3_hub": (9, 0),  # FlashAttention 3 -> Hopper (SM90)
    "flash_4_hub": (10, 0),  # FlashAttention 4 -> Blackwell (SM100)
}


def _cuda_capability() -> Optional[tuple[int, int]]:
    """(major, minor) compute capability of the active CUDA device, or None if unknown."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return tuple(torch.cuda.get_device_capability())  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return None


def _backend_arch_supported(backend: str) -> bool:
    """False only when ``backend`` needs a known-higher CUDA arch than this device has.

    Unknown capability (no CUDA / detection failure) returns True so we never block on a
    guess -- diffusers' own set-time check still guards the package, and a genuine run-time
    failure falls back to native."""
    required = _MIN_CUDA_CAPABILITY.get(backend)
    if required is None:
        return True
    have = _cuda_capability()
    if have is None:
        return True
    return have >= required


def _is_cuda_nvidia(target: Any) -> bool:
    """CUDA device on an NVIDIA (non-ROCm) build -- where cuDNN attention applies."""
    if getattr(target, "device", None) != "cuda":
        return False
    try:
        import torch
        return getattr(torch.version, "hip", None) is None
    except Exception:  # noqa: BLE001
        return False


def select_attention_backend(
    target: Any, requested: Optional[str], *, speed_active: bool
) -> Optional[str]:
    """The dispatcher backend name to apply, or None to leave the diffusers default.

    An explicit alias is honored verbatim (apply falls back if its kernel is unavailable).
    ``auto`` upgrades to cuDNN on NVIDIA CUDA only when a speed profile is active (so
    ``off`` stays bit-identical); everywhere else it returns None (native default)."""
    alias = normalize_attention_backend(requested)
    if alias != ATTN_AUTO:
        backend = _ALIASES[alias]
        if backend == "native":
            return None
        # An arch-gated kernel (flash3/flash4) on a card that can't run it would set fine
        # then crash mid-generation, so drop it to the native default up front.
        if not _backend_arch_supported(backend):
            return None
        return backend
    # auto
    if speed_active and _is_cuda_nvidia(target) and _cudnn_attention_supported():
        return "_native_cudnn"
    return None


def _cudnn_attention_supported() -> bool:
    """cuDNN fused SDPA needs Ampere+ (SM80). On pre-SM80 NVIDIA cards (T4 SM75 /
    V100 SM70) diffusers accepts ``_native_cudnn`` at set time but the kernel fails at
    the first generation, so gate the auto-cuDNN upgrade on capability. Unknown
    capability allows it (diffusers' set-time check + the run-time fallback still guard)."""
    have = _cuda_capability()
    return have is None or have >= (8, 0)


def apply_attention_backend(
    pipe: Any,
    backend: Optional[str],
    *,
    logger: Any = None,
) -> Optional[str]:
    """Set ``backend`` on ``pipe.transformer`` via the diffusers dispatcher.

    Returns the backend actually engaged, or None when left at the native default (either
    because ``backend`` was None or because the requested kernel was unavailable -> graceful
    fallback, never a load failure).

    diffusers keeps a *process-wide* active attention backend that ``set_attention_backend``
    also updates, and a fresh transformer's processors follow it (their ``_attention_backend``
    defaults to None). So a load that wants native must restore it explicitly: otherwise it
    silently inherits a backend an earlier load pinned (e.g. cuDNN under a speed profile),
    breaking the bit-identical/``off`` guarantee. Best-effort throughout."""
    transformer = getattr(pipe, "transformer", None)
    fn = getattr(transformer, "set_attention_backend", None)
    if not callable(fn):
        return None
    if backend is not None:
        try:
            fn(backend)
            if logger is not None:
                logger.info("diffusion.attention: backend=%s", backend)
            return backend
        except Exception as exc:  # noqa: BLE001 — unavailable kernel -> restore native below
            _warn(logger, backend, exc)
    # No backend requested, or the requested one failed: pin the native default so a stale
    # process-wide backend from a previous load can't leak into this one.
    _restore_native_backend(fn, logger)
    return None


def _active_attention_backend() -> Optional[str]:
    """The diffusers process-wide active attention backend name, or None if undeterminable."""
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry

        # get_active_backend() returns an AttentionBackend enum (or None), NOT a tuple:
        # unpacking it as `name, _` raises ValueError (swallowed below), so this always
        # returned None and the native-restore short-circuit never fired.
        backend = _AttentionBackendRegistry.get_active_backend()
        if backend is None:
            return None
        return getattr(backend, "value", str(backend))
    except Exception:  # noqa: BLE001
        return None


def _restore_native_backend(set_backend_fn: Any, logger: Any) -> None:
    """Force the native default when the global active backend isn't already native."""
    if _active_attention_backend() == ATTN_NATIVE:
        return  # already native -> avoid redundant work and an extra dispatcher warning
    try:
        set_backend_fn(ATTN_NATIVE)
    except Exception as exc:  # noqa: BLE001 — best-effort restore
        _warn(logger, ATTN_NATIVE, exc)


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.attention: %s unavailable (%s); using default", what, exc)
