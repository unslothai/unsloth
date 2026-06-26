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
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized:
        return ATTN_AUTO
    if normalized not in ATTN_ALIASES:
        raise ValueError(
            f"Unsupported attention_backend '{value}'. Use one of: {', '.join(ATTN_ALIASES)}."
        )
    return normalized


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
    target: Any,
    requested: Optional[str],
    *,
    speed_active: bool,
) -> Optional[str]:
    """The dispatcher backend name to apply, or None to leave the diffusers default.

    An explicit alias is honored verbatim (apply falls back if its kernel is unavailable).
    ``auto`` upgrades to cuDNN on NVIDIA CUDA only when a speed profile is active (so
    ``off`` stays bit-identical); everywhere else it returns None (native default)."""
    alias = normalize_attention_backend(requested)
    if alias != ATTN_AUTO:
        backend = _ALIASES[alias]
        return None if backend == "native" else backend
    # auto
    if speed_active and _is_cuda_nvidia(target):
        return "_native_cudnn"
    return None


def apply_attention_backend(
    pipe: Any,
    backend: Optional[str],
    *,
    logger: Any = None,
) -> Optional[str]:
    """Set ``backend`` on ``pipe.transformer`` via the diffusers dispatcher.

    Returns the backend actually engaged, or None when left at the default (either because
    ``backend`` was None or because the requested kernel was unavailable -> graceful
    fallback to the diffusers default, never a load failure). Best-effort."""
    if backend is None:
        return None
    transformer = getattr(pipe, "transformer", None)
    fn = getattr(transformer, "set_attention_backend", None)
    if not callable(fn):
        return None
    try:
        fn(backend)
        if logger is not None:
            logger.info("diffusion.attention: backend=%s", backend)
        return backend
    except Exception as exc:  # noqa: BLE001 — unavailable kernel -> diffusers default
        _warn(logger, backend, exc)
        return None


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.attention: %s unavailable (%s); using default", what, exc)
