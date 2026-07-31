# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Select the diffusion transformer's attention backend.

diffusers' ``transformer.set_attention_backend(name)`` dispatcher swaps the SDPA kernel,
validating hardware/package at set time (default ``native`` = ``F.scaled_dot_product_attention``).
Attention is bandwidth-bound, so a better kernel is a real win orthogonal to weight quantisation
(it speeds the QK/PV matmuls torchao never touches) and composes with torch.compile.

  auto  - the best *exact* backend for the device. On NVIDIA CUDA that is cuDNN fused attention
          (``_native_cudnn``), ~1.18x end-to-end on B200, LPIPS ~0.004 (below the noise floor).
          Elsewhere stays ``native``. Only upgrades when a speed profile is active, so
          ``speed_mode=off`` stays bit-identical.
  native - force the default SDPA (bit-identical reference).
  cudnn  - cuDNN fused attention (exact; NVIDIA).
  flash / flash3 / flash4 - FlashAttention 2 / 3 (Hopper) / 4 (SM100); exact, kernel-gated.
  sage   - SageAttention (INT8 QK); quantized, small quality cost, consumer-friendly.
  xformers / aiter - memory-efficient (NVIDIA) / AITER (AMD ROCm).

Best-effort: an unavailable backend falls back to the diffusers default. torch/diffusers lazy.
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
    """Lower/strip a requested backend; None / "" / "auto" -> "auto". Raises ValueError for an
    unsupported alias so a bad request is rejected cheaply."""
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


# Backends diffusers validates by package at set time but whose kernels need a specific CUDA arch at run time. Gate by a (min, max-exclusive) capability range: FA3 is Hopper-SM90 only, FA4 is Blackwell+.
_ARCH_CAPABILITY: dict[str, tuple[tuple[int, int], Optional[tuple[int, int]]]] = {
    "_flash_3_hub": ((9, 0), (10, 0)),  # FlashAttention 3 -> Hopper (SM90) only
    "flash_4_hub": ((10, 0), None),  # FlashAttention 4 -> Blackwell (SM100)+
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
    """False only when ``backend`` needs a CUDA arch outside this device's range. Unknown
    capability returns True (never block on a guess; the run-time failure falls back to native)."""
    bounds = _ARCH_CAPABILITY.get(backend)
    if bounds is None:
        return True
    have = _cuda_capability()
    if have is None:
        return True
    low, high = bounds
    return have >= low and (high is None or have < high)


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

    An explicit alias is honored (apply falls back if its kernel is unavailable). ``auto``
    upgrades to cuDNN on NVIDIA CUDA only when a speed profile is active (so ``off`` stays
    bit-identical); elsewhere returns None (native)."""
    alias = normalize_attention_backend(requested)
    if alias != ATTN_AUTO:
        backend = _ALIASES[alias]
        if backend == "native":
            return None
        # AITER is the AMD ROCm kernel: honor it on a ROCm target, else the NVIDIA-only guard below drops the one backend that works there.
        if backend == "aiter":
            if getattr(target, "device", None) == "cuda" and not _is_cuda_nvidia(target):
                return backend
            return None
        # cuDNN / flash* / sage are CUDA+NVIDIA-only; elsewhere the first generation crashes.
        if not _is_cuda_nvidia(target):
            return None
        # An arch-gated kernel (flash3/flash4) on a card that can't run it sets fine then crashes.
        if not _backend_arch_supported(backend):
            return None
        # cuDNN fused SDPA needs Ampere+ (SM80); gate an explicit request like the auto path.
        if backend == "_native_cudnn" and not _cudnn_attention_supported():
            return None
        return backend
    # auto
    if speed_active and _is_cuda_nvidia(target) and _cudnn_attention_supported():
        return "_native_cudnn"
    return None


def _cudnn_attention_supported() -> bool:
    """cuDNN fused SDPA needs Ampere+ (SM80); on pre-SM80 cards (T4/V100) diffusers accepts it
    then fails at generation, so gate the upgrade on capability. Unknown capability allows it."""
    have = _cuda_capability()
    return have is None or have >= (8, 0)


# Optional kernels installable on demand: dispatcher name -> (probe module, pip package). Wheels only (--only-binary=:all:), since a source build needs a CUDA toolchain the host may lack.
_INSTALLABLE_BACKENDS: dict[str, tuple[str, str]] = {
    "sage": ("sageattention", "sageattention>=2.1.1"),
    "flash": ("flash_attn", "flash-attn"),
    "_flash_3_hub": ("kernels", "kernels"),  # FA3/FA4 from the HF kernels hub
    "flash_4_hub": ("kernels", "kernels"),
    "xformers": ("xformers", "xformers"),
}

# On-demand install gate (mirrors UNSLOTH_DIFFUSION_SD_CPP_INSTALL): auto (default) / 1 installs a missing package when a gated backend is requested; 0 never installs and falls back to native.
_ATTENTION_INSTALL_ENV = "UNSLOTH_DIFFUSION_ATTENTION_INSTALL"

# Packages a pip install was already attempted for in THIS process. The loader pre-installs outside its locks, so a recorded attempt stops apply re-running the 600s install under _generate_lock.
_INSTALL_ATTEMPTED: set[str] = set()


def _pip_requirement(backend: str, package: str) -> str:
    """Requirement to hand pip, carrying any floor the dispatcher enforces at set time.

    PyPI's newest ``sageattention`` wheel is 1.0.6 while diffusers refuses anything below
    ``_REQUIRED_SAGE_VERSION`` (2.1.1) — an unpinned install therefore always "succeeds", writes
    an unusable 1.0.6 into the running venv, and is then rejected with "the version is too old".
    Pinning makes pip resolve nothing instead of installing something we will not use. Re-read the
    floor from diffusers so a future bump tracks automatically."""
    if backend != "sage":
        return package
    try:
        from diffusers.models.attention_dispatch import _REQUIRED_SAGE_VERSION as floor
        if isinstance(floor, str) and floor.strip():
            return f"sageattention>={floor.strip()}"
    except Exception:  # noqa: BLE001 — older/newer diffusers may not expose it; keep the static pin
        pass
    return package


def _ensure_attention_backend_installed(backend: str, logger: Any = None) -> None:
    """Best-effort wheel-only install of the package ``backend`` needs, when allowed.

    Called after arch gating, so only for a backend that could work here. Failure is swallowed:
    the subsequent set_attention_backend raises on the missing package and falls back to native."""
    import importlib.util
    import os

    spec = _INSTALLABLE_BACKENDS.get(backend)
    if spec is None:
        return
    module, package = spec
    package = _pip_requirement(backend, package)
    gate = os.environ.get(_ATTENTION_INSTALL_ENV, "auto").strip().lower()
    if gate in ("0", "false", "no", "off"):
        return
    try:
        if importlib.util.find_spec(module) is not None:
            return
    except Exception:  # noqa: BLE001 — a broken install probes as missing; try the install
        pass
    # Attempt each install once per process, else the in-lock apply re-runs it under _generate_lock and blocks unload/cancel.
    if package in _INSTALL_ATTEMPTED:
        return
    _INSTALL_ATTEMPTED.add(package)
    import subprocess
    import sys

    if logger is not None:
        logger.info(
            "diffusion.attention: installing %s for backend=%s (wheel-only)", package, backend
        )
    try:
        subprocess.run(
            # --no-deps: install ONLY this kernel wheel, since xformers/flash-attn pin an exact torch and normal resolution would replace the running one. An ABI mismatch just fails to import.
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--only-binary",
                ":all:",
                "--no-deps",
                package,
            ],
            capture_output = True,
            timeout = 600,
            check = True,
        )
        # The import system caches directory listings, so invalidate the finder caches or the next find_spec can miss the new wheel.
        importlib.invalidate_caches()
    except Exception as exc:  # noqa: BLE001 — no wheel / no network -> native fallback
        if logger is not None:
            # CalledProcessError.str() shows only the exit code; surface stderr so the fallback is diagnosable.
            stderr = getattr(exc, "stderr", None)
            if stderr:
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors = "replace")
                logger.warning(
                    "diffusion.attention: could not install %s; pip failed with: %s",
                    package,
                    stderr.strip() or str(exc),
                )
            else:
                logger.warning(
                    "diffusion.attention: could not install %s (%s); falling back to default",
                    package,
                    exc,
                )


def _attention_dits(pipe: Any) -> list:
    """Every DiT the denoise loop runs: the primary ``transformer`` plus a second expert some
    families carry (Ideogram's ``unconditional_transformer``, an MoE ``transformer_2``). The
    backend must be set on ALL of them, else the second DiT keeps the native default."""
    dits: list = []
    for attr in ("transformer", "transformer_2", "unconditional_transformer"):
        m = getattr(pipe, attr, None)
        if m is not None and m not in dits:
            dits.append(m)
    return dits


def apply_attention_backend(
    pipe: Any,
    backend: Optional[str],
    *,
    logger: Any = None,
) -> Optional[str]:
    """Set ``backend`` on EVERY denoiser DiT via the diffusers dispatcher.

    Returns the backend engaged, or None when left at native (``backend`` was None or the kernel
    was unavailable -> graceful fallback, never a load failure).

    diffusers keeps a process-wide active backend that ``set_attention_backend`` also updates, and
    a fresh transformer's processors follow it (default None). So a load wanting native must
    restore it explicitly, else it inherits a backend an earlier load pinned (e.g. cuDNN under a
    speed profile), breaking the ``off`` guarantee. Best-effort."""
    setters = [
        s
        for s in (getattr(t, "set_attention_backend", None) for t in _attention_dits(pipe))
        if callable(s)
    ]
    if not setters:
        return None
    if backend is not None:
        _ensure_attention_backend_installed(backend, logger)
        engaged = False
        for fn in setters:
            try:
                fn(backend)
                engaged = True
            except Exception as exc:  # noqa: BLE001 — unavailable kernel -> restore native below
                _warn(logger, backend, exc)
        if engaged:
            # set_attention_backend also pins the backend process-wide. Each DiT's processors keep it locally, so reset the global to native ONCE, else a later component inherits this kernel.
            _reset_global_backend_to_native(logger)
            if logger is not None:
                logger.info("diffusion.attention: backend=%s", backend)
            return backend
    # No backend requested, or every set failed: pin native so a stale process-wide backend cannot leak in. One reset covers every fresh DiT.
    _restore_native_backend(setters[0], logger)
    return None


def _active_attention_backend() -> Optional[str]:
    """The diffusers process-wide active attention backend name, or None if undeterminable."""
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry

        # get_active_backend() returns (AttentionBackendName, fn) or None; read element 0's .value, not the tuple.
        active = _AttentionBackendRegistry.get_active_backend()
        if active is None:
            return None
        name = active[0] if isinstance(active, tuple) else active
        return getattr(name, "value", str(name))
    except Exception:  # noqa: BLE001
        return None


def _reset_global_backend_to_native(logger: Any) -> None:
    """Reset the process-wide active backend to native after a successful per-transformer set, so
    a later unconfigured component doesn't inherit this kernel (the DiT's own processors keep it).
    Best-effort: if the diffusers internals move, the prior (leaking) behavior is unchanged."""
    if _active_attention_backend() == ATTN_NATIVE:
        return
    try:
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _AttentionBackendRegistry,
        )
        _AttentionBackendRegistry.set_active_backend(AttentionBackendName.NATIVE)
    except Exception:  # noqa: BLE001 — best-effort; leave the global as-is on any change
        pass


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
