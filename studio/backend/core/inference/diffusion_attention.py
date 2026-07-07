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
# and then crashes mid-generation. Gate them up front by a (min, max-exclusive) compute
# capability range. FlashAttention 3 is a Hopper-SM90 rewrite with no Blackwell kernel, so it
# needs an upper bound: an explicit flash3 on a B200 (SM100) must drop to native instead of
# setting fine then crashing at generation. FlashAttention 4 is Blackwell+ (no upper bound).
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
    """False only when ``backend`` needs a CUDA arch outside this device's supported range.

    Unknown capability (no CUDA / detection failure) returns True so we never block on a
    guess -- diffusers' own set-time check still guards the package, and a genuine run-time
    failure falls back to native."""
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

    An explicit alias is honored verbatim (apply falls back if its kernel is unavailable).
    ``auto`` upgrades to cuDNN on NVIDIA CUDA only when a speed profile is active (so
    ``off`` stays bit-identical); everywhere else it returns None (native default)."""
    alias = normalize_attention_backend(requested)
    if alias != ATTN_AUTO:
        backend = _ALIASES[alias]
        if backend == "native":
            return None
        # Every explicit kernel here (cuDNN / flash* / sage) is CUDA+NVIDIA-only; on
        # ROCm / MPS / CPU diffusers accepts the name at set time and the first
        # generation crashes, so drop to the native default up front.
        if not _is_cuda_nvidia(target):
            return None
        # An arch-gated kernel (flash3/flash4) on a card that can't run it would set fine
        # then crash mid-generation, so drop it to the native default up front.
        if not _backend_arch_supported(backend):
            return None
        # cuDNN fused SDPA needs Ampere+ (SM80); diffusers accepts it on pre-SM80 cards
        # (T4/V100) then fails at the first generation, so apply the same gate to an
        # explicit cuDNN request as the auto path already does.
        if backend == "_native_cudnn" and not _cudnn_attention_supported():
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


# Optional-kernel backends the loader may install on demand: dispatcher name ->
# (probe module, pip package). Only wheels are ever installed (--only-binary=:all:):
# a source build of flash-attn or sageattention takes tens of minutes and needs a
# CUDA toolchain, which a Studio host cannot be assumed to have -- no wheel for this
# python/torch/cuda combo means the request falls back to the native default exactly
# as an uninstallable kernel does today. cuDNN/native need nothing (ship with torch).
_INSTALLABLE_BACKENDS: dict[str, tuple[str, str]] = {
    "sage": ("sageattention", "sageattention"),
    "flash": ("flash_attn", "flash-attn"),
    "_flash_3_hub": ("kernels", "kernels"),  # FA3/FA4 stream from the HF kernels hub
    "flash_4_hub": ("kernels", "kernels"),
    "xformers": ("xformers", "xformers"),
}

# Gate for the on-demand install, mirroring UNSLOTH_DIFFUSION_SD_CPP_INSTALL:
#   auto (default) / 1 - install the missing package when a gated backend is requested
#   0                  - never install; a missing kernel falls back to native
_ATTENTION_INSTALL_ENV = "UNSLOTH_DIFFUSION_ATTENTION_INSTALL"

# Packages a pip install has already been attempted for in THIS process (success or
# failure). The loader pre-installs the kernel OUTSIDE its locks and then re-resolves the
# same backend under _generate_lock, where apply_attention_backend would otherwise call
# pip a SECOND time -- for a package with no matching wheel / an offline host that repeat
# runs the full (up to 600s) install while holding the load lock, blocking unload/cancel/
# new loads for exactly the failure the pre-install was added to keep off the lock. Record
# each attempt so a retry is a no-op and set_attention_backend falls back to native at once.
_INSTALL_ATTEMPTED: set[str] = set()


def _ensure_attention_backend_installed(backend: str, logger: Any = None) -> None:
    """Best-effort wheel-only install of the package ``backend`` needs, when allowed.

    Called after arch gating (select_attention_backend already dropped kernels this
    card cannot run), so an install attempt is only made for a backend that could
    actually work here. Failure is logged and swallowed: the subsequent
    set_attention_backend raises on the still-missing package and the load falls
    back to the native default, same as before this hook existed."""
    import importlib.util
    import os

    spec = _INSTALLABLE_BACKENDS.get(backend)
    if spec is None:
        return
    module, package = spec
    gate = os.environ.get(_ATTENTION_INSTALL_ENV, "auto").strip().lower()
    if gate in ("0", "false", "no", "off"):
        return
    try:
        if importlib.util.find_spec(module) is not None:
            return
    except Exception:  # noqa: BLE001 — a broken install probes as missing; try the install
        pass
    # Only ever attempt each package's install once per process. The loader pre-installs
    # this backend outside its locks; if that failed (no wheel / offline) the module is
    # still missing here, so without this guard the in-lock apply path would re-run the
    # whole install under _generate_lock and block unload/cancel. A recorded attempt makes
    # the retry a no-op -> set_attention_backend raises on the missing package -> native.
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
            # --no-deps: install ONLY this best-effort kernel wheel, never its declared
            # dependencies. xformers/flash-attn pin an exact torch (e.g. torch==2.x), so
            # normal resolution would upgrade/replace the running torch/triton and leave
            # later loads on a different, possibly CUDA-mismatched dependency stack. Without
            # its deps an ABI-incompatible kernel simply fails to import -> native fallback,
            # which is the same best-effort outcome as an uninstallable wheel.
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
        # The wheel just landed in site-packages, but the import system caches each
        # directory's listing; the very next find_spec / import in this same process can
        # still miss the freshly installed package when the install lands within the
        # directory mtime's resolution -- silently falling back to native on the first
        # use. Invalidate the finder caches so set_attention_backend picks it up now.
        importlib.invalidate_caches()
    except Exception as exc:  # noqa: BLE001 — no wheel / no network -> native fallback
        if logger is not None:
            # A failed pip install raises CalledProcessError whose str() shows only the
            # exit code and command; the real reason (no matching wheel, resolver error)
            # is in exc.stderr. Surface it so a fallback to native is diagnosable.
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
    """Every DiT the denoise loop runs each step: the primary ``transformer`` plus a second
    expert some families carry (Ideogram's ``unconditional_transformer`` for its dual-branch
    CFG, an MoE ``transformer_2``). The attention backend must be set on ALL of them, else the
    second DiT keeps the native default while status reports the requested kernel as engaged."""
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
    """Set ``backend`` on EVERY denoiser DiT (``pipe.transformer`` plus a second expert such as
    Ideogram's ``unconditional_transformer``) via the diffusers dispatcher.

    Returns the backend actually engaged, or None when left at the native default (either
    because ``backend`` was None or because the requested kernel was unavailable -> graceful
    fallback, never a load failure).

    diffusers keeps a *process-wide* active attention backend that ``set_attention_backend``
    also updates, and a fresh transformer's processors follow it (their ``_attention_backend``
    defaults to None). So a load that wants native must restore it explicitly: otherwise it
    silently inherits a backend an earlier load pinned (e.g. cuDNN under a speed profile),
    breaking the bit-identical/``off`` guarantee. Best-effort throughout."""
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
            # set_attention_backend also pins the backend in diffusers' process-wide registry.
            # Each DiT's own processors now keep it locally (their _attention_backend is now
            # explicit), so reset the global default back to native ONCE -- otherwise a later
            # component whose processors are unconfigured (backend None) inherits this kernel.
            _reset_global_backend_to_native(logger)
            if logger is not None:
                logger.info("diffusion.attention: backend=%s", backend)
            return backend
    # No backend requested, or every set failed: pin the native default so a stale process-wide
    # backend from a previous load can't leak into this one. Fresh DiTs follow the process-wide
    # backend, so one reset via any DiT's setter covers them all.
    _restore_native_backend(setters[0], logger)
    return None


def _active_attention_backend() -> Optional[str]:
    """The diffusers process-wide active attention backend name, or None if undeterminable."""
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry

        # get_active_backend() returns a (AttentionBackendName, fn) tuple (or None), so
        # take element 0 and read its .value (e.g. "native"); reading .value off the
        # tuple itself would yield a junk string that never compares equal to a name.
        active = _AttentionBackendRegistry.get_active_backend()
        if active is None:
            return None
        name = active[0] if isinstance(active, tuple) else active
        return getattr(name, "value", str(name))
    except Exception:  # noqa: BLE001
        return None


def _reset_global_backend_to_native(logger: Any) -> None:
    """Reset diffusers' process-wide active attention backend to native after a
    successful per-transformer set, so a later component whose processors are
    unconfigured (backend None) does not inherit this transformer's kernel. The
    transformer's own processors keep the backend just set. Best-effort and silent:
    if the diffusers internals move, the prior (leaking) behavior is unchanged."""
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
