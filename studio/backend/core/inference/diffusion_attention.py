# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Select the diffusion transformer's attention backend.

diffusers' ``transformer.set_attention_backend(name)`` dispatcher swaps the SDPA kernel,
validating hardware/package at set time (default ``native`` = ``F.scaled_dot_product_attention``).
Attention is bandwidth-bound, so a better kernel is a real win orthogonal to weight quantisation
(it speeds the QK/PV matmuls torchao never touches) and composes with torch.compile.

  auto  - the best *exact* backend for the device. On NVIDIA CUDA that is cuDNN fused attention
          (``_native_cudnn``). Torch's SDPA dispatch is a HEURISTIC, not a guarantee, and that is
          the reason to pin it: re-measured on torch 2.12 / B200 at Qwen-Image's 1024px shape
          (B=1 H=24 N=4352 D=128 bf16) the default ALREADY lands on cuDNN, so pinning is
          bitwise-identical and near free (compiled 1.599s -> 1.572s, 1.02x; eager 0.93x, the
          per-call sdpa_kernel wrapper cost, which compile folds away) -- but FLASH and EFFICIENT
          at that same shape run 3.9x and 9.0x slower. Pinning is insurance against the heuristic
          picking one of those on another card, head_dim or torch build. Elsewhere stays
          ``native``. Only upgrades when a speed profile is active, so ``speed_mode=off`` stays
          bit-identical.
  native - force the default SDPA (bit-identical reference).
  cudnn  - cuDNN fused attention (exact; NVIDIA).
  flash / flash3 / flash4 - FlashAttention 2 / 3 (Hopper) / 4 (SM100); exact, kernel-gated.
  sage   - SageAttention 2 (INT8 QK); quantized, small quality cost. A pip SageAttention 2 when installed, else the
           kernels-community build from the Hugging Face kernels hub (``sage_hub``, Ampere / Ada / Hopper).
  xformers / aiter - memory-efficient (NVIDIA) / AITER (AMD ROCm).
  flash on ROCm - only when the DAO-AILab ROCm build imports and passes an on-card check; ``auto`` stays native.

Best-effort: an unavailable backend falls back to the diffusers default. torch/diffusers lazy.
"""

from __future__ import annotations

import functools
import os
import re
import threading
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


# Backends diffusers validates by package at set time but whose kernels need a specific CUDA arch at run time. Gate by a
# (min, max-exclusive) capability range: FA3 is Hopper-SM90 only, FA4 is Blackwell+.
_ARCH_CAPABILITY: dict[str, tuple[tuple[int, int], Optional[tuple[int, int]]]] = {
    "_flash_3_hub": ((9, 0), (10, 0)),  # Hopper (SM90) only
    "flash_4_hub": (
        (10, 0),
        None,
    ),  # SM100+; sm110 / sm120 unverified, so _fa4_kernel_runs also checks each card
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

        # Shared with the stub installer: torch.version.hip alone misreads AMD wheels that only tag __version__,
        # dropping aiter and pointing cuDNN/xformers (stubbed there) at a ROCm card.
        from core._torchao_stub import _module_is_rocm
        return not _module_is_rocm(torch)
    except Exception:  # noqa: BLE001
        return False


# What the native SDPA dispatch can actually run (#8225). torch's ``flash_sdp_enabled()`` /
# ``mem_efficient_sdp_enabled()`` report the USER TOGGLE, not whether a kernel exists for this device. On the ROCm
# build in #8225 (gfx1200, torch 2.11+rocm7) both answer True while every dispatch to them raises "No available
# kernel. Aborting execution.", so the dispatcher degrades silently to MATH -- the one backend that materialises the
# whole B x heads x N x N score matrix, which is how a 3.4 GB Q4_K_M video model asked a 16 GB card for a single 66.54
# GiB allocation. Probe actual execution first, then intersect that capability with the current process flags.
SDPA_FLASH = "flash"
SDPA_MEM_EFFICIENT = "mem_efficient"
SDPA_CUDNN = "cudnn"
SDPA_MATH = "math"

# Backends whose working set is O(N): they never materialise the score matrix.
_SDPA_SUBQUADRATIC = (SDPA_FLASH, SDPA_MEM_EFFICIENT, SDPA_CUDNN)

_SDPA_PROBE_LOCK = threading.Lock()
# (indexed device, dtype name) -> capability; process flags are applied on every read.
_SDPA_PROBE_CACHE: dict[tuple[str, str], tuple[str, ...]] = {}


# An incomplete ROCm probe is retried, but not by every check of the same load (up to 90 s per attempt).
_ROCM_PROBE_RETRY_S = 120.0
_ROCM_PROBE_INCOMPLETE: dict[tuple[str, str], float] = {}


def _probe_rocm_sdpa_kernels(device: str, dtype: Any) -> tuple[str, ...]:
    import time

    key = (device, str(dtype))
    last = _ROCM_PROBE_INCOMPLETE.get(key)
    if last is not None and time.monotonic() - last < _ROCM_PROBE_RETRY_S:
        return ()
    available = _run_rocm_sdpa_children(device, dtype)
    if available:
        _ROCM_PROBE_INCOMPLETE.pop(key, None)
    else:
        _ROCM_PROBE_INCOMPLETE[key] = time.monotonic()
    return available


def _run_rocm_sdpa_children(device: str, dtype: Any) -> tuple[str, ...]:
    """Isolate each backend's HIP error state from Studio and from the other probes."""
    from concurrent.futures import ThreadPoolExecutor
    import json
    from pathlib import Path
    import subprocess
    import sys
    from utils.child_stdio import utf8_child_env
    from utils.native_path_leases import child_env_without_native_path_secret
    from utils.subprocess_compat import windows_hidden_subprocess_kwargs
    from utils.torch_device_probe import ROCM_DLL_DIRS_ENV_VAR, _rocm_dll_directories
    from .rocm_sdpa_probe import RESULT_PREFIX

    # Same bootstrap as the device probe: a fresh interpreter lacks main.py's ROCm DLL registrations.
    child_env = child_env_without_native_path_secret()
    dll_directories = _rocm_dll_directories()
    if dll_directories:
        child_env[ROCM_DLL_DIRS_ENV_VAR] = os.pathsep.join(dll_directories)
    child_env = utf8_child_env(child_env)

    def probe(backend: str) -> str:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("rocm_sdpa_probe.py")),
                    device,
                    str(dtype).removeprefix("torch."),
                    backend,
                ],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 45,
                env = child_env,
                **windows_hidden_subprocess_kwargs(),
            )
            replies = [
                line[len(RESULT_PREFIX) :]
                for line in result.stdout.splitlines()
                if line.startswith(RESULT_PREFIX)
            ]
            return json.loads(replies[-1]) if result.returncode == 0 and replies else "unknown"
        except Exception:
            return "unknown"

    if probe("MATH") != "available":
        return ()
    # One child per backend, run concurrently: an in-process fused failure would poison Studio.
    candidates = (
        (SDPA_FLASH, "FLASH_ATTENTION"),
        (SDPA_MEM_EFFICIENT, "EFFICIENT_ATTENTION"),
    )
    with ThreadPoolExecutor(max_workers = 2) as executor:
        statuses = list(executor.map(probe, (backend for _, backend in candidates)))
    available = [SDPA_MATH]
    for (name, _), status in zip(candidates, statuses):
        if status not in ("available", "unavailable", "failed"):
            return ()
        if status == "available":
            available.append(name)
    return tuple(available)


def _enabled_sdpa_kernels(kernels: tuple[str, ...]) -> tuple[str, ...]:
    if not kernels:
        return ()
    import torch

    flags = {
        SDPA_FLASH: "flash_sdp_enabled",
        SDPA_MEM_EFFICIENT: "mem_efficient_sdp_enabled",
        SDPA_CUDNN: "cudnn_sdp_enabled",
        SDPA_MATH: "math_sdp_enabled",
    }
    return tuple(k for k in kernels if getattr(torch.backends.cuda, flags[k], lambda: True)())


def _probe_sdpa_kernels(device: str, dtype: Any) -> tuple[str, ...]:
    """Probe execution; ROCm runs in fresh children so launch errors stay isolated."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    candidates = (
        (SDPA_FLASH, getattr(SDPBackend, "FLASH_ATTENTION", None)),
        (SDPA_MEM_EFFICIENT, getattr(SDPBackend, "EFFICIENT_ATTENTION", None)),
        (SDPA_CUDNN, getattr(SDPBackend, "CUDNN_ATTENTION", None)),
        (SDPA_MATH, getattr(SDPBackend, "MATH", None)),
    )
    from core._torchao_stub import _module_is_rocm

    if device.split(":")[0] == "cuda" and _module_is_rocm(torch):
        return _probe_rocm_sdpa_kernels(device, dtype)
    # Small enough to be free, but shaped like real attention: the fused kernels reject head_dim they cannot serve, and
    # a degenerate 1-element tensor would not exercise that.
    q = torch.zeros((1, 2, 8, 64), device = device, dtype = dtype)
    available: list[str] = []
    for name, backend in candidates:
        if backend is None:
            continue
        try:
            with sdpa_kernel([backend]):
                out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            # Windows ROCm raises a failed fused launch only at the next checked kernel: take it here.
            out.float().sum().item()
            available.append(name)
        except Exception:  # noqa: BLE001 - "No available kernel" is the answer, not an error
            continue
    return tuple(available)


def available_sdpa_kernels(target: Any) -> tuple[str, ...]:
    """The SDPA backends that actually EXECUTE on ``target``, cheapest source of truth available.

    Empty when the probe could not run at all (no torch, no device, an allocator failure) -- an
    unanswerable probe must never be read as "only math", which is a claim about the hardware."""
    return _enabled_sdpa_kernels(_sdpa_capability(target))


def _sdpa_capability(target: Any) -> tuple[str, ...]:
    device = str(getattr(target, "torch_device", None) or getattr(target, "device", "") or "")
    if device == "cuda":
        ordinal = getattr(target, "ordinal", None)
        device = f"cuda:{ordinal}" if ordinal is not None else _indexed_cuda_device(device)
    if not device:
        return ()
    dtype = getattr(target, "dtype", None)
    if dtype is None:
        try:
            import torch

            # fp16 rather than fp32: the fused kernels are half-precision only, so probing at fp32 would report "math
            # only" on hardware where flash is perfectly healthy.
            dtype = torch.float16
        except Exception:  # noqa: BLE001
            return ()
    key = (device, str(dtype))
    with _SDPA_PROBE_LOCK:
        cached = _SDPA_PROBE_CACHE.get(key)
        if cached is not None:
            return cached
    try:
        available = _probe_sdpa_kernels(device, dtype)
    except Exception:  # noqa: BLE001 - a probe is a diagnostic; it may never fail a load
        available = ()
    # memoize only an ANSWER: an empty result means the probe itself could not complete (a transient allocator failure
    # while the device was full, exactly when this warning matters most)
    if not available:
        return ()
    with _SDPA_PROBE_LOCK:
        return _SDPA_PROBE_CACHE.setdefault(key, available)


def sdpa_math_only(target: Any) -> bool:
    """True only when MATH ran and every sub-quadratic backend refused.

    A probe that answered nothing at all returns False: silence is not evidence."""
    available = available_sdpa_kernels(target)
    if not available:
        return False
    return SDPA_MATH in available and not any(k in available for k in _SDPA_SUBQUADRATIC)


def sdpa_subquadratic_confirmed(target: Any) -> bool:
    """True only when a sub-quadratic SDPA kernel was seen to run; an unanswered probe is False."""
    return any(k in _SDPA_SUBQUADRATIC for k in available_sdpa_kernels(target))


SDPA_MATH_ONLY_MESSAGE = (
    "attention has no fused kernel on this device, so it will run on the SDPA math backend, "
    "which materialises the full attention score matrix (batch x heads x tokens x tokens, 4 bytes "
    "per element). Peak VRAM then grows with the SQUARE of the token count -- resolution x frames "
    "for video -- and can exceed the card by many times even for a small model. Lower the "
    "resolution or the frame count, or install a backend with a working kernel for this GPU."
)


def warn_if_sdpa_math_only(target: Any, logger: Any = None) -> bool:
    """Log the math-only diagnosis at LOAD, not 70 seconds into a doomed generation.

    Returns whether the warning fired, so callers can carry it into their resolved controls."""
    if not sdpa_math_only(target):
        return False
    if logger is not None:
        logger.warning(
            "diffusion.attention.math_only: device=%s dtype=%s kernels=%s -- %s",
            getattr(target, "device", None),
            getattr(target, "dtype", None),
            ",".join(available_sdpa_kernels(target)) or "none",
            SDPA_MATH_ONLY_MESSAGE,
        )
    return True


ROCM_FUSED_SDPA_ALLOW_ENV = "UNSLOTH_ALLOW_ROCM_FUSED_SDPA"
_ROCM_GUARD_DISABLED: set[str] = set()


def guard_rocm_fused_sdpa(target: Any, logger: Any = None) -> tuple[str, ...]:
    """Disable failed ROCm fused backends only after math has been verified in isolation."""
    if not _is_cuda_rocm(target):
        return ()
    if os.environ.get(ROCM_FUSED_SDPA_ALLOW_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        return ()
    import torch

    from types import SimpleNamespace
    from .rocm_bf16 import rocm_bf16_supported

    device = str(getattr(target, "torch_device", None) or getattr(target, "device", "cuda"))
    ordinal = getattr(target, "ordinal", None)
    if device == "cuda":
        device = f"cuda:{ordinal}" if ordinal is not None else _indexed_cuda_device(device)
    dtype = getattr(target, "dtype", None)
    if dtype not in (torch.float16, torch.bfloat16):
        dtype = (
            torch.bfloat16
            if rocm_bf16_supported(torch, torch.device(device).index)
            else torch.float16
        )
    available = _sdpa_capability(SimpleNamespace(device = device, dtype = dtype))
    if SDPA_MATH not in available:
        return ()
    disabled = []
    for name, enabled, switch in (
        (SDPA_FLASH, torch.backends.cuda.flash_sdp_enabled, torch.backends.cuda.enable_flash_sdp),
        (
            SDPA_MEM_EFFICIENT,
            torch.backends.cuda.mem_efficient_sdp_enabled,
            torch.backends.cuda.enable_mem_efficient_sdp,
        ),
    ):
        if name in available:
            # The flags are process-wide: a healthy card gets back only what this guard turned off for another.
            if name in _ROCM_GUARD_DISABLED and not enabled():
                switch(True)
            _ROCM_GUARD_DISABLED.discard(name)
        elif enabled():
            switch(False)
            _ROCM_GUARD_DISABLED.add(name)
            disabled.append(name)
    if disabled:
        (logger or _module_logger()).warning(
            "diffusion.attention: disabled unavailable ROCm SDPA kernels %s; using the remaining kernels. "
            "If fused kernels are expected on this GPU, run 'unsloth studio update' to repair "
            "the AMD device packages. %s=1 skips this check.",
            ", ".join(disabled),
            ROCM_FUSED_SDPA_ALLOW_ENV,
        )
    return tuple(disabled)


def select_attention_backend(
    target: Any,
    requested: Optional[str],
    *,
    speed_active: bool,
    family: Any = None,
    speed_unset: bool = False,
) -> Optional[str]:
    """The dispatcher backend name to apply, or None to leave the diffusers default.

    An explicit alias is honored (apply falls back if its kernel is unavailable). ``auto``
    upgrades to cuDNN on NVIDIA CUDA only when a speed profile is active (so ``off`` stays
    bit-identical), and to verified ROCm flash for ``ROCM_AUTO_FLASH_FAMILIES`` on gfx11 unless speed is
    explicitly ``off``; elsewhere returns None (native)."""
    alias = normalize_attention_backend(requested)
    if alias != ATTN_AUTO:
        backend = _ALIASES[alias]
        if backend == "native":
            return None
        # AITER is the AMD ROCm kernel: honor it on a ROCm target, else the NVIDIA-only guard below drops the one
        # backend that works there
        if backend == "aiter":
            if getattr(target, "device", None) == "cuda" and not _is_cuda_nvidia(target):
                return backend
            return None
        # cuDNN / flash* / sage are CUDA+NVIDIA-only (elsewhere the first generation crashes), except verified ROCm FA2.
        if not _is_cuda_nvidia(target):
            if backend == "flash" and _is_cuda_rocm(target) and _rocm_flash_attn_runs(target):
                return backend
            return None
        # An arch-gated kernel (flash3/flash4) on a card that can't run it sets fine then crashes.
        if not _backend_arch_supported(backend):
            return None
        # cuDNN fused SDPA needs Ampere+ (SM80); gate an explicit request like the auto path.
        if backend == "_native_cudnn" and not _cudnn_attention_supported():
            return None
        return backend
    if speed_active and _is_cuda_nvidia(target) and _cudnn_attention_supported():
        return "_native_cudnn"
    if speed_active and ROCM_AUTO_FLASH and _is_cuda_rocm(target) and _rocm_flash_attn_runs(target):
        return "flash"
    if (speed_active or speed_unset) and _rocm_auto_flash(target, family):
        return "flash"
    return None


def auto_attention_reason(engaged: Optional[str]) -> str:
    """Status reason for an ``auto`` pick that engaged ``engaged`` (None = native)."""
    if not engaged:
        return "diffusers default"
    if "cudnn" in engaged:
        return "cuDNN fused attention upgrade"
    if engaged == "flash":
        return "verified ROCm flash attention"
    return f"{engaged} attention upgrade"


def _is_cuda_rocm(target: Any) -> bool:
    return getattr(target, "device", None) == "cuda" and not _is_cuda_nvidia(target)


# ``auto`` -> verified ROCm flash under a speed profile. Off: gfx1151 gains over AOTriton SDPA were too small/narrow.
ROCM_AUTO_FLASH = False
# Also with speed unset (the UI default). gfx1151 klein 1024^2: 7.37-7.40 s vs SDPA 8.04-8.09 s, LPIPS 0.008.
# FLUX.1 measured LPIPS 0.048 vs SDPA (over the 0.02 bar): stays native.
ROCM_AUTO_FLASH_FAMILIES = frozenset({"flux.2-klein"})
ROCM_AUTO_FLASH_ARCHES = ("gfx11",)


def _rocm_gfx_arch(target: Any) -> str:
    try:
        import torch
        from utils.hardware.hardware import _props_gfx_arch

        device = torch.device(
            _indexed_cuda_device(
                str(getattr(target, "torch_device", None) or getattr(target, "device", None) or "")
            )
        )
        index = device.index if device.index is not None else torch.cuda.current_device()
        return _props_gfx_arch(torch.cuda.get_device_properties(index))
    except Exception:  # noqa: BLE001
        return ""


def _flash_attn_installed() -> bool:
    """Without importing it, so an ``auto`` load on a card without flash_attn logs nothing."""
    import sys

    if "flash_attn" in sys.modules:
        return sys.modules["flash_attn"] is not None
    try:
        import importlib.util
        return importlib.util.find_spec("flash_attn") is not None
    except Exception:  # noqa: BLE001
        return False


def _rocm_auto_flash(target: Any, family: Any) -> bool:
    if not _is_cuda_rocm(target):
        return False
    name = family if isinstance(family, str) else getattr(family, "name", None)
    if name not in ROCM_AUTO_FLASH_FAMILIES:
        return False
    if not _rocm_gfx_arch(target).startswith(ROCM_AUTO_FLASH_ARCHES):
        return False
    return _flash_attn_installed() and _rocm_flash_attn_runs(target)


# Max abs error of the probe's flash_attn output against an fp32 reference (bf16 eps is ~4e-3).
_ROCM_FLASH_PROBE_TOL = 2e-2
# (device, dtype) -> whether flash_attn ran and matched. Only answers are cached; an unaskable probe is retried.
_ROCM_FLASH_PROBE_CACHE: dict[tuple[str, str], bool] = {}
_ROCM_FLASH_MISSING_LOGGED: list[bool] = []

_ROCM_FLASH_HINT = (
    "the PyPI flash-attn wheel is the NVIDIA CUDA build; on ROCm install the DAO-AILab flash-attention ROCm build "
    "(github.com/Dao-AILab/flash-attention, CK backend, or FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE for Triton) "
    "or use attention_backend=aiter"
)


def _module_logger() -> Any:
    try:
        from loggers import get_logger
        return get_logger(__name__)
    except Exception:  # noqa: BLE001
        import logging
        return logging.getLogger(__name__)


def _run_rocm_flash_probe(device: str, dtype: Any) -> bool:
    """Tiny ``flash_attn_func`` on ``device`` vs an fp32 reference; False when it raises or disagrees."""
    import torch
    from flash_attn import flash_attn_func

    if dtype not in (torch.float16, torch.bfloat16):
        # diffusers' flash backend is half precision only, and the CK build raises on fp32 at the first call
        if dtype is not None:
            return False
        dtype = torch.bfloat16
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn((1, 128, 2, 64), generator = gen).to(device, dtype) for _ in range(3))
    try:
        out = flash_attn_func(q, k, v)
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - no kernel for this arch is the answer
        return False
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    ref = torch.softmax(qf @ kf.transpose(-1, -2) * qf.shape[-1] ** -0.5, dim = -1) @ vf
    err = (out.float() - ref.transpose(1, 2)).abs().max()
    return bool(torch.isfinite(err).item() and err.item() <= _ROCM_FLASH_PROBE_TOL)


def _rocm_flash_attn_runs(target: Any) -> bool:
    """True only when flash_attn imports and its kernel ran and matched on this ROCm card at this dtype."""
    device = _indexed_cuda_device(
        str(getattr(target, "torch_device", None) or getattr(target, "device", None) or "")
    )
    dtype = getattr(target, "dtype", None)
    key = (device, str(dtype))
    cached = _ROCM_FLASH_PROBE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import flash_attn  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        if not _ROCM_FLASH_MISSING_LOGGED:
            _ROCM_FLASH_MISSING_LOGGED.append(True)
            _module_logger().warning(
                "diffusion.attention: flash requested on ROCm but flash_attn is not importable (%s); %s. "
                "Using the default backend",
                exc,
                _ROCM_FLASH_HINT,
            )
        return False
    try:
        ok = _run_rocm_flash_probe(device, dtype)
    except Exception:  # noqa: BLE001 - OOM / device trouble: no answer, keep native this time
        return False
    if not ok:
        _module_logger().warning(
            "diffusion.attention: flash_attn did not run or did not match SDPA on %s (%s); using the default backend",
            device,
            dtype,
        )
    return _ROCM_FLASH_PROBE_CACHE.setdefault(key, ok)


def _cudnn_attention_supported() -> bool:
    """cuDNN fused SDPA needs Ampere+ (SM80); on pre-SM80 cards (T4/V100) diffusers accepts it
    then fails at generation, so gate the upgrade on capability. Unknown capability allows it."""
    have = _cuda_capability()
    return have is None or have >= (8, 0)


# Arch support varies by sageattn build, so a run per card is the gate. (device, dtype, head_dim) -> "" or why not.
_SAGE_PROBE_CACHE: dict[tuple[str, str, int], str] = {}

_SAGE_MAX_HEAD_DIM = 128  # sageattn raises above 128
_SAGE_DTYPE_NAMES = ("float16", "bfloat16", "fp16", "bf16", "half")
# SageAttention 2 lands near cos 0.9999 / rel-L1 0.01-0.03 vs fp32 here; a wrong-arch or broken build is far outside.
_SAGE_MIN_COSINE = 0.99
_SAGE_MAX_REL_L1 = 0.08


def _run_sage_probe(
    device: str,
    dtype: Any,
    head_dim: int = 128,
    sageattn: Any = None,
) -> str:
    """Empty when ``sageattn`` matches fp32 SDPA at ``head_dim``, else why not. Raises when unaskable (import, OOM).

    ``sageattn`` defaults to the pip package's. Random inputs with a per-channel K offset like real keys: a zero tensor
    proves the launch, not the numbers."""
    import torch

    if sageattn is None:
        from sageattention import sageattn

    if dtype not in (torch.float16, torch.bfloat16):
        dtype = torch.float16
    shape = (1, 128, 2, int(head_dim))
    gen = torch.Generator(device = "cpu").manual_seed(0)
    q, k, v = (torch.randn(shape, generator = gen) for _ in range(3))
    k = k + 2.0 * torch.randn((1, 1, 1, shape[-1]), generator = gen)
    q, k, v = (t.to(device = device, dtype = dtype) for t in (q, k, v))
    try:
        out = sageattn(q, k, v, tensor_layout = "NHD")
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    try:
        ref = torch.nn.functional.scaled_dot_product_attention(
            *(t.float().transpose(1, 2) for t in (q, k, v))
        ).transpose(1, 2)
        got = out.float()
        if tuple(got.shape) != tuple(ref.shape):
            return f"self-check: output shape {tuple(got.shape)} != {tuple(ref.shape)}"
        if not bool(torch.isfinite(got).all()):
            return "self-check: non-finite output"
        cos = float(torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim = 0))
        rel = float((got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-12))
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001
        return f"self-check: {type(exc).__name__}: {exc}"
    if cos < _SAGE_MIN_COSINE or rel > _SAGE_MAX_REL_L1:
        return f"self-check: cosine {cos:.4f}, relative L1 {rel:.4f} vs fp32 at head_dim {head_dim}"
    return ""


def _indexed_cuda_device(device: str) -> str:
    """Bare "cuda" -> the card pinned on this thread, so one card's verdict is never reused for another."""
    if device != "cuda":
        return device
    try:
        import torch
        return f"cuda:{torch.cuda.current_device()}"
    except Exception:  # noqa: BLE001
        return device


def _sage_probe_head_dims(head_dims: Any) -> tuple[int, ...]:
    """Head dims Sage serves (<= 128; larger run native per call), 128 when none are known."""
    dims = sorted(
        {
            int(d)
            for d in (head_dims or ())
            if isinstance(d, int) and not isinstance(d, bool) and 0 < d <= _SAGE_MAX_HEAD_DIM
        }
    )
    return tuple(dims) or (_SAGE_MAX_HEAD_DIM,)


def _sage_kernel_runs(
    target: Any,
    logger: Any = None,
    head_dims: Any = None,
) -> Optional[bool]:
    """True when ``sageattn`` passed its self-check at every head dim it serves; False when it raised, failed, or is
    not importable (not cached: a later install may fix it); None (unaskable) keeps the request."""
    device = str(getattr(target, "torch_device", None) or getattr(target, "device", None) or "")
    if not device.startswith("cuda"):
        return None
    device = _indexed_cuda_device(device)
    dtype = getattr(target, "dtype", None)
    error = ""
    for head_dim in _sage_probe_head_dims(head_dims):
        key = (device, str(dtype), head_dim)
        error = _SAGE_PROBE_CACHE.get(key)
        if error is None:
            try:
                error = _run_sage_probe(device, dtype, head_dim)
            except ImportError as exc:
                error = f"sageattention could not be imported ({type(exc).__name__}: {exc})"
                break
            except Exception:  # noqa: BLE001
                return None
            error = _SAGE_PROBE_CACHE.setdefault(key, error)
        if error:
            break
    if error and logger is not None:
        logger.warning(
            "diffusion.attention: SageAttention does not run on this GPU (%s); using the default backend",
            error,
        )
    return not error


def _runs_in_float32(target: Any) -> bool:
    """fp32 pipeline (video families promote fp16 on cards without bf16): Sage never applies."""
    dtype = getattr(target, "dtype", None)
    if dtype is None:
        return False
    name = str(dtype).replace("torch.", "").lower()
    return name in ("float32", "fp32", "float")


# Calls Sage cannot take (attn_mask, head_dim > 128, dtype) run native instead of raising mid-generation. The checks
# read only shapes, dtypes and mask presence, so torch.compile specialises on them without a graph break.
_SAGE_GUARD_ATTR = "_unsloth_sage_guard"
_SAGE_ROUTED: dict[str, int] = {}
_SAGE_ROUTED_LOGGED: set[str] = set()


def _sage_reroute_reason(query: Any, key: Any, value: Any, attn_mask: Any) -> Optional[str]:
    import torch

    if attn_mask is not None:
        return "attn_mask"
    if not all(isinstance(t, torch.Tensor) for t in (query, key, value)):
        return "inputs"
    if (
        query.dtype not in (torch.float16, torch.bfloat16)
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        return "dtype"
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return "rank"
    head_dim = query.shape[-1]
    if head_dim > _SAGE_MAX_HEAD_DIM or key.shape[-1] != head_dim or value.shape[-1] != head_dim:
        return "head_dim"
    if key.shape[2] == 0 or query.shape[2] % key.shape[2] != 0:
        return "heads"
    if query.device.type != "cuda":
        return "device"
    return None


def _note_reroute(label: str, reason: str, counts: dict, logged: set) -> None:
    """Count and log once per reason. Eager only: under torch.compile it would be a traced side effect."""
    try:
        import torch
        if torch.compiler.is_compiling():
            return
    except Exception:  # noqa: BLE001
        pass
    counts[reason] = counts.get(reason, 0) + 1
    if reason not in logged:
        logged.add(reason)
        import logging
        logging.getLogger(__name__).warning(
            "diffusion.attention: %s cannot take this attention call (%s); running it on the default "
            "backend instead",
            label,
            reason,
        )


def _note_sage_reroute(reason: str) -> None:
    _note_reroute("SageAttention", reason, _SAGE_ROUTED, _SAGE_ROUTED_LOGGED)


# sageattn() reads the arch via torch.cuda calls Dynamo cannot trace, so a fullgraph compile failed and the load ran
# eager. An opaque custom op keeps the block compiled.
_SAGE_OP_NAME = "unsloth_studio::sage_attention_nhd"
# Own op for the hub build, so a pip SageAttention 2 installed later never runs through it.
_SAGE_HUB_OP_NAME = "unsloth_studio::sage_hub_attention_nhd"
_SAGE_OPS: dict[str, dict[str, Any]] = {}
_SAGE_OP_LOCK = threading.Lock()


def _sage_custom_op(sage_fn: Any, op_name: str = _SAGE_OP_NAME) -> Optional[Any]:
    """An opaque ``torch.library`` op around a diffusers attention function (NHD), or None. Used for Sage and FA4."""
    with _SAGE_OP_LOCK:
        slot = _SAGE_OPS.setdefault(op_name, {})
        if "op" in slot:
            slot["fn"] = sage_fn
            return slot["op"]
        try:
            import torch

            slot["fn"] = sage_fn

            def _impl(query, key, value, is_causal, scale):
                out = slot["fn"](
                    query = query, key = key, value = value, is_causal = is_causal, scale = scale
                )
                # Head dims under 64 come back as a padded slice; the fake describes a contiguous tensor.
                return out.contiguous()

            # custom_op infers the schema from real types, not `from __future__ import annotations` strings.
            _impl.__annotations__ = {
                "query": torch.Tensor,
                "key": torch.Tensor,
                "value": torch.Tensor,
                "is_causal": bool,
                "scale": Optional[float],
                "return": torch.Tensor,
            }
            _op = torch.library.custom_op(op_name, mutates_args = ())(_impl)

            @_op.register_fake
            def _(query, key, value, is_causal, scale):
                return query.new_empty((*query.shape[:-1], value.shape[-1]))

            slot["op"] = _op
        except Exception:  # noqa: BLE001 - old torch: call the function directly
            slot["op"] = None
        return slot["op"]


def _install_dispatch_guard(
    backend: str,
    reroute_reason: Any,
    note: Any,
    make_op: Any = None,
) -> bool:
    """Wrap diffusers' registered ``backend`` so calls its kernel cannot take run native. Idempotent.

    False when the registry is not where expected: the caller then does not engage the backend. ``make_op(fn)``
    optionally returns an opaque op for plain (no lse, no context-parallel) calls."""
    try:
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _AttentionBackendRegistry,
        )

        backends = _AttentionBackendRegistry._backends
        name = AttentionBackendName(backend)
        kernel_fn = backends[name]
        native_fn = backends[AttentionBackendName.NATIVE]
    except Exception:  # noqa: BLE001
        return False
    if getattr(kernel_fn, _SAGE_GUARD_ATTR, False):
        return True
    if not callable(kernel_fn) or not callable(native_fn):
        return False

    # Eagerly: the guard runs inside compiled graphs, where a registration lock cannot trace.
    op = make_op(kernel_fn) if make_op is not None else None
    names = ("query", "key", "value", "attn_mask")

    def _guarded(*args: Any, **kwargs: Any) -> Any:
        bound = dict(zip(names, args))
        bound.update({n: kwargs[n] for n in names if n in kwargs})
        reason = reroute_reason(
            bound.get("query"), bound.get("key"), bound.get("value"), bound.get("attn_mask")
        )
        if reason is None:
            extra = {k: v for k, v in kwargs.items() if k not in names}
            plain = (
                op is not None
                and len(args) <= 4
                and not extra.get("return_lse", False)
                and extra.get("_parallel_config") is None
                and set(extra) <= {"is_causal", "scale", "return_lse", "_parallel_config"}
            )
            if plain:
                return op(
                    bound["query"],
                    bound["key"],
                    bound["value"],
                    bool(extra.get("is_causal", False)),
                    extra.get("scale"),
                )
            return kernel_fn(*args, **kwargs)
        note(reason)
        return native_fn(*args, **kwargs)

    setattr(_guarded, _SAGE_GUARD_ATTR, True)
    _guarded.__wrapped__ = kernel_fn  # type: ignore[attr-defined]
    try:
        backends[name] = _guarded
    except Exception:  # noqa: BLE001
        return False
    return True


def _install_sage_dispatch_guard() -> bool:
    return _install_dispatch_guard(
        "sage", lambda *a: _sage_reroute_reason(*a), _note_sage_reroute, _sage_custom_op
    )


# diffusers' flash_4_hub raises on any attn_mask too; head dims are checked at load (_fa4_kernel_runs).
_FA4_ROUTED: dict[str, int] = {}
_FA4_ROUTED_LOGGED: set[str] = set()


def _fa4_reroute_reason(query: Any, key: Any, value: Any, attn_mask: Any) -> Optional[str]:
    import torch

    if attn_mask is not None:
        return "attn_mask"
    if not all(isinstance(t, torch.Tensor) for t in (query, key, value)):
        return "inputs"
    if (
        query.dtype not in (torch.float16, torch.bfloat16)
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        return "dtype"
    if query.device.type != "cuda":
        return "device"
    return None


def _note_fa4_reroute(reason: str) -> None:
    _note_reroute("FlashAttention 4", reason, _FA4_ROUTED, _FA4_ROUTED_LOGGED)


# The hub FA4 launch (CuTe DSL, tvm-ffi) breaks a fullgraph compile 5 times per Flux block: keep it opaque, like Sage.
_FA4_OP_NAME = "unsloth_studio::flash_4_hub_attention_nhd"


def _install_fa4_dispatch_guard() -> bool:
    return _install_dispatch_guard(
        "flash_4_hub",
        lambda *a: _fa4_reroute_reason(*a),
        _note_fa4_reroute,
        lambda fn: _sage_custom_op(fn, _FA4_OP_NAME),
    )


# Whether the hub FA4 build runs correctly here, once per (device, dtype, head_dim) through diffusers' dispatch: catches
# cards with no code (sm110 / sm120 unverified) and API drift. An exact kernel, so the bound is tight.
_FA4_PROBE_CACHE: dict[tuple[str, str, int], str] = {}
_FA4_MIN_COSINE = 0.999
_FA4_MAX_REL_L1 = 0.02


def _run_fa4_probe(
    device: str,
    dtype: Any,
    head_dim: int = 128,
) -> str:
    """Empty when flash_4_hub matches fp32 SDPA at ``head_dim``, else why not. Raises on OOM."""
    import torch
    from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn

    if dtype not in (torch.float16, torch.bfloat16):
        dtype = torch.bfloat16
    gen = torch.Generator(device = "cpu").manual_seed(0)
    q, k, v = (torch.randn((1, 256, 2, int(head_dim)), generator = gen) for _ in range(3))
    q, k, v = (t.to(device = device, dtype = dtype) for t in (q, k, v))
    try:
        out = dispatch_attention_fn(q, k, v, backend = AttentionBackendName.FLASH_4_HUB)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        if isinstance(out, tuple):
            out = out[0]
        ref = torch.nn.functional.scaled_dot_product_attention(
            *(t.float().transpose(1, 2) for t in (q, k, v))
        ).transpose(1, 2)
        got = out.float()
        if tuple(got.shape) != tuple(ref.shape):
            return f"self-check: output shape {tuple(got.shape)} != {tuple(ref.shape)}"
        if not bool(torch.isfinite(got).all()):
            return "self-check: non-finite output"
        cos = float(torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim = 0))
        rel = float((got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-12))
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    if cos < _FA4_MIN_COSINE or rel > _FA4_MAX_REL_L1:
        return f"self-check: cosine {cos:.5f}, relative L1 {rel:.4f} vs fp32 at head_dim {head_dim}"
    return ""


def _fa4_kernel_runs(
    target: Any,
    logger: Any = None,
    head_dims: Any = None,
) -> Optional[bool]:
    """True when flash_4_hub passed at every DiT head dim; False with a logged reason; None unaskable."""
    device = str(getattr(target, "torch_device", None) or getattr(target, "device", None) or "")
    if not device.startswith("cuda"):
        return None
    device = _indexed_cuda_device(device)
    dtype = getattr(target, "dtype", None)
    dims = sorted(
        {
            int(d)
            for d in (head_dims or ())
            if isinstance(d, int) and not isinstance(d, bool) and d > 0
        }
    )
    error = ""
    for head_dim in dims or [128]:
        key = (device, str(dtype), head_dim)
        error = _FA4_PROBE_CACHE.get(key)
        if error is None:
            try:
                error = _run_fa4_probe(device, dtype, head_dim)
            except Exception:  # noqa: BLE001
                return None
            error = _FA4_PROBE_CACHE.setdefault(key, error)
        if error:
            break
    if error and logger is not None:
        logger.warning(
            "diffusion.attention: FlashAttention 4 does not run correctly on this GPU (%s); using the default backend",
            error,
        )
    return not error


def _version_tuple(text: str) -> tuple[int, ...]:
    m = re.match(r"\s*(\d+(?:\.\d+)*)", str(text))
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def _sage_version_too_old() -> Optional[str]:
    """Why an installed sageattention is below diffusers' floor (PyPI's 1.0.6 is SageAttention 1), else None. Not
    cached: a later install may replace it."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version("sageattention")
        except PackageNotFoundError:
            return None
        floor = "2.1.1"
        try:
            from diffusers.models.attention_dispatch import _REQUIRED_SAGE_VERSION
            if isinstance(_REQUIRED_SAGE_VERSION, str) and _REQUIRED_SAGE_VERSION.strip():
                floor = _REQUIRED_SAGE_VERSION.strip()
        except Exception:  # noqa: BLE001
            pass
        if _version_tuple(installed) and _version_tuple(installed) < _version_tuple(floor):
            return f"sageattention {installed} is older than {floor}, the SageAttention 2 release diffusers needs"
    except Exception:  # noqa: BLE001
        return None
    return None


def _sage_usable(
    pipe: Any,
    target: Any,
    logger: Any = None,
) -> bool:
    """Whether an explicit ``sage`` request may engage: not fp32, kernel passes at the DiT head dims, guard in place."""
    if target is not None and _runs_in_float32(target):
        if logger is not None:
            logger.warning(
                "diffusion.attention: SageAttention needs fp16 or bf16 and this pipeline runs in float32; "
                "using the default backend"
            )
        return False
    too_old = _sage_version_too_old()
    if too_old:
        if logger is not None:
            logger.warning("diffusion.attention: %s; using the default backend", too_old)
        return False
    head_dims = _dit_head_dims(pipe)
    if head_dims and min(head_dims) > _SAGE_MAX_HEAD_DIM:
        if logger is not None:
            logger.warning(
                "diffusion.attention: SageAttention serves head_dim <= %d and this model uses %s; "
                "using the default backend",
                _SAGE_MAX_HEAD_DIM,
                ",".join(str(d) for d in sorted(head_dims)),
            )
        return False
    if target is not None and _sage_kernel_runs(target, logger, head_dims) is False:
        return False
    if not _install_sage_dispatch_guard():
        if logger is not None:
            logger.warning(
                "diffusion.attention: this diffusers build does not expose the sage backend the way Unsloth expects, "
                "so masked attention calls could not be routed around it; using the default backend"
            )
        return False
    return True


# PyPI has no SageAttention 2 (only 1.0.6, which diffusers refuses), so ``sage`` without a pip SageAttention 2 runs the
# kernels-hub build (sm80 / 89 / 90) via diffusers' ``sage_hub``. diffusers 0.40 asks for version 1 builds, which stop
# at torch 2.10, so version 2 (torch 2.9 to 2.12) is fetched first.
SAGE_HUB_BACKEND = "sage_hub"
_SAGE_HUB_REPO = "kernels-community/sage-attention"
_SAGE_HUB_VERSIONS = (2, 1)
_SAGE_HUB_PROBE_CACHE: dict[tuple[str, str, int], str] = {}
_SAGE_HUB_LOCK = threading.Lock()
# A failed fetch, remembered so the in-lock apply does not repeat it. Cleared by a restart, like _INSTALL_ATTEMPTED.
_SAGE_HUB_FAILED: list[str] = []
SAGE_SOURCE_HINT = (
    "SageAttention 2 is not published on PyPI (pip only has SageAttention 1.0.6), so on this platform it needs a "
    "source build or a community wheel"
)


def _pip_sage2_installed() -> bool:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            installed = version("sageattention")
        except PackageNotFoundError:
            return False
    except Exception:  # noqa: BLE001
        return False
    return bool(_version_tuple(installed)) and _sage_version_too_old() is None


def _load_sage_hub_kernel() -> tuple[Any, str]:
    """``(sageattn, "")`` from the hub build, set in diffusers' ``sage_hub`` slot, or ``(None, why)``. Never raises."""
    with _SAGE_HUB_LOCK:
        try:
            from diffusers.models.attention_dispatch import (
                _HUB_KERNELS_REGISTRY,
                AttentionBackendName,
            )
            config = _HUB_KERNELS_REGISTRY[AttentionBackendName(SAGE_HUB_BACKEND)]
        except Exception as exc:  # noqa: BLE001
            return None, f"this diffusers build has no sage_hub backend ({type(exc).__name__})"
        if callable(getattr(config, "kernel_fn", None)):
            return config.kernel_fn, ""
        if _SAGE_HUB_FAILED:
            return None, _SAGE_HUB_FAILED[0]
        try:
            from kernels import get_kernel
        except Exception as exc:  # noqa: BLE001
            return None, f"the kernels package is not importable ({type(exc).__name__}: {exc})"
        errors = []
        versions = list(_SAGE_HUB_VERSIONS)
        default = getattr(config, "version", None)
        if isinstance(default, int) and default not in versions:
            versions.append(default)
        for ver in versions:
            try:
                fn = getattr(get_kernel(_SAGE_HUB_REPO, version = ver), "sageattn", None)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"version {ver}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            if callable(fn):
                config.kernel_fn = fn
                return fn, ""
            errors.append(f"version {ver}: no sageattn in the build")
        _SAGE_HUB_FAILED.append(f"no {_SAGE_HUB_REPO} build loads here ({'; '.join(errors)})")
        return None, _SAGE_HUB_FAILED[0]


def _sage_hub_kernel_runs(
    target: Any,
    sageattn: Any,
    head_dims: Any = None,
) -> tuple[Optional[bool], str]:
    """``(True, "")`` when the hub ``sageattn`` passed at every head dim, ``(False, why)``, or ``(None, "")`` unaskable."""
    device = str(getattr(target, "torch_device", None) or getattr(target, "device", None) or "")
    if not device.startswith("cuda"):
        return None, ""
    device = _indexed_cuda_device(device)
    dtype = getattr(target, "dtype", None)
    for head_dim in _sage_probe_head_dims(head_dims):
        key = (device, str(dtype), head_dim)
        error = _SAGE_HUB_PROBE_CACHE.get(key)
        if error is None:
            try:
                error = _run_sage_probe(device, dtype, head_dim, sageattn = sageattn)
            except Exception:  # noqa: BLE001
                return None, ""
            error = _SAGE_HUB_PROBE_CACHE.setdefault(key, error)
        if error:
            return False, error
    return True, ""


def _install_sage_hub_dispatch_guard() -> bool:
    return _install_dispatch_guard(
        SAGE_HUB_BACKEND,
        lambda *a: _sage_reroute_reason(*a),
        _note_sage_reroute,
        lambda fn: _sage_custom_op(fn, _SAGE_HUB_OP_NAME),
    )


def _sage_hub_backend(
    pipe: Any,
    target: Any,
    logger: Any = None,
) -> Optional[str]:
    """``sage_hub`` when the hub build passes its self-check here (before any DiT is switched), else None, logged."""

    def _decline(why: str) -> None:
        if logger is not None:
            logger.warning(
                "diffusion.attention: SageAttention unavailable: %s. %s. Using the default backend",
                why,
                SAGE_SOURCE_HINT,
            )

    if target is not None and _runs_in_float32(target):
        if logger is not None:
            logger.warning(
                "diffusion.attention: SageAttention needs fp16 or bf16 and this pipeline runs in float32; "
                "using the default backend"
            )
        return None
    head_dims = _dit_head_dims(pipe)
    if head_dims and min(head_dims) > _SAGE_MAX_HEAD_DIM:
        if logger is not None:
            logger.warning(
                "diffusion.attention: SageAttention serves head_dim <= %d and this model uses %s; "
                "using the default backend",
                _SAGE_MAX_HEAD_DIM,
                ",".join(str(d) for d in sorted(head_dims)),
            )
        return None
    sageattn, why = _load_sage_hub_kernel()
    if sageattn is None:
        _decline(f"the Hugging Face kernels-hub build could not be loaded ({why})")
        return None
    if target is not None:
        runs, why = _sage_hub_kernel_runs(target, sageattn, head_dims)
        if runs is False:
            _decline(
                f"the Hugging Face kernels-hub build does not run correctly on this GPU ({why})"
            )
            return None
    if not _install_sage_hub_dispatch_guard():
        if logger is not None:
            logger.warning(
                "diffusion.attention: this diffusers build does not expose the sage_hub backend the way Unsloth "
                "expects, so masked attention calls could not be routed around it; using the default backend"
            )
        return None
    if logger is not None:
        logger.info(
            "diffusion.attention: SageAttention 2 from the Hugging Face kernels hub (%s) passed its self-check",
            _SAGE_HUB_REPO,
        )
    return SAGE_HUB_BACKEND


# Only cutlass-dsl 4.4 / 4.5 load the hub FA4 build (4.6+ lacks cute.core.ThrMma, 4.3 PipelineClcFetchAsync).
FA4_CUTLASS_DSL_MIN = (4, 4)
FA4_CUTLASS_DSL_MAX_EXCL = (4, 6)
FA4_CUTLASS_DSL_SPEC = "nvidia-cutlass-dsl>=4.4,<4.6"
FA4_TVM_FFI_SPEC = "apache-tvm-ffi>=0.1.6,<0.2,!=0.1.8,!=0.1.8.post0"
FA4_KERNELS_MIN = (0, 12, 3)
FA4_KERNELS_SPEC = "kernels==0.12.3"


def _dist_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:  # noqa: BLE001
        return None


def fa4_cutlass_dsl_ok(installed: Optional[str]) -> bool:
    have = _version_tuple(installed or "")[:2]
    return bool(have) and FA4_CUTLASS_DSL_MIN <= have < FA4_CUTLASS_DSL_MAX_EXCL


def _cutlass_dsl_dependents() -> list[str]:
    found = []
    try:
        from importlib.metadata import distributions
        for dist in distributions():
            for req in dist.requires or ():
                head = (
                    re.split(r"[\s;\[<>=!~]", req.strip(), maxsplit = 1)[0].lower().replace("_", "-")
                )
                if head == "nvidia-cutlass-dsl" and "extra ==" not in req:
                    found.append(
                        f"{dist.metadata['Name']} {dist.version} ({req.split(';')[0].strip()})"
                    )
                    break
    except Exception:  # noqa: BLE001
        pass
    return sorted(set(found))


def _fa4_python_deps_plan() -> tuple[list[str], Optional[str]]:
    """``(requirements, refusal)`` for the hub FA4 build. An out-of-range cutlass-dsl is never replaced (a user's quack
    or a newer FlashInfer may need it): the request is refused instead."""
    reqs: list[str] = []
    kernels = _dist_version("kernels")
    if (
        kernels is not None
        and _version_tuple(kernels)
        and _version_tuple(kernels) < FA4_KERNELS_MIN
    ):
        # Same dependencies as 0.12.1, so --no-deps is safe.
        reqs.append(FA4_KERNELS_SPEC)
    cutlass = _dist_version("nvidia-cutlass-dsl")
    if cutlass is None:
        reqs.append(FA4_CUTLASS_DSL_SPEC)
    elif not fa4_cutlass_dsl_ok(cutlass):
        users = [d for d in _cutlass_dsl_dependents() if not d.lower().startswith("flash-attn")]
        return [], (
            f"FlashAttention 4 from the kernels hub loads only with nvidia-cutlass-dsl 4.4.x or 4.5.x and {cutlass} is "
            "installed"
            + (f" (required by {', '.join(users)})" if users else "")
            + "; Unsloth does not replace it"
        )
    if _dist_version("apache-tvm-ffi") is None:
        reqs.append(FA4_TVM_FFI_SPEC)
    if _dist_version("einops") is None:
        reqs.append("einops")
    return reqs, None


def _pinned_constraints_file() -> Optional[str]:
    """Every installed distribution pinned, so a dependency install cannot move anything."""
    try:
        from .diffusion_nvfp4_install import _write_constraints, installed_distributions
        return _write_constraints(installed_distributions())
    except Exception:  # noqa: BLE001
        return None


def _refresh_diffusers_kernels_version(logger: Any = None) -> None:
    """diffusers reads the kernels version once at import and gates flash_4_hub on it: refresh it after an upgrade, only
    while ``kernels`` is not imported yet (an imported old version runs until a restart)."""
    import sys

    if "kernels" in sys.modules:
        if logger is not None:
            logger.warning(
                "diffusion.attention: kernels was upgraded to %s for FlashAttention 4 and takes effect after Studio "
                "restarts; this load uses the default backend",
                FA4_KERNELS_SPEC.split("==", 1)[1],
            )
        return
    try:
        from diffusers.utils import import_utils
        installed = _dist_version("kernels")
        if installed and getattr(import_utils, "_kernels_version", None) not in (None, installed):
            import_utils._kernels_version = installed
            import_utils._kernels_available = True
            # diffusers memoizes is_kernels_version, so a check made before the upgrade would still answer False.
            cache_clear = getattr(
                getattr(import_utils, "is_kernels_version", None), "cache_clear", None
            )
            if callable(cache_clear):
                cache_clear()
    except Exception:  # noqa: BLE001
        pass


# cutlass-dsl reaches sys.path through a .pth file (libs-base in 4.4 / 4.5, the main dist in 4.6+).
_PTH_DISTRIBUTIONS = ("nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl")


def _activate_installed_pth_files(names: tuple[str, ...] = _PTH_DISTRIBUTIONS) -> None:
    """Run the .pth files of ``names`` now: the interpreter reads them only at startup."""
    try:
        import site
        import sys
        from importlib.metadata import PackageNotFoundError, distribution
    except Exception:  # noqa: BLE001
        return
    for name in names:
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001
            continue
        for entry in dist.files or ():
            if not str(entry).endswith(".pth") or "/" in str(entry).replace("\\", "/"):
                continue
            try:
                path = os.fspath(dist.locate_file(entry))
                site.addpackage(os.path.dirname(path), os.path.basename(path), set(sys.path))
            except Exception:  # noqa: BLE001
                continue


def _ensure_fa4_python_deps(logger: Any = None) -> Optional[str]:
    """Install the hub FA4 build's imports at versions it loads with. Returns the refusal reason, or None."""
    import importlib
    import subprocess
    import sys

    reqs, refusal = _fa4_python_deps_plan()
    if refusal is not None:
        if logger is not None:
            logger.warning("diffusion.attention: %s; using the default backend", refusal)
        return refusal
    if not reqs:
        # An NVFP4 FlashInfer install earlier in this process may have added cutlass-dsl without running its .pth.
        _activate_installed_pth_files()
        return None
    key = "fa4-deps:" + ",".join(reqs)
    if key in _INSTALL_ATTEMPTED:
        return None
    _INSTALL_ATTEMPTED.add(key)
    kernels_upgrade = [r for r in reqs if r == FA4_KERNELS_SPEC]
    deps = [r for r in reqs if r != FA4_KERNELS_SPEC]
    if logger is not None:
        logger.info(
            "diffusion.attention: installing %s for backend=flash_4_hub (wheel-only)",
            ", ".join(reqs),
        )
    base = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--only-binary",
        ":all:",
    ]
    constraints = _pinned_constraints_file() if deps else None
    try:
        if kernels_upgrade:
            subprocess.run(
                base + ["--no-deps", *kernels_upgrade], capture_output = True, timeout = 600, check = True
            )
            _refresh_diffusers_kernels_version(logger)
        if deps:
            # Resolved (cutlass-dsl needs its libs and cuda-python); the constraints add only new packages.
            cmd = base + (["-c", constraints] if constraints else []) + deps
            subprocess.run(cmd, capture_output = True, timeout = 600, check = True)
        importlib.invalidate_caches()
        _activate_installed_pth_files()
    except Exception as exc:  # noqa: BLE001 - FA4 then falls back at set time
        if logger is not None:
            stderr = getattr(exc, "stderr", None)
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors = "replace")
            logger.warning(
                "diffusion.attention: could not install %s for FlashAttention 4: %s",
                ", ".join(reqs),
                _redacted_for_log((stderr or "").strip()[-1500:]) or _redacted_for_log(str(exc)),
            )
    finally:
        if constraints:
            try:
                os.unlink(constraints)
            except OSError:
                pass
    return None


# Optional kernels installable on demand: dispatcher name -> (probe module, pip package). Wheels only
# (--only-binary=:all:), since a source build needs a CUDA toolchain the host may lack.
_INSTALLABLE_BACKENDS: dict[str, tuple[str, str]] = {
    # Never ``sageattention`` from PyPI (only SageAttention 1 there): the hub build is the installable one.
    "sage": ("kernels", "kernels"),
    "flash": ("flash_attn", "flash-attn"),
    "_flash_3_hub": ("kernels", "kernels"),  # FA3/FA4 from the HF kernels hub
    "flash_4_hub": ("kernels", "kernels"),
    # Never handed to pip as a name -- see _MATCHED_WHEEL_BACKENDS below; the package string survives only for logging
    # and for the _INSTALL_ATTEMPTED bookkeeping.
    "xformers": ("xformers", "xformers"),
}

# On-demand install gate (mirrors UNSLOTH_DIFFUSION_SD_CPP_INSTALL): auto (default) / 1 installs a missing package
# when a gated backend is requested; 0 never installs and falls back to native.
_ATTENTION_INSTALL_ENV = "UNSLOTH_DIFFUSION_ATTENTION_INSTALL"

# Packages a pip install was already attempted for in THIS process. The loader pre-installs outside its locks, so a
# recorded attempt stops apply re-running the 600s install under _generate_lock.
_INSTALL_ATTEMPTED: set[str] = set()

# Backends whose wheel must be resolved against the RUNNING torch build instead of handed to pip as a name. Only
# DETERMINISTIC answers are memoised (a URL, or a refusal that depends purely on the resident torch, which cannot change
# under a running interpreter). A probe timeout is transient and is NOT cached: caching it would turn one loaded-machine
# hiccup into "no xFormers for the rest of this Unsloth session".
_MATCHED_WHEEL_BACKENDS = frozenset({"xformers"})
_XFORMERS_WHEEL_TARGET: Optional[tuple[Optional[str], Optional[str]]] = None
_XFORMERS_WHEEL_LOCK = threading.Lock()


# Scheme-qualified URLs only; nothing else in these messages can carry a credential.
_URL_IN_TEXT = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s'\"<>]+")


def _redacted_for_log(text: str) -> str:
    """Every URL in ``text``, stripped of userinfo / query / fragment.

    Takes free text, not just a URL, because pip echoes the URL it was handed back in its
    stderr -- redacting only the name in the log line would leave the secret in the body.
    """
    try:
        from utils.wheel_utils import redact_url_credentials
    except Exception:  # noqa: BLE001 -- redaction must never be the thing that breaks a log
        return text
    return _URL_IN_TEXT.sub(lambda m: redact_url_credentials(m.group(0)), text)


def _xformers_wheel_target() -> tuple[Optional[str], Optional[str]]:
    """Resolve the xFormers wheel built for the resident torch: (URL, refusal reason).

    xformers' compiled extension is linked against ONE exact (torch, CUDA) pair, and next to any
    other pair ``torch.ops.load_library`` raises -- which xformers/_cpp_lib.py then downgrades to a
    log warning, so the import "succeeds" with memory-efficient attention, SwiGLU and the sparse ops
    silently gone. That is invisible to ``find_spec`` and to pip, and PyPI publishes only the
    CUDA-12.8 flavour, so a plain ``pip install xformers`` beside a cu130 torch installs the broken
    combination every time.

    So resolve the exact download.pytorch.org wheel instead, and when no wheel matches return a
    reason rather than a URL: installing nothing leaves the caller on torch SDPA, which is strictly
    better than an extension that cannot load.

    The URL is not HEAD-checked here. This can run under ``_generate_lock`` (the video loader has no
    out-of-lock pre-install hop), so it must not add network round trips to a path that already
    blocks unload/cancel; a wrong row surfaces as a pip failure instead, and the matrix has a
    live-URL test behind it.
    """
    global _XFORMERS_WHEEL_TARGET
    with _XFORMERS_WHEEL_LOCK:
        if _XFORMERS_WHEEL_TARGET is not None:
            return _XFORMERS_WHEEL_TARGET
        try:
            from utils.wheel_utils import probe_torch_wheel_env, xformers_wheel_url

            # include_windows: this is the one resolver that HAS win_amd64 wheels upstream. timeout matches the other
            # probe_torch_wheel_env callers.
            env = probe_torch_wheel_env(timeout = 30, include_windows = True)
        except Exception as exc:  # noqa: BLE001 -- must never break a model load
            return (None, f"the xFormers wheel could not be resolved ({exc})")
        if env is None:
            # Ambiguous: a platform wheel_platform_tag() does not name (macOS, Windows on ARM) which is deterministic,
            # or a probe that timed out on a busy box which is transient. Not cached, so the next request can settle
            # it. Linux aarch64 is NOT here: it gets a platform_tag and so lands on the branch below.
            return (
                None,
                "torch could not be probed, or this platform has no xFormers wheel "
                "(macOS / Windows on ARM)",
            )
        url = xformers_wheel_url(env)
        if url is None:
            # Name the platform. Linux aarch64 reaches here with a perfectly ordinary torch, and reporting only the
            # torch and CUDA would read as "upstream never built this pair" when the truth is "upstream never built it
            # for this arch".
            target = (
                None,
                f"no xFormers wheel is published for torch "
                f"{env.get('torch_version') or 'unknown'} with CUDA "
                f"{env.get('cuda_version') or 'none'} on "
                f"{env.get('platform_tag') or 'this platform'}",
            )
        else:
            target = (url, None)
        _XFORMERS_WHEEL_TARGET = target
        return target


# The huggingface_hub floor the current `kernels` wheels declare (kernels >= 0.14.1 requires huggingface-hub >= 1.10.0).
# A (major, minor) pair, compared against the resident hub below.
_KERNELS_HUB_FLOOR = (1, 10)


def _kernels_hub_compatible() -> bool:
    """Whether installing the ``kernels`` package is SAFE next to the resident huggingface_hub.

    Current ``kernels`` wheels declare ``huggingface_hub >= 1.10`` and build their dependency tables
    against that API, and with an older hub the breakage is NOT contained to the requested backend:
    ``import kernels`` raises at module scope, and diffusers imports ``kernels`` whenever it is
    installed, so EVERY later pipeline import in every process fails until the package is
    uninstalled. Measured with kernels 0.16.0: hub 1.0.0-1.2.4 raise
    ``StrictDataclassFieldValidationError`` on ``import kernels``, and 1.3-1.9 merely happen to work
    today, below the floor kernels supports. The whole 1.x range under 1.10 is therefore refused
    rather than trusted, since the install is unpinned. An undeterminable hub version allows the
    install, keeping the previous behaviour.

    A ``--no-deps`` install cannot self-correct here: pip writes the wheel without ever reading its
    ``Requires-Dist``, so this predicate is the only thing enforcing that floor.
    """
    try:
        import re
        from importlib.metadata import version

        m = re.match(r"\s*(\d+)(?:\.(\d+))?", version("huggingface_hub"))
        if m is None:
            return True
        return (int(m.group(1)), int(m.group(2) or 0)) >= _KERNELS_HUB_FLOOR
    except Exception:  # noqa: BLE001 - unknown hub -> keep the previous permissive behaviour
        return True


def _ensure_attention_backend_installed(backend: str, logger: Any = None) -> Optional[str]:
    """Best-effort install of ``backend``'s package, plus the hub FA4 build's dependencies for ``flash_4_hub``. Returns
    the refusal reason, or None; see ``_ensure_backend_package``."""
    if backend == "sage" and _pip_sage2_installed():
        return None
    reason = _ensure_backend_package(backend, logger)
    if reason is not None or backend not in ("flash_4_hub", "sage"):
        return reason
    if os.environ.get(_ATTENTION_INSTALL_ENV, "auto").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return None
    try:
        import importlib.util
        if importlib.util.find_spec("kernels") is None:
            return None
    except Exception:  # noqa: BLE001
        return None
    if backend == "sage":
        # Fetched here, outside the loader's locks; apply finds it in diffusers' registry.
        _load_sage_hub_kernel()
        return None
    return _ensure_fa4_python_deps(logger)


def _ensure_backend_package(backend: str, logger: Any = None) -> Optional[str]:
    """Best-effort wheel-only install of the package ``backend`` needs, when allowed.

    Called after arch gating, so only for a backend that could work here. Failure is swallowed: the
    subsequent set_attention_backend raises on the missing package and falls back to native.

    Returns the reason the install was REFUSED (a policy decision, e.g. no CUDA-matched xFormers
    wheel exists for the resident torch), or None when nothing stood in the way -- the install ran,
    was skipped as already present, or merely failed. Every refusal is also logged at warning level;
    the return value is there so a caller that wants to surface the reason can, and both current
    callers deliberately ignore it.
    """
    import importlib.util
    import os
    import sys

    spec = _INSTALLABLE_BACKENDS.get(backend)
    if spec is None:
        return None
    module, package = spec
    gate = os.environ.get(_ATTENTION_INSTALL_ENV, "auto").strip().lower()
    if gate in ("0", "false", "no", "off"):
        return None
    # A hidden package stays unusable until restart; skip without recording an attempt.
    if module in sys.modules and sys.modules[module] is None:
        reason = f"{module} is disabled in this process because it requires a different torch"
        if logger is not None:
            logger.warning(
                "diffusion.attention: not installing %s for backend=%s: %s; restart Studio "
                "once the installer has removed it. Using the default backend",
                package,
                backend,
                reason,
            )
        return reason
    # Refusing is a POLICY decision, not a failed attempt, so it is checked before the _INSTALL_ATTEMPTED memo below and
    # records nothing: a later request on a fixed environment must still be able to install. Scoped to kernels (which
    # sage and flash3 / flash4 install); the flash-attn / xformers wheels do not import huggingface_hub at module scope.
    if package == "kernels" and not _kernels_hub_compatible():
        if logger is not None:
            logger.warning(
                "diffusion.attention: not installing 'kernels' for backend=%s — the resident "
                "huggingface_hub is below %d.%d and a kernels install would break every later "
                "diffusers pipeline import; using the default backend",
                backend,
                *_KERNELS_HUB_FLOOR,
            )
        return "the resident huggingface_hub is too old for the kernels package"
    try:
        if importlib.util.find_spec(module) is not None:
            # Present is present, including a MISMATCHED xformers: find_spec sees the package, so nothing below runs and
            # the wrong-CUDA build stays. That is deliberate here. Repairing means reinstalling a package the user may
            # have built or pinned, and this can run under _generate_lock, so a 100 MB download would block unload and
            # cancel. install.ps1 is where the repair belongs -- it compares cpp_lib.json against the resident torch and
            # passes --reinstall-package, outside any request. What this branch prevents is Unsloth CREATING the
            # mismatch, which is how it got made in the first place.
            return None
    except Exception:  # noqa: BLE001 - a broken install probes as missing; try the install
        pass
    # PyPI flash-attn is the CUDA build: never pip it onto a ROCm torch. Policy, so no _INSTALL_ATTEMPTED record.
    if backend == "flash" and _torch_is_rocm():
        reason = "flash-attn on PyPI is the NVIDIA CUDA build"
        if logger is not None:
            logger.warning(
                "diffusion.attention: not installing %s for backend=%s on ROCm: %s; %s. Using the default backend",
                package,
                backend,
                reason,
                _ROCM_FLASH_HINT,
            )
        return reason
    # XFormers ships a compiled extension tied to one exact (torch, CUDA) pair, so the name `xformers` is not a safe
    # thing to hand pip: PyPI serves only the CUDA-12.8 build and --no-deps below stops pip from ever reading its
    # `Requires-Dist: torch==X`. Resolve the matching wheel URL instead, and REFUSE when there is none -- like the
    # kernels gate above this is policy, so it is checked before the _INSTALL_ATTEMPTED memo and records nothing there
    # (a refused backend never burns its one install attempt).
    if backend in _MATCHED_WHEEL_BACKENDS:
        wheel_url, refusal = _xformers_wheel_target()
        if wheel_url is None:
            if logger is not None:
                logger.warning(
                    "diffusion.attention: not installing %s for backend=%s — %s; an unpinned "
                    "install would land an extension that cannot load next to the resident "
                    "torch and would disable memory-efficient attention silently. Using the "
                    "default backend",
                    package,
                    backend,
                    refusal,
                )
            return refusal
        package = wheel_url
    # What pip gets and what the log gets are not the same string. UNSLOTH_PYTORCH_MIRROR may carry userinfo or a
    # token for a private index, and it is baked into the wheel URL, so logging the URL verbatim writes that secret
    # into the backend log.
    display = _redacted_for_log(package)
    # attempt each install once per process, else the in-lock apply re-runs it under _generate_lock and blocks
    # unload/cancel
    if package in _INSTALL_ATTEMPTED:
        return None
    _INSTALL_ATTEMPTED.add(package)
    import subprocess

    if logger is not None:
        logger.info(
            "diffusion.attention: installing %s for backend=%s (wheel-only)", display, backend
        )
    try:
        subprocess.run(
            # --no-deps: install ONLY this kernel wheel, since xformers/flash-attn pin an exact torch and normal
            # resolution would replace the running one. It also means pip never reads the wheel's `Requires-Dist:
            # torch==X`, so nothing here would catch an ABI mismatch -- for xformers, whose mismatch is SILENT, the
            # URL was resolved against the running torch above precisely so there is nothing left to catch.
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
        # The import system caches directory listings, so invalidate the finder caches or the next find_spec can miss
        # the new wheel.
        importlib.invalidate_caches()
        if package == "kernels":
            _refresh_diffusers_kernels_version(logger)
    except Exception as exc:  # noqa: BLE001 - no wheel / no network -> native fallback
        if logger is not None:
            # CalledProcessError.str() shows only the exit code; surface stderr so the fallback is diagnosable.
            stderr = getattr(exc, "stderr", None)
            if stderr:
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors = "replace")
                logger.warning(
                    "diffusion.attention: could not install %s; pip failed with: %s",
                    display,
                    # pip echoes the URL it was given, so redact the body too rather than only the name in front of it.
                    _redacted_for_log(stderr.strip()) or str(exc),
                )
            else:
                logger.warning(
                    "diffusion.attention: could not install %s (%s); falling back to default",
                    display,
                    _redacted_for_log(str(exc)),
                )
    return None


def _torch_is_rocm() -> bool:
    try:
        import torch

        from core._torchao_stub import _module_is_rocm
        return _module_is_rocm(torch)
    except Exception:  # noqa: BLE001
        return False


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


# head_dim 256 (Ideogram 4) has no cuDNN kernel on torch 2.11-2.13 (cuDNN 9.19/9.20); a pinned cuDNN then has no fallback.
_CUDNN_HEAD_DIM_CACHE: dict[tuple[str, str, int], bool] = {}


def _dit_head_dims(pipe: Any) -> set[int]:
    """The attention head dims of every denoiser DiT: ``config.attention_head_dim``, else any module's ``head_dim``."""
    dims: set[int] = set()
    for dit in _attention_dits(pipe):
        config = getattr(dit, "config", None)
        value = None
        try:
            value = config.get("attention_head_dim") if hasattr(config, "get") else None
        except Exception:  # noqa: BLE001
            value = None
        if isinstance(value, int) and not isinstance(value, bool):
            dims.add(value)
            continue
        if isinstance(value, (list, tuple)) and value and all(isinstance(v, int) for v in value):
            dims.update(value)
            continue
        modules = getattr(dit, "modules", None)
        if not callable(modules):
            continue
        try:
            for module in modules():
                hd = getattr(module, "head_dim", None)
                if isinstance(hd, int) and not isinstance(hd, bool) and hd > 0:
                    dims.add(hd)
        except Exception:  # noqa: BLE001
            continue
    return dims


CUDNN_HEAD_DIM_PROBE_ENV = "UNSLOTH_DIFFUSION_CUDNN_HEAD_DIM_PROBE"


def _run_cudnn_head_dim_probe(device: str, dtype: Any, head_dim: int) -> bool:
    """True when cuDNN attention serves ``head_dim`` per SDPA's own gate (launches no kernel); raises when unaskable."""
    import torch

    if dtype not in (torch.float16, torch.bfloat16):
        dtype = torch.bfloat16
    q = torch.empty((1, 2, 8, int(head_dim)), device = device, dtype = dtype)
    try:
        from torch.backends.cuda import SDPAParams, can_use_cudnn_attention
    except ImportError:
        SDPAParams = can_use_cudnn_attention = None
    if SDPAParams is not None and can_use_cudnn_attention is not None:
        try:
            params = SDPAParams(q, q, q, None, 0.0, False, False)
        except TypeError:  # torch < 2.5 has no enable_gqa argument
            params = SDPAParams(q, q, q, None, 0.0, False)
        return bool(can_use_cudnn_attention(params, False))
    from torch.nn.attention import SDPBackend, sdpa_kernel

    try:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - "No available kernel" is the answer
        return False
    return True


def _cudnn_runs_head_dim(target: Any, head_dim: int) -> Optional[bool]:
    """Whether cuDNN attention runs at ``head_dim`` on ``target``; None when unaskable (not cached)."""
    device = str(getattr(target, "torch_device", None) or getattr(target, "device", None) or "")
    if not device.startswith("cuda"):
        return None
    device = _indexed_cuda_device(device)
    dtype = getattr(target, "dtype", None)
    key = (device, str(dtype), int(head_dim))
    cached = _CUDNN_HEAD_DIM_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        ran = _run_cudnn_head_dim_probe(device, dtype, head_dim)
    except Exception:  # noqa: BLE001 - no torch, no device, allocator trouble: no answer about the kernel
        return None
    return _CUDNN_HEAD_DIM_CACHE.setdefault(key, bool(ran))


def _cudnn_serves_pipe(
    pipe: Any,
    target: Any,
    logger: Any = None,
) -> bool:
    """False only when cuDNN demonstrably lacks a DiT head_dim. ``UNSLOTH_DIFFUSION_CUDNN_HEAD_DIM_PROBE=0`` skips it."""
    if (os.environ.get(CUDNN_HEAD_DIM_PROBE_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return True
    missing = sorted(d for d in _dit_head_dims(pipe) if _cudnn_runs_head_dim(target, d) is False)
    if missing and logger is not None:
        logger.warning(
            "diffusion.attention: cuDNN attention has no kernel for head_dim %s with this torch/cuDNN build; "
            "keeping the default SDPA dispatch",
            ",".join(str(d) for d in missing),
        )
    return not missing


def _configure_native_attention(pipe: Any, target: Any, logger: Any) -> None:
    rocm = _is_cuda_rocm(target)
    if rocm:
        # An earlier incomplete probe left the flags alone; apply a retry's answer before dispatching.
        try:
            guard_rocm_fused_sdpa(target, logger)
        except Exception as exc:
            _warn(logger, "ROCm fused SDPA check", exc)
    if rocm and sdpa_math_only(target):
        try:
            from .diffusion_qwenimage21_math import install
            if install(pipe, target, logger):
                if logger is not None:
                    logger.warning(
                        "diffusion.attention: Qwen-Image-2.1 is using bounded math attention; "
                        "check the AMD device packages if fused kernels are expected on this GPU"
                    )
                return
        except Exception as exc:
            _warn(logger, "bounded Qwen-Image-2.1 math attention", exc)
    warn_if_sdpa_math_only(target, logger)


def apply_attention_backend(
    pipe: Any,
    backend: Optional[str],
    *,
    logger: Any = None,
    target: Any = None,
) -> Optional[str]:
    """Set ``backend`` on EVERY denoiser DiT via the diffusers dispatcher.

    Returns the backend engaged, or None when left at native (``backend`` was None or the kernel
    was unavailable -> graceful fallback, never a load failure).

    ``target`` is the resolved device target. Given, and only when the result is native, the SDPA
    backends are probed and a math-only device is reported here rather than discovered as a
    six-figure-MiB allocation mid-generation (#8225). Optional so existing callers are unaffected.

    diffusers keeps a process-wide active backend that ``set_attention_backend`` also updates, and
    a fresh transformer's processors follow it (default None). So a load wanting native must
    restore it explicitly, else it inherits a backend an earlier load pinned (e.g. cuDNN under a
    speed profile), breaking the ``off`` guarantee. Best-effort."""
    if target is not None:
        try:
            guard_rocm_fused_sdpa(target, logger)
        except Exception as exc:
            _warn(logger, "ROCm fused SDPA check", exc)
    setters = [
        s
        for s in (getattr(t, "set_attention_backend", None) for t in _attention_dits(pipe))
        if callable(s)
    ]
    if not setters:
        # A U-Net pipeline (SDXL) exposes no dispatcher setter, so there is no backend to set -- but its attention still
        # runs through torch SDPA, so the math-only diagnosis applies exactly as it does to a DiT. Report it here or an
        # SDXL load gets no warning at all.
        if target is not None:
            _configure_native_attention(pipe, target, logger)
        return None
    if backend is not None:
        _ensure_attention_backend_installed(backend, logger)
        if backend == "sage":
            if _pip_sage2_installed():
                if not _sage_usable(pipe, target, logger):
                    backend = None
            else:
                too_old = _sage_version_too_old()
                if too_old and logger is not None:
                    logger.info("diffusion.attention: ignoring %s", too_old)
                backend = _sage_hub_backend(pipe, target, logger)
        # Re-verify at the dtype the pipeline runs in: selection may have seen the pre-promotion (fp16) dtype.
        if (
            backend == "flash"
            and target is not None
            and _is_cuda_rocm(target)
            and not _rocm_flash_attn_runs(target)
        ):
            backend = None
        if (
            backend == "_native_cudnn"
            and target is not None
            and not _cudnn_serves_pipe(pipe, target, logger)
        ):
            backend = None
        if backend == "flash_4_hub" and not _install_fa4_dispatch_guard():
            if logger is not None:
                logger.warning(
                    "diffusion.attention: this diffusers build does not expose the flash_4_hub backend the way "
                    "Unsloth expects, so masked attention calls could not be routed around it; using the default "
                    "backend"
                )
            backend = None
    if backend is not None:
        engaged = False
        for fn in setters:
            try:
                fn(backend)
                engaged = True
            except Exception as exc:  # noqa: BLE001 - unavailable kernel -> restore native below
                _warn(logger, backend, exc)
        if (
            engaged
            and backend == "flash_4_hub"
            and target is not None
            and _fa4_kernel_runs(target, logger, _dit_head_dims(pipe)) is False
        ):
            # set_attention_backend fetches the kernel, so probe after it and undo the pin on failure.
            for fn in setters:
                try:
                    fn(ATTN_NATIVE)
                except Exception as exc:  # noqa: BLE001
                    _warn(logger, ATTN_NATIVE, exc)
            engaged = False
        if engaged:
            _tag_dits(pipe, backend)
            # set_attention_backend also pins the backend process-wide. Each DiT's processors keep it locally, so reset
            # the global to native ONCE, else a later component inherits this kernel.
            _reset_global_backend_to_native(logger)
            if logger is not None:
                logger.info("diffusion.attention: backend=%s", backend)
            return backend
    # No backend requested, or every set failed: pin native so a stale process-wide backend cannot leak in. One reset
    # covers every fresh DiT.
    _tag_dits(pipe, None)
    _restore_native_backend(setters[0], logger)
    # Native means torch's SDPA dispatch decides per call, and on a device with no fused kernel that decision is MATH.
    # Say so now; the flags this would otherwise be read off lie (#8225).
    if target is not None:
        _configure_native_attention(pipe, target, logger)
    return None


# Read by graph_eligible: Sage under a replayed graph renders noise.
ATTENTION_BACKEND_ATTR = "_unsloth_attention_backend"


def _tag_dits(pipe: Any, backend: Optional[str]) -> None:
    for dit in _attention_dits(pipe):
        try:
            setattr(dit, ATTENTION_BACKEND_ATTR, backend)
        except Exception:  # noqa: BLE001
            pass


def _active_attention_backend() -> Optional[str]:
    """The diffusers process-wide active attention backend name, or None if undeterminable."""
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry

        # get_active_backend() returns (AttentionBackendName, fn) or None; read element 0's .value
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
    except Exception:  # noqa: BLE001 - best-effort; leave the global as-is on any change
        pass


def _restore_native_backend(set_backend_fn: Any, logger: Any) -> None:
    """Force the native default when the global active backend isn't already native."""
    if _active_attention_backend() == ATTN_NATIVE:
        return  # already native -> avoid redundant work and an extra dispatcher warning
    try:
        set_backend_fn(ATTN_NATIVE)
    except Exception as exc:  # noqa: BLE001 - best-effort restore
        _warn(logger, ATTN_NATIVE, exc)


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.attention: %s unavailable (%s); using default", what, exc)


# HunyuanVideo-1.5 joint-attention padding trim (accuracy-exact speed win). HunyuanVideo15AttnProcessor2_0 runs a
# JOINT [video ; text] self-attention and, on every block and step, materialises a dense [B,1,N,N] boolean mask so the
# video never attends to padded text. A dense bool attn_mask costs most of what the fused kernels are for: at the
# production shape (N~=50k, 121 frames 480p, B200) the SAME attention is 296 ms with the dense mask against 15 ms with
# attn_mask=None, and end to end a 121-frame 832x480 10-step render goes 353.8s -> 33.9s. The text is ~99.5% padding
# (a t2v prompt fills ~9 of ~1985 slots). The fix is exact: the model already masks the padded text and DISCARDS its
# attention output (only the video split feeds proj_out), so removing the padded tokens before attention changes
# nothing for the video. "Exact" means no information is discarded, NOT bit-reproducible: swapping masked for fused
# SDPA perturbs each step at bf16 rounding scale and 10 denoising steps amplify that, so the finished video is a
# different sample. That is intrinsic to the kernel change, not to the trim, since rendering the SAME dense-mask path
# under two different exact SDPA kernels diverges more. Whole-video LPIPS cannot judge a kernel change at this step
# count; the single-forward relative error can. scripts/sdpa_mask_backend_probe.py re-measures all of it. Done in an
# eager forward pre-hook (outside the compiled blocks): drop the all-zero image stream (t2v), trim the mllm/byt5
# streams to their globally-valid columns, and, when nothing partially-padded remains, flag the DiT so the processor
# skips the dense mask and runs the fused path. Mixed-padding batches fall back to the stock dense mask. SHAPE NOTE:
# trimmed length varies per prompt; safe because ``max`` compiles with dynamic=None (one generalising recompile). A
# static (dynamic=False) compile would recompile per prompt length and hit dynamo's recompile limit under fullgraph.
_HUNYUAN15_TRANSFORMER_CLS = "HunyuanVideo15Transformer3DModel"
_HUNYUAN15_PROCESSOR_CLS = "HunyuanVideo15AttnProcessor2_0"
_NULL_ATTN_FLAG = "_unsloth_null_attn_mask"

_NULL_PROCESSOR_CACHE: dict = {}


def _set_hunyuan_null_mask(module: Any, enabled: bool) -> None:
    """Set the null-mask flag on every block's attention of ``module``. The flag is valid ONLY for
    the forward whose pre-hook removed the padding, so a post-hook clears it back to False after
    each call (see the module note and _hunyuan_trim_post_hook)."""
    for blk in getattr(module, "transformer_blocks", []):
        attn = getattr(blk, "attn", None)
        if attn is not None:
            setattr(attn, _NULL_ATTN_FLAG, enabled)


def _hunyuan_null_mask_state(module: Any) -> bool:
    """The null-mask flag this call's pre-hook set (read by the graph layer as part of its key)."""
    for blk in getattr(module, "transformer_blocks", []):
        attn = getattr(blk, "attn", None)
        if attn is not None:
            return bool(getattr(attn, _NULL_ATTN_FLAG, False))
    return False


def _null_mask_processor_cls():
    """Build (once, lazily) a HunyuanVideo15AttnProcessor2_0 subclass whose ``__call__`` runs
    attn_mask=None when the DiT is flagged (padding already removed by the pre-hook); otherwise it
    delegates to the stock processor, so a mixed-padding batch and future diffusers changes stay
    correct."""
    cached = _NULL_PROCESSOR_CACHE.get("cls")
    if cached is not None:
        return cached

    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15AttnProcessor2_0,
    )

    class _HunyuanNullMaskProcessor(HunyuanVideo15AttnProcessor2_0):
        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states = None,
            attention_mask = None,
            image_rotary_emb = None,
        ):
            # Fast path only when the pre-hook removed all padding (attn_mask redundant); a constant python bool so
            # torch.compile const-folds the branch (no graph break).
            if not getattr(attn, _NULL_ATTN_FLAG, False):
                return super().__call__(
                    attn,
                    hidden_states,
                    encoder_hidden_states = encoder_hidden_states,
                    attention_mask = attention_mask,
                    image_rotary_emb = image_rotary_emb,
                )

            # Null path = the stock body with the mask block removed and attn_mask=None.
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim = 1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim = 1)

            if encoder_hidden_states is not None:
                encoder_query = attn.add_q_proj(encoder_hidden_states)
                encoder_key = attn.add_k_proj(encoder_hidden_states)
                encoder_value = attn.add_v_proj(encoder_hidden_states)

                encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
                encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
                encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

                if attn.norm_added_q is not None:
                    encoder_query = attn.norm_added_q(encoder_query)
                if attn.norm_added_k is not None:
                    encoder_key = attn.norm_added_k(encoder_key)

                query = torch.cat([query, encoder_query], dim = 1)
                key = torch.cat([key, encoder_key], dim = 1)
                value = torch.cat([value, encoder_value], dim = 1)

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask = None,
                dropout_p = 0.0,
                is_causal = False,
                backend = self._attention_backend,
                parallel_config = self._parallel_config,
            )

            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                enc_len = encoder_hidden_states.shape[1]
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, :-enc_len],
                    hidden_states[:, -enc_len:],
                )
                if getattr(attn, "to_out", None) is not None:
                    hidden_states = attn.to_out[0](hidden_states)
                    hidden_states = attn.to_out[1](hidden_states)
                if getattr(attn, "to_add_out", None) is not None:
                    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # Always the 2-tuple, matching the stock processor's return contract (it returns (hidden_states,
            # encoder_hidden_states) outside its own `if`), so the calling block unpacks identically on either path.
            return hidden_states, encoder_hidden_states

    _NULL_PROCESSOR_CACHE["cls"] = _HunyuanNullMaskProcessor
    return _HunyuanNullMaskProcessor


def _trim_stream(states, mask):
    """Drop the columns of a [B, S, D] text stream + its [B, S] mask that are padding for EVERY
    batch element (globally invalid). Returns (states, mask, all_valid): all_valid is True when
    the trimmed stream has NO partially-padded column left (so it needs no attention mask)."""
    if states is None or mask is None or mask.dim() != 2:
        return states, mask, True  # nothing to mask -> treat as no-padding
    mb = mask.bool()
    keep = mb.any(dim = 0)  # column valid for at least one batch element
    if not bool(keep.all()):
        states = states[:, keep]
        mask = mask[:, keep]
        mb = mb[:, keep]
    # All remaining slots valid for every element (vacuously True for a 0-length stream, fine for an unused secondary
    # stream e.g. byt5 in t2v).
    all_valid = bool(mb.all().item())
    return states, mask, all_valid


_TRIM_MEMO_ATTR = "_unsloth_trim_memo"


def _trim_plan(module: Any, kwargs: dict) -> dict:
    """The trim decisions for this call's image / mask tensors: the host reads (``_trim_stream``'s) made once per
    set of inputs. A pipeline hands the SAME prompt tensors to every step, so a step after the first reuses the plan
    and makes no host wait (a CUDA-graph replay of the step then runs without one). Keyed on the tensors themselves,
    held here, and their version counters, so new or edited inputs plan afresh."""
    import torch

    names = ("image_embeds", "encoder_attention_mask", "encoder_attention_mask_2")
    srcs = tuple(kwargs.get(n) for n in names)
    # An inference tensor (renders run under torch.inference_mode) has no version counter and reading one raises;
    # it is keyed on identity alone. It cannot be written outside inference mode, and the pipeline hands every step
    # the encoder's own outputs, which nothing edits in place.
    versions = tuple(
        ("inference" if t.is_inference() else t._version) if torch.is_tensor(t) else None
        for t in srcs
    )
    memo = module.__dict__.setdefault(_TRIM_MEMO_ATTR, [])
    for held, held_versions, plan in memo:
        if held_versions == versions and all(a is b for a, b in zip(held, srcs)):
            return plan
    image = srcs[0]
    plan = {
        "t2v": bool(image is not None and image.numel() > 0 and bool(torch.all(image == 0).item()))
    }
    for name, mask in zip(names[1:], srcs[1:]):
        if mask is None or not torch.is_tensor(mask) or mask.dim() != 2:
            plan[name] = (None, True)
            continue
        mb = mask.bool()
        keep = mb.any(dim = 0)  # column valid for at least one batch element
        index = None
        if not bool(keep.all()):
            index = keep.nonzero().squeeze(1)
            mb = mb.index_select(1, index)
        # vacuously True for a 0-length stream, fine for an unused secondary stream (byt5 in t2v)
        plan[name] = (index, bool(mb.all().item()))
    memo.insert(0, (srcs, versions, plan))
    del memo[4:]  # the CFG branches of one render
    return plan


def _apply_trim(states: Any, mask: Any, decided: tuple) -> tuple:
    """``_trim_stream`` with its decisions already made: the same columns, gathered on the device."""
    if states is None or mask is None or mask.dim() != 2:
        return states, mask, True
    index, all_valid = decided
    if index is not None:
        states = states.index_select(1, index)
        mask = mask.index_select(1, index)
    return states, mask, all_valid


def _hunyuan_trim_pre_hook(module, args, kwargs):
    """Eager forward pre-hook: strip padded text tokens so the joint attention runs fused.

    - Drop the image stream when it is entirely zero (t2v): those ~729 tokens are pure padding. This
    is upstream's own t2v sentinel (``is_t2v = torch.all(image_embeds == 0)``), and ``torch.all`` of
    an empty tensor is vacuously True, so emptying the axis keeps it True.

    - Trim the mllm/byt5 text streams to their globally-valid columns.

    - Flag every block's attention so the null-mask processor skips the dense mask when nothing
    partially-padded remains; otherwise leave the flag False and the stock dense-mask path handles
    the residual padding.

    This hook is the correctness choke point: the null-mask flag is valid only because the padding
    was removed HERE, on the same call. It fires on ``module(...)`` (``__call__``), which the
    pipeline/guider/cache_context/compile all use. Do NOT invoke a hooked DiT via
    ``module.forward(...)`` directly: that skips pre-hooks, so a stale True flag would null the mask
    over un-trimmed padding and corrupt the output.

    The three ``.item()`` reads below are host syncs, but this hook runs eagerly outside the
    compiled blocks (~3 syncs against a ~1.3 s forward), so they must stay here and not be folded
    into the graph. Best-effort: any anomaly leaves the inputs untouched and the flag False.
    """
    import torch

    original = dict(kwargs)
    try:
        null_ok = True

        plan = _trim_plan(module, kwargs)
        image = kwargs.get("image_embeds")
        if plan["t2v"]:
            # All-zero image == "no image" (t2v). Emptying the token axis removes the 729 padded image tokens; is_t2v
            # stays True in forward (all() of empty is vacuously True).
            kwargs["image_embeds"] = image[:, :0]

        for skey, mkey, required in (
            ("encoder_hidden_states", "encoder_attention_mask", True),
            ("encoder_hidden_states_2", "encoder_attention_mask_2", False),
        ):
            # Only touch streams passed by keyword (the pipeline always does); never write back an absent key (a
            # positional encoder_hidden_states would collide). An absent REQUIRED primary stream drops the fast path;
            # an absent optional byt5 is fine.
            if skey not in kwargs:
                null_ok = null_ok and not required
                continue
            states, mask, all_valid = _apply_trim(kwargs.get(skey), kwargs.get(mkey), plan[mkey])
            kwargs[skey] = states
            kwargs[mkey] = mask
            null_ok = null_ok and all_valid

        # The primary mllm stream flows through the TokenRefiner's own attention, whose pooling divides by the mask sum;
        # never hand it a 0-length sequence (pathological empty prompt). Revert and take the stock dense-mask path.
        primary = kwargs.get("encoder_hidden_states")
        if primary is not None and primary.dim() == 3 and primary.shape[1] == 0:
            kwargs.clear()
            kwargs.update(original)
            null_ok = False

        _set_hunyuan_null_mask(module, null_ok)
        return args, kwargs
    except Exception:  # noqa: BLE001 - optimisation only; never break the forward
        # We may have trimmed some kwargs before failing. Restore the caller's untrimmed inputs so the stock dense-mask
        # path (flag False) runs on exactly what it expects.
        kwargs.clear()
        kwargs.update(original)
        _set_hunyuan_null_mask(module, False)
        return args, kwargs


def _hunyuan_trim_post_hook(module, _args, output):
    """Clear the null-mask flag after each hooked forward, scoping the authorisation to exactly the
    call whose pre-hook removed the padding. Registered with ``always_call=True`` so the flag is
    also cleared when the forward raises -- otherwise a latched True would null the mask over
    un-trimmed padding on any later direct ``module.forward(...)``. Returns the output unchanged."""
    _set_hunyuan_null_mask(module, False)
    return output


def _install_null_processors(dit: Any, logger: Any) -> bool:
    """Swap every stock block attention processor on ``dit`` for the null-mask subclass. Only
    touches blocks whose processor is exactly the stock class (so a diffusers change or an
    already-installed run is a no-op). Preserves any pinned attention backend."""
    try:
        cls = _null_mask_processor_cls()
    except Exception as exc:  # noqa: BLE001 - diffusers moved / unavailable -> skip
        _warn(logger, "hunyuan_attn_trim", exc)
        return False
    installed = 0
    for blk in getattr(dit, "transformer_blocks", []):
        attn = getattr(blk, "attn", None)
        proc = getattr(attn, "processor", None) if attn is not None else None
        if proc is None:
            continue
        if isinstance(proc, cls):
            installed += 1
            continue
        if type(proc).__name__ != _HUNYUAN15_PROCESSOR_CLS:
            continue
        new = cls()
        # carry over any backend/parallel config the stock processor already held
        new._attention_backend = getattr(proc, "_attention_backend", None)
        new._parallel_config = getattr(proc, "_parallel_config", None)
        try:
            attn.set_processor(new)
        except Exception:  # noqa: BLE001 - fall back to direct assignment
            attn.processor = new
        installed += 1
    return installed > 0


def install_hunyuan_attention_trim(
    pipe: Any,
    family: Any,
    *,
    logger: Any = None,
) -> bool:
    """HunyuanVideo-1.5 only: make the joint attention skip padded text tokens (see module note).

    Installs a null-mask processor on every denoiser DiT block plus an eager pre-hook that trims the
    padded text/image streams each forward. Exact for the video output (the fused-vs-masked SDPA
    swap is the only numeric change). Returns True when engaged; No-op (False) for any other family,
    an unexpected class, or any failure -- the stock dense-mask path stays, so correctness never
    depends on this. Call BEFORE apply_attention_backend so the kernel pins onto the new processor.

    The caller must NOT install this when the denoiser blocks are compiled with static shapes: the
    trimmed text length varies per prompt (see the SHAPE NOTE in the module header)."""
    if getattr(family, "transformer_class", None) != _HUNYUAN15_TRANSFORMER_CLS:
        return False
    engaged = False
    for dit in _attention_dits(pipe):
        if type(dit).__name__ != _HUNYUAN15_TRANSFORMER_CLS:
            continue
        if not _install_null_processors(dit, logger):
            continue
        # installation and every idle period start in the conservative state
        _set_hunyuan_null_mask(dit, False)
        # The flag picks the attention branch inside the forward, so a CUDA graph keys on it (diffusion_cuda_graph).
        dit._unsloth_graph_key_extra = functools.partial(_hunyuan_null_mask_state, dit)
        if getattr(dit, "_unsloth_trim_hook", None) is None:
            pre_handle = None
            try:
                pre_handle = dit.register_forward_pre_hook(_hunyuan_trim_pre_hook, with_kwargs = True)
                # always_call: clear the flag even when the forward raises, so an exception can never leave the
                # null-mask authorisation latched for a later direct forward.
                post_handle = dit.register_forward_hook(_hunyuan_trim_post_hook, always_call = True)
                dit._unsloth_trim_hook = (pre_handle, post_handle)
            except Exception as exc:  # noqa: BLE001 - optimisation only
                if pre_handle is not None:
                    pre_handle.remove()
                _set_hunyuan_null_mask(dit, False)
                _warn(logger, "hunyuan_attn_trim", exc)
                continue
        engaged = True
    if engaged and logger is not None:
        logger.info("diffusion.attention: hunyuan padded-text trim engaged")
    return engaged
