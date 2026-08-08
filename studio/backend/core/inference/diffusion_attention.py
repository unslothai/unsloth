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
  sage   - SageAttention (INT8 QK); quantized, small quality cost, consumer-friendly.
  xformers / aiter - memory-efficient (NVIDIA) / AITER (AMD ROCm).

Best-effort: an unavailable backend falls back to the diffusers default. torch/diffusers lazy.
"""

from __future__ import annotations

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

        # Shared with the stub installer: torch.version.hip alone misreads AMD wheels that only tag
        # __version__, dropping aiter and pointing cuDNN/xformers (stubbed there) at a ROCm card.
        from core._torchao_stub import _module_is_rocm
        return not _module_is_rocm(torch)
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
    # Never handed to pip as a name -- see _MATCHED_WHEEL_BACKENDS below; the package
    # string survives only for logging and for the _INSTALL_ATTEMPTED bookkeeping.
    "xformers": ("xformers", "xformers"),
}

# On-demand install gate (mirrors UNSLOTH_DIFFUSION_SD_CPP_INSTALL): auto (default) / 1 installs a missing package when a gated backend is requested; 0 never installs and falls back to native.
_ATTENTION_INSTALL_ENV = "UNSLOTH_DIFFUSION_ATTENTION_INSTALL"

# Packages a pip install was already attempted for in THIS process. The loader pre-installs outside its locks, so a recorded attempt stops apply re-running the 600s install under _generate_lock.
_INSTALL_ATTEMPTED: set[str] = set()

# Backends whose wheel must be resolved against the RUNNING torch build instead of handed
# to pip as a name. Only DETERMINISTIC answers are memoised (a URL, or a refusal that
# depends purely on the resident torch, which cannot change under a running interpreter).
# A probe timeout is transient and is deliberately NOT cached: caching it would turn one
# loaded-machine hiccup into "no xFormers for the rest of this Studio session".
_MATCHED_WHEEL_BACKENDS = frozenset({"xformers"})
_XFORMERS_WHEEL_TARGET: Optional[tuple[Optional[str], Optional[str]]] = None
_XFORMERS_WHEEL_LOCK = threading.Lock()


def _xformers_wheel_target() -> tuple[Optional[str], Optional[str]]:
    """Resolve the xFormers wheel built for the resident torch: (URL, refusal reason).

    xformers' compiled extension is linked against ONE exact (torch, CUDA) pair, and next
    to any other pair ``torch.ops.load_library`` raises -- which xformers/_cpp_lib.py then
    downgrades to a log warning, so the import "succeeds" with memory-efficient attention,
    SwiGLU and the sparse ops silently gone. That is invisible to ``find_spec`` and to pip.
    PyPI publishes only the CUDA-12.8 flavour, so a plain ``pip install xformers`` beside a
    cu130 torch installs the broken combination every time.

    So resolve the exact download.pytorch.org wheel instead, and when no wheel matches
    return a reason rather than a URL: installing nothing leaves the caller on torch SDPA,
    which is strictly better than installing an extension that cannot load.

    The URL is not HEAD-checked here. This can run under ``_generate_lock`` (the video
    loader has no out-of-lock pre-install hop, unlike the image one), so it must not add
    network round trips to a path that already blocks unload/cancel; a wrong row surfaces
    as a pip failure instead, and the matrix has a live-URL test behind it.
    """
    global _XFORMERS_WHEEL_TARGET
    with _XFORMERS_WHEEL_LOCK:
        if _XFORMERS_WHEEL_TARGET is not None:
            return _XFORMERS_WHEEL_TARGET
        try:
            from utils.wheel_utils import probe_torch_wheel_env, xformers_wheel_url

            # include_windows: this is the one resolver that HAS win_amd64 wheels
            # upstream. timeout matches the other probe_torch_wheel_env callers.
            env = probe_torch_wheel_env(timeout = 30, include_windows = True)
        except Exception as exc:  # noqa: BLE001 -- must never break a model load
            return (None, f"the xFormers wheel could not be resolved ({exc})")
        if env is None:
            # Ambiguous: a platform wheel_platform_tag() does not name (macOS, Windows on
            # ARM) which is deterministic, or a probe that timed out on a busy box which is
            # transient. Not cached, so the next request can settle it. Linux aarch64 is
            # NOT here -- it gets a platform_tag and so lands on the branch below.
            return (
                None,
                "torch could not be probed, or this platform has no xFormers wheel "
                "(macOS / Windows on ARM)",
            )
        url = xformers_wheel_url(env)
        if url is None:
            # Name the platform. Linux aarch64 reaches here with a perfectly ordinary
            # torch, and reporting only the torch and CUDA would read as "upstream never
            # built this pair" when the truth is "upstream never built it for this arch".
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


# The huggingface_hub floor the current `kernels` wheels declare (kernels >= 0.14.1 requires
# huggingface-hub >= 1.10.0). A (major, minor) pair, compared against the resident hub below.
_KERNELS_HUB_FLOOR = (1, 10)


def _kernels_hub_compatible() -> bool:
    """Whether installing the ``kernels`` package is SAFE next to the resident huggingface_hub.

    Current ``kernels`` wheels declare ``huggingface_hub >= 1.10`` and build their dependency
    tables against that API, and with an older hub the breakage is NOT contained to the requested
    backend: ``import kernels`` raises at module scope, and diffusers imports ``kernels`` whenever
    it is installed, so EVERY later pipeline import in every process fails until the package is
    uninstalled. Measured with kernels 0.16.0: hub 1.0.0-1.2.4 raise
    ``StrictDataclassFieldValidationError`` on ``import kernels`` (the strict dataclasses only
    learned ``str | None`` unions in hub 1.3.0), and 1.3-1.9 merely happen to work today, below
    the floor kernels supports. The whole 1.x range under 1.10 is therefore refused rather than
    trusted, since the install is unpinned and a future kernels may use any 1.10 API. The
    requested hub backend falls back to native instead. An undeterminable hub version allows the
    install, which keeps the previous behaviour.

    A ``--no-deps`` install cannot self-correct here: pip writes the wheel without ever reading
    its ``Requires-Dist``, so this predicate is the only thing enforcing that floor."""
    try:
        import re
        from importlib.metadata import version

        m = re.match(r"\s*(\d+)(?:\.(\d+))?", version("huggingface_hub"))
        if m is None:
            return True
        return (int(m.group(1)), int(m.group(2) or 0)) >= _KERNELS_HUB_FLOOR
    except Exception:  # noqa: BLE001 — unknown hub -> keep the previous permissive behaviour
        return True


def _ensure_attention_backend_installed(backend: str, logger: Any = None) -> Optional[str]:
    """Best-effort wheel-only install of the package ``backend`` needs, when allowed.

    Called after arch gating, so only for a backend that could work here. Failure is swallowed:
    the subsequent set_attention_backend raises on the missing package and falls back to native.

    Returns the reason the install was REFUSED (a policy decision, e.g. no CUDA-matched
    xFormers wheel exists for the resident torch), or None when nothing stood in the way --
    the install ran, was skipped as already present, or merely failed. Every refusal is also
    logged at warning level; the return value is there so a caller that wants to surface the
    reason (rather than silently falling back to native) can, and both current callers
    deliberately ignore it."""
    import importlib.util
    import os

    spec = _INSTALLABLE_BACKENDS.get(backend)
    if spec is None:
        return None
    module, package = spec
    package = _pip_requirement(backend, package)
    gate = os.environ.get(_ATTENTION_INSTALL_ENV, "auto").strip().lower()
    if gate in ("0", "false", "no", "off"):
        return None
    # Refusing is a POLICY decision, not a failed attempt, so it is checked before the
    # _INSTALL_ATTEMPTED memo below and records nothing: a later request on a fixed environment
    # must still be able to install. Scoped to kernels; the sage / flash-attn / xformers wheels
    # do not import huggingface_hub at module scope.
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
            # Present is present, including a MISMATCHED xformers: find_spec sees the
            # package, so nothing below runs and the wrong-CUDA build stays. That is
            # deliberate here. Repairing means reinstalling a package the user may have
            # built or pinned on purpose, and this can run under _generate_lock, so a
            # 100 MB download would block unload and cancel. install.ps1 is where the
            # repair belongs -- it compares cpp_lib.json against the resident torch and
            # passes --reinstall-package, outside any request. What this branch prevents
            # is Studio CREATING the mismatch, which is how it got made in the first place.
            return None
    except Exception:  # noqa: BLE001 — a broken install probes as missing; try the install
        pass
    # xFormers ships a compiled extension tied to one exact (torch, CUDA) pair, so the name
    # `xformers` is not a safe thing to hand pip: PyPI serves only the CUDA-12.8 build and
    # --no-deps below deliberately stops pip from ever reading its `Requires-Dist: torch==X`.
    # Resolve the matching wheel URL instead, and REFUSE when there is none -- like the
    # kernels gate above this is policy, so it is checked before the _INSTALL_ATTEMPTED memo
    # and records nothing there (a refused backend never burns its one install attempt).
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
    # Attempt each install once per process, else the in-lock apply re-runs it under _generate_lock and blocks unload/cancel.
    if package in _INSTALL_ATTEMPTED:
        return None
    _INSTALL_ATTEMPTED.add(package)
    import subprocess
    import sys

    if logger is not None:
        logger.info(
            "diffusion.attention: installing %s for backend=%s (wheel-only)", package, backend
        )
    try:
        subprocess.run(
            # --no-deps: install ONLY this kernel wheel, since xformers/flash-attn pin an exact torch and normal resolution would replace the running one. It also means pip never reads the wheel's `Requires-Dist: torch==X`, so nothing here would catch an ABI mismatch -- for xformers, whose mismatch is SILENT (the extension fails to load and _cpp_lib.py logs a warning), the URL was resolved against the running torch above precisely so there is nothing left to catch.
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
    return None


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


# --------------------------------------------------------------------------------------
# HunyuanVideo-1.5 joint-attention padding trim (accuracy-exact speed win)
#
# HunyuanVideo15AttnProcessor2_0 runs a JOINT [video ; text] self-attention and, on EVERY block
# and step, materialises a dense [B,1,N,N] boolean mask so the video never attends to padded text.
# But a dense bool attn_mask costs most of what the fused kernels are for. Established by output
# identity, not by timing, since dispatch overhead makes timings alone ambiguous: FLASH refuses a
# non-null mask outright, MATH OOMs on the 75.5 GiB [1,16,N,N] score matrix, and the default
# dispatch is BITWISE-equal to forced cuDNN (296.11 vs 296.00 ms), so cuDNN is what runs -- on a
# masked path 20x slower than its own unmasked one. On a B200 at the production shape (N~=50k, 121
# frames 480p) the SAME attention is 296 ms WITH the dense mask vs 15 ms with attn_mask=None. (An
# aside worth knowing before optimising here: forced EFFICIENT does the masked attention in 168 ms,
# so the dispatcher's masked pick is not even the fastest available one.)
# (torch 2.12; scripts/sdpa_mask_backend_probe.py re-measures it, and also shows MATH OOMing on the
# 75.5 GiB score matrix and FLASH refusing a dense mask outright). END TO END that is 10.4x: a full
# 121-frame 832x480 10-step render goes 353.8s -> 33.9s, medians of 3, reproduced across two runs
# purely to mask padding. And the text is ~99.5% padding: a t2v prompt fills only ~9 of ~1985 slots
# (image 729 + byt5 256 + mllm 1000, almost all zero-padded).
#
# The fix is exact: the model already masks the padded text and DISCARDS its attention output (only
# the video split feeds proj_out), so removing the padded tokens before attention changes nothing
# for the video. "Exact" here means no information is discarded, NOT bit-reproducible: swapping
# masked for fused SDPA perturbs each step at bf16 rounding scale (one DiT forward on identical
# inputs differs by 6.6e-3 relative, cosine 0.99998) and 10 denoising steps amplify that
# chaotically, so the finished video is visibly a different sample. That is intrinsic to the kernel
# change, not to the trim: rendering the SAME dense-mask path under two different exact SDPA kernels
# diverges MORE (LPIPS 0.303 vs the trim's 0.285, SSIM 0.744 vs 0.767, over 13 sampled frames).
# Whole-video LPIPS cannot judge a kernel change at this step count; the single-forward relative
# error is the metric that can.
# Done in an eager forward pre-hook (outside the compiled blocks): drop the all-zero
# image stream (t2v), trim the mllm/byt5 streams to their globally-valid columns, and -- when
# nothing partially-padded remains (the common batch-1 / per-branch call) -- flag the DiT so the
# processor skips the dense mask and runs the fused path. The only numeric change is the SDPA kernel
# (masked -> fused), on par with the shipped cuDNN backend swap. Mixed-padding batches fall back to
# the stock dense mask.
#
# SHAPE NOTE: the trimmed text length is prompt-dependent, so the compiled blocks see a new shape
# per prompt. That is free on the default speed tier (compiled with dynamic=True) but not on ``max``
# (dynamic=False), where each length is its own graph and a fullgraph region hard-errors once
# dynamo's recompile limit is reached. The caller therefore only installs the trim on a tier that
# compiles dynamically; see the call site in video.py.
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
            # Fast path only when the pre-hook removed all padding (attn_mask redundant); a
            # constant python bool so torch.compile const-folds the branch (no graph break).
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

            # Always the 2-tuple, matching the stock processor's return contract (it returns
            # (hidden_states, encoder_hidden_states) outside its own `if`), so the calling block
            # unpacks identically on either path.
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
    # All remaining slots valid for every element (vacuously True for a 0-length stream, fine
    # for an unused secondary stream e.g. byt5 in t2v).
    all_valid = bool(mb.all().item())
    return states, mask, all_valid


def _hunyuan_trim_pre_hook(module, args, kwargs):
    """Eager forward pre-hook: strip padded text tokens so the joint attention runs fused.

    - Drop the image stream when it is entirely zero (t2v): those ~729 tokens are pure padding.
      This is upstream's own t2v sentinel (``is_t2v = torch.all(image_embeds == 0)``), and
      ``torch.all`` of an empty tensor is vacuously True, so emptying the axis keeps it True.
    - Trim the mllm/byt5 text streams to their globally-valid columns.
    - Flag every block's attention so the null-mask processor skips the dense mask when nothing
      partially-padded remains (the batch-1 / per-guidance-branch case); otherwise leave the
      flag False and the stock dense-mask path handles the residual padding correctly.

    This hook is the correctness choke point: the null-mask flag is valid only because the padding
    was removed HERE, on the same call. It fires on ``module(...)`` (``__call__``), which the
    pipeline/guider/cache_context/compile all use. Do NOT invoke a hooked DiT via
    ``module.forward(...)`` directly: that skips pre-hooks, so a stale True flag would null the mask
    over un-trimmed padding and corrupt the output.

    The three ``.item()`` reads below are host syncs, but this hook runs eagerly outside the
    compiled blocks (~3 syncs against a ~1.3 s forward), so they must stay here and not be folded
    into the graph.

    Best-effort: any anomaly leaves the inputs untouched and the flag False."""
    import torch

    original = dict(kwargs)
    try:
        null_ok = True

        image = kwargs.get("image_embeds")
        if image is not None and image.numel() > 0 and bool(torch.all(image == 0).item()):
            # All-zero image == "no image" (t2v). Emptying the token axis removes the 729 padded
            # image tokens; is_t2v stays True in forward (all() of empty is vacuously True).
            kwargs["image_embeds"] = image[:, :0]

        for skey, mkey, required in (
            ("encoder_hidden_states", "encoder_attention_mask", True),
            ("encoder_hidden_states_2", "encoder_attention_mask_2", False),
        ):
            # Only touch streams passed by keyword (the pipeline always does); never write back an
            # absent key (a positional encoder_hidden_states would collide). An absent REQUIRED
            # primary stream drops the fast path; an absent optional byt5 is fine.
            if skey not in kwargs:
                null_ok = null_ok and not required
                continue
            states, mask, all_valid = _trim_stream(kwargs.get(skey), kwargs.get(mkey))
            kwargs[skey] = states
            kwargs[mkey] = mask
            null_ok = null_ok and all_valid

        # The primary mllm stream flows through the TokenRefiner's own attention, whose pooling
        # divides by the mask sum; never hand it a 0-length sequence (pathological empty prompt).
        # Revert and take the stock dense-mask path.
        primary = kwargs.get("encoder_hidden_states")
        if primary is not None and primary.dim() == 3 and primary.shape[1] == 0:
            kwargs.clear()
            kwargs.update(original)
            null_ok = False

        _set_hunyuan_null_mask(module, null_ok)
        return args, kwargs
    except Exception:  # noqa: BLE001 — optimisation only; never break the forward
        # We may have trimmed some kwargs before failing. Restore the caller's untrimmed inputs so
        # the stock dense-mask path (flag False) runs on exactly what it expects.
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
    except Exception as exc:  # noqa: BLE001 — diffusers moved / unavailable -> skip
        _warn(logger, "hunyuan_attn_trim", exc)
        return False
    installed = 0
    for blk in getattr(dit, "transformer_blocks", []):
        attn = getattr(blk, "attn", None)
        proc = getattr(attn, "processor", None) if attn is not None else None
        if proc is None:
            continue
        if isinstance(proc, cls):
            installed += 1  # already ours (idempotent)
            continue
        if type(proc).__name__ != _HUNYUAN15_PROCESSOR_CLS:
            continue  # unknown processor -> leave it alone
        new = cls()
        # carry over any backend/parallel config the stock processor already held
        new._attention_backend = getattr(proc, "_attention_backend", None)
        new._parallel_config = getattr(proc, "_parallel_config", None)
        try:
            attn.set_processor(new)
        except Exception:  # noqa: BLE001 — fall back to direct assignment
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
        # Installation (and every idle period between generations) starts in the conservative
        # state: the flag is only ever True inside the exact forward its pre-hook trimmed.
        _set_hunyuan_null_mask(dit, False)
        if getattr(dit, "_unsloth_trim_hook", None) is None:
            pre_handle = None
            try:
                pre_handle = dit.register_forward_pre_hook(_hunyuan_trim_pre_hook, with_kwargs = True)
                # always_call: clear the flag even when the forward raises, so an exception can
                # never leave the null-mask authorisation latched for a later direct forward.
                post_handle = dit.register_forward_hook(_hunyuan_trim_post_hook, always_call = True)
                dit._unsloth_trim_hook = (pre_handle, post_handle)
            except Exception as exc:  # noqa: BLE001 — optimisation only
                if pre_handle is not None:
                    pre_handle.remove()
                _set_hunyuan_null_mask(dit, False)
                _warn(logger, "hunyuan_attn_trim", exc)
                continue
        engaged = True
    if engaged and logger is not None:
        logger.info("diffusion.attention: hunyuan padded-text trim engaged")
    return engaged
