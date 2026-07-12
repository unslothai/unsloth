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
        # AITER is the AMD ROCm kernel, not an NVIDIA one: honor it on a ROCm (AMD) CUDA
        # target and drop it everywhere else (diffusers' own set-time check rejects it off
        # ROCm anyway). Without this special-case the NVIDIA-only guard below would silently
        # drop the one explicit backend that only ever works on ROCm.
        if backend == "aiter":
            if getattr(target, "device", None) == "cuda" and not _is_cuda_nvidia(target):
                return backend
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


def _kernels_hub_compatible(logger: Any = None) -> bool:
    """Whether installing the ``kernels`` package is SAFE next to the resident huggingface_hub.
    Every ``kernels`` release (>= 0.13) needs hub >= 1.0's strict-dataclass API, and with an older
    hub the breakage is NOT contained: ``import kernels`` raises at module scope and diffusers
    imports ``kernels`` whenever installed, so EVERY later pipeline import crashes until it is
    uninstalled (measured: hub 0.36 + kernels 0.13/0.16 both brick the HunyuanVideo-1.5 import).
    So hub < 1.0 stacks must not auto-install it; the hub backend falls back to native. An
    undeterminable hub version allows the install (previous behaviour)."""
    try:
        from importlib.metadata import version
        return int(version("huggingface_hub").split(".", 1)[0]) >= 1
    except Exception:  # noqa: BLE001 -- unknown hub -> keep the previous behaviour
        return True


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
    if package == "kernels" and not _kernels_hub_compatible(logger):
        if logger is not None:
            logger.warning(
                "diffusion.attention: not installing 'kernels' for backend=%s -- the "
                "resident huggingface_hub is < 1.0 and a kernels install would break "
                "every later diffusers pipeline import; using the default backend",
                backend,
            )
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


# --------------------------------------------------------------------------------------
# HunyuanVideo-1.5 joint-attention padding trim (accuracy-exact speed win)
#
# HunyuanVideo15AttnProcessor2_0 runs a JOINT [video ; text] self-attention and, on EVERY block
# and step, materialises a dense [B,1,N,N] boolean mask so the video never attends to padded text.
# But a dense bool attn_mask DISABLES every fused SDPA kernel (flash rejects it; cuDNN/efficient
# fall back) and forces the slow math path: on a B200 at the production shape (N~=50k, 121 frames
# 480p) the SAME attention is 421 ms WITH the dense mask vs 19 ms with attn_mask=None -- a ~22x tax
# purely to mask padding. And the text is ~99.5% padding: a t2v prompt fills only ~9 of ~1985 slots
# (image 729 + byt5 256 + mllm 1000, almost all zero-padded).
#
# The fix is exact: the model already masks the padded text and DISCARDS its attention output (only
# the video split feeds proj_out), so removing the padded tokens before attention changes nothing
# for the video. Done in an eager forward pre-hook (outside the compiled blocks): drop the all-zero
# image stream (t2v), trim the mllm/byt5 streams to their globally-valid columns, and -- when
# nothing partially-padded remains (the common batch-1 / per-branch call) -- flag the DiT so the
# processor skips the dense mask and runs the fused path. The only numeric change is the SDPA kernel
# (masked -> fused), on par with the shipped cuDNN backend swap. Mixed-padding batches fall back to
# the stock dense mask.
_HUNYUAN15_TRANSFORMER_CLS = "HunyuanVideo15Transformer3DModel"
_HUNYUAN15_PROCESSOR_CLS = "HunyuanVideo15AttnProcessor2_0"
_NULL_ATTN_FLAG = "_unsloth_null_attn_mask"

_NULL_PROCESSOR_CACHE: dict = {}


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
                return hidden_states, encoder_hidden_states

            return hidden_states

    _NULL_PROCESSOR_CACHE["cls"] = _HunyuanNullMaskProcessor
    return _HunyuanNullMaskProcessor


def _trim_stream(states, mask):
    """Drop the columns of a [B, S, D] text stream + its [B, S] mask that are padding for EVERY
    batch element (globally invalid). Returns (states, mask, all_valid): all_valid is True when
    the trimmed stream has NO partially-padded column left (so it needs no attention mask)."""
    import torch

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
    - Trim the mllm/byt5 text streams to their globally-valid columns.
    - Flag every block's attention so the null-mask processor skips the dense mask when nothing
      partially-padded remains (the batch-1 / per-guidance-branch case); otherwise leave the
      flag False and the stock dense-mask path handles the residual padding correctly.

    This hook is the correctness choke point: the null-mask flag is valid only because the padding
    was removed HERE, on the same call. It fires on ``module(...)`` (``__call__``), which the
    pipeline/guider/cache_context/compile all use. Do NOT invoke a hooked DiT via
    ``module.forward(...)`` directly: that skips pre-hooks, so a stale True flag would null the mask
    over un-trimmed padding and corrupt the output.

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

        # The primary mllm stream flows through the TokenRefiner's own attention; never hand it a
        # 0-length sequence (pathological empty prompt). Revert and take the stock dense-mask path.
        primary = kwargs.get("encoder_hidden_states")
        if primary is not None and primary.dim() == 3 and primary.shape[1] == 0:
            kwargs.clear()
            kwargs.update(original)
            null_ok = False

        for blk in getattr(module, "transformer_blocks", []):
            attn = getattr(blk, "attn", None)
            if attn is not None:
                setattr(attn, _NULL_ATTN_FLAG, null_ok)

        return args, kwargs
    except Exception:  # noqa: BLE001 — optimisation only; never break the forward
        # We may have trimmed some kwargs before failing. Restore the caller's untrimmed inputs so
        # the stock dense-mask path (flag False) runs on exactly what it expects.
        kwargs.clear()
        kwargs.update(original)
        for blk in getattr(module, "transformer_blocks", []):
            attn = getattr(blk, "attn", None)
            if attn is not None:
                setattr(attn, _NULL_ATTN_FLAG, False)
        return args, kwargs


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
    padded text/image streams each forward. Bit-exact for the video output (the fused-vs-masked SDPA
    swap is the only numeric change). Returns True when engaged; No-op (False) for any other family,
    an unexpected class, or any failure -- the stock dense-mask path stays, so correctness never
    depends on this. Call BEFORE apply_attention_backend so the kernel pins onto the new processor."""
    if getattr(family, "transformer_class", None) != _HUNYUAN15_TRANSFORMER_CLS:
        return False
    engaged = False
    for dit in _attention_dits(pipe):
        if type(dit).__name__ != _HUNYUAN15_TRANSFORMER_CLS:
            continue
        if not _install_null_processors(dit, logger):
            continue
        if getattr(dit, "_unsloth_trim_hook", None) is None:
            try:
                handle = dit.register_forward_pre_hook(_hunyuan_trim_pre_hook, with_kwargs = True)
                dit._unsloth_trim_hook = handle
            except Exception as exc:  # noqa: BLE001 — optimisation only
                _warn(logger, "hunyuan_attn_trim", exc)
                continue
        engaged = True
    if engaged and logger is not None:
        logger.info("diffusion.attention: hunyuan padded-text trim engaged")
    return engaged
