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


# Backends diffusers validates only by package at set time but whose kernels need a specific
# CUDA arch at run time (so an explicit request on the wrong card sets fine then crashes
# mid-generation). Gate by a (min, max-exclusive) capability range: FA3 is Hopper-SM90 only
# (upper bound, so flash3 on a B200 drops to native), FA4 is Blackwell+ (no upper bound).
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
        # AITER is the AMD ROCm kernel: honor it on a ROCm CUDA target, drop it elsewhere (else
        # the NVIDIA-only guard below would drop the one backend that only works on ROCm).
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


def attention_backend_supported_on_device(backend: Optional[str], device_index: int) -> bool:
    """Whether an already-resolved dispatcher backend can actually RUN on CUDA ``device_index``.

    ``select_attention_backend`` arch-gates a backend against the ACTIVE device, but a CFG-parallel
    replica lives on a possibly HETEROGENEOUS second GPU (FA3 is Hopper-SM90 only, FA4 needs
    Blackwell-SM100, cuDNN needs Ampere+). Installing the primary-resolved backend there without
    re-checking would set fine then crash on the replica's first attention kernel. Re-applies the
    same arch gate to a specific device index. None (native) is always fine; an unqueryable
    capability returns True (best-effort, matching ``_backend_arch_supported``)."""
    if backend is None:
        return True
    try:
        import torch
        have = tuple(torch.cuda.get_device_capability(device_index))  # type: ignore[assignment]
    except Exception:  # noqa: BLE001 -- unqueryable device: don't block on a guess
        return True
    bounds = _ARCH_CAPABILITY.get(backend)
    if bounds is not None:
        low, high = bounds
        if not (have >= low and (high is None or have < high)):
            return False
    if backend == "_native_cudnn" and have < (8, 0):
        return False
    return True


# Optional-kernel backends installable on demand: dispatcher name -> (probe module, pip
# package). Wheels only (--only-binary=:all:): a source build needs a CUDA toolchain a Studio
# host may lack; no wheel means a native fallback. cuDNN/native ship with torch.
_INSTALLABLE_BACKENDS: dict[str, tuple[str, str]] = {
    "sage": ("sageattention", "sageattention"),
    "flash": ("flash_attn", "flash-attn"),
    "_flash_3_hub": ("kernels", "kernels"),  # FA3/FA4 from the HF kernels hub
    "flash_4_hub": ("kernels", "kernels"),
    "xformers": ("xformers", "xformers"),
}

# On-demand install gate (mirrors UNSLOTH_DIFFUSION_SD_CPP_INSTALL):
#   auto (default) / 1 - install the missing package when a gated backend is requested
#   0                  - never install; a missing kernel falls back to native
_ATTENTION_INSTALL_ENV = "UNSLOTH_DIFFUSION_ATTENTION_INSTALL"

# Packages a pip install was already attempted for in THIS process (success or failure). The
# loader pre-installs outside its locks, then re-resolves under _generate_lock where apply would
# otherwise call pip a SECOND time -- a no-wheel/offline host would re-run the full 600s install
# holding the load lock, blocking unload/cancel. A recorded attempt makes the retry a no-op.
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

    Called after arch gating, so only for a backend that could work here. Failure is swallowed:
    the subsequent set_attention_backend raises on the missing package and falls back to native."""
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
    # Attempt each install once per process (see _INSTALL_ATTEMPTED): else the in-lock apply path
    # re-runs the whole install under _generate_lock and blocks unload/cancel.
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
            # --no-deps: install ONLY this kernel wheel. xformers/flash-attn pin an exact torch,
            # so normal resolution would replace the running torch/triton. Without deps an
            # ABI-incompatible kernel just fails to import -> native fallback.
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
        # The import system caches directory listings, so the next find_spec can miss the wheel
        # just installed (mtime resolution). Invalidate the finder caches so it's picked up now.
        importlib.invalidate_caches()
    except Exception as exc:  # noqa: BLE001 — no wheel / no network -> native fallback
        if logger is not None:
            # CalledProcessError.str() shows only the exit code; the real reason is in stderr.
            # Surface it so the native fallback is diagnosable.
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
            # set_attention_backend also pins the backend process-wide. Each DiT's processors now
            # keep it locally, so reset the global to native ONCE, else a later unconfigured
            # component inherits this kernel.
            _reset_global_backend_to_native(logger)
            if logger is not None:
                logger.info("diffusion.attention: backend=%s", backend)
            return backend
    # No backend requested, or every set failed: pin native so a stale process-wide backend can't
    # leak in. Fresh DiTs follow the global, so one reset via any setter covers them all.
    _restore_native_backend(setters[0], logger)
    return None


def _active_attention_backend() -> Optional[str]:
    """The diffusers process-wide active attention backend name, or None if undeterminable."""
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry

        # get_active_backend() returns (AttentionBackendName, fn) or None; take element 0 and
        # read its .value ("native"), not off the tuple (which never compares equal to a name).
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
