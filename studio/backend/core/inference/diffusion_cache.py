# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in step caching for the diffusion transformer (First-Block-Cache).

Across denoising steps a DiT's output changes little once the trajectory settles, so most of
the transformer can be reused. First-Block-Cache (FBCache) computes the first block, and if
its residual barely changed from the previous step (within ``threshold``) it skips the
remaining blocks and reuses their cached output. diffusers ships it natively
(``transformer.enable_cache(FirstBlockCacheConfig(...))`` for CacheMixin models, or the
standalone ``apply_first_block_cache`` hook).

Measured on Flux.1-dev (28 steps, 1024px, B200): ~1.4x on top of torch.compile (2.83 ->
2.03 s) at LPIPS ~0.08 vs the no-cache output -- deep inside the speed-for-quality bar.

OFF by default and a deliberate per-load opt-in, because the win scales with step count: a
few-step distilled model (e.g. Z-Image-Turbo at ~8 steps) has almost no headroom and a
single skipped step is a large fraction of the trajectory, so caching is for many-step
models (Flux / Qwen-Image). It composes with torch.compile only with ``fullgraph=False``
(the cache's compiler-disabled decision is a graph break), which the speed layer switches to
automatically when a cache is engaged. Best-effort: an incompatible model (e.g. a transformer
whose block signature the hook does not recognise) is caught and the load proceeds uncached.
torch / diffusers imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional

TC_OFF = "off"
TC_AUTO = "auto"
TC_FBCACHE = "fbcache"
TC_MAGCACHE = "magcache"
TC_MODES = (TC_FBCACHE, TC_MAGCACHE)

# FBCache residual thresholds: higher skips more steps (faster, lower quality). The dense
# bf16 default; a quantised transformer shifts the residual distribution, so it needs a
# higher threshold for the cache to trigger at all (per ParaAttention's fp8 guidance).
DEFAULT_FBCACHE_THRESHOLD = 0.08
QUANT_FBCACHE_THRESHOLD = 0.12

# MagCache (diffusers >= 0.39): skips whole steps from a PRE-CALIBRATED residual-magnitude
# curve with an accumulated-error budget, a consecutive-skip cap, and a no-skip retention
# window over the early steps -- so unlike FBCache the divergence from the uncached
# trajectory is bounded. Measured on HunyuanVideo-1.5-720p (B200, 50 steps, 720p clip):
# threshold 0.12 = 1.5x end-to-end at LPIPS 0.147 vs the same uncached stack with the SAME
# composition (FBCache at its 0.08 default reached 2.4x but LPIPS 0.54: a brighter,
# visibly different clip -- why the fbcache auto policy excludes this family).
DEFAULT_MAGCACHE_THRESHOLD = 0.12
MAGCACHE_MAX_SKIP_STEPS = 3
MAGCACHE_RETENTION_RATIO = 0.2

# The auto policy's step-count bar: FBCache's win scales with step count (each skipped
# step is a larger quality hit on a short trajectory), so auto engages it only at 20+
# steps -- full "dev"-style schedules (28+) qualify, distilled turbo models (4-9) never do.
FBCACHE_MIN_STEPS = 20


# Per-family MagCache magnitude-ratio curves (MagCacheConfig.mag_ratios), calibrated with
# diffusers' calibrate mode on the family base checkpoints at the default 50-step schedule
# (720p clip, B200). The curve is checkpoint-dependent but highly stable where it matters:
# the CFG cond/uncond branches differ by <= 0.014 and a 30-step calibration matches the
# 50-step curve within 0.027 after nearest-interpolation, so ONE curve per family is
# enough -- diffusers interpolates it to the actual step count. Conditional-branch curve
# per the MagCache calibration guidance.
_MAGCACHE_720P_RATIOS = (
    1.0,
    1.0226,
    1.0093,
    1.001,
    1.0008,
    1.0001,
    0.9995,
    1.0003,
    0.9998,
    0.9993,
    0.9994,
    0.9993,
    0.9997,
    1.0002,
    0.9994,
    0.9985,
    0.9987,
    0.9997,
    0.9979,
    0.9987,
    0.9985,
    0.9982,
    0.9977,
    0.998,
    0.9979,
    0.9971,
    0.9968,
    0.9967,
    0.9964,
    0.9965,
    0.9959,
    0.9954,
    0.995,
    0.9938,
    0.9942,
    0.9924,
    0.9924,
    0.9907,
    0.9905,
    0.9878,
    0.9867,
    0.9845,
    0.9808,
    0.9773,
    0.9715,
    0.9652,
    0.9529,
    0.9347,
    0.9011,
    0.83,
)
_MAGCACHE_480P_RATIOS = (
    1.0,
    1.0077,
    1.0138,
    1.0043,
    1.0029,
    0.9986,
    0.9966,
    1.0,
    1.0006,
    0.9996,
    0.9993,
    0.9986,
    1.0,
    0.9993,
    0.9966,
    0.9986,
    0.9988,
    0.9991,
    0.998,
    0.9977,
    0.9976,
    0.9971,
    0.9973,
    0.9969,
    0.996,
    0.9961,
    0.9949,
    0.9958,
    0.9933,
    0.9942,
    0.9941,
    0.9926,
    0.9929,
    0.9916,
    0.9923,
    0.9887,
    0.99,
    0.9882,
    0.9865,
    0.9833,
    0.9827,
    0.9791,
    0.9763,
    0.9718,
    0.9657,
    0.9563,
    0.9454,
    0.9264,
    0.8967,
    0.8382,
)
_MAGCACHE_FAMILY_RATIOS: dict[str, tuple[float, ...]] = {
    "hunyuanvideo-1.5": _MAGCACHE_480P_RATIOS,
    "hunyuanvideo-1.5-720p": _MAGCACHE_720P_RATIOS,
}

# Families whose AUTO step-cache decision engages MagCache instead of FBCache. On
# HunyuanVideo-1.5 FBCache free-runs (no skip cap, no error budget) and derails the
# trajectory (LPIPS 0.54 + a luma shift at its default threshold), while MagCache holds
# the same composition at 1.5x -- see the constants above. Every other family keeps the
# measured FBCache default. An EXPLICIT "fbcache"/"magcache" request always wins.
_FAMILY_AUTO_CACHE_MODE: dict[str, str] = {
    "hunyuanvideo-1.5": TC_MAGCACHE,
    "hunyuanvideo-1.5-720p": TC_MAGCACHE,
}


def auto_cache_mode(family: Optional[str]) -> str:
    """The cache mode the AUTO policy engages for ``family`` (mode only; the step-count
    bar and the engage call are the caller's job). MagCache additionally needs a
    calibrated ratio curve: a family routed here without one runs uncached (the
    apply_step_cache magcache branch checks), never silently falls back to FBCache."""
    return _FAMILY_AUTO_CACHE_MODE.get(str(family or "").strip().lower(), TC_FBCACHE)


def normalize_transformer_cache(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested cache mode; None / "" / "none" / "off" -> None (disabled),
    "auto" -> TC_AUTO (the loader decides from the step count).

    Raises ValueError for an unsupported value so a bad request is rejected cheaply."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized in ("none", "off"):
        return None
    if normalized == TC_AUTO:
        return TC_AUTO
    if normalized not in TC_MODES:
        raise ValueError(
            f"Unsupported transformer_cache '{value}'. Use one of: off, auto, "
            f"{', '.join(TC_MODES)}."
        )
    return normalized


# Transformer block classes whose FBCache metadata is missing from the installed
# diffusers. The First-Block-Cache hook reads each block's (hidden_states,
# encoder_hidden_states) return layout from TransformerBlockRegistry; diffusers 0.39
# registers the HunyuanVideo 1.0 blocks but not the 1.5 ones, so enable_cache raises
# "Model class HunyuanVideo15TransformerBlock not registered" on a DiT that is
# otherwise fully cache-compatible: CacheMixin, one homogeneous ``transformer_blocks``
# list of residual-additive dual-stream blocks returning (hidden_states,
# encoder_hidden_states) -- the exact layout of the registered 1.0 block. Keyed by the
# TRANSFORMER class name so only a family that needs the patch pays for it, and probed
# via TransformerBlockRegistry.get first so a diffusers release that ships the
# registration natively makes this a no-op.
#   transformer class -> ((block module, block class, hs index, ehs index), ...)
_EXTRA_BLOCK_METADATA: dict[str, tuple[tuple[str, str, int, Optional[int]], ...]] = {
    "HunyuanVideo15Transformer3DModel": (
        (
            "diffusers.models.transformers.transformer_hunyuan_video15",
            "HunyuanVideo15TransformerBlock",
            0,
            1,
        ),
    ),
}


def _ensure_block_metadata_registered(transformer: Any, logger: Any = None) -> None:
    """Register the missing FBCache block metadata for ``transformer``'s family (see
    ``_EXTRA_BLOCK_METADATA``). Best-effort: a failure just leaves enable_cache to raise
    its own error and the load runs uncached, exactly as before this patch."""
    specs = _EXTRA_BLOCK_METADATA.get(type(transformer).__name__)
    if not specs:
        return
    try:
        import importlib

        from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
        for module_name, cls_name, hs_index, ehs_index in specs:
            block_cls = getattr(importlib.import_module(module_name), cls_name)
            try:
                TransformerBlockRegistry.get(block_cls)
                continue  # a newer diffusers registers it natively
            except ValueError:
                pass
            TransformerBlockRegistry.register(
                block_cls,
                TransformerBlockMetadata(
                    return_hidden_states_index = hs_index,
                    return_encoder_hidden_states_index = ehs_index,
                ),
            )
            if logger is not None:
                logger.info("diffusion.cache: registered %s block metadata for fbcache", cls_name)
    except Exception as exc:  # noqa: BLE001 -- best-effort; enable_cache surfaces the real error
        _warn(logger, "block metadata registration", exc)


def _invalidate_child_registry_cache(transformer: Any) -> None:
    """Drop the HookRegistry's cached child-registry list after (un)installing hooks.

    ``cache_context`` propagates the state context through ``_get_child_registries``,
    which diffusers 0.39 caches on first use. An UNCACHED generation already calls
    ``cache_context`` (the pipeline wraps every denoise call), creating the
    transformer-level registry with an EMPTY cached child list -- so a later
    ``enable_cache`` (the auto step-count toggle engaging FBCache mid-session) installs
    block hooks that ``_set_context`` never reaches, and the first cached forward dies
    with "No context is set". Invalidate the stale cache so the next ``cache_context``
    rebuilds it over the freshly hooked blocks. Best-effort and cheap (one attribute)."""
    registry = getattr(transformer, "_diffusers_hook", None)
    if registry is not None and getattr(registry, "_child_registries_cache", None) is not None:
        try:
            registry._child_registries_cache = None
        except Exception:  # noqa: BLE001 -- diffusers internals moved; leave as-is
            pass


def _pipeline_opens_cache_context(pipe: Any) -> bool:
    """Whether the pipeline enters ``transformer.cache_context(...)`` in its denoise loop.
    The First-Block-Cache hook requires it at run time, and a CacheMixin transformer alone
    does NOT guarantee it: Flux Kontext / img2img / inpaint / controlnet reuse the CacheMixin
    FluxTransformer2DModel but never open a cache_context. Read from the pipeline ``__call__``
    source, resolved off the instance so a per-expert proxy view (``_SecondDiTView``)
    delegates to the real pipe; if it cannot be read, report False so the cache stays off."""
    import inspect

    call = getattr(pipe, "__call__", None)
    if call is None:
        return False
    try:
        src = inspect.getsource(call)
    except (OSError, TypeError):
        return False
    # Match the actual call `cache_context(` -- a bare mention in a comment/docstring lacks
    # the paren, so this does not false-positive on prose.
    return "cache_context(" in src


def apply_step_cache(
    pipe: Any,
    *,
    mode: Optional[str],
    threshold: Optional[float] = None,
    quant_active: bool = False,
    family: Optional[str] = None,
    steps: Optional[int] = None,
    logger: Any = None,
) -> Optional[str]:
    """Engage step caching on ``pipe.transformer``. Returns the mode actually engaged, or
    None when disabled / unsupported (the load then runs uncached). ``threshold`` overrides
    the default; ``quant_active`` raises the FBCache default so the cache still triggers on
    a quantised transformer. The magcache mode additionally needs ``family`` (to look up the
    calibrated ratio curve) and ``steps`` (MagCache interpolates that curve over the
    configured step count and sizes its no-skip retention window from it). Best-effort:
    never raises for an incompatible model."""
    mode = normalize_transformer_cache(mode)
    if mode is None or mode == TC_AUTO:
        # AUTO must be resolved by the loader (step-count policy) before reaching the
        # engage call; treat a stray auto as off rather than crashing the load.
        return None
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    if mode == TC_MAGCACHE:
        thr = threshold if threshold is not None else DEFAULT_MAGCACHE_THRESHOLD
    else:
        thr = (
            threshold
            if threshold is not None
            else (QUANT_FBCACHE_THRESHOLD if quant_active else DEFAULT_FBCACHE_THRESHOLD)
        )
    # Engage only via the transformer's native enable_cache (the diffusers CacheMixin path):
    # the lower-level apply_first_block_cache hook would install on a non-CacheMixin
    # transformer too (e.g. Z-Image), whose pipeline opens no cache_context and would crash
    # the first generation -- so a model without enable_cache runs uncached per the
    # best-effort contract instead of being reported as cached and then failing.
    enable_cache = getattr(transformer, "enable_cache", None)
    if not callable(enable_cache):
        _warn(logger, mode, RuntimeError("transformer has no cache_context (not a CacheMixin)"))
        return None
    # A CacheMixin transformer is necessary but NOT sufficient: the First-Block-Cache hook
    # raises "No context is set" on the first forward unless the PIPELINE wraps its denoise
    # loop in transformer.cache_context(...). Flux Kontext / img2img / inpaint / controlnet
    # reuse the CacheMixin FluxTransformer2DModel yet their __call__ opens no cache_context,
    # so engaging FBCache there would crash every default generation -- run uncached instead.
    if not _pipeline_opens_cache_context(pipe):
        _warn(
            logger, mode, RuntimeError("pipeline __call__ opens no cache_context; running uncached")
        )
        return None
    # Some cache-compatible block classes are missing from the installed diffusers'
    # FBCache metadata registry (HunyuanVideo-1.5); register them before enable_cache.
    # Both hook families (FBCache / MagCache) read the same block metadata.
    _ensure_block_metadata_registered(transformer, logger)
    try:
        if mode == TC_MAGCACHE:
            ratios = _MAGCACHE_FAMILY_RATIOS.get(str(family or "").strip().lower())
            if ratios is None:
                # No silent FBCache fallback: the family was routed to magcache exactly
                # because FBCache derails it, so an uncalibrated checkpoint runs uncached.
                _warn(
                    logger,
                    mode,
                    RuntimeError(f"no calibrated mag_ratios for family '{family}'"),
                )
                return None
            if not steps or int(steps) <= 0:
                _warn(logger, mode, RuntimeError("magcache needs the step count to engage"))
                return None
            from diffusers.hooks import MagCacheConfig

            config: Any = MagCacheConfig(
                threshold = thr,
                max_skip_steps = MAGCACHE_MAX_SKIP_STEPS,
                retention_ratio = MAGCACHE_RETENTION_RATIO,
                num_inference_steps = int(steps),
                mag_ratios = list(ratios),
            )
            # The curve is interpolated over the CONFIGURED step count, so the marker
            # carries it: the auto toggle re-engages on a step-count change.
            marker = f"{mode}@{thr}#s{int(steps)}"
        else:
            try:
                from diffusers import FirstBlockCacheConfig
            except ImportError:  # older diffusers exports it only from diffusers.hooks
                from diffusers.hooks import FirstBlockCacheConfig

            config = FirstBlockCacheConfig(threshold = thr)
            marker = f"{mode}@{thr}"
        enable_cache(config)
        # A prior uncached generation may have frozen an empty child-registry list on
        # the transformer's HookRegistry; the block hooks just installed would then
        # never receive the cache context. Must follow every enable_cache.
        _invalidate_child_registry_cache(transformer)
        try:
            transformer._unsloth_step_cache = marker
        except Exception:  # noqa: BLE001 — marker is best-effort
            pass
        if logger is not None:
            logger.info("diffusion.cache: %s engaged (threshold=%s)", mode, thr)
        return mode
    except Exception as exc:  # noqa: BLE001 — incompatible model -> run uncached
        # enable_cache can fail after hooking some blocks; drop any partial hooks so
        # the reported-uncached model doesn't actually run half-cached.
        try:
            transformer.disable_cache()
        except Exception:  # noqa: BLE001
            pass
        _warn(logger, mode, exc)
        return None


def effective_denoise_steps(steps: int, strength: Optional[float]) -> int:
    """The number of steps diffusers ACTUALLY denoises for a request.

    An image-conditioned workflow with ``strength`` < 1 (img2img / upscale / inpaint) runs
    only a fraction of ``num_inference_steps``: diffusers' ``get_timesteps`` computes
    ``init_timestep = min(int(num_inference_steps * strength), num_inference_steps)`` and
    denoises exactly ``init_timestep`` steps -- the product is FLOORED, not rounded. The auto
    step-cache policy must key on THIS count -- e.g. a 28-step upscale at strength 0.35 runs
    ``int(9.8) = 9`` real steps, exactly the short trajectory FBCache should stay off (each
    skipped step is a large quality hit). ``strength`` None (txt2img / reference) or >= 1 -> the
    full count.
    """
    s = int(steps)
    if strength is None or float(strength) >= 1.0:
        return s
    return max(1, min(int(s * float(strength)), s))


def effective_request_strength(
    request_strength: Optional[float],
    has_init_image: bool,
    pipe_accepts_strength: bool,
    pipe_default_strength: Any,
) -> Optional[float]:
    """The strength the pipe will ACTUALLY apply, for keying the auto step-cache policy.

    Only image-conditioned pipelines that take ``strength`` apply it (txt2img / a pipe without
    the kwarg run the full trajectory -> None). When the request omits ``strength`` the loader
    does NOT pass the kwarg, so the pipe runs its OWN signature default (< 1 for every img2img /
    inpaint pipeline here, e.g. 0.6); the policy must key on that default, not the full step
    count, or FBCache engages on a fraction of the advertised steps. A non-numeric default
    (``inspect.Parameter.empty``) falls back to the full count (None).
    """
    if not (has_init_image and pipe_accepts_strength):
        return None
    if request_strength is not None:
        return request_strength
    return pipe_default_strength if isinstance(pipe_default_strength, (int, float)) else None


def _disengage_step_cache(
    transformer: Any,
    *,
    reason: str,
    logger: Any = None,
) -> bool:
    """disable_cache + clear the marker; True when the transformer is now uncached."""
    disable_cache = getattr(transformer, "disable_cache", None)
    if not callable(disable_cache):
        return False
    try:
        disable_cache()
        transformer._unsloth_step_cache = None
        if logger is not None:
            logger.info("diffusion.cache: step cache disengaged (%s)", reason)
        return True
    except Exception as exc:  # noqa: BLE001 -- keep the cache rather than crash
        _warn(logger, "step cache disable", exc)
        return False


def maybe_toggle_step_cache(
    pipe: Any,
    *,
    steps: int,
    quant_active: bool = False,
    threshold: Optional[float] = None,
    mode: str = TC_FBCACHE,
    family: Optional[str] = None,
    logger: Any = None,
) -> Optional[str]:
    """Generation-time enable/disable for an AUTO cache decision, keyed on the actual
    step count: engage ``mode`` (the family's auto cache mode) at ``FBCACHE_MIN_STEPS``
    or more, run uncached below it. Idempotent (the ``_unsloth_step_cache`` marker tracks
    the engaged state), so calling it on every generation is cheap -- except a magcache
    step-count change, which re-engages so the ratio curve is re-interpolated over the
    actual schedule. Only the loader's auto path calls this; an explicit user choice is
    never toggled. Returns the mode now active (or None when uncached)."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    engaged = getattr(transformer, "_unsloth_step_cache", None)
    want = int(steps) >= FBCACHE_MIN_STEPS
    if (
        want
        and engaged
        and mode == TC_MAGCACHE
        and f"#s{int(steps)}" not in str(engaged)
        and _disengage_step_cache(
            transformer, reason = f"magcache re-interpolating for {steps} steps", logger = logger
        )
    ):
        engaged = None
    if want and not engaged:
        return apply_step_cache(
            pipe,
            mode = mode,
            threshold = threshold,
            quant_active = quant_active,
            family = family,
            steps = steps,
            logger = logger,
        )
    if not want and engaged:
        if _disengage_step_cache(
            transformer,
            reason = f"auto: {steps} steps < {FBCACHE_MIN_STEPS}",
            logger = logger,
        ):
            return None
        return mode
    return mode if engaged else None


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.cache: %s unavailable (%s); running uncached", what, exc)
