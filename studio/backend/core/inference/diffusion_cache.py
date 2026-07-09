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
TC_MODES = (TC_FBCACHE,)

# FBCache residual thresholds: higher skips more steps (faster, lower quality). The dense
# bf16 default; a quantised transformer shifts the residual distribution, so it needs a
# higher threshold for the cache to trigger at all (per ParaAttention's fp8 guidance).
DEFAULT_FBCACHE_THRESHOLD = 0.08
QUANT_FBCACHE_THRESHOLD = 0.12

# The auto policy's step-count bar: FBCache's win scales with step count (each skipped
# step is a larger quality hit on a short trajectory), so auto engages it only at 20+
# steps -- full "dev"-style schedules (28+) qualify, distilled turbo models (4-9) never do.
FBCACHE_MIN_STEPS = 20


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
    logger: Any = None,
) -> Optional[str]:
    """Engage step caching on ``pipe.transformer``. Returns the mode actually engaged, or
    None when disabled / unsupported (the load then runs uncached). ``threshold`` overrides
    the default; ``quant_active`` raises the default so the cache still triggers on a
    quantised transformer. Best-effort: never raises for an incompatible model."""
    mode = normalize_transformer_cache(mode)
    if mode is None or mode == TC_AUTO:
        # AUTO must be resolved by the loader (step-count policy) before reaching the
        # engage call; treat a stray auto as off rather than crashing the load.
        return None
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
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
        _warn(logger, mode, RuntimeError("pipeline __call__ opens no cache_context; running uncached"))
        return None
    try:
        try:
            from diffusers import FirstBlockCacheConfig
        except ImportError:  # older diffusers exports it only from diffusers.hooks
            from diffusers.hooks import FirstBlockCacheConfig

        config = FirstBlockCacheConfig(threshold = thr)
        enable_cache(config)
        try:
            transformer._unsloth_step_cache = f"{mode}@{thr}"
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


def maybe_toggle_step_cache(
    pipe: Any,
    *,
    steps: int,
    quant_active: bool = False,
    threshold: Optional[float] = None,
    logger: Any = None,
) -> Optional[str]:
    """Generation-time enable/disable for an AUTO cache decision, keyed on the actual
    step count: engage FBCache at ``FBCACHE_MIN_STEPS`` or more, run uncached below it.
    Idempotent (the ``_unsloth_step_cache`` marker tracks the engaged state), so calling
    it on every generation is cheap. Only the loader's auto path calls this; an explicit
    user choice is never toggled. Returns the mode now active (or None when uncached)."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    engaged = getattr(transformer, "_unsloth_step_cache", None)
    want = int(steps) >= FBCACHE_MIN_STEPS
    if want and not engaged:
        return apply_step_cache(
            pipe,
            mode = TC_FBCACHE,
            threshold = threshold,
            quant_active = quant_active,
            logger = logger,
        )
    if not want and engaged:
        disable_cache = getattr(transformer, "disable_cache", None)
        if callable(disable_cache):
            try:
                disable_cache()
                transformer._unsloth_step_cache = None
                if logger is not None:
                    logger.info(
                        "diffusion.cache: fbcache disengaged (auto: %s steps < %s)",
                        steps,
                        FBCACHE_MIN_STEPS,
                    )
                return None
            except Exception as exc:  # noqa: BLE001 — keep the cache rather than crash
                _warn(logger, "fbcache disable", exc)
        return TC_FBCACHE
    return TC_FBCACHE if engaged else None


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.cache: %s unavailable (%s); running uncached", what, exc)
