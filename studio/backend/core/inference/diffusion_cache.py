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
TC_FBCACHE = "fbcache"
TC_MODES = (TC_FBCACHE,)

# FBCache residual thresholds: higher skips more steps (faster, lower quality). The dense
# bf16 default; a quantised transformer shifts the residual distribution, so it needs a
# higher threshold for the cache to trigger at all (per ParaAttention's fp8 guidance).
DEFAULT_FBCACHE_THRESHOLD = 0.08
QUANT_FBCACHE_THRESHOLD = 0.12


def normalize_transformer_cache(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested cache mode; None / "" / "none" / "off" -> None (disabled).

    Raises ValueError for an unsupported value so a bad request is rejected cheaply."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized in ("none", "off"):
        return None
    if normalized not in TC_MODES:
        raise ValueError(
            f"Unsupported transformer_cache '{value}'. Use one of: off, {', '.join(TC_MODES)}."
        )
    return normalized


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
    if mode is None:
        return None
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    thr = (
        threshold
        if threshold is not None
        else (QUANT_FBCACHE_THRESHOLD if quant_active else DEFAULT_FBCACHE_THRESHOLD)
    )
    # Only engage via the transformer's native enable_cache (the diffusers CacheMixin path).
    # That mixin is present exactly when the pipeline wraps the transformer call in a
    # cache_context, which the First-Block-Cache hook requires at run time. The lower-level
    # apply_first_block_cache hook would install on a non-CacheMixin transformer too (e.g.
    # Z-Image), but its pipeline opens no cache_context, so the first generation would crash
    # inside the hook -- so a model without enable_cache runs uncached per the best-effort
    # contract instead of being reported as cached and then failing.
    enable_cache = getattr(transformer, "enable_cache", None)
    if not callable(enable_cache):
        _warn(logger, mode, RuntimeError("transformer has no cache_context (not a CacheMixin)"))
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


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.cache: %s unavailable (%s); running uncached", what, exc)
