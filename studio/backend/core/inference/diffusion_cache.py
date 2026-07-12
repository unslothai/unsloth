# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in step caching for the diffusion transformer (First-Block-Cache).

Once a DiT's trajectory settles its output changes little across steps, so most of the
transformer can be reused. FBCache computes the first block and, if its residual barely changed
from the previous step (within ``threshold``), skips the rest and reuses their cached output.
diffusers ships it natively (``transformer.enable_cache(FirstBlockCacheConfig(...))``).

Measured on Flux.1-dev (28 steps, 1024px, B200): ~1.4x on top of torch.compile (2.83 -> 2.03 s)
at LPIPS ~0.08 -- deep inside the speed-for-quality bar.

OFF by default: the win scales with step count, so a few-step distilled model (Z-Image-Turbo
~8 steps) has almost no headroom and caching is for many-step models (Flux / Qwen-Image). It
composes with torch.compile only at ``fullgraph=False`` (the cache's compiler-disabled decision
is a graph break), which the speed layer switches to automatically. Best-effort: an incompatible
model is caught and the load proceeds uncached. torch / diffusers imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional

TC_OFF = "off"
TC_AUTO = "auto"
TC_FBCACHE = "fbcache"
TC_MODES = (TC_FBCACHE,)

# FBCache residual thresholds: higher skips more steps (faster, lower quality). Quantised
# transformers shift the residual distribution, so they need a higher threshold to trigger at
# all (per ParaAttention's fp8 guidance).
DEFAULT_FBCACHE_THRESHOLD = 0.08
QUANT_FBCACHE_THRESHOLD = 0.12

# Auto step-count bar: FBCache's win scales with step count, so auto engages only at 20+ steps
# ("dev" schedules 28+ qualify, distilled turbo 4-9 never do).
FBCACHE_MIN_STEPS = 20


def normalize_transformer_cache(value: Optional[str]) -> Optional[str]:
    """Lower/strip a cache mode; None / "" / "none" / "off" -> None, "auto" -> TC_AUTO (loader
    decides from step count). Raises ValueError for an unsupported value."""
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


def _invalidate_child_registry_cache(transformer: Any) -> None:
    """Drop the HookRegistry's cached child-registry list after (un)installing hooks.

    ``cache_context`` propagates state through ``_get_child_registries``, which diffusers 0.39
    caches on first use. An uncached generation already calls it, creating an EMPTY cached child
    list -- so a later ``enable_cache`` installs block hooks ``_set_context`` never reaches and the
    first cached forward dies with "No context is set". Invalidate so the next ``cache_context``
    rebuilds it over the freshly hooked blocks. Best-effort."""
    registry = getattr(transformer, "_diffusers_hook", None)
    if registry is not None and getattr(registry, "_child_registries_cache", None) is not None:
        try:
            registry._child_registries_cache = None
        except Exception:  # noqa: BLE001 -- diffusers internals moved; leave as-is
            pass


# diffusers cache hook names whose compute branch we re-point at a compiled inner forward
# (leader = measuring first block, block = the rest); both share the fn_ref layout.
_CACHE_HOOK_NAMES = (
    "mag_cache_leader_block_hook",
    "mag_cache_block_hook",
    "fbc_leader_block_hook",
    "fbc_block_hook",
)


def _compile_hooked_block_inners(transformer: Any, logger: Any = None) -> int:
    """Restore the regional compile on cache-hooked blocks' COMPUTED steps.

    ``enable_cache`` replaces each block's ``forward`` with the hook's ``new_forward`` (stashing
    the bound method in ``fn_ref.original_forward``), whose skip decision is data-dependent Python:
    MagCache ``@torch.compiler.disable``s the whole thing (compute runs EAGER), and even FBCache's
    traceable ``new_forward`` graph-breaks around its disabled decision, which on some archs
    (Qwen-Image) drops the compute call out of the compiled region -- so ``_compiled_call_impl`` is
    never reached and the cache forfeits the compile win on every computed step. A ``torch.compile``d
    callable re-enables dynamo for its own extent even inside a disabled frame, so re-pointing
    ``original_forward`` at a compiled wrapper restores compiled compute steps while the skip
    decision stays eager. Measured: Qwen-Image FBCache computed steps 91.8 -> 71.2 ms (uncached
    compiled rate), 1.21x end-to-end; FLUX.1-dev neutral (its new_forward traces); video DiT
    MagCache 39.4 -> 26.9 s at 50 steps.

    Only speed-layer-compiled blocks are armed (``_compiled_call_impl`` guard) and only when
    ``original_forward`` is a plain bound method (a stacked hook chain is skipped). Idempotent via
    ``_unsloth_orig_inner``; best-effort. Returns the number armed."""
    try:
        import torch
    except Exception:  # noqa: BLE001 -- no torch, nothing to arm
        return 0
    armed = 0
    try:
        for module in transformer.modules():
            registry = getattr(module, "_diffusers_hook", None)
            if registry is None or getattr(module, "_compiled_call_impl", None) is None:
                continue
            hooks = getattr(registry, "hooks", None) or {}
            for name in _CACHE_HOOK_NAMES:
                hook = hooks.get(name)
                fn_ref = getattr(hook, "fn_ref", None) if hook is not None else None
                orig = getattr(fn_ref, "original_forward", None)
                if orig is None or getattr(hook, "_unsloth_orig_inner", None) is not None:
                    continue
                if getattr(orig, "__self__", None) is None:
                    continue  # not a plain bound method; arming would miss the block
                # fullgraph=False / dynamic=True: a cache is active (its decision graph-breaks) and
                # this matches the default tier. Dynamo caches per code object, so re-arming after
                # a toggle is ~free (~0.03 s).
                fn_ref.original_forward = torch.compile(orig, fullgraph = False, dynamic = True)
                hook._unsloth_orig_inner = orig
                armed += 1
    except Exception as exc:  # noqa: BLE001 -- best-effort: the cache still works eager
        _warn(logger, "cache-hook inner compile", exc)
        return armed
    if armed and logger is not None:
        logger.info(
            "diffusion.cache: %d cache-hooked block(s) armed with compiled inner forwards",
            armed,
        )
    return armed


def _restore_hooked_block_inners(transformer: Any) -> None:
    """Undo ``_compile_hooked_block_inners``: restore the bound methods and clear the markers.
    MUST run before ``disable_cache`` -- ``remove_hook`` splices ``original_forward`` back into
    ``module.forward``, so a leftover compiled wrapper would pin a stale callable on the uncached
    path."""
    try:
        modules = list(transformer.modules())
    except Exception:  # noqa: BLE001 -- not a torch module (tests/fakes): nothing armed
        return
    for module in modules:
        registry = getattr(module, "_diffusers_hook", None)
        if registry is None:
            continue
        hooks = getattr(registry, "hooks", None) or {}
        for name in _CACHE_HOOK_NAMES:
            hook = hooks.get(name)
            orig = getattr(hook, "_unsloth_orig_inner", None) if hook is not None else None
            if orig is None:
                continue
            try:
                hook.fn_ref.original_forward = orig
                hook._unsloth_orig_inner = None
            except Exception:  # noqa: BLE001 -- per-hook best-effort
                pass


def _pipeline_opens_cache_context(pipe: Any) -> bool:
    """Whether the pipeline enters ``transformer.cache_context(...)`` in its denoise loop. The
    FBCache hook requires it at run time, and a CacheMixin transformer alone doesn't guarantee it
    (Flux Kontext / img2img / inpaint / controlnet reuse FluxTransformer2DModel but open none).
    Read from ``__call__`` source; False when unreadable so the cache stays off."""
    import inspect

    call = getattr(pipe, "__call__", None)
    if call is None:
        return False
    try:
        src = inspect.getsource(call)
    except (OSError, TypeError):
        return False
    # Match the call `cache_context(` (the paren avoids a false positive on prose).
    return "cache_context(" in src


def apply_step_cache(
    pipe: Any,
    *,
    mode: Optional[str],
    threshold: Optional[float] = None,
    quant_active: bool = False,
    logger: Any = None,
) -> Optional[str]:
    """Engage step caching on ``pipe.transformer``. Returns the mode engaged, or None when
    disabled / unsupported (runs uncached). ``threshold`` overrides the default; ``quant_active``
    raises it so the cache triggers on a quantised transformer. Best-effort."""
    mode = normalize_transformer_cache(mode)
    if mode is None or mode == TC_AUTO:
        # AUTO is resolved by the loader before this; treat a stray auto as off.
        return None
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    thr = (
        threshold
        if threshold is not None
        else (QUANT_FBCACHE_THRESHOLD if quant_active else DEFAULT_FBCACHE_THRESHOLD)
    )
    # Engage only via the native enable_cache (CacheMixin path): the lower-level
    # apply_first_block_cache hook would also install on a non-CacheMixin transformer (e.g.
    # Z-Image) whose pipeline opens no cache_context and crashes generation. So a model without
    # enable_cache runs uncached instead of being reported cached then failing.
    enable_cache = getattr(transformer, "enable_cache", None)
    if not callable(enable_cache):
        _warn(logger, mode, RuntimeError("transformer has no cache_context (not a CacheMixin)"))
        return None
    # A CacheMixin transformer is necessary but not sufficient: the hook raises "No context is set"
    # unless the PIPELINE wraps its denoise loop in cache_context(...). Flux Kontext / img2img /
    # inpaint / controlnet reuse FluxTransformer2DModel yet open none, so run uncached instead.
    if not _pipeline_opens_cache_context(pipe):
        _warn(
            logger, mode, RuntimeError("pipeline __call__ opens no cache_context; running uncached")
        )
        return None
    try:
        try:
            from diffusers import FirstBlockCacheConfig
        except ImportError:  # older diffusers exports it only from diffusers.hooks
            from diffusers.hooks import FirstBlockCacheConfig

        config = FirstBlockCacheConfig(threshold = thr)
        enable_cache(config)
        # enable_cache after the pipe ran leaves a stale cached child-registry list; the new block
        # hooks would never receive the cache context. Must follow every enable_cache.
        _invalidate_child_registry_cache(transformer)
        # If blocks are already regionally compiled (toggle path: compile ran at load), re-point
        # the fresh hooks' compute branch at compiled inners; the load path is armed by
        # _compile_repeated_blocks. No-op when nothing is compiled.
        _compile_hooked_block_inners(transformer, logger)
        try:
            transformer._unsloth_step_cache = f"{mode}@{thr}"
        except Exception:  # noqa: BLE001 — marker is best-effort
            pass
        if logger is not None:
            logger.info("diffusion.cache: %s engaged (threshold=%s)", mode, thr)
        return mode
    except Exception as exc:  # noqa: BLE001 — incompatible model -> run uncached
        # enable_cache can fail after hooking some blocks; drop partial hooks so the
        # reported-uncached model isn't half-cached. Restore armed compiled inners FIRST
        # (remove_hook splices original_forward back into module.forward).
        _restore_hooked_block_inners(transformer)
        try:
            transformer.disable_cache()
        except Exception:  # noqa: BLE001
            pass
        _warn(logger, mode, exc)
        return None


def effective_denoise_steps(steps: int, strength: Optional[float]) -> int:
    """The number of steps diffusers ACTUALLY denoises for a request.

    An image-conditioned workflow with ``strength`` < 1 (img2img / upscale / inpaint) denoises
    only ``init_timestep = min(int(num_inference_steps * strength), num_inference_steps)`` steps
    -- FLOORED, not rounded. The auto step-cache policy keys on THIS count (e.g. a 28-step upscale
    at strength 0.35 runs int(9.8) = 9 steps, the short trajectory FBCache should stay off).
    ``strength`` None or >= 1 -> the full count.
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

    Only image-conditioned pipelines taking ``strength`` apply it (else full trajectory -> None).
    When the request omits it the loader doesn't pass the kwarg, so the pipe uses its OWN signature
    default (< 1 for every img2img / inpaint pipeline, e.g. 0.6); the policy keys on that default,
    else FBCache engages on a fraction of the advertised steps. A non-numeric default -> None.
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
    """Generation-time enable/disable for an AUTO cache decision, keyed on the step count: engage
    FBCache at ``FBCACHE_MIN_STEPS``+, else uncached. Idempotent (``_unsloth_step_cache`` marker),
    so per-generation calls are cheap. Only the auto path calls this. Returns the active mode."""
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
                # Restore before remove_hook splices original_forward back, so compiled
                # wrappers don't leak onto the uncached path.
                _restore_hooked_block_inners(transformer)
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
