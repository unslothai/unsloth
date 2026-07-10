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


# diffusers' cache hook registry names whose compute branch we re-point at a compiled
# inner forward (leader = the measuring first block, block = the remaining ones); both
# hook families share the fn_ref layout.
_CACHE_HOOK_NAMES = (
    "mag_cache_leader_block_hook",
    "mag_cache_block_hook",
    "fbc_leader_block_hook",
    "fbc_block_hook",
)


def _compile_hooked_block_inners(transformer: Any, logger: Any = None) -> int:
    """Restore the regional compile on cache-hooked blocks' COMPUTED steps.

    ``enable_cache`` replaces each block's ``forward`` with the hook's ``new_forward``
    (stashing the pre-hook bound method in ``fn_ref.original_forward``), whose skip
    decision is data-dependent Python: MagCache ``@torch.compiler.disable``s the whole
    ``new_forward`` (recursive -- the compute branch runs EAGER), and even FBCache's
    traceable ``new_forward`` graph-breaks around its disabled threshold decision,
    which on some archs (measured: Qwen-Image) drops the compute branch's call into
    ``original_forward`` out of the compiled region -- the block's regional compile
    artifact (``_compiled_call_impl``) is never reached and the cache forfeits the
    compile win on every non-skipped step. An explicitly ``torch.compile``d callable
    re-enables
    dynamo for its own extent even inside a disabled frame, so re-pointing
    ``fn_ref.original_forward`` at a compiled wrapper of the same bound method restores
    compiled compute steps while the skip decision stays eager exactly as designed.
    Measured (B200, scripts/image_speedmem_bench.py): Qwen-Image FBCache computed steps
    91.8 -> 71.2 ms (= the uncached compiled rate), 1.21x end to end; FLUX.1-dev is
    neutral (its FBCache ``new_forward`` happens to trace, so computed steps were
    already compiled -- same-process armed vs unarmed latents bit-identical); on the
    video DiT balanced MagCache went 39.4 -> 26.9 s at 50 steps.

    Only blocks the speed layer actually compiled are armed (``_compiled_call_impl``
    guard -- eager tiers stay untouched), and only when ``original_forward`` is a plain
    bound method (a stacked hook chain, e.g. offload, captures a partial and is
    skipped). Idempotent via the ``_unsloth_orig_inner`` marker; best-effort. Returns
    the number of hooks armed."""
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
                    continue  # not the plain bound method; arming would miss the block
                # fullgraph=False / dynamic=True: a cache is active by definition (its
                # decision points graph-break) and this matches the default tier the
                # regional compile used. Dynamo caches per code object, so re-arming
                # after a toggle is effectively free (~0.03 s).
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
    """Undo ``_compile_hooked_block_inners``: put the plain bound methods back and clear
    the markers. MUST run before ``disable_cache`` -- ``remove_hook`` splices
    ``fn_ref.original_forward`` back into ``module.forward``, and leaving the compiled
    wrapper there would pin a stale compiled callable onto the uncached path."""
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
        # enable_cache AFTER the pipe has already run leaves a stale cached child-registry
        # list on the transformer's HookRegistry; the block hooks just installed would then
        # never receive the cache context. Must follow every enable_cache.
        _invalidate_child_registry_cache(transformer)
        # If the blocks are already regionally compiled (the generation-time toggle
        # path: compile ran at load), re-point the fresh hooks' compute branch at
        # compiled inners; the load path (cache before compile) is armed by
        # _compile_repeated_blocks instead. No-op when nothing is compiled.
        _compile_hooked_block_inners(transformer, logger)
        try:
            transformer._unsloth_step_cache = f"{mode}@{thr}"
        except Exception:  # noqa: BLE001 — marker is best-effort
            pass
        if logger is not None:
            logger.info("diffusion.cache: %s engaged (threshold=%s)", mode, thr)
        return mode
    except Exception as exc:  # noqa: BLE001 — incompatible model -> run uncached
        # enable_cache can fail after hooking some blocks; drop any partial hooks so
        # the reported-uncached model doesn't actually run half-cached. Any armed
        # compiled inners must be restored FIRST (remove_hook splices original_forward
        # back into module.forward).
        _restore_hooked_block_inners(transformer)
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
                # Before remove_hook splices fn_ref.original_forward back into
                # module.forward: the compiled inner wrappers must not leak onto the
                # uncached path.
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
