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

# FBCache residual thresholds: higher skips more steps (faster, lower quality). Quantised transformers shift the
# residual distribution, so they need a higher threshold.
DEFAULT_FBCACHE_THRESHOLD = 0.08
QUANT_FBCACHE_THRESHOLD = 0.12

# Auto step-count bar: FBCache's win scales with step count, so auto engages only at 20+ steps ("dev" schedules
# qualify, distilled turbo never does).
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


# Block classes diffusers ships without first-block-cache metadata, and the metadata they want.
# The pair is (index of hidden_states in the block's return, index of encoder_hidden_states or None).
#
# Qwen-Image-2.1 is single stream: its block takes ``hidden_states, modulation, rotary_emb, ...``
# and returns the hidden states alone, so 0 / None. Without the entry ``enable_cache`` raises
# "Model class QwenImage21TransformerBlock not registered." and every load of the family renders
# uncached, which is the whole step-cache saving gone on a 20+ step model, silently.
_UNREGISTERED_BLOCK_METADATA: dict = {
    (
        "diffusers.models.transformers.transformer_qwenimage21",
        "QwenImage21TransformerBlock",
    ): (0, None),
}


def register_unregistered_transformer_blocks(logger: Any = None) -> tuple:
    """Add our own first-block-cache metadata for block classes diffusers has not registered.

    Idempotent, best-effort, and never overwrites: a class diffusers registers later wins, since
    upstream's own metadata is authoritative and ours exists only to fill the gap until it lands.
    An import failure means that diffusers does not have the class at all, which is not an error
    here; the family simply is not installed.
    """
    added: list = []
    try:
        from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
    except Exception:  # noqa: BLE001 - an older diffusers has no registry to fill
        return ()
    for (module_name, class_name), (
        hidden_index,
        encoder_index,
    ) in _UNREGISTERED_BLOCK_METADATA.items():
        try:
            import importlib
            block_cls = getattr(importlib.import_module(module_name), class_name, None)
        except Exception:  # noqa: BLE001 - this diffusers does not ship the family
            continue
        if block_cls is None:
            continue
        try:
            TransformerBlockRegistry.get(block_cls)
            continue  # already known, ours would be a downgrade
        except Exception:  # noqa: BLE001 - "not registered" is the case we are here for
            pass
        try:
            TransformerBlockRegistry.register(
                model_class = block_cls,
                metadata = TransformerBlockMetadata(
                    return_hidden_states_index = hidden_index,
                    return_encoder_hidden_states_index = encoder_index,
                ),
            )
            added.append(class_name)
        except Exception as exc:  # noqa: BLE001 - registration is an optimisation, never a gate
            if logger is not None:
                logger.debug("could not register %s for step caching: %s", class_name, exc)
    return tuple(added)


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


# diffusers cache hook names whose compute branch we re-point at a compiled inner forward (leader = measuring first
# block, block = the rest); both share the fn_ref layout.
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
                # fullgraph=False / dynamic=True: a cache is active (its decision graph-breaks) and this matches the
                # default tier. Dynamo caches per code object, so re-arming after a toggle is ~free.
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


def _unhook_first_block_cache(transformer: Any) -> bool:
    """Take the First-Block-Cache hooks off directly, and say whether they are KNOWN to be gone.

    diffusers cannot be asked: ``enable_cache`` sets ``_cache_config`` only after
    ``apply_first_block_cache`` returns and ``disable_cache`` warns and returns when it is None, so a
    raise part-way through hooking leaves hooks live behind ``is_cache_enabled is False``.
    ``remove_hook`` skips unregistered names and recurses, so it is safe whatever got installed.
    False means only "cannot verify": the caller must keep the marker, not assume uncached.
    """
    # `_cache_config` is generic across every cache CacheMixin supports, so a live MagCache / PAB
    # reaches here looking like a broken FBC one; tearing it down is not ours to do. Matched by name
    # because the config class is exported from two modules depending on the diffusers version.
    config = getattr(transformer, "_cache_config", None)
    if config is not None and type(config).__name__ != "FirstBlockCacheConfig":
        return False
    try:
        from diffusers.hooks import HookRegistry
        from diffusers.hooks.first_block_cache import _FBC_BLOCK_HOOK, _FBC_LEADER_BLOCK_HOOK
    except Exception:  # noqa: BLE001 - private names; an unknown layout means we cannot verify
        return False
    try:
        registry = HookRegistry.check_if_exists_or_initialize(transformer)
        registry.remove_hook(_FBC_LEADER_BLOCK_HOOK, recurse = True)
        registry.remove_hook(_FBC_BLOCK_HOOK, recurse = True)
        # Left dangling by a partial engage, and it is what diffusers keys every later call on.
        transformer._cache_config = None
    except Exception:  # noqa: BLE001
        return False
    return True


def _first_block_cache_is_hooked(transformer: Any) -> bool:
    """Whether FBCache's hook names are registered ANYWHERE under *transformer* right now.

    Asked BEFORE an engage, so a failure afterwards can tell our own half-finished install from
    someone else's working cache: the public ``apply_first_block_cache`` hooks without setting
    ``_cache_config``, so our ``enable_cache`` raises on the duplicate name having changed nothing.
    """
    try:
        from diffusers.hooks.first_block_cache import _FBC_BLOCK_HOOK, _FBC_LEADER_BLOCK_HOOK
        names = (_FBC_LEADER_BLOCK_HOOK, _FBC_BLOCK_HOOK)
        for module in transformer.modules():
            registry = getattr(module, "_diffusers_hook", None)
            hooks = getattr(registry, "hooks", None) or {} if registry is not None else {}
            if any(name in hooks for name in names):
                return True
    except Exception:  # noqa: BLE001 - private names, or not a torch module: cannot tell
        return False
    return False


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


def _reuses_prefix_kv(pipe: Any, transformer: Any) -> bool:
    """Whether the denoise loop feeds the blocks a SHORTER sequence after the first step.

    A transformer that caches the prompt/condition prefix K and V runs its first step over the
    whole joint sequence and every later step over the target tokens alone, so the per-block
    sequence length changes between step 0 and step 1. FBCache compares the first block's residual
    against the previous step's and reuses the remaining blocks' cached residual, and both are
    plain elementwise ops on a stored tensor, so the length change makes them raise:

        RuntimeError: The size of tensor a (4096) must match the size of tensor b (4297)
                      at non-singleton dimension 1

    (Qwen-Image-2.1 at 1024px: 4096 image tokens against 4096 + 201 prompt tokens.) The cache is
    not merely unsupported here, it takes the generation down at the second step, so refuse it.

    Read structurally rather than by family name, because the shape is shared: the transformer's
    forward accepts a ``kv_cache_mode`` and the pipeline passes one. Qwen-Image-2.1, FLUX.2 klein
    KV and Wan-Animate-2 all match today, and a family that adopts prefix reuse later is covered
    without touching this file.

    Conservative on purpose. A checkpoint that carries the parameter but never populates the cache
    (the pipeline gates on its own config) keeps a constant length and would have been safe, and it
    loses the cache anyway. That costs speed on a model we have not seen; guessing the other way
    costs a failed render on one we have."""
    import inspect

    forward = getattr(transformer, "forward", None)
    if forward is None:
        return False
    try:
        if "kv_cache_mode" not in inspect.signature(forward).parameters:
            return False
    except (TypeError, ValueError):  # not introspectable: assume no prefix reuse
        return False
    call = getattr(pipe, "__call__", None)
    if call is None:
        return False
    try:
        src = inspect.getsource(call)
    except (OSError, TypeError):
        # The transformer takes the parameter and the loop cannot be read, so whether it is driven
        # is unknown. Unknown is the crashing side here, so treat it as driven.
        return True
    return "kv_cache_mode" in src


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
    # Before enable_cache, which is what raises on a block class the registry has never seen.
    register_unregistered_transformer_blocks(logger)
    thr = (
        threshold
        if threshold is not None
        else (QUANT_FBCACHE_THRESHOLD if quant_active else DEFAULT_FBCACHE_THRESHOLD)
    )
    # Engage only via the native enable_cache (CacheMixin path): the lower-level apply_first_block_cache hook would
    # also install on a non-CacheMixin transformer whose pipeline opens no cache_context, and crash generation.
    enable_cache = getattr(transformer, "enable_cache", None)
    if not callable(enable_cache):
        _warn(logger, mode, RuntimeError("transformer has no cache_context (not a CacheMixin)"))
        return None
    # CacheMixin is necessary but not sufficient: the hook raises "No context is set" unless the
    # PIPELINE opens cache_context(...). Flux Kontext / img2img / inpaint / controlnet open none.
    if not _pipeline_opens_cache_context(pipe):
        _warn(
            logger, mode, RuntimeError("pipeline __call__ opens no cache_context; running uncached")
        )
        return None
    # Prefix KV reuse shortens the block sequence after the first step, which the cache's stored
    # residuals cannot be subtracted from. Checked before enable_cache: engaging here does not fail
    # at load, it fails at step 2 of the user's generation.
    if _reuses_prefix_kv(pipe, transformer):
        _warn(
            logger,
            mode,
            RuntimeError(
                "transformer reuses a prefix KV cache, so the block sequence length changes "
                "after the first step; running uncached"
            ),
        )
        return None
    # enable_cache RAISES when is_cache_enabled, so without this a redundant call lands in the
    # recovery branch below and loses the cache. Different settings must disable first.
    if getattr(transformer, "is_cache_enabled", False):
        prior = getattr(transformer, "_unsloth_step_cache", None)
        live = getattr(transformer, "_cache_config", None)
        # ONLY the live config authorises the no-op, never the marker: a transformer reconfigured
        # elsewhere keeps whatever marker we last wrote, so honouring it would report settings the
        # model is not running.
        if (
            type(live).__name__ == "FirstBlockCacheConfig"
            and getattr(live, "threshold", None) == thr
        ):
            # May not be a cache we installed, so the post-enable integration cannot be assumed
            # done. Both idempotent; skipping the first strands the child-registry list.
            _invalidate_child_registry_cache(transformer)
            _compile_hooked_block_inners(transformer, logger)
            try:
                transformer._unsloth_step_cache = f"{mode}@{thr}"
            except Exception:  # noqa: BLE001 - marker is best-effort
                pass
            return mode
        try:
            _restore_hooked_block_inners(transformer)
            transformer.disable_cache()
        except Exception as exc:  # noqa: BLE001 - cannot re-configure -> finish the teardown ourselves
            # disable_cache removes leader and block hooks in SEPARATE calls and clears
            # _cache_config only after both, so a raise between them leaves block hooks with no
            # leader while is_cache_enabled still reads True. Finish the removal by name.
            removed = _unhook_first_block_cache(transformer)
            if not removed:
                # Cannot verify, so report what was last known engaged and leave the marker alone.
                _warn(logger, mode, exc)
                return prior.split("@")[0] if isinstance(prior, str) else None
            # Fully unhooked, so drop the stale marker and engage at the new settings.
            try:
                transformer._unsloth_step_cache = None
            except Exception:  # noqa: BLE001 - marker is best-effort
                pass
    # Asked BEFORE the engage: afterwards our own partial install and someone else's finished one
    # look identical. Outside the try so the recovery below can always read it.
    hooked_before = _first_block_cache_is_hooked(transformer)
    try:
        try:
            from diffusers import FirstBlockCacheConfig
        except ImportError:  # older diffusers exports it only from diffusers.hooks
            from diffusers.hooks import FirstBlockCacheConfig

        config = FirstBlockCacheConfig(threshold = thr)
        enable_cache(config)
        # enable_cache after the pipe ran leaves a stale cached child-registry list, so the new block hooks would never
        # receive the cache context. Must follow every enable_cache.
        _invalidate_child_registry_cache(transformer)
        # If blocks are already regionally compiled (toggle path), re-point the fresh hooks' compute branch at
        # compiled inners; the load path is armed by _compile_repeated_blocks.
        _compile_hooked_block_inners(transformer, logger)
        try:
            transformer._unsloth_step_cache = f"{mode}@{thr}"
        except Exception:  # noqa: BLE001 - marker is best-effort
            pass
        if logger is not None:
            logger.info("diffusion.cache: %s engaged (threshold=%s)", mode, thr)
        return mode
    except Exception as exc:  # noqa: BLE001 - incompatible model -> run uncached
        if hooked_before:
            # FBCache installed through the low-level apply_first_block_cache, which is why
            # _cache_config was None. register_hook refuses a duplicate name BEFORE changing
            # anything, so those hooks are intact and tearing them down would cost a healthy cache.
            _invalidate_child_registry_cache(transformer)
            _compile_hooked_block_inners(transformer, logger)
            # The marker MUST be set before reporting success: it is what keeps the CUDA graph
            # wrapper eager over these live hooks. Mode only; the threshold is its installer's.
            try:
                transformer._unsloth_step_cache = mode
            except Exception:  # noqa: BLE001 - marker is best-effort
                pass
            _warn(logger, mode, exc)
            return mode
        # enable_cache can fail part-hooked; restore armed compiled inners FIRST
        _restore_hooked_block_inners(transformer)
        disabled = True
        try:
            transformer.disable_cache()
        except Exception:  # noqa: BLE001 - _unhook_first_block_cache below is what actually decides
            disabled = False
        # Neither `disabled` nor is_cache_enabled can be trusted here: enable_cache assigns
        # _cache_config LAST, so a raise part-way through hooking has both say "not caching".
        del disabled
        removed = _unhook_first_block_cache(transformer)
        # The marker tracks the HOOKS, not intent, so it comes off only once they are KNOWN gone:
        # left set it only forces eager, cleared over live hooks it permits a captured graph.
        if removed:
            try:
                transformer._unsloth_step_cache = None
            except Exception:  # noqa: BLE001 - marker is best-effort
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
            # Read BEFORE the teardown: it is what says whether diffusers' own disable_cache can
            # do the removal, and disable_cache clears it.
            ours = (
                type(getattr(transformer, "_cache_config", None)).__name__
                == "FirstBlockCacheConfig"
            )
            try:
                # Restore before remove_hook splices original_forward back, so compiled wrappers do not leak onto the
                # uncached path.
                _restore_hooked_block_inners(transformer)
                disable_cache()
                # disable_cache removes nothing when _cache_config is None, which is exactly an
                # adopted low-level cache, so trusting it would clear the marker over live hooks.
                # THAT is the case the verification exists for, so it only gets a veto there. A
                # live FirstBlockCacheConfig means diffusers just removed both hooks and cleared
                # the config itself, and disable_cache returning without raising is the evidence;
                # demanding the private-name sweep succeed on top of it makes every disengage
                # depend on `diffusers.hooks.first_block_cache` staying importable, and when it is
                # not, FBCache stays engaged on short trajectories forever -- the exact quality
                # regression the auto policy exists to avoid.
                if not _unhook_first_block_cache(transformer) and not ours:
                    _warn(
                        logger,
                        "fbcache disable",
                        RuntimeError("cache hooks could not be verified removed"),
                    )
                    return TC_FBCACHE
                transformer._unsloth_step_cache = None
                if logger is not None:
                    logger.info(
                        "diffusion.cache: fbcache disengaged (auto: %s steps < %s)",
                        steps,
                        FBCACHE_MIN_STEPS,
                    )
                return None
            except Exception as exc:  # noqa: BLE001 - keep the cache rather than crash
                _warn(logger, "fbcache disable", exc)
        return TC_FBCACHE
    return TC_FBCACHE if engaged else None


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.cache: %s unavailable (%s); running uncached", what, exc)
