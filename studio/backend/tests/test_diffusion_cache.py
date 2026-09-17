# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for opt-in step caching (First-Block-Cache).

``diffusers`` is stubbed via ``sys.modules`` (the module under test imports
``FirstBlockCacheConfig`` lazily), and the pipeline is a fake that records the engaged config.
So normalisation, the CacheMixin (``enable_cache``) gating, threshold selection, and the
best-effort failure handling are all exercised without torch or a real diffusers model.
"""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.diffusion_cache import (
    DEFAULT_FBCACHE_THRESHOLD,
    QUANT_FBCACHE_THRESHOLD,
    TC_FBCACHE,
    apply_step_cache,
    normalize_transformer_cache,
)


# ── normalize_transformer_cache ────────────────────────────────────────────────────
def test_normalize_disabled_values_are_none():
    for value in (None, "", "  ", "none", "off", "OFF", "None"):
        assert normalize_transformer_cache(value) is None


def test_normalize_fbcache_and_casing():
    assert normalize_transformer_cache("fbcache") == TC_FBCACHE
    assert normalize_transformer_cache("FBCache") == TC_FBCACHE
    assert normalize_transformer_cache("  fbcache  ") == TC_FBCACHE


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_transformer_cache("deepcache")


# ── apply_step_cache ───────────────────────────────────────────────────────────────
class _Config:
    def __init__(self, threshold):
        self.threshold = threshold


class _MixinTransformer:
    """A CacheMixin-style transformer: exposes ``enable_cache``."""

    def __init__(self, *, fail = False):
        self.fail = fail
        self.enabled_with = None

    def enable_cache(self, config):
        if self.fail:
            raise RuntimeError("block signature not recognised")
        self.enabled_with = config


class _NonCacheMixinTransformer:
    """A transformer with no ``enable_cache`` (not a CacheMixin) -> must run uncached.

    Its pipeline opens no ``cache_context``, so installing FBCache would crash at generation;
    the load runs uncached instead (e.g. Z-Image)."""


class _CtxPipe:
    """A pipeline whose denoise loop opens ``transformer.cache_context(...)`` (like FluxPipeline)
    -- the First-Block-Cache hook needs it, so FBCache may engage here."""

    def __init__(self, transformer):
        self.transformer = transformer

    def __call__(self, *args, **kwargs):
        with self.transformer.cache_context("cond"):
            return None


class _NoCtxPipe:
    """A pipeline that never enters a caching context (like FluxKontextPipeline / img2img /
    inpaint / controlnet, which reuse the CacheMixin FluxTransformer2DModel): FBCache must NOT
    engage or the hook raises "No context is set" on the first forward."""

    def __init__(self, transformer):
        self.transformer = transformer

    def __call__(self, *args, **kwargs):
        return None


def _pipe(transformer):
    return _CtxPipe(transformer)


def _stub_diffusers(monkeypatch, *, hook_recorder = None):
    diffusers = types.ModuleType("diffusers")
    diffusers.FirstBlockCacheConfig = _Config
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)

    hooks = types.ModuleType("diffusers.hooks")

    def _apply_first_block_cache(transformer, config):
        if hook_recorder is not None:
            hook_recorder["transformer"] = transformer
            hook_recorder["config"] = config

    hooks.apply_first_block_cache = _apply_first_block_cache
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)


def test_disabled_mode_is_noop(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer()
    assert apply_step_cache(_pipe(t), mode = None) is None
    assert apply_step_cache(_pipe(t), mode = "off") is None
    assert t.enabled_with is None


def test_enable_cache_path_default_threshold(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer()
    engaged = apply_step_cache(_pipe(t), mode = "fbcache")
    assert engaged == TC_FBCACHE
    assert t.enabled_with.threshold == DEFAULT_FBCACHE_THRESHOLD
    assert t._unsloth_step_cache == f"fbcache@{DEFAULT_FBCACHE_THRESHOLD}"


def test_quant_active_raises_default_threshold(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer()
    apply_step_cache(_pipe(t), mode = "fbcache", quant_active = True)
    assert t.enabled_with.threshold == QUANT_FBCACHE_THRESHOLD


def test_explicit_threshold_overrides_quant(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer()
    apply_step_cache(_pipe(t), mode = "fbcache", threshold = 0.2, quant_active = True)
    assert t.enabled_with.threshold == 0.2


def test_non_cachemixin_runs_uncached(monkeypatch):
    # A transformer without enable_cache (e.g. Z-Image) must NOT get the standalone hook: its pipeline opens no cache_context, so it runs uncached instead of crashing.
    rec: dict = {}
    _stub_diffusers(monkeypatch, hook_recorder = rec)
    t = _NonCacheMixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
    assert rec == {}  # the standalone hook was never called


def test_pipeline_without_cache_context_runs_uncached(monkeypatch):
    # A CacheMixin transformer whose PIPELINE never opens a cache_context (Flux Kontext / img2img / inpaint / controlnet reuse
    # FluxTransformer2DModel) must run uncached, else the First-Block-Cache hook raises "No context is set" on the first forward.
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer()
    assert apply_step_cache(_NoCtxPipe(t), mode = "fbcache") is None
    assert t.enabled_with is None  # enable_cache was never called


def test_incompatible_model_runs_uncached(monkeypatch):
    # enable_cache raising (e.g. unrecognised block signature) must not fail the load.
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer(fail = True)
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None


def test_enable_cache_failure_rolls_back_partial_hooks(monkeypatch):
    # enable_cache can raise after hooking some blocks, so the failure path calls disable_cache rather than run half-cached.
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer(fail = True)
    t.disabled = False
    t.disable_cache = lambda: setattr(t, "disabled", True)
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
    assert t.disabled is True


def test_config_import_falls_back_to_hooks_module(monkeypatch):
    # Older diffusers exports FirstBlockCacheConfig only from diffusers.hooks.
    diffusers = types.ModuleType("diffusers")  # no FirstBlockCacheConfig attribute
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    hooks = types.ModuleType("diffusers.hooks")
    hooks.FirstBlockCacheConfig = _Config
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    t = _MixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "fbcache") == TC_FBCACHE
    assert t.enabled_with.threshold == DEFAULT_FBCACHE_THRESHOLD


def test_missing_transformer_is_none(monkeypatch):
    _stub_diffusers(monkeypatch)
    pipe = types.SimpleNamespace(transformer = None)
    assert apply_step_cache(pipe, mode = "fbcache") is None


def test_diffusers_unavailable_runs_uncached(monkeypatch):
    # No diffusers import: best-effort returns None and the load proceeds uncached. Block the hooks module too, since the
    # config import falls back to it and an earlier real import may have cached it in sys.modules.
    monkeypatch.setitem(sys.modules, "diffusers", None)
    monkeypatch.setitem(sys.modules, "diffusers.hooks", None)
    t = _MixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None


# ── the auto policy: normalize("auto") + generation-time toggling ──────────────────
from core.inference.diffusion_cache import (  # noqa: E402
    FBCACHE_MIN_STEPS,
    TC_AUTO,
    effective_denoise_steps,
    effective_request_strength,
    maybe_toggle_step_cache,
)


# ── effective_denoise_steps (strength-aware step count for the auto policy) ─────────
def test_effective_steps_txt2img_is_full_count():
    # No strength (txt2img / reference) -> the full requested step count.
    assert effective_denoise_steps(28, None) == 28
    assert effective_denoise_steps(28, 1.0) == 28  # full redraw denoises every step


def test_effective_steps_low_strength_shrinks_below_the_bar():
    # A 28-step upscale at strength 0.35 denoises int(9.8) = 9 steps, below FBCACHE_MIN_STEPS, so auto must not engage FBCache.
    eff = effective_denoise_steps(28, 0.35)
    assert eff == 9
    assert eff < FBCACHE_MIN_STEPS


def test_effective_request_strength_uses_pipe_default_when_omitted():
    import inspect

    # txt2img or a pipe without the strength kwarg gives the full trajectory (None).
    assert effective_request_strength(None, False, True, 0.6) is None
    assert effective_request_strength(0.5, True, False, None) is None
    # img2img with an explicit strength -> that value.
    assert effective_request_strength(0.2, True, True, 0.6) == 0.2
    # img2img with an OMITTED strength uses the pipe's signature default, so auto keys on the real trajectory: int(28 * 0.6) = 16 steps.
    s = effective_request_strength(None, True, True, 0.6)
    assert s == 0.6
    assert effective_denoise_steps(28, s) == 16
    # A non-numeric signature default falls back to the full count.
    assert effective_request_strength(None, True, True, inspect.Parameter.empty) is None
    assert effective_request_strength(None, True, True, None) is None


def test_effective_steps_matches_diffusers_get_timesteps():
    # Mirror diffusers exactly: it denoises min(int(steps * strength), steps) steps (floored).
    for steps, strength in [(28, 0.35), (28, 0.8), (50, 0.5), (20, 0.99), (30, 0.1)]:
        expected = max(1, min(int(steps * strength), steps))
        assert effective_denoise_steps(steps, strength) == expected


def test_toggle_stays_off_for_low_strength_workflow(monkeypatch):
    # End to end: a 28-step request would engage FBCache, but at strength 0.35 the effective ~10 steps keep it uncached.
    _stub_diffusers(monkeypatch)
    t = _ToggleTransformer()
    mode = maybe_toggle_step_cache(_pipe(t), steps = effective_denoise_steps(28, 0.35))
    assert mode is None and t.enables == 0


class _ToggleTransformer(_MixinTransformer):
    """CacheMixin-style fake with the disable side too, counting transitions."""

    def __init__(self):
        super().__init__()
        self.enables = 0
        self.disables = 0

    def enable_cache(self, config):
        super().enable_cache(config)
        self.enables += 1

    def disable_cache(self):
        self.disables += 1


def test_normalize_auto_is_a_distinct_state():
    assert normalize_transformer_cache("auto") == TC_AUTO
    assert normalize_transformer_cache(" AUTO ") == TC_AUTO


def test_apply_treats_stray_auto_as_off(monkeypatch):
    # AUTO is resolved by the loader; if it ever reaches the engage call the load runs uncached instead of crashing.
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "auto") is None
    assert t.enabled_with is None


def test_toggle_engages_at_the_step_bar(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _ToggleTransformer()
    mode = maybe_toggle_step_cache(_pipe(t), steps = FBCACHE_MIN_STEPS)
    assert mode == TC_FBCACHE and t.enables == 1
    assert t.enabled_with.threshold == DEFAULT_FBCACHE_THRESHOLD
    assert t._unsloth_step_cache


def test_toggle_uses_quant_threshold(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _ToggleTransformer()
    maybe_toggle_step_cache(_pipe(t), steps = 28, quant_active = True)
    assert t.enabled_with.threshold == QUANT_FBCACHE_THRESHOLD


def test_toggle_is_idempotent_when_engaged(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _ToggleTransformer()
    maybe_toggle_step_cache(_pipe(t), steps = 28)
    mode = maybe_toggle_step_cache(_pipe(t), steps = 28)
    assert mode == TC_FBCACHE and t.enables == 1 and t.disables == 0


def test_toggle_disengages_below_the_bar(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _ToggleTransformer()
    maybe_toggle_step_cache(_pipe(t), steps = 28)
    mode = maybe_toggle_step_cache(_pipe(t), steps = 8)
    assert mode is None and t.disables == 1
    assert not t._unsloth_step_cache
    # and it stays off on repeat calls (no flapping disable calls).
    assert maybe_toggle_step_cache(_pipe(t), steps = 8) is None
    assert t.disables == 1


def test_toggle_reengages_after_a_disable(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _ToggleTransformer()
    maybe_toggle_step_cache(_pipe(t), steps = 28)
    maybe_toggle_step_cache(_pipe(t), steps = 8)
    mode = maybe_toggle_step_cache(_pipe(t), steps = 24)
    assert mode == TC_FBCACHE and t.enables == 2


def test_toggle_noop_without_cache_support(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _NonCacheMixinTransformer()
    assert maybe_toggle_step_cache(_pipe(t), steps = 28) is None
    assert maybe_toggle_step_cache(_pipe(t), steps = 8) is None


def test_toggle_noop_without_transformer():
    assert maybe_toggle_step_cache(types.SimpleNamespace(), steps = 28) is None


# ── compiled cache-hook inners (regional compile x step cache composition) ──────────
import functools  # noqa: E402

from core.inference.diffusion_cache import (  # noqa: E402
    _compile_hooked_block_inners,
    _invalidate_child_registry_cache,
    _restore_hooked_block_inners,
)


class _BoundInner:
    """Provides a plain bound method for fn_ref.original_forward (__self__ present)."""

    def forward(self, *args, **kwargs):
        return "eager"


def _hooked_block(
    *,
    compiled = True,
    hook_name = "fbc_block_hook",
    bound = True,
):
    inner = _BoundInner()
    orig = inner.forward if bound else functools.partial(_BoundInner.forward, inner)
    hook = types.SimpleNamespace(fn_ref = types.SimpleNamespace(original_forward = orig))
    block = types.SimpleNamespace(
        _diffusers_hook = types.SimpleNamespace(hooks = {hook_name: hook}),
        _compiled_call_impl = object() if compiled else None,
    )
    return block, hook, orig


def _fake_dit(blocks):
    return types.SimpleNamespace(modules = lambda: [types.SimpleNamespace()] + blocks)


def _stub_torch_compile(monkeypatch):
    compiled_calls = []

    def _compile(fn, **kwargs):
        compiled_calls.append((fn, kwargs))
        wrapper = lambda *a, **k: fn(*a, **k)  # noqa: E731
        wrapper._unsloth_test_compiled_of = fn
        return wrapper

    torch = types.ModuleType("torch")
    torch.compile = _compile
    monkeypatch.setitem(sys.modules, "torch", torch)
    return compiled_calls


def test_arming_swaps_inner_for_compiled_wrapper(monkeypatch):
    calls = _stub_torch_compile(monkeypatch)
    block, hook, orig = _hooked_block()
    assert _compile_hooked_block_inners(_fake_dit([block])) == 1
    assert hook.fn_ref.original_forward is not orig
    assert hook.fn_ref.original_forward._unsloth_test_compiled_of is orig
    assert hook._unsloth_orig_inner is orig
    # The inner compile must match the cache-active tier: graph-breakable + dynamic.
    assert calls[0][1] == {"fullgraph": False, "dynamic": True}


def test_arming_is_idempotent(monkeypatch):
    _stub_torch_compile(monkeypatch)
    block, hook, _ = _hooked_block()
    dit = _fake_dit([block])
    assert _compile_hooked_block_inners(dit) == 1
    once = hook.fn_ref.original_forward
    assert _compile_hooked_block_inners(dit) == 0  # marker short-circuits
    assert hook.fn_ref.original_forward is once


def test_arming_skips_uncompiled_blocks(monkeypatch):
    # An eager-tier load has no _compiled_call_impl: the hook must stay untouched, else compiling the inner adds compile the user declined.
    _stub_torch_compile(monkeypatch)
    block, hook, orig = _hooked_block(compiled = False)
    assert _compile_hooked_block_inners(_fake_dit([block])) == 0
    assert hook.fn_ref.original_forward is orig


def test_arming_skips_partial_captured_inner(monkeypatch):
    # A stacked hook chain (e.g. group offload) captures a functools.partial, not the bound method; arming would compile the wrong layer.
    _stub_torch_compile(monkeypatch)
    block, hook, orig = _hooked_block(bound = False)
    assert _compile_hooked_block_inners(_fake_dit([block])) == 0
    assert hook.fn_ref.original_forward is orig


def test_arming_covers_every_cache_hook_family(monkeypatch):
    # FBCache is the image cache today, but the hook-name table already covers the MagCache layout, so a future mode arms for free.
    _stub_torch_compile(monkeypatch)
    names = (
        "mag_cache_leader_block_hook",
        "mag_cache_block_hook",
        "fbc_leader_block_hook",
        "fbc_block_hook",
    )
    blocks = [_hooked_block(hook_name = n)[0] for n in names]
    assert _compile_hooked_block_inners(_fake_dit(blocks)) == len(names)


def test_restore_puts_the_exact_original_back(monkeypatch):
    _stub_torch_compile(monkeypatch)
    block, hook, orig = _hooked_block()
    dit = _fake_dit([block])
    _compile_hooked_block_inners(dit)
    _restore_hooked_block_inners(dit)
    assert hook.fn_ref.original_forward is orig
    assert hook._unsloth_orig_inner is None


def test_restore_tolerates_fakes_without_modules():
    _restore_hooked_block_inners(_MixinTransformer())  # no .modules(): no-op


def test_apply_step_cache_arms_compiled_blocks_on_toggle(monkeypatch):
    # The generation-time toggle engages the cache AFTER the load compiled the blocks, so apply_step_cache must arm the fresh hooks itself.
    _stub_diffusers(monkeypatch)
    _stub_torch_compile(monkeypatch)
    block, hook, orig = _hooked_block()

    class _T(_MixinTransformer):
        def modules(self):
            return [block]

    t = _T()
    engaged = apply_step_cache(_pipe(t), mode = "fbcache")
    assert engaged == TC_FBCACHE
    assert hook.fn_ref.original_forward is not orig
    assert hook._unsloth_orig_inner is orig


def test_toggle_disable_restores_inners_before_disable(monkeypatch):
    # remove_hook splices fn_ref.original_forward back into module.forward, so the compiled wrapper must be swapped out BEFORE disable_cache.
    _stub_diffusers(monkeypatch)
    order = []

    class _T(_ToggleTransformer):
        def disable_cache(self):
            super().disable_cache()
            order.append("disable")

        def modules(self):
            order.append("restore-walk")
            return []

    t = _T()
    maybe_toggle_step_cache(_pipe(t), steps = 28)
    mode = maybe_toggle_step_cache(_pipe(t), steps = 8)
    assert mode is None and t.disables == 1
    assert order[-2:] == ["restore-walk", "disable"]


def test_enable_failure_restores_inners_before_partial_disable(monkeypatch):
    # enable_cache can fail after hooking (and arming) some blocks, so the partial-hook cleanup must un-arm them before disable_cache.
    _stub_diffusers(monkeypatch)
    order = []

    class _T(_ToggleTransformer):
        def enable_cache(self, config):
            raise RuntimeError("block signature not recognised")

        def disable_cache(self):
            super().disable_cache()
            order.append("disable")

        def modules(self):
            order.append("restore-walk")
            return []

    t = _T()
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
    assert order == ["restore-walk", "disable"]


# ── stale child-registry cache invalidation (mid-session enable) ────────────────────


def test_enable_invalidates_stale_child_registry_cache(monkeypatch):
    # diffusers 0.39 caches the child-registry list on first cache_context use and an UNCACHED generation already populates it
    # (empty), so a later toggle-time enable_cache would install hooks the context never reaches ("No context is set").
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer()
    t._diffusers_hook = types.SimpleNamespace(_child_registries_cache = ["stale"])
    assert apply_step_cache(_pipe(t), mode = "fbcache") == TC_FBCACHE
    assert t._diffusers_hook._child_registries_cache is None


def test_invalidate_child_registry_cache_tolerates_absence():
    _invalidate_child_registry_cache(types.SimpleNamespace())  # no registry: no-op
    reg = types.SimpleNamespace(_child_registries_cache = None)
    _invalidate_child_registry_cache(types.SimpleNamespace(_diffusers_hook = reg))
    assert reg._child_registries_cache is None


# ── a second engage must not destroy the cache the first one installed ────────────
class _RealisticCacheMixin:
    """``CacheMixin`` as diffusers actually implements it, which the simple stub above does not model.

    The two behaviours that matter: ``enable_cache`` RAISES when a cache is already enabled ("To apply a new
    caching technique, please disable the existing one first"), and ``is_cache_enabled`` reports the live
    state. Without them a redundant ``apply_step_cache`` looks harmless in tests while tearing the running
    cache down in production.
    """

    def __init__(self):
        self.enabled_with = None
        self.disable_calls = 0
        self.enable_calls = 0

    @property
    def is_cache_enabled(self):
        return self.enabled_with is not None

    def enable_cache(self, config):
        self.enable_calls += 1
        if self.is_cache_enabled:
            raise ValueError(
                "Caching has already been enabled with <class 'FirstBlockCacheConfig'>."
            )
        self.enabled_with = config

    def disable_cache(self):
        self.disable_calls += 1
        self.enabled_with = None


def test_a_redundant_engage_keeps_the_running_cache(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _RealisticCacheMixin()
    assert apply_step_cache(_pipe(t), mode = "fbcache") == TC_FBCACHE
    first = t.enabled_with

    # Same settings a second time: the cache must still be on, on the SAME config object, and diffusers must
    # not have been asked to enable or disable anything.
    assert apply_step_cache(_pipe(t), mode = "fbcache") == TC_FBCACHE
    assert t.is_cache_enabled is True
    assert t.enabled_with is first
    assert (t.enable_calls, t.disable_calls) == (1, 0)
    assert t._unsloth_step_cache == f"fbcache@{DEFAULT_FBCACHE_THRESHOLD}"


def test_re_engaging_at_a_new_threshold_reconfigures(monkeypatch):
    # Different settings are a real request, so the old cache comes off and the new one goes on. This is the
    # only order diffusers accepts, and it must leave the marker describing the NEW threshold.
    _stub_diffusers(monkeypatch)
    t = _RealisticCacheMixin()
    apply_step_cache(_pipe(t), mode = "fbcache")
    assert apply_step_cache(_pipe(t), mode = "fbcache", threshold = 0.5) == TC_FBCACHE
    assert t.is_cache_enabled is True
    assert t.enabled_with.threshold == 0.5
    assert (t.enable_calls, t.disable_calls) == (2, 1)
    assert t._unsloth_step_cache == "fbcache@0.5"


def test_a_failed_re_engage_clears_the_marker(monkeypatch):
    # The marker is read far from here as "this transformer step-caches", so it must not outlive a failed
    # engage. Re-engaging at a new threshold is the only way to reach that state: the first engage set the
    # marker, the second drops the old cache and then fails, leaving a transformer that does NOT cache and a
    # marker that says it does.
    _stub_diffusers(monkeypatch)
    t = _RealisticCacheMixin()
    apply_step_cache(_pipe(t), mode = "fbcache")
    assert t._unsloth_step_cache == f"fbcache@{DEFAULT_FBCACHE_THRESHOLD}"

    real_enable = t.enable_cache

    def _boom(config):
        real_enable(config)
        raise RuntimeError("block signature not recognised")

    t.enable_cache = _boom
    assert apply_step_cache(_pipe(t), mode = "fbcache", threshold = 0.5) is None
    assert t._unsloth_step_cache is None


def test_a_stale_marker_without_hooks_re_engages(monkeypatch):
    # The reverse desync: a marker left set on a transformer whose hooks are gone. The cache is genuinely off,
    # so the engage must proceed and rewrite the marker rather than trust it and no-op.
    _stub_diffusers(monkeypatch)
    t = _RealisticCacheMixin()
    t._unsloth_step_cache = f"fbcache@{DEFAULT_FBCACHE_THRESHOLD}"
    assert apply_step_cache(_pipe(t), mode = "fbcache") == TC_FBCACHE
    assert t.is_cache_enabled is True
    assert t.enable_calls == 1


class _CleanupAlsoFailsTransformer:
    """``enable_cache`` hooks some blocks and THEN raises, so the cache reads as enabled only after the
    attempt; the cleanup ``disable_cache`` raises too, leaving those hooks live. ``is_cache_enabled``
    reports that honestly, and it is the only signal the recovery branch has to go on."""

    def __init__(self, *, enabled_after_failure = True):
        self.enabled_after_failure = enabled_after_failure
        self.attempted = False
        self.disable_attempted = False

    @property
    def is_cache_enabled(self):
        # Uncached until the part-way failure, so the re-entry guard at the top is not the path here.
        return self.attempted and self.enabled_after_failure

    def enable_cache(self, config):
        self.attempted = True
        raise RuntimeError("block signature not recognised")

    def disable_cache(self):
        self.disable_attempted = True
        raise RuntimeError("cannot unhook")

    def cache_context(self, *_a, **_k):
        raise AssertionError("not used")


def test_a_failed_cleanup_keeps_the_marker_rather_than_claiming_uncached(monkeypatch):
    """If disable_cache ALSO raises, hooks may still be live. Clearing the marker there would let the
    CUDA graph wrapper capture a cache-hooked forward: wrong output, where keeping it only forces eager."""
    _stub_diffusers(monkeypatch)
    t = _CleanupAlsoFailsTransformer(enabled_after_failure = True)
    t._unsloth_step_cache = "fbcache@0.1"

    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
    assert t.disable_attempted is True
    # Preserved: the transformer is still caching, so nothing downstream may treat it as uncached.
    assert t._unsloth_step_cache == "fbcache@0.1"


def test_a_failed_cleanup_that_did_unhook_still_clears_the_marker(monkeypatch):
    """disable_cache raising but the model reporting the cache OFF means the hooks went anyway, so the
    marker must come off; leaving it would force eager on a transformer that does not cache."""
    _stub_diffusers(monkeypatch)
    t = _CleanupAlsoFailsTransformer(enabled_after_failure = False)
    t._unsloth_step_cache = "fbcache@0.1"

    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
    assert t._unsloth_step_cache is None


class _ReconfigureCleanupPartlyFails:
    """Already caching. The re-configure ``disable_cache`` takes the hooks off and THEN raises, so the
    cache reads as off afterwards even though the call failed."""

    def __init__(self, *, enabled_after_disable):
        self.enabled = True
        self.enabled_after_disable = enabled_after_disable
        self.enabled_with = None

    @property
    def is_cache_enabled(self):
        return self.enabled

    def disable_cache(self):
        self.enabled = self.enabled_after_disable
        raise RuntimeError("hook registry went away mid-teardown")

    def enable_cache(self, config):
        self.enabled_with = config
        self.enabled = True

    def cache_context(self, *_a, **_k):
        raise AssertionError("not used")


def test_a_reconfigure_whose_cleanup_unhooked_anyway_re_engages(monkeypatch):
    """disable_cache raised, but the cache reads as OFF, so it did the thing this call wanted. Keeping
    the old marker there would pin the transformer to the eager path and block every later re-engage."""
    _stub_diffusers(monkeypatch)
    t = _ReconfigureCleanupPartlyFails(enabled_after_disable = False)
    t._unsloth_step_cache = "fbcache@0.1"

    assert apply_step_cache(_pipe(t), mode = "fbcache", threshold = 0.42) == TC_FBCACHE
    assert t.enabled_with.threshold == 0.42
    assert t._unsloth_step_cache == "fbcache@0.42"


def test_a_reconfigure_that_could_not_unhook_keeps_reporting_the_prior_mode(monkeypatch):
    """The other half: the cache is still enabled, so the hooks are live and the honest answer is the
    mode that is STILL running, not None and not the mode that was asked for."""
    _stub_diffusers(monkeypatch)
    t = _ReconfigureCleanupPartlyFails(enabled_after_disable = True)
    t._unsloth_step_cache = "fbcache@0.1"

    assert apply_step_cache(_pipe(t), mode = "fbcache", threshold = 0.42) == TC_FBCACHE
    assert t.enabled_with is None  # never re-engaged
    assert t._unsloth_step_cache == "fbcache@0.1"  # marker still describes the live hooks
