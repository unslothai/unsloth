# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for First-Block-Cache step caching, with diffusers stubbed in sys.modules."""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_cache as dc

from core.inference.diffusion_cache import (
    DEFAULT_FBCACHE_THRESHOLD,
    QUANT_FBCACHE_THRESHOLD,
    TC_FBCACHE,
    apply_step_cache,
    normalize_transformer_cache,
)


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


class FirstBlockCacheConfig:  # noqa: N801 - the name diffusers exports, and what the guard matches on
    def __init__(self, threshold):
        self.threshold = threshold


_Config = FirstBlockCacheConfig


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
    """Without enable_cache (not a CacheMixin) the load runs uncached; FBCache would crash at generation."""


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


class _PrefixKVTransformer(_MixinTransformer):
    """A CacheMixin transformer caching the prompt/condition prefix K and V (Qwen-Image-2.1, FLUX.2
    klein KV, Wan-Animate-2); blocks see the joint sequence once, then target tokens."""

    def forward(
        self,
        hidden_states,
        *,
        kv_cache = None,
        kv_cache_mode = None,
    ):
        return hidden_states


class _PrefixKVPipe(_CtxPipe):
    """A denoise loop that drives the prefix cache: prefill once, then decode. FBCache must not
    engage, or step 2 dies subtracting a 4096-row residual from a 4297-row one."""

    def __call__(self, *args, **kwargs):
        with self.transformer.cache_context("cond"):
            self.transformer(None, kv_cache = object(), kv_cache_mode = "extract")
            self.transformer(None, kv_cache = object(), kv_cache_mode = "cached")
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

    # diffusers removes FBCache by name via HookRegistry, not via _cache_config.
    class _Registry:
        instances: dict = {}
        removed: list = []

        def __init__(self, module):
            self.module = module

        @classmethod
        def check_if_exists_or_initialize(cls, module):
            return cls.instances.setdefault(id(module), cls(module))

        def remove_hook(
            self,
            name,
            recurse = True,
        ):
            type(self).removed.append(name)

    _Registry.instances = {}
    _Registry.removed = []
    hooks.HookRegistry = _Registry
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)

    fbc = types.ModuleType("diffusers.hooks.first_block_cache")
    fbc._FBC_LEADER_BLOCK_HOOK = "fbc_leader_block_hook"
    fbc._FBC_BLOCK_HOOK = "fbc_block_hook"
    monkeypatch.setitem(sys.modules, "diffusers.hooks.first_block_cache", fbc)
    hooks.first_block_cache = fbc
    return _Registry


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
    rec: dict = {}
    _stub_diffusers(monkeypatch, hook_recorder = rec)
    t = _NonCacheMixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
    assert rec == {}


def test_pipeline_without_cache_context_runs_uncached(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer()
    assert apply_step_cache(_NoCtxPipe(t), mode = "fbcache") is None
    assert t.enabled_with is None


def test_prefix_kv_transformer_runs_uncached_without_the_length_guard(monkeypatch):
    # Prefix-KV models shrink the sequence after step 0, so FBCache's residual subtraction raises.
    _stub_diffusers(monkeypatch)
    monkeypatch.setattr(dc, "install_fbcache_length_guard", lambda: False)
    t = _PrefixKVTransformer()
    assert apply_step_cache(_PrefixKVPipe(t), mode = "fbcache", length_changes_ok = True) is None
    assert t.enabled_with is None


def test_prefix_kv_transformer_is_cached_once_the_length_guard_is_in(monkeypatch):
    _stub_diffusers(monkeypatch)
    monkeypatch.setattr(dc, "install_fbcache_length_guard", lambda: True)
    t = _PrefixKVTransformer()
    assert apply_step_cache(_PrefixKVPipe(t), mode = "fbcache") is None
    assert t.enabled_with is None
    assert apply_step_cache(_PrefixKVPipe(t), mode = "fbcache", length_changes_ok = True) == "fbcache"
    assert t.enabled_with is not None


def test_length_guard_recomputes_on_a_length_change_and_defers_otherwise(monkeypatch):
    torch = pytest.importorskip("torch")
    fbc = pytest.importorskip("diffusers.hooks.first_block_cache")
    monkeypatch.setattr(
        fbc.FBCHeadBlockHook,
        "_should_compute_remaining_blocks",
        fbc.FBCHeadBlockHook._should_compute_remaining_blocks,
    )
    assert dc.install_fbcache_length_guard() is True
    guarded = fbc.FBCHeadBlockHook._should_compute_remaining_blocks
    assert dc.install_fbcache_length_guard() is True
    assert fbc.FBCHeadBlockHook._should_compute_remaining_blocks is guarded

    state = types.SimpleNamespace(
        head_block_residual = torch.ones(1, 4297, 8), tail_block_residuals = None
    )
    hook = fbc.FBCHeadBlockHook.__new__(fbc.FBCHeadBlockHook)
    hook.state_manager = types.SimpleNamespace(get_state = lambda: state)
    hook.threshold = 0.1
    assert hook._should_compute_remaining_blocks(torch.ones(1, 4096, 8)) is True
    state.head_block_residual = torch.ones(1, 4096, 8)
    assert hook._should_compute_remaining_blocks(torch.ones(1, 4096, 8)) is False
    assert hook._should_compute_remaining_blocks(torch.full((1, 4096, 8), 2.0)) is True
    state.tail_block_residuals = (torch.ones(1, 4297, 8), None)
    assert hook._should_compute_remaining_blocks(torch.ones(1, 4096, 8)) is True


def test_prefix_kv_guard_does_not_catch_an_ordinary_transformer(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "fbcache") == "fbcache"
    assert t.enabled_with is not None


def test_prefix_kv_guard_ignores_a_parameter_the_loop_never_drives(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _PrefixKVTransformer()
    assert apply_step_cache(_CtxPipe(t), mode = "fbcache") == "fbcache"
    assert t.enabled_with is not None


def test_incompatible_model_runs_uncached(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer(fail = True)
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None


def test_enable_cache_failure_rolls_back_partial_hooks(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer(fail = True)
    t.disabled = False
    t.disable_cache = lambda: setattr(t, "disabled", True)
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
    assert t.disabled is True


def test_config_import_falls_back_to_hooks_module(monkeypatch):
    diffusers = types.ModuleType("diffusers")
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
    # Block diffusers.hooks too: the config import falls back to it and may be in sys.modules.
    monkeypatch.setitem(sys.modules, "diffusers", None)
    monkeypatch.setitem(sys.modules, "diffusers.hooks", None)
    t = _MixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None


from core.inference.diffusion_cache import (  # noqa: E402
    FBCACHE_MIN_STEPS,
    TC_AUTO,
    effective_denoise_steps,
    effective_request_strength,
    maybe_toggle_step_cache,
)


def test_effective_steps_txt2img_is_full_count():
    assert effective_denoise_steps(28, None) == 28
    assert effective_denoise_steps(28, 1.0) == 28


def test_effective_steps_low_strength_shrinks_below_the_bar():
    # 28 steps at strength 0.35 denoise int(9.8) = 9 steps, below FBCACHE_MIN_STEPS.
    eff = effective_denoise_steps(28, 0.35)
    assert eff == 9
    assert eff < FBCACHE_MIN_STEPS


def test_effective_request_strength_uses_pipe_default_when_omitted():
    import inspect

    assert effective_request_strength(None, False, True, 0.6) is None
    assert effective_request_strength(0.5, True, False, None) is None
    assert effective_request_strength(0.2, True, True, 0.6) == 0.2
    s = effective_request_strength(None, True, True, 0.6)
    assert s == 0.6
    assert effective_denoise_steps(28, s) == 16
    assert effective_request_strength(None, True, True, inspect.Parameter.empty) is None
    assert effective_request_strength(None, True, True, None) is None


def test_effective_steps_matches_diffusers_get_timesteps():
    # Mirror diffusers exactly: it denoises min(int(steps * strength), steps) steps (floored).
    for steps, strength in [(28, 0.35), (28, 0.8), (50, 0.5), (20, 0.99), (30, 0.1)]:
        expected = max(1, min(int(steps * strength), steps))
        assert effective_denoise_steps(steps, strength) == expected


def test_toggle_stays_off_for_low_strength_workflow(monkeypatch):
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
    assert maybe_toggle_step_cache(_pipe(t), steps = 8) is None
    assert t.disables == 1


class _DiffusersLikeTransformer(_ToggleTransformer):
    """Fake records _cache_config as diffusers does; the cache layer trusts that, not its own marker."""

    def enable_cache(self, config):
        super().enable_cache(config)
        self._cache_config = config
        self.is_cache_enabled = True

    def disable_cache(self):
        super().disable_cache()
        self._cache_config = None
        self.is_cache_enabled = False


def _hide_private_hook_names(monkeypatch):
    """Make ``diffusers.hooks.first_block_cache`` unimportable, as a shifted diffusers would."""
    monkeypatch.setitem(sys.modules, "diffusers.hooks.first_block_cache", None)


def test_toggle_disengages_even_when_the_private_hook_names_are_gone(monkeypatch):
    """Our own FBCache must disengage via diffusers' disable_cache, not depend on private hook names."""
    _stub_diffusers(monkeypatch)
    t = _DiffusersLikeTransformer()
    maybe_toggle_step_cache(_pipe(t), steps = 28)
    assert t._cache_config is not None, "the engage did not leave a live config to key on"

    _hide_private_hook_names(monkeypatch)
    mode = maybe_toggle_step_cache(_pipe(t), steps = 8)
    assert mode is None, "a cache diffusers removed itself is still being reported as engaged"
    assert t.disables == 1
    assert not t._unsloth_step_cache


def test_toggle_keeps_an_adopted_cache_it_cannot_verify_removed(monkeypatch):
    """No live config means disable_cache removed nothing; an unverifiable adopted cache stays engaged."""
    _stub_diffusers(monkeypatch)
    t = _ToggleTransformer()
    maybe_toggle_step_cache(_pipe(t), steps = 28)

    _hide_private_hook_names(monkeypatch)
    assert maybe_toggle_step_cache(_pipe(t), steps = 8) == TC_FBCACHE
    assert t._unsloth_step_cache, "the marker came off hooks that were never verified gone"


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
    assert calls[0][1] == {"fullgraph": False, "dynamic": True}


def test_arming_is_idempotent(monkeypatch):
    _stub_torch_compile(monkeypatch)
    block, hook, _ = _hooked_block()
    dit = _fake_dit([block])
    assert _compile_hooked_block_inners(dit) == 1
    once = hook.fn_ref.original_forward
    assert _compile_hooked_block_inners(dit) == 0
    assert hook.fn_ref.original_forward is once


def test_arming_skips_uncompiled_blocks(monkeypatch):
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
    _restore_hooked_block_inners(_MixinTransformer())


def test_apply_step_cache_arms_compiled_blocks_on_toggle(monkeypatch):
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
    # remove_hook splices original_forward back, so unwrap the compiled inner BEFORE disable_cache.
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
    # Two walks (pre-engage probe, then restore); the restore must precede disable_cache.
    assert order == ["restore-walk", "restore-walk", "disable"]


def test_enable_invalidates_stale_child_registry_cache(monkeypatch):
    # diffusers 0.39 caches the child-registry list on first cache_context use, even uncached.
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer()
    t._diffusers_hook = types.SimpleNamespace(_child_registries_cache = ["stale"])
    assert apply_step_cache(_pipe(t), mode = "fbcache") == TC_FBCACHE
    assert t._diffusers_hook._child_registries_cache is None


def test_invalidate_child_registry_cache_tolerates_absence():
    _invalidate_child_registry_cache(types.SimpleNamespace())
    reg = types.SimpleNamespace(_child_registries_cache = None)
    _invalidate_child_registry_cache(types.SimpleNamespace(_diffusers_hook = reg))
    assert reg._child_registries_cache is None


class _RealisticCacheMixin:
    """CacheMixin as diffusers implements it: enable_cache raises if a cache is already enabled."""

    def __init__(self):
        self.enabled_with = None
        # diffusers' is_cache_enabled is `_cache_config is not None`, assigned last in enable_cache.
        self._cache_config = None
        self.disable_calls = 0
        self.enable_calls = 0

    @property
    def is_cache_enabled(self):
        return self._cache_config is not None

    def enable_cache(self, config):
        self.enable_calls += 1
        if self.is_cache_enabled:
            raise ValueError(
                "Caching has already been enabled with <class 'FirstBlockCacheConfig'>."
            )
        self.enabled_with = config
        self._cache_config = config

    def disable_cache(self):
        self.disable_calls += 1
        self.enabled_with = None
        self._cache_config = None


def test_a_redundant_engage_keeps_the_running_cache(monkeypatch):
    _stub_diffusers(monkeypatch)
    t = _RealisticCacheMixin()
    assert apply_step_cache(_pipe(t), mode = "fbcache") == TC_FBCACHE
    first = t.enabled_with

    assert apply_step_cache(_pipe(t), mode = "fbcache") == TC_FBCACHE
    assert t.is_cache_enabled is True
    assert t.enabled_with is first
    assert (t.enable_calls, t.disable_calls) == (1, 0)
    assert t._unsloth_step_cache == f"fbcache@{DEFAULT_FBCACHE_THRESHOLD}"


def test_re_engaging_at_a_new_threshold_reconfigures(monkeypatch):
    # Disable then enable is the only order diffusers accepts; the marker must show the new threshold.
    _stub_diffusers(monkeypatch)
    t = _RealisticCacheMixin()
    apply_step_cache(_pipe(t), mode = "fbcache")
    assert apply_step_cache(_pipe(t), mode = "fbcache", threshold = 0.5) == TC_FBCACHE
    assert t.is_cache_enabled is True
    assert t.enabled_with.threshold == 0.5
    assert (t.enable_calls, t.disable_calls) == (2, 1)
    assert t._unsloth_step_cache == "fbcache@0.5"


def test_a_failed_re_engage_clears_the_marker(monkeypatch):
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
        return self.attempted and self.enabled_after_failure

    def enable_cache(self, config):
        self.attempted = True
        raise RuntimeError("block signature not recognised")

    def disable_cache(self):
        self.disable_attempted = True
        raise RuntimeError("cannot unhook")

    def cache_context(self, *_a, **_k):
        raise AssertionError("not used")


def test_a_failed_cleanup_removes_the_hooks_itself_then_clears_the_marker(monkeypatch):
    """disable_cache raising does not end it: the hooks come off by NAME through the registry, which is
    the layer under diffusers' own teardown, so the marker can be cleared on evidence rather than hope."""
    registry = _stub_diffusers(monkeypatch)
    t = _CleanupAlsoFailsTransformer(enabled_after_failure = True)
    t._unsloth_step_cache = "fbcache@0.1"

    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
    assert t.disable_attempted is True
    assert registry.removed == ["fbc_leader_block_hook", "fbc_block_hook"]
    assert t._unsloth_step_cache is None


def test_a_silently_partial_engage_still_has_its_hooks_taken_off(monkeypatch):
    """A partial engage leaves hooks live with is_cache_enabled False; removal must go by name."""
    registry = _stub_diffusers(monkeypatch)

    class _PartlyHooked:
        _cache_config = None
        disable_calls = 0

        @property
        def is_cache_enabled(self):
            return self._cache_config is not None

        def enable_cache(self, config):
            raise RuntimeError("apply_first_block_cache died after hooking 3 of 57 blocks")

        def disable_cache(self):
            type(self).disable_calls += 1

        def cache_context(self, *_a, **_k):
            raise AssertionError("not used")

    t = _PartlyHooked()
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
    assert registry.removed == ["fbc_leader_block_hook", "fbc_block_hook"]
    assert t._unsloth_step_cache is None


def test_an_unverifiable_teardown_keeps_the_marker(monkeypatch):
    """Without a registry to remove hooks through, their removal can't be verified, so the marker
    stays: clearing it would let a graph capture a partly-hooked forward."""
    _stub_diffusers(monkeypatch)
    monkeypatch.delitem(sys.modules, "diffusers.hooks.first_block_cache", raising = False)
    monkeypatch.setitem(sys.modules, "diffusers.hooks", types.ModuleType("diffusers.hooks"))
    sys.modules["diffusers.hooks"].apply_first_block_cache = lambda *_a, **_k: None
    sys.modules["diffusers.hooks"].FirstBlockCacheConfig = _Config

    t = _CleanupAlsoFailsTransformer(enabled_after_failure = True)
    t._unsloth_step_cache = "fbcache@0.1"
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
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


def test_a_reconfigure_that_cannot_be_verified_keeps_reporting_the_prior_mode(monkeypatch):
    """With no registry to remove through, teardown cannot be shown complete; re-engaging over
    possibly live hooks is the dangerous move, so the last known mode is reported."""
    _stub_diffusers(monkeypatch)
    monkeypatch.delitem(sys.modules, "diffusers.hooks.first_block_cache", raising = False)
    monkeypatch.setitem(sys.modules, "diffusers.hooks", types.ModuleType("diffusers.hooks"))
    sys.modules["diffusers.hooks"].apply_first_block_cache = lambda *_a, **_k: None
    sys.modules["diffusers.hooks"].FirstBlockCacheConfig = _Config

    t = _ReconfigureCleanupPartlyFails(enabled_after_disable = True)
    t._unsloth_step_cache = "fbcache@0.1"

    assert apply_step_cache(_pipe(t), mode = "fbcache", threshold = 0.42) == TC_FBCACHE
    assert t.enabled_with is None
    assert t._unsloth_step_cache == "fbcache@0.1"


def test_a_reconfigure_that_dies_between_the_two_hook_removals_finishes_the_job(monkeypatch):
    """A raise between the two hook removals must be finished; leftover block hooks are worse than none."""
    registry = _stub_diffusers(monkeypatch)

    class FirstBlockCacheConfig:  # noqa: N801 - matched by NAME, so the name is the fixture
        pass

    class _DiesMidTeardown:
        _cache_config = FirstBlockCacheConfig()

        @property
        def is_cache_enabled(self):
            return self._cache_config is not None

        def disable_cache(self):
            raise RuntimeError("registry mutated while removing fbc_block_hook")

        def enable_cache(self, config):
            self.enabled_with = config

        def cache_context(self, *_a, **_k):
            raise AssertionError("not used")

    t = _DiesMidTeardown()
    t.enabled_with = None
    t._unsloth_step_cache = "fbcache@0.1"

    assert apply_step_cache(_pipe(t), mode = "fbcache", threshold = 0.3) == TC_FBCACHE
    assert registry.removed == ["fbc_leader_block_hook", "fbc_block_hook"]
    assert t.enabled_with.threshold == 0.3
    assert t._unsloth_step_cache == "fbcache@0.3"


def test_another_cache_type_is_left_alone(monkeypatch):
    """MagCache, FasterCache and PAB share the generic _cache_config, so they look like broken FBC
    configs; FBC hook names miss them, and clearing state would forget a live cache."""
    registry = _stub_diffusers(monkeypatch)

    class _MagCacheConfig:
        pass

    class _RunningMagCache:
        def __init__(self):
            self._cache_config = _MagCacheConfig()
            self.enabled_with = None

        @property
        def is_cache_enabled(self):
            return self._cache_config is not None

        def disable_cache(self):
            raise RuntimeError("teardown of the other cache failed")

        def enable_cache(self, config):
            self.enabled_with = config

        def cache_context(self, *_a, **_k):
            raise AssertionError("not used")

    t = _RunningMagCache()
    t._unsloth_step_cache = "magcache@0.1"

    assert apply_step_cache(_pipe(t), mode = "fbcache") == "magcache"
    assert registry.removed == []
    assert isinstance(t._cache_config, _MagCacheConfig)
    assert t.enabled_with is None


def test_a_lost_marker_does_not_cost_a_healthy_cache(monkeypatch):
    """A lost own marker must not tear down a healthy cache; the live diffusers config decides."""
    registry = _stub_diffusers(monkeypatch)

    class FirstBlockCacheConfig:  # noqa: N801 - matched by NAME
        def __init__(self, threshold):
            self.threshold = threshold

    class _CachedByHand:
        def __init__(self):
            self._cache_config = FirstBlockCacheConfig(DEFAULT_FBCACHE_THRESHOLD)
            self.disable_calls = 0

        @property
        def is_cache_enabled(self):
            return self._cache_config is not None

        def disable_cache(self):
            self.disable_calls += 1
            self._cache_config = None

        def enable_cache(self, config):
            self._cache_config = config

        def cache_context(self, *_a, **_k):
            raise AssertionError("not used")

    t = _CachedByHand()
    engaged = apply_step_cache(_pipe(t), mode = "fbcache")

    assert engaged == TC_FBCACHE
    assert t.disable_calls == 0
    assert registry.removed == []
    assert t._unsloth_step_cache == f"fbcache@{DEFAULT_FBCACHE_THRESHOLD}"


def test_a_stale_marker_cannot_authorize_the_no_op(monkeypatch):
    """Only the live config decides: a transformer reconfigured elsewhere keeps the old marker, and
    trusting it would report success for settings it is not running."""
    _stub_diffusers(monkeypatch)

    class FirstBlockCacheConfig:  # noqa: N801 - matched by NAME
        def __init__(self, threshold):
            self.threshold = threshold

    class _Reconfigured:
        def __init__(self):
            self._cache_config = FirstBlockCacheConfig(0.5)
            self.disable_calls = 0

        @property
        def is_cache_enabled(self):
            return self._cache_config is not None

        def disable_cache(self):
            self.disable_calls += 1
            self._cache_config = None

        def enable_cache(self, config):
            self._cache_config = config

        def cache_context(self, *_a, **_k):
            raise AssertionError("not used")

    t = _Reconfigured()
    t._unsloth_step_cache = f"fbcache@{DEFAULT_FBCACHE_THRESHOLD}"

    assert apply_step_cache(_pipe(t), mode = "fbcache") == TC_FBCACHE
    assert t.disable_calls == 1
    assert t._cache_config.threshold == DEFAULT_FBCACHE_THRESHOLD


def test_an_adopted_cache_gets_the_post_enable_integration(monkeypatch):
    """An adopted live cache still gets post-enable integration, or cache_context reaches no block."""
    _stub_diffusers(monkeypatch)
    import core.inference.diffusion_cache as dc

    ran = []
    monkeypatch.setattr(dc, "_invalidate_child_registry_cache", lambda t: ran.append("invalidate"))
    monkeypatch.setattr(
        dc, "_compile_hooked_block_inners", lambda t, log = None: ran.append("compile")
    )

    class FirstBlockCacheConfig:  # noqa: N801 - matched by NAME
        def __init__(self, threshold):
            self.threshold = threshold

    class _CachedByHand:
        def __init__(self):
            self._cache_config = FirstBlockCacheConfig(DEFAULT_FBCACHE_THRESHOLD)

        @property
        def is_cache_enabled(self):
            return self._cache_config is not None

        def disable_cache(self):
            raise AssertionError("the healthy cache must not be torn down")

        def enable_cache(self, config):
            raise AssertionError("the healthy cache must not be rebuilt")

        def cache_context(self, *_a, **_k):
            raise AssertionError("not used")

    assert apply_step_cache(_pipe(_CachedByHand()), mode = "fbcache") == TC_FBCACHE
    assert ran == ["invalidate", "compile"]


def test_a_cache_installed_through_the_low_level_api_is_not_torn_down(monkeypatch):
    """Hooks from low-level apply_first_block_cache lack _cache_config; recovery must leave them alone."""
    registry = _stub_diffusers(monkeypatch)
    import core.inference.diffusion_cache as dc

    monkeypatch.setattr(dc, "_first_block_cache_is_hooked", lambda t: True)

    class _HookedTheOtherWay:
        is_cache_enabled = False

        def enable_cache(self, config):
            raise ValueError("Hook with name fbc_leader_block_hook already exists in the registry.")

        def disable_cache(self):
            raise AssertionError("must not be reached: there is no config for diffusers to act on")

        def cache_context(self, *_a, **_k):
            raise AssertionError("not used")

    t = _HookedTheOtherWay()
    assert apply_step_cache(_pipe(t), mode = "fbcache") == TC_FBCACHE
    assert registry.removed == []


def test_a_low_level_cache_is_integrated_and_marked_before_being_reported(monkeypatch):
    """Adopting low-level hooks must integrate and set the marker first, as for a live config: a
    clear marker over live hooks lets GraphedForward capture a data-dependent cached forward."""
    registry = _stub_diffusers(monkeypatch)
    import core.inference.diffusion_cache as dc

    monkeypatch.setattr(dc, "_first_block_cache_is_hooked", lambda t: True)
    ran = []
    monkeypatch.setattr(dc, "_invalidate_child_registry_cache", lambda t: ran.append("invalidate"))
    monkeypatch.setattr(
        dc, "_compile_hooked_block_inners", lambda t, log = None: ran.append("compile")
    )

    class _HookedTheOtherWay:
        is_cache_enabled = False

        def enable_cache(self, config):
            raise ValueError("Hook with name fbc_leader_block_hook already exists in the registry.")

        def disable_cache(self):
            raise AssertionError("must not be reached: there is no config for diffusers to act on")

        def cache_context(self, *_a, **_k):
            raise AssertionError("not used")

    t = _HookedTheOtherWay()
    assert apply_step_cache(_pipe(t), mode = "fbcache") == TC_FBCACHE
    assert ran == ["invalidate", "compile"]
    # Truthy keeps the CUDA graph wrapper eager over live hooks; mode-only since threshold is unknown.
    assert t._unsloth_step_cache == TC_FBCACHE
    assert registry.removed == []


def test_auto_disengage_unhooks_an_adopted_cache_instead_of_trusting_disable_cache(monkeypatch):
    """An adopted cache has no ``_cache_config``, so ``disable_cache`` removes nothing; clearing
    the marker on that call alone would leave it clear over live hooks."""
    registry = _stub_diffusers(monkeypatch)

    class _AdoptedLowLevel:
        _cache_config = None
        _unsloth_step_cache = TC_FBCACHE

        def __init__(self):
            self.disable_calls = 0

        def disable_cache(self):
            self.disable_calls += 1

    t = _AdoptedLowLevel()
    assert maybe_toggle_step_cache(_pipe(t), steps = 8) is None
    assert t.disable_calls == 1
    assert registry.removed == ["fbc_leader_block_hook", "fbc_block_hook"]
    assert t._unsloth_step_cache is None


def test_auto_disengage_keeps_the_marker_when_the_hooks_cannot_be_verified_gone(monkeypatch):
    """Unverifiable teardown keeps the marker: slow eager beats a graph captured over a cache."""
    _stub_diffusers(monkeypatch)
    import core.inference.diffusion_cache as dc

    monkeypatch.setattr(dc, "_unhook_first_block_cache", lambda t: False)

    class _Unverifiable:
        _cache_config = None
        _unsloth_step_cache = TC_FBCACHE

        def disable_cache(self):
            pass

    t = _Unverifiable()
    assert maybe_toggle_step_cache(_pipe(t), steps = 8) == TC_FBCACHE
    assert t._unsloth_step_cache == TC_FBCACHE


def test_the_hook_probe_sees_the_names_the_low_level_api_installs(monkeypatch):
    """The probe is the whole basis for the teardown decision, so it is pinned to the real hook
    names: an upstream rename must fail here, not tear down other caches."""
    _stub_diffusers(monkeypatch)
    import types as _types

    from diffusers.hooks.first_block_cache import _FBC_BLOCK_HOOK, _FBC_LEADER_BLOCK_HOOK
    from core.inference.diffusion_cache import _first_block_cache_is_hooked

    bare = _types.SimpleNamespace(modules = lambda: [_types.SimpleNamespace()])
    assert _first_block_cache_is_hooked(bare) is False

    for name in (_FBC_LEADER_BLOCK_HOOK, _FBC_BLOCK_HOOK):
        block = _types.SimpleNamespace(
            _diffusers_hook = _types.SimpleNamespace(hooks = {name: object()})
        )
        hooked = _types.SimpleNamespace(
            modules = lambda block = block: [_types.SimpleNamespace(), block]
        )
        assert _first_block_cache_is_hooked(hooked) is True


def test_auto_tier_is_the_speed_layers_max():
    from core.inference.diffusion_speed import SPEED_MAX
    assert dc.AUTO_STEP_CACHE_TIER == SPEED_MAX


@pytest.mark.parametrize("tier", ["off", "eager", "default", None])
def test_auto_resolves_uncached_below_max(tier):
    assert not dc.auto_step_cache_allowed(tier)
    assert dc.resolve_auto_step_cache(tier, 50) is None


def test_auto_resolves_fbcache_on_max_at_the_step_bar():
    assert dc.resolve_auto_step_cache("max", dc.FBCACHE_MIN_STEPS) == TC_FBCACHE
    assert dc.resolve_auto_step_cache("max", dc.FBCACHE_MIN_STEPS - 1) is None


def test_step_cache_supported_mirrors_the_engage_refusals(monkeypatch):
    _stub_diffusers(monkeypatch)
    assert dc.step_cache_supported(_pipe(_MixinTransformer()))
    assert not dc.step_cache_supported(_pipe(_NonCacheMixinTransformer()))
    assert not dc.step_cache_supported(_NoCtxPipe(_MixinTransformer()))
    assert not dc.step_cache_supported(_PrefixKVPipe(_PrefixKVTransformer()))
    assert not dc.step_cache_supported(types.SimpleNamespace(transformer = None))
