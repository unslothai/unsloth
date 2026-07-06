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


def _pipe(transformer):
    return types.SimpleNamespace(transformer = transformer)


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
    # A transformer without enable_cache (e.g. Z-Image) must NOT install the standalone hook
    # -- its pipeline opens no cache_context, so it runs uncached instead of crashing at gen.
    rec: dict = {}
    _stub_diffusers(monkeypatch, hook_recorder = rec)
    t = _NonCacheMixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
    assert rec == {}  # the standalone hook was never called


def test_incompatible_model_runs_uncached(monkeypatch):
    # enable_cache raising (e.g. unrecognised block signature) must not fail the load.
    _stub_diffusers(monkeypatch)
    t = _MixinTransformer(fail = True)
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None


def test_enable_cache_failure_rolls_back_partial_hooks(monkeypatch):
    # enable_cache can raise after hooking some blocks; the reported-uncached model
    # must not actually run half-cached, so the failure path calls disable_cache.
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
    # no diffusers import -> best-effort returns None, load proceeds uncached. Block the
    # hooks module too: the config import falls back to diffusers.hooks, which a REAL
    # earlier import in the test session may have left cached in sys.modules.
    monkeypatch.setitem(sys.modules, "diffusers", None)
    monkeypatch.setitem(sys.modules, "diffusers.hooks", None)
    t = _MixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None


# ── the auto policy: normalize("auto") + generation-time toggling ──────────────────
from core.inference.diffusion_cache import (  # noqa: E402
    FBCACHE_MIN_STEPS,
    TC_AUTO,
    effective_denoise_steps,
    maybe_toggle_step_cache,
)


# ── effective_denoise_steps (strength-aware step count for the auto policy) ─────────
def test_effective_steps_txt2img_is_full_count():
    # No strength (txt2img / reference) -> the full requested step count.
    assert effective_denoise_steps(28, None) == 28
    assert effective_denoise_steps(28, 1.0) == 28  # full redraw denoises every step


def test_effective_steps_low_strength_shrinks_below_the_bar():
    # A 28-step upscale at strength 0.35 denoises int(9.8) = 9 steps (diffusers get_timesteps
    # floors the product), which is below FBCACHE_MIN_STEPS -> the auto policy must NOT engage
    # FBCache there.
    eff = effective_denoise_steps(28, 0.35)
    assert eff == 9
    assert eff < FBCACHE_MIN_STEPS


def test_effective_steps_matches_diffusers_get_timesteps():
    # Mirror diffusers exactly: it denoises init_timestep = min(int(num_inference_steps *
    # strength), num_inference_steps) steps (the product is floored, not rounded).
    for steps, strength in [(28, 0.35), (28, 0.8), (50, 0.5), (20, 0.99), (30, 0.1)]:
        expected = max(1, min(int(steps * strength), steps))
        assert effective_denoise_steps(steps, strength) == expected


def test_toggle_stays_off_for_low_strength_workflow(monkeypatch):
    # End to end: a 28-step request would engage FBCache, but at strength 0.35 the
    # effective ~10 steps keep it uncached.
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
    # AUTO must be resolved by the loader; if it ever reaches the engage call the
    # load runs uncached instead of crashing.
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
