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
