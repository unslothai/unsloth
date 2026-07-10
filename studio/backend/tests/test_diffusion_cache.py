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
    # A transformer without enable_cache (e.g. Z-Image) must NOT install the standalone hook
    # -- its pipeline opens no cache_context, so it runs uncached instead of crashing at gen.
    rec: dict = {}
    _stub_diffusers(monkeypatch, hook_recorder = rec)
    t = _NonCacheMixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "fbcache") is None
    assert rec == {}  # the standalone hook was never called


def test_pipeline_without_cache_context_runs_uncached(monkeypatch):
    # A CacheMixin transformer whose PIPELINE never opens a cache_context (Flux Kontext /
    # img2img / inpaint / controlnet reuse the CacheMixin FluxTransformer2DModel) must run
    # uncached -- otherwise the First-Block-Cache hook raises "No context is set" on the
    # first forward, crashing every default generation.
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
    effective_request_strength,
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


def test_effective_request_strength_uses_pipe_default_when_omitted():
    import inspect

    # txt2img (no init image) or a pipe without the strength kwarg -> full trajectory (None).
    assert effective_request_strength(None, False, True, 0.6) is None
    assert effective_request_strength(0.5, True, False, None) is None
    # img2img with an explicit strength -> that value.
    assert effective_request_strength(0.2, True, True, 0.6) == 0.2
    # img2img with an OMITTED strength -> the pipe's own signature default (< 1), so the auto
    # policy keys on the real (short) trajectory, not the full step count. This is the fix:
    # int(28 * 0.6) = 16 real steps, not 28.
    s = effective_request_strength(None, True, True, 0.6)
    assert s == 0.6
    assert effective_denoise_steps(28, s) == 16
    # A non-numeric signature default (inspect.Parameter.empty) falls back to the full count.
    assert effective_request_strength(None, True, True, inspect.Parameter.empty) is None
    assert effective_request_strength(None, True, True, None) is None


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


# ── FBCache block-metadata registration (HunyuanVideo-1.5) ─────────────────────────
from core.inference.diffusion_cache import (  # noqa: E402
    _ensure_block_metadata_registered,
    _invalidate_child_registry_cache,
)


def _stub_hunyuan15_registry(monkeypatch):
    """Stub the two diffusers modules the registration helper imports: the FBCache
    metadata registry (diffusers.hooks._helpers) and the HunyuanVideo-1.5 transformer
    module carrying the block class. Returns (registry_cls, block_cls)."""

    class _Metadata:
        def __init__(
            self,
            return_hidden_states_index = None,
            return_encoder_hidden_states_index = None,
        ):
            self.return_hidden_states_index = return_hidden_states_index
            self.return_encoder_hidden_states_index = return_encoder_hidden_states_index

    class _Registry:
        registry: dict = {}

        @classmethod
        def get(cls, model_class):
            if model_class not in cls.registry:
                raise ValueError(f"Model class {model_class} not registered.")
            return cls.registry[model_class]

        @classmethod
        def register(cls, model_class, metadata):
            cls.registry[model_class] = metadata

    class HunyuanVideo15TransformerBlock:  # the sentinel block class
        pass

    helpers = types.ModuleType("diffusers.hooks._helpers")
    helpers.TransformerBlockMetadata = _Metadata
    helpers.TransformerBlockRegistry = _Registry
    monkeypatch.setitem(sys.modules, "diffusers.hooks._helpers", helpers)

    blocks = types.ModuleType("diffusers.models.transformers.transformer_hunyuan_video15")
    blocks.HunyuanVideo15TransformerBlock = HunyuanVideo15TransformerBlock
    monkeypatch.setitem(
        sys.modules,
        "diffusers.models.transformers.transformer_hunyuan_video15",
        blocks,
    )
    return _Registry, HunyuanVideo15TransformerBlock


class HunyuanVideo15Transformer3DModel(_MixinTransformer):
    """A CacheMixin-style fake whose CLASS NAME keys the extra-metadata table."""


def test_hunyuan15_block_metadata_is_registered(monkeypatch):
    registry, block_cls = _stub_hunyuan15_registry(monkeypatch)
    _ensure_block_metadata_registered(HunyuanVideo15Transformer3DModel())
    meta = registry.registry[block_cls]
    # The 1.5 dual-stream block returns (hidden_states, encoder_hidden_states) -- the
    # same layout as the natively registered HunyuanVideo 1.0 block.
    assert meta.return_hidden_states_index == 0
    assert meta.return_encoder_hidden_states_index == 1


def test_hunyuan15_registration_defers_to_a_native_one(monkeypatch):
    # A diffusers release that ships the registration natively must win: the helper
    # probes TransformerBlockRegistry.get first and never overwrites.
    registry, block_cls = _stub_hunyuan15_registry(monkeypatch)
    native = object()
    registry.registry[block_cls] = native
    _ensure_block_metadata_registered(HunyuanVideo15Transformer3DModel())
    assert registry.registry[block_cls] is native


def test_registration_noop_for_other_families(monkeypatch):
    registry, _ = _stub_hunyuan15_registry(monkeypatch)
    _ensure_block_metadata_registered(_MixinTransformer())
    assert registry.registry == {}


def test_registration_failure_is_swallowed(monkeypatch):
    # diffusers internals moved / import fails -> best-effort no-op; enable_cache then
    # surfaces its own error and the load runs uncached, exactly as before the patch.
    monkeypatch.setitem(sys.modules, "diffusers.hooks._helpers", None)
    _ensure_block_metadata_registered(HunyuanVideo15Transformer3DModel())  # no raise


# ── stale child-registry invalidation after enable_cache ───────────────────────────
def test_invalidate_child_registry_cache_clears_stale_list():
    # An UNCACHED generation's cache_context call froze an EMPTY child list on the
    # transformer's HookRegistry; enable_cache installs block hooks _set_context would
    # then never reach ("No context is set" on the first cached forward). The helper
    # drops the stale cache so the next cache_context rebuilds it over the new hooks.
    t = _MixinTransformer()
    t._diffusers_hook = types.SimpleNamespace(_child_registries_cache = [])
    _invalidate_child_registry_cache(t)
    assert t._diffusers_hook._child_registries_cache is None


def test_invalidate_child_registry_cache_noops():
    _invalidate_child_registry_cache(_MixinTransformer())  # no _diffusers_hook
    t = _MixinTransformer()
    t._diffusers_hook = types.SimpleNamespace(_child_registries_cache = None)
    _invalidate_child_registry_cache(t)  # nothing cached yet
    assert t._diffusers_hook._child_registries_cache is None


def test_apply_step_cache_registers_and_invalidates_for_hunyuan15(monkeypatch):
    _stub_diffusers(monkeypatch)
    registry, block_cls = _stub_hunyuan15_registry(monkeypatch)
    t = HunyuanVideo15Transformer3DModel()
    t._diffusers_hook = types.SimpleNamespace(_child_registries_cache = [])
    engaged = apply_step_cache(_pipe(t), mode = "fbcache")
    assert engaged == TC_FBCACHE
    assert t.enabled_with.threshold == DEFAULT_FBCACHE_THRESHOLD
    assert block_cls in registry.registry  # metadata registered before enable_cache
    assert t._diffusers_hook._child_registries_cache is None  # stale cache dropped


# ── magcache mode (per-family auto cache) ──────────────────────────────────────────
from core.inference.diffusion_cache import (  # noqa: E402
    DEFAULT_MAGCACHE_THRESHOLD,
    MAGCACHE_MAX_SKIP_STEPS,
    MAGCACHE_RETENTION_RATIO,
    TC_MAGCACHE,
    _MAGCACHE_FAMILY_RATIOS,
    auto_cache_mode,
)


class _MagConfig:
    def __init__(self, threshold, max_skip_steps, retention_ratio, num_inference_steps, mag_ratios):
        self.threshold = threshold
        self.max_skip_steps = max_skip_steps
        self.retention_ratio = retention_ratio
        self.num_inference_steps = num_inference_steps
        self.mag_ratios = mag_ratios


def _stub_diffusers_with_magcache(monkeypatch):
    _stub_diffusers(monkeypatch)
    hooks = sys.modules["diffusers.hooks"]
    hooks.MagCacheConfig = _MagConfig


def test_normalize_accepts_magcache():
    assert normalize_transformer_cache("magcache") == TC_MAGCACHE
    assert normalize_transformer_cache("MagCache") == TC_MAGCACHE


def test_auto_cache_mode_per_family():
    # HunyuanVideo-1.5: FBCache free-runs (no cap / no error budget) and derails the
    # trajectory (measured LPIPS 0.54 at its default threshold), so auto engages the
    # bounded MagCache there. Wan2.2-TI2V-5B: both modes hold composition, but MagCache
    # dominates the accuracy/speed frontier (1.65x at pairwise LPIPS 0.034 vs FBCache's
    # 1.49x at 0.031; 1.73x/0.044 vs 1.71x/0.083 at the fast points), so auto engages
    # MagCache with its calibrated curve. Wan2.2-A14B measured the OTHER way (FBCache
    # 0.12 at 2.88x/0.128 dominates balanced MagCache's 1.80x/0.145; the 16-step
    # high-noise expert starves MagCache's budget), so the MoE stays on FBCache. Every
    # other family keeps the measured FBCache default.
    assert auto_cache_mode("hunyuanvideo-1.5") == TC_MAGCACHE
    assert auto_cache_mode("hunyuanvideo-1.5-720p") == TC_MAGCACHE
    assert auto_cache_mode("HunyuanVideo-1.5-720p") == TC_MAGCACHE
    assert auto_cache_mode("wan2.2-ti2v-5b") == TC_MAGCACHE
    for other in (None, "", "flux", "wan2.2-t2v-a14b", "ltx-2", "z-image"):
        assert auto_cache_mode(other) == TC_FBCACHE


def test_magcache_families_have_calibrated_ratios():
    # Every family the auto policy routes to magcache must ship a calibrated curve, or
    # the auto default silently runs uncached (apply_step_cache checks the table).
    from core.inference.diffusion_cache import _FAMILY_AUTO_CACHE_MODE
    for fam, mode in _FAMILY_AUTO_CACHE_MODE.items():
        if mode == TC_MAGCACHE:
            ratios = _MAGCACHE_FAMILY_RATIOS[fam]
            assert len(ratios) == 50  # the default 50-step schedule they were calibrated on
            assert all(0.5 < r < 1.5 for r in ratios)


def test_magcache_engages_with_family_curve(monkeypatch):
    _stub_diffusers_with_magcache(monkeypatch)
    t = _MixinTransformer()
    engaged = apply_step_cache(_pipe(t), mode = "magcache", family = "hunyuanvideo-1.5-720p", steps = 50)
    assert engaged == TC_MAGCACHE
    cfg = t.enabled_with
    assert cfg.threshold == DEFAULT_MAGCACHE_THRESHOLD
    assert cfg.max_skip_steps == MAGCACHE_MAX_SKIP_STEPS
    assert cfg.retention_ratio == MAGCACHE_RETENTION_RATIO
    assert cfg.num_inference_steps == 50
    assert cfg.mag_ratios == list(_MAGCACHE_FAMILY_RATIOS["hunyuanvideo-1.5-720p"])
    # The marker carries the step count so the auto toggle re-engages on a change.
    assert t._unsloth_step_cache == f"magcache@{DEFAULT_MAGCACHE_THRESHOLD}#s50"


def test_magcache_without_calibration_runs_uncached(monkeypatch):
    # No silent FBCache fallback: the family was routed to magcache exactly because
    # FBCache derails it, so an uncalibrated family must run uncached instead.
    _stub_diffusers_with_magcache(monkeypatch)
    t = _MixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "magcache", family = "flux", steps = 50) is None
    assert t.enabled_with is None


def test_magcache_without_steps_runs_uncached(monkeypatch):
    _stub_diffusers_with_magcache(monkeypatch)
    t = _MixinTransformer()
    assert apply_step_cache(_pipe(t), mode = "magcache", family = "hunyuanvideo-1.5-720p") is None
    assert t.enabled_with is None


def test_magcache_explicit_threshold_wins(monkeypatch):
    _stub_diffusers_with_magcache(monkeypatch)
    t = _MixinTransformer()
    apply_step_cache(
        _pipe(t),
        mode = "magcache",
        family = "hunyuanvideo-1.5-720p",
        steps = 30,
        threshold = 0.24,
    )
    assert t.enabled_with.threshold == 0.24


def test_toggle_engages_family_magcache(monkeypatch):
    _stub_diffusers_with_magcache(monkeypatch)
    t = _ToggleTransformer()
    mode = maybe_toggle_step_cache(
        _pipe(t), steps = 30, mode = TC_MAGCACHE, family = "hunyuanvideo-1.5-720p"
    )
    assert mode == TC_MAGCACHE and t.enables == 1
    assert t.enabled_with.num_inference_steps == 30


def test_toggle_magcache_reengages_on_step_change(monkeypatch):
    # MagCache interpolates its calibrated curve over the CONFIGURED step count, so a
    # step-count change must disable + re-enable; the same count stays idempotent.
    _stub_diffusers_with_magcache(monkeypatch)
    t = _ToggleTransformer()
    maybe_toggle_step_cache(_pipe(t), steps = 30, mode = TC_MAGCACHE, family = "hunyuanvideo-1.5-720p")
    maybe_toggle_step_cache(_pipe(t), steps = 30, mode = TC_MAGCACHE, family = "hunyuanvideo-1.5-720p")
    assert t.enables == 1 and t.disables == 0  # idempotent at the same count
    mode = maybe_toggle_step_cache(
        _pipe(t), steps = 50, mode = TC_MAGCACHE, family = "hunyuanvideo-1.5-720p"
    )
    assert mode == TC_MAGCACHE and t.disables == 1 and t.enables == 2
    assert t.enabled_with.num_inference_steps == 50


def test_toggle_magcache_disengages_below_bar(monkeypatch):
    _stub_diffusers_with_magcache(monkeypatch)
    t = _ToggleTransformer()
    maybe_toggle_step_cache(_pipe(t), steps = 30, mode = TC_MAGCACHE, family = "hunyuanvideo-1.5-720p")
    mode = maybe_toggle_step_cache(
        _pipe(t), steps = 8, mode = TC_MAGCACHE, family = "hunyuanvideo-1.5-720p"
    )
    assert mode is None and t.disables == 1


# ── per-expert magcache curves (dual-expert MoE, Wan2.2-A14B) ───────────────────────
from core.inference.diffusion_cache import (  # noqa: E402
    _MAGCACHE_CALIBRATION_STEPS,
    _magcache_ratio_key,
)


def test_magcache_ratio_key_primary_and_expert():
    # The primary transformer resolves the bare family key (back-compat with every
    # single-DiT family); a second expert resolves "family::expert".
    assert _magcache_ratio_key("wan2.2-t2v-a14b", None) == "wan2.2-t2v-a14b"
    assert _magcache_ratio_key("wan2.2-t2v-a14b", "transformer") == "wan2.2-t2v-a14b"
    assert (
        _magcache_ratio_key("Wan2.2-T2V-A14B", "transformer_2")
        == "wan2.2-t2v-a14b::transformer_2"
    )


def test_magcache_expert_resolves_its_own_curve(monkeypatch):
    _stub_diffusers_with_magcache(monkeypatch)
    from core.inference import diffusion_cache as dc_mod

    primary_curve = tuple([1.0] * 15)
    expert_curve = tuple([0.99] * 35)
    monkeypatch.setitem(dc_mod._MAGCACHE_FAMILY_RATIOS, "fam-moe", primary_curve)
    monkeypatch.setitem(
        dc_mod._MAGCACHE_FAMILY_RATIOS, "fam-moe::transformer_2", expert_curve
    )
    t = _MixinTransformer()
    engaged = apply_step_cache(
        _pipe(t), mode = "magcache", family = "fam-moe", steps = 50,
        expert = "transformer_2",
    )
    assert engaged == TC_MAGCACHE
    assert t.enabled_with.mag_ratios == list(expert_curve)


def test_magcache_expert_subcurve_scales_step_count(monkeypatch):
    # An expert sub-curve covers only that expert's slice of the calibration schedule
    # (the hook counts the expert's OWN forwards from 0), so the configured step count
    # scales by steps / calibration-steps: a 35-of-50 sub-curve at a 30-step request
    # configures round(35 * 30 / 50) = 21 steps -- NOT the full 30.
    _stub_diffusers_with_magcache(monkeypatch)
    from core.inference import diffusion_cache as dc_mod

    expert_curve = tuple([0.99] * 35)
    monkeypatch.setitem(
        dc_mod._MAGCACHE_FAMILY_RATIOS, "fam-moe::transformer_2", expert_curve
    )
    t = _MixinTransformer()
    apply_step_cache(
        _pipe(t), mode = "magcache", family = "fam-moe", steps = 30,
        expert = "transformer_2",
    )
    assert t.enabled_with.num_inference_steps == round(35 * 30 / _MAGCACHE_CALIBRATION_STEPS)
    # At the calibration step count itself the sub-curve maps 1:1.
    t2 = _MixinTransformer()
    apply_step_cache(
        _pipe(t2), mode = "magcache", family = "fam-moe",
        steps = _MAGCACHE_CALIBRATION_STEPS, expert = "transformer_2",
    )
    assert t2.enabled_with.num_inference_steps == 35


def test_magcache_full_curve_keeps_requested_steps(monkeypatch):
    # A full 50-entry curve interpolates to the requested count directly (the
    # single-DiT behaviour is unchanged by the expert plumbing).
    _stub_diffusers_with_magcache(monkeypatch)
    t = _MixinTransformer()
    apply_step_cache(
        _pipe(t), mode = "magcache", family = "wan2.2-ti2v-5b", steps = 30,
        expert = "transformer",
    )
    assert t.enabled_with.num_inference_steps == 30
    assert len(t.enabled_with.mag_ratios) == _MAGCACHE_CALIBRATION_STEPS


def test_magcache_expert_without_curve_runs_uncached(monkeypatch):
    # A second expert with no calibrated sub-curve must run uncached, NOT silently
    # reuse the primary's curve (the experts split the schedule; the curves differ).
    _stub_diffusers_with_magcache(monkeypatch)
    from core.inference import diffusion_cache as dc_mod

    monkeypatch.setitem(dc_mod._MAGCACHE_FAMILY_RATIOS, "fam-moe", tuple([1.0] * 15))
    t = _MixinTransformer()
    assert (
        apply_step_cache(
            _pipe(t), mode = "magcache", family = "fam-moe", steps = 50,
            expert = "transformer_2",
        )
        is None
    )
    assert t.enabled_with is None


def test_toggle_threads_expert_through(monkeypatch):
    _stub_diffusers_with_magcache(monkeypatch)
    from core.inference import diffusion_cache as dc_mod

    expert_curve = tuple([0.98] * 35)
    monkeypatch.setitem(
        dc_mod._MAGCACHE_FAMILY_RATIOS, "fam-moe::transformer_2", expert_curve
    )
    t = _ToggleTransformer()
    mode = maybe_toggle_step_cache(
        _pipe(t), steps = 50, mode = TC_MAGCACHE, family = "fam-moe",
        expert = "transformer_2",
    )
    assert mode == TC_MAGCACHE
    assert t.enabled_with.mag_ratios == list(expert_curve)


# ── cache quality presets (speed/accuracy knob) ────────────────────────────────────
from core.inference.diffusion_cache import (  # noqa: E402
    CACHE_QUALITY_LEVELS,
    CQ_BALANCED,
    CQ_FAST,
    CQ_QUALITY,
    _FBCACHE_QUALITY_THRESHOLDS,
    _MAGCACHE_QUALITY_PRESETS,
    normalize_cache_quality,
)


def test_normalize_cache_quality_unset_and_auto_are_none():
    for value in (None, "", "  ", "auto", "AUTO"):
        assert normalize_cache_quality(value) is None


def test_normalize_cache_quality_levels_and_casing():
    assert normalize_cache_quality("quality") == CQ_QUALITY
    assert normalize_cache_quality("  Balanced ") == CQ_BALANCED
    assert normalize_cache_quality("FAST") == CQ_FAST


def test_normalize_cache_quality_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_cache_quality("ultra")


def test_quality_preset_tables_cover_every_level():
    # A missing preset row would KeyError at engage time; the tables and the public
    # levels tuple must stay in lockstep.
    assert set(_MAGCACHE_QUALITY_PRESETS) == set(CACHE_QUALITY_LEVELS)
    assert set(_FBCACHE_QUALITY_THRESHOLDS) == set(CACHE_QUALITY_LEVELS)


def test_balanced_presets_match_the_preknob_defaults():
    # "balanced" IS the pre-knob shipped behaviour: a load without the knob must be
    # byte-identical to the round-1 defaults.
    assert _MAGCACHE_QUALITY_PRESETS[CQ_BALANCED] == (
        DEFAULT_MAGCACHE_THRESHOLD,
        MAGCACHE_MAX_SKIP_STEPS,
        MAGCACHE_RETENTION_RATIO,
    )
    assert _FBCACHE_QUALITY_THRESHOLDS[CQ_BALANCED] == (
        DEFAULT_FBCACHE_THRESHOLD,
        QUANT_FBCACHE_THRESHOLD,
    )


def test_magcache_quality_preset_engages_conservative_params(monkeypatch):
    # Calibrated on HunyuanVideo-1.5-720p (50 steps): thr 0.06 / cap 2 / retention 0.3 =
    # 1.11x at pairwise LPIPS 0.057 vs balanced's 1.49x at 0.126.
    _stub_diffusers_with_magcache(monkeypatch)
    t = _MixinTransformer()
    engaged = apply_step_cache(
        _pipe(t),
        mode = "magcache",
        family = "hunyuanvideo-1.5-720p",
        steps = 50,
        quality = "quality",
    )
    assert engaged == TC_MAGCACHE
    thr, cap, retention = _MAGCACHE_QUALITY_PRESETS[CQ_QUALITY]
    assert t.enabled_with.threshold == thr
    assert t.enabled_with.max_skip_steps == cap
    assert t.enabled_with.retention_ratio == retention


def test_magcache_explicit_threshold_beats_the_preset(monkeypatch):
    # The preset still supplies the skip cap / retention window, but a pinned threshold
    # wins (the documented contract of transformer_cache_threshold).
    _stub_diffusers_with_magcache(monkeypatch)
    t = _MixinTransformer()
    apply_step_cache(
        _pipe(t),
        mode = "magcache",
        family = "hunyuanvideo-1.5-720p",
        steps = 50,
        quality = "fast",
        threshold = 0.05,
    )
    assert t.enabled_with.threshold == 0.05
    assert t.enabled_with.max_skip_steps == _MAGCACHE_QUALITY_PRESETS[CQ_FAST][1]


def test_fbcache_quality_preset_thresholds(monkeypatch):
    _stub_diffusers(monkeypatch)
    dense_thr, quant_thr = _FBCACHE_QUALITY_THRESHOLDS[CQ_QUALITY]
    t = _MixinTransformer()
    apply_step_cache(_pipe(t), mode = "fbcache", quality = "quality")
    assert t.enabled_with.threshold == dense_thr
    t2 = _MixinTransformer()
    apply_step_cache(_pipe(t2), mode = "fbcache", quality = "quality", quant_active = True)
    assert t2.enabled_with.threshold == quant_thr


def test_apply_step_cache_rejects_bad_quality(monkeypatch):
    _stub_diffusers(monkeypatch)
    with pytest.raises(ValueError):
        apply_step_cache(_pipe(_MixinTransformer()), mode = "fbcache", quality = "bogus")


def test_toggle_threads_quality_through(monkeypatch):
    _stub_diffusers_with_magcache(monkeypatch)
    t = _ToggleTransformer()
    maybe_toggle_step_cache(
        _pipe(t),
        steps = 30,
        mode = TC_MAGCACHE,
        family = "hunyuanvideo-1.5-720p",
        quality = "quality",
    )
    assert t.enabled_with.threshold == _MAGCACHE_QUALITY_PRESETS[CQ_QUALITY][0]
    assert t.enabled_with.max_skip_steps == _MAGCACHE_QUALITY_PRESETS[CQ_QUALITY][1]


# ── compiled cache-hook inners (regional compile x step cache composition) ──────────
import functools  # noqa: E402

from core.inference.diffusion_cache import (  # noqa: E402
    _compile_hooked_block_inners,
    _restore_hooked_block_inners,
    auto_cache_quality,
)


def test_auto_cache_quality_per_family():
    assert auto_cache_quality("hunyuanvideo-1.5") == CQ_QUALITY
    assert auto_cache_quality("HunyuanVideo-1.5-720p") == CQ_QUALITY
    for other in (None, "", "flux", "wan2.2-ti2v-5b", "ltx-2"):
        assert auto_cache_quality(other) == CQ_BALANCED


class _BoundInner:
    """Provides a plain bound method for fn_ref.original_forward (__self__ present)."""

    def forward(self, *args, **kwargs):
        return "eager"


def _hooked_block(
    *,
    compiled = True,
    hook_name = "mag_cache_block_hook",
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
    # An eager-tier load has no _compiled_call_impl: the hook must stay untouched
    # (compiling the inner would ADD compile where the user chose eager).
    _stub_torch_compile(monkeypatch)
    block, hook, orig = _hooked_block(compiled = False)
    assert _compile_hooked_block_inners(_fake_dit([block])) == 0
    assert hook.fn_ref.original_forward is orig


def test_arming_skips_partial_captured_inner(monkeypatch):
    # A stacked hook chain (e.g. group offload) captures a functools.partial, not the
    # plain bound method; arming would compile the wrong layer of the chain.
    _stub_torch_compile(monkeypatch)
    block, hook, orig = _hooked_block(bound = False)
    assert _compile_hooked_block_inners(_fake_dit([block])) == 0
    assert hook.fn_ref.original_forward is orig


def test_arming_covers_every_cache_hook_family(monkeypatch):
    # FBCache is the image cache today, but the hook-name table already covers the
    # MagCache layout too (same fn_ref shape), so a future mode arms for free.
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


def test_disengage_restores_inners_before_disable(monkeypatch):
    # remove_hook splices fn_ref.original_forward back into module.forward, so the
    # compiled wrapper must be swapped out BEFORE disable_cache runs.
    from core.inference import diffusion_cache as dc_mod

    order = []

    class _T(_MixinTransformer):
        def disable_cache(self):
            order.append("disable")

        def modules(self):
            order.append("restore-walk")
            return []

    t = _T()
    t._unsloth_step_cache = "magcache@0.12#s50"
    assert dc_mod._disengage_step_cache(t, reason = "test") is True
    assert order == ["restore-walk", "disable"]


def test_apply_step_cache_arms_compiled_blocks_on_toggle(monkeypatch):
    # The generation-time toggle engages the cache AFTER the load already compiled the
    # blocks; apply_step_cache must arm the fresh hooks itself.
    _stub_diffusers_with_magcache(monkeypatch)
    _stub_torch_compile(monkeypatch)
    block, hook, orig = _hooked_block()

    class _T(_MixinTransformer):
        def modules(self):
            return [block]

    t = _T()
    engaged = apply_step_cache(_pipe(t), mode = "magcache", family = "hunyuanvideo-1.5-720p", steps = 50)
    assert engaged == TC_MAGCACHE
    assert hook.fn_ref.original_forward is not orig
    assert hook._unsloth_orig_inner is orig


def test_toggle_disable_restores_inners_before_disable(monkeypatch):
    # remove_hook splices fn_ref.original_forward back into module.forward, so the
    # compiled wrapper must be swapped out BEFORE disable_cache runs.
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
    # enable_cache can fail after hooking (and arming) some blocks; the partial-hook
    # cleanup must un-arm them before disable_cache splices original_forward back.
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
    # diffusers 0.39 caches the child-registry list on first cache_context use; an
    # UNCACHED generation already populates it (empty), so a later toggle-time
    # enable_cache would install hooks the context never reaches ("No context is set").
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
