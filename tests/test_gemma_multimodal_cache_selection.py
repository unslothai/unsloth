# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cache selection for multimodal generation (unslothai/unsloth#6028).

A static cache makes transformers skip mask materialisation at prefill and rely
on `is_causal`, which drops the bidirectional image/audio block overlay the Gemma
multimodal families build. Image tokens then attend causally and spatial
grounding degrades. `generate` must fall back to the dynamic cache for those
requests, and only those.
"""

import sys
import types

import torch

import pytest

from unsloth.models.vision import (
    _BIDIRECTIONAL_MASK_BUILDERS,
    _MEDIA_TOKEN_TYPES,
    _STATIC_CACHE_IMPLEMENTATIONS,
    _dynamic_cache_choice,
    _needs_bidirectional_multimodal_mask,
)

# Both helpers landed in transformers 5.10; earlier versions expose neither.
_FIRST_VERSION_WITH_BLOCK_MASK = "5.10.0"


def _make_model(
    module_name,
    has_helper,
    helper = None,
):
    module = types.ModuleType(module_name)
    if has_helper:
        setattr(module, helper or _BIDIRECTIONAL_MASK_BUILDERS[0], lambda *args, **kwargs: None)
    sys.modules[module_name] = module

    model_cls = type("Model", (), {})
    model_cls.__module__ = module_name
    return model_cls()


@pytest.fixture
def gemma_like(request):
    name = f"_fake_gemma_{request.node.name}"
    yield _make_model(name, True)
    sys.modules.pop(name, None)


@pytest.fixture
def causal_vlm(request):
    name = f"_fake_qwen_{request.node.name}"
    yield _make_model(name, False)
    sys.modules.pop(name, None)


@pytest.mark.parametrize("media_kwarg", ["pixel_values", "pixel_values_videos"])
def test_media_request_on_gemma_uses_dynamic_cache(gemma_like, media_kwarg):
    assert _needs_bidirectional_multimodal_mask(gemma_like, {media_kwarg: object()})


def test_audio_only_keeps_the_static_cache(gemma_like):
    """get_block_sequence_ids_for_mask blocks token types 1 and 2 only, so audio
    stays causal and must not lose the compiled path."""
    assert not _needs_bidirectional_multimodal_mask(gemma_like, {"input_features": object()})


def test_text_only_keeps_static_cache(gemma_like):
    assert not _needs_bidirectional_multimodal_mask(gemma_like, {"input_ids": object()})


def test_none_media_kwarg_keeps_static_cache(gemma_like):
    assert not _needs_bidirectional_multimodal_mask(gemma_like, {"pixel_values": None})


def test_causal_vlm_keeps_static_cache(causal_vlm):
    """Qwen2-VL, Llava and PaliGemma have no bidirectional block overlay, so they
    must not lose the static-cache path."""
    assert not _needs_bidirectional_multimodal_mask(causal_vlm, {"pixel_values": object()})


def test_unknown_module_does_not_raise():
    orphan_cls = type("Orphan", (), {})
    orphan_cls.__module__ = "_module_that_does_not_exist"
    assert not _needs_bidirectional_multimodal_mask(orphan_cls(), {"pixel_values": object()})


@pytest.mark.parametrize("helper", _BIDIRECTIONAL_MASK_BUILDERS)
def test_either_mask_builder_gates_the_guard(request, helper):
    """The two names span transformers 5.10 to current, so either one alone has
    to be enough."""
    name = f"_fake_helper_{request.node.name}"
    model = _make_model(name, True, helper = helper)
    try:
        assert _needs_bidirectional_multimodal_mask(model, {"pixel_values": object()})
    finally:
        sys.modules.pop(name, None)


def test_real_gemma_modules_expose_the_helper():
    """Pin the upstream symbols the guard keys on, so a rename is caught here.

    These helpers landed in transformers 5.10. Before that neither name exists
    (measured on 4.57.6 and 5.5.0), the guard is inert by design, and Gemma keeps
    the static path it always had, so there is nothing to pin.
    """
    transformers = pytest.importorskip("transformers")
    from packaging.version import InvalidVersion, Version

    try:
        new_enough = Version(transformers.__version__) >= Version(_FIRST_VERSION_WITH_BLOCK_MASK)
    except InvalidVersion:
        pytest.skip(f"cannot order version {transformers.__version__!r}")
    if not new_enough:
        pytest.skip(f"no block mask helper before transformers {_FIRST_VERSION_WITH_BLOCK_MASK}")

    seen = False
    for name in ("gemma3", "gemma4", "gemma4_unified"):
        try:
            module = __import__(f"transformers.models.{name}.modeling_{name}", fromlist = ["x"])
        except Exception:
            continue
        seen = True
        assert any(hasattr(module, h) for h in _BIDIRECTIONAL_MASK_BUILDERS), name
    if not seen:
        pytest.skip("no Gemma multimodal module in this transformers version")


def test_causal_vlms_never_expose_a_mask_builder():
    """The other half of the gate: these must stay on the static path on every
    version, so neither name may appear on them."""
    pytest.importorskip("transformers")
    for name in ("qwen2_vl", "llava", "paligemma"):
        try:
            module = __import__(f"transformers.models.{name}.modeling_{name}", fromlist = ["x"])
        except Exception:
            continue
        assert not any(hasattr(module, h) for h in _BIDIRECTIONAL_MASK_BUILDERS), name


class _Cfg:
    def __init__(self, bidir):
        self.use_bidirectional_attention = bidir

    def get_text_config(self):
        return self


def _gemma4_shaped(module, bidir):
    """A model whose create_masks_for_generate takes mm_token_type_ids, which is
    the Gemma 4 form and the one upstream gates on the config value."""

    class Model:
        config = _Cfg(bidir)

        @staticmethod
        def create_masks_for_generate(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            mm_token_type_ids = None,
            **kwargs,
        ):
            return None

    Model.__module__ = module.__name__
    return Model()


def _gemma3_shaped(module):
    """Gemma 3 takes token_type_ids and overlays regardless of the config flag,
    which it sets to False."""

    class Model:
        config = _Cfg(False)

        @staticmethod
        def create_masks_for_generate(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            token_type_ids = None,
            **kwargs,
        ):
            return None

    Model.__module__ = module.__name__
    return Model()


def test_causal_gemma4_variant_keeps_the_static_cache(gemma_like):
    """E2B / E4B leave use_bidirectional_attention None, so upstream builds no
    overlay and they must not lose the compiled path."""
    module = sys.modules[type(gemma_like).__module__]
    assert not _needs_bidirectional_multimodal_mask(
        _gemma4_shaped(module, None), {"pixel_values": object()}
    )


def test_vision_gemma4_variant_is_gated(gemma_like):
    module = sys.modules[type(gemma_like).__module__]
    assert _needs_bidirectional_multimodal_mask(
        _gemma4_shaped(module, "vision"), {"pixel_values": object()}
    )


def test_gemma3_is_gated_despite_the_flag_being_false(gemma_like):
    module = sys.modules[type(gemma_like).__module__]
    assert _needs_bidirectional_multimodal_mask(_gemma3_shaped(module), {"pixel_values": object()})


def test_precomputed_embeds_with_media_tokens_are_gated(gemma_like):
    """inputs_embeds carries no media kwarg; the token types are the signal."""
    module = sys.modules[type(gemma_like).__module__]
    model = _gemma4_shaped(module, "vision")
    ids = torch.tensor([[0, 0, 1, 1, 0]])
    assert _needs_bidirectional_multimodal_mask(model, {"mm_token_type_ids": ids})


def test_text_only_token_types_keep_the_static_cache(gemma_like):
    """The processor emits mm_token_type_ids for text-only prompts too, all
    zeros, so presence alone must not cost the static path."""
    module = sys.modules[type(gemma_like).__module__]
    model = _gemma4_shaped(module, "vision")
    ids = torch.zeros(1, 5, dtype = torch.long)
    assert not _needs_bidirectional_multimodal_mask(model, {"mm_token_type_ids": ids})


def test_gemma3_token_type_ids_are_read(gemma_like):
    module = sys.modules[type(gemma_like).__module__]
    model = _gemma3_shaped(module)
    assert _needs_bidirectional_multimodal_mask(
        model, {"token_type_ids": torch.tensor([[0, 1, 0]])}
    )
    assert not _needs_bidirectional_multimodal_mask(
        model, {"token_type_ids": torch.zeros(1, 3, dtype = torch.long)}
    )


def test_non_tensor_token_types_do_not_raise(gemma_like):
    module = sys.modules[type(gemma_like).__module__]
    model = _gemma4_shaped(module, "vision")
    assert not _needs_bidirectional_multimodal_mask(model, {"mm_token_type_ids": "not a tensor"})


def _resolve_cache_choice(force_dynamic, cache_implementation, kwargs):
    """The assignment block from unsloth_base_fast_generate, isolated."""
    forced = _dynamic_cache_choice(kwargs) if force_dynamic else None
    if "generation_config" in kwargs:
        kwargs["generation_config"].cache_implementation = (
            forced if force_dynamic else cache_implementation
        )
        return kwargs["generation_config"].cache_implementation
    kwargs["cache_implementation"] = forced if force_dynamic else cache_implementation
    return kwargs["cache_implementation"]


class _GenCfg:
    cache_implementation = "static"


def test_media_request_pins_the_literal_dynamic():
    """None is refilled from the model default by _prepare_generation_config, so
    the guard has to name the cache it wants."""
    assert _resolve_cache_choice(True, None, {}) == "dynamic"
    assert _resolve_cache_choice(True, None, {"generation_config": _GenCfg()}) == "dynamic"


def test_caller_generation_config_is_overridden():
    """The reported hole: a caller-supplied config whose field says static."""
    cfg = _GenCfg()
    assert cfg.cache_implementation == "static"
    _resolve_cache_choice(True, None, {"generation_config": cfg})
    assert cfg.cache_implementation == "dynamic"


def test_text_only_still_gets_the_static_cache():
    assert _resolve_cache_choice(False, "static", {}) == "static"
    cfg = _GenCfg()
    _resolve_cache_choice(False, "static", {"generation_config": cfg})
    assert cfg.cache_implementation == "static"


def test_none_is_preserved_when_not_forcing():
    """bfloat16 mixed precision clears the cache by setting None; that path must
    keep its own meaning."""
    assert _resolve_cache_choice(False, None, {}) is None


def test_dynamic_is_a_cache_implementation_transformers_accepts():
    """Pin the literal against upstream so a rename does not silently no-op."""
    import inspect

    generation_utils = pytest.importorskip("transformers.generation.utils")
    prepare = getattr(generation_utils.GenerationMixin, "_prepare_cache_for_generation", None)
    if prepare is None:
        pytest.skip("no _prepare_cache_for_generation in this transformers")
    source = inspect.getsource(prepare)
    assert '"dynamic"' in source or "'dynamic'" in source
    static = getattr(generation_utils, "ALL_STATIC_CACHE_IMPLEMENTATIONS", ())
    assert "dynamic" not in static


def test_audio_token_type_alone_keeps_the_static_cache(gemma_like):
    """Audio marks a different token type than the blocked 1 and 2."""
    module = sys.modules[type(gemma_like).__module__]
    model = _gemma4_shaped(module, "vision")
    audio_only = torch.tensor([[0, 3, 3, 0]])
    assert not _needs_bidirectional_multimodal_mask(model, {"mm_token_type_ids": audio_only})


def test_image_token_type_beside_audio_is_gated(gemma_like):
    module = sys.modules[type(gemma_like).__module__]
    model = _gemma4_shaped(module, "vision")
    mixed = torch.tensor([[0, 3, 1, 0]])
    assert _needs_bidirectional_multimodal_mask(model, {"mm_token_type_ids": mixed})


def test_video_token_type_is_gated(gemma_like):
    module = sys.modules[type(gemma_like).__module__]
    model = _gemma4_shaped(module, "vision")
    assert _needs_bidirectional_multimodal_mask(
        model, {"mm_token_type_ids": torch.tensor([[0, 2, 0]])}
    )


def test_explicit_static_kwarg_is_overridden_alongside_the_config():
    """Generation kwargs are applied after the config merge, so an explicit
    cache_implementation would otherwise outlive the config assignment."""
    cfg = _GenCfg()
    kwargs = {"generation_config": cfg, "cache_implementation": "static"}
    _resolve_cache_choice(True, None, kwargs)
    if kwargs.get("cache_implementation") is not None:
        kwargs["cache_implementation"] = "dynamic"
    assert cfg.cache_implementation == "dynamic"
    assert kwargs["cache_implementation"] == "dynamic"


def test_blocked_token_types_match_upstream():
    """Pin the values upstream actually blocks, so a change there is caught."""
    import inspect

    modeling = pytest.importorskip("transformers.models.gemma4_unified.modeling_gemma4_unified")
    builder = getattr(modeling, "get_block_sequence_ids_for_mask", None)
    if builder is None:
        pytest.skip("no block mask builder in this transformers version")
    source = inspect.getsource(builder)
    for value in _MEDIA_TOKEN_TYPES:
        assert f"== {value}" in source
    assert "== 3" not in source  # audio is not blocked


def _force_gate(local_cache_implementation, kwargs, is_media):
    """The force decision from unsloth_base_fast_generate, isolated.

    Deliberately independent of the local default: clearing it does not make the
    effective cache dynamic, since kwargs and the caller's generation_config are
    applied afterwards.
    """
    return kwargs.get("past_key_values") is None and is_media


def test_mixed_precision_still_forces_dynamic_for_media():
    """UNSLOTH_BFLOAT16_MIXED_PRECISION clears the local default to None. An
    explicit static kwarg would otherwise survive and bring the bug back."""
    kwargs = {"generation_config": _GenCfg(), "cache_implementation": "static"}
    assert _force_gate(None, kwargs, is_media = True)
    _resolve_cache_choice(True, None, kwargs)
    kwargs["cache_implementation"] = "dynamic"
    assert kwargs["generation_config"].cache_implementation == "dynamic"
    assert kwargs["cache_implementation"] == "dynamic"


def test_a_caller_supplied_cache_is_never_overridden():
    kwargs = {"past_key_values": object()}
    assert not _force_gate("static", kwargs, is_media = True)


def test_text_only_never_forces_even_with_a_cleared_default():
    assert not _force_gate(None, {}, is_media = False)


@pytest.mark.parametrize("requested", ["offloaded", "quantized"])
def test_a_growing_cache_the_caller_asked_for_survives(requested):
    """offloaded and quantized grow per step, so they keep the media mask.
    Downgrading them to plain dynamic would throw away the memory they buy."""
    assert _dynamic_cache_choice({"cache_implementation": requested}) == requested
    cfg = _GenCfg()
    cfg.cache_implementation = requested
    assert _dynamic_cache_choice({"generation_config": cfg}) == requested
    assert _resolve_cache_choice(True, None, {"generation_config": cfg}) == requested


@pytest.mark.parametrize("requested", ["static", "offloaded_static"])
def test_a_static_cache_the_caller_asked_for_is_replaced(requested):
    assert _dynamic_cache_choice({"cache_implementation": requested}) == "dynamic"
    cfg = _GenCfg()
    cfg.cache_implementation = requested
    assert _resolve_cache_choice(True, None, {"generation_config": cfg}) == "dynamic"


def test_the_kwarg_outranks_the_config_when_both_are_given():
    cfg = _GenCfg()
    cfg.cache_implementation = "offloaded"
    assert (
        _dynamic_cache_choice({"generation_config": cfg, "cache_implementation": "static"})
        == "dynamic"
    )


def test_static_implementations_match_upstream():
    """A name upstream treats as static must be replaced, not preserved."""
    configuration_utils = pytest.importorskip("transformers.generation.configuration_utils")
    upstream = getattr(configuration_utils, "ALL_STATIC_CACHE_IMPLEMENTATIONS", None)
    if upstream is None:
        pytest.skip("no ALL_STATIC_CACHE_IMPLEMENTATIONS in this transformers")
    assert set(upstream) <= set(_STATIC_CACHE_IMPLEMENTATIONS)
    assert "dynamic" not in _STATIC_CACHE_IMPLEMENTATIONS
