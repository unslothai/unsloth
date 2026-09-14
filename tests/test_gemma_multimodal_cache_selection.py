"""Cache selection for multimodal generation (unslothai/unsloth#6028).

A static cache makes transformers skip mask materialisation at prefill and rely
on `is_causal`, which drops the bidirectional image/audio block overlay the Gemma
multimodal families build. Image tokens then attend causally and spatial
grounding degrades. `generate` must fall back to the dynamic cache for those
requests, and only those.
"""
import sys
import types

import pytest

from unsloth.models.vision import (
    _BIDIRECTIONAL_MASK_BUILDERS,
    _needs_bidirectional_multimodal_mask,
)


def _make_model(module_name, has_helper, helper=None):
    module = types.ModuleType(module_name)
    if has_helper:
        setattr(module, helper or _BIDIRECTIONAL_MASK_BUILDERS[0],
                lambda *args, **kwargs: None)
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


@pytest.mark.parametrize(
    "media_kwarg", ["pixel_values", "pixel_values_videos", "input_features"]
)
def test_media_request_on_gemma_uses_dynamic_cache(gemma_like, media_kwarg):
    assert _needs_bidirectional_multimodal_mask(gemma_like, {media_kwarg: object()})


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
    model = _make_model(name, True, helper=helper)
    try:
        assert _needs_bidirectional_multimodal_mask(model, {"pixel_values": object()})
    finally:
        sys.modules.pop(name, None)


def test_real_gemma_modules_expose_the_helper():
    """Pin the upstream symbols the guard keys on, so a rename is caught here."""
    pytest.importorskip("transformers")
    seen = False
    for name in ("gemma3", "gemma4", "gemma4_unified"):
        try:
            module = __import__(
                f"transformers.models.{name}.modeling_{name}", fromlist=["x"]
            )
        except Exception:
            continue
        seen = True
        assert any(hasattr(module, h) for h in _BIDIRECTIONAL_MASK_BUILDERS), name
    if not seen:
        pytest.skip("no Gemma multimodal module in this transformers version")
