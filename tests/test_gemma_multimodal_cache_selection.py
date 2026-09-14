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

import pytest

from unsloth.models.vision import (
    _BIDIRECTIONAL_MASK_BUILDERS,
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


@pytest.mark.parametrize("media_kwarg", ["pixel_values", "pixel_values_videos", "input_features"])
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
