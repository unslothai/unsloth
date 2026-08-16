# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does a reload request that OMITS disable_vision inherit it, or reset it?

``tensor_parallel`` needed explicit ``model_fields_set`` handling for exactly
this shape (a non-Optional bool defaulting False, where "omitted" and "explicit
false" are indistinguishable by getattr). ``disable_vision`` has the same shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement  # noqa: F401,E402  (installs the import stubs)

from core.inference.llama_cpp import GgufLoadIntent  # noqa: E402
from models.inference import LoadRequest  # noqa: E402
from routes.inference import _gguf_request_intent  # noqa: E402


def _intent_from(request: LoadRequest, source: GgufLoadIntent) -> GgufLoadIntent:
    return _gguf_request_intent(
        source,
        request,
        chat_template_override = None,
        extra_args = None,
        gpu_ids = None,
        n_parallel = 1,
    )


def test_an_omitted_disable_vision_is_not_an_explicit_false():
    """The request model must be able to tell the two apart at all."""
    omitted = LoadRequest(model_path = "m")
    explicit = LoadRequest(model_path = "m", disable_vision = False)
    assert "disable_vision" not in omitted.model_fields_set
    assert "disable_vision" in explicit.model_fields_set


def test_a_reload_omitting_the_field_resets_it_rather_than_inheriting():
    """Pinned deliberately: omitted means False, it does NOT inherit.

    ``llama_extra_args`` and ``gpu_memory_mode`` inherit from the resident
    intent when omitted, so the asymmetry is worth stating rather than
    discovering. It is the right way round here, and the alternative is worse:
    ``model_override_load_kwargs`` only emits ``disable_vision`` when the
    override store holds a True, so a user who REMOVES the override sends a
    payload with the key absent. If absence inherited, turning Vision back on
    from the model page could never take effect on a resident model.

    The cost is that a third-party client driving /load directly has to resend
    the field on every reload of a text-only-loaded vision model, or the
    projector comes back. Every in-product load path sends it unconditionally
    (shared-composer.tsx, use-chat-model-runtime.ts, chat-adapter.ts), so this
    is an API-shape property rather than a live defect.
    """
    resident = GgufLoadIntent(model_identifier = "m", is_vision = True, disable_vision = True)
    request = LoadRequest(model_path = "m", max_seq_length = 8192)

    assert _intent_from(request, resident).disable_vision is False


def test_an_explicit_false_does_turn_vision_back_on():
    resident = GgufLoadIntent(model_identifier = "m", is_vision = True, disable_vision = True)
    request = LoadRequest(model_path = "m", disable_vision = False)

    assert _intent_from(request, resident).disable_vision is False


def test_a_fresh_load_still_defaults_off():
    fresh = GgufLoadIntent(model_identifier = "m", is_vision = True)
    request = LoadRequest(model_path = "m")

    assert _intent_from(request, fresh).disable_vision is False
