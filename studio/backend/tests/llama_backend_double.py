# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One shared attribute surface for the llama.cpp backend doubles.

Nine hand-written doubles across five test files re-declared the same attribute block, so a new
attribute read off ``llama_backend`` in production breaks whichever doubles were not updated. #8700
did exactly that with ``context_length``: 19 red tests, some a plain ``AttributeError`` under
TestClient, some a 20-second timeout in the slot-release tests that reads as "the slot was never
released".

Inherit from this class rather than re-declaring the block, and add new shared attributes here.
``test_llama_backend_double.py`` keeps it honest in both directions.
"""

from __future__ import annotations

from typing import Optional

from models.inference import _InferenceRuntimeFields

# Taken from the model, not listed: an Optional field added there needs no edit here, and a
# required one is named by the canary.
_RUNTIME_FIELDS = frozenset(_InferenceRuntimeFields.model_fields)


class FakeLlamaCppBackend:
    """The attributes ``routes/inference.py`` reads off a loaded GGUF backend.

    Subclasses override what their scenario needs (notably ``supports_tools``, which selects the
    tool loop) and supply the generator methods: behaviour is per-test, the attribute surface is not.
    """

    is_loaded = True
    model_identifier = "test/model.gguf"
    is_vision = False
    supports_tools = False
    # Read unguarded on the chat-completions path for the monitor's context-usage readout. None is
    # what the real property answers before a model is loaded; context-usage tests set a number.
    context_length: Optional[int] = None

    # Runtime fields /status mirrors, at LlamaCppBackend.__init__'s values. None is not one of
    # them: the response rejects it. Re-declared by hand in the per-test fakes until now, which
    # is how four reasoning fields added in one place reached CI as three red tests naming none.
    is_diffusion = False
    supports_reasoning = False
    reasoning_always_on = False
    reasoning_style = "enable_thinking"
    reasoning_effort_levels: list = []
    reasoning_budget = -1
    reasoning_budget_message = ""
    supports_preserve_thinking = False
    tensor_parallel = False
    gpu_memory_mode = "auto"
    gpu_layers = -1
    n_cpu_moe = 0
    n_moe_layers = 0
    # Private: the only name the real backend has for these, so the one production falls back to.
    _is_audio = False
    _has_audio_input = False
    _has_video_input = False
    _disable_vision = False
    _vision_disabled_by_user = False
    _requested_reasoning_budget = -1
    _requested_reasoning_budget_message = ""
    # Read straight off the backend, not through the model-field loop, so an absent one is an
    # AttributeError before the drift check can name it.
    requested_spec_mode = None
    requested_parallel_slots = 1
    effective_parallel_slots = 1
    requested_n_ctx = 0
    requested_extra_args = None
    spec_fallback_reason = None
    spec_drafter_kind = None
    last_load_warning = None

    def __getattr__(self, name):
        """Runtime fields answer as the real backend does; anything else raises.

        Scoped on purpose. A blanket None reads right until a caller defaults an absent
        capability to something else: the route falls back to ``supports_tools`` for
        ``supports_tool_passthrough``, and None is not absent, so the default never runs.

        The private slot comes first, since ``_llama_runtime_fields`` falls back to ``_name``
        and a public None would hide the slot holding the value.
        """
        if name not in _RUNTIME_FIELDS:
            raise AttributeError(name)
        try:
            return object.__getattribute__(self, f"_{name}")
        except AttributeError:
            return None
