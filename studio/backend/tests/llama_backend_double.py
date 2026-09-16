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

    # The runtime fields /status mirrors off the backend, at the values LlamaCppBackend.__init__
    # gives them. These are not Optional in the response, so a double that answers None for an
    # unset attribute makes /status fail validation rather than report a default. The same block
    # was being re-declared by hand in the per-test fakes, which is how four reasoning fields
    # added in one place reached CI as three red tests naming none of them.
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
    # Private, because that is the only name the real backend has for these and so the name
    # _llama_runtime_fields falls back to. A public one here would be an attribute production
    # cannot read.
    _is_audio = False
    _has_audio_input = False
    _has_video_input = False
    _disable_vision = False
    _vision_disabled_by_user = False
    _requested_reasoning_budget = -1
    _requested_reasoning_budget_message = ""

    def __getattr__(self, name):
        """Anything unset reads as the real backend's None, except where a private slot holds it.

        Several doubles carried a blanket None for this, which reads right until a runtime field
        lives only under ``_name`` on the real backend: ``_llama_runtime_fields`` looks up the
        public name first and falls back to the private one, so answering None publicly hides the
        slot that has the value, and /status is handed a None its response model rejects.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        if not name.startswith("_"):
            try:
                return object.__getattribute__(self, f"_{name}")
            except AttributeError:
                pass
        return None
