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
    _is_audio = False
    is_vision = False
    supports_tools = False
    # Read unguarded on the chat-completions path for the monitor's context-usage readout. None is
    # what the real property answers before a model is loaded; context-usage tests set a number.
    context_length: Optional[int] = None
