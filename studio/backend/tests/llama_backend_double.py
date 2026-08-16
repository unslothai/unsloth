# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One shared attribute surface for the llama.cpp backend doubles.

Nine hand-written doubles across five test files each re-declared the same block of class
attributes, with no shared base and no ``spec=``. That is fine until production reads one more
attribute off ``llama_backend``: the doubles that happen to be updated keep passing and the rest
fail, far from the change that caused it.

That is exactly what #8700 did. It added
``_monitor_perf_callback(monitor_id, llama_backend.context_length)`` to the chat-completions route,
updated the five test files it ran locally, and missed three others. The result was 19 red tests in
two different shapes: a plain ``AttributeError`` where the route is driven through TestClient, and a
20-second ``asyncio.wait_for`` timeout in the slot-release tests, where the same AttributeError is
swallowed into the response task and only shows up as "the slot was never released".

``context_length`` itself is not special. It is a public property of ``LlamaCppBackend``, has been
since well before #8700, and the real object always has it (it answers ``None`` before a model is
loaded, which every caller already handles). The doubles were simply under-specified.

Inherit from this class rather than re-declaring the block, and add new shared attributes here so
every double gains them at once. ``test_llama_backend_double.py`` keeps this honest from both
directions: nothing declared here may be absent from the real backend, and the route must still
serve a request when driven with a bare double.
"""

from __future__ import annotations

from typing import Optional


class FakeLlamaCppBackend:
    """The attributes ``routes/inference.py`` reads off a loaded GGUF backend.

    Subclasses override what their scenario needs (``supports_tools`` above all, which selects the
    tool loop over the plain generator) and supply the generator methods themselves: behaviour is
    per-test, the attribute surface is not.
    """

    is_loaded = True
    model_identifier = "test/model.gguf"
    _is_audio = False
    is_vision = False
    supports_tools = False
    # Read unguarded on the chat-completions path to size the monitor's context-usage readout.
    # None is what the real property answers before a model is loaded, so it is the honest default;
    # tests that assert on context usage set a number instead.
    context_length: Optional[int] = None
