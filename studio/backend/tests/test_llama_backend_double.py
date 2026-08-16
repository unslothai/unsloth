# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The shared backend double keeps up with the real backend.

Two directions, because the double can rot either way.

Downward: the double must not claim attributes the real ``LlamaCppBackend`` does not have, or the
tests pass against a backend that cannot exist.

Upward, which is the one that actually bit: the route must still serve a request when driven with a
bare double. #8700 added an unguarded ``llama_backend.context_length`` read to the chat-completions
path and updated five test files, missing three. That produced 19 failures in two unrelated-looking
shapes -- a plain ``AttributeError`` under TestClient, and a 20-second timeout in the slot-release
tests, where the same error is swallowed into the response task and reads as "the slot was never
released". Neither shape names the missing attribute.

The canary below fails in ONE place, at the point of the change, with the attribute in the message.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
import routes.inference as inference_route

from .llama_backend_double import FakeLlamaCppBackend


def test_the_double_claims_nothing_the_real_backend_lacks():
    """Every attribute the double declares exists on the real backend."""
    from core.inference.llama_cpp import LlamaCppBackend

    declared = {
        name
        for name in vars(FakeLlamaCppBackend)
        if not name.startswith("__") and name != "_abc_impl"
    }
    # Instance attributes set in __init__ are not on the class, so check the source too: the
    # question is "does the real backend have this concept", not "is it a class attribute".
    import inspect

    source = inspect.getsource(LlamaCppBackend)
    missing = sorted(
        name
        for name in declared
        if not hasattr(LlamaCppBackend, name) and f"self.{name}" not in source
    )
    assert missing == [], (
        f"the double declares {missing}, which the real LlamaCppBackend does not have -- "
        f"either the attribute was renamed in production or the double invented it"
    )


def test_a_bare_double_can_still_serve_a_chat_completion(monkeypatch):
    """The canary: drive the real route with nothing but the shared double.

    If production starts reading a new attribute off ``llama_backend``, this fails here with the
    attribute named, instead of scattering AttributeErrors and timeouts across five files.
    """

    class _Backend(FakeLlamaCppBackend):
        def generate_chat_completion(self, **kwargs):
            yield "hi"
            yield {"type": "metadata", "usage": {}, "timings": {}}

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())

    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "tester"

    with TestClient(app) as client:
        response = client.post(
            "/chat/completions",
            json = {
                "model": "test/model.gguf",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

    assert response.status_code == 200, (
        f"the route could not be served with the shared double: {response.text[:400]}\n"
        f"If this is an AttributeError, production began reading a new attribute off "
        f"llama_backend -- add it to FakeLlamaCppBackend rather than to one test's fake."
    )
