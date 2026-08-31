# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The shared backend double keeps up with the real backend, in both directions.

Downward: the double must not claim attributes ``LlamaCppBackend`` lacks, or the tests pass against
a backend that cannot exist.

Upward, the one that bit: the route must still serve a request driven by a bare double. #8700 added
an unguarded ``context_length`` read and updated five of eight test files, giving 19 failures split
between ``AttributeError`` and 20-second timeouts, neither naming the attribute. The canary below
fails in one place instead, with the attribute in the message.
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
    # Attributes set in __init__ are not on the class, so check the source too.
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
    """The canary: drive the real route with nothing but the shared double, so a newly read
    attribute fails here by name rather than scattering errors across five files."""

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
