# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The shared double must not claim attributes LlamaCppBackend lacks, and routes must tolerate it."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from models.inference import _InferenceRuntimeFields
import routes.inference as inference_route

from .llama_backend_double import FakeLlamaCppBackend


def test_status_runtime_fields_survive_a_double_that_answers_none():
    """Runtime fields that reject None need a real value on the double, since /status mirrors them all."""
    fields = inference_route._llama_runtime_fields(FakeLlamaCppBackend())
    # supplied by the route, not the backend, so _llama_runtime_fields excuses it
    fields["requires_trust_remote_code"] = False

    try:
        _InferenceRuntimeFields(**fields)
    except ValidationError as error:
        rejected = sorted({str(item["loc"][0]) for item in error.errors()})
        raise AssertionError(
            f"/status cannot answer {rejected} from a backend double: the field is not Optional, "
            "so None is not a value it accepts. Give it the value LlamaCppBackend.__init__ uses "
            "in FakeLlamaCppBackend, rather than to one test's fake."
        ) from None


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
