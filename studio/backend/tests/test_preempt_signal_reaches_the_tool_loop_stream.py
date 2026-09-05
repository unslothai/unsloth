# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pause signal reaches the client on the path the GUI actually takes.

Every Studio chat carries tools, so every GUI chat streams through the tool-loop consumer of
``openai_chat_completions``, not the plain one. The plain consumer had forwarded the
generator's ``{"type": "preempt"}`` events as ``: preempt-paused`` / ``: preempt-resumed``
since the signal was written. The tool-loop consumer had no branch for them, so they fell
through to the content diff, which reads ``text`` off a dict that has none, and the client
saw nothing.

Measured with four browser sessions on the 4B model at ``-c 8192``: nine pauses and nine
resumes in the server log, "Paused while another chat finishes" shown zero times, every chat
finished. The feature worked; the user could not tell.

These drive the real ASGI route with a fake backend, so a consumer that swallows the event
fails here rather than in a browser.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
import routes.inference as inference_route

from .llama_backend_double import FakeLlamaCppBackend


class _PausingToolBackend(FakeLlamaCppBackend):
    supports_tools = True
    context_length = 8192

    def generate_chat_completion_with_tools(self, **kwargs):
        yield {"type": "content", "text": "Introduction: The"}
        yield {"type": "preempt", "state": "paused"}
        yield {"type": "preempt", "state": "resumed"}
        yield {"type": "content", "text": "Introduction: The Paradigm"}
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            "timings": {"prompt_n": 11, "predicted_n": 5},
            "finish_reason": "stop",
        }


class _PausingPlainBackend(FakeLlamaCppBackend):
    supports_tools = False
    context_length = 8192

    def generate_chat_completion(self, **kwargs):
        yield "Introduction: The"
        yield {"type": "preempt", "state": "paused"}
        yield {"type": "preempt", "state": "resumed"}
        yield "Introduction: The Paradigm"
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            "timings": {"prompt_n": 11, "predicted_n": 5},
            "finish_reason": "stop",
        }


def _client(monkeypatch, backend, *, tools: bool):
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: tools)
    if tools:

        async def _fake_select(payload, **_kwargs):
            return [{"type": "function", "function": {"name": "python"}}]

        monkeypatch.setattr(inference_route, "_select_request_tools", _fake_select)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _payload(tools: bool):
    body = {"messages": [{"role": "user", "content": "write me an essay"}], "stream": True}
    if tools:
        body["enable_tools"] = True
    return body


class TestTheToolLoopStream:
    def test_the_pause_and_the_resume_are_announced(self, monkeypatch):
        response = _client(monkeypatch, _PausingToolBackend(), tools = True).post(
            "/chat/completions", json = _payload(tools = True)
        )
        assert response.status_code == 200
        body = response.text
        assert ": preempt-paused" in body, (
            "the GUI path swallowed the pause: the user sees a half-written answer stop "
            "dead with no explanation, indistinguishable from a wedged backend"
        )
        assert ": preempt-resumed" in body
        assert body.index(": preempt-paused") < body.index(": preempt-resumed")

    def test_the_text_around_the_pause_still_arrives(self, monkeypatch):
        response = _client(monkeypatch, _PausingToolBackend(), tools = True).post(
            "/chat/completions", json = _payload(tools = True)
        )
        body = response.text
        assert "Introduction: The" in body
        assert " Paradigm" in body
        assert "data: [DONE]" in body

    def test_the_signal_is_a_comment_not_a_data_event(self, monkeypatch):
        """Readers that predate the signal must see nothing new: no chunk schema change."""
        response = _client(monkeypatch, _PausingToolBackend(), tools = True).post(
            "/chat/completions", json = _payload(tools = True)
        )
        for line in response.text.splitlines():
            if "preempt" in line:
                assert line.startswith(":"), line


class TestThePlainStreamStillDoes:
    def test_the_pause_and_the_resume_are_announced(self, monkeypatch):
        response = _client(monkeypatch, _PausingPlainBackend(), tools = False).post(
            "/chat/completions", json = _payload(tools = False)
        )
        assert response.status_code == 200
        body = response.text
        assert ": preempt-paused" in body
        assert ": preempt-resumed" in body
        assert " Paradigm" in body
