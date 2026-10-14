# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-token residency is Studio's own preemptor's input, and nobody else's.

When llama-server parks and restores slots itself, Studio picks no victims, so a per-token
callback only fed a ledger nothing plans against, and it made every stream ask the server for
``timings_per_token``: a ``timings`` object serialised into every SSE chunk. Off, there is no
callback either. Only ``UNSLOTH_LLAMA_PREEMPT_MODE=studio`` (or an opt-in on a server that does
not park) needs it.
"""

import contextlib
import copy
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import llama_preemption as preemption
from core.inference.llama_admission import reset_llama_admission_queues
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_preemption import reset_preemption_controllers
import routes.inference as inference_route
from state.tool_policy import reset_tool_policy, set_tool_policy
from .llama_backend_double import FakeLlamaCppBackend


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(preemption.PREEMPT_ENV, raising = False)
    monkeypatch.delenv(preemption.PREEMPT_MODE_ENV, raising = False)
    preemption.note_server_parking(False)
    reset_tool_policy()
    reset_preemption_controllers()
    reset_llama_admission_queues()
    yield
    preemption.note_server_parking(False)
    reset_tool_policy()
    reset_preemption_controllers()
    reset_llama_admission_queues()


class _Backend(FakeLlamaCppBackend):
    supports_tools = True
    base_url = "http://127.0.0.1:10358/"
    context_length = 16384
    _kv_cache_context_total = 16384
    effective_parallel_slots = 4
    _kv_cache_unified = True

    def __init__(self, *, server_parks):
        self.calls = []
        self.server_preempts_kv = server_parks

    def generate_chat_completion(self, **kwargs):
        self.calls.append(("plain", kwargs))
        yield "the answer"

    def generate_chat_completion_with_tools(self, **kwargs):
        self.calls.append(("tool_loop", kwargs))
        yield {"type": "content", "text": "the answer"}


def _one_chat(monkeypatch, *, server_parks, tool_loop):
    backend = _Backend(server_parks = server_parks)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    if tool_loop:
        set_tool_policy(True)
    body = {
        "model": "test/model.gguf",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "max_tokens": 512,
    }
    if not tool_loop:
        body["tool_choice"] = "none"
        body["enabled_tools"] = []
    response = TestClient(app).post(
        "/chat/completions", json = body, headers = {"X-Unsloth-Events": "1"}
    )
    assert response.status_code == 200
    entry, kwargs = backend.calls[0]
    assert entry == ("tool_loop" if tool_loop else "plain")
    return kwargs


@pytest.mark.parametrize("tool_loop", [False, True])
def test_a_parking_server_is_handed_no_token_callback(monkeypatch, tool_loop):
    preemption.note_server_parking(True)
    assert preemption.preemption_enabled() is True
    kwargs = _one_chat(monkeypatch, server_parks = True, tool_loop = tool_loop)
    assert kwargs.get("on_tokens") is None


@pytest.mark.parametrize("tool_loop", [False, True])
def test_preemption_off_is_handed_no_token_callback(monkeypatch, tool_loop):
    assert preemption.preemption_enabled() is False
    kwargs = _one_chat(monkeypatch, server_parks = False, tool_loop = tool_loop)
    assert kwargs.get("on_tokens") is None


@pytest.mark.parametrize("tool_loop", [False, True])
def test_studio_owned_preemption_still_watches_the_tokens(monkeypatch, tool_loop):
    monkeypatch.setenv(preemption.PREEMPT_MODE_ENV, preemption.PREEMPT_MODE_STUDIO)
    assert preemption.preemption_enabled() is True
    kwargs = _one_chat(monkeypatch, server_parks = False, tool_loop = tool_loop)
    assert callable(kwargs.get("on_tokens"))


def test_the_observer_keeps_reporting_tool_states_on_a_parking_server():
    """Tool states are per event, not per token, and the progress clock reads them."""
    preemption.note_server_parking(True)
    backend = SimpleNamespace(
        base_url = "http://127.0.0.1:10358/",
        context_length = 16384,
        _kv_cache_context_total = 16384,
        effective_parallel_slots = 4,
        _kv_cache_unified = True,
        server_preempts_kv = True,
    )
    refresh, observe, note_state = inference_route._openai_llama_residency_observer(
        llama_backend = backend, completion_id = "chat-1"
    )
    assert observe is None
    assert refresh(None) is None
    assert note_state(preemption.ParticipantState.TOOLS_RUNNING) == (
        preemption.ParticipantState.TOOLS_RUNNING
    )


def _sse(delta):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _sent_payload(monkeypatch, **kwargs):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48851
    backend._api_key = None
    backend._effective_context_length = 16384
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False
    payloads = []

    @contextlib.contextmanager
    def fake_stream(_client, _url, payload, _cancel, headers = None, first_token_deadline = None):
        payloads.append(copy.deepcopy(payload))
        chunks = [_sse({"content": "hi"}), "data: [DONE]\n"]
        yield type("FakeResponse", (), {"status_code": 200, "chunks": chunks})()

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream)
    monkeypatch.setattr(
        backend,
        "_iter_text_cancellable",
        lambda response, _cancel, first_token_deadline = None: iter(response.chunks),
    )
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "hello"}], **kwargs
        )
    )
    return payloads[0]


def test_without_a_callback_the_wire_asks_for_no_per_token_timings(monkeypatch):
    assert "timings_per_token" not in _sent_payload(monkeypatch)


def test_with_a_callback_the_wire_asks_for_them(monkeypatch):
    assert _sent_payload(monkeypatch, on_tokens = lambda _n: None)["timings_per_token"] is True
