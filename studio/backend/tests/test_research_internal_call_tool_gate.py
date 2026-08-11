# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deep Research's internal hop must never enter the local tool loop.

Research puts gathered web/document text into its prompts and posts them back through
/v1/chat/completions with an internal key. `unsloth run --enable-tools` sets a process-wide
policy that overrides a per-request `enable_tools`, and an omitted `enabled_tools` resolves
to every built-in (python/terminal included), so without a hard opt-out an injected page
could steer that hop into executing tools. These tests pin the opt-out at the route, where
the decision is actually made, and pin that the opt-out costs an ordinary run nothing.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
import routes.inference as inference_route
from state.tool_policy import reset_tool_policy, set_tool_policy


@pytest.fixture(autouse = True)
def _clean_policy():
    reset_tool_policy()
    yield
    reset_tool_policy()


class _Backend:
    """Records which generation entry point the route picked."""

    is_loaded = True
    model_identifier = "test/model.gguf"
    _is_audio = False
    is_vision = False
    supports_tools = True

    def __init__(self):
        self.calls = []

    def generate_chat_completion(self, **kwargs):
        self.calls.append(("plain", kwargs))
        yield "the answer"

    def generate_chat_completion_with_tools(self, **kwargs):
        self.calls.append(("tool_loop", kwargs))
        yield {"type": "content", "text": "the answer"}


def _client(monkeypatch, backend):
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _research_payload(opt_out: bool):
    """The payload ResearchSupervisor._stream_completion builds, with the opt-out on or off."""
    body = {
        "model": "test/model.gguf",
        "messages": [
            {"role": "user", "content": "<untrusted_web_evidence>...</untrusted_web_evidence>"}
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.2,
        "max_tokens": 512,
    }
    if opt_out:
        body["tool_choice"] = "none"
        body["enabled_tools"] = []
    return body


def _entry_point(monkeypatch, *, policy, opt_out):
    backend = _Backend()
    if policy is not None:
        set_tool_policy(policy)
    response = _client(monkeypatch, backend).post(
        "/chat/completions", json = _research_payload(opt_out)
    )
    assert response.status_code == 200
    assert "the answer" in response.text
    return backend.calls[0][0], backend.calls[0][1]


def test_forced_tool_policy_would_reach_the_tool_loop_without_the_opt_out(monkeypatch):
    # Guards the test itself: it must fail if the route ever stops forcing tools on here,
    # otherwise the assertion below would pass for the wrong reason.
    entry, kwargs = _entry_point(monkeypatch, policy = True, opt_out = False)
    assert entry == "tool_loop"
    assert {t["function"]["name"] for t in kwargs["tools"]} >= {"python", "terminal"}


@pytest.mark.parametrize("policy", [None, True, False])
def test_the_research_payload_never_enters_the_tool_loop(monkeypatch, policy):
    entry, kwargs = _entry_point(monkeypatch, policy = policy, opt_out = True)
    assert entry == "plain"
    assert not kwargs.get("tools")


@pytest.mark.parametrize("policy", [None, False])
def test_the_opt_out_changes_nothing_a_default_install_does(monkeypatch, policy):
    # Without --enable-tools the hop was already tool-free, so the two fields must not
    # perturb what the model is handed: same entry point, same generation kwargs.
    before_entry, before_kwargs = _entry_point(monkeypatch, policy = policy, opt_out = False)
    reset_tool_policy()
    after_entry, after_kwargs = _entry_point(monkeypatch, policy = policy, opt_out = True)

    assert (before_entry, after_entry) == ("plain", "plain")
    # cancel_event is a fresh object per request, so identity differences say nothing.
    drop = {"cancel_event"}
    assert {k: v for k, v in before_kwargs.items() if k not in drop} == {
        k: v for k, v in after_kwargs.items() if k not in drop
    }


def test_json_mode_research_calls_send_llama_server_an_unchanged_body():
    # The JSON-mode phases take the llama-server passthrough instead of the loop above, so
    # pin the wire body there too: no tools means no tool_choice is forwarded at all, and
    # Unsloth-only extensions never leak upstream.
    from models.inference import ChatCompletionRequest

    class _PassthroughBackend:
        supports_tools = True
        supports_tool_passthrough = True
        markup_profile = None

        def _request_reasoning_kwargs(self, enable_thinking, reasoning_effort, preserve):
            return None

    backend = _PassthroughBackend()
    bodies = []
    for opt_out in (False, True):
        payload = ChatCompletionRequest(
            **_research_payload(opt_out), response_format = {"type": "json_object"}
        )
        assert inference_route._takes_tool_passthrough(payload, backend) is True
        bodies.append(
            inference_route._build_openai_passthrough_body(payload, llama_backend = backend)
        )

    assert bodies[0] == bodies[1]
    assert "tool_choice" not in bodies[1] and "tools" not in bodies[1]
    assert "enabled_tools" not in bodies[1] and "enable_tools" not in bodies[1]
    assert json.loads(json.dumps(bodies[1]))["response_format"] == {"type": "json_object"}
