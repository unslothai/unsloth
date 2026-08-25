# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted tools that survive a turn the Studio loop runs.

Images and Fetch have their own pills, no local implementation, and no
relationship to Search / Code / RAG. So a request can legitimately mix them with
a Studio tool, and the loop has to forward those names to the provider instead
of withholding the whole hosted surface: the alternative is a lit toggle for a
tool the model is never offered.

Search and code execution are the opposite case. Studio runs those itself once
the loop is up, so forwarding them too would run both sides of one tool and bill
the provider for its half.
"""

import asyncio

from types import SimpleNamespace

import pytest

from core.inference.providers import hosted_only_tools, provider_hosted_tools


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _FakeExternalClient:
    last: dict = {}

    def __init__(self, **kwargs):
        _FakeExternalClient.last = {"ctor": kwargs}

    def stream_chat_completion(self, **kwargs):
        async def gen():
            yield "data: [DONE]\n\n"

        return gen()

    async def close(self):
        return None


class _LoopEntered(Exception):
    """stream_with_studio_tools was reached; carries the transport it was given."""


def _request():
    async def is_disconnected():
        return False

    return SimpleNamespace(
        headers = {},
        state = SimpleNamespace(skip_api_monitor = True),
        is_disconnected = is_disconnected,
    )


def _install(monkeypatch, provider_type: str):
    from core.inference.providers import get_base_url
    from routes import inference as inf

    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda _pid: {
            "id": _pid,
            "provider_type": provider_type,
            "base_url": get_base_url(provider_type) or "http://127.0.0.1:8080/v1",
            "display_name": "Saved connection",
            "is_enabled": True,
        },
    )
    monkeypatch.setattr(inf, "resolve_provider_api_key_or_400", lambda *a, **k: "k")
    monkeypatch.setattr(inf, "ExternalProviderClient", _FakeExternalClient)

    def _loop_raiser(transport, **_kwargs):
        raise _LoopEntered(transport)

    monkeypatch.setattr(inf, "stream_with_studio_tools", _loop_raiser)
    return inf


def _payload(**overrides):
    from models.inference import ChatCompletionRequest

    base = dict(
        messages = [{"role": "user", "content": "draw me a chart of this"}],
        provider_id = "saved-1",
        external_model = "gpt-5.4",
        stream = True,
        enable_tools = True,
    )
    base.update(overrides)
    return ChatCompletionRequest(**base)


def _loop_transport(monkeypatch, provider_type: str, selection: list[str], **overrides):
    """Run the route and return the transport the loop was handed."""
    inf = _install(monkeypatch, provider_type)

    async def go():
        resp = await inf._proxy_to_external_provider(
            _payload(enabled_tools = selection, **overrides), _request(), current_subject = "t"
        )
        return [chunk async for chunk in resp.body_iterator]

    with pytest.raises(_LoopEntered) as excinfo:
        _drive(go())
    return excinfo.value.args[0]


@pytest.fixture(autouse = True)
def _clean_policy():
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    yield
    reset_tool_policy()


# ── the helper ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "selection, expected",
    [
        (["python", "terminal", "image_generation"], ["image_generation"]),
        (["search_knowledge_base", "image_generation"], ["image_generation"]),
        # web_search is Studio's own once the loop runs, so it never rides along.
        (["web_search", "python", "image_generation"], ["image_generation"]),
        (["python", "terminal"], []),
        (["web_search"], []),
        # Order and duplicates come from the client; the forwarded list is stable.
        (
            ["image_generation", "python", "image_generation"],
            ["image_generation"],
        ),
    ],
)
def test_only_the_hosted_tools_studio_cannot_run_ride_along(selection, expected):
    assert hosted_only_tools("openai", selection) == expected


def test_a_provider_without_that_tool_is_not_offered_it():
    """openai has no web_fetch, so asking for one must not invent it."""
    assert "web_fetch" not in provider_hosted_tools("openai")
    assert hosted_only_tools("openai", ["python", "web_fetch"]) == []
    assert hosted_only_tools("anthropic", ["python", "web_fetch"]) == ["web_fetch"]


@pytest.mark.parametrize("provider_type", ["llama_cpp", "vllm", "ollama", "custom"])
def test_a_self_hosted_server_is_sent_no_hosted_names_at_all(provider_type):
    """These declare no hosted tools, and an unknown name is a 400 from some of
    them, so the filter has to be empty rather than pass-through."""
    assert hosted_only_tools(provider_type, ["python", "image_generation"]) == []


def test_an_absent_or_malformed_selection_is_not_a_crash():
    assert hosted_only_tools("openai", None) == []
    assert hosted_only_tools(None, ["image_generation"]) == []
    assert hosted_only_tools("openai", [None, 3, "image_generation"]) == ["image_generation"]


# ── the route ────────────────────────────────────────────────────────


@pytest.mark.parametrize("provider_type", ["openai", "gemini"])
def test_images_plus_a_studio_tool_still_reaches_the_provider(monkeypatch, provider_type):
    """The regression in one line: Images plus Code took the Studio loop, and the
    loop used to withhold every hosted name, so image_generation vanished while
    its toggle stayed on."""
    transport = _loop_transport(
        monkeypatch, provider_type, ["python", "terminal", "image_generation"]
    )
    assert transport._request_kwargs["enabled_tools"] == ["image_generation"]


def test_automatic_rag_does_not_cost_the_user_their_image_tool(monkeypatch):
    """A project with automatic RAG selects the loop without the user touching a
    tool pill, which is the quietest way to lose Images."""
    transport = _loop_transport(
        monkeypatch,
        "openai",
        ["search_knowledge_base", "image_generation"],
        # The route drops the RAG tool without a scope, and no scope means no
        # loop at all, so the automatic-RAG turn has to carry one to be the case
        # this is about.
        rag_scope = {"kb_id": "kb-1"},
    )
    assert transport._request_kwargs["enabled_tools"] == ["image_generation"]


def test_the_loop_keeps_its_own_search(monkeypatch):
    """Studio's web_search is running locally this turn, so the provider must not
    be asked to run its own as well."""
    transport = _loop_transport(monkeypatch, "openai", ["web_search", "python"])
    assert transport._request_kwargs["enabled_tools"] is None


def test_a_self_hosted_loop_is_still_sent_no_tool_flags(monkeypatch):
    transport = _loop_transport(monkeypatch, "llama_cpp", ["web_search", "python"])
    assert transport._request_kwargs["enabled_tools"] is None
