# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Upgrade-skew guards: registry payload and schema, no sqlite migration, opt-in response_format."""

import asyncio
import json
import sqlite3

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import (
    PROVIDER_REGISTRY,
    list_available_providers,
    provider_runs_local_tools,
)


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(agen):
    return [line async for line in agen]


def _mock_http_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))


def _capturing_handler(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    return handler


# hidden in the registry; the UI surfaces them via CUSTOM_PROVIDER_PRESETS
SELF_HOSTED_PRESETS = ("custom", "vllm", "ollama", "llama_cpp")

# keys old cached frontends read off every registry row; do not drop or rename
LEGACY_REGISTRY_KEYS = frozenset(
    {
        "provider_type",
        "display_name",
        "base_url",
        "default_models",
        "model_capabilities",
        "supports_streaming",
        "supports_vision",
        "supports_tool_calling",
        "model_list_mode",
        "auth_kind",
        "base_url_editable",
        "model_ids_editable",
    }
)


def test_registry_default_still_hides_self_hosted_presets():
    """Self-hosted presets stay hidden by default, since old bundles ignore the hidden field."""
    types = {entry["provider_type"] for entry in list_available_providers()}
    for preset in SELF_HOSTED_PRESETS:
        assert preset not in types, (
            f"{preset} is hidden and must not appear in the default /registry "
            "payload; a cached pre-change bundle would render it as a duplicate "
            "dropdown entry"
        )


def test_registry_include_hidden_returns_presets_flagged():
    """``include_hidden=true`` is how a bundle that *does* know asks."""
    entries = {
        entry["provider_type"]: entry for entry in list_available_providers(include_hidden = True)
    }
    for preset in SELF_HOSTED_PRESETS:
        assert preset in entries, f"{preset} missing from include_hidden payload"
        assert entries[preset]["hidden"] is True
        assert entries[preset]["supports_studio_tools"] is True


def test_hidden_flag_matches_the_registry_source_of_truth():
    """Every row's ``hidden`` mirrors the registry, so the UI filter is total."""
    for entry in list_available_providers(include_hidden = True):
        expected = bool(PROVIDER_REGISTRY[entry["provider_type"]].get("hidden"))
        assert entry["hidden"] is expected


def test_visible_rows_are_identical_with_and_without_include_hidden():
    """Asking for hidden rows must not perturb the rows the old bundle reads."""
    default_rows = list_available_providers()
    widened = {
        entry["provider_type"]: entry for entry in list_available_providers(include_hidden = True)
    }
    for row in default_rows:
        assert row == widened[row["provider_type"]]


def test_registry_default_hides_oauth_providers_from_legacy_clients():
    """v0.1.701-beta sends a bare request and renders every row as an API-key form (#8722)."""
    types = {entry["provider_type"] for entry in list_available_providers()}
    assert "openai_codex" not in types


@pytest.mark.parametrize(
    "include_hidden, include_oauth", [(True, False), (False, True), (True, True)]
)
def test_oauth_aware_clients_get_oauth_providers(include_hidden, include_oauth):
    """Released bundles v0.1.702-beta+ render OAuth but only send include_hidden."""
    entries = {
        entry["provider_type"]: entry
        for entry in list_available_providers(
            include_hidden = include_hidden, include_oauth = include_oauth
        )
    }
    assert entries["openai_codex"]["auth_kind"] == "chatgpt_oauth"


@pytest.mark.parametrize("include_hidden", [False, True])
@pytest.mark.parametrize("include_oauth", [False, True])
def test_registry_capabilities_preserve_api_key_rows_and_exclude_managed_providers(
    include_hidden, include_oauth
):
    rows = list_available_providers(include_hidden = include_hidden, include_oauth = include_oauth)
    api_key_rows = [row for row in rows if row["auth_kind"] == "api_key"]
    assert api_key_rows == [
        row
        for row in list_available_providers(include_hidden = include_hidden)
        if row["auth_kind"] == "api_key"
    ]
    assert all(not PROVIDER_REGISTRY[row["provider_type"]].get("managed") for row in rows)


@pytest.mark.parametrize(
    "query, expects_oauth, expects_hidden",
    [
        ("", False, False),
        ("?include_hidden=true", True, True),
        ("?include_oauth=false", False, False),
        ("?include_oauth=true", True, False),
        ("?include_hidden=true&include_oauth=true", True, True),
    ],
)
def test_registry_endpoint_requires_oauth_client_capability(query, expects_oauth, expects_hidden):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes.providers import router

    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.include_router(router, prefix = "/api/providers")
    with TestClient(app) as client:
        response = client.get(f"/api/providers/registry{query}")

    assert response.status_code == 200
    entries = {row["provider_type"]: row for row in response.json()}
    assert ("openai_codex" in entries) is expects_oauth
    if expects_oauth:
        assert entries["openai_codex"]["auth_kind"] == "chatgpt_oauth"
    for preset in SELF_HOSTED_PRESETS:
        assert (preset in entries) is expects_hidden
    assert all(not PROVIDER_REGISTRY[provider_type].get("managed") for provider_type in entries)


def test_registry_rows_keep_every_pre_change_key():
    """Additive only. A cached bundle reads these keys off every row."""
    for entry in list_available_providers(include_hidden = True):
        missing = LEGACY_REGISTRY_KEYS - set(entry)
        assert not missing, f"{entry['provider_type']} lost legacy keys {missing}"


def test_registry_entry_schema_tolerates_a_pre_change_payload():
    """Missing supports_studio_tools defaults to False, so an old backend degrades closed."""
    from models.providers import ProviderRegistryEntry

    legacy_payload = {
        "provider_type": "openai",
        "display_name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_models": ["gpt-4o"],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
    }
    entry = ProviderRegistryEntry(**legacy_payload)
    assert entry.supports_studio_tools is False
    assert entry.hidden is False


def test_anthropic_is_studio_tools_capable():
    assert provider_runs_local_tools("anthropic") is True
    entry = next(row for row in list_available_providers() if row["provider_type"] == "anthropic")
    assert entry["supports_studio_tools"] is True
    assert entry["supports_tool_calling"] is True


def test_openai_codex_keeps_the_capability_it_already_had():
    """The pre-change behaviour is a strict subset of the new one."""
    assert provider_runs_local_tools("openai_codex") is True


@pytest.mark.parametrize("provider_type", SELF_HOSTED_PRESETS)
def test_self_hosted_presets_run_studio_tools(provider_type):
    assert provider_runs_local_tools(provider_type) is True


@pytest.mark.parametrize("provider_type", [None, "", "not_a_provider", "  "])
def test_unknown_provider_types_degrade_closed(provider_type):
    """An unrecognised type must never arm the loop."""
    assert provider_runs_local_tools(provider_type) is False


def test_capability_flag_agrees_with_the_registry_entry():
    for entry in list_available_providers(include_hidden = True):
        assert entry["supports_studio_tools"] is provider_runs_local_tools(entry["provider_type"])


def test_llm_providers_schema_gains_no_column():
    """llm_providers gains no column; the capability is derived at read time, not stored."""
    from storage import providers_db

    conn = sqlite3.connect(":memory:")
    try:
        providers_db._ensure_schema(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_providers)")}
    finally:
        conn.close()

    assert columns >= {
        "id",
        "provider_type",
        "display_name",
        "base_url",
        "is_enabled",
        "created_at",
        "updated_at",
        "models_json",
        "available_models_json",
    }
    # the capability must stay registry-derived; a column would need a migration
    assert not [
        column
        for column in columns
        if "studio_tool" in column or "local_tool" in column or "tool_execution" in column
    ]


def test_response_format_is_omitted_when_the_caller_does_not_ask(monkeypatch):
    """response_format stays absent unless asked; TGI and older LM Studio reject the text default."""
    captured: dict = {}
    _mock_http_client(monkeypatch, _capturing_handler(captured))

    async def run():
        client = ExternalProviderClient(
            provider_type = "custom",
            base_url = "http://custom.example/v1",
            api_key = "",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}],
                model = "local-model",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 64,
            )
        )
        await client.close()

    _drive(run())
    assert "response_format" not in captured["body"]


def test_response_format_is_forwarded_verbatim_when_requested(monkeypatch):
    """Structured-output requests used to be dropped silently on this path."""
    captured: dict = {}
    _mock_http_client(monkeypatch, _capturing_handler(captured))

    async def run():
        client = ExternalProviderClient(
            provider_type = "custom",
            base_url = "http://custom.example/v1",
            api_key = "",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}],
                model = "local-model",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 64,
                response_format = {"type": "json_object"},
            )
        )
        await client.close()

    _drive(run())
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_gemini_translates_response_format_to_a_response_mime_type(monkeypatch):
    """Gemini maps response_format to a generationConfig MIME type, or the planner parses prose."""
    captured: dict = {}
    _mock_http_client(monkeypatch, _capturing_handler(captured))

    async def run():
        client = ExternalProviderClient(
            provider_type = "gemini",
            base_url = "https://generativelanguage.googleapis.com/v1beta",
            api_key = "k",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "Return only strict JSON"}],
                model = "gemini-3-pro",
                tool_choice = "none",
                enabled_tools = [],
                response_format = {"type": "json_object"},
            )
        )
        await client.close()

    _drive(run())
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "tools" not in captured["body"]


def test_gemini_skips_the_json_mime_type_when_tools_are_sent(monkeypatch):
    """Gemini 400s on "Function calling with a response mime type ... unsupported"."""
    captured: dict = {}
    _mock_http_client(monkeypatch, _capturing_handler(captured))

    async def run():
        client = ExternalProviderClient(
            provider_type = "gemini",
            base_url = "https://generativelanguage.googleapis.com/v1beta",
            api_key = "k",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "gemini-3-pro",
                tools = [
                    {
                        "type": "function",
                        "function": {"name": "web_search", "parameters": {"type": "object"}},
                    }
                ],
                tool_choice = "auto",
                response_format = {"type": "json_object"},
            )
        )
        await client.close()

    _drive(run())
    assert "tools" in captured["body"]
    assert "responseMimeType" not in captured["body"].get("generationConfig", {})


@pytest.mark.parametrize(
    "response_format, expected",
    [
        ({"type": "json_object"}, {"type": "json_object"}),
        (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "plan",
                    "schema": {"type": "object", "properties": {}},
                    "strict": True,
                },
            },
            {
                "type": "json_schema",
                "name": "plan",
                "schema": {"type": "object", "properties": {}},
                "strict": True,
            },
        ),
    ],
)
def test_openai_responses_translates_response_format_to_text_format(
    monkeypatch, response_format, expected
):
    """/v1/responses carries structured output on ``text.format``, never response_format."""
    captured: dict = {}
    _mock_http_client(monkeypatch, _capturing_handler(captured))

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "k",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "Return only strict JSON"}],
                model = "gpt-5.1",
                response_format = response_format,
            )
        )
        await client.close()

    _drive(run())
    assert captured["body"]["text"]["format"] == expected
    assert "response_format" not in captured["body"]


def test_a_non_scalar_provider_type_is_not_a_registry_lookup_crash():
    """The value arrives straight from a request body; dict.get would TypeError."""
    assert provider_runs_local_tools(["vllm"]) is False
    assert provider_runs_local_tools({"provider": "vllm"}) is False
    assert provider_runs_local_tools(None) is False
    assert provider_runs_local_tools("vllm") is True
