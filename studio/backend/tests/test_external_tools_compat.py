# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Upgrade / version-skew guards for the studio-tools-on-every-provider change.

These tests do not exercise the tool loop itself (``test_studio_tool_loop.py``
owns that). They pin the contract at the seams where an *existing* install can
break during an upgrade, because each of those seams is a place where the two
halves of Studio are versioned independently:

* the ``/api/providers/registry`` payload, read by a browser that may still be
  running a JS bundle from before this capability existed (old FE + new BE);
* the ``ProviderRegistryEntry`` schema, which a new bundle parses from a
  backend that may predate the new fields (new FE + old BE);
* the ``llm_providers`` sqlite schema, which this change must not migrate;
* ``response_format``, newly forwarded on the OpenAI-compatible path, which
  must stay opt-in because not every OpenAI-compatible server tolerates it.
"""

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


# ── helpers ──────────────────────────────────────────────────────────


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


# The four self-hosted presets. They are ``hidden`` in the registry and are
# surfaced by the UI through CUSTOM_PROVIDER_PRESETS rather than the dropdown.
SELF_HOSTED_PRESETS = ("custom", "vllm", "ollama", "llama_cpp")

# Keys the pre-change bundle already read off every registry row. Dropping or
# renaming any of them breaks a cached bundle even though the server is new.
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


# ── 1a. old frontend + new backend ───────────────────────────────────


def test_registry_default_still_hides_self_hosted_presets():
    """The default payload is byte-for-byte the *set* the old bundle expected.

    A browser holding a pre-change bundle filters the provider dropdown on a
    hardcoded ``HIDDEN_PROVIDER_TYPES`` set that contains only ``qwen``; it has
    no idea to filter on a ``hidden`` field. If the default response started
    including the self-hosted presets, that bundle would render vLLM / Ollama /
    llama.cpp / Custom as four extra dropdown entries duplicating the custom
    presets it already lists above the separator. Hence: opt-in.
    """
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


def test_registry_rows_keep_every_pre_change_key():
    """Additive only. A cached bundle reads these keys off every row."""
    for entry in list_available_providers(include_hidden = True):
        missing = LEGACY_REGISTRY_KEYS - set(entry)
        assert not missing, f"{entry['provider_type']} lost legacy keys {missing}"


# ── 1b. new frontend + old backend ───────────────────────────────────


def test_registry_entry_schema_tolerates_a_pre_change_payload():
    """A new bundle against an old backend gets no ``supports_studio_tools``.

    The pydantic model must default it to False rather than reject the row, so
    the capability degrades *closed*: pills stay off instead of arming a tool
    loop the old backend cannot run.
    """
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


# ── capability allowlist ─────────────────────────────────────────────


def test_anthropic_is_not_studio_tools_capable():
    """``_stream_anthropic`` never forwards caller function-tool schemas.

    Advertising the capability would hand the loop a catalog the model never
    sees, so every turn would look like a model that declined to call a tool.
    """
    assert provider_runs_local_tools("anthropic") is False


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


# ── 1c. no DB migration ──────────────────────────────────────────────


def test_llm_providers_schema_gains_no_column():
    """Existing sqlite rows need no migration; the capability is not persisted.

    It is derived from the registry at read time, so an install upgrading in
    place keeps its ``llm_providers`` rows verbatim. This test fails the moment
    somebody adds a column, which is the point: that would need a migration
    story this change deliberately does not have.
    """
    from storage import providers_db

    conn = sqlite3.connect(":memory:")
    try:
        providers_db._ensure_schema(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_providers)")}
    finally:
        conn.close()

    assert columns == {
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


# ── 4. response_format stays opt-in ──────────────────────────────────


def test_response_format_is_omitted_when_the_caller_does_not_ask(monkeypatch):
    """Not every OpenAI-compatible server tolerates ``response_format``.

    TGI types it as a Rust enum with no ``text`` variant and 422s on the
    OpenAI-default ``{"type": "text"}``; LM Studio before 0.3.18 400s on the
    same. Studio talks to those through the ``custom`` preset, so the field has
    to stay absent unless a caller explicitly asked for structured output.
    """
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
