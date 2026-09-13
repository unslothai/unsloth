# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parallel Search MCP provider: defaults, formatting, policy, fallback."""

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference import parallel_search as ps  # noqa: E402
from routes.chat_history import ChatSettingsPayload  # noqa: E402


def test_provider_defaults_to_duckduckgo(monkeypatch):
    monkeypatch.setattr("storage.studio_db.list_chat_settings", lambda: {}, raising = False)
    assert ps.web_search_provider() == ps.DUCK_PROVIDER_ID
    assert ps.parallel_api_key() is None


def test_provider_selects_parallel_and_trims_key(monkeypatch):
    monkeypatch.setattr(
        "storage.studio_db.list_chat_settings",
        lambda: {
            "webSearchProvider": "parallel",
            "parallelSearchApiKey": "  secret  ",
        },
        raising = False,
    )
    assert ps.web_search_provider() == ps.PARALLEL_PROVIDER_ID
    assert ps.parallel_api_key() == "secret"


def test_parallel_search_formats_results(monkeypatch):
    def fake_call(
        tool,
        arguments,
        api_key,
        timeout,
        deadline = None,
    ):
        assert tool == "web_search"
        return {
            "content": [
                {
                    "text": (
                        '{"results": ['
                        '{"title": "A", "url": "https://example.com/a",'
                        ' "excerpts": ["first excerpt"]},'
                        '{"title": "B", "url": "https://example.com/b",'
                        ' "excerpts": ["second excerpt"]}]}'
                    )
                }
            ]
        }

    monkeypatch.setattr(ps, "_call_tool", fake_call)
    text = ps.parallel_web_search("hello", max_results = 5, timeout = 5)
    assert "Title: A" in text
    assert "URL: https://example.com/a" in text
    assert "Snippet: first excerpt" in text
    assert "IMPORTANT" in text


def test_parallel_search_respects_website_policy(monkeypatch):
    def fake_call(
        tool,
        arguments,
        api_key,
        timeout,
        deadline = None,
    ):
        return {
            "content": [
                {
                    "text": (
                        '{"results": ['
                        '{"title": "Blocked", "url": "https://blocked.example/x",'
                        ' "excerpts": ["nope"]},'
                        '{"title": "Ok", "url": "https://allowed.example/y",'
                        ' "excerpts": ["yep"]}]}'
                    )
                }
            ]
        }

    monkeypatch.setattr(ps, "_call_tool", fake_call)
    text = ps.parallel_web_search(
        "hello",
        timeout = 5,
        website_policy = {
            "allowedDomains": ["allowed.example"],
            "blockedDomains": ["blocked.example"],
        },
    )
    assert "allowed.example" in text
    assert "blocked.example" not in text


def test_settings_payload_accepts_provider_and_cleans_key():
    payload = ChatSettingsPayload.model_validate(
        {"webSearchProvider": "parallel", "parallelSearchApiKey": "  k  "}
    )
    assert payload.webSearchProvider == "parallel"
    assert payload.parallelSearchApiKey == "k"


def test_settings_payload_rejects_unknown_provider():
    with pytest.raises(Exception):
        ChatSettingsPayload.model_validate({"webSearchProvider": "bing"})


def test_settings_payload_empty_key_means_unset():
    payload = ChatSettingsPayload.model_validate({"parallelSearchApiKey": "   "})
    assert payload.parallelSearchApiKey is None
