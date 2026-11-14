# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from routes import inference as inference_route
from routes import providers as providers_route

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_external_provider_sampling_over_the_wire import _run, _Server  # noqa: E402


class _Listing(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    body: dict = {}
    seen_keys: list = []

    def log_message(self, *args, **kwargs) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        type(self).seen_keys.append(
            self.headers.get("x-api-key") or self.headers.get("authorization")
        )
        raw = json.dumps(self.body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


_ANTHROPIC = {
    "data": [
        {"id": "claude-opus-4-8", "type": "model", "max_input_tokens": 1_000_000},
        {"id": "claude-haiku-4-5", "type": "model", "max_input_tokens": 200_000},
    ],
    "has_more": False,
}
_OPENROUTER = {
    "data": [
        {"id": "openai/gpt-4.1", "context_length": 1_047_576, "architecture": {}},
        {"id": "qwen/qwen3-coder", "context_length": 262_144, "architecture": {}},
        {
            "id": "xiaomi/mimo-v2.6-flash",
            "context_length": 1_050_000,
            "top_provider": {"context_length": 1_048_576},
        },
    ]
}


def _window(
    monkeypatch,
    provider_type,
    body,
    model,
    *,
    may_use_saved = True,
):
    handler = type("Listing", (_Listing,), {"body": body, "seen_keys": []})
    with _Server(handler = handler) as upstream:
        row = {
            "id": "c1",
            "provider_type": provider_type,
            "base_url": upstream.base_url,
            "is_enabled": True,
            "models": [model],
        }
        monkeypatch.setattr(providers_route.providers_db, "get_provider", lambda _id: row)
        monkeypatch.setattr(
            providers_route.credential_secrets, "get_provider_api_key", lambda _id: "sk-saved"
        )
        monkeypatch.setattr(
            inference_route, "_request_may_use_saved_provider_key", lambda _r: may_use_saved
        )
        answer = {}

        async def go() -> None:
            answer.update(await providers_route.load_served_window("c1", model, request = object()))

        _run(go)
    return answer, handler.seen_keys


@pytest.mark.parametrize(
    "provider_type, body, model, window",
    [
        ("anthropic", _ANTHROPIC, "claude-opus-4-8", 1_000_000),
        ("openrouter", _OPENROUTER, "openai/gpt-4.1", 1_047_576),
        ("openrouter", _OPENROUTER, "qwen/qwen3-coder", 262_144),
        ("openrouter", _OPENROUTER, "xiaomi/mimo-v2.6-flash", 1_048_576),
    ],
)
def test_a_hosted_connection_reports_its_catalogued_window(
    monkeypatch, provider_type, body, model, window
):
    answer, _ = _window(monkeypatch, provider_type, body, model)
    assert answer == {"context_length": window}


def test_a_caller_without_saved_key_access_gets_no_window_and_sends_no_key(monkeypatch):
    answer, seen = _window(
        monkeypatch, "anthropic", _ANTHROPIC, "claude-opus-4-8", may_use_saved = False
    )
    assert answer == {"context_length": None}
    assert not any(key and "sk-saved" in key for key in seen)
