# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the /v1/containers CRUD client methods.

Covers:
- list / create / delete all send ``OpenAI-Beta: containers=v1``. Without
  it, OpenAI silently no-ops the DELETE but still returns 200
  ``{"deleted": true}``.
- ``delete_openai_container`` raises when the body omits
  ``{"deleted": true}``, even on a 2xx response.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _mock_http_client(monkeypatch, handler):
    """Wire `handler` for the shared `_http_client` AND any per-call
    `httpx.AsyncClient(...)`. delete_openai_container creates a fresh
    AsyncClient (see external_provider.delete_openai_container), so we
    must also intercept that constructor."""
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))
    real_async_client = httpx.AsyncClient

    def _patched_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(ep_mod.httpx, "AsyncClient", _patched_async_client)


def _make_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "openai",
        base_url = "https://api.openai.com/v1",
        api_key = "sk-test",
    )


def test_list_sends_openai_beta_header(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json = {"data": [{"id": "cntr_x", "name": "auto"}]},
        )

    _mock_http_client(monkeypatch, handler)
    result = _drive(_make_client().list_openai_containers())

    assert result == [{"id": "cntr_x", "name": "auto"}]
    assert seen["headers"].get("openai-beta") == "containers=v1"
    assert seen["url"] == "https://api.openai.com/v1/containers"


def test_create_sends_openai_beta_header(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json = {"id": "cntr_new", "name": "analysis"})

    _mock_http_client(monkeypatch, handler)
    result = _drive(_make_client().create_openai_container(name = "analysis", ttl_minutes = 30))

    assert result == {"id": "cntr_new", "name": "analysis"}
    assert seen["headers"].get("openai-beta") == "containers=v1"
    assert seen["body"]["name"] == "analysis"
    assert seen["body"]["expires_after"] == {"anchor": "last_active_at", "minutes": 30}


def test_delete_sends_openai_beta_header_and_accepts_confirmation(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["url"] = str(request.url)
        seen["method"] = request.method
        return httpx.Response(
            200,
            json = {"id": "cntr_x", "object": "container.deleted", "deleted": True},
        )

    _mock_http_client(monkeypatch, handler)
    _drive(_make_client().delete_openai_container("cntr_x"))

    assert seen["method"] == "DELETE"
    assert seen["url"] == "https://api.openai.com/v1/containers/cntr_x"
    assert seen["headers"].get("openai-beta") == "containers=v1"


def test_delete_raises_when_response_lacks_deleted_true(monkeypatch):
    """OpenAI returns 200 ``{"deleted": true}`` even when the request is
    silently rejected (e.g. before we sent OpenAI-Beta). Guard: when the
    body omits ``deleted: true``, surface an error so the UI reports the
    failure instead of false success."""

    def handler(request: httpx.Request) -> httpx.Response:
        # 200 but no deleted flag — unexpected payload shape.
        return httpx.Response(200, json = {"id": "cntr_x", "object": "container"})

    _mock_http_client(monkeypatch, handler)

    with pytest.raises(httpx.HTTPError, match = "did not confirm container deletion"):
        _drive(_make_client().delete_openai_container("cntr_x"))


def test_delete_raises_when_deleted_is_false(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json = {"id": "cntr_x", "object": "container.deleted", "deleted": False},
        )

    _mock_http_client(monkeypatch, handler)

    with pytest.raises(httpx.HTTPError, match = "did not confirm container deletion"):
        _drive(_make_client().delete_openai_container("cntr_x"))


def test_delete_raises_when_body_is_not_json(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content = b"<html>OK</html>")

    _mock_http_client(monkeypatch, handler)

    with pytest.raises(httpx.HTTPError, match = "did not confirm container deletion"):
        _drive(_make_client().delete_openai_container("cntr_x"))


def test_delete_propagates_openai_4xx(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json = {"error": {"message": "not found"}})

    _mock_http_client(monkeypatch, handler)

    with pytest.raises(httpx.HTTPStatusError):
        _drive(_make_client().delete_openai_container("cntr_missing"))


def test_external_chat_route_resolves_saved_provider_key(monkeypatch):
    from routes import inference as inf_mod
    from models.inference import ChatCompletionRequest

    class ResolverReached(Exception):
        pass

    monkeypatch.setattr(
        inf_mod.providers_db,
        "get_provider",
        lambda _provider_id: {
            "provider_type": "mistral",
            "base_url": "https://api.mistral.ai/v1",
            "display_name": "Mistral",
            "is_enabled": True,
        },
    )

    def resolve(subject, provider_id, encrypted_api_key):
        raise ResolverReached(subject, provider_id, encrypted_api_key)

    monkeypatch.setattr(inf_mod, "resolve_provider_api_key_or_400", resolve)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hello"}],
        provider_id = "provider-1",
        external_model = "mistral-large-latest",
    )

    with pytest.raises(ResolverReached) as reached:
        _drive(inf_mod._proxy_to_external_provider(payload, None, current_subject = "alice"))
    assert reached.value.args == ("alice", "provider-1", None)


def test_container_client_uses_saved_provider_key(monkeypatch):
    from routes import inference as inf_mod
    from models.inference import OpenAIContainerRequest

    calls: list[tuple[str, str | None, str | None]] = []

    def resolve(subject, provider_id, encrypted_api_key):
        calls.append((subject, provider_id, encrypted_api_key))
        return "saved-key"

    monkeypatch.setattr(inf_mod, "resolve_provider_api_key_or_400", resolve)

    monkeypatch.setattr(
        inf_mod.providers_db,
        "get_provider",
        lambda _provider_id: {
            "provider_type": "openai",
            "base_url": "https://api.openai.com/v1",
            "display_name": "OpenAI",
            "is_enabled": True,
        },
    )
    client = inf_mod._resolve_openai_cloud_client(
        OpenAIContainerRequest(
            provider_id = "provider-1",
            provider_base_url = "https://attacker.invalid/v1",
        ),
        current_subject = "alice",
    )

    assert client.api_key == "saved-key"

    assert client.base_url == "https://api.openai.com/v1"
    assert calls == [("alice", "provider-1", None)]
    _drive(client.close())


def test_list_route_filters_expired_containers(monkeypatch):
    """OpenAI keeps containers in /v1/containers with status="expired"
    after their idle TTL passes — unusable but still listed. The list
    route must drop them so the picker shows only usable containers."""
    from routes import inference as inf_mod
    from models.inference import OpenAIContainerRequest

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json = {
                "data": [
                    {"id": "cntr_active", "name": "live", "status": "running"},
                    {"id": "cntr_dead", "name": "old", "status": "expired"},
                    {"id": "cntr_unknown", "name": "no-status"},
                ],
            },
        )

    _mock_http_client(monkeypatch, handler)

    def fake_resolve(_body):
        return _make_client()

    monkeypatch.setattr(inf_mod, "_resolve_openai_cloud_client", fake_resolve)

    body = OpenAIContainerRequest(
        encrypted_api_key = "enc",
        provider_base_url = "https://api.openai.com/v1",
    )
    response = _drive(inf_mod.list_openai_containers(body, current_subject = "u"))
    ids = [c.id for c in response.containers]
    assert "cntr_active" in ids
    assert "cntr_unknown" in ids  # missing status is treated as usable
    assert "cntr_dead" not in ids
