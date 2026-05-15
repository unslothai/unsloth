# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the /v1/containers CRUD client methods.

Covers:
- All three calls (list / create / delete) send
  ``OpenAI-Beta: containers=v1``. Without it, OpenAI silently no-ops
  the DELETE while still returning 200 ``{"deleted": true}``.
- ``delete_openai_container`` raises when the response body does not
  report ``{"deleted": true}``, even on a 2xx response.
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
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))


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
    result = _drive(
        _make_client().create_openai_container(name = "analysis", ttl_minutes = 30)
    )

    assert result == {"id": "cntr_new", "name": "analysis"}
    assert seen["headers"].get("openai-beta") == "containers=v1"
    assert seen["body"]["name"] == "analysis"
    assert seen["body"]["expires_after"] == {
        "anchor": "last_active_at",
        "minutes": 30,
    }


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
    silently rejected (e.g. before we started sending OpenAI-Beta).
    Defensive guard: when the body omits ``deleted: true``, surface it
    as an error so the UI can report the failure instead of falsely
    reporting success."""

    def handler(request: httpx.Request) -> httpx.Response:
        # 200 but no deleted flag — simulate an unexpected payload shape.
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
