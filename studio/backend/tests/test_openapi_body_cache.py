# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/openapi.json is rendered once per schema, off the loop, byte-identical to JSONResponse."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse


@pytest.fixture
def main_module():
    import main
    return main


def test_body_and_headers_match_a_plain_json_response(main_module):
    client = TestClient(main_module.app)
    served = client.get("/openapi.json")
    expected = JSONResponse(main_module.app.openapi())
    assert served.status_code == 200
    assert served.content == expected.body
    assert served.headers["content-type"] == expected.headers["content-type"]
    assert served.headers["content-length"] == expected.headers["content-length"]


def test_repeat_fetches_render_once(main_module, monkeypatch):
    client = TestClient(main_module.app)
    client.get("/openapi.json")
    renders = []
    real = JSONResponse.render

    def counted(self, content):
        if isinstance(content, dict) and "paths" in content:
            renders.append(1)
        return real(self, content)

    monkeypatch.setattr(JSONResponse, "render", counted)
    for _ in range(3):
        assert client.get("/openapi.json").status_code == 200
    assert renders == []


def test_a_rebuilt_schema_is_served_afresh(main_module):
    app = main_module.app
    client = TestClient(app)
    client.get("/openapi.json")
    saved = app.openapi_schema
    path = "/api/__openapi_cache_probe"
    app.add_api_route(path, lambda: {"ok": True}, methods = ["GET"])
    try:
        # Some FastAPI versions keep the cached schema after a route change: reset it explicitly.
        app.openapi_schema = None
        assert path in client.get("/openapi.json").json()["paths"]
        assert client.get("/openapi.json").content == JSONResponse(app.openapi()).body
    finally:
        app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != path]
        app.openapi_schema = saved


def test_root_path_variant_is_cached_separately(main_module):
    app = main_module.app
    plain = TestClient(app).get("/openapi.json").json()
    prefixed = TestClient(app, root_path = "/studio").get("/openapi.json").json()
    again = TestClient(app).get("/openapi.json").json()
    assert "servers" not in plain and "servers" not in again
    assert prefixed["servers"][0] == {"url": "/studio"}
