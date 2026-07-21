# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route-level tests for the preview settings endpoints (rotate + sharing toggle)."""

from pathlib import Path
import sys
import types as _types


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings


@pytest.fixture
def client(monkeypatch):
    # Stub the persistence helpers so the endpoints don't touch the real DBs.
    calls: dict = {"enabled": True}

    def _set(value):
        calls["set"] = bool(value)
        calls["enabled"] = bool(value)
        return bool(value)

    monkeypatch.setattr(
        settings, "get_preview_sharing_enabled", lambda: calls["enabled"]
    )
    monkeypatch.setattr(settings, "set_preview_sharing_enabled", _set)
    monkeypatch.setattr(
        settings,
        "rotate_preview_link_secret",
        lambda: calls.__setitem__("rotated", True),
    )

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False), calls


def test_rotate_preview_links(client):
    c, calls = client
    r = c.post("/preview-links/rotate")
    assert r.status_code == 200
    assert r.json() == {"rotated": True}
    assert calls.get("rotated") is True


def test_get_preview_sharing(client):
    c, _ = client
    r = c.get("/preview-sharing")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert "default_enabled" in body


def test_put_preview_sharing_disables(client):
    c, calls = client
    r = c.put("/preview-sharing", json = {"enabled": False})
    assert r.status_code == 200
    assert r.json()["enabled"] is False
    assert calls["set"] is False


def test_put_preview_sharing_rejects_non_bool(client):
    # Pydantic rejects a non-bool body (422) before the handler runs.
    c, _ = client
    r = c.put("/preview-sharing", json = {"enabled": "maybe"})
    assert r.status_code == 422
