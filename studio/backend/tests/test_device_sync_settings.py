# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.device_sync_settings as sync  # noqa: E402
from auth.authentication import get_current_subject  # noqa: E402
from routes import settings as settings_routes  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


@pytest.fixture
def client(monkeypatch, tmp_path):
    store: dict = {}
    monkeypatch.setattr("storage.studio_db.get_app_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings", lambda d, read_back = True: store.update(d)
    )
    monkeypatch.setattr(
        sync, "background_image_path", lambda: tmp_path / "appearance" / "chat-background"
    )
    monkeypatch.setattr(settings_routes, "background_image_path", sync.background_image_path)
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.include_router(settings_routes.router, prefix = "/api/settings")
    return TestClient(app)


def test_client_preferences_start_unsaved_and_merge_by_section(client):
    assert client.get("/api/settings/client-preferences").json()["saved"] is False
    client.put("/api/settings/client-preferences", json = {"browser": {"searchEngine": "brave"}})
    body = client.put("/api/settings/client-preferences", json = {"interfaceScale": 125}).json()
    assert body["saved"] is True
    assert body["interfaceScale"] == 125
    assert body["browser"]["searchEngine"] == "brave"
    assert isinstance(body["updatedAt"], int)


def test_client_preferences_reject_out_of_range_values(client):
    assert (
        client.put("/api/settings/client-preferences", json = {"interfaceScale": 400}).status_code
        == 422
    )
    bad = {"browser": {"searchEngine": "javascript:alert(1)"}}
    assert client.put("/api/settings/client-preferences", json = bad).status_code == 422


def test_background_image_round_trips_and_follows_its_own_bytes(client):
    assert client.get("/api/settings/chat-background").json()["saved"] is False
    put = client.put(
        "/api/settings/chat-background/image", content = PNG, headers = {"content-type": "text/html"}
    )
    assert put.status_code == 200
    meta = put.json()["image"]
    assert meta["type"] == "image/png" and meta["size"] == len(PNG)
    image = client.get("/api/settings/chat-background/image")
    assert image.content == PNG
    assert image.headers["content-type"] == "image/png"
    assert image.headers["x-content-type-options"] == "nosniff"


def test_background_rejects_non_images(client):
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    assert client.put("/api/settings/chat-background/image", content = svg).status_code == 400
    assert client.put("/api/settings/chat-background/image", content = b"").status_code == 400


def test_adjustments_need_an_image_and_keep_it(client):
    adjustments = {"adjustments": {"effect": "Blur", "visibility": 60}}
    assert client.put("/api/settings/chat-background", json = adjustments).status_code == 409
    client.put("/api/settings/chat-background/image", content = PNG)
    body = client.put("/api/settings/chat-background", json = adjustments).json()
    assert body["adjustments"] == {"effect": "Blur", "visibility": 60}
    assert body["image"]["size"] == len(PNG)


def test_removing_the_background_leaves_a_dated_empty_record(client, tmp_path):
    client.put("/api/settings/chat-background/image", content = PNG)
    body = client.delete("/api/settings/chat-background").json()
    assert body["saved"] is True and body["image"] is None and isinstance(body["updatedAt"], int)
    assert not (tmp_path / "appearance" / "chat-background").exists()
    assert client.get("/api/settings/chat-background/image").status_code == 404
