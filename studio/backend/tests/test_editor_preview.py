# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import zipfile

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image

from routes import editor_preview

BASE = "/api/editor-preview"


@pytest.fixture
def app():
    api = FastAPI()
    api.include_router(editor_preview.router, prefix=BASE)

    async def ui_user(request: Request):
        editor_preview.require_local(request)
        return request.headers.get("x-test-user", "alice")

    api.dependency_overrides[editor_preview.preview_user] = ui_user
    return api


@pytest.fixture
def client(app):
    with TestClient(app, base_url="http://127.0.0.1:3101", client=("127.0.0.1", 5000)) as http:
        yield http


def pair(client, editor):
    response = client.post(BASE + "/sessions/" + editor)
    assert response.status_code == 201
    session = response.json()
    endpoint, token = session["pairing"].split()
    return session["id"], endpoint, {
        "Authorization": "Bearer " + token,
        "X-Unsloth-Preview-Instance": "a" * 32,
    }


@pytest.mark.parametrize("editor", ["blender", "godot", "unity", "unreal", "roblox"])
def test_pairing_is_editor_scoped_and_archive_has_source(client, editor):
    response = client.get(BASE + "/adapters/" + editor)
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "README.md" in archive.namelist()
        assert len(archive.namelist()) > 1
    session_id, endpoint, headers = pair(client, editor)
    assert client.get(endpoint + "/input").status_code == 401
    assert client.get(endpoint + "/input", headers=headers).json()["active"] is False
    another = {**headers, "X-Unsloth-Preview-Instance": "b" * 32}
    assert client.get(endpoint + "/input", headers=another).status_code == 409
    assert client.delete(BASE + "/sessions/" + session_id, headers={"x-test-user": "bob"}).status_code == 404
    assert client.delete(BASE + "/sessions/" + session_id).status_code == 204
    assert client.get(endpoint + "/input", headers=headers).status_code == 401


def test_frame_exchange_sends_input_once_and_keeps_the_newest_frame(app, client):
    session_id, endpoint, headers = pair(client, "roblox")
    assert client.get(endpoint + "/input", headers=headers).status_code == 200
    state = app.state.editor_previews[session_id]
    state.viewer = True
    path = BASE + "/sessions/" + session_id + "/input"

    image = io.BytesIO()
    Image.new("RGB", (4, 4), "green").save(image, "PNG")
    assert client.post(endpoint + "/exchange", headers=headers, content=b"invalid").status_code == 415
    assert client.post(endpoint + "/exchange", headers=headers, content=image.getvalue()).status_code == 200
    assert client.post(path, json={"orbit_x": .2}).status_code == 204
    assert client.post(path, json={"orbit_x": .15, "zoom": -.1}).status_code == 204
    result = client.post(endpoint + "/exchange", headers=headers, content=image.getvalue())
    assert result.json()["orbit_x"] == pytest.approx(.35)
    assert result.json()["zoom"] == pytest.approx(-.1)
    assert state.frame == image.getvalue()
    assert client.get(endpoint + "/input", headers=headers).json()["orbit_x"] == 0


def test_denies_a_nonlocal_browser(app):
    with TestClient(app, base_url="http://127.0.0.1:3101", client=("192.0.2.1", 5000)) as client:
        assert client.post(BASE + "/sessions/blender").status_code == 403
