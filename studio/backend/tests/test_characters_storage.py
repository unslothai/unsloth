# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Character storage: transactions, portrait files, scene edits and the shared image helpers."""

import asyncio
import io
import shutil

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from auth.authentication import get_current_subject
from routes.characters import router
from storage import characters_db as db
from storage import studio_db
from utils import device_sync_settings
from utils.image_files import read_limited_body, sniff_image_type
from utils.paths.storage_roots import account_path


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buffer, format = "PNG")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    app = FastAPI()
    app.include_router(router, prefix = "/api/characters")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    with TestClient(app) as test_client:
        yield test_client


def _character(client, **fields) -> dict:
    return client.post("/api/characters", json = {"name": "Ann", **fields}).json()


def test_a_book_sent_by_the_editor_is_normalized_like_an_import(client):
    created = _character(client, characterBook = {"entries": 5})
    assert created["characterBook"]["entries"] == []
    assert client.get(f"/api/characters/{created['id']}/export?format=json").status_code == 200
    patched = client.patch(
        f"/api/characters/{created['id']}",
        json = {"characterBook": {"entries": [{"keys": "fern", "content": 7}]}},
    ).json()
    entry = patched["characterBook"]["entries"][0]
    assert entry["keys"] == ["fern"] and entry["content"] == "7"


def test_a_character_deleted_while_a_chat_is_made_is_a_404_and_leaves_no_chat(client, monkeypatch):
    character = _character(client)
    before = {t["id"] for t in studio_db.list_chat_threads()}
    real = db.get_characters

    def racing(ids):
        found = real(ids)
        db.delete_character(character["id"])
        return found

    monkeypatch.setattr(db, "get_characters", racing)
    response = client.post("/api/characters/chats", json = {"characterIds": [character["id"]]})
    assert response.status_code == 404
    assert {t["id"] for t in studio_db.list_chat_threads()} == before


def test_a_persona_deleted_mid_edit_is_a_400(client, monkeypatch):
    character = _character(client)
    thread_id = client.post(
        "/api/characters/chats", json = {"characterIds": [character["id"]]}
    ).json()["threadId"]
    persona = client.post("/api/characters/personas", json = {"name": "Pat"}).json()
    db.delete_persona(persona["id"])
    response = client.put(f"/api/characters/chats/{thread_id}", json = {"personaId": persona["id"]})
    assert response.status_code == 400
    missing = client.post(
        "/api/characters/chats",
        json = {"characterIds": [character["id"]], "personaId": persona["id"]},
    )
    assert missing.status_code == 400


def test_a_large_group_opens_in_a_handful_of_connections(client, monkeypatch):
    greetings = [f"g{i}" for i in range(64)]
    ids = [_character(client, firstMes = "hi", alternateGreetings = greetings)["id"] for _ in range(16)]
    opened = []
    real = studio_db.get_connection

    def counting(*args, **kwargs):
        opened.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(studio_db, "get_connection", counting)
    response = client.post("/api/characters/chats", json = {"characterIds": ids})
    assert response.status_code == 201
    assert len(opened) < 20
    messages = studio_db.list_chat_messages(response.json()["threadId"])
    assert len(messages) == 1 + 15 + 16 * 65


def test_a_greeting_records_its_index_in_the_card(client):
    character = _character(client, firstMes = "Main.", alternateGreetings = ["  ", "Third."])
    created = client.post("/api/characters/chats", json = {"characterIds": [character["id"]]}).json()
    replies = {
        m["content"][0]["text"]: m["metadata"]["roleplay"]["greeting"]
        for m in studio_db.list_chat_messages(created["threadId"])
        if m["role"] == "assistant"
    }
    assert replies == {"Main.": 0, "Third.": 2}
    assert created["greetingIds"] == [created["headId"]]


def test_an_empty_scene_edit_does_not_turn_a_plain_chat_into_a_character_chat(client):
    studio_db.upsert_chat_thread({"id": "plain", "modelType": "base", "createdAt": 1})
    assert client.put("/api/characters/chats/plain", json = {}).status_code == 404
    assert (
        client.put("/api/characters/chats/plain", json = {"settings": {"view": "novel"}}).status_code
        == 404
    )
    assert db.get_chat_scene("plain") is None
    character = _character(client)
    added = client.put("/api/characters/chats/plain", json = {"add": [character["id"]]})
    assert added.status_code == 200 and added.json()["cast"][0]["characterId"] == character["id"]


def test_reading_a_portrait_does_not_recreate_the_folder(client):
    character = _character(client)
    folder = account_path("characters")
    shutil.rmtree(folder, ignore_errors = True)
    assert db.read_image("characters", character["id"]) is None
    db.delete_image("characters", character["id"])
    assert not folder.exists()


def test_a_portrait_is_served_as_the_type_its_bytes_say(client):
    character = _character(client)
    assert client.put(f"/api/characters/{character['id']}/avatar", content = _png()).status_code == 200
    gif = b"GIF89a" + b"\x00" * 20
    db.write_image("characters", character["id"], gif)
    response = client.get(f"/api/characters/{character['id']}/avatar")
    assert response.headers["content-type"] == "image/gif"


def test_a_portrait_for_a_missing_row_writes_no_file(client):
    assert db.set_character_avatar("nope", _png(), "image/png") is None
    assert db.set_persona_avatar("nope", _png(), "image/png") is None
    assert not (account_path("characters") / "nope.img").exists()
    assert not (account_path("personas") / "nope.img").exists()


def test_an_archived_chat_does_not_move_its_character_up(client):
    old = _character(client, name = "Old")
    new = _character(client, name = "New")
    thread_id = client.post("/api/characters/chats", json = {"characterIds": [old["id"]]}).json()[
        "threadId"
    ]
    client.post("/api/characters/chats", json = {"characterIds": [new["id"]]})
    thread = studio_db.get_chat_thread(thread_id)
    studio_db.upsert_chat_thread(
        {**thread, "id": thread_id, "archived": True, "updatedAt": db.now_ms() + 60_000}
    )
    listed = client.get("/api/characters").json()["characters"]
    names = [c["name"] for c in listed if c["name"] in ("Old", "New")]
    assert names == ["New", "Old"]


def test_an_export_names_the_file_for_every_client(client):
    character = _character(client, name = "Mära")
    disposition = client.get(f"/api/characters/{character['id']}/export").headers[
        "content-disposition"
    ]
    assert 'filename="M?ra.png"' in disposition and "filename*=UTF-8''M%C3%A4ra.png" in disposition


def test_image_helpers():
    assert sniff_image_type(_png()) == "image/png"
    assert sniff_image_type(b"GIF89a....") == "image/gif"
    assert sniff_image_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff_image_type(b"nope") is None
    with pytest.raises(ValueError):
        device_sync_settings.save_background_image(b"GIF89a" + b"\x00" * 20)

    class _Request:
        headers = {}

        async def stream(self):
            yield b"x" * 6
            yield b"x" * 6

    with pytest.raises(HTTPException) as refused:
        asyncio.run(read_limited_body(_Request(), 10, "too big"))
    assert refused.value.status_code == 413
    assert asyncio.run(read_limited_body(_Request(), 12, "too big")) == b"x" * 12
