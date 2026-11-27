# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import hashlib
import re
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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
def settings_db(monkeypatch, tmp_path):
    database = tmp_path / "studio.db"

    def connect():
        connection = sqlite3.connect(database, timeout = 5)
        connection.row_factory = sqlite3.Row
        return connection

    with connect() as connection:
        connection.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value_json TEXT, updated_at TEXT)"
        )
    monkeypatch.setattr("storage.studio_db.get_connection", connect)
    return database


@pytest.fixture
def client(monkeypatch, tmp_path, settings_db):
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


def test_download_prompts_follow_the_account(client):
    prompts = {"askBeforeDownloading": False, "askBeforeProgramDownloads": False}
    client.put("/api/settings/client-preferences", json = {"browser": prompts})
    body = client.get("/api/settings/client-preferences").json()
    assert {k: body["browser"][k] for k in prompts} == prompts


_FRONTEND = _BACKEND.parent / "frontend" / "src" / "features"


def _synced_store_fields(
    store: str,
    interface: str,
    skip = frozenset(),
) -> set[str]:
    source = (_FRONTEND / store).read_text(encoding = "utf-8")
    body = re.search(rf"interface {interface} \{{\n(.*?)\n\}}", source, re.S).group(1)
    fields = re.findall(r"^  (\w+)\??: (?!\()", body, re.M)
    return set(fields) - skip


# The sync hook sends every store field and the server ignores unknown keys, so a field missing here never syncs.
def test_every_synced_store_field_is_saved():
    hook = (_FRONTEND / "settings/hooks/use-device-settings-sync.ts").read_text(encoding = "utf-8")
    local_chat = set(
        re.findall(r'"(\w+)"', re.search(r"LOCAL_CHAT_PREFS = new Set\(\[(.*?)\]", hook).group(1))
    )
    stores = {
        settings_routes.BrowserPrefsPayload: (
            "browser/prefs-store.ts",
            "BrowserPrefsState",
            frozenset(),
        ),
        settings_routes.ChatPrefsPayload: (
            "chat/stores/chat-preferences-store.ts",
            "ChatPreferencesState",
            frozenset(local_chat),
        ),
        settings_routes.PlusMenuPrefsPayload: (
            "chat/stores/plus-menu-prefs-store.ts",
            "PlusMenuPrefsState",
            frozenset(),
        ),
    }
    for model, args in stores.items():
        fields = _synced_store_fields(*args)
        assert fields, args
        assert fields <= set(model.model_fields), (model.__name__, fields - set(model.model_fields))


def test_client_preferences_reject_out_of_range_values(client):
    assert (
        client.put("/api/settings/client-preferences", json = {"interfaceScale": 400}).status_code
        == 422
    )
    bad = {"browser": {"searchEngine": "javascript:alert(1)"}}
    assert client.put("/api/settings/client-preferences", json = bad).status_code == 422


def test_chat_menu_time_and_spellcheck_preferences_follow_the_account(client):
    chat = {
        "sendShortcut": "mod-enter",
        "showReplyTimes": True,
        "pastedTextMinChars": 8000,
        "generationStatsUpdates": "realtime",
    }
    plus_menu = {"pins": {"mcp": True}, "order": ["mcp", "skills"], "pinnedPromptIds": ["p1"]}
    client.put("/api/settings/client-preferences", json = {"chat": chat, "plusMenu": plus_menu})
    client.put("/api/settings/client-preferences", json = {"timeFormat": "24h", "spellcheck": False})
    body = client.get("/api/settings/client-preferences").json()
    assert {k: body["chat"][k] for k in chat} == chat
    assert body["plusMenu"]["pins"] == {"mcp": True}
    assert body["plusMenu"]["order"] == ["mcp", "skills"]
    assert body["plusMenu"]["pinnedPromptIds"] == ["p1"]
    assert body["timeFormat"] == "24h"
    assert body["spellcheck"] is False


@pytest.mark.parametrize(
    "bad",
    [
        {"chat": {"sendShortcut": "alt-enter"}},
        {"chat": {"newChatComposerPosition": 101}},
        {"plusMenu": {"order": "mcp"}},
        {"timeFormat": "36h"},
        {"chat": {"generationStatsUpdates": "live"}},
    ],
)
def test_synced_preferences_reject_invalid_values(client, bad):
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


def test_concurrent_preference_sections_are_both_saved(monkeypatch, settings_db):
    import storage.studio_db as db

    sync.update_client_preferences({"interfaceScale": 100, "browser": {"searchEngine": "google"}})
    original = db.get_app_setting
    readers = threading.Barrier(2)
    calls = threading.local()

    def simultaneous_read(key, fallback = None):
        value = original(key, fallback)
        if not getattr(calls, "read", False):
            calls.read = True
            readers.wait(timeout = 5)
        return value

    monkeypatch.setattr(db, "get_app_setting", simultaneous_read)
    with ThreadPoolExecutor(max_workers = 2) as executor:
        scale = executor.submit(sync.update_client_preferences, {"interfaceScale": 125})
        browser = executor.submit(
            sync.update_client_preferences, {"browser": {"searchEngine": "brave"}}
        )
        scale.result(timeout = 5)
        browser.result(timeout = 5)
    stored = original(sync.CLIENT_PREFERENCES_KEY)
    assert stored["interfaceScale"] == 125
    assert stored["browser"]["searchEngine"] == "brave"


@pytest.mark.parametrize("operation", ["upload", "clear", "adjust", "read"])
def test_background_mutations_serialize_file_and_metadata(
    monkeypatch, tmp_path, settings_db, operation
):
    monkeypatch.setattr(sync, "background_image_path", lambda: tmp_path / "chat-background")
    monkeypatch.setattr(settings_routes, "background_image_path", sync.background_image_path)
    first_writing = threading.Event()
    release_first = threading.Event()
    second_writing = threading.Event()
    second_started = threading.Event()
    original = sync._write

    def delayed_write(key, value):
        if threading.current_thread().name.endswith("_0"):
            first_writing.set()
            assert release_first.wait(timeout = 5)
        else:
            second_writing.set()
        return original(key, value)

    monkeypatch.setattr(sync, "_write", delayed_write)

    def second_operation():
        second_started.set()
        if operation == "upload":
            return sync.save_background_image(PNG + b"second")
        if operation == "clear":
            return sync.clear_background()
        if operation == "read":
            return settings_routes.get_chat_background_image("unsloth")
        return sync.save_background_adjustments({"visibility": 70})

    with ThreadPoolExecutor(max_workers = 2, thread_name_prefix = "background") as executor:
        first = executor.submit(sync.save_background_image, PNG)
        try:
            assert first_writing.wait(timeout = 5)
            second = executor.submit(second_operation)
            assert second_started.wait(timeout = 5)
            assert not second_writing.wait(timeout = 0.2)
        finally:
            release_first.set()
        first.result(timeout = 5)
        result = second.result(timeout = 5)
    if operation == "read":
        assert result.body == PNG
    stored = sync.get_chat_background()
    path = sync.background_image_path()
    if operation == "clear":
        assert stored["image"] is None
        assert not path.exists()
    else:
        assert stored["image"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        if operation == "adjust":
            assert stored["adjustments"] == {"visibility": 70}


def test_client_preferences_keep_window_material(client):
    client.put("/api/settings/client-preferences", json = {"spellcheck": False})
    material = {
        "material": "liquid-3d",
        "style": "liquid-3d",
        "menus": False,
        "strength": {"light": 60, "dark": 20},
        "lens": "magnify",
        "scope": "chats",
    }
    client.put("/api/settings/client-preferences", json = {"material": material})
    body = client.get("/api/settings/client-preferences").json()
    assert body["material"] == material
    assert body["spellcheck"] is False


@pytest.mark.parametrize(
    "bad",
    [
        {"material": "glass"},
        {"style": "solid"},
        {"strength": {"light": 101}},
        {"lens": "zoom"},
        {"scope": "everywhere"},
    ],
)
def test_client_preferences_reject_bad_window_material(client, bad):
    response = client.put("/api/settings/client-preferences", json = {"material": bad})
    assert response.status_code == 422
