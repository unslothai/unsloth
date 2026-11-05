# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Character cards (V1, V2, V3 as PNG, JSON and CHARX), the character routes and character chats."""

import base64
import io
import json
import struct
import uuid
import zipfile
import zlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from auth.authentication import get_current_subject
from core.characters import cards
from routes.characters import router
from storage import characters_db as db
from storage import studio_db


def _png(color = (200, 80, 40), size = (40, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format = "PNG")
    return buffer.getvalue()


def _chunk(kind: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I", len(body))
        + kind
        + body
        + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    )


def _with_text(
    png: bytes,
    key: str,
    card: dict,
    kind: bytes = b"tEXt",
) -> bytes:
    payload = base64.b64encode(json.dumps(card).encode())
    if kind == b"tEXt":
        body = key.encode() + b"\x00" + payload
    elif kind == b"zTXt":
        body = key.encode() + b"\x00\x00" + zlib.compress(payload)
    else:
        body = key.encode() + b"\x00\x00\x00\x00\x00" + payload
    iend = png.rindex(b"IEND") - 4
    return png[:iend] + _chunk(kind, body) + png[iend:]


V2 = {
    "spec": "chara_card_v2",
    "spec_version": "2.0",
    "data": {
        "name": "Mira",
        "description": "{{char}} tends the greenhouse.",
        "personality": "patient",
        "scenario": "A greenhouse at dusk.",
        "first_mes": "Hello {{user}}, mind the ferns.",
        "mes_example": "<START>\n{{user}}: Hi\n{{char}}: Hello.",
        "creator_notes": "Gentle.",
        "system_prompt": "",
        "post_history_instructions": "",
        "alternate_greetings": ["Oh, a visitor.", "  "],
        "tags": ["fantasy", "cozy", "fantasy"],
        "creator": "someone",
        "character_version": "1.2",
        "extensions": {"talkativeness": "0.8", "fav": True, "custom_tool": {"keep": 1}},
        "character_book": {
            "entries": [
                {
                    "keys": ["fern"],
                    "content": "Ferns are sacred.",
                    "enabled": True,
                    "insertion_order": 5,
                }
            ]
        },
        "unknown_v2_field": "survives",
    },
}


def test_reads_a_v2_png_card():
    parsed = cards.parse_card_file(_with_text(_png(), "chara", V2), "mira.png")
    assert parsed.source_format == "png"
    assert parsed.spec == "chara_card_v2"
    fields = parsed.fields
    assert fields["name"] == "Mira"
    assert fields["alternate_greetings"] == ["Oh, a visitor."]
    assert fields["tags"] == ["fantasy", "cozy"]
    assert fields["talkativeness"] == 0.8
    assert fields["favorite"] is True
    assert fields["character_book"]["entries"][0]["keys"] == ["fern"]
    assert fields["character_book"]["entries"][0]["position"] == "before_char"
    # The stored portrait drops the card chunk.
    assert b"chara" not in parsed.avatar
    assert parsed.avatar_type == "image/png"


@pytest.mark.parametrize("kind", [b"tEXt", b"zTXt", b"iTXt"])
def test_reads_every_png_text_chunk_kind(kind):
    parsed = cards.parse_card_file(_with_text(_png(), "chara", V2, kind), "x.png")
    assert parsed.fields["name"] == "Mira"


@pytest.mark.parametrize("kind", [b"zTXt", b"iTXt"])
def test_compressed_card_text_is_inflated_only_up_to_the_card_limit(monkeypatch, kind):
    monkeypatch.setattr(cards, "MAX_CARD_BYTES", 4096)
    big = {**V2, "data": {**V2["data"], "description": "x" * 20_000}}
    payload = base64.b64encode(json.dumps(big).encode())
    flags = b"\x00" if kind == b"zTXt" else b"\x01\x00\x00\x00"
    body = b"chara\x00" + flags + zlib.compress(payload)
    png = _png()
    iend = png.rindex(b"IEND") - 4
    assert cards.read_png_card(png[:iend] + _chunk(kind, body) + png[iend:])[0] is None


def test_prefers_the_v3_chunk_over_v2():
    v3 = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {**V2["data"], "name": "Mira V3", "group_only_greetings": []},
    }
    png = _with_text(_with_text(_png(), "chara", V2), "ccv3", v3)
    parsed = cards.parse_card_file(png, "x.png")
    assert parsed.fields["name"] == "Mira V3"
    assert parsed.spec == "chara_card_v3"


def test_reads_a_v1_json_card():
    v1 = {
        "name": "Old",
        "description": "d",
        "personality": "p",
        "scenario": "s",
        "first_mes": "hi",
        "mes_example": "",
    }
    parsed = cards.parse_card_file(json.dumps(v1).encode(), "old.json")
    assert parsed.spec == "chara_card_v1"
    assert parsed.fields["first_mes"] == "hi"
    assert parsed.avatar is None


def test_reads_a_charx_archive_with_its_icon():
    v3 = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {
            **V2["data"],
            "assets": [
                {
                    "type": "icon",
                    "uri": "embeded://assets/icon/images/main.png",
                    "name": "main",
                    "ext": "png",
                }
            ],
        },
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("card.json", json.dumps(v3))
        archive.writestr("assets/icon/images/main.png", _png((1, 2, 3)))
    parsed = cards.parse_card_file(buffer.getvalue(), "mira.charx")
    assert parsed.source_format == "charx"
    assert parsed.avatar_type == "image/png"


@pytest.mark.parametrize(
    "data, message",
    [
        (b"", "empty"),
        (b"not a card", "not a PNG"),
        (_png(), "no character card"),
        (json.dumps({"description": "nameless"}).encode(), "no name"),
    ],
)
def test_rejects_files_that_are_not_cards(data, message):
    with pytest.raises(cards.CardError, match = message):
        cards.parse_card_file(data, "x")


def test_export_png_round_trips_unknown_fields_and_carries_both_chunks():
    parsed = cards.parse_card_file(_with_text(_png(), "chara", V2), "mira.png")
    character = {
        "name": "Mira (edited)",
        "description": parsed.fields["description"],
        "firstMes": parsed.fields["first_mes"],
        "alternateGreetings": parsed.fields["alternate_greetings"],
        "tags": ["cozy"],
        "talkativeness": 0.3,
        "favorite": False,
        "extensions": parsed.fields["extensions"],
        "characterBook": parsed.fields["character_book"],
        "card": parsed.card,
    }
    exported = cards.export_png(character, parsed.avatar)
    card, chunks = cards.read_png_card(exported)
    assert chunks == ["ccv3", "chara"]
    assert card["spec"] == "chara_card_v3"
    assert card["data"]["name"] == "Mira (edited)"
    assert card["data"]["unknown_v2_field"] == "survives"
    assert card["data"]["extensions"]["custom_tool"] == {"keep": 1}
    assert card["data"]["extensions"]["talkativeness"] == 0.3
    v2 = cards.build_card_v2(character)
    assert v2["spec"] == "chara_card_v2" and "group_only_greetings" not in v2["data"]
    reparsed = cards.parse_card_file(exported, "again.png")
    assert reparsed.fields["name"] == "Mira (edited)"


def test_export_carries_an_edited_lorebook_placement_into_the_entry_extensions():
    entry = {"keys": ["k"], "content": "c", "extensions": {"position": 4, "depth": 4, "keep": 1}}
    card = {**V2, "data": {**V2["data"], "character_book": {"entries": [entry]}}}
    parsed = cards.parse_card_file(json.dumps(card).encode(), "mira.json")
    book = parsed.fields["character_book"]
    assert book["entries"][0]["position"] == "at_depth"
    book["entries"][0].update(position = "after_char", depth = 0)
    character = {"name": "Mira", "characterBook": book, "card": parsed.card}
    exported = cards.build_card_v3(character)
    ext = exported["data"]["character_book"]["entries"][0]["extensions"]
    assert ext == {"position": 1, "depth": 0, "keep": 1}
    again = cards.parse_card_file(json.dumps(exported).encode(), "again.json")
    assert again.fields["character_book"]["entries"][0]["position"] == "after_char"
    assert book["entries"][0]["extensions"]["position"] == 4


def test_export_converts_a_missing_or_jpeg_portrait_to_png():
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buffer, format = "JPEG")
    for avatar in (None, buffer.getvalue()):
        exported = cards.export_png({"name": "X"}, avatar)
        assert exported.startswith(cards.PNG_SIGNATURE)
        assert cards.read_png_card(exported)[0]["data"]["name"] == "X"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    app = FastAPI()
    app.include_router(router, prefix = "/api/characters")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    with TestClient(app) as test_client:
        yield test_client


def _import(
    client,
    data: bytes,
    name: str = "card.png",
):
    return client.post(f"/api/characters/import?filename={name}", content = data)


def test_import_list_edit_export_and_delete(client):
    response = _import(client, _with_text(_png(), "chara", V2))
    assert response.status_code == 201, response.text
    body = response.json()
    character = body["character"]
    assert body["format"] == "png" and character["hasAvatar"] is True
    assert "card" not in character and character["cardSpec"] == "chara_card_v2"
    character_id = character["id"]

    listed = client.get("/api/characters").json()["characters"]
    assert [c["id"] for c in listed if "Starter" not in c["tags"]] == [character_id]

    patched = client.patch(
        f"/api/characters/{character_id}", json = {"name": "Mira II", "clear": ["characterBook"]}
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "Mira II" and patched.json()["characterBook"] is None

    avatar = client.get(f"/api/characters/{character_id}/avatar")
    assert avatar.status_code == 200 and avatar.headers["content-type"] == "image/png"

    exported = client.get(f"/api/characters/{character_id}/export?format=json")
    assert exported.status_code == 200
    assert exported.json()["data"]["name"] == "Mira II"
    assert exported.json()["data"]["unknown_v2_field"] == "survives"
    assert "Mira%20II.json" in exported.headers["content-disposition"]

    png = client.get(f"/api/characters/{character_id}/export?format=png")
    assert cards.read_png_card(png.content)[0]["data"]["name"] == "Mira II"

    copy = client.post(f"/api/characters/{character_id}/duplicate")
    assert (
        copy.status_code == 201
        and copy.json()["name"] == "Mira II (copy)"
        and copy.json()["hasAvatar"]
    )

    assert client.delete(f"/api/characters/{character_id}").status_code == 204
    assert client.get(f"/api/characters/{character_id}").status_code == 404
    assert client.get(f"/api/characters/{character_id}/avatar").status_code == 404


def test_import_rejects_a_plain_image(client):
    response = _import(client, _png())
    assert response.status_code == 400
    assert "no character card" in response.json()["detail"]


def test_create_validates_and_uploads_a_portrait(client):
    assert client.post("/api/characters", json = {"name": "  "}).status_code == 422
    created = client.post("/api/characters", json = {"name": "Rowan", "firstMes": "Yo."}).json()
    assert created["hasAvatar"] is False
    bad = client.put(f"/api/characters/{created['id']}/avatar", content = b"nope")
    assert bad.status_code == 400
    good = client.put(f"/api/characters/{created['id']}/avatar", content = _png())
    assert good.json()["hasAvatar"] is True and good.json()["avatarRev"] == 1
    cleared = client.delete(f"/api/characters/{created['id']}/avatar")
    assert cleared.json()["hasAvatar"] is False


def test_personas_keep_exactly_one_default(client):
    first = client.post("/api/characters/personas", json = {"name": "Alex"}).json()
    assert first["isDefault"] is True
    second = client.post("/api/characters/personas", json = {"name": "Sam", "isDefault": True}).json()
    personas = {p["id"]: p for p in client.get("/api/characters/personas").json()["personas"]}
    assert personas[second["id"]]["isDefault"] and not personas[first["id"]]["isDefault"]
    assert client.delete(f"/api/characters/personas/{second['id']}").status_code == 204
    assert client.get("/api/characters/personas").json()["personas"][0]["isDefault"] is True
    missing = client.patch(f"/api/characters/personas/{second['id']}", json = {"isDefault": True})
    assert missing.status_code == 404
    assert client.get("/api/characters/personas").json()["personas"][0]["isDefault"] is True


def _messages(thread_id: str) -> list[dict]:
    return studio_db.list_chat_messages(thread_id)


def test_a_one_on_one_chat_opens_with_its_greetings_as_siblings(client):
    character = _import(client, _with_text(_png(), "chara", V2)).json()["character"]
    persona = client.post("/api/characters/personas", json = {"name": "Alex"}).json()
    created = client.post(
        "/api/characters/chats",
        json = {"characterIds": [character["id"]], "personaId": persona["id"]},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    thread_id = body["threadId"]
    messages = _messages(thread_id)
    opening = [m for m in messages if m["role"] == "user"]
    assert len(opening) == 1 and opening[0]["metadata"]["roleplay"] == {"kind": "opening"}
    greetings = [m for m in messages if m["role"] == "assistant"]
    assert {m["parentId"] for m in greetings} == {opening[0]["id"]}
    texts = {m["content"][0]["text"] for m in greetings}
    assert texts == {"Hello Alex, mind the ferns.", "Oh, a visitor."}
    # The main greeting is the newest row, so a chat with no saved branch opens on it.
    newest = max(messages, key = lambda m: m["createdAt"])
    assert newest["content"][0]["text"] == "Hello Alex, mind the ferns."
    assert body["greetingIds"] == [newest["id"]] and body["headId"] == newest["id"]

    thread = studio_db.get_chat_thread(thread_id)
    assert thread["title"] == "Mira"
    assert thread["settings"]["toolsEnabled"] is False

    scene = client.get(f"/api/characters/chats/{thread_id}").json()
    assert scene["personaId"] == persona["id"]
    assert scene["cast"] == [{"characterId": character["id"], "position": 0, "muted": False}]
    assert client.get("/api/characters/chat-index").json()["chats"] == {
        thread_id: [character["id"]]
    }
    chats = client.get(f"/api/characters/{character['id']}/chats").json()["chats"]
    assert [c["threadId"] for c in chats] == [thread_id]
    assert client.get(f"/api/characters/{character['id']}").json()["lastChatAt"] is not None


def test_a_chat_stores_the_greetings_the_client_resolved(client):
    character = client.post(
        "/api/characters",
        json = {
            "name": "Bo",
            "firstMes": "{{random::Hi::Yo}} {{user}}",
            "alternateGreetings": ["{{// x}}"],
        },
    ).json()

    def opening(greetings):
        body = {"characterIds": [character["id"]], "userName": "Pat", "greetings": greetings}
        created = client.post("/api/characters/chats", json = body)
        assert created.status_code == 201, created.text
        rows = _messages(created.json()["threadId"])
        return sorted(m["content"][0]["text"] for m in rows if m["role"] == "assistant")

    assert opening({character["id"]: ["Yo Pat", ""]}) == ["Yo Pat"]
    # A list that no longer matches the card falls back to the names only.
    assert opening({character["id"]: ["stale"]}) == ["{{// x}}", "{{random::Hi::Yo}} Pat"]


def test_a_group_chat_chains_each_members_greeting_through_a_turn_marker(client):
    a = client.post("/api/characters", json = {"name": "Ann", "firstMes": "Hi from {{char}}."}).json()
    b = client.post(
        "/api/characters",
        json = {"name": "Bo", "firstMes": "Bo here.", "alternateGreetings": ["Bo again."]},
    ).json()
    c = client.post("/api/characters", json = {"name": "Cy"}).json()
    body = client.post(
        "/api/characters/chats",
        json = {"characterIds": [a["id"], b["id"], c["id"]], "userName": "Pat"},
    ).json()
    messages = {m["id"]: m for m in _messages(body["threadId"])}
    head = messages[body["headId"]]
    path = []
    node = head
    while node:
        path.append(node)
        node = messages.get(node["parentId"]) if node["parentId"] else None
    path.reverse()
    kinds = [(m["role"], m["metadata"]["roleplay"]["kind"]) for m in path]
    # Cy has no greeting, so the chain ends at Cy's marker.
    assert kinds == [
        ("user", "opening"),
        ("assistant", "reply"),
        ("user", "turn"),
        ("assistant", "reply"),
        ("user", "turn"),
    ]
    assert path[1]["content"][0]["text"] == "Hi from Ann."
    assert path[2]["metadata"]["roleplay"]["speakerId"] == b["id"]
    assert path[3]["content"][0]["text"] == "Bo here."
    assert studio_db.get_chat_thread(body["threadId"])["title"] == "Ann, Bo & Cy"


def test_scene_updates_cast_persona_and_settings(client):
    a = client.post("/api/characters", json = {"name": "Ann"}).json()
    b = client.post("/api/characters", json = {"name": "Bo"}).json()
    thread_id = client.post("/api/characters/chats", json = {"characterIds": [a["id"]]}).json()[
        "threadId"
    ]
    updated = client.put(
        f"/api/characters/chats/{thread_id}",
        json = {
            "add": [b["id"]],
            "mute": {a["id"]: True},
            "settings": {
                "replyOrder": "list",
                "authorsNote": {"text": "Keep it short.", "depth": 2},
                "view": "novel",
            },
        },
    )
    assert updated.status_code == 200, updated.text
    scene = updated.json()
    assert [m["characterId"] for m in scene["cast"]] == [a["id"], b["id"]]
    assert scene["cast"][0]["muted"] is True
    assert scene["settings"]["authorsNote"] == {"text": "Keep it short.", "depth": 2}
    assert scene["settings"]["view"] == "novel"
    assert client.get(f"/api/characters/chats/{thread_id}").json()["settings"]["view"] == "novel"
    assert (
        client.put(f"/api/characters/chats/{thread_id}", json = {"add": ["missing"]}).status_code
        == 404
    )
    assert (
        client.put(f"/api/characters/chats/{thread_id}", json = {"personaId": "missing"}).status_code
        == 400
    )
    assert (
        client.put(
            f"/api/characters/chats/{thread_id}", json = {"settings": {"replyOrder": "random"}}
        ).status_code
        == 422
    )
    assert (
        client.put(
            f"/api/characters/chats/{thread_id}", json = {"settings": {"view": "theater"}}
        ).status_code
        == 422
    )
    assert client.put("/api/characters/chats/nope", json = {"settings": {}}).status_code == 404
    assert client.get("/api/characters/chats/nope").status_code == 204


def test_an_ordinary_chat_has_no_scene(client):
    studio_db.upsert_chat_thread({"id": "plain", "modelType": "base", "createdAt": 1})
    assert client.get("/api/characters/chats/plain").status_code == 204


def test_deleting_a_character_keeps_its_chats_and_deleting_a_chat_drops_its_scene(client):
    a = client.post("/api/characters", json = {"name": "Ann", "firstMes": "Hi."}).json()
    thread_id = client.post("/api/characters/chats", json = {"characterIds": [a["id"]]}).json()[
        "threadId"
    ]
    assert client.delete(f"/api/characters/{a['id']}").status_code == 204
    assert studio_db.get_chat_thread(thread_id) is not None
    assert client.get("/api/characters/chat-index").json()["chats"] == {thread_id: []}
    studio_db.delete_chat_threads([thread_id])
    assert client.get("/api/characters/chat-index").json()["chats"] == {}
    assert db.get_chat_scene(thread_id) is None


def test_a_chat_needs_known_characters(client):
    assert client.post("/api/characters/chats", json = {"characterIds": ["nope"]}).status_code == 404
    assert client.post("/api/characters/chats", json = {"characterIds": []}).status_code == 422


def test_fill_names_handles_both_macro_spellings():
    assert (
        db.fill_names("{{char}} & <USER> & {{ User }}", "Ann", "Pat\\1") == "Ann & Pat\\1 & Pat\\1"
    )


def test_malformed_cards_are_read_or_refused_cleanly():
    nan_card = b'{"name": "Nan", "extensions": {"x": NaN, "talkativeness": Infinity}, "alternate_greetings": "Hi there"}'
    parsed = cards.parse_card_file(nan_card, "nan.json")
    assert parsed.fields["extensions"]["x"] is None
    assert parsed.fields["talkativeness"] == 0.5
    assert parsed.fields["alternate_greetings"] == ["Hi there"]
    json.dumps(parsed.card, allow_nan = False)
    odd = {
        "name": "Odd",
        "alternate_greetings": 5,
        "character_book": {
            "entries": [{"keys": ["a"], "content": "b", "extensions": {"selectiveLogic": [1]}}]
        },
    }
    fields = cards.parse_card_file(json.dumps(odd).encode(), "odd.json").fields
    assert fields["alternate_greetings"] == []
    assert fields["character_book"]["entries"][0]["selective_logic"] == "and_any"
    with pytest.raises(cards.CardError):
        cards.parse_card_file(b"PK\x03\x04garbage", "broken.charx")


def test_a_named_lorebook_is_refused_not_imported_as_a_character():
    for entries in (
        {"0": {"uid": 0, "key": ["Gull Rock"], "content": "A lighthouse."}},
        [{"keys": ["a"], "content": "b"}],
    ):
        book = {"name": "Saltmere Harbor", "entries": entries}
        with pytest.raises(cards.CardError, match = "lorebook"):
            cards.parse_card_file(json.dumps(book).encode(), "harbor-lore.json")
    # A V1 card keeps importing even with a stray top-level entries key.
    card = {"name": "Vee", "description": "A card.", "entries": []}
    assert cards.parse_card_file(json.dumps(card).encode(), "vee.json").fields["name"] == "Vee"


def test_export_survives_a_damaged_portrait():
    truncated = _png()[:-12]  # no IEND
    exported = cards.export_png({"name": "Cut"}, truncated)
    assert cards.read_png_card(exported)[0]["data"]["name"] == "Cut"


@pytest.mark.parametrize("value", [b"NaN", b"1e309", b"-1e400"])
def test_a_non_finite_card_imports_and_the_library_still_lists(client, value):
    response = client.post(
        "/api/characters/import?filename=nan.json",
        content = b'{"name": "Nan", "extensions": {"x": ' + value + b', "y": 1.5}}',
    )
    assert response.status_code == 201
    assert response.json()["character"]["extensions"] == {"x": None, "y": 1.5}
    assert client.get("/api/characters").status_code == 200


def test_a_fork_keeps_the_cast_and_scene(client):
    a = client.post("/api/characters", json = {"name": "Ann", "firstMes": "Hi."}).json()
    created = client.post(
        "/api/characters/chats",
        json = {"characterIds": [a["id"]], "settings": {"scenario": "A dock."}},
    ).json()
    fork = studio_db.fork_chat_thread(
        created["threadId"], created["headId"], "forked", 5, lambda: uuid.uuid4().hex
    )
    assert fork is not None
    scene = client.get("/api/characters/chats/forked").json()
    assert [m["characterId"] for m in scene["cast"]] == [a["id"]]
    assert scene["settings"] == {"scenario": "A dock."}


def test_starter_characters_arrive_once_with_portraits_and_stay_deleted(client):
    from core.characters.starters import STARTERS

    listed = client.get("/api/characters").json()["characters"]
    starters = [c for c in listed if "Starter" in c["tags"]]
    assert len(starters) == len(STARTERS)
    assert all(c["hasAvatar"] and c["creator"] == "Unsloth Studio" for c in starters)
    rin = next(c for c in starters if c["name"] == "Rin Takeda")
    full = client.get(f"/api/characters/{rin['id']}").json()
    assert full["firstMes"] and full["alternateGreetings"] and full["characterBook"]["entries"]
    assert client.get(f"/api/characters/{rin['id']}/avatar").headers["content-type"] == "image/webp"

    # A second load adds nothing, and a deleted starter does not come back.
    assert client.delete(f"/api/characters/{rin['id']}").status_code == 204
    again = client.get("/api/characters").json()["characters"]
    assert len([c for c in again if "Starter" in c["tags"]]) == len(STARTERS) - 1
    assert all(c["name"] != "Rin Takeda" for c in again)


def test_starter_cards_are_original_and_export_cleanly(client):
    from core.characters.starters import STARTER_DIR, STARTERS

    for starter in STARTERS:
        card = json.loads((STARTER_DIR / f"{starter}.json").read_text())
        data = card["data"]
        assert card["spec"] == "chara_card_v2" and data["creator"] == "Unsloth Studio"
        text = json.dumps(data)
        assert "\u2014" not in text and "\u2013" not in text
        assert (STARTER_DIR / f"{starter}.webp").read_bytes()[8:12] == b"WEBP"
    character = client.get("/api/characters").json()["characters"][0]
    exported = client.get(f"/api/characters/{character['id']}/export?format=png")
    assert exported.status_code == 200 and exported.content.startswith(b"\x89PNG")


def test_imported_tags_fit_the_editor_and_charx_assets_do_not_leak_into_exports(client):
    card = {**V2, "data": {**V2["data"], "tags": ["x" * 150, "ok"]}}
    imported = _import(client, json.dumps(card).encode(), "long.json").json()["character"]
    assert imported["tags"] == ["x" * 100, "ok"]
    assert (
        client.patch(
            f"/api/characters/{imported['id']}", json = {"tags": imported["tags"]}
        ).status_code
        == 200
    )

    assets = [
        {
            "type": "icon",
            "uri": "embeded://assets/icon/images/main.png",
            "name": "main",
            "ext": "png",
        },
        {"type": "emotion", "uri": "embeded://assets/emotion/joy.png", "name": "joy", "ext": "png"},
        {"type": "background", "uri": "https://example.com/bg.png", "name": "bg", "ext": "png"},
    ]
    v3 = {"spec": "chara_card_v3", "spec_version": "3.0", "data": {**V2["data"], "assets": assets}}
    exported = cards.build_card_v3({"name": "Mira", "card": v3})
    assert exported["data"]["assets"] == [
        {"type": "icon", "uri": "ccdefault:", "name": "main", "ext": "png"},
        assets[2],
    ]


def test_charx_icon_candidates_are_read_once_each(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        refs = [{"type": "icon", "uri": "embeded://assets/x.bin", "name": "main"}] * 128
        archive.writestr("card.json", json.dumps({**V2, "data": {**V2["data"], "assets": refs}}))
        archive.writestr("assets/x.bin", b"\0" * 4096)
    reads = []
    original = zipfile.ZipFile.open

    def counted(self, name, *args, **kwargs):
        reads.append(getattr(name, "filename", name))
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", counted)
    cards.read_charx(buffer.getvalue())
    assert reads.count("assets/x.bin") == 1


def test_repeated_compressed_card_chunks_are_not_all_inflated(monkeypatch):
    calls = []
    real = zlib.decompressobj
    monkeypatch.setattr(cards.zlib, "decompressobj", lambda *a: calls.append(1) or real(*a))
    png = _png()
    for _ in range(64):
        png = _with_text(png, "chara", V2, b"zTXt")
    card, chunks = cards.read_png_card(png)
    assert card["data"]["name"] == "Mira" and chunks == ["chara"]
    assert len(calls) == 1
    monkeypatch.setattr(cards, "MAX_CARD_BYTES", 4096)
    bomb = _png()
    for _ in range(8):
        # Not a card, so each one is inflated and spends the budget.
        bomb = _with_text(bomb, "ccv3", "x" * 3000, b"zTXt")
    calls.clear()
    assert cards.read_png_card(_with_text(bomb, "chara", V2, b"zTXt"))[0] is None


def test_scene_settings_merge_so_tabs_keep_each_others_edits(client):
    character = client.post("/api/characters", json = {"name": "Ann", "firstMes": "Hi."}).json()
    thread_id = client.post(
        "/api/characters/chats", json = {"characterIds": [character["id"]]}
    ).json()["threadId"]
    note = {"authorsNote": {"text": "Rain.", "depth": 2}}
    client.put(f"/api/characters/chats/{thread_id}", json = {"settings": note})
    scene = client.put(
        f"/api/characters/chats/{thread_id}", json = {"settings": {"view": "stage"}}
    ).json()
    assert scene["settings"] == {**note, "view": "stage"}
    url = f"/api/characters/chats/{thread_id}"
    depth = client.put(url, json = {"settings": {"authorsNote": {"depth": 5}}}).json()
    assert depth["settings"]["authorsNote"] == {"text": "Rain.", "depth": 5}
    cleared = client.put(
        f"/api/characters/chats/{thread_id}", json = {"settings": {"authorsNote": None}}
    )
    assert cleared.json()["settings"] == {"view": "stage"}


def test_cast_edits_apply_by_id_and_keep_a_group_speaker(client):
    ids = [
        client.post("/api/characters", json = {"name": n}).json()["id"] for n in ("Ann", "Bo", "Cy")
    ]
    a, b, c = ids
    thread_id = client.post("/api/characters/chats", json = {"characterIds": ids}).json()["threadId"]
    url = f"/api/characters/chats/{thread_id}"
    # Two tabs, each from its own stale view: both edits survive.
    client.put(url, json = {"remove": [a]})
    scene = client.put(url, json = {"mute": {b: True, a: True}}).json()
    assert [(m["characterId"], m["muted"]) for m in scene["cast"]] == [(b, True), (c, False)]
    assert client.put(url, json = {"mute": {c: True}}).status_code == 409
    assert client.get(url).json()["cast"][1]["muted"] is False
    assert client.put(url, json = {"remove": [c]}).status_code == 200


def test_continuing_an_older_chat_moves_its_character_up(client):
    old = client.post("/api/characters", json = {"name": "Old"}).json()
    new = client.post("/api/characters", json = {"name": "New"}).json()
    thread_id = client.post("/api/characters/chats", json = {"characterIds": [old["id"]]}).json()[
        "threadId"
    ]
    client.post("/api/characters/chats", json = {"characterIds": [new["id"]]})
    thread = studio_db.get_chat_thread(thread_id)
    later = db.now_ms() + 60_000
    studio_db.upsert_chat_thread({**thread, "id": thread_id, "updatedAt": later})
    listed = client.get("/api/characters").json()["characters"]
    names = [c["name"] for c in listed if c["name"] in ("Old", "New")]
    assert names == ["Old", "New"]
    assert next(c for c in listed if c["name"] == "Old")["lastChatAt"] == later


def test_deleting_the_only_unmuted_member_keeps_a_group_speaker(client):
    ids = [
        client.post("/api/characters", json = {"name": n}).json()["id"] for n in ("Ann", "Bo", "Cy")
    ]
    thread_id = client.post("/api/characters/chats", json = {"characterIds": ids}).json()["threadId"]
    url = f"/api/characters/chats/{thread_id}"
    client.put(url, json = {"mute": {ids[1]: True, ids[2]: True}})
    assert client.delete(f"/api/characters/{ids[0]}").status_code == 204
    assert [m["muted"] for m in client.get(url).json()["cast"]] == [False, True]


def test_a_card_with_malformed_extensions_still_exports(client):
    card = {**V2, "data": {**V2["data"], "extensions": "broken"}}
    imported = _import(client, json.dumps(card).encode(), "odd.json").json()["character"]
    for fmt in ("json", "png"):
        assert (
            client.get(f"/api/characters/{imported['id']}/export?format={fmt}").status_code == 200
        )


def test_a_charx_member_is_not_inflated_past_its_declared_size():
    import tracemalloc

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("card.json", b'{"name": "x"}' + b" " * (16 * 1024 * 1024))
    data = bytearray(buffer.getvalue())
    # Declare 22 bytes in both headers, as a crafted archive would.
    struct.pack_into("<I", data, data.find(b"PK\x03\x04") + 22, 22)
    struct.pack_into("<I", data, data.find(b"PK\x01\x02") + 24, 22)
    tracemalloc.start()
    try:
        with pytest.raises(cards.CardError):
            cards.read_charx(bytes(data))
        assert tracemalloc.get_traced_memory()[1] < 4 * 1024 * 1024
    finally:
        tracemalloc.stop()


def test_a_chat_with_a_missing_member_is_refused(client):
    ann = client.post("/api/characters", json = {"name": "Ann"}).json()
    created = client.post("/api/characters/chats", json = {"characterIds": [ann["id"], "missing"]})
    assert created.status_code == 404
