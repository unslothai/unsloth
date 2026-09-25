# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import io
import json
import os
import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from auth.authentication import (  # noqa: E402
    get_current_subject,
    request_admitted_without_credential,
)
from core import library  # noqa: E402
from routes import library as library_routes  # noqa: E402
from storage import library_db  # noqa: E402

from .test_rag_native_drop_upload import SECRET, _sign  # noqa: E402

_CLIP = bytes(range(100))
_SANDBOX_ID = "sandbox:t-lib:report.txt"
_SANDBOX_A = "sandbox:t-lib:a.txt"


def _app(subject) -> TestClient:
    """The router alone; with a ``subject``, as a caller signed in with the UI or a key."""
    app = FastAPI()
    if subject is not None:
        app.dependency_overrides[get_current_subject] = subject
        app.dependency_overrides[request_admitted_without_credential] = lambda: False
    app.include_router(library_routes.router, prefix = "/api/library")
    return TestClient(app)


@pytest.fixture
def client(monkeypatch):
    # Only the Library's own uploads: the other sources read stores a test seeds for itself.
    monkeypatch.setattr(library, "_SOURCES", (library._upload_items,))
    library.invalidate_listing()
    library._THUMBNAILS.forget()
    return _app(lambda: "unsloth")


async def _signed_in_subject(_request, _token):
    return "unsloth"


@pytest.fixture
def signed_in(monkeypatch):
    monkeypatch.setattr(library_routes, "subject_for_header_or_query_token", _signed_in_subject)


def _send(client, files, **form):
    return client.post("/api/library/uploads", files = [("files", f) for f in files], data = form)


def _upload(client, *files):
    response = _send(client, files)
    assert response.status_code == 200, response.text
    return response.json()["ids"]


def _items(client):
    body = client.get("/api/library").json()
    return {item["id"]: item for item in body["items"]}, body["folders"]


def _post(client, path, **body):
    return client.post(f"/api/library/{path}", json = body)


def _patch(client, **body):
    assert client.patch("/api/library/items", json = body).status_code == 200


def _favorites(client):
    return client.get("/api/library/favorites").json()["ids"]


def _delete(client, item_id) -> int:
    return _post(client, "items/delete", id = item_id).status_code


def _download(client, item_id):
    return client.get("/api/library/items/download", params = {"id": item_id})


def _thumbnail(client, item_id):
    return client.get("/api/library/items/thumbnail", params = {"id": item_id})


def _folder(client, name, parent_id) -> str:
    return _post(client, "folders", name = name, parentId = parent_id).json()["id"]


def _ref(item_id: str) -> str:
    return item_id.split(":", 1)[1]


def _fail(*_args, **_kwargs):
    raise sqlite3.OperationalError("database is locked")


def _file(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def _gallery_image(prompt: str) -> str:
    from PIL import Image

    from core.inference import image_gallery

    meta = {"prompt": prompt, "width": 4, "height": 4, "steps": 1, "guidance": 1, "seed": 1}
    return image_gallery.save(Image.new("RGB", (4, 4)), {**meta, "created_at": 1})["id"]


def _gallery_video(prompt: str) -> str:
    from core.inference import video_gallery

    keys = ("width", "height", "num_frames", "fps", "duration_s", "steps", "guidance", "seed")
    meta = {key: 1 for key in keys}
    return video_gallery.save(
        b"\0\0\0\x18ftypmp42", {**meta, "prompt": prompt, "created_at": 1_700_000_000}
    )["id"]


def _gallery_audio(prompt: str) -> str:
    from core.inference import audio_gallery
    meta = {"model": "sample-tts", "audio_type": "snac", "sample_rate": 24000, "duration_s": 1.0}
    return audio_gallery.save(
        b"RIFF0000WAVE", {**meta, "prompt": prompt, "created_at": 1_700_000_000}
    )["id"]


def _sandbox_chat(
    name,
    body,
    thread = "t-lib",
):
    """A file ``name`` in the chat's sandbox: (sandbox directory, its path)."""
    from core.inference.tools import resolve_sandbox_workdir
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": thread, "title": "T", "modelType": "base", "modelId": "m", "createdAt": 1}
    )
    directory = resolve_sandbox_workdir(thread)
    path = os.path.join(directory, name)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    Path(path).write_bytes(body)
    return directory, path


def _png(width, height, mode) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    color = (200, 40, 40, 128) if mode == "RGBA" else (200, 40, 40)
    Image.new(mode, (width, height), color).save(buf, format = "PNG")
    return buf.getvalue()


def _thumbnail_size(client, item_id):
    from PIL import Image

    response = _thumbnail(client, item_id)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/webp"
    return Image.open(io.BytesIO(response.content)).size


def test_uploads_are_listed_and_typed_by_their_extension_not_the_client(client, monkeypatch):
    import mimetypes

    # (as sent, listed name, stored type)
    uploads = [
        (("note.md", b"# hi", "text/markdown"), "note.md", "text/markdown"),
        # A client path keeps its last segment.
        (("C:\\Users\\me\\notes.txt", b"<script>", "text/html"), "notes.txt", "text/plain"),
        (("report.pdf", b"%PDF-1.7", "text/html"), "report.pdf", "application/pdf"),
        # A list of types, or one a browser would run, is not stored for an unknown extension.
        (("track.xyz", b"x", "audio/x, text/html"), "track.xyz", "application/octet-stream"),
        (("data.bin2", b"x", "application/xhtml+xml"), "data.bin2", "application/octet-stream"),
    ]
    ids = _upload(client, *(sent for sent, _, _ in uploads))
    items, folders = _items(client)
    assert folders == []
    assert [(items[i]["name"], items[i]["contentType"]) for i in ids] == [
        (name, stored) for _, name, stored in uploads
    ]
    note = items[ids[0]]
    assert note["source"] == "uploaded" and note["sizeBytes"] == 4
    assert note["favorite"] is False and note["folderId"] is None
    # By a fixed map, whatever the OS registry says: Windows apps remap types, and mimetypes calls
    # .ts a video.
    monkeypatch.setattr(mimetypes, "guess_type", lambda *_a, **_k: ("video/mp2t", None))
    types = ("app.ts", "view.tsx", "mod.mts", "photo.JPG", "mystery.xyz")
    assert [library._guess_type(name) for name in types] == [
        *["text/typescript"] * 3,
        "image/jpeg",
        "application/octet-stream",
    ]


@pytest.mark.parametrize(
    "stored, inline",
    [
        ("image/png", True),
        ("audio/mpeg", True),
        ("image/png; charset=binary", True),
        # Served on the app origin, so markup never comes back as something a browser renders.
        ("text/html", False),
        ("image/svg+xml", False),
        ("audio/x, text/html", False),
    ],
)
def test_only_an_exact_raster_or_media_type_is_served_inline(client, stored, inline):
    [item] = _upload(client, ("page.xyz", b"<script>1</script>", "application/octet-stream"))
    conn = library_db.get_connection()
    conn.execute("UPDATE library_uploads SET content_type = ? WHERE id = ?", (stored, _ref(item)))
    conn.commit()
    conn.close()
    response = client.get(_items(client)[0][item]["fileUrl"])
    assert response.headers["x-content-type-options"] == "nosniff"
    assert ("attachment" not in response.headers.get("content-disposition", "")) is inline
    expected = library.media_type(stored) if inline else "application/octet-stream"
    assert response.headers["content-type"] == expected


def test_rename_favorite_and_folders_are_an_overlay(client):
    [image] = _upload(client, ("photo.png", b"", "image/png"))
    folder = _folder(client, "Work", None)
    _patch(client, id = image, name = "photo.txt", favorite = True, folderId = folder)
    item = _items(client)[0][image]
    assert (item["name"], item["favorite"], item["folderId"]) == ("photo.txt", True, folder)
    # The type follows the file's own name, so a binary renamed to .txt never opens as text.
    assert item["fileName"] == "photo.png"
    # Leaving the key off leaves it where it is; an explicit null moves it back out.
    _patch(client, id = image, favorite = False)
    assert _items(client)[0][image]["folderId"] == folder
    _patch(client, id = image, folderId = None)
    assert _items(client)[0][image]["folderId"] is None
    response = client.patch("/api/library/items", json = {"id": image, "folderId": "nope"})
    assert response.status_code == 404
    # Folders nest; a folder never moves under itself, and a deleted one hands its contents up.
    inner = _folder(client, "Inner", folder)
    child = _folder(client, "Child", inner)
    for parent in (inner, folder):
        response = client.patch(f"/api/library/folders/{folder}", json = {"parentId": parent})
        assert response.status_code == 400
        assert response.json()["detail"] == "A folder cannot be moved into itself"
    [note] = _send(client, [("note.md", b"", "text/markdown")], folderId = inner).json()["ids"]
    assert client.delete(f"/api/library/folders/{inner}").status_code == 200
    items, folders = _items(client)
    assert items[note]["folderId"] == folder
    assert {f["id"]: f["parentId"] for f in folders} == {folder: None, child: folder}


def test_notes_are_saved_in_their_own_encoding_and_deletable(client):
    # Windows PowerShell 5 writes UTF-16LE with a BOM and CRLF endings.
    [note] = _upload(client, ("log.txt", "\ufeffold\r\n".encode("utf-16le"), "text/plain"))
    url = f"/api/library/uploads/{_ref(note)}/text"
    response = client.put(url, json = {"text": "\ufeffnew\r\n", "encoding": "utf-16le"})
    assert response.status_code == 200
    item = _items(client)[0][note]
    assert client.get(item["fileUrl"]).content == b"\xff\xfen\x00e\x00w\x00\r\x00\n\x00"
    assert client.put(url, json = {"text": "x", "encoding": "latin-1"}).status_code == 422
    assert client.put(url, json = {"text": "hello"}).status_code == 200
    assert _items(client)[0][note]["sizeBytes"] == 5
    assert _delete(client, note) == 200
    assert note not in _items(client)[0]
    assert client.get(item["fileUrl"]).status_code == 404
    assert _delete(client, note) == 404
    assert client.put(url, json = {"text": "x"}).status_code == 404


def test_chat_attachments_holding_bytes_are_known_by_their_type_in_any_case(client, monkeypatch):
    import storage.studio_db as studio_db

    def attachment(attachment_id, content_type):
        return {"messageId": "m", "id": attachment_id, "type": "file", "contentType": content_type}

    # A compare chat's audio part is typed by the inventory at best, and may carry no type at all.
    voice = {"messageId": "m:1", "id": "voice", "type": "audio", "contentType": None}
    monkeypatch.setattr(library, "_SOURCES", (library._attachment_items,))
    monkeypatch.setattr(
        studio_db,
        "list_chat_attachments",
        lambda: [
            attachment("clip", "Video/WebM; codecs=vp9"),
            attachment("words", "Text/Plain"),
            voice,
        ],
    )
    items = _items(client)[0]
    clip, words = items["attachment:m:clip"], items["attachment:m:words"]
    assert (clip["contentType"], clip["textOnly"]) == ("video/webm", False)
    assert (words["contentType"], words["textOnly"]) == ("text/plain", True)
    # Any string is a message id: encoded, so the colon in it is not the one after it.
    assert items["attachment:m%3A1:voice"]["textOnly"] is False
    deleted = []
    monkeypatch.setattr(studio_db, "delete_chat_attachment", lambda *ids: deleted.append(ids) or 1)
    assert _delete(client, "attachment:m%3A1:voice") == 200
    assert deleted == [("m:1", "voice")]


def test_a_chat_audio_part_is_typed_by_its_format_or_its_bytes():
    from storage.studio_db import _content_part_attachments

    def typed(audio):
        [part] = _content_part_attachments(json.dumps([{"type": "audio", "audio": audio}]))
        return part["contentType"]

    wav = base64.b64encode(b"RIFF\0\0\0\0WAVEfmt ").decode()
    assert typed(wav) == "audio/wav"
    assert typed(base64.b64encode(b"ID3\3\0\0\0\0\0\0\0\0").decode()) == "audio/mpeg"
    assert typed({"data": wav, "format": "flac"}) == "audio/flac"
    assert typed(f"data:audio/ogg;base64,{wav}") == "audio/ogg"
    assert typed(base64.b64encode(b"not a sound at all").decode()) is None


def test_a_gallery_file_that_cannot_be_deleted_keeps_its_name_star_and_folder(client, monkeypatch):
    from core.inference import audio_gallery

    monkeypatch.setattr(library, "_SOURCES", (library._audio_items,))
    item_id = f"audio:{_gallery_audio('Kept')}"
    _patch(client, id = item_id, name = "Renamed", favorite = True)
    # As on Windows while another app has it open: the gallery keeps it for another try.
    monkeypatch.setattr(audio_gallery, "delete", lambda _ref: False)
    assert _delete(client, item_id) == 500
    item = _items(client)[0][item_id]
    assert (item["name"], item["favorite"]) == ("Renamed", True)


def test_a_sandbox_file_that_is_gone_takes_no_rename_a_later_file_would_inherit(
    client, signed_in, monkeypatch
):
    monkeypatch.setattr(library, "_SOURCES", (library._sandbox_items,))
    _directory, path = _sandbox_chat("a.txt", b"first")
    os.unlink(path)
    response = client.patch("/api/library/items", json = {"id": _SANDBOX_A, "name": "Old"})
    assert response.status_code == 404
    _sandbox_chat("a.txt", b"another file at the same path")
    assert _items(client)[0][_SANDBOX_A]["name"] == "a.txt"


def test_a_sandbox_file_of_a_chat_whose_id_has_a_colon_is_reachable(client, signed_in, monkeypatch):
    monkeypatch.setattr(library, "_SOURCES", (library._sandbox_items,))
    _sandbox_chat("a.txt", b"mine", thread = "imported:1")
    [item_id] = _items(client)[0]
    assert item_id == "sandbox:imported%3A1:a.txt"
    assert _download(client, item_id).content == b"mine"
    assert _delete(client, item_id) == 200
    assert _items(client)[0] == {}


def test_a_sandbox_file_written_twice_in_a_second_is_a_new_version(client, signed_in, monkeypatch):
    monkeypatch.setattr(library, "_SOURCES", (library._sandbox_items,))
    _directory, path = _sandbox_chat("frame.png", b"one")
    versions = []
    for ns in (1_700_000_000_100_000_000, 1_700_000_000_600_000_000):
        os.utime(path, ns = (ns, ns))
        library.invalidate_listing()
        versions.append(_items(client)[0]["sandbox:t-lib:frame.png"]["updatedAt"])
    assert versions == [1_700_000_000_100, 1_700_000_000_600]


def test_generated_audio_and_video_are_listed_and_deleted(client, monkeypatch):
    import routes.video as video_routes
    from core.inference import audio_gallery, video_gallery

    monkeypatch.setattr(library, "_SOURCES", (library._audio_items, library._video_items))
    audio = _gallery_audio("Hello from the Library")
    video, shelved = _gallery_video("A calm sea"), _gallery_video("Archived")
    video_gallery.set_flags(shelved, archived = True)
    items = _items(client)[0]
    item = items[f"audio:{audio}"]
    assert (item["name"], item["contentType"]) == ("Hello from the Library.wav", "audio/wav")
    assert (item["source"], item["createdAt"]) == ("generated", 1_700_000_000_000)
    item = items[f"video:{video}"]
    assert (item["name"], item["contentType"]) == ("A calm sea.mp4", "video/mp4")
    # Off the Video page's shelf, still the Library's.
    assert (item["archived"], items[f"video:{shelved}"]["archived"]) == (False, True)
    assert _delete(client, f"audio:{audio}") == 200
    assert audio_gallery.audio_path(audio) is None

    # A clip whose job cannot be dropped stays listed, to delete again.
    forgotten = []
    monkeypatch.setattr(video_routes, "_forget_terminal_video", forgotten.append)
    monkeypatch.setattr(video_routes, "_forget_openai_job", lambda ref: False)
    response = _post(client, "items/delete", id = f"video:{video}")
    assert response.status_code == 500 and "video job" in response.json()["detail"]
    assert video_gallery.video_path(video) is not None
    monkeypatch.setattr(video_routes, "_forget_openai_job", lambda ref: forgotten.append(ref) or 1)
    assert _delete(client, f"video:{video}") == 200
    assert video_gallery.video_path(video) is None
    # Same cleanup as the Video page, so no ghost card comes back.
    assert forgotten == [video, video]


def test_favorites_list_only_stars_the_source_still_has(client, monkeypatch):
    from core.inference import image_gallery

    [starred, plain] = _upload(client, ("a.txt", b"a", "text/plain"), ("b.txt", b"b", "text/plain"))
    image = f"image:{_gallery_image('A star')}"
    for item_id, favorite in ((starred, True), (plain, False), (image, True), ("image:gone", True)):
        _patch(client, id = item_id, favorite = favorite)
    reads = []
    real = image_gallery._read_meta
    monkeypatch.setattr(image_gallery, "_read_meta", lambda path: reads.append(path) or real(path))
    # Deleted by its gallery, not the Library: the star stays in the overlay but is not listed.
    assert set(_favorites(client)) == {starred, image}
    # The gallery pages ask this for every star each time they open, so no PNG is decoded.
    assert reads == []
    image_gallery.image_path(_ref(image)).unlink()
    assert _favorites(client) == [starred]
    # A delete drops the item's row, even when its source had already lost it.
    assert (_delete(client, starred), _delete(client, "image:gone")) == (200, 404)
    assert set(library_db.list_entries()) == {plain, image}
    assert _delete(client, "elsewhere:x") == 400


def test_leftovers_of_a_crash_are_swept_from_the_uploads_folder(client):
    import time

    [kept, interrupted] = _upload(
        client, ("k.txt", b"k", "text/plain"), ("i.txt", b"i", "text/plain")
    )
    directory = library.uploads_dir()
    # A delete that stopped after setting its file aside: the row still lists it.
    aside = directory / f".{_ref(interrupted)}.deleting"
    os.replace(directory / _ref(interrupted), aside)
    stale = [directory / ".0123.tmp", directory / f".{'a' * 32}.deleting"]
    fresh = directory / ".4567.tmp"
    for path in (*stale, fresh):
        path.write_bytes(b"x")
    old = time.time() - 2 * library._LEFTOVER_AGE_SECONDS
    for path in (*stale, aside):
        os.utime(path, (old, old))
    library._swept_at.clear()
    items = _items(client)[0]
    assert {kept, interrupted} <= set(items)
    names = {path.name for path in directory.iterdir()}
    assert names == {_ref(kept), _ref(interrupted), fresh.name}
    assert client.get(items[interrupted]["fileUrl"]).content == b"i"


def test_fine_tuned_models_are_listed_but_not_deleted_here(client, monkeypatch):
    import shutil

    from utils.paths.storage_roots import outputs_root

    monkeypatch.setattr(library, "_SOURCES", (library._model_items,))
    outputs_root().mkdir(parents = True, exist_ok = True)
    stamp = library._model_stamp()
    run = outputs_root() / "library-test-run"
    run.mkdir()
    (run / "adapter_config.json").write_text('{"base_model_name_or_path": "unsloth/base"}')
    (run / "adapter_model.safetensors").write_bytes(b"x" * 10)
    try:
        # A new run changes the stamp the remembered listing is kept by.
        assert library._model_stamp() != stamp
        [item] = [item for item in _items(client)[0].values() if item["name"] == run.name]
        assert item["contentType"] == library.MODEL_CONTENT_TYPE
        assert (item["model"]["origin"], item["model"]["exportType"]) == ("training", "lora")
        assert item["sizeBytes"] >= 10
        assert _delete(client, item["id"]) == 400
        assert run.is_dir()
    finally:
        shutil.rmtree(run)


def test_the_slow_sources_are_remembered_briefly_and_forgotten_on_a_write(client, monkeypatch):
    calls = []

    def walked():
        calls.append(1)
        fields = {"source": "generated", "content_type": "text/plain", "file_url": ""}
        return [
            library._item("sandbox:t:a.txt", name = "a.txt", size_bytes = 1, created_at = 1, **fields)
        ]

    def broken():
        raise RuntimeError("gallery unreadable")

    monkeypatch.setattr(library, "_sandbox_items", walked)
    # A failing source is skipped rather than emptying the Library.
    monkeypatch.setattr(library, "_SOURCES", (library._remembered("_sandbox_items", 60), broken))
    # Listed only, never written: the star is kept for the file it would find.
    monkeypatch.setattr(library, "fingerprint", lambda _item_id: "1:1")
    _patch(client, id = "sandbox:t:a.txt", favorite = True)
    for _ in range(2):
        assert _items(client)[0]["sandbox:t:a.txt"]["favorite"] is True
    assert len(calls) == 1
    # Another account's listing is never this one's.
    monkeypatch.setattr(library, "_account_key", lambda: "other")
    _items(client)
    _items(client)
    assert len(calls) == 2
    library.invalidate_listing()
    _items(client)
    assert len(calls) == 3


def test_an_overlay_row_stays_with_the_file_it_was_made_for(client, monkeypatch, tmp_path):
    monkeypatch.setattr(library, "_SOURCES", (library._sandbox_items,))
    _directory, path = _sandbox_chat("report.txt", b"v1")

    def make_again():
        # Moved aside rather than unlinked, so no filesystem can hand the new file the old inode.
        os.replace(path, tmp_path / f"old-{len(os.listdir(tmp_path))}")
        Path(path).write_bytes(b"new")
        library.invalidate_listing()

    # A row from before fingerprints is adopted by the file at its path.
    library_db.update_entry(_SANDBOX_ID, favorite = True)
    assert library_db.list_entries()[_SANDBOX_ID]["fingerprint"] is None
    item = _items(client)[0][_SANDBOX_ID]
    assert item["favorite"] is True and "_fingerprint" not in item
    assert library_db.list_entries()[_SANDBOX_ID]["fingerprint"] == library.fingerprint(_SANDBOX_ID)

    # Edited in place, as a tool appending to its output: still the same file.
    _patch(client, id = _SANDBOX_ID, name = "Q3")
    with open(path, "ab") as handle:
        handle.write(b" and v2")
    library.invalidate_listing()
    item = _items(client)[0][_SANDBOX_ID]
    assert (item["favorite"], item["name"]) == (True, "Q3")
    assert _favorites(client) == [_SANDBOX_ID]

    # Made again at the path: the star is not the new file's, and a write drops the old row first.
    make_again()
    assert _favorites(client) == []
    _patch(client, id = _SANDBOX_ID, folderId = None)
    entry = library_db.list_entries()[_SANDBOX_ID]
    assert (entry["name"], entry["favorite"]) == (None, False)

    # Nor does a listing carry it over: it prunes the row.
    _patch(client, id = _SANDBOX_ID, favorite = True, name = "Old")
    make_again()
    item = _items(client)[0][_SANDBOX_ID]
    assert (item["favorite"], item["name"]) == (False, "report.txt")
    assert _SANDBOX_ID not in library_db.list_entries()


def test_an_older_database_is_upgraded_and_checked_once(client, monkeypatch, tmp_path):
    (tmp_path / "real").mkdir()
    conn = sqlite3.connect(tmp_path / "real" / "studio.db")
    conn.execute(
        "CREATE TABLE library_entries (item_id TEXT NOT NULL PRIMARY KEY, name TEXT, "
        "favorite INTEGER NOT NULL DEFAULT 0, folder_id TEXT, updated_at INTEGER NOT NULL)"
    )
    conn.execute("INSERT INTO library_entries (item_id, favorite, updated_at) VALUES ('x', 1, 1)")
    conn.commit()
    conn.close()
    # A home reached through a link: the unresolved path is not the key the schema was saved as.
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory = True)
    monkeypatch.setattr(library_db, "studio_db_path", lambda: tmp_path / "link" / "studio.db")
    # Its rows are kept and gain the fingerprint column.
    assert library_db.list_entries()["x"] == {
        "name": None,
        "favorite": True,
        "folderId": None,
        "updatedAt": 1,
        "fingerprint": None,
    }
    runs = []
    monkeypatch.setattr(library_db, "_ensure_schema", runs.append)
    for _ in range(3):
        library_db.get_connection().close()
    assert runs == []


def test_a_failed_write_leaves_the_uploads_as_they_were(client, monkeypatch):
    [note] = _upload(client, ("n.md", b"old", "text/markdown"))
    for name, call in (
        # The note's row cannot be updated: the old note goes back.
        ("touch_upload", lambda: library.write_upload_text(_ref(note), "new")),
        # The file is set aside but its row cannot be dropped: the file comes back.
        ("delete_upload", lambda: library.delete_item(note)),
        # A new upload's row cannot be written: its file goes.
        ("insert_upload", lambda: library.save_upload("a.txt", "text/plain", [b"hi"])),
    ):
        with monkeypatch.context() as patched:
            patched.setattr(library_db, name, _fail)
            with pytest.raises(sqlite3.OperationalError):
                call()
    assert library.upload_path(_ref(note)).read_bytes() == b"old"
    assert [path.name for path in library.uploads_dir().iterdir()] == [_ref(note)]
    assert note in _items(client)[0]


def test_a_file_held_open_on_windows_answers_409(client, monkeypatch):
    [note] = _upload(client, ("n.md", b"old", "text/markdown"))
    url = f"/api/library/uploads/{_ref(note)}/text"

    def held(*_args):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(library.os, "replace", held)
    response = client.put(url, json = {"text": "new"})
    assert response.status_code == 409
    assert response.json()["detail"] == "The file is in use. Close it and try again."
    assert _delete(client, note) == 409

    def broken(*_args):
        raise OSError(5, "I/O error", "/secret/path")

    monkeypatch.setattr(library.os, "replace", broken)
    for response in (client.put(url, json = {"text": "new"}), _post(client, "items/delete", id = note)):
        assert response.status_code == 500 and "/secret" not in response.text
    # A save that failed leaves the previous text and nothing staged.
    assert [path.name for path in library.uploads_dir().iterdir()] == [_ref(note)]
    assert library.upload_path(_ref(note)).read_bytes() == b"old"


def test_concurrent_deletes_and_note_saves_leave_nothing_behind(client):
    from concurrent.futures import ThreadPoolExecutor

    [upload, note] = _upload(
        client, ("twice.txt", b"hi", "text/plain"), ("n.md", b"old", "text/markdown")
    )
    with ThreadPoolExecutor(8) as pool:
        deletes = [pool.submit(library.delete_item, upload) for _ in range(4)]
        saves = [pool.submit(library.write_upload_text, _ref(note), f"v{i}") for i in range(6)]
        saves.append(pool.submit(library.delete_item, note))
        [job.result() for job in saves]
    assert [job.result() for job in deletes].count(True) == 1
    assert not any(library.uploads_dir().iterdir())
    assert _items(client)[0] == {}


@pytest.fixture
def lease_secret(monkeypatch):
    import utils.native_path_leases as leases

    secret = base64.urlsafe_b64encode(SECRET).decode("ascii").rstrip("=")
    monkeypatch.setenv(leases.LEASE_SECRET_ENV, secret)
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)
    yield
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)


@pytest.mark.parametrize("swap", ["link", "file"])
def test_a_desktop_drop_swapped_after_its_check_is_refused(
    client, lease_secret, tmp_path, monkeypatch, swap
):
    path = _file(tmp_path / "notes.txt", "mine")
    outside = _file(tmp_path / "outside.txt", "your")
    lease = _sign(path)
    real = library._verify_native

    def swapped_after_the_check(*args, **kwargs):
        grant = real(*args, **kwargs)
        if kwargs.get("consume"):
            path.unlink()
            if swap == "link":
                path.symlink_to(outside)
            else:
                os.replace(outside, path)
        return grant

    monkeypatch.setattr(library, "_verify_native", swapped_after_the_check)
    assert _send(client, [], nativePathLeases = [lease]).status_code == 400
    assert _items(client)[0] == {}


def test_a_desktop_drop_is_read_from_its_signed_path_once_the_whole_batch_checks(
    client, lease_secret, tmp_path
):
    lease = _sign(_file(tmp_path / "Quarterly report.pdf", "%PDF-1.7 report"))
    forged = _sign(_file(tmp_path / "other.txt", "hi"), secret = b"x" * 32)
    # One bad grant refuses the batch before any grant is spent or any file kept.
    response = _send(
        client, [("kept.md", b"# hi", "text/markdown")], nativePathLeases = [lease, forged]
    )
    assert response.status_code == 400
    # A generic reason: the grant's own can name a path in the workspace.
    assert response.json()["detail"] == "The dropped file could not be read. Drop it again."
    assert _items(client)[0] == {} and list(library.uploads_dir().iterdir()) == []
    folder = _folder(client, "Work", None)
    response = _send(client, [], nativePathLeases = [lease], folderId = folder)
    assert response.status_code == 200, response.text
    [item_id] = response.json()["ids"]
    item = _items(client)[0][item_id]
    assert (item["name"], item["contentType"]) == ("Quarterly report.pdf", "application/pdf")
    assert item["folderId"] == folder
    assert client.get(item["fileUrl"]).content == b"%PDF-1.7 report"
    assert _send(client, []).status_code == 400


def test_a_failed_batch_keeps_nothing_and_gives_its_grants_back(
    client, lease_secret, tmp_path, monkeypatch
):
    leases = [_sign(_file(tmp_path / "a.txt", "a")), _sign(_file(tmp_path / "b.txt", "b"))]
    original = library.save_upload
    calls = []

    def fail_second(*args):
        calls.append(args[0])
        if len(calls) == 2:
            raise OSError("disk full")
        return original(*args)

    def gone(*_args, **_kwargs):
        raise KeyError("folder")

    with monkeypatch.context() as patched:
        patched.setattr(library, "save_upload", fail_second)
        with pytest.raises(OSError):
            _send(client, [], nativePathLeases = leases)
    assert _items(client)[0] == {}
    # A folder deleted meanwhile undoes the batch too.
    folder = _folder(client, "Work", None)
    with monkeypatch.context() as patched:
        patched.setattr(library_db, "update_entry", gone)
        plan = ("plan.md", b"# plan", "text/markdown")
        response = _send(client, [plan], nativePathLeases = leases, folderId = folder)
        assert response.status_code == 404
    assert _items(client)[0] == {}
    # The same grants work on a retry: nothing a failed batch spent stays spent.
    response = _send(client, [], nativePathLeases = leases)
    assert response.status_code == 200 and len(response.json()["ids"]) == 2
    # And a grant that was used is still single use.
    assert _send(client, [], nativePathLeases = leases[:1]).status_code == 400


def test_an_oversized_desktop_drop_is_refused_before_copying(client, monkeypatch, tmp_path):
    big = _file(tmp_path / "big.bin", "x" * 10)
    monkeypatch.setattr(library_routes, "_MAX_UPLOAD_BYTES", 4)
    opened = lambda lease: ("big.bin", "application/octet-stream", open(big, "rb"))  # noqa: E731
    monkeypatch.setattr(library, "open_native_upload", opened)

    def copied(*_args):
        raise AssertionError("copied an oversized drop")

    monkeypatch.setattr(library, "save_upload", copied)
    # Refused by the batch's check, before any grant is spent, and again when read, if it grew.
    for size in (10, 1):
        monkeypatch.setattr(
            library, "check_native_upload", lambda lease, size = size: ("big.bin", size)
        )
        assert _send(client, [], nativePathLeases = ["lease"]).status_code == 413


@pytest.fixture
def project(tmp_path, monkeypatch):
    import storage.studio_db as studio_db

    root = tmp_path / "Projects" / "demo"
    (root / "sandbox").mkdir(parents = True)
    record = {"id": "p1", "rootPath": str(root), "sandboxPath": str(root / "sandbox")}
    monkeypatch.setattr(
        studio_db, "ensure_chat_project_workspace", lambda pid: record if pid == "p1" else None
    )
    return root / "sandbox"


def test_an_upload_is_copied_into_a_project_under_its_own_name(client, project, tmp_path):
    from core.inference import gallery_projects

    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    add = lambda item_id, project_id: _post(  # noqa: E731
        client, "items/project", id = item_id, projectId = project_id
    )
    assert add(note, "p1").json() == {"already": False}
    [copied] = (project / "files").iterdir()
    assert copied.name.startswith("plan-") and copied.suffix == ".md"
    assert copied.read_bytes() == b"# plan"
    # Adding it again finds the copy instead of making another.
    assert add(note, "p1").json() == {"already": True}
    assert add(note, "missing").status_code == 404
    assert add("upload:0123456789abcdef0123456789abcdef", "p1").status_code == 404
    assert add("attachment:m:a", "p1").status_code == 400
    assert add("model:training:/tmp/run", "p1").status_code == 400
    # A name Windows cannot hold is refused on every OS; an open file copies from its descriptor.
    source = _file(tmp_path / "x.txt", "x")
    for bad in ("a:b.txt", "CON.txt", "trailing.", "tab\there.txt"):
        with pytest.raises(ValueError):
            gallery_projects.copy_into_project(source, "p1", "files", bad)
    with open(source, "rb") as handle:
        result = gallery_projects.copy_into_project(handle, "p1", "files", "x-1.txt")
    assert Path(result["path"]).read_text() == "x"


@pytest.mark.parametrize(
    "name, safe",
    [
        ("report.csv", "report.csv"),
        ("a<b>c:d|e?.txt", "a b c d e.txt"),
        ("CON.txt", "_CON.txt"),
        ("lpt1", "_lpt1"),
        ("notes. . .", "notes"),
        (".env", "env"),
        ("line\nbreak\ttab.md", "line break tab.md"),
        ("...", "file"),
        ("C:\\Users\\me\\x.txt", "x.txt"),
        # Cut by UTF-8 bytes, a whole character at a time: 80 sloths are 320.
        pytest.param("\U0001f9a5" * 80 + ".txt", "\U0001f9a5" * 50 + ".txt", id = "sloths"),
    ],
)
def test_names_written_to_disk_are_valid_on_windows(name, safe):
    from core.inference.gallery_projects import _bad_name, _tmp_name

    assert library.safe_file_name(name) == safe
    project_name = library.safe_file_name(name, item_id = "upload:x")
    assert not _bad_name(project_name), project_name
    # Add to project writes its copy under a longer temp name first.
    assert len(_tmp_name(project_name).encode()) <= 255


def test_items_download_as_attachments_under_the_name_they_were_given(client, monkeypatch):
    [upload] = _upload(client, ("re:port*q3?.html", b"<script>1</script>", "text/html"))
    # Checks its own credentials, header or query, rather than the overridden dependency.
    assert _download(client, upload).status_code == 401
    monkeypatch.setattr(library_routes, "subject_for_header_or_query_token", _signed_in_subject)
    response = _download(client, upload)
    assert (response.status_code, response.content) == (200, b"<script>1</script>")
    assert response.headers["content-type"] == "application/octet-stream"
    # Stored as a bare id, so the header names it after the upload, not the file on disk.
    disposition = response.headers["content-disposition"]
    assert disposition.startswith('attachment; filename="re port q3.html"')
    assert response.headers["content-length"] == "18"
    assert response.headers["x-content-type-options"] == "nosniff"
    head = client.head("/api/library/items/download", params = {"id": upload})
    assert (head.status_code, head.content) == (200, b"")
    assert _download(client, "upload:" + "0" * 32).status_code == 404
    assert _download(client, "attachment:m:a").status_code == 400


def test_generated_media_download_under_their_prompt(client, signed_in, project):
    from core.inference import gallery_projects, image_gallery

    def disposition(item_id):
        response = _download(client, item_id)
        assert response.status_code == 200
        return response.headers["content-disposition"]

    image = _gallery_image('A red fox: "at dawn"')
    assert disposition(f"image:{image}").startswith('attachment; filename="A red fox at dawn.png"')
    # The UTF-8 name rides along for a prompt ASCII cannot hold.
    image = _gallery_image("Café ☕ at noon")
    assert f"filename*=UTF-8''{quote('Café ☕ at noon.png')}" in disposition(f"image:{image}")
    # No prompt: the id, never an empty name.
    image = _gallery_image("  ")
    assert f'filename="{image}.png"' in disposition(f"image:{image}")
    assert 'filename="Hello there.wav"' in disposition(f"audio:{_gallery_audio('Hello there')}")

    # A project copy keeps the stored name, as the Images page's own Add to project does, so
    # adding from either place finds the other's copy; two alike stay two files.
    first, second = _gallery_image("Same prompt"), _gallery_image("Same prompt")
    for image in (first, second):
        response = _post(client, "items/project", id = f"image:{image}", projectId = "p1")
        assert response.json() == {"already": False}
    names = sorted(path.name for path in (project / "images").iterdir())
    assert names == sorted(f"{image}.png" for image in (first, second))
    path = image_gallery.owned_image_path(first)
    assert gallery_projects.copy_into_project(path, "p1", "images")["already"] is True


def test_a_sandbox_file_is_reachable_by_id_only_as_the_listing_walks_it(
    client, signed_in, monkeypatch
):
    import routes.inference as inference

    monkeypatch.setattr(library, "_SOURCES", (library._sandbox_items,))
    # The walk counts dotfiles too, which the Library then hides: .env, a.txt, b.txt.
    monkeypatch.setattr(inference, "_MAX_SNAPSHOT_FILES", 3)
    directory, _path = _sandbox_chat("a.txt", b"mine")
    for name in ("b.txt", "c.txt", ".env", "d1/d2/d3/d4/deep.txt"):
        _sandbox_chat(name, b"x")
    walks = []
    real = inference._sandbox_listing_names
    monkeypatch.setattr(
        inference, "_sandbox_listing_names", lambda path: walks.append(path) or real(path)
    )
    status = lambda name: _download(client, f"sandbox:t-lib:{name}").status_code  # noqa: E731

    def listed(*_args):
        raise AssertionError("a per-item route listed the sandboxes")

    with monkeypatch.context() as patched:
        # A card's file never lists every chat, and its sandbox is walked once for every card.
        patched.setattr(library, "_sandbox_sessions", listed)
        response = _download(client, "sandbox:t-lib:a.txt")
        assert response.content == b"mine"
        assert 'filename="a.txt"' in response.headers["content-disposition"]
        assert status("b.txt") == 200 and len(walks) == 1
        # Past the cap, hidden, too deep, outside, or in no chat.
        for name in ("c.txt", ".env", "d1/d2/d3/d4/deep.txt", "../x", ".hidden"):
            assert status(name) == 404, name
        assert _download(client, "sandbox:nope:a.txt").status_code == 404
    assert set(_items(client)[0]) == {"sandbox:t-lib:a.txt", "sandbox:t-lib:b.txt"}
    # The listing leaves its walk for the per-item routes.
    library.invalidate_listing()
    _items(client)
    walks.clear()
    assert (status("a.txt"), status("c.txt"), walks) == (200, 404, [])
    for item_id in ("sandbox:t-lib:.env", "sandbox:t-lib:c.txt", "sandbox:not-a-chat:a.txt"):
        assert _delete(client, item_id) == 404, item_id
    assert os.path.exists(os.path.join(directory, ".env"))
    assert os.path.exists(os.path.join(directory, "c.txt"))
    assert _delete(client, "sandbox:t-lib:a.txt") == 200
    assert not os.path.exists(os.path.join(directory, "a.txt"))


@pytest.mark.parametrize("route", ["download", "stream"])
def test_a_sandbox_file_swapped_for_a_link_after_the_check_is_not_read(
    client, signed_in, monkeypatch, tmp_path, route
):
    secret = _file(tmp_path / "secret.mp3", "secret")
    _directory, path = _sandbox_chat("song.mp3", b"mine")
    item_id = "sandbox:t-lib:song.mp3"
    if route == "stream":
        url = _stream_url(client, item_id)
        assert _app(None).get(url).content == b"mine"
    real = library._sandbox_path
    swapped = []

    def swap_after_check(ref):
        checked = real(ref)
        if not swapped:
            # Tool code racing the route: the name checked is a link by the time it is opened.
            os.unlink(path)
            os.symlink(secret, path)
            swapped.append(ref)
        return checked

    monkeypatch.setattr(library, "_sandbox_path", swap_after_check)
    response = _app(None).get(url) if route == "stream" else _download(client, item_id)
    assert response.status_code == 404
    assert b"secret" not in response.content


def test_a_projects_own_files_are_listed(client, monkeypatch, tmp_path):
    from core.inference.tools import resolve_sandbox_workdir
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects home"))
    monkeypatch.setattr(library, "_SOURCES", (library._sandbox_items,))
    project = studio_db.upsert_chat_project(
        {"id": "p-lib", "name": "Research", "createdAt": 1, "updatedAt": 1}
    )
    # Studio gives every project a folder of its own, so the column is never empty.
    assert project["rootPath"]
    directory = resolve_sandbox_workdir("project-p-lib")
    os.makedirs(os.path.join(directory, "files"), exist_ok = True)
    Path(directory, "files", "notes.txt").write_text("x")
    item = _items(client)[0]["sandbox:project-p-lib:files/notes.txt"]
    assert item["threadTitle"] == "Research"

    # One pointed at a folder of the user's own keeps its files out of the Library.
    own = tmp_path / "My code"
    conn = studio_db.get_connection()
    conn.execute("UPDATE chat_projects SET root_path = ? WHERE id = 'p-lib'", (str(own),))
    conn.commit()
    conn.close()
    assert not library._studio_project_root(str(own))
    assert "sandbox:project-p-lib:files/notes.txt" not in _items(client)[0]


@pytest.fixture
def revealed(monkeypatch):
    import utils.paths.file_manager as file_manager
    import utils.paths.path_utils as path_utils

    calls = []
    # CI runs on a headless Linux, where no file manager is reported.
    monkeypatch.setattr(file_manager, "file_manager_kind", lambda: "files")
    monkeypatch.setattr(path_utils, "reveal_in_file_manager", lambda path: calls.append(str(path)))
    return calls


def test_reveal_opens_an_items_own_file_or_a_location_by_key(client, revealed):
    reveal = lambda item_id: _post(client, "items/reveal", id = item_id).status_code  # noqa: E731
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    assert reveal("attachment:m:a") == 400
    assert reveal("upload:0123456789abcdef0123456789abcdef") == 404
    # A model id carries its path, so one outside the outputs and exports roots is not opened.
    assert reveal("model:training:/etc") == 404
    assert _post(client, "locations/reveal", key = "/etc").status_code == 404
    assert revealed == []
    assert reveal(note) == 200
    locations = client.get("/api/library/locations").json()["locations"]
    paths = {entry["key"]: entry["path"] for entry in locations}
    assert set(paths) == {"uploads", "images", "videos", "audio", "fineTunes", "exports"}
    assert _post(client, "locations/reveal", key = "images").status_code == 200
    assert revealed == [str(library.upload_path(_ref(note))), paths["images"]]


def test_reveal_is_refused_to_a_managed_account_and_where_no_file_manager_is(
    client, revealed, monkeypatch
):
    import utils.paths.file_manager as file_manager
    import utils.paths.path_utils as path_utils

    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    reveal = lambda: (  # noqa: E731
        _post(client, "items/reveal", id = note).status_code,
        _post(client, "locations/reveal", key = "images").status_code,
    )
    with monkeypatch.context() as managed:
        managed.setattr(library_routes.account_access, "managed_account", lambda: True)
        assert reveal() == (403, 403)
    monkeypatch.setattr(file_manager, "file_manager_kind", lambda: None)
    assert reveal() == (503, 503)
    assert revealed == []

    # Popen not finding xdg-open raises what a vanished file would; the file is there, so 503.
    def no_launcher(path):
        raise FileNotFoundError(2, "No such file or directory: 'xdg-open'")

    monkeypatch.setattr(file_manager, "file_manager_kind", lambda: "files")
    monkeypatch.setattr(path_utils, "reveal_in_file_manager", no_launcher)
    response = _post(client, "items/reveal", id = note)
    assert response.status_code == 503
    assert response.json()["detail"] == "No file manager is available on this machine"


@pytest.mark.parametrize(
    "platform, os_name, wsl, container, display, expected",
    [
        ("darwin", "posix", False, False, None, "finder"),
        ("win32", "nt", False, False, None, "explorer"),
        # WSL reveals in the Windows host's Explorer.
        ("linux", "posix", True, False, None, "explorer"),
        ("linux", "posix", False, False, None, None),
        ("linux", "posix", False, False, "WAYLAND_DISPLAY", "files"),
        # Never in a container, even with a display.
        ("linux", "posix", False, True, "DISPLAY", None),
    ],
)
def test_file_manager_is_named_only_where_a_window_can_appear(
    monkeypatch, platform, os_name, wsl, container, display, expected
):
    from utils.paths import file_manager, path_utils

    monkeypatch.setattr(file_manager.sys, "platform", platform)
    monkeypatch.setattr(file_manager.os, "name", os_name)
    monkeypatch.setattr(path_utils, "_IS_WSL", wsl)
    monkeypatch.setattr(file_manager, "_in_container", lambda: container)
    for name in ("DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising = False)
    if display:
        monkeypatch.setenv(display, ":0")
    assert file_manager.file_manager_kind() == expected


def test_explorer_gets_the_documented_select_command(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    import utils.paths.path_utils as path_utils

    target = tmp_path / "a b" / "c.txt"
    target.parent.mkdir()
    target.write_text("x")
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda command, *args, **kwargs: calls.append(command))
    with monkeypatch.context() as windows:
        windows.setattr(path_utils.sys, "platform", "win32")
        windows.setattr(path_utils.os, "name", "nt")
        path_utils.reveal_in_file_manager(target)
    # One string: a list quotes "/select,<path>" whole, which Explorer misreads when it has a space.
    assert calls.pop() == f'explorer /select,"{target}"'
    # WSL interop quotes each argument itself, so there the switch and the path go apart.
    windows_path = "C:\\Users\\me\\a b\\c.txt"
    monkeypatch.setattr(path_utils, "_IS_WSL", True)
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout = f"{windows_path}\n")
    )
    assert path_utils._wsl_reveal_in_explorer(target, is_file = True)
    assert calls == [["explorer.exe", "/select,", windows_path]]


@pytest.mark.parametrize(
    "path, root, inside",
    [
        (r"\\?\C:\Users\me\Studio\library\x", r"C:\Users\me\Studio", True),
        (r"C:\Users\me\Studio\library\x", r"\\?\c:\users\ME\studio", True),
        (r"\\?\UNC\server\share\Studio\x", r"\\server\share\Studio", True),
        (r"C:\Users\me\Studio2\x", r"C:\Users\me\Studio", False),
        (r"D:\Studio\x", r"C:\Studio", False),
        (r"C:\Studio", r"C:\Studio", False),
        ("C:\\x", "C:\\", True),
    ],
)
def test_containment_ignores_the_windows_long_path_prefix(path, root, inside):
    import ntpath

    from utils.paths.path_utils import is_path_within
    assert is_path_within(path, root, pathmod = ntpath) is inside


@pytest.mark.parametrize(
    "size, mode, thumbnail",
    [
        ((3000, 2000), "RGB", (640, 427)),
        ((1000, 5000), "RGBA", (640, 960)),
        ((4000, 500), "RGB", (640, 427)),
        # Never scaled up.
        ((100, 80), "RGB", (100, 80)),
    ],
)
def test_image_thumbnail_is_bounded_and_cropped_as_the_card_shows_it(client, size, mode, thumbnail):
    [image] = _upload(client, ("pic.png", _png(*size, mode), "application/octet-stream"))
    assert _thumbnail_size(client, image) == thumbnail


def _clip(width, height) -> bytes:
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = "mp4") as out:
        stream = out.add_stream("libx264", rate = 4)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        pixels = np.full((height, width, 3), 80, dtype = np.uint8)
        out.mux(stream.encode(av.VideoFrame.from_ndarray(pixels, format = "rgb24")))
        out.mux(stream.encode())
    return buf.getvalue()


def test_a_tall_clip_thumbnail_is_bounded_in_height_too(client):
    # Small to decode, but a card showing it at full height would hold every row.
    [clip] = _upload(client, ("tall.mp4", _clip(32, 4000), "video/mp4"))
    assert _thumbnail_size(client, clip) == (8, 960)


def test_video_upload_thumbnail_is_its_first_frame(client):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = "mp4") as out:
        stream = out.add_stream("libx264", rate = 4)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(4):
            pixels = np.full((48, 64, 3), index * 40, dtype = np.uint8)
            out.mux(stream.encode(av.VideoFrame.from_ndarray(pixels, format = "rgb24")))
        out.mux(stream.encode())
    [clip] = _upload(client, ("clip.mp4", buf.getvalue(), "video/mp4"))
    assert _thumbnail_size(client, clip) == (64, 48)


def test_what_is_not_a_small_raster_or_clip_has_no_thumbnail(client, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(library, "_THUMBNAIL_MAX_PIXELS", 100 * 100)
    ppm = io.BytesIO()
    Image.new("RGB", (8, 8)).save(ppm, format = "PPM")
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 8 8\nshowpage\n"
    undecodable = _upload(
        client,
        ("huge.png", _png(101, 100, "RGB"), "image/png"),
        ("broken.png", b"not a png", "image/png"),
        # A format Pillow knows but a card does not need is never handed to its decoder.
        ("pic.png", ppm.getvalue(), "image/png"),
        ("pic.jpg", eps, "image/jpeg"),
        ("broken.mp4", b"not a video", "video/mp4"),
    )
    for item_id in undecodable:
        assert _thumbnail(client, item_id).status_code == 501, item_id
    no_picture = _upload(
        client, ("note.md", b"# hi", "text/markdown"), ("icon.svg", b"<svg/>", "image/svg+xml")
    )
    for item_id in (*no_picture, "upload:" + "0" * 32, "model:training:/tmp/x", "image:missing"):
        assert _thumbnail(client, item_id).status_code == 404, item_id


def test_a_video_thumbnail_reads_its_container_by_type(client, monkeypatch):
    from core.inference import video_gallery

    seen = []
    monkeypatch.setattr(
        video_gallery,
        "first_frame_webp",
        lambda source, **kwargs: seen.append((hasattr(source, "read"), kwargs)) or b"webp",
    )
    [clip, playlist, flv] = _upload(
        client,
        ("clip.webm", b"x", "video/webm"),
        ("clip.m3u8", b"#EXTM3U", "video/mp4"),
        ("clip.flv", b"x", "video/x-flv"),
    )
    assert [_thumbnail(client, item_id).status_code for item_id in (clip, playlist)] == [200, 200]
    # A descriptor, never a name to reopen, and a demuxer forced rather than probed, so a playlist
    # sent as mp4 is read as mp4 and never followed as HLS.
    assert [(is_stream, kwargs["container"]) for is_stream, kwargs in seen] == [
        (True, "webm"),
        (True, "mp4"),
    ]
    assert seen[0][1]["max_pixels"] == library._THUMBNAIL_MAX_PIXELS
    # A container with no demuxer on the list has no picture.
    assert _thumbnail(client, flv).status_code == 404


def test_an_attachment_is_read_for_its_thumbnail_only_within_the_decodes_count(client, monkeypatch):
    free = []

    def media(_ref):
        # A 64 MiB clip and its base64 are held from here, so this has to wait its turn too.
        free.append(library._THUMBNAIL_DECODES._value)
        return "image/png", _png(8, 8, "RGB")

    monkeypatch.setattr(library, "_attachment_media", media)
    assert _thumbnail(client, "attachment:m:a").status_code == 200
    assert free == [1]


def test_thumbnails_are_cached_by_version_and_attachments_have_them_too(client, monkeypatch):
    import storage.studio_db as studio_db

    calls = []
    real = library._decode
    monkeypatch.setattr(
        library, "_decode", lambda mime, source: calls.append(mime) or real(mime, source)
    )
    [image] = _upload(client, ("a.png", _png(40, 30, "RGB"), "image/png"))
    for _ in range(3):
        assert _thumbnail_size(client, image) == (40, 30)
    assert calls == ["image/png"]
    # Another account can have an item of the same id and version: its sign-in keys it apart.
    assert _thumbnail(client, image).headers["vary"] == "Authorization"
    # A new version of the file is a new size and mtime, so a new picture.
    library.upload_path(_ref(image)).write_bytes(_png(30, 20, "RGB"))
    assert _thumbnail_size(client, image) == (30, 20)
    assert len(calls) == 2

    data = base64.b64encode(_png(120, 90, "RGB")).decode("ascii")
    attachment = {
        "type": "image",
        "contentType": "image/png",
        "content": [{"type": "image", "image": f"data:image/png;base64,{data}"}],
    }
    monkeypatch.setattr(studio_db, "get_chat_attachment", lambda *_ids: attachment)
    for _ in range(2):
        assert _thumbnail_size(client, "attachment:m:a") == (120, 90)
    assert len(calls) == 3
    attachment["content"] = [{"type": "text", "text": "just words"}]
    assert _thumbnail(client, "attachment:m:a").status_code == 404


def _mint(client, item_id):
    return client.get("/api/library/items/stream-url", params = {"id": item_id})


def _stream_url(client, item_id):
    response = _mint(client, item_id)
    assert response.status_code == 200, response.text
    url = response.json()["url"]
    assert url.startswith("/api/library/items/stream?")
    return url


def _stream(item_id, token):
    # The real dependencies: the stream route must not need the bearer the mint does.
    return _app(None).get("/api/library/items/stream", params = {"id": item_id, "token": token})


def test_audio_and_video_stream_from_a_signed_link_with_ranges(client):
    [clip, song] = _upload(
        client, ("clip.mp4", _CLIP, "video/mp4"), ("song.mp3", b"ID3" + b"x" * 7, "audio/mpeg")
    )
    url = _stream_url(client, clip)
    stream = _app(None)
    whole = stream.get(url)
    assert (whole.status_code, whole.content) == (200, _CLIP)
    for header, value in (
        ("content-type", "video/mp4"),
        ("x-content-type-options", "nosniff"),
        ("cache-control", "private"),
        ("accept-ranges", "bytes"),
        ("content-length", "100"),
    ):
        assert whole.headers[header] == value, header
    for header, body, content_range in (
        ("bytes=10-19", _CLIP[10:20], "bytes 10-19/100"),
        ("bytes=95-", _CLIP[95:], "bytes 95-99/100"),
        ("bytes=90-500", _CLIP[90:], "bytes 90-99/100"),
        ("bytes=-5", _CLIP[-5:], "bytes 95-99/100"),
        ("bytes=-500", _CLIP, "bytes 0-99/100"),
    ):
        response = stream.get(url, headers = {"Range": header})
        assert (response.status_code, response.content) == (206, body), header
        assert response.headers["content-range"] == content_range, header
        assert response.headers["content-length"] == str(len(body)), header
    # Several ranges, a malformed one, or an If-Range with nothing to match may be ignored.
    for headers in (
        {"Range": "bytes=0-1,5-6"},
        {"Range": "bytes=5-1"},
        {"Range": "items=0-1"},
        {"Range": "bytes=a-b"},
        {"Range": "bytes=0-1", "If-Range": '"etag"'},
    ):
        response = stream.get(url, headers = headers)
        assert (response.status_code, response.content) == (200, _CLIP), headers
    for header in ("bytes=100-", "bytes=-0"):
        response = stream.get(url, headers = {"Range": header})
        assert response.status_code == 416, header
        assert response.headers["content-range"] == "bytes */100"
    head = stream.head(url, headers = {"Range": "bytes=0-9"})
    assert (head.status_code, head.content, head.headers["content-length"]) == (206, b"", "10")
    # Audio, and generated media under its gallery's type.
    for item_id, content_type, body in (
        (song, "audio/mpeg", b"ID3" + b"x" * 7),
        (f"video:{_gallery_video('A calm sea')}", "video/mp4", b"\0\0\0\x18ftypmp42"),
        (f"audio:{_gallery_audio('Hello')}", "audio/wav", b"RIFF0000WAVE"),
    ):
        response = stream.get(_stream_url(client, item_id))
        assert (response.status_code, response.headers["content-type"]) == (200, content_type)
        assert response.content == body


def test_only_audio_and_video_items_get_a_stream_link(client):
    [note, image, unknown] = _upload(
        client,
        ("note.md", b"# hi", "text/markdown"),
        ("pic.png", b"\x89PNG", "image/png"),
        # A declared type the extension map does not know is stored, but never streamed.
        ("clip.xyz", b"x", "video/x-custom"),
    )
    # A sandbox file, too, streams only under an audio or video name.
    _sandbox_chat("notes.txt", b"text")
    mint = lambda item_id: _mint(client, item_id).status_code  # noqa: E731
    for item_id in (note, image, unknown, "sandbox:t-lib:notes.txt", "attachment:m:a"):
        assert mint(item_id) == 400, item_id
    for item_id in ("model:training:/x", "elsewhere:x"):
        assert mint(item_id) == 400, item_id
    assert (mint("upload:" + "0" * 32), mint("video:nope")) == (404, 404)
    # A link can never be made for one, so the stream refuses it even with a valid signature.
    response = _stream(note, library_routes._sign_stream_id(note))
    assert response.status_code == 404 and b"# hi" not in response.content
    # Minted only for a caller signed in with the UI or a key.
    assert _mint(_app(None), note).status_code in (401, 403)
    client.app.dependency_overrides[request_admitted_without_credential] = lambda: True
    assert mint(note) == 403


def test_a_tampered_expired_foreign_or_other_kind_of_link_is_refused(client, monkeypatch):
    import routes.video as video_routes

    [clip, other] = _upload(
        client, ("clip.mp4", _CLIP, "video/mp4"), ("other.mp4", b"other", "video/mp4")
    )
    token = library_routes._sign_stream_id(clip)
    assert _stream(clip, token).status_code == 200
    target, expires, signature = token.rsplit(".", 2)
    for bad in (
        f"{target}.{expires}.{'0' * len(signature)}",
        f"{target}.{int(expires) + 60}.{signature}",
        f"{library_routes._sign_stream_id(other).rsplit('.', 2)[0]}.{expires}.{signature}",
        "nonsense",
        "",
    ):
        assert _stream(clip, bad).status_code in (401, 422), bad
    # Names one item: another's id with it reads nothing.
    response = _stream(other, token)
    assert response.status_code == 401 and b"other" not in response.content

    # A Video page link never opens this route, nor one of these the Video page's.
    video = _gallery_video("A calm sea")
    app = FastAPI()
    app.include_router(video_routes.router, prefix = "/api/inference")
    video_page = TestClient(app).get
    video_token = video_routes._sign_video_id(video)
    signed_url = f"/api/inference/video/gallery/{video}/file-signed"
    assert _stream(f"video:{video}", video_token).status_code == 401
    library_token = library_routes._sign_stream_id(video)
    assert video_page(signed_url, params = {"token": library_token}).status_code == 401
    # Each on its own route still plays.
    assert video_page(signed_url, params = {"token": video_token}).status_code == 200

    monkeypatch.setattr(library_routes, "_STREAM_LINK_TTL", -1)
    assert _stream(clip, library_routes._sign_stream_id(clip)).status_code == 401


@pytest.fixture
def two_accounts(monkeypatch, tmp_path):
    from auth import policy, storage as auth_storage
    from utils.account_context import AccountContext

    alice = AccountContext("a" * 32, "alice")
    bob = AccountContext("b" * 32, "bob")
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(auth_storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    connection = auth_storage.get_connection()
    with connection:
        for account in (alice, bob):
            connection.execute(
                "INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret,"
                " account_id, role, is_active) VALUES (?, 'salt', 'hash', 'secret', ?, 'user', 1)",
                (account.username, account.account_id),
            )
    connection.close()
    yield alice, bob
    policy.invalidate_account_cache()


def _account_client(account):
    from utils.account_context import bind_account, reset_account
    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    return _app(subject)


def test_a_stream_link_reads_only_its_own_accounts_item(client, two_accounts):
    from utils.account_context import OWNER, run_as

    alice, bob = two_accounts
    mine = run_as(alice, library.save_upload, "clip.mp4", "video/mp4", iter([b"alice"]))
    theirs = run_as(bob, library.save_upload, "clip.mp4", "video/mp4", iter([b"bob"]))
    mine_id, theirs_id = f"upload:{mine['id']}", f"upload:{theirs['id']}"
    sign = lambda account, item_id: run_as(account, library_routes._sign_stream_id, item_id)  # noqa: E731
    with _account_client(alice) as as_alice, _account_client(bob) as as_bob:
        url = _stream_url(as_alice, mine_id)
        # Bob cannot mint a link to Alice's item, however he spells it.
        assert _mint(as_bob, mine_id).status_code == 404
    # The link is the credential: it plays Alice's file for whoever holds it, in her account.
    response = _app(None).get(url)
    assert (response.status_code, response.content) == (200, b"alice")
    response = _stream(theirs_id, sign(alice, theirs_id))
    assert response.status_code == 404 and b"bob" not in response.content
    assert _stream(theirs_id, sign(alice, mine_id)).status_code == 401
    # An owner's link resolves in the owner's own Library, where Alice's upload is not.
    assert _stream(mine_id, sign(OWNER, mine_id)).status_code == 404
