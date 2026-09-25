# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import io
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
import hub.storage.scan_folders as _scan_folders  # noqa: E402

from .test_rag_native_drop_upload import SECRET, _sign  # noqa: E402

# Before any fixture swaps them for tests that use temp folders.
_REAL_DENIED = _scan_folders.is_denied_system_path
_REAL_SCRATCH = library._scratch_and_system_folders
_REAL_MOVE_TARGET = library._move_target

_CLIP = bytes(range(100))
_SANDBOX_ID = "sandbox:t-lib:report.txt"


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


def _sandbox_chat(name, body):
    """A file ``name`` in chat t-lib's sandbox: (sandbox directory, its path)."""
    from core.inference.tools import resolve_sandbox_workdir
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": "t-lib", "title": "T", "modelType": "base", "modelId": "m", "createdAt": 1}
    )
    directory = resolve_sandbox_workdir("t-lib")
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


def test_opening_an_item_records_when(client):
    [note] = _upload(client, ("note.md", b"hi", "text/markdown"))
    assert _items(client)[0][note]["openedAt"] is None
    assert _post(client, "items/opened", id = note).status_code == 200
    opened = _items(client)[0][note]
    assert opened["openedAt"] >= opened["createdAt"]
    assert not opened["favorite"]


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


def test_the_listing_reports_the_library_disk(client):
    disk = client.get("/api/library").json()["disk"]
    assert disk["totalBytes"] > 0
    assert 0 <= disk["freeBytes"] <= disk["totalBytes"]
    # Everything sits under one test home here, so every source is on the measured disk.
    assert set(disk["sources"]) == {
        "upload",
        "attachment",
        "image",
        "video",
        "audio",
        "model:training",
        "model:exported",
        "sandbox",
    }


def test_fine_tunes_and_exports_are_placed_on_disks_separately(client, monkeypatch):
    from utils.paths.storage_roots import exports_root

    real = library._device
    exports = str(exports_root())
    monkeypatch.setattr(library, "_device", lambda path: -1 if str(path) == exports else real(path))
    sources = client.get("/api/library").json()["disk"]["sources"]
    assert "model:training" in sources and "model:exported" not in sources


def test_a_source_on_another_disk_is_left_out_of_the_bar(client, monkeypatch):
    real = library._device
    uploads = library.uploads_dir()
    monkeypatch.setattr(
        library, "_device", lambda path: real(uploads) if str(path) == str(uploads) else -1
    )
    assert client.get("/api/library").json()["disk"]["sources"] == ["upload"]


def test_a_broken_source_root_leaves_the_listing_and_the_rest(client, monkeypatch):
    def broken():
        raise ImportError("no sandbox")

    monkeypatch.setitem(library._SOURCE_ROOTS, "sandbox", broken)
    response = client.get("/api/library")
    assert response.status_code == 200
    sources = response.json()["disk"]["sources"]
    assert "sandbox" not in sources and "upload" in sources


def test_a_gguf_export_counts_every_quantization(client, monkeypatch):
    import shutil

    from utils.paths.storage_roots import exports_root

    monkeypatch.setattr(library, "_SOURCES", (library._model_items,))
    run = exports_root() / "library-test-gguf"
    run.mkdir(parents = True)
    (run / "model.Q4_K_M.gguf").write_bytes(b"x" * 10)
    (run / "model.Q8_0.gguf").write_bytes(b"x" * 20)
    try:
        [item] = [item for item in _items(client)[0].values() if item["name"] == run.name]
        assert item["sizeBytes"] == 30
        # Its star is kept by the listed file, which is what a lookup by id checks it against.
        _patch(client, id = item["id"], favorite = True)
        assert _items(client)[0][item["id"]]["favorite"] is True
        assert item["id"] in _favorites(client)
    finally:
        shutil.rmtree(run)


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
        # Once the models route has deleted it, the Library forgets its favorite too.
        _patch(client, id = item["id"], favorite = True)
        shutil.rmtree(run)
        assert _delete(client, item["id"]) == 200
        assert _favorites(client) == []
    finally:
        shutil.rmtree(run, ignore_errors = True)


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
    # Its rows are kept and gain the opened and fingerprint columns.
    assert library_db.list_entries()["x"] == {
        "name": None,
        "favorite": True,
        "folderId": None,
        "updatedAt": 1,
        "openedAt": None,
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
    ],
)
def test_names_written_to_disk_are_valid_on_windows(name, safe):
    from core.inference.gallery_projects import _bad_name

    assert library.safe_file_name(name) == safe
    project_name = library.safe_file_name(name, item_id = "upload:x")
    assert not _bad_name(project_name), project_name


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


# ── Moving a kind of file ────────────────────────────────────────


@pytest.fixture(autouse = True)
def _temp_folders_are_ordinary(monkeypatch):
    # macOS keeps pytest's temp folders under /private/var, which the real check refuses.
    import hub.storage.scan_folders as scan_folders

    monkeypatch.setattr(
        scan_folders, "is_denied_system_path", lambda path: path.startswith(("/etc", "/usr"))
    )
    # And pytest's temp folders are temporary folders, which Library moves refuse too.
    real = library._scratch_and_system_folders
    monkeypatch.setattr(
        library,
        "_scratch_and_system_folders",
        lambda: [
            folder for folder in real() if folder.startswith(("/usr", "/opt", "/Applications"))
        ],
    )


def _move(client, key, path):
    return client.post("/api/library/locations/move", json = {"key": key, "path": path})


def _location(client, key):
    locations = client.get("/api/library/locations").json()["locations"]
    return next(entry for entry in locations if entry["key"] == key)


def test_images_move_to_a_new_folder_and_back(client, tmp_path):
    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    (old / "a.png").write_bytes(b"png")
    (old / "a.json").write_text("{}")
    new = tmp_path / "Pictures" / "Unsloth"
    new.parent.mkdir()

    response = _move(client, "images", str(new))
    assert response.status_code == 200, response.text
    assert sorted(p.name for p in new.iterdir()) == ["a.json", "a.png"]
    assert not any(old.iterdir())
    assert image_gallery.gallery_dir() == new.resolve()
    location = _location(client, "images")
    assert {key: location[key] for key in ("key", "path", "movable", "custom", "available")} == {
        "key": "images",
        "path": str(new.resolve()),
        "movable": True,
        "custom": True,
        "available": True,
    }
    assert location["disk"]["freeBytes"] <= location["disk"]["totalBytes"]
    assert location["device"] == _location(client, "fineTunes")["device"]

    assert _move(client, "images", None).status_code == 200
    assert (old / "a.png").read_bytes() == b"png"
    assert image_gallery.gallery_dir() == old
    assert _location(client, "images")["custom"] is False


def test_uploads_follow_their_folder(client, tmp_path):
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    assert _move(client, "uploads", str(tmp_path / "uploads")).status_code == 200
    upload_id = note.split(":", 1)[1]
    assert client.get(f"/api/library/uploads/{upload_id}/file").content == b"# plan"


def test_a_move_needs_an_empty_ordinary_folder(client, tmp_path):
    from core.inference import video_gallery

    # A folder that already holds a same-named "Unsloth Images" with files in it.
    full = tmp_path / "full"
    (full / "Unsloth Images").mkdir(parents = True)
    (full / "Unsloth Images" / "keep.txt").write_text("mine")
    assert _move(client, "images", str(full)).status_code == 400
    assert _move(client, "images", "relative/folder").status_code == 400
    assert _move(client, "images", "/etc/unsloth-images").status_code == 400
    assert _move(client, "images", str(tmp_path / "missing" / "deeper")).status_code == 400
    # Inside another kind's folder, where its listing would pick the files up.
    assert _move(client, "images", str(video_gallery.gallery_dir() / "images")).status_code == 400
    assert (full / "Unsloth Images" / "keep.txt").read_text() == "mine"


def test_a_folder_with_files_gets_a_named_folder_inside(client, tmp_path):
    from core.inference import image_gallery

    (image_gallery.gallery_dir() / "a.png").write_bytes(b"png")
    pictures = tmp_path / "Pictures"
    pictures.mkdir()
    (pictures / "holiday.jpg").write_bytes(b"jpg")
    assert _move(client, "images", str(pictures)).status_code == 200
    assert image_gallery.gallery_dir() == (pictures / "Unsloth Images").resolve()
    assert (pictures / "Unsloth Images" / "a.png").read_bytes() == b"png"
    assert (pictures / "holiday.jpg").read_bytes() == b"jpg"


def test_moving_back_to_a_default_that_holds_files_keeps_them_listed(client, tmp_path):
    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    (old / "a.png").write_bytes(b"png")
    assert _move(client, "images", str(tmp_path / "elsewhere")).status_code == 200
    (old / "stray.txt").write_text("left behind")
    assert _move(client, "images", None).status_code == 200
    assert image_gallery.gallery_dir() == (old / "Unsloth Images").resolve()
    assert (image_gallery.gallery_dir() / "a.png").read_bytes() == b"png"
    assert _location(client, "images")["custom"] is True


def test_a_named_subfolder_that_is_another_kinds_folder_is_refused(client, tmp_path):
    from core.inference import image_gallery, video_gallery

    data = tmp_path / "data"
    data.mkdir()
    assert _move(client, "videos", str(data / "Unsloth Images")).status_code == 200
    (data / "notes.txt").write_text("mine")
    (image_gallery.gallery_dir() / "a.png").write_bytes(b"png")
    assert _move(client, "images", str(data)).status_code == 400
    assert video_gallery.gallery_dir() == (data / "Unsloth Images").resolve()
    assert (image_gallery.gallery_dir() / "a.png").read_bytes() == b"png"


def test_a_chat_sandbox_cannot_hold_moved_files(client):
    from core.inference.tools import sandbox_root

    session = Path(sandbox_root()) / "chat-1"
    session.mkdir(parents = True, exist_ok = True)
    assert _move(client, "uploads", str(session / "uploads")).status_code == 400
    assert _move(client, "images", sandbox_root()).status_code == 400
    assert not (session / "uploads").exists()


def test_a_named_subfolder_linking_into_another_kinds_folder_is_refused(client, tmp_path):
    from core.inference import video_gallery

    data = tmp_path / "data"
    data.mkdir()
    (data / "notes.txt").write_text("mine")
    (data / "Unsloth Images").symlink_to(video_gallery.gallery_dir(), target_is_directory = True)
    assert _move(client, "images", str(data)).status_code == 400
    assert not any(video_gallery.gallery_dir().iterdir())


def test_a_named_subfolder_linking_into_a_credential_folder_is_refused(client, tmp_path):
    from core.inference import image_gallery

    ssh = tmp_path / "home" / ".ssh"
    ssh.mkdir(parents = True)
    data = tmp_path / "data"
    data.mkdir()
    (data / "notes.txt").write_text("mine")
    (data / "Unsloth Images").symlink_to(ssh, target_is_directory = True)
    (image_gallery.gallery_dir() / "a.png").write_bytes(b"png")
    response = _move(client, "images", str(data))
    assert response.status_code == 400
    assert "credential" in response.json()["detail"]
    assert not any(ssh.iterdir())
    assert (image_gallery.gallery_dir() / "a.png").read_bytes() == b"png"


def test_an_unplugged_folder_is_not_made_again(client, tmp_path):
    import shutil

    from core.inference import image_gallery
    from utils.paths.relocations import LocationUnavailable

    drive = tmp_path / "Drive" / "Images"
    drive.parent.mkdir()
    assert _move(client, "images", str(drive)).status_code == 200
    shutil.rmtree(drive.parent)
    with pytest.raises(LocationUnavailable):
        image_gallery.gallery_dir()
    assert not drive.parent.exists()
    # Settings still shows it, will not make it to reveal it, refuses to move files it cannot
    # reach, and can reset it.
    assert _location(client, "images")["path"] == str(drive.resolve())
    response = client.post("/api/library/locations/reveal", json = {"key": "images"})
    assert response.status_code == 409
    assert not drive.parent.exists()
    assert _move(client, "images", str(tmp_path / "elsewhere")).status_code == 400
    assert _move(client, "images", None).status_code == 200
    assert _location(client, "images")["custom"] is False
    assert image_gallery.gallery_dir().is_dir()


def test_fine_tunes_and_exports_do_not_move(client, tmp_path):
    for key in ("fineTunes", "exports", "somewhere"):
        assert _move(client, key, str(tmp_path / key)).status_code == 400
    assert _location(client, "fineTunes")["movable"] is False


def test_only_the_installation_owner_can_move(client, tmp_path, monkeypatch):
    monkeypatch.setattr(library_routes.account_access, "managed_account", lambda: True)
    assert _move(client, "images", str(tmp_path / "elsewhere")).status_code == 403
    assert not (tmp_path / "elsewhere").exists()


def test_a_failed_move_puts_everything_back(client, tmp_path, monkeypatch):
    import errno
    import shutil

    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    for name in ("a.png", "b.png", "c.png"):
        (old / name).write_bytes(name.encode())
    (old / "d").mkdir()
    (old / "d" / "e.png").write_bytes(b"e")

    # Another drive: nothing renames, so every entry is copied, and the second copy runs out of
    # space part way, leaving half a file behind it.
    def cross_device(_src, _dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    real_copy = shutil.copy2
    calls = []

    def filling_copy(src, dst, **kwargs):
        calls.append(src)
        if len(calls) == 2:
            Path(dst).write_bytes(b"par")
            raise OSError(28, "No space left on device")
        return real_copy(src, dst, **kwargs)

    monkeypatch.setattr(library, "_rename", cross_device)
    monkeypatch.setattr(library.shutil, "copy2", filling_copy)
    target = tmp_path / "small-disk"
    response = _move(client, "images", str(target))
    assert response.status_code == 500
    assert "No space left" in response.json()["detail"]
    monkeypatch.setattr(library.shutil, "copy2", real_copy)
    assert sorted(p.name for p in old.iterdir()) == ["a.png", "b.png", "c.png", "d"]
    assert (old / "a.png").read_bytes() == b"a.png"
    assert not any(target.iterdir())
    assert image_gallery.gallery_dir() == old
    assert _location(client, "images")["custom"] is False


def test_a_file_another_program_holds_open_stays_put_whole(client, tmp_path, monkeypatch):
    import errno

    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    (old / "a.png").write_bytes(b"a")
    (old / "d").mkdir()
    (old / "d" / "keep.png").write_bytes(b"keep")
    (old / "d" / "open.png").write_bytes(b"open")

    def cross_device(_src, _dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    real_unlink = library._unlink

    def locked(path):
        # Windows copies an open file but will not delete it (ERROR_SHARING_VIOLATION).
        if Path(path).name == "open.png":
            exc = PermissionError(errno.EACCES, "The process cannot access the file")
            exc.winerror = 32
            raise exc
        real_unlink(path)

    monkeypatch.setattr(library, "_rename", cross_device)
    monkeypatch.setattr(library, "_unlink", locked)
    target = tmp_path / "other-disk"
    response = _move(client, "images", str(target))
    assert response.status_code == 500
    assert (
        "open.png is in use by another program. Close it and try again."
        in response.json()["detail"]
    )
    monkeypatch.setattr(library, "_unlink", real_unlink)
    # Everything back, and no copy of anything left in the target.
    assert (old / "a.png").read_bytes() == b"a"
    assert sorted(p.name for p in (old / "d").iterdir()) == ["keep.png", "open.png"]
    assert (old / "d" / "open.png").read_bytes() == b"open"
    assert not any(target.iterdir())
    assert image_gallery.gallery_dir() == old


def _fill(folder, count = 3):
    for i in range(count):
        (folder / f"{i}.png").write_bytes(b"png%d" % i)


def _files(folder):
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())


def _unresolved_target(monkeypatch):
    # resolve() keeps the spelling it is given: a case alias on a case-insensitive disk, or a bind
    # mount, names a folder by a path that is not its own. A link that resolve() is kept from
    # following stands in for either on any disk.
    monkeypatch.setattr(library, "_move_target", lambda raw: Path(raw))


def _case_insensitive(folder):
    return os.path.isdir(str(folder.parent / folder.name.upper()))


def test_another_spelling_of_the_current_folder_moves_nothing(client, tmp_path, monkeypatch):
    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    _fill(old)
    alias = tmp_path / "alias"
    alias.symlink_to(old, target_is_directory = True)
    _unresolved_target(monkeypatch)
    assert _move(client, "images", str(alias)).status_code == 200
    assert _files(old) == ["0.png", "1.png", "2.png"]
    assert image_gallery.gallery_dir() == old


def test_a_case_alias_of_the_current_folder_moves_nothing(client):
    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    if not _case_insensitive(old):
        pytest.skip("this disk tells case apart")
    _fill(old)
    assert _move(client, "images", str(old.parent / old.name.upper())).status_code == 200
    assert _files(old) == ["0.png", "1.png", "2.png"]
    assert image_gallery.gallery_dir() == old


def test_another_spelling_of_another_kinds_folder_is_refused(client, tmp_path, monkeypatch):
    from core.inference import image_gallery, video_gallery

    _fill(image_gallery.gallery_dir())
    videos = video_gallery.gallery_dir()
    (videos / "v.mp4").write_bytes(b"v")
    alias = tmp_path / "alias"
    alias.symlink_to(videos, target_is_directory = True)
    _unresolved_target(monkeypatch)
    assert _move(client, "images", str(alias)).status_code == 400
    if _case_insensitive(videos):
        monkeypatch.setattr(library, "_move_target", _REAL_MOVE_TARGET)
        response = _move(client, "images", str(videos.parent / videos.name.upper()))
        assert response.status_code == 400
    assert _files(videos) == ["v.mp4"]
    assert _files(image_gallery.gallery_dir()) == ["0.png", "1.png", "2.png"]


def test_unsloths_own_folder_and_folders_inside_the_current_one_are_refused(client):
    from core.inference import image_gallery
    from utils.paths import studio_root

    old = image_gallery.gallery_dir()
    _fill(old)
    (old / "sub").mkdir()
    for target in (studio_root(), studio_root() / "elsewhere", old / "sub", old / "new"):
        assert _move(client, "images", str(target)).status_code == 400, target
    assert _files(old) == ["0.png", "1.png", "2.png"]
    assert not (studio_root() / "elsewhere").exists()
    assert image_gallery.gallery_dir() == old


def test_a_move_never_moves_a_folder_into_itself(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _fill(source)
    (source / "Unsloth Images").mkdir()
    log = library._MoveLog()
    library._move_entries(source, source / "Unsloth Images", log)
    assert _files(source) == [f"Unsloth Images/{i}.png" for i in range(3)]


def test_reset_again_after_a_reset_into_a_full_default_is_a_no_op(client, tmp_path):
    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    (old / "a.png").write_bytes(b"png")
    assert _move(client, "images", str(tmp_path / "elsewhere")).status_code == 200
    (old / "stray.txt").write_text("left behind")
    assert _move(client, "images", None).status_code == 200
    response = _move(client, "images", None)
    assert response.status_code == 200, response.text
    assert image_gallery.gallery_dir() == (old / "Unsloth Images").resolve()
    assert (old / "Unsloth Images" / "a.png").read_bytes() == b"png"
    assert (old / "stray.txt").read_text() == "left behind"


def _cross_device(monkeypatch):
    import errno
    def cross_device(_src, _dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(library, "_rename", cross_device)


def _save_during_move(monkeypatch, save):
    """Calls `save(new_folder)` right after the move records the new folder."""
    from utils.paths import relocations

    real = relocations.set_chosen

    def set_and_save(key, path):
        real(key, path)
        if path is not None:
            save(Path(path))

    monkeypatch.setattr(relocations, "set_chosen", set_and_save)


def test_a_merge_across_drives_keeps_what_the_new_folder_already_holds(
    client, tmp_path, monkeypatch
):
    from core.inference import video_gallery

    old = video_gallery.gallery_dir()
    (old / ".jobs").mkdir()
    (old / ".jobs" / "old.json").write_text("old job")
    (old / "same.txt").write_text("same")
    (old / "clash.txt").write_text("from the old folder")
    (old / "v.mp4").write_bytes(b"v")

    def save(new):
        # A video job and two files saved into the new folder before the move reaches them.
        (new / ".jobs").mkdir()
        (new / ".jobs" / "new.json").write_text("running job")
        (new / "same.txt").write_text("same")
        (new / "clash.txt").write_text("from the new folder")

    _cross_device(monkeypatch)
    _save_during_move(monkeypatch, save)
    new = tmp_path / "videos"
    assert _move(client, "videos", str(new)).status_code == 200
    assert (new / ".jobs" / "new.json").read_text() == "running job"
    assert (new / ".jobs" / "old.json").read_text() == "old job"
    assert (new / "same.txt").read_text() == "same"
    assert (new / "clash.txt").read_text() == "from the new folder"
    assert (new / "clash (2).txt").read_text() == "from the old folder"
    assert (new / "v.mp4").read_bytes() == b"v"
    assert _files(old) == []


def test_a_failed_move_takes_back_what_was_saved_meanwhile(client, tmp_path, monkeypatch):
    import shutil

    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    _fill(old)
    real_copy = shutil.copy2
    calls = []

    def filling_copy(src, dst, **kwargs):
        calls.append(src)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return real_copy(src, dst, **kwargs)

    _cross_device(monkeypatch)
    monkeypatch.setattr(library.shutil, "copy2", filling_copy)
    _save_during_move(monkeypatch, lambda new: (new / "saved-meanwhile.png").write_bytes(b"new"))
    target = tmp_path / "target"
    assert _move(client, "images", str(target)).status_code == 500
    monkeypatch.setattr(library.shutil, "copy2", real_copy)
    assert _files(old) == ["0.png", "1.png", "2.png", "saved-meanwhile.png"]
    assert _files(target) == []
    assert image_gallery.gallery_dir() == old


def test_an_upload_finishing_during_a_move_lands_in_the_new_folder(client, tmp_path):
    new = tmp_path / "uploads"

    def chunks():
        yield b"first half, "
        # Moved while the upload is still writing to the old folder.
        library.move_location("uploads", str(new))
        yield b"second half"

    record = library.save_upload("slow.txt", "text/plain", chunks())
    assert (new / record["id"]).read_bytes() == b"first half, second half"
    assert client.get(f"/api/library/uploads/{record['id']}/file").content == (
        b"first half, second half"
    )
    assert not [
        p for p in library._location_default("uploads").iterdir() if p.name.endswith(".tmp")
    ]


def test_a_gallery_save_still_writing_is_waited_for(client, tmp_path):
    import threading
    import time

    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    (old / "a.png").write_bytes(b"a")
    writing = old / ".late.png.tmp"
    writing.write_bytes(b"late")

    def finish():
        time.sleep(0.5)
        os.replace(writing, old / "late.png")

    writer = threading.Thread(target = finish)
    writer.start()
    new = tmp_path / "images"
    assert _move(client, "images", str(new)).status_code == 200
    writer.join()
    assert _files(new) == ["a.png", "late.png"]
    assert _files(old) == []


def test_a_file_deleted_during_the_move_is_skipped(client, tmp_path, monkeypatch):
    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    _fill(old)
    real = library._rename

    def rename(src, dst):
        if Path(src).name == "1.png":
            # Deleted from the Library just before the move reached it.
            Path(src).unlink()
            raise FileNotFoundError(2, "No such file or directory")
        return real(src, dst)

    monkeypatch.setattr(library, "_rename", rename)
    new = tmp_path / "images"
    assert _move(client, "images", str(new)).status_code == 200
    assert _files(new) == ["0.png", "2.png"]
    assert image_gallery.gallery_dir() == new.resolve()


def test_a_drive_without_room_for_the_files_is_refused_up_front(client, tmp_path, monkeypatch):
    import shutil

    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    (old / "big.png").write_bytes(b"x" * 1000)
    target = tmp_path / "small-drive" / "images"
    target.parent.mkdir()
    real_device = library._device
    monkeypatch.setattr(
        library,
        "_device",
        lambda path: -1 if str(path).startswith(str(target.parent)) else real_device(path),
    )
    monkeypatch.setattr(
        library.shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(10**6, 10**6, 10)
    )
    response = _move(client, "images", str(target))
    assert response.status_code == 400
    assert "free" in response.json()["detail"]
    assert not target.exists()
    assert (old / "big.png").stat().st_size == 1000


def test_temporary_and_cache_folders_cannot_hold_library_files(monkeypatch):
    import tempfile

    import hub.storage.scan_folders as scan_folders

    # The real lists, temp folders and all.
    monkeypatch.setattr(scan_folders, "is_denied_system_path", _REAL_DENIED)
    monkeypatch.setattr(
        library, "_scratch_and_system_folders", lambda: [*_REAL_SCRATCH(), "/scratch-for-test"]
    )
    for folder in (Path(tempfile.gettempdir()) / "images", Path("/var/tmp/images")):
        with pytest.raises(ValueError, match = "cannot hold these files"):
            library._refuse_denied(folder.resolve())
    # Library-only: the model download folder's check lets it be.
    assert not _REAL_DENIED("/scratch-for-test/images")
    with pytest.raises(ValueError, match = "Temporary"):
        library._refuse_denied(Path("/scratch-for-test/images"))
    library._refuse_denied(Path.home() / "Pictures" / "Unsloth")


def _mounted(monkeypatch, mounts):
    real = os.path.ismount
    monkeypatch.setattr(os.path, "ismount", lambda path: str(path) in mounts or real(path))


def test_a_drive_unplugged_from_a_fixed_mount_point_is_unavailable(client, tmp_path, monkeypatch):
    from core.inference import image_gallery
    from utils.paths.relocations import LocationUnavailable

    (image_gallery.gallery_dir() / "a.png").write_bytes(b"png")
    drive = tmp_path / "mnt-usb"
    drive.mkdir()
    mounts = {str(drive.resolve())}
    _mounted(monkeypatch, mounts)
    assert _move(client, "images", str(drive)).status_code == 200
    # A mount point gets a named folder, so the drive's own top level stays the user's.
    assert image_gallery.gallery_dir() == (drive / "Unsloth Images").resolve()

    # Unplugged: the mount point is an ordinary empty folder on the system disk again.
    mounts.clear()
    with pytest.raises(LocationUnavailable):
        image_gallery.gallery_dir()
    location = _location(client, "images")
    assert location["available"] is False and location["disk"] is None
    assert client.post("/api/library/locations/reveal", json = {"key": "images"}).status_code == 409
    response = _move(client, "images", None)
    assert response.status_code == 200
    assert response.json()["leftBehind"] == str((drive / "Unsloth Images").resolve())
    assert _location(client, "images")["custom"] is False


def test_a_folder_chosen_before_mount_points_were_recorded_still_resolves(client, tmp_path):
    from core.inference import image_gallery
    from storage.studio_db import upsert_app_settings
    from utils.paths import relocations

    folder = tmp_path / "old-choice"
    folder.mkdir()
    upsert_app_settings({"library.locations": {"images": str(folder)}}, read_back = False)
    relocations.forget_cache()
    try:
        assert image_gallery.gallery_dir() == folder
        assert _location(client, "images")["custom"] is True
    finally:
        upsert_app_settings({"library.locations": {}}, read_back = False)
        relocations.forget_cache()


def test_a_folder_chosen_before_mount_points_learns_its_mount_once_present(
    client, tmp_path, monkeypatch
):
    from core.inference import image_gallery
    from storage.studio_db import get_app_setting, upsert_app_settings
    from utils.paths import relocations
    from utils.paths.relocations import LocationUnavailable

    drive = tmp_path / "mnt-usb"
    folder = drive / "Unsloth Images"
    mounts = set()
    _mounted(monkeypatch, mounts)
    upsert_app_settings({"library.locations": {"images": str(folder)}}, read_back = False)
    try:
        # Its drive is out: nothing to learn yet, so the bare path is kept as it was.
        relocations.forget_cache()
        with pytest.raises(LocationUnavailable):
            image_gallery.gallery_dir()
        assert get_app_setting("library.locations", {})["images"] == str(folder)

        # Back in: the mount is recorded, and saved.
        folder.mkdir(parents = True)
        mounts.add(str(drive.resolve()))
        relocations.forget_cache()
        assert image_gallery.gallery_dir() == folder
        saved = get_app_setting("library.locations", {})["images"]
        assert saved == {"path": str(folder), "mount": str(drive.resolve())}

        # Unplugged again, its empty mount point left behind: now that is caught.
        mounts.clear()
        relocations.forget_cache()
        with pytest.raises(LocationUnavailable):
            image_gallery.gallery_dir()
    finally:
        upsert_app_settings({"library.locations": {}}, read_back = False)
        relocations.forget_cache()


def test_an_unplugged_folder_is_reported_once_and_the_bar_falls_back(client, tmp_path, monkeypatch):
    import shutil

    logged = []

    class Logger:
        def debug(self, message, *args, **kwargs):
            logged.append(message % args)

        def info(self, message, *args, **kwargs):
            logged.append(message % args)

        def warning(self, message, *args, **kwargs):
            logged.append(message % args)

    drive = tmp_path / "Drive"
    drive.mkdir()
    assert _move(client, "uploads", str(drive / "uploads")).status_code == 200
    assert _move(client, "images", str(drive / "images")).status_code == 200
    shutil.rmtree(drive)
    monkeypatch.setattr(library, "logger", Logger())
    monkeypatch.setattr(library, "_reported_unavailable", set())
    for _ in range(3):
        disk = client.get("/api/library").json()["disk"]
        assert disk is not None and disk["totalBytes"] > 0
    assert len(logged) == len(set(logged)) == 2
    assert all("location_unavailable" in line for line in logged)


def test_folders_on_a_disk_without_file_ids_compare_by_spelling(tmp_path, monkeypatch):
    # FAT and some network shares report st_ino 0 for everything: equal ids must not make two
    # different folders "the same", which would refuse every move onto such a drive.
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    real_stat = os.stat

    def no_ids(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        return os.stat_result((result.st_mode, 0, *tuple(result)[2:]))

    monkeypatch.setattr(library.os, "stat", no_ids)
    assert not library._same_folder(a, b)
    assert not library._inside(b / "x", a)
    assert library._same_folder(a, tmp_path / "a")
    assert library._inside(a / "x", a)


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
