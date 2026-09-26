# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import contextlib
import errno
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
    authenticated_via_api_key,
    get_current_subject,
    request_admitted_without_credential,
)
from core import library  # noqa: E402
from routes import library as library_routes  # noqa: E402
from storage import library_db  # noqa: E402
import hub.storage.scan_folders as _scan_folders  # noqa: E402

from .test_rag_native_drop_upload import SECRET, _sign  # noqa: E402

_REAL_DENIED = _scan_folders.is_denied_system_path
_REAL_SCRATCH = library._scratch_and_system_folders

_CLIP = bytes(range(100))
_SANDBOX_ID = "sandbox:t-lib:report.txt"
_SANDBOX_A = "sandbox:t-lib:a.txt"


def _app(subject) -> TestClient:
    """The router alone; with a ``subject``, as a caller signed in with the UI or a key."""
    app = FastAPI()
    if subject is not None:
        app.dependency_overrides[get_current_subject] = subject
        app.dependency_overrides[request_admitted_without_credential] = lambda: False
        app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.include_router(library_routes.router, prefix = "/api/library")
    return TestClient(app)


@pytest.fixture
def client(monkeypatch):
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

    uploads = [
        (("note.md", b"# hi", "text/markdown"), "note.md", "text/markdown"),
        (("C:\\Users\\me\\notes.txt", b"<script>", "text/html"), "notes.txt", "text/plain"),
        (("report.pdf", b"%PDF-1.7", "text/html"), "report.pdf", "application/pdf"),
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
    assert item["openedAt"] is None
    assert _post(client, "items/opened", id = image).status_code == 200
    item = _items(client)[0][image]
    assert item["createdAt"] <= item["openedAt"] and item["favorite"] is True
    for missing in ("upload:gone", "sandbox:t-lib:gone.txt"):
        assert _post(client, "items/opened", id = missing).status_code == 404
    assert "upload:gone" not in library_db.list_entries()
    # The type follows the file's own name, so a binary renamed to .txt never opens as text.
    assert item["fileName"] == "photo.png"
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
    assert items["attachment:m%3A1:voice"]["textOnly"] is False
    deleted = []
    monkeypatch.setattr(studio_db, "delete_chat_attachment", lambda *ids: deleted.append(ids) or 1)
    assert _delete(client, "attachment:m%3A1:voice") == 200
    assert deleted == [("m:1", "voice")]


def test_deleting_a_chat_attachment_from_the_library_sweeps_its_original(client, monkeypatch):
    import storage.studio_db as studio_db
    from core import chat_originals

    sweeps = []
    monkeypatch.setattr(chat_originals, "sweep", lambda force = False: sweeps.append(force))
    monkeypatch.setattr(studio_db, "delete_chat_attachment", lambda *ids: True)
    assert _delete(client, "attachment:m:doc") == 200
    assert sweeps == [False]


def test_an_original_sent_many_times_counts_once_toward_disk_usage(client, monkeypatch):
    import storage.studio_db as studio_db

    def sent(
        attachment_id,
        sha256,
        has_original = True,
    ):
        return {
            "messageId": "m",
            "id": attachment_id,
            "type": "document",
            "contentType": "application/pdf",
            "sizeBytes": 1000,
            "originalSha256": sha256,
            "hasOriginal": has_original,
        }

    monkeypatch.setattr(library, "_SOURCES", (library._attachment_items,))
    monkeypatch.setattr(
        studio_db,
        "list_chat_attachments",
        lambda: [
            sent("a", "1" * 64),
            sent("b", "1" * 64),
            sent("c", "2" * 64),
            sent("d", "1" * 64),
        ],
    )
    items = _items(client)[0]
    usage = [items[f"attachment:m:{name}"].get("storageBytes", 1000) for name in "abcd"]
    assert usage == [1000, 0, 1000, 0]
    assert all(items[f"attachment:m:{name}"]["sizeBytes"] == 1000 for name in "abcd")


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


def test_a_sandbox_delete_takes_only_the_file_it_was_listed_as(client, signed_in, monkeypatch):
    monkeypatch.setattr(library, "_SOURCES", (library._sandbox_items,))
    _directory, path = _sandbox_chat("a.txt", b"listed")
    listed = _items(client)[0][_SANDBOX_A]["fingerprint"]
    # Held open so ext4 cannot hand the new file the same inode; Windows refuses to unlink an
    # open file and never reuses a file id, so nothing is held there.
    with open(path, "rb") if os.name != "nt" else contextlib.nullcontext():
        os.unlink(path)
        _sandbox_chat("a.txt", b"made since")
    response = _post(client, "items/delete", id = _SANDBOX_A, fingerprint = listed)
    assert response.status_code == 409
    assert Path(path).read_bytes() == b"made since"
    library.invalidate_listing()
    current = _items(client)[0][_SANDBOX_A]["fingerprint"]
    assert current != listed
    assert _post(client, "items/delete", id = _SANDBOX_A, fingerprint = current).status_code == 200
    assert not os.path.exists(path)


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


def test_a_clip_counts_its_recipe_toward_what_it_takes_on_disk(client, monkeypatch):
    from core.inference import audio_gallery

    monkeypatch.setattr(library, "_SOURCES", (library._audio_items, library._image_items))
    audio, image = _gallery_audio("A recipe beside it"), _gallery_image("No recipe")
    items = _items(client)[0]
    wav = audio_gallery.audio_path(audio)
    item = items[f"audio:{audio}"]
    assert item["sizeBytes"] == wav.stat().st_size
    assert item["storageBytes"] == wav.stat().st_size + wav.with_suffix(".json").stat().st_size
    assert "storageBytes" not in items[f"image:{image}"]


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
    assert (item["archived"], items[f"video:{shelved}"]["archived"]) == (False, True)
    assert _delete(client, f"audio:{audio}") == 200
    assert audio_gallery.audio_path(audio) is None

    forgotten = []
    monkeypatch.setattr(video_routes, "_forget_terminal_video", forgotten.append)
    monkeypatch.setattr(video_routes, "_forget_openai_job", lambda ref: False)
    response = _post(client, "items/delete", id = f"video:{video}")
    assert response.status_code == 500 and "video job" in response.json()["detail"]
    assert video_gallery.video_path(video) is not None
    monkeypatch.setattr(video_routes, "_forget_openai_job", lambda ref: forgotten.append(ref) or 1)
    assert _delete(client, f"video:{video}") == 200
    assert video_gallery.video_path(video) is None
    assert forgotten == [video, video]


def test_the_listing_reports_the_library_disk_and_the_sources_on_it(client, monkeypatch):
    from utils.paths.storage_roots import exports_root

    def sources():
        response = client.get("/api/library")
        assert response.status_code == 200
        disk = response.json()["disk"]
        assert 0 <= disk["freeBytes"] <= disk["totalBytes"] and disk["totalBytes"] > 0
        return set(disk["sources"])

    every = {"upload", "attachment", "image", "video", "audio", "sandbox"}
    every |= {"model:training", "model:exported"}
    assert sources() == every
    real, exports, uploads = library._device, str(exports_root()), str(library.uploads_dir())
    monkeypatch.setattr(library, "_device", lambda path: -1 if str(path) == exports else real(path))
    assert sources() == every - {"model:exported"}
    monkeypatch.setitem(library._SOURCE_ROOTS, "sandbox", _fail)
    assert sources() == every - {"model:exported", "sandbox"}
    monkeypatch.setattr(library, "_device", lambda path: 1 if str(path) == uploads else -1)
    assert sources() == {"upload"}


def test_favorites_list_only_stars_the_source_still_has(client, monkeypatch):
    from core.inference import image_gallery

    [starred, plain] = _upload(client, ("a.txt", b"a", "text/plain"), ("b.txt", b"b", "text/plain"))
    image = f"image:{_gallery_image('A star')}"
    for item_id, favorite in ((starred, True), (plain, False), (image, True), ("image:gone", True)):
        _patch(client, id = item_id, favorite = favorite)
    reads = []
    real = image_gallery._read_meta
    monkeypatch.setattr(image_gallery, "_read_meta", lambda path: reads.append(path) or real(path))
    assert set(_favorites(client)) == {starred, image}
    assert reads == []
    image_gallery.image_path(_ref(image)).unlink()
    assert _favorites(client) == [starred]
    assert (_delete(client, starred), _delete(client, "image:gone")) == (200, 404)
    assert set(library_db.list_entries()) == {plain, image}
    assert _delete(client, "elsewhere:x") == 400


def test_leftovers_of_a_crash_are_swept_from_the_uploads_folder(client):
    import time

    [kept, interrupted] = _upload(
        client, ("k.txt", b"k", "text/plain"), ("i.txt", b"i", "text/plain")
    )
    directory = library.uploads_dir()
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

    from utils.paths.storage_roots import exports_root, outputs_root

    monkeypatch.setattr(library, "_SOURCES", (library._model_items,))
    outputs_root().mkdir(parents = True, exist_ok = True)
    stamp = library._model_stamp()
    run, gguf = outputs_root() / "library-test-run", exports_root() / "library-test-gguf"
    run.mkdir()
    (run / "adapter_config.json").write_text('{"base_model_name_or_path": "unsloth/base"}')
    (run / "adapter_model.safetensors").write_bytes(b"x" * 10)
    gguf.mkdir(parents = True)
    (gguf / "model.Q4_K_M.gguf").write_bytes(b"x" * 10)
    (gguf / "model.Q8_0.gguf").write_bytes(b"x" * 20)
    try:
        assert library._model_stamp() != stamp
        items = {item["name"]: item for item in _items(client)[0].values()}
        item, export = items[run.name], items[gguf.name]
        assert item["contentType"] == library.MODEL_CONTENT_TYPE
        assert (item["model"]["origin"], item["model"]["exportType"]) == ("training", "lora")
        assert item["sizeBytes"] >= 10 and export["sizeBytes"] == 30
        for starred in (item, export):
            _patch(client, id = starred["id"], favorite = True)
        assert _items(client)[0][export["id"]]["favorite"] is True
        assert set(_favorites(client)) == {item["id"], export["id"]}
        assert _delete(client, item["id"]) == 400
        assert run.is_dir()
        shutil.rmtree(run)
        assert _delete(client, item["id"]) == 200
        assert _favorites(client) == [export["id"]]
    finally:
        shutil.rmtree(run, ignore_errors = True)
        shutil.rmtree(gguf, ignore_errors = True)


def test_training_runs_map_only_folders_directly_under_outputs(monkeypatch):
    """A newer run in an external folder of the same name must not claim the managed model."""
    import sqlite3

    from storage import studio_db
    from utils.paths.storage_roots import outputs_root

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE training_runs (id TEXT, output_dir TEXT, started_at TEXT)")
    conn.executemany(
        "INSERT INTO training_runs VALUES (?, ?, ?)",
        [
            ("managed", str(outputs_root() / "my-run"), "2026-01-01"),
            ("external", "/tmp/elsewhere/my-run", "2026-02-01"),
            ("nested", str(outputs_root() / "group" / "other"), "2026-03-01"),
        ],
    )
    monkeypatch.setattr(studio_db, "get_connection", lambda: conn)
    assert library._training_runs_by_dir() == {"my-run": "managed"}


def test_an_api_key_lists_fine_tunes_by_reference_and_can_still_act_on_them(client, monkeypatch):
    import shutil

    from utils.paths.storage_roots import outputs_root

    monkeypatch.setattr(library, "_SOURCES", (library._model_items,))
    outputs_root().mkdir(parents = True, exist_ok = True)
    run = outputs_root() / "library-key-run"
    run.mkdir()
    local_base = str(outputs_root() / "my-local-base")
    (run / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": local_base}))
    (run / "adapter_model.safetensors").write_bytes(b"x" * 10)
    try:
        library.invalidate_listing()
        client.app.dependency_overrides[library_routes.authenticated_via_api_key] = lambda: True
        body = client.get("/api/library").text
        assert str(run) not in body and local_base not in body
        [item] = [i for i in json.loads(body)["items"] if i["name"] == run.name]
        _patch(client, id = item["id"], favorite = True)
        assert str(run) not in json.dumps(_favorites(client))
        client.app.dependency_overrides[library_routes.authenticated_via_api_key] = lambda: False
        assert f"model:training:{run}" in _favorites(client)
    finally:
        shutil.rmtree(run)


def test_a_file_part_kept_as_a_data_url_decodes():
    import base64

    from routes.chat_history import _decode_attachment_base64

    clip = base64.b64encode(_CLIP).decode()
    assert _decode_attachment_base64(f"data:video/mp4;base64,{clip}") == _CLIP
    assert _decode_attachment_base64(clip) == _CLIP


def test_gallery_iso_dates_and_suffixless_chat_media_names():
    assert library._to_ms("2026-09-25T10:00:00Z") == 1790330400000
    assert library._to_ms(1790330400) == 1790330400000
    assert library._to_ms("not a date") == 0
    assert library._named_for_type("Chat image", "image/png") == "Chat image.png"
    assert library._named_for_type("Chat audio", "audio/wav") == "Chat audio.wav"
    assert library._named_for_type("photo.jpg", "image/jpeg") == "photo.jpg"
    assert library._named_for_type("notes", "application/pdf") == "notes"


def test_an_empty_parent_or_folder_id_is_refused(client):
    assert _post(client, "folders", name = "x", parentId = "").status_code == 422
    assert (
        client.patch("/api/library/items", json = {"id": "upload:x", "folderId": ""}).status_code
        == 422
    )


def test_an_upload_file_is_revalidated_after_a_note_save(client):
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    response = client.get(_items(client)[0][note]["fileUrl"])
    assert "no-cache" in response.headers["cache-control"]


def test_the_listing_memo_is_bounded():
    assert library._LISTING.size > 0


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
        os.replace(path, tmp_path / f"old-{len(os.listdir(tmp_path))}")
        Path(path).write_bytes(b"new")
        library.invalidate_listing()

    library_db.update_entry(_SANDBOX_ID, favorite = True)
    assert library_db.list_entries()[_SANDBOX_ID]["fingerprint"] is None
    item = _items(client)[0][_SANDBOX_ID]
    assert item["favorite"] is True and "_fingerprint" not in item
    assert library_db.list_entries()[_SANDBOX_ID]["fingerprint"] == library.fingerprint(_SANDBOX_ID)

    _patch(client, id = _SANDBOX_ID, name = "Q3")
    with open(path, "ab") as handle:
        handle.write(b" and v2")
    library.invalidate_listing()
    item = _items(client)[0][_SANDBOX_ID]
    assert (item["favorite"], item["name"]) == (True, "Q3")
    assert _favorites(client) == [_SANDBOX_ID]

    make_again()
    assert _favorites(client) == []
    _patch(client, id = _SANDBOX_ID, folderId = None)
    entry = library_db.list_entries()[_SANDBOX_ID]
    assert (entry["name"], entry["favorite"]) == (None, False)

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
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory = True)
    monkeypatch.setattr(library_db, "studio_db_path", lambda: tmp_path / "link" / "studio.db")
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
        ("touch_upload", lambda: library.write_upload_text(_ref(note), "new")),
        ("delete_upload", lambda: library.delete_item(note)),
        ("insert_upload", lambda: library.save_upload("a.txt", "text/plain", [b"hi"])),
    ):
        with monkeypatch.context() as patched:
            patched.setattr(library_db, name, _fail)
            with pytest.raises(sqlite3.OperationalError):
                call()
    assert library.upload_path(_ref(note)).read_bytes() == b"old"
    assert [path.name for path in library.uploads_dir().iterdir()] == [_ref(note)]
    assert note in _items(client)[0]


def test_a_note_save_keeps_the_old_file_on_disk_not_in_memory(client, monkeypatch):
    from pathlib import Path

    [note] = _upload(client, ("n.md", b"old", "text/markdown"))
    path = library.upload_path(_ref(note))
    read = Path.read_bytes
    with monkeypatch.context() as patched:
        patched.setattr(
            Path, "read_bytes", lambda self: pytest.fail("read") if self == path else read(self)
        )
        patched.setattr(library_db, "touch_upload", _fail)
        with pytest.raises(sqlite3.OperationalError):
            library.write_upload_text(_ref(note), "new")
    assert path.read_bytes() == b"old"
    assert library.write_upload_text(_ref(note), "new")
    assert path.read_bytes() == b"new"
    assert [p.name for p in library.uploads_dir().iterdir()] == [_ref(note)]


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
    response = _send(
        client, [("kept.md", b"# hi", "text/markdown")], nativePathLeases = [lease, forged]
    )
    assert response.status_code == 400
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
    folder = _folder(client, "Work", None)
    with monkeypatch.context() as patched:
        patched.setattr(library_db, "update_entry", gone)
        plan = ("plan.md", b"# plan", "text/markdown")
        response = _send(client, [plan], nativePathLeases = leases, folderId = folder)
        assert response.status_code == 404
    assert _items(client)[0] == {}
    response = _send(client, [], nativePathLeases = leases)
    assert response.status_code == 200 and len(response.json()["ids"]) == 2
    assert _send(client, [], nativePathLeases = leases[:1]).status_code == 400


def test_an_oversized_desktop_drop_is_refused_before_copying(client, monkeypatch, tmp_path):
    big = _file(tmp_path / "big.bin", "x" * 10)
    monkeypatch.setattr(library_routes, "_MAX_UPLOAD_BYTES", 4)
    opened = lambda lease: ("big.bin", "application/octet-stream", open(big, "rb"))  # noqa: E731
    monkeypatch.setattr(library, "open_native_upload", opened)

    def copied(*_args):
        raise AssertionError("copied an oversized drop")

    monkeypatch.setattr(library, "save_upload", copied)
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
    assert add(note, "p1").json() == {"already": True}
    assert add(note, "missing").status_code == 404
    assert add("upload:0123456789abcdef0123456789abcdef", "p1").status_code == 404
    assert add("attachment:m:a", "p1").status_code == 400
    assert add("model:training:/tmp/run", "p1").status_code == 400
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
        pytest.param("\U0001f9a5" * 80 + ".txt", "\U0001f9a5" * 50 + ".txt", id = "sloths"),
    ],
)
def test_names_written_to_disk_are_valid_on_windows(name, safe):
    from core.inference.gallery_projects import _bad_name, _tmp_name

    assert library.safe_file_name(name) == safe
    project_name = library.safe_file_name(name, item_id = "upload:x")
    assert not _bad_name(project_name), project_name
    assert len(_tmp_name(project_name).encode()) <= 255


def test_items_download_as_attachments_under_the_name_they_were_given(client, monkeypatch):
    [upload] = _upload(client, ("re:port*q3?.html", b"<script>1</script>", "text/html"))
    assert _download(client, upload).status_code == 401
    monkeypatch.setattr(library_routes, "subject_for_header_or_query_token", _signed_in_subject)
    response = _download(client, upload)
    assert (response.status_code, response.content) == (200, b"<script>1</script>")
    assert response.headers["content-type"] == "application/octet-stream"
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
    image = _gallery_image("Café ☕ at noon")
    assert f"filename*=UTF-8''{quote('Café ☕ at noon.png')}" in disposition(f"image:{image}")
    # No prompt: the id, never an empty name.
    image = _gallery_image("  ")
    assert f'filename="{image}.png"' in disposition(f"image:{image}")
    assert 'filename="Hello there.wav"' in disposition(f"audio:{_gallery_audio('Hello there')}")

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
        for name in ("c.txt", ".env", "d1/d2/d3/d4/deep.txt", "../x", ".hidden"):
            assert status(name) == 404, name
        assert _download(client, "sandbox:nope:a.txt").status_code == 404
    assert set(_items(client)[0]) == {"sandbox:t-lib:a.txt", "sandbox:t-lib:b.txt"}
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


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason = "no FIFOs on this OS")
def test_a_file_swapped_for_a_fifo_is_refused_without_waiting_for_a_writer(tmp_path):
    import threading

    fifo = tmp_path / "out.png"
    os.mkfifo(fifo)
    outcome = []

    def open_it():
        try:
            library._open_regular(str(fifo)).close()
            outcome.append("opened")
        except LookupError:
            outcome.append("refused")

    worker = threading.Thread(target = open_it, daemon = True)
    worker.start()
    worker.join(5)
    if worker.is_alive():
        # Unblock the stuck open so the thread ends, then fail.
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
    assert outcome == ["refused"]


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

    own = tmp_path / "My code"
    conn = studio_db.get_connection()
    conn.execute("UPDATE chat_projects SET root_path = ? WHERE id = 'p-lib'", (str(own),))
    conn.commit()
    conn.close()
    assert not library._studio_project_root(str(own))
    assert "sandbox:project-p-lib:files/notes.txt" not in _items(client)[0]


@pytest.fixture(autouse = True)
def _temp_folders_are_ordinary(monkeypatch):
    # macOS keeps pytest's temp folders under /private/var, which the real checks refuse, and they
    # are temporary folders, which Library moves refuse too.
    monkeypatch.setattr(
        _scan_folders,
        "is_denied_system_path",
        lambda path: path.startswith(("/etc", "/private/etc", "/usr")),
    )
    monkeypatch.setattr(
        library,
        "_scratch_and_system_folders",
        lambda: [f for f in _REAL_SCRATCH() if f.startswith(("/usr", "/opt", "/Applications"))],
    )


def _move(client, key, path):
    return client.post("/api/library/locations/move", json = {"key": key, "path": path})


def _location(client, key = None):
    locations = {e["key"]: e for e in client.get("/api/library/locations").json()["locations"]}
    return locations[key] if key else [entry["path"] for entry in locations.values()]


def _images():
    from core.inference import image_gallery
    return image_gallery.gallery_dir()


def _videos():
    from core.inference import video_gallery
    return video_gallery.gallery_dir()


def _fill(folder):
    for i in range(3):
        (folder / f"{i}.png").write_bytes(b"png%d" % i)


def _files(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def _cross_device(_src, _dst):
    raise OSError(errno.EXDEV, "Invalid cross-device link")


def _save_during_move(monkeypatch, save):
    """Calls `save(new_folder)` right after the move records the new folder."""
    from utils.paths import relocations

    real = relocations.set_chosen

    def record(
        key,
        path,
        moving_from = None,
    ):
        real(key, path, moving_from)
        if moving_from is not None:
            save(Path(path))

    monkeypatch.setattr(relocations, "set_chosen", record)


def _mounted(monkeypatch, mounts):
    real = os.path.ismount
    monkeypatch.setattr(os.path, "ismount", lambda path: str(path) in mounts or real(path))


def test_images_move_to_a_new_folder_and_back(client, tmp_path):
    old = _images()
    (old / "a.png").write_bytes(b"png")
    (old / "a.json").write_text("{}")
    new = tmp_path / "Pictures" / "Unsloth"
    new.parent.mkdir()

    response = _move(client, "images", str(new))
    assert response.status_code == 200, response.text
    assert _files(new) == ["a.json", "a.png"] and _files(old) == []
    assert _images() == new.resolve()
    location = _location(client, "images")
    assert [location[key] for key in ("path", "movable", "custom", "available")] == [
        str(new.resolve()),
        True,
        True,
        True,
    ]
    assert location["disk"]["freeBytes"] <= location["disk"]["totalBytes"]
    fine_tunes = _location(client, "fineTunes")
    assert location["device"] == fine_tunes["device"] and fine_tunes["movable"] is False

    assert _move(client, "images", None).status_code == 200
    assert _files(old) == ["a.json", "a.png"] and _images() == old
    assert _location(client, "images")["custom"] is False

    (new / "holiday.jpg").write_bytes(b"jpg")
    assert _move(client, "images", str(new)).status_code == 200
    assert _images() == (new / "Unsloth Images").resolve()
    assert _files(new) == ["Unsloth Images/a.json", "Unsloth Images/a.png", "holiday.jpg"]
    (old / "stray.txt").write_text("left behind")
    for _ in range(2):
        response = _move(client, "images", None)
        assert response.status_code == 200, response.text
    assert _images() == (old / "Unsloth Images").resolve()
    assert _files(old) == ["Unsloth Images/a.json", "Unsloth Images/a.png", "stray.txt"]
    assert _location(client, "images")["custom"] is True


def _made(folder, file = None):
    folder.mkdir(parents = True, exist_ok = True)
    if file:
        (folder / file).write_text("mine")
    return folder


def _named_folder(tmp_path, link_to = None):
    """A folder holding a file of the user's, whose "Unsloth Images" links to `link_to`, or with
    none, is the videos' folder."""
    data = _made(tmp_path / "data", "notes.txt")
    if link_to is None:
        library.move_location("videos", str(data / "Unsloth Images"))
    else:
        (data / "Unsloth Images").symlink_to(link_to, target_is_directory = True)
    return data


def _sandbox():
    from core.inference.tools import sandbox_root
    return Path(sandbox_root())


def _read_only(tmp_path):
    if getattr(os, "geteuid", lambda: 0)() == 0:
        pytest.skip("root writes anywhere, and Windows keeps no write bit on folders")
    (tmp_path / "locked").mkdir(mode = 0o500)
    return tmp_path / "locked" / "images"


def _studio(sub = ""):
    from utils.paths import studio_root
    return studio_root() / sub


@pytest.mark.parametrize(
    "key, target, detail",
    [
        ("images", lambda _: "relative/folder", "absolute"),
        (
            "images",
            lambda _: os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "unsloth-images")
            if os.name == "nt"
            else "/etc/unsloth-images",
            "System",
        ),
        ("images", lambda tmp: tmp / "missing" / "deeper", "parent folder"),
        ("images", _read_only, "cannot write"),
        (
            "images",
            lambda tmp: _made(tmp / "full" / "Unsloth Images", "keep").parent,
            "holds files",
        ),
        ("images", lambda _: _videos() / "images", "inside another Unsloth folder"),
        ("uploads", lambda _: _made(_sandbox() / "chat-1") / "uploads", "inside another Unsloth"),
        ("images", lambda _: _sandbox(), "inside another Unsloth folder"),
        ("images", lambda _: _images() / "sub", "inside another Unsloth folder"),
        ("images", lambda _: _images() / "new", "inside another Unsloth folder"),
        ("images", lambda _: _studio(), "Unsloth's own folder"),
        ("images", lambda _: _studio("elsewhere"), "Unsloth's own folder"),
        ("images", _named_folder, "inside another Unsloth folder"),
        ("images", lambda tmp: _named_folder(tmp, _videos()), "inside another Unsloth folder"),
        ("images", lambda tmp: _named_folder(tmp, _made(tmp / "home" / ".ssh")), "credential"),
        ("fineTunes", lambda tmp: tmp / "fine-tunes", "stay where they are"),
        ("exports", lambda tmp: tmp / "exports", "stay where they are"),
        ("somewhere", lambda tmp: tmp / "somewhere", "stay where they are"),
    ],
)
def test_a_folder_that_cannot_take_the_files_is_refused(client, tmp_path, key, target, detail):
    def tree():
        return {
            str(path)
            for root in (tmp_path, _studio())
            for path in root.rglob("*")
            if ".db" not in path.name
        }

    _fill(_images())
    (_images() / "sub").mkdir()
    target, folders = str(target(tmp_path)), _location(client)
    before = tree()
    response = _move(client, key, target)
    assert response.status_code == 400
    assert detail in response.json()["detail"]
    assert (tree(), _location(client)) == (before, folders)


def test_a_flag_set_during_a_move_keeps_the_ones_set_before(client, tmp_path, monkeypatch):
    from core.inference import gallery_flags, image_gallery

    kept = _gallery_image("archived long ago")
    image_gallery.set_flags(kept, archived = True)
    fresh = _gallery_image("archived during the move")

    # The images first and the store last, the order that loses it (directory order is the OS's).
    def store_last(
        source,
        target,
        log,
        only = None,
    ):
        for entry in sorted(source.iterdir(), key = lambda p: p.name.startswith(".")):
            if (only is None or only(entry)) and not library._inside(target, entry):
                if entry.name == ".flags.json":
                    assert image_gallery.set_flags(fresh, archived = True) is not None
                library._move_entry(entry, target / entry.name, log)

    monkeypatch.setattr(library, "_move_entries", store_last)
    assert _move(client, "images", str(tmp_path / "Pics")).status_code == 200
    flags = gallery_flags.read(_images())
    assert gallery_flags.is_archived(flags, kept) and gallery_flags.is_archived(flags, fresh)
    assert not list(_images().glob(".flags (*"))
    image_gallery.clear()
    assert {p.stem for p in _images().glob("*.png")} == {kept, fresh}


def test_only_the_installation_owner_can_move(client, tmp_path, monkeypatch):
    monkeypatch.setattr(library_routes.account_access, "managed_account", lambda: True)
    assert _move(client, "images", str(tmp_path / "elsewhere")).status_code == 403
    assert not (tmp_path / "elsewhere").exists()


@pytest.mark.parametrize("mount_point", [False, True])
def test_an_unplugged_folder_is_not_made_again_and_can_be_reset(
    client, tmp_path, monkeypatch, mount_point
):
    import shutil

    from utils.paths.relocations import LocationUnavailable

    (_images() / "a.png").write_bytes(b"png")
    drive = tmp_path / "Drive"
    drive.mkdir()
    mounts = {str(drive.resolve())} if mount_point else set()
    _mounted(monkeypatch, mounts)
    assert _move(client, "images", str(drive)).status_code == 200
    assert _move(client, "uploads", str(tmp_path / "Uploads")).status_code == 200
    folder = (drive / "Unsloth Images" if mount_point else drive).resolve()
    assert _images() == folder

    if mount_point:
        mounts.clear()
    else:
        shutil.rmtree(drive)
    shutil.rmtree(tmp_path / "Uploads")
    with pytest.raises(LocationUnavailable):
        _images()
    assert drive.exists() is mount_point
    logged = []
    for level in ("debug", "info", "warning"):
        monkeypatch.setattr(
            library.logger, level, lambda message, *a, **_: logged.append(message % a)
        )
    for _ in range(3):
        assert client.get("/api/library").json()["disk"]["totalBytes"] > 0
    assert len(logged) == len(set(logged)) == 2
    assert all("location_unavailable" in line for line in logged)
    location = _location(client, "images")
    assert (location["path"], location["available"], location["disk"]) == (str(folder), False, None)
    assert _post(client, "locations/reveal", key = "images").status_code == 409
    assert drive.exists() is mount_point
    assert _move(client, "images", str(tmp_path / "elsewhere")).status_code == 400
    response = _move(client, "images", None)
    assert (response.status_code, response.json()["leftBehind"]) == (200, str(folder))
    assert _location(client, "images")["custom"] is False and _images().is_dir()


@pytest.mark.parametrize(
    "failure, detail",
    [
        ("full", "No space left"),
        ("in use", "open.png is in use by another program. Close it and try again."),
    ],
)
def test_a_failed_move_puts_everything_back(client, tmp_path, monkeypatch, failure, detail):
    import shutil

    old = _images()
    _fill(old)
    (old / "d").mkdir()
    (old / "d" / "open.png").write_bytes(b"open")
    real_copy, real_unlink = shutil.copy2, library._unlink
    copies = []

    def filling_copy(src, dst, **kwargs):
        copies.append(src)
        if len(copies) == 2:
            Path(dst).write_bytes(b"par")
            raise OSError(28, "No space left on device")
        return real_copy(src, dst, **kwargs)

    def locked(path):
        if Path(path).name == "open.png":
            exc = PermissionError(errno.EACCES, "The process cannot access the file")
            exc.winerror = 32
            raise exc
        real_unlink(path)

    monkeypatch.setattr(library, "_rename", _cross_device)
    if failure == "full":
        monkeypatch.setattr(library.shutil, "copy2", filling_copy)
    else:
        monkeypatch.setattr(library, "_unlink", locked)
    _save_during_move(monkeypatch, lambda new: (new / "saved-meanwhile.png").write_bytes(b"new"))
    target = tmp_path / "other-drive"
    response = _move(client, "images", str(target))
    assert response.status_code == 500 and detail in response.json()["detail"]
    assert _files(old) == ["0.png", "1.png", "2.png", "d/open.png", "saved-meanwhile.png"]
    assert (old / "1.png").read_bytes() == b"png1" and _files(target) == []
    assert _images() == old and _location(client, "images")["custom"] is False


@pytest.mark.parametrize("spelling", ["link", "case"])
def test_another_spelling_of_a_folder_is_that_folder(client, tmp_path, monkeypatch, spelling):
    images, videos = _images(), _videos()
    _fill(images)
    (videos / "v.mp4").write_bytes(b"v")
    if spelling == "case" and not (images.parent / images.name.upper()).is_dir():
        pytest.skip("this disk tells case apart")
    if spelling == "link":
        # resolve() keeps the spelling it is given: a case alias on a case-insensitive disk, or a
        # bind mount, names a folder by a path that is not its own. A link that resolve() is kept
        # from following stands in for either on any disk.
        monkeypatch.setattr(library, "_move_target", lambda raw: Path(raw))

    def alias(folder):
        if spelling == "case":
            return folder.parent / folder.name.upper()
        (tmp_path / folder.name).symlink_to(folder, target_is_directory = True)
        return tmp_path / folder.name

    # The current folder under another name: nothing moves (a move onto itself would empty it).
    assert _move(client, "images", str(alias(images))).status_code == 200
    assert _move(client, "images", str(alias(videos))).status_code == 400
    assert _files(images) == ["0.png", "1.png", "2.png"] and _files(videos) == ["v.mp4"]
    assert _images() == images


def test_a_move_never_moves_a_folder_into_itself(tmp_path):
    _fill(tmp_path)
    (tmp_path / "Unsloth Images").mkdir()
    library._move_entries(tmp_path, tmp_path / "Unsloth Images", library._MoveLog())
    assert _files(tmp_path) == [f"Unsloth Images/{i}.png" for i in range(3)]


def test_a_move_across_drives_merges_skips_and_waits(client, tmp_path, monkeypatch):
    import threading
    import time

    old = _videos()
    (old / ".jobs").mkdir()
    (old / ".jobs" / "old.json").write_text("old job")
    (old / "same.txt").write_text("same")
    (old / "clash.txt").write_text("from the old folder")
    (old / "v.mp4").write_text("v")
    (old / "gone.mp4").write_text("gone")
    (old / ".late.mp4.tmp").write_text("late")

    def save(new):
        (new / ".jobs").mkdir()
        (new / ".jobs" / "new.json").write_text("running job")
        (new / "same.txt").write_text("same")
        (new / "clash.txt").write_text("from the new folder")

    def rename(src, dst):
        if Path(src).name == "gone.mp4":
            Path(src).unlink()
            raise FileNotFoundError(2, "No such file or directory")
        _cross_device(src, dst)

    def finish():
        time.sleep(0.5)
        os.replace(old / ".late.mp4.tmp", old / "late.mp4")

    monkeypatch.setattr(library, "_rename", rename)
    _save_during_move(monkeypatch, save)
    writer = threading.Thread(target = finish)
    writer.start()
    new = tmp_path / "videos"
    assert _move(client, "videos", str(new)).status_code == 200
    writer.join()
    assert {name: (new / name).read_text() for name in _files(new)} == {
        ".jobs/new.json": "running job",
        ".jobs/old.json": "old job",
        "same.txt": "same",
        "clash.txt": "from the new folder",
        "clash (2).txt": "from the old folder",
        "v.mp4": "v",
        "late.mp4": "late",
    }
    assert _files(old) == [] and _videos() == new.resolve()


def test_an_upload_finishing_during_a_move_lands_in_the_new_folder(client, tmp_path):
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    new = tmp_path / "uploads"

    def chunks():
        yield b"first half, "
        library.move_location("uploads", str(new))
        yield b"second half"

    slow = f"upload:{library.save_upload('slow.txt', 'text/plain', chunks())['id']}"
    items = _items(client)[0]
    served = [client.get(items[item_id]["fileUrl"]).content for item_id in (slow, note)]
    assert served == [b"first half, second half", b"# plan"]
    assert _files(new) == sorted(_ref(item_id) for item_id in (slow, note))
    assert not [p for p in library._location_default("uploads").iterdir() if p.suffix == ".tmp"]


def _move_after_lookup(monkeypatch, new):
    """The first lookup of an upload's file starts a move of the uploads folder, and gives it a
    moment to finish, before answering with where the file was."""
    import threading

    real = library.upload_path
    movers = []

    def lookup(upload_id):
        path = real(upload_id)
        if not movers:
            movers.append(
                threading.Thread(target = library.move_location, args = ("uploads", str(new)))
            )
            movers[0].start()
            movers[0].join(0.5)
        return path

    monkeypatch.setattr(library, "upload_path", lookup)
    return movers


def test_an_upload_deleted_or_saved_as_a_move_finishes_is_not_left_behind(
    client, tmp_path, monkeypatch
):
    gone, kept = _upload(
        client, ("gone.txt", b"gone", "text/plain"), ("kept.txt", b"kept", "text/plain")
    )
    new = tmp_path / "uploads"
    movers = _move_after_lookup(monkeypatch, new)
    assert _delete(client, gone) == 200
    movers[0].join()
    assert _files(new) == [_ref(kept)]
    assert set(_items(client)[0]) == {kept}

    new = tmp_path / "uploads-again"
    movers = _move_after_lookup(monkeypatch, new)
    assert library.write_upload_text(kept.partition(":")[2], "edited", "utf-8")
    movers[0].join()
    assert (new / _ref(kept)).read_text() == "edited"


def test_a_drive_without_room_for_the_files_is_refused_up_front(client, tmp_path, monkeypatch):
    import shutil

    (_images() / "big.png").write_bytes(b"x" * 1000)
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
    assert response.status_code == 400 and "free" in response.json()["detail"]
    assert not target.exists()
    assert (_images() / "big.png").stat().st_size == 1000


def test_a_named_folder_linked_onto_a_full_drive_is_refused_up_front(client, tmp_path, monkeypatch):
    import shutil

    (_images() / "big.png").write_bytes(b"x" * 1000)
    picked = tmp_path / "Pictures"
    _made(picked, "holiday.jpg")
    other = tmp_path / "small-drive"
    _made(other / "images")
    (picked / "Unsloth Images").symlink_to(other / "images", target_is_directory = True)
    real_device = library._device
    monkeypatch.setattr(
        library,
        "_device",
        lambda path: -1 if str(path).startswith(str(other.resolve())) else real_device(path),
    )
    real_usage = shutil.disk_usage
    monkeypatch.setattr(
        library.shutil,
        "disk_usage",
        lambda path: shutil._ntuple_diskusage(10**6, 10**6, 10)
        if str(path).startswith(str(other.resolve()))
        else real_usage(path),
    )
    response = _move(client, "images", str(picked))
    assert response.status_code == 400 and "free" in response.json()["detail"]
    assert _files(other) == [] and (_images() / "big.png").stat().st_size == 1000


def test_temporary_and_cache_folders_cannot_hold_library_files(monkeypatch):
    import tempfile

    monkeypatch.setattr(_scan_folders, "is_denied_system_path", _REAL_DENIED)
    monkeypatch.setattr(
        library, "_scratch_and_system_folders", lambda: [*_REAL_SCRATCH(), "/scratch-for-test"]
    )
    scratch = (
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Temp" / "images"
        if os.name == "nt"
        else Path("/var/tmp/images")
    )
    for folder in (Path(tempfile.gettempdir()) / "images", scratch):
        with pytest.raises(ValueError, match = "cannot hold these files"):
            library._refuse_denied(folder.resolve())
    assert not _REAL_DENIED("/scratch-for-test/images")
    with pytest.raises(ValueError, match = "Temporary"):
        library._refuse_denied(Path("/scratch-for-test/images"))
    library._refuse_denied(Path.home() / "Pictures" / "Unsloth")


def test_a_folder_chosen_before_mount_points_learns_its_mount_once_present(
    client, tmp_path, monkeypatch
):
    from storage.studio_db import get_app_setting, upsert_app_settings
    from utils.paths.relocations import LocationUnavailable, forget_cache

    drive = tmp_path / "mnt-usb"
    folder, mount = drive / "Unsloth Images", str(drive.resolve())
    mounts = set()
    _mounted(monkeypatch, mounts)
    upsert_app_settings({"library.locations": {"images": str(folder)}}, read_back = False)
    forget_cache()
    with pytest.raises(LocationUnavailable):
        _images()
    assert get_app_setting("library.locations", {})["images"] == str(folder)
    folder.mkdir(parents = True)
    mounts.add(mount)
    forget_cache()
    assert _images() == folder and _location(client, "images")["custom"] is True
    assert get_app_setting("library.locations", {})["images"] == {
        "path": str(folder),
        "mount": mount,
    }
    mounts.clear()
    forget_cache()
    with pytest.raises(LocationUnavailable):
        _images()


def test_uploads_left_on_a_reset_drive_are_hidden_until_their_files_are_back(client, tmp_path):
    [upload] = _upload(client, ("k.txt", b"k", "text/plain"))
    drive = tmp_path / "Drive"
    assert _move(client, "uploads", str(drive)).status_code == 200
    drive.rename(tmp_path / "unplugged")
    assert _move(client, "uploads", None).status_code == 200
    assert _items(client)[0] == {}
    (tmp_path / "unplugged" / _ref(upload)).rename(library.uploads_dir() / _ref(upload))
    assert set(_items(client)[0]) == {upload}


class _Crash(BaseException):
    """The process dying: nothing the move does on a failure runs."""


def _crash_moving(monkeypatch, key, path):
    """Starts moving `key`'s files to `path` and crashes after the first."""
    real = library._move_entry

    def crash_after_one(entry, dest, log):
        if log.moved:
            raise _Crash
        real(entry, dest, log)

    monkeypatch.setattr(library, "_move_entry", crash_after_one)
    with pytest.raises(_Crash):
        library.move_location(key, str(path))
    monkeypatch.setattr(library, "_move_entry", real)


def test_a_move_cut_short_is_finished_on_the_next_start(client, tmp_path, monkeypatch):
    from storage.studio_db import get_app_setting
    from utils.paths.relocations import forget_cache

    old, new = _images(), tmp_path / "Pictures"
    _fill(old)
    _crash_moving(monkeypatch, "images", new)
    assert len(_files(new)) == 1 and len(_files(old)) == 2
    forget_cache()
    assert _images() == new.resolve() and len(_files(new)) == 3
    assert "moving_from" not in str(get_app_setting("library.locations", {}))
    assert _location(client, "images")["custom"] is True


def test_a_move_cut_short_waits_for_the_drive_it_was_moving_onto(client, tmp_path, monkeypatch):
    import shutil

    from storage.studio_db import get_app_setting
    from utils.paths.relocations import forget_cache

    old, new = _images(), tmp_path / "Pictures"
    _fill(old)
    _crash_moving(monkeypatch, "images", new)
    shutil.move(new, tmp_path / "unplugged")
    forget_cache()
    assert _images() == old and len(_files(old)) == 2
    assert _location(client, "images")["available"] is True
    assert "moving_from" in str(get_app_setting("library.locations", {}))
    shutil.move(tmp_path / "unplugged", new)
    assert _images() == old and len(_files(old)) == 2
    forget_cache()
    assert _images() == new.resolve() and len(_files(new)) == 3
    assert "moving_from" not in str(get_app_setting("library.locations", {}))

    _crash_moving(monkeypatch, "images", tmp_path / "Other")
    shutil.move(tmp_path / "Other", tmp_path / "unplugged")
    forget_cache()
    response = _move(client, "images", str(new))
    assert response.status_code == 200, response.text
    assert response.json()["leftBehind"] == str((tmp_path / "Other").resolve())
    assert _images() == new.resolve()
    assert "moving_from" not in str(get_app_setting("library.locations", {}))


def test_a_copy_across_drives_cut_short_leaves_no_partial_file_for_the_next_start(
    client, tmp_path, monkeypatch
):
    import shutil

    from utils.paths.relocations import forget_cache

    old, new = _images(), tmp_path / "Pictures"
    _fill(old)
    monkeypatch.setattr(library, "_rename", _cross_device)
    real_copy, real_discard = shutil.copy2, library._discard

    def dying_copy(_src, dst, **_kwargs):
        Path(dst).write_bytes(b"pn")
        raise _Crash

    monkeypatch.setattr(library.shutil, "copy2", dying_copy)
    monkeypatch.setattr(library, "_discard", lambda _path: None)
    with pytest.raises(_Crash):
        library.move_location("images", str(new))
    assert [name for name in os.listdir(new) if not name.startswith(".")] == []
    monkeypatch.setattr(library.shutil, "copy2", real_copy)
    monkeypatch.setattr(library, "_discard", real_discard)
    forget_cache()
    assert _images() == new.resolve()
    assert _files(new) == ["0.png", "1.png", "2.png"]
    assert (new / "0.png").read_bytes() == b"png0"


def test_an_open_cannot_land_between_a_delete_and_its_row(client, monkeypatch):
    import threading

    [upload] = _upload(client, ("a.txt", b"a", "text/plain"))
    _patch(client, id = upload, favorite = True)
    real = library.item_exists

    def a_delete_meanwhile(item_id, *args):
        found = real(item_id, *args)
        deleting = threading.Thread(target = library.delete_item, args = (upload,))
        deleting.start()
        deleting.join(0.3)
        threads.append(deleting)
        return found

    threads = []
    monkeypatch.setattr(library, "item_exists", a_delete_meanwhile)
    assert _post(client, "items/opened", id = upload).status_code == 200
    threads[0].join()
    assert upload not in library_db.list_entries()


def test_a_new_move_finishes_one_cut_short_first_or_waits_for_its_drive(
    client, tmp_path, monkeypatch
):
    import shutil

    from utils.paths.relocations import forget_cache

    old, new, other = _images(), tmp_path / "Pictures", tmp_path / "Other"
    _fill(old)
    _crash_moving(monkeypatch, "images", new)
    shutil.move(new, tmp_path / "unplugged")
    forget_cache()
    response = _move(client, "images", str(other))
    assert response.status_code == 400 and str(new.resolve()) in response.json()["detail"]
    shutil.move(tmp_path / "unplugged", new)
    assert _move(client, "images", str(other)).status_code == 200
    assert _images() == other.resolve() and len(_files(other)) == 3
    assert _files(old) == [] and _files(new) == []


def test_a_move_cut_short_waits_for_the_drive_it_was_leaving(client, tmp_path, monkeypatch):
    import shutil

    from storage.studio_db import get_app_setting
    from utils.paths.relocations import forget_cache

    drive = (tmp_path / "Drive").resolve()
    drive.mkdir()
    mounts = {str(drive)}
    _mounted(monkeypatch, mounts)
    source, target = drive / "Pictures", tmp_path / "Pictures"
    assert _move(client, "images", str(source)).status_code == 200
    _fill(_images())
    _crash_moving(monkeypatch, "images", target)
    shutil.move(source, tmp_path / "unplugged")
    mounts.clear()
    forget_cache()
    assert _images() == target.resolve() and len(_files(target)) == 1
    assert "moving_from" in str(get_app_setting("library.locations", {}))
    shutil.move(tmp_path / "unplugged", source)
    mounts.add(str(drive))
    forget_cache()
    assert _images() == target.resolve() and len(_files(target)) == 3
    assert "moving_from" not in str(get_app_setting("library.locations", {}))


def test_a_folder_choice_read_before_a_move_saved_does_not_replace_it(
    client, tmp_path, monkeypatch
):
    import storage.studio_db as studio_db
    from utils.paths import relocations

    real = studio_db.get_app_setting
    newer = (tmp_path / "Pictures").resolve()
    newer.mkdir()
    reads = []

    def read_while_a_move_saves(*args, **kwargs):
        value = real(*args, **kwargs)
        reads.append(value)
        if len(reads) == 1:
            relocations.set_chosen("images", newer)
        return value

    relocations.forget_cache()
    monkeypatch.setattr(studio_db, "get_app_setting", read_while_a_move_saves)
    relocations.chosen("images")
    assert relocations.chosen("images") == newer


def test_a_folder_choice_that_cannot_be_read_fails_rather_than_saving_to_the_default(
    client, tmp_path, monkeypatch
):
    import storage.studio_db as studio_db
    from utils.paths.relocations import forget_cache

    new = tmp_path / "Pictures"
    assert _move(client, "images", str(new)).status_code == 200
    real = studio_db.get_app_setting

    def failing(message):
        def read(*_args, **_kwargs):
            raise sqlite3.OperationalError(message)

        return read

    monkeypatch.setattr(studio_db, "get_app_setting", failing("database is locked"))
    forget_cache()
    with pytest.raises(sqlite3.OperationalError):
        _images()
    monkeypatch.setattr(studio_db, "get_app_setting", failing("no such table: app_settings"))
    forget_cache()
    assert _images() == _studio("images")
    monkeypatch.setattr(studio_db, "get_app_setting", real)
    forget_cache()
    assert _images() == new.resolve()


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
    assert not library._same_folder(a, b) and not library._inside(b / "x", a)
    assert library._same_folder(a, tmp_path / "a") and library._inside(a / "x", a)


@pytest.fixture
def revealed(monkeypatch):
    import utils.paths.file_manager as file_manager
    import utils.paths.path_utils as path_utils

    calls = []
    monkeypatch.setattr(file_manager, "file_manager_kind", lambda: "files")
    monkeypatch.setattr(path_utils, "reveal_in_file_manager", lambda path: calls.append(str(path)))
    return calls


def test_reveal_opens_an_items_own_file_or_a_location_by_key(client, revealed):
    reveal = lambda item_id: _post(client, "items/reveal", id = item_id).status_code  # noqa: E731
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    assert reveal("attachment:m:a") == 400
    assert reveal("upload:0123456789abcdef0123456789abcdef") == 404
    assert reveal("model:training:/etc") == 404
    assert _post(client, "locations/reveal", key = "/etc").status_code == 404
    assert revealed == []
    assert reveal(note) == 200
    locations = client.get("/api/library/locations").json()["locations"]
    paths = {entry["key"]: entry["path"] for entry in locations}
    assert set(paths) == {"uploads", "images", "videos", "audio", "fineTunes", "exports"}
    assert _post(client, "locations/reveal", key = "images").status_code == 200
    assert revealed == [str(library.upload_path(_ref(note))), paths["images"]]


def test_locations_hide_host_paths_from_an_api_key(client):
    signed_in = client.get("/api/library/locations").json()["locations"]
    assert all(os.path.isabs(entry["path"]) for entry in signed_in)
    client.app.dependency_overrides[authenticated_via_api_key] = lambda: True
    keyed = client.get("/api/library/locations").json()["locations"]
    assert [entry["key"] for entry in keyed] == [entry["key"] for entry in signed_in]
    for entry, shown in zip(keyed, signed_in):
        assert entry["path"] != shown["path"]
        assert shown["path"] not in json.dumps(entry)


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

    def no_launcher(path):
        raise FileNotFoundError(2, "No such file or directory: 'xdg-open'")

    monkeypatch.setattr(file_manager, "file_manager_kind", lambda: "files")
    monkeypatch.setattr(path_utils, "reveal_in_file_manager", no_launcher)
    response = _post(client, "items/reveal", id = note)
    assert response.status_code == 503
    assert response.json()["detail"] == "No file manager is available on this machine"


def _os_named(name):
    # A copy of os for one module: setting the real os.name to "nt" breaks pathlib in pytest itself.
    import types

    fake = types.ModuleType("os")
    fake.__dict__.update(os.__dict__)
    fake.name = name
    return fake


@pytest.mark.parametrize(
    "platform, os_name, wsl, container, display, expected",
    [
        ("darwin", "posix", False, False, None, "finder"),
        ("win32", "nt", False, False, None, "explorer"),
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
    monkeypatch.setattr(file_manager, "os", _os_named(os_name))
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
        windows.setattr(path_utils, "os", _os_named("nt"))
        path_utils.reveal_in_file_manager(target)
    assert calls.pop() == f'explorer /select,"{target}"'
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
    assert _thumbnail(client, flv).status_code == 404


def test_an_attachment_is_read_for_its_thumbnail_only_within_the_decodes_count(client, monkeypatch):
    free = []

    def media(_ref):
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
    assert _thumbnail(client, image).headers["vary"] == "Authorization"
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
        f"{target}.{expires}." + "\u00e9" * len(signature),
        f"{target}.{int(expires) + 60}.{signature}",
        f"{library_routes._sign_stream_id(other).rsplit('.', 2)[0]}.{expires}.{signature}",
        "nonsense",
        "",
    ):
        assert _stream(clip, bad).status_code in (401, 422), bad
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
        assert _mint(as_bob, mine_id).status_code == 404
    response = _app(None).get(url)
    assert (response.status_code, response.content) == (200, b"alice")
    response = _stream(theirs_id, sign(alice, theirs_id))
    assert response.status_code == 404 and b"bob" not in response.content
    assert _stream(theirs_id, sign(alice, mine_id)).status_code == 401
    assert _stream(mine_id, sign(OWNER, mine_id)).status_code == 404
