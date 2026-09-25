# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3
import sys
from pathlib import Path

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


@pytest.fixture
def client(monkeypatch):
    # Only the Library's own uploads: the other sources read chat, gallery and sandbox stores this
    # test does not seed.
    monkeypatch.setattr(library, "_SOURCES", (library._upload_items,))
    library.invalidate_listing()
    library._thumbnail_cache.clear()
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.dependency_overrides[request_admitted_without_credential] = lambda: False
    app.include_router(library_routes.router, prefix = "/api/library")
    return TestClient(app)


def _upload(
    client,
    *files,
    folder_id = None,
):
    response = client.post(
        "/api/library/uploads",
        files = [("files", file) for file in files],
        data = {"folderId": folder_id} if folder_id else {},
    )
    assert response.status_code == 200, response.text
    return response.json()["ids"]


def _items(client):
    body = client.get("/api/library").json()
    return {item["id"]: item for item in body["items"]}, body["folders"]


def test_uploads_are_listed_as_uploaded_items(client):
    [note, image] = _upload(
        client,
        ("note.md", b"# hi", "text/markdown"),
        ("pic.png", b"\x89PNG", "image/png"),
    )
    items, folders = _items(client)
    assert folders == []
    assert items[note]["name"] == "note.md"
    assert items[note]["source"] == "uploaded"
    assert items[note]["sizeBytes"] == 4
    assert items[image]["contentType"] == "image/png"
    assert not items[note]["favorite"] and items[note]["folderId"] is None


def test_raster_uploads_render_inline_and_markup_downloads(client):
    [image, page] = _upload(
        client,
        ("pic.png", b"\x89PNG", "image/png"),
        ("page.html", b"<script>alert(1)</script>", "text/html"),
    )
    items, _ = _items(client)
    inline = client.get(items[image]["fileUrl"])
    assert inline.headers["content-type"] == "image/png"
    # Served on the app origin, so markup must never come back as something a browser renders.
    download = client.get(items[page]["fileUrl"])
    assert download.headers["content-type"] == "application/octet-stream"
    assert download.headers["x-content-type-options"] == "nosniff"
    assert "attachment" in download.headers["content-disposition"]


def test_rename_favorite_and_move_are_an_overlay(client):
    [note] = _upload(client, ("note.md", b"", "text/markdown"))
    folder = client.post("/api/library/folders", json = {"name": "Work"}).json()

    patch = {"id": note, "name": "Plans.md", "favorite": True, "folderId": folder["id"]}
    assert client.patch("/api/library/items", json = patch).status_code == 200
    item = _items(client)[0][note]
    assert (item["name"], item["favorite"], item["folderId"]) == ("Plans.md", True, folder["id"])

    # An explicit null moves it back out; leaving the key off leaves it where it is.
    client.patch("/api/library/items", json = {"id": note, "favorite": False})
    assert _items(client)[0][note]["folderId"] == folder["id"]
    client.patch("/api/library/items", json = {"id": note, "folderId": None})
    assert _items(client)[0][note]["folderId"] is None


def test_a_rename_leaves_the_file_name(client):
    """The type is read from the file's own name, so a binary renamed to .txt never opens as text."""
    [image] = _upload(client, ("photo.png", b"png", "image/png"))
    client.patch("/api/library/items", json = {"id": image, "name": "photo.txt"})
    item = _items(client)[0][image]
    assert (item["name"], item["fileName"]) == ("photo.txt", "photo.png")


def test_moving_into_a_missing_folder_is_refused(client):
    [note] = _upload(client, ("note.md", b"", "text/markdown"))
    response = client.patch("/api/library/items", json = {"id": note, "folderId": "nope"})
    assert response.status_code == 404


def test_a_folder_cannot_move_into_its_own_subtree(client):
    outer = client.post("/api/library/folders", json = {"name": "Outer"}).json()
    inner = client.post(
        "/api/library/folders", json = {"name": "Inner", "parentId": outer["id"]}
    ).json()
    response = client.patch(f"/api/library/folders/{outer['id']}", json = {"parentId": inner["id"]})
    assert response.status_code == 400
    assert response.json()["detail"] == "A folder cannot be moved into itself"
    response = client.patch(f"/api/library/folders/{outer['id']}", json = {"parentId": outer["id"]})
    assert response.status_code == 400


def test_deleting_a_folder_moves_its_contents_up(client):
    outer = client.post("/api/library/folders", json = {"name": "Outer"}).json()
    inner = client.post(
        "/api/library/folders", json = {"name": "Inner", "parentId": outer["id"]}
    ).json()
    child = client.post(
        "/api/library/folders", json = {"name": "Child", "parentId": inner["id"]}
    ).json()
    [note] = _upload(client, ("note.md", b"", "text/markdown"), folder_id = inner["id"])

    assert client.delete(f"/api/library/folders/{inner['id']}").status_code == 200
    items, folders = _items(client)
    assert items[note]["folderId"] == outer["id"]
    assert {f["id"]: f["parentId"] for f in folders} == {
        outer["id"]: None,
        child["id"]: outer["id"],
    }


def test_a_utf16_note_is_saved_back_as_utf16(client):
    # Windows PowerShell 5 writes UTF-16LE with a BOM and CRLF endings.
    [note] = _upload(client, ("log.txt", "\ufeffold\r\n".encode("utf-16le"), "text/plain"))
    upload_id = note.removeprefix("upload:")
    response = client.put(
        f"/api/library/uploads/{upload_id}/text",
        json = {"text": "\ufeffnew\r\n", "encoding": "utf-16le"},
    )
    assert response.status_code == 200
    items, _ = _items(client)
    assert client.get(items[note]["fileUrl"]).content == b"\xff\xfen\x00e\x00w\x00\r\x00\n\x00"
    response = client.put(
        f"/api/library/uploads/{upload_id}/text", json = {"text": "x", "encoding": "latin-1"}
    )
    assert response.status_code == 422


def test_notes_are_editable_and_deletable(client):
    [note] = _upload(client, ("note.md", b"", "text/markdown"))
    upload_id = note.removeprefix("upload:")
    assert (
        client.put(f"/api/library/uploads/{upload_id}/text", json = {"text": "hello"}).status_code
        == 200
    )
    items, _ = _items(client)
    assert items[note]["sizeBytes"] == 5
    assert client.get(items[note]["fileUrl"]).content == b"hello"

    assert client.post("/api/library/items/delete", json = {"id": note}).status_code == 200
    assert note not in _items(client)[0]
    assert client.get(items[note]["fileUrl"]).status_code == 404
    assert client.post("/api/library/items/delete", json = {"id": note}).status_code == 404


def test_unknown_item_kinds_are_rejected(client):
    response = client.post("/api/library/items/delete", json = {"id": "elsewhere:x"})
    assert response.status_code == 400


def test_a_failing_source_does_not_empty_the_library(client, monkeypatch):
    [note] = _upload(client, ("note.md", b"", "text/markdown"))

    def broken():
        raise RuntimeError("gallery unreadable")

    monkeypatch.setattr(library, "_SOURCES", (library._upload_items, broken))
    assert note in _items(client)[0]


def test_generated_audio_is_listed_and_deleted(client, monkeypatch):
    from core.inference import audio_gallery

    monkeypatch.setattr(library, "_SOURCES", (library._audio_items,))
    record = audio_gallery.save(
        b"RIFF0000WAVE",
        {
            "prompt": "Hello from the Library",
            "model": "sample-tts",
            "audio_type": "snac",
            "sample_rate": 24000,
            "duration_s": 1.0,
            "created_at": 1_700_000_000,
        },
    )
    item_id = f"audio:{record['id']}"
    item = _items(client)[0][item_id]
    assert (item["name"], item["contentType"], item["source"]) == (
        "Hello from the Library.wav",
        "audio/wav",
        "generated",
    )
    assert item["createdAt"] == 1_700_000_000_000
    assert client.post("/api/library/items/delete", json = {"id": item_id}).status_code == 200
    assert audio_gallery.audio_path(record["id"]) is None


def test_generated_video_is_listed_and_deleted(client, monkeypatch):
    from core.inference import video_gallery

    monkeypatch.setattr(library, "_SOURCES", (library._video_items,))
    meta = {
        key: 1
        for key in (
            "width",
            "height",
            "num_frames",
            "fps",
            "duration_s",
            "steps",
            "guidance",
            "seed",
        )
    }
    record = video_gallery.save(
        b"\0\0\0\x18ftypmp42", {**meta, "prompt": "A calm sea", "created_at": 1_700_000_000}
    )
    import routes.video as video_routes

    forgotten = []
    monkeypatch.setattr(video_routes, "_forget_terminal_video", forgotten.append)
    monkeypatch.setattr(
        video_routes, "_forget_openai_job", lambda ref: forgotten.append(ref) or True
    )
    item_id = f"video:{record['id']}"
    item = _items(client)[0][item_id]
    assert (item["name"], item["contentType"]) == ("A calm sea.mp4", "video/mp4")
    assert client.post("/api/library/items/delete", json = {"id": item_id}).status_code == 200
    assert video_gallery.video_path(record["id"]) is None
    # Same cleanup as the Video page, so no ghost card comes back.
    assert forgotten == [record["id"], record["id"]]


def test_a_video_whose_job_cannot_be_dropped_stays_to_delete_again(client, monkeypatch):
    from core.inference import video_gallery
    import routes.video as video_routes

    keys = ("width", "height", "num_frames", "fps", "duration_s", "steps", "guidance", "seed")
    meta = {key: 1 for key in keys}
    record = video_gallery.save(
        b"\0\0\0\x18ftypmp42", {**meta, "prompt": "A calm sea", "created_at": 1_700_000_000}
    )
    item_id = f"video:{record['id']}"
    monkeypatch.setattr(video_routes, "_forget_terminal_video", lambda ref: None)
    monkeypatch.setattr(video_routes, "_forget_openai_job", lambda ref: False)
    response = client.post("/api/library/items/delete", json = {"id": item_id})
    assert response.status_code == 500
    assert "video job" in response.json()["detail"]
    assert video_gallery.video_path(record["id"]) is not None

    monkeypatch.setattr(video_routes, "_forget_openai_job", lambda ref: True)
    assert client.post("/api/library/items/delete", json = {"id": item_id}).status_code == 200
    assert video_gallery.video_path(record["id"]) is None


def test_favorites_lists_only_favorite_ids(client):
    [starred, plain] = _upload(client, ("a.txt", b"a", "text/plain"), ("b.txt", b"b", "text/plain"))
    client.patch("/api/library/items", json = {"id": starred, "favorite": True})
    client.patch("/api/library/items", json = {"id": plain, "favorite": False})
    # Deleted by its gallery, not the Library: the star stays in the overlay but is not listed.
    client.patch("/api/library/items", json = {"id": "image:gone", "favorite": True})
    assert client.get("/api/library/favorites").json() == {"ids": [starred]}


def test_deleting_an_item_drops_its_overlay_row(client):
    from storage import library_db

    [note] = _upload(client, ("n.txt", b"n", "text/plain"))
    client.patch("/api/library/items", json = {"id": note, "favorite": True})
    client.patch("/api/library/items", json = {"id": "image:gone", "favorite": True})
    assert client.post("/api/library/items/delete", json = {"id": note}).status_code == 200
    # Already gone from its source: the delete still clears what the Library kept about it.
    assert client.post("/api/library/items/delete", json = {"id": "image:gone"}).status_code == 404
    assert library_db.list_entries() == {}


def test_leftovers_of_a_crash_are_swept_from_the_uploads_folder(client):
    import os
    import time

    [kept, interrupted] = _upload(
        client, ("k.txt", b"k", "text/plain"), ("i.txt", b"i", "text/plain")
    )
    directory = library.uploads_dir()
    interrupted_id = interrupted.split(":", 1)[1]
    # A delete that stopped after setting its file aside: the row still lists it.
    os.replace(directory / interrupted_id, directory / f".{interrupted_id}.deleting")
    stale = [directory / ".0123.tmp", directory / f".{'a' * 32}.deleting"]
    fresh = directory / ".4567.tmp"
    for path in (*stale, fresh):
        path.write_bytes(b"x")
    old = time.time() - 2 * library._LEFTOVER_AGE_SECONDS
    for path in (*stale, directory / f".{interrupted_id}.deleting"):
        os.utime(path, (old, old))
    library._swept_at.clear()
    items = _items(client)[0]
    assert {kept, interrupted} <= set(items)
    names = {path.name for path in directory.iterdir()}
    assert names == {kept.split(":", 1)[1], interrupted_id, fresh.name}
    assert client.get(items[interrupted]["fileUrl"]).content == b"i"


def test_fine_tuned_models_are_listed_but_not_deleted_here(client, monkeypatch):
    from utils.paths.storage_roots import outputs_root

    monkeypatch.setattr(library, "_SOURCES", (library._model_items,))
    run = outputs_root() / "library-test-run"
    run.mkdir(parents = True)
    (run / "adapter_config.json").write_text('{"base_model_name_or_path": "unsloth/base"}')
    (run / "adapter_model.safetensors").write_bytes(b"x" * 10)
    try:
        items = [item for item in _items(client)[0].values() if item["name"] == run.name]
        [item] = items
        assert item["contentType"] == library.MODEL_CONTENT_TYPE
        assert item["model"]["origin"] == "training"
        assert item["model"]["exportType"] == "lora"
        assert item["sizeBytes"] >= 10
        response = client.post("/api/library/items/delete", json = {"id": item["id"]})
        assert response.status_code == 400
        assert run.is_dir()
    finally:
        import shutil
        shutil.rmtree(run)


# ── Desktop drops ────────────────────────────────────────────────


@pytest.fixture
def lease_secret(monkeypatch):
    import base64

    import utils.native_path_leases as leases

    from .test_rag_native_drop_upload import SECRET

    monkeypatch.setenv(
        leases.LEASE_SECRET_ENV,
        base64.urlsafe_b64encode(SECRET).decode("ascii").rstrip("="),
    )
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)
    yield
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)


def test_a_desktop_drop_is_read_from_its_signed_path(client, lease_secret, tmp_path):
    from .test_rag_native_drop_upload import _sign

    dropped = tmp_path / "Quarterly report.pdf"
    dropped.write_bytes(b"%PDF-1.7 report")
    folder = client.post("/api/library/folders", json = {"name": "Work"}).json()

    response = client.post(
        "/api/library/uploads",
        data = {"nativePathLeases": [_sign(dropped)], "folderId": folder["id"]},
    )
    assert response.status_code == 200, response.text
    [item_id] = response.json()["ids"]
    item = _items(client)[0][item_id]
    assert (item["name"], item["contentType"]) == ("Quarterly report.pdf", "application/pdf")
    assert item["folderId"] == folder["id"]
    assert client.get(item["fileUrl"]).content == b"%PDF-1.7 report"


def test_a_desktop_drop_needs_a_valid_grant(client, lease_secret, tmp_path):
    from .test_rag_native_drop_upload import _sign

    dropped = tmp_path / "notes.txt"
    dropped.write_text("hi")
    forged = _sign(dropped, secret = b"x" * 32)
    response = client.post("/api/library/uploads", data = {"nativePathLeases": [forged]})
    assert response.status_code == 400
    assert _items(client)[0] == {}


def test_a_failed_batch_keeps_none_of_its_files(client, lease_secret, tmp_path):
    from .test_rag_native_drop_upload import _sign

    dropped = tmp_path / "notes.txt"
    dropped.write_text("hi")
    response = client.post(
        "/api/library/uploads",
        files = [("files", ("kept.md", b"# hi", "text/markdown"))],
        data = {"nativePathLeases": [_sign(dropped, secret = b"x" * 32)]},
    )
    assert response.status_code == 400
    assert _items(client)[0] == {}
    assert [path for path in library.uploads_dir().iterdir()] == []


def test_an_empty_upload_is_refused(client):
    assert client.post("/api/library/uploads", data = {}).status_code == 400


def test_a_client_supplied_path_keeps_only_its_last_segment(client):
    [item_id] = _upload(client, ("C:\\Users\\me\\report.txt", b"x", "text/plain"))
    assert _items(client)[0][item_id]["name"] == "report.txt"


# ── Add to project ───────────────────────────────────────────────


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


def test_an_upload_is_copied_into_a_project_under_its_own_name(client, project):
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    body = {"id": note, "projectId": "p1"}
    first = client.post("/api/library/items/project", json = body)
    assert first.status_code == 200, first.text
    assert first.json() == {"already": False}
    [copied] = (project / "files").iterdir()
    assert copied.name.startswith("plan-") and copied.suffix == ".md"
    assert copied.read_bytes() == b"# plan"
    # Adding it again finds the copy instead of making another.
    assert client.post("/api/library/items/project", json = body).json() == {"already": True}


def test_add_to_project_refuses_what_it_cannot_copy(client, project):
    post = lambda item_id, project_id = "p1": client.post(  # noqa: E731
        "/api/library/items/project", json = {"id": item_id, "projectId": project_id}
    )
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    assert post(note, "missing").status_code == 404
    assert post("upload:0123456789abcdef0123456789abcdef").status_code == 404
    assert post("attachment:m:a").status_code == 400
    assert post("model:training:/tmp/run").status_code == 400


# ── Reveal in Finder ─────────────────────────────────────────────


@pytest.fixture
def revealed(monkeypatch):
    import utils.paths.path_utils as path_utils

    import utils.paths.file_manager as file_manager

    calls = []
    # CI runs on a headless Linux, where no file manager is reported.
    monkeypatch.setattr(file_manager, "file_manager_kind", lambda: "files")
    monkeypatch.setattr(path_utils, "reveal_in_file_manager", lambda path: calls.append(str(path)))
    return calls


def test_reveal_opens_the_items_own_file(client, revealed):
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    assert client.post("/api/library/items/reveal", json = {"id": note}).status_code == 200
    assert revealed == [str(library.upload_path(note.split(":", 1)[1]))]


def test_reveal_refuses_what_has_no_file_or_is_outside_studio(client, revealed):
    post = lambda item_id: client.post("/api/library/items/reveal", json = {"id": item_id})  # noqa: E731
    assert post("attachment:m:a").status_code == 400
    assert post("upload:0123456789abcdef0123456789abcdef").status_code == 404
    # A model id carries its path, so one outside the outputs and exports roots is not opened.
    assert post("model:training:/etc").status_code == 404
    assert revealed == []


def test_only_the_installation_owner_can_reveal(client, revealed, monkeypatch):
    monkeypatch.setattr(library_routes.account_access, "managed_account", lambda: True)
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    assert client.post("/api/library/items/reveal", json = {"id": note}).status_code == 403
    assert client.post("/api/library/locations/reveal", json = {"key": "images"}).status_code == 403
    assert revealed == []


def test_locations_are_listed_and_revealed_by_key(client, revealed):
    locations = client.get("/api/library/locations").json()["locations"]
    paths = {entry["key"]: entry["path"] for entry in locations}
    assert set(paths) == {"uploads", "images", "videos", "audio", "fineTunes", "exports"}
    assert client.post("/api/library/locations/reveal", json = {"key": "images"}).status_code == 200
    assert revealed == [paths["images"]]
    assert client.post("/api/library/locations/reveal", json = {"key": "/etc"}).status_code == 404


# ── Review fixes ─────────────────────────────────────────────────


def test_archived_gallery_items_stay_in_the_library(client, monkeypatch):
    from core.inference import video_gallery

    monkeypatch.setattr(library, "_SOURCES", (library._video_items,))
    meta = {
        k: 1
        for k in ("width", "height", "num_frames", "fps", "duration_s", "steps", "guidance", "seed")
    }
    record = video_gallery.save(
        b"\0\0\0\x18ftypmp42", {**meta, "prompt": "Archived", "created_at": 1}
    )
    video_gallery.set_flags(record["id"], archived = True)
    assert _items(client)[0][f"video:{record['id']}"]["archived"] is True


def test_items_download_as_attachments(client, monkeypatch):
    [upload] = _upload(client, ("page.html", b"<script>1</script>", "text/html"))
    # Checks its own credentials, header or query, rather than the overridden dependency.
    assert client.get("/api/library/items/download", params = {"id": upload}).status_code == 401

    async def signed_in(_request, _token):
        return "unsloth"

    monkeypatch.setattr(library_routes, "subject_for_header_or_query_token", signed_in)
    response = client.get("/api/library/items/download", params = {"id": upload})
    assert response.status_code == 200
    assert response.content == b"<script>1</script>"
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"].startswith("attachment")
    assert client.head("/api/library/items/download", params = {"id": upload}).status_code == 200
    assert (
        client.get("/api/library/items/download", params = {"id": "upload:" + "0" * 32}).status_code
        == 404
    )
    assert (
        client.get("/api/library/items/download", params = {"id": "attachment:m:a"}).status_code
        == 400
    )


def test_an_oversized_desktop_drop_is_refused_before_copying(client, monkeypatch, tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 10)
    monkeypatch.setattr(library_routes, "_MAX_UPLOAD_BYTES", 4)
    monkeypatch.setattr(
        library,
        "open_native_upload",
        lambda lease: ("big.bin", "application/octet-stream", open(big, "rb")),
    )

    def copied(*_args):
        raise AssertionError("copied an oversized drop")

    monkeypatch.setattr(library, "save_upload", copied)
    # Refused by the batch's check, before any grant is spent...
    monkeypatch.setattr(library, "check_native_upload", lambda lease: ("big.bin", 10))
    response = client.post("/api/library/uploads", data = {"nativePathLeases": ["lease"]})
    assert response.status_code == 413
    # ...and again when read, for a file that grew since.
    monkeypatch.setattr(library, "check_native_upload", lambda lease: ("big.bin", 1))
    response = client.post("/api/library/uploads", data = {"nativePathLeases": ["lease"]})
    assert response.status_code == 413


def test_concurrent_deletes_of_one_upload_leave_nothing_behind(client):
    from concurrent.futures import ThreadPoolExecutor

    [upload] = _upload(client, ("twice.txt", b"hi", "text/plain"))
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: library.delete_item(upload), range(4)))
    assert results.count(True) == 1
    assert not any(library.uploads_dir().iterdir())
    assert upload not in _items(client)[0]


def test_a_failed_note_save_keeps_the_old_text(client, monkeypatch):
    from storage import library_db

    [note] = _upload(client, ("n.md", b"old", "text/markdown"))
    upload_id = note.split(":", 1)[1]

    def broken(*_args):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(library_db, "touch_upload", broken)
    with pytest.raises(sqlite3.OperationalError):
        library.write_upload_text(upload_id, "new")
    assert library.upload_path(upload_id).read_bytes() == b"old"
    assert [p.name for p in library.uploads_dir().iterdir()] == [upload_id]


def test_a_note_save_racing_its_delete_leaves_nothing_behind(client):
    from concurrent.futures import ThreadPoolExecutor

    [note] = _upload(client, ("n.md", b"old", "text/markdown"))
    upload_id = note.split(":", 1)[1]
    with ThreadPoolExecutor(8) as pool:
        jobs = [pool.submit(library.write_upload_text, upload_id, f"v{i}") for i in range(6)]
        jobs.append(pool.submit(library.delete_item, note))
        [job.result() for job in jobs]
    assert not any(library.uploads_dir().iterdir())


def test_a_failed_upload_record_leaves_no_file(client, monkeypatch):
    from storage import library_db

    def broken(*_args):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(library_db, "insert_upload", broken)
    with pytest.raises(sqlite3.OperationalError):
        library.save_upload("a.txt", "text/plain", [b"hi"])
    assert not any(library.uploads_dir().iterdir())


def test_a_failed_upload_delete_keeps_the_file_and_its_row(client, monkeypatch):
    from storage import library_db

    [upload] = _upload(client, ("keep.txt", b"hi", "text/plain"))
    path = library.upload_path(upload.split(":", 1)[1])

    def broken(*_args):
        raise sqlite3.OperationalError("database is locked")

    original = library_db.delete_upload
    monkeypatch.setattr(library_db, "delete_upload", broken)
    with pytest.raises(sqlite3.OperationalError):
        library.delete_item(upload)
    assert path.read_bytes() == b"hi"
    monkeypatch.setattr(library_db, "delete_upload", original)
    assert upload in _items(client)[0]


def test_a_crafted_sandbox_id_reaches_only_listed_files(client, monkeypatch):
    import os

    from core.inference.tools import resolve_sandbox_workdir
    from storage import studio_db

    monkeypatch.setattr(library, "_SOURCES", (library._sandbox_items,))
    studio_db.upsert_chat_thread(
        {"id": "t-lib", "title": "T", "modelType": "base", "modelId": "m", "createdAt": 1}
    )
    directory = resolve_sandbox_workdir("t-lib")
    os.makedirs(directory, exist_ok = True)
    for name in ("report.txt", ".secret"):
        with open(os.path.join(directory, name), "w") as handle:
            handle.write("x")
    delete = lambda item_id: client.post("/api/library/items/delete", json = {"id": item_id})  # noqa: E731
    assert "sandbox:t-lib:report.txt" in _items(client)[0]
    # Hidden files and unknown sessions are not the Library's to touch.
    assert delete("sandbox:t-lib:.secret").status_code == 404
    assert delete("sandbox:not-a-chat:report.txt").status_code == 404
    assert os.path.exists(os.path.join(directory, ".secret"))
    assert delete("sandbox:t-lib:report.txt").status_code == 200
    assert not os.path.exists(os.path.join(directory, "report.txt"))


def test_a_failed_note_save_keeps_the_previous_text(client, monkeypatch):
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    upload_id = note.split(":", 1)[1]

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(library.os, "replace", fail)
    with pytest.raises(OSError):
        library.write_upload_text(upload_id, "# new plan")
    path = library.upload_path(upload_id)
    assert path.read_bytes() == b"# plan"
    assert [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_a_batch_whose_folder_vanishes_keeps_nothing(client, monkeypatch):
    from storage import library_db

    folder = client.post("/api/library/folders", json = {"name": "Work"}).json()

    def gone(*_args, **_kwargs):
        raise KeyError(folder["id"])

    monkeypatch.setattr(library_db, "update_entry", gone)
    response = client.post(
        "/api/library/uploads",
        files = [("files", ("plan.md", b"# plan", "text/markdown"))],
        data = {"folderId": folder["id"]},
    )
    assert response.status_code == 404
    assert _items(client)[0] == {}


def _clip(width = 64, height = 48) -> bytes:
    import io

    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = "mp4") as out:
        stream = out.add_stream("libx264", rate = 4)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        for index in range(4):
            frame = av.VideoFrame.from_ndarray(
                np.full((height, width, 3), index * 40, dtype = np.uint8), format = "rgb24"
            )
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return buf.getvalue()


def test_video_upload_thumbnail_is_its_first_frame(client):
    from PIL import Image
    import io

    [clip] = _upload(client, ("clip.mp4", _clip(), "video/mp4"))
    response = client.get("/api/library/items/thumbnail", params = {"id": clip})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/webp"
    assert Image.open(io.BytesIO(response.content)).size == (64, 48)


def _png(
    width,
    height,
    mode = "RGB",
) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new(mode, (width, height), (200, 40, 40, 128) if mode == "RGBA" else (200, 40, 40)).save(
        buf, format = "PNG"
    )
    return buf.getvalue()


def _thumbnail_size(client, item_id):
    import io

    from PIL import Image

    response = client.get("/api/library/items/thumbnail", params = {"id": item_id})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/webp"
    return Image.open(io.BytesIO(response.content)).size


def test_image_thumbnail_is_bounded_and_cropped_as_the_card_shows_it(client):
    [big] = _upload(client, ("big.png", _png(3000, 2000), "image/png"))
    [tall] = _upload(client, ("tall.png", _png(1000, 5000, "RGBA"), "image/png"))
    [wide] = _upload(client, ("wide.png", _png(4000, 500), "image/png"))
    [small] = _upload(client, ("small.png", _png(100, 80), "application/octet-stream"))
    assert _thumbnail_size(client, big) == (640, 427)
    assert _thumbnail_size(client, tall) == (640, 960)
    assert _thumbnail_size(client, wide) == (640, 427)
    # Never scaled up.
    assert _thumbnail_size(client, small) == (100, 80)


def test_an_image_too_large_to_decode_has_no_thumbnail(client, monkeypatch):
    monkeypatch.setattr(library, "_THUMBNAIL_MAX_PIXELS", 100 * 100)
    [image] = _upload(client, ("huge.png", _png(101, 100), "image/png"))
    response = client.get("/api/library/items/thumbnail", params = {"id": image})
    assert response.status_code == 501


def test_undecodable_image_has_no_thumbnail(client):
    [image] = _upload(client, ("broken.png", b"not a png", "image/png"))
    response = client.get("/api/library/items/thumbnail", params = {"id": image})
    assert response.status_code == 501


def test_thumbnail_is_only_for_pictures(client):
    [note] = _upload(client, ("note.md", b"# hi", "text/markdown"))
    [svg] = _upload(client, ("icon.svg", b"<svg/>", "image/svg+xml"))
    for item_id in (note, svg, "upload:" + "0" * 32, "model:training:/tmp/x", "image:missing"):
        response = client.get("/api/library/items/thumbnail", params = {"id": item_id})
        assert response.status_code == 404, item_id


def test_undecodable_video_has_no_thumbnail(client):
    pytest.importorskip("av")
    [clip] = _upload(client, ("broken.mp4", b"not a video", "video/mp4"))
    response = client.get("/api/library/items/thumbnail", params = {"id": clip})
    assert response.status_code == 501


# ── Second review ────────────────────────────────────────────────


def test_explorer_gets_the_documented_select_command(tmp_path, monkeypatch):
    import subprocess

    import utils.paths.path_utils as path_utils

    target = tmp_path / "a b" / "c.txt"
    target.parent.mkdir()
    target.write_text("x")
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda command, *args, **kwargs: calls.append(command))
    monkeypatch.setattr(path_utils.sys, "platform", "win32")
    monkeypatch.setattr(path_utils.os, "name", "nt")
    path_utils.reveal_in_file_manager(target)
    monkeypatch.undo()
    # One string: a list quotes "/select,<path>" whole, which Explorer misreads when it has a space.
    assert calls == [f'explorer /select,"{target}"']


def test_wsl_hands_explorer_the_select_switch_and_path_apart(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    import utils.paths.path_utils as path_utils

    target = tmp_path / "a b" / "c.txt"
    target.parent.mkdir()
    target.write_text("x")
    calls = []

    def fake_run(command, *args, **kwargs):
        return SimpleNamespace(stdout = "C:\\Users\\me\\a b\\c.txt\n")

    monkeypatch.setattr(path_utils, "_IS_WSL", True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", lambda command, *args, **kwargs: calls.append(command))
    assert path_utils._wsl_reveal_in_explorer(target, is_file = True)
    assert calls == [["explorer.exe", "/select,", "C:\\Users\\me\\a b\\c.txt"]]
    assert subprocess.list2cmdline(calls[0]) == 'explorer.exe /select, "C:\\Users\\me\\a b\\c.txt"'


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


def test_a_projects_own_files_are_listed(client, monkeypatch, tmp_path):
    import os

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
    with open(os.path.join(directory, "files", "notes.txt"), "w") as handle:
        handle.write("x")
    item = _items(client)[0]["sandbox:project-p-lib:files/notes.txt"]
    assert item["threadTitle"] == "Research"

    # One pointed at a folder of the user's own keeps its files out of the Library.
    own = tmp_path / "My code"
    conn = studio_db.get_connection()
    conn.execute("UPDATE chat_projects SET root_path = ? WHERE id = 'p-lib'", (str(own),))
    conn.commit()
    conn.close()
    assert not library._studio_project_root(str(own), library._project_workspaces())
    assert "sandbox:project-p-lib:files/notes.txt" not in _items(client)[0]


def test_types_come_from_a_fixed_extension_map(client, monkeypatch):
    import mimetypes

    # Whatever the OS registry says: Windows apps remap types, and mimetypes calls .ts a video.
    monkeypatch.setattr(mimetypes, "guess_type", lambda *_a, **_k: ("video/mp2t", None))
    for name in ("app.ts", "view.tsx", "mod.mts"):
        assert library._guess_type(name) == "text/typescript", name
    assert library._guess_type("photo.JPG") == "image/jpeg"
    assert library._guess_type("mystery.xyz") == "application/octet-stream"


def test_an_upload_is_typed_by_its_extension_not_the_client(client):
    [page, pdf, odd, blob] = _upload(
        client,
        ("notes.txt", b"<script>1</script>", "text/html"),
        ("report.pdf", b"%PDF-1.7", "text/html"),
        ("track.xyz", b"x", "audio/x, text/html"),
        ("data.bin2", b"x", "application/xhtml+xml"),
    )
    items, _ = _items(client)
    assert items[page]["contentType"] == "text/plain"
    assert items[pdf]["contentType"] == "application/pdf"
    # A list of types, or a type a browser would run, is not stored for an unknown extension.
    assert items[odd]["contentType"] == "application/octet-stream"
    assert items[blob]["contentType"] == "application/octet-stream"


def test_only_an_exact_media_type_is_served_inline(client, monkeypatch):
    from storage import library_db

    [clip] = _upload(client, ("clip.xyz", b"x", "application/octet-stream"))
    upload_id = clip.split(":", 1)[1]
    url = f"/api/library/uploads/{upload_id}/file"
    conn = library_db.get_connection()
    for stored, inline in (
        ("audio/x, text/html", False),
        ("audio/mpeg", True),
        ("image/png; charset=binary", True),
        ("image/svg+xml", False),
    ):
        conn.execute(
            "UPDATE library_uploads SET content_type = ? WHERE id = ?", (stored, upload_id)
        )
        conn.commit()
        response = client.get(url)
        assert response.headers["x-content-type-options"] == "nosniff", stored
        disposition = response.headers.get("content-disposition", "")
        assert ("attachment" not in disposition) is inline, stored
        if not inline:
            assert response.headers["content-type"] == "application/octet-stream"
    conn.close()


def test_a_failed_batch_gives_its_desktop_grants_back(client, lease_secret, tmp_path, monkeypatch):
    from .test_rag_native_drop_upload import _sign

    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("a")
    second.write_text("b")
    leases = [_sign(first), _sign(second)]
    original = library.save_upload
    calls = []

    def fail_second(*args):
        calls.append(args[0])
        if len(calls) == 2:
            raise OSError("disk full")
        return original(*args)

    monkeypatch.setattr(library, "save_upload", fail_second)
    with pytest.raises(OSError):
        client.post("/api/library/uploads", data = {"nativePathLeases": leases})
    assert _items(client)[0] == {}
    # The same grants work on a retry: nothing the failed batch spent stays spent.
    monkeypatch.setattr(library, "save_upload", original)
    response = client.post("/api/library/uploads", data = {"nativePathLeases": leases})
    assert response.status_code == 200, response.text
    assert len(response.json()["ids"]) == 2
    # And a grant that was used is still single use.
    response = client.post("/api/library/uploads", data = {"nativePathLeases": leases[:1]})
    assert response.status_code == 400


def test_a_bad_grant_refuses_the_batch_before_any_grant_is_spent(client, lease_secret, tmp_path):
    from .test_rag_native_drop_upload import _sign

    good = tmp_path / "good.txt"
    good.write_text("x")
    lease = _sign(good)
    forged = _sign(good, secret = b"x" * 32)
    response = client.post("/api/library/uploads", data = {"nativePathLeases": [lease, forged]})
    assert response.status_code == 400
    # A generic reason: the grant's own can name a path in the workspace.
    assert response.json()["detail"] == "The dropped file could not be read. Drop it again."
    response = client.post("/api/library/uploads", data = {"nativePathLeases": [lease]})
    assert response.status_code == 200, response.text


@pytest.fixture
def signed_in(monkeypatch):
    async def subject(_request, _token):
        return "unsloth"

    monkeypatch.setattr(library_routes, "subject_for_header_or_query_token", subject)


def test_an_upload_downloads_under_the_name_it_was_given(client, signed_in):
    # Stored as a bare id, so the header names it after the upload, not the file on disk.
    [upload] = _upload(client, ("re:port*q3?.csv", b"a,b", "text/csv"))
    response = client.get("/api/library/items/download", params = {"id": upload})
    assert response.status_code == 200
    assert response.content == b"a,b"
    disposition = response.headers["content-disposition"]
    assert disposition.startswith('attachment; filename="re port q3.csv"')
    assert response.headers["content-length"] == "3"
    assert response.headers["x-content-type-options"] == "nosniff"


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
    assert library.safe_file_name(name) == safe
    project_name = library._project_name(name, "upload:x")
    from core.inference.gallery_projects import _bad_name

    assert not _bad_name(project_name), project_name


def test_add_to_project_refuses_a_name_windows_cannot_hold(tmp_path, monkeypatch, project):
    from core.inference import gallery_projects as gp

    source = tmp_path / "x.txt"
    source.write_text("x")
    for bad in ("a:b.txt", "CON.txt", "trailing.", "tab\there.txt"):
        with pytest.raises(ValueError):
            gp.copy_into_project(source, "p1", "files", bad)
    # An open file copies from its descriptor, under the name it is given.
    with open(source, "rb") as handle:
        result = gp.copy_into_project(handle, "p1", "files", "x-1.txt")
    assert Path(result["path"]).read_text() == "x"


def test_an_image_attachment_has_a_thumbnail(client, monkeypatch):
    import base64

    import storage.studio_db as studio_db

    data = base64.b64encode(_png(120, 90)).decode("ascii")
    attachment = {
        "type": "image",
        "contentType": "image/png",
        "content": [{"type": "image", "image": f"data:image/png;base64,{data}"}],
    }
    monkeypatch.setattr(
        studio_db, "get_chat_attachment", lambda message_id, attachment_id: attachment
    )
    assert _thumbnail_size(client, "attachment:m:a") == (120, 90)
    attachment["content"] = [{"type": "text", "text": "just words"}]
    library._thumbnail_cache.clear()
    response = client.get("/api/library/items/thumbnail", params = {"id": "attachment:m:a"})
    assert response.status_code == 404


def test_thumbnails_open_only_raster_formats(client):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format = "PPM")
    # A format Pillow knows but a card does not need is never handed to its decoder.
    [ppm] = _upload(client, ("pic.png", buf.getvalue(), "image/png"))
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 8 8\nshowpage\n"
    [ps] = _upload(client, ("pic.jpg", eps, "image/jpeg"))
    for item_id in (ppm, ps):
        response = client.get("/api/library/items/thumbnail", params = {"id": item_id})
        assert response.status_code == 501, item_id


def test_a_video_thumbnail_reads_its_container_by_type(client, monkeypatch):
    from core.inference import video_gallery

    seen = []
    monkeypatch.setattr(
        video_gallery,
        "first_frame_webp",
        lambda source, **kwargs: seen.append((hasattr(source, "read"), kwargs)) or b"webp",
    )
    [clip] = _upload(client, ("clip.webm", b"x", "video/webm"))
    [playlist] = _upload(client, ("clip.m3u8", b"#EXTM3U", "video/mp4"))
    [avi] = _upload(client, ("clip.flv", b"x", "video/x-flv"))
    for item_id in (clip, playlist):
        assert client.get("/api/library/items/thumbnail", params = {"id": item_id}).status_code == 200
    # A descriptor, never a name to reopen, and a demuxer forced rather than probed, so a playlist
    # sent as mp4 is read as mp4 and never followed as HLS.
    assert [(is_stream, kwargs["container"]) for is_stream, kwargs in seen] == [
        (True, "webm"),
        (True, "mp4"),
    ]
    assert seen[0][1]["max_pixels"] == library._THUMBNAIL_MAX_PIXELS
    # A container with no demuxer on the list has no picture.
    assert client.get("/api/library/items/thumbnail", params = {"id": avi}).status_code == 404


def test_thumbnails_are_cached_by_version_and_decoded_a_few_at_a_time(client, monkeypatch):
    calls = []
    real = library._picture

    def counted(mime_type, source):
        calls.append(mime_type)
        return real(mime_type, source)

    monkeypatch.setattr(library, "_picture", counted)
    [image] = _upload(client, ("a.png", _png(40, 30), "image/png"))
    for _ in range(3):
        assert _thumbnail_size(client, image) == (40, 30)
    assert calls == ["image/png"]
    # A new version of the file is a new size and mtime, so a new picture.
    library.upload_path(image.split(":", 1)[1]).write_bytes(_png(30, 20))
    assert _thumbnail_size(client, image) == (30, 20)
    assert len(calls) == 2


def _sandbox_chat(
    monkeypatch,
    name = "report.txt",
    body = b"x",
):
    import os

    from core.inference.tools import resolve_sandbox_workdir
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": "t-lib", "title": "T", "modelType": "base", "modelId": "m", "createdAt": 1}
    )
    directory = resolve_sandbox_workdir("t-lib")
    os.makedirs(directory, exist_ok = True)
    path = os.path.join(directory, name)
    with open(path, "wb") as handle:
        handle.write(body)
    return directory, path


def test_a_sandbox_file_swapped_for_a_link_after_the_check_is_not_read(
    client, signed_in, monkeypatch, tmp_path
):
    import os

    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    directory, path = _sandbox_chat(monkeypatch, body = b"mine")
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
    response = client.get("/api/library/items/download", params = {"id": "sandbox:t-lib:report.txt"})
    assert response.status_code == 404
    assert b"secret" not in response.content


def test_a_sandbox_file_is_found_without_listing_every_chat(client, signed_in, monkeypatch):
    directory, _path = _sandbox_chat(monkeypatch, body = b"mine")

    def listed(*_args):
        raise AssertionError("a per-item route listed the sandboxes")

    monkeypatch.setattr(library, "_sandbox_sessions", listed)
    response = client.get("/api/library/items/download", params = {"id": "sandbox:t-lib:report.txt"})
    assert response.status_code == 200
    assert response.content == b"mine"
    assert 'filename="report.txt"' in response.headers["content-disposition"]
    for crafted in ("sandbox:t-lib:../x", "sandbox:t-lib:.hidden", "sandbox:nope:report.txt"):
        response = client.get("/api/library/items/download", params = {"id": crafted})
        assert response.status_code == 404, crafted


def test_a_file_held_open_on_windows_answers_409(client, monkeypatch):
    [note] = _upload(client, ("n.md", b"old", "text/markdown"))
    upload_id = note.split(":", 1)[1]

    def held(*_args):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(library.os, "replace", held)
    response = client.put(f"/api/library/uploads/{upload_id}/text", json = {"text": "new"})
    assert response.status_code == 409
    assert response.json()["detail"] == "The file is in use. Close it and try again."
    response = client.post("/api/library/items/delete", json = {"id": note})
    assert response.status_code == 409

    def broken(*_args):
        raise OSError(5, "I/O error", "/secret/path")

    monkeypatch.setattr(library.os, "replace", broken)
    response = client.post("/api/library/items/delete", json = {"id": note})
    assert response.status_code == 500
    assert "/secret" not in response.text


def test_a_missing_file_manager_is_not_a_missing_file(client, monkeypatch):
    import utils.paths.file_manager as file_manager
    import utils.paths.path_utils as path_utils

    monkeypatch.setattr(file_manager, "file_manager_kind", lambda: "files")

    def no_launcher(path):
        raise FileNotFoundError(2, "No such file or directory: 'xdg-open'")

    monkeypatch.setattr(path_utils, "reveal_in_file_manager", no_launcher)
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    response = client.post("/api/library/items/reveal", json = {"id": note})
    assert response.status_code == 503
    assert response.json()["detail"] == "No file manager is available on this machine"


def test_a_host_with_no_file_manager_refuses_to_reveal(client, revealed, monkeypatch):
    import utils.paths.file_manager as file_manager

    monkeypatch.setattr(file_manager, "file_manager_kind", lambda: None)
    [note] = _upload(client, ("plan.md", b"# plan", "text/markdown"))
    response = client.post("/api/library/items/reveal", json = {"id": note})
    assert response.status_code == 503
    response = client.post("/api/library/locations/reveal", json = {"key": "images"})
    assert response.status_code == 503
    assert revealed == []


def test_the_slow_sources_are_remembered_briefly_and_forgotten_on_a_write(client, monkeypatch):
    calls = []

    def walked():
        calls.append(1)
        return [
            library._item(
                "sandbox:t:a.txt",
                name = "a.txt",
                source = "generated",
                content_type = "text/plain",
                size_bytes = 1,
                created_at = 1,
                file_url = "",
            )
        ]

    monkeypatch.setattr(library, "_sandbox_items", walked)
    remembered = library._remembered("_sandbox_items", 60)
    monkeypatch.setattr(library, "_SOURCES", (remembered,))
    client.patch("/api/library/items", json = {"id": "sandbox:t:a.txt", "favorite": True})
    assert _items(client)[0]["sandbox:t:a.txt"]["favorite"] is True
    assert _items(client)[0]["sandbox:t:a.txt"]["favorite"] is True
    assert len(calls) == 1
    # Another account's listing is never this one's.
    account = ["other"]
    monkeypatch.setattr(library, "_account_key", lambda: account[0])
    _items(client)
    assert len(calls) == 2
    _items(client)
    assert len(calls) == 2
    library.invalidate_listing()
    _items(client)
    assert len(calls) == 3


def test_a_new_fine_tune_changes_the_model_stamp(client):
    import shutil

    from utils.paths.storage_roots import outputs_root

    outputs_root().mkdir(parents = True, exist_ok = True)
    before = library._model_stamp()
    run = outputs_root() / "stamp-run"
    run.mkdir()
    try:
        assert library._model_stamp() != before
    finally:
        shutil.rmtree(run)


def test_the_schema_is_checked_once_per_database(client, monkeypatch, tmp_path):
    from storage import library_db

    # A home reached through a link: the unresolved path is not the key the schema was saved as.
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory = True)
    monkeypatch.setattr(library_db, "studio_db_path", lambda: tmp_path / "link" / "studio.db")
    library_db.get_connection().close()
    runs = []
    monkeypatch.setattr(library_db, "_ensure_schema", lambda conn: runs.append(conn))
    for _ in range(3):
        library_db.get_connection().close()
    assert runs == []


def test_an_overlay_row_stays_with_the_file_it_was_made_for(client, monkeypatch, tmp_path):
    import os

    from storage import library_db

    monkeypatch.setattr(library, "_SOURCES", (library._sandbox_items,))
    directory, path = _sandbox_chat(monkeypatch, body = b"v1")
    item_id = "sandbox:t-lib:report.txt"
    client.patch("/api/library/items", json = {"id": item_id, "favorite": True, "name": "Q3"})
    assert _items(client)[0][item_id]["favorite"] is True

    # Edited in place, as a tool appending to its output: still the same file.
    with open(path, "ab") as handle:
        handle.write(b" and v2")
    library.invalidate_listing()
    assert (_items(client)[0][item_id]["favorite"], _items(client)[0][item_id]["name"]) == (
        True,
        "Q3",
    )
    assert client.get("/api/library/favorites").json() == {"ids": [item_id]}

    # Deleted outside the Library and made again at the path. Moved aside rather than unlinked,
    # so no filesystem can hand the new file the old one's inode.
    os.replace(path, tmp_path / "old-report.txt")
    with open(path, "wb") as handle:
        handle.write(b"new")
    assert client.get("/api/library/favorites").json() == {"ids": []}
    library.invalidate_listing()
    item = _items(client)[0][item_id]
    assert (item["favorite"], item["name"]) == (False, "report.txt")
    # Pruned, so starring the new file starts from nothing.
    assert item_id not in library_db.list_entries()
    client.patch("/api/library/items", json = {"id": item_id, "favorite": True})
    assert library_db.list_entries()[item_id]["name"] is None


def test_a_patch_for_a_new_file_at_the_path_drops_the_old_files_row(client, monkeypatch, tmp_path):
    import os

    from storage import library_db

    directory, path = _sandbox_chat(monkeypatch)
    item_id = "sandbox:t-lib:report.txt"
    client.patch("/api/library/items", json = {"id": item_id, "name": "Old name"})
    os.replace(path, tmp_path / "old.txt")
    with open(path, "wb") as handle:
        handle.write(b"new")
    library.invalidate_listing()
    # No listing in between: the write itself sees the row belongs to another file.
    client.patch("/api/library/items", json = {"id": item_id, "favorite": True})
    entry = library_db.list_entries()[item_id]
    assert (entry["name"], entry["favorite"]) == (None, True)


def test_a_row_from_before_fingerprints_is_adopted(client, monkeypatch):
    from storage import library_db

    monkeypatch.setattr(library, "_SOURCES", (library._sandbox_items,))
    _sandbox_chat(monkeypatch)
    item_id = "sandbox:t-lib:report.txt"
    library_db.update_entry(item_id, favorite = True)
    assert library_db.list_entries()[item_id]["fingerprint"] is None
    assert _items(client)[0][item_id]["favorite"] is True
    assert library_db.list_entries()[item_id]["fingerprint"] == library.fingerprint(item_id)
    assert "_fingerprint" not in _items(client)[0][item_id]


def test_an_older_database_gains_the_fingerprint_column(client, monkeypatch, tmp_path):
    from storage import library_db

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE library_entries (item_id TEXT NOT NULL PRIMARY KEY, name TEXT, "
        "favorite INTEGER NOT NULL DEFAULT 0, folder_id TEXT, updated_at INTEGER NOT NULL)"
    )
    conn.execute("INSERT INTO library_entries (item_id, favorite, updated_at) VALUES ('x', 1, 1)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(library_db, "studio_db_path", lambda: db)
    assert library_db.list_entries()["x"] == {
        "name": None,
        "favorite": True,
        "folderId": None,
        "updatedAt": 1,
        "fingerprint": None,
    }


def _gallery_image(prompt: str) -> str:
    from PIL import Image

    from core.inference import image_gallery

    record = image_gallery.save(
        Image.new("RGB", (4, 4)),
        {
            "prompt": prompt,
            "width": 4,
            "height": 4,
            "steps": 1,
            "guidance": 1,
            "seed": 1,
            "created_at": 1,
        },
    )
    return record["id"]


def test_generated_media_download_under_their_prompt(client, signed_in, project):
    from urllib.parse import quote

    from core.inference import audio_gallery

    image = _gallery_image('A red fox: "at dawn"')
    response = client.get("/api/library/items/download", params = {"id": f"image:{image}"})
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith(
        'attachment; filename="A red fox at dawn.png"'
    )
    # The UTF-8 name rides along for a prompt ASCII cannot hold.
    image = _gallery_image("Café ☕ at noon")
    disposition = client.get(
        "/api/library/items/download", params = {"id": f"image:{image}"}
    ).headers["content-disposition"]
    assert f"filename*=UTF-8''{quote('Café ☕ at noon.png')}" in disposition
    # No prompt: the id, never an empty name.
    image = _gallery_image("  ")
    disposition = client.get(
        "/api/library/items/download", params = {"id": f"image:{image}"}
    ).headers["content-disposition"]
    assert f'filename="{image}.png"' in disposition

    clip = audio_gallery.save(
        b"RIFF0000WAVE",
        {
            "prompt": "Hello there",
            "model": "sample-tts",
            "audio_type": "snac",
            "sample_rate": 24000,
            "duration_s": 1.0,
            "created_at": 1,
        },
    )
    disposition = client.get(
        "/api/library/items/download", params = {"id": f"audio:{clip['id']}"}
    ).headers["content-disposition"]
    assert 'filename="Hello there.wav"' in disposition

    # A project copy keeps the stored name, as the Images page's own Add to project does, so
    # adding from either place finds the other's copy; two alike stay two files.
    from core.inference import gallery_projects, image_gallery

    first, second = _gallery_image("Same prompt"), _gallery_image("Same prompt")
    for image in (first, second):
        response = client.post(
            "/api/library/items/project", json = {"id": f"image:{image}", "projectId": "p1"}
        )
        assert response.json() == {"already": False}
    names = sorted(path.name for path in (project / "images").iterdir())
    assert names == sorted(f"{image}.png" for image in (first, second))
    # What the Images page's own route does with the same image.
    path = image_gallery.owned_image_path(first)
    assert gallery_projects.copy_into_project(path, "p1", "images")["already"] is True


def test_a_sandbox_file_past_the_listing_cap_is_not_reachable_by_id(client, signed_in, monkeypatch):
    import os

    import routes.inference as inference

    monkeypatch.setattr(library, "_SOURCES", (library._sandbox_items,))
    # The walk counts dotfiles too, which the Library then hides: .env, a.txt, b.txt.
    monkeypatch.setattr(inference, "_MAX_SNAPSHOT_FILES", 3)
    directory, _path = _sandbox_chat(monkeypatch, name = "a.txt")
    for name in ("b.txt", "c.txt", ".env"):
        with open(os.path.join(directory, name), "w") as handle:
            handle.write("x")
    os.makedirs(os.path.join(directory, "d1", "d2", "d3", "d4"))
    with open(os.path.join(directory, "d1", "d2", "d3", "d4", "deep.txt"), "w") as handle:
        handle.write("x")
    download = lambda name: client.get(  # noqa: E731
        "/api/library/items/download", params = {"id": f"sandbox:t-lib:{name}"}
    ).status_code
    # Cold: this sandbox is walked once, and the answer serves every card after it.
    walks = []
    real = inference._sandbox_listing_names
    monkeypatch.setattr(
        inference, "_sandbox_listing_names", lambda path: walks.append(path) or real(path)
    )
    assert (download("a.txt"), download("b.txt")) == (200, 200)
    assert len(walks) == 1
    for name in ("c.txt", ".env", "d1/d2/d3/d4/deep.txt"):
        assert download(name) == 404, name
    assert set(_items(client)[0]) == {"sandbox:t-lib:a.txt", "sandbox:t-lib:b.txt"}
    # The listing leaves its walk for the per-item routes.
    library.invalidate_listing()
    _items(client)
    walks.clear()
    assert download("a.txt") == 200 and download("c.txt") == 404
    assert walks == []
    response = client.post("/api/library/items/delete", json = {"id": "sandbox:t-lib:c.txt"})
    assert response.status_code == 404
    assert os.path.exists(os.path.join(directory, "c.txt"))


# ── Streamed previews ────────────────────────────────────────────

_CLIP = bytes(range(100))


def _stream_url(client, item_id):
    response = client.get("/api/library/items/stream-url", params = {"id": item_id})
    assert response.status_code == 200, response.text
    url = response.json()["url"]
    assert url.startswith("/api/library/items/stream?")
    return url


def _unauthenticated():
    # The real dependencies: the stream route must not need the bearer the mint does.
    app = FastAPI()
    app.include_router(library_routes.router, prefix = "/api/library")
    return TestClient(app)


def test_audio_and_video_stream_from_a_signed_link_with_ranges(client):
    [clip, song] = _upload(
        client, ("clip.mp4", _CLIP, "video/mp4"), ("song.mp3", b"ID3" + b"x" * 7, "audio/mpeg")
    )
    url = _stream_url(client, clip)
    stream = _unauthenticated()
    whole = stream.get(url)
    assert whole.status_code == 200
    assert whole.content == _CLIP
    assert whole.headers["content-type"] == "video/mp4"
    assert whole.headers["x-content-type-options"] == "nosniff"
    assert whole.headers["cache-control"] == "private"
    assert whole.headers["accept-ranges"] == "bytes"
    assert whole.headers["content-length"] == "100"
    for header, status, body, content_range in (
        ("bytes=10-19", 206, _CLIP[10:20], "bytes 10-19/100"),
        ("bytes=95-", 206, _CLIP[95:], "bytes 95-99/100"),
        ("bytes=90-500", 206, _CLIP[90:], "bytes 90-99/100"),
        ("bytes=-5", 206, _CLIP[-5:], "bytes 95-99/100"),
        ("bytes=-500", 206, _CLIP, "bytes 0-99/100"),
    ):
        response = stream.get(url, headers = {"Range": header})
        assert response.status_code == status, header
        assert response.content == body, header
        assert response.headers["content-range"] == content_range, header
        assert response.headers["content-length"] == str(len(body)), header
    # Several ranges, or a malformed one, may be ignored: the whole file.
    for header in ("bytes=0-1,5-6", "bytes=5-1", "items=0-1", "bytes=a-b"):
        response = stream.get(url, headers = {"Range": header})
        assert (response.status_code, response.content) == (200, _CLIP), header
    for header in ("bytes=100-", "bytes=-0"):
        response = stream.get(url, headers = {"Range": header})
        assert response.status_code == 416, header
        assert response.headers["content-range"] == "bytes */100"
    head = stream.head(url, headers = {"Range": "bytes=0-9"})
    assert head.status_code == 206 and head.content == b""
    assert head.headers["content-length"] == "10"
    audio = stream.get(_stream_url(client, song))
    assert (audio.status_code, audio.headers["content-type"]) == (200, "audio/mpeg")


def _gallery_video():
    from core.inference import video_gallery
    meta = {
        key: 1
        for key in (
            "width",
            "height",
            "num_frames",
            "fps",
            "duration_s",
            "steps",
            "guidance",
            "seed",
        )
    }
    return video_gallery.save(
        b"\0\0\0\x18ftypmp42", {**meta, "prompt": "A calm sea", "created_at": 1_700_000_000}
    )


def test_generated_media_streams_under_its_gallery_type(client):
    from core.inference import audio_gallery

    video = _gallery_video()
    audio = audio_gallery.save(
        b"RIFF0000WAVE",
        {
            "prompt": "Hello",
            "model": "sample-tts",
            "audio_type": "snac",
            "sample_rate": 24000,
            "duration_s": 1.0,
            "created_at": 1_700_000_000,
        },
    )
    stream = _unauthenticated()
    response = stream.get(_stream_url(client, f"video:{video['id']}"))
    assert (response.status_code, response.headers["content-type"]) == (200, "video/mp4")
    assert response.content == b"\0\0\0\x18ftypmp42"
    response = stream.get(_stream_url(client, f"audio:{audio['id']}"))
    assert (response.status_code, response.headers["content-type"]) == (200, "audio/wav")


def test_only_audio_and_video_items_get_a_stream_link(client):
    [note, image, unknown] = _upload(
        client,
        ("note.md", b"# hi", "text/markdown"),
        ("pic.png", b"\x89PNG", "image/png"),
        # A declared type the extension map does not know is stored, but never streamed.
        ("clip.xyz", b"x", "video/x-custom"),
    )
    for item_id in (note, image, unknown, "attachment:m:a", "model:training:/x", "elsewhere:x"):
        response = client.get("/api/library/items/stream-url", params = {"id": item_id})
        assert response.status_code == 400, item_id
    missing = client.get("/api/library/items/stream-url", params = {"id": "upload:" + "0" * 32})
    assert missing.status_code == 404
    gone = client.get("/api/library/items/stream-url", params = {"id": "video:nope"})
    assert gone.status_code == 404
    # A link can never be made for one, so the stream refuses it even with a valid signature.
    token = library_routes._sign_stream_id(note)
    response = _unauthenticated().get(
        "/api/library/items/stream", params = {"id": note, "token": token}
    )
    assert response.status_code == 404
    assert b"# hi" not in response.content


def test_a_stream_link_is_minted_only_for_a_signed_in_caller(client):
    [clip] = _upload(client, ("clip.mp4", _CLIP, "video/mp4"))
    response = _unauthenticated().get("/api/library/items/stream-url", params = {"id": clip})
    assert response.status_code in (401, 403)
    client.app.dependency_overrides[request_admitted_without_credential] = lambda: True
    response = client.get("/api/library/items/stream-url", params = {"id": clip})
    assert response.status_code == 403


def test_a_tampered_expired_or_foreign_stream_link_is_refused(client, monkeypatch):
    [clip, other] = _upload(
        client, ("clip.mp4", _CLIP, "video/mp4"), ("other.mp4", b"other", "video/mp4")
    )
    token = library_routes._sign_stream_id(clip)
    stream = _unauthenticated()
    get = lambda item_id, value: stream.get(  # noqa: E731
        "/api/library/items/stream", params = {"id": item_id, "token": value}
    )
    assert get(clip, token).status_code == 200
    target, expires, signature = token.split(".")
    for bad in (
        f"{target}.{expires}.{'0' * len(signature)}",
        f"{target}.{int(expires) + 60}.{signature}",
        f"{library_routes._sign_stream_id(other).split('.')[0]}.{expires}.{signature}",
        "nonsense",
        "",
    ):
        assert get(clip, bad).status_code in (401, 422), bad
    # Names one item: another's id with it reads nothing.
    response = get(other, token)
    assert response.status_code == 401 and b"other" not in response.content
    monkeypatch.setattr(library_routes, "_STREAM_LINK_TTL", -1)
    assert get(clip, library_routes._sign_stream_id(clip)).status_code == 401


def test_media_links_of_one_kind_never_open_the_other(client):
    import routes.video as video_routes

    record = _gallery_video()
    item_id = f"video:{record['id']}"
    app = FastAPI()
    app.include_router(library_routes.router, prefix = "/api/library")
    app.include_router(video_routes.router, prefix = "/api/inference")
    both = TestClient(app)
    video_token = video_routes._sign_video_id(record["id"])
    response = both.get("/api/library/items/stream", params = {"id": item_id, "token": video_token})
    assert response.status_code == 401
    library_token = library_routes._sign_stream_id(record["id"])
    response = both.get(
        f"/api/inference/video/gallery/{record['id']}/file-signed",
        params = {"token": library_token},
    )
    assert response.status_code == 401
    # Each on its own route still plays.
    assert (
        both.get(
            f"/api/inference/video/gallery/{record['id']}/file-signed",
            params = {"token": video_token},
        ).status_code
        == 200
    )


def test_a_sandbox_clip_swapped_for_a_link_after_the_check_is_not_streamed(
    client, monkeypatch, tmp_path
):
    import os

    secret = tmp_path / "secret.mp3"
    secret.write_bytes(b"secret")
    _directory, path = _sandbox_chat(monkeypatch, name = "song.mp3", body = b"mine")
    item_id = "sandbox:t-lib:song.mp3"
    url = _stream_url(client, item_id)
    stream = _unauthenticated()
    assert stream.get(url).content == b"mine"
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
    response = stream.get(url)
    assert response.status_code == 404
    assert b"secret" not in response.content
    # A sandbox file is streamed only under an audio or video name.
    _sandbox_chat(monkeypatch, name = "notes.txt", body = b"text")
    response = client.get("/api/library/items/stream-url", params = {"id": "sandbox:t-lib:notes.txt"})
    assert response.status_code == 400


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

    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[request_admitted_without_credential] = lambda: False
    app.include_router(library_routes.router, prefix = "/api/library")
    return TestClient(app)


def test_a_stream_link_reads_only_its_own_accounts_item(client, two_accounts):
    from utils.account_context import OWNER, run_as

    alice, bob = two_accounts
    mine = run_as(alice, library.save_upload, "clip.mp4", "video/mp4", iter([b"alice"]))
    theirs = run_as(bob, library.save_upload, "clip.mp4", "video/mp4", iter([b"bob"]))
    mine_id, theirs_id = f"upload:{mine['id']}", f"upload:{theirs['id']}"
    with _account_client(alice) as as_alice, _account_client(bob) as as_bob:
        url = _stream_url(as_alice, mine_id)
        # Bob cannot mint a link to Alice's item, however he spells it.
        assert (
            as_bob.get("/api/library/items/stream-url", params = {"id": mine_id}).status_code == 404
        )
    stream = _unauthenticated()
    # The link is the credential: it plays Alice's file for whoever holds it, in her account.
    response = stream.get(url)
    assert (response.status_code, response.content) == (200, b"alice")
    token = run_as(alice, library_routes._sign_stream_id, theirs_id)
    response = stream.get("/api/library/items/stream", params = {"id": theirs_id, "token": token})
    assert response.status_code == 404 and b"bob" not in response.content
    alice_token = run_as(alice, library_routes._sign_stream_id, mine_id)
    response = stream.get(
        "/api/library/items/stream", params = {"id": theirs_id, "token": alice_token}
    )
    assert response.status_code == 401
    # An owner's link resolves in the owner's own Library, where Alice's upload is not.
    owner_token = run_as(OWNER, library_routes._sign_stream_id, mine_id)
    response = stream.get("/api/library/items/stream", params = {"id": mine_id, "token": owner_token})
    assert response.status_code == 404


def test_favorites_check_a_gallery_file_without_decoding_it(client, monkeypatch):
    from core.inference import image_gallery

    image = _gallery_image("A star")
    client.patch("/api/library/items", json = {"id": f"image:{image}", "favorite": True})
    reads = []
    real = image_gallery._read_meta
    monkeypatch.setattr(image_gallery, "_read_meta", lambda path: reads.append(path) or real(path))
    # The Images, Video and Audio pages ask this for every star each time they open.
    assert f"image:{image}" in client.get("/api/library/favorites").json()["ids"]
    assert reads == []
    image_gallery.image_path(image).unlink()
    assert f"image:{image}" not in client.get("/api/library/favorites").json()["ids"]
