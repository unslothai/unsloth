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

from auth.authentication import get_current_subject  # noqa: E402
from core import library  # noqa: E402
from routes import library as library_routes  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # Only the Library's own uploads: the other sources read chat, gallery and sandbox stores this
    # test does not seed.
    monkeypatch.setattr(library, "_SOURCES", (library._upload_items,))
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
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


def test_opening_an_item_records_when(client):
    [note] = _upload(client, ("note.md", b"hi", "text/markdown"))
    assert _items(client)[0][note]["openedAt"] is None
    assert client.post("/api/library/items/opened", json = {"id": note}).status_code == 200
    opened = _items(client)[0][note]
    assert opened["openedAt"] >= opened["createdAt"]
    assert not opened["favorite"]


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
        "model",
        "sandbox",
    }


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
    finally:
        shutil.rmtree(run)


def test_favorites_lists_only_favorite_ids(client):
    client.patch("/api/library/items", json = {"id": "image:abc", "favorite": True})
    client.patch("/api/library/items", json = {"id": "image:def", "favorite": False})
    assert client.get("/api/library/favorites").json() == {"ids": ["image:abc"]}


def test_fine_tuned_models_are_listed_but_not_deleted_here(client, monkeypatch):
    import shutil

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
        # Once the models route has deleted it, the Library forgets its favorite too.
        client.patch("/api/library/items", json = {"id": item["id"], "favorite": True})
        shutil.rmtree(run)
        response = client.post("/api/library/items/delete", json = {"id": item["id"]})
        assert response.status_code == 200
        assert client.get("/api/library/favorites").json() == {"ids": []}
    finally:
        shutil.rmtree(run, ignore_errors = True)


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

    calls = []
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
        library, "open_native_upload", lambda lease: ("big.bin", "application/octet-stream", open(big, "rb"))
    )

    def copied(*_args):
        raise AssertionError("copied an oversized drop")

    monkeypatch.setattr(library, "save_upload", copied)
    response = client.post("/api/library/uploads", data = {"nativePathLeases": ["lease"]})
    assert response.status_code == 413


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


def test_thumbnail_is_only_for_videos(client):
    [note] = _upload(client, ("note.md", b"# hi", "text/markdown"))
    for item_id in (note, "upload:" + "0" * 32, "model:training:/tmp/x", "image:missing"):
        response = client.get("/api/library/items/thumbnail", params = {"id": item_id})
        assert response.status_code == 404, item_id


def test_undecodable_video_has_no_thumbnail(client):
    pytest.importorskip("av")
    [clip] = _upload(client, ("broken.mp4", b"not a video", "video/mp4"))
    response = client.get("/api/library/items/thumbnail", params = {"id": clip})
    assert response.status_code == 501


# ── Moving a kind of file ────────────────────────────────────────


@pytest.fixture(autouse = True)
def _temp_folders_are_ordinary(monkeypatch):
    # macOS keeps pytest's temp folders under /private/var, which the real check refuses.
    import hub.storage.scan_folders as scan_folders
    monkeypatch.setattr(
        scan_folders, "is_denied_system_path", lambda path: path.startswith(("/etc", "/usr"))
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
    assert _location(client, "images") == {
        "key": "images",
        "path": str(new.resolve()),
        "movable": True,
        "custom": True,
    }

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


def test_fine_tunes_and_exports_do_not_move(client, tmp_path):
    for key in ("fineTunes", "exports", "somewhere"):
        assert _move(client, key, str(tmp_path / key)).status_code == 400
    assert _location(client, "fineTunes")["movable"] is False


def test_only_the_installation_owner_can_move(client, tmp_path, monkeypatch):
    monkeypatch.setattr(library_routes.account_access, "managed_account", lambda: True)
    assert _move(client, "images", str(tmp_path / "elsewhere")).status_code == 403
    assert not (tmp_path / "elsewhere").exists()


def test_a_failed_move_puts_everything_back(client, tmp_path, monkeypatch):
    import shutil

    from core.inference import image_gallery

    old = image_gallery.gallery_dir()
    for name in ("a.png", "b.png", "c.png"):
        (old / name).write_bytes(name.encode())
    real_move = shutil.move
    calls = []

    def flaky_move(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return real_move(src, dst)

    monkeypatch.setattr(library.shutil, "move", flaky_move)
    response = _move(client, "images", str(tmp_path / "small-disk"))
    assert response.status_code == 500
    assert "No space left" in response.json()["detail"]
    monkeypatch.setattr(library.shutil, "move", real_move)
    assert sorted(p.name for p in old.iterdir()) == ["a.png", "b.png", "c.png"]
    assert image_gallery.gallery_dir() == old
    assert _location(client, "images")["custom"] is False
