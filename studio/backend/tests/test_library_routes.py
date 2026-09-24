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


def test_favorites_lists_only_favorite_ids(client):
    client.patch("/api/library/items", json = {"id": "image:abc", "favorite": True})
    client.patch("/api/library/items", json = {"id": "image:def", "favorite": False})
    assert client.get("/api/library/favorites").json() == {"ids": ["image:abc"]}


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
    assert f"video:{record['id']}" in _items(client)[0]


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
