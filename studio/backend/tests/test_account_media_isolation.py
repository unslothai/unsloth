# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account boundaries exercised through the real media object routes and stores."""

from __future__ import annotations


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from auth import policy, storage as auth_storage
from auth.authentication import get_current_subject
from core.inference import audio_gallery, image_gallery, search_images, video_gallery
from routes import inference, video
from utils.account_context import OWNER, AccountContext, bind_account, reset_account, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(search_images, "_account_states", {})
    monkeypatch.setattr(video, "_managed_jobs", {})
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(auth_storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    connection = auth_storage.get_connection()
    with connection:
        for account in (ALICE, BOB):
            connection.execute(
                "INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret,"
                " account_id, role, is_active) VALUES (?, 'salt', 'hash', 'secret', ?, 'user', 1)",
                (account.username, account.account_id),
            )
    connection.close()
    yield
    policy.invalidate_account_cache()


def _save(kind):
    meta = {
        "prompt": "private prompt",
        "model": "private/model",
        "created_at": 100.0,
        "width": 512,
        "height": 512,
        "steps": 5,
        "guidance": 1.0,
        "seed": 1,
        "num_frames": 49,
        "fps": 24,
        "duration_s": 2.0,
    }
    if kind == "images":
        return image_gallery.save(Image.new("RGB", (8, 8)), meta)
    if kind == "audio":
        meta.update(audio_type = "snac", sample_rate = 24000, created_at = "2026-08-06T00:00:00Z")
        return audio_gallery.save(b"RIFF\x24\x00\x00\x00WAVEfmt ", meta)
    meta["created_at"] = "2026-08-06T00:00:00Z"
    return video_gallery.save(b"\x00\x00\x00\x18ftypmp42", meta)


def _client(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.include_router(video.router, prefix = "/api/inference")
    app.include_router(video.openai_router, prefix = "/v1")
    return TestClient(app)


@pytest.mark.parametrize("kind", ["images", "audio", "video"])
@pytest.mark.parametrize("account,other", [(ALICE, BOB), (BOB, ALICE)])
def test_gallery_object_routes_do_not_resolve_another_accounts_ids(kind, account, other):
    record = run_as(account, _save, kind)
    root = f"/api/inference/{kind}/gallery"
    with _client(other) as client:
        assert client.get(f"{root}/{record['id']}/file").status_code == 404
        assert client.patch(f"{root}/{record['id']}", json = {"starred": True}).status_code == 404
        assert client.delete(f"{root}/{record['id']}").status_code == 404
        assert record["id"] not in client.get(root).text
        if kind == "video":
            assert client.get(f"{root}/{record['id']}/signed-url").status_code == 404
            assert client.get(f"{root}/{record['id']}/export").status_code == 404
    with _client(account) as client:
        assert client.get(f"{root}/{record['id']}/file").status_code == 200
        assert record["id"] in client.get(root).text


@pytest.mark.parametrize(
    "kind,module", [("images", image_gallery), ("audio", audio_gallery), ("video", video_gallery)]
)
def test_clear_and_paths_are_account_scoped(kind, module, tmp_path):
    a = run_as(ALICE, _save, kind)
    b = run_as(BOB, _save, kind)
    assert run_as(ALICE, module.clear) == 1
    path_fn = {
        "images": image_gallery.owned_image_path,
        "audio": audio_gallery.owned_audio_path,
        "video": video_gallery.owned_video_path,
    }[kind]
    assert run_as(ALICE, path_fn, a["id"]) is None
    assert run_as(BOB, path_fn, b["id"]).is_file()
    folder = "videos" if kind == "video" else kind
    assert run_as(OWNER, module.gallery_dir) == tmp_path / folder
    assert run_as(BOB, module.gallery_dir) == tmp_path / "accounts" / BOB.account_id / folder


@pytest.mark.parametrize("kind,module", [("images", inference), ("video", video)])
def test_signed_links_authorize_only_the_original_account_and_media(kind, module):
    record = run_as(ALICE, _save, kind)
    sign = module._sign_image_id if kind == "images" else module._sign_video_id
    token = run_as(ALICE, sign, record["id"])
    root = f"/api/inference/{kind}/gallery"
    with _client(BOB) as client:
        assert (
            client.get(f"{root}/{record['id']}/file-signed", params = {"token": token}).status_code
            == 200
        )
        assert client.get(f"{root}/guessed/file-signed", params = {"token": token}).status_code == 401
        tampered = token.replace(ALICE.account_id, BOB.account_id)
        assert (
            client.get(f"{root}/{record['id']}/file-signed", params = {"token": tampered}).status_code
            == 401
        )
    verify = (
        module._verify_image_link_token if kind == "images" else module._verify_video_link_token
    )
    owner_token = run_as(OWNER, sign, record["id"])
    assert owner_token.split(".")[0] == record["id"]
    assert verify(owner_token) == record["id"]


def test_search_ids_and_clear_fences_are_private(monkeypatch):
    raw = [{"thumbnail": "https://example.com/image.jpg", "url": "https://example.com/page"}]
    monkeypatch.setattr(search_images, "_fetch_thumbnail_bytes", lambda *args: b"thumbnail")
    image_id = run_as(ALICE, search_images.register_images, raw)[0]["id"]
    assert run_as(BOB, search_images.lookup_image, image_id) is None
    assert run_as(BOB, search_images.thumbnail_bytes, image_id) is None
    before = run_as(ALICE, search_images.cache_generation)
    run_as(BOB, search_images.snapshot_and_fence_registrations)
    run_as(BOB, search_images.clear_cache)
    assert run_as(ALICE, search_images.cache_generation) == before
    assert run_as(ALICE, search_images.thumbnail_bytes, image_id) == b"thumbnail"
    with _client(BOB) as client:
        assert client.get(f"/api/inference/search-images/{image_id}").status_code == 404
    run_as(ALICE, search_images.clear_cache)
    assert run_as(ALICE, search_images.thumbnail_bytes, image_id) is None


def test_openai_video_jobs_are_private_in_memory_and_after_rehydration():
    job = video._VideoJob(
        id = "private-job",
        created_at = 100,
        prompt = "private",
        model = "private/model",
        size = "512x512",
        seconds = "2",
        status = "failed",
    )
    run_as(ALICE, video._remember_job, job)
    assert run_as(BOB, video._lookup_video, job.id) is None
    assert run_as(BOB, video._all_videos) == []
    assert run_as(ALICE, video._lookup_video, job.id).id == job.id
    video._managed_jobs.clear()
    assert run_as(BOB, video._lookup_video, job.id) is None
    assert run_as(ALICE, video._lookup_video, job.id).id == job.id
    with _client(BOB) as client:
        assert client.get(f"/v1/videos/{job.id}").status_code == 404
        assert client.get(f"/v1/videos/{job.id}/content").status_code == 404
        assert client.delete(f"/v1/videos/{job.id}").status_code == 404


def test_clearing_a_managed_chat_history_reaps_only_its_own_thumbnails(monkeypatch):
    """The clear reaps every account's private thumbnails, not just the owner's."""
    from routes import chat_history

    monkeypatch.setattr(search_images, "_fetch_thumbnail_bytes", lambda *args: b"thumbnail")
    raw = [{"thumbnail": "https://example.com/i.jpg", "url": "https://example.com/p"}]
    ids = {
        account: run_as(account, search_images.register_images, raw)[0]["id"]
        for account in (ALICE, BOB)
    }
    snapshot = run_as(ALICE, chat_history._snapshot_chat_images)
    assert snapshot == {ids[ALICE]}
    run_as(ALICE, search_images.clear_cache, snapshot)
    assert run_as(ALICE, search_images.thumbnail_bytes, ids[ALICE]) is None
    assert run_as(BOB, search_images.thumbnail_bytes, ids[BOB]) == b"thumbnail"


def test_a_failed_replacement_load_leaves_residency_with_the_resident_model(monkeypatch):
    import threading

    from core.inference import gpu_arbiter, video as video_module

    backend = video_module.get_video_backend()
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", None)
    monkeypatch.setattr(gpu_arbiter, "_prior_account", None, raising = False)
    resident = {**backend.status(), "loaded": True, "repo_id": "alice/private-video"}
    monkeypatch.setattr(backend, "status", lambda: resident)
    monkeypatch.setattr(backend, "_state", object())
    unloaded: list[str] = []
    monkeypatch.setattr(backend, "unload", lambda: unloaded.append("unload") or resident)
    run_as(ALICE, gpu_arbiter.acquire_for, gpu_arbiter.VIDEO, lambda: None)
    run_as(BOB, gpu_arbiter.acquire_for, gpu_arbiter.VIDEO, lambda: None)
    monkeypatch.setattr(
        video_module,
        "_detect_load_family",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    run_as(
        BOB,
        backend._run_load,
        repo_id = "bob/model",
        _load_token = backend._load_token,
        _cancel_event = threading.Event(),
    )
    assert gpu_arbiter.owner_account() == ALICE.account_id
    with _client(BOB) as client:
        assert client.get("/api/inference/video/status").json() == {"loaded": True, "yours": False}
        assert client.post("/api/inference/video/unload").status_code == 404
    assert unloaded == []
    with _client(ALICE) as client:
        assert client.get("/api/inference/video/status").json()["repo_id"] == "alice/private-video"
