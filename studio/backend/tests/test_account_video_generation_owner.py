# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A background clip belongs to the account that started it, not to the model's loader."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy, storage as auth_storage
from auth.authentication import get_current_subject
from core.inference import gpu_arbiter
from hub.services.models import account_access
from routes import video
from utils.account_context import AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")

CLIP = {
    "id": "clip-1",
    "url": "/api/inference/video/gallery/clip-1/file",
    "prompt": "bob's private prompt",
    "width": 512,
    "height": 512,
    "num_frames": 49,
    "fps": 24,
    "duration_s": 2.0,
    "steps": 5,
    "guidance": 1.0,
    "seed": 1,
    "created_at": "2026-08-06T00:00:00Z",
}


class FakeVideoBackend:
    def __init__(self):
        self.started: list[str] = []
        self.cancelled: list[str] = []

    def status(self):
        return {
            "loaded": True,
            "repo_id": "public/video-model",
            "defaults": {"fps": 24, "num_frames": 49, "frame_step": 4, "frame_offset": 1},
        }

    def generation_snapshot(self):
        return self.status(), self

    def begin_generate(self, **kwargs):
        from utils.account_context import current_account
        self.started.append(current_account().username)
        return {"queued": 1, "width": 512, "height": 512, "num_frames": 49, "fps": 24}

    def generate_progress(self):
        return {"active": False, "phase": "completed", "step": 5, "total": 5, "video": CLIP}

    def cancel_generate(self, *args, **kwargs):
        from utils.account_context import current_account
        self.cancelled.append(current_account().username)
        return True


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(auth_storage, "_bootstrap_password", None)
    monkeypatch.setattr(video, "_generation_account", None, raising = False)
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
    monkeypatch.setattr(gpu_arbiter, "current_owner", lambda: gpu_arbiter.VIDEO)
    monkeypatch.setattr(gpu_arbiter, "owner_account", lambda: ALICE.account_id)
    monkeypatch.setattr(account_access, "repo_is_public", lambda *args, **kwargs: True)
    yield
    policy.invalidate_account_cache()


@pytest.fixture
def backend(monkeypatch):
    import core.inference.video as video_module

    fake = FakeVideoBackend()
    monkeypatch.setattr(video_module, "get_video_backend", lambda: fake)
    return fake


def _client(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.include_router(video.router, prefix = "/api/inference")
    app.include_router(video.openai_router, prefix = "/v1")
    return TestClient(app)


def test_the_starting_account_owns_its_clip_and_the_models_loader_does_not(backend):
    generate = "/api/inference/video/generate"
    progress = "/api/inference/video/generate-progress"
    cancel = "/api/inference/video/generate/cancel"

    with _client(BOB) as client:
        assert client.post(generate, json = {"prompt": "p", "steps": 5}).status_code == 200
        assert backend.started == ["bob"]
        assert client.get(progress).json()["video"]["prompt"] == CLIP["prompt"]

    with _client(ALICE) as client:
        assert client.get(progress).json() == {"loaded": True, "yours": False}
        assert client.post(cancel).json() == {"cancelled": False}
    assert backend.cancelled == []

    with _client(BOB) as client:
        assert client.post(cancel).json() == {"cancelled": True}
    assert backend.cancelled == ["bob"]


def test_a_clip_started_on_the_openai_route_belongs_to_that_account_too(backend):
    progress = "/api/inference/video/generate-progress"
    cancel = "/api/inference/video/generate/cancel"

    with _client(BOB) as client:
        created = client.post(
            "/v1/videos",
            json = {
                "prompt": "p",
                "model": "public/video-model",
                "seconds": "2",
                "size": "512x512",
            },
        )
        assert created.status_code == 200, created.text
        assert backend.started == ["bob"]
        assert client.get(progress).json()["video"]["prompt"] == CLIP["prompt"]

    with _client(ALICE) as client:
        assert client.get(progress).json() == {"loaded": True, "yours": False}
        assert client.post(cancel).json() == {"cancelled": False}
    assert backend.cancelled == []


def test_the_installation_owner_does_not_see_a_managed_accounts_clip(backend):
    """The owner administers the machine, which does not extend to another account's prompt."""
    from utils.account_context import OWNER

    generate = "/api/inference/video/generate"
    progress = "/api/inference/video/generate-progress"
    cancel = "/api/inference/video/generate/cancel"

    with _client(BOB) as client:
        assert client.post(generate, json = {"prompt": "p", "steps": 5}).status_code == 200
        assert client.get(progress).json()["video"]["prompt"] == CLIP["prompt"]

    with _client(OWNER) as client:
        assert client.get(progress).json() == {"loaded": True, "yours": False}
        assert client.post(cancel).json() == {"cancelled": False}
    assert backend.cancelled == []
