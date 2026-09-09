# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A clip belongs to the account that reserved the job, from the moment the reservation is taken."""

from __future__ import annotations

import threading
from types import SimpleNamespace

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


class ReservingVideoBackend:
    """Reserves the job under its own lock, then resolves inputs, as the real backend does."""

    def __init__(self):
        self.cancelled: list[str] = []
        self.reserved = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self._job_account: str | None = None

    def status(self):
        return {
            "loaded": True,
            "repo_id": "public/video-model",
            "defaults": {"fps": 24, "num_frames": 49, "frame_step": 4, "frame_offset": 1},
        }

    def generate_job_account(self):
        return self._job_account

    def generation_snapshot(self):
        return self.status(), self

    def begin_generate(self, **kwargs):
        from utils.account_context import current_account_id

        self._job_account = current_account_id()
        self.reserved.set()
        assert self.release.wait(20)
        return {"queued": 1, "width": 512, "height": 512, "num_frames": 49, "fps": 24}

    def generate_progress(self):
        return {"active": True, "phase": "denoising", "step": 2, "total": 5, "video": CLIP}

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
    monkeypatch.setattr(gpu_arbiter, "current_owner", lambda: gpu_arbiter.VIDEO)
    monkeypatch.setattr(gpu_arbiter, "owner_account", lambda: ALICE.account_id)
    monkeypatch.setattr(account_access, "repo_is_public", lambda *args, **kwargs: True)
    yield


@pytest.fixture
def backend(monkeypatch):
    import core.inference.video as video_module

    fake = ReservingVideoBackend()
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
    return TestClient(app)


def test_ownership_holds_from_the_reservation_not_from_the_end_of_begin_generate(backend):
    generate = "/api/inference/video/generate"
    progress = "/api/inference/video/generate-progress"
    cancel = "/api/inference/video/generate/cancel"

    with _client(ALICE) as client:
        assert client.post(generate, json = {"prompt": "p", "steps": 5}).status_code == 200

    backend.reserved.clear()
    backend.release.clear()
    started = {}

    def run():
        with _client(BOB) as client:
            started["response"] = client.post(generate, json = {"prompt": "p", "steps": 5})

    thread = threading.Thread(target = run, daemon = True)
    thread.start()
    try:
        assert backend.reserved.wait(20)
        with _client(ALICE) as client:
            assert client.post(cancel).json() == {"cancelled": False}
            assert client.get(progress).json() == {"loaded": True, "yours": False}
        assert backend.cancelled == []
        with _client(BOB) as client:
            assert client.get(progress).json()["video"]["prompt"] == CLIP["prompt"]
    finally:
        backend.release.set()
        thread.join(20)
    assert started["response"].status_code == 200


def test_the_real_backend_records_the_account_inside_the_locked_reservation():
    import functools

    from core.inference.video import VideoBackend, _VideoResolvedInputs

    backend = VideoBackend()
    family = SimpleNamespace(
        name = "fam",
        default_fps = 24,
        default_num_frames = 49,
        frame_step = 4,
        frame_offset = 1,
    )
    state = SimpleNamespace(family = family, h3_task = None, engine = "diffusers", repo_id = "r")
    backend._state = state
    backend._resolve_keyframes = lambda *a, **k: (None, None, 512, 512, "t2v")
    backend._resolve_references = lambda *a, **k: None
    backend._resolve_flow_shifts = lambda *a, **k: (None, None)
    seen = {}

    def worker(**kwargs):
        seen["at_start"] = backend.generate_job_account()

    backend._run_generate = worker
    import core.inference.video as video_module

    original = video_module.validate_video_request_shape
    video_module.validate_video_request_shape = lambda *a, **k: None
    try:
        from utils.account_context import run_as
        run_as(BOB, functools.partial(backend.begin_generate, prompt = "p", steps = 5))
    finally:
        video_module.validate_video_request_shape = original
    assert seen["at_start"] == BOB.account_id
    assert backend.generate_job_account() == BOB.account_id


def test_cancel_rechecks_the_authorized_reservation_under_the_lock():
    """The route authorizes on the loop and cancels in a thread; a successor that reserved in between must not receive the stale cancel."""
    import threading

    from core.inference.video import VideoBackend

    backend = VideoBackend()
    event = threading.Event()
    backend._active_generate_cancel = event
    backend._generate_job_account = BOB.account_id
    assert backend.cancel_generate(expected_account = ALICE.account_id) is False
    assert not event.is_set()
    assert backend.cancel_generate(expected_account = BOB.account_id) is True
    assert event.is_set()
