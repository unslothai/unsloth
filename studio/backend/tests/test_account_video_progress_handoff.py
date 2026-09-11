# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video generate-progress must not hand one account the successor's job.

The route reads the reservation for its visibility check and then reads the progress.
``begin_generate`` runs in a worker thread (asyncio.to_thread), so another account can take the
reservation between those two reads with no await in between: ALICE's clip finished, BOB reserved,
and ALICE's poll then returns BOB's fresh progress instead of ALICE's terminal record. The Video
page merges that terminal record to show the saved clip, so the handoff both leaks BOB's job and
makes ALICE's completed clip vanish until the gallery is refreshed.
"""

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

ALICE_CLIP = {
    "id": "alice-clip",
    "url": "/api/inference/video/gallery/alice-clip/file",
    "prompt": "alice's private prompt",
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

ALICE_TERMINAL = {
    "active": False,
    "phase": "completed",
    "step": 5,
    "total": 5,
    "video": ALICE_CLIP,
}
BOB_FRESH = {
    "active": True,
    "phase": "queued",
    "step": 0,
    "total": 0,
    "eta_seconds": None,
    "video_id": "bob-clip",
}


class HandoffVideoBackend:
    """ALICE owns the reservation when the route authorizes; BOB reserves right after that read.

    The handoff fires once, immediately after the FIRST read of the reservation, which is the
    earliest interleaving a worker-thread ``begin_generate`` can produce. ``generate_progress``
    mirrors the real backend: the reservation and the progress come out of ONE lock, so an
    ``expected_account`` the caller authorized is rechecked against the owner held right now.
    """

    def __init__(self, hand_off: bool = True):
        self._hand_off = hand_off
        self.job_account = ALICE.account_id
        self.progress = dict(ALICE_TERMINAL)

    def status(self):
        return {"loaded": True, "repo_id": "public/video-model"}

    def generate_job_account(self):
        current = self.job_account
        if self._hand_off:
            self._hand_off = False
            self.job_account = BOB.account_id
            self.progress = dict(BOB_FRESH)
        return current

    def generate_progress(self, expected_account = None):
        if expected_account is not None and self.job_account != expected_account:
            return None
        return dict(self.progress)


class SoloVideoBackend:
    """A single-user backend with no reservation at all: the old path must be untouched."""

    def __init__(self):
        self.progress = dict(ALICE_TERMINAL)

    def status(self):
        return {"loaded": True, "repo_id": "public/video-model"}

    def generate_progress(self):
        return dict(self.progress)


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


def _install(monkeypatch, backend):
    import core.inference.video as video_module
    monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)


PROGRESS = "/api/inference/video/generate-progress"


def test_progress_does_not_follow_a_reservation_that_changed_hands(monkeypatch):
    """BOB reserves after ALICE's poll passed authorization; ALICE must not get BOB's job."""
    backend = HandoffVideoBackend(hand_off = True)
    _install(monkeypatch, backend)

    with _client(ALICE) as client:
        body = client.get(PROGRESS).json()

    assert backend.job_account == BOB.account_id
    assert body.get("video_id") != "bob-clip", "alice received bob's job progress"
    assert body == {"loaded": True, "yours": False}


def test_progress_still_returns_the_owners_own_terminal_record(monkeypatch):
    """No handoff: ALICE keeps seeing the completed clip the Video page merges."""
    backend = HandoffVideoBackend(hand_off = False)
    _install(monkeypatch, backend)

    with _client(ALICE) as client:
        body = client.get(PROGRESS).json()

    assert body["phase"] == "completed"
    assert body["video"]["id"] == ALICE_CLIP["id"]


def test_single_user_backend_without_a_reservation_is_unchanged(monkeypatch):
    backend = SoloVideoBackend()
    _install(monkeypatch, backend)

    with _client(ALICE) as client:
        body = client.get(PROGRESS).json()

    assert body["phase"] == "completed"
    assert body["video"]["id"] == ALICE_CLIP["id"]


def test_real_backend_returns_none_when_the_reservation_moved():
    """The recheck lives under the same lock as the progress read."""
    from core.inference.video import VideoBackend

    backend = VideoBackend()
    backend._gen = dict(ALICE_TERMINAL)
    backend._generate_job_account = BOB.account_id
    assert backend.generate_progress(expected_account = ALICE.account_id) is None
    assert backend.generate_progress(expected_account = BOB.account_id)["phase"] == "completed"
    assert backend.generate_progress()["phase"] == "completed"
