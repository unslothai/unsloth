# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An idle cancel must not stop a render another account starts in the gap.

Before the first generation of the process the backend holds no reservation
(``generate_job_account()`` is None, and it is never cleared afterwards), so the
cancel route takes its no-reservation branch and calls ``cancel_generate()`` with
no expectation at all. ``begin_generate`` runs on a worker thread, so another
account can reserve between the route's read and the executor call, and an
unexpecting cancel sets whatever cancel event is current -- the foreign render's.
"""

from __future__ import annotations

import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy, storage as auth_storage
from auth.authentication import get_current_subject
from core.inference import gpu_arbiter
from core.inference.video import VideoBackend
from hub.services.models import account_access
from routes import video
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


def _arm(backend: VideoBackend, account: AccountContext) -> threading.Event:
    """What begin_generate commits under the lock: the account plus the cancel handle."""
    event = threading.Event()
    backend._generate_job_account = account.account_id
    backend._active_generate_cancel = event
    return event


def _reserve_after_first_read(backend: VideoBackend, account: AccountContext) -> threading.Event:
    """Return the real reservation now (None), then let ``account`` reserve."""
    event = threading.Event()
    original = backend.generate_job_account
    fired = {"done": False}

    def reading():
        current = original()
        if not fired["done"]:
            fired["done"] = True
            backend._generate_job_account = account.account_id
            backend._active_generate_cancel = event
        return current

    backend.generate_job_account = reading
    return event


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
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


def test_idle_cancel_does_not_stop_a_render_another_account_starts_in_the_gap(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    backend = VideoBackend()
    _install(monkeypatch, backend)
    bob_event = _reserve_after_first_read(backend, BOB)

    with _client(ALICE) as client:
        body = client.post("/api/inference/video/generate/cancel").json()

    assert backend._generate_job_account == BOB.account_id
    assert not bob_event.is_set(), "alice's idle cancel stopped bob's render"
    assert body == {"cancelled": False}


def test_idle_cancel_still_stops_the_callers_own_render_started_in_the_gap(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    backend = VideoBackend()
    _install(monkeypatch, backend)
    alice_event = _reserve_after_first_read(backend, ALICE)

    with _client(ALICE) as client:
        body = client.post("/api/inference/video/generate/cancel").json()

    assert body == {"cancelled": True}
    assert alice_event.is_set()


def test_single_user_cancel_still_works_and_is_a_no_op_when_idle(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    monkeypatch.setattr(gpu_arbiter, "owner_account", lambda: OWNER.account_id)
    backend = VideoBackend()
    _install(monkeypatch, backend)

    with _client(OWNER) as client:
        assert client.post("/api/inference/video/generate/cancel").json() == {"cancelled": False}
        event = _arm(backend, OWNER)
        assert client.post("/api/inference/video/generate/cancel").json() == {"cancelled": True}
    assert event.is_set()
