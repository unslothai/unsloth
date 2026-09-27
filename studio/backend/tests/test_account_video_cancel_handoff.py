# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cancel authorized against one account's job must not cancel the successor's render.

The cancel route reads the backend's reservation several times (the visibility check, the
foreign-work check, and the expected_account it hands back). ``begin_generate`` runs in a worker
thread (asyncio.to_thread), so another account can take the reservation between two of those reads
with no await in between. The backend then rechecks against the account it was GIVEN, so a stale
read makes the recheck a no-op and the requester cancels a render it never owned.
"""

from __future__ import annotations

import threading

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


class HandoffVideoBackend:
    """ALICE's job is current when the route authorizes; BOB reserves right after that read.

    The handoff fires once, immediately after the FIRST read of the reservation, which is the
    earliest interleaving a worker-thread begin_generate can produce: ALICE's clip finished and
    BOB's reservation committed while the cancel request was still on the loop.

    ``cancel_generate`` is the real backend's recheck (core/inference/video.py): it compares
    ``expected_account`` with the reservation it holds NOW and sets the cancel event on a match.
    """

    def __init__(self, hand_off: bool = True):
        self._hand_off = hand_off
        self.job_account = ALICE.account_id
        self.cancel_event = threading.Event()

    def status(self):
        return {"loaded": True, "repo_id": "public/video-model"}

    def generate_job_account(self):
        current = self.job_account
        if self._hand_off:
            self._hand_off = False
            self.job_account = BOB.account_id
        return current

    def cancel_generate(
        self,
        expected_video_id = None,
        expected_account = None,
    ):
        if expected_account is not None and self.job_account != expected_account:
            return False
        self.cancel_event.set()
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


def test_cancel_does_not_follow_a_reservation_that_changed_hands(monkeypatch):
    """BOB reserves after ALICE's cancel passed authorization; BOB's render must survive."""
    backend = HandoffVideoBackend(hand_off = True)
    _install(monkeypatch, backend)

    with _client(ALICE) as client:
        body = client.post("/api/inference/video/generate/cancel").json()

    assert backend.job_account == BOB.account_id
    assert not backend.cancel_event.is_set(), "alice cancelled bob's render"
    assert body == {"cancelled": False}


def test_cancel_still_stops_the_requesting_accounts_own_job(monkeypatch):
    """No handoff: the ordinary cancel keeps working."""
    backend = HandoffVideoBackend(hand_off = False)
    _install(monkeypatch, backend)

    with _client(ALICE) as client:
        body = client.post("/api/inference/video/generate/cancel").json()

    assert body == {"cancelled": True}
    assert backend.cancel_event.is_set()
