# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import allow_ambient_hf_token, get_current_subject
from core.inference import gpu_arbiter
from hub.services.models import account_access as access
from routes import video
from state import active_generations
from utils.account_context import AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")
PUBLIC = {"org/public-video"}


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_resident_components", {}, raising = False)
    monkeypatch.setattr(access, "_generation_accounts", {})
    monkeypatch.setattr(access, "repo_is_public", lambda repo_id, *a, **k: repo_id in PUBLIC)
    monkeypatch.setattr(gpu_arbiter, "_owner", "video")
    monkeypatch.setattr(gpu_arbiter, "_owner_account", BOB.account_id)
    active_generations.reset_for_tests()
    yield
    active_generations.reset_for_tests()


def client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[allow_ambient_hf_token] = lambda: False
    app.include_router(video.openai_router, prefix = "/v1")
    return TestClient(app)


def _multipart(fields):
    boundary = "----unsloth-test-boundary"
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def test_openai_video_resident_replaced_after_authorization(monkeypatch):
    from core.inference import video as video_module
    from core.inference.video_families import VIDEO_MODEL_CHANGED_MSG

    resident = SimpleNamespace(repo_id = "org/public-video", family = "wan")
    replacement = SimpleNamespace(repo_id = "bob/private-video", family = "wan")
    box = {"state": resident, "reads": 0}
    reserved = []

    def _status_of(state):
        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "family": state.family,
            "base_repo": None,
            "defaults": {
                "num_frames": 17,
                "fps": 16,
                "frame_step": 4,
                "frame_offset": 1,
                "resolution_presets": [[320, 320]],
            },
        }

    def _advance():
        state = box["state"]
        box["reads"] += 1
        if box["reads"] == 1:
            box["state"] = replacement
        return state

    def status():
        return _status_of(_advance())

    def generation_snapshot():
        state = _advance()
        return _status_of(state), state

    def begin_generate(**kwargs):
        expected = kwargs.get("expected_state")
        if expected is not None and expected is not box["state"]:
            raise RuntimeError(VIDEO_MODEL_CHANGED_MSG)
        reserved.append(box["state"].repo_id)
        return {"width": 320, "height": 320, "num_frames": 17, "fps": 16}

    backend = SimpleNamespace(
        status = status,
        generation_snapshot = generation_snapshot,
        begin_generate = begin_generate,
        generate_progress = lambda: {"active": False},
    )
    monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)
    body, content_type = _multipart({"prompt": "a sloth", "size": "320x320"})
    with client_for(ALICE) as client:
        response = client.post("/v1/videos", content = body, headers = {"Content-Type": content_type})
    print("STATUS", response.status_code, response.json())
    print("RESERVED", reserved)
    assert response.status_code == 404, response.json()
    assert reserved == []
