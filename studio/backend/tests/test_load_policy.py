# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The load policy comes from the credential the auth layer admitted, over a real HTTP request."""

import secrets
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from auth import authentication, policy, storage
from core.inference.load_policy import LoadPolicy, current_load_policy

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield storage
    policy.invalidate_account_cache()


@pytest.fixture
def client(auth_db):
    app = FastAPI()

    @app.get("/policy")
    async def read_policy(_subject: str = Depends(authentication.get_current_subject)):
        return {"policy": current_load_policy().value}

    with TestClient(app) as test_client:
        yield test_client


def _policy(client, key):
    response = client.get("/policy", headers = {"Authorization": f"Bearer {key}"})
    assert response.status_code == 200, response.text
    return response.json()["policy"]


def test_the_chat_tools_key_may_only_load_beside(client):
    key, _row = storage.create_api_key(
        username = "unsloth", name = storage.STUDIO_CHAT_TOOLS_KEY_NAME, internal = True
    )
    assert _policy(client, key) == LoadPolicy.BESIDE_ONLY.value


@pytest.mark.parametrize(
    "name, internal",
    [
        ("cli", False),
        # A user's own key cannot opt into, or out of, the chat's policy by its name.
        (storage.STUDIO_CHAT_TOOLS_KEY_NAME, False),
        ("Data recipe", True),
    ],
)
def test_every_other_key_loads_as_before(client, name, internal):
    key, _row = storage.create_api_key(username = "unsloth", name = name, internal = internal)
    assert _policy(client, key) == LoadPolicy.ANY.value


def test_routes_and_media_switch_do_not_read_the_mcp_caller():
    adapters = ("routes/inference.py", "routes/video.py", "core/inference/media_auto_switch.py")
    for relative in adapters:
        source = (BACKEND / relative).read_text()
        assert "studio_mcp" not in source, relative
