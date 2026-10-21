# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import secrets

import pytest
from fastapi.testclient import TestClient

from auth import policy, storage
from utils.account_context import OWNER, bind_account, reset_account

from .support import make_app


@pytest.fixture
def isolated_auth(tmp_path, monkeypatch):
    home = tmp_path / "install"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path / "Documents"))
    monkeypatch.delenv("UNSLOTH_STUDIO_PROJECTS_HOME", raising = False)
    monkeypatch.setattr(storage, "DB_PATH", home / "auth" / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", home / "auth" / ".bootstrap_password")
    monkeypatch.setattr(storage, "_credential_encryption_key_cache", None)
    policy.invalidate_account_cache()
    token = bind_account(OWNER)
    try:
        yield storage
    finally:
        reset_account(token)
        policy.invalidate_account_cache()


@pytest.fixture
def accounts(isolated_auth):
    for username in ("unsloth", "alice", "bob"):
        isolated_auth.create_initial_user(username, "account-password", secrets.token_urlsafe(32))
    return {name: isolated_auth.get_account(name) for name in ("unsloth", "alice", "bob")}


@pytest.fixture
def account_client(isolated_auth):
    with TestClient(make_app()) as client:
        yield client
