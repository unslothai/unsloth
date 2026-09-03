# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deep Research inference must not be attributed to a third-party API caller.

The supervisor reaches the local chat-completions endpoint with a minted sk-unsloth key,
so without the internal-key check every research step opened the API monitor overlay.
"""

from __future__ import annotations

import pytest

from auth import storage as auth_storage
from routes.inference import _request_used_api_key


class _Request:
    def __init__(self, authorization: str | None):
        self.headers = {} if authorization is None else {"authorization": authorization}


@pytest.fixture(autouse = True)
def auth_home(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(auth_storage, "_bootstrap_password", None)
    monkeypatch.setattr(auth_storage, "_api_key_pbkdf2_salt_cache", None)
    auth_storage._reset_api_key_hash_cache()
    auth_storage.create_initial_user(
        username = "researcher",
        password = "human-password-123",
        jwt_secret = "test-secret",
    )
    yield tmp_path
    auth_storage._reset_api_key_hash_cache()


def test_internal_key_is_not_reported_as_api_traffic():
    raw_key, _row = auth_storage.create_api_key(
        username = "researcher",
        name = "deep-research workflow",
        internal = True,
    )
    assert auth_storage.is_internal_api_key(raw_key) is True
    assert _request_used_api_key(_Request(f"Bearer {raw_key}")) is False


def test_user_key_is_still_reported_as_api_traffic():
    raw_key, _row = auth_storage.create_api_key(username = "researcher", name = "my key")
    assert auth_storage.is_internal_api_key(raw_key) is False
    assert _request_used_api_key(_Request(f"Bearer {raw_key}")) is True


def test_session_jwt_and_missing_header_are_not_api_traffic():
    assert _request_used_api_key(_Request("Bearer eyJhbGciOiJIUzI1NiJ9.body.sig")) is False
    assert _request_used_api_key(_Request(None)) is False


def test_unknown_key_is_treated_as_third_party():
    # An unrecognised key cannot be Studio's own, so it must keep its monitor attribution.
    assert auth_storage.is_internal_api_key("sk-unsloth-deadbeefdeadbeef") is False
    assert _request_used_api_key(_Request("Bearer sk-unsloth-deadbeefdeadbeef")) is True


def test_probe_failure_keeps_the_row_attributed(monkeypatch):
    def explode(_raw_key):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(auth_storage, "is_internal_api_key", explode)
    assert _request_used_api_key(_Request("Bearer sk-unsloth-deadbeefdeadbeef")) is True
