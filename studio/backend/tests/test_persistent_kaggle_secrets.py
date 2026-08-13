# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persistent admin password + API key from env (Kaggle secrets)."""

import secrets

import pytest

from auth import storage
from auth.hashing import verify_password


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    monkeypatch.setattr(storage, "_credential_encryption_key_cache", None)
    monkeypatch.delenv("UNSLOTH_STUDIO_ADMIN_PASSWORD", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_AUTH_TOKEN", raising = False)
    yield


def test_ensure_default_admin_uses_persistent_password(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ADMIN_PASSWORD", "kaggle-admin-secret")
    assert storage.ensure_default_admin() is True
    row = storage.get_user_and_secret(storage.DEFAULT_ADMIN_USERNAME)
    assert row is not None
    salt, pwd_hash, _jwt, must_change = row
    assert must_change is False
    assert verify_password("kaggle-admin-secret", salt, pwd_hash)
    assert storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME) is False


def test_ensure_default_admin_reapplies_password_on_existing_user(monkeypatch):
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "old-password-xx",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = True,
    )
    monkeypatch.setenv("UNSLOTH_STUDIO_ADMIN_PASSWORD", "kaggle-admin-secret")
    assert storage.ensure_default_admin() is False
    row = storage.get_user_and_secret(storage.DEFAULT_ADMIN_USERNAME)
    salt, pwd_hash, _jwt, must_change = row
    assert must_change is False
    assert verify_password("kaggle-admin-secret", salt, pwd_hash)


def test_ensure_api_key_from_raw_is_idempotent():
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
    )
    raw = "sk-unsloth-" + "ab" * 16
    assert storage.ensure_api_key_from_raw(raw) is True
    assert storage.validate_api_key(raw) == storage.DEFAULT_ADMIN_USERNAME
    assert storage.ensure_api_key_from_raw(raw) is True
    assert storage.validate_api_key(raw) == storage.DEFAULT_ADMIN_USERNAME


def test_apply_persistent_api_key_from_env(monkeypatch):
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
    )
    raw = "sk-unsloth-" + "cd" * 16
    monkeypatch.setenv("UNSLOTH_STUDIO_AUTH_TOKEN", raw)
    assert storage.apply_persistent_api_key() is True
    assert storage.validate_api_key(raw) == storage.DEFAULT_ADMIN_USERNAME


def test_ensure_default_admin_seeds_api_key(monkeypatch):
    raw = "sk-unsloth-" + "ef" * 16
    monkeypatch.setenv("UNSLOTH_STUDIO_ADMIN_PASSWORD", "kaggle-admin-secret")
    monkeypatch.setenv("UNSLOTH_STUDIO_AUTH_TOKEN", raw)
    storage.ensure_default_admin()
    assert storage.validate_api_key(raw) == storage.DEFAULT_ADMIN_USERNAME


def test_persistent_api_key_rejects_non_unsloth_prefix(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_AUTH_TOKEN", "sk-other-not-valid")
    assert storage.persistent_api_key() is None
    assert storage.apply_persistent_api_key() is False
