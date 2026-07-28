# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A password rotation must not leave a session minted from the replaced credential.

`unsloth studio reset-password` rotates in place against a live server, so a login
can verify the old password, have the rotation land, and only then mint its tokens.
Issuance is bound to the credential version that was verified, so such a login gets
tokens that are already dead rather than a session that outlives the reset.
"""

import secrets
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from auth import storage
from auth.authentication import ALGORITHM, create_access_token, create_refresh_token


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    yield


@pytest.fixture
def admin():
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "old-password-123",
        jwt_secret = secrets.token_urlsafe(64),
    )
    return storage.DEFAULT_ADMIN_USERNAME


def _verified_secret(username):
    return storage.get_user_and_secret(username)[2]


def test_access_token_from_the_replaced_credential_is_rejected(admin):
    secret = _verified_secret(admin)

    storage.update_password(admin, "new-password-456", revoke_refresh_tokens = True)
    token = create_access_token(subject = admin, secret = secret)

    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode(token, storage.get_jwt_secret(admin), algorithms = [ALGORITHM])


def test_refresh_token_from_the_replaced_credential_is_rejected(admin):
    secret = _verified_secret(admin)

    # Inserted AFTER the rotation's DELETE, so revocation alone cannot catch it.
    storage.update_password(admin, "new-password-456", revoke_refresh_tokens = True)
    token = create_refresh_token(subject = admin, secret = secret)

    assert storage.verify_refresh_token(token) is None
    assert storage.consume_refresh_token(token) is None


def test_a_rejected_refresh_token_is_dropped(admin):
    secret = _verified_secret(admin)
    storage.update_password(admin, "new-password-456", revoke_refresh_tokens = True)
    token = create_refresh_token(subject = admin, secret = secret)

    storage.verify_refresh_token(token)

    conn = storage.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) AS c FROM refresh_tokens").fetchone()["c"] == 0
    finally:
        conn.close()


def test_tokens_from_the_current_credential_still_work(admin):
    secret = _verified_secret(admin)

    access = create_access_token(subject = admin, secret = secret)
    refresh = create_refresh_token(subject = admin, secret = secret)

    jwt.decode(access, storage.get_jwt_secret(admin), algorithms = [ALGORITHM])
    assert storage.verify_refresh_token(refresh) == (admin, False)


def test_unstamped_legacy_tokens_still_verify(admin):
    # Rows written before the secret_gen column existed must not log users out.
    token = secrets.token_urlsafe(48)
    expires_at = (datetime.now(timezone.utc) + timedelta(days = 7)).isoformat()
    storage.save_refresh_token(token, admin, expires_at, secret_gen = None)
    conn = storage.get_connection()
    try:
        conn.execute("UPDATE refresh_tokens SET secret_gen = NULL")
        conn.commit()
    finally:
        conn.close()

    assert storage.verify_refresh_token(token) == (admin, False)
