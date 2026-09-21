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

from auth import hashing, storage
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


def test_refresh_cannot_outlive_a_rotation_it_raced(admin):
    # /refresh consumes, then mints. A rotation landing in between must not let
    # the replacement pair be signed with the credential that just replaced it.
    secret = _verified_secret(admin)
    token = create_refresh_token(subject = admin, secret = secret)
    consumed = storage.consume_refresh_token(token)
    assert consumed is not None
    _username, _is_desktop, consumed_secret = consumed

    storage.update_password(admin, "new-password-456", revoke_refresh_tokens = True)
    access = create_access_token(subject = admin, secret = consumed_secret)
    refresh = create_refresh_token(subject = admin, secret = consumed_secret)

    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode(access, storage.get_jwt_secret(admin), algorithms = [ALGORITHM])
    assert storage.verify_refresh_token(refresh) is None


def test_desktop_login_cannot_outlive_a_rotation_it_raced(admin):
    # The reset deletes the desktop secret, so a desktop-login that validated it
    # just beforehand must not mint a session that survives.
    raw = storage.create_desktop_secret()
    verified = storage.validate_desktop_secret_with_credential(raw)
    assert verified is not None
    _username, verified_secret = verified

    storage.update_password(admin, "new-password-456", revoke_refresh_tokens = True)
    access = create_access_token(subject = admin, desktop = True, secret = verified_secret)
    refresh = create_refresh_token(subject = admin, desktop = True, secret = verified_secret)

    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode(access, storage.get_jwt_secret(admin), algorithms = [ALGORITHM])
    assert storage.verify_refresh_token(refresh) is None


def test_change_password_cannot_overwrite_a_rotation_it_raced(admin):
    # A change-password that verified the old hash must not clobber a reset that
    # committed while it was in flight.
    _salt, verified_hash, _secret, _must_change = storage.get_user_and_secret(admin)

    storage.update_password(admin, "reset-by-the-cli-789", revoke_refresh_tokens = True)

    assert not storage.update_password(
        admin,
        "attacker-chosen-000",
        revoke_refresh_tokens = True,
        expect_password_hash = verified_hash,
    )
    salt, pwd_hash, _s, _m = storage.get_user_and_secret(admin)
    assert hashing.verify_password("reset-by-the-cli-789", salt, pwd_hash)


def test_api_key_creation_from_a_revoked_credential_is_refused(admin):
    generation = storage.credential_generation(_verified_secret(admin))

    storage.update_password(admin, "new-password-456", revoke_refresh_tokens = True)

    with pytest.raises(storage.CredentialRotated):
        storage.create_api_key(username = admin, name = "k", expect_gen = generation)
    conn = storage.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) AS c FROM api_keys").fetchone()["c"] == 0
    finally:
        conn.close()


def test_api_key_creation_under_the_current_credential_still_works(admin):
    generation = storage.credential_generation(_verified_secret(admin))

    raw_key, _row = storage.create_api_key(username = admin, name = "k", expect_gen = generation)

    assert storage.validate_api_key(raw_key) == admin


def test_change_password_tokens_are_bound_to_its_own_write(admin):
    # The tokens returned to a successful change-password must be signed with the
    # secret that write produced, not whatever a later reset put in the DB.
    _salt, verified_hash, _secret, _must = storage.get_user_and_secret(admin)
    new_secret = storage.update_password(
        admin,
        "chosen-by-the-user",
        revoke_refresh_tokens = True,
        expect_password_hash = verified_hash,
    )
    assert new_secret is not None

    storage.update_password(admin, "reset-by-the-cli-789", revoke_refresh_tokens = True)
    access = create_access_token(subject = admin, secret = new_secret)
    refresh = create_refresh_token(subject = admin, secret = new_secret)

    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode(access, storage.get_jwt_secret(admin), algorithms = [ALGORITHM])
    assert storage.verify_refresh_token(refresh) is None


def test_internal_api_key_minting_honours_the_request_generation(admin):
    generation = storage.credential_generation(_verified_secret(admin))
    storage.update_password(admin, "new-password-456", revoke_refresh_tokens = True)

    with pytest.raises(storage.CredentialRotated):
        storage.create_api_key(
            username = admin,
            name = "data-recipe workflow",
            internal = True,
            expect_gen = generation,
        )


def test_api_key_auth_reports_the_version_the_key_was_valid_under(admin):
    # The generation must come from the same transaction as the key check, or a
    # revoked key could hand a route the post-reset generation and mint again.
    raw, _row = storage.create_api_key(username = admin, name = "agent")
    verified = storage.validate_api_key_with_credential(raw)
    assert verified is not None
    _user, secret = verified
    generation = storage.credential_generation(secret)

    storage.update_password(admin, "new-password-456", revoke_refresh_tokens = True)
    conn = storage.get_connection()
    try:
        conn.execute("DELETE FROM api_keys")
        conn.commit()
    finally:
        conn.close()

    assert storage.validate_api_key(raw) is None
    with pytest.raises(storage.CredentialRotated):
        storage.create_api_key(username = admin, name = "after", expect_gen = generation)


def test_consuming_a_legacy_token_reports_the_pre_reset_credential(admin):
    # An unstamped row has no generation to compare, so consume must read the
    # credential inside the delete transaction rather than after committing it.
    token = secrets.token_urlsafe(48)
    expires_at = (datetime.now(timezone.utc) + timedelta(days = 7)).isoformat()
    storage.save_refresh_token(token, admin, expires_at, secret_gen = None)
    conn = storage.get_connection()
    try:
        conn.execute("UPDATE refresh_tokens SET secret_gen = NULL")
        conn.commit()
    finally:
        conn.close()

    consumed = storage.consume_refresh_token(token)
    assert consumed is not None
    _username, _is_desktop, consumed_secret = consumed

    storage.update_password(admin, "new-password-456", revoke_refresh_tokens = True)
    access = create_access_token(subject = admin, secret = consumed_secret)
    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode(access, storage.get_jwt_secret(admin), algorithms = [ALGORITHM])


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
