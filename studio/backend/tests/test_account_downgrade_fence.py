# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A build without account support cannot authenticate a managed account.

Downgrading an install that has managed accounts must not hand those accounts
the owner's data. The older build reads the legacy credential columns and looks
hashes up by their plain digest; a managed account's real credentials live in
columns it never reads and its hashes carry a prefix it never computes, so every
one of its credentials gets 401 there. The owner's row is unchanged, and this
build reads the real values, so an upgrade brings the accounts straight back.
"""

import secrets
import sqlite3

import jwt
import pytest

from auth import policy, storage
from auth.hashing import verify_password


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield storage
    policy.invalidate_account_cache()


def _legacy(username: str) -> sqlite3.Row:
    """The row exactly as a build without account support reads it."""
    conn = sqlite3.connect(storage.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT password_salt, password_hash, jwt_secret, must_change_password "
            "FROM auth_user WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()


def _legacy_lookup(table: str, column: str, digest: str):
    conn = sqlite3.connect(storage.DB_PATH)
    try:
        return conn.execute(f"SELECT username FROM {table} WHERE {column} = ?", (digest,)).fetchone()
    finally:
        conn.close()


def _set_up_alice(password: str = "alice-password-1") -> dict:
    account = storage.issue_account_setup_code(username = "alice")
    record = storage.authenticate_account_login("alice", account["setup_code"])
    assert record is not None
    _salt, pwd_hash, secret, must_change = record
    assert must_change
    new_secret = storage.update_account_password(
        "alice", password, expect_password_hash = pwd_hash, expect_secret = secret
    )
    assert new_secret
    return account["account"]


def test_owner_row_is_byte_for_byte_legacy(auth_db):
    legacy = _legacy("unsloth")
    assert verify_password("owner-password", legacy["password_salt"], legacy["password_hash"])
    assert storage.get_jwt_secret("unsloth") == legacy["jwt_secret"]
    raw, _row = storage.create_api_key("unsloth", name = "cli")
    assert _legacy_lookup("api_keys", "key_hash", storage._pbkdf2_api_key(raw)) is not None
    storage.save_refresh_token("tok", "unsloth", "2999-01-01T00:00:00+00:00")
    assert _legacy_lookup("refresh_tokens", "token_hash", storage._hash_token("tok")) is not None


def test_managed_password_never_verifies_on_the_legacy_columns(auth_db):
    _set_up_alice()
    assert storage.authenticate_account_login("alice", "alice-password-1") is not None
    legacy = _legacy("alice")
    assert not verify_password("alice-password-1", legacy["password_salt"], legacy["password_hash"])
    assert legacy["password_hash"] == "managed-account"


def test_managed_tokens_do_not_verify_with_the_legacy_secret(auth_db):
    _set_up_alice()
    real = storage.get_jwt_secret("alice")
    legacy = _legacy("alice")["jwt_secret"]
    assert real and legacy and real != legacy
    token = jwt.encode({"sub": "alice"}, real, algorithm = "HS256")
    assert jwt.decode(token, real, algorithms = ["HS256"])["sub"] == "alice"
    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(token, legacy, algorithms = ["HS256"])


def test_managed_api_keys_and_refresh_tokens_are_invisible_to_a_plain_lookup(auth_db):
    _set_up_alice()
    raw, _row = storage.create_api_key("alice", name = "cli")
    assert _legacy_lookup("api_keys", "key_hash", storage._pbkdf2_api_key(raw)) is None
    verified = storage.validate_api_key_account(raw)
    assert verified is not None and verified[0]["username"] == "alice"

    storage.save_refresh_token("alice-refresh", "alice", "2999-01-01T00:00:00+00:00")
    assert _legacy_lookup("refresh_tokens", "token_hash", storage._hash_token("alice-refresh")) is None
    assert storage.verify_refresh_token("alice-refresh") == ("alice", False)
    consumed = storage.consume_refresh_token("alice-refresh")
    assert consumed is not None and consumed[0] == "alice"
    assert consumed[2] == storage.get_jwt_secret("alice")


def test_every_managed_credential_path_stays_fenced(auth_db):
    alice = _set_up_alice()
    # Regenerated setup code, password change, revocation, reactivation: the
    # legacy hash keeps the sentinel and the legacy secret is never the real one.
    storage.issue_account_setup_code(account_id = alice["account_id"])
    assert _legacy("alice")["password_hash"] == "managed-account"
    storage.set_account_active(alice["account_id"], False)
    storage.set_account_active(alice["account_id"], True)
    assert _legacy("alice")["jwt_secret"] != storage.get_jwt_secret("alice")
    storage.update_password("alice", "alice-password-2")
    assert _legacy("alice")["password_hash"] == "managed-account"
    assert storage.get_user_and_secret("alice") is not None
    salt, pwd_hash, _secret, _change = storage.get_user_and_secret("alice")
    assert verify_password("alice-password-2", salt, pwd_hash)


def test_rows_written_before_the_fence_are_moved_behind_it(auth_db):
    """A managed row and its keys created by an earlier build of this branch."""
    conn = sqlite3.connect(storage.DB_PATH)
    try:
        conn.execute(
            "INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret, must_change_password, "
            "account_id, role, is_active, created_at) VALUES ('bob', 's', 'h', 'legacy-secret', 0, 'bob-id', 'user', 1, 'now')"
        )
        conn.execute(
            "INSERT INTO api_keys (username, key_prefix, key_hash, name, created_at) VALUES ('bob', 'p', 'plainhash', 'k', 'now')"
        )
        conn.execute(
            "INSERT INTO refresh_tokens (token_hash, username, expires_at) VALUES ('plaintoken', 'bob', '2999-01-01T00:00:00+00:00')"
        )
        conn.commit()
        storage._fence_managed_credentials(conn)
        conn.commit()
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM auth_user WHERE username = 'bob'").fetchone()
        assert row["account_jwt_secret"] == "legacy-secret" and row["jwt_secret"] != "legacy-secret"
        assert row["account_password_hash"] == "h" and row["password_hash"] == "managed-account"
        assert conn.execute("SELECT key_hash FROM api_keys WHERE username = 'bob'").fetchone()[0] == "account:plainhash"
        assert conn.execute("SELECT token_hash FROM refresh_tokens WHERE username = 'bob'").fetchone()[0] == "account:plaintoken"
        # Idempotent.
        storage._fence_managed_credentials(conn)
        assert conn.execute("SELECT key_hash FROM api_keys WHERE username = 'bob'").fetchone()[0] == "account:plainhash"
        owner = conn.execute("SELECT * FROM auth_user WHERE username = 'unsloth'").fetchone()
        assert owner["account_jwt_secret"] is None
    finally:
        conn.close()
    assert storage.get_jwt_secret("bob") == "legacy-secret"
