# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Old named-column readers ignore new identity/setup columns and accounts/.

This proves owner data compatibility, not safe multi-account service by an old build:
old auth also ignores is_active and cannot enforce the new isolation policy.
"""

import sqlite3
from contextlib import closing

from auth import policy
from utils.account_context import run_as
from utils.paths import workspace_root

from .seed import OLD_AUTH_COLUMNS, SENTINEL, THREAD_ID, old_auth_row, seed_legacy_install


def test_old_named_column_reads_and_owner_password_write_remain_valid(isolated_auth):
    home = isolated_auth.DB_PATH.parent.parent
    original = seed_legacy_install(home)
    before = old_auth_row(isolated_auth.DB_PATH)
    assert isolated_auth.get_account("unsloth").account_id == "owner"
    isolated_auth.create_initial_user("alice", "alice-password", "alice-jwt-secret" * 3)
    alice = isolated_auth.get_account("alice")
    managed = run_as(alice, workspace_root) / "outputs" / "adapter.bin"
    managed.parent.mkdir(parents = True)
    managed.write_bytes(b"private-alice-model")
    with closing(sqlite3.connect(isolated_auth.DB_PATH)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(auth_user)")}
        assert set(OLD_AUTH_COLUMNS) < columns
        assert {"account_id", "is_active", "setup_code_hash", "setup_code_expires_at"} <= columns
        # The old password-change statement names its columns and username.
        conn.execute(
            "UPDATE auth_user SET password_salt=?, password_hash=?, jwt_secret=?, "
            "must_change_password=0 WHERE username=?",
            ("old-reset-salt", "old-reset-hash", "old-reset-secret", "unsloth"),
        )
        conn.commit()
    after = old_auth_row(isolated_auth.DB_PATH)
    assert after[:2] == before[:2]
    assert after[2:5] == ("old-reset-salt", "old-reset-hash", "old-reset-secret")
    assert isolated_auth.get_account("alice") == alice
    assert isolated_auth.get_user_record("alice")["jwt_secret"] == "alice-jwt-secret" * 3
    with closing(sqlite3.connect(home / "studio.db")) as conn:
        assert conn.execute(
            "SELECT title FROM chat_threads WHERE id=?", (THREAD_ID,)
        ).fetchone() == (SENTINEL,)
        assert conn.execute(
            "SELECT value_json FROM chat_settings WHERE key='theme'"
        ).fetchone() == ('"dark"',)
    assert (home / "studio.db").read_bytes() == original["studio.db"]
    assert managed.read_bytes() == b"private-alice-model"
    assert policy.login_mode() == "multi"


def test_old_auth_read_ignores_deactivation_explicitly(isolated_auth, accounts):
    with closing(sqlite3.connect(isolated_auth.DB_PATH)) as conn:
        conn.execute("UPDATE auth_user SET is_active=0 WHERE username='alice'")
        conn.commit()
        legacy = conn.execute(
            "SELECT password_salt,password_hash,jwt_secret,must_change_password "
            "FROM auth_user WHERE username=?",
            ("alice",),
        ).fetchone()
    assert legacy is not None
    assert isolated_auth.get_user_record("alice")["is_active"] == 0
