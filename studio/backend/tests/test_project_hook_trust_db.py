# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused coverage for hash-bound project hook trust persistence."""

from __future__ import annotations

import sqlite3

import pytest

from storage import project_hook_trust_db as trust_db
from storage import studio_db


HASH_A = "a" * 64
HASH_B = "b" * 64
IDENTITY_A = (101, 202)
IDENTITY_B = (303, 404)
WORKSPACE_REVISION = 7


@pytest.fixture(autouse = True)
def _reset_hook_trust_schema():
    trust_db._ready_databases.clear()
    yield
    trust_db._ready_databases.clear()


def _create_project(project_id: str = "hooks-project") -> str:
    conn = studio_db.get_connection()
    try:
        conn.execute(
            """
            INSERT INTO chat_projects (
                id, name, instructions, archived, created_at, updated_at
            )
            VALUES (?, ?, '', 0, 1, 1)
            """,
            (project_id, "Hooks Project"),
        )
        conn.commit()
    finally:
        conn.close()
    return project_id


def _state(
    project_id: str,
    content_hash: str | None,
    *,
    identity: tuple[int, int] = IDENTITY_A,
    workspace_revision: int = WORKSPACE_REVISION,
) -> dict:
    return trust_db.get_project_hook_trust_state(
        project_id,
        content_hash,
        workspace_identity = identity,
        workspace_revision = workspace_revision,
    )


def _trust(
    project_id: str,
    content_hash: str,
    expected_revision: int,
    *,
    identity: tuple[int, int] = IDENTITY_A,
    workspace_revision: int = WORKSPACE_REVISION,
) -> dict:
    return trust_db.trust_project_hooks(
        project_id,
        content_hash,
        workspace_identity = identity,
        workspace_revision = workspace_revision,
        expected_revision = expected_revision,
    )


def _toggle(
    project_id: str,
    content_hash: str,
    handler_id: str,
    *,
    enabled: bool,
    expected_revision: int,
    identity: tuple[int, int] = IDENTITY_A,
    workspace_revision: int = WORKSPACE_REVISION,
) -> dict:
    return trust_db.set_project_hook_handler_enabled(
        project_id,
        content_hash,
        handler_id,
        workspace_identity = identity,
        workspace_revision = workspace_revision,
        enabled = enabled,
        expected_revision = expected_revision,
    )


def test_table_is_lazy_and_has_project_cascade():
    project_id = _create_project()
    conn = studio_db.get_connection()
    try:
        before = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'project_hook_trust'"
        ).fetchone()
    finally:
        conn.close()
    assert before is None

    assert _state(project_id, HASH_A) == {
        "projectId": project_id,
        "contentHash": HASH_A,
        "trusted": False,
        "disabledHandlerIds": [],
        "revision": 0,
    }

    conn = studio_db.get_connection()
    try:
        foreign_keys = conn.execute("PRAGMA foreign_key_list(project_hook_trust)").fetchall()
    finally:
        conn.close()
    assert any(row[2] == "chat_projects" and row[6] == "CASCADE" for row in foreign_keys)


def test_trusts_only_the_exact_rediscovered_hash_and_uses_cas():
    project_id = _create_project()

    trusted = _trust(project_id, HASH_A, 0)

    assert trusted == {
        "projectId": project_id,
        "contentHash": HASH_A,
        "trusted": True,
        "disabledHandlerIds": [],
        "revision": 1,
    }
    assert _state(project_id, HASH_A) == trusted
    assert _state(project_id, HASH_B) == {
        "projectId": project_id,
        "contentHash": HASH_B,
        "trusted": False,
        "disabledHandlerIds": [],
        "revision": 1,
    }
    assert _state(project_id, None)["trusted"] is False
    with pytest.raises(trust_db.ProjectHookTrustConflictError, match = "revision"):
        _trust(project_id, HASH_B, 0)


def test_trust_is_bound_to_workspace_identity_and_revision():
    project_id = _create_project()
    trusted = _trust(project_id, HASH_A, 0)

    identity_drift = _state(project_id, HASH_A, identity = IDENTITY_B)
    revision_drift = _state(
        project_id,
        HASH_A,
        workspace_revision = WORKSPACE_REVISION + 1,
    )

    assert identity_drift["trusted"] is False
    assert revision_drift["trusted"] is False
    assert identity_drift["revision"] == trusted["revision"]
    assert revision_drift["revision"] == trusted["revision"]
    with pytest.raises(trust_db.ProjectHookTrustConflictError, match = "workspace changed"):
        _toggle(
            project_id,
            HASH_A,
            "PreToolUse:0:0",
            identity = IDENTITY_B,
            enabled = False,
            expected_revision = trusted["revision"],
        )

    rebound = _trust(
        project_id,
        HASH_A,
        trusted["revision"],
        identity = IDENTITY_B,
        workspace_revision = WORKSPACE_REVISION + 1,
    )
    assert rebound["trusted"] is True
    assert rebound["revision"] == trusted["revision"] + 1


def test_retrusting_changed_source_resets_stale_disabled_ids():
    project_id = _create_project()
    trusted = _trust(project_id, HASH_A, 0)
    disabled = _toggle(
        project_id,
        HASH_A,
        "PreToolUse:0:0",
        enabled = False,
        expected_revision = trusted["revision"],
    )
    assert disabled["disabledHandlerIds"] == ["PreToolUse:0:0"]

    changed = _state(project_id, HASH_B)
    assert changed["trusted"] is False
    assert changed["disabledHandlerIds"] == []
    retrusted = _trust(project_id, HASH_B, changed["revision"])

    assert retrusted["trusted"] is True
    assert retrusted["disabledHandlerIds"] == []
    assert retrusted["revision"] == disabled["revision"] + 1
    with pytest.raises(trust_db.ProjectHookTrustConflictError, match = "changed"):
        _toggle(
            project_id,
            HASH_A,
            "PreToolUse:0:0",
            enabled = False,
            expected_revision = retrusted["revision"],
        )


def test_handler_toggle_is_hash_bound_idempotent_and_revisioned():
    project_id = _create_project()
    trusted = _trust(project_id, HASH_A, 0)

    disabled = _toggle(
        project_id,
        HASH_A,
        "PostToolUse:1:2",
        enabled = False,
        expected_revision = trusted["revision"],
    )
    assert disabled["disabledHandlerIds"] == ["PostToolUse:1:2"]
    assert disabled["revision"] == 2

    unchanged = _toggle(
        project_id,
        HASH_A,
        "PostToolUse:1:2",
        enabled = False,
        expected_revision = disabled["revision"],
    )
    assert unchanged == disabled

    enabled = _toggle(
        project_id,
        HASH_A,
        "PostToolUse:1:2",
        enabled = True,
        expected_revision = unchanged["revision"],
    )
    assert enabled["disabledHandlerIds"] == []
    assert enabled["revision"] == 3

    with pytest.raises(trust_db.ProjectHookTrustConflictError, match = "revision"):
        _toggle(
            project_id,
            HASH_A,
            "PreToolUse:0:0",
            enabled = False,
            expected_revision = 2,
        )


def test_revoke_preserves_monotonic_revision_and_clears_preferences():
    project_id = _create_project()
    trusted = _trust(project_id, HASH_A, 0)
    disabled = _toggle(
        project_id,
        HASH_A,
        "Stop:0:0",
        enabled = False,
        expected_revision = trusted["revision"],
    )

    revoked = trust_db.revoke_project_hook_trust(
        project_id,
        expected_revision = disabled["revision"],
    )

    assert revoked == {
        "projectId": project_id,
        "contentHash": None,
        "trusted": False,
        "disabledHandlerIds": [],
        "revision": 3,
    }
    assert _state(project_id, HASH_A) == {**revoked, "contentHash": HASH_A}
    with pytest.raises(trust_db.ProjectHookTrustConflictError, match = "revision"):
        _trust(project_id, HASH_A, 0)

    retrusted = _trust(project_id, HASH_A, 3)
    assert retrusted["trusted"] is True
    assert retrusted["revision"] == 4


def test_project_delete_cascades_hook_trust_row():
    project_id = _create_project()
    _trust(project_id, HASH_A, 0)

    conn = studio_db.get_connection()
    try:
        conn.execute("DELETE FROM chat_projects WHERE id = ?", (project_id,))
        conn.commit()
        row = conn.execute(
            "SELECT 1 FROM project_hook_trust WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    finally:
        conn.close()

    assert row is None


def test_rejects_missing_projects_invalid_hashes_and_unbounded_handler_ids():
    with pytest.raises(ValueError, match = "Project does not exist"):
        _trust("missing", HASH_A, 0)

    project_id = _create_project()
    with pytest.raises(ValueError, match = "lowercase SHA-256"):
        _trust(project_id, "A" * 64, 0)
    trusted = _trust(project_id, HASH_A, 0)
    with pytest.raises(ValueError, match = "size limit"):
        _toggle(
            project_id,
            HASH_A,
            "x" * (trust_db.MAX_HANDLER_ID_BYTES + 1),
            enabled = False,
            expected_revision = trusted["revision"],
        )
    with pytest.raises(ValueError, match = "enabled state"):
        _toggle(
            project_id,
            HASH_A,
            "Stop:0:0",
            enabled = 1,
            expected_revision = trusted["revision"],
        )


def test_disabled_handler_count_is_bounded(tmp_path, monkeypatch):
    _ = tmp_path
    project_id = _create_project()
    trusted = _trust(project_id, HASH_A, 0)
    monkeypatch.setattr(trust_db, "MAX_DISABLED_HANDLER_IDS", 1)
    first = _toggle(
        project_id,
        HASH_A,
        "PreToolUse:0:0",
        enabled = False,
        expected_revision = trusted["revision"],
    )

    with pytest.raises(ValueError, match = "exceed"):
        _toggle(
            project_id,
            HASH_A,
            "PostToolUse:0:0",
            enabled = False,
            expected_revision = first["revision"],
        )


def test_corrupt_persisted_preferences_fail_closed():
    project_id = _create_project()
    _trust(project_id, HASH_A, 0)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE project_hook_trust SET disabled_handler_ids_json = ? WHERE project_id = ?",
            ('["Stop:0:0", "Stop:0:0"]', project_id),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(trust_db.ProjectHookTrustStateError, match = "preferences"):
        _state(project_id, HASH_A)
    assert trust_db.get_project_hook_trust_record(project_id) == {
        "projectId": project_id,
        "storedContentHash": HASH_A,
        "hasStoredTrust": True,
        "revision": 1,
    }

    revoked = trust_db.revoke_project_hook_trust(project_id, expected_revision = 1)
    assert revoked["trusted"] is False
    assert revoked["revision"] == 2
    assert _state(project_id, HASH_A)["trusted"] is False


def test_legacy_unbound_trust_is_untrusted_and_can_be_revoked():
    project_id = _create_project()
    conn = studio_db.get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE project_hook_trust (
                project_id TEXT NOT NULL PRIMARY KEY
                    REFERENCES chat_projects(id) ON DELETE CASCADE,
                trusted_content_hash TEXT,
                disabled_handler_ids_json TEXT NOT NULL DEFAULT '[]',
                revision INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )
        conn.execute(
            """
            INSERT INTO project_hook_trust (
                project_id, trusted_content_hash, disabled_handler_ids_json,
                revision, updated_at
            ) VALUES (?, ?, ?, 4, 1)
            """,
            (project_id, HASH_A, '["Stop:0:0"]'),
        )
        conn.commit()
    finally:
        conn.close()

    state = _state(project_id, HASH_A)
    assert state["trusted"] is False
    assert state["disabledHandlerIds"] == []
    assert state["revision"] == 4

    revoked = trust_db.revoke_project_hook_trust(project_id, expected_revision = 4)
    assert revoked["revision"] == 5
    conn = studio_db.get_connection()
    try:
        row = conn.execute(
            """
            SELECT trusted_content_hash, workspace_device_id, workspace_file_id,
                   workspace_revision, disabled_handler_ids_json
            FROM project_hook_trust
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
    finally:
        conn.close()
    assert tuple(row) == (None, None, None, None, "[]")


def test_schema_constraint_rejects_non_sha256_hashes():
    project_id = _create_project()
    _state(project_id, None)
    conn = studio_db.get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO project_hook_trust (
                    project_id, trusted_content_hash, disabled_handler_ids_json,
                    revision, updated_at
                ) VALUES (?, ?, '[]', 1, 1)
                """,
                (project_id, "not-a-hash"),
            )
        conn.rollback()
    finally:
        conn.close()
