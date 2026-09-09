# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused invariants for durable project verification state."""

from __future__ import annotations

import os
import re
import sqlite3
import threading

import pytest

from core.agent_workspace import verification_state as state
from core.agent_workspace.verification_context import AgentWorkspaceError
from storage import studio_db


IDENTITY = (101, 202)
OTHER_IDENTITY = (303, 404)
WORKSPACE_REVISION = 7
OWNER = "backend-owner-a"
OTHER_OWNER = "backend-owner-b"


@pytest.fixture(autouse = True)
def _reset_verification_schema_cache():
    state._ready_databases.clear()
    yield
    state._ready_databases.clear()


def _create_project(project_id: str = "verification-project") -> str:
    connection = studio_db.get_connection()
    try:
        connection.execute(
            """
            INSERT INTO chat_projects (
                id, name, instructions, archived, created_at, updated_at
            ) VALUES (?, 'Verification Project', '', 0, 1, 1)
            """,
            (project_id,),
        )
        connection.commit()
    finally:
        connection.close()
    return project_id


def _check(
    name: str = "tests",
    command: str = "python -m pytest",
    *,
    required: bool = True,
    timeout: int = 60,
    log_limit: int = 4096,
) -> dict:
    return {
        "name": name,
        "kind": "custom",
        "command": command,
        "required": required,
        "timeoutSeconds": timeout,
        "logLimitBytes": log_limit,
    }


def _save(
    project_id: str,
    checks: list[dict],
    expected_revision: int = 0,
) -> dict:
    return state.set_verification_config(
        project_id,
        checks,
        workspace_identity = IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
        expected_revision = expected_revision,
    )


def _begin(
    project_id: str,
    profile: dict,
    *,
    owner_id: str = OWNER,
) -> dict:
    return state.begin_verification_run(
        project_id,
        owner_id = owner_id,
        config_revision = profile["revision"],
        config_hash = profile["configHash"],
        checks = profile["checks"],
        workspace_identity = IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
    )


def _result(
    check: dict,
    run: dict,
    status: str,
    *,
    output: str = "",
    output_bytes: int | None = None,
    output_truncated: bool = False,
    started_at: int | None = None,
    error: str | None = None,
) -> dict:
    started = run["startedAt"] if started_at is None else started_at
    terminal = status != "running"
    completed = started + 10 if terminal else None
    if status == "passed":
        exit_code = 0
    elif status == "failed":
        exit_code = 1
    else:
        exit_code = None
    result = {
        "name": check["name"],
        "kind": check["kind"],
        "command": check["command"],
        "required": check["required"],
        "status": status,
        "exitCode": exit_code,
        "output": output,
        "outputBytes": len(output.encode("utf-8")) if output_bytes is None else output_bytes,
        "outputTruncated": output_truncated,
        "timeoutSeconds": check["timeoutSeconds"],
        "startedAt": started,
        "completedAt": completed,
        "durationMs": 10 if terminal else None,
    }
    if error is not None:
        result["error"] = error
    return result


def _complete(
    project_id: str,
    run: dict,
    results: list[dict],
    *,
    owner_id: str = OWNER,
    terminal_status: str | None = None,
    error: str | None = None,
) -> dict:
    return state.complete_verification_run(
        project_id,
        run["id"],
        results,
        owner_id = owner_id,
        terminal_status = terminal_status,
        error = error,
    )


def test_config_is_workspace_bound_hash_verified_and_exact_cas():
    project_id = _create_project()
    empty = state.get_verification_config(project_id)
    assert empty["checks"] == []
    assert empty["revision"] == 0
    assert empty["workspaceDeviceId"] is None
    assert empty["sourceFreshness"] == "unverified"

    profile = _save(project_id, [_check()])
    assert profile["revision"] == 1
    assert len(profile["configHash"]) == 64
    assert profile["workspaceDeviceId"] == IDENTITY[0]
    assert profile["workspaceFileId"] == IDENTITY[1]
    assert profile["workspaceRevision"] == WORKSPACE_REVISION
    assert state.verification_config_matches(
        project_id,
        revision = 1,
        config_hash = profile["configHash"],
        workspace_identity = IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
    )
    assert not state.verification_config_matches(
        project_id,
        revision = 1,
        config_hash = profile["configHash"],
        workspace_identity = OTHER_IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
    )

    assert _save(project_id, profile["checks"], expected_revision = 1) == profile
    with pytest.raises(state.VerificationConflictError, match = "revision"):
        _save(project_id, [_check(command = "python -m pytest -q")], expected_revision = 0)

    rebound = state.set_verification_config(
        project_id,
        profile["checks"],
        workspace_identity = OTHER_IDENTITY,
        workspace_revision = WORKSPACE_REVISION + 1,
        expected_revision = 1,
    )
    assert rebound["revision"] == 2
    assert rebound["configHash"] == profile["configHash"]
    assert rebound["workspaceDeviceId"] == OTHER_IDENTITY[0]


def test_config_revision_cannot_regress_or_be_replaced():
    project_id = _create_project()
    first = _save(project_id, [_check()])
    current = _save(
        project_id,
        [_check(command = "python -m pytest -q")],
        expected_revision = first["revision"],
    )
    assert current["revision"] == 2
    other_project = _create_project("other-config-authority-project")
    other = _save(other_project, [_check(command = "python -m pytest other")])

    connection = studio_db.get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError, match = "revision must increase"):
            connection.execute(
                """
                UPDATE agent_verification_configs
                SET checks_json = '[]'
                WHERE project_id = ?
                """,
                (project_id,),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match = "revision must increase"):
            connection.execute(
                """
                UPDATE agent_verification_configs
                SET revision = revision - 1
                WHERE project_id = ?
                """,
                (project_id,),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match = "identity is immutable"):
            connection.execute(
                """
                UPDATE OR REPLACE agent_verification_configs
                SET project_id = ?, revision = revision + 1
                WHERE project_id = ?
                """,
                (other_project, project_id),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match = "cannot be deleted"):
            connection.execute(
                "DELETE FROM agent_verification_configs WHERE project_id = ?",
                (project_id,),
            )
        connection.rollback()
        row = connection.execute(
            "SELECT * FROM agent_verification_configs WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        assert row is not None
        columns = list(row.keys())
        with pytest.raises(sqlite3.IntegrityError, match = "cannot be replaced"):
            connection.execute(
                f"""
                INSERT OR REPLACE INTO agent_verification_configs ({", ".join(columns)})
                VALUES ({", ".join("?" for _column in columns)})
                """,
                tuple(row[column] for column in columns),
            )
        connection.rollback()
    finally:
        connection.close()

    exact = state.get_verification_config(project_id)
    assert exact == current
    assert state.get_verification_config(other_project) == other


def test_schema_refresh_replaces_hostile_same_name_authority_objects():
    project_id = _create_project()
    _save(project_id, [_check()])
    trigger_names = (
        "trg_agent_verification_configs_no_replace",
        "trg_agent_verification_configs_no_delete",
        "trg_agent_verification_configs_revision_monotonic",
        "trg_agent_verification_runs_evidence_revision",
        "trg_agent_verification_runs_terminal_lease_evidence",
        "trg_agent_verification_runs_revision_monotonic",
        "trg_agent_verification_runs_no_replace",
        "trg_agent_verification_runs_no_delete",
        "trg_agent_verification_runs_identity_immutable",
        "trg_agent_verification_runs_lifecycle_immutable",
        "trg_agent_verification_project_deletions_no_replace",
        "trg_agent_verification_project_deletions_no_delete",
        "trg_agent_verification_project_deletions_lifecycle",
    )
    index_names = (
        "idx_agent_verification_runs_one_active_project",
        "idx_agent_verification_runs_project_history",
        "idx_agent_verification_runs_expired_lease",
    )
    connection = studio_db.get_connection()
    try:
        for trigger_name in trigger_names:
            connection.execute(f"DROP TRIGGER {trigger_name}")
            connection.execute(
                f"""
                CREATE TRIGGER {trigger_name}
                AFTER UPDATE ON agent_verification_runs
                BEGIN
                    SELECT 1;
                END
                """
            )
        for index_name in index_names:
            connection.execute(f"DROP INDEX {index_name}")
            connection.execute(f"CREATE INDEX {index_name} ON agent_verification_runs(id)")
        connection.commit()
    finally:
        connection.close()

    state._ready_databases.clear()
    assert state.get_verification_config(project_id)["revision"] == 1

    connection = studio_db.get_connection()
    try:
        rows = connection.execute(
            f"""
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE name IN ({", ".join("?" for _name in trigger_names + index_names)})
            """,
            trigger_names + index_names,
        ).fetchall()
    finally:
        connection.close()
    objects = {str(row["name"]): row for row in rows}
    assert set(objects) == set(trigger_names + index_names)

    normalized = {name: " ".join(str(row["sql"]).lower().split()) for name, row in objects.items()}
    assert "select 1;" not in " ".join(normalized.values())
    assert (
        "before insert on agent_verification_configs"
        in normalized["trg_agent_verification_configs_no_replace"]
    )
    assert (
        "before delete on agent_verification_configs"
        in normalized["trg_agent_verification_configs_no_delete"]
    )
    assert (
        "before update on agent_verification_configs"
        in normalized["trg_agent_verification_configs_revision_monotonic"]
    )
    assert (
        "after update of id, project_id, owner_id, status, terminal_hint"
        in normalized["trg_agent_verification_runs_evidence_revision"]
    )
    assert (
        "after update of heartbeat_at, lease_expires_at"
        in normalized["trg_agent_verification_runs_terminal_lease_evidence"]
    )
    assert (
        "before update of evidence_revision"
        in normalized["trg_agent_verification_runs_revision_monotonic"]
    )
    assert (
        "before insert on agent_verification_runs"
        in normalized["trg_agent_verification_runs_no_replace"]
    )
    assert (
        "before delete on agent_verification_runs"
        in normalized["trg_agent_verification_runs_no_delete"]
    )
    assert (
        "verification run identity is immutable"
        in normalized["trg_agent_verification_runs_identity_immutable"]
    )
    assert (
        "verification run lifecycle is immutable"
        in normalized["trg_agent_verification_runs_lifecycle_immutable"]
    )
    assert (
        "before insert on agent_verification_project_deletions"
        in normalized["trg_agent_verification_project_deletions_no_replace"]
    )
    assert (
        "before delete on agent_verification_project_deletions"
        in normalized["trg_agent_verification_project_deletions_no_delete"]
    )
    assert (
        "before update on agent_verification_project_deletions"
        in normalized["trg_agent_verification_project_deletions_lifecycle"]
    )
    assert (
        "create unique index idx_agent_verification_runs_one_active_project "
        "on agent_verification_runs(project_id) where status = 'running'"
        == normalized["idx_agent_verification_runs_one_active_project"]
    )
    assert (
        "on agent_verification_runs(project_id, history_sequence desc) "
        "where status != 'running'" in normalized["idx_agent_verification_runs_project_history"]
    )
    assert (
        "on agent_verification_runs(status, lease_expires_at)"
        in normalized["idx_agent_verification_runs_expired_lease"]
    )


def test_schema_rejects_hostile_fresh_name_trigger_on_owned_table():
    project_id = _create_project()
    _save(project_id, [_check()])
    connection = studio_db.get_connection()
    try:
        connection.execute(
            """
            CREATE TRIGGER hostile_fresh_verification_trigger
            AFTER UPDATE OF heartbeat_at ON agent_verification_runs
            BEGIN
                UPDATE agent_verification_runs
                SET lease_expires_at = 0
                WHERE id = NEW.id;
            END
            """
        )
        connection.commit()
    finally:
        connection.close()

    state._ready_databases.clear()
    with pytest.raises(state.VerificationStateError, match = "authority objects"):
        state.get_verification_config(project_id)


def test_schema_rejects_hostile_fresh_name_index_on_owned_table():
    project_id = _create_project()
    _save(project_id, [_check()])
    connection = studio_db.get_connection()
    try:
        connection.execute(
            """
            CREATE INDEX hostile_fresh_verification_index
            ON agent_verification_runs(owner_id)
            """
        )
        connection.commit()
    finally:
        connection.close()

    state._ready_databases.clear()
    with pytest.raises(state.VerificationStateError, match = "authority objects"):
        state.get_verification_config(project_id)


@pytest.mark.parametrize(
    "table_name",
    ("agent_verification_configs", "agent_verification_runs"),
)
def test_schema_rejects_same_column_weaker_authority_tables(table_name):
    project_id = _create_project()
    state.get_verification_config(project_id)
    weak_name = f"{table_name}_weak"
    connection = studio_db.get_connection()
    try:
        original_columns = [
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        ]
        connection.execute(f"CREATE TABLE {weak_name} AS SELECT * FROM {table_name} WHERE 0")
        connection.execute(f"DROP TABLE {table_name}")
        connection.execute(f"ALTER TABLE {weak_name} RENAME TO {table_name}")
        weak_columns = [
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        ]
        assert weak_columns == original_columns
        assert not any(
            int(row[5]) for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        )
        if table_name == "agent_verification_configs":
            assert (
                connection.execute("PRAGMA foreign_key_list(agent_verification_configs)").fetchall()
                == []
            )
        else:
            weak_sql = str(
                connection.execute(
                    "SELECT sql FROM sqlite_master WHERE name = ?",
                    (table_name,),
                ).fetchone()[0]
            ).upper()
            assert re.search(r"\bCHECK\s*\(", weak_sql) is None
            assert re.search(r"\bUNIQUE\b", weak_sql) is None
            assert connection.execute("PRAGMA index_list(agent_verification_runs)").fetchall() == []
        connection.commit()
    finally:
        connection.close()

    state._ready_databases.clear()
    with pytest.raises(state.VerificationStateError, match = "schema is incompatible"):
        state.get_verification_config(project_id)


def test_schema_normalization_does_not_collapse_non_sql_whitespace():
    project_id = _create_project()
    state.get_verification_config(project_id)
    non_sql_whitespace = chr(0x00A0)
    weak_definition = state._RUN_TABLE_SQL.replace(
        "owner_id TEXT NOT NULL",
        f"owner_id TEXT NOT{non_sql_whitespace}NULL",
    ).replace("CREATE TABLE IF NOT EXISTS", "CREATE TABLE", 1)
    connection = studio_db.get_connection()
    try:
        connection.execute("DROP TABLE agent_verification_runs")
        connection.execute(weak_definition)
        owner_column = next(
            row
            for row in connection.execute("PRAGMA table_info(agent_verification_runs)").fetchall()
            if row[1] == "owner_id"
        )
        assert int(owner_column[3]) == 0
        connection.commit()
    finally:
        connection.close()

    state._ready_databases.clear()
    with pytest.raises(state.VerificationStateError, match = "schema is incompatible"):
        state.get_verification_config(project_id)


def test_live_schema_change_invalidates_cached_readiness():
    project_id = _create_project()
    state.get_verification_config(project_id)
    connection = studio_db.get_connection()
    try:
        previous_schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
        connection.execute(
            "CREATE TABLE weak_runs AS SELECT * FROM agent_verification_runs WHERE 0"
        )
        connection.execute("DROP TABLE agent_verification_runs")
        connection.execute("ALTER TABLE weak_runs RENAME TO agent_verification_runs")
        connection.commit()
        assert connection.execute("PRAGMA schema_version").fetchone()[0] > previous_schema_version
    finally:
        connection.close()

    with pytest.raises(state.VerificationStateError, match = "schema is incompatible"):
        state.get_verification_config(project_id)


def test_cached_read_attests_trigger_even_with_restored_schema_version():
    project_id = _create_project()
    state.get_verification_config(project_id)
    database_path = os.path.realpath(str(studio_db.studio_db_path()))
    cached_key = next(key for key in state._ready_databases if key[0] == database_path)
    connection = studio_db.get_connection()
    try:
        connection.execute("DROP TRIGGER trg_agent_verification_configs_no_delete")
        connection.execute(f"PRAGMA schema_version = {cached_key[3]}")
        connection.commit()
        assert connection.execute("PRAGMA schema_version").fetchone()[0] == cached_key[3]
    finally:
        connection.close()

    with pytest.raises(state.VerificationStateError, match = "authority objects"):
        state.get_verification_config(project_id)


def test_committed_wal_ddl_before_write_is_rejected_after_writer_lock():
    project_id = _create_project()
    state.get_verification_config(project_id)
    guarded = state._connection()
    mutator = studio_db.get_connection()
    try:
        assert guarded.in_transaction
        mutator.execute("DROP TRIGGER trg_agent_verification_configs_no_delete")
        mutator.commit()
        with pytest.raises(state.VerificationStateError, match = "authority objects"):
            state._begin_write(guarded)
    finally:
        guarded.rollback()
        guarded.close()
        mutator.close()


def test_same_path_database_replacement_invalidates_cached_readiness(tmp_path):
    project_id = _create_project()
    state.get_verification_config(project_id)
    source = studio_db.get_connection()
    database_path = state._database_key(source)[0]
    original_identity = os.stat(database_path).st_dev, os.stat(database_path).st_ino
    replacement_path = tmp_path / "replacement.db"
    replacement = sqlite3.connect(replacement_path)
    try:
        source.backup(replacement)
        replacement.commit()
    finally:
        replacement.close()
        source.close()

    os.replace(replacement_path, database_path)
    replacement_identity = os.stat(database_path).st_dev, os.stat(database_path).st_ino
    assert replacement_identity != original_identity
    with pytest.raises(state.VerificationStateError, match = "identity changed"):
        state.get_verification_config(project_id)


def test_initial_database_creation_identity_race_fails_closed(tmp_path, monkeypatch):
    database_path = os.path.realpath(str(studio_db.studio_db_path()))
    replacement_path = tmp_path / "initial-create-replacement.db"
    replacement = sqlite3.connect(replacement_path)
    replacement.close()
    original_get_connection = studio_db.get_connection

    def replace_after_open(*args, **kwargs):
        connection = original_get_connection(*args, **kwargs)
        os.replace(replacement_path, database_path)
        return connection

    monkeypatch.setattr(studio_db, "get_connection", replace_after_open)
    with pytest.raises(state.VerificationStateError, match = "changed while it was being opened"):
        state.get_verification_config("initial-create-race")


def test_concurrent_first_open_serializes_schema_cache_refresh(monkeypatch):
    original_ensure_schema = state._ensure_schema
    ensure_entered = threading.Event()
    release_ensure = threading.Event()
    call_lock = threading.Lock()
    blocked = False

    def barriered_ensure_schema(connection):
        nonlocal blocked
        with call_lock:
            should_block = not blocked
            blocked = True
        if should_block:
            ensure_entered.set()
            if not release_ensure.wait(timeout = 2):
                raise AssertionError("schema initialization barrier timed out")
        return original_ensure_schema(connection)

    monkeypatch.setattr(state, "_ensure_schema", barriered_ensure_schema)
    results = []
    failures = []

    def open_state():
        try:
            results.append(state.get_verification_config("concurrent-open"))
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            failures.append(exc)

    first = threading.Thread(target = open_state)
    second = threading.Thread(target = open_state)
    first.start()
    assert ensure_entered.wait(timeout = 2)
    second.start()
    release_ensure.set()
    first.join(timeout = 2)
    second.join(timeout = 2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert len(results) == 2
    assert all(result["revision"] == 0 for result in results)


def test_database_identity_race_fails_closed(monkeypatch):
    project_id = _create_project()
    real_identity = state._database_file_identity
    calls = 0

    def changing_identity(path):
        nonlocal calls
        calls += 1
        identity = real_identity(path)
        return identity if calls < 3 else (identity[0], identity[1] + 1)

    monkeypatch.setattr(state, "_database_file_identity", changing_identity)
    with pytest.raises(state.VerificationStateError, match = "changed during schema inspection"):
        state.get_verification_config(project_id)


def test_config_and_result_inputs_are_strict_and_bounded(monkeypatch):
    project_id = _create_project()
    with pytest.raises(AgentWorkspaceError, match = "At most"):
        _save(project_id, [_check(name = f"check-{index}") for index in range(33)])
    with pytest.raises(AgentWorkspaceError, match = "unknown or missing"):
        _save(project_id, [{**_check(), "typo": True}])
    with pytest.raises(AgentWorkspaceError, match = "boolean"):
        _save(project_id, [{**_check(), "required": 1}])
    with pytest.raises(AgentWorkspaceError, match = "integer"):
        _save(project_id, [{**_check(), "timeoutSeconds": "60"}])
    with pytest.raises(AgentWorkspaceError, match = "size limit"):
        _save(project_id, [_check(command = "x" * (state.MAX_CHECK_COMMAND_BYTES + 1))])
    with pytest.raises(AgentWorkspaceError, match = "configuration"):
        _save(
            project_id,
            [
                _check(name = f"large-{index}", command = f"echo {index};" + "x" * 16_000)
                for index in range(9)
            ],
        )
    with pytest.raises(AgentWorkspaceError, match = "format controls"):
        _save(project_id, [_check(command = "printf safe\u202e")])
    for unsafe_command in (
        "printf safe \u034f# comment; printf HACKED",
        "printf safe \ufe0f# comment; printf HACKED",
        "printf safe \U000e0100# comment; printf HACKED",
        "printf safe \U0001343f# comment; printf HACKED",
    ):
        with pytest.raises(AgentWorkspaceError, match = "default-ignorable"):
            _save(project_id, [_check(command = unsafe_command)])
    for unsafe_command in ("printf safe\x1b", "printf safe\u0085"):
        with pytest.raises(AgentWorkspaceError, match = "unsafe control"):
            _save(project_id, [_check(command = unsafe_command)])
    for unsafe_command in (
        "printf safe\u00a0# comment; printf HACKED",
        "printf safe\u2028printf HACKED",
        "printf safe\u2029printf HACKED",
    ):
        with pytest.raises(AgentWorkspaceError, match = "Unicode whitespace"):
            _save(project_id, [_check(command = unsafe_command)])
    multiline = _save(
        project_id,
        [_check(command = "printf first\n\tprintf second")],
    )
    assert multiline["checks"][0]["command"] == "printf first\n\tprintf second"

    profile = _save(
        project_id,
        [_check(log_limit = 1024)],
        expected_revision = multiline["revision"],
    )
    run = _begin(project_id, profile)
    oversized = _result(
        profile["checks"][0],
        run,
        "running",
        output = "x" * 1025,
    )
    with pytest.raises(AgentWorkspaceError, match = "per-check"):
        state.update_verification_run_progress(
            project_id,
            run["id"],
            [oversized],
            owner_id = OWNER,
        )

    inconsistent = _result(
        profile["checks"][0],
        run,
        "running",
        output = "trusted evidence",
        output_bytes = 0,
    )
    with pytest.raises(AgentWorkspaceError, match = "byte metadata"):
        state.update_verification_run_progress(
            project_id,
            run["id"],
            [inconsistent],
            owner_id = OWNER,
        )
    missing_notice = _result(
        profile["checks"][0],
        run,
        "passed",
        output = "trusted evidence",
        output_truncated = True,
    )
    with pytest.raises(AgentWorkspaceError, match = "capture notice"):
        state.update_verification_run_progress(
            project_id,
            run["id"],
            [missing_notice],
            owner_id = OWNER,
        )
    fake_notice = _result(
        profile["checks"][0],
        run,
        "passed",
        output = (
            "trusted evidence\n"
            "[Process output was truncated. The capture limit was 1024 bytes.]\n"
        ),
        output_bytes = 1,
        output_truncated = True,
    )
    with pytest.raises(AgentWorkspaceError, match = "byte metadata"):
        state.update_verification_run_progress(
            project_id,
            run["id"],
            [fake_notice],
            owner_id = OWNER,
        )
    impossible_capture = _result(
        profile["checks"][0],
        run,
        "passed",
        output = (
            f"{'x' * 100}\n[Process output was truncated. The capture limit was 1 bytes.]\n"
        ),
        output_bytes = 100,
        output_truncated = True,
    )
    with pytest.raises(AgentWorkspaceError, match = "byte metadata"):
        state.update_verification_run_progress(
            project_id,
            run["id"],
            [impossible_capture],
            owner_id = OWNER,
        )
    replacement = _result(
        profile["checks"][0],
        run,
        "passed",
        output = "\ufffd",
        output_bytes = 1,
    )
    assert (
        state._validate_result(
            replacement,
            profile["checks"][0],
            run_started_at = run["startedAt"],
        )["outputBytes"]
        == 1
    )

    monkeypatch.setattr(state, "MAX_RUN_RESULT_BYTES", 5)
    bounded = _result(profile["checks"][0], run, "running", output = "123456")
    with pytest.raises(AgentWorkspaceError, match = "run limit"):
        state.update_verification_run_progress(
            project_id,
            run["id"],
            [bounded],
            owner_id = OWNER,
        )


def test_run_pins_exact_config_workspace_owner_and_single_active_row():
    project_id = _create_project()
    profile = _save(project_id, [_check("lint"), _check("tests")])
    run = _begin(project_id, profile)

    assert run["ownerId"] == OWNER
    assert run["configHash"] == profile["configHash"]
    assert run["workspaceDeviceId"] == IDENTITY[0]
    assert run["workspaceFileId"] == IDENTITY[1]
    assert run["workspaceRevision"] == WORKSPACE_REVISION
    assert run["heartbeatAt"] == run["startedAt"]
    assert run["leaseExpiresAt"] > run["heartbeatAt"]
    assert run["sourceFreshness"] == "unverified"
    assert run["historySequence"] is None
    assert state.active_verification_run(project_id)["id"] == run["id"]

    with pytest.raises(state.VerificationConflictError, match = "already running"):
        _begin(project_id, profile, owner_id = OTHER_OWNER)
    with pytest.raises(state.VerificationConflictError, match = "another owner"):
        state.update_verification_run_progress(
            project_id,
            run["id"],
            [],
            owner_id = OTHER_OWNER,
        )
    assert state.verification_run_may_spawn(
        project_id,
        run["id"],
        OWNER,
        revision = profile["revision"],
        config_hash = profile["configHash"],
        workspace_identity = IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
    )
    assert not state.verification_run_may_spawn(
        project_id,
        run["id"],
        OTHER_OWNER,
        revision = profile["revision"],
        config_hash = profile["configHash"],
        workspace_identity = IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
    )
    assert not state.verification_run_may_spawn(
        project_id,
        run["id"],
        OWNER,
        revision = profile["revision"],
        config_hash = profile["configHash"],
        workspace_identity = OTHER_IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
    )

    with pytest.raises(state.VerificationConflictError, match = "locked while a run is active"):
        _save(
            project_id,
            [_check("lint", command = "ruff check ."), _check("tests")],
            expected_revision = profile["revision"],
        )


def test_pre_spawn_uses_time_sampled_after_writer_authority(monkeypatch):
    clock = [10_000]
    monkeypatch.setattr(state, "_now_ms", lambda: clock[0])
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    run = _begin(project_id, profile)
    original_begin_write = state._begin_write

    def delayed_begin_write(connection):
        original_begin_write(connection)
        clock[0] = run["leaseExpiresAt"] + 1

    monkeypatch.setattr(state, "_begin_write", delayed_begin_write)

    assert not state.verification_run_may_spawn(
        project_id,
        run["id"],
        OWNER,
        revision = profile["revision"],
        config_hash = profile["configHash"],
        workspace_identity = IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
    )
    exact = state.get_verification_run(project_id, run["id"])
    assert exact is not None
    assert exact["status"] == "interrupted"


def test_progress_is_an_exact_ordered_prefix_with_one_append_only_running_tail():
    project_id = _create_project()
    profile = _save(project_id, [_check("lint"), _check("tests")])
    run = _begin(project_id, profile)
    lint, tests = profile["checks"]

    running = _result(lint, run, "running", output = "a")
    state.update_verification_run_progress(
        project_id,
        run["id"],
        [running],
        owner_id = OWNER,
    )
    growing = _result(lint, run, "running", output = "abc")
    state.update_verification_run_progress(
        project_id,
        run["id"],
        [growing],
        owner_id = OWNER,
    )

    regressed = _result(lint, run, "running", output = "abX")
    with pytest.raises(state.VerificationConflictError, match = "append-only"):
        state.update_verification_run_progress(
            project_id,
            run["id"],
            [regressed],
            owner_id = OWNER,
        )
    with pytest.raises(AgentWorkspaceError, match = "exact configured check prefix"):
        state.update_verification_run_progress(
            project_id,
            run["id"],
            [_result(tests, run, "running")],
            owner_id = OWNER,
        )

    passed_lint = _result(lint, run, "passed", output = "abc")
    state.update_verification_run_progress(
        project_id,
        run["id"],
        [passed_lint],
        owner_id = OWNER,
    )
    with pytest.raises(state.VerificationConflictError, match = "cannot change"):
        state.update_verification_run_progress(
            project_id,
            run["id"],
            [_result(lint, run, "failed", output = "abc")],
            owner_id = OWNER,
        )

    running_tests = _result(tests, run, "running", started_at = run["startedAt"] + 20)
    updated = state.update_verification_run_progress(
        project_id,
        run["id"],
        [passed_lint, running_tests],
        owner_id = OWNER,
    )
    assert [item["status"] for item in updated["results"]] == ["passed", "running"]

    overlapping_tests = _result(
        tests,
        run,
        "running",
        started_at = passed_lint["completedAt"] - 1,
    )
    with pytest.raises(AgentWorkspaceError, match = "sequential and non-overlapping"):
        state.update_verification_run_progress(
            project_id,
            run["id"],
            [passed_lint, overlapping_tests],
            owner_id = OWNER,
        )


def test_control_heavy_output_stays_within_the_bounded_canonical_json_envelope():
    project_id = _create_project()
    profile = _save(project_id, [_check(log_limit = 1024 * 1024)])
    run = _begin(project_id, profile)
    output = "\x00" * (1024 * 1024)

    updated = state.update_verification_run_progress(
        project_id,
        run["id"],
        [_result(profile["checks"][0], run, "running", output = output)],
        owner_id = OWNER,
    )

    assert updated["results"][0]["output"] == output


@pytest.mark.parametrize(
    ("result_status", "terminal_hint", "expected"),
    [
        ("passed", None, "passed"),
        ("failed", None, "failed"),
        ("cancelled", None, "cancelled"),
        ("timed_out", None, "timed_out"),
        ("blocked", None, "blocked"),
        (None, None, "blocked"),
        (None, "interrupted", "interrupted"),
    ],
)
def test_terminal_status_is_derived_from_required_evidence(
    result_status: str | None, terminal_hint: str | None, expected: str
):
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    run = _begin(project_id, profile)
    results = (
        [_result(profile["checks"][0], run, result_status)] if result_status is not None else []
    )
    finished = _complete(
        project_id,
        run,
        results,
        terminal_status = terminal_hint,
        error = "execution stopped" if terminal_hint == "interrupted" else None,
    )
    assert finished["status"] == expected
    assert finished["completedAt"] is not None
    assert state.active_verification_run(project_id) is None


def test_optional_failure_does_not_override_complete_required_evidence():
    project_id = _create_project()
    profile = _save(project_id, [_check("required"), _check("optional", required = False)])
    run = _begin(project_id, profile)
    required = _result(profile["checks"][0], run, "passed")
    finished = _complete(
        project_id,
        run,
        [
            required,
            _result(
                profile["checks"][1],
                run,
                "failed",
                started_at = required["completedAt"],
            ),
        ],
    )
    assert finished["status"] == "passed"
    assert finished["historySequence"] == 1


def test_durable_cancel_wins_completion_and_project_cancel_is_idempotent():
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    run = _begin(project_id, profile)

    cancelled, accepted = state.request_verification_cancel(project_id, run["id"])
    assert accepted is True
    assert cancelled["cancelRequested"] is True
    repeated, accepted_again = state.request_project_verification_cancel(project_id)
    assert accepted_again is True
    assert repeated["id"] == run["id"]
    assert state.heartbeat_verification_run(project_id, run["id"], OWNER) is True
    assert not state.verification_run_may_spawn(
        project_id,
        run["id"],
        OWNER,
        revision = profile["revision"],
        config_hash = profile["configHash"],
        workspace_identity = IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
    )

    finished = _complete(
        project_id,
        run,
        [_result(profile["checks"][0], run, "passed")],
        terminal_status = "failed",
        error = "late worker failure",
    )
    assert finished["status"] == "cancelled"
    assert finished["cancelRequested"] is True
    assert finished["error"] is None
    terminal, terminal_accepted = state.request_verification_cancel(project_id, run["id"])
    assert terminal["status"] == "cancelled"
    assert terminal_accepted is False
    assert state.request_project_verification_cancel(project_id) == (None, False)


def test_cancel_after_lease_expiry_preserves_ownership_failure(monkeypatch):
    clock = [1_000]
    monkeypatch.setattr(state, "_now_ms", lambda: clock[0])
    exact_project = _create_project("expired-exact-cancel")
    exact_profile = _save(exact_project, [_check()])
    exact_run = _begin(exact_project, exact_profile)

    clock[0] = exact_run["leaseExpiresAt"] + 1
    exact, accepted = state.request_verification_cancel(
        exact_project,
        exact_run["id"],
    )
    assert accepted is False
    assert exact["status"] == "interrupted"
    assert exact["cancelRequested"] is False
    assert exact["historySequence"] is not None

    project_cancel = _create_project("expired-project-cancel")
    project_profile = _save(project_cancel, [_check()])
    project_run = _begin(project_cancel, project_profile)
    clock[0] = project_run["leaseExpiresAt"] + 1

    assert state.request_project_verification_cancel(project_cancel) == (None, False)
    recovered = state.get_verification_run(project_cancel, project_run["id"])
    assert recovered is not None
    assert recovered["status"] == "interrupted"
    assert recovered["cancelRequested"] is False
    assert recovered["historySequence"] > exact["historySequence"]


def test_lease_reconciliation_preserves_progress_and_does_not_touch_live_sibling(monkeypatch):
    clock = [1_000]
    monkeypatch.setattr(state, "_now_ms", lambda: clock[0])
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    run = _begin(project_id, profile)
    running = _result(profile["checks"][0], run, "running", output = "partial")
    state.update_verification_run_progress(
        project_id,
        run["id"],
        [running],
        owner_id = OWNER,
    )

    clock[0] = run["leaseExpiresAt"] - 1
    assert state.reconcile_interrupted_verification_runs() == 0
    assert state.active_verification_run(project_id)["ownerId"] == OWNER
    with pytest.raises(state.VerificationConflictError, match = "another owner"):
        state.heartbeat_verification_run(project_id, run["id"], OTHER_OWNER)

    clock[0] = run["leaseExpiresAt"] + 1
    assert state.reconcile_interrupted_verification_runs() == 1
    recovered = state.get_verification_run(project_id, run["id"])
    assert recovered["status"] == "interrupted"
    assert recovered["results"] == [running]
    assert recovered["error"] == "Verification ownership expired before completion."
    assert state.active_verification_run(project_id) is None

    cancelled_run = _begin(project_id, profile, owner_id = OTHER_OWNER)
    state.request_project_verification_cancel(project_id)
    clock[0] = cancelled_run["leaseExpiresAt"] + 1
    assert state.reconcile_interrupted_verification_runs() == 1
    cancelled = state.get_verification_run(project_id, cancelled_run["id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancelRequested"] is True


def test_backward_clock_does_not_regress_run_or_deletion_leases(monkeypatch):
    clock = [10_000]
    monkeypatch.setattr(state, "_now_ms", lambda: clock[0])
    project_id = _create_project()
    profile = _save(project_id, [_check("first"), _check("second")])
    run = _begin(project_id, profile)
    initial_heartbeat = run["heartbeatAt"]
    initial_expiry = run["leaseExpiresAt"]

    clock[0] = 5_000
    assert state.heartbeat_verification_run(project_id, run["id"], OWNER) is False
    assert state.verification_run_may_spawn(
        project_id,
        run["id"],
        OWNER,
        revision = profile["revision"],
        config_hash = profile["configHash"],
        workspace_identity = IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
    )
    first_result = _result(profile["checks"][0], run, "passed")
    progressed = state.update_verification_run_progress(
        project_id,
        run["id"],
        [first_result],
        owner_id = OWNER,
    )
    assert progressed["heartbeatAt"] == initial_heartbeat
    assert progressed["leaseExpiresAt"] == initial_expiry
    assert progressed["updatedAt"] >= first_result["completedAt"]

    fence = state.begin_verification_project_deletion(project_id, "clock-fence")
    fence_heartbeat = fence["heartbeatAt"]
    fence_expiry = fence["leaseExpiresAt"]
    clock[0] = 4_000
    repeated = state.begin_verification_project_deletion(project_id, "clock-fence")
    heartbeat = state.heartbeat_verification_project_deletion(
        project_id,
        "clock-fence",
        fence["revision"],
    )
    assert repeated["heartbeatAt"] == fence_heartbeat
    assert repeated["leaseExpiresAt"] == fence_expiry
    assert heartbeat["heartbeatAt"] == fence_heartbeat
    assert heartbeat["leaseExpiresAt"] == fence_expiry

    state.finish_verification_project_deletion(
        project_id,
        "clock-fence",
        fence["revision"],
    )
    clock[0] = initial_expiry + 1
    assert state.reconcile_interrupted_verification_runs() == 1
    reconciled = state.get_verification_run(project_id, run["id"])
    assert reconciled["status"] == "interrupted"
    assert reconciled["completedAt"] >= first_result["completedAt"]


def test_leased_deletion_fence_blocks_admission_and_pre_spawn(monkeypatch):
    clock = [10_000]
    monkeypatch.setattr(state, "_now_ms", lambda: clock[0])
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    run = _begin(project_id, profile)

    fence = state.begin_verification_project_deletion(project_id, "delete-a")
    assert fence["leaseExpiresAt"] > fence["heartbeatAt"]
    same = state.begin_verification_project_deletion(project_id, "delete-a")
    assert same["fenceId"] == "delete-a"
    with pytest.raises(state.VerificationConflictError, match = "another owner"):
        state.begin_verification_project_deletion(project_id, "delete-b")
    with pytest.raises(state.VerificationConflictError, match = "retirement is in progress"):
        _save(project_id, profile["checks"], expected_revision = profile["revision"])
    with pytest.raises(state.VerificationConflictError, match = "deletion is in progress"):
        _begin(project_id, profile, owner_id = OTHER_OWNER)
    assert not state.verification_run_may_spawn(
        project_id,
        run["id"],
        OWNER,
        revision = profile["revision"],
        config_hash = profile["configHash"],
        workspace_identity = IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
    )

    heartbeat = state.heartbeat_verification_project_deletion(
        project_id,
        "delete-a",
        fence["revision"],
    )
    assert heartbeat["fenceId"] == "delete-a"
    assert (
        state.finish_verification_project_deletion(
            project_id,
            "delete-a",
            fence["revision"],
        )
        is True
    )
    assert (
        state.finish_verification_project_deletion(
            project_id,
            "delete-a",
            fence["revision"],
        )
        is False
    )

    expiring = state.begin_verification_project_deletion(project_id, "delete-old")
    clock[0] = expiring["leaseExpiresAt"] + 1
    replacement = state.begin_verification_project_deletion(project_id, "delete-new")
    assert replacement["fenceId"] == "delete-new"
    with pytest.raises(state.VerificationConflictError, match = "another owner"):
        state.finish_verification_project_deletion(
            project_id,
            "delete-old",
            expiring["revision"],
        )


def test_deletion_fence_revision_blocks_delete_and_replace_aba():
    project_id = _create_project()
    first = state.begin_verification_project_deletion(project_id, "delete-a")
    assert first["active"] is True
    assert first["revision"] == 1

    connection = studio_db.get_connection()
    try:
        row = connection.execute(
            "SELECT * FROM agent_verification_project_deletions WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        assert row is not None
        with pytest.raises(sqlite3.IntegrityError, match = "cannot be deleted"):
            connection.execute(
                "DELETE FROM agent_verification_project_deletions WHERE project_id = ?",
                (project_id,),
            )
        connection.rollback()
        columns = list(row.keys())
        replacement = [row[column] for column in columns]
        replacement[columns.index("fence_id")] = "delete-stolen"
        with pytest.raises(sqlite3.IntegrityError, match = "cannot be replaced"):
            connection.execute(
                f"""
                INSERT OR REPLACE INTO agent_verification_project_deletions (
                    {", ".join(columns)}
                ) VALUES ({", ".join("?" for _column in columns)})
                """,
                tuple(replacement),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match = "lifecycle is invalid"):
            connection.execute(
                """
                UPDATE agent_verification_project_deletions
                SET fence_id = ?, revision = revision + 1
                WHERE project_id = ?
                """,
                ("delete-stolen", project_id),
            )
        connection.rollback()
    finally:
        connection.close()

    assert (
        state.finish_verification_project_deletion(
            project_id,
            "delete-a",
            first["revision"],
        )
        is True
    )
    connection = studio_db.get_connection()
    try:
        released = connection.execute(
            """
            SELECT active, revision FROM agent_verification_project_deletions
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        assert released is not None
        assert released["active"] == 0
        assert released["revision"] == 1
        with pytest.raises(sqlite3.IntegrityError, match = "cannot be deleted"):
            connection.execute(
                "DELETE FROM agent_verification_project_deletions WHERE project_id = ?",
                (project_id,),
            )
        connection.rollback()
    finally:
        connection.close()

    second = state.begin_verification_project_deletion(project_id, "delete-b")
    assert second["active"] is True
    assert second["revision"] == 2
    assert second["fenceId"] == "delete-b"
    assert (
        state.finish_verification_project_deletion(
            project_id,
            "delete-b",
            second["revision"],
        )
        is True
    )

    third = state.begin_verification_project_deletion(project_id, "delete-a")
    assert third["active"] is True
    assert third["revision"] == 3
    assert third["fenceId"] == "delete-a"
    with pytest.raises(state.VerificationConflictError, match = "expired or changed"):
        state.heartbeat_verification_project_deletion(
            project_id,
            "delete-a",
            first["revision"],
        )
    with pytest.raises(state.VerificationConflictError, match = "another owner"):
        state.finish_verification_project_deletion(
            project_id,
            "delete-a",
            first["revision"],
        )


def test_archived_projects_reject_config_start_and_pre_spawn():
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    run = _begin(project_id, profile)
    connection = studio_db.get_connection()
    try:
        connection.execute(
            "UPDATE chat_projects SET archived = 1 WHERE id = ?",
            (project_id,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(state.VerificationConflictError, match = "Archived projects"):
        _save(project_id, profile["checks"], expected_revision = profile["revision"])
    with pytest.raises(state.VerificationConflictError, match = "Archived projects"):
        _begin(project_id, profile, owner_id = OTHER_OWNER)
    assert not state.verification_run_may_spawn(
        project_id,
        run["id"],
        OWNER,
        revision = profile["revision"],
        config_hash = profile["configHash"],
        workspace_identity = IDENTITY,
        workspace_revision = WORKSPACE_REVISION,
    )


def test_corrupt_json_and_hashes_fail_closed_without_fallback():
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    connection = studio_db.get_connection()
    try:
        connection.execute(
            """
            UPDATE agent_verification_configs
            SET checks_json = '[]', revision = revision + 1
            WHERE project_id = ?
            """,
            (project_id,),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(state.VerificationStateError, match = "hash"):
        state.get_verification_config(project_id)

    other_project = _create_project("corrupt-run-project")
    other_profile = _save(other_project, [_check()])
    run = _begin(other_project, other_profile)
    connection = studio_db.get_connection()
    try:
        connection.execute(
            """
            UPDATE agent_verification_runs
            SET results_json = ?
            WHERE id = ?
            """,
            ('[{"name":"tests","name":"duplicate"}]', run["id"]),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(state.VerificationStateError, match = "invalid JSON"):
        state.get_verification_run(other_project, run["id"])


def test_history_summary_never_claims_unloaded_terminal_evidence():
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    run = _begin(project_id, profile)
    finished = _complete(
        project_id,
        run,
        [_result(profile["checks"][0], run, "passed", output = "verified")],
    )

    [summary] = state.list_verification_run_summaries(project_id)
    assert summary["id"] == finished["id"]
    assert summary["status"] == "unverified"
    assert summary["evidenceStatus"] == "not_loaded"
    assert "checks" not in summary
    assert "results" not in summary
    assert "output" not in repr(summary)
    initial_evidence_revision = summary["evidenceRevision"]

    connection = studio_db.get_connection()
    try:
        connection.execute(
            "UPDATE agent_verification_runs SET results_json = '[]' WHERE id = ?",
            (run["id"],),
        )
        connection.commit()
    finally:
        connection.close()

    [corrupt_summary] = state.list_verification_run_summaries(project_id)
    assert corrupt_summary["status"] == "unverified"
    assert corrupt_summary["evidenceStatus"] == "not_loaded"
    assert corrupt_summary["evidenceRevision"] == initial_evidence_revision + 1
    assert (
        state.get_verification_run_evidence_revision(project_id, run["id"])
        == initial_evidence_revision + 1
    )
    with pytest.raises(state.VerificationStateError, match = "status is inconsistent"):
        state.get_verification_run(project_id, run["id"])


def test_evidence_revision_ignores_lease_heartbeats(monkeypatch):
    clock = [10_000]
    monkeypatch.setattr(state, "_now_ms", lambda: clock[0])
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    run = _begin(project_id, profile)
    initial_revision = state.get_verification_run_evidence_revision(project_id, run["id"])

    clock[0] += 5_000
    assert state.heartbeat_verification_run(project_id, run["id"], OWNER) is False
    assert state.get_verification_run_evidence_revision(project_id, run["id"]) == initial_revision

    running = _result(profile["checks"][0], run, "running", output = "progress")
    state.update_verification_run_progress(
        project_id,
        run["id"],
        [running],
        owner_id = OWNER,
    )
    progress_revision = state.get_verification_run_evidence_revision(project_id, run["id"])
    assert progress_revision is not None
    assert initial_revision is not None
    assert progress_revision == initial_revision + 1

    def reject_writer_authority(_connection):
        raise AssertionError("unchanged marker poll acquired writer authority")

    monkeypatch.setattr(state, "_begin_write", reject_writer_authority)
    assert state.get_verification_run_evidence_revision(project_id, run["id"]) == progress_revision


def test_heartbeat_uses_bounded_ownership_metadata(monkeypatch):
    clock = [10_000]
    monkeypatch.setattr(state, "_now_ms", lambda: clock[0])
    project_id = _create_project("bounded-heartbeat-project")
    profile = _save(project_id, [_check(log_limit = 2 * 1024 * 1024)])
    run = _begin(project_id, profile)
    connection = studio_db.get_connection()
    try:
        connection.execute(
            "UPDATE agent_verification_runs SET results_json = ? WHERE id = ?",
            ("x" * (4 * 1024 * 1024), run["id"]),
        )
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setattr(
        state,
        "_run_from_row",
        lambda _row: pytest.fail("heartbeat decoded verification evidence"),
    )

    clock[0] += 5_000
    assert state.heartbeat_verification_run(project_id, run["id"], OWNER) is False
    with pytest.raises(state.VerificationConflictError, match = "another owner"):
        state.heartbeat_verification_run(project_id, run["id"], OTHER_OWNER)
    connection = studio_db.get_connection()
    try:
        row = connection.execute(
            "SELECT heartbeat_at, lease_expires_at FROM agent_verification_runs WHERE id = ?",
            (run["id"],),
        ).fetchone()
    finally:
        connection.close()
    assert row["heartbeat_at"] == clock[0]
    assert row["lease_expires_at"] >= clock[0] + state.LEASE_DURATION_MS


def test_evidence_revision_cannot_regress_or_be_replaced():
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    run = _begin(project_id, profile)
    current = state.update_verification_run_progress(
        project_id,
        run["id"],
        [_result(profile["checks"][0], run, "running", output = "trusted")],
        owner_id = OWNER,
    )
    assert current["evidenceRevision"] == 2

    connection = studio_db.get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError, match = "revision must increase"):
            connection.execute(
                """
                UPDATE agent_verification_runs
                SET results_json = '[]', evidence_revision = evidence_revision - 1
                WHERE id = ?
                """,
                (run["id"],),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match = "cannot be deleted"):
            connection.execute(
                "DELETE FROM agent_verification_runs WHERE id = ?",
                (run["id"],),
            )
        connection.rollback()
        row = connection.execute(
            "SELECT * FROM agent_verification_runs WHERE id = ?",
            (run["id"],),
        ).fetchone()
        assert row is not None
        columns = list(row.keys())
        placeholders = ", ".join("?" for _column in columns)
        with pytest.raises(sqlite3.IntegrityError, match = "cannot be replaced"):
            connection.execute(
                f"""
                INSERT OR REPLACE INTO agent_verification_runs ({", ".join(columns)})
                VALUES ({placeholders})
                """,
                tuple(row[column] for column in columns),
            )
        connection.rollback()
    finally:
        connection.close()

    exact = state.get_verification_run(project_id, run["id"])
    assert exact is not None
    assert exact["evidenceRevision"] == 2
    assert exact["results"][0]["output"] == "trusted"

    finished = _complete(
        project_id,
        run,
        [_result(profile["checks"][0], run, "passed", output = "trusted")],
    )
    successor = _begin(project_id, profile)
    other_project = _create_project("immutable-run-project")
    connection = studio_db.get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError, match = "cannot be deleted"):
            connection.execute(
                "DELETE FROM agent_verification_runs WHERE id = ?",
                (finished["id"],),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match = "identity is immutable"):
            connection.execute(
                "UPDATE OR REPLACE agent_verification_runs SET id = ? WHERE id = ?",
                (successor["id"], finished["id"]),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match = "identity is immutable"):
            connection.execute(
                "UPDATE agent_verification_runs SET project_id = ? WHERE id = ?",
                (other_project, finished["id"]),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match = "lifecycle is immutable"):
            connection.execute(
                """
                UPDATE OR REPLACE agent_verification_runs
                SET status = 'running', completed_at = NULL, history_sequence = NULL
                WHERE id = ?
                """,
                (finished["id"],),
            )
        connection.rollback()
        active_row = connection.execute(
            "SELECT * FROM agent_verification_runs WHERE id = ?",
            (successor["id"],),
        ).fetchone()
        assert active_row is not None
        columns = list(active_row.keys())
        active_values = [active_row[column] for column in columns]
        active_values[columns.index("id")] = "fresh-active-replacement-id"
        with pytest.raises(sqlite3.IntegrityError, match = "cannot be replaced"):
            connection.execute(
                f"""
                INSERT OR REPLACE INTO agent_verification_runs ({", ".join(columns)})
                VALUES ({", ".join("?" for _column in columns)})
                """,
                tuple(active_values),
            )
        connection.rollback()
        terminal_row = connection.execute(
            "SELECT * FROM agent_verification_runs WHERE id = ?",
            (finished["id"],),
        ).fetchone()
        assert terminal_row is not None
        columns = list(terminal_row.keys())
        terminal_values = [terminal_row[column] for column in columns]
        terminal_values[columns.index("id")] = "fresh-terminal-replacement-id"
        with pytest.raises(sqlite3.IntegrityError, match = "cannot be replaced"):
            connection.execute(
                f"""
                INSERT OR REPLACE INTO agent_verification_runs ({", ".join(columns)})
                VALUES ({", ".join("?" for _column in columns)})
                """,
                tuple(terminal_values),
            )
        connection.rollback()
        rows = connection.execute(
            "SELECT id FROM agent_verification_runs WHERE project_id = ?",
            (project_id,),
        ).fetchall()
    finally:
        connection.close()
    assert {row["id"] for row in rows} == {finished["id"], successor["id"]}


def test_terminal_lease_corruption_advances_conditional_evidence_marker():
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    run = _begin(project_id, profile)
    finished = _complete(
        project_id,
        run,
        [_result(profile["checks"][0], run, "passed")],
    )
    marker = finished["evidenceRevision"]

    connection = studio_db.get_connection()
    try:
        connection.execute(
            """
            UPDATE agent_verification_runs
            SET heartbeat_at = started_at - 1
            WHERE id = ?
            """,
            (run["id"],),
        )
        connection.commit()
    finally:
        connection.close()

    assert state.get_verification_run_evidence_revision(project_id, run["id"]) == marker + 1
    with pytest.raises(state.VerificationStateError, match = "timestamps are invalid"):
        state.get_verification_run(project_id, run["id"])


def test_large_result_read_does_not_hold_the_writer_lock(monkeypatch):
    project_id = _create_project()
    profile = _save(project_id, [_check(log_limit = 1024 * 1024)])
    run = _begin(project_id, profile)
    state.update_verification_run_progress(
        project_id,
        run["id"],
        [_result(profile["checks"][0], run, "running", output = "x" * (512 * 1024))],
        owner_id = OWNER,
    )

    reader_entered = threading.Event()
    release_reader = threading.Event()
    reader_identity: list[int] = []
    reader_errors: list[BaseException] = []
    heartbeat_done = threading.Event()
    heartbeat_errors: list[BaseException] = []
    original_run_from_row = state._run_from_row

    def blocked_reader(row):
        if reader_identity and threading.get_ident() == reader_identity[0]:
            reader_entered.set()
            if not release_reader.wait(timeout = 5):
                raise AssertionError("reader release timed out")
        return original_run_from_row(row)

    monkeypatch.setattr(state, "_run_from_row", blocked_reader)

    def read_large_result():
        reader_identity.append(threading.get_ident())
        try:
            state.get_verification_run(project_id, run["id"])
        except BaseException as exc:  # noqa: BLE001 - surfaced in the owning test thread
            reader_errors.append(exc)

    def heartbeat():
        try:
            state.heartbeat_verification_run(project_id, run["id"], OWNER)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the owning test thread
            heartbeat_errors.append(exc)
        finally:
            heartbeat_done.set()

    reader_thread = threading.Thread(target = read_large_result)
    heartbeat_thread = threading.Thread(target = heartbeat)
    reader_thread.start()
    assert reader_entered.wait(timeout = 2)
    heartbeat_thread.start()
    heartbeat_completed_while_reader_was_blocked = heartbeat_done.wait(timeout = 1)
    release_reader.set()
    reader_thread.join(timeout = 5)
    heartbeat_thread.join(timeout = 5)

    assert heartbeat_completed_while_reader_was_blocked
    assert not reader_thread.is_alive()
    assert not heartbeat_thread.is_alive()
    assert reader_errors == []
    assert heartbeat_errors == []


def test_expired_large_result_is_reconciled_before_lock_free_decode(monkeypatch):
    clock = [20_000]
    monkeypatch.setattr(state, "_now_ms", lambda: clock[0])
    project_id = _create_project()
    profile = _save(project_id, [_check(log_limit = 1024 * 1024)])
    run = _begin(project_id, profile)
    state.update_verification_run_progress(
        project_id,
        run["id"],
        [_result(profile["checks"][0], run, "running", output = "x" * (512 * 1024))],
        owner_id = OWNER,
    )

    clock[0] = run["leaseExpiresAt"] + 1
    sibling_project = _create_project("reconciliation-writer-project")
    sibling_profile = _save(sibling_project, [_check()])
    sibling = _begin(sibling_project, sibling_profile, owner_id = OTHER_OWNER)

    reader_entered = threading.Event()
    release_reader = threading.Event()
    reader_identity: list[int] = []
    reader_records: list[dict] = []
    reader_errors: list[BaseException] = []
    writer_done = threading.Event()
    writer_errors: list[BaseException] = []
    original_run_from_row = state._run_from_row

    def blocked_reader(row):
        if reader_identity and threading.get_ident() == reader_identity[0]:
            reader_entered.set()
            if not release_reader.wait(timeout = 5):
                raise AssertionError("reader release timed out")
        return original_run_from_row(row)

    monkeypatch.setattr(state, "_run_from_row", blocked_reader)

    def read_expired_result():
        reader_identity.append(threading.get_ident())
        try:
            record = state.get_verification_run(project_id, run["id"])
            assert record is not None
            reader_records.append(record)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the owning test thread
            reader_errors.append(exc)

    def heartbeat_sibling():
        try:
            state.heartbeat_verification_run(sibling_project, sibling["id"], OTHER_OWNER)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the owning test thread
            writer_errors.append(exc)
        finally:
            writer_done.set()

    reader_thread = threading.Thread(target = read_expired_result)
    writer_thread = threading.Thread(target = heartbeat_sibling)
    reader_thread.start()
    assert reader_entered.wait(timeout = 2)
    writer_thread.start()
    writer_completed_while_reader_was_blocked = writer_done.wait(timeout = 1)
    release_reader.set()
    reader_thread.join(timeout = 5)
    writer_thread.join(timeout = 5)

    assert writer_completed_while_reader_was_blocked
    assert reader_errors == []
    assert writer_errors == []
    assert reader_records[0]["status"] == "interrupted"


def test_history_is_pruned_by_project_and_global_evidence_bytes(monkeypatch):
    project_id = _create_project()
    profile = _save(project_id, [_check(log_limit = 4096)])

    def complete_run(owned_project: str, owned_profile: dict) -> str:
        run = _begin(owned_project, owned_profile)
        _complete(
            owned_project,
            run,
            [_result(owned_profile["checks"][0], run, "passed", output = "x" * 2048)],
        )
        return run["id"]

    first_id = complete_run(project_id, profile)
    connection = studio_db.get_connection()
    try:
        evidence_bytes = connection.execute(
            f"""
            SELECT {state._RUN_EVIDENCE_BYTES_SQL}
            FROM agent_verification_runs WHERE id = ?
            """,
            (first_id,),
        ).fetchone()[0]
    finally:
        connection.close()

    monkeypatch.setattr(state, "MAX_PROJECT_RUN_HISTORY_BYTES", evidence_bytes * 2)
    second_id = complete_run(project_id, profile)
    third_id = complete_run(project_id, profile)
    project_history = state.list_verification_runs(project_id)
    assert {record["id"] for record in project_history} == {second_id, third_id}

    monkeypatch.setattr(state, "MAX_PROJECT_RUN_HISTORY_BYTES", evidence_bytes * 10)
    monkeypatch.setattr(state, "MAX_GLOBAL_RUN_HISTORY_BYTES", evidence_bytes * 2)
    other_project = _create_project("global-history-project")
    other_profile = _save(other_project, [_check(log_limit = 4096)])
    newest_id = complete_run(other_project, other_profile)

    connection = studio_db.get_connection()
    try:
        rows = connection.execute(
            f"""
            SELECT id, {state._RUN_EVIDENCE_BYTES_SQL} AS evidence_bytes
            FROM agent_verification_runs WHERE status != 'running'
            """
        ).fetchall()
    finally:
        connection.close()
    assert newest_id in {row["id"] for row in rows}
    assert len(rows) == 2
    assert sum(row["evidence_bytes"] for row in rows) <= evidence_bytes * 2


def test_retention_keeps_the_newest_insert_when_the_wall_clock_moves_backward(monkeypatch):
    clock = [30_000]
    monkeypatch.setattr(state, "_now_ms", lambda: clock[0])
    project_id = _create_project()
    profile = _save(project_id, [_check(log_limit = 4096)])

    older = _begin(project_id, profile)
    older_finished = _complete(
        project_id,
        older,
        [_result(profile["checks"][0], older, "passed", output = "x" * 2048)],
    )
    connection = studio_db.get_connection()
    try:
        evidence_bytes = connection.execute(
            f"""
            SELECT {state._RUN_EVIDENCE_BYTES_SQL}
            FROM agent_verification_runs WHERE id = ?
            """,
            (older["id"],),
        ).fetchone()[0]
    finally:
        connection.close()

    monkeypatch.setattr(state, "MAX_PROJECT_RUN_HISTORY_BYTES", evidence_bytes)
    clock[0] = 20_000
    newer = _begin(project_id, profile)
    completed = _complete(
        project_id,
        newer,
        [_result(profile["checks"][0], newer, "passed", output = "x" * 2048)],
    )

    assert completed["id"] == newer["id"]
    assert completed["historySequence"] > older_finished["historySequence"]
    assert completed["completedAt"] >= completed["results"][0]["completedAt"]
    assert state.get_verification_run(project_id, older["id"]) is None
    assert state.get_verification_run(project_id, newer["id"])["status"] == "passed"
    assert state.list_verification_runs(project_id)[0]["id"] == newer["id"]


def test_long_running_completion_is_retained_as_newest_global_history(monkeypatch):
    old_project = _create_project("long-running-history")
    old_profile = _save(old_project, [_check(log_limit = 4096)])
    old_run = _begin(old_project, old_profile)

    def complete_project(project_id: str) -> dict:
        profile = _save(project_id, [_check(log_limit = 4096)])
        run = _begin(project_id, profile)
        return _complete(
            project_id,
            run,
            [_result(profile["checks"][0], run, "passed", output = "x" * 2048)],
        )

    first = complete_project(_create_project("newer-terminal-first"))
    connection = studio_db.get_connection()
    try:
        evidence_bytes = connection.execute(
            f"""
            SELECT {state._RUN_EVIDENCE_BYTES_SQL}
            FROM agent_verification_runs WHERE id = ?
            """,
            (first["id"],),
        ).fetchone()[0]
    finally:
        connection.close()
    monkeypatch.setattr(state, "MAX_GLOBAL_RUN_HISTORY_BYTES", evidence_bytes * 2)
    second = complete_project(_create_project("newer-terminal-second"))
    completed_old = _complete(
        old_project,
        old_run,
        [_result(old_profile["checks"][0], old_run, "passed", output = "x" * 2048)],
    )

    assert completed_old["historySequence"] > second["historySequence"]
    assert second["historySequence"] > first["historySequence"]
    assert state.get_verification_run(first["projectId"], first["id"]) is None
    assert state.get_verification_run(second["projectId"], second["id"]) is not None
    assert state.get_verification_run(old_project, old_run["id"])["status"] == "passed"


def test_bulk_reconciliation_orders_rollback_runs_by_durable_insertion(monkeypatch):
    clock = [30_000]
    monkeypatch.setattr(state, "_now_ms", lambda: clock[0])
    older_project = _create_project("rollback-expiry-older")
    older_profile = _save(older_project, [_check()])
    older = _begin(older_project, older_profile)

    clock[0] = 20_000
    newer_project = _create_project("rollback-expiry-newer")
    newer_profile = _save(newer_project, [_check()])
    newer = _begin(newer_project, newer_profile)
    connection = studio_db.get_connection()
    try:
        evidence_bytes = connection.execute(
            f"""
            SELECT {state._RUN_EVIDENCE_BYTES_SQL}
            FROM agent_verification_runs WHERE id = ?
            """,
            (newer["id"],),
        ).fetchone()[0]
    finally:
        connection.close()
    interrupted_error_bytes = len(
        "Verification ownership expired before completion.".encode("utf-8")
    )
    monkeypatch.setattr(
        state,
        "MAX_GLOBAL_RUN_HISTORY_BYTES",
        evidence_bytes + interrupted_error_bytes,
    )

    clock[0] = older["leaseExpiresAt"] + 1
    assert state.reconcile_interrupted_verification_runs() == 2
    assert state.get_verification_run(older_project, older["id"]) is None
    retained = state.get_verification_run(newer_project, newer["id"])
    assert retained is not None
    assert retained["status"] == "interrupted"
    assert retained["historySequence"] == 2


def test_prune_failure_rolls_back_candidates_and_exact_delete_guard(monkeypatch):
    project_id = _create_project("prune-rollback-project")
    profile = _save(project_id, [_check()])
    run = _begin(project_id, profile)
    _complete(
        project_id,
        run,
        [_result(profile["checks"][0], run, "passed")],
    )
    connection = state._connection()
    try:
        state._begin_write(connection)
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
        monkeypatch.setattr(state, "MAX_RUN_HISTORY", 0)

        def fail_after_delete_guard_drop(_connection):
            raise RuntimeError("forced delete guard recreation failure")

        monkeypatch.setattr(
            state,
            "_create_run_delete_trigger",
            fail_after_delete_guard_drop,
        )
        with pytest.raises(RuntimeError, match = "forced delete guard"):
            state._prune_runs_locked(connection, project_id)
        connection.rollback()
    finally:
        connection.close()

    inspection = studio_db.get_connection()
    try:
        assert (
            inspection.execute(
                "SELECT 1 FROM agent_verification_runs WHERE id = ?",
                (run["id"],),
            ).fetchone()
            is not None
        )
        trigger_sql = inspection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'trigger'
                AND name = 'trg_agent_verification_runs_no_delete'
            """
        ).fetchone()[0]
        assert state._normalize_schema_sql(trigger_sql) == state._normalize_schema_sql(
            state._RUN_NO_DELETE_TRIGGER_SQL
        )
        assert inspection.execute("PRAGMA schema_version").fetchone()[0] == schema_version
    finally:
        inspection.close()


def test_successful_prune_restores_guard_and_invalidates_schema_cookie(monkeypatch):
    project_id = _create_project("prune-schema-cookie-project")
    profile = _save(project_id, [_check()])
    monkeypatch.setattr(state, "MAX_RUN_HISTORY", 1)
    first = _begin(project_id, profile)
    _complete(
        project_id,
        first,
        [_result(profile["checks"][0], first, "passed")],
    )
    cached_version = max(key[3] for key in state._ready_databases)
    second = _begin(project_id, profile)
    _complete(
        project_id,
        second,
        [_result(profile["checks"][0], second, "passed")],
    )

    assert state.get_verification_run(project_id, first["id"]) is None
    assert state.get_verification_run(project_id, second["id"]) is not None
    connection = state._connection()
    try:
        assert connection.in_transaction
        assert connection.authority_key[3] > cached_version
        state._verify_schema_objects(connection)
    finally:
        connection.close()


def test_retention_foreign_keys_and_project_cascade(monkeypatch):
    assert "sqlite_sequence" not in {
        table_name for table_name, _definition in state._TABLE_DEFINITIONS
    }
    project_id = _create_project()
    profile = _save(project_id, [_check()])
    monkeypatch.setattr(state, "MAX_RUN_HISTORY", 2)
    run_ids = []
    for _index in range(4):
        run = _begin(project_id, profile)
        run_ids.append(run["id"])
        _complete(
            project_id,
            run,
            [_result(profile["checks"][0], run, "passed")],
        )

    history = state.list_verification_runs(project_id, limit = 100)
    assert len(history) == 2
    assert {record["id"] for record in history} == set(run_ids[-2:])
    assert state.get_verification_run(project_id, run_ids[0]) is None

    fence = state.begin_verification_project_deletion(project_id, "delete-cascade")
    connection = studio_db.get_connection()
    try:
        config_fk = connection.execute(
            "PRAGMA foreign_key_list(agent_verification_configs)"
        ).fetchall()
        runs_fk = connection.execute("PRAGMA foreign_key_list(agent_verification_runs)").fetchall()
        fence_fk = connection.execute(
            "PRAGMA foreign_key_list(agent_verification_project_deletions)"
        ).fetchall()
        assert all(
            any(row[2] == "chat_projects" and row[6] == "CASCADE" for row in rows)
            for rows in (config_fk, runs_fk, fence_fk)
        )
        # Raw parent deletion is a storage-owner operation and must retain its
        # declared cascade semantics across every verification authority guard.
        connection.execute("DELETE FROM chat_projects WHERE id = ?", (project_id,))
        connection.commit()
    finally:
        connection.close()

    assert state.get_verification_config(project_id)["revision"] == 0
    assert state.list_verification_runs(project_id) == []
    assert (
        state.finish_verification_project_deletion(
            project_id,
            "delete-cascade",
            fence["revision"],
        )
        is False
    )


def test_schema_constraints_reject_invalid_dynamic_state():
    project_id = _create_project()
    state.get_verification_config(project_id)
    connection = studio_db.get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO agent_verification_configs (
                    project_id, workspace_device_id, workspace_file_id,
                    workspace_revision, checks_json, config_hash, revision, updated_at
                ) VALUES (?, 1, 2, 0, '[]', ?, 1, 1)
                """,
                (project_id, "not-a-sha256"),
            )
        connection.rollback()
    finally:
        connection.close()
