# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import shlex
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import HTTPException

from core.agent_workspace import state
from core.agent_workspace import background as background_module
from core.agent_workspace import verification as verification_module
from core.agent_workspace.background import BackgroundTaskManager
from core.agent_workspace.common import AgentWorkspaceError
from core.agent_workspace.common import (
    run_bounded,
    workspace_fingerprint,
    workspace_fingerprint_complete,
)
from core.agent_workspace.state import (
    begin_verification_run,
    claim_background_task,
    create_background_task,
    create_plan,
    finish_verification_run,
    get_background_task,
    get_verification_config,
    set_verification_config,
    update_background_task,
    update_plan_status,
    update_plan_task,
)
from core.agent_workspace.verification import (
    GOAL_COMPLETION_VERIFICATION_DETAIL,
    execute_check,
    require_goal_completion_verification,
    run_project_verification,
    verification_run_with_freshness,
)
from routes import chat_history
from storage import studio_db
from storage.api_usage_db import ApiUsageReceipt, record_api_usage


def _folder_project(root: Path, project_id: str = "project") -> dict:
    metadata = root.stat()
    return studio_db.upsert_chat_project(
        {
            "id": project_id,
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "goal": "Ship the workspace",
            "goalStatus": "active",
            "goalUpdatedAt": 7,
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )


def _python_command(source: str) -> str:
    return f"{shlex.quote(str(Path(sys.executable).resolve()))} -c {shlex.quote(source)}"


def _required_check(name: str = "test") -> dict:
    return {
        "name": name,
        "kind": "test",
        "command": _python_command("print('passed')"),
        "required": True,
        "timeoutSeconds": 10,
        "logLimitBytes": 1024,
    }


def _record_verification(
    root: Path,
    *,
    config_revision: int,
    status: str = "passed",
    result_status: str = "passed",
    names: tuple[str, ...] = ("test",),
    worktree_id: str | None = None,
    fingerprint: str | None = None,
) -> dict:
    evidence = fingerprint or workspace_fingerprint(root)
    run = begin_verification_run(
        "project",
        evidence,
        worktree_id,
        config_revision = config_revision,
    )
    return finish_verification_run(
        run["id"],
        status,
        evidence,
        [{"name": name, "required": True, "status": result_status} for name in names],
    )


def test_verification_persists_bounded_evidence_and_detects_staleness(
    tmp_path, local_verification_execution_boundary
):
    _folder_project(tmp_path)
    (tmp_path / "source.txt").write_text("one", encoding = "utf-8")
    checks = [
        {
            "name": "test",
            "kind": "test",
            "command": _python_command("print('passed')"),
            "required": True,
            "timeoutSeconds": 10,
            "logLimitBytes": 1024,
        }
    ]
    first_config = set_verification_config("project", checks, require_for_goal_completion = True)
    second_config = set_verification_config("project", checks, require_for_goal_completion = True)

    assert first_config["revision"] == 1
    assert second_config["revision"] == 2
    assert get_verification_config("project") == {
        "projectId": "project",
        "checks": checks,
        "requireForGoalCompletion": True,
        "revision": 2,
        "updatedAt": second_config["updatedAt"],
    }
    run = run_project_verification("project")
    assert run["status"] == "passed"
    assert run["configRevision"] == 2
    assert run["results"][0]["exitCode"] == 0
    assert run["results"][0]["output"] == "passed\n"
    assert verification_run_with_freshness(run["id"])["stale"] is False

    (tmp_path / "source.txt").write_text("two", encoding = "utf-8")
    assert verification_run_with_freshness(run["id"])["stale"] is True


def test_pre_agent_studio_database_migrates_with_api_usage_and_agent_state(tmp_path, monkeypatch):
    studio_home = tmp_path / "studio"
    projects_home = tmp_path / "projects"
    studio_home.mkdir()
    projects_home.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(projects_home))
    db_path = studio_db.studio_db_path()
    db_path.parent.mkdir(parents = True, exist_ok = True)

    legacy = sqlite3.connect(db_path)
    try:
        legacy.executescript(
            """
            CREATE TABLE chat_projects (
                id TEXT NOT NULL PRIMARY KEY,
                name TEXT NOT NULL,
                instructions TEXT,
                root_path TEXT,
                archived INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            INSERT INTO chat_projects VALUES (
                'legacy', 'Legacy', 'keep', NULL, 0, 1, 1
            );
            CREATE TABLE api_usage_events (
                id TEXT NOT NULL PRIMARY KEY,
                subject TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            ) WITHOUT ROWID;
            INSERT INTO api_usage_events VALUES (
                'old-usage', 'subject', '/v1/chat/completions', 'old-model',
                'completed', 2, 3, 5, 1
            );
            """
        )
        legacy.commit()
    finally:
        legacy.close()

    studio_db._schema_ready = False
    state._READY_DATABASES.discard(str(db_path))
    conn = studio_db.get_connection()
    try:
        project_columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_projects)")}
        assert {
            "workspace_kind",
            "workspace_device_id",
            "workspace_file_id",
            "managed_root_device_id",
            "managed_root_file_id",
            "goal",
            "goal_status",
            "goal_updated_at",
            "goal_revision",
        } <= project_columns
        assert (
            conn.execute(
                "SELECT total_tokens FROM api_usage_events WHERE id = 'old-usage'"
            ).fetchone()[0]
            == 5
        )
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(api_usage_events)").fetchall()}
        assert "idx_api_usage_events_subject_created_at" in indexes
    finally:
        conn.close()

    project = studio_db.ensure_chat_project_workspace("legacy")
    assert project is not None
    assert project["workspaceKind"] == "managed"
    assert project["goalRevision"] == 0
    assert project["managedRootDeviceId"] is not None
    assert project["managedRootFileId"] is not None
    assert project["workspaceDeviceId"] is not None
    assert project["workspaceFileId"] is not None
    root = Path(project["rootPath"])
    assert root.is_dir()
    assert (root / "sandbox").is_dir()

    verification = set_verification_config("legacy", [], require_for_goal_completion = False)
    plan = create_plan(
        "legacy",
        "Migration plan",
        None,
        [{"title": "Verify migration"}],
    )
    task = create_background_task("legacy", "verification", {})
    assert verification["revision"] == 1
    assert plan["tasks"][0]["title"] == "Verify migration"
    assert task["status"] == "queued"

    assert record_api_usage(
        ApiUsageReceipt(
            id = "new-usage",
            subject = "subject",
            endpoint = "/v1/responses",
            model = "new-model",
            status = "completed",
            prompt_tokens = 7,
            completion_tokens = 11,
            total_tokens = 18,
            created_at = 2,
        )
    )

    deleted = studio_db.delete_chat_project("legacy", delete_files = False)
    assert deleted is not None
    assert deleted["id"] == "legacy"
    assert root.is_dir()

    conn = state.connection()
    try:
        expected_agent_tables = {
            "agent_verification_configs",
            "agent_verification_runs",
            "agent_plans",
            "agent_plan_tasks",
            "agent_background_tasks",
            "agent_git_checkpoints",
            "agent_worktrees",
        }
        agent_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'agent_%'"
            ).fetchall()
        }
        assert expected_agent_tables <= agent_tables
        for table in (
            "agent_verification_configs",
            "agent_plans",
            "agent_plan_tasks",
            "agent_background_tasks",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM api_usage_events").fetchone()[0] == 2
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_verification_state_schema_migrates_policy_and_run_revisions(tmp_path):
    _folder_project(tmp_path)
    fingerprint = workspace_fingerprint(tmp_path)
    conn = studio_db.get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE agent_verification_configs (
                project_id TEXT NOT NULL PRIMARY KEY,
                checks_json TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE agent_verification_runs (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL,
                worktree_id TEXT,
                status TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                final_fingerprint TEXT,
                results_json TEXT NOT NULL,
                started_at INTEGER NOT NULL,
                completed_at INTEGER
            );
            """
        )
        conn.execute(
            """
            INSERT INTO agent_verification_configs(project_id, checks_json, updated_at)
            VALUES ('project', '[]', 1)
            """
        )
        conn.execute(
            """
            INSERT INTO agent_verification_runs(
                id, project_id, worktree_id, status, source_fingerprint,
                final_fingerprint, results_json, started_at, completed_at
            ) VALUES ('legacy-run', 'project', NULL, 'passed', ?, ?, '[]', 1, 2)
            """,
            (fingerprint, fingerprint),
        )
        conn.commit()
        key = state._database_key(conn)
    finally:
        conn.close()

    state._READY_DATABASES.discard(key)
    migrated = state.connection()
    try:
        config_columns = {
            row[1]
            for row in migrated.execute("PRAGMA table_info(agent_verification_configs)").fetchall()
        }
        run_columns = {
            row[1]
            for row in migrated.execute("PRAGMA table_info(agent_verification_runs)").fetchall()
        }
    finally:
        migrated.close()

    assert {"require_for_goal_completion", "revision"} <= config_columns
    assert "config_revision" in run_columns
    assert get_verification_config("project")["requireForGoalCompletion"] is False
    assert get_verification_config("project")["revision"] == 0
    assert state.get_verification_run("legacy-run")["configRevision"] == 0


def test_verification_config_revision_is_monotonic_under_concurrent_saves(tmp_path):
    _folder_project(tmp_path)
    ready = state.connection()
    ready.close()
    barrier = threading.Barrier(4)

    def save(index: int) -> int:
        barrier.wait(timeout = 5)
        config = set_verification_config(
            "project",
            [_required_check(f"check-{index}")],
            require_for_goal_completion = bool(index % 2),
        )
        return config["revision"]

    with ThreadPoolExecutor(max_workers = 4) as pool:
        revisions = list(pool.map(save, range(4)))

    assert sorted(revisions) == [1, 2, 3, 4]
    assert get_verification_config("project")["revision"] == 4


def test_verification_config_rejects_stale_saves_and_invalid_checks(tmp_path):
    _folder_project(tmp_path)
    initial = set_verification_config(
        "project",
        [_required_check("unit")],
        expected_revision = 0,
    )

    with pytest.raises(AgentWorkspaceError, match = "another session"):
        set_verification_config(
            "project",
            [_required_check("lint")],
            expected_revision = 0,
        )
    assert get_verification_config("project")["checks"][0]["name"] == "unit"

    updated = set_verification_config(
        "project",
        [_required_check("lint")],
        expected_revision = initial["revision"],
    )
    assert updated["revision"] == 2

    blank = _required_check("blank")
    blank["command"] = "   "
    with pytest.raises(AgentWorkspaceError, match = "cannot be blank"):
        set_verification_config("project", [blank])
    with pytest.raises(AgentWorkspaceError, match = "must be unique"):
        set_verification_config(
            "project",
            [_required_check("Test"), _required_check(" test ")],
        )


def test_verification_run_rejects_config_drift_and_unknown_selections(tmp_path):
    _folder_project(tmp_path)
    initial = set_verification_config("project", [_required_check("unit")])

    with pytest.raises(AgentWorkspaceError, match = "not configured"):
        run_project_verification(
            "project",
            ["unit", "typo"],
            config_revision = initial["revision"],
        )

    changed = set_verification_config("project", [_required_check("unit")])
    with pytest.raises(AgentWorkspaceError, match = "changed after"):
        run_project_verification(
            "project",
            ["unit"],
            config_revision = initial["revision"],
        )
    assert changed["revision"] == initial["revision"] + 1


def test_goal_completion_gate_accepts_only_current_complete_primary_evidence(tmp_path):
    _folder_project(tmp_path)
    checks = [_required_check("test"), _required_check("lint")]
    config = set_verification_config("project", checks, require_for_goal_completion = True)

    _record_verification(
        tmp_path,
        config_revision = config["revision"],
        names = ("test", "lint"),
    )
    require_goal_completion_verification("project")

    (tmp_path / "changed.py").write_text("changed", encoding = "utf-8")
    with pytest.raises(AgentWorkspaceError, match = "fresh passing verification run") as blocked:
        require_goal_completion_verification("project")
    assert str(blocked.value) == GOAL_COMPLETION_VERIFICATION_DETAIL
    assert str(tmp_path) not in str(blocked.value)


def test_goal_completion_gate_rejects_worktree_only_and_partial_evidence(tmp_path):
    _folder_project(tmp_path)
    checks = [_required_check("test"), _required_check("lint")]
    config = set_verification_config("project", checks, require_for_goal_completion = True)

    _record_verification(
        tmp_path,
        config_revision = config["revision"],
        names = ("test", "lint"),
        worktree_id = "worktree-only",
    )
    with pytest.raises(AgentWorkspaceError, match = "main project workspace"):
        require_goal_completion_verification("project")

    _record_verification(
        tmp_path,
        config_revision = config["revision"],
        names = ("test",),
    )
    with pytest.raises(AgentWorkspaceError, match = "every required check"):
        require_goal_completion_verification("project")


def test_goal_completion_gate_uses_latest_primary_run(tmp_path):
    _folder_project(tmp_path)
    config = set_verification_config(
        "project", [_required_check()], require_for_goal_completion = True
    )
    passing = _record_verification(tmp_path, config_revision = config["revision"])
    failing = _record_verification(
        tmp_path,
        config_revision = config["revision"],
        status = "failed",
        result_status = "failed",
    )
    conn = state.connection()
    try:
        conn.execute(
            "UPDATE agent_verification_runs SET started_at = ? WHERE id IN (?, ?)",
            (passing["startedAt"], passing["id"], failing["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(AgentWorkspaceError, match = "fresh passing verification run"):
        require_goal_completion_verification("project")


def test_goal_completion_gate_is_optional(tmp_path):
    _folder_project(tmp_path)
    set_verification_config("project", [_required_check()], require_for_goal_completion = False)

    require_goal_completion_verification("project")


def test_goal_completion_gate_rejects_config_changes_and_no_checks(tmp_path):
    _folder_project(tmp_path)
    checks = [_required_check()]
    initial = set_verification_config("project", checks, require_for_goal_completion = True)
    _record_verification(tmp_path, config_revision = initial["revision"])

    changed = set_verification_config("project", checks, require_for_goal_completion = True)
    assert changed["revision"] > initial["revision"]
    with pytest.raises(AgentWorkspaceError, match = "verification-setting change"):
        require_goal_completion_verification("project")

    empty = set_verification_config("project", [], require_for_goal_completion = True)
    assert empty["revision"] > changed["revision"]
    with pytest.raises(AgentWorkspaceError, match = "fresh passing verification run"):
        require_goal_completion_verification("project")


def test_concurrent_goal_edits_use_server_revision_and_admit_one(tmp_path, monkeypatch):
    project = _folder_project(tmp_path)
    barrier = threading.Barrier(2)
    real_get = chat_history.get_chat_project

    def synchronized_get(project_id):
        value = real_get(project_id)
        barrier.wait(timeout = 5)
        return value

    monkeypatch.setattr(chat_history, "get_chat_project", synchronized_get)

    def patch(goal):
        try:
            return chat_history.patch_project(
                project["id"],
                chat_history.ChatProjectPatch(
                    goal = goal,
                    goalUpdatedAt = 1,
                    updatedAt = 1,
                ),
                current_subject = "tester",
            )
        except Exception as exc:  # noqa: BLE001 - the result records the losing request
            return exc

    with ThreadPoolExecutor(max_workers = 2) as pool:
        results = list(pool.map(patch, ("first", "second")))

    successes = [result for result in results if isinstance(result, chat_history.ChatProject)]
    conflicts = [result for result in results if isinstance(result, HTTPException)]
    stored = studio_db.get_chat_project(project["id"])

    assert len(successes) == 1
    assert len(conflicts) == 1
    assert conflicts[0].status_code == 409
    assert stored["goal"] in {"first", "second"}
    assert stored["goalRevision"] == 1
    assert stored["goalUpdatedAt"] > 1
    assert stored["updatedAt"] > 1


def test_project_create_route_cannot_upsert_or_bypass_goal_completion(tmp_path):
    project = _folder_project(tmp_path)
    set_verification_config(
        project["id"],
        [_required_check()],
        require_for_goal_completion = True,
    )

    with pytest.raises(HTTPException) as blocked:
        chat_history.save_project(
            chat_history.ChatProjectCreate(
                id = project["id"],
                name = "Replacement",
                goal = "Overwrite the goal",
                goalStatus = "completed",
                goalUpdatedAt = 999,
                archived = False,
                createdAt = 999,
                updatedAt = 999,
            ),
            current_subject = "tester",
        )

    assert blocked.value.status_code == 409
    stored = studio_db.get_chat_project(project["id"])
    assert stored["name"] == "Project"
    assert stored["goal"] == "Ship the workspace"
    assert stored["goalStatus"] == "active"
    assert stored["goalRevision"] == 0


def test_goal_completion_rejects_policy_change_between_check_and_commit(tmp_path, monkeypatch):
    project = _folder_project(tmp_path)
    initial = set_verification_config(
        project["id"],
        [],
        require_for_goal_completion = False,
    )
    real_require = chat_history.require_goal_completion_verification

    def change_policy_after_check(project_id):
        checked_revision = real_require(project_id)
        assert checked_revision == initial["revision"]
        set_verification_config(
            project_id,
            [_required_check()],
            require_for_goal_completion = True,
        )
        return checked_revision

    monkeypatch.setattr(
        chat_history,
        "require_goal_completion_verification",
        change_policy_after_check,
    )

    with pytest.raises(HTTPException, match = "verification policy changed") as blocked:
        chat_history.patch_project(
            project["id"],
            chat_history.ChatProjectPatch(goalStatus = "completed"),
            current_subject = "tester",
        )

    assert blocked.value.status_code == 409
    stored = studio_db.get_chat_project(project["id"])
    assert stored["goalStatus"] == "active"
    assert stored["goalRevision"] == 0


@pytest.mark.parametrize(
    ("run_status", "result_status"),
    [
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("failed", "timed_out"),
    ],
)
def test_goal_completion_gate_rejects_failed_cancelled_and_timed_out_results(
    tmp_path, run_status, result_status
):
    _folder_project(tmp_path)
    config = set_verification_config(
        "project", [_required_check()], require_for_goal_completion = True
    )
    _record_verification(
        tmp_path,
        config_revision = config["revision"],
        status = run_status,
        result_status = result_status,
    )

    with pytest.raises(AgentWorkspaceError, match = "fresh passing verification run"):
        require_goal_completion_verification("project")


def test_goal_completion_gate_rejects_incomplete_fingerprint_evidence(tmp_path):
    _folder_project(tmp_path)
    config = set_verification_config(
        "project", [_required_check()], require_for_goal_completion = True
    )
    _record_verification(
        tmp_path,
        config_revision = config["revision"],
        fingerprint = "badc0ffe" + "0" * 56,
    )

    with pytest.raises(AgentWorkspaceError, match = "fresh passing verification run"):
        require_goal_completion_verification("project")


def test_verification_timeout_cancels_process_and_bounds_output(
    tmp_path, local_verification_execution_boundary
):
    cancel = threading.Event()
    timer = threading.Timer(0.15, cancel.set)
    timer.start()
    try:
        result = execute_check(
            {
                "name": "cancel",
                "command": _python_command(
                    "import sys,time; print('x'*5000); sys.stdout.flush(); time.sleep(10)"
                ),
                "timeoutSeconds": 5,
                "logLimitBytes": 1024,
            },
            root = tmp_path,
            cancel_event = cancel,
            run_id = "cancel-run",
        )
    finally:
        timer.cancel()

    assert result["status"] == "cancelled"
    assert result["outputTruncated"] is True
    assert len(result["output"].encode("utf-8")) <= 1024


def test_verification_cancelled_while_waiting_for_workspace_slot(tmp_path, monkeypatch):
    cancel = threading.Event()

    class WaitingBoundary:
        closed = False

        def acquire_execution_slot(self, cancel_event):
            assert cancel_event is cancel
            cancel_event.set()
            return False

        def close(self):
            self.closed = True

    boundary = WaitingBoundary()
    monkeypatch.setattr(
        verification_module.ProjectExecutionBoundary,
        "open",
        lambda *_args, **_kwargs: boundary,
    )
    monkeypatch.setattr(
        verification_module,
        "spawn_on_lifetime_thread",
        lambda *_args, **_kwargs: pytest.fail("cancelled checks must not spawn"),
    )

    result = execute_check(
        _required_check(),
        root = tmp_path,
        cancel_event = cancel,
        run_id = "waiting-run",
    )

    assert result["status"] == "cancelled"
    assert result["exitCode"] is None
    assert result["output"] == ""
    assert boundary.closed is True


def test_verification_is_stale_when_workspace_changes_during_run(
    tmp_path, local_verification_execution_boundary
):
    _folder_project(tmp_path)
    (tmp_path / "source.txt").write_text("before", encoding = "utf-8")
    set_verification_config(
        "project",
        [
            {
                "name": "mutating-check",
                "kind": "test",
                "command": _python_command(
                    "from pathlib import Path; Path('source.txt').write_text('during')"
                ),
                "required": True,
                "timeoutSeconds": 10,
                "logLimitBytes": 1024,
            }
        ],
    )

    run = run_project_verification("project")
    refreshed = verification_run_with_freshness(run["id"])

    assert run["changedDuringRun"] is True
    assert run["stale"] is True
    assert refreshed["changedDuringRun"] is True
    assert refreshed["stale"] is True


def test_non_git_fingerprint_hashes_content_when_size_and_mtime_are_restored(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"first")
    original_times = source.stat().st_atime_ns, source.stat().st_mtime_ns
    before = workspace_fingerprint(tmp_path)

    source.write_bytes(b"other")
    os.utime(source, ns = original_times)
    after = workspace_fingerprint(tmp_path)

    assert workspace_fingerprint_complete(before) is True
    assert workspace_fingerprint_complete(after) is True
    assert after != before


def test_large_untracked_content_marks_verification_evidence_unverifiable(
    tmp_path, local_verification_execution_boundary
):
    code, output, _ = run_bounded(["git", "init", "-q"], cwd = tmp_path)
    assert code == 0, output
    (tmp_path / "large.bin").write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    fingerprint = workspace_fingerprint(tmp_path)

    assert workspace_fingerprint_complete(fingerprint) is False

    _folder_project(tmp_path)
    set_verification_config(
        "project",
        [
            {
                "name": "test",
                "kind": "test",
                "command": _python_command("print('passed')"),
                "required": True,
                "timeoutSeconds": 10,
                "logLimitBytes": 1024,
            }
        ],
    )
    run = run_project_verification("project")

    assert run["status"] == "passed"
    assert run["evidenceComplete"] is False
    assert run["unverifiable"] is True
    assert run["stale"] is True


def test_plans_are_goal_linked_durable_and_revision_guarded(tmp_path):
    _folder_project(tmp_path)
    plan = create_plan(
        "project",
        "Implementation",
        "Ship the workspace",
        [{"title": "Build it"}, {"title": "Verify it"}],
        goal_updated_at = 7,
    )
    assert plan["goalUpdatedAt"] == 7
    assert plan["revision"] == 0

    updated = update_plan_task(
        plan["id"],
        plan["tasks"][0]["id"],
        status = "completed",
        expected_revision = 0,
    )
    assert updated["revision"] == 1
    assert updated["tasks"][0]["status"] == "completed"

    with pytest.raises(AgentWorkspaceError, match = "another session"):
        update_plan_status(plan["id"], "completed", expected_revision = 0)


def test_plan_task_blocker_clears_explicitly_and_when_work_resumes(tmp_path):
    _folder_project(tmp_path)
    plan = create_plan(
        "project",
        "Implementation",
        "Ship the workspace",
        [{"title": "Build it", "status": "blocked", "blocker": "Need input"}],
    )
    task_id = plan["tasks"][0]["id"]

    cleared = update_plan_task(
        plan["id"],
        task_id,
        blocker = None,
        expected_revision = 0,
    )
    assert cleared["tasks"][0]["blocker"] is None
    assert cleared["completionSummary"]["blockers"] == []

    blocked = update_plan_task(
        plan["id"],
        task_id,
        status = "blocked",
        blocker = "Waiting",
        expected_revision = 1,
    )
    resumed = update_plan_task(
        plan["id"],
        task_id,
        status = "running",
        expected_revision = blocked["revision"],
    )
    assert resumed["tasks"][0]["blocker"] is None
    assert resumed["completionSummary"]["blockers"] == []


def test_restart_recovery_interrupts_running_but_preserves_queued(tmp_path):
    _folder_project(tmp_path)
    queued = create_background_task("project", "verification", {})
    running = create_background_task("project", "verification", {})
    update_background_task(running["id"], "running")

    conn = state.connection()
    try:
        key = state._database_key(conn)
    finally:
        conn.close()
    state._READY_DATABASES.discard(key)
    conn = state.connection()
    conn.close()

    assert get_background_task(queued["id"])["status"] == "queued"
    assert get_background_task(running["id"])["status"] == "interrupted"


def test_restart_recovery_cancels_queued_descendants_of_terminal_parent(tmp_path):
    _folder_project(tmp_path)
    parent = create_background_task("project", "agent", {})
    child = create_background_task(
        "project",
        "agent",
        {},
        parent_task_id = parent["id"],
        root_task_id = parent["id"],
    )
    grandchild = create_background_task(
        "project",
        "agent",
        {},
        parent_task_id = child["id"],
        root_task_id = parent["id"],
    )
    update_background_task(parent["id"], "cancelled")

    conn = state.connection()
    try:
        key = state._database_key(conn)
    finally:
        conn.close()
    state._READY_DATABASES.discard(key)
    conn = state.connection()
    conn.close()

    for task in (child, grandchild):
        recovered = get_background_task(task["id"])
        assert recovered["status"] == "cancelled"
        assert recovered["cancelRequested"] is True
        assert recovered["error"] == "Parent task stopped before this child could run."


def test_start_rejects_and_cancels_queued_child_of_terminal_parent(tmp_path):
    _folder_project(tmp_path)
    parent = create_background_task("project", "agent", {})
    child = create_background_task(
        "project",
        "agent",
        {},
        parent_task_id = parent["id"],
        root_task_id = parent["id"],
    )
    update_background_task(parent["id"], "cancelled")

    with pytest.raises(AgentWorkspaceError, match = "parent agent is no longer active"):
        claim_background_task(child["id"])

    recovered = get_background_task(child["id"])
    assert recovered["status"] == "cancelled"
    assert recovered["cancelRequested"] is True


def test_terminal_background_state_rejects_late_worker_update(tmp_path):
    _folder_project(tmp_path)
    task = create_background_task("project", "verification", {})
    update_background_task(task["id"], "cancelled")

    with pytest.raises(AgentWorkspaceError, match = "cannot transition"):
        update_background_task(task["id"], "completed", result = {"ok": True})


def test_concurrent_background_start_claims_task_once(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    call_count = 0
    call_lock = threading.Lock()

    def fake_verification(*args, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        entered.set()
        release.wait(timeout = 5)
        return {"status": "passed"}

    monkeypatch.setattr(background_module, "run_project_verification", fake_verification)
    manager = BackgroundTaskManager(max_workers = 1)
    task = manager.enqueue_verification("project", start = False)
    barrier = threading.Barrier(3)
    results = []

    def start_once():
        barrier.wait()
        try:
            results.append(("started", manager.start(task["id"])))
        except AgentWorkspaceError as exc:
            results.append(("rejected", str(exc)))

    workers = [threading.Thread(target = start_once) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout = 5)

    assert [kind for kind, _ in results].count("started") == 1
    assert [kind for kind, _ in results].count("rejected") == 1
    assert entered.wait(timeout = 2)
    assert call_count == 1
    release.set()
    manager._executor.shutdown(wait = True)


def test_queued_verification_is_bound_to_its_config_revision(tmp_path):
    _folder_project(tmp_path)
    initial = set_verification_config("project", [_required_check("unit")])
    manager = BackgroundTaskManager(max_workers = 1)
    try:
        queued = manager.enqueue_verification("project", start = False)
        assert queued["payload"]["configRevision"] == initial["revision"]
        set_verification_config("project", [_required_check("lint")])

        manager.start(queued["id"])
        current = get_background_task(queued["id"])
        for _ in range(300):
            if current["status"] in {"failed", "completed", "cancelled"}:
                break
            time.sleep(0.01)
            current = get_background_task(queued["id"])

        assert current["status"] == "failed"
        assert "changed after" in current["error"]
    finally:
        manager._executor.shutdown(wait = True)


def test_project_deletion_cancels_queued_tasks_and_waits_for_running_worker(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    entered = threading.Event()
    finished = threading.Event()

    def cancellable_verification(*args, cancel_event, **kwargs):
        entered.set()
        assert cancel_event.wait(timeout = 5)
        time.sleep(0.05)
        finished.set()
        return {"status": "cancelled"}

    monkeypatch.setattr(background_module, "run_project_verification", cancellable_verification)
    manager = BackgroundTaskManager(max_workers = 1)
    try:
        running = manager.enqueue_verification("project", start = True)
        queued = manager.enqueue_verification("project", start = False)
        assert entered.wait(timeout = 2)

        manager.begin_project_deletion("project")
        try:
            with pytest.raises(AgentWorkspaceError, match = "deletion is in progress"):
                manager.enqueue_verification("project", start = False)
            stopped = manager.cancel_project_tasks_and_wait("project", timeout_seconds = 5)
        finally:
            manager.finish_project_deletion("project")

        assert finished.is_set()
        assert {task["id"] for task in stopped} == {running["id"], queued["id"]}
        assert get_background_task(running["id"])["status"] == "cancelled"
        assert get_background_task(queued["id"])["status"] == "cancelled"
    finally:
        manager._executor.shutdown(wait = True)


def test_project_deletion_fence_rejects_new_foreground_verification(tmp_path):
    from core.agent_workspace import verification as verification_module

    _folder_project(tmp_path)
    set_verification_config(
        "project",
        [
            {
                "name": "test",
                "kind": "test",
                "command": _python_command("print('passed')"),
                "required": True,
                "timeoutSeconds": 10,
                "logLimitBytes": 1024,
            }
        ],
    )

    verification_module.begin_project_deletion("project")
    try:
        with pytest.raises(AgentWorkspaceError, match = "deletion is in progress"):
            run_project_verification("project")
    finally:
        verification_module.finish_project_deletion("project")


def test_verification_child_environment_excludes_backend_credentials(
    tmp_path, monkeypatch, local_verification_execution_boundary
):
    real_home = str(tmp_path / "real-home")
    secrets = {
        "OPENAI_API_KEY": "openai-secret",
        "GH_TOKEN": "github-secret",
        "UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET": "lease-secret",
        "SSH_AUTH_SOCK": str(tmp_path / "agent.sock"),
        "HTTP_PROXY": "https://proxy-user:proxy-pass@example.invalid",
        "HOME": real_home,
        "USERPROFILE": real_home,
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    names = [*secrets, "TMP", "TEMP", "TMPDIR"]
    source = "import json,os; print(json.dumps({k:os.environ.get(k) for k in %r}))" % names

    result = execute_check(
        {
            "name": "environment",
            "command": _python_command(source),
            "timeoutSeconds": 5,
            "logLimitBytes": 8192,
        },
        root = tmp_path,
        cancel_event = threading.Event(),
        run_id = "environment-run",
    )
    child = json.loads(result["output"])

    for name in (
        "OPENAI_API_KEY",
        "GH_TOKEN",
        "UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET",
        "SSH_AUTH_SOCK",
        "HTTP_PROXY",
    ):
        assert child[name] is None
    assert child["HOME"] != real_home
    assert child["USERPROFILE"] != real_home
    assert child["TMP"] == child["HOME"]
    assert child["TEMP"] == child["HOME"]
    assert child["TMPDIR"] == child["HOME"]
    assert not Path(child["HOME"]).is_relative_to(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason = "POSIX process-group behavior")
def test_bounded_runner_reaps_grandchild_after_shell_leader_exits(tmp_path):
    from core.agent_workspace.common import run_bounded
    from utils.process_lifetime import _group_has_members

    started = time.monotonic()
    code, output, _ = run_bounded(
        ["/bin/sh", "-c", 'printf \'%s \' "$$"; sleep 30 & echo "$!"'],
        cwd = tmp_path,
        timeout_seconds = 5,
        output_limit = 1024,
    )
    group_id, child_pid = [int(value) for value in output.split()]

    assert code == 0
    assert child_pid > 1
    assert time.monotonic() - started < 4
    assert not _group_has_members(group_id)


def test_windows_cleanup_uses_captured_pid_after_leader_exit(monkeypatch):
    from core.agent_workspace import common

    calls = []

    class ExitedProcess:
        pid = 42

        @staticmethod
        def poll():
            return 0

        @staticmethod
        def kill():
            raise AssertionError("An exited leader must not be signalled directly")

    monkeypatch.setattr(common.os, "name", "nt")
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda argv, **kwargs: calls.append((argv, kwargs)),
    )

    common._terminate_bounded_process(ExitedProcess())

    assert calls[0][0] == ["taskkill", "/PID", "42", "/T", "/F"]
