# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import shlex
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from core.agent_workspace.common import AgentWorkspaceError
from core.agent_workspace.project_automation import (
    create_lifecycle_hook,
    create_project_rule,
    create_schedule,
    delete_project_rule,
    get_schedule,
    install_project_skill,
    lease_due_schedules,
    list_lifecycle_hook_runs,
    list_project_rules,
    list_schedule_runs,
    next_schedule_time,
    render_project_skills_catalog,
    reconcile_schedule_run,
    render_project_rules_guidance,
    render_project_skills_guidance,
    resolve_project_rule,
    run_lifecycle_hooks,
    skill_digest,
    update_project_rule,
    update_project_skill,
    update_schedule,
)
from storage import studio_db


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
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )


def _python_command(source: str) -> str:
    executable = shlex.quote(str(Path(sys.executable).resolve()))
    return f"{executable} -c {shlex.quote(source)}"


def _rendered_json(guidance: str) -> list[dict]:
    return json.loads(guidance.split("\n", 1)[1])


def test_rules_are_revisioned_validated_and_rendered_with_a_hard_bound(tmp_path):
    _folder_project(tmp_path)
    rule = create_project_rule(
        "project",
        name = "No destructive shell",
        tool_name = "terminal",
        effect = "deny",
        match_kind = "prefix",
        argument_pattern = "rm ",
        guidance = "Keep destructive commands behind an explicit project decision.",
        priority = 100,
    )
    assert rule["revision"] == 0
    assert list_project_rules("project", enabled_only = True) == [rule]

    updated = update_project_rule(
        rule["id"],
        expected_revision = 0,
        effect = "prompt",
    )
    assert updated is not None
    assert updated["revision"] == 1
    assert resolve_project_rule("project", "terminal", "rm generated.txt") == updated
    assert resolve_project_rule("project", "terminal", "git status") is None
    assert _rendered_json(render_project_rules_guidance("project", limit = 2048))[0] == {
        "arguments": "rm ",
        "effect": "prompt",
        "guidance": "Keep destructive commands behind an explicit project decision.",
        "match": "prefix",
        "name": "No destructive shell",
        "priority": 100,
        "tool": "terminal",
    }
    assert len(render_project_rules_guidance("project", limit = 256).encode()) <= 256

    with pytest.raises(AgentWorkspaceError, match = "another session"):
        update_project_rule(rule["id"], expected_revision = 0, effect = "allow")
    with pytest.raises(AgentWorkspaceError, match = "argument pattern"):
        create_project_rule(
            "project",
            name = "Invalid",
            tool_name = "terminal",
            effect = "allow",
            match_kind = "exact",
        )
    assert delete_project_rule(updated["id"], expected_revision = 1)


def test_skills_require_a_matching_digest_and_only_enabled_guidance_renders(tmp_path):
    _folder_project(tmp_path)
    guidance = "Read the repository map before proposing a cross-cutting change."
    with pytest.raises(AgentWorkspaceError, match = "pinned SHA-256"):
        install_project_skill(
            "project",
            name = "Repository mapper",
            source = "project:skills/repository-mapper/SKILL.md",
            guidance = guidance,
            content_digest = "0" * 64,
        )

    skill = install_project_skill(
        "project",
        name = "Repository mapper",
        description = "Maps architectural truth sources.",
        source = "project:skills/repository-mapper/SKILL.md",
        guidance = guidance,
        content_digest = skill_digest(guidance),
        enabled = False,
    )
    assert render_project_skills_guidance("project") == ""
    enabled = update_project_skill(skill["id"], expected_revision = 0, enabled = True)
    assert enabled is not None
    rendered = _rendered_json(render_project_skills_guidance("project", limit = 4096))
    assert rendered == [
        {
            "description": "Maps architectural truth sources.",
            "guidance": guidance,
            "name": "Repository mapper",
            "sha256": skill_digest(guidance),
            "source": "project:skills/repository-mapper/SKILL.md",
        }
    ]
    with pytest.raises(AgentWorkspaceError, match = "requires a new pinned digest"):
        update_project_skill(enabled["id"], expected_revision = 1, guidance = "changed")
    catalog = _rendered_json(render_project_skills_catalog("project", limit = 4096))
    assert catalog == [
        {
            "description": "Maps architectural truth sources.",
            "id": enabled["id"],
            "name": "Repository mapper",
            "sha256": skill_digest(guidance),
            "source": "project:skills/repository-mapper/SKILL.md",
        }
    ]
    assert guidance not in render_project_skills_catalog("project")


def test_lifecycle_hooks_use_the_project_boundary_and_persist_each_result(
    tmp_path, local_verification_execution_boundary
):
    _folder_project(tmp_path)
    create_lifecycle_hook(
        "project",
        name = "advisory",
        event = "before_agent",
        command = _python_command("print('advisory'); raise SystemExit(2)"),
        position = 0,
        required = False,
    )
    create_lifecycle_hook(
        "project",
        name = "passing",
        event = "before_agent",
        command = _python_command("print('inside', end='')"),
        position = 1,
    )
    create_lifecycle_hook(
        "project",
        name = "blocking",
        event = "before_agent",
        command = _python_command("raise SystemExit(3)"),
        position = 2,
    )
    create_lifecycle_hook(
        "project",
        name = "never reached",
        event = "before_agent",
        command = _python_command("raise SystemExit(0)"),
        position = 3,
    )

    result = run_lifecycle_hooks("project", "before_agent")
    assert result["status"] == "failed"
    assert result["requiredFailure"] is True
    assert [run["hookName"] for run in result["runs"]] == ["advisory", "passing", "blocking"]
    assert result["runs"][1]["result"]["output"] == "inside"
    durable = list_lifecycle_hook_runs("project", invocation_id = result["invocationId"])
    assert len(durable) == 3
    assert {run["status"] for run in durable} == {"failed", "passed"}

    with pytest.raises(AgentWorkspaceError, match = "event is invalid"):
        create_lifecycle_hook(
            "project",
            name = "git pre-commit",
            event = "pre-commit",
            command = "true",
        )


def test_schedule_leases_are_atomic_and_reconcile_to_the_next_occurrence(tmp_path):
    _folder_project(tmp_path)
    schedule = create_schedule(
        "project",
        name = "Hourly verification",
        task_kind = "verification",
        payload = {"selectedNames": ["test"]},
        cadence = {"kind": "hourly", "minute": 0},
        timezone_name = "UTC",
        misfire_policy = "run_once",
        current_time_ms = 0,
    )
    assert schedule["nextRunAt"] == 0

    barrier = threading.Barrier(2)

    def claim(owner: str) -> tuple[str, list[dict]]:
        barrier.wait()
        return owner, lease_due_schedules(
            owner,
            current_time_ms = 500,
            lease_ms = 1000,
            misfire_grace_ms = 1000,
        )

    with ThreadPoolExecutor(max_workers = 2) as pool:
        claims = list(pool.map(claim, ("worker-a", "worker-b")))
    winners = [(owner, leases) for owner, leases in claims if leases]
    assert len(winners) == 1
    winner, leases = winners[0]
    assert leases[0]["payload"] == {"selectedNames": ["test"]}
    assert lease_due_schedules("worker-c", current_time_ms = 500, lease_ms = 1000) == []

    completion = reconcile_schedule_run(
        leases[0]["runId"],
        winner,
        status = "completed",
        current_time_ms = 1000,
    )
    assert completion["run"]["status"] == "completed"
    assert completion["schedule"]["nextRunAt"] == 3_600_000
    assert completion["schedule"]["leased"] is False
    assert list_schedule_runs("project")[0]["scheduledFor"] == 0

    with pytest.raises(AgentWorkspaceError, match = "another session"):
        update_schedule(
            schedule["id"],
            expected_revision = 0,
            name = "stale edit",
            current_time_ms = 1000,
        )


def test_schedule_skip_and_expired_lease_reconciliation_are_durable(tmp_path):
    _folder_project(tmp_path)
    skipped = create_schedule(
        "project",
        name = "Skip stale",
        task_kind = "agent",
        payload = {"prompt": "status"},
        cadence = {"kind": "hourly", "minute": 0},
        timezone_name = "UTC",
        misfire_policy = "skip",
        current_time_ms = 0,
    )
    assert (
        lease_due_schedules(
            "worker",
            current_time_ms = 7_200_000,
            misfire_grace_ms = 1000,
        )
        == []
    )
    after_skip = get_schedule(skipped["id"])
    assert after_skip is not None
    assert after_skip["lastStatus"] == "skipped"
    assert after_skip["nextRunAt"] == 10_800_000
    assert list_schedule_runs("project")[0]["status"] == "skipped"

    one_shot = create_schedule(
        "project",
        name = "Lease recovery",
        task_kind = "agent",
        payload = {"prompt": "continue"},
        cadence = {"kind": "once", "at": 20_000_000},
        timezone_name = "UTC",
        misfire_policy = "run_once",
        current_time_ms = 10_000_000,
    )
    first = lease_due_schedules(
        "worker-a",
        current_time_ms = 20_000_000,
        lease_ms = 1000,
        misfire_grace_ms = 10_000,
    )
    assert [lease["id"] for lease in first] == [one_shot["id"]]
    second = lease_due_schedules(
        "worker-b",
        current_time_ms = 20_001_001,
        lease_ms = 1000,
        misfire_grace_ms = 10_000,
    )
    assert [lease["id"] for lease in second] == [one_shot["id"]]
    statuses = [run["status"] for run in list_schedule_runs("project")]
    assert "interrupted" in statuses
    assert "leased" in statuses


def test_weekly_schedule_uses_the_named_timezone_and_skips_nonexistent_local_time():
    zone = ZoneInfo("America/New_York")
    friday = datetime(2026, 8, 28, 12, 0, tzinfo = zone)
    monday = datetime(2026, 8, 31, 9, 30, tzinfo = zone)
    assert next_schedule_time(
        {"kind": "weekly", "weekday": 0, "hour": 9, "minute": 30},
        "America/New_York",
        int(friday.timestamp() * 1000),
    ) == int(monday.timestamp() * 1000)

    before_spring_forward = datetime(2026, 3, 7, 3, 0, tzinfo = zone)
    monday_after_gap = datetime(2026, 3, 9, 2, 30, tzinfo = zone)
    assert next_schedule_time(
        {"kind": "daily", "hour": 2, "minute": 30},
        "America/New_York",
        int(before_spring_forward.timestamp() * 1000),
    ) == int(monday_after_gap.timestamp() * 1000)
