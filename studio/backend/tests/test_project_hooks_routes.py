# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated transport coverage for project hook review and trust controls."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from routes import project_hooks
from storage import project_hook_trust_db, studio_db


@pytest.fixture(autouse = True)
def _supported_command_runtime(monkeypatch):
    if os.name == "nt":
        pytest.skip("Hook source review requires native POSIX descriptor discovery")
    monkeypatch.setattr(
        project_hooks,
        "execution_status",
        lambda: {"available": True, "backend": "test", "reason": None},
    )


def _create_project(project_id: str = "hooks-route-project") -> str:
    connection = studio_db.get_connection()
    try:
        connection.execute(
            """
            INSERT INTO chat_projects (
                id, name, instructions, archived, created_at, updated_at
            ) VALUES (?, ?, '', 0, 1, 1)
            """,
            (project_id, "Hooks Route Project"),
        )
        connection.commit()
    finally:
        connection.close()
    return project_id


def _write_hooks(root: Path, command: str = "python3 .codex/check.py") -> None:
    path = root / ".codex" / "hooks.json"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(
        json.dumps(
            {
                "description": "Repository checks",
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": command,
                                    "timeout": 30,
                                }
                            ],
                        }
                    ]
                },
            },
            separators = (",", ":"),
        ),
        encoding = "utf-8",
    )


@contextmanager
def _workspace(root: Path, revision: int = 7):
    metadata = root.stat()
    yield SimpleNamespace(
        root = root,
        device_id = metadata.st_dev,
        file_id = metadata.st_ino,
        revision = revision,
    )


def _bind_project(
    monkeypatch,
    project_id: str,
    root: Path,
    *,
    revision: int = 7,
) -> None:
    monkeypatch.setattr(
        project_hooks,
        "get_chat_project",
        lambda requested: (
            {
                "id": project_id,
                "archived": False,
                "workspaceAvailable": True,
                "workspaceRevision": revision,
            }
            if requested == project_id
            else None
        ),
    )
    monkeypatch.setattr(
        project_hooks,
        "project_workspace_access",
        lambda requested: (
            _workspace(root, revision)
            if requested == project_id
            else pytest.fail("opened the wrong project")
        ),
    )


def test_routes_require_explicit_exact_hash_trust_and_revisioned_handler_controls(
    tmp_path, monkeypatch
):
    project_id = _create_project()
    root = tmp_path / "repository"
    root.mkdir()
    _write_hooks(root)
    _bind_project(monkeypatch, project_id, root)

    discovered = project_hooks.project_hooks(project_id, _current_subject = "tester")

    assert discovered["exists"] is True
    assert discovered["workspaceAvailable"] is True
    assert discovered["workspaceRevision"] == 7
    assert discovered["storedTrust"] is None
    assert discovered["trusted"] is False
    assert discovered["revision"] == 0
    assert discovered["sourcePath"] == ".codex/hooks.json"
    handler = discovered["hooks"]["PreToolUse"][0]["hooks"][0]
    assert handler["enabled"] is True
    assert handler["active"] is False
    assert handler["command"] == "python3 .codex/check.py"

    trusted = project_hooks.project_hooks_trust(
        project_id,
        project_hooks.TrustProjectHooksRequest(
            contentHash = discovered["contentHash"],
            workspaceRevision = discovered["workspaceRevision"],
            revision = discovered["revision"],
        ),
        _current_subject = "tester",
    )

    assert trusted["trusted"] is True
    assert trusted["workspaceRevision"] == 7
    assert trusted["revision"] == 1
    assert trusted["storedTrust"] == {
        "contentHash": trusted["contentHash"],
        "revision": trusted["revision"],
    }
    assert trusted["hooks"]["PreToolUse"][0]["hooks"][0]["enabled"] is True
    assert trusted["hooks"]["PreToolUse"][0]["hooks"][0]["active"] is True

    disabled = project_hooks.project_hook_handler(
        project_id,
        handler["id"],
        project_hooks.SetProjectHookHandlerRequest(
            contentHash = trusted["contentHash"],
            workspaceRevision = trusted["workspaceRevision"],
            revision = trusted["revision"],
            enabled = False,
        ),
        _current_subject = "tester",
    )

    assert disabled["revision"] == 2
    assert disabled["hooks"]["PreToolUse"][0]["hooks"][0]["enabled"] is False
    assert disabled["hooks"]["PreToolUse"][0]["hooks"][0]["active"] is False

    revoked = project_hooks.project_hooks_revoke(
        project_id,
        project_hooks.RevokeProjectHooksRequest(revision = disabled["revision"]),
        _current_subject = "tester",
    )
    assert revoked["trusted"] is False
    assert revoked["revision"] == 3
    assert revoked["storedTrust"] is None
    assert revoked["hooks"]["PreToolUse"][0]["hooks"][0]["enabled"] is True
    assert revoked["hooks"]["PreToolUse"][0]["hooks"][0]["active"] is False


def test_api_keys_cannot_grant_or_change_hook_authority(monkeypatch):
    monkeypatch.setattr(
        project_hooks,
        "get_chat_project",
        lambda _project_id: pytest.fail("API-key mutations must fail before project access"),
    )
    trust_request = project_hooks.TrustProjectHooksRequest(
        contentHash = "a" * 64,
        workspaceRevision = 0,
        revision = 0,
    )
    handler_request = project_hooks.SetProjectHookHandlerRequest(
        contentHash = "a" * 64,
        workspaceRevision = 0,
        revision = 0,
        enabled = True,
    )

    with pytest.raises(HTTPException) as trust_denied:
        project_hooks.project_hooks_trust(
            "project",
            trust_request,
            via_api_key = True,
            _current_subject = "api-key",
        )
    assert trust_denied.value.status_code == 403

    with pytest.raises(HTTPException) as handler_denied:
        project_hooks.project_hook_handler(
            "project",
            "PreToolUse:0:0",
            handler_request,
            via_api_key = True,
            _current_subject = "api-key",
        )
    assert handler_denied.value.status_code == 403

    with pytest.raises(HTTPException) as revoke_denied:
        project_hooks.project_hooks_revoke(
            "project",
            project_hooks.RevokeProjectHooksRequest(revision = 1),
            via_api_key = True,
            _current_subject = "api-key",
        )
    assert revoke_denied.value.status_code == 403


def test_changed_file_is_untrusted_and_stale_mutations_fail(tmp_path, monkeypatch):
    project_id = _create_project("hooks-route-drift")
    root = tmp_path / "repository"
    root.mkdir()
    _write_hooks(root, "printf first")
    _bind_project(monkeypatch, project_id, root)
    first = project_hooks.project_hooks(project_id, _current_subject = "tester")
    trusted = project_hooks.project_hooks_trust(
        project_id,
        project_hooks.TrustProjectHooksRequest(
            contentHash = first["contentHash"],
            workspaceRevision = first["workspaceRevision"],
            revision = first["revision"],
        ),
        _current_subject = "tester",
    )

    _write_hooks(root, "printf changed")
    changed = project_hooks.project_hooks(project_id, _current_subject = "tester")

    assert changed["contentHash"] != first["contentHash"]
    assert changed["workspaceRevision"] == 7
    assert changed["trusted"] is False
    assert changed["revision"] == trusted["revision"]
    assert changed["hooks"]["PreToolUse"][0]["hooks"][0]["enabled"] is True
    assert changed["hooks"]["PreToolUse"][0]["hooks"][0]["active"] is False
    with pytest.raises(HTTPException) as stale:
        project_hooks.project_hooks_trust(
            project_id,
            project_hooks.TrustProjectHooksRequest(
                contentHash = first["contentHash"],
                workspaceRevision = changed["workspaceRevision"],
                revision = changed["revision"],
            ),
            _current_subject = "tester",
        )
    assert stale.value.status_code == 409
    assert "changed" in stale.value.detail


def test_unknown_handlers_and_stale_revisions_do_not_mutate_state(tmp_path, monkeypatch):
    project_id = _create_project("hooks-route-conflict")
    root = tmp_path / "repository"
    root.mkdir()
    _write_hooks(root)
    _bind_project(monkeypatch, project_id, root)
    current = project_hooks.project_hooks(project_id, _current_subject = "tester")
    trusted = project_hooks.project_hooks_trust(
        project_id,
        project_hooks.TrustProjectHooksRequest(
            contentHash = current["contentHash"],
            workspaceRevision = current["workspaceRevision"],
            revision = current["revision"],
        ),
        _current_subject = "tester",
    )

    with pytest.raises(HTTPException) as missing:
        project_hooks.project_hook_handler(
            project_id,
            "PreToolUse:9:9",
            project_hooks.SetProjectHookHandlerRequest(
                contentHash = trusted["contentHash"],
                workspaceRevision = trusted["workspaceRevision"],
                revision = trusted["revision"],
                enabled = False,
            ),
            _current_subject = "tester",
        )
    assert missing.value.status_code == 404

    with pytest.raises(HTTPException) as conflict:
        project_hooks.project_hooks_revoke(
            project_id,
            project_hooks.RevokeProjectHooksRequest(revision = 0),
            _current_subject = "tester",
        )
    assert conflict.value.status_code == 409
    state = project_hook_trust_db.get_project_hook_trust_state(
        project_id,
        trusted["contentHash"],
        workspace_identity = (root.stat().st_dev, root.stat().st_ino),
        workspace_revision = 7,
    )
    assert state["trusted"] is True
    assert state["revision"] == trusted["revision"]


def test_missing_or_archived_projects_are_hidden(monkeypatch):
    monkeypatch.setattr(project_hooks, "get_chat_project", lambda _project_id: None)

    with pytest.raises(HTTPException) as missing:
        project_hooks.project_hooks("missing", _current_subject = "tester")

    assert missing.value.status_code == 404


def test_workspace_revision_change_invalidates_trust_without_applying_preferences(
    tmp_path, monkeypatch
):
    project_id = _create_project("hooks-route-workspace-revision")
    root = tmp_path / "repository"
    root.mkdir()
    _write_hooks(root)
    _bind_project(monkeypatch, project_id, root, revision = 10)
    current = project_hooks.project_hooks(project_id, _current_subject = "tester")
    trusted = project_hooks.project_hooks_trust(
        project_id,
        project_hooks.TrustProjectHooksRequest(
            contentHash = current["contentHash"],
            workspaceRevision = current["workspaceRevision"],
            revision = current["revision"],
        ),
        _current_subject = "tester",
    )
    disabled = project_hooks.project_hook_handler(
        project_id,
        current["hooks"]["PreToolUse"][0]["hooks"][0]["id"],
        project_hooks.SetProjectHookHandlerRequest(
            contentHash = trusted["contentHash"],
            workspaceRevision = trusted["workspaceRevision"],
            revision = trusted["revision"],
            enabled = False,
        ),
        _current_subject = "tester",
    )
    assert disabled["hooks"]["PreToolUse"][0]["hooks"][0]["enabled"] is False

    _bind_project(monkeypatch, project_id, root, revision = 11)
    changed = project_hooks.project_hooks(project_id, _current_subject = "tester")
    handler = changed["hooks"]["PreToolUse"][0]["hooks"][0]

    assert changed["trusted"] is False
    assert changed["workspaceRevision"] == 11
    assert handler["enabled"] is True
    assert handler["active"] is False

    monkeypatch.setattr(
        project_hooks,
        "trust_project_hooks",
        lambda *_args, **_kwargs: pytest.fail("stale workspace trust reached storage"),
    )
    with pytest.raises(HTTPException) as stale_trust:
        project_hooks.project_hooks_trust(
            project_id,
            project_hooks.TrustProjectHooksRequest(
                contentHash = changed["contentHash"],
                workspaceRevision = trusted["workspaceRevision"],
                revision = changed["revision"],
            ),
            _current_subject = "tester",
        )
    assert stale_trust.value.status_code == 409
    assert "workspace changed" in stale_trust.value.detail.lower()

    monkeypatch.setattr(
        project_hooks,
        "set_project_hook_handler_enabled",
        lambda *_args, **_kwargs: pytest.fail("stale workspace toggle reached storage"),
    )
    with pytest.raises(HTTPException) as stale_toggle:
        project_hooks.project_hook_handler(
            project_id,
            handler["id"],
            project_hooks.SetProjectHookHandlerRequest(
                contentHash = changed["contentHash"],
                workspaceRevision = trusted["workspaceRevision"],
                revision = changed["revision"],
                enabled = True,
            ),
            _current_subject = "tester",
        )
    assert stale_toggle.value.status_code == 409
    assert "workspace changed" in stale_toggle.value.detail.lower()


def test_trust_rediscovers_source_before_rendering_snapshot(tmp_path, monkeypatch):
    project_id = _create_project("hooks-route-source-race")
    root = tmp_path / "repository"
    root.mkdir()
    _write_hooks(root, "printf first")
    _bind_project(monkeypatch, project_id, root)
    current = project_hooks.project_hooks(project_id, _current_subject = "tester")
    original_discover = project_hooks._discover_workspace
    calls = 0

    def drifting_discover(workspace):
        nonlocal calls
        discovered = original_discover(workspace)
        calls += 1
        if calls == 1:
            _write_hooks(root, "printf changed")
        return discovered

    monkeypatch.setattr(project_hooks, "_discover_workspace", drifting_discover)
    raced = project_hooks.project_hooks_trust(
        project_id,
        project_hooks.TrustProjectHooksRequest(
            contentHash = current["contentHash"],
            workspaceRevision = current["workspaceRevision"],
            revision = current["revision"],
        ),
        _current_subject = "tester",
    )

    assert calls == 2
    assert raced["contentHash"] != current["contentHash"]
    assert raced["trusted"] is False
    assert raced["revision"] == 1
    handler = raced["hooks"]["PreToolUse"][0]["hooks"][0]
    assert handler["enabled"] is True
    assert handler["active"] is False


def test_trust_and_toggle_hold_one_workspace_access_lease(tmp_path, monkeypatch):
    project_id = _create_project("hooks-route-lease")
    root = tmp_path / "repository"
    root.mkdir()
    _write_hooks(root)
    _bind_project(monkeypatch, project_id, root)
    current = project_hooks.project_hooks(project_id, _current_subject = "tester")
    metadata = root.stat()
    lease = {"active": False, "enters": 0}

    @contextmanager
    def guarded_access(requested):
        assert requested == project_id
        assert lease["active"] is False
        lease["active"] = True
        lease["enters"] += 1
        try:
            yield SimpleNamespace(
                root = root,
                device_id = metadata.st_dev,
                file_id = metadata.st_ino,
                revision = 7,
            )
        finally:
            lease["active"] = False

    original_discover = project_hooks._discover_workspace

    def guarded_discover(workspace):
        assert lease["active"] is True
        return original_discover(workspace)

    monkeypatch.setattr(project_hooks, "project_workspace_access", guarded_access)
    monkeypatch.setattr(project_hooks, "_discover_workspace", guarded_discover)
    trusted = project_hooks.project_hooks_trust(
        project_id,
        project_hooks.TrustProjectHooksRequest(
            contentHash = current["contentHash"],
            workspaceRevision = current["workspaceRevision"],
            revision = current["revision"],
        ),
        _current_subject = "tester",
    )
    assert lease == {"active": False, "enters": 1}

    lease["enters"] = 0
    project_hooks.project_hook_handler(
        project_id,
        current["hooks"]["PreToolUse"][0]["hooks"][0]["id"],
        project_hooks.SetProjectHookHandlerRequest(
            contentHash = trusted["contentHash"],
            workspaceRevision = trusted["workspaceRevision"],
            revision = trusted["revision"],
            enabled = False,
        ),
        _current_subject = "tester",
    )
    assert lease == {"active": False, "enters": 1}


def test_revoke_succeeds_when_workspace_cannot_be_discovered(tmp_path, monkeypatch):
    project_id = _create_project("hooks-route-unavailable-revoke")
    root = tmp_path / "repository"
    root.mkdir()
    _write_hooks(root)
    _bind_project(monkeypatch, project_id, root)
    current = project_hooks.project_hooks(project_id, _current_subject = "tester")
    trusted = project_hooks.project_hooks_trust(
        project_id,
        project_hooks.TrustProjectHooksRequest(
            contentHash = current["contentHash"],
            workspaceRevision = current["workspaceRevision"],
            revision = current["revision"],
        ),
        _current_subject = "tester",
    )

    monkeypatch.setattr(
        project_hooks,
        "get_chat_project",
        lambda requested: (
            {
                "id": project_id,
                "archived": False,
                "workspaceAvailable": False,
                "workspaceRevision": 7,
            }
            if requested == project_id
            else None
        ),
    )
    monkeypatch.setattr(
        project_hooks,
        "project_workspace_access",
        lambda _requested: pytest.fail("offline hook snapshots must not discover the workspace"),
    )
    unavailable = project_hooks.project_hooks(project_id, _current_subject = "tester")
    assert unavailable["workspaceAvailable"] is False
    assert unavailable["workspaceRevision"] == 7
    assert unavailable["storedTrust"] == {
        "contentHash": trusted["contentHash"],
        "revision": trusted["revision"],
    }

    revoked = project_hooks.project_hooks_revoke(
        project_id,
        project_hooks.RevokeProjectHooksRequest(revision = unavailable["storedTrust"]["revision"]),
        _current_subject = "tester",
    )

    assert revoked["workspaceAvailable"] is False
    assert revoked["storedTrust"] is None
    assert revoked["trusted"] is False
    assert revoked["revision"] == trusted["revision"] + 1
    assert revoked["exists"] is False
    assert revoked["contentHash"] is None
    assert all(not groups for groups in revoked["hooks"].values())
    state = project_hook_trust_db.get_project_hook_trust_state(
        project_id,
        current["contentHash"],
        workspace_identity = (root.stat().st_dev, root.stat().st_ino),
        workspace_revision = 7,
    )
    assert state["trusted"] is False
    assert state["revision"] == revoked["revision"]


def test_get_falls_back_only_when_workspace_access_itself_fails(tmp_path, monkeypatch):
    project_id = _create_project("hooks-route-stale-workspace-health")
    root = tmp_path / "repository"
    root.mkdir()
    _write_hooks(root)
    _bind_project(monkeypatch, project_id, root)
    current = project_hooks.project_hooks(project_id, _current_subject = "tester")
    trusted = project_hooks.project_hooks_trust(
        project_id,
        project_hooks.TrustProjectHooksRequest(
            contentHash = current["contentHash"],
            workspaceRevision = current["workspaceRevision"],
            revision = current["revision"],
        ),
        _current_subject = "tester",
    )

    @contextmanager
    def unavailable_access(_requested):
        raise project_hooks.AgentWorkspaceError("Project workspace is unavailable.")
        yield

    monkeypatch.setattr(project_hooks, "project_workspace_access", unavailable_access)
    unavailable = project_hooks.project_hooks(project_id, _current_subject = "tester")
    assert unavailable["workspaceAvailable"] is False
    assert unavailable["workspaceRevision"] == 7
    assert unavailable["storedTrust"] == {
        "contentHash": trusted["contentHash"],
        "revision": trusted["revision"],
    }

    _bind_project(monkeypatch, project_id, root)

    def malformed_hooks(_workspace):
        raise project_hooks.AgentWorkspaceError("Project hooks must be valid JSON.")

    monkeypatch.setattr(project_hooks, "_discover_workspace", malformed_hooks)
    with pytest.raises(HTTPException) as malformed:
        project_hooks.project_hooks(project_id, _current_subject = "tester")
    assert malformed.value.status_code == 409
    assert "valid JSON" in malformed.value.detail


def test_trust_state_read_does_not_open_or_parse_the_workspace(tmp_path, monkeypatch):
    project_id = _create_project("hooks-route-independent-trust")
    root = tmp_path / "repository"
    root.mkdir()
    _write_hooks(root)
    _bind_project(monkeypatch, project_id, root)
    current = project_hooks.project_hooks(project_id, _current_subject = "tester")
    trusted = project_hooks.project_hooks_trust(
        project_id,
        project_hooks.TrustProjectHooksRequest(
            contentHash = current["contentHash"],
            workspaceRevision = current["workspaceRevision"],
            revision = current["revision"],
        ),
        _current_subject = "tester",
    )

    monkeypatch.setattr(
        project_hooks,
        "project_workspace_access",
        lambda _requested: pytest.fail("trust-state reads must not open the workspace"),
    )
    monkeypatch.setattr(
        project_hooks,
        "_discover_workspace",
        lambda _workspace: pytest.fail("trust-state reads must not parse hooks.json"),
    )

    state = project_hooks.project_hooks_trust_state(
        project_id,
        _current_subject = "tester",
    )

    assert state == {
        "storedTrust": {
            "contentHash": trusted["contentHash"],
            "revision": trusted["revision"],
        },
        "revision": trusted["revision"],
    }


def test_capability_probe_failure_after_trust_returns_saved_unavailable_state(
    tmp_path, monkeypatch
):
    project_id = _create_project("probe-failure")
    root = tmp_path / "repository"
    root.mkdir()
    _write_hooks(root)
    _bind_project(monkeypatch, project_id, root)
    current = project_hooks.project_hooks(project_id, _current_subject = "tester")

    def failed_probe():
        raise RuntimeError("native probe failed")

    monkeypatch.setattr(project_hooks, "execution_status", failed_probe)
    saved = project_hooks.project_hooks_trust(
        project_id,
        project_hooks.TrustProjectHooksRequest(
            contentHash = current["contentHash"],
            workspaceRevision = current["workspaceRevision"],
            revision = current["revision"],
        ),
        _current_subject = "tester",
    )
    assert saved["trusted"]
    assert saved["execution"]["available"] is False
    assert saved["revision"] > current["revision"]
