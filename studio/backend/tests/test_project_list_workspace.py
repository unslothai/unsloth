# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GET /api/chat/projects backfills a missing workspace, and only that.

The sidebar, the Projects page and every gallery menu list projects, so ensuring each row's
workspace on every list cost several connections and mkdirs per project per call.
"""

from routes import chat_history


def _row(id, root_path):
    return {
        "id": id,
        "name": id,
        "instructions": "",
        "rootPath": root_path,
        "sandboxPath": f"{root_path}/sandbox" if root_path else None,
        "workingFolder": None,
        "archived": False,
        "createdAt": 1,
        "updatedAt": 2,
        "lastActivityAt": 5,
    }


def test_only_a_project_without_a_root_is_ensured(monkeypatch):
    rows = [_row("rooted", "/tmp/rooted"), _row("legacy", None)]
    monkeypatch.setattr(chat_history, "list_chat_projects", lambda include_archived = False: rows)
    ensured = []

    def ensure(project_id):
        ensured.append(project_id)
        return {**_row(project_id, "/tmp/backfilled"), "lastActivityAt": None}

    monkeypatch.setattr(chat_history, "ensure_chat_project_workspace", ensure)

    projects = chat_history.list_projects(include_archived = False, current_subject = "owner").projects

    assert ensured == ["legacy"]
    by_id = {project.id: project for project in projects}
    assert by_id["rooted"].rootPath == "/tmp/rooted"
    assert by_id["legacy"].rootPath == "/tmp/backfilled"
    # Activity comes from the listing, which the ensured record does not carry.
    assert by_id["legacy"].lastActivityAt == 5


def test_a_row_deleted_mid_backfill_still_lists(monkeypatch):
    rows = [_row("gone", None)]
    monkeypatch.setattr(chat_history, "list_chat_projects", lambda include_archived = False: rows)
    monkeypatch.setattr(chat_history, "ensure_chat_project_workspace", lambda project_id: None)

    projects = chat_history.list_projects(include_archived = False, current_subject = "owner").projects

    assert [project.id for project in projects] == ["gone"]
