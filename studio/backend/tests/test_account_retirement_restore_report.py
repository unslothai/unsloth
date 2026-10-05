# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A retirement rollback that cannot rename a root back says so and keeps the root listed,
instead of reporting a clean rollback with the data stranded under its deleted name."""

import os

from auth import storage
from studio.backend.tests.test_account_lifecycle import auth_env, headers, matrix  # noqa: F401
from utils.account_context import run_as
from utils.paths import storage_roots


def test_a_failed_restore_is_reported_with_the_stranded_path(matrix, monkeypatch):
    client, _, accounts = matrix
    account = storage.get_account("alice")
    roots = [
        run_as(account, root)
        for root in (
            storage_roots.workspace_root,
            storage_roots.project_workspaces_root,
            storage_roots.tmp_root,
        )
    ]
    for root in roots:
        root.mkdir(parents = True, exist_ok = True)
        (root / "private.txt").write_text("keep")
    calls = []

    class RenameFailsLateAndOnce(type(roots[0])):
        @staticmethod
        def rename(source, destination):
            calls.append((str(source), str(destination)))
            # The last move aside fails, then the first rename back fails as well.
            if len(calls) in (3, 4):
                raise PermissionError("locked directory")
            os.rename(source, destination)

    monkeypatch.setattr(accounts, "Path", RenameFailsLateAndOnce)
    response = client.delete(f"/api/accounts/{account.account_id}", headers = headers())
    print(f"delete: {response.status_code} {response.text}")
    assert response.status_code == 409
    stranded = [p for root in roots for p in root.parent.iterdir() if "-deleted-" in p.name]
    assert len(stranded) == 1, stranded
    assert str(stranded[0]) in response.text
    assert storage.get_user_record("alice")["is_active"] == 0
