# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's own project workspace is inside its directory boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import policy
from hub.services.models import account_access as access
from utils.account_context import AccountContext, run_as
from utils.paths.storage_roots import project_workspaces_root

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_public_repos", {})
    monkeypatch.setattr(
        access,
        "HfApi",
        lambda: SimpleNamespace(
            repo_info = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
        ),
    )


def _alice_project(tmp_path):
    project = run_as(ALICE, project_workspaces_root) / "demo-1234"
    project.mkdir(parents = True, exist_ok = True)
    return project


def test_private_directory_admits_the_accounts_own_project_workspace(tmp_path):
    project = _alice_project(tmp_path)
    # The same path the account boundary and model visibility already accept.
    assert run_as(ALICE, access.model_visible, str(project))
    assert run_as(ALICE, access.private_directory, str(project), "") == str(project)


def test_private_directory_still_refuses_another_accounts_project_workspace(tmp_path):
    project = _alice_project(tmp_path)
    with pytest.raises(HTTPException):
        run_as(BOB, access.private_directory, str(project), "")


def test_private_directory_owner_and_workspace_paths_unchanged(tmp_path):
    from utils.paths.storage_roots import workspace_root

    alice_outputs = str(run_as(ALICE, workspace_root) / "outputs")
    assert run_as(ALICE, access.private_directory, alice_outputs, "outputs") == alice_outputs
    with pytest.raises(HTTPException):
        run_as(ALICE, access.private_directory, str(tmp_path / "elsewhere"), "outputs")
