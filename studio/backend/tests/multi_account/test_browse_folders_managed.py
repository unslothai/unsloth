# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's folder browser is confined to its workspace, suggestions included."""

import pytest

import storage.studio_db as studio_db
from routes import models as models_routes
from utils.account_context import AccountContext, run_as
from utils.paths import storage_roots
from utils.paths.storage_roots import workspace_root

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def _denylist_inert(monkeypatch):
    # macOS tmp_path is under denied /private/var.
    monkeypatch.setattr(studio_db, "is_denied_system_path", lambda _p: False)


def test_managed_browse_suggests_only_the_workspace(isolated_auth):
    root = run_as(ALICE, workspace_root)
    (root / "models").mkdir(parents = True)
    response = run_as(ALICE, models_routes.browse_folders, None, False, "alice")
    assert response.current == str(root.resolve())
    assert response.parent is None
    assert response.suggestions == [str(root.resolve())]
    assert [entry.name for entry in response.entries] == ["models"]


def test_managed_browse_opens_when_the_install_root_is_a_symlink(
    isolated_auth, tmp_path, monkeypatch
):
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(relocated, target_is_directory = True)
    monkeypatch.setattr(storage_roots, "studio_root", lambda: linked)
    root = run_as(ALICE, workspace_root)
    (root / "models").mkdir(parents = True)
    response = run_as(ALICE, models_routes.browse_folders, None, False, "alice")
    assert response.current == str(root.resolve())
    assert [entry.name for entry in response.entries] == ["models"]
    nested = run_as(ALICE, models_routes.browse_folders, str(root / "models"), False, "alice")
    assert nested.current == str((root / "models").resolve())
