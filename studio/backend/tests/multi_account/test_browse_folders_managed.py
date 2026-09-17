# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's folder browser is confined to its workspace, suggestions included."""

from routes import models as models_routes
from utils.account_context import AccountContext, run_as
from utils.paths.storage_roots import workspace_root

ALICE = AccountContext("a" * 32, "alice")


def test_managed_browse_suggests_only_the_workspace(isolated_auth):
    root = run_as(ALICE, workspace_root)
    (root / "models").mkdir(parents = True)
    response = run_as(ALICE, models_routes.browse_folders, None, False, "alice")
    assert response.current == str(root.resolve())
    assert response.parent is None
    assert response.suggestions == [str(root.resolve())]
    assert [entry.name for entry in response.entries] == ["models"]
