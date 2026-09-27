# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An overridden sandbox home is a private root: retirement renames it aside too."""

from pathlib import Path

import pytest

from auth import storage
from utils.account_context import run_as
from utils.paths import storage_roots

from .test_account_lifecycle import auth_env, matrix  # noqa: F401


@pytest.fixture
def sandbox_home(tmp_path, monkeypatch, matrix):  # noqa: F811
    """UNSLOTH_STUDIO_SANDBOX_HOME puts the account sandbox outside workspace_root()."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sandboxes"))
    return matrix


def _make_private_roots(account):
    for root in (
        storage_roots.workspace_root,
        storage_roots.project_workspaces_root,
        storage_roots.tmp_root,
    ):
        run_as(account, root).mkdir(parents = True, exist_ok = True)


def test_delete_retires_an_overridden_sandbox_root(sandbox_home):
    _, _, accounts = sandbox_home
    from core.inference import tools

    alice = storage.get_account("alice")
    sandbox = Path(run_as(alice, tools.sandbox_root))
    assert sandbox not in run_as(alice, storage_roots.workspace_root).parents
    sandbox.mkdir(parents = True, exist_ok = True)
    (sandbox / "tool-output.txt").write_text("alice")
    _make_private_roots(alice)

    storage.delete_account(alice.account_id, accounts.retire_account_roots)

    assert not sandbox.exists()
    retired = list(sandbox.parent.glob(sandbox.name + "-deleted-*"))
    assert len(retired) == 1
    assert (retired[0] / "tool-output.txt").read_text() == "alice"


def test_the_retirement_fence_covers_an_overridden_sandbox_root(sandbox_home):
    """A finalizer outliving the delete must not recreate the sandbox either."""
    _, _, accounts = sandbox_home
    from core.inference import tools

    alice = storage.get_account("alice")
    sandbox = Path(run_as(alice, tools.sandbox_root))
    sandbox.mkdir(parents = True, exist_ok = True)
    _make_private_roots(alice)
    storage.delete_account(alice.account_id, accounts.retire_account_roots)

    assert run_as(alice, storage_roots._under_managed_workspace, sandbox) is True
    with pytest.raises(storage_roots.RetiredAccountError):
        run_as(alice, storage_roots.ensure_dir, sandbox / "late")
    assert not sandbox.exists()
