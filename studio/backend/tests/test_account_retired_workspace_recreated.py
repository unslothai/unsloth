# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A raw mkdir that outlives deletion must not un-fence the retired account."""

import pytest

from storage import studio_db
from utils.account_context import OWNER, AccountContext, run_as
from utils.paths import storage_roots as roots

ALICE = AccountContext("11111111111111111111111111111111", "alice")


@pytest.fixture(autouse = True)
def fresh_retirement_tombstones(monkeypatch):
    from core.training import account_jobs
    monkeypatch.setattr(account_jobs, "_retired", set())


@pytest.fixture
def account_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path / "Documents"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    studio_db.close_wal_keeper()
    studio_db.reset_schema_state_for_tests()
    yield
    studio_db.close_wal_keeper()
    studio_db.reset_schema_state_for_tests()


def test_a_raw_mkdir_after_deletion_does_not_unfence_the_account(account_home, monkeypatch):
    """``import-example``'s failure recovery calls ``folder.mkdir(parents = True)`` directly, so a
    request that outlives DELETE puts the workspace back. Every later guarded creation must still
    refuse: the tombstone, not the directory's existence, says the account is gone."""
    from routes.accounts import retire_account_roots

    monkeypatch.setattr("core.inference.mcp_client.close_mcp_sessions", lambda: None)
    monkeypatch.setattr("core.inference.mcp_client.invalidate_tool_cache", lambda: None)
    root = run_as(ALICE, roots.workspace_root)
    run_as(ALICE, studio_db.get_connection).close()
    retire_account_roots(ALICE)
    assert not root.exists()

    # routes/training.py restore_folded(), verbatim, on a folder under the retired workspace.
    folder = run_as(ALICE, lambda: roots.datasets_root() / "tuxemon")
    folder.mkdir(parents = True, exist_ok = True)
    assert root.exists()

    studio_db.reset_schema_state_for_tests()
    with pytest.raises(roots.RetiredAccountError):
        run_as(ALICE, lambda: roots.ensure_dir(roots.outputs_root()))
    with pytest.raises(roots.RetiredAccountError):
        run_as(ALICE, studio_db.get_connection).close()
    assert not (root / "studio.db").exists()

    # Single-user / owner installs are untouched: the owner is never retired.
    assert run_as(OWNER, lambda: roots.ensure_dir(roots.outputs_root())).is_dir()
