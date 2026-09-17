# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression: a Landlock confinement offered to a managed account must scope signals.

Drop in as studio/backend/tests/test_account_tool_confinement_signal_abi.py.
"""

import subprocess
import sys
import time

import pytest

from auth import policy
from core.inference import tool_confinement, tools
from utils.account_context import AccountContext, run_as

from .test_account_lifecycle import auth_env, matrix  # noqa: F401

BOB = AccountContext("bob-id", "bob")


@pytest.fixture(autouse = True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_ALLOW_UNCONFINED_TOOLS", raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_NPROC", "4000000")
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(tools, "_workdirs", {})
    monkeypatch.setattr(tools, "_active_sessions", {})
    monkeypatch.setattr(tools, "_pending_removals", {})
    monkeypatch.setattr(tools, "_removing_sessions", set())
    monkeypatch.setattr(tools, "_legacy_sandbox_migrated", True)
    monkeypatch.setattr(tools, "_start_detached_sweep", lambda: None)
    monkeypatch.setattr(tools, "_legacy_sandbox_root", lambda: str(tmp_path / "legacy"))


@pytest.mark.skipif(
    sys.platform != "linux" or tool_confinement.landlock_abi() < 6,
    reason = "needs a kernel able to apply the ABI 6 signal scope",
)
@pytest.mark.parametrize("abi", (3, 4, 5))
def test_pre_scope_landlock_abi_never_confines_a_managed_account(abi, monkeypatch):
    """Below ABI 6 signals are unscoped, so the fs sandbox alone must not count as confinement."""
    monkeypatch.setattr(tool_confinement, "landlock_abi", lambda: abi)
    assert tool_confinement._linux_confinement(tools._SANDBOX_SITE_DIR) is None
    with pytest.raises(tool_confinement.ToolConfinementUnavailable):
        run_as(BOB, tools._account_confinement)


@pytest.mark.skipif(
    sys.platform != "linux" or tool_confinement.landlock_abi() < 6,
    reason = "needs a kernel able to apply the ABI 6 signal scope",
)
def test_a_confined_managed_child_cannot_kill_a_foreign_process_on_a_pre_scope_abi(monkeypatch):
    """Same behaviour, observed: an ABI 5 host must not hand a managed tool a signal-capable box."""
    monkeypatch.setattr(tool_confinement, "landlock_abi", lambda: 5)
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        try:
            run_as(BOB, tools._account_confinement)
        except tool_confinement.ToolConfinementUnavailable:
            return  # refused: the gap cannot be reached
        out = run_as(
            BOB,
            tools._python_exec,
            "import os, signal\n"
            f"try:\n"
            f"    os.kill({victim.pid}, signal.SIGKILL); print('KILLED_FOREIGN')\n"
            "except OSError as e:\n"
            "    print('signal denied', e.errno)\n",
            session_id = "chat",
        )
        time.sleep(1)
        assert "KILLED_FOREIGN" not in out, out
        assert victim.poll() is None, "a managed tool killed a foreign same-UID process"
    finally:
        victim.kill()
