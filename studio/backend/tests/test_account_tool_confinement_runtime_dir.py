# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sys

import pytest

from core.inference import tool_confinement, tools
from utils.account_context import run_as

from .test_account_tool_confinement import BOB, LANDLOCK, auth_env, isolated, matrix  # noqa: F401


def _runtime_dir() -> str:
    return f"/run/user/{os.getuid()}"


def test_landlock_rules_exclude_the_per_user_runtime_dir(tmp_path):
    """/run is granted wholesale, but XDG_RUNTIME_DIR is 0700 for the Studio Unix user:
    without an exclusion every managed account reads the owner's runtime secrets."""
    if sys.platform != "linux":
        pytest.skip("Linux rule builder")
    runtime = _runtime_dir()
    if not os.path.isdir(runtime):
        pytest.skip("no per-user runtime dir on this host")
    run_as(BOB, tools._get_workdir, "chat")
    rules = run_as(BOB, tool_confinement._landlock_rules, 3, tools._SANDBOX_SITE_DIR)
    assert all(not tool_confinement._contains(p, runtime) for p, _ in rules), [
        p for p, _ in rules if tool_confinement._contains(p, runtime)
    ]
    # The rest of /run stays readable: /etc/resolv.conf usually points into it.
    resolv = os.path.realpath("/etc/resolv.conf")
    if resolv.startswith("/run/") and os.path.exists(resolv):
        assert any(tool_confinement._contains(p, resolv) for p, _ in rules)


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_managed_child_cannot_read_the_per_user_runtime_dir(tmp_path):
    runtime = _runtime_dir()
    if not os.path.isdir(runtime):
        pytest.skip("no per-user runtime dir on this host")
    run_as(BOB, tools._get_workdir, "chat")
    out = run_as(BOB, tools._bash_exec, f"ls -a {runtime}; echo rc=$?", session_id = "chat")
    assert "rc=0" not in out, out
