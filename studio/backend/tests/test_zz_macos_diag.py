# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import subprocess
import sys

import pytest

from core.inference import _diag_tc_main, tool_confinement, tools
from utils.account_context import run_as

from .test_account_tool_confinement import ALICE, isolated  # noqa: F401


@pytest.mark.skipif(sys.platform != "darwin", reason = "diag")
def test_diag(monkeypatch):
    workdir = run_as(ALICE, tools._get_workdir, "chat")
    report = [f"exe={sys.executable} prefix={sys.prefix} base={sys.base_prefix}"]
    for label, fn in (
        ("head", tool_confinement.macos_profile),
        ("main", _diag_tc_main.macos_profile),
    ):
        monkeypatch.setattr(tool_confinement, "macos_profile", fn)
        confinement = run_as(ALICE, tools._account_confinement)
        for argv in (
            ["/usr/bin/true"],
            [sys.executable, "-c", "print('hi')"],
            ["/bin/sh", "-c", "python3 -c 'print(1)'"],
        ):
            r = subprocess.run(
                confinement.wrap(argv), cwd = workdir, capture_output = True, text = True, timeout = 60
            )
            report.append(
                f"{label} {argv[0]} rc={r.returncode} out={r.stdout[-200:]!r} err={r.stderr[-400:]!r}"
            )
    log = subprocess.run(
        ["log", "show", "--last", "2m", "--predicate", "sender == 'Sandbox'", "--style", "compact"],
        capture_output = True,
        text = True,
        timeout = 60,
    ).stdout
    report.append("DENY:" + "\n".join(l for l in log.splitlines() if "deny" in l)[-3000:])
    assert False, "\n".join(report)
