# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Windows signing toolchain check has to fail closed.

An unsigned or half-signed installer is the failure this workflow exists to
prevent, so the step that proves `trusted-signing-cli` works must not be able to
report success when it does not. The check previously ended both of its branches
in `|| Write-Output "..."`, which exits 0, so a missing or broken binary passed
and only surfaced later inside the Tauri bundling step.
"""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"


def _verify_step():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    return next(
        step
        for step in workflow["jobs"]["build"]["steps"]
        if step.get("name") == "Verify trusted-signing-cli"
    )


def test_the_check_exits_non_zero_on_every_failure_path():
    run = _verify_step()["run"]
    # Not on PATH, cannot be started, and started but exited non-zero.
    assert run.count("exit 1") == 3


def test_a_binary_that_cannot_start_is_caught():
    # A truncated cache entry gives "Exec format error", which is a terminating
    # PowerShell error, not a native exit code. Without the catch the step dies
    # on that line and never prints the cache-bump recovery.
    run = _verify_step()["run"]
    assert "try {" in run
    assert "catch {" in run


def test_the_launch_error_is_flattened_to_one_line():
    # A PowerShell error spans message, offending line and caret. An annotation
    # is truncated at the first newline, so an unflattened message would drop
    # the recovery guidance that follows it.
    run = _verify_step()["run"]
    assert "-replace '\\s+', ' '" in run


def test_failures_are_not_swallowed_by_a_fallback_message():
    run = _verify_step()["run"]
    assert "|| Write-Output" not in run


def test_the_native_exit_code_is_inspected():
    # $ErrorActionPreference does not trap a native binary exiting non-zero,
    # so $LASTEXITCODE has to be read explicitly for the check to mean anything.
    run = _verify_step()["run"]
    assert "$LASTEXITCODE" in run


def test_the_operator_is_told_how_to_recover_from_a_bad_cache():
    # The binary is restored from a cache keyed on its version, so a corrupt
    # entry survives until the key changes. Failing closed without saying that
    # would turn a rare cache fault into an unexplained release outage.
    run = _verify_step()["run"]
    assert run.count("bump the key") == 3


def test_the_check_still_only_runs_on_windows():
    assert _verify_step()["if"] == "matrix.platform == 'windows-latest'"
