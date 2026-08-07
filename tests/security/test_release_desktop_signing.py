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


def test_the_check_exits_non_zero_on_both_failure_paths():
    run = _verify_step()["run"]
    # One for "not on PATH", one for "on PATH but exited non-zero".
    assert run.count("exit 1") == 2


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
    assert run.count("bump the key") == 2


def test_the_check_still_only_runs_on_windows():
    assert _verify_step()["if"] == "matrix.platform == 'windows-latest'"
