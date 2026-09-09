# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Windows signing toolchain has to be verified, and has to fail closed.

`trusted-signing-cli` runs with the Azure Trusted Signing and Tauri signing
secrets in the environment, so two things must hold: the binary is the one we
pinned, and a check that cannot prove that cannot report success.

Both have regressed before. The install step once restored the executable from
an actions/cache and skipped the build on a hit, leaving a spoofable
`--version` as the only gate. The verify step once ended both branches in
`|| Write-Output "..."`, which exits 0, so a broken binary passed and surfaced
later inside Tauri bundling.
"""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"


def _build_steps():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    return workflow["jobs"]["build"]["steps"]


def _step(name: str):
    return next(step for step in _build_steps() if step.get("name") == name)


def _verify_step():
    return _step("Verify trusted-signing-cli")


def test_the_signing_cli_is_pinned_by_url_and_digest():
    step = _step("Install trusted-signing-cli")
    assert (
        "/releases/download/0.10.0/trusted-signing-cli.exe"
        in step["env"]["TRUSTED_SIGNING_CLI_URL"]
    )
    assert len(step["env"]["TRUSTED_SIGNING_CLI_SHA256"]) == 64


def test_both_steps_name_the_shell_they_need():
    # -MaximumRetryCount / -RetryIntervalSec are PowerShell 6+ only, and every other Windows step here says pwsh
    # outright rather than inheriting the interpreter from a runner default.
    for name in ("Install trusted-signing-cli", "Verify trusted-signing-cli"):
        assert _step(name)["shell"] == "pwsh", name


def test_a_digest_mismatch_stops_the_release_before_anything_is_signed():
    run = _step("Install trusted-signing-cli")["run"]
    assert "Get-FileHash" in run
    assert "if ($actual -ne $expected)" in run
    assert "exit 1" in run
    # A rejected download left on disk is still reachable by name from PATH.
    assert "Remove-Item -Force $dest" in run


def test_no_step_restores_the_signing_cli_from_a_cache():
    # A cache is not an integrity mechanism: restore plus skip-on-hit is what would let a poisoned entry sign releases.
    for step in _build_steps():
        uses = step.get("uses", "")
        if not uses.startswith("actions/cache"):
            continue
        assert "trusted-signing-cli" not in yaml.safe_dump(step)


def test_the_verified_binary_is_the_one_that_signs():
    # The signing script calls the tool by bare name, and rust-cache restores ~/.cargo/bin, which can hold an unverified
    # copy of the same name. PATH has to resolve to the digest-checked file.
    run = _verify_step()["run"]
    assert "$verified" in run
    assert "[IO.Path]::GetFullPath($cli.Source) -ne $verified" in run


def test_the_check_exits_non_zero_on_every_failure_path():
    run = _verify_step()["run"]
    # Not on PATH, resolved to an unverified copy, cannot be started, and started but exited non-zero.
    assert run.count("exit 1") == 4


def test_a_binary_that_cannot_start_is_caught():
    # A truncated download raises a terminating PowerShell error, not a native exit code, so without the catch the step
    # dies before explaining why.
    run = _verify_step()["run"]
    assert "try {" in run
    assert "catch {" in run


def test_the_launch_error_is_flattened_to_one_line():
    # A PowerShell error spans message, offending line and caret; an annotation stops at the first newline, dropping
    # the guidance that follows.
    run = _verify_step()["run"]
    assert "-replace '\\s+', ' '" in run


def test_failures_are_not_swallowed_by_a_fallback_message():
    run = _verify_step()["run"]
    assert "|| Write-Output" not in run


def test_the_native_exit_code_is_inspected():
    # $ErrorActionPreference does not trap a native non-zero exit, so
    # $LASTEXITCODE has to be read explicitly.
    run = _verify_step()["run"]
    assert "$LASTEXITCODE" in run


def test_every_failure_says_what_it_was():
    # Failing closed without saying why turns a rare fetch fault into an unexplained outage, and the four causes need
    # different fixes.
    run = _verify_step()["run"]
    assert run.count("::error::") == 4


def test_the_check_still_only_runs_on_windows():
    assert _verify_step()["if"] == "matrix.platform == 'windows-latest'"
