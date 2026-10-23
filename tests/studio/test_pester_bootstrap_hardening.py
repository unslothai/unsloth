# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pins Pester bootstrap to PSResourceGet, with retries and an import check, since PSGallery flakes."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "studio-windows-inference-smoke.yml"
_GUARD_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pester-guard-ci.yml"


def _pester_steps() -> list[dict]:
    """Matches steps by the Install Pester name, not the job id, so a job rename cannot break it."""
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding = "utf-8"))
    for job in workflow["jobs"].values():
        steps = job.get("steps") or []
        if any("Install Pester" in (s.get("name") or "") for s in steps):
            return steps
    raise AssertionError("no job in the workflow installs Pester")


def _bootstrap_step() -> dict:
    for step in _pester_steps():
        if "Install Pester" in (step.get("name") or ""):
            return step
    raise AssertionError("no Pester install step")


def test_bootstrap_step_exists_and_runs_under_pwsh():
    step = _bootstrap_step()
    assert step["shell"] == "pwsh"
    assert step["run"].strip()


def test_registration_failures_are_never_silenced():
    """The exact regression: a swallowed Register-PSRepository hid the real error."""
    run = _bootstrap_step()["run"]
    seen = set()
    for line in run.splitlines():
        stripped = line.strip()
        for cmd in ("Register-PSRepository", "Register-PSResourceRepository"):
            if not stripped.startswith(cmd):
                continue
            seen.add(cmd)
            assert "SilentlyContinue" not in stripped, (
                "registering the gallery must fail loudly, not silently leave it unregistered: "
                f"{stripped}"
            )
    # Otherwise deleting both registrations would pass on an unregistered-PSGallery runner.
    assert seen == {
        "Register-PSRepository",
        "Register-PSResourceRepository",
    }, f"both registration paths must stay present, found {sorted(seen)}"
    assert "$ErrorActionPreference = 'Stop'" in run


def test_psresourceget_is_preferred_over_the_nuget_bootstrap():
    run = _bootstrap_step()["run"]
    # Match the invocation: the `Get-Command` probe alone would satisfy a substring check.
    assert "Install-PSResource -Name" in run, "the PSResourceGet branch must actually install"
    assert (
        "$usePSResourceGet = $hasPSResourceGet" in run
    ), "PSResourceGet must be the initial choice, not just a reachable fallback"
    assert run.index("Install-PSResource -Name") < run.index(
        "Install-Module "
    ), "PSResourceGet must be tried before the nuget.exe-backed Install-Module path"


def test_install_is_retried_and_then_fails_loudly():
    run = _bootstrap_step()["run"]
    assert "-le 3" in run, "expected a bounded retry loop"
    assert "Start-Sleep" in run, "expected backoff between attempts"
    assert "if ($attempt -eq 3) { throw }" in run, "the last attempt must rethrow"


def test_a_failing_client_is_swapped_rather_than_retried_three_times():
    """PSGallery has served 500s to PSResourceGet while Install-Module kept working."""
    run = _bootstrap_step()["run"]
    assert (
        "if ($hasPSResourceGet) { $usePSResourceGet = -not $usePSResourceGet }" in run
    ), "a failed attempt must swap install clients, not retry the same one"


def test_module_presence_is_verified_after_install():
    run = _bootstrap_step()["run"]
    assert "failed to import" in run, "expected a post-import version assertion"
    assert (
        "still not present after install" in run
    ), "an install that reports success but leaves no usable module must fail"


def test_the_guard_runs_from_the_workflow_it_guards():
    """No pytest workflow filters on this file, so the job must run the guard itself."""
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding = "utf-8"))
    on = workflow.get("on") or workflow.get(True)
    # as_posix(): this runs on windows-latest, where str() gives backslashes.
    assert _WORKFLOW.relative_to(REPO_ROOT).as_posix() in on["pull_request"]["paths"]
    assert any(
        Path(__file__).name in (s.get("run") or "") for s in _pester_steps()
    ), "the Pester phase must run this guard, or a workflow-only edit skips it entirely"


def test_editing_the_guard_runs_it_on_windows():
    """The heavy workflow has no job-level gating, so the guard gets its own cheap one."""
    workflow = yaml.safe_load(_GUARD_WORKFLOW.read_text(encoding = "utf-8"))
    on = workflow.get("on") or workflow.get(True)
    assert Path(__file__).resolve().relative_to(REPO_ROOT).as_posix() in on["pull_request"]["paths"]
    jobs = list(workflow["jobs"].values())
    assert len(jobs) == 1, "keep this workflow to one job, it exists to be cheap"
    assert jobs[0]["runs-on"] == "windows-latest", "the bug this catches is Windows-only"
    assert any(Path(__file__).name in (s.get("run") or "") for s in jobs[0]["steps"])


def test_network_is_skipped_when_the_image_already_satisfies_the_minimum():
    """The runner ships Pester 5.x; the common path should not touch PSGallery."""
    run = _bootstrap_step()["run"]
    assert "if (-not $installed)" in run
