# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards the Windows Pester bootstrap against the PSGallery flake coming back.

The setup.ps1 Pester job installs its own Pester. Twice now that install has
broken CI on unrelated PRs, and both times the failure was hidden:

1. PSGallery is intermittently missing from the repository list on GitHub's
   Windows runners, so `Set-PSRepository PSGallery` died with "No repository
   with the name 'PSGallery' was found." (#6892)

2. The guard added for (1) called `Register-PSRepository -Default` with
   `-ErrorAction SilentlyContinue`. On a runner where the legacy
   PackageManagement provider cannot bootstrap nuget.exe, that call fails with
   "NuGet.Commands.CommandException: Missing option value for: '-source'" --
   silently. The next line then died with the misleading message from (1), so
   the logs pointed at the wrong cause.

The bootstrap now prefers PSResourceGet (which resolves PSGallery over HTTPS and
never shells out to nuget.exe), retries, and verifies the module actually
imported. These tests fail if any of that is removed.
"""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "studio-windows-inference-smoke.yml"


def _bootstrap_step() -> dict:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding = "utf-8"))
    steps = workflow["jobs"]["pester"]["steps"]
    for step in steps:
        if "Install Pester" in (step.get("name") or ""):
            return step
    raise AssertionError("no Pester install step in the pester job")


def test_bootstrap_step_exists_and_runs_under_pwsh():
    step = _bootstrap_step()
    assert step["shell"] == "pwsh"
    assert step["run"].strip()


def test_registration_failures_are_never_silenced():
    """The exact regression: a swallowed Register-PSRepository hid the real error."""
    run = _bootstrap_step()["run"]
    for line in run.splitlines():
        stripped = line.strip()
        if stripped.startswith(("Register-PSRepository", "Register-PSResourceRepository")):
            assert "SilentlyContinue" not in stripped, (
                "registering the gallery must fail loudly, not silently leave it unregistered: "
                f"{stripped}"
            )
    assert "$ErrorActionPreference = 'Stop'" in run


def test_psresourceget_is_preferred_over_the_nuget_bootstrap():
    run = _bootstrap_step()["run"]
    assert "Install-PSResource" in run
    # The legacy path may remain as a fallback, but must not be the only option.
    assert run.index("Install-PSResource") < run.index(
        "Install-Module"
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


def test_network_is_skipped_when_the_image_already_satisfies_the_minimum():
    """The runner ships Pester 5.x; the common path should not touch PSGallery."""
    run = _bootstrap_step()["run"]
    assert "if (-not $installed)" in run
