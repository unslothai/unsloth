# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every Playwright driver under tests/studio is invoked by some workflow.

These suites are standalone scripts, not pytest files, so nothing collects them:
a driver runs only because a workflow step or a .github/scripts helper names it.
Delete that line, or add a driver and forget one, and the suite runs nowhere
while every job stays green. Two were already in that state when this was
written, listed below with a reason each.

Same shape as test_ci_shell_suite_coverage.py, which guards tests/sh for the same
failure: the list of what CI runs drifting behind the directory it runs from.
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DRIVERS = sorted((REPO / "tests" / "studio").glob("playwright_*.py"))

# Drivers no workflow runs, each with why. Shrinking this list is the point;
# growing it needs a reason written here.
NOT_IN_CI = {
    # Needs the Tauri desktop shell (it serves a page under the real Tauri CSP),
    # which no Linux runner in this repo builds. tests/studio/
    # test_tauri_python_tool_images.py asserts the policy statically instead.
    "playwright_tauri_python_tool_images.py",
    # Drives the Train page pickers, which need a dataset and a model resolved
    # through huggingface_hub; the UI workflows deliberately boot API-only with
    # one 254 MiB GGUF and no network model resolution.
    "playwright_train_pickers.py",
}


def _executable_text(path: Path) -> str:
    """The parts of a workflow that RUN something: step `run` bodies and `uses` refs.

    Reading the whole file counts a driver named in `on.pull_request.paths` as an
    invocation. studio-frontend-ci.yml names playwright_strip_ansi_smoke.py in both
    its trigger list and its step, so deleting the step alone would leave the guard
    green while the suite runs nowhere. Trigger paths say when CI runs, not what it
    runs. Reported on PR #9060.
    """
    document = yaml.safe_load(path.read_text(encoding = "utf-8"))
    if not isinstance(document, dict):
        return ""
    parts: list[str] = []
    for job in (document.get("jobs") or {}).values():
        if not isinstance(job, dict):
            continue
        parts.append(str(job.get("uses", "")))
        for step in job.get("steps") or []:
            if isinstance(step, dict):
                parts.append(str(step.get("run", "")))
                parts.append(str(step.get("uses", "")))
                parts.append(str(step.get("with", "")))
    return "\n".join(parts)


def _ci_text() -> str:
    """Everything CI could name a driver from, reachable from something that runs.

    Two ways a name can look like coverage without being it, both live in this repo:
    a driver named only in a workflow's trigger `paths`, and a driver named only by a
    helper script no workflow calls. So the executable fields of the workflows seed
    the text, and a helper joins only once something already in it names the helper --
    repeatedly, since one helper may call another.
    """
    helpers = [
        path
        for directory in ((REPO / ".github" / "scripts"), (REPO / ".github" / "actions"))
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    parts = [
        _executable_text(path)
        for path in sorted((REPO / ".github" / "workflows").rglob("*"))
        if path.is_file() and path.suffix in (".yml", ".yaml")
    ]
    text = "\n".join(parts)
    remaining = list(helpers)
    added = True
    while added:
        added = False
        for path in list(remaining):
            rel = path.relative_to(REPO).as_posix()
            names = {rel, path.name}
            # A composite action is referenced by its DIRECTORY
            # (`uses: ./.github/actions/install-unsloth-local`), never by the
            # action.yml inside it, so matching only the file path never opens one
            # and a driver it launches reads as an orphan. Reported on PR #9060.
            if path.name in ("action.yml", "action.yaml"):
                names.add(path.parent.relative_to(REPO).as_posix())
            if any(name in text for name in names):
                parts.append(path.read_text(encoding = "utf-8", errors = "replace"))
                remaining.remove(path)
                text = "\n".join(parts)
                added = True
    return text


def test_every_playwright_driver_is_invoked_by_ci():
    text = _ci_text()
    orphans = sorted(
        driver.name
        for driver in DRIVERS
        if driver.name not in NOT_IN_CI and driver.name not in text
    )
    assert not orphans, (
        f"{len(orphans)} Playwright suite(s) under tests/studio are not named by any workflow "
        f"or CI script, so they run nowhere and every job stays green: {orphans}. Add the step, "
        f"or add the file to NOT_IN_CI with the reason."
    )


def test_the_exemptions_are_still_exempt_and_still_exist():
    """An exemption that outlives its file, or its reason, quietly shrinks the check."""
    names = {driver.name for driver in DRIVERS}
    missing = sorted(NOT_IN_CI - names)
    assert not missing, f"NOT_IN_CI names files that no longer exist: {missing}"
    text = _ci_text()
    now_covered = sorted(name for name in NOT_IN_CI if name in text)
    assert not now_covered, (
        f"{now_covered} are exempted from CI coverage but CI now names them. Remove them from "
        f"NOT_IN_CI so the check keeps guarding them."
    )


def test_the_linux_job_still_drives_all_three_browser_engines():
    """The repo-wide check cannot see this job disappear.

    run-studio-indicator-browser.sh is named by the Mac and Windows UI workflows too,
    so deleting all three calls from the Linux workflow leaves every guard above green
    while the Chromium/Firefox/WebKit coverage this job exists for is gone. Reported on
    PR #9060. Asserted against the job, not the file: a step moved back into ui-smoke
    would put it behind the 30-minute limit this change moved it out of.
    """
    document = yaml.safe_load(
        (REPO / ".github" / "workflows" / "studio-ui-smoke.yml").read_text(encoding = "utf-8")
    )
    job = document["jobs"]["ui-indicator"]
    runs = "\n".join(
        str(step.get("run", "")) for step in job["steps"] if isinstance(step, dict)
    )
    missing = [
        engine
        for engine in ("chromium", "firefox", "webkit")
        if f"run-studio-indicator-browser.sh 18899 {engine}" not in runs
    ]
    assert not missing, (
        f"the ui-indicator job no longer drives {missing}. That is the cross-browser "
        f"coverage this job was split out to keep, and the repo-wide check above cannot "
        f"see it go: the Mac and Windows workflows name the same helper."
    )


def test_the_scan_reads_the_workflows_it_claims_to():
    """A scan that read nothing would pass both checks above on anything."""
    assert len(DRIVERS) > 10, f"only found {len(DRIVERS)} drivers; the glob is wrong"
    text = _ci_text()
    assert "run-studio-indicator-browser.sh" in text
    assert "playwright_loaded_models_indicator.py" in text, (
        "the indicator driver is named by a CI script, so a scan that misses it is not "
        "reading .github/scripts"
    )
    # A composite action is reached by its directory, so a walk that only matched the
    # action.yml path would never open one. install-unsloth-local is the repo's only
    # composite action and several workflows use it.
    assert "actions/install-unsloth-local" in text
    # A line from inside that action's body, not from any workflow, so this fails if the
    # walk matched the `uses:` reference without ever opening the action.
    assert "The POSIX `install.sh --local --no-torch` bootstrap" in text, (
        "the composite action's own contents are not in the text, so a driver launched "
        "from inside one would read as an orphan"
    )
