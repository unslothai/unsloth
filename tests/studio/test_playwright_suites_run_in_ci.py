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

import re
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


def _uncommented(text: str) -> str:
    """``text`` with ``#`` comments removed, so a disabled command stops counting.

    Commenting an invocation out is how one gets disabled, and this scan reads `run:`
    bodies verbatim, so `# python tests/studio/x.py` used to match. Shell, YAML and
    Python all take `#` to end of line.
    """
    return "\n".join(re.sub(r"(?:^|(?<=\s))#.*", "", line) for line in text.splitlines())


def _invoked(name: str, text: str) -> bool:
    """Whether ``text`` RUNS ``name``, rather than merely mentioning it.

    Substring presence is not coverage: report.py names playwright_chat_ui.py in a
    result description, so deleting every real invocation could leave this green on
    prose. Every driver and helper here is run as an argument to an interpreter, so
    that is what is matched.
    """
    pattern = (
        rf"(?:^|[\s;&|(])(?:python3?|node|bash|sh)\s+(?:-\S+\s+)*[^\s;&|<>'\"]*{re.escape(name)}\b"
    )
    return re.search(pattern, _uncommented(text), re.M) is not None


def _executable_text(path: Path) -> str:
    """The parts of a workflow that RUN something: step `run` bodies and `uses` refs.

    Trigger paths say when CI runs, not what it runs. studio-frontend-ci.yml names
    playwright_strip_ansi_smoke.py in both, so reading the whole file left deleting
    the step alone undetected.
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

    The workflows' executable fields seed the text, and a helper joins only once
    something already in it names the helper, repeatedly since one helper may call
    another. A helper no workflow calls is not coverage.
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
            # A composite action is referenced by its DIRECTORY, never by the
            # action.yml inside it, so matching the file path never opens one.
            if path.name in ("action.yml", "action.yaml"):
                reached = path.parent.relative_to(REPO).as_posix() in text
            else:
                reached = _invoked(path.name, text) or _invoked(rel, text)
            if reached:
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
        if driver.name not in NOT_IN_CI and not _invoked(driver.name, text)
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
    now_covered = sorted(name for name in NOT_IN_CI if _invoked(name, text))
    assert not now_covered, (
        f"{now_covered} are exempted from CI coverage but CI now names them. Remove them from "
        f"NOT_IN_CI so the check keeps guarding them."
    )


def test_the_linux_job_still_drives_all_three_browser_engines():
    """The repo-wide check cannot see this job disappear.

    The Mac and Windows UI workflows name the same helper, so deleting all three calls
    from the Linux one leaves every guard above green. Asserted against the job, not
    the file: a step moved back into ui-smoke lands behind the 30-minute limit this
    change moved it out of.
    """
    document = yaml.safe_load(
        (REPO / ".github" / "workflows" / "studio-ui-smoke.yml").read_text(encoding = "utf-8")
    )
    job = document["jobs"]["ui-indicator"]
    # Uncommented for the same reason the repo-wide scan is: commenting a line out is how
    # an invocation gets disabled, and a raw substring test reads `# bash ...engine` as
    # coverage. Reported on this PR: all three could be commented out with this green.
    runs = _uncommented(
        "\n".join(str(step.get("run", "")) for step in job["steps"] if isinstance(step, dict))
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
    # And the disabled form of each of those lines does not read as coverage, or the
    # check above passes on three commented-out commands.
    disabled = _uncommented("\n".join(f"# bash x.sh 18899 {e}" for e in ("chromium", "webkit")))
    assert "18899 chromium" not in disabled and "18899 webkit" not in disabled


def test_the_scan_reads_the_workflows_it_claims_to():
    """A scan that read nothing would pass both checks above on anything."""
    assert len(DRIVERS) > 10, f"only found {len(DRIVERS)} drivers; the glob is wrong"
    text = _ci_text()
    assert "run-studio-indicator-browser.sh" in text
    assert "playwright_loaded_models_indicator.py" in text, (
        "the indicator driver is named by a CI script, so a scan that misses it is not "
        "reading .github/scripts"
    )
    assert "actions/install-unsloth-local" in text
    # From inside that action's body, not any workflow, so this fails if the walk
    # matched the `uses:` reference without opening the action.
    assert "The POSIX `install.sh --local --no-torch` bootstrap" in text, (
        "the composite action's own contents are not in the text, so a driver launched "
        "from inside one would read as an orphan"
    )
