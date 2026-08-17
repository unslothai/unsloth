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


def _ci_text() -> str:
    """Everything CI could name a driver from, reachable from a workflow.

    Reachability is the whole point. Reading every file under .github/scripts
    unconditionally counts an orphaned helper as coverage: delete the workflow step
    that calls run-studio-indicator-browser.sh, leave the script in the tree, and
    the driver it names is still in this text while the suite runs nowhere. That is
    the regression these tests exist to catch. So the workflows seed the text, and
    a helper joins only once something already in it names the helper -- repeatedly,
    since one helper may call another.
    """
    helpers = [
        path
        for directory in ((REPO / ".github" / "scripts"), (REPO / ".github" / "actions"))
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    parts = [
        path.read_text(encoding = "utf-8", errors = "replace")
        for path in sorted((REPO / ".github" / "workflows").rglob("*"))
        if path.is_file()
    ]
    text = "\n".join(parts)
    remaining = list(helpers)
    added = True
    while added:
        added = False
        for path in list(remaining):
            rel = path.relative_to(REPO).as_posix()
            if rel in text or path.name in text:
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


def test_the_scan_reads_the_workflows_it_claims_to():
    """A scan that read nothing would pass both checks above on anything."""
    assert len(DRIVERS) > 10, f"only found {len(DRIVERS)} drivers; the glob is wrong"
    text = _ci_text()
    assert "run-studio-indicator-browser.sh" in text
    assert "playwright_loaded_models_indicator.py" in text, (
        "the indicator driver is named by a CI script, so a scan that misses it is not "
        "reading .github/scripts"
    )
