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
from fnmatch import fnmatch
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DRIVERS = sorted((REPO / "tests" / "studio").glob("playwright_*.py"))

# Drivers no workflow runs, each with why.
NOT_IN_CI = {
    # Needs the Tauri desktop shell (it serves a page under the real Tauri CSP), which no Linux runner in this repo
    # builds.
    # tests/studio/ test_tauri_python_tool_images.py asserts the policy statically instead.
    "playwright_tauri_python_tool_images.py",
    # Drives the Train page pickers, which need a dataset and a model resolved through huggingface_hub;
    # the UI workflows deliberately boot API-only with one 254 MiB GGUF and no network model resolution.
    "playwright_train_pickers.py",
    # A measurement harness rather than a gate: it prints the per-N cost table #8977 was sized from and deliberately
    # sets no budget, and the sizes that make the curve mean anything (to 500 messages under 6x CPU throttling) cost
    # tens of minutes.
    # The part of it that can go wrong silently, the verdict in harness_failures, is driven without a browser by
    # test_autoscroll_harness_contract.py, which CI does run.
    # What it asserts that CANNOT go stale silently, the flag wiring and the absence of a measurement in the unmeasured
    # primitive, is covered without a browser by studio/frontend/tests/reasoning-grid-collapse.test.ts, which CI does
    # run.
    "playwright_thread_weight.py",
    # The same shape as playwright_thread_weight.py: a measurement harness, not a
    # gate. It prints what a collapsible toggle costs against document size, and it
    # deliberately sets no budget, because the number is hardware-dependent and a
    # threshold here would be flaky rather than informative. The cells that make the
    # O(total layout objects) curve readable run 100k+ element documents and cost
    # minutes. Run by hand when that curve needs re-measuring. What it asserts that
    # CANNOT go stale silently, the flag wiring and the absence of a measurement in
    # the unmeasured primitive, is covered without a browser by
    # studio/frontend/tests/reasoning-grid-collapse.test.ts, which CI does run.
    "playwright_collapse_layout.py",
    # Half of what it asserts is about the engine Frontend CI does not install.
    # It proves the thread's fast copy path byte for byte against the real clipboard on BOTH engines: Chromium answers
    # and must match, WebKit must refuse and its refusal must be backed by a measured divergence.
    # What CI does run is studio/frontend/tests/thread-fast-copy.test.ts, which pins the gate's branches and the patch's
    # bookkeeping but, by its own docstring, cannot see how a real engine serialises anything.
    "playwright_thread_fast_copy.py",
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


def test_mcp_argument_driver_launches_the_managed_studio_python():
    document = yaml.safe_load(
        (REPO / ".github" / "workflows" / "studio-ui-smoke.yml").read_text(encoding = "utf-8")
    )
    steps = document["jobs"]["ui-smoke"]["steps"]
    run = next(
        str(step["run"])
        for step in steps
        if step.get("name") == "MCP arguments end to end (Playwright)"
    )
    driver = (REPO / "tests" / "studio" / "playwright_mcp_arguments.py").read_text(encoding = "utf-8")

    assert 'export STUDIO_MCP_PYTHON="$studio_home/unsloth_studio/bin/python"' in run
    assert 'os.environ.get("STUDIO_MCP_PYTHON", sys.executable)' in driver


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


def test_tool_activity_install_enforces_the_script_allowlist():
    document = yaml.safe_load(
        (REPO / ".github" / "workflows" / "studio-ui-smoke.yml").read_text(encoding = "utf-8")
    )
    steps = document["jobs"]["ui-smoke"]["steps"]
    run = next(
        str(step["run"])
        for step in steps
        if step.get("name") == "Tool activity collapse regression (Playwright)"
    )
    upgrade = "npm install -g npm@^11"
    version_gate = "11.1[6-9].*|11.[2-9][0-9].*|1[2-9].*"
    install = "npm --prefix studio/frontend ci --strict-allow-scripts"
    assert upgrade in run
    assert version_gate in run
    assert install in run
    assert run.index(upgrade) < run.index(version_gate) < run.index(install)


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
    # Two narrowings, each closing a way this check could pass on nothing:
    #   uncommented -- commenting a line out is how an invocation gets disabled, and a raw substring
    #     test reads `# bash ...engine` as coverage.
    #   only the steps that invoke the helper -- scanning every run in the job would read
    #     `playwright install --with-deps chromium firefox webkit` as coverage, and that step names
    #     all three engines whether or not any of them is ever driven.
    runs = _uncommented(
        "\n".join(
            str(step.get("run", ""))
            for step in job["steps"]
            if isinstance(step, dict)
            and "run-studio-indicator-browser.sh" in str(step.get("run", ""))
        )
    )
    # Matched as "the helper is invoked, and each engine is named as a bare argument to
    # it", not as the literal `...sh 18899 <engine>` this once was. The engines now run
    # concurrently from a loop over "<port> <engine>" pairs, each with its own port, so the
    # old form no longer appears anywhere even though all three still run.
    #
    # The property being guarded is unchanged, and is the one that matters: the Mac and
    # Windows UI workflows name the same helper, so dropping an engine HERE is invisible to
    # the repo-wide scan above. What the relaxation gives up is the port literal, which
    # this check was never really about; test_indicator_browsers_run_in_parallel.py asserts
    # the ports are distinct, which is the property the number was standing in for.
    assert (
        "run-studio-indicator-browser.sh" in runs
    ), "the ui-indicator job no longer invokes the cross-browser indicator helper at all"
    missing = [
        engine
        for engine in ("chromium", "firefox", "webkit")
        if not re.search(rf"(?<![\w-]){re.escape(engine)}(?![\w-])", runs)
    ]
    assert not missing, (
        f"the ui-indicator job no longer drives {missing}. That is the cross-browser "
        f"coverage this job was split out to keep, and the repo-wide check above cannot "
        f"see it go: the Mac and Windows workflows name the same helper."
    )
    # And the disabled form does not read as coverage, or the check above passes on commented-out commands.
    disabled = _uncommented(
        "\n".join(
            f"# bash run-studio-indicator-browser.sh 18899 {e}" for e in ("chromium", "webkit")
        )
    )
    assert "chromium" not in disabled and "webkit" not in disabled


def test_no_build_gate_sits_behind_a_browser_smoke():
    """A smoke failure must not decide whether the build gates report.

    Every step carries an implicit `if: success()`, so a job stops at its first failing
    step and skips the rest. The ANSI smoke is intermittently red, and while it ran ahead
    of them the build and the three bundle assertions never reported at all on those runs.
    The smokes each start their own vite dev server and read nothing out of `dist/`, so
    they belong last. Asserted by step index, since the ordering is the whole guarantee.
    """
    document = yaml.safe_load(
        (REPO / ".github" / "workflows" / "studio-frontend-ci.yml").read_text(encoding = "utf-8")
    )
    names = [str(step.get("name", "")) for step in document["jobs"]["build"]["steps"]]
    gates = [
        "Build",
        "Built bundle must not contain Unsloth's unstable_Provider call site",
        "Bundle size budget (75 MB)",
        "Startup bundle budget",
    ]
    missing = [gate for gate in gates if gate not in names]
    assert not missing, f"renamed or deleted build gates: {missing}; update this list"
    first_smoke = min(index for index, name in enumerate(names) if name.startswith("Browser smoke"))
    late = [gate for gate in gates if names.index(gate) > first_smoke]
    assert not late, (
        f"{late} run after {names[first_smoke]!r}, so a red browser smoke skips them and the "
        f"checks that decide whether the app ships never report. Move them above the smokes."
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
    assert "actions/install-unsloth-local" in text
    # From inside that action's body, not any workflow, so this fails if the walk matched the `uses:` reference without
    # opening the action.
    assert "The POSIX `install.sh --local --no-torch` bootstrap" in text, (
        "the composite action's own contents are not in the text, so a driver launched "
        "from inside one would read as an orphan"
    )


def test_every_smoke_report_is_covered_by_the_failure_upload():
    """A smoke that fails must have its own diagnostic in the artifact.

    The upload runs `if: failure()`, so the ONE report worth having is the one the
    smoke that just failed wrote. Four of the five write `logs/playwright-<name>`;
    the settings smoke writes a JSON report under its own name, so a bare
    `logs/playwright-*` path uploaded every report except that one.
    """
    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "studio-frontend-ci.yml").read_text(encoding = "utf-8")
    )
    steps = workflow["jobs"]["build"]["steps"]
    upload = next(s for s in steps if s.get("name") == "Upload browser smoke artifacts")
    patterns = [line.strip() for line in str(upload["with"]["path"]).splitlines() if line.strip()]

    # Every logs/ path the smokes CI runs actually write, read from their source.
    run = " ".join(str(s.get("run", "")) for s in steps)
    smokes = [d for d in DRIVERS if d.name in run]
    assert len(smokes) >= 5, f"expected the browser smokes to be wired up, found {len(smokes)}"

    uncovered = []
    for driver in smokes:
        text = driver.read_text(encoding = "utf-8")
        for out in sorted(set(re.findall(r'"(logs/[^"]+)"', text))):
            stem = out.split("%")[0].split("{")[0]
            if not any(fnmatch(stem, p) or stem.startswith(p.rstrip("*")) for p in patterns):
                uncovered.append(f"{driver.name} -> {out}")
    assert not uncovered, (
        f"these smoke reports are not in the failure upload: {uncovered}; a failing smoke "
        f"would upload every report except its own. Upload patterns: {patterns}"
    )


def test_a_continue_on_error_smoke_can_still_upload_its_report():
    """`failure()` cannot see a smoke that is allowed to fail.

    `continue-on-error: true` rewrites a step's CONCLUSION to success while leaving its
    OUTCOME as failure, so a bare `if: failure()` upload is skipped on exactly the runs
    where the non-blocking smoke is the only thing that failed, which is when its report
    is the whole point. Each such smoke must be named in the upload condition.
    """
    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "studio-frontend-ci.yml").read_text(encoding = "utf-8")
    )
    steps = workflow["jobs"]["build"]["steps"]
    upload = next(s for s in steps if s.get("name") == "Upload browser smoke artifacts")
    condition = str(upload.get("if", ""))

    lenient = [
        s
        for s in steps
        if s.get("continue-on-error") and str(s.get("name", "")).startswith("Browser smoke")
    ]
    assert lenient, "expected at least one continue-on-error browser smoke; did one get renamed?"

    unseen = []
    for step in lenient:
        step_id = step.get("id")
        if not step_id:
            unseen.append(f"{step['name']!r} has no id, so the upload cannot reference it")
        elif f"steps.{step_id}.outcome" not in condition:
            unseen.append(f"{step['name']!r} (id {step_id}) is not in the upload condition")
    assert not unseen, (
        f"{unseen}; a continue-on-error smoke that fails alone leaves conclusion=success, so "
        f"`{condition}` skips the upload and its report is lost."
    )
