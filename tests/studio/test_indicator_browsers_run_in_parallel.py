# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The three loaded-models-indicator engines run concurrently, and stay isolated.

`Loaded-models indicator (cross-browser)` was the longest Linux job in CI at ~1000s, and
~870s of that was one step running the same suite three times in a row, once per engine.
The runs are disjoint -- each boots its own server and drives its own browser -- so the
serialisation bought nothing but wall-clock.

What made it non-trivial is that `run-studio-indicator-browser.sh` does this:

    rm -rf "$studio_home/auth"        # so the boot mints a fresh password
    ...
    old_password=$(cat "$studio_home/auth/.bootstrap_password")

On a shared `$studio_home` that is a destructive race: one engine's wipe lands between
another's mint and its read, and the reader either fails to open the file or logs in with a
password that is no longer valid. It would not fail cleanly or every time, which is exactly
why it is asserted here rather than left to review.

The isolation is a per-engine `UNSLOTH_STUDIO_HOME`, and it is cheap only because that
variable selects a DATA root:

  * `unsloth` on PATH still resolves the installed venv, so no reinstall.
  * the frontend is served from a PACKAGE-relative path (`_DEFAULT_FRONTEND_PATH` in
    `studio/backend/run.py` is `<pkg>/../frontend/dist`), NOT from `studio_root()`, so no
    per-engine frontend build.
  * the suite is `UNSLOTH_API_ONLY` with the four /status endpoints stubbed via
    `page.route`, so a fresh home needs no model, no GPU and no llama.cpp build.

That third point is what limits this pattern to this suite. A job whose home holds a
downloaded model must not copy it: `test_a_fresh_home_is_only_safe_for_an_api_only_suite`
pins the property the cheapness depends on.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-ui-smoke.yml"
SCRIPT = REPO / ".github" / "scripts" / "run-studio-indicator-browser.sh"

ENGINES = ("chromium", "firefox", "webkit")


def _indicator_step() -> dict:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    for job in doc["jobs"].values():
        for step in job.get("steps") or []:
            if "loaded-models indicator" in (step.get("name") or "").lower():
                return step
    raise AssertionError("no step in studio-ui-smoke.yml runs the loaded-models indicator")


def test_the_engines_are_launched_concurrently_rather_than_one_after_another():
    run = str(_indicator_step().get("run", ""))
    assert "&" in run and "wait" in run, (
        "the indicator step no longer backgrounds its engine runs. Three sequential runs of "
        "this suite is ~870s of the job's ~1000s, for work that is fully disjoint."
    )


def test_every_engine_still_runs():
    run = str(_indicator_step().get("run", ""))
    missing = [e for e in ENGINES if e not in run]
    assert not missing, (
        f"{missing} no longer run in the indicator step. Parallelising a suite must not "
        f"quietly drop an engine -- webkit and firefox are the ones that catch the layout "
        f"regressions chromium does not."
    )


def test_each_engine_gets_its_own_port():
    run = str(_indicator_step().get("run", ""))
    ports = re.findall(r"\b(\d{4,5})\s+\w+\b", run)
    assert len(set(ports)) >= len(ENGINES), (
        f"the engines do not have distinct ports ({ports}); concurrent servers cannot share "
        f"one bind address, and the second boot would fail its health wait"
    )


def test_each_engine_gets_its_own_studio_home():
    """The auth wipe/mint/read window is the race, and it is silent when it loses."""
    run = str(_indicator_step().get("run", ""))
    assert "UNSLOTH_STUDIO_HOME" in run, (
        "the indicator step runs its engines concurrently against a shared studio home. "
        "run-studio-indicator-browser.sh does `rm -rf $studio_home/auth` and then reads back "
        "$studio_home/auth/.bootstrap_password, so the engines would race on each other's "
        "credentials."
    )
    # The home has to VARY, and vary by the same token that selects the engine. A fixed
    # path would be the shared home again under a new name. Resolved through one level of
    # shell indirection, because the path is built into a local before it is exported.
    assignments = dict(re.findall(r"(?m)^\s*(\w+)=(.+?)\s*\\?$", run))
    assignment = next((l for l in run.splitlines() if "UNSLOTH_STUDIO_HOME=" in l), "")
    value = assignment.split("UNSLOTH_STUDIO_HOME=", 1)[1]
    seen, frontier = set(), re.findall(r"\$\{?(\w+)\}?", value)
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        frontier += re.findall(r"\$\{?(\w+)\}?", assignments.get(name, ""))
    engine_vars = set(
        re.findall(r"\$\{?(\w+)\}?", run[run.index("run-studio-indicator-browser.sh") :][:200])
    )
    assert seen & engine_vars, (
        f"UNSLOTH_STUDIO_HOME ({assignment.strip()!r}) does not vary by the same variable the "
        f"engine does, so the concurrent runs still share one home and race on its auth dir"
    )


def test_the_script_still_derives_the_home_it_wipes_from_the_environment():
    """The isolation above is only real while the script honours the override."""
    src = SCRIPT.read_text(encoding = "utf-8")
    assert 'studio_home="${UNSLOTH_STUDIO_HOME:-' in src, (
        "run-studio-indicator-browser.sh no longer takes its studio home from "
        "UNSLOTH_STUDIO_HOME, so the per-engine homes in the workflow are ignored and the "
        "concurrent runs share one after all"
    )
    assert 'rm -rf "$studio_home/auth"' in src, (
        "the auth wipe this isolation exists for is gone; if the script no longer wipes and "
        "re-mints, re-check whether the per-engine homes are still needed"
    )


def test_a_failure_in_one_engine_is_still_a_failure_of_the_step():
    """Backgrounding makes `set -e` stop protecting the step. This is the classic hole."""
    run = str(_indicator_step().get("run", ""))
    assert re.search(r"\bwait\b", run), "the step does not wait on its background jobs at all"
    assert re.search(r"exit\s+\"?\$", run) or "::error::" in run, (
        "the step backgrounds its engines but never propagates their exit status, so a "
        "failing engine would leave the step green"
    )


def test_all_engines_are_waited_on_before_the_step_gives_up():
    """A bail-on-first-failure would report one engine and hide the other two."""
    run = str(_indicator_step().get("run", ""))
    assert not re.search(r"wait[^\n]*\|\|\s*exit", run), (
        "the step exits on the first failing engine, so a run where two engines regress "
        "reports only one and the second surfaces days later"
    )


def test_each_isolated_home_still_reaches_the_installed_studio_venv():
    """A bare empty UNSLOTH_STUDIO_HOME is not a data root, it is an EMPTY INSTALL.

    This is the trap that made the first cut of this change fail every engine.
    UNSLOTH_STUDIO_HOME selects the CLI's install root: `_studio_venv_python()` resolves
    $UNSLOTH_STUDIO_HOME/unsloth_studio/bin/python and `_find_run_py()` globs under the
    same directory, so pointing the variable at a fresh directory makes the launch print
    "Unsloth Studio not set up. Run install.sh first." and exit BEFORE it binds a port.
    All three engines then fail identically, which reads like a bug in the suite rather
    than in the isolation.

    So each per-engine home must link the venv install.sh already built. Asserted because
    the failure mode is loud but deeply misleading about its own cause.
    """
    run = str(_indicator_step().get("run", ""))
    assert "unsloth_studio" in run, (
        "the per-engine homes no longer connect to the installed studio venv, so every "
        "launch will exit with 'Unsloth Studio not set up' before binding its port"
    )
    assert re.search(r"ln -sf?n?\s", run), (
        "the venv is no longer symlinked into each per-engine home. Copying it instead "
        "would make three copies of the install and cost more than the serialisation this "
        "change removed."
    )


def test_the_cli_still_resolves_the_venv_from_the_studio_home():
    """The symlink is only correct while the CLI looks there. Pin the path it uses."""
    src = (REPO / "unsloth_cli" / "commands" / "studio.py").read_text(encoding = "utf-8")
    assert 'STUDIO_HOME / "unsloth_studio"' in src, (
        "unsloth_cli no longer resolves the studio venv at STUDIO_HOME/unsloth_studio, so "
        "the symlink the indicator step creates may point at the wrong place; re-check "
        "what the CLI expects before trusting the isolated homes"
    )


def test_a_fresh_home_is_only_safe_for_an_api_only_suite():
    """Per-engine homes are cheap because nothing expensive lives in a home HERE.

    If this suite ever needed a real model, three fresh homes would mean three downloads
    and the parallel version would be slower than the sequential one it replaced.
    """
    src = SCRIPT.read_text(encoding = "utf-8")
    assert "UNSLOTH_API_ONLY=1" in src, (
        "run-studio-indicator-browser.sh no longer boots API-only. A fresh per-engine "
        "UNSLOTH_STUDIO_HOME is only cheap while the home holds no model and no llama.cpp "
        "build; re-measure before keeping the parallel launch."
    )


def test_the_frontend_is_served_from_the_package_not_the_studio_home():
    """The load-bearing fact: a fresh home does not mean a fresh frontend build."""
    run_py = (REPO / "studio" / "backend" / "run.py").read_text(encoding = "utf-8")
    line = next((l for l in run_py.splitlines() if l.startswith("_DEFAULT_FRONTEND_PATH")), None)
    assert line, "run.py no longer defines _DEFAULT_FRONTEND_PATH"
    assert "__file__" in line, (
        f"the default frontend path is no longer package-relative ({line.strip()!r}). If it "
        f"now resolves under studio_root(), each per-engine UNSLOTH_STUDIO_HOME needs its "
        f"own frontend build and the parallel indicator step stops being cheap."
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_each_engine_keeps_a_distinct_artifact_path(engine):
    """Shared log paths would interleave three engines into one unreadable file."""
    src = SCRIPT.read_text(encoding = "utf-8")
    for pattern in ("logs/playwright-indicator-$slug", "logs/studio-indicator-$slug.log"):
        assert (
            pattern in src
        ), f"{pattern} is no longer per-engine, so concurrent runs write over each other"
