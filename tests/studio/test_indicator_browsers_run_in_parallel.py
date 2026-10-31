# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parallel engines need their own UNSLOTH_STUDIO_HOME; a shared one lets one wipe another's auth."""

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
    # The home must vary by the engine token; resolved through one level of shell indirection.
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
    """Each per-engine home must link the installed unsloth_studio venv, or launches exit before binding."""
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
    """Fresh per-engine homes are only cheap while the suite stays API-only with no downloaded model."""
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
