# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards the files Backend CI deliberately keeps out of the parallel pytest run.

The repo-cpu-tests job runs tests/ under `-n 4`. Two groups cannot go through it:

- tests/studio/load_freeze asserts UPPER bounds on real elapsed time (a /health
  burst under 250 ms while a 600 ms blocking probe runs, a 100-request burst
  under 350 ms). Those bounds are the contract, so they cannot be loosened, and a
  pytest worker descheduled by the other three inflates them.
- the hardware-spoof files mutate hardware.py module globals, so they leak into
  whatever shares their worker.

Both are ignored from the parallel invocation and run again in their own serial
step. That is two edits held together by nothing, and dropping the second one is
silent: the job stays green while the tests stop running. These tests fail if the
ignore appears without a step that runs the same path, or the other way round.
"""

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "studio-backend-ci.yml"

# (ignored path, why it cannot share a worker)
ISOLATED = [
    ("tests/studio/load_freeze", "wall-clock latency bounds"),
    ("tests/studio/test_hardware_dispatch_matrix.py", "mutates hardware.py globals"),
    ("tests/studio/test_is_mlx_dispatch_gate.py", "mutates hardware.py globals"),
    ("tests/studio/test_xpu_spoof_pipeline.py", "mutates hardware.py globals"),
]


def _pytest_commands(text: str) -> list[str]:
    """Every `python -m pytest ...` invocation in the workflow, line joins resolved.

    Read off the raw text rather than the parsed YAML: a `run:` block is one
    scalar and the interesting structure is inside it, so parsing buys nothing and
    would make this depend on the job/step layout instead of the commands.
    """
    joined = re.sub(r"\\\s*\n\s*", " ", text)
    return [
        line.strip()
        for line in joined.splitlines()
        if "python -m pytest" in line and not line.lstrip().startswith("#")
    ]


# Two different trees are run in parallel now, and the isolation below belongs to exactly
# one of them. The repo-root job runs `tests/` from the checkout; the matrix job runs the
# backend's own suite with `working-directory: studio/backend`, so `tests/studio/...` is
# not a path that exists for it and demanding those ignores would be nonsense.
#
# Told apart by what they ignore, because that is the thing both the ignores and this
# scan are about: only the repo-root run excludes directories that live at the repo root.
REPO_ROOT_MARKER = "--ignore=tests/qlora"


def _over_the_repo_root(command: str) -> bool:
    return REPO_ROOT_MARKER in command


# The same pairing, for the backend matrix run. Ignoring a file from the parallel run and
# running it again serially is two edits held together by nothing, and dropping the second
# is silent: the job stays green while the tests stop running.
BACKEND_ISOLATED = [
    ("tests/test_streaming_stripper.py", "times itself against a reference in the same process"),
]

BACKEND_MARKER = "--ignore=tests/test_studio_api.py"


def _over_the_backend(command: str) -> bool:
    return BACKEND_MARKER in command


@pytest.mark.parametrize("path, reason", ISOLATED, ids = [p for p, _ in ISOLATED])
def test_an_isolated_path_is_ignored_by_every_parallel_pytest_run(path, reason):
    for command in _pytest_commands(WORKFLOW.read_text(encoding = "utf-8")):
        if " -n " not in f" {command} " or not _over_the_repo_root(command):
            continue
        assert f"--ignore={path}" in command, (
            f"{path} ({reason}) is not ignored by a parallel pytest run in "
            f"{WORKFLOW.name}, so it shares four workers on the runner's four vCPUs: {command}"
        )


@pytest.mark.parametrize("path, reason", ISOLATED, ids = [p for p, _ in ISOLATED])
def test_an_isolated_path_still_runs_in_a_serial_step(path, reason):
    """Ignoring it is half the change. Without this, the tests silently stop running."""
    serial = [
        command
        for command in _pytest_commands(WORKFLOW.read_text(encoding = "utf-8"))
        if " -n " not in f" {command} "
        and re.search(rf"(?<![\w/]){re.escape(path)}(?![\w/])", command)
    ]
    assert serial, (
        f"{path} is ignored from the parallel run ({reason}) and no serial pytest step runs "
        f"it, so it runs nowhere in {WORKFLOW.name} while the job stays green."
    )


def test_the_command_scan_sees_the_parallel_run_and_the_serial_steps():
    """Pin the parser: a scan that matched nothing would pass both tests above."""
    commands = _pytest_commands(WORKFLOW.read_text(encoding = "utf-8"))
    parallel = [command for command in commands if " -n " in f" {command} "]
    assert len(parallel) == 2, (
        f"expected two parallel pytest runs, the backend matrix and repo-cpu-tests, got "
        f"{parallel}. If a job stopped running in parallel, say so here rather than "
        f"letting this scan quietly cover one run."
    )
    root = [command for command in parallel if _over_the_repo_root(command)]
    assert len(root) == 1, (
        f"expected exactly one parallel run over the repo root, got {root}. The isolation "
        f"checks above apply to that one, and a scan that matched none of them would pass "
        f"on nothing."
    )
    # The line joins have to be resolved, or the parallel command reads as `pytest tests/ -q`
    # with none of its --ignore flags and the first test above passes on nothing.
    assert "--ignore=" in root[0]
    assert len(commands) > 1, "no serial pytest steps found; the ignore checks cannot fail"


def test_the_backend_matrix_still_runs_in_parallel():
    """The matrix leg was 23.3 minutes serial and is the longest job in the repo.

    Measured over the same tree before it was turned on: 1322.6s serial against 343.0s at
    -n 4, with the two failure sets equal name for name, so nothing in the backend suite
    depends on the order it runs in. Asserted here because dropping the flag would show up
    only as CI slowly getting slower again, which nothing reports.
    """
    backend = [
        command
        for command in _pytest_commands(WORKFLOW.read_text(encoding = "utf-8"))
        if "--ignore=tests/test_studio_api.py" in command
    ]
    assert backend, "the backend matrix pytest step is gone or was renamed past this scan"
    assert " -n " in f" {backend[0]} ", (
        f"the backend matrix leg is running serially again, which costs about 17 minutes "
        f"per leg on every pull request and every push to main: {backend[0]}"
    )


@pytest.mark.parametrize("path, reason", BACKEND_ISOLATED, ids = [p for p, _ in BACKEND_ISOLATED])
def test_a_backend_isolated_path_is_ignored_by_the_parallel_run(path, reason):
    """Relative timing cannot survive four workers on four vCPUs.

    Observed on staging: the 3.10 leg reported "early markup cost 1.354s against the
    reference's 0.854s" while 3.13 passed the same commit. One side of the ratio was
    descheduled, not slower.
    """
    parallel = [
        command
        for command in _pytest_commands(WORKFLOW.read_text(encoding = "utf-8"))
        if " -n " in f" {command} " and _over_the_backend(command)
    ]
    assert parallel, "the backend parallel run is gone or was renamed past this scan"
    assert f"--ignore={path}" in parallel[0], (
        f"{path} ({reason}) is back in the backend parallel run, where its measurements "
        f"compare a descheduled worker against an undescheduled one: {parallel[0]}"
    )


@pytest.mark.parametrize("path, reason", BACKEND_ISOLATED, ids = [p for p, _ in BACKEND_ISOLATED])
def test_a_backend_isolated_path_still_runs_serially(path, reason):
    """Ignoring it is half the change; without this it runs nowhere and the job is green."""
    serial = [
        command
        for command in _pytest_commands(WORKFLOW.read_text(encoding = "utf-8"))
        if " -n " not in f" {command} "
        and re.search(rf"(?<![\w/]){re.escape(path)}(?![\w/])", command)
    ]
    assert serial, (
        f"{path} is ignored from the backend parallel run ({reason}) and no serial step "
        f"runs it, so it runs nowhere in {WORKFLOW.name} while the job stays green."
    )
