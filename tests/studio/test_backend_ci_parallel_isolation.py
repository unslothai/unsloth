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


@pytest.mark.parametrize("path, reason", ISOLATED, ids = [p for p, _ in ISOLATED])
def test_an_isolated_path_is_ignored_by_every_parallel_pytest_run(path, reason):
    for command in _pytest_commands(WORKFLOW.read_text(encoding = "utf-8")):
        if " -n " not in f" {command} ":
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
    assert len(parallel) == 1, f"expected exactly one parallel pytest run, got {parallel}"
    assert "tests/" in parallel[0]
    # The line joins have to be resolved, or the parallel command reads as `pytest tests/ -q`
    # with none of its --ignore flags and the first test above passes on nothing.
    assert "--ignore=" in parallel[0]
    assert len(commands) > 1, "no serial pytest steps found; the ignore checks cannot fail"
