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

import ast
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
    ("tests/test_llama_cpp_wait_for_vram_settle.py", "asserts elapsed < 0.05"),
    ("tests/test_tool_xml_strip.py", "asserts a regex benchmark under 0.1s"),
]

# Below this, an elapsed-time bound is inside the range of a single scheduler quantum, so
# under four workers on four vCPUs it measures the scheduler as much as the code. Above it
# there is enough headroom to survive being descheduled. Twenty-two backend files assert
# some elapsed bound and serialising all of them would give back most of what -n 4 buys,
# so the line is drawn where the measurement stops being about the code.
TIGHT_BOUND_S = 0.1

BACKEND_TESTS = Path(__file__).resolve().parents[2] / "studio" / "backend" / "tests"
_CLOCKS = ("monotonic", "perf_counter", "process_time", "time")


def _tight_elapsed_bounds(path: Path) -> list[str]:
    """Asserts of the form ``elapsed < <= TIGHT_BOUND_S``, where ``elapsed`` came from a clock.

    Read with ast rather than a regex, so the name has to actually be assigned from a
    difference of two clock readings. Grepping for `< 0.05` would match a tolerance on a
    float, and grepping for `elapsed` would match a variable that holds anything.
    """
    try:
        tree = ast.parse(path.read_text(encoding = "utf-8", errors = "replace"))
    except SyntaxError:
        return []
    timed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
            if not isinstance(node.value.op, ast.Sub):
                continue
            reads_clock = any(
                isinstance(inner, ast.Call) and getattr(inner.func, "attr", "") in _CLOCKS
                for inner in ast.walk(node.value)
            )
            if reads_clock:
                timed.update(t.id for t in node.targets if isinstance(t, ast.Name))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        for cmp_node in ast.walk(node.test):
            if not isinstance(cmp_node, ast.Compare):
                continue
            if not (isinstance(cmp_node.left, ast.Name) and cmp_node.left.id in timed):
                continue
            for op, bound in zip(cmp_node.ops, cmp_node.comparators):
                if not isinstance(op, (ast.Lt, ast.LtE)):
                    continue
                if isinstance(bound, ast.Constant) and isinstance(bound.value, (int, float)):
                    if bound.value <= TIGHT_BOUND_S:
                        found.append(f"{path.name}:{node.lineno} {cmp_node.left.id} < {bound.value}")
    return found

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


def test_every_tight_elapsed_bound_is_isolated():
    """The rule, applied by scanning rather than by memory.

    Two of the entries above were found by review rather than by CI: they passed on
    staging and would have flaked later. A new test asserting a 20ms bound would do the
    same. This finds them, so adding one forces the isolation instead of buying a flake.
    """
    isolated = {path for path, _ in BACKEND_ISOLATED}
    stray = {}
    for path in sorted(BACKEND_TESTS.glob("*.py")):
        bounds = _tight_elapsed_bounds(path)
        if bounds and f"tests/{path.name}" not in isolated:
            stray[path.name] = bounds
    assert not stray, (
        f"these backend tests assert an elapsed-time bound at or below {TIGHT_BOUND_S}s "
        f"and still run under -n 4, where four workers share four vCPUs and a bound that "
        f"small measures the scheduler as much as the code: {stray}. Either add the file "
        f"to BACKEND_ISOLATED and to both halves of the workflow, or give the assertion "
        f"enough headroom to survive being descheduled."
    )


def test_the_tight_bound_scan_finds_the_known_ones():
    """A scan that matched nothing would pass the test above on an empty set."""
    found = {
        path.name: _tight_elapsed_bounds(path)
        for path in sorted(BACKEND_TESTS.glob("*.py"))
        if _tight_elapsed_bounds(path)
    }
    assert "test_llama_cpp_wait_for_vram_settle.py" in found, found
    assert "test_tool_xml_strip.py" in found, found
