# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A crashed xdist worker must fail the CI step quickly and name the test, not wedge it.

Under ``--dist loadgroup`` / ``loadscope`` / ``loadfile`` (all ``LoadScopeScheduling``), xdist 3.8 requeues a
crashed worker's unit, crashed test included, and hands the replacement worker exactly one unit. A worker never
starts its last queued test until it gets a second one or a shutdown, so the controller waits forever and the
step ends at the outer ``timeout`` 30 minutes later with nothing named. ``--timeout-method=thread`` ends a hung
worker by design, so any hang turned into that. ``--max-worker-restart=0`` makes the crash end the run, and
``faulthandler_timeout`` (written to fd 2, which reaches the log; pytest-timeout's own dump is lost under xdist)
shows where the test hung.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import textwrap
import time

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_SCOPED_DIST = re.compile(r"--dist[= ]load(?:group|scope|file)\b")


def _pytest_runs():
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
        for job_name, job in (workflow.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                # One shell command per line once the backslash continuations are joined.
                for command in re.sub(r"\\\n", " ", step.get("run") or "").splitlines():
                    if "pytest" in command and _SCOPED_DIST.search(command):
                        yield f"{path.name}:{job_name}:{step.get('name')}", command


_RUNS = list(_pytest_runs())


def test_the_scan_sees_the_known_scoped_dist_steps():
    files = {name.split(":")[0] for name, _ in _RUNS}
    assert {
        "studio-backend-ci.yml",
        "distributed-cpu-ci.yml",
        "consolidated-tests-ci.yml",
        "version-compat-ci.yml",
    } <= files, files


@pytest.mark.parametrize("name,command", _RUNS, ids = [name for name, _ in _RUNS])
def test_every_scoped_dist_run_fails_fast_on_a_crashed_worker(name, command):
    assert "--max-worker-restart=0" in command, (
        f"{name} runs xdist with a scope-based --dist but lets it restart crashed workers; "
        "the replacement never runs its one requeued unit, so a crash wedges the step until the outer timeout"
    )
    per_test = re.search(r"--timeout[= ](\d+)", command)
    if per_test:
        dump = re.search(r"faulthandler_timeout=(\d+)", command)
        assert dump and int(dump.group(1)) < int(per_test.group(1)), (
            f"{name}: --timeout={per_test.group(1)} ends a hung worker, and under xdist pytest-timeout's stack dump "
            "is lost; set -o faulthandler_timeout below it so the log shows where the test hung"
        )


def test_a_crashed_loadgroup_worker_ends_the_run_and_names_the_test(tmp_path):
    pytest.importorskip("xdist")
    pytest.importorskip("pytest_timeout")
    (tmp_path / "test_wedge.py").write_text(
        textwrap.dedent(
            """
            import time
            import pytest

            def test_hangs():
                time.sleep(600)

            @pytest.mark.parametrize("i", range(6))
            def test_ok(i):
                time.sleep(0.5)
            """
        ),
        encoding = "utf-8",
    )
    flags = "-n 2 --dist loadgroup --max-worker-restart=0 --timeout=8 --timeout-method=thread"
    started = time.monotonic()
    try:
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:randomly",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                "-o",
                "faulthandler_timeout=4",
                *flags.split(),
                "test_wedge.py",
            ],
            cwd = tmp_path,
            capture_output = True,
            text = True,
            timeout = 150,
        )
    except subprocess.TimeoutExpired as exc:
        partial = (
            exc.stdout.decode(errors = "replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout or ""
        )
        pytest.fail(f"the xdist controller wedged after the worker crash:\n{partial[-2000:]}")
    out = done.stdout + done.stderr
    assert done.returncode != 0, out[-3000:]
    assert "crashed while running 'test_wedge.py::test_hangs'" in out, out[-3000:]
    assert (
        'test_wedge.py", line 6 in test_hangs' in out
    ), f"no stack dump names the hung line:\n{out[-3000:]}"
    assert time.monotonic() - started < 120, out[-3000:]
