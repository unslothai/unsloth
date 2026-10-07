# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The unsloth_zoo revision notice must never be the reason a Core cell goes red.

It only says which revision was tested, and by the time it runs the clone, the install and
the whole suite have passed. Under `set -euxo pipefail` that is easy to get wrong: for
`head="$(a | b)"` the pipeline's status becomes the assignment's and `-e` ends the step, so
a `git ls-remote` that loses a DNS lookup would take a green cell with it.

Run rather than read: the block is extracted from the action and executed under the same
shell flags, which is the only way to answer a question about `set -e` semantics.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import textwrap
from pathlib import Path

import pytest
import yaml


_REPO = Path(__file__).resolve().parents[2]
_ACTION = _REPO / ".github" / "actions" / "core-cpu-setup" / "action.yml"

# Executed, not read: needs bash and GNU timeout, which stock macOS lacks. The action
# only runs on ubuntu-24.04, so skipping loses no coverage.
_MISSING = [tool for tool in ("bash", "timeout") if shutil.which(tool) is None]

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason = f"needs {' and '.join(_MISSING)} to execute the extracted block",
)


def _clone_step() -> dict:
    steps = yaml.safe_load(_ACTION.read_text(encoding = "utf-8"))["runs"]["steps"]
    for step in steps:
        if "run" in step and "unsloth-zoo" in step["run"]:
            return step
    raise AssertionError(f"no unsloth_zoo clone step in {_ACTION}")


def _revision_notice_block() -> str:
    """The lines from the resolved-revision lookup to the end of the step."""
    body = _clone_step()["run"]
    start = body.index('head="$(')
    block = body[start:]
    # A ${{ }} left in would run as literal text and quietly test the wrong thing.
    assert "${{" not in block, f"unexpected GitHub expression in the notice block:\n{block}"
    return block


def test_the_step_declares_the_shell_flags_this_test_reproduces():
    """If the step stops running under -e / pipefail, this test is checking a phantom."""
    body = _clone_step()["run"]
    flags = [line.strip() for line in body.splitlines() if line.strip().startswith("set -")]
    assert flags, "the clone step no longer sets shell flags; re-point this test"
    assert any("e" in f.split()[1] and "pipefail" in f for f in flags), flags
    assert _clone_step().get("shell") == "bash", _clone_step().get("shell")


def _lookup_timeout_seconds() -> int:
    """The bound the action puts on the lookup, read from the action itself."""
    block = _revision_notice_block()
    found = re.search(r"timeout\s+(\d+)\s+git ls-remote", block)
    assert found, f"the lookup is no longer bounded by `timeout`:\n{block}"
    seconds = int(found.group(1))
    assert 0 < seconds <= 30, f"an unreasonable bound for a diagnostic: {seconds}s"
    return seconds


def _run_notice(
    tmp_path: Path,
    *,
    git_exit: int,
    git_stdout: str,
    git_sleep: int = 0,
) -> subprocess.CompletedProcess:
    """Execute the notice block with `git` stubbed, under the step's own flags."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    git = stub_dir / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "ls-remote" ]; then\n'
        f"  sleep {git_sleep}\n"
        # %b, not %s: the repr writes tabs as escapes, and cut -f1 needs a real tab.
        f"  printf '%b' {git_stdout!r}\n"
        f"  exit {git_exit}\n"
        "fi\n"
        "echo 0000000000000000000000000000000000000000\n",
        encoding = "utf-8",
    )
    git.chmod(0o755)

    script = tmp_path / "notice.sh"
    script.write_text(
        "set -euxo pipefail\n"
        'RUNNER_TEMP="$1"\n'
        + textwrap.dedent(_revision_notice_block())
        + "\necho NOTICE_BLOCK_SURVIVED\n",
        encoding = "utf-8",
    )
    # The caller's real PATH, not a fixed one, so the skip check and the run agree.
    env = {
        "PATH": f"{stub_dir}{os.pathsep}{os.environ.get('PATH', os.defpath)}",
        "RUNNER_TEMP": str(tmp_path),
    }
    return subprocess.run(
        ["bash", str(script), str(tmp_path)],
        capture_output = True,
        text = True,
        timeout = 60,
        env = env,
    )


def test_a_failing_remote_lookup_does_not_end_the_step(tmp_path):
    """The regression this file exists for: a blip on the lookup must be survivable."""
    proc = _run_notice(tmp_path, git_exit = 128, git_stdout = "")
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        "a failing git ls-remote ended the step. Under `set -euxo pipefail` the pipeline's "
        "status becomes the assignment's, so the lookup has to absorb its own failure "
        f"inside the substitution:\n{combined[-2000:]}"
    )
    assert "NOTICE_BLOCK_SURVIVED" in proc.stdout, combined[-2000:]
    assert "::warning" not in combined, (
        "an unanswerable lookup must stay quiet rather than claim the revision is stale:\n"
        + combined[-2000:]
    )


def test_a_lookup_that_answers_nothing_also_stays_quiet(tmp_path):
    """Exit 0 with empty output, which a proxy returning a 200 and no body would give."""
    proc = _run_notice(tmp_path, git_exit = 0, git_stdout = "")
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    assert "::warning" not in proc.stdout + proc.stderr


def test_a_revision_behind_main_is_reported(tmp_path):
    """Not vacuous: the case the notice was added for must still produce the warning."""
    proc = _run_notice(
        tmp_path,
        git_exit = 0,
        git_stdout = "1111111111111111111111111111111111111111\trefs/heads/main\n",
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined[-2000:]
    assert "::warning" in combined, f"a stale revision produced no warning:\n{combined[-2000:]}"
    assert "1111111111111111111111111111111111111111" in combined, combined[-2000:]


def test_the_matching_revision_is_not_reported_as_stale(tmp_path):
    """The other direction, so the warning cannot degrade into always firing."""
    same = "0000000000000000000000000000000000000000"
    proc = _run_notice(tmp_path, git_exit = 0, git_stdout = f"{same}\trefs/heads/main\n")
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined[-2000:]
    assert (
        "::warning" not in combined
    ), f"the pinned revision equals main and was still called stale:\n{combined[-2000:]}"


def test_the_lookup_absorbs_its_failure_inside_the_substitution():
    """Structural companion to the behavioural tests above.

    They would also pass if someone moved the guard to `|| true` AFTER the closing paren,
    which silences the assignment's status but leaves the pipeline's own failure to be
    caught by pipefail first in some shells. Pin the shape that is actually correct.
    """
    block = _revision_notice_block()
    lookup = re.search(r'head="\$\((.*?)\)"', block, flags = re.S)
    assert lookup, f"the resolved-revision lookup changed shape:\n{block}"
    assert "|| true" in lookup.group(1), (
        "the lookup must absorb its own failure inside the command substitution, not "
        f"after it:\n{lookup.group(0)}"
    )


def test_a_hanging_remote_lookup_does_not_hold_the_step(tmp_path):
    """A lookup that stalls rather than fails must not run out the job's clock.

    `|| true` cannot help: nothing has exited, so nothing is absorbed. Takes the bound in
    real time, which is why it is the slowest test here.
    """
    bound = _lookup_timeout_seconds()
    began = time.monotonic()
    proc = _run_notice(tmp_path, git_exit = 0, git_stdout = "", git_sleep = bound * 6)
    elapsed = time.monotonic() - began

    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined[-2000:]
    assert "NOTICE_BLOCK_SURVIVED" in proc.stdout, combined[-2000:]
    assert elapsed < bound + 20, (
        f"the block took {elapsed:.1f}s against a {bound}s bound, so the lookup is not "
        "actually bounded and a stalled connection would hold the step open"
    )
    assert "::warning" not in combined, (
        "a lookup that never answered must stay quiet rather than claim staleness:\n"
        + combined[-2000:]
    )


def test_the_warning_does_not_prescribe_a_remedy_that_cannot_work(tmp_path):
    """A pinned dispatch, a moving main and a stale re-run are indistinguishable by the
    time the action sees a sha, and re-running is the remedy for only the last. So the
    message has to hold for all three rather than assert one."""
    proc = _run_notice(
        tmp_path,
        git_exit = 0,
        git_stdout = "1111111111111111111111111111111111111111\trefs/heads/main\n",
    )
    combined = proc.stdout + proc.stderr
    warning = next((line for line in combined.splitlines() if "::warning" in line), None)
    assert warning is not None, combined[-2000:]
    lowered = warning.lower()
    assert "pinned" in lowered or "deliberate" in lowered, (
        "the warning tells the reader to re-run without allowing that the ref may have "
        f"been pinned on purpose, which re-running will not change:\n{warning}"
    )
    # A first attempt can mismatch too: main may advance between resolve-zoo-ref and this step.
    assert (
        "advanced" in lowered
    ), f"the warning does not allow for main moving after the resolve job:\n{warning}"
    assert (
        "re-run all jobs" in lowered
    ), f"the warning no longer names the remedy for the stale-re-run case:\n{warning}"


def test_the_suite_that_runs_this_guard_triggers_on_the_action_it_guards():
    """A guard absent for the change it guards is not a guard.

    Backend CI runs tests/studio and is path-filtered; without .github/actions in that
    filter an action-only PR ran Core, which does not execute tests/studio, and skipped the
    suite that does. test_local_actions_are_in_path_filters.py enforces that a workflow
    lists the actions it `uses:`; this is the other direction, the ones its TESTS read.
    """
    workflow = yaml.safe_load(
        (_REPO / ".github" / "workflows" / "studio-backend-ci.yml").read_text(encoding = "utf-8")
    )
    triggers = workflow.get(True) or workflow.get("on") or {}
    paths = (triggers.get("pull_request") or {}).get("paths") or []
    assert paths, "Backend CI lost its paths filter; this guard no longer applies"

    action = _ACTION.relative_to(_REPO).parent.as_posix()
    covered = [
        p
        for p in paths
        if action.startswith(p.rstrip("*").rstrip("/")) and p.rstrip().endswith("**")
    ]
    assert covered, (
        f"Backend CI runs this test but does not trigger on {action}, so a PR that "
        "changes only the action will not run the guard that covers it. Add "
        f"'.github/actions/**' to its paths:\n  " + "\n  ".join(paths)
    )
