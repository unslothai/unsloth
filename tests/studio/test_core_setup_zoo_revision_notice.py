# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The unsloth_zoo revision notice must never be the reason a Core cell goes red.

It exists only to say which revision was tested, because `gh run rerun --failed` reuses
the sha `resolve-zoo-ref` handed out on the first attempt and a re-run therefore re-tests
a revision that can be days old. A diagnostic that can itself fail the job would cost more
than it explains: the clone, the install and the whole suite have already passed by the
time it runs.

The hazard is not hypothetical. The step runs under `set -euxo pipefail`, and for
`head="$(a | b)"` pipefail hands `a`'s status to the pipeline, the pipeline's status
becomes the assignment's, and `-e` ends the step. So a `git ls-remote` that loses a DNS
lookup takes a green cell down with it.

Run rather than read: the block is extracted from the action and executed under the same
shell flags with a `git` that fails, which is the only way to answer a question about
`set -e` semantics.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml


_REPO = Path(__file__).resolve().parents[2]
_ACTION = _REPO / ".github" / "actions" / "core-cpu-setup" / "action.yml"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason = "needs bash to execute the extracted block"
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
    # The step interpolates nothing in this tail, but assert that rather than assume it:
    # a ${{ }} left in would run as literal text and quietly test the wrong thing.
    assert "${{" not in block, f"unexpected GitHub expression in the notice block:\n{block}"
    return block


def test_the_step_declares_the_shell_flags_this_test_reproduces():
    """If the step stops running under -e / pipefail, this test is checking a phantom."""
    body = _clone_step()["run"]
    flags = [line.strip() for line in body.splitlines() if line.strip().startswith("set -")]
    assert flags, "the clone step no longer sets shell flags; re-point this test"
    assert any("e" in f.split()[1] and "pipefail" in f for f in flags), flags
    assert _clone_step().get("shell") == "bash", _clone_step().get("shell")


def _run_notice(tmp_path: Path, *, git_exit: int, git_stdout: str) -> subprocess.CompletedProcess:
    """Execute the notice block with `git` stubbed, under the step's own flags."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    git = stub_dir / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "ls-remote" ]; then\n'
        # %b, not %s: the repr below writes the tab and newline as backslash escapes, and
        # ls-remote output is tab-separated. With %s they stay literal, cut -f1 finds no
        # field separator and hands back the whole line, which reads as a mismatch.
        f"  printf '%b' {git_stdout!r}\n"
        f"  exit {git_exit}\n"
        "fi\n"
        # rev-parse, standing in for the clone this test does not make.
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
    env = {"PATH": f"{stub_dir}:/usr/bin:/bin", "RUNNER_TEMP": str(tmp_path)}
    return subprocess.run(
        ["bash", str(script), str(tmp_path)],
        capture_output = True, text = True, timeout = 60, env = env,
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
    assert "::warning" not in combined, (
        f"the pinned revision equals main and was still called stale:\n{combined[-2000:]}"
    )


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
