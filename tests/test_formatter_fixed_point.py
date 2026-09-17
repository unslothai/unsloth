# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""main must be a fixed point of its own formatting hook.

pre-commit runs `ruff-format-with-kwargs` on the files a PR touches, so a file
that lands unformatted is never looked at again: the next PR to edit it inherits
a red `pre-commit.ci - pr` for a diff it did not write, and the author goes
looking for a defect that is not in their change. Two files reached main that
way and sat there, one of them for months.

The check is the real thing rather than `ruff format --check`. The hook is
`enforce_kwargs_spacing --pre`, then `ruff format`, then `enforce_kwargs_spacing`
again, and ruff only covers the middle pass, so a file can pass a ruff check
cleanly and still be rewritten by the hook. It is run over copies, so a failing
run reports the drift instead of quietly fixing it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = str(_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import enforce_kwargs_spacing  # noqa: E402
from run_ruff_format import (  # noqa: E402
    CONFIG,
    installed_ruff_version,
    pinned_ruff_version,
    version_mismatch,
)

#: What Windows' CreateProcess accepts for a whole command line, in characters. Documented by
#: Microsoft as the lpCommandLine cap and unchanged since Windows XP. POSIX has no equivalent
#: single limit (execve fails with E2BIG against ARG_MAX, ~2 MB on the runners), so Windows is
#: the binding constraint and the only number worth pinning.
_WINDOWS_COMMAND_LINE_LIMIT = 32767

#: Characters a single formatter invocation may build up to. The whole tracked set is ~2650
#: files whose copied paths under a pytest tmp_path come to ~325,000 characters -- 9.9x the cap
#: above -- so the single-invocation form this replaced could never run on Windows at all: it
#: died at CreateProcess with `[WinError 206] The filename or extension is too long` before ruff
#: opened one file. Nobody had seen it fail because every job that schedules this file is
#: ubuntu-24.04, where execve's ARG_MAX is ~2 MB and the same argv fits easily.
#:
#: A LENGTH budget rather than a file COUNT. A count has to be re-tuned every time the tracked
#: set or the path depth moves, and silently loses its margin in between: 200 files measured
#: 26,687 characters here, only 1.2x under the cap, so a fifth more files would have put it back
#: over. Accumulating by length is self-adjusting and needs no number kept in step with the
#: repo. 8000 leaves 4x headroom for the argv-quoting model below being wrong.
#:
#: Batching at all is safe because the hook is per-file: it formats each path independently, so
#: N calls over disjoint batches do exactly what one call over the union does.
_FORMAT_ARGV_BUDGET = 8000

_HOOK_ID = "ruff-format-with-kwargs"
# The spacing pass refuses to rewrite itself ("skip modifying this script to
# avoid self-edit loops"), so it is not a file the hook keeps at a fixed point
# and checking it would fail main over a rewrite that never happens. Taken from
# the module rather than spelled out, so moving or renaming it does not turn
# this into a stale exclusion of nothing.
_SELF_SKIPPED = Path(enforce_kwargs_spacing.__file__).resolve()
# The paths the hook is pointed at. `types: [python]` is what pre-commit filters
# on, and the repo tracks no .pyi, so this is the same set.
_TRACKED_GLOBS = ("*.py", "*.pyi")


def hook_exclude_pattern(config_text: str, hook_id: str) -> str | None:
    """The `exclude:` regex the named hook is configured with, or None.

    Read out of .pre-commit-config.yaml rather than copied here. A second copy of
    the exclusion list is a second thing to forget, and forgetting it in this
    direction is the expensive one: this test would format a file the hook never
    touches and fail main over it.

    Scanned rather than parsed with PyYAML, matching how the version pin is read
    next door: the block is found by its `- id:` and abandoned at the next `- id:`
    or `- repo:`, which keeps the ruff hook's own `exclude: '\\.ipynb$'` out.
    """
    lines = config_text.splitlines()
    inside = False
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(rf"-\s*id:\s*{re.escape(hook_id)}", stripped):
            inside = True
            continue
        if inside:
            if stripped.startswith("- id:") or stripped.startswith("- repo:"):
                break
            # The quoted form is tried first and keeps its contents verbatim: a
            # regex may contain a `#`, and treating that as a comment would
            # silently truncate the exclusion to a prefix that matches nothing.
            quoted = re.fullmatch(r"exclude:\s*(['\"])(.*)\1\s*(?:#.*)?", stripped)
            if quoted:
                return quoted.group(2)
            bare = re.fullmatch(r"exclude:\s*(\S+)\s*(?:#.*)?", stripped)
            if bare:
                return bare.group(1)
    return None


def eligible_files(root: Path) -> list[str]:
    """Every tracked Python file the hook would be handed, repo-relative."""
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", *_TRACKED_GLOBS],
        capture_output = True,
        text = True,
        check = True,
    )
    tracked = [name for name in out.stdout.split("\0") if name]
    pattern = hook_exclude_pattern(CONFIG.read_text(encoding = "utf-8"), _HOOK_ID)
    assert (
        pattern
    ), f"{CONFIG.name} no longer gives {_HOOK_ID} an exclude; the filter below is blind"
    excluded = re.compile(pattern)
    return [
        name
        for name in tracked
        if not excluded.search(name) and (root / name).resolve() != _SELF_SKIPPED
    ]


def _pinned_ruff_reason() -> str | None:
    """Why this cannot be checked here, or None when it can.

    ruff's formatting is not stable across releases, so another ruff answers a
    different question, and the formatter refuses to run under one anyway.
    """
    pinned = pinned_ruff_version(CONFIG.read_text(encoding = "utf-8")) if CONFIG.exists() else None
    installed = installed_ruff_version()
    if installed is None:
        return "ruff is not installed here, and the formatter cannot run without it"
    if version_mismatch(pinned, installed):
        return f"the repo is formatted with ruff {pinned}, this environment has {installed}"
    return None


def guard_verdict(ruff_reason: str | None, in_ci: bool) -> str:
    """`run`, `skip` or `fail`.

    Skipping is for a contributor who has not installed the pinned ruff; making
    them install one to run the rest of the suite would be rude. In CI it is the
    wrong answer: the runner installs the pin in a step of its own, so a missing
    ruff there means that step moved, was renamed, or a new job started calling
    `pytest tests/` without it, and the guard would go green having checked
    nothing. That is the failure this whole file exists to stop, applied to
    itself, and it costs nothing to notice.
    """
    if ruff_reason is None:
        return "run"
    return "fail" if in_ci else "skip"


def running_in_ci(environ: dict[str, str] | None = None) -> bool:
    """GitHub Actions sets both; `CI` alone covers the other providers."""
    env = os.environ if environ is None else environ
    return bool(env.get("GITHUB_ACTIONS") or env.get("CI"))


_RUFF_REASON = _pinned_ruff_reason()
_VERDICT = guard_verdict(_RUFF_REASON, running_in_ci())


class TestTheExcludeComesFromTheConfig:
    """The filter has to track the hook, not a copy of it made once."""

    def test_the_real_hook_still_names_an_exclude(self):
        pattern = hook_exclude_pattern(CONFIG.read_text(encoding = "utf-8"), _HOOK_ID)
        assert pattern, f"no exclude found for {_HOOK_ID}"
        re.compile(pattern)

    def test_it_reads_the_named_hook_and_not_a_neighbour(self):
        # The ruff hook above ours carries `exclude: '\.ipynb$'`. Picking up the
        # first exclude in the file would format the vendored tree and fail main.
        text = (
            "repos:\n"
            "  - repo: https://example.invalid/ruff\n"
            "    hooks:\n"
            "      - id: ruff\n"
            "        exclude: '\\.ipynb$'\n"
            "  - repo: local\n"
            "    hooks:\n"
            "      - id: ruff-format-with-kwargs\n"
            "        exclude: '^vendor/'\n"
            "      - id: something-else\n"
            "        exclude: '^other/'\n"
        )
        assert hook_exclude_pattern(text, "ruff-format-with-kwargs") == "^vendor/"
        assert hook_exclude_pattern(text, "ruff") == "\\.ipynb$"
        assert hook_exclude_pattern(text, "something-else") == "^other/"

    def test_a_hook_without_an_exclude_answers_none(self):
        text = "      - id: ruff-format-with-kwargs\n        entry: python x.py\n      - id: next\n"
        assert hook_exclude_pattern(text, "ruff-format-with-kwargs") is None

    def test_the_vendored_tree_and_the_generated_files_are_out(self):
        # Named because they are the ones the hook skips deliberately: reformatting
        # the vendored copy breaks its digest test.
        names = eligible_files(_ROOT)
        assert names
        assert not [n for n in names if n.startswith("studio/backend/vendor/")]
        assert not [n for n in names if n.endswith("chat_templates.py")]

    def test_the_spacing_pass_is_not_asked_to_rewrite_itself(self):
        # It declines by path identity, so a copy of it under another name would
        # be rewritten and reported as drift the hook will never produce.
        names = eligible_files(_ROOT)
        assert _SELF_SKIPPED.is_file()
        assert not [n for n in names if (_ROOT / n).resolve() == _SELF_SKIPPED]
        # And the rest of scripts/ is still in scope, including the hook entry
        # point, which the spacing pass does rewrite.
        assert "scripts/run_ruff_format.py" in names


class TestTheGuardCannotGoGreenHavingCheckedNothing:
    """A skip is a pass to everything that reads CI, so CI may not be allowed one."""

    def test_a_usable_ruff_runs_the_check(self):
        assert guard_verdict(None, in_ci = False) == "run"
        assert guard_verdict(None, in_ci = True) == "run"

    def test_a_contributor_without_the_pin_is_only_skipped(self):
        assert guard_verdict("ruff is not installed here", in_ci = False) == "skip"

    def test_the_same_gap_in_ci_is_a_failure(self):
        # Whichever reason it is: no ruff means the guard checked nothing, and a
        # mismatched ruff means the workflow drifted off the pin it installs.
        assert guard_verdict("ruff is not installed here", in_ci = True) == "fail"
        assert guard_verdict("the repo is formatted with ruff 0.6.9", in_ci = True) == "fail"

    def test_ci_is_detected_from_either_variable(self):
        assert running_in_ci({"GITHUB_ACTIONS": "true"}) is True
        assert running_in_ci({"CI": "true"}) is True
        assert running_in_ci({}) is False
        # Unset-but-present is how some runners spell "not CI".
        assert running_in_ci({"CI": ""}) is False


def formatter_argvs(copies: list[str], head: list[str] | None = None) -> "list[list[str]]":
    """Every command line the fixed-point guard will run, in order.

    Batches are accumulated until adding the next path would take the command line past
    `_FORMAT_ARGV_BUDGET`, so a single path longer than the budget still gets its own call
    rather than being dropped -- the caller would rather run one over-long command and see the
    OS refuse it than silently skip a file.

    The guard and the Windows-limit test below both go through this, deliberately. A test that
    only checked the budget constant would be measuring a number while the caller did something
    else, and deleting the batching at the call site would leave it green. Here there is one
    definition of what actually gets executed, so the limit test cannot drift away from the run.
    """
    head = head or [sys.executable, str(_ROOT / "scripts" / "run_ruff_format.py")]
    base = _command_line_length(head)
    argvs: list[list[str]] = []
    batch: list[str] = []
    used = base
    for path in copies:
        cost = _command_line_length([path])
        if batch and used + cost > _FORMAT_ARGV_BUDGET:
            argvs.append([*head, *batch])
            batch, used = [], base
        batch.append(path)
        used += cost
    if batch:
        argvs.append([*head, *batch])
    return argvs


def _command_line_length(argv: list[str]) -> int:
    """What Windows counts against its command-line cap for this argv.

    CreateProcess is handed ONE string, so the cost is the arguments joined by the separating
    spaces, plus a pair of quotes around every argument a runner path forces (the hosted image
    checks out under `D:\\a\\unsloth\\unsloth`, no spaces, but `C:\\Users\\RUNNER~1\\AppData\\
    Local\\Temp` is where tmp_path lands and a user name with a space is normal off CI). Counted
    with the quotes always, because this is a headroom check and the cheap direction to be wrong
    in is pessimistic.
    """
    return sum(len(arg) + 3 for arg in argv)


@pytest.mark.skipif(_VERDICT == "skip", reason = _RUFF_REASON or "")
def test_the_formatter_invocation_fits_in_a_windows_command_line():
    """The guard below must be able to START on Windows, not only pass on Linux.

    It used to pass every tracked file as one argv. That is ~2650 paths and, under a Windows
    tmp_path, roughly 325,000 characters against CreateProcess's 32,767 -- 9.9x over, so the
    call died with `[WinError 206] The filename or extension is too long` before ruff opened a
    single file. It had never been caught because every job that schedules this file is
    ubuntu-24.04, where execve's ARG_MAX is ~2 MB and the same argv fits with room to spare.

    So the limit is asserted here rather than left to a Windows runner to discover: this runs
    in the existing Linux job, needs no second platform, and goes red the moment someone
    reverts the batching or raises _FORMAT_ARGV_BUDGET past what the cap allows. Computed from the
    REAL file list and a realistic Windows tmp_path prefix, not from a remembered number, so
    the file set growing is what moves it.
    """
    names = eligible_files(_ROOT)
    assert len(names) > 1000, f"only {len(names)} files matched; the file list has gone vacuous"

    # A hosted Windows runner's pytest tmp_path. Longer than the Linux equivalent, which is the
    # point: the platform with the smallest cap also has the longest prefix.
    prefix = "C:\\Users\\RUNNER~1\\AppData\\Local\\Temp\\pytest-of-runner\\pytest-999\\test_0"
    copies = [prefix + "\\" + name for name in names]

    argvs = formatter_argvs(copies)
    # Every path is still formatted: batching may not silently drop the tail.
    assert [arg for argv in argvs for arg in argv[2:]] == copies, "batching lost or reordered files"

    worst = max(_command_line_length(argv) for argv in argvs)
    assert worst < _WINDOWS_COMMAND_LINE_LIMIT, (
        f"the widest of the {len(argvs)} formatter batches builds a {worst}-character command "
        f"line against Windows' {_WINDOWS_COMMAND_LINE_LIMIT}-character CreateProcess limit, so "
        f"this guard would die with WinError 206 on Windows before formatting anything. Lower "
        f"_FORMAT_ARGV_BUDGET (currently {_FORMAT_ARGV_BUDGET} over {len(names)} tracked files)."
    )
    # Non-vacuous: the batching has to be doing something. One call with everything on it is the
    # shape that was broken, and it must still be over the cap -- if it ever fits, this test is
    # measuring nothing and the batching can go.
    unbatched = _command_line_length([sys.executable, "run_ruff_format.py", *copies])
    assert unbatched > _WINDOWS_COMMAND_LINE_LIMIT, (
        f"the whole tracked set now builds a {unbatched}-character command line, under the "
        f"{_WINDOWS_COMMAND_LINE_LIMIT} cap, so batching is no longer load-bearing and this "
        "test no longer proves anything. Delete both, or say why they stay."
    )


@pytest.mark.skipif(_VERDICT == "skip", reason = _RUFF_REASON or "")
def test_every_tracked_python_file_is_already_formatted(tmp_path):
    """Run the hook over copies of the whole tracked set and expect no rewrite."""
    if _VERDICT == "fail":
        pytest.fail(
            f"this guard cannot run in CI: {_RUFF_REASON}.\n"
            "  The runner installs the pinned ruff in its own step (see the "
            "'Install the pinned ruff (formatter fixed-point guard)' step in "
            ".github/workflows/studio-backend-ci.yml).\n"
            "  Skipping here would report a green formatting guard that checked no files."
        )
    names = eligible_files(_ROOT)
    assert len(names) > 1000, f"only {len(names)} files matched; the file list has gone vacuous"

    # ruff reads line-length and extend-exclude from the root pyproject.toml, and
    # finds it by walking up from each file. Without this copy the run happens at
    # ruff's default 88 columns and every long line looks like drift.
    shutil.copy2(_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")

    originals: dict[str, bytes] = {}
    copies: list[str] = []
    for name in names:
        source = _ROOT / name
        target = tmp_path / name
        target.parent.mkdir(parents = True, exist_ok = True)
        originals[name] = source.read_bytes()
        target.write_bytes(originals[name])
        copies.append(str(target))

    # Batched, and through the same builder the Windows-limit test above measures. One call with
    # all ~2650 paths on it is 9.9x over Windows' CreateProcess cap and dies with WinError 206.
    for argv in formatter_argvs(copies):
        run = subprocess.run(argv, capture_output = True, text = True)
        assert run.returncode == 0, f"the formatter itself failed:\n{run.stdout}\n{run.stderr}"

    drifted = [name for name in names if (tmp_path / name).read_bytes() != originals[name]]
    assert not drifted, (
        "these tracked files are not a fixed point of the ruff-format-with-kwargs hook, "
        "so pre-commit.ci will fail the next PR that edits them:\n"
        + "\n".join(f"  {name}" for name in drifted)
        + "\n  Fix: python scripts/run_ruff_format.py "
        + " ".join(drifted)
    )
