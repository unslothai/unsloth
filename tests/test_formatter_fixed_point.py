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

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
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

# Windows' CreateProcess lpCommandLine cap in characters; the binding constraint across OSes.
_WINDOWS_COMMAND_LINE_LIMIT = 32767

# Length budget per formatter invocation. All tracked paths together are ~10x the Windows cap
# (WinError 206). A length, not a file count, so it self-adjusts; 8000 leaves 4x headroom.
# Batching is safe because the hook formats each path independently.
_FORMAT_ARGV_BUDGET = 8000

_HOOK_ID = "ruff-format-with-kwargs"
# The spacing pass refuses to rewrite itself, so it is not kept at a fixed point.
_SELF_SKIPPED = Path(enforce_kwargs_spacing.__file__).resolve()
# `types: [python]` is what pre-commit filters on, and the repo tracks no .pyi.
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
            # The quoted form keeps its contents verbatim: a regex may contain a `#`.
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
        # The ruff hook above ours carries its own exclude; the first one in the file is the wrong one.
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
        # Reformatting the vendored copy breaks its digest test.
        names = eligible_files(_ROOT)
        assert names
        assert not [n for n in names if n.startswith("studio/backend/vendor/")]
        assert not [n for n in names if n.endswith("chat_templates.py")]

    def test_the_spacing_pass_is_not_asked_to_rewrite_itself(self):
        # It declines by path identity, so a renamed copy would be rewritten.
        names = eligible_files(_ROOT)
        assert _SELF_SKIPPED.is_file()
        assert not [n for n in names if (_ROOT / n).resolve() == _SELF_SKIPPED]
        assert "scripts/run_ruff_format.py" in names


class TestTheGuardCannotGoGreenHavingCheckedNothing:
    """A skip is a pass to everything that reads CI, so CI may not be allowed one."""

    def test_a_usable_ruff_runs_the_check(self):
        assert guard_verdict(None, in_ci = False) == "run"
        assert guard_verdict(None, in_ci = True) == "run"

    def test_a_contributor_without_the_pin_is_only_skipped(self):
        assert guard_verdict("ruff is not installed here", in_ci = False) == "skip"

    def test_the_same_gap_in_ci_is_a_failure(self):
        # No ruff means the guard checked nothing; a mismatched ruff means the workflow drifted.
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

    # A hosted Windows runner's tmp_path: the smallest cap also has the longest prefix.
    prefix = "C:\\Users\\RUNNER~1\\AppData\\Local\\Temp\\pytest-of-runner\\pytest-999\\test_0"
    copies = [prefix + "\\" + name for name in names]

    argvs = formatter_argvs(copies)
    assert [arg for argv in argvs for arg in argv[2:]] == copies, "batching lost or reordered files"

    worst = max(_command_line_length(argv) for argv in argvs)
    assert worst < _WINDOWS_COMMAND_LINE_LIMIT, (
        f"the widest of the {len(argvs)} formatter batches builds a {worst}-character command "
        f"line against Windows' {_WINDOWS_COMMAND_LINE_LIMIT}-character CreateProcess limit, so "
        f"this guard would die with WinError 206 on Windows before formatting anything. Lower "
        f"_FORMAT_ARGV_BUDGET (currently {_FORMAT_ARGV_BUDGET} over {len(names)} tracked files)."
    )
    # Non-vacuous: if the unbatched call ever fits under the cap, batching can go.
    unbatched = _command_line_length([sys.executable, "run_ruff_format.py", *copies])
    assert unbatched > _WINDOWS_COMMAND_LINE_LIMIT, (
        f"the whole tracked set now builds a {unbatched}-character command line, under the "
        f"{_WINDOWS_COMMAND_LINE_LIMIT} cap, so batching is no longer load-bearing and this "
        "test no longer proves anything. Delete both, or say why they stay."
    )


# Each batch spawns ruff and the spacing pass, so stopping one must kill its whole tree.
_HAS_PROC = os.path.isdir("/proc/self")

_OWN_GROUP = (
    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    if os.name == "nt"
    else {"start_new_session": True}
)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` and everything it started, then reap it."""
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            )
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if proc.poll() is None:
        proc.kill()
    proc.wait()


@contextlib.contextmanager
def _timeout_signal_held():
    """Defer SIGALRM (pytest-timeout's signal method) until the block has finished.

    Masking the thread would not do: under xdist the kernel can hand the signal to another
    thread, and CPython still runs the Python handler here at the next bytecode. So the handler
    itself is swapped for one that only records the signal, and the signal is raised again once
    the real handler is back. Windows has no SIGALRM, and pytest-timeout's thread method there
    ends the whole process rather than raising into this one.
    """
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return
    held = []
    previous = signal.signal(signal.SIGALRM, lambda signum, frame: held.append(signum))
    try:
        yield
    finally:
        signal.signal(signal.SIGALRM, previous)
        if held:
            signal.raise_signal(signal.SIGALRM)


def _run_side_by_side(argvs: list[list[str]], log_dir: Path) -> list[tuple[int, str]]:
    """Run `argvs` at most `os.cpu_count()` at a time; (returncode, output) for each, in order.

    Driven from the calling thread rather than a thread pool: pytest-timeout raises in this
    thread, and a pool's shutdown would wait on worker threads blocked in subprocess.run, so a
    stalled formatter would outlive the per-test timeout and hold the job to its own. Here the
    exception lands in the polling loop and the finally kills whatever is still running.
    Output goes to files so a chatty child can never block on a full pipe.
    """
    log_dir.mkdir(parents = True, exist_ok = True)
    limit = max(1, min(len(argvs), os.cpu_count() or 1))
    results: list[tuple[int, str] | None] = [None] * len(argvs)
    running: dict[int, tuple[subprocess.Popen, Path]] = {}
    queued = list(enumerate(argvs))
    try:
        while queued or running:
            while queued and len(running) < limit:
                index, argv = queued.pop(0)
                log = log_dir / f"{index}.log"
                # SIGALRM held across the launch so a timeout cannot land before the process is recorded.
                with _timeout_signal_held(), open(log, "wb") as sink:
                    running[index] = (
                        subprocess.Popen(argv, stdout = sink, stderr = subprocess.STDOUT, **_OWN_GROUP),
                        log,
                    )
            for index, (proc, log) in list(running.items()):
                if proc.poll() is not None:
                    results[index] = (
                        proc.returncode,
                        log.read_text(encoding = "utf-8", errors = "replace"),
                    )
                    del running[index]
            if running:
                time.sleep(0.05)
    finally:
        for proc, _ in running.values():
            _kill_tree(proc)
    return [result for result in results if result is not None]


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

    # ruff reads line-length from the root pyproject.toml, found by walking up from each file.
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

    # Batched to stay under the Windows cap, and run side by side to fit pytest-timeout's 330 s.
    for code, output in _run_side_by_side(formatter_argvs(copies), tmp_path / "formatter-logs"):
        assert code == 0, f"the formatter itself failed:\n{output}"

    drifted = [name for name in names if (tmp_path / name).read_bytes() != originals[name]]
    assert not drifted, (
        "these tracked files are not a fixed point of the ruff-format-with-kwargs hook, "
        "so pre-commit.ci will fail the next PR that edits them:\n"
        + "\n".join(f"  {name}" for name in drifted)
        + "\n  Fix: python scripts/run_ruff_format.py "
        + " ".join(drifted)
    )


def test_a_timeout_in_the_polling_loop_kills_the_running_formatters(tmp_path, monkeypatch):
    """pytest-timeout raises in the thread that polls, so whatever it interrupts must not leave
    a formatter running (the job would then wait on it until its own, much longer, timeout)."""
    started = []
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        started.append(real_popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(subprocess, "Popen", popen)
    pid_file = tmp_path / "grandchild.pid"
    wrapper = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        # Published whole: a reader never sees the file created but not yet written.
        f"open({str(pid_file)!r} + '.tmp', 'w').write(str(child.pid))\n"
        f"import os; os.replace({str(pid_file)!r} + '.tmp', {str(pid_file)!r})\n"
        "child.wait()\n"
    )

    def timeout(_seconds):
        deadline = real_monotonic() + 30
        while _read_pid(pid_file) is None and real_monotonic() < deadline:
            real_sleep(0.05)
        raise RuntimeError("stand-in for pytest-timeout")

    real_sleep, real_monotonic = time.sleep, time.monotonic
    monkeypatch.setattr(time, "sleep", timeout)
    with pytest.raises(RuntimeError, match = "stand-in"):
        _run_side_by_side([[sys.executable, "-c", wrapper]], tmp_path / "logs")
    assert started, "nothing was launched, so this proves nothing"
    assert all(proc.poll() is not None for proc in started), "a formatter outlived the timeout"
    grandchild = _read_pid(pid_file)
    assert grandchild is not None, "the batch never started its child"
    # A grandchild in uninterruptible I/O dies only once scheduled, so wait as long as the launch
    # may take. The verdict is the last probe taken.
    deadline = real_monotonic() + 30
    alive = _alive(grandchild)
    while alive and real_monotonic() < deadline:
        real_sleep(0.05)
        alive = _alive(grandchild)
    assert not alive, "the formatter's own child outlived the timeout"


@pytest.mark.skipif(not _HAS_PROC, reason = "reads Linux /proc")
def test_a_pid_reaped_between_the_two_probes_is_not_alive(monkeypatch):
    """The signal probe can see a zombie that is reaped before /proc/<pid>/stat is opened.

    That window used to read as alive, so the timeout test above failed on a process that had
    already been killed and reaped (Repo tests (CPU, rest) on #12097).
    """
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not Path(f"/proc/{child.pid}").exists(), "the child was not reaped"
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    assert _alive(child.pid) is False


@pytest.mark.skipif(os.name == "nt", reason = "POSIX signal probe")
def test_without_proc_a_pid_the_signal_probe_finds_is_alive(monkeypatch):
    """Where /proc is not mounted (macOS), a missing /proc entry proves nothing.

    Reading it as death would pass the timeout test above even if the group kill missed.
    """
    # A pid with no /proc entry, as on macOS, that the signal probe still finds.
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    monkeypatch.setattr(sys.modules[__name__], "_HAS_PROC", False)
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    assert _alive(child.pid) is True


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output = True, text = True
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if not _HAS_PROC:
        return True
    # A killed orphan is a zombie until reaped; a pid with no /proc entry left is gone.
    try:
        with open(f"/proc/{pid}/stat", encoding = "utf-8") as stat:
            return stat.read().split(") ", 1)[1][0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError:
        return True


def test_side_by_side_results_come_back_in_order(tmp_path):
    argvs = [[sys.executable, "-c", f"import sys; print({i}); sys.exit({i % 2})"] for i in range(5)]
    results = _run_side_by_side(argvs, tmp_path)
    assert [code for code, _ in results] == [0, 1, 0, 1, 0]
    assert [output.strip() for _, output in results] == ["0", "1", "2", "3", "4"]


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason = "pytest-timeout raises no signal here")
def test_a_timeout_during_launch_still_kills_the_new_formatter(tmp_path, monkeypatch):
    """The window between Popen returning and the process being recorded: a timeout landing
    there must still leave the finally something to kill."""
    started = []
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        started.append(real_popen(*args, **kwargs))
        signal.raise_signal(signal.SIGALRM)  # arrives the moment Popen has returned
        return started[-1]

    def timeout(_signum, _frame):
        raise RuntimeError("stand-in for pytest-timeout")

    previous = signal.signal(signal.SIGALRM, timeout)
    monkeypatch.setattr(subprocess, "Popen", popen)
    try:
        with pytest.raises(RuntimeError, match = "stand-in"):
            _run_side_by_side(
                [[sys.executable, "-c", "import time; time.sleep(120)"]], tmp_path / "logs"
            )
    finally:
        signal.signal(signal.SIGALRM, previous)
    assert started, "nothing was launched, so this proves nothing"
    assert all(
        proc.poll() is not None for proc in started
    ), "the new formatter outlived the timeout"
