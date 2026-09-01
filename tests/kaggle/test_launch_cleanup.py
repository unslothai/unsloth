# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A killed launcher must not leave a Kaggle kernel billing.

`release()` deletes every kernel the process pushed, and it is the budget
control rather than a tidy-up: a kernel left behind bills to its own ceiling
with nobody reading the result, and that ceiling has been observed not to stop
a wedged one. But release() used to be reachable only from `finish()`, which
only runs on paths that RETURN. Ctrl-C re-raised past it, and SIGTERM -- what
`kill` and a cancelled GitHub Actions workflow send -- terminated the process
without running anything.

These tests drive real subprocesses and real signals, because the property is
about process death and nothing weaker would show it.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"
sys.path.insert(0, str(CI_DIR))

import launch  # noqa: E402


class _StubKaggleApi:
    """A client that can say WHICH account it is, because the real one can.

    `launch.py` reads the owner off the authenticated client and refuses to push
    when it cannot: a kernel id is `<owner>/<slug>`, CI holds more than one
    account, and a kernel pushed under the wrong name cannot be deleted under
    the other. A bare `object()` models a client that never authenticated, which
    is a different test from the ones below.
    """

    CONFIG_NAME_USER = "username"

    def __init__(self, username = "me"):
        self.config_values = {self.CONFIG_NAME_USER: username}


def _stub_api(*_args, **_kwargs):
    return _StubKaggleApi()


def _fake_kaggle(bin_dir: Path, record: Path) -> None:
    """A `kaggle` on PATH that only records what it was asked to delete."""
    bin_dir.mkdir(parents = True, exist_ok = True)
    shim = bin_dir / "kaggle"
    shim.write_text(
        textwrap.dedent(f"""\
        #!{sys.executable}
        import sys, pathlib
        pathlib.Path({str(record)!r}).open("a").write(" ".join(sys.argv[1:]) + "\\n")
        """),
        encoding = "utf-8",
    )
    shim.chmod(0o755)


# Arm faulthandler against a FILE, not stderr.
# PYTHONFAULTHANDLER sends the dump to fd 2, and every launcher here has fd 2 redirected onto the stdout pipe.
# One of these tests deliberately fills that pipe and stops draining it, which is exactly the hang worth diagnosing, and
# a raw write bypasses Python's io lock but not pipe backpressure: SIGABRT would kill the child with nothing written and
# the failure message would be as empty as before.
_FAULT_PREAMBLE = """\
import faulthandler as _faulthandler, os as _os
_fault_dump = open(_os.environ["LAUNCH_FAULT_DUMP"], "w", buffering = 1)
_faulthandler.enable(file = _fault_dump)
"""


def _runner(tmp_path: Path, body: str) -> subprocess.Popen:
    """Run `body` against the real launch module, with a fake kaggle CLI."""
    record = tmp_path / "kaggle_calls.txt"
    _fake_kaggle(tmp_path / "bin", record)
    script = tmp_path / "runner.py"
    script.write_text(
        _FAULT_PREAMBLE
        + textwrap.dedent(f"""\
        import sys, time
        sys.path.insert(0, {str(CI_DIR)!r})
        import launch
        launch.INFLIGHT = __import__("pathlib").Path({str(tmp_path / "inflight.json")!r})
        {body}
        """),
        encoding = "utf-8",
    )
    env = _child_env(tmp_path / "bin")
    return subprocess.Popen(
        [sys.executable, str(script)],
        env = env,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
    )


# A guard against a launcher that never dies, not a latency target: four of these subprocess tests now run at once on a
# four-core runner, where the SIGINT case was observed to need over 30 seconds purely to be scheduled.
_DEATH_BUDGET_SEC = 120

# How long the launcher stalls while a test signals it.
# Must OUTLAST the budget above: a handler that swallows its signal leaves the process asleep and then resuming, so a
# shorter stall lets it wake, run finish(), delete through the ordinary path and exit inside the wait, passing every
_STALL_SEC = 900


def _await_ready(proc: subprocess.Popen) -> None:
    """Wait for the runner to say it is where the test wants it, past whatever
    the launcher logged on the way there."""
    for _ in range(50):
        if proc.stdout.readline().strip() == "READY":
            return
    raise AssertionError("the runner never reached its READY point")


def _fault_dump(tmp_path: Path) -> Path:
    """Where the child writes its stacks. One per test, beside its other artefacts."""
    return tmp_path / "fault.txt"


def _child_env(bin_dir: Path) -> dict:
    """The launcher's environment, pointing faulthandler at a file.

    `_wait_for_death` sends SIGABRT before it kills a launcher that overstayed its
    budget, and faulthandler turns that into every thread's stack. The destination is a
    FILE rather than stderr: fd 2 is redirected onto the stdout pipe here, one of these
    tests deliberately fills that pipe and stops draining it, and a raw write bypasses
    Python's io lock but not pipe backpressure. Dumping there would block, the child
    would die with nothing written, and the failure message would be as empty as the one
    this exists to replace. A file has no reader to block on.
    """
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "LAUNCH_FAULT_DUMP": str(_fault_dump(bin_dir.parent)),
    }


def _wait_for_death(proc: subprocess.Popen, tmp_path: Path | None = None) -> None:
    """Wait out the budget, and say where the launcher was if it never dies.

    A hang and a wrong exit status are different faults, and the bare TimeoutExpired
    named neither. Killing first is what lets the pipe reach EOF so the tail reads.

    SIGABRT before SIGKILL, because the tail on its own has already proved too coarse:
    CI produced "(nothing logged after READY)" on this file, which is the same output
    whether the handler never ran, ran and was refused the pipe, or ran and blocked
    inside a delete. With faulthandler armed in the child, SIGABRT writes the stack it
    is actually stuck on to a file, and then ends the process, so the answer costs
    nothing extra when the hang does not happen.

    The file matters rather than being an implementation detail: the pipe is one of the
    things that can be the hang, so a diagnostic that travels down it can be silenced by
    the very fault it is describing.
    """
    try:
        proc.wait(timeout = _DEATH_BUDGET_SEC)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            proc.send_signal(signal.SIGABRT)
            proc.wait(timeout = 10)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout = 30)
        raise AssertionError(
            f"the launcher was still alive {_DEATH_BUDGET_SEC}s after its signal. "
            f"Launcher said: {_tail(proc)}\n"
            f"Stack at the abort: {_stacks(tmp_path)}"
        ) from None


def _tail(proc: subprocess.Popen) -> str:
    """Whatever the launcher logged after READY, for a failure message.

    `_install_release_handlers` logs "received signal N" the moment its handler runs,
    which separates "the signal never reached the handler" from "the handler ran and
    the process still exited 0". CI has produced the second symptom on a runner where
    it does not reproduce locally, and this output was being discarded.

    Read only after the process has exited, so it cannot block: the pipe is at EOF.
    """
    try:
        return (proc.stdout.read() or "").strip() or "(nothing logged after READY)"
    except Exception as exc:  # noqa: BLE001 -- a diagnostic must not mask the failure
        return f"(could not read the launcher's output: {type(exc).__name__}: {exc})"


def _stacks(tmp_path: Path | None) -> str:
    """The child's own stacks, written by faulthandler on the SIGABRT above."""
    if tmp_path is None:
        return "(not requested: this call site passed no tmp_path)"
    dump = _fault_dump(tmp_path)
    if not dump.is_file():
        return "(no dump: the child died before faulthandler could write, or never armed it)"
    return dump.read_text(encoding = "utf-8", errors = "replace").strip() or "(dump empty)"


def _deletions(tmp_path: Path) -> list[str]:
    record = tmp_path / "kaggle_calls.txt"
    if not record.is_file():
        return []
    return [l for l in record.read_text().splitlines() if l.startswith("kernels delete")]


def _push_ok(slug: str):
    """A push that files `slug` the way the real one does: into the caller's
    own list and into the registry, before it returns anything."""

    def _impl(
        notebook,
        user,
        kernel_timeout_sec,
        accelerator = "NvidiaTeslaT4",
        attempted = None,
    ):
        attempted = [] if attempted is None else attempted
        attempted.append(slug)
        launch._inflight_add(slug)
        return {"ok": True, "slug": slug, "attempts": attempted}

    return _impl


def _run_main(
    tmp_path: Path,
    monkeypatch,
    push_impl = None,
    kaggle = _fake_kaggle,
    argv_extra: tuple[str, ...] = (),
    notebooks: tuple[str, ...] = ("a.ipynb",),
) -> dict:
    """Drive the real main() with the network stubbed out, and hand back the
    launch_result.json it wrote.

    Cleanup lives INSIDE main() now -- `release()` is a closure over `result`
    and `args` -- so this is how a test reaches it. Every path out of main()
    goes through finish() -> release(), and the result file is release()'s own
    record of what it concluded about each slug, so asserting on the file
    asserts on the real thing rather than on a re-implementation of it.

    Only the network and the clock are replaced. The push loop, the entry
    bookkeeping, `_slugs_filed`, `delete_kernel` and the registry are the
    production ones, and `kaggle` on PATH is what decides whether a delete is
    confirmed.
    """
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    kaggle(tmp_path / "bin", tmp_path / "kaggle_calls.txt")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    # The retry backoffs, not the retries:
    monkeypatch.setattr(launch, "PUSH_BACKOFF_SEC", 0)
    monkeypatch.setattr(launch, "DELETE_BACKOFF_SEC", 0)
    monkeypatch.setattr(launch, "_api", _stub_api)
    monkeypatch.setattr(launch, "wait", lambda *a, **kw: "COMPLETE")
    monkeypatch.setattr(
        launch,
        "fetch_evidence",
        lambda *a, **kw: {"notebooks": [], "log": None, "truncated": False},
    )
    # Installing the real handlers here would leave a SIGTERM disposition and an atexit callback on the pytest
    monkeypatch.setattr(launch, "_install_release_handlers", lambda release: None)
    if push_impl is not None:
        monkeypatch.setattr(launch, "push", push_impl)
    outdir = tmp_path / "out"
    argv = ["launch.py", "--user", "me", "--outdir", str(outdir), *argv_extra]
    for notebook in notebooks:
        argv += ["--notebook", notebook]
    monkeypatch.setattr(sys, "argv", argv)
    launch.main()
    return json.loads((outdir / "launch_result.json").read_text(encoding = "utf-8"))


# --------------------------------------------------------------------------
def test_a_pushed_kernel_is_recorded_before_anything_else_can_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    launch._inflight_add("me/k-1")
    entries = launch._inflight_read()
    assert [e["slug"] for e in entries] == ["me/k-1"]
    assert entries[0]["pid"] == os.getpid()


def test_deleting_a_kernel_takes_it_out_of_the_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    launch._inflight_add("me/k-1")
    launch._inflight_drop("me/k-1")
    assert launch._inflight_read() == []


def test_a_live_owner_is_never_swept(tmp_path, monkeypatch):
    """Two launchers run concurrently. Deleting the other one's kernel would
    destroy a legitimate run and report its absence as a code failure."""
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    launch._inflight_write([{"slug": "me/live", "pid": os.getpid() + 0, "at": 0}])
    assert launch.sweep_orphans() == []
    assert [e["slug"] for e in launch._inflight_read()] == ["me/live"]


def test_a_dead_owners_kernel_is_reclaimed(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    _fake_kaggle(tmp_path / "bin", tmp_path / "kaggle_calls.txt")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    launch._inflight_write([{"slug": "me/orphan", "pid": dead.pid, "at": 0}])
    assert launch.sweep_orphans() == ["me/orphan"]
    assert launch._inflight_read() == []
    assert any("me/orphan" in c for c in _deletions(tmp_path))


def test_a_failed_delete_keeps_the_entry_for_next_time(tmp_path, monkeypatch):
    """Forgetting a kernel we could not delete is how one bills forever."""
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    # A pid that cannot be running: claim one and let it exit.
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    launch._inflight_write([{"slug": "me/orphan", "pid": dead.pid, "at": 0}])

    def _boom(*a, **kw):
        raise OSError("kaggle is unreachable")

    monkeypatch.setattr(launch.subprocess, "run", _boom)
    assert launch.sweep_orphans() == []
    assert [e["slug"] for e in launch._inflight_read()] == ["me/orphan"]


def _stalling_push(monkeypatch, tmp_path) -> Path:
    """Every `kaggle kernels push` runs out of wall clock and is killed."""
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    monkeypatch.setattr(launch, "PUSH_BACKOFF_SEC", 0)
    monkeypatch.setattr(launch, "DELETE_BACKOFF_SEC", 0)
    notebook = tmp_path / "kernel.ipynb"
    notebook.write_text("{}", encoding = "utf-8")

    def _stall(*a, **kw):
        raise subprocess.TimeoutExpired(cmd = ["kaggle"], timeout = 600)

    monkeypatch.setattr(launch.subprocess, "run", _stall)
    return notebook


def test_a_stalled_push_is_reported_as_infra_not_raised(tmp_path, monkeypatch):
    """Every other Kaggle transport failure returns a reason and exits 0.

    A `kaggle kernels push` that stalls past the 600s ceiling raises
    TimeoutExpired, and letting it escape ends the process before
    `finish()` writes launch_result.json: the launch step goes red and the
    reporter never gets to call the run NOT RUN. Red is reserved for a payload
    that ran on a T4 and failed an assertion.

    The stall is a retryable attempt like any other transport failure -- the
    exhausted retries are what the caller finally hears about -- so what this
    asserts is that push() RETURNS a reason, and that the reason still carries
    the stall rather than describing it as some other refusal.
    """
    notebook = _stalling_push(monkeypatch, tmp_path)
    pushed = launch.push(notebook, "me", 3600)
    assert pushed["ok"] is False
    assert pushed["reason"] == "push_failed"
    assert "timed out" in pushed["detail"], pushed["detail"]


def test_a_timed_out_push_does_not_forget_the_kernel_it_may_have_created(tmp_path, monkeypatch):
    """A stalled CLI says nothing about whether Kaggle took the push.

    Kaggle can accept it and start billing before the response is lost, and
    the slug is ours and already decided at that point. Forgetting it left a
    running kernel that `finish()` could not delete and that no later orphan
    sweep could see either, so it billed to its 70 minute ceiling with nobody
    reading the result.

    Two records have to hold it, because they cover different deaths: the
    caller's `attempted` list, which release() reconciles on the way out, and
    the on-disk registry, which is all that survives a kill.
    """
    notebook = _stalling_push(monkeypatch, tmp_path)
    owned: list[str] = []
    pushed = launch.push(notebook, "me", 3600, attempted = owned)

    # Not "slug": the caller must not WAIT on a kernel that may not exist.
    assert pushed.get("slug") is None
    assert pushed["attempts"] == owned
    assert len(owned) == launch.PUSH_ATTEMPTS
    assert all(s.startswith("me/unsloth-t4-ci-") for s in owned), owned
    assert sorted(e["slug"] for e in launch._inflight_read()) == sorted(owned)


def test_a_push_that_raises_still_leaves_its_slug_with_the_caller(tmp_path, monkeypatch):
    """The slug is decided BEFORE the call that may have created the kernel.

    A push reaches the network, and not everything it can raise is foreseen: a
    malformed response decodes with strict error handling and raises
    UnicodeDecodeError, the runner can answer OSError. Anything that unwinds
    past the return takes the slug with it unless the caller already owns the
    list, and a slug Kaggle may have just accepted and nothing can name is the
    same leak the timeout used to be.
    """
    notebook = _stalling_push(monkeypatch, tmp_path)

    def _explode(*a, **kw):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(launch.subprocess, "run", _explode)
    owned: list[str] = []
    with pytest.raises(UnicodeDecodeError):
        launch.push(notebook, "me", 3600, attempted = owned)
    assert len(owned) == 1 and owned[0].startswith("me/unsloth-t4-ci-")


def test_a_kernel_only_a_timeout_knows_about_is_still_deleted(tmp_path, monkeypatch):
    """Cleanup is the budget control, so it has to act on the slugs a stalled
    push left behind and not only on a confirmed one.

    End to end through the real push loop: every attempt stalls, so nothing is
    ever confirmed, and the only names for the sessions Kaggle may have started
    are the ones push() filed. All of them have to be deleted and all of them
    have to leave the registry.
    """
    real_run = subprocess.run
    notebook = tmp_path / "kernel.ipynb"
    notebook.write_text("{}", encoding = "utf-8")

    def _stall_pushes_only(cmd, *a, **kw):
        if "push" in cmd:
            raise subprocess.TimeoutExpired(cmd = cmd, timeout = launch.PUSH_SUBPROCESS_TIMEOUT_SEC)
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(launch.subprocess, "run", _stall_pushes_only)
    result = _run_main(tmp_path, monkeypatch, notebooks = (str(notebook),))

    entry = result["kernels"][0]
    filed = entry["attempted"]
    assert len(filed) == launch.PUSH_ATTEMPTS
    # Never waited on: no attempt was ever confirmed.
    assert entry["slug"] is None
    assert result["verdict"] == "infra"
    deleted = _deletions(tmp_path)
    for slug in filed:
        assert any(slug in c for c in deleted), f"{slug} was never deleted; it may keep billing"
    assert entry["released"] is True
    assert result["unreleased"] == []
    assert launch._inflight_read() == []


def test_a_corrupt_registry_does_not_take_the_run_down(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    (tmp_path / "inflight.json").write_text("{not json", encoding = "utf-8")
    assert launch._inflight_read() == []
    assert launch.sweep_orphans() == []


# --------------------------------------------------------------------------
def _waiting_launcher(outdir: Path) -> str:
    return "\n".join(
        [
            "",
            "        class _Api:",
            "            CONFIG_NAME_USER = 'username'",
            "            config_values = {'username': 'me'}",
            "        launch._api = lambda *a, **k: _Api()",
            "        launch.sweep_orphans = lambda *a, **k: []",
            "        def _push(notebook, user, kernel_timeout_sec,",
            "                  accelerator='NvidiaTeslaT4', attempted=None):",
            "            attempted = [] if attempted is None else attempted",
            "            attempted.append('me/k-1')",
            "            launch._inflight_add('me/k-1')",
            "            return {'ok': True, 'slug': 'me/k-1', 'attempts': attempted}",
            "        launch.push = _push",
            "        def _wait(*a, **kw):",
            "            print('READY', flush=True)",
            "            time.sleep(%d)" % _STALL_SEC,
            "        launch.wait = _wait",
            "        sys.argv = ['launch.py', '--notebook', 'a.ipynb', '--user', 'me',",
            f"                    '--outdir', {str(outdir)!r}]",
            "        launch.main()",
        ]
    )


def _flooding_launcher(outdir: Path) -> str:
    """A launcher parked inside a write to a stdout nobody is draining.

    Two things differ from `_waiting_launcher`. It writes until the pipe is full instead
    of sleeping, so when the signal lands the main thread is asleep in the kernel holding
    the buffer lock rather than merely idle. And its delete logs on the way through, so
    the handler meets a blocking `_log` it did not call itself: `delete_kernel` reports a
    refused delete through the ordinary path, and a stall there is a stall before the
    retry and before the `finally` that re-raises the signal. A fake deletion that
    succeeds silently never reaches that.
    """
    return (
        _waiting_launcher(outdir)
        .replace(
            "            time.sleep(%d)" % _STALL_SEC,
            "\n".join(
                [
                    "            while True:",
                    "                sys.stdout.write('x' * 65536)",
                    "                sys.stdout.flush()",
                ]
            ),
        )
        .replace(
            "        launch.main()",
            "\n".join(
                [
                    "        _real_delete = launch.delete_kernel",
                    "        def _noisy_delete(slug):",
                    "            launch._log(f'deleting {slug}')",
                    "            return _real_delete(slug)",
                    "        launch.delete_kernel = _noisy_delete",
                    "        launch.main()",
                ]
            ),
        )
    )


@pytest.mark.parametrize("signame", ["SIGINT", "SIGTERM"])
def test_a_signalled_launcher_deletes_its_kernels(tmp_path, signame):
    """SIGTERM is what `kill` and an Actions cancel send; SIGINT is Ctrl-C.
    Before the handlers, neither deleted anything.
    """
    proc = _runner(tmp_path, _waiting_launcher(tmp_path / "out"))
    try:
        _await_ready(proc)
        proc.send_signal(getattr(signal, signame))
        _wait_for_death(proc, tmp_path)
    finally:
        if proc.poll() is None:
            proc.kill()
    logged = _tail(proc)
    assert any(
        "me/k-1" in c for c in _deletions(tmp_path)
    ), f"{signame} left the kernel behind; it would bill to its ceiling. Launcher said: {logged}"
    assert json.loads((tmp_path / "inflight.json").read_text()) == []
    # On the signal path, not on the way out of an ordinary run:
    assert proc.returncode == -getattr(signal, signame), (
        f"the kernel was deleted, but the launcher exited {proc.returncode} rather than "
        f"dying of {signame}, so nothing here says the signal is what did it. "
        f"Launcher said: {logged}"
    )


def test_the_exit_status_still_says_it_was_killed(tmp_path):
    """A handler that swallows the signal and exits 0 makes a cancelled job
    look like a completed one."""
    proc = _runner(tmp_path, _waiting_launcher(tmp_path / "out"))
    try:
        _await_ready(proc)
        proc.send_signal(signal.SIGTERM)
        _wait_for_death(proc, tmp_path)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == -signal.SIGTERM, (
        f"expected death by SIGTERM, got returncode {proc.returncode}. "
        f"Launcher said: {_tail(proc)}"
    )


def test_the_exit_status_survives_a_release_that_fails(tmp_path):
    """The way the status above was actually observed to come back 0.

    A delete raising inside the handler propagated into the main thread, where
    main() catches BaseException and reaches release() by RETURNING. A first
    delete that failed and a second that worked -- a transient OSError from a
    subprocess spawn on a loaded runner -- therefore turned a cancelled run into
    `exit 0`. The retry still deletes the kernel; what this pins is that the exit
    status does not depend on the delete having worked.
    """
    proc = _runner(
        tmp_path,
        _waiting_launcher(tmp_path / "out").replace(
            "        launch.main()",
            "\n".join(
                [
                    "        _calls = []",
                    "        _real_delete = launch.delete_kernel",
                    "        def _flaky_delete(slug):",
                    "            _calls.append(slug)",
                    "            if len(_calls) == 1:",
                    "                raise OSError(11, 'Resource temporarily unavailable')",
                    "            return _real_delete(slug)",
                    "        launch.delete_kernel = _flaky_delete",
                    "        launch.main()",
                ]
            ),
        ),
    )
    try:
        _await_ready(proc)
        proc.send_signal(signal.SIGTERM)
        _wait_for_death(proc, tmp_path)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == -signal.SIGTERM, (
        f"a release() that raised turned SIGTERM into returncode {proc.returncode}; "
        f"a cancelled job would read as a completed one. Launcher said: {_tail(proc)}"
    )
    assert any("me/k-1" in c for c in _deletions(tmp_path))


def test_the_stall_outlasts_the_death_budget():
    """The relationship the signal tests rest on, asserted rather than assumed.

    If the sleep is the shorter of the two, a launcher that ignored its signal
    wakes up, finishes normally and exits inside the wait, so the tests pass on
    the behaviour they forbid. Tuning one number without the other is silent.
    """
    assert _STALL_SEC > _DEATH_BUDGET_SEC, (
        f"a launcher that swallows its signal wakes after {_STALL_SEC}s and exits "
        f"normally inside the {_DEATH_BUDGET_SEC}s wait, so the signal tests would "
        f"pass without any signal handling at all"
    )


def test_the_handler_survives_its_own_logging_failing(tmp_path):
    """The reentrancy that a contended runner produced, made deterministic.

    A signal handler runs on the main thread wherever it was, and if that was inside a
    write to stdout the interpreter refuses the second one: ``RuntimeError: reentrant
    call inside <_io.BufferedWriter name='<stdout>'>``. Captured on a staging runner,
    where it escaped the handler before it could re-raise the signal and the launcher
    exited 1. Nothing about the timing is reproduced here; the failure it causes is,
    by making the handler's own log call raise.
    """
    proc = _runner(
        tmp_path,
        _waiting_launcher(tmp_path / "out").replace(
            "        launch.main()",
            "\n".join(
                [
                    "        _real_emit = launch._emit",
                    "        def _reentrant(line):",
                    "            if 'received signal' in line:",
                    "                raise RuntimeError(",
                    '                    "reentrant call inside <_io.BufferedWriter "',
                    "                    \"name='<stdout>'>\")",
                    "            _real_emit(line)",
                    "        launch._emit = _reentrant",
                    "        launch.main()",
                ]
            ),
        ),
    )
    try:
        _await_ready(proc)
        proc.send_signal(signal.SIGTERM)
        _wait_for_death(proc, tmp_path)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == -signal.SIGTERM, (
        f"a log call that raised inside the handler turned SIGTERM into returncode "
        f"{proc.returncode}. Launcher said: {_tail(proc)}"
    )
    # And the retry still did the budget control the handler exists for.
    # And the kernel is still deleted:
    assert any("me/k-1" in c for c in _deletions(tmp_path))


def test_the_handler_survives_a_stdout_nobody_is_draining(tmp_path):
    """Raising is not the only way a log line stops the handler; blocking is the other.

    stdout in CI is a pipe. If whatever collects it stops reading, the pipe fills and a
    write goes to sleep in the kernel instead of failing, so no ``except`` and no
    ``finally`` runs. The handler announces itself BEFORE calling ``release()``, so the
    kernels would keep billing until something killed the launcher from outside, which is
    the one outcome this file exists to prevent. The same backpressure is what makes the
    reentrancy above likely, so the two arrive together.

    Reproduced exactly: the launcher writes until the pipe is full and this test never
    reads a byte of it, so the main thread is asleep inside a write, holding the buffer
    lock, when the signal lands. Both the buffered path and the raw descriptor are then
    unavailable, and the line has to be dropped rather than waited on.
    """
    proc = _runner(tmp_path, _flooding_launcher(tmp_path / "out"))
    try:
        _await_ready(proc)
        time.sleep(2)
        proc.send_signal(signal.SIGTERM)
        _wait_for_death(proc, tmp_path)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert any("me/k-1" in c for c in _deletions(tmp_path)), (
        "a full stdout pipe stopped the handler before release(), so the kernel stayed up "
        "and billed. The diagnostic must be dropped rather than blocked on."
    )
    assert json.loads((tmp_path / "inflight.json").read_text()) == []
    assert proc.returncode == -signal.SIGTERM, (
        f"the launcher exited {proc.returncode} rather than dying of SIGTERM after its "
        f"logging blocked"
    )


def test_a_reentrant_log_inside_the_delete_retries_does_not_abandon_them(tmp_path):
    """The transitive path, where a dropped line costs a whole retry budget.

    ``delete_kernel`` reports a refused delete through the ordinary ``_log``. If that
    raises the reentrancy above, the exception propagates out of ``delete_kernel`` and
    out of ``release()``, so a kernel Kaggle would have accepted on the third attempt is
    never asked a third time. Distinct from the handler's own lines, which were already
    wrapped, and from a full pipe, which drops rather than raises.

    Here the shim refuses twice and accepts on the third, and every ``_emit`` raises
    while the handler is running, so the retries only complete if the failure is
    swallowed where it happens.
    """
    record = tmp_path / "kaggle_calls.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    shim = bin_dir / "kaggle"
    shim.write_text(
        textwrap.dedent(f"""\
        #!{sys.executable}
        import sys, pathlib
        record = pathlib.Path({str(record)!r})
        record.open("a").write(" ".join(sys.argv[1:]) + "\\n")
        seen = sum(1 for l in record.read_text().splitlines() if l.startswith("kernels delete"))
        if seen < 3:            # refuse the first two, accept the third
            sys.stderr.write("500 Server Error\\n")
            sys.exit(1)
        """),
        encoding = "utf-8",
    )
    shim.chmod(0o755)

    script = tmp_path / "runner.py"
    script.write_text(
        _FAULT_PREAMBLE
        + textwrap.dedent(f"""\
        import sys, time
        sys.path.insert(0, {str(CI_DIR)!r})
        import launch
        launch.INFLIGHT = __import__("pathlib").Path({str(tmp_path / "inflight.json")!r})
        launch.DELETE_BACKOFF_SEC = 0
        _real_emit = launch._emit
        def _reentrant(line):
            if launch._IN_SIGNAL_HANDLER:
                raise RuntimeError(
                    "reentrant call inside <_io.BufferedWriter name='<stdout>'>")
            _real_emit(line)
        launch._emit = _reentrant
        def release():
            if launch.delete_kernel("me/k-1"):
                launch._inflight_drop("me/k-1")
        launch._inflight_add("me/k-1")
        launch._install_release_handlers(release)
        print("READY", flush=True)
        time.sleep({_STALL_SEC})
    """),
        encoding = "utf-8",
    )
    env = _child_env(bin_dir)
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        env = env,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
    )
    try:
        _await_ready(proc)
        proc.send_signal(signal.SIGTERM)
        _wait_for_death(proc, tmp_path)
    finally:
        if proc.poll() is None:
            proc.kill()

    attempts = _deletions(tmp_path)
    assert len(attempts) >= 3, (
        f"only {len(attempts)} delete attempts were made. A log line raising inside "
        f"delete_kernel abandoned the retries, so a kernel Kaggle would have released "
        f"on the third attempt keeps billing. Launcher said: {_tail(proc)}"
    )
    assert json.loads((tmp_path / "inflight.json").read_text()) == []


def test_the_leaked_kernel_warning_does_not_strand_the_handler(tmp_path):
    """The branch that only opens when cleanup has already failed.

    ``release()`` ends by warning about kernels it could not delete. That line is
    emitted on exactly the path where a kernel is still billing, so a raw write there
    blocks on a full pipe before the ``finally`` can re-raise the signal: the launcher
    neither dies nor reports, and the kernel runs on. Every other test in this file has
    a deletion that eventually succeeds, which leaves the warning unreached.
    """
    record = tmp_path / "kaggle_calls.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    shim = bin_dir / "kaggle"
    shim.write_text(
        textwrap.dedent(f"""\
        #!{sys.executable}
        import sys, pathlib
        pathlib.Path({str(record)!r}).open("a").write(" ".join(sys.argv[1:]) + "\\n")
        sys.stderr.write("500 Server Error\\n")
        sys.exit(1)
        """),
        encoding = "utf-8",
    )
    shim.chmod(0o755)

    body = _flooding_launcher(tmp_path / "out").replace(
        "        launch.main()", "        launch.DELETE_BACKOFF_SEC = 0\n        launch.main()"
    )
    script = tmp_path / "runner.py"
    script.write_text(
        _FAULT_PREAMBLE
        + textwrap.dedent(f"""\
        import sys, time
        sys.path.insert(0, {str(CI_DIR)!r})
        import launch
        launch.INFLIGHT = __import__("pathlib").Path({str(tmp_path / "inflight.json")!r})
        {body}
    """),
        encoding = "utf-8",
    )
    env = _child_env(bin_dir)
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        env = env,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
    )
    try:
        _await_ready(proc)
        time.sleep(2)
        proc.send_signal(signal.SIGTERM)
        _wait_for_death(proc, tmp_path)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == -signal.SIGTERM, (
        f"the launcher exited {proc.returncode} rather than dying of SIGTERM. Its "
        f"warning about the kernel it could not delete is the last thing it writes, "
        f"and on a full pipe that write is where it stopped."
    )
    # It really did reach the branch, so the assertion above is not vacuous.
    assert _deletions(tmp_path), "no delete was attempted, so nothing could leak"


def _failing_kaggle(bin_dir: Path, record: Path) -> None:
    """A `kaggle` that records the call and then refuses it, so nothing is ever
    released and release() reaches its leaked-kernel warning."""
    bin_dir.mkdir(parents = True, exist_ok = True)
    shim = bin_dir / "kaggle"
    shim.write_text(
        textwrap.dedent(f"""\
        #!{sys.executable}
        import sys, pathlib
        pathlib.Path({str(record)!r}).open("a").write(" ".join(sys.argv[1:]) + "\\n")
        sys.stderr.write("500 Server Error\\n")
        sys.exit(1)
        """),
        encoding = "utf-8",
    )
    shim.chmod(0o755)


def test_the_leaked_warning_is_still_a_github_annotation(tmp_path, monkeypatch, capsys):
    """Routing it through the safe writer must not add the [launch] prefix.

    GitHub matches an annotation from the START of the line, so a prefixed one is an
    ordinary log line that nothing surfaces, and this warning exists precisely to be
    surfaced: it is the only thing that tells a human a kernel is still billing.

    Driven through main() with a kaggle that always refuses, so this reads what
    release() actually wrote. Calling the writer directly would pass whatever the call
    site does, which is the thing that can regress.
    """
    _run_main(tmp_path, monkeypatch, push_impl = _push_ok("me/k-1"), kaggle = _failing_kaggle)
    lines = capsys.readouterr().out.splitlines()
    warnings = [l for l in lines if "Kaggle kernels may still be running" in l]
    assert warnings, (
        f"release() never warned about the kernel it could not delete. Output was: " f"{lines[-8:]}"
    )
    assert warnings[0].startswith("::warning"), (
        f"the annotation was written as {warnings[0]!r}. GitHub matches ::warning at the "
        f"start of the line, so a prefix silently demotes the one message that says a "
        f"kernel is still billing into an ordinary log line nobody sees."
    )


def test_an_unhandled_exception_still_deletes(tmp_path):
    """atexit covers the path no signal handler sees.

    Driven through `_install_release_handlers` directly rather than through
    main(), because main() catches BaseException and reaches release() by
    RETURNING: the interpreter shutdown this registration exists for is the
    one nothing inside main() gets to see. The callable is the real
    delete_kernel plus the real registry drop, so an atexit hook that runs
    but does not finish still fails.
    """
    proc = _runner(
        tmp_path,
        """
        def release():
            if launch.delete_kernel("me/k-1"):
                launch._inflight_drop("me/k-1")
        launch._inflight_add("me/k-1")
        launch._install_release_handlers(release)
        raise RuntimeError("boom")
    """,
    )
    _wait_for_death(proc, tmp_path)
    assert any("me/k-1" in c for c in _deletions(tmp_path))
    assert json.loads((tmp_path / "inflight.json").read_text()) == []


def test_kill_9_leaves_it_for_the_sweep(tmp_path):
    """Nothing in-process survives SIGKILL. What must survive is the record,
    so the next launcher can reclaim it."""
    inflight = tmp_path / "inflight.json"
    proc = _runner(
        tmp_path,
        f"""
        launch._inflight_add("me/k-9")
        print("READY", flush=True)
        time.sleep({_STALL_SEC})
    """,
    )
    try:
        assert proc.stdout.readline().strip() == "READY"
        proc.send_signal(signal.SIGKILL)
        _wait_for_death(proc, tmp_path)
    finally:
        if proc.poll() is None:
            proc.kill()
    entries = json.loads(inflight.read_text())
    assert [e["slug"] for e in entries] == ["me/k-9"]
    assert not _deletions(tmp_path), "SIGKILL cannot have run our handler"


def _failing_kaggle(bin_dir: Path, record: Path) -> None:
    """A `kaggle` that records the call and then refuses, like a transient
    API rejection. `subprocess.run` does not raise on that, so nothing but the
    return code separates it from a delete that worked."""
    bin_dir.mkdir(parents = True, exist_ok = True)
    shim = bin_dir / "kaggle"
    shim.write_text(
        textwrap.dedent(f"""\
        #!{sys.executable}
        import sys, pathlib
        pathlib.Path({str(record)!r}).open("a").write(" ".join(sys.argv[1:]) + "\\n")
        sys.exit(1)
        """),
        encoding = "utf-8",
    )
    shim.chmod(0o755)


def test_a_delete_kaggle_refuses_is_not_recorded_as_reclaimed(tmp_path, monkeypatch):
    """A nonzero exit means the kernel may still be running and still
    billing. Dropping its registry entry would leave nothing to reclaim it."""
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    _failing_kaggle(tmp_path / "bin", tmp_path / "kaggle_calls.txt")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    launch._inflight_write([{"slug": "me/orphan", "pid": dead.pid, "at": 0}])

    assert launch.sweep_orphans() == []
    assert [e["slug"] for e in launch._inflight_read()] == ["me/orphan"]


def test_a_release_kaggle_refuses_is_not_marked_released(tmp_path, monkeypatch):
    """The same rule as the sweep, on the path that runs while the kernel is
    known to be up: an unconfirmed delete reads as STILL BILLING.

    So the entry is not marked released, the registry keeps the slug for a
    later sweep, and the run says out loud which kernels a human has to go and
    delete.
    """
    result = _run_main(tmp_path, monkeypatch, push_impl = _push_ok("me/k-1"), kaggle = _failing_kaggle)
    entry = result["kernels"][0]
    assert entry["released"] is False
    assert entry["released_slugs"] == []
    assert result["unreleased"] == ["me/k-1"]
    assert [e["slug"] for e in launch._inflight_read()] == ["me/k-1"]
    # Retried rather than written off, since a refusal says nothing about the kernel;
    # see DELETE_ATTEMPTS.
    assert len(_deletions(tmp_path)) == launch.DELETE_ATTEMPTS


def test_a_successful_release_is_still_recorded(tmp_path, monkeypatch):
    result = _run_main(tmp_path, monkeypatch, push_impl = _push_ok("me/k-1"))
    entry = result["kernels"][0]
    assert entry["released"] is True
    assert entry["released_slugs"] == ["me/k-1"]
    assert result["unreleased"] == []
    assert _deletions(tmp_path) == ["kernels delete me/k-1 -y"]
    assert launch._inflight_read() == []


def test_a_deliberately_kept_kernel_is_not_swept_away_later(tmp_path, monkeypatch):
    """--keep-kernel left the entry naming a pid that dies with the launcher,
    so the next invocation called it an orphan and deleted exactly what the
    flag asked to keep."""
    _run_main(
        tmp_path,
        monkeypatch,
        push_impl = _push_ok("me/kept"),
        argv_extra = ("--keep-kernel",),
    )
    assert not _deletions(tmp_path)
    # The owner is now gone, which is what makes the next sweep interested.
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    entries = launch._inflight_read()
    entries[0]["pid"] = dead.pid
    launch._inflight_write(entries)

    assert launch.sweep_orphans() == []
    assert not _deletions(tmp_path)
    assert [e["slug"] for e in launch._inflight_read()] == ["me/kept"]


def test_a_kernel_pushed_before_the_signal_is_still_deleted(tmp_path):
    """SIGTERM between the first successful push and the end of the push loop
    used to find no kernel list at all, and left a running kernel behind.

    Both halves of that window are covered here. `me/k-1` is a push that
    RETURNED, so its entry is complete. `me/k-2` is the slug the second push
    filed and had not returned yet when the signal arrived: it exists only in
    the caller-owned `attempted` list the launcher hands to push(), which is
    the whole reason that list is the caller's. A kernel Kaggle may already
    have accepted is exactly as billable as a confirmed one.
    """
    body = "\n".join(
        [
            "",
            "        class _Api:",
            "            CONFIG_NAME_USER = 'username'",
            "            config_values = {'username': 'me'}",
            "        launch._api = lambda *a, **k: _Api()",
            "        launch.sweep_orphans = lambda *a, **k: []",
            "        calls = []",
            "        def _push(notebook, user, kernel_timeout_sec,",
            "                  accelerator='NvidiaTeslaT4', attempted=None):",
            "            calls.append(notebook)",
            "            attempted = [] if attempted is None else attempted",
            "            slug = 'me/k-%d' % len(calls)",
            "            attempted.append(slug)",
            "            if len(calls) == 1:",
            "                return {'ok': True, 'slug': slug, 'attempts': attempted}",
            "            print('READY', flush=True)",
            "            time.sleep(%d)" % _STALL_SEC,
            "        launch.push = _push",
            "        sys.argv = ['launch.py', '--notebook', 'a.ipynb', '--notebook', 'b.ipynb',",
            f"                    '--user', 'me', '--outdir', {str(tmp_path / 'out')!r}]",
            "        launch.main()",
        ]
    )
    proc = _runner(tmp_path, body)
    try:
        _await_ready(proc)
        proc.send_signal(signal.SIGTERM)
        _wait_for_death(proc, tmp_path)
    finally:
        if proc.poll() is None:
            proc.kill()
    deleted = _deletions(tmp_path)
    assert any(
        "me/k-1" in c for c in deleted
    ), "the kernel pushed before the signal was left running"
    assert any(
        "me/k-2" in c for c in deleted
    ), "the slug the in-flight push had already filed was left running"
