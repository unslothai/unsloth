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


def _runner(tmp_path: Path, body: str) -> subprocess.Popen:
    """Run `body` against the real launch module, with a fake kaggle CLI."""
    record = tmp_path / "kaggle_calls.txt"
    _fake_kaggle(tmp_path / "bin", record)
    script = tmp_path / "runner.py"
    script.write_text(
        textwrap.dedent(f"""\
        import sys, time
        sys.path.insert(0, {str(CI_DIR)!r})
        import launch
        launch.INFLIGHT = __import__("pathlib").Path({str(tmp_path / "inflight.json")!r})
        {body}
        """),
        encoding = "utf-8",
    )
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.Popen(
        [sys.executable, str(script)],
        env = env,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
    )


def _await_ready(proc: subprocess.Popen) -> None:
    """Wait for the runner to say it is where the test wants it, past whatever
    the launcher logged on the way there."""
    for _ in range(50):
        if proc.stdout.readline().strip() == "READY":
            return
    raise AssertionError("the runner never reached its READY point")


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
    # The retry backoffs, not the retries: every attempt still runs.
    monkeypatch.setattr(launch, "PUSH_BACKOFF_SEC", 0)
    monkeypatch.setattr(launch, "DELETE_BACKOFF_SEC", 0)
    monkeypatch.setattr(launch, "_api", lambda: object())
    monkeypatch.setattr(launch, "wait", lambda *a, **kw: "COMPLETE")
    monkeypatch.setattr(
        launch,
        "fetch_evidence",
        lambda *a, **kw: {"notebooks": [], "log": None, "truncated": False},
    )
    # Installing the real handlers here would leave a SIGTERM disposition and
    # an atexit callback on the pytest interpreter for the rest of the session.
    # They have their own tests, which drive a subprocess for that reason.
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
# the registry
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
    # Our own pid counts as alive.
    assert launch.sweep_orphans() == []
    assert [e["slug"] for e in launch._inflight_read()] == ["me/live"]


def test_a_dead_owners_kernel_is_reclaimed(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    _fake_kaggle(tmp_path / "bin", tmp_path / "kaggle_calls.txt")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    # A pid that cannot be running: claim one and let it exit.
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    launch._inflight_write([{"slug": "me/orphan", "pid": dead.pid, "at": 0}])
    assert launch.sweep_orphans() == ["me/orphan"]
    assert launch._inflight_read() == []
    assert any("me/orphan" in c for c in _deletions(tmp_path))


def test_a_failed_delete_keeps_the_entry_for_next_time(tmp_path, monkeypatch):
    """Forgetting a kernel we could not delete is how one bills forever."""
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
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
# the signals
# --------------------------------------------------------------------------
# A launcher that has pushed its kernel and is now waiting on it, which is
# where a cancelled workflow finds it: almost all of a run's wall clock is
# spent here, with the kernel up and billing. main() installs the real
# handlers over the real release(), so the signal is delivered to production
# code and nothing about cleanup is re-implemented in the runner.
def _waiting_launcher(outdir: Path) -> str:
    return "\n".join(
        [
            "",
            "        launch._api = lambda: object()",
            "        launch.sweep_orphans = lambda: []",
            "        def _push(notebook, user, kernel_timeout_sec,",
            "                  accelerator='NvidiaTeslaT4', attempted=None):",
            "            attempted = [] if attempted is None else attempted",
            "            attempted.append('me/k-1')",
            "            launch._inflight_add('me/k-1')",
            "            return {'ok': True, 'slug': 'me/k-1', 'attempts': attempted}",
            "        launch.push = _push",
            "        def _wait(*a, **kw):",
            "            print('READY', flush=True)",
            "            time.sleep(60)",
            "        launch.wait = _wait",
            "        sys.argv = ['launch.py', '--notebook', 'a.ipynb', '--user', 'me',",
            f"                    '--outdir', {str(outdir)!r}]",
            "        launch.main()",
        ]
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
        proc.wait(timeout = 30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert any(
        "me/k-1" in c for c in _deletions(tmp_path)
    ), f"{signame} left the kernel behind; it would bill to its ceiling"
    # And the registry agrees, so no later sweep chases a kernel that is gone.
    assert json.loads((tmp_path / "inflight.json").read_text()) == []


def test_the_exit_status_still_says_it_was_killed(tmp_path):
    """A handler that swallows the signal and exits 0 makes a cancelled job
    look like a completed one."""
    proc = _runner(tmp_path, _waiting_launcher(tmp_path / "out"))
    try:
        _await_ready(proc)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout = 30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert (
        proc.returncode == -signal.SIGTERM
    ), f"expected death by SIGTERM, got returncode {proc.returncode}"


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
    proc.wait(timeout = 30)
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
        time.sleep(60)
    """,
    )
    try:
        assert proc.stdout.readline().strip() == "READY"
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout = 30)
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
    # Retried rather than written off, since a refusal says nothing about the
    # kernel; see DELETE_ATTEMPTS.
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
            "        launch._api = lambda: object()",
            "        launch.sweep_orphans = lambda: []",
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
            "            time.sleep(60)",
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
        proc.wait(timeout = 30)
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
