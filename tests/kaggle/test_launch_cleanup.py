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


def _deletions(tmp_path: Path) -> list[str]:
    record = tmp_path / "kaggle_calls.txt"
    if not record.is_file():
        return []
    return [l for l in record.read_text().splitlines() if l.startswith("kernels delete")]


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


def test_a_stalled_push_is_reported_as_infra_not_raised(tmp_path, monkeypatch):
    """Every other Kaggle transport failure returns a reason and exits 0.

    A `kaggle kernels push` that stalls past the 600s ceiling raises
    TimeoutExpired, and letting it escape ends the process before
    `finish()` writes launch_result.json: the launch step goes red and the
    reporter never gets to call the run NOT RUN. Red is reserved for a payload
    that ran on a T4 and failed an assertion.
    """
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    notebook = tmp_path / "kernel.ipynb"
    notebook.write_text("{}", encoding = "utf-8")

    def _stall(*a, **kw):
        raise subprocess.TimeoutExpired(cmd = ["kaggle"], timeout = 600)

    monkeypatch.setattr(launch.subprocess, "run", _stall)
    pushed = launch.push(notebook, "me", 3600)
    assert pushed["ok"] is False
    assert pushed["reason"] == "push_timeout"


def test_a_timed_out_push_does_not_forget_the_kernel_it_may_have_created(tmp_path, monkeypatch):
    """A stalled CLI says nothing about whether Kaggle took the push.

    Kaggle can accept it and start billing before the response is lost, and
    the slug is ours and already decided at that point. Forgetting it left a
    running kernel that `finish()` could not delete and that no later orphan
    sweep could see either, so it billed to its 70 minute ceiling with nobody
    reading the result.
    """
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    notebook = tmp_path / "kernel.ipynb"
    notebook.write_text("{}", encoding = "utf-8")

    def _stall(*a, **kw):
        raise subprocess.TimeoutExpired(cmd = ["kaggle"], timeout = 600)

    monkeypatch.setattr(launch.subprocess, "run", _stall)
    pushed = launch.push(notebook, "me", 3600)

    slug = pushed["orphan_slug"]
    assert slug.startswith("me/unsloth-t4-ci-")
    # Not "slug": the caller must not WAIT on a kernel that may not exist.
    assert pushed.get("slug") is None
    assert [e["slug"] for e in launch._inflight_read()] == [slug]


def test_a_kernel_only_a_timeout_knows_about_is_still_deleted(tmp_path, monkeypatch):
    """release_kernels is the budget control, so it has to act on the slug a
    timed out push left behind and not only on a confirmed one."""
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    _fake_kaggle(tmp_path / "bin", tmp_path / "kaggle_calls.txt")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    launch._inflight_add("me/k-timeout")
    result = {"kernels": [{"slug": None, "orphan_slug": "me/k-timeout"}]}

    class A:
        keep_kernel = False

    launch.release_kernels(result, A())
    assert _deletions(tmp_path) == ["kernels delete me/k-timeout -y"]
    assert launch._inflight_read() == []


def test_a_corrupt_registry_does_not_take_the_run_down(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    (tmp_path / "inflight.json").write_text("{not json", encoding = "utf-8")
    assert launch._inflight_read() == []
    assert launch.sweep_orphans() == []


# --------------------------------------------------------------------------
# the signals
# --------------------------------------------------------------------------
@pytest.mark.parametrize("signame", ["SIGINT", "SIGTERM"])
def test_a_signalled_launcher_deletes_its_kernels(tmp_path, signame):
    """SIGTERM is what `kill` and an Actions cancel send; SIGINT is Ctrl-C.
    Before the handlers, neither deleted anything.
    """
    proc = _runner(
        tmp_path,
        """
        result = {"kernels": [{"slug": "me/k-1", "released": False}]}
        class A:
            keep_kernel = False
        launch._install_release_handlers(result, A())
        print("READY", flush=True)
        time.sleep(60)
    """,
    )
    try:
        assert proc.stdout.readline().strip() == "READY"
        proc.send_signal(getattr(signal, signame))
        proc.wait(timeout = 30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert any(
        "me/k-1" in c for c in _deletions(tmp_path)
    ), f"{signame} left the kernel behind; it would bill to its ceiling"


def test_the_exit_status_still_says_it_was_killed(tmp_path):
    """A handler that swallows the signal and exits 0 makes a cancelled job
    look like a completed one."""
    proc = _runner(
        tmp_path,
        """
        result = {"kernels": [{"slug": "me/k-1", "released": False}]}
        class A:
            keep_kernel = False
        launch._install_release_handlers(result, A())
        print("READY", flush=True)
        time.sleep(60)
    """,
    )
    try:
        assert proc.stdout.readline().strip() == "READY"
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout = 30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert (
        proc.returncode == -signal.SIGTERM
    ), f"expected death by SIGTERM, got returncode {proc.returncode}"


def test_an_unhandled_exception_still_deletes(tmp_path):
    """atexit covers the path no signal handler sees."""
    proc = _runner(
        tmp_path,
        """
        result = {"kernels": [{"slug": "me/k-1", "released": False}]}
        class A:
            keep_kernel = False
        launch._install_release_handlers(result, A())
        raise RuntimeError("boom")
    """,
    )
    proc.wait(timeout = 30)
    assert any("me/k-1" in c for c in _deletions(tmp_path))


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
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    _failing_kaggle(tmp_path / "bin", tmp_path / "kaggle_calls.txt")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    launch._inflight_add("me/k-1")
    result = {"kernels": [{"slug": "me/k-1", "released": False}]}

    class A:
        keep_kernel = False

    launch.release_kernels(result, A())
    assert result["kernels"][0]["released"] is False
    assert [e["slug"] for e in launch._inflight_read()] == ["me/k-1"]


def test_a_successful_release_is_still_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    _fake_kaggle(tmp_path / "bin", tmp_path / "kaggle_calls.txt")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    launch._inflight_add("me/k-1")
    result = {"kernels": [{"slug": "me/k-1", "released": False}]}

    class A:
        keep_kernel = False

    launch.release_kernels(result, A())
    assert result["kernels"][0]["released"] is True
    assert launch._inflight_read() == []


def test_a_deliberately_kept_kernel_is_not_swept_away_later(tmp_path, monkeypatch):
    """--keep-kernel left the entry naming a pid that dies with the launcher,
    so the next invocation called it an orphan and deleted exactly what the
    flag asked to keep."""
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    _fake_kaggle(tmp_path / "bin", tmp_path / "kaggle_calls.txt")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    launch._inflight_add("me/kept")
    result = {"kernels": [{"slug": "me/kept", "released": False}]}

    class A:
        keep_kernel = True

    launch.release_kernels(result, A())
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
    used to find no kernel list at all, and left a running kernel behind."""
    body = "\n".join(
        [
            "",
            "        launch._api = lambda: object()",
            "        launch.sweep_orphans = lambda: []",
            "        calls = []",
            "        def _push(notebook, user, kernel_timeout_sec, accelerator='NvidiaTeslaT4'):",
            "            calls.append(notebook)",
            "            if len(calls) == 1:",
            "                return {'ok': True, 'slug': 'me/k-1'}",
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
        for _ in range(50):
            if proc.stdout.readline().strip() == "READY":
                break
        else:
            raise AssertionError("the second push never started")
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout = 30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert any(
        "me/k-1" in c for c in _deletions(tmp_path)
    ), "the kernel pushed before the signal was left running"
