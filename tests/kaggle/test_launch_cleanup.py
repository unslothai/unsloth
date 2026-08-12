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
