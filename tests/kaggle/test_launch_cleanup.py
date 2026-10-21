# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""release() must also run on Ctrl-C and SIGTERM, since a kernel left running bills to its ceiling."""

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


def _shared_setup_1(proc, tmp_path):
    try:
        _await_ready(proc)
        proc.send_signal(signal.SIGTERM)
        _wait_for_death(proc, tmp_path)
    finally:
        if proc.poll() is None:
            proc.kill()


REPO_ROOT = Path(__file__).resolve().parents[2]
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"
sys.path.insert(0, str(CI_DIR))

import launch  # noqa: E402

# One xdist worker for this file: on a four-core runner, parallel workers stop the death budget
# measuring the handler and start measuring scheduling. Needs `--dist loadgroup`.
pytestmark = pytest.mark.xdist_group(name = "kaggle_launch_signals")


class _StubKaggleApi:
    """launch.py refuses to push without an owner from the client, so the stub must carry a username."""

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


# Faulthandler to a FILE: fd 2 is the stdout pipe, which one test deliberately stops draining.
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


# A hang guard, not a latency target: scheduling on a loaded four-core runner can exceed 30s.
_DEATH_BUDGET_SEC = 120

# Must outlast the budget, or a handler that swallows its signal wakes and exits normally.
_STALL_SEC = 900

# Sleep in slices: a signal landing just before a long sleep's syscall is handled only after
# the whole sleep. The slices still sum to _STALL_SEC.
_STALL_SLICE_SEC = 0.5
_STALL = (
    f"for _slice in range({int(_STALL_SEC / _STALL_SLICE_SEC)}): time.sleep({_STALL_SLICE_SEC})"
)


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
    """Faulthandler dumps to a file, not stderr: stderr shares the stdout pipe, which one test fills."""
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "LAUNCH_FAULT_DUMP": str(_fault_dump(bin_dir.parent)),
    }


def _wait_for_death(proc: subprocess.Popen, tmp_path: Path | None = None) -> None:
    """SIGABRT before SIGKILL so faulthandler writes the stuck stack to a file, not the stdout pipe."""
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
    """Read only after exit, when the pipe is at EOF, so the read cannot block on a full pipe."""
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
        **kwargs,
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
    """Drives the real main() with only the network and clock stubbed; returns launch_result.json."""
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    kaggle(tmp_path / "bin", tmp_path / "kaggle_calls.txt")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(launch, "PUSH_BACKOFF_SEC", 0)
    monkeypatch.setattr(launch, "DELETE_BACKOFF_SEC", 0)
    monkeypatch.setattr(launch, "_api", _stub_api)
    monkeypatch.setattr(launch, "wait", lambda *a, **kw: "COMPLETE")
    monkeypatch.setattr(
        launch,
        "fetch_evidence",
        lambda *a, **kw: {"notebooks": [], "log": None, "truncated": False},
    )
    # The real handlers would leave SIGTERM and atexit state on the pytest process.
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
    """A stalled push must return a reason: an escaping TimeoutExpired skips launch_result.json."""
    notebook = _stalling_push(monkeypatch, tmp_path)
    pushed = launch.push(notebook, "me", 3600)
    assert pushed["ok"] is False
    assert pushed["reason"] == "push_failed"
    assert "timed out" in pushed["detail"], pushed["detail"]


def test_a_timed_out_push_does_not_forget_the_kernel_it_may_have_created(tmp_path, monkeypatch):
    """A timed-out push may still have created a kernel, so its slug must stay in the attempted list."""
    notebook = _stalling_push(monkeypatch, tmp_path)
    owned: list[str] = []
    pushed = launch.push(notebook, "me", 3600, attempted = owned)

    # The caller must not wait on a kernel that may not exist.
    assert pushed.get("slug") is None
    assert pushed["attempts"] == owned
    assert len(owned) == launch.PUSH_ATTEMPTS
    assert all(s.startswith("me/unsloth-t4-ci-") for s in owned), owned
    assert sorted(e["slug"] for e in launch._inflight_read()) == sorted(owned)


def test_a_push_that_raises_still_leaves_its_slug_with_the_caller(tmp_path, monkeypatch):
    """A push can raise past the return, so its slug is recorded before the call that may create it."""
    notebook = _stalling_push(monkeypatch, tmp_path)

    def _explode(*a, **kw):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(launch.subprocess, "run", _explode)
    owned: list[str] = []
    with pytest.raises(UnicodeDecodeError):
        launch.push(notebook, "me", 3600, attempted = owned)
    assert len(owned) == 1 and owned[0].startswith("me/unsloth-t4-ci-")


def test_a_kernel_only_a_timeout_knows_about_is_still_deleted(tmp_path, monkeypatch):
    """With every attempt stalled, push() filed the only names for kernels Kaggle may have started."""
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


# A launcher waiting on its pushed kernel, where a cancelled workflow finds it. main() installs
# the real handlers, so cleanup is production code.
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
            "                  accelerator='NvidiaTeslaT4', attempted=None, **kwargs):",
            "            attempted = [] if attempted is None else attempted",
            "            attempted.append('me/k-1')",
            "            launch._inflight_add('me/k-1')",
            "            return {'ok': True, 'slug': 'me/k-1', 'attempts': attempted}",
            "        launch.push = _push",
            "        def _wait(*a, **kw):",
            "            print('READY', flush=True)",
            "            " + _STALL,
            "        launch.wait = _wait",
            "        sys.argv = ['launch.py', '--notebook', 'a.ipynb', '--user', 'me',",
            f"                    '--outdir', {str(outdir)!r}]",
            "        launch.main()",
        ]
    )


def _flooding_launcher(outdir: Path) -> str:
    """Writes until the stdout pipe is full, so the signal lands while the main thread holds the lock."""
    return (
        _waiting_launcher(outdir)
        .replace(
            "            " + _STALL,
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
    # Checks the signal path: finish() alone would satisfy the deletion assertion.
    assert proc.returncode == -getattr(signal, signame), (
        f"the kernel was deleted, but the launcher exited {proc.returncode} rather than "
        f"dying of {signame}, so nothing here says the signal is what did it. "
        f"Launcher said: {logged}"
    )


def test_the_exit_status_still_says_it_was_killed(tmp_path):
    """A handler that swallows the signal and exits 0 makes a cancelled job
    look like a completed one."""
    proc = _runner(tmp_path, _waiting_launcher(tmp_path / "out"))
    _shared_setup_1(proc, tmp_path)
    assert proc.returncode == -signal.SIGTERM, (
        f"expected death by SIGTERM, got returncode {proc.returncode}. "
        f"Launcher said: {_tail(proc)}"
    )


def test_the_exit_status_survives_a_release_that_fails(tmp_path):
    """A delete raising inside the handler must not turn a cancelled run into exit 0."""
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
    _shared_setup_1(proc, tmp_path)
    assert proc.returncode == -signal.SIGTERM, (
        f"a release() that raised turned SIGTERM into returncode {proc.returncode}; "
        f"a cancelled job would read as a completed one. Launcher said: {_tail(proc)}"
    )
    assert any("me/k-1" in c for c in _deletions(tmp_path))


def test_the_stall_outlasts_the_death_budget():
    """The stall must outlast the death budget, or a launcher that ignores its signal passes the test."""
    assert _STALL_SEC > _DEATH_BUDGET_SEC, (
        f"a launcher that swallows its signal wakes after {_STALL_SEC}s and exits "
        f"normally inside the {_DEATH_BUDGET_SEC}s wait, so the signal tests would "
        f"pass without any signal handling at all"
    )


def test_the_stall_is_sliced_so_a_signal_at_ready_is_not_deferred():
    """Stalls are sliced to 1s: one long sleep would defer a signal handler for the whole stall."""
    ns = {"time": __import__("types").SimpleNamespace(sleep = lambda s: slept.append(s))}
    slept: list[float] = []
    exec(_STALL, ns)
    assert max(slept) <= 1.0, f"a {max(slept)}s slice can hold a pending handler that long"
    assert sum(slept) == pytest.approx(
        _STALL_SEC
    ), "the slices must still add up to the whole stall"
    source = Path(__file__).read_text(encoding = "utf-8")
    long_sleeps = source.count("time.sleep(%d)" + chr(34) + " % _STALL_SEC") + source.count(
        "time.sleep({" + "_STALL_SEC})"
    )
    assert long_sleeps == 0, "a stub went back to one long sleep; stall through _STALL instead"


def test_the_handler_survives_its_own_logging_failing(tmp_path):
    """A signal handler's stdout write can hit a reentrant RuntimeError, which must not stop cleanup."""
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
    _shared_setup_1(proc, tmp_path)
    assert proc.returncode == -signal.SIGTERM, (
        f"a log call that raised inside the handler turned SIGTERM into returncode "
        f"{proc.returncode}. Launcher said: {_tail(proc)}"
    )
    assert any("me/k-1" in c for c in _deletions(tmp_path))


def test_the_handler_survives_a_stdout_nobody_is_draining(tmp_path):
    """A full stdout pipe blocks rather than raises, so the handler must drop its log line, not wait."""
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
    """A reentrant log inside a delete must not escape, or the remaining retries are abandoned."""
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
        {_STALL}
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
    _shared_setup_1(proc, tmp_path)

    attempts = _deletions(tmp_path)
    assert len(attempts) >= 3, (
        f"only {len(attempts)} delete attempts were made. A log line raising inside "
        f"delete_kernel abandoned the retries, so a kernel Kaggle would have released "
        f"on the third attempt keeps billing. Launcher said: {_tail(proc)}"
    )
    assert json.loads((tmp_path / "inflight.json").read_text()) == []


def test_the_leaked_kernel_warning_does_not_strand_the_handler(tmp_path):
    """The leaked-kernel warning is reached only after failed deletes, so its write must not block."""
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
    assert _deletions(tmp_path), "no delete was attempted, so nothing could leak"


def test_the_leaked_warning_is_still_a_github_annotation(tmp_path, monkeypatch, capsys):
    """The warning must stay unprefixed, since GitHub surfaces only annotations that start the line."""
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
    """atexit covers the exit path that main() never sees, so the real delete must run from it."""
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
        {_STALL}
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
    """An unconfirmed delete must not mark the kernel released; its slug stays in the registry."""
    result = _run_main(tmp_path, monkeypatch, push_impl = _push_ok("me/k-1"), kaggle = _failing_kaggle)
    entry = result["kernels"][0]
    assert entry["released"] is False
    assert entry["released_slugs"] == []
    assert result["unreleased"] == ["me/k-1"]
    assert [e["slug"] for e in launch._inflight_read()] == ["me/k-1"]
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
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    entries = launch._inflight_read()
    entries[0]["pid"] = dead.pid
    launch._inflight_write(entries)

    assert launch.sweep_orphans() == []
    assert not _deletions(tmp_path)
    assert [e["slug"] for e in launch._inflight_read()] == ["me/kept"]


def test_a_kernel_pushed_before_the_signal_is_still_deleted(tmp_path):
    """A slug whose push never returned lives only in the caller-owned attempted list."""
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
            "                  accelerator='NvidiaTeslaT4', attempted=None, **kwargs):",
            "            calls.append(notebook)",
            "            attempted = [] if attempted is None else attempted",
            "            slug = 'me/k-%d' % len(calls)",
            "            attempted.append(slug)",
            "            if len(calls) == 1:",
            "                return {'ok': True, 'slug': slug, 'attempts': attempted}",
            "            print('READY', flush=True)",
            "            " + _STALL,
            "        launch.push = _push",
            "        sys.argv = ['launch.py', '--notebook', 'a.ipynb', '--notebook', 'b.ipynb',",
            f"                    '--user', 'me', '--outdir', {str(tmp_path / 'out')!r}]",
            "        launch.main()",
        ]
    )
    proc = _runner(tmp_path, body)
    _shared_setup_1(proc, tmp_path)
    deleted = _deletions(tmp_path)
    assert any(
        "me/k-1" in c for c in deleted
    ), "the kernel pushed before the signal was left running"
    assert any(
        "me/k-2" in c for c in deleted
    ), "the slug the in-flight push had already filed was left running"
