# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No child outlives the Studio that spawned it.

The chain this closes: a tool call runs under a shell wrapper, the kill path
reaped only the wrapper on Windows, and the orphaned venv python then made
`unsloth studio update` refuse to run until it was killed by hand.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

IS_WINDOWS = sys.platform == "win32"


def _alive(pid: int) -> bool:
    if IS_WINDOWS:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output = True,
            text = True,
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill(pid: int) -> None:
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output = True)
        else:
            os.kill(pid, 9)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. Killing a tool call takes the payload with it
# ---------------------------------------------------------------------------
def test_tool_kill_takes_the_shell_payload_with_it(tmp_path):
    from core.inference.tools import _get_shell_cmd, _kill_process_tree

    pidfile = tmp_path / "child.pid"
    payload = (
        f"import os,time,pathlib;"
        f"pathlib.Path(r'{pidfile}').write_text(str(os.getpid()));"
        f"time.sleep(120)"
    )
    command = f'"{sys.executable}" -c "{payload}"'
    argv = _get_shell_cmd(command)
    print(f"\n[{sys.platform}] shell wrapper argv[0] = {argv[0]}")

    kwargs = {}
    if not IS_WINDOWS:
        kwargs["start_new_session"] = True  # what the real spawn does on POSIX
    proc = subprocess.Popen(argv, **kwargs)

    for _ in range(100):
        if pidfile.is_file() and pidfile.read_text().strip():
            break
        time.sleep(0.1)
    grandchild = int(pidfile.read_text().strip())
    assert _alive(grandchild)

    try:
        _kill_process_tree(proc)
        proc.wait(timeout = 10)
        time.sleep(2.0)
        survived = _alive(grandchild)
        print(f"payload pid {grandchild} alive after _kill_process_tree: {survived}")
        # Windows reaches the payload via taskkill /T /F, POSIX via killpg. It
        # used to survive on Windows, orphaning the venv python that then blocked
        # `unsloth studio update`.
        assert not survived, "the payload under the shell wrapper was orphaned"
    finally:
        _kill(grandchild)


# ---------------------------------------------------------------------------
# 2. One surviving venv process blocks `unsloth studio update` (Windows)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not IS_WINDOWS, reason = "the update gate is Windows-only")
def test_update_gate_blocks_on_a_single_orphan(tmp_path):
    from unsloth_cli import _studio_runtime_gate

    studio_home = tmp_path / "studio_home"
    venv = studio_home / "unsloth_studio"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check = True)
    venv_python = venv / "Scripts" / "python.exe"
    assert venv_python.is_file()

    # No orphan: the gate lets the update through.
    _studio_runtime_gate.ensure_managed_environment_is_idle(studio_home)

    orphan = subprocess.Popen([str(venv_python), "-c", "import time; time.sleep(120)"])
    try:
        time.sleep(2.0)
        with pytest.raises(RuntimeError) as excinfo:
            _studio_runtime_gate.ensure_managed_environment_is_idle(studio_home)
        print(f"\nupdate gate said: {excinfo.value}")
        assert "in use by" in str(excinfo.value)
        assert str(orphan.pid) in str(excinfo.value)
    finally:
        _kill(orphan.pid)


@pytest.mark.skipif(IS_WINDOWS, reason = "contrast case for POSIX")
def test_update_gate_is_a_noop_on_posix(tmp_path):
    """On Linux/macOS nothing checks for a running Studio before an update."""
    from unsloth_cli import _studio_runtime_gate
    _studio_runtime_gate.ensure_managed_environment_is_idle(tmp_path / "anything")


# ---------------------------------------------------------------------------
# 3. Shutdown paths that are not signals
# ---------------------------------------------------------------------------
def test_console_close_runs_the_graceful_shutdown():
    """Closing the console window is not a signal, so it needs its own handler."""
    import run

    assert hasattr(run, "_install_windows_console_handler")
    if not IS_WINDOWS:
        assert run._install_windows_console_handler(lambda *a: None) is False
        return

    calls = []
    assert run._install_windows_console_handler(lambda *a: calls.append(a)) is True


def test_windows_job_status_is_reported():
    """A silent failure meant nobody could tell the guarantee was off."""
    from utils import process_lifetime

    process_lifetime.initialize_parent_lifetime()
    in_force, detail = process_lifetime.windows_job_status()
    print(f"\n[{sys.platform}] job status: in_force={in_force} detail={detail!r}")
    assert detail != "not attempted"
    if IS_WINDOWS or sys.platform.startswith("linux"):
        assert in_force, detail
    else:
        # macOS: no kernel mechanism, so it must say so rather than imply cover.
        assert not in_force


def test_macos_style_orphans_are_recorded_and_reaped(tmp_path, monkeypatch):
    """The on-disk child record is the only reaper macOS has after a crash."""
    from utils import process_lifetime as pl

    records = tmp_path / "children"
    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(records))
    pl._tracked_pids.clear()

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        pl.adopt_pid(child.pid)
        record = records / f"{os.getpid()}.json"
        assert record.is_file(), "the child was not recorded"

        # Pretend a previous Studio wrote this and died. The identity is what
        # decides, not the pid: a dead pid is recycled fast on a busy machine
        # (macOS especially), and an owner whose start time no longer matches is
        # a different process, so its recorded children are orphans.
        import json

        payload = json.loads(record.read_text())
        payload["owner_identity"] = "a-previous-studio-that-is-gone"
        record.write_text(json.dumps(payload))

        reaped = pl.reap_recorded_children()
        print(f"\nreaped: {reaped}")
        assert child.pid in reaped
        # Reap our own zombie so the liveness check means something.
        child.wait(timeout = 10)
        assert not _alive(child.pid)
        assert not record.exists(), "the record should be consumed"
    finally:
        _kill(child.pid)


def test_liveness_probe_does_not_kill_what_it_probes():
    """os.kill(pid, 0) is TerminateProcess on Windows, so the probe cannot use it."""
    from utils import process_lifetime as pl

    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        for _ in range(3):
            assert pl._pid_alive(victim.pid) is True
            time.sleep(0.3)
        assert victim.poll() is None, "the liveness probe killed the process"
    finally:
        _kill(victim.pid)

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout = 30)
    assert pl._pid_alive(dead.pid) is False


def test_a_second_studio_does_not_erase_the_first_record(tmp_path, monkeypatch):
    """Two Studios can share a home; one record per owner keeps both tracked."""
    from utils import process_lifetime as pl

    records = tmp_path / "children"
    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(records))
    pl._tracked_pids.clear()

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        pl.adopt_pid(child.pid)
        # A sibling Studio's record, written under its own pid.
        import json

        other = records / "424242.json"
        other.write_text(
            json.dumps(
                {
                    "owner_pid": 424242,
                    "owner_identity": "a-studio-that-is-gone",
                    "children": [],
                }
            )
        )
        pl.adopt_pid(child.pid)  # rewrites ours only
        assert other.is_file(), "a sibling's record was erased"
        assert (records / f"{os.getpid()}.json").is_file()
    finally:
        _kill(child.pid)


def test_a_live_owner_is_never_reaped(tmp_path, monkeypatch):
    """Two Studios at once must not kill each other's children."""
    from utils import process_lifetime as pl

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    pl._tracked_pids.clear()

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        pl.adopt_pid(child.pid)
        # owner_pid is this process, which is very much alive.
        assert pl.reap_recorded_children() == []
        assert _alive(child.pid)
    finally:
        _kill(child.pid)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        "[]",
        "null",
        '{"children": "nope"}',
        '{"children": [1, 2]}',
        '{"children": [{"pid": "x"}]}',
    ],
)
def test_a_malformed_record_never_blocks_startup(tmp_path, monkeypatch, content):
    """The sweep runs before the server binds, so it must not raise on a record
    written by an older build or truncated by a power cut."""
    from utils import process_lifetime as pl

    directory = tmp_path / "children"
    directory.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(directory))
    (directory / "700000.json").write_text(content)
    assert pl.reap_recorded_children() == []


def test_identity_separates_two_processes_started_together(tmp_path):
    """Linux starttime has 10ms granularity, so identity carries the command
    name too: a recycled pid has to match both."""
    from utils import process_lifetime as pl

    first = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    second = subprocess.Popen(["sleep", "30"]) if not IS_WINDOWS else None
    try:
        assert pl._pid_identity(first.pid) == pl._pid_identity(first.pid)
        if second is not None:
            assert pl._pid_identity(first.pid) != pl._pid_identity(second.pid)
    finally:
        _kill(first.pid)
        if second is not None:
            _kill(second.pid)


def test_ctrl_c_is_passed_on_by_the_console_handler():
    """Ctrl+C and Ctrl+Break must report "not handled" so Python's own signal
    handler still runs. Returning True (or raising inside the ctypes callback,
    where the BOOL result is then undefined) would leave Studio unstoppable."""
    import run

    assert run._console_event_is_shutdown(0) is False  # CTRL_C_EVENT
    assert run._console_event_is_shutdown(1) is False  # CTRL_BREAK_EVENT
    for close_event in (2, 5, 6):  # CLOSE, LOGOFF, SHUTDOWN
        assert run._console_event_is_shutdown(close_event) is True


def test_a_surviving_child_stays_in_the_record(tmp_path, monkeypatch):
    """forget_pid is for confirmed exits; a survivor must remain reapable."""
    from utils import process_lifetime as pl

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    pl._tracked_pids.clear()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        pl.adopt_pid(child.pid)
        assert child.pid in pl._tracked_pids
        # A failed kill leaves poll() as None, so llama_cpp must not forget it.
        assert child.poll() is None
    finally:
        _kill(child.pid)


def test_concurrent_adopts_all_survive(tmp_path, monkeypatch):
    """Two threads adopting at once must not lose either pid."""
    import json
    import threading

    from utils import process_lifetime as pl

    directory = tmp_path / "children"
    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(directory))
    pl._tracked_pids.clear()
    pids = list(range(900000, 900040))
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"id-{pid}")

    def adopt(chunk):
        for pid in chunk:
            pl.adopt_pid(pid)

    threads = [threading.Thread(target = adopt, args = (pids[i::4],)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    record = json.loads((directory / f"{os.getpid()}.json").read_text())
    assert sorted(entry["pid"] for entry in record["children"]) == pids
    pl._tracked_pids.clear()


def test_a_survivor_keeps_its_record_through_a_clean_shutdown(tmp_path, monkeypatch):
    """terminate_all cannot confirm every exit, and the record is the only
    handle the next startup has on what is left."""
    from utils import process_lifetime as pl

    directory = tmp_path / "children"
    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(directory))
    pl._tracked_pids.clear()

    stubborn = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        pl.adopt_pid(stubborn.pid)
        # Signalling silently does nothing, as an unkillable child would.
        monkeypatch.setattr(pl, "_posix_terminate", lambda pid, timeout = 5.0: None)
        monkeypatch.setattr(pl.os, "kill", lambda pid, sig: None)

        survivors = pl.terminate_all()
        assert survivors == [stubborn.pid]

        pl.clear_breadcrumb()
        record = directory / f"{os.getpid()}.json"
        assert record.is_file(), "the only handle on the survivor was deleted"
        import json

        assert [e["pid"] for e in json.loads(record.read_text())["children"]] == [stubborn.pid]
    finally:
        _kill(stubborn.pid)
    pl._tracked_pids.clear()


def test_a_confirmed_shutdown_still_clears_the_record(tmp_path, monkeypatch):
    from utils import process_lifetime as pl

    directory = tmp_path / "children"
    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(directory))
    pl._tracked_pids.clear()

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pl.adopt_pid(child.pid)
    assert pl.terminate_all(timeout = 3.0) == []
    pl.clear_breadcrumb()
    assert list(directory.glob("*.json")) == []


def test_a_live_owner_survives_an_unreadable_identity(tmp_path, monkeypatch):
    """A `ps` that failed for a moment must not cost a running Studio its
    sidecars."""
    import json

    from utils import process_lifetime as pl

    directory = tmp_path / "children"
    directory.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(directory))

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (directory / "123456.json").write_text(
            json.dumps(
                {
                    "owner_pid": os.getppid(),  # alive
                    "owner_identity": "recorded-when-it-started",
                    "children": [{"pid": child.pid, "identity": pl._pid_identity(child.pid)}],
                }
            )
        )
        # The lookup fails only for the owner, as a transient `ps` error would.
        real = pl._pid_identity
        monkeypatch.setattr(
            pl,
            "_pid_identity",
            lambda pid: None if pid == os.getppid() else real(pid),
        )
        assert pl.reap_recorded_children() == []
        assert child.poll() is None, "a live Studio's child was killed"
    finally:
        _kill(child.pid)


def test_an_unverifiable_captured_pid_is_not_taskkilled(monkeypatch):
    """The delayed Windows path fails closed: the job object covers the tree."""
    from core.inference import tools

    ran = []
    monkeypatch.setattr(
        tools, "_windows_taskkill_tree", lambda pid, identity = None: ran.append(pid) or True
    )
    tools._killpg_captured(("windows-tree", 4321, None))
    assert ran == [], "signalled a pid it could not verify"
    tools._killpg_captured(("windows-tree", 4321, "created-at-t0"))
    assert ran == [4321]


def test_the_status_is_not_claimed_when_prctl_is_blocked(monkeypatch):
    """Under a seccomp or container policy that rejects prctl, children can
    survive a hard parent exit; the diagnostic must not say otherwise."""
    from utils import process_lifetime as pl

    monkeypatch.setattr(pl, "_is_windows", lambda: False)
    monkeypatch.setattr(pl, "_is_linux", lambda: True)
    monkeypatch.setattr(pl, "_pdeathsig_available", lambda: False)
    monkeypatch.setattr(pl, "_initialized", False)
    pl.initialize_parent_lifetime()
    in_force, detail = pl.windows_job_status()
    assert in_force is False
    assert "prctl" in detail

    monkeypatch.setattr(pl, "_pdeathsig_available", lambda: True)
    monkeypatch.setattr(pl, "_initialized", False)
    pl.initialize_parent_lifetime()
    assert pl.windows_job_status()[0] is True


def test_the_probe_does_not_arm_anything(monkeypatch):
    """PR_GET_PDEATHSIG is read-only, so probing must leave our own setting be."""
    import ctypes

    from utils import process_lifetime as pl

    if not pl._is_linux():
        pytest.skip("Linux only")
    libc = ctypes.CDLL("libc.so.6", use_errno = True)
    before = ctypes.c_int(0)
    libc.prctl(2, ctypes.byref(before), 0, 0, 0)
    assert pl._pdeathsig_available() is True
    after = ctypes.c_int(0)
    libc.prctl(2, ctypes.byref(after), 0, 0, 0)
    assert after.value == before.value


def test_the_sweep_snapshot_is_taken_under_the_lock():
    """Writes hold _record_lock, so the read has to as well."""
    import ast
    import inspect

    from utils import process_lifetime as pl

    tree = ast.parse(inspect.getsource(pl.terminate_all))
    body = ast.dump(tree)
    assert "_record_lock" in body
    # The snapshot must not be a bare list() over the live dict.
    source = inspect.getsource(pl.terminate_all)
    assert "for pid, identity in list(_tracked_pids.items())" not in source


def test_the_probe_fails_when_only_the_read_is_permitted(monkeypatch):
    """seccomp filters prctl on its first argument, so a working GET says
    nothing about SET."""
    import ctypes

    from utils import process_lifetime as pl

    class _Libc:
        def prctl(self, op, *rest):
            return 0 if op == 2 else -1  # GET ok, SET rejected

    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: _Libc())
    assert pl._pdeathsig_available() is False


def test_sd_cli_is_recorded_while_it_runs(tmp_path, monkeypatch):
    """It holds VRAM for the length of a generation and macOS has no
    parent-death signal, so a crash mid-run must leave something to reap."""
    from core.inference import sd_cpp_engine

    adopted, forgotten = [], []
    monkeypatch.setattr(sd_cpp_engine, "adopt_pid", adopted.append)
    monkeypatch.setattr(sd_cpp_engine, "forget_pid", forgotten.append)
    monkeypatch.setattr(sd_cpp_engine, "runtime_env", lambda binary, base = None: dict(os.environ))

    engine = sd_cpp_engine.SdCppEngine.__new__(sd_cpp_engine.SdCppEngine)
    engine.binary = sys.executable
    out = tmp_path / "image.png"
    engine._run(
        [sys.executable, "-c", f"open({str(out)!r}, 'wb').write(b'x')"],
        str(out),
        timeout = 60,
        env = None,
        on_log = None,
    )
    assert len(adopted) == 1 and adopted[0] > 0
    assert forgotten == adopted, "a finished run must not stay on the record"


def test_the_diffusion_runner_is_recorded_too():
    """The llama-server path adopts through _record_server_pid; this one spawns
    directly and was invisible to the sweep."""
    import ast
    import inspect

    from core.inference import llama_cpp

    source = inspect.getsource(llama_cpp)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.dump(node)
        if "diffusion-stdout" not in body:
            continue
        assert "adopt_pid" in body, f"{node.name} spawns a runner it never records"
        break
    else:
        raise AssertionError("could not find the diffusion runner spawn")


def test_a_windows_tool_tree_dies_with_its_job(monkeypatch):
    """taskkill cannot reach a tree whose root already exited, which is the
    case this capture exists for; the job handle still can."""
    from core.inference import tools

    class _Job:
        def __init__(self):
            self.terminated = 0

        def terminate(self):
            self.terminated += 1
            return True

    taskkilled = []
    monkeypatch.setattr(
        tools, "_windows_taskkill_tree", lambda pid, identity = None: taskkilled.append(pid) or True
    )
    job = _Job()
    tools._killpg_captured(("windows-job", job))
    assert job.terminated == 1
    assert taskkilled == [], "the job is the whole reach; no pid revalidation needed"


def test_the_windows_breadcrumb_fallback_takes_the_whole_tree(tmp_path, monkeypatch):
    """Killing the leader alone strands its workers, and the record naming them
    is deleted straight after."""
    import json

    from utils import process_lifetime as pl

    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(pl, "_identity_or_none", lambda pid: "same" if pid == 4242 else None)
    trees = []
    monkeypatch.setattr(pl, "_windows_terminate_tree", trees.append)

    record = tmp_path / "999.json"
    record.write_text(
        json.dumps(
            {
                "owner_pid": 999,
                "owner_identity": None,
                "children": [{"pid": 4242, "identity": "same"}],
            }
        ),
        encoding = "utf-8",
    )
    killed = pl._reap_one_record(record, timeout = 1.0)
    assert killed == [4242]
    assert trees == [4242], "only the leader was signalled"


def test_the_diffusion_runner_leads_its_own_group():
    """The shim spawns the visual server, and _posix_terminate only reaches a
    tree through killpg, which needs the recorded pid to be a group leader."""
    import ast
    import inspect

    from core.inference import llama_cpp

    tree = ast.parse(inspect.getsource(llama_cpp))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or "diffusion-stdout" not in ast.dump(node):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if getattr(call.func, "attr", None) != "Popen":
                continue
            assert any(
                kw.arg == "start_new_session" for kw in call.keywords
            ), "the runner shares Studio's process group, so its child is unreachable"
            return
    raise AssertionError("could not find the diffusion runner spawn")


def test_a_group_leader_is_reaped_with_its_children(tmp_path, monkeypatch):
    """What start_new_session buys: killpg takes the grandchild too."""
    from utils import process_lifetime as pl

    if pl._is_windows():
        pytest.skip("POSIX only")

    marker = tmp_path / "grandchild.pid"
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time, pathlib\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n"
            "time.sleep(60)\n",
        ],
        start_new_session = True,
    )
    try:
        for _ in range(100):
            if marker.is_file():
                break
            time.sleep(0.1)
        grandchild = int(marker.read_text())
        pl._posix_terminate(leader.pid, timeout = 5.0)
        time.sleep(0.5)
        assert not _alive(grandchild), "the visual server survived its runner"
    finally:
        _kill(leader.pid)


def test_every_restart_path_checks_the_rearm():
    """Skip & Restart is offered on every error, so gating only the recovery
    path still lets a user start a backend under a disarmed job."""
    hook = (
        Path(__file__).resolve().parents[2] / "frontend" / "src" / "hooks" / "use-tauri-update.ts"
    )
    source = hook.read_text(encoding = "utf-8")
    starts = [
        block for block in source.split("async function ")[1:] if 'invoke("start_server"' in block
    ]
    assert starts, "no restart path found"
    for block in starts:
        name = block.split("(")[0]
        assert "cleanupRearmedRef.current" in block, f"{name} restarts without checking the re-arm"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
