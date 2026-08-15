# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No child outlives the Studio that spawned it.

The chain this closes: a tool call runs under a shell wrapper, the kill path
reaped only the wrapper on Windows, and the orphaned venv python then made
`unsloth studio update` refuse to run until it was killed by hand.
"""

import json
import os
import signal
import subprocess
import threading
import sys
import time
from pathlib import Path

import pytest

IS_WINDOWS = sys.platform == "win32"


def _load_installer_module():
    """install_llama_prebuilt.py, imported from the repo without installing it."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "install_llama_prebuilt.py"
    name = "install_llama_prebuilt_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: its dataclasses resolve annotations through
    # sys.modules, and a module that is not in there fails to build them.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


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


def test_identity_survives_a_child_renaming_itself():
    """Identity is only ever compared for one pid, and comm is mutable: a worker
    calling prctl(PR_SET_NAME) or setproctitle would otherwise read as recycled
    and be dropped unsignalled."""
    from utils import process_lifetime as pl

    if not pl._is_linux():
        pytest.skip("Linux only")

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import ctypes, time\n"
            "time.sleep(0.3)\n"
            'ctypes.CDLL("libc.so.6").prctl(15, b"renamed-worker", 0, 0, 0)\n'
            "time.sleep(30)\n",
        ]
    )
    try:
        before = pl._pid_identity(child.pid)
        with open(f"/proc/{child.pid}/comm", encoding = "utf-8") as fh:
            name_before = fh.read().strip()
        time.sleep(1.0)
        with open(f"/proc/{child.pid}/comm", encoding = "utf-8") as fh:
            assert fh.read().strip() != name_before, "the child did not rename itself"
        after = pl._pid_identity(child.pid)
        assert before == after
        assert pl._same_identity(before, after)
    finally:
        _kill(child.pid)


def test_a_record_written_with_a_process_name_still_matches():
    """Records from an earlier build carry starttime:comm."""
    from utils import process_lifetime as pl

    if not pl._is_linux():
        pytest.skip("Linux only")
    assert pl._same_identity("196794665:python3", "196794665") is True
    assert pl._same_identity("196794665:python3", "196794999") is False


def test_a_group_outliving_its_leader_is_still_reaped(tmp_path, monkeypatch):
    """The shim can crash while the visual server holds the GPU; the record's
    pid is then gone and the group is the only handle left."""
    import json

    from utils import process_lifetime as pl

    if pl._is_windows():
        pytest.skip("POSIX only")

    marker = tmp_path / "grandchild.pid"
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, pathlib\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n",
        ],
        start_new_session = True,
    )
    try:
        leader.wait(timeout = 30)
        for _ in range(100):
            if marker.is_file():
                break
            time.sleep(0.1)
        grandchild = int(marker.read_text())
        assert _alive(grandchild)

        record = tmp_path / "900000.json"
        record.write_text(
            json.dumps(
                {
                    "owner_pid": 900000,
                    "owner_identity": None,
                    "children": [
                        {"pid": leader.pid, "identity": "gone", "pgid": leader.pid},
                    ],
                }
            ),
            encoding = "utf-8",
        )
        pl._reap_one_record(record, timeout = 5.0)
        time.sleep(0.5)
        assert not _alive(grandchild), "the visual server survived its dead leader"
    finally:
        _kill(leader.pid)


def test_a_child_sharing_studios_group_is_never_recorded():
    """Recording that group would make the sweep kill Studio and every sibling."""
    from utils import process_lifetime as pl

    if pl._is_windows():
        pytest.skip("POSIX only")
    shared = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    own = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session = True
    )
    try:
        assert pl._own_process_group(shared.pid) is None
        assert pl._own_process_group(own.pid) == own.pid
    finally:
        _kill(shared.pid)
        _kill(own.pid)


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
    killed, _deferred = pl._reap_one_record(record, timeout = 1.0)
    assert killed == [4242]
    assert trees == [4242], "only the leader was signalled"


def test_the_diffusion_runner_stays_in_the_backend_group():
    """The desktop stops this backend by signalling its process group and
    force-kills it five seconds later, so a session of its own would leave the
    shim and the visual server holding the GPU until the next launch."""
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
            assert not any(
                kw.arg == "start_new_session" for kw in call.keywords
            ), "a session of its own is out of reach of the group the desktop stops"
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


def test_forgetting_a_leader_keeps_its_live_group(tmp_path, monkeypatch):
    """The shim can exit before the visual server it started; dropping the
    record then loses the only handle on that group."""
    from utils import process_lifetime as pl

    if pl._is_windows():
        pytest.skip("POSIX only")

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    pl._tracked_pids.clear()
    pl._tracked_pgids.clear()

    marker = tmp_path / "grandchild.pid"
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, pathlib\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n",
        ],
        start_new_session = True,
    )
    try:
        pl.adopt_pid(leader.pid)
        assert pl._tracked_pgids.get(leader.pid) == leader.pid
        leader.wait(timeout = 30)
        for _ in range(100):
            if marker.is_file():
                break
            time.sleep(0.1)
        grandchild = int(marker.read_text())

        pl.forget_pid(leader.pid)
        assert leader.pid in pl._tracked_pids, "the group was dropped with its dead leader"

        # And the backstop then takes it.
        pl.terminate_all(timeout = 5.0)
        time.sleep(0.5)
        assert not _alive(grandchild)
    finally:
        _kill(leader.pid)


def test_forgetting_a_leader_with_no_group_left_drops_it(tmp_path, monkeypatch):
    from utils import process_lifetime as pl

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    pl._tracked_pids.clear()
    pl._tracked_pgids.clear()

    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session = True)
    child.wait(timeout = 30)
    pl.adopt_pid(child.pid)
    pl.forget_pid(child.pid)
    assert child.pid not in pl._tracked_pids
    assert child.pid not in pl._tracked_pgids


def test_the_windows_backstop_takes_the_whole_tree(tmp_path, monkeypatch):
    """Same gap the startup sweep had: killing the leader alone strands its
    workers, and clear_breadcrumb removes the record straight after."""
    from utils import process_lifetime as pl

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "same")
    monkeypatch.setattr(pl, "_identity_or_none", lambda pid: "same")
    trees = []
    monkeypatch.setattr(pl, "_windows_terminate_tree", trees.append)
    killed_singly = []
    monkeypatch.setattr(pl.os, "kill", lambda pid, sig: killed_singly.append(pid))

    pl._tracked_pids.clear()
    pl._tracked_pgids.clear()
    pl._tracked_pids[4242] = "same"
    pl.terminate_all(timeout = 1.0)
    assert trees == [4242]
    assert killed_singly == [], "the leader was signalled on its own"


def test_the_windows_identity_probe_prototypes_every_handle_call():
    """Without argtypes ctypes marshals a 64-bit HANDLE as c_int, so the handle
    is truncated and the probe leaks one per call."""
    import ast
    import inspect

    from utils import process_lifetime as pl

    source = inspect.getsource(pl._pid_identity)
    tree = ast.parse(source)
    prototyped = {
        node.targets[0].value.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr in ("argtypes", "restype")
        and isinstance(node.targets[0].value, ast.Attribute)
    }
    for call in ("OpenProcess", "GetProcessTimes", "CloseHandle"):
        assert call in prototyped, f"{call} is called without a signature"


def test_a_retained_child_keeps_its_group(tmp_path, monkeypatch):
    """terminate_all pops the group before deciding; a pid put back without it
    leaves nothing able to reach a descendant once the leader exits."""
    from utils import process_lifetime as pl

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    pl._tracked_pids.clear()
    pl._tracked_pgids.clear()

    # Unverifiable identity, the transient `ps` failure case.
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
    pl._tracked_pids[4242] = "recorded"
    pl._tracked_pgids[4242] = 4242
    pl.terminate_all(timeout = 1.0)
    assert pl._tracked_pids.get(4242) == "recorded"
    assert pl._tracked_pgids.get(4242) == 4242, "the group was dropped with the retained pid"

    # And a confirmed survivor keeps it too.
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "recorded")
    monkeypatch.setattr(pl, "_identity_or_none", lambda pid: "recorded")
    monkeypatch.setattr(pl, "_posix_terminate", lambda pid, timeout = 5.0: None)
    monkeypatch.setattr(pl, "_windows_terminate_tree", lambda pid: None)
    assert pl.terminate_all(timeout = 1.0) == [4242]
    assert pl._tracked_pgids.get(4242) == 4242


def test_a_worker_record_is_revisited_after_its_owner_dies(tmp_path, monkeypatch):
    """A worker's record is skipped while its owner lives, and that owner can be
    terminated later in the same sweep by the backend's own record."""
    import json

    from utils import process_lifetime as pl

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    directory = tmp_path / "children"
    directory.mkdir(parents = True)

    alive = {9000, 9001}  # 9000 is the worker, 9001 its llama-server
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid in alive)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_identity_or_none", lambda pid: "same" if pid in alive else None)
    monkeypatch.setattr(pl, "_is_windows", lambda: False)

    def terminate(pid, timeout = 5.0):
        alive.discard(pid)

    monkeypatch.setattr(pl, "_posix_terminate", terminate)

    # Sorted lexicographically, so the worker's record is read first.
    (directory / "9000.json").write_text(
        json.dumps(
            {
                "owner_pid": 9000,
                "owner_identity": "same",
                "children": [{"pid": 9001, "identity": "same"}],
            }
        ),
        encoding = "utf-8",
    )
    (directory / "9999.json").write_text(
        json.dumps(
            {
                "owner_pid": 9999,
                "owner_identity": "a-backend-that-crashed",
                "children": [{"pid": 9000, "identity": "same"}],
            }
        ),
        encoding = "utf-8",
    )

    killed = pl.reap_recorded_children(timeout = 1.0)
    assert sorted(killed) == [9000, 9001], killed
    assert list(directory.glob("*.json")) == []


def test_a_group_that_survives_sigkill_keeps_its_record(monkeypatch):
    """Reporting it resolved deletes the last handle on something still holding
    the GPU."""
    from utils import process_lifetime as pl

    if pl._is_windows():
        pytest.skip("POSIX only")

    def killpg(pgid, sig):
        if sig == 0:
            return None  # always still there
        if sig == signal.SIGKILL:
            raise PermissionError("not permitted")

    monkeypatch.setattr(pl.os, "killpg", killpg)
    assert pl._reap_orphaned_group(4242, 4242, timeout = 0.2) is False


def test_a_failed_taskkill_falls_through_to_the_leader(monkeypatch):
    """check=False does not raise, so the status is the only signal."""
    import subprocess as sp

    from utils import process_lifetime as pl

    monkeypatch.setattr(pl.os, "kill", lambda pid, sig: fallbacks.append(pid))
    fallbacks = []

    class _Result:
        returncode = 1

    monkeypatch.setattr(sp, "run", lambda *a, **k: _Result())
    pl._windows_terminate_tree(4242)
    assert fallbacks == [4242], "a failed taskkill was treated as success"

    class _Ok:
        returncode = 0

    fallbacks.clear()
    monkeypatch.setattr(sp, "run", lambda *a, **k: _Ok())
    pl._windows_terminate_tree(4242)
    assert fallbacks == []


def test_a_zombie_owner_does_not_shield_its_children(tmp_path, monkeypatch):
    """os.kill(pid, 0) succeeds for a zombie, and the sweep runs once, so its
    sidecars would survive until another restart."""
    import json

    from utils import process_lifetime as pl

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    directory = tmp_path / "children"
    directory.mkdir(parents = True)

    monkeypatch.setattr(pl, "_is_windows", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: pid == 9500)
    monkeypatch.setattr(pl, "_identity_or_none", lambda pid: "same")
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "same")
    terminated = []
    monkeypatch.setattr(pl, "_posix_terminate", lambda pid, timeout = 5.0: terminated.append(pid))

    (directory / "9500.json").write_text(
        json.dumps(
            {
                "owner_pid": 9500,
                "owner_identity": "same",
                "children": [{"pid": 9501, "identity": "same"}],
            }
        ),
        encoding = "utf-8",
    )

    pl.reap_recorded_children(timeout = 1.0)
    assert terminated == [9501], "a zombie Studio kept its orphans alive"


def test_a_group_that_cannot_be_taken_keeps_its_record(tmp_path, monkeypatch):
    """_reap_orphaned_group returning False has to leave the breadcrumb, or the
    next startup has no handle on the survivors."""
    import json

    from utils import process_lifetime as pl

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    directory = tmp_path / "children"
    directory.mkdir(parents = True)

    monkeypatch.setattr(pl, "_is_windows", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(pl, "_identity_or_none", lambda pid: None)
    monkeypatch.setattr(pl, "_reap_orphaned_group", lambda pgid, pid, timeout: False)
    monkeypatch.setattr(pl, "_group_has_members", lambda pgid: True)

    record = directory / "9600.json"
    record.write_text(
        json.dumps(
            {
                "owner_pid": 9600,
                "owner_identity": "gone",
                "children": [{"pid": 9601, "identity": "same", "pgid": 9601}],
            }
        ),
        encoding = "utf-8",
    )

    pl._reap_one_record(record, timeout = 1.0)
    assert record.is_file(), "the last handle on a live group was deleted"


def test_the_backstop_keeps_a_group_it_could_not_take(monkeypatch):
    from utils import process_lifetime as pl

    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(pl, "_reap_orphaned_group", lambda pgid, pid, timeout: False)
    monkeypatch.setattr(pl, "_group_has_members", lambda pgid: True)
    pl._tracked_pids.clear()
    pl._tracked_pgids.clear()
    pl._tracked_pids[9700] = "same"
    pl._tracked_pgids[9700] = 9700

    assert pl.terminate_all(timeout = 1.0) == [9700]
    assert pl._tracked_pids.get(9700) == "same"
    assert pl._tracked_pgids.get(9700) == 9700


def test_every_child_starting_path_checks_the_rearm():
    """The backend updater is a child too, and a retry re-enters installUpdate."""
    hook = (
        Path(__file__).resolve().parents[2] / "frontend" / "src" / "hooks" / "use-tauri-update.ts"
    )
    source = hook.read_text(encoding = "utf-8")
    for block in source.split("async function ")[1:]:
        name = block.split("(")[0]
        if 'invoke("start_server"' in block or 'invoke("start_backend_update"' in block:
            assert "crashCleanupReady()" in block, f"{name} starts a child without the check"


def test_a_tool_subprocess_is_recorded_while_it_runs(tmp_path, monkeypatch):
    """A force quit mid-call on macOS would otherwise leave a session-leading
    tool with nothing able to find it."""
    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools
    from utils import process_lifetime as pl

    pl._tracked_pids.clear()
    pl._tracked_pgids.clear()
    tools._workdirs.clear()

    seen = []
    real_adopt = pl.adopt_pid

    def watching_adopt(pid):
        seen.append(pid)
        return real_adopt(pid)

    monkeypatch.setattr(pl, "adopt_pid", watching_adopt)
    tools._python_exec("print('hi')", session_id = "__LOCALID_adopt01")
    assert seen, "the tool subprocess was never recorded"
    # And it is not left on the record once it has exited.
    assert all(pid not in pl._tracked_pids for pid in seen), pl._tracked_pids


def test_a_terminated_leader_leaving_a_group_keeps_its_record(tmp_path, monkeypatch):
    """The dead-leader branch covered only a leader that was already gone."""
    import json

    from utils import process_lifetime as pl

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    directory = tmp_path / "children"
    directory.mkdir(parents = True)

    monkeypatch.setattr(pl, "_is_windows", lambda: False)
    # Alive on the way in, gone after the terminate, but its group lives on.
    states = iter([True, True, False, False, False])
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: next(states, False))
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_identity_or_none", lambda pid: "same")
    monkeypatch.setattr(pl, "_posix_terminate", lambda pid, timeout = 5.0: None)
    monkeypatch.setattr(pl, "_group_has_members", lambda pgid: pgid == 9801)

    record = directory / "9800.json"
    record.write_text(
        json.dumps(
            {
                "owner_pid": 9800,
                "owner_identity": "gone",
                "children": [{"pid": 9801, "identity": "same", "pgid": 9801}],
            }
        ),
        encoding = "utf-8",
    )

    pl._reap_one_record(record, timeout = 1.0)
    assert record.is_file(), "the group outlived its leader and lost its record"


def test_cloudflared_is_adopted_before_stop_can_reach_it():
    """Publishing _proc first lets a concurrent stop reap and forget it, and the
    adoption would then record whatever inherited the pid."""
    import inspect

    from cloudflare_tunnel import CloudflareTunnel

    source = inspect.getsource(CloudflareTunnel)
    start = source.index("_adopt_pid(proc.pid)")
    publish = source.index("self._proc = proc")
    assert start < publish, "_proc is published before the adoption"
    # Both inside the same `with self._lock:` block.
    lock = source.rindex("with self._lock:", 0, start)
    assert source.index("self._proc = proc", lock) < source.index("threading.Thread(", lock)


def test_a_group_of_only_zombies_is_not_alive():
    """killpg(pgid, 0) succeeds for a zombie leader, so the plain probe would
    keep a finished group's record forever."""
    from utils import process_lifetime as pl

    if pl._is_windows():
        pytest.skip("POSIX only")

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session = True
    )
    try:
        assert pl._group_member_pids(child.pid) == [child.pid]
        assert pl._group_has_members(child.pid) is True
        child.kill()
        time.sleep(0.3)
        assert pl._pid_is_zombie(child.pid) is True
        assert pl._group_has_members(child.pid) is False
    finally:
        _kill(child.pid)
        child.wait(timeout = 10)


def test_a_zombie_child_is_not_signalled_or_waited_on(tmp_path, monkeypatch):
    """Under a non-reaping PID 1 a zombie answers every probe, so terminating it
    burns the full grace period per record and reaps nothing."""
    import json

    from utils import process_lifetime as pl

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    directory = tmp_path / "children"
    directory.mkdir(parents = True)

    monkeypatch.setattr(pl, "_is_windows", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: pid == 9901)
    monkeypatch.setattr(pl, "_identity_or_none", lambda pid: None)
    monkeypatch.setattr(pl, "_group_has_members", lambda pgid: False)
    terminated = []
    monkeypatch.setattr(pl, "_posix_terminate", lambda pid, timeout = 5.0: terminated.append(pid))

    record = directory / "9900.json"
    record.write_text(
        json.dumps(
            {
                "owner_pid": 9900,
                "owner_identity": "gone",
                "children": [{"pid": 9901, "identity": "same", "pgid": 9901}],
            }
        ),
        encoding = "utf-8",
    )

    started = time.monotonic()
    pl._reap_one_record(record, timeout = 5.0)
    assert terminated == [], "a zombie was signalled"
    assert time.monotonic() - started < 1.0, "startup waited out the grace period"


def test_the_component_installer_is_recorded_while_it_runs():
    """It rewrites files in place, so one surviving a crash overlaps the next
    launch."""
    import ast
    import inspect

    from utils.prebuilt import update_flow

    tree = ast.parse(inspect.getsource(update_flow.stream_installer))
    body = ast.dump(tree)
    assert "adopt_pid" in body, "the installer is never recorded"
    assert "forget_pid" in body, "the record is never cleared"


def test_a_fork_child_does_not_inherit_a_held_record_lock():
    """A fork while another thread was adopting leaves it held here with nobody
    to release it, and the first adoption after that blocks forever."""
    from utils import process_lifetime as pl

    held_spawner = pl._spawner_lock
    held_record = pl._record_lock
    held_record.acquire()
    try:
        pl._reset_after_fork()
        assert pl._record_lock is not held_record, "the child kept the locked record lock"
        assert pl._spawner_lock is not held_spawner
        assert pl._record_lock.acquire(blocking = False), "the fresh lock is not free"
        pl._record_lock.release()
    finally:
        held_record.release()


def test_the_update_hook_asks_the_native_side_after_a_remount():
    """A webview reload rebuilds the hook with its ref back at true while the
    native job can still be disarmed."""
    hook = (
        Path(__file__).resolve().parents[2] / "frontend" / "src" / "hooks" / "use-tauri-update.ts"
    ).read_text(encoding = "utf-8")
    assert "cleanupCheckedRef" in hook
    gate = hook[hook.index("async function crashCleanupReady") :]
    gate = gate[: gate.index("\n  }")]
    assert '"desktop_update_cleanup_armed"' in gate, "the gate still trusts the ref alone"
    assert gate.index("cleanupCheckedRef") < gate.index(
        "if (cleanupRearmedRef.current) return true"
    )

    tauri = Path(__file__).resolve().parents[2] / "src-tauri" / "src"
    assert "desktop_update_cleanup_armed" in (tauri / "main.rs").read_text(encoding = "utf-8")
    assert "kill_on_close_armed" in (tauri / "windows_job.rs").read_text(encoding = "utf-8")


def test_a_transient_identity_failure_is_not_recorded_as_none(monkeypatch):
    """None is permanent: an entry with no identity is never signalled, so that
    child would survive every later launch."""
    from utils import process_lifetime as pl

    answers = iter([None, None, "started-at"])
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: next(answers, "started-at"))
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    assert pl._identity_for_record(os.getpid()) == "started-at"


def test_identity_capture_stops_once_the_child_is_gone(monkeypatch):
    from utils import process_lifetime as pl

    calls = []

    def _identity(pid):
        calls.append(pid)
        return None

    monkeypatch.setattr(pl, "_pid_identity", _identity)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)
    assert pl._identity_for_record(4242) is None
    assert calls == [4242], "kept retrying a process that had already exited"


def test_an_unanswerable_cleanup_query_counts_as_disarmed():
    hook = (
        Path(__file__).resolve().parents[2] / "frontend" / "src" / "hooks" / "use-tauri-update.ts"
    ).read_text(encoding = "utf-8")
    gate = hook[hook.index("async function crashCleanupReady") :]
    gate = gate[: gate.index("\n  }")]
    assert "cleanupRearmedRef.current = !isTauri;" in gate, "a failed query still reads as armed"


def test_the_component_installer_stays_in_the_backend_group():
    """The desktop stop path force-kills this backend's group, so a session of
    its own would let the installer keep rewriting files after the app reports
    the backend stopped."""
    import inspect

    from utils.prebuilt import update_flow

    source = inspect.getsource(update_flow.stream_installer)
    assert "start_new_session = " not in source
    # Still recorded, which is what the macOS sweep has to work from.
    spawn = source.index("subprocess.Popen(")
    assert "adopt_pid(proc.pid)" in source[spawn:]


def test_the_owner_identity_is_retried_and_then_kept(monkeypatch):
    from utils import process_lifetime as pl

    monkeypatch.setattr(pl, "_owner_identity", None)
    answers = iter([None, "started-at"])
    calls = []

    def _identity(pid):
        calls.append(pid)
        return next(answers, "started-at")

    monkeypatch.setattr(pl, "_pid_identity", _identity)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    assert pl._own_identity() == "started-at"
    assert len(calls) == 2, calls
    # Kept, so the record lock never waits on another probe.
    assert pl._own_identity() == "started-at"
    assert len(calls) == 2, calls


def test_a_fork_child_does_not_keep_the_parents_identity(monkeypatch):
    from utils import process_lifetime as pl

    monkeypatch.setattr(pl, "_owner_identity", "the-parents")
    pl._reset_after_fork()
    assert pl._owner_identity is None, "the child would record its parent's identity"


def test_a_fork_child_does_not_claim_the_parents_children(tmp_path, monkeypatch):
    """Its record would name processes it never started, and a later startup
    would reap them while their real parent is still running."""
    from utils import process_lifetime as pl

    monkeypatch.setattr(pl, "_tracked_pids", {4321: "started-at"})
    monkeypatch.setattr(pl, "_tracked_pgids", {4321: 4321})
    pl._reset_after_fork()
    assert pl._tracked_pids == {}
    assert pl._tracked_pgids == {}


def test_a_missing_child_identity_is_filled_in_later(tmp_path, monkeypatch):
    """None is permanent otherwise: nothing signals an entry it cannot verify,
    so that child outlives every shutdown."""
    import json

    from utils import process_lifetime as pl

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    monkeypatch.setattr(pl, "_tracked_pids", {5150: None})
    monkeypatch.setattr(pl, "_tracked_pgids", {})
    monkeypatch.setattr(pl, "_owner_identity", "owner")
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "found-later")

    pl._write_breadcrumb()
    record = pl._breadcrumb_file()
    written = json.loads(record.read_text(encoding = "utf-8"))
    assert written["children"][0]["identity"] == "found-later"
    # Written back, so it costs one probe rather than one per write.
    assert pl._tracked_pids[5150] == "found-later"


def test_an_exited_child_is_not_probed_again(monkeypatch):
    from utils import process_lifetime as pl

    calls = []
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: calls.append(pid))
    assert pl._refreshed_identity(9999, None) is None
    assert calls == []


def test_adoption_installs_the_fork_reset(monkeypatch):
    """The Linux spawn path is the only other place that registers it, so on
    macOS nothing would have."""
    from utils import process_lifetime as pl

    monkeypatch.setattr(pl, "_fork_reset_installed", False)
    registered = []
    monkeypatch.setattr(os, "register_at_fork", lambda **kw: registered.append(kw))
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "started-at")
    monkeypatch.setattr(pl, "_own_process_group", lambda pid: None)
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    monkeypatch.setattr(pl, "_is_windows", lambda: False)

    pl.adopt_pid(os.getpid())
    assert registered and registered[0]["after_in_child"] is pl._reset_after_fork
    pl.forget_pid(os.getpid())


def test_a_group_of_nothing_but_a_zombie_costs_no_grace_period(monkeypatch):
    """killpg(pgid, 0) answers for a zombie, and where pid 1 does not reap it
    stays that way, so every stale record would wait out the timeout."""
    import time as _time

    from utils import process_lifetime as pl

    monkeypatch.setattr(pl, "_is_windows", lambda: False)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(pl, "_group_has_members", lambda pgid: False)

    started = _time.monotonic()
    assert pl._reap_orphaned_group(4711, 4711, timeout = 5.0) is False
    assert _time.monotonic() - started < 1.0, "waited out the grace period"


def test_an_unanswerable_group_query_is_not_an_empty_group(monkeypatch):
    """Reporting empty for a failed ps lets forget_pid drop the only record of
    a live descendant."""
    import subprocess as _subprocess

    from utils import process_lifetime as pl

    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_is_windows", lambda: False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)

    class _Failed:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(_subprocess, "run", lambda *a, **k: _Failed())
    assert pl._group_member_pids(4711) is None
    assert pl._group_has_members(4711) is True, "a failed query read as gone"


def test_the_visual_server_goes_down_with_an_ordinary_stop():
    """The desktop shutdown only waits for this path, so what the shim started
    has to be taken down here rather than by the next startup sweep."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend

    source = inspect.getsource(LlamaCppBackend._kill_process)
    assert "_collect_descendants" in source
    assert "_terminate_descendants" in source
    # Named before the wait: the shim's children are reparented once it exits.
    assert source.index("_collect_descendants") < source.index("self._process.terminate()")
    assert source.index("self._process.terminate()") < source.index("_terminate_descendants")


@pytest.mark.skipif(os.name == "nt", reason = "posix termination path")
def test_an_exited_child_does_not_burn_the_shutdown_timeout():
    """A child that exited but has not been waited on answers signal 0 like a
    live one, so the wait would spend its whole budget on a dead process. With
    several tracked, that is the whole shutdown."""
    from utils.process_lifetime import _pid_is_zombie, _posix_terminate

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        for _ in range(100):
            if _pid_is_zombie(proc.pid):
                break
            time.sleep(0.05)
        assert _pid_is_zombie(proc.pid), "the child never became reapable"

        started = time.monotonic()
        _posix_terminate(proc.pid, timeout = 5.0)
        elapsed = time.monotonic() - started
        assert elapsed < 2.0, f"waited {elapsed:.1f}s on a process that had exited"
        # Still ours to reap: the wait must not consume the exit status the
        # owner is about to read.
        assert proc.wait(timeout = 5) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout = 5)


@pytest.mark.skipif(os.name == "nt", reason = "posix termination path")
def test_a_running_child_is_still_terminated():
    """The shortcut above must not stop the ordinary kill."""
    from utils.process_lifetime import _posix_terminate

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        started = time.monotonic()
        _posix_terminate(proc.pid, timeout = 5.0)
        assert time.monotonic() - started < 5.0
        assert proc.wait(timeout = 5) != 0, "exited on its own, so this proves nothing"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout = 5)


@pytest.mark.skipif(os.name == "nt", reason = "posix process groups")
def test_a_group_is_waited_on_while_a_member_is_still_running():
    """The leader going first must not cut the wait short: the group is what
    holds the GPU, and a live member is the reason for the SIGKILL."""
    from utils.process_lifetime import _posix_terminate

    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time;"
            "subprocess.Popen([sys.executable, '-c', 'import signal, time;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']);"
            "time.sleep(30)",
        ],
        start_new_session = True,
    )
    try:
        time.sleep(1.0)  # let the grandchild exist before anything is signalled
        started = time.monotonic()
        _posix_terminate(leader.pid, timeout = 2.0)
        elapsed = time.monotonic() - started
        assert elapsed >= 1.5, f"gave up after {elapsed:.1f}s with a member still running"
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except Exception:
            pass
        if leader.poll() is None:
            leader.wait(timeout = 5)


@pytest.mark.skipif(os.name == "nt", reason = "posix process groups")
def test_a_live_leader_answers_without_scanning_every_process(monkeypatch):
    """Enumerating a group reads the state of every process on the machine, and
    this runs on each stop; a running leader already settles the question."""
    from utils import process_lifetime as lifetime

    leader = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session = True,
    )
    try:
        for _ in range(50):
            if lifetime._pid_alive(leader.pid):
                break
            time.sleep(0.05)

        def refuse(pgid):
            raise AssertionError("scanned every process for a group whose leader is alive")

        monkeypatch.setattr(lifetime, "_group_member_pids", refuse)
        assert lifetime._group_has_members(leader.pid) is True
    finally:
        leader.kill()
        leader.wait(timeout = 5)


@pytest.mark.skipif(os.name == "nt", reason = "posix process groups")
def test_a_group_outliving_its_leader_is_still_found():
    """The short-circuit must not become the whole answer: the case this record
    exists for is a leader that exited while its child holds the GPU."""
    from utils import process_lifetime as lifetime

    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys;"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']);",
        ],
        start_new_session = True,
    )
    survivor = None
    try:
        leader.wait(timeout = 30)  # the leader goes, its child stays
        members = lifetime._group_member_pids(leader.pid) or []
        survivor = next((pid for pid in members if pid != leader.pid), None)
        assert survivor, members
        assert lifetime._group_has_members(leader.pid) is True
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except Exception:
            pass
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout = 5)


@pytest.mark.skipif(os.name == "nt", reason = "posix process groups")
def test_forgetting_an_untracked_pid_costs_nothing():
    """Every tool call ends here, most of them with nothing recorded: neither
    the group scan nor the record rewrite has anything to do."""
    from utils import process_lifetime as lifetime

    scanned = []
    written = []
    original_scan = lifetime._group_member_pids
    original_write = lifetime._write_breadcrumb
    lifetime._group_member_pids = lambda pgid: (scanned.append(pgid), original_scan(pgid))[1]
    lifetime._write_breadcrumb = lambda: (written.append(1), original_write())[1]
    try:
        lifetime.forget_pid(999_001)
    finally:
        lifetime._group_member_pids = original_scan
        lifetime._write_breadcrumb = original_write

    assert scanned == []
    assert written == [], "rewrote the record for a pid it never held"


def test_a_validation_server_the_installer_started_is_recorded(monkeypatch, tmp_path):
    """It is a grandchild: a parent-death signal reaches the installer only, and
    the record of the installer pid alone never finds the server holding the GPU."""
    from utils.prebuilt import update_flow

    adopted, forgotten = [], []
    monkeypatch.setattr(update_flow, "adopt_pid", lambda pid: adopted.append(pid))
    monkeypatch.setattr(update_flow, "forget_pid", lambda pid: forgotten.append(pid))

    class FakeProc:
        pid = 4321
        stdout = iter(
            [
                "downloading 10% (1/10)\n",
                "UNSLOTH_INSTALLER_CHILD started 9911\n",
                "validating\n",
                "UNSLOTH_INSTALLER_CHILD stopped 9911\n",
                "downloading 100% (10/10)\n",
            ]
        )

        def wait(self):
            return 0

        def poll(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(update_flow.subprocess, "Popen", lambda *a, **k: FakeProc())
    update_flow.stream_installer(
        ["x"],
        {},
        timeout_seconds = 30,
        job = {},
        job_lock = threading.Lock(),
    )

    assert adopted == [4321, 9911], adopted  # the installer, then its server
    # Dropped on the stop line, before the installer's own record goes.
    assert forgotten[0] == 9911, forgotten


def test_a_validation_server_left_running_stays_recorded(monkeypatch):
    """The installer killed before it could report the stop is exactly the case
    the record exists for, so nothing may drop it."""
    from utils.prebuilt import update_flow

    adopted, forgotten = [], []
    monkeypatch.setattr(update_flow, "adopt_pid", lambda pid: adopted.append(pid))
    monkeypatch.setattr(update_flow, "forget_pid", lambda pid: forgotten.append(pid))

    class FakeProc:
        pid = 4321
        stdout = iter(["UNSLOTH_INSTALLER_CHILD started 9912\n"])

        def wait(self):
            return 0

        def poll(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(update_flow.subprocess, "Popen", lambda *a, **k: FakeProc())
    update_flow.stream_installer(
        ["x"],
        {},
        timeout_seconds = 30,
        job = {},
        job_lock = threading.Lock(),
    )

    assert adopted == [4321, 9912], adopted  # the installer, then its server
    assert 9912 not in forgotten, "dropped the only handle on a running server"


def test_the_installer_announces_and_groups_its_validation_server():
    """The other half of the handoff, in the script Studio runs."""
    source = (Path(__file__).resolve().parents[2] / "install_llama_prebuilt.py").read_text(
        encoding = "utf-8"
    )

    assert '_announce_child("started", process.pid)' in source
    assert '_announce_child("stopped", process.pid)' in source
    # Its own group, so a server that starts something of its own is still
    # reachable through the one pid that gets recorded.
    assert '"start_new_session": True' in source
    assert "_terminate_validation_server" in source
    assert "killpg" in source


def test_a_backend_never_starts_under_a_disarmed_job():
    """The UI gate runs per update action, but a webview remount starts a backend
    on its own, so the check belongs where the spawn happens."""
    process_rs = (
        Path(__file__).resolve().parents[2] / "src-tauri" / "src" / "process.rs"
    ).read_text(encoding = "utf-8")

    start = process_rs.index("pub fn start_backend(")
    spawn = process_rs.index("resolve_backend_binary()", start)
    gate = process_rs.index("kill_on_close_armed()", start)
    assert gate < spawn, "the armed check runs after the backend is resolved and spawned"
    assert "resume_after_update_installer()" in process_rs[start:spawn]
    # Fail closed: a re-arm that will not take must stop the start.
    assert "Refusing to start the backend with crash cleanup disarmed" in process_rs


def test_one_malformed_record_does_not_stop_the_whole_sweep(tmp_path, monkeypatch):
    """A record is a file on disk: anything that parses as JSON reaches the
    identity comparison, and the sweep runs inside one try in run_server."""
    from utils import process_lifetime as lifetime

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path))
    victim = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session = True,
    )
    # Live, so its identity is readable and the record's is what gets compared
    # against it. Unverifiable, so it must survive the sweep.
    decoy = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session = True,
    )
    try:
        for index, shape in enumerate((123, ["a"], {"b": 1}, None)):
            (tmp_path / f"bad_{index}.json").write_text(
                json.dumps(
                    {
                        "owner_pid": 999_001,
                        "owner_identity": shape,
                        "children": [{"pid": decoy.pid, "identity": shape, "pgid": None}],
                    }
                ),
                encoding = "utf-8",
            )
        # A real orphan behind them, from an owner that is gone.
        (tmp_path / "good.json").write_text(
            json.dumps(
                {
                    "owner_pid": 999_003,
                    "owner_identity": "1",
                    "children": [
                        {
                            "pid": victim.pid,
                            "identity": lifetime._pid_identity(victim.pid),
                            "pgid": victim.pid,
                        }
                    ],
                }
            ),
            encoding = "utf-8",
        )

        killed = lifetime.reap_recorded_children(timeout = 5.0)
        assert victim.pid in killed, killed
        victim.wait(timeout = 10)
        assert decoy.poll() is None, "signalled a pid it could not verify"
    finally:
        for proc in (victim, decoy):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout = 5)


def test_a_malformed_identity_is_never_treated_as_a_match():
    """Unverifiable has to stay unverifiable: an identity that compares equal to
    anything would signal whatever holds that pid now."""
    from utils.process_lifetime import _recorded_identity, _same_identity

    for shape in (123, ["1"], {"a": 1}, None, ""):
        assert _recorded_identity(shape) is None, shape
        assert _same_identity(shape, "1") is False
        assert _same_identity("1", shape) is False
    assert _recorded_identity("1") == "1"


@pytest.mark.skipif(os.name == "nt", reason = "posix process groups")
def test_a_group_of_zombies_does_not_hold_up_the_startup_sweep(monkeypatch):
    """Where pid 1 does not reap (a container), a member that exits on the
    SIGTERM keeps answering killpg(pgid, 0) for the rest of the grace period."""
    from utils import process_lifetime as lifetime

    leader = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session = True,
    )
    try:
        # As a non-reaping pid 1 leaves things: signalled, exited, still listed.
        alive = {"value": True}
        real_members = lifetime._group_member_pids
        monkeypatch.setattr(
            lifetime,
            "_group_member_pids",
            lambda pgid: real_members(pgid) if alive["value"] else [],
        )
        monkeypatch.setattr(lifetime, "_pid_is_zombie", lambda pid: not alive["value"])

        def fake_killpg(pgid, sig):
            if sig == signal.SIGTERM:
                alive["value"] = False  # it exited, and nobody waited on it
            return None  # the pid is still in the table, so the probe succeeds

        monkeypatch.setattr(lifetime.os, "killpg", fake_killpg)

        started = time.monotonic()
        assert lifetime._reap_orphaned_group(leader.pid, leader.pid, 5.0) is True
        elapsed = time.monotonic() - started
        assert elapsed < 2.0, f"waited {elapsed:.1f}s on a group that had already exited"
    finally:
        leader.kill()
        leader.wait(timeout = 5)


def test_rearming_on_backend_start_also_clears_the_cleanup_guard():
    """Re-enabling the job is only half of it: the pre-exit hook has already run
    its cleanup, and the guard left set makes the next attempt suspend
    kill-on-close without stopping this backend first."""
    process_rs = (
        Path(__file__).resolve().parents[2] / "src-tauri" / "src" / "process.rs"
    ).read_text(encoding = "utf-8")

    start = process_rs.index("pub fn start_backend(")
    spawn = process_rs.index("resolve_backend_binary()", start)
    guard = process_rs[start:spawn]
    assert "resume_after_update_installer()" in guard
    assert "reset_termination_cleanup()" in guard, "re-armed the job but left the guard set"
    assert guard.index("resume_after_update_installer()") < guard.index(
        "reset_termination_cleanup()"
    )


def test_a_tree_taskkill_could_not_take_keeps_its_record(tmp_path, monkeypatch):
    """This fallback runs when the job object was unavailable, so the record is
    the only handle on those workers: reading the dead leader as the tree being
    gone would strand them for good."""
    from utils import process_lifetime as lifetime

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path))
    monkeypatch.setattr(lifetime, "_is_windows", lambda: True)
    monkeypatch.setattr(lifetime, "_identity_or_none", lambda pid: "1")
    monkeypatch.setattr(lifetime, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(lifetime, "_reap_orphaned_group", lambda pgid, pid, timeout: False)
    monkeypatch.setattr(lifetime, "_group_has_members", lambda pgid: False)

    # Alive until it is signalled, gone afterwards: the leader dies, and only
    # taskkill's exit status says whether its workers went with it.
    state = {"leader": True, "tree": False}
    monkeypatch.setattr(lifetime, "_pid_alive", lambda pid: state["leader"])

    def fake_tree(pid):
        state["leader"] = False
        return state["tree"]

    monkeypatch.setattr(lifetime, "_windows_terminate_tree", fake_tree)

    def write_record():
        (tmp_path / "previous.json").write_text(
            json.dumps(
                {
                    "owner_pid": 999_101,
                    "owner_identity": "an-owner-that-is-gone",
                    "children": [{"pid": 999_102, "identity": "1", "pgid": None}],
                }
            ),
            encoding = "utf-8",
        )

    write_record()
    assert lifetime.reap_recorded_children(timeout = 1.0) == [999_102]
    assert (tmp_path / "previous.json").exists(), "dropped the only handle on a live tree"

    # And once taskkill takes the tree, the record is consumed.
    state.update(leader = True, tree = True)
    write_record()
    assert lifetime.reap_recorded_children(timeout = 1.0) == [999_102]
    assert not (tmp_path / "previous.json").exists()


def test_a_failed_tree_kill_keeps_the_pid_in_the_shutdown_record(monkeypatch):
    """terminate_all reports what is still up so the caller can keep it in the
    crash record; a leader-only kill leaves the workers with nothing naming
    them."""
    from utils import process_lifetime as lifetime

    monkeypatch.setattr(lifetime, "_is_windows", lambda: True)
    monkeypatch.setattr(lifetime, "_pid_identity", lambda pid: "1")
    monkeypatch.setattr(lifetime, "_identity_or_none", lambda pid: "1")
    monkeypatch.setattr(lifetime, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(lifetime, "_group_has_members", lambda pgid: False)
    monkeypatch.setattr(lifetime, "_write_breadcrumb", lambda: None)

    state = {"leader": True, "tree": False}
    monkeypatch.setattr(lifetime, "_pid_alive", lambda pid: state["leader"])

    def fake_tree(pid):
        state["leader"] = False
        return state["tree"]

    monkeypatch.setattr(lifetime, "_windows_terminate_tree", fake_tree)

    def track():
        with lifetime._record_lock:
            lifetime._tracked_pids.clear()
            lifetime._tracked_pgids.clear()
            lifetime._tracked_pids[999_201] = "1"

    track()
    assert lifetime.terminate_all(timeout = 1.0) == [999_201]
    assert 999_201 in lifetime._tracked_pids, "stopped tracking a tree that may still be up"

    state.update(leader = True, tree = True)
    track()
    assert lifetime.terminate_all(timeout = 1.0) == []
    assert 999_201 not in lifetime._tracked_pids

    with lifetime._record_lock:
        lifetime._tracked_pids.clear()
        lifetime._tracked_pgids.clear()


@pytest.mark.skipif(os.name == "nt", reason = "posix process groups")
def test_the_installer_waits_for_its_validation_group_not_the_leader(tmp_path):
    """A child that ignores SIGTERM outlives the leader's exit, and announcing
    the stop is what drops the only record of it."""
    installer = _load_installer_module()

    script = tmp_path / "leader.py"
    script.write_text(
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "child = subprocess.Popen([sys.executable, '-c',"
        " 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        " time.sleep(30)'])\n"
        "print(child.pid, flush = True)\n"
        "time.sleep(30)\n",
        encoding = "utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout = subprocess.PIPE,
        text = True,
        start_new_session = True,
    )
    try:
        grandchild = int(proc.stdout.readline().strip())
        assert _alive(grandchild)

        gone = installer._terminate_validation_server(proc)
        assert gone is True, "reported a stop with the group still up"
        time.sleep(0.3)
        assert not _alive(grandchild), "the leader went and its child stayed"
    finally:
        for pid in (proc.pid, locals().get("grandchild")):
            if pid:
                _kill(pid)
        if proc.poll() is None:
            proc.wait(timeout = 5)


@pytest.mark.skipif(os.name == "nt", reason = "posix process groups")
def test_a_validation_group_that_will_not_die_is_not_announced_as_stopped(monkeypatch):
    """False keeps the pid recorded, which is the whole point of announcing it."""
    installer = _load_installer_module()

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session = True,
    )
    try:
        # Signals go nowhere: what an unkillable member looks like from here.
        monkeypatch.setattr(installer.os, "killpg", lambda pgid, sig: None)
        assert installer._terminate_validation_server(proc, grace = 0.2) is False
    finally:
        monkeypatch.undo()
        proc.kill()
        proc.wait(timeout = 5)


def test_an_installer_timeout_takes_its_announced_children_with_it(monkeypatch):
    """This process keeps running after the error, so no startup sweep is coming
    before the retry."""
    from utils.prebuilt import update_flow

    terminated = []
    monkeypatch.setattr(update_flow, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(update_flow, "forget_pid", lambda pid: None)
    monkeypatch.setattr(update_flow, "terminate_pid", lambda pid: terminated.append(pid))

    class FakeProc:
        pid = 4321

        def __init__(self):
            self.killed = False
            self.stdout = self._lines()

        def _lines(self):
            yield "UNSLOTH_INSTALLER_CHILD started 9931\n"
            while not self.killed:
                time.sleep(0.05)

        def wait(self):
            return -9

        def poll(self):
            return -9

        def kill(self):
            self.killed = True

    monkeypatch.setattr(update_flow.subprocess, "Popen", lambda *a, **k: FakeProc())
    with pytest.raises(RuntimeError, match = "timed out"):
        update_flow.stream_installer(
            ["x"],
            {},
            timeout_seconds = 1,
            job = {},
            job_lock = threading.Lock(),
        )

    assert terminated == [9931], terminated


def test_no_process_group_id_is_kept_past_the_group_it_names():
    """A pid is reusable once nothing holds the number as a process group any
    more, so a group id cached across an unload eventually names a stranger."""
    source = (
        Path(__file__).resolve().parents[1] / "core" / "inference" / "llama_cpp.py"
    ).read_text(encoding = "utf-8")

    assert "_diffusion_pgid" not in source, "a kept group id outlives its group"


def test_a_failed_installer_takes_its_announced_children_with_it(monkeypatch):
    """A nonzero exit leaves this process running, and its own live record
    shields those children from a sweep that will not run anyway."""
    from utils.prebuilt import update_flow

    terminated = []
    monkeypatch.setattr(update_flow, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(update_flow, "forget_pid", lambda pid: None)
    monkeypatch.setattr(update_flow, "terminate_pid", lambda pid: terminated.append(pid))

    class FakeProc:
        pid = 4321
        stdout = iter(
            [
                "UNSLOTH_INSTALLER_CHILD started 9941\n",
                "boom\n",
            ]
        )

        def wait(self):
            return 3

        def poll(self):
            return 3

        def kill(self):
            pass

    monkeypatch.setattr(update_flow.subprocess, "Popen", lambda *a, **k: FakeProc())
    with pytest.raises(update_flow.InstallerExit):
        update_flow.stream_installer(
            ["x"],
            {},
            timeout_seconds = 30,
            job = {},
            job_lock = threading.Lock(),
        )

    assert terminated == [9941], terminated


def test_a_server_the_installer_reported_stopped_is_not_killed_twice(monkeypatch):
    """The ordinary path: it said the group was gone, so there is nothing left
    to terminate."""
    from utils.prebuilt import update_flow

    terminated = []
    monkeypatch.setattr(update_flow, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(update_flow, "forget_pid", lambda pid: None)
    monkeypatch.setattr(update_flow, "terminate_pid", lambda pid: terminated.append(pid))

    class FakeProc:
        pid = 4321
        stdout = iter(
            [
                "UNSLOTH_INSTALLER_CHILD started 9942\n",
                "UNSLOTH_INSTALLER_CHILD stopped 9942\n",
            ]
        )

        def wait(self):
            return 0

        def poll(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(update_flow.subprocess, "Popen", lambda *a, **k: FakeProc())
    update_flow.stream_installer(
        ["x"],
        {},
        timeout_seconds = 30,
        job = {},
        job_lock = threading.Lock(),
    )
    assert terminated == []


@pytest.mark.skipif(os.name == "nt", reason = "posix process groups")
def test_terminate_pid_takes_the_group_once_its_leader_has_gone(tmp_path, monkeypatch):
    """The installer announces a server that has already exited while a child of
    its own still holds the GPU. getpgid stops answering for the reaped leader,
    so only the recorded group can still reach that child, and this backend
    keeps running: no sweep is coming to catch it."""
    from utils import process_lifetime as lifetime

    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path))
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys;"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']);",
        ],
        start_new_session = True,
    )
    survivor = None
    try:
        lifetime.adopt_pid(leader.pid)  # while it still leads a group we can read
        assert lifetime._tracked_pgids.get(leader.pid) == leader.pid
        leader.wait(timeout = 30)  # the leader goes and is reaped; its child stays
        members = lifetime._group_member_pids(leader.pid) or []
        survivor = next((pid for pid in members if pid != leader.pid), None)
        assert survivor, members

        lifetime.terminate_pid(leader.pid, timeout = 5.0)
        for _ in range(100):
            if not _alive(survivor):
                break
            time.sleep(0.05)
        assert not _alive(survivor), "left a group member holding the GPU"
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except Exception:
            pass
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout = 5)
        with lifetime._record_lock:
            lifetime._tracked_pids.clear()
            lifetime._tracked_pgids.clear()


def test_terminate_pid_keeps_a_record_taskkill_could_not_confirm(monkeypatch):
    """Windows has no group to fall back on, so a tree kill that failed leaves
    the record as the only thing naming those workers."""
    from utils import process_lifetime as lifetime

    monkeypatch.setattr(lifetime, "_is_windows", lambda: True)
    monkeypatch.setattr(lifetime, "_group_has_members", lambda pgid: False)
    monkeypatch.setattr(lifetime, "_write_breadcrumb", lambda: None)

    state = {"tree": False}
    monkeypatch.setattr(lifetime, "_windows_terminate_tree", lambda pid: state["tree"])

    def track():
        with lifetime._record_lock:
            lifetime._tracked_pids.clear()
            lifetime._tracked_pgids.clear()
            lifetime._tracked_pids[999_301] = "1"

    try:
        track()
        lifetime.terminate_pid(999_301, timeout = 1.0)
        assert 999_301 in lifetime._tracked_pids, "dropped the only handle on a live tree"

        state["tree"] = True  # and a confirmed kill still consumes the record
        track()
        lifetime.terminate_pid(999_301, timeout = 1.0)
        assert 999_301 not in lifetime._tracked_pids
    finally:
        with lifetime._record_lock:
            lifetime._tracked_pids.clear()
            lifetime._tracked_pgids.clear()


def test_announced_children_survive_two_threads_draining_at_once():
    """The watchdog and the reader thread both drain this set, and cancelling a
    timer does not stop a callback that already began. A bare
    `while announced: announced.pop()` raises KeyError out of whichever thread
    loses that race, replacing the installer error the caller should see."""
    import inspect

    from utils.prebuilt import update_flow

    announced = update_flow.AnnouncedChildren()
    assert announced.take() is None, "an empty drain must end the loop, not raise"

    pids = list(range(9000, 9500))
    for pid in pids:
        announced.add(pid)

    taken: list[int] = []
    failures: list[BaseException] = []
    guard = threading.Lock()
    start = threading.Barrier(4)

    def drain():
        try:
            start.wait(timeout = 10)
            while True:
                pid = announced.take()
                if pid is None:
                    return
                with guard:
                    taken.append(pid)
        except BaseException as exc:  # noqa: BLE001 - reported below
            failures.append(exc)

    threads = [threading.Thread(target = drain) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 30)

    assert not failures, failures
    assert sorted(taken) == pids, "a pid was taken twice or lost"

    # And the installer stream drains through it rather than a bare set.
    source = inspect.getsource(update_flow.stream_installer)
    assert "AnnouncedChildren()" in source
    assert "announced.take()" in source


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "PR_SET_PDEATHSIG")
def test_a_validation_server_dies_with_a_killed_installer(tmp_path):
    """The server runs in a session of its own, which takes it out of the group
    the desktop stop path kills, so it arms the parent-death signal instead: a
    SIGKILL mid-validation must not leave it holding the GPU and the staged
    files until some later startup sweeps the breadcrumb."""
    studio_dir = Path(__file__).resolve().parents[2]
    script = tmp_path / "spawn_validation_server.py"
    script.write_text(
        "import importlib.util, os, subprocess, sys, time\n"
        "studio = sys.argv[1]\n"
        "sys.path.insert(0, studio)\n"
        "spec = importlib.util.spec_from_file_location(\n"
        "    'installer_under_test', os.path.join(studio, 'install_llama_prebuilt.py')\n"
        ")\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['installer_under_test'] = module\n"
        "spec.loader.exec_module(module)\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "    **module._validation_server_kwargs(),\n"
        ")\n"
        "print(child.pid, flush = True)\n"
        "time.sleep(120)\n",
        encoding = "utf-8",
    )

    installer = subprocess.Popen(
        [sys.executable, str(script), str(studio_dir)],
        stdout = subprocess.PIPE,
        text = True,
    )
    server_pid = None
    try:
        line = installer.stdout.readline().strip()
        assert line.isdigit(), f"the installer never started a server: {line!r}"
        server_pid = int(line)
        assert _alive(server_pid)

        installer.kill()  # the crash case: no cooperative shutdown runs
        installer.wait(timeout = 10)
        for _ in range(200):
            if not _alive(server_pid):
                break
            time.sleep(0.05)
        assert not _alive(server_pid), "outlived the installer that started it"
    finally:
        if installer.poll() is None:
            installer.kill()
            installer.wait(timeout = 5)
        if server_pid is not None and _alive(server_pid):
            try:
                os.kill(server_pid, signal.SIGKILL)
            except OSError:
                pass


@pytest.mark.skipif(os.name == "nt", reason = "posix termination path")
def test_a_child_in_our_own_group_still_takes_its_children_down(tmp_path):
    """killpg is not available for a child that shares this process's group, so
    without the walk the diffusion shim dies and the visual server keeps the GPU."""
    from utils import process_lifetime as pl

    marker = tmp_path / "grandchild.pid"
    shim = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time, pathlib\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n"
            "time.sleep(60)\n",
        ],
    )  # deliberately in this process's group, as the runner now is
    grandchild = None
    try:
        for _ in range(200):
            if marker.is_file():
                break
            time.sleep(0.05)
        grandchild = int(marker.read_text())
        pl._posix_terminate(shim.pid, timeout = 5.0)
        for _ in range(100):
            if not _alive(grandchild):
                break
            time.sleep(0.05)
        assert not _alive(grandchild), "the visual server survived its runner"
    finally:
        _kill(shim.pid)
        if grandchild is not None:
            _kill(grandchild)


@pytest.mark.skipif(os.name == "nt", reason = "posix identity check")
def test_terminate_pid_leaves_a_pid_that_is_no_longer_ours_alone(monkeypatch):
    """An announced child can exit without the line that clears it, so the pid
    reached here may already belong to someone else's tree."""
    from utils import process_lifetime as pl

    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        monkeypatch.setitem(pl._tracked_pids, stranger.pid, "1")  # an impossible start time
        monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
        pl.terminate_pid(stranger.pid, timeout = 1.0)
        time.sleep(0.5)
        # poll(), not a signal-0 probe: a killed child of this process answers
        # that probe as a zombie until it is waited on.
        assert stranger.poll() is None, "signalled a pid the record does not match"
    finally:
        _kill(stranger.pid)
        stranger.wait(timeout = 5)


@pytest.mark.skipif(os.name == "nt", reason = "posix group handling")
def test_a_validation_server_without_a_death_signal_stays_in_the_group(monkeypatch):
    """macOS has no parent-death signal, and the record only exists once the
    backend has read the announcement that follows the spawn, so a session of
    its own leaves a window in which nothing can reach the server."""
    import importlib

    studio = Path(__file__).resolve().parents[2]
    if str(studio) not in sys.path:
        sys.path.insert(0, str(studio))
    ilp = importlib.import_module("install_llama_prebuilt")

    monkeypatch.setattr(sys, "platform", "darwin")
    assert ilp._validation_server_kwargs() == {}, "no death signal to pair the session with"

    server = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        **ilp._validation_server_kwargs(),
    )
    signalled: list = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: signalled.append((pgid, sig)))
    try:
        ilp._terminate_validation_server(server, grace = 1.0)
        assert signalled == [], f"signalled the group it shares with the installer: {signalled}"
        assert server.poll() is not None, "the server was left running"
    finally:
        monkeypatch.undo()
        _kill(server.pid)
        server.wait(timeout = 5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
