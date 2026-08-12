# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Isolated device-allocation probe (#8474), every boundary mocked.

The fault this guards against cannot be provoked in CI -- there is no AMD silicon and no
ROCm here -- so the child is faked and the thing under test is the CLASSIFICATION: that a
child killed by a signal, a Windows native fault, any nonzero exit, a timeout and a failed
spawn all come back as "do not allocate in this process", and that a healthy child is the
only thing that comes back ok.
"""

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from utils import device_allocation_probe as probe_mod  # noqa: E402
from utils.device_allocation_probe import (  # noqa: E402
    DeviceAllocationProbeResult,
    describe_exit,
    probe_torch_device_allocation,
)


@pytest.fixture(autouse = True)
def _clear_cache():
    probe_mod._clear_probe_cache()
    yield
    probe_mod._clear_probe_cache()


class _FakeProc:
    """Popen stand-in. ``timeouts`` is how many communicate() calls raise before one
    returns, so a test can drive the terminate -> grace -> kill ladder."""

    def __init__(
        self,
        returncode = 0,
        stderr = "",
        timeouts = 0,
    ):
        self.returncode = returncode
        self.pid = 4242
        self._stderr = stderr
        self._timeouts = timeouts
        self.calls: list[str] = []

    def communicate(self, timeout = None):
        self.calls.append("communicate")
        if self._timeouts > 0:
            self._timeouts -= 1
            raise subprocess.TimeoutExpired(cmd = "probe", timeout = timeout or 0)
        return "ok\n", self._stderr

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")


def _patch_popen(
    monkeypatch,
    proc,
    recorder = None,
):
    def fake_popen(argv, **kwargs):
        if recorder is not None:
            recorder.append((argv, kwargs))
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return recorder


# --- the passing case -------------------------------------------------------------


def test_healthy_child_passes(monkeypatch):
    _patch_popen(monkeypatch, _FakeProc(returncode = 0))
    result = probe_torch_device_allocation("cuda:0")
    assert isinstance(result, DeviceAllocationProbeResult)
    assert result.ok is True
    assert result.device == "cuda:0"
    assert result.returncode == 0
    assert result.reason is None


def test_child_runs_this_interpreter_with_hidden_window_kwargs(monkeypatch):
    from utils.subprocess_compat import windows_hidden_subprocess_kwargs

    calls: list = []
    _patch_popen(monkeypatch, _FakeProc(), calls)
    probe_torch_device_allocation("cuda:1")

    argv, kwargs = calls[0]
    assert argv[0] == sys.executable
    assert argv[1] == "-c"
    assert argv[-2] == "cuda:1"  # the device is passed as argv, never interpolated
    for key, value in windows_hidden_subprocess_kwargs().items():
        assert key in kwargs
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert "shell" not in kwargs


def test_child_allocates_writes_and_synchronizes(monkeypatch):
    calls: list = []
    _patch_popen(monkeypatch, _FakeProc(), calls)
    probe_torch_device_allocation("cuda:0")

    script = calls[0][0][2]
    assert "torch.empty(1, device = device)" in script
    assert "tensor.zero_()" in script  # an allocation that is never written can be elided
    assert "torch.cuda.synchronize" in script  # else an async fault escapes the child


def test_child_env_strips_the_native_path_secret(monkeypatch):
    from utils import native_path_leases

    monkeypatch.setenv(native_path_leases.LEASE_SECRET_ENV, "super-secret")
    calls: list = []
    _patch_popen(monkeypatch, _FakeProc(), calls)
    probe_torch_device_allocation("cuda:0")

    env = calls[0][1]["env"]
    assert native_path_leases.LEASE_SECRET_ENV not in env
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_windows_child_registers_the_rocm_dll_directories(monkeypatch, tmp_path):
    # os.add_dll_directory registrations are process-local, so a fresh child does not
    # inherit main.py's. Without this a healthy Windows AMD GPU cannot even import torch in
    # the child, and the fail-closed verdict would condemn it to CPU for the process.
    rocm_bin = tmp_path / "rocm" / "bin"
    rocm_bin.mkdir(parents = True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HIP_PATH", str(tmp_path / "rocm"))
    calls: list = []
    _patch_popen(monkeypatch, _FakeProc(), calls)

    probe_torch_device_allocation("cuda:0")

    env = calls[0][1]["env"]
    assert str(rocm_bin) in env[probe_mod.ROCM_DLL_DIRS_ENV_VAR].split(os.pathsep)
    script = calls[0][0][2]
    # The registration has to precede the import, not merely be present.
    assert script.index("add_dll_directory") < script.index("import torch")


def test_child_is_tracked_without_a_pre_exec_hook(monkeypatch):
    # preexec_fn would fork this multithreaded server and run Python before exec, where a
    # lock another uvicorn thread held at fork time is still held; the child can hang there
    # while the parent waits inside Popen, before any timeout applies. Python's docs call
    # it unsafe in the presence of threads. Shutdown tracking must not reintroduce it.
    from utils import process_lifetime

    adopted: list = []
    forgotten: list = []
    monkeypatch.setattr(process_lifetime, "adopt_pid", adopted.append)
    monkeypatch.setattr(process_lifetime, "forget_pid", forgotten.append)

    calls: list = []
    proc = _FakeProc()
    _patch_popen(monkeypatch, proc, calls)

    probe_torch_device_allocation("cuda:0")

    assert "preexec_fn" not in calls[0][1]
    assert adopted == [proc.pid]
    assert forgotten == [proc.pid]  # exited cleanly, so the record is released


def test_the_child_bounds_its_own_life(monkeypatch):
    # With no parent-death hook, an orphaned probe has to stop by itself rather than sit on
    # the GPU the next backend will want.
    calls: list = []
    _patch_popen(monkeypatch, _FakeProc(), calls)

    probe_torch_device_allocation("cuda:0")

    argv = calls[0][0]
    assert float(argv[-1]) == probe_mod._CHILD_SELF_LIMIT_SECONDS
    # Past the parent's own deadline, so in a normal run the parent always decides first.
    assert probe_mod._CHILD_SELF_LIMIT_SECONDS > probe_mod.PROBE_TIMEOUT_SECONDS
    script = argv[2]
    assert "os._exit" in script  # a wedged driver will not run a clean shutdown
    assert script.index("threading.Timer") < script.index("import torch")
    # A Timer is a Thread and is NOT daemon by default. Without this the child cannot exit
    # until the watchdog fires, so every probe on every host times out. Caught live.
    assert "daemon = True" in script


def test_the_child_script_exits_promptly(tmp_path):
    # Runs the real child, with torch stubbed out, so the watchdog cannot silently make a
    # healthy probe hang. This is the check that caught the non-daemon Timer.
    stub = tmp_path / "torch.py"
    stub.write_text(
        "class _Dev:\n"
        "    type = 'cpu'\n"
        "class _T:\n"
        "    device = _Dev()\n"
        "    def zero_(self):\n"
        "        pass\n"
        "def empty(*a, **k):\n"
        "    return _T()\n"
    )
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    done = subprocess.run(
        [sys.executable, "-c", probe_mod._CHILD_SCRIPT, "cpu", "300"],
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_a_stuck_child_stays_adopted_until_the_reaper_collects_it(monkeypatch):
    from utils import process_lifetime

    forgotten: list = []
    monkeypatch.setattr(process_lifetime, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(process_lifetime, "forget_pid", forgotten.append)

    proc = _FakeProc(returncode = -int(signal.SIGKILL), timeouts = 3)
    proc.returncode = None  # never exited: still alive after the kill
    reaped = threading.Event()

    def _wait():
        reaped.set()

    proc.wait = _wait  # type: ignore[method-assign]
    _patch_popen(monkeypatch, proc)

    probe_torch_device_allocation("cuda:0")

    assert reaped.wait(timeout = 5)
    # Released by the reaper once it was really gone, not while it was still running.
    for _ in range(50):
        if forgotten:
            break
        threading.Event().wait(0.05)
    assert forgotten == [proc.pid]


def test_rocm_dll_directories_are_empty_off_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HIP_PATH", str(tmp_path))
    assert probe_mod._rocm_dll_directories() == []


def test_non_windows_child_env_has_no_dll_variable(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    calls: list = []
    _patch_popen(monkeypatch, _FakeProc(), calls)
    probe_torch_device_allocation("cuda:0")
    assert probe_mod.ROCM_DLL_DIRS_ENV_VAR not in calls[0][1]["env"]


def test_rocm_dll_directories_prefers_the_newest_version(monkeypatch, tmp_path):
    root = tmp_path / "AMD" / "ROCm"
    for version in ("6.3", "10.0", "7.0"):
        (root / version / "bin").mkdir(parents = True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("HIP_PATH", raising = False)
    monkeypatch.delenv("ROCM_PATH", raising = False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))

    found = probe_mod._rocm_dll_directories()

    # Numeric sort, so 10.0 outranks 7.0 rather than sorting as a string.
    assert [Path(p).parent.name for p in found] == ["10.0", "7.0", "6.3"]


# --- every way it fails closed ---------------------------------------------------


def test_sigsegv_fails_closed(monkeypatch):
    _patch_popen(monkeypatch, _FakeProc(returncode = -int(signal.SIGSEGV)))
    result = probe_torch_device_allocation("cuda:0")
    assert result.ok is False
    assert "SIGSEGV" in result.reason


@pytest.mark.parametrize("returncode", [-int(signal.SIGABRT), -int(signal.SIGILL), -4, -7, -8])
def test_other_hard_faults_fail_closed(monkeypatch, returncode):
    _patch_popen(monkeypatch, _FakeProc(returncode = returncode))
    assert probe_torch_device_allocation("cuda:0").ok is False


def test_windows_unsigned_access_violation_fails_closed(monkeypatch):
    _patch_popen(monkeypatch, _FakeProc(returncode = 0xC0000005))
    result = probe_torch_device_allocation("cuda:0")
    assert result.ok is False
    assert "0xC0000005" in result.reason


def test_windows_signed_access_violation_fails_closed(monkeypatch):
    # The same status handed back as a signed 32-bit int, which is how it can arrive.
    _patch_popen(monkeypatch, _FakeProc(returncode = 0xC0000005 - 0x100000000))
    result = probe_torch_device_allocation("cuda:0")
    assert result.ok is False
    assert "0xC0000005" in result.reason


def test_ordinary_nonzero_exit_fails_closed(monkeypatch):
    # No torch, a bad device string, an ImportError: not a driver fault, but still no
    # evidence this device can allocate, so it must not be used in-process either.
    _patch_popen(monkeypatch, _FakeProc(returncode = 1, stderr = "ModuleNotFoundError: torch"))
    result = probe_torch_device_allocation("cuda:0")
    assert result.ok is False
    assert "exit code 1" in result.reason


def test_timeout_fails_closed_and_is_not_reported_as_a_fault(monkeypatch):
    # One timeout on the real wait, then terminate is enough. The returncode after our
    # own terminate must NOT be described as the driver's doing.
    proc = _FakeProc(returncode = -int(signal.SIGTERM), timeouts = 1)
    _patch_popen(monkeypatch, proc)
    result = probe_torch_device_allocation("cuda:0")
    assert result.ok is False
    assert result.reason == "probe timed out"
    assert "SIGTERM" not in result.reason
    assert "terminate" in proc.calls


def test_timeout_escalates_to_kill_and_always_reaps(monkeypatch):
    proc = _FakeProc(returncode = -int(signal.SIGKILL), timeouts = 3)
    _patch_popen(monkeypatch, proc)
    assert probe_torch_device_allocation("cuda:0").ok is False
    assert proc.calls.count("terminate") == 1
    assert proc.calls.count("kill") == 1
    # terminate drain, kill drain: the child is waited on, never left as a zombie.
    assert proc.calls.count("communicate") == 3


def test_a_child_that_outlives_sigkill_is_still_reaped(monkeypatch):
    # SIGKILL is not delivered while a task sits in an uninterruptible driver wait, which
    # is exactly the wedged-GPU case this module exists to survive. Dropping the last
    # reference there would leave an unreaped stray, so it is handed to a reaper instead.
    proc = _FakeProc(returncode = -int(signal.SIGKILL), timeouts = 3)
    waited = threading.Event()
    proc.wait = lambda: waited.set()  # type: ignore[method-assign]
    _patch_popen(monkeypatch, proc)

    assert probe_torch_device_allocation("cuda:0").ok is False
    assert waited.wait(timeout = 5), "abandoned child was never waited on"


def test_a_child_that_dies_within_the_grace_needs_no_reaper(monkeypatch):
    proc = _FakeProc(returncode = -int(signal.SIGTERM), timeouts = 1)
    before = threading.active_count()
    _patch_popen(monkeypatch, proc)

    probe_torch_device_allocation("cuda:0")

    assert "kill" not in proc.calls
    assert threading.active_count() <= before


def test_a_read_failure_during_teardown_does_not_escape(monkeypatch):
    # The first wait times out, so teardown starts; the post-kill read then fails. That
    # error is raised inside the timeout handling, so it needs the same treatment, and the
    # child is not confirmed dead, so it still has to reach the reaper.
    proc = _FakeProc(returncode = None, timeouts = 1)
    reaped = threading.Event()
    proc.wait = lambda: reaped.set()  # type: ignore[method-assign]
    original = proc.communicate

    def _fail_after_first(timeout = None):
        try:
            return original(timeout = timeout)
        except subprocess.TimeoutExpired:
            raise
        finally:
            proc.communicate = _boom  # type: ignore[method-assign]

    def _boom(timeout = None):
        proc.calls.append("communicate")
        raise OSError("handle is invalid")

    proc.communicate = _fail_after_first  # type: ignore[method-assign]
    _patch_popen(monkeypatch, proc)

    result = probe_torch_device_allocation("cuda:0")

    assert result.ok is False
    assert result.reason == "probe timed out"
    assert reaped.wait(timeout = 5), "unconfirmed child was never handed to the reaper"


def test_a_read_failure_fails_closed_and_tears_the_child_down(monkeypatch):
    # A Windows pipe/handle failure, or a read error under resource pressure. This function
    # promises never to raise and to treat every unknown outcome as unsafe, and the child
    # may still be running, so it has to be torn down rather than left to its watchdog.
    proc = _FakeProc()

    def _boom(timeout = None):
        proc.calls.append("communicate")
        raise OSError("handle is invalid")

    proc.communicate = _boom  # type: ignore[method-assign]
    _patch_popen(monkeypatch, proc)

    result = probe_torch_device_allocation("cuda:0")

    assert result.ok is False
    assert "could not be read" in result.reason
    assert "terminate" in proc.calls


def test_spawn_failure_fails_closed_without_raising(monkeypatch):
    def boom(argv, **kwargs):
        raise OSError("no fork headroom")

    monkeypatch.setattr(subprocess, "Popen", boom)
    result = probe_torch_device_allocation("cuda:0")
    assert result.ok is False
    assert result.returncode is None
    assert "could not start" in result.reason


# --- memoization ------------------------------------------------------------------


def test_result_is_cached_per_process(monkeypatch):
    calls: list = []
    _patch_popen(monkeypatch, _FakeProc(), calls)
    first = probe_torch_device_allocation("cuda:0")
    second = probe_torch_device_allocation("cuda:0")
    assert first is second
    assert len(calls) == 1


def test_each_device_is_probed_separately(monkeypatch):
    calls: list = []
    _patch_popen(monkeypatch, _FakeProc(), calls)
    probe_torch_device_allocation("cuda:0")
    probe_torch_device_allocation("cuda:1")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "var",
    [
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
    ],
)
def test_device_identity_env_change_invalidates_the_verdict(monkeypatch, var):
    # HSA_OVERRIDE_GFX_VERSION is the spoof behind #7331: the same "cuda:0" can mean
    # different kernels before and after it changes, so a cached verdict cannot stand.
    calls: list = []
    _patch_popen(monkeypatch, _FakeProc(), calls)
    monkeypatch.delenv(var, raising = False)
    probe_torch_device_allocation("cuda:0")
    monkeypatch.setenv(var, "11.0.0")
    probe_torch_device_allocation("cuda:0")
    assert len(calls) == 2


def test_unrelated_env_change_does_not_invalidate_the_verdict(monkeypatch):
    calls: list = []
    _patch_popen(monkeypatch, _FakeProc(), calls)
    probe_torch_device_allocation("cuda:0")
    monkeypatch.setenv("SOME_UNRELATED_VARIABLE", "1")
    probe_torch_device_allocation("cuda:0")
    assert len(calls) == 1


def test_no_verdict_survives_the_process(monkeypatch, tmp_path):
    # A cached negative that outlived a driver repair would pin a healthy host to CPU with
    # no way to tell why, so the memo is process-local. Asserted as the property that
    # matters: a fresh process re-probes. (The studio home does gain a child-lifetime
    # record here, which is the reaper's, not a verdict.)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    calls: list = []
    _patch_popen(monkeypatch, _FakeProc(returncode = -int(signal.SIGSEGV)), calls)

    probe_torch_device_allocation("cuda:0")
    probe_mod._clear_probe_cache()  # stands in for the next process starting cold
    probe_torch_device_allocation("cuda:0")

    assert len(calls) == 2
    assert not list(tmp_path.rglob("*probe*"))


# --- exit description -------------------------------------------------------------


def test_describe_exit_is_quiet_about_success():
    assert describe_exit(0) is None
    assert describe_exit(None) is None


def test_describe_exit_names_the_signal():
    assert "SIGSEGV" in describe_exit(-int(signal.SIGSEGV))


def test_parent_side_never_imports_torch():
    # This module is imported by the RAG embedder, which is deliberately torch-optional
    # and runs in the lean main process (see tests/test_startup_defers_torch.py). Walk the
    # AST rather than the text: the child script is a string literal here and the prose
    # around it names the import it is careful to place correctly.
    import ast

    source = (_BACKEND_DIR / "utils" / "device_allocation_probe.py").read_text(encoding = "utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported
