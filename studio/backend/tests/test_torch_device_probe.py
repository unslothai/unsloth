# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the out-of-process torch allocation probe."""

import ast
import os
import signal
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

from utils import process_lifetime, torch_device_probe


def _spoof_os_name(monkeypatch, name: str) -> None:
    """Give the probe a private ``os`` whose ``name`` is spoofed.

    The probe branches on ``os.name`` for every status interpretation, but the
    real ``os`` module is shared with pytest's terminal reporter and the xdist
    transport, which keep reading it while a test runs. Mutating it in-process
    flipped pathlib to Windows flavour inside pytest's own report formatting
    (``ValueError: ... is not in the subpath of '\\repo\\...'``) and took the
    worker down, killing or wedging the whole run. A module-private copy keeps
    the spoof scoped to the code under test.
    """
    fake_os = types.ModuleType(os.__name__)
    fake_os.__dict__.update(os.__dict__)
    fake_os.name = name
    monkeypatch.setattr(torch_device_probe, "os", fake_os)


def _spoof_sys_platform(monkeypatch, platform: str) -> None:
    """Give the probe a private copy of ``sys`` with the platform spoofed.

    Same reason as ``_spoof_os_name``: the real ``sys`` is shared with pytest
    and the xdist transport; on Windows the spoof also defeats the
    ``sys.platform == "win32"`` guard in process_lifetime's liveness probe.
    """
    fake_sys = types.ModuleType(sys.__name__)
    fake_sys.__dict__.update(sys.__dict__)
    fake_sys.platform = platform
    monkeypatch.setattr(torch_device_probe, "sys", fake_sys)


# A child that dies of SIGSEGV is still handed to the host's core_pattern handler
# (apport on Ubuntu), which reads the whole core before the child is reaped: a
# multi-MB write and roughly 4x the wall time per fault, on every run of this suite.
# Marking the child non-dumpable first keeps the SIGSEGV the test needs and writes
# no core. RLIMIT_CORE = 0 does NOT work here, because a piped core_pattern ignores
# it; PR_SET_DUMPABLE is the only thing that suppresses the dump. prctl is
# Linux-only, so the call is guarded and simply does nothing elsewhere.
_SUPPRESS_CORE = (
    "import ctypes\n"
    "try:\n"
    "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 0\n"
    "except Exception:\n"
    "    pass\n"
)
_CRASHING_SCRIPT = _SUPPRESS_CORE + "ctypes.string_at(0)\n"


@pytest.fixture(autouse = True)
def _fresh_probe(monkeypatch):
    monkeypatch.setenv(torch_device_probe.DISABLE_ENV_VAR, "0")
    monkeypatch.setattr(process_lifetime, "adopt_pid", lambda _pid: None)
    monkeypatch.setattr(process_lifetime, "forget_pid", lambda _pid: None)
    torch_device_probe.device_can_allocate.cache_clear()
    yield
    torch_device_probe.device_can_allocate.cache_clear()


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode = 0,
        stderr = "",
        timeouts = 0,
    ):
        self.returncode = returncode
        self.pid = 4242
        self.stderr = stderr
        self.timeouts = timeouts
        self.calls: list[str] = []

    def communicate(self, timeout = None):
        self.calls.append("communicate")
        if self.timeouts:
            self.timeouts -= 1
            raise subprocess.TimeoutExpired("probe", timeout or 0)
        return None, self.stderr

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")

    def wait(self):
        self.calls.append("wait")


def _patch_popen(
    monkeypatch,
    process,
    calls = None,
):
    def _popen(argv, **kwargs):
        if calls is not None:
            calls.append((argv, kwargs))
        return process

    monkeypatch.setattr(subprocess, "Popen", _popen)


def _run_script(monkeypatch, script):
    monkeypatch.setattr(torch_device_probe, "_PROBE_SCRIPT", script)


def test_probe_script_is_valid_python():
    compile(torch_device_probe._PROBE_SCRIPT, "<probe>", "exec")


def test_probe_script_initializes_blas_and_synchronizes():
    script = torch_device_probe._PROBE_SCRIPT
    assert "torch.ones" in script
    assert "tensor @ tensor" in script
    assert ".item()" in script


def test_child_that_crashes_marks_the_device_unusable(monkeypatch):
    _run_script(monkeypatch, _CRASHING_SCRIPT)
    assert torch_device_probe.device_can_allocate("cuda") is False


def test_clean_child_marks_the_device_usable(monkeypatch):
    _run_script(monkeypatch, "pass")
    assert torch_device_probe.device_can_allocate("cuda") is True


def test_child_raising_an_exception_does_not_condemn_the_device(monkeypatch):
    _run_script(monkeypatch, "raise RuntimeError('no torch here')")
    assert torch_device_probe.device_can_allocate("cuda") is True


def test_hung_child_marks_the_device_unusable(monkeypatch):
    _run_script(monkeypatch, "import time; time.sleep(30)")
    monkeypatch.setattr(torch_device_probe, "PROBE_TIMEOUT_SECONDS", 1.0)
    assert torch_device_probe.device_can_allocate("cuda") is False


def test_unspawnable_probe_does_not_claim_the_accelerator_works(monkeypatch):
    # A probe that never ran proves nothing, and the two ways of being wrong are not
    # symmetric: CPU costs embedding speed, the accelerator costs the backend.
    def _no_spawn(*_args, **_kwargs):
        raise OSError("fork failed")

    monkeypatch.setattr(subprocess, "Popen", _no_spawn)
    assert torch_device_probe.device_can_allocate("cuda") is False


def test_an_unrunnable_probe_still_leaves_cpu_available(monkeypatch):
    # The opposite trade for CPU: it cannot fault a GPU driver, so a probe that never ran
    # says nothing against it. Condemning it would push the caller past its CPU fallback
    # to the GGUF backend, changing the embedding space over a passing failure to fork.
    def _no_spawn(*_args, **_kwargs):
        raise OSError("fork failed")

    monkeypatch.setattr(subprocess, "Popen", _no_spawn)
    assert torch_device_probe.device_can_allocate("cpu") is True


def test_unreadable_probe_result_cleans_up_and_does_not_claim_the_device_works(monkeypatch):
    process = _FakeProcess()

    def _broken_communicate(timeout = None):
        raise OSError("pipe failed")

    process.communicate = _broken_communicate
    _patch_popen(monkeypatch, process)

    assert torch_device_probe.device_can_allocate("cuda") is False
    assert "terminate" in process.calls


def test_a_read_failure_during_teardown_does_not_escape(monkeypatch):
    # The post-kill read used to sit inside the timeout branch, where the trailing
    # except OSError was a sibling and could not catch it. A pipe failure there escaped
    # device_can_allocate, so a device that really did time out raised instead of
    # returning False, and the child never reached the reaper.
    process = _FakeProcess(returncode = None, timeouts = 1)
    reaped = threading.Event()
    process.wait = lambda: reaped.set()
    original = process.communicate

    def _fail_after_first(timeout = None):
        try:
            return original(timeout = timeout)
        finally:
            process.communicate = _boom

    def _boom(timeout = None):
        process.calls.append("communicate")
        raise OSError("pipe failed")

    process.communicate = _fail_after_first
    _patch_popen(monkeypatch, process)

    assert torch_device_probe.device_can_allocate("cuda") is False
    assert reaped.wait(timeout = 5), "unconfirmed child was never handed to the reaper"


def test_result_is_cached_per_device(monkeypatch):
    calls: list = []
    _patch_popen(monkeypatch, _FakeProcess(), calls)

    assert torch_device_probe.device_can_allocate("cuda") is True
    assert torch_device_probe.device_can_allocate("cuda") is True
    assert torch_device_probe.device_can_allocate("cpu") is True
    assert [call[0][-2] for call in calls] == ["cuda", "cpu"]


@pytest.mark.parametrize(
    "variable",
    [
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_OVERRIDE_GFX_VERSION",
        # _TORCH_DEVICE maps DeviceType.XPU to "xpu", so the probe runs on Intel too and
        # its selectors move the silicon underneath a cached verdict just as the rest do.
        "ZE_AFFINITY_MASK",
        "ONEAPI_DEVICE_SELECTOR",
    ],
)
def test_device_identity_change_invalidates_cache(monkeypatch, variable):
    calls: list = []
    _patch_popen(monkeypatch, _FakeProcess(), calls)
    monkeypatch.delenv(variable, raising = False)

    assert torch_device_probe.device_can_allocate("cuda") is True
    monkeypatch.setenv(variable, "changed")
    assert torch_device_probe.device_can_allocate("cuda") is True
    assert len(calls) == 2


def test_every_visibility_mask_hardware_honours_is_part_of_the_cache_key():
    """The two lists have to move together, or a cached verdict outlives its device.

    hardware.py decides whether a visibility mask is filtering the device set. Any variable
    it counts there renames the silicon behind "cuda", so a verdict cached before the change
    would describe a GPU that is no longer the one being asked about. Read out of the source
    rather than imported, since that module reaches for torch.
    """
    source = Path(torch_device_probe.__file__).with_name("hardware") / "hardware.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    masks = {
        node.value
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef) and function.name == "_rocm_visibility_mask_active"
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isupper()
    }

    assert masks, "hardware._rocm_visibility_mask_active no longer lists its variables"
    assert masks <= set(torch_device_probe._DEVICE_IDENTITY_ENV_VARS)


def test_disable_env_var_skips_the_child(monkeypatch):
    def _no_spawn(*_args, **_kwargs):
        raise AssertionError("probe spawned despite opt-out")

    monkeypatch.setenv(torch_device_probe.DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(subprocess, "Popen", _no_spawn)
    assert torch_device_probe.device_can_allocate("cuda") is True


def test_child_uses_selected_device_without_preexec(monkeypatch):
    calls: list = []
    _patch_popen(monkeypatch, _FakeProcess(), calls)

    assert torch_device_probe.device_can_allocate("xpu") is True
    argv, kwargs = calls[0]
    assert argv[0] == sys.executable
    assert argv[-2] == "xpu"
    assert "preexec_fn" not in kwargs


def test_a_child_that_hit_its_own_deadline_is_a_failed_probe(monkeypatch):
    # A child that stopped itself hung, and a hang is a device failure. Neither form was
    # recognised before: SIGALRM is not a hard fault so it fell through _died_by_signal,
    # and the Windows status is an ordinary non-zero exit. Both read as a healthy device,
    # which let the parent make the allocation the probe stands in front of.
    _spoof_os_name(monkeypatch, "posix")
    _patch_popen(monkeypatch, _FakeProcess(returncode = -torch_device_probe._SIGALRM_NUMBER))
    assert torch_device_probe.device_can_allocate("cuda") is False


def test_the_windows_watchdog_status_is_a_failed_probe(monkeypatch):
    _spoof_os_name(monkeypatch, "nt")
    _patch_popen(monkeypatch, _FakeProcess(returncode = torch_device_probe._WATCHDOG_EXIT_STATUS))
    assert torch_device_probe.device_can_allocate("cuda") is False


def test_a_windows_crt_abort_is_a_failed_probe(monkeypatch):
    # A native abort() on Windows leaves plain exit status 3, not an NTSTATUS, so nothing
    # else here recognises it and the crashing device was being reported as usable.
    _spoof_os_name(monkeypatch, "nt")
    _patch_popen(
        monkeypatch,
        _FakeProcess(returncode = torch_device_probe._WINDOWS_ABORT_EXIT_STATUS),
    )
    assert torch_device_probe.device_can_allocate("cuda") is False


def test_the_abort_status_is_read_as_a_crash_only_on_windows(monkeypatch):
    # Elsewhere 3 is just an exit status a child chose, and an abort arrives as SIGABRT.
    _spoof_os_name(monkeypatch, "posix")
    assert (
        torch_device_probe._died_by_signal(torch_device_probe._WINDOWS_ABORT_EXIT_STATUS) is False
    )


def test_the_abort_status_matches_the_one_llama_cpp_already_uses():
    # Same CRT convention, two readers; a divergence here would be silent.
    source = Path(torch_device_probe.__file__).parents[1] / "core" / "inference" / "llama_cpp.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    (function,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_is_abort_exit"
    ]
    statuses = {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    }
    assert statuses == {torch_device_probe._WINDOWS_ABORT_EXIT_STATUS}


def test_the_windows_watchdog_uses_the_status_the_parent_looks_for(monkeypatch):
    # The child writes the number and the parent matches on the constant; they have to agree.
    assert (
        f"os._exit({torch_device_probe._WATCHDOG_EXIT_STATUS})" in torch_device_probe._PROBE_SCRIPT
    )


def test_the_child_deadline_does_not_depend_on_the_gil(monkeypatch):
    # The deadline exists for a torch that hangs in a native call, and that is exactly when
    # a threading.Timer cannot fire: its callback needs the GIL, which a long C call never
    # returns to the interpreter to release. SIGALRM with no handler is enforced by the
    # kernel, so it runs no Python at all.
    script = torch_device_probe._PROBE_SCRIPT
    assert "signal.alarm" in script
    assert script.index("signal.alarm") < script.index("import torch")
    # Windows has no alarm, so the timer stays as the fallback there.
    assert "threading.Timer" in script


def test_the_kernel_enforces_the_child_deadline():
    # Proves the mechanism rather than trusting it: no handler is installed, so the default
    # disposition terminates the process, and the exit is the signal itself.
    if not hasattr(signal, "alarm"):
        pytest.skip("POSIX only")
    done = subprocess.run(
        [sys.executable, "-c", "import signal, time; signal.alarm(1); time.sleep(30)"],
        capture_output = True,
        timeout = 60,
    )
    assert done.returncode == -int(signal.SIGALRM)


def test_an_inherited_sigalrm_disposition_cannot_disarm_the_deadline():
    # exec keeps an inherited SIG_IGN and an inherited blocked mask, so a supervisor that
    # ignores or blocks SIGALRM would leave the deadline unenforceable and an orphaned probe
    # running against a hung driver forever. The child restores the disposition itself.
    if not hasattr(signal, "alarm"):
        pytest.skip("POSIX only")
    prologue = torch_device_probe._PROBE_SCRIPT.split("if sys.platform")[0]
    child = prologue + "\nimport time\ntime.sleep(30)\n"
    hostile = (
        "import os, signal, sys\n"
        "signal.signal(signal.SIGALRM, signal.SIG_IGN)\n"
        "signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})\n"
        "os.execv(sys.executable, [sys.executable, '-c', sys.argv[1], 'cpu', '1'])\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", hostile, child],
        capture_output = True,
        timeout = 60,
    )
    assert done.returncode == -int(signal.SIGALRM)


def test_child_has_its_own_deadline(monkeypatch):
    calls: list = []
    _patch_popen(monkeypatch, _FakeProcess(), calls)

    torch_device_probe.device_can_allocate("cuda")
    argv = calls[0][0]
    assert float(argv[-1]) > torch_device_probe.PROBE_TIMEOUT_SECONDS
    script = argv[2]
    assert script.index("threading.Timer") < script.index("import torch")
    assert "daemon = True" in script
    assert "os._exit" in script


def test_child_is_tracked_until_it_exits(monkeypatch):
    adopted: list[int] = []
    forgotten: list[int] = []
    monkeypatch.setattr(process_lifetime, "adopt_pid", adopted.append)
    monkeypatch.setattr(process_lifetime, "forget_pid", forgotten.append)
    process = _FakeProcess()
    _patch_popen(monkeypatch, process)

    torch_device_probe.device_can_allocate("cuda")
    assert adopted == [process.pid]
    assert forgotten == [process.pid]


def test_timeout_escalates_and_reaps_a_survivor(monkeypatch):
    forgotten: list[int] = []
    monkeypatch.setattr(process_lifetime, "forget_pid", forgotten.append)
    process = _FakeProcess(returncode = None, timeouts = 3)
    reaped = threading.Event()

    def _wait():
        process.calls.append("wait")
        reaped.set()

    process.wait = _wait
    _patch_popen(monkeypatch, process)

    assert torch_device_probe.device_can_allocate("cuda") is False
    assert process.calls[:5] == ["communicate", "terminate", "communicate", "kill", "communicate"]
    assert reaped.wait(timeout = 5)
    for _ in range(50):
        if forgotten:
            break
        threading.Event().wait(0.02)
    assert forgotten == [process.pid]


def test_timeout_that_terminates_does_not_kill(monkeypatch):
    process = _FakeProcess(returncode = -int(signal.SIGTERM), timeouts = 1)
    _patch_popen(monkeypatch, process)

    assert torch_device_probe.device_can_allocate("cuda") is False
    assert "terminate" in process.calls
    assert "kill" not in process.calls


def test_windows_child_registers_rocm_dll_directories_before_torch(monkeypatch, tmp_path):
    rocm_bin = tmp_path / "rocm" / "bin"
    rocm_bin.mkdir(parents = True)
    _spoof_sys_platform(monkeypatch, "win32")
    monkeypatch.setenv("HIP_PATH", str(tmp_path / "rocm"))
    calls: list = []
    _patch_popen(monkeypatch, _FakeProcess(), calls)

    torch_device_probe.device_can_allocate("cuda")
    argv, kwargs = calls[0]
    assert str(rocm_bin) in kwargs["env"][torch_device_probe.ROCM_DLL_DIRS_ENV_VAR]
    assert argv[2].index("add_dll_directory") < argv[2].index("import torch")


def test_windows_rocm_directories_use_numeric_version_order(monkeypatch, tmp_path):
    for version in ("6.3", "10.0", "7.0"):
        (tmp_path / "AMD" / "ROCm" / version / "bin").mkdir(parents = True)
    _spoof_sys_platform(monkeypatch, "win32")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("HIP_PATH", raising = False)
    monkeypatch.delenv("ROCM_PATH", raising = False)

    found = torch_device_probe._rocm_dll_directories()
    assert [Path(path).parent.name for path in found] == ["10.0", "7.0", "6.3"]


@pytest.mark.parametrize(
    "returncode, on_windows, expected",
    [
        (-11, False, True),
        (-6, False, True),
        # Something else killed the probe; that says nothing about the device. The repo
        # makes the same exclusion in LlamaCppBackend._is_signal_crash.
        (-9, False, False),
        (-15, False, False),
        (-2, False, False),
        (0, False, False),
        (1, False, False),
        (3221225477, True, True),
        (3221226505, True, True),
        (1, True, False),
        (3221225477, False, False),
    ],
)
def test_died_by_signal(monkeypatch, returncode, on_windows, expected):
    _spoof_os_name(monkeypatch, "nt" if on_windows else "posix")
    assert torch_device_probe._died_by_signal(returncode) is expected


@pytest.mark.parametrize("killer", [9, 15, 1])
def test_a_killed_probe_is_not_a_pass_for_an_accelerator(monkeypatch, killer):
    # Not a hard fault, so it is no evidence against the device, but it is not the clean
    # run that earns a pass either. Importing torch and building its device context is
    # enough to trip a cgroup OOM on its own, and reading that as a pass sends the caller
    # on to a much larger load in this process.
    _spoof_os_name(monkeypatch, "posix")
    _patch_popen(monkeypatch, _FakeProcess(returncode = -killer))
    assert torch_device_probe.device_can_allocate("cuda") is False


def test_a_killed_probe_still_leaves_cpu_available(monkeypatch):
    # Same no-verdict trade as a probe that never ran: CPU cannot fault a GPU driver, and
    # condemning it would push the caller past its CPU fallback to a different backend.
    _spoof_os_name(monkeypatch, "posix")
    _patch_popen(monkeypatch, _FakeProcess(returncode = -9))
    assert torch_device_probe.device_can_allocate("cpu") is True


def test_a_clean_child_is_still_a_pass(monkeypatch):
    _spoof_os_name(monkeypatch, "posix")
    _patch_popen(monkeypatch, _FakeProcess(returncode = 0))
    assert torch_device_probe.device_can_allocate("cuda") is True


def test_real_torch_allocates_on_cpu():
    pytest.importorskip("torch")
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            torch_device_probe._PROBE_SCRIPT,
            "cpu",
            str(torch_device_probe._CHILD_SELF_LIMIT_SECONDS),
        ],
        capture_output = True,
        timeout = torch_device_probe.PROBE_TIMEOUT_SECONDS,
    )
    assert probe.returncode == 0, probe.stderr
