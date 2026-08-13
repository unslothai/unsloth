# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the out-of-process torch allocation probe."""

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from utils import process_lifetime, torch_device_probe


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
    _run_script(monkeypatch, "import ctypes; ctypes.string_at(0)")
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
        "HSA_OVERRIDE_GFX_VERSION",
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
    monkeypatch.setattr(sys, "platform", "win32")
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
    monkeypatch.setattr(sys, "platform", "win32")
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
    monkeypatch.setattr(torch_device_probe.os, "name", "nt" if on_windows else "posix")
    assert torch_device_probe._died_by_signal(returncode) is expected


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
