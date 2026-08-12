# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the out-of-process torch allocation probe.

Crash cases use real access violations because driver faults bypass exceptions.
"""

import subprocess
import sys

import pytest

from utils import torch_device_probe


@pytest.fixture(autouse = True)
def _fresh_probe(monkeypatch):
    monkeypatch.setenv(torch_device_probe.DISABLE_ENV_VAR, "0")
    torch_device_probe.device_can_allocate.cache_clear()
    yield
    torch_device_probe.device_can_allocate.cache_clear()


def _run_script(monkeypatch, script):
    monkeypatch.setattr(torch_device_probe, "_PROBE_SCRIPT", script)


def test_probe_script_is_valid_python():
    compile(torch_device_probe._PROBE_SCRIPT, "<probe>", "exec")


def test_child_that_crashes_marks_the_device_unusable(monkeypatch):
    _run_script(monkeypatch, "import ctypes; ctypes.string_at(0)")
    assert torch_device_probe.device_can_allocate("cuda") is False


def test_clean_child_marks_the_device_usable(monkeypatch):
    _run_script(monkeypatch, "pass")
    assert torch_device_probe.device_can_allocate("cuda") is True


def test_child_raising_an_exception_does_not_condemn_the_device(monkeypatch):
    # Let the in-process loader report ordinary Python errors with full context.
    _run_script(monkeypatch, "raise RuntimeError('no torch here')")
    assert torch_device_probe.device_can_allocate("cuda") is True


def test_hung_child_marks_the_device_unusable(monkeypatch):
    _run_script(monkeypatch, "import time; time.sleep(30)")
    monkeypatch.setattr(torch_device_probe, "PROBE_TIMEOUT_SECONDS", 1.0)
    assert torch_device_probe.device_can_allocate("cuda") is False


def test_unspawnable_probe_does_not_condemn_the_device(monkeypatch):
    def _no_spawn(*_a, **_k):
        raise OSError("fork failed")

    monkeypatch.setattr(subprocess, "run", _no_spawn)
    assert torch_device_probe.device_can_allocate("cuda") is True


def test_result_is_cached_per_device(monkeypatch):
    spawns = []
    real_run = subprocess.run

    def _counting_run(argv, **kwargs):
        spawns.append(argv[-1])
        return real_run(argv, **kwargs)

    _run_script(monkeypatch, "pass")
    monkeypatch.setattr(subprocess, "run", _counting_run)

    assert torch_device_probe.device_can_allocate("cuda") is True
    assert torch_device_probe.device_can_allocate("cuda") is True
    assert torch_device_probe.device_can_allocate("cpu") is True
    assert spawns == ["cuda", "cpu"]


def test_disable_env_var_skips_the_child(monkeypatch):
    def _no_spawn(*_a, **_k):
        raise AssertionError("probe spawned a child despite the opt-out")

    monkeypatch.setenv(torch_device_probe.DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(subprocess, "run", _no_spawn)
    assert torch_device_probe.device_can_allocate("cuda") is True


def test_device_is_passed_to_the_child(monkeypatch):
    _run_script(monkeypatch, "import sys; sys.exit(0 if sys.argv[1] == 'xpu' else 7)")
    assert torch_device_probe.device_can_allocate("xpu") is True


@pytest.mark.parametrize(
    "returncode, on_windows, expected",
    [
        (-11, False, True),  # SIGSEGV
        (-6, False, True),  # SIGABRT
        (0, False, False),
        (1, False, False),  # ordinary Python failure
        (3221225477, True, True),  # 0xC0000005 access violation
        (3221226505, True, True),  # 0xC0000409 fastfail
        (1, True, False),
        (3221225477, False, False),  # not a status code on POSIX
    ],
)
def test_died_by_signal(monkeypatch, returncode, on_windows, expected):
    monkeypatch.setattr(torch_device_probe.os, "name", "nt" if on_windows else "posix")
    assert torch_device_probe._died_by_signal(returncode) is expected


def test_real_torch_allocates_on_cpu():
    pytest.importorskip("torch")
    probe = subprocess.run(
        [sys.executable, "-c", torch_device_probe._PROBE_SCRIPT, "cpu"],
        capture_output = True,
        timeout = torch_device_probe.PROBE_TIMEOUT_SECONDS,
    )
    assert not torch_device_probe._died_by_signal(probe.returncode), probe.stderr
