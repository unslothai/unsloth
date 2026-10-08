# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The runtime OpenBLAS cap for Windows ROCm, whose rocm-openblas.dll ignores OPENBLAS_NUM_THREADS (#12942).

Runs anywhere: sys.platform and ctypes' Windows entry points are faked."""

import ctypes
import os
import sys

import pytest

import utils.cpu_threads as cpu_threads
from utils.cpu_threads import configure_cpu_threads


class _FakeSetter:
    def __init__(self, calls):
        self.calls = calls
        self.argtypes = None
        self.restype = None

    def __call__(self, n):
        self.calls.append(n)


class _FakeLib:
    def __init__(
        self,
        calls,
        has_setter = True,
    ):
        if has_setter:
            self.openblas_set_num_threads = _FakeSetter(calls)


class _Calls(list):
    loaded: dict


@pytest.fixture
def fake_windows(monkeypatch):
    """win32 with rocm-openblas.dll loaded (handle 0x1234); returns the list of setter calls."""
    calls = _Calls()
    loaded = {"rocm-openblas.dll": 0x1234}

    class _GetModuleHandleW:
        restype = None

        def __call__(self, name):
            return loaded.get(name, 0)

    class _Kernel32:
        def __init__(self):
            self.GetModuleHandleW = _GetModuleHandleW()

    def fake_windll(name, use_last_error = False):
        assert name == "kernel32"
        return _Kernel32()

    def fake_cdll(name, handle = None):
        assert handle == loaded[name], "must bind the already loaded module, never load a new one"
        return _FakeLib(calls)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", fake_windll, raising = False)
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    calls.loaded = loaded
    return calls


@pytest.fixture
def clean_thread_env():
    """configure_cpu_threads() writes the real os.environ; monkeypatch.delenv only restores names that existed."""
    saved = dict(os.environ)
    for name in (
        "UNSLOTH_CPU_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.pop(name, None)
    yield
    os.environ.clear()
    os.environ.update(saved)


# The defect: on Windows ROCm the env default never reaches the loaded DLL. Fails before the fix.
def test_process_configuration_caps_a_loaded_rocm_openblas(
    fake_windows, clean_thread_env, monkeypatch
):
    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch") or object())

    configure_cpu_threads()

    assert fake_windows == [1]


def test_user_openblas_value_is_what_reaches_the_dll(fake_windows, clean_thread_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch") or object())
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "6")

    configure_cpu_threads()

    assert fake_windows == [6]


def test_unsloth_cpu_threads_reaches_the_dll(fake_windows, clean_thread_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch") or object())
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "4")

    configure_cpu_threads()

    assert fake_windows == [4]


def test_cap_waits_for_torch_and_fires_after_its_module_body(
    fake_windows, clean_thread_env, monkeypatch, tmp_path
):
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_text("LOADED = True\n", encoding = "utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    seen = []
    real_apply = cpu_threads.apply_openblas_runtime_cap

    def spy():
        seen.append(getattr(sys.modules.get("torch"), "LOADED", False))
        return real_apply()

    monkeypatch.setattr(cpu_threads, "apply_openblas_runtime_cap", spy)

    configure_cpu_threads()
    assert fake_windows == [] and seen == []  # nothing loaded yet, so nothing to cap

    import torch  # noqa: F401 -- the fake package above

    assert seen == [True]
    assert fake_windows == [1]


def test_install_is_idempotent(fake_windows, monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    before = len(sys.meta_path)

    assert cpu_threads.install_openblas_runtime_cap()
    assert cpu_threads.install_openblas_runtime_cap()

    assert len(sys.meta_path) == before + 1


def test_nothing_happens_when_the_dll_is_not_loaded(fake_windows, clean_thread_env, monkeypatch):
    fake_windows.loaded.clear()
    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch") or object())

    configure_cpu_threads()

    assert fake_windows == []


@pytest.mark.parametrize("raw", ["0", "-2", "abc"])
def test_invalid_openblas_value_is_not_pushed(fake_windows, clean_thread_env, monkeypatch, raw):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", raw)

    assert cpu_threads.apply_openblas_runtime_cap() == []
    assert fake_windows == []


def test_a_failing_setter_never_breaks_startup(fake_windows, clean_thread_env, monkeypatch):
    raised = []

    def boom(name, handle = None):
        raised.append(name)
        raise OSError("bad module")

    monkeypatch.setattr(ctypes, "CDLL", boom)
    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch") or object())

    configure_cpu_threads()  # must not raise

    assert raised == ["rocm-openblas.dll"]


def test_cap_is_clamped_to_the_core_count(fake_windows, clean_thread_env, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch") or object())
    os.environ["OPENBLAS_NUM_THREADS"] = "64"

    configure_cpu_threads()

    assert fake_windows == [8]


def test_off_windows_nothing_is_installed(monkeypatch, clean_thread_env):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    before = list(sys.meta_path)

    configure_cpu_threads()

    assert sys.meta_path == before
    assert cpu_threads.apply_openblas_runtime_cap() == []


def test_a_mapping_env_stays_pure(fake_windows, monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    before = list(sys.meta_path)

    configure_cpu_threads({})

    assert sys.meta_path == before
