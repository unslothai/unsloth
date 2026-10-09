# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The runtime OpenBLAS cap for Windows ROCm, whose rocm-openblas.dll ignores OPENBLAS_NUM_THREADS (#12942).

Runs anywhere: sys.platform and ctypes' Windows entry points are faked."""

import ctypes
import os
import sys
import types

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
    # Pinned: the cap is clamped to the core count, and CI runners can have as few as 3.
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setattr(ctypes, "WinDLL", fake_windll, raising = False)
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    # On a real Windows host importing a worker module has already installed the hook; start from none.
    monkeypatch.setattr(
        sys,
        "meta_path",
        [
            finder
            for finder in sys.meta_path
            if not getattr(finder, cpu_threads._OPENBLAS_CAP_SENTINEL, False)
        ],
    )
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
        "UNSLOTH_OPENBLAS_DEFAULTED",
    ):
        os.environ.pop(name, None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _fake_torch(threads):
    return types.SimpleNamespace(get_num_threads = lambda: threads)


# The defect: on Windows ROCm a user's OPENBLAS_NUM_THREADS never reaches the loaded DLL. Fails before the fix.
def test_user_openblas_one_reaches_the_dll(fake_windows, clean_thread_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(12))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")

    configure_cpu_threads()

    assert fake_windows == [1]


# Studio's own default of 1 is meant for numpy; the DLL is torch's CPU BLAS and gets torch's thread count.
def test_studio_default_gives_the_dll_torchs_thread_count(
    fake_windows, clean_thread_env, monkeypatch
):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(12))

    configure_cpu_threads()

    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert fake_windows == [12]


# A spawned worker only sees the inherited env, so the marker must tell it the 1 is Studio's default.
def test_a_worker_inheriting_the_default_gives_the_dll_torchs_thread_count(
    fake_windows, clean_thread_env, monkeypatch
):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(12))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("UNSLOTH_OPENBLAS_DEFAULTED", "1")

    assert cpu_threads.install_openblas_runtime_cap()

    assert fake_windows == [12]


def test_user_openblas_value_is_what_reaches_the_dll(fake_windows, clean_thread_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch") or object())
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


def test_torch_keeps_its_own_loader_and_reload_does_not_stack(
    fake_windows, clean_thread_env, monkeypatch, tmp_path
):
    import importlib

    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_text("LOADED = True\n", encoding = "utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "torch", raising = False)

    configure_cpu_threads()
    import torch

    # pkg_resources picks its provider by type(module.__loader__), so the wrapper must not stay visible.
    assert type(torch.__loader__) is importlib.machinery.SourceFileLoader
    assert torch.__spec__.loader is torch.__loader__
    importlib.reload(torch)
    assert type(torch.__loader__) is importlib.machinery.SourceFileLoader
    assert fake_windows == [1, 1]


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


# Spawned workers inherit OPENBLAS_NUM_THREADS but never re-run run.py when Desktop starts the backend through
# the CLI, so each long-lived worker module installs the cap itself, at import, before its first torch import.
@pytest.mark.parametrize("worker", ["core/training/worker.py", "core/inference/worker.py"])
def test_long_lived_workers_install_the_cap_at_import(worker):
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parent.parent / worker).read_text(encoding = "utf-8"))
    calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "install_openblas_runtime_cap"
    ]
    assert len(calls) == 1
