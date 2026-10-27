# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime OpenBLAS cap for Windows ROCm (#12942); sys.platform and ctypes' Windows entry points are faked."""

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
    # Pinned: the cap clamps to the core count and the default fits free memory.
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setattr(cpu_threads, "_usable_cpus", lambda: 32)
    monkeypatch.setattr(cpu_threads, "_openblas_memory_headroom", lambda: None)
    monkeypatch.setattr(ctypes, "WinDLL", fake_windll, raising = False)
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    # A worker import on real Windows has already installed the hook.
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
    """monkeypatch.delenv only restores names that existed."""
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


# The defect: a user's OPENBLAS_NUM_THREADS never reached the loaded DLL.
def test_user_openblas_one_reaches_the_dll(fake_windows, clean_thread_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(12))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")

    configure_cpu_threads()

    assert fake_windows == [1]


def test_studio_default_gives_the_dll_torchs_thread_count(
    fake_windows, clean_thread_env, monkeypatch
):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(12))

    configure_cpu_threads()

    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
    assert fake_windows == [12]


@pytest.mark.parametrize("headroom_mb, expected", [(None, 12), (64 << 10, 12), (2560, 2), (500, 1)])
def test_studio_default_fits_the_dll_to_the_memory_left(
    fake_windows, clean_thread_env, monkeypatch, headroom_mb, expected
):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(12))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.setenv("UNSLOTH_OPENBLAS_DEFAULTED", "8")
    monkeypatch.setattr(
        cpu_threads,
        "_openblas_memory_headroom",
        lambda: None if headroom_mb is None else headroom_mb << 20,
    )

    cpu_threads.apply_openblas_runtime_cap()

    assert fake_windows == [expected]


def test_a_user_value_reaches_the_dll_however_short_memory_is(
    fake_windows, clean_thread_env, monkeypatch
):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(12))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "6")
    monkeypatch.setattr(cpu_threads, "_openblas_memory_headroom", lambda: 100 << 20)

    cpu_threads.apply_openblas_runtime_cap()

    assert fake_windows == [6]


def test_a_worker_inheriting_the_default_gives_the_dll_torchs_thread_count(
    fake_windows, clean_thread_env, monkeypatch
):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(12))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.setenv("UNSLOTH_OPENBLAS_DEFAULTED", "8")

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
    (package / "__init__.py").write_text(
        "LOADED = True\ndef get_num_threads():\n    return 12\n", encoding = "utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    seen = []
    real_apply = cpu_threads.apply_openblas_runtime_cap

    def spy():
        seen.append(getattr(sys.modules.get("torch"), "LOADED", False))
        return real_apply()

    monkeypatch.setattr(cpu_threads, "apply_openblas_runtime_cap", spy)

    configure_cpu_threads()
    assert fake_windows == [] and seen == []

    import torch  # noqa: F401 -- the fake package above

    assert seen == [True]
    assert fake_windows == [12]


def test_torch_keeps_its_own_loader_and_reload_does_not_stack(
    fake_windows, clean_thread_env, monkeypatch, tmp_path
):
    import importlib.util

    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_text(
        "LOADED = True\ndef get_num_threads():\n    return 12\n", encoding = "utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "torch", raising = False)

    # e.g. beartype.claw's SourceFileLoader subclass.
    native = type(importlib.util.find_spec("torch").loader)
    assert native is not cpu_threads._TorchLoader
    configure_cpu_threads()
    import torch

    assert type(torch.__loader__) is native
    assert torch.__spec__.loader is torch.__loader__
    importlib.reload(torch)
    assert type(torch.__loader__) is native
    assert fake_windows == [12, 12]


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

    configure_cpu_threads()

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


# Desktop-spawned workers never run run.py, so each worker module installs the cap at import.
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
