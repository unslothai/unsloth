# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows CPU affinity reaches the thread pools (#14313); sys.platform and kernel32 are faked."""

import ctypes
import os
import sys

import pytest

import utils.cpu_threads as cpu_threads


def _kernel32(
    process_mask,
    system_mask,
    ok = True,
):
    class _GetProcessAffinityMask:
        argtypes = restype = None

        def __call__(self, handle, process, system):
            process._obj.value = process_mask
            system._obj.value = system_mask
            return ok

    class _GetCurrentProcess:
        restype = None

        def __call__(self):
            return -1

    class _Kernel32:
        GetProcessAffinityMask = _GetProcessAffinityMask()
        GetCurrentProcess = _GetCurrentProcess()

    return _Kernel32()


@pytest.fixture
def windows_mask(monkeypatch):
    """Set the faked (process, system) affinity masks: a 32-CPU host, unrestricted until told otherwise."""
    masks = {"process": (1 << 32) - 1, "system": (1 << 32) - 1, "ok": True}

    def fake_windll(name, use_last_error = False):
        assert name == "kernel32"
        return _kernel32(masks["process"], masks["system"], masks["ok"])

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", fake_windll, raising = False)
    # Python on Windows: process_cpu_count is cpu_count and there is no sched_getaffinity.
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 32, raising = False)
    monkeypatch.delattr(os, "sched_getaffinity", raising = False)
    monkeypatch.setattr(cpu_threads, "_openblas_memory_headroom", lambda: None)
    monkeypatch.setattr(cpu_threads, "_torch_threads_unfitted", None, raising = False)
    monkeypatch.setattr(cpu_threads, "_fitted_affinity", None, raising = False)
    return masks


@pytest.mark.parametrize(
    "process, system, ok, expected",
    [
        (0b1, (1 << 32) - 1, True, 1),
        (0b1100, (1 << 32) - 1, True, 2),
        ((1 << 32) - 1, (1 << 32) - 1, True, None),
        # threads in several processor groups
        (0, 0, True, None),
        (0b1, (1 << 32) - 1, False, None),
    ],
)
def test_windows_affinity_reader(windows_mask, process, system, ok, expected):
    windows_mask.update(process = process, system = system, ok = ok)

    assert cpu_threads._windows_affinity_cpus() == expected


# The defect: one allowed CPU of 32 still sized every pool for 32.
def test_usable_cpus_follows_a_windows_affinity_mask(windows_mask):
    windows_mask["process"] = 0b1

    assert cpu_threads._usable_cpus() == 1
    assert cpu_threads.default_openblas_threads() == 1


def test_unrestricted_windows_keeps_the_core_count(windows_mask):
    assert cpu_threads._usable_cpus() == 32
    assert cpu_threads.default_openblas_threads() == 8


def test_an_unreadable_mask_keeps_the_core_count(windows_mask, monkeypatch):
    def broken(name, use_last_error = False):
        raise OSError("no kernel32")

    monkeypatch.setattr(ctypes, "WinDLL", broken, raising = False)

    assert cpu_threads._usable_cpus() == 32


class _Torch:
    def __init__(self, threads):
        self.threads = threads
        self.set_calls = []

    def get_num_threads(self):
        return self.threads

    def set_num_threads(self, n):
        self.set_calls.append(n)
        self.threads = n


@pytest.fixture
def dll_calls(windows_mask, monkeypatch):
    calls = []
    monkeypatch.setattr(
        cpu_threads,
        "apply_openblas_runtime_cap",
        lambda: calls.append(cpu_threads._openblas_thread_target()),
    )
    saved = {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "UNSLOTH_OPENBLAS_DEFAULTED")}
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["UNSLOTH_OPENBLAS_DEFAULTED"] = "8"
    yield calls
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# Task Manager narrows a running backend, then widens it again.
def test_fit_follows_affinity_narrowed_and_widened_at_runtime(windows_mask, dll_calls, monkeypatch):
    torch = _Torch(16)
    monkeypatch.setitem(sys.modules, "torch", torch)

    cpu_threads.fit_threads_to_affinity()
    assert torch.set_calls == [] and dll_calls == []

    windows_mask["process"] = 0b1
    cpu_threads.fit_threads_to_affinity()
    cpu_threads.fit_threads_to_affinity()
    assert torch.set_calls == [1]
    assert dll_calls == [1]

    windows_mask["process"] = 0b1111
    cpu_threads.fit_threads_to_affinity()
    assert torch.set_calls == [1, 4]
    assert dll_calls == [1, 4]

    windows_mask["process"] = (1 << 32) - 1
    cpu_threads.fit_threads_to_affinity()
    assert torch.set_calls == [1, 4, 16]
    assert dll_calls == [1, 4, 16]


def test_fit_never_raises_a_smaller_user_count(windows_mask, dll_calls, monkeypatch):
    torch = _Torch(2)
    monkeypatch.setitem(sys.modules, "torch", torch)
    windows_mask["process"] = 0b1111

    cpu_threads.fit_threads_to_affinity()

    assert torch.set_calls == []
    assert dll_calls == [2]


def test_a_user_openblas_value_is_clamped_to_the_allowed_cpus(windows_mask, dll_calls, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _Torch(16))
    os.environ["OPENBLAS_NUM_THREADS"] = "6"
    windows_mask["process"] = 0b11

    cpu_threads.fit_threads_to_affinity()

    assert dll_calls == [2]


def test_fit_is_a_noop_before_torch_or_off_windows(windows_mask, dll_calls, monkeypatch):
    windows_mask["process"] = 0b1
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    cpu_threads.fit_threads_to_affinity()

    torch = _Torch(16)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(sys, "platform", "linux")
    cpu_threads.fit_threads_to_affinity()

    assert torch.set_calls == [] and dll_calls == []


def test_a_failing_torch_never_breaks_the_batch(windows_mask, dll_calls, monkeypatch):
    class _Broken:
        def get_num_threads(self):
            raise RuntimeError("torch went away")

    monkeypatch.setitem(sys.modules, "torch", _Broken())
    windows_mask["process"] = 0b1

    cpu_threads.fit_threads_to_affinity()

    assert dll_calls == []


# A process launched already narrowed gets fitted pools right after torch imports.
def test_torch_import_fits_an_inherited_affinity(windows_mask, dll_calls, monkeypatch):
    torch = _Torch(16)
    monkeypatch.setitem(sys.modules, "torch", torch)
    windows_mask["process"] = 0b1

    assert cpu_threads.install_openblas_runtime_cap()

    assert torch.set_calls == [1]
    assert dll_calls == [1, 1]


def test_every_cpu_embedding_batch_refits(monkeypatch):
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "core/rag/embeddings.py").read_text(
        encoding = "utf-8"
    )
    encode = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_st_encode"
    )
    called = [
        node.func.id
        for node in ast.walk(encode)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "fit_threads_to_affinity" in called
