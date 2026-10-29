# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows + AMD HIP charges commit for GPU buffers it never touches; a load that leaves the
system short of commit says so instead of exhausting the page file silently (#16015)."""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

import core.inference.diffusion_memory as diffusion_memory
from core.inference import llama_cpp as llama_cpp_module
from core.inference.llama_cpp import LlamaCppBackend

_MIB = 1024 * 1024


def test_report_numbers_warn_with_page_file_advice():
    # The report: ~15.9 GiB untouched commit on llama-server, 1.2 GB of commit left.
    message = LlamaCppBackend._hip_commit_headroom_message(16_265, 1_150)
    assert message is not None
    assert "15.9 GiB" in message
    assert "1.1 GiB" in message
    assert "at least 16 GiB" in message
    assert "Virtual memory" in message


@pytest.mark.parametrize(
    ("untouched_mib", "headroom_mib"),
    [
        (16_265, 4_096),  # enough commit left
        (16_265, None),  # headroom unreadable
        (2_047, 0),  # the child holds no meaningful untouched commit
    ],
)
def test_no_warning_while_there_is_room_or_nothing_to_blame(untouched_mib, headroom_mib):
    assert LlamaCppBackend._hip_commit_headroom_message(untouched_mib, headroom_mib) is None


class _FakeProcess:
    def __init__(self, pid):
        self.pid = pid

    def memory_info(self):
        return _types.SimpleNamespace(private = 16_500 * _MIB, wset = 235 * _MIB)


def _windows_child(
    monkeypatch,
    backends,
    headroom_mib = 1_150,
):
    monkeypatch.setattr(llama_cpp_module.sys, "platform", "win32")
    monkeypatch.setattr(
        LlamaCppBackend, "_installed_ggml_backends", staticmethod(lambda binary = None: backends)
    )
    monkeypatch.setattr(diffusion_memory, "_windows_commit_headroom_mib", lambda: headroom_mib)
    import psutil

    monkeypatch.setattr(psutil, "Process", _FakeProcess)


def test_hip_child_short_of_commit_gets_the_notice(monkeypatch):
    _windows_child(monkeypatch, frozenset({"base", "cpu", "hip"}))
    message = LlamaCppBackend._windows_hip_commit_notice(1234, "llama-server.exe")
    assert message is not None and "15.9 GiB" in message


@pytest.mark.parametrize("backends", [frozenset({"cpu", "cuda"}), frozenset({"cpu", "vulkan"})])
def test_non_hip_builds_are_not_measured(monkeypatch, backends):
    _windows_child(monkeypatch, backends)
    assert LlamaCppBackend._windows_hip_commit_notice(1234, "llama-server.exe") is None


def test_off_windows_or_without_a_child_nothing_is_measured(monkeypatch):
    _windows_child(monkeypatch, frozenset({"hip"}))
    assert LlamaCppBackend._windows_hip_commit_notice(None, "llama-server.exe") is None
    monkeypatch.setattr(llama_cpp_module.sys, "platform", "linux")
    assert LlamaCppBackend._windows_hip_commit_notice(1234, "llama-server.exe") is None


def test_a_failed_probe_never_fails_the_load(monkeypatch):
    _windows_child(monkeypatch, frozenset({"hip"}))
    import psutil

    def _gone(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", _gone)
    assert LlamaCppBackend._windows_hip_commit_notice(1234, "llama-server.exe") is None
