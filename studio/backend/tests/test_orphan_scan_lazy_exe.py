# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The psutil orphan scan (Windows / macOS) resolves exe only for llama-server-named processes,
with the same candidates and kill decisions as reading exe for every process."""

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

from core.inference import llama_cpp as llama_cpp_module
from core.inference.llama_cpp import LlamaCppBackend

_NO_PROCFS = "/unsloth-test-no-such-proc-root"


def _fake_psutil():
    """psutil stand-in with the real process_iter contract: attrs are read through the
    methods, AccessDenied / ZombieProcess become None, NoSuchProcess drops the process."""
    mod = types.ModuleType("psutil")
    mod.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    mod.ZombieProcess = type("ZombieProcess", (mod.NoSuchProcess,), {})
    mod.AccessDenied = type("AccessDenied", (Exception,), {})
    mod.procs = []
    mod.exe_calls = []

    def process_iter(attrs = None):
        for proc in list(mod.procs):
            try:
                info = {}
                for attr in attrs or ():
                    try:
                        info[attr] = proc.pid if attr == "pid" else getattr(proc, attr)()
                    except (mod.AccessDenied, mod.ZombieProcess):
                        info[attr] = None
                proc.info = info
                yield proc
            except mod.NoSuchProcess:
                continue

    mod.process_iter = process_iter
    return mod


class _Proc:
    def __init__(
        self,
        mod,
        killed,
        pid,
        name,
        exe,
        kill_error = None,
    ):
        self._mod, self._killed, self.pid = mod, killed, pid
        self._name, self._exe, self._kill_error = name, exe, kill_error

    def _value(self, value):
        if isinstance(value, str) and value.startswith("raise:"):
            raise getattr(self._mod, value[len("raise:") :])(self.pid)
        return value

    def name(self):
        return self._value(self._name)

    def exe(self):
        self._mod.exe_calls.append(self.pid)
        return self._value(self._exe)

    def kill(self):
        if self._kill_error:
            raise getattr(self._mod, self._kill_error)(self.pid)
        self._killed.append(self.pid)


def _scan(tmp_path: Path):
    owned = tmp_path / "owned" / "llama-server"
    owned.parent.mkdir(parents = True)
    owned.write_text("x")
    foreign = tmp_path / "foreign" / "llama-server"
    mod = _fake_psutil()
    killed: list[int] = []
    base = os.getpid() + 1000
    me = os.getpid()
    rows = [
        (base + 1, "llama-server", str(owned)),  # owned orphan: killed
        (base + 2, "python3", "/usr/bin/python3"),  # not a server
        (base + 3, "LLAMA-SERVER.EXE", str(owned)),  # case-insensitive name: killed
        (base + 4, "llama-server", "raise:AccessDenied"),
        (base + 5, "llama-server", "raise:ZombieProcess"),
        (base + 6, "llama-server", "raise:NoSuchProcess"),  # exited mid-scan
        (base + 7, "llama-server", ""),
        (base + 8, "raise:AccessDenied", str(owned)),  # unreadable name
        (base + 9, "raise:NoSuchProcess", str(owned)),
        (me, "llama-server", str(owned)),  # this process
        (base + 11, "llama-server", str(foreign)),  # not ours
        (base + 12, "llama-server", str(owned)),  # live parent: spared
        (base + 13, "llama-server-cuda", str(owned)),  # prefix match: killed
        (base + 15, "svchost.exe", "raise:AccessDenied"),
        (base + 16, "kernel_task", "raise:ZombieProcess"),
    ]
    mod.procs = [_Proc(mod, killed, *row) for row in rows]
    mod.procs.append(_Proc(mod, killed, base + 14, "llama-server", str(owned), "NoSuchProcess"))
    with (
        patch.dict(sys.modules, {"psutil": mod}),
        patch.dict(os.environ, {"LLAMA_SERVER_PATH": str(owned)}),
        patch.object(llama_cpp_module, "_PROC_ROOT", _NO_PROCFS),
        patch.object(llama_cpp_module, "_is_signalable_pid", lambda pid: True),
        patch.object(LlamaCppBackend, "_reap_recorded_pid", staticmethod(lambda: 0)),
        patch.object(
            LlamaCppBackend, "_pid_parent_is_alive", staticmethod(lambda pid: pid == base + 12)
        ),
    ):
        n = LlamaCppBackend._kill_orphaned_servers()
    return n, killed, mod.exe_calls, base


def test_psutil_scan_kill_decisions_unchanged(tmp_path: Path) -> None:
    n, killed, _calls, base = _scan(tmp_path)
    assert killed == [base + 1, base + 3, base + 13]
    assert n == 3


def test_psutil_scan_resolves_exe_only_for_server_names(tmp_path: Path) -> None:
    _n, _killed, calls, base = _scan(tmp_path)
    offsets = [pid - base for pid in calls]
    assert offsets == [1, 3, 4, 5, 6, 7, 11, 12, 13, 14]
