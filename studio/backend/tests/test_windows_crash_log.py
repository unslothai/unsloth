# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import windows_crash_log as wcl  # noqa: E402

_NS = "http://schemas.microsoft.com/win/2004/08/events/event"


def _event(
    pid_hex: str,
    module: str = "amdvlk64.dll",
    *,
    named: bool = False,
) -> str:
    values = [
        "llama-server.exe",
        "0.0.0.0",
        "6a000000",
        module,
        "32.0.21045.5002",
        "6b000000",
        "c0000005",
        "0000000000123abc",
        pid_hex,
        "01db0000000000",
        r"C:\llama-server.exe",
    ]
    tag = "Data Name='x'" if named else "Data"
    data = "".join(f"<{tag}>{value}</Data>" for value in values)
    return f"<Event xmlns='{_NS}'><EventData>{data}</EventData></Event>"


def test_fault_comes_from_the_event_with_the_crashed_pid():
    xml_text = _event("0x10", "other.dll") + _event("0x74c")
    assert wcl._fault_from_events(xml_text, 0x74C) == (
        "module=amdvlk64.dll version=32.0.21045.5002 "
        "exception=0xc0000005 offset=0x0000000000123abc"
    )


def test_prefixed_hex_values_are_kept_as_they_are():
    xml_text = _event("0x74c").replace(">c0000005<", ">0xc0000005<")
    assert "exception=0xc0000005 " in wcl._fault_from_events(xml_text, 0x74C)


def test_named_insertion_strings_parse_the_same_way():
    assert "module=amdvlk64.dll" in wcl._fault_from_events(_event("0x74c", named = True), 0x74C)


def test_no_fault_for_another_pid_short_or_malformed_events():
    assert wcl._fault_from_events(_event("0x10"), 0x74C) is None
    assert wcl._fault_from_events(_event("not-hex"), 0x74C) is None
    assert wcl._fault_from_events(f"<Event xmlns='{_NS}'><Data>x</Data></Event>", 1) is None
    assert wcl._fault_from_events("<Event", 1) is None


def test_wait_retries_until_the_event_is_written(monkeypatch):
    answers = iter([None, None, "module=a.dll"])
    monkeypatch.setattr(wcl, "_query_fault", lambda pid: next(answers))
    monkeypatch.setattr(wcl.time, "sleep", lambda s: None)
    assert wcl._wait_for_fault(1) == "module=a.dll"


def test_wait_gives_up_after_its_attempts(monkeypatch):
    calls = []
    monkeypatch.setattr(wcl, "_query_fault", lambda pid: calls.append(pid))
    monkeypatch.setattr(wcl.time, "sleep", lambda s: None)
    assert wcl._wait_for_fault(1) is None
    assert len(calls) == wcl._EVENT_WAIT_ATTEMPTS


def test_query_failure_reads_as_no_fault(monkeypatch):
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("wevtutil", 5)

    monkeypatch.setattr(wcl.subprocess, "run", timed_out)
    assert wcl._query_fault(1) is None


class _Key:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_winreg(
    monkeypatch,
    *,
    values = None,
    subkeys = None,
):
    values = values or {}
    subkeys = subkeys or {}
    registry = types.SimpleNamespace(HKEY_LOCAL_MACHINE = "HKLM", HKEY_CURRENT_USER = "HKCU")

    def open_key(parent, name):
        path = f"{getattr(parent, 'path', parent)}\\{name}"
        if path not in values and path not in subkeys:
            raise FileNotFoundError(path)
        return _Key(path)

    def query_value(key, name):
        try:
            return values[key.path][name], 1
        except KeyError:
            raise FileNotFoundError(name) from None

    registry.OpenKey = open_key
    registry.QueryInfoKey = lambda key: (
        len(subkeys.get(key.path, ())),
        len(values.get(key.path, {})),
        0,
    )
    registry.EnumKey = lambda key, index: subkeys[key.path][index]
    registry.QueryValueEx = query_value
    registry.EnumValue = lambda key, index: (*list(values[key.path].items())[index], 4)
    monkeypatch.setitem(sys.modules, "winreg", registry)


def test_gpu_drivers_skip_subkeys_that_are_not_adapters(monkeypatch):
    root = f"HKLM\\{wcl._DISPLAY_ADAPTERS_KEY}"
    _fake_winreg(
        monkeypatch,
        subkeys = {root: ["0000", "Properties"]},
        values = {
            f"{root}\\0000": {"DriverDesc": "AMD Radeon RX 5700 XT", "DriverVersion": "32.0.1"},
            f"{root}\\Properties": {},
        },
    )
    assert wcl._gpu_drivers() == ["AMD Radeon RX 5700 XT 32.0.1"]


def test_implicit_layers_list_enabled_manifests_by_file_name_from_both_hives(monkeypatch):
    _fake_winreg(
        monkeypatch,
        values = {
            f"HKLM\\{wcl._IMPLICIT_LAYERS_KEY}": {
                r"C:\Tools\VkLayer_a.json": 0,
                r"C:\Old\b.json": 1,
            },
            f"HKCU\\{wcl._IMPLICIT_LAYERS_KEY}": {r"C:\Users\me\VkLayer_c.json": 0},
        },
    )
    assert wcl._enabled_implicit_layers() == ["VkLayer_a.json", "VkLayer_c.json"]


def test_details_append_after_the_attempt_end_line(tmp_path, monkeypatch):
    log_path = tmp_path / "llama.log"
    log_path.write_text(
        "[studio] attempt end reason=exited exit_code=3221225477\n", encoding = "utf-8"
    )
    monkeypatch.setattr(wcl, "_wait_for_fault", lambda pid: None)
    monkeypatch.setattr(wcl, "_gpu_drivers", lambda: ["GPU 1.0"])
    monkeypatch.setattr(wcl, "_enabled_implicit_layers", lambda: [])
    wcl._append_details(log_path, 1)
    assert log_path.read_text(encoding = "utf-8").splitlines()[1:] == [
        "[studio] crash module=unknown",
        "[studio] gpu drivers: GPU 1.0",
        "[studio] vulkan implicit layers: none",
    ]


def test_record_is_a_no_op_off_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(wcl.sys, "platform", "linux")
    monkeypatch.setattr(wcl, "_append_details", lambda *a: (_ for _ in ()).throw(AssertionError))
    wcl.record_crash_details(tmp_path / "llama.log", 1)
    assert not list(tmp_path.iterdir())


def test_record_runs_off_the_calling_thread_on_windows(tmp_path, monkeypatch):
    started = []

    class _Thread:
        def __init__(self, target, args, daemon, name):
            started.append((target, args, daemon))

        def start(self):
            pass

    monkeypatch.setattr(wcl.sys, "platform", "win32")
    monkeypatch.setattr(wcl.threading, "Thread", _Thread)
    wcl.record_crash_details(tmp_path / "llama.log", 7)
    assert started == [(wcl._append_details, (tmp_path / "llama.log", 7), True)]
