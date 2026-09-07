# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from core.inference import srt_adapter


@pytest.mark.parametrize("ports", [[55080, 55089], [1, 2], [55080, 55300], [True, 55089]])
def test_installed_windows_range_is_validated_before_launch(tmp_path, monkeypatch, ports):
    monkeypatch.setattr(srt_adapter.sys, "platform", "win32")
    monkeypatch.setattr(srt_adapter, "RUNTIME", tmp_path)
    monkeypatch.setattr(srt_adapter, "read_roots", lambda executable: [str(tmp_path)])
    (tmp_path / "installed-runtime-settings.json").write_text(
        json.dumps({"windowsProxyPortRange": ports})
    )
    if ports == [55080, 55089]:
        request = srt_adapter.request_for([sys.executable, "-c", "print(1)"], str(tmp_path), {}, 30)
        assert request["windowsProxyPortRange"] == ports
    else:
        with pytest.raises(srt_adapter.SrtError, match = "Invalid installed"):
            srt_adapter.request_for([sys.executable, "-c", "print(1)"], str(tmp_path), {}, 30)


@pytest.fixture
def mount_table(monkeypatch):
    monkeypatch.setattr(
        Path, "read_text", lambda *args, **kwargs: "1 0 0:1 / / rw - ext4 /dev/root rw\n"
    )


def test_workdir_hardlink_is_refused(tmp_path, mount_table):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    sentinel = tmp_path / "host.txt"
    sentinel.write_bytes(b"benign host control")
    os.link(sentinel, work / "alias.txt")
    with pytest.raises(srt_adapter.SrtError, match = "hardlinked"):
        srt_adapter.validate_roots([str(runtime)], str(work))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason = "native FIFO creation requires POSIX")
def test_runtime_fifo_is_refused(tmp_path, mount_table):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    os.mkfifo(runtime / "host-ipc")
    with pytest.raises(srt_adapter.SrtError, match = "IPC/device"):
        srt_adapter.validate_roots([str(runtime)], str(work))


def test_nested_runtime_mount_is_masked(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    nested = runtime / "foreign"
    nested.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(
        Path, "read_text", lambda *args, **kwargs: f"1 0 0:1 / {nested} rw - ext4 /dev/foreign rw\n"
    )
    assert srt_adapter.validate_roots([str(runtime)], str(work)) == [str(nested)]


def test_nested_workdir_mount_is_refused(tmp_path, monkeypatch):
    nested = tmp_path / "foreign"
    nested.mkdir()
    monkeypatch.setattr(
        Path, "read_text", lambda *args, **kwargs: f"1 0 0:1 / {nested} rw - ext4 /dev/foreign rw\n"
    )
    with pytest.raises(srt_adapter.SrtError, match = "nested host mount"):
        srt_adapter.validate_roots([], str(tmp_path))


def test_selected_venv_executable_is_not_replaced_by_base_python(tmp_path, monkeypatch):
    executable = str(tmp_path / "venv" / "bin" / "python")
    monkeypatch.setattr(srt_adapter, "read_roots", lambda path: [str(tmp_path / "venv")])
    monkeypatch.setattr(srt_adapter, "validate_roots", lambda roots, workdir: [])
    request = srt_adapter.request_for([executable, "-c", "print(1)"], str(tmp_path), {}, 5)
    assert request["executable"] == executable
    assert request["argv"] == ["-c", "print(1)"]


def test_extra_trust_alias_mounts_only_canonical_target(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    target = tmp_path / "certificates" / "ca.pem"
    alias = runtime / "ssl" / "cert.pem"
    original_realpath = os.path.realpath
    monkeypatch.setattr(srt_adapter, "read_roots", lambda path: [str(runtime)])
    monkeypatch.setattr(srt_adapter, "validate_roots", lambda roots, workdir: [])
    monkeypatch.setattr(
        srt_adapter.os.path,
        "realpath",
        lambda path: str(target) if str(path) == str(alias) else original_realpath(path),
    )
    request = srt_adapter.request_for(
        [str(runtime / "python")], str(tmp_path), {}, 5, additional_read_roots = [str(alias)]
    )
    assert str(target) in request["readRoots"]
    assert str(alias) not in request["readRoots"]


def test_sysconfig_data_does_not_admit_entire_system_prefix(monkeypatch):
    monkeypatch.setattr(srt_adapter.sys, "platform", "linux")
    monkeypatch.setattr(srt_adapter.sysconfig, "get_config_var", lambda name: "x86_64-linux-gnu")
    monkeypatch.setattr(
        srt_adapter.sysconfig, "get_paths", lambda: {"data": "/", "stdlib": "/usr/lib/python"}
    )
    monkeypatch.setattr(srt_adapter.sys, "prefix", "/usr")
    monkeypatch.setattr(srt_adapter.sys, "base_prefix", "/usr")
    monkeypatch.setattr(srt_adapter.os.path, "exists", lambda path: True)
    monkeypatch.setattr(srt_adapter.os.path, "realpath", lambda path: path)
    assert "/" not in srt_adapter.read_roots("/usr/bin/python")
    assert "/usr" not in srt_adapter.read_roots("/usr/bin/python")


def test_read_roots_exclude_unselected_sdk_and_interpreter_trees(monkeypatch):
    monkeypatch.setattr(srt_adapter.sys, "platform", "linux")
    monkeypatch.setattr(
        srt_adapter.sysconfig,
        "get_paths",
        lambda: {"stdlib": "/usr/lib/python3.12", "purelib": "/selected/env/lib/site-packages"},
    )
    monkeypatch.setattr(srt_adapter.sysconfig, "get_config_var", lambda name: "x86_64-linux-gnu")
    monkeypatch.setattr(
        srt_adapter.site, "getsitepackages", lambda: ["/usr/lib/python3/dist-packages"]
    )
    monkeypatch.setattr(srt_adapter.sys, "prefix", "/selected/env")
    monkeypatch.setattr(srt_adapter.sys, "base_prefix", "/usr")
    monkeypatch.setattr(srt_adapter.os.path, "exists", lambda path: True)
    monkeypatch.setattr(srt_adapter.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(srt_adapter.os.path, "realpath", lambda path: path)
    monkeypatch.setattr(srt_adapter.os.path, "abspath", lambda path: path)
    roots = srt_adapter.read_roots("/usr/bin/python3.12")
    assert not {"/usr/lib", "/lib", "/usr/local/lib"}.intersection(roots)
    assert {
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/python3.12",
        "/usr/lib/python3/dist-packages",
        "/selected/env",
    }.issubset(roots)


@pytest.mark.parametrize(
    "receipt,valid",
    [
        (b'{"v":1,"event":"exit","code":0,"signal":null,"reason":"completed"}\n', True),
        (b"", False),
        (b"not json\n", False),
        (b'{"v":1,"event":"error"}\n', False),
        (b'{"v":1,"event":"exit","code":0,"signal":null,"reason":"timeout"}\n', False),
    ],
)
def test_success_requires_private_completion_receipt(receipt, valid):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, receipt)
    os.close(write_fd)
    proc = SimpleNamespace(_srt_control_fd = read_fd, _srt_control_pending = b"", poll = lambda: 0)
    try:
        if valid:
            srt_adapter.verify_success(proc)
        else:
            with pytest.raises(srt_adapter.SrtError):
                srt_adapter.verify_success(proc)
    finally:
        srt_adapter.release_control(proc)
