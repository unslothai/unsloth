# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Atomic engine status writes under Windows file-sharing contention."""

import errno
import json
import sys
import threading

import pytest

from core.inference import engine_install as install


def _windows_error(code):
    error = PermissionError(errno.EACCES, "file is open")
    error.winerror = code
    return error


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows read handles deny deletion")
def test_atomic_json_waits_for_real_windows_reader(tmp_path, monkeypatch):
    path = tmp_path / "vllm.job.json"
    path.write_text('{"state":"queued"}', encoding = "utf-8")
    replace = install.os.replace
    denied = threading.Event()
    errors = []

    def observed_replace(source, destination):
        try:
            return replace(source, destination)
        except PermissionError:
            denied.set()
            raise

    monkeypatch.setattr(install.os, "replace", observed_replace)

    def writer():
        try:
            install._atomic_json(path, {"state": "running"})
        except Exception as error:
            errors.append(error)

    with path.open(encoding = "utf-8") as reader:
        assert json.load(reader) == {"state": "queued"}
        thread = threading.Thread(target = writer)
        thread.start()
        saw_denial = denied.wait(5)
    # Close the reader before joining, allowing the retried rename to complete.
    thread.join(5)
    assert not thread.is_alive()
    assert saw_denial, "Windows did not refuse replacement of the open file"
    assert not errors
    assert json.loads(path.read_text()) == {"state": "running"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("code", [5, 32, 33])
def test_atomic_json_retries_windows_contention(tmp_path, monkeypatch, code):
    monkeypatch.setattr(install.sys, "platform", "win32")
    path = tmp_path / "vllm.job.json"
    path.write_text('{"state":"queued"}')
    replace = install.os.replace
    calls = []

    def transient(source, destination):
        calls.append(source)
        if len(calls) == 1:
            assert json.loads(path.read_text()) == {"state": "queued"}
            raise _windows_error(code)
        return replace(source, destination)

    monkeypatch.setattr(install.os, "replace", transient)
    install._atomic_json(path, {"state": "running"})
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert json.loads(path.read_text()) == {"state": "running"}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_json_permanent_denial_is_bounded_and_preserves_previous(tmp_path, monkeypatch):
    monkeypatch.setattr(install.sys, "platform", "win32")
    path = tmp_path / "vllm.job.json"
    path.write_text('{"state":"queued"}')
    clock = iter([0, 0.25, 0.5, 0.75, 1.0])
    monkeypatch.setattr(install.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(install.time, "sleep", lambda _: None)
    calls = []
    error = _windows_error(5)

    def denied(source, destination):
        calls.append(source)
        raise error

    monkeypatch.setattr(install.os, "replace", denied)
    with pytest.raises(PermissionError) as caught:
        install._atomic_json(path, {"state": "running"})
    assert caught.value is error
    assert len(calls) == 4
    assert json.loads(path.read_text()) == {"state": "queued"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("platform,code", [("linux", 5), ("win32", 112)])
def test_atomic_json_other_errors_are_not_retried(tmp_path, monkeypatch, platform, code):
    monkeypatch.setattr(install.sys, "platform", platform)
    path = tmp_path / "vllm.job.json"
    error = _windows_error(code)
    calls = []

    def denied(source, destination):
        calls.append(source)
        raise error

    monkeypatch.setattr(install.os, "replace", denied)
    with pytest.raises(PermissionError) as caught:
        install._atomic_json(path, {"state": "running"})
    assert caught.value is error
    assert len(calls) == 1
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))
