# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native deadline-owned shell discovery; not Terminal sandbox qualification."""

from dataclasses import replace
import ctypes
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from test_preparation import assert_dead, BASE_PYTHON
from core.inference.os_sandbox import ToolLaunchPlan
from core.inference.windows_sandbox import terminal_runtime as runtime
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows Terminal scanner")


@pytest.fixture
def spec(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    shell = str(Path(os.environ["SystemRoot"]) / "System32/cmd.exe")
    return ToolLaunchPlan(
        (shell, "/d", "/c", "echo PAYLOAD_RAN>payload-ran"),
        str(work),
        {},
        execution_kind = "terminal",
    )


def test_real_worker_discovers_shell_without_executing_it(spec, monkeypatch):
    original = runtime._run_worker
    workers = []

    def run(argv, *args, **kwargs):
        assert argv[1:4] == ["-I", "-S", "-B"]
        assert Path(argv[4]).name == "terminal_runtime_worker.py"
        assert argv[0] == BASE_PYTHON
        data, pid = original(argv, *args, **kwargs)
        workers.append(pid)
        return data, pid

    monkeypatch.setattr(runtime, "_run_worker", run)
    monkeypatch.setattr(runtime.lpac, "_runtime_roots", lambda *args: pytest.fail("broker scan"))
    before = (spec.argv, dict(spec.env), spec.workdir)
    result = runtime.inspect_terminal_runtime(spec)
    assert os.path.samefile(result.argv[0], spec.argv[0])
    assert result.argv[1:] == spec.argv[1:]
    assert result.workdir == spec.workdir
    assert result.runtime_roots == (str(Path(spec.argv[0]).parent.resolve()),)
    assert result.acl_roots == ()
    assert (spec.argv, spec.env, spec.workdir) == before
    assert not hasattr(result, "qualified") and not hasattr(result, "execution_record")
    assert len(workers) == 1
    assert_dead(workers[0])
    assert not list(Path(spec.workdir).iterdir())


def test_relative_cmd_uses_only_the_supplied_absolute_path(spec):
    request = replace(
        spec, argv = ("cmd", *spec.argv[1:]), env = {"PATH": str(Path(spec.argv[0]).parent)}
    )
    result = runtime.inspect_terminal_runtime(request)
    assert os.path.samefile(result.argv[0], spec.argv[0])
    assert not list(Path(spec.workdir).iterdir())


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), 121, "30"])
def test_invalid_timeout_starts_nothing(spec, monkeypatch, timeout):
    monkeypatch.setattr(runtime, "_run_worker", lambda *a, **k: pytest.fail("worker started"))
    with pytest.raises(WindowsRuntimeError, match = "timeout"):
        runtime.inspect_terminal_runtime(spec, timeout = timeout)


@pytest.mark.parametrize(
    "change",
    [
        {"execution_kind": "python"},
        {"execution_kind": None},
        {"requested_mode": "limited"},
        {"requested_mode": "full"},
        {"close_fds": False},
        {"terminate_descendants": False},
        {"argv": ()},
        {"argv": ("cmd",)},
        {"argv": ("cmd\0",)},
        {"argv": ("x" * 16385,)},
        {"workdir": "relative"},
        {"env": {"bad\0": "value"}},
    ],
)
def test_invalid_plan_starts_nothing(spec, monkeypatch, change):
    monkeypatch.setattr(runtime, "_run_worker", lambda *a, **k: pytest.fail("worker started"))
    with pytest.raises(WindowsRuntimeError):
        runtime.inspect_terminal_runtime(replace(spec, **change))


def test_cancel_before_discovery_never_captures_or_launches(spec, monkeypatch):
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(runtime, "_capture_broker_runtime", lambda: pytest.fail("capture"))
    with pytest.raises(WindowsRuntimeError) as caught:
        runtime.inspect_terminal_runtime(spec, cancel = cancel)
    assert caught.value.code == "WINDOWS_SANDBOX_CANCELLED"


def test_explicit_cancellation_signal_is_preserved(spec, monkeypatch):
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(runtime, "_capture_broker_runtime", lambda: pytest.fail("capture"))
    with pytest.raises(WindowsRuntimeError) as caught:
        runtime.inspect_terminal_runtime(spec, cancel = cancel)
    assert caught.value.code == "WINDOWS_SANDBOX_CANCELLED"


@pytest.mark.parametrize("escape", ["hardlink", "symlink"])
def test_real_worker_rejects_workdir_escape_before_any_payload(spec, tmp_path, escape):
    outside = tmp_path / "outside"
    outside.write_text("host sentinel", encoding = "utf-8")
    target = Path(spec.workdir) / "escape"
    if escape == "hardlink":
        os.link(outside, target)
    else:
        target.symlink_to(outside)
    with pytest.raises(WindowsRuntimeError, match = "boundary|reparse"):
        runtime.inspect_terminal_runtime(spec)
    assert not (Path(spec.workdir) / "payload-ran").exists()
    assert outside.read_text(encoding = "utf-8") == "host sentinel"


def test_real_worker_rejects_runtime_workdir_overlap(spec):
    shell = Path(spec.workdir) / "cmd.exe"
    shell.touch()
    with pytest.raises(WindowsRuntimeError, match = "overlap"):
        runtime.inspect_terminal_runtime(replace(spec, argv = (str(shell),)))


def test_real_worker_scans_shell_files_without_loading_them(spec, tmp_path):
    binary = tmp_path / "Git with spaces λ" / "bin"
    binary.mkdir(parents = True)
    shell = binary / "bash.exe"
    shell.write_bytes(b"deliberately not an executable")
    utilities = binary.parent / "usr/bin"
    utilities.mkdir(parents = True)
    request = replace(spec, argv = (str(shell), "-c", "touch payload-ran"))
    result = runtime.inspect_terminal_runtime(request)
    assert set(result.runtime_roots) == {str(binary), str(utilities)}
    assert result.acl_roots == result.runtime_roots
    assert not list(Path(spec.workdir).iterdir())
    assert shell.read_bytes() == b"deliberately not an executable"


@pytest.mark.parametrize(
    "data",
    [b"", b"{}", b"[]", b"null", b"bad", b"\xff", b"x" * 65537],
    ids = ["empty", "object", "array", "null", "malformed", "invalid-utf8", "oversized"],
)
def test_malformed_worker_output_is_not_a_runtime(data, spec):
    request = runtime._input({"argv": list(spec.argv), "workdir": spec.workdir, "env": spec.env})
    with pytest.raises(WindowsRuntimeError):
        runtime._response(data, 123, "a" * 64, request)


@pytest.mark.parametrize("reason", ["timeout", "cancel"])
def test_blocked_worker_is_killed_without_broker_fallback(spec, tmp_path, monkeypatch, reason):
    original = runtime._run_worker
    cancel = threading.Event()
    calls = []

    def blocked(argv, environment, directory, **kwargs):
        calls.append(argv)
        # Enter the real fixed scanner before blocking in filesystem validation.
        # There is no production test flag or replacement success result.
        source = f"""
import sys, os
sys.path.insert(0, {str(Path(runtime.__file__).parents[3])!r})
from core.inference.windows_sandbox import terminal_runtime_worker, terminal_runtime
def blocked_scan(*args):
    with open({str(tmp_path / "pid")!r}, 'w') as output:
        output.write(str(os.getpid()))
    read, write = os.pipe()
    os.read(read, 1)
    open({str(tmp_path / "unexpected-completion")!r}, 'w').close()
terminal_runtime.lpac._validate_runtime_trees = blocked_scan
sys.argv = [terminal_runtime_worker.__file__, sys.argv[1]]
raise SystemExit(terminal_runtime_worker.main())
"""
        return original(
            [argv[0], "-I", "-S", "-B", "-c", source, argv[-1]], environment, directory, **kwargs
        )

    def stop():
        deadline = time.monotonic() + 4
        while not (tmp_path / "pid").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        cancel.set()

    monkeypatch.setattr(runtime, "_run_worker", blocked)
    monkeypatch.setattr(runtime, "_inspect", lambda *args: pytest.fail("broker fallback"))
    observer = threading.Thread(target = stop) if reason == "cancel" else None
    if observer is not None:
        observer.start()
    try:
        with pytest.raises(WindowsRuntimeError) as caught:
            runtime.inspect_terminal_runtime(spec, timeout = 5 if observer else 1, cancel = cancel)
        expected = (
            "WINDOWS_SANDBOX_CANCELLED" if observer else "WINDOWS_SANDBOX_PREPARATION_TIMEOUT"
        )
        assert caught.value.code == expected
    finally:
        if observer is not None:
            observer.join(5)
    assert len(calls) == 1
    assert_dead(int((tmp_path / "pid").read_text()))
    assert not (tmp_path / "unexpected-completion").exists()
    assert not list(Path(spec.workdir).iterdir())


@pytest.mark.parametrize(
    "field",
    [
        "nonce",
        "pid",
        "schema",
        "request_digest",
        "argv",
        "workdir",
        "extra",
        "runtime_roots",
        "acl_roots",
    ],
)
def test_real_worker_response_tampering_is_rejected(spec, monkeypatch, field):
    original = runtime._run_worker

    def tamper(*args, **kwargs):
        data, pid = original(*args, **kwargs)
        value = json.loads(data)
        if field == "argv":
            value["result"]["argv"][0] = str(Path(spec.workdir) / "different.exe")
        elif field == "workdir":
            value["result"]["workdir"] = str(Path(spec.workdir).parent)
        elif field == "runtime_roots":
            value["result"]["runtime_roots"].append(str(Path(spec.workdir).parent))
        elif field == "acl_roots":
            value["result"]["acl_roots"] = value["result"]["runtime_roots"]
        else:
            value[field] = "wrong"
        return json.dumps(value).encode(), pid

    monkeypatch.setattr(runtime, "_run_worker", tamper)
    with pytest.raises(WindowsRuntimeError):
        runtime.inspect_terminal_runtime(spec)
    assert not list(Path(spec.workdir).iterdir())


def test_cleanup_failure_ownership_reaches_the_enclosing_launch_owner(spec, monkeypatch):
    failure = WindowsRuntimeError("WINDOWS_SANDBOX_CLEANUP_FAILED", "fixed failure")
    failure.retained_process = object()
    failure.retained_control_handles = (123,)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(runtime, "_run_worker", fail)
    with pytest.raises(WindowsRuntimeError) as caught:
        runtime.inspect_terminal_runtime(spec)
    assert caught.value is failure
    assert caught.value.retained_process is failure.retained_process
    assert caught.value.retained_control_handles == (123,)


def test_actual_failed_worker_channel_close_retains_handles_and_returns_no_result(
    spec, monkeypatch
):
    from core.inference.windows_sandbox import preparation

    api = runtime.lpac._api().kernel32
    original_close, original_pipe = api.CloseHandle, preparation._WorkerChannels.pipe
    targets, remaining = set(), ()

    def pipe(owner):
        read, write = original_pipe(owner)
        if not targets:
            targets.add(read)
        return read, write

    def close(handle):
        if handle in targets:
            ctypes.set_last_error(5)
            return False
        return original_close(handle)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(preparation._WorkerChannels, "pipe", pipe)
            patch.setattr(api, "CloseHandle", close)
            with pytest.raises(WindowsRuntimeError) as caught:
                runtime.inspect_terminal_runtime(spec)
            remaining = caught.value.retained_control_handles
            assert caught.value.code == "WINDOWS_SANDBOX_CLEANUP_FAILED"
            assert set(remaining) == targets
    finally:
        # These handles are explicitly transferred to the enclosing owner on
        # failure. The test is that owner; never discard or fabricate cleanup.
        for handle in remaining or tuple(targets):
            assert original_close(handle)
    assert not list(Path(spec.workdir).iterdir())
