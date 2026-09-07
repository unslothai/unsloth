# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native bounded Terminal preparation, independent of backend qualification."""

from dataclasses import replace
import ctypes
import json
import os
from pathlib import Path
import sys
import subprocess
import threading
import time

import pytest

from test_terminal_job import KWARGS, lpac, spawn_prepared_launch
from test_preparation import assert_dead, BACKEND
from core.inference.os_sandbox import ToolLaunchPlan
from core.inference.windows_sandbox import identity, terminal_launch as launch

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason = "Native Windows Terminal preparation"
)


@pytest.fixture
def spec(tmp_path, monkeypatch):
    local, work = tmp_path / "LocalAppData", tmp_path / "work"
    local.mkdir()
    work.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    yield ToolLaunchPlan(
        (
            str(Path(os.environ["SystemRoot"]) / "System32/cmd.exe"),
            "/d",
            "/c",
            "echo TERMINAL_READY",
        ),
        str(work),
        {},
        execution_kind = "terminal",
    )
    for owner in tuple(launch._pending_cleanup):
        owner.cleanup()
    assert not launch._pending_cleanup
    assert not list(local.rglob("*.json"))


def kwargs(prepared):
    return {**KWARGS, "cwd": prepared.workdir, "env": prepared.env}


def test_real_terminal_preparation_launch_and_cleanup(spec):
    prepared = launch.prepare_terminal_launch(spec)
    owner = prepared.spawn_callback.__self__
    profile, journal = Path(owner.identity.profile_folder), owner.reservation.path
    try:
        assert prepared.execution_record is None
        assert owner.pins.handles and journal.exists()
        assert all(not os.get_handle_inheritable(handle) for handle in owner.pins.handles.values())
        assert identity._read(journal)["purpose"] == "terminal"
        assert str(Path(sys.executable).parent) not in prepared.env["PATH"].split(os.pathsep)
        process = spawn_prepared_launch(prepared, **kwargs(prepared))
        assert process.stdout.readline().strip() == "TERMINAL_READY"
        assert process.wait(timeout = 5) == 0
        assert prepared.execution_record is None  # Only the qualified backend can attach one.
    finally:
        prepared.cleanup()
    assert not prepared.cleanup_diagnostics
    assert owner.closed and not owner.pins.handles
    assert not profile.exists() and not journal.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_kind", "python"),
        ("requested_mode", "limited"),
        ("requested_mode", "full"),
        ("close_fds", False),
        ("terminate_descendants", False),
    ],
)
def test_invalid_terminal_plan_starts_no_worker(spec, monkeypatch, field, value):
    monkeypatch.setattr(launch, "_run_worker", lambda *a, **kw: pytest.fail("worker started"))
    with pytest.raises(launch.WindowsRuntimeError):
        launch.prepare_terminal_launch(replace(spec, **{field: value}))


def test_cancelled_terminal_preparation_starts_nothing(spec, monkeypatch):
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(launch, "_run_worker", lambda *a, **kw: pytest.fail("worker started"))
    with pytest.raises(launch.WindowsRuntimeError, match = "cancelled"):
        launch.prepare_terminal_launch(spec, cancel = cancel)


def test_terminal_preparation_preserves_explicit_cancellation_signal(spec):
    cancel = threading.Event()
    prepared = launch.prepare_terminal_launch(spec, cancel = cancel)
    try:
        assert prepared.spawn_callback.__self__.cancel is cancel
    finally:
        prepared.cleanup()
    assert not prepared.cleanup_diagnostics


def test_terminal_launch_rejects_changed_argv_without_payload(spec):
    prepared = launch.prepare_terminal_launch(spec)
    owner = prepared.spawn_callback.__self__
    prepared.argv = (*prepared.argv[:-1], "echo BAD>payload-ran")
    try:
        with pytest.raises(launch.WindowsRuntimeError, match = "changed its prepared"):
            spawn_prepared_launch(prepared, **kwargs(prepared))
        assert owner.closed and not Path(spec.workdir, "payload-ran").exists()
        assert prepared.execution_record is None
    finally:
        prepared.cleanup()


def test_terminal_environment_removes_case_aliases_and_python_paths(spec):
    prepared = launch.prepare_terminal_launch(
        replace(
            spec,
            env = {
                "Path": "C:\\untrusted",
                "home": "C:\\secret",
                "tmp": "C:\\secret",
                "PYTHONPATH": "C:\\secret",
                "PYTHONHOME": "C:\\secret",
                "CUSTOM": "preserve",
            },
        )
    )
    try:
        assert prepared.env["CUSTOM"] == "preserve"
        assert not any(k.upper().startswith("PYTHON") for k in prepared.env)
        assert len({k.upper() for k in prepared.env}) == len(prepared.env)
        assert "C:\\secret" not in prepared.env.values()
        assert "C:\\untrusted" not in prepared.env["PATH"]
    finally:
        prepared.cleanup()


@pytest.mark.parametrize(
    "field",
    [
        "nonce",
        "pid",
        "request_digest",
        "schema",
        "argv",
        "runtime_roots",
        "sid",
        "journal",
        "traverse",
        "pins",
        "pin_handle",
        "swapped_handles",
    ],
)
def test_terminal_handoff_tampering_fails_closed_and_recovers(spec, monkeypatch, field):
    original = launch._run_worker
    seen = []

    def run(*args, **kwargs):
        transfer = kwargs["transfer"]

        def altered(process, data):
            seen.append(process.pid)
            value = json.loads(data)
            if field == "argv":
                value["result"]["argv"][-1] = "echo BAD>payload-ran"
            elif field == "runtime_roots":
                value["result"]["runtime_roots"] = [spec.workdir]
            elif field == "sid":
                value["identity"]["record"]["sid"] = "S-1-15-2-1"
            elif field == "journal":
                value["identity"]["path"] = str(
                    Path(spec.workdir)
                    / "python-bootstrap-v2"
                    / Path(value["identity"]["path"]).name
                )
            elif field == "traverse":
                value["identity"]["traverse"] = []
            elif field == "pins":
                value["pins"].pop()
            elif field == "pin_handle":
                value["pins"][-1][1] = 1
            elif field == "swapped_handles":
                value["pins"][0][1], value["pins"][-1][1] = (
                    value["pins"][-1][1],
                    value["pins"][0][1],
                )
            else:
                value[field] = "wrong"
            return transfer(process, json.dumps(value).encode())

        return original(*args, **{**kwargs, "transfer": altered})

    monkeypatch.setattr(launch, "_run_worker", run)
    with pytest.raises(launch.WindowsRuntimeError):
        launch.prepare_terminal_launch(spec)
    assert len(seen) == 1
    assert_dead(seen[0])
    assert not Path(spec.workdir, "payload-ran").exists()


@pytest.mark.parametrize("reason", ["timeout", "cancel"])
def test_terminal_interrupted_grant_worker_is_reaped_and_recovered(
    spec, tmp_path, monkeypatch, reason
):
    original = launch._run_worker
    marker = tmp_path / "granted-worker"
    cancel = threading.Event()
    source = f"""
import sys, os
from pathlib import Path
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox import terminal_prepare_worker
from core.inference.windows_sandbox import terminal_native as lpac
original = lpac._grant_modify
def blocked(path, sid):
    original(path, sid)
    Path({str(marker)!r}).write_text(str(os.getpid()))
    read, write = os.pipe()
    os.read(read, 1)
    Path({str(tmp_path / "late-write")!r}).touch()
lpac._grant_modify = blocked
sys.argv = [terminal_prepare_worker.__file__, sys.argv[1]]
raise SystemExit(terminal_prepare_worker.main())
"""

    def run(argv, *args, **kwargs):
        return original([argv[0], "-I", "-S", "-B", "-c", source, argv[-1]], *args, **kwargs)

    def stop():
        deadline = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        cancel.set()

    observer = threading.Thread(target = stop) if reason == "cancel" else None
    monkeypatch.setattr(launch, "_run_worker", run)
    if observer is not None:
        observer.start()
    try:
        with pytest.raises(launch.WindowsRuntimeError) as error:
            launch.prepare_terminal_launch(spec, timeout = 5 if observer else 1.5, cancel = cancel)
        assert error.value.code == (
            "WINDOWS_SANDBOX_CANCELLED" if observer else "WINDOWS_SANDBOX_PREPARATION_TIMEOUT"
        )
    finally:
        if observer is not None:
            observer.join(5)
    assert_dead(int(marker.read_text()))
    assert not (tmp_path / "late-write").exists()


def test_native_terminal_spawn_failure_never_executes_or_returns_record(spec, monkeypatch):
    prepared = launch.prepare_terminal_launch(
        replace(spec, argv = (*spec.argv[:-1], "echo BAD>payload-ran"))
    )
    owner = prepared.spawn_callback.__self__
    original = lpac._api().kernel32.CreateProcessW

    def create(application, *args):
        if application == prepared.argv[0]:
            ctypes.set_last_error(5)
            return False
        return original(application, *args)  # Fixed profile cleanup still runs.

    try:
        with monkeypatch.context() as patch:
            patch.setattr(lpac._api().kernel32, "CreateProcessW", create)
            with pytest.raises(OSError, match = "CreateProcessW"):
                spawn_prepared_launch(prepared, **kwargs(prepared))
        assert owner.closed and prepared.execution_record is None
        assert not Path(spec.workdir, "payload-ran").exists()
    finally:
        prepared.cleanup()


def test_terminal_workdir_write_and_host_read_denial_have_positive_control(spec, tmp_path):
    sentinel = tmp_path / "host secret.txt"
    sentinel.write_text("HOST_SECRET_CONTROL")
    command = "echo WRITABLE>created & type %HOST_SENTINEL%"
    selected = replace(
        spec, argv = (*spec.argv[:-1], command), env = {"HOST_SENTINEL": f'"{sentinel}"'}
    )
    control = subprocess.run(
        selected.argv,
        cwd = selected.workdir,
        env = {**os.environ, **selected.env},
        capture_output = True,
        text = True,
        timeout = 5,
    )
    assert control.returncode == 0 and "HOST_SECRET_CONTROL" in control.stdout
    Path(spec.workdir, "created").unlink()  # This test's positive-control output only.
    prepared = launch.prepare_terminal_launch(selected)
    try:
        process = spawn_prepared_launch(prepared, **kwargs(prepared))
        output = process.stdout.read()
        assert process.wait(timeout = 5) != 0
        assert "HOST_SECRET_CONTROL" not in output
        assert Path(spec.workdir, "created").read_text().strip() == "WRITABLE"
    finally:
        prepared.cleanup()


@pytest.mark.parametrize("reason", ["timeout", "cancel", "disconnect"])
def test_terminal_streams_before_owned_cleanup_reaps_process(spec, reason):
    prepared = launch.prepare_terminal_launch(
        replace(
            spec, argv = (*spec.argv[:-1], "echo STREAMING & for /l %i in (1,1,1000000000) do @rem")
        )
    )
    owner = prepared.spawn_callback.__self__
    process = None
    try:
        process = spawn_prepared_launch(prepared, **kwargs(prepared))
        assert process.stdout.readline().strip() == "STREAMING"
        assert process.poll() is None
        if reason == "timeout":
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout = 0.01)
        elif reason == "disconnect":
            process.stdout.close()
        # The shared tool loop invokes this same owned cleanup on cancellation,
        # timeout and disconnect; no process polling supervisor is introduced.
    finally:
        prepared.cleanup()
    assert process is not None
    assert_dead(process.pid)
    assert owner.closed and not prepared.cleanup_diagnostics


@pytest.mark.parametrize("failure", ["exit", "hang"])
def test_terminal_failure_after_acknowledged_handoff_recovers_everything(
    spec, monkeypatch, failure
):
    original = launch._run_worker
    transferred = []
    source = f"""
import sys, os
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox import terminal_prepare_worker
sys.argv = [terminal_prepare_worker.__file__, sys.argv[1]]
assert terminal_prepare_worker.main() == 0
if {failure!r} == 'hang':
    read, write = os.pipe()
    os.read(read, 1)
raise SystemExit(3)
"""

    def run(argv, *args, **kwargs):
        transfer = kwargs["transfer"]

        def capture(process, data):
            result = transfer(process, data)
            transferred.append((process.pid, transfer.__self__))
            return result

        return original(
            [argv[0], "-I", "-S", "-B", "-c", source, argv[-1]],
            *args,
            **{**kwargs, "transfer": capture},
        )

    monkeypatch.setattr(launch, "_run_worker", run)
    with pytest.raises(launch.WindowsRuntimeError):
        launch.prepare_terminal_launch(spec, timeout = 1.5)
    assert len(transferred) == 1
    pid, owner = transferred[0]
    assert_dead(pid)
    assert owner.closed and not owner.pins.handles
    assert not owner.reservation.path.exists()
    assert not Path(owner.identity.profile_folder).exists()


def test_terminal_retains_failed_native_channel_close_until_retry(spec, monkeypatch):
    from core.inference.windows_sandbox import preparation

    api = lpac._api().kernel32
    original_pipe, original_close = preparation._WorkerChannels.pipe, api.CloseHandle
    targets = set()
    owner = None

    def pipe(channels):
        read, write = original_pipe(channels)
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
            with pytest.raises(launch.WindowsRuntimeError) as error:
                launch.prepare_terminal_launch(spec)
            owner = error.value.retained_launch
            assert owner in launch._pending_cleanup and owner.handles == targets
            assert owner.reservation.path.exists() and owner.pins.handles and not owner.closed
            with pytest.raises(OSError):
                launch.prepare_terminal_launch(spec)
            assert len(launch._pending_cleanup) == 1
        owner.cleanup()
        assert owner.closed and not owner.handles and not owner.pins.handles
        assert not owner.reservation.path.exists()
    finally:
        if owner is not None:
            owner.cleanup()


def test_terminal_cancelled_after_preparation_never_starts_payload(spec):
    cancel = threading.Event()
    prepared = launch.prepare_terminal_launch(
        replace(spec, argv = (*spec.argv[:-1], "echo BAD>payload-ran")), cancel = cancel
    )
    cancel.set()
    try:
        with pytest.raises(launch.WindowsRuntimeError, match = "cancelled"):
            spawn_prepared_launch(prepared, **kwargs(prepared))
        assert prepared.execution_record is None
        assert not Path(spec.workdir, "payload-ran").exists()
    finally:
        prepared.cleanup()


def test_terminal_pins_non_system_shell_until_cleanup(spec, tmp_path):
    shell = tmp_path / "shell with spaces-é"
    shell.mkdir()
    executable = shell / "cmd.exe"
    executable.write_bytes(b"static fixture, never executed")
    prepared = launch.prepare_terminal_launch(replace(spec, argv = (str(executable),)))
    owner = prepared.spawn_callback.__self__
    try:
        assert executable in owner.pins.handles
        with pytest.raises(PermissionError):
            executable.write_bytes(b"replacement")
        with pytest.raises(PermissionError):
            shell.rename(tmp_path / "renamed-shell")
    finally:
        prepared.cleanup()
    executable.write_bytes(b"replacement after cleanup")
    assert owner.closed and not owner.pins.handles


def test_terminal_cleanup_waits_for_native_creation_ownership(spec, monkeypatch):
    prepared = launch.prepare_terminal_launch(spec)
    owner = prepared.spawn_callback.__self__
    original = lpac._api().kernel32.CreateProcessW
    entered, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    failures, processes = [], []

    def create(application, *args):
        result = original(application, *args)
        if application == prepared.argv[0] and result:
            entered.set()
            assert release.wait(4)
        return result

    def start():
        try:
            processes.append(spawn_prepared_launch(prepared, **kwargs(prepared)))
        except BaseException as error:
            failures.append(error)

    def stop():
        try:
            owner.cleanup()
            cleaned.set()
        except BaseException as error:
            failures.append(error)

    threads = []
    try:
        with monkeypatch.context() as patch:
            patch.setattr(lpac._api().kernel32, "CreateProcessW", create)
            threads.append(threading.Thread(target = start))
            threads[0].start()
            assert entered.wait(3)
            threads.append(threading.Thread(target = stop))
            threads[1].start()
            assert not cleaned.wait(0.1)
            assert owner.identity.sid and owner.pins.handles and owner.reservation.path.exists()
            release.set()
            for thread in threads:
                thread.join(5)
            assert all(not thread.is_alive() for thread in threads)
        assert not failures and len(processes) == 1
        assert cleaned.is_set() and owner.closed
        assert_dead(processes[0].pid)
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
        prepared.cleanup()


def test_terminal_invocations_use_distinct_fresh_private_temp(spec):
    first = launch.prepare_terminal_launch(
        replace(spec, argv = (*spec.argv[:-1], 'echo PRIVATE>"%TMP%\\per-invocation"'))
    )
    second = None
    try:
        process = spawn_prepared_launch(first, **kwargs(first))
        assert process.wait(timeout = 5) == 0
        marker = Path(first.env["TMP"]) / "per-invocation"
        assert marker.read_text().strip() == "PRIVATE"
        second = launch.prepare_terminal_launch(
            replace(
                spec,
                argv = (
                    *spec.argv[:-1],
                    'if exist "%TMP%\\per-invocation" (exit /b 7) else (echo FRESH)',
                ),
            )
        )
        assert second.env["TMP"] != first.env["TMP"]
        process = spawn_prepared_launch(second, **kwargs(second))
        assert process.stdout.readline().strip() == "FRESH"
        assert process.wait(timeout = 5) == 0
        assert marker.exists()  # First invocation's temp still exists during the second.
    finally:
        if second is not None:
            second.cleanup()
        first.cleanup()
    assert not Path(first.env["TMP"]).exists()
    assert second is not None and not Path(second.env["TMP"]).exists()


@pytest.mark.parametrize("flags", [("/c",), ("/d", "/c"), ("/d", "/s", "/c")])
def test_terminal_preserves_quoted_cmd_metacharacters(spec, flags):
    command = 'echo "A & B">"quoted & file-é.txt"'
    selected = replace(spec, argv = (spec.argv[0], *flags, command))
    prepared = launch.prepare_terminal_launch(selected)
    try:
        process = spawn_prepared_launch(prepared, **kwargs(prepared))
        output = process.stdout.read()
        assert process.wait(timeout = 5) == 0, output
        assert Path(spec.workdir, "quoted & file-é.txt").read_text().strip() == '"A & B"'
        assert os.path.samefile(prepared.argv[0], selected.argv[0])
        assert prepared.argv[1:] == selected.argv[1:]
    finally:
        prepared.cleanup()


@pytest.mark.skipif(
    os.environ.get("UNSLOTH_TEST_TERMINAL_COMPATIBILITY") != "1",
    reason = "Opt-in native qualification diagnostic; this OS failed Terminal compatibility",
)
def test_terminal_runs_quoted_workdir_batch_path(spec):
    script = Path(spec.workdir) / "quoted script-é.cmd"
    script.write_bytes(b"@echo off\r\necho BATCH_OK\r\n")
    prepared = launch.prepare_terminal_launch(replace(spec, argv = (*spec.argv[:-1], f'"{script}"')))
    try:
        # Explicit fixture control, not a retry of a failed sandbox payload.
        control = subprocess.run(
            launch._command_line(prepared.argv),
            executable = prepared.argv[0],
            cwd = prepared.workdir,
            env = prepared.env,
            capture_output = True,
            text = True,
            timeout = 5,
            creationflags = subprocess.CREATE_NO_WINDOW,
        )
        assert control.returncode == 0, control.stderr
        assert control.stdout.strip() == "BATCH_OK"
        process = spawn_prepared_launch(prepared, **kwargs(prepared))
        output = process.stdout.read()
        assert process.wait(timeout = 5) == 0, output
        assert output.strip() == "BATCH_OK"
    finally:
        prepared.cleanup()


@pytest.mark.skipif(
    os.environ.get("UNSLOTH_TEST_TERMINAL_COMPATIBILITY") != "1",
    reason = "Opt-in native qualification diagnostic; this OS failed Terminal compatibility",
)
def test_terminal_can_launch_same_sandbox_shell_child(spec):
    command = f'"{spec.argv[0]}" /d /c echo CHILD_OK'
    prepared = launch.prepare_terminal_launch(replace(spec, argv = (*spec.argv[:-1], command)))
    try:
        control = subprocess.run(
            launch._command_line(prepared.argv),
            executable = prepared.argv[0],
            cwd = prepared.workdir,
            env = prepared.env,
            capture_output = True,
            text = True,
            timeout = 5,
            creationflags = subprocess.CREATE_NO_WINDOW,
        )
        assert control.returncode == 0, control.stderr
        assert control.stdout.strip() == "CHILD_OK"
        process = spawn_prepared_launch(prepared, **kwargs(prepared))
        output = process.stdout.read()
        assert process.wait(timeout = 5) == 0, output
        assert output.strip() == "CHILD_OK"
    finally:
        prepared.cleanup()
