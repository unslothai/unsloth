# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Live production creation primitive, not complete backend qualification."""

import ctypes
from ctypes import wintypes as W
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import time

import pefile
import pytest

import test_python_host as host_fixtures
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox import native_process
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

python_runtime = host_fixtures.python_runtime
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows process lane")


def test_production_creation_starts_real_python_under_verified_gate(
    python_runtime, tmp_path, monkeypatch
):
    original = native_process._create
    observed = []

    def create(*args):
        process = original(*args)
        api = lpac._api().kernel32
        # Both processes are already members on return from CreateProcess,
        # without the test or launcher calling AssignProcessToJobObject.
        belongs = W.BOOL()
        assert api.IsProcessInJob(
            process._handle, process._unsloth_job._handle, ctypes.byref(belongs)
        )
        assert belongs.value and process.poll() is None
        handle = api.OpenProcess(0x100000, False, process.pid)
        assert handle
        observed.append(handle)
        return process

    monkeypatch.setattr(native_process, "_create", create)
    monkeypatch.setattr(
        lpac._api().kernel32,
        "AssignProcessToJobObject",
        lambda *args: pytest.fail("Job ownership was assigned after creation"),
    )
    try:
        with host_fixtures.python_launch(
            python_runtime,
            tmp_path,
            "import asyncio, sqlite3; assert asyncio.run(asyncio.sleep(0, result=7)) == 7; print('PRODUCTION_CREATION_OK')",
            production_creation = True,
        ) as launch:
            assert len(observed) == 2
            assert lpac._api().kernel32.WaitForSingleObject(observed[0], 0) == 0
            assert "PRODUCTION_CREATION_OK" in host_fixtures.run(launch)
    finally:
        for handle in observed:
            lpac._api().kernel32.CloseHandle(handle)


@pytest.mark.parametrize("case", ["cancel", "expired", "pseudo", "duplicate", "invalid_handle"])
def test_invalid_creation_plan_never_creates_a_process(case, monkeypatch):
    import threading

    cancel = threading.Event()
    if case == "cancel":
        cancel.set()
    kwargs = {"stdin": 1, "stdout": 2, "control_handles": (3,), "cancel": cancel}
    if case == "expired":
        kwargs["timeout"] = 0
    if case == "pseudo":
        kwargs["stdin"] = -1
    if case == "duplicate":
        kwargs["control_handles"] = (1,)
    monkeypatch.setattr(
        native_process, "_create", lambda *args: pytest.fail("Invalid plan spawned")
    )
    with pytest.raises(WindowsRuntimeError):
        native_process.create_suspended_host(
            r"E:\trusted\host.exe",
            (r"E:\trusted\host.exe",),
            None,
            {},
            r"E:\private\temp",
            **kwargs,
        )


def test_real_noninheritable_channel_is_rejected_without_spawn(monkeypatch):
    import msvcrt
    import os

    descriptors = [os.open(os.devnull, os.O_RDWR) for _ in range(3)]
    try:
        for fd in descriptors:
            os.set_inheritable(fd, False)
        handles = tuple(msvcrt.get_osfhandle(fd) for fd in descriptors)
        monkeypatch.setattr(
            native_process, "_create", lambda *args: pytest.fail("Invalid channel spawned")
        )
        with pytest.raises(WindowsRuntimeError, match = "not inheritable"):
            native_process.create_suspended_host(
                r"E:\trusted\host.exe",
                (r"E:\trusted\host.exe",),
                None,
                {},
                r"E:\private\temp",
                stdin = handles[0],
                stdout = handles[1],
                control_handles = (handles[2],),
            )
    finally:
        for fd in descriptors:
            os.close(fd)


@pytest.mark.parametrize(
    "failure", ["target_creation", "token_duplicate", "token_attach", "cancel_after_target"]
)
def test_creation_failure_reaps_every_started_process(
    python_runtime, tmp_path, monkeypatch, failure
):
    import threading
    import native_support

    original = native_process._create
    entrypoint = native_process.create_suspended_host
    cancel, observed = threading.Event(), []

    def create(*args):
        if failure == "target_creation" and args[-1] is None:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_NATIVE_LAUNCH_FAILED", "fixed target failure"
            )
        process = original(*args)
        handle = lpac._api().kernel32.OpenProcess(0x100001, False, process.pid)
        assert handle
        observed.append(handle)
        if failure == "cancel_after_target" and args[-1] is None:
            cancel.set()
        return process

    monkeypatch.setattr(native_process, "_create", create)
    monkeypatch.setattr(
        native_support,
        "create_suspended_host",
        lambda *a, **kw: entrypoint(*a, **kw, cancel = cancel),
    )
    api = lpac._api()
    if failure in ("token_duplicate", "token_attach"):
        name = "DuplicateTokenEx" if failure == "token_duplicate" else "SetThreadToken"

        def fail(*args):
            ctypes.set_last_error(5)
            return False

        monkeypatch.setattr(api.advapi32, name, fail)
    try:
        with pytest.raises((WindowsRuntimeError, OSError)):
            with host_fixtures.python_launch(
                python_runtime, tmp_path, "print('MUST_NOT_RUN')", production_creation = True
            ):
                pytest.fail("Failed native creation returned a launch")
        assert len(observed) == (2 if failure in ("token_attach", "cancel_after_target") else 1)
        for handle in observed:
            assert api.kernel32.WaitForSingleObject(handle, 5000) == 0
    finally:
        for handle in observed:
            api.kernel32.TerminateProcess(handle, 1)
            api.kernel32.WaitForSingleObject(handle, 5000)
            api.kernel32.CloseHandle(handle)


@pytest.mark.parametrize("stop_at", [1, 2], ids = ["donor-created", "target-created"])
def test_parent_death_before_create_returns_kills_job_member(python_runtime, tmp_path, stop_at):
    tests = Path(__file__).parent
    backend = tests.parents[1] / "backend"
    marker, identity_marker = tmp_path / "created", tmp_path / "identity"
    runtime = python_runtime
    source = f"""
import sys, os, ctypes, json
from pathlib import Path
from types import SimpleNamespace
sys.path[:0] = [{str(tests)!r}, {str(backend)!r}]
sys.path.extend([{str(Path(pefile.__file__).parent)!r}, {sysconfig.get_path("purelib")!r}])
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox.content import RuntimeContentStore
from core.inference.windows_sandbox.preparation import _decode
from core.inference.windows_sandbox.runtime import RuntimeDescriptor
from test_python_host import python_launch
from core.inference.windows_sandbox.identity import InvocationReservation
original_identity = InvocationReservation.create
def identity(self, workdir, **kwargs):
    result = original_identity(self, workdir, **kwargs)
    Path({str(identity_marker)!r}).write_text(result.manifest_path)
    return result
InvocationReservation.create = identity
kernel = lpac._api().kernel32
original_create = kernel.CreateProcessW
calls = 0
def create(*args):
    global calls
    result = original_create(*args)
    if result:
        calls += 1
        if calls == {stop_at}:
            info = ctypes.cast(args[-1], ctypes.POINTER(lpac._PROCESS_INFORMATION)).contents
            Path({str(marker)!r}).write_text(json.dumps([os.getpid(), info.dwProcessId]))
            r, w = os.pipe()
            os.read(r, 1)
    return result
kernel.CreateProcessW = create
runtime = SimpleNamespace(
    store=RuntimeContentStore({str(runtime.store.root)!r}), digest={runtime.digest!r},
    descriptor=_decode(RuntimeDescriptor, json.loads({json.dumps(asdict(runtime.descriptor))!r})),
    binary=Path({str(runtime.binary)!r}), native_images={runtime.native_images!r})
with python_launch(runtime, Path({str(tmp_path)!r}), "print('MUST_NOT_RUN')", production_creation=True):
    raise AssertionError('creation fixture escaped its stop')
"""
    owner = subprocess.Popen(
        # Observe the actual broker, not a Windows venv redirector process.
        [sys._base_executable, "-I", "-c", source],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    child_handle = None
    try:
        deadline = time.monotonic() + 30
        while not marker.exists() and owner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), owner.communicate(timeout = 5)
        broker_pid, child_pid = json.loads(marker.read_text())
        assert broker_pid == owner.pid
        child_handle = lpac._api().kernel32.OpenProcess(0x100001, False, child_pid)
        assert child_handle
        owner.kill()
        output, errors = owner.communicate(timeout = 10)
        assert b"MUST_NOT_RUN" not in output
        assert lpac._api().kernel32.WaitForSingleObject(child_handle, 5000) == 0
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout = 10)
        if child_handle:
            api = lpac._api().kernel32
            api.TerminateProcess(child_handle, 1)
            assert api.WaitForSingleObject(child_handle, 5000) == 0
            api.CloseHandle(child_handle)
        runtime.store.recover_readers()
        if identity_marker.exists():
            _cleanup_fixture_identity(Path(identity_marker.read_text()), tmp_path)


def _cleanup_fixture_identity(manifest, boundary):
    """Recover only the exact identity created by this killed fixture."""
    from core.inference.windows_sandbox.identity import (
        InvocationRecipe,
        cleanup_recipe,
        _derived_sid,
    )

    data = json.loads(manifest.read_bytes())
    recipe = InvocationRecipe(data["moniker"], data["owner_pid"], data["owner_created"])
    assert manifest.name == recipe.filename()
    assert data["workdir"] == str(boundary / "work")
    assert lpac._process_identity(data["owner_pid"]) != (data["owner_pid"], data["owner_created"])
    # The diagnostic driver is the fixture's sole grant outside its recorded workdir.
    with _derived_sid(recipe.moniker) as (sid, _):
        lpac._revoke_sid(str(boundary / "driver.exe"), sid)
    cleanup_recipe(recipe, str(manifest))
    assert not manifest.exists()


def test_process_adapter_failure_reaps_unreturned_native_process(
    python_runtime, tmp_path, monkeypatch
):
    api = lpac._api().kernel32
    original = api.CreateProcessW
    observed = []

    def create(*args):
        result = original(*args)
        if result:
            info = ctypes.cast(args[-1], ctypes.POINTER(lpac._PROCESS_INFORMATION)).contents
            handle = api.OpenProcess(0x100001, False, info.dwProcessId)
            assert handle
            observed.append(handle)
        return result

    def fail(*args):
        raise RuntimeError("fixed process adapter failure")

    monkeypatch.setattr(api, "CreateProcessW", create)
    monkeypatch.setattr(lpac, "WindowsLpacProcess", fail)
    try:
        with pytest.raises(RuntimeError, match = "fixed process adapter failure"):
            with host_fixtures.python_launch(
                python_runtime, tmp_path, "print('MUST_NOT_RUN')", production_creation = True
            ):
                pytest.fail("Unowned process was returned")
        assert len(observed) == 1
        assert api.WaitForSingleObject(observed[0], 5000) == 0
    finally:
        for handle in observed:
            api.TerminateProcess(handle, 1)
            api.WaitForSingleObject(handle, 5000)
            api.CloseHandle(handle)
