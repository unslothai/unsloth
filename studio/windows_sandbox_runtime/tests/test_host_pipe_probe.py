# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Live host endpoint controls; unavailable endpoints cannot prove isolation."""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes as W
import secrets
import sys
from types import SimpleNamespace

import pytest

from test_launch import run_harness, installed_runtime, runtime_wheel
from core.inference.windows_sandbox import probe
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows named-pipe probe")


@contextmanager
def host_owner():
    owner = SimpleNamespace(nonce = secrets.token_bytes(32), handles = set())
    api = probe._host_pipe_api()
    try:
        yield owner
    finally:
        for handle in tuple(owner.handles):
            assert api.CloseHandle(handle)
            owner.handles.remove(handle)
        # The OS removes this exact instance after its last handle closes.
        handle = api.CreateFileW(
            probe.HOST_PIPE_PREFIX + owner.nonce.hex(), 0xC0000000, 0, None, 3, 0, None
        )
        error = ctypes.get_last_error()
        if handle != ctypes.c_void_p(-1).value:
            assert api.CloseHandle(handle)
        assert handle == ctypes.c_void_p(-1).value and error == 2, error


def test_host_pipe_controls_exchange_bytes_and_release_instance():
    with host_owner() as owner:
        server = probe._prepare_host_pipe(owner)
        assert owner.handles == {server}
        probe._verify_host_pipe(owner, server)
        assert len(owner.handles) == 2
        api = probe._host_pipe_api()
        api.GetHandleInformation.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD)]
        api.GetHandleInformation.restype = W.BOOL
        for handle in owner.handles:
            flags = W.DWORD()
            assert api.GetHandleInformation(handle, ctypes.byref(flags))
            assert not flags.value & 1, "Probe endpoint is inheritable"


def test_unrestricted_host_cannot_pass_pipe_denial_probe():
    with host_owner() as owner:
        probe._prepare_host_pipe(owner)
        with pytest.raises(AssertionError, match = "Host named pipe was accessible"):
            exec(
                probe._HOST_PIPE_SOURCE,
                dict(
                    kernel = probe._host_pipe_api(),
                    ctypes = ctypes,
                    checks = [],
                    HOST_PIPE_PATH = probe.HOST_PIPE_PREFIX + owner.nonce.hex(),
                ),
            )


def test_pipe_name_collision_never_adopts_or_closes_original_instance():
    with host_owner() as owner:
        server = probe._prepare_host_pipe(owner)
        with pytest.raises(WindowsRuntimeError, match = "creation"):
            probe._prepare_host_pipe(owner)
        assert owner.handles == {server}
        probe._verify_host_pipe(owner, server)


def test_missing_pipe_cannot_pass_denial_probe():
    with host_owner() as owner:
        with pytest.raises(AssertionError, match = "Host pipe denial failed"):
            exec(
                probe._HOST_PIPE_SOURCE,
                dict(
                    kernel = probe._host_pipe_api(),
                    ctypes = ctypes,
                    checks = [],
                    HOST_PIPE_PATH = probe.HOST_PIPE_PREFIX + owner.nonce.hex(),
                ),
            )


@pytest.mark.parametrize("error", [2, 6, 87, 231])
def test_other_pipe_errors_are_not_denial(error):
    def opened(*args):
        ctypes.set_last_error(error)
        return ctypes.c_void_p(-1).value

    checks = []
    with pytest.raises(AssertionError, match = "Host pipe denial failed"):
        exec(
            probe._HOST_PIPE_SOURCE,
            dict(
                kernel = SimpleNamespace(CreateFileW = opened),
                ctypes = ctypes,
                checks = checks,
                HOST_PIPE_PATH = "fixed test endpoint",
            ),
        )
    assert not checks


@pytest.mark.parametrize("operation", ["WriteFile", "PeekNamedPipe", "ReadFile"])
def test_host_control_checks_native_io_results(monkeypatch, operation):
    with host_owner() as owner:
        server = probe._prepare_host_pipe(owner)
        api = probe._host_pipe_api()
        original = getattr(api, operation)

        def fail(*args):
            assert original(*args)
            return 0

        with monkeypatch.context() as patch:
            patch.setattr(api, operation, fail)
            with pytest.raises(WindowsRuntimeError, match = "control failed"):
                probe._verify_host_pipe(owner, server)
        assert len(owner.handles) == 2


@pytest.mark.parametrize("mode", ["short_write", "short_read", "corrupt_read"])
def test_host_control_checks_byte_counts_and_contents(monkeypatch, mode):
    with host_owner() as owner:
        server = probe._prepare_host_pipe(owner)
        api = probe._host_pipe_api()
        operation = "WriteFile" if mode == "short_write" else "ReadFile"
        original = getattr(api, operation)

        def corrupt(handle, buffer, size, count, overlap):
            result = original(handle, buffer, size, count, overlap)
            assert result
            if mode == "corrupt_read":
                buffer[0] = bytes([buffer.raw[0] ^ 1])
            else:
                ctypes.cast(count, ctypes.POINTER(W.DWORD)).contents.value -= 1
            return result

        with monkeypatch.context() as patch:
            patch.setattr(api, operation, corrupt)
            with pytest.raises(WindowsRuntimeError, match = "control failed"):
                probe._verify_host_pipe(owner, server)


@pytest.mark.parametrize(
    "mode", ["prepare_after_create", "missing_endpoint", "host_open", "positive_control", "close"]
)
def test_pipe_control_failure_blocks_observations_and_preserves_cleanup(
    installed_runtime, tmp_path, mode
):
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
from core.inference.windows_sandbox import probe,launch,identity
import ctypes
owners,servers=[],[]
original_prepare,original_verify = probe._prepare_host_pipe,probe._verify_host_pipe
kernel=probe._host_pipe_api()
original_close=kernel.CloseHandle
def prepare(owner):
    owners.append(owner)
    server=original_prepare(owner)
    servers.append(server)
    if {mode!r} == 'prepare_after_create':
        raise probe.WindowsRuntimeError('WINDOWS_SANDBOX_PROBE_FAILED','injected pipe preparation failure')
    if {mode!r} == 'missing_endpoint':
        owner._close_handle(server)
    return server
def verify(owner,server):
    if {mode!r} == 'host_open':
        owner._close_handle(server)
    original_verify(owner,server)
    if {mode!r} == 'positive_control':
        raise probe.WindowsRuntimeError('WINDOWS_SANDBOX_PROBE_FAILED','injected pipe positive control failure')
    if {mode!r} == 'close':
        def close(handle):
            if handle == server:
                ctypes.set_last_error(5)
                return 0
            return original_close(handle)
        kernel.CloseHandle=close
probe._prepare_host_pipe,probe._verify_host_pipe=prepare,verify
try:
    try:
        probe.run_python_probe(sys.executable,root/'cache')
    except probe.WindowsRuntimeError as error:
        assert len(owners)==1
        owner=owners[0]
        assert owner.started == ({mode!r} != 'prepare_after_create')
        if {mode!r} == 'close':
            assert error.code == 'WINDOWS_SANDBOX_CLEANUP_FAILED',str(error)
            assert error.retained_launch is owner
            assert owner in launch._pending_cleanup and servers[0] in owner.handles
            assert not owner.closed
        else:
            assert error.code == 'WINDOWS_SANDBOX_PROBE_FAILED',str(error)
            assert owner.closed and not owner.handles
            if {mode!r} == 'missing_endpoint':
                assert 'Host pipe denial failed' in str(error),str(error)
            if {mode!r} == 'host_open':
                assert 'Host named-pipe open control failed' in str(error),str(error)
    else:
        raise AssertionError('Failed pipe returned observations')
finally:
    probe._prepare_host_pipe,probe._verify_host_pipe=original_prepare,original_verify
    kernel.CloseHandle=original_close
    for owner in owners:
        owner.cleanup()
assert owner.closed and not owner.handles and not launch._pending_cleanup
assert not owner.pins.handles and not owner.file_pins.handles
assert not owner.reservation.path.exists()
with identity._derived_sid(owner.reservation.recipe.moniker) as (_,sid):
    assert identity._profile_path(sid) is None
assert not list((root/'cache'/'.readers').iterdir()) and not list(work.iterdir())
handle=kernel.CreateFileW(probe.HOST_PIPE_PREFIX+owner.nonce.hex(),0xc0000000,0,None,3,0,None)
error=ctypes.get_last_error()
if handle != ctypes.c_void_p(-1).value:
    assert kernel.CloseHandle(handle)
assert handle == ctypes.c_void_p(-1).value and error==2,error
print('PIPE_FAILURE_CLEANED_WITHOUT_OBSERVATIONS')
""",
    )
    assert "PIPE_FAILURE_CLEANED_WITHOUT_OBSERVATIONS" in output
