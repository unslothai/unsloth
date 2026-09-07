# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Actual native gate enforcement. Not Python/runtime qualification."""

from dataclasses import replace
import ctypes
from ctypes import wintypes as W
import os
from pathlib import Path
import statistics
import sys
import threading
import time

import pytest

from native_support import native_launch
from core.inference.windows_sandbox import protocol
from core.inference.windows_sandbox.protocol import (
    authorize_startup,
    _pipe_api,
    STATUS,
    acknowledgement,
    parse_startup_status,
)
from core.inference.windows_sandbox.profiles import WindowsRuntimeError


@pytest.fixture(scope = "module")
def gate_binary():
    if os.name != "nt":
        pytest.skip("Windows native gate lane")
    path = Path(os.environ["UNSLOTH_TEST_GATE_BINARY"]).resolve()
    assert path.is_file(), "Explicit development/CI native gate build is required"
    return path


def authorize(launch, **kwargs):
    launch.handles.difference_update((launch.status, launch.acknowledgement))
    return authorize_startup(
        launch.process,
        launch.status,
        launch.acknowledgement,
        kwargs.pop("binding", launch.binding),
        timeout = kwargs.pop("timeout", 5),
        **kwargs,
    )


def test_native_gate_enters_lpac_after_supported_donor_drop(gate_binary, tmp_path):
    with native_launch(gate_binary, tmp_path) as launch:
        assert not launch.sentinel.exists()
        assert authorize(launch) == launch.binding
        assert launch.process.wait(timeout = 5) == 0
        assert launch.sentinel.read_bytes() == b"AFTER_GATE"


def test_native_gate_accepts_live_thread_without_impersonation(gate_binary, tmp_path):
    with native_launch(gate_binary, tmp_path, thread_mode = "clean-thread") as launch:
        authorize(launch)
        assert launch.process.wait(timeout = 5) == 0
        assert launch.sentinel.read_bytes() == b"AFTER_GATE"


def test_native_gate_detects_impersonation_on_other_live_thread(gate_binary, tmp_path):
    with native_launch(gate_binary, tmp_path, thread_mode = "retained-thread") as launch:
        with pytest.raises(WindowsRuntimeError, match = "stage 8"):
            authorize(launch)
        assert launch.process.poll() is not None
        assert not launch.sentinel.exists()


@pytest.mark.parametrize(
    "invalid,stage",
    [
        ({"attach_token": False}, 2),
        ({"capability": "internetClient"}, 2),
        ({"limit": 2}, 6),
        ({"optout": False}, 7),
        ({"aap_granted": False}, 3),
    ],
)
def test_native_gate_rejects_wrong_authority_before_payload(gate_binary, tmp_path, invalid, stage):
    with native_launch(gate_binary, tmp_path, **invalid) as launch:
        with pytest.raises(WindowsRuntimeError, match = f"stage {stage}"):
            authorize(launch)
        assert launch.process.poll() is not None
        assert not launch.sentinel.exists()


@pytest.mark.parametrize("field", ["nonce", "content_digest"])
def test_parent_binding_rejection_never_acknowledges_payload(gate_binary, tmp_path, field):
    with native_launch(gate_binary, tmp_path) as launch:
        with pytest.raises(WindowsRuntimeError, match = "PROTOCOL_MISMATCH"):
            authorize(launch, binding = replace(launch.binding, **{field: b"x" * 32}))
        assert launch.process.poll() is not None
        assert not launch.sentinel.exists()


def test_cancelled_native_startup_never_runs_payload(gate_binary, tmp_path):
    cancelled = threading.Event()
    cancelled.set()
    with native_launch(gate_binary, tmp_path) as launch:
        with pytest.raises(WindowsRuntimeError, match = "CANCELLED"):
            authorize(launch, cancel = cancelled)
        assert launch.process.poll() is not None
        assert not launch.sentinel.exists()


def read_ready(launch, *, consume = True):
    """Bounded test observer; never acknowledges or establishes availability."""
    api = _pipe_api()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        available = W.DWORD()
        assert api.PeekNamedPipe(launch.status, None, 0, None, ctypes.byref(available), None)
        if available.value:
            assert available.value == STATUS.size
            data, count = ctypes.create_string_buffer(STATUS.size), W.DWORD()
            if consume:
                assert api.ReadFile(launch.status, data, len(data), ctypes.byref(count), None)
            else:
                assert api.PeekNamedPipe(
                    launch.status, data, len(data), ctypes.byref(count), None, None
                )
            assert count.value == STATUS.size
            parse_startup_status(data.raw, launch.binding)
            return
        time.sleep(0.005)
    pytest.fail("Native driver never reached READY")


def test_complete_status_without_eof_times_out_and_kills_owner(gate_binary, tmp_path):
    with native_launch(gate_binary, tmp_path, hold_status_writer = True) as launch:
        read_ready(launch, consume = False)
        with pytest.raises(WindowsRuntimeError, match = "STARTUP_TIMEOUT"):
            authorize(launch, timeout = 0.3)
        assert launch.process.poll() is not None
        assert not launch.sentinel.exists()


@pytest.mark.parametrize("attack", ["empty", "truncated", "wrong_binding", "excess"])
def test_native_rejects_missing_or_invalid_parent_ack(gate_binary, tmp_path, attack):
    with native_launch(gate_binary, tmp_path) as launch:
        read_ready(launch)
        assert launch.process.poll() is None
        assert not launch.sentinel.exists()
        data = acknowledgement(launch.binding)
        if attack == "empty":
            data = b""
        elif attack == "truncated":
            data = data[:-1]
        elif attack == "wrong_binding":
            data = acknowledgement(replace(launch.binding, nonce = b"x" * 32))
        elif attack == "excess":
            data += b"x"
        api, count = _pipe_api(), W.DWORD()
        if data:
            assert api.WriteFile(launch.acknowledgement, data, len(data), ctypes.byref(count), None)
            assert count.value == len(data)
        assert api.CloseHandle(launch.acknowledgement)
        launch.handles.remove(launch.acknowledgement)
        assert launch.process.wait(timeout = 5) == 94
        assert not launch.sentinel.exists()


def test_closing_job_while_waiting_for_ack_terminates_native_owner(gate_binary, tmp_path):
    with native_launch(gate_binary, tmp_path) as launch:
        read_ready(launch)
        assert launch.process.poll() is None
        launch.process._unsloth_job.close()
        # Kill-on-close's exit status need not be nonzero. The process handle
        # becoming signaled and the absent sentinel are the lifecycle proof.
        launch.process.wait(timeout = 5)
        assert launch.process.poll() is not None
        assert not launch.sentinel.exists()


def test_failed_reaping_retains_ack_writer_and_cannot_release_payload(
    gate_binary, tmp_path, monkeypatch
):
    with native_launch(gate_binary, tmp_path) as launch:
        api = _pipe_api()

        class CloseFailure:
            def __getattr__(self, name):
                return getattr(api, name)

            def CloseHandle(self, handle):
                if handle == launch.acknowledgement:
                    ctypes.set_last_error(5)
                    return False
                return api.CloseHandle(handle)

        def failed_wait(**_kwargs):
            raise TimeoutError("injected reaping failure")

        with monkeypatch.context() as patch:
            patch.setattr(protocol, "_pipe_api", CloseFailure)
            patch.setattr(launch.process, "terminate", lambda: None)
            patch.setattr(launch.process, "wait", failed_wait)
            with pytest.raises(WindowsRuntimeError, match = "CLEANUP_FAILED") as failure:
                authorize(launch)
        retained = failure.value.retained_control_handles
        launch.handles.update(retained)
        assert retained == (launch.acknowledgement,)
        assert launch.process.poll() is None
        assert not launch.sentinel.exists()
        # Restore real APIs and reap before releasing the retained writer.
        launch.process.terminate()
        launch.process.wait(timeout = 5)
        assert not launch.sentinel.exists()


def test_twenty_native_gate_launches_report_component_cost(gate_binary, tmp_path, record_property):
    samples = []
    for index in range(20):
        directory = tmp_path / str(index)
        directory.mkdir()
        start = time.perf_counter()
        with native_launch(gate_binary, directory) as launch:
            authorize(launch)
            samples.append((time.perf_counter() - start) * 1000)
            assert launch.process.wait(timeout = 5) == 0
            assert launch.sentinel.read_bytes() == b"AFTER_GATE"
        assert launch.sentinel.read_bytes() == b"AFTER_GATE"
    record_property("evidence", "NATIVE_GATE_COMPONENT_ONLY_NOT_PYTHON_STARTUP")
    record_property("windows_build", sys.getwindowsversion().build)
    record_property("samples_ms", repr(samples))
    record_property("median_ms", statistics.median(samples))
    record_property("p95_nearest_rank_ms", sorted(samples)[18])


@pytest.mark.parametrize("channel", ["status", "acknowledgement"])
def test_failed_control_close_retains_only_unclosed_handle(
    gate_binary, tmp_path, monkeypatch, channel
):
    with native_launch(gate_binary, tmp_path) as launch:
        api = _pipe_api()
        failed_handle = getattr(launch, channel)
        closed = []

        class CloseFailure:
            def __getattr__(self, name):
                return getattr(api, name)

            def CloseHandle(self, handle):
                if handle == failed_handle:
                    ctypes.set_last_error(5)
                    return False
                result = api.CloseHandle(handle)
                if result:
                    closed.append(handle)
                return result

        try:
            with monkeypatch.context() as patch:
                patch.setattr(protocol, "_pipe_api", CloseFailure)
                with pytest.raises(WindowsRuntimeError, match = "CLEANUP_FAILED") as caught:
                    authorize(launch)
            assert caught.value.retained_control_handles == (failed_handle,)
            assert failed_handle not in closed
            other = launch.acknowledgement if channel == "status" else launch.status
            assert closed == [other]
            assert launch.process.poll() is not None
            assert not launch.sentinel.exists()
        finally:
            # Only this handle's close was rejected; the other slot is no longer ours.
            launch.handles.add(failed_handle)
