# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fixed native startup status and parent acknowledgement, never stdout parsing.

The launcher alone owns these anonymous pipes. A successful exchange does not
itself qualify a runtime: trusted artifacts, startup actions and the complete
runtime enforcement probe remain separate admission requirements.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes as W
from dataclasses import dataclass
import math
import secrets
import struct
import time

from .profiles import PYTHON_PROFILE, WindowsRuntimeError
from .native_io import pipe_api as _pipe_api

VERSION = 1
STATUS_MAGIC = b"USLPAC1\0"
ACK_MAGIC = b"USLACK1\0"
STATUS = struct.Struct("<8s6I32s32s32s")
ACK = struct.Struct("<8s2I32s32s32s")
READY = 1
FAILED = 2
REQUIRED_GATE_CHECKS = 0x1FF


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_PROTOCOL_MISMATCH", message)


@dataclass(frozen = True)
class LaunchBinding:
    pid: int
    nonce: bytes
    profile_digest: bytes
    content_digest: bytes

    def __post_init__(self):
        if type(self.pid) is not int or not 0 < self.pid <= 0xFFFFFFFF:
            raise _invalid("Invalid native startup PID.")
        for value in (self.nonce, self.profile_digest, self.content_digest):
            if type(value) is not bytes or len(value) != 32:
                raise _invalid("Invalid native startup binding length.")


def new_launch_nonce():
    return secrets.token_bytes(32)


def parse_startup_status(data, binding: LaunchBinding):
    if type(data) is not bytes or len(data) != STATUS.size:
        raise _invalid("Native startup status has the wrong size.")
    magic, version, phase, pid, checks, error, stage, nonce, profile, content = STATUS.unpack(data)
    if (
        magic != STATUS_MAGIC
        or version != VERSION
        or pid != binding.pid
        or nonce != binding.nonce
        or profile != binding.profile_digest
        or content != binding.content_digest
        or phase not in (READY, FAILED)
        or checks & ~REQUIRED_GATE_CHECKS
        or not 1 <= stage <= 16
    ):
        raise _invalid("Native startup status does not match this invocation.")
    if phase == FAILED:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_STARTUP_FAILED",
            f"Native startup failed at stage {stage} (WinError {error}).",
        )
    if error or stage != 16 or checks != REQUIRED_GATE_CHECKS:
        raise _invalid("Native startup did not prove every required gate check.")
    # This immutable binding is not an execution record or a qualification.
    return binding


def acknowledgement(binding: LaunchBinding):
    return ACK.pack(
        ACK_MAGIC, VERSION, 0, binding.nonce, binding.profile_digest, binding.content_digest
    )


def authorize_startup(
    process,
    status_handle,
    acknowledgement_handle,
    binding,
    *,
    timeout,
    cancel = None,
):
    """Own parent control handles, wait for bounded status/EOF, then acknowledge.

    Both native control channels are separate from process.stdout. A failed
    exchange terminates and reaps the Job-owned process, never retries it. The
    acknowledgement is small enough for the newly created empty pipe buffer.
    The launcher must not expose its writer to any other process/thread.
    """
    from .native_bindings import _api

    api = _api().kernel32
    pending = [status_handle, acknowledgement_handle]
    release_channels = True
    try:
        api = _pipe_api()
        if (
            not isinstance(binding, LaunchBinding)
            or process.pid != binding.pid
            or not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or not 0 < timeout <= 120
            or binding.profile_digest != bytes.fromhex(PYTHON_PROFILE.digest)
        ):
            raise _invalid("Invalid startup owner, profile or deadline.")
        deadline = time.monotonic() + timeout
        data = bytearray()
        while True:
            if cancel is not None and cancel.is_set():
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_CANCELLED", "Native startup was cancelled."
                )
            if time.monotonic() >= deadline:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_STARTUP_TIMEOUT", "Native startup exceeded its deadline."
                )
            available = W.DWORD()
            if not api.PeekNamedPipe(status_handle, None, 0, None, ctypes.byref(available), None):
                error = ctypes.get_last_error()
                if error == 109:  # closed writer, not a live-but-empty pipe
                    break
                raise _invalid(f"Native status pipe query failed ({error}).")
            if available.value:
                if available.value > STATUS.size - len(data):
                    raise _invalid("Native startup emitted excess control data.")
                buffer = ctypes.create_string_buffer(available.value)
                count = W.DWORD()
                if (
                    not api.ReadFile(status_handle, buffer, len(buffer), ctypes.byref(count), None)
                    or not count.value
                ):
                    raise _invalid("Native startup status read failed.")
                data.extend(buffer.raw[: count.value])
            else:
                time.sleep(min(0.005, max(0, deadline - time.monotonic())))
        parse_startup_status(bytes(data), binding)
        if cancel is not None and cancel.is_set():
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_CANCELLED", "Native startup was cancelled before acknowledgement."
            )
        if time.monotonic() >= deadline or process.poll() is not None:
            raise _invalid("Native startup owner exited or expired before acknowledgement.")
        if not api.CloseHandle(status_handle):
            raise _invalid("Native startup status channel closure failed.")
        pending.remove(status_handle)
        response = acknowledgement(binding)
        written = W.DWORD()
        if not api.WriteFile(
            acknowledgement_handle, response, len(response), ctypes.byref(written), None
        ) or written.value != len(response):
            raise _invalid("Native startup acknowledgement failed.")
        # The child requires acknowledgement EOF before payload entry. Treat
        # failed closure as failed startup, not a successful launch callback.
        while pending:
            if not api.CloseHandle(pending[0]):
                raise _invalid("Native startup channel closure failed.")
            pending.pop(0)
        return binding
    except BaseException as original:
        try:
            process.terminate()
            process.wait(timeout = 5)
        except Exception as cleanup:
            # Do not deliver acknowledgement EOF to a possibly live process
            # when termination/reaping failed. Keep ownership with the error;
            # the launch owner must reap the Job before releasing these handles.
            release_channels = False
            failure = WindowsRuntimeError(
                "WINDOWS_SANDBOX_CLEANUP_FAILED",
                f"Startup failed and process cleanup failed: {cleanup}",
            )
            failure.retained_control_handles = tuple(pending)
            raise failure from original
        raise
    finally:
        errors = []
        retained = []
        for handle in pending if release_channels else ():
            if not api.CloseHandle(handle):
                errors.append(ctypes.get_last_error())
                retained.append(handle)
        if errors:
            failure = WindowsRuntimeError(
                "WINDOWS_SANDBOX_CLEANUP_FAILED",
                f"Startup control handle cleanup failed: {errors}",
            )
            failure.retained_control_handles = tuple(retained)
            raise failure
