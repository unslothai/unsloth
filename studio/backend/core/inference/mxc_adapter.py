# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Authenticated control protocol for the Windows-native MXC supervisor."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import threading
import time

from . import mxc_pipe, mxc_runtime

MAX_REQUEST = 262_144
MAX_CONTROL = 65_536
STARTUP_TIMEOUT_SECONDS = 45


class MxcAdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        code: str = "mxc_error",
        may_have_started: bool = False,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.may_have_started = may_have_started


def _validated_event(line: bytes, request: dict, token: str, expected: str) -> dict:
    if len(line) > MAX_CONTROL:
        raise MxcAdapterError("MXC control frame exceeds the protocol bound", stage="protocol")
    try:
        event = json.loads(line)
    except (ValueError, UnicodeError) as exc:
        raise MxcAdapterError("malformed MXC control frame", stage="protocol") from exc
    if not isinstance(event, dict) or event.get("v") != mxc_runtime.RUNNER_PROTOCOL_VERSION:
        raise MxcAdapterError("unsupported MXC control protocol", stage="protocol")
    if event.get("runId") != request["runId"] or not secrets.compare_digest(
        str(event.get("token", "")), token
    ):
        raise MxcAdapterError("MXC control authentication failed", stage="protocol")
    kind = event.get("event")
    if kind == "ERROR":
        raise MxcAdapterError(
            str(event.get("message") or "MXC supervisor rejected the launch"),
            stage=str(event.get("stage") or "launch"),
            code=str(event.get("code") or "mxc_error"),
        )
    if kind != expected:
        raise MxcAdapterError(f"out-of-order MXC control event: {kind!r}", stage="protocol")
    if kind == "STARTED" and event.get("backendTier") != "base-container":
        raise MxcAdapterError(
            "MXC STARTED did not carry BaseContainer execution evidence",
            stage="admission",
            code="missing_effective_tier",
        )
    return event


def _stop_helper(proc, connection=None) -> None:
    if connection is not None:
        connection.close()
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def spawn(request: dict, *, cancel_event=None, popen_kwargs: dict | None = None):
    """Start the supervisor and return only after authenticated STARTED."""
    token = secrets.token_hex(32)
    request = dict(request)
    request["token"] = token
    encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(encoded) > MAX_REQUEST:
        raise MxcAdapterError("MXC launch request exceeds the protocol bound", stage="policy")

    proc = None
    connection = None
    writer = None
    runtime_lease = None
    with mxc_pipe.PrivatePipeServer(buffer_size=MAX_CONTROL) as listener:
        request["controlPipe"] = listener.name
        encoded = (
            json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        if len(encoded) > MAX_REQUEST:
            raise MxcAdapterError("MXC launch request exceeds the protocol bound", stage="policy")
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS

        def check_wait() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise MxcAdapterError(
                    "MXC launch cancelled before STARTED", stage="startup", code="cancelled"
                )
            if time.monotonic() >= deadline:
                raise MxcAdapterError(
                    "MXC supervisor STARTED acknowledgement timed out",
                    stage="startup",
                    code="timeout",
                )

        try:
            runtime_lease = mxc_runtime.acquire_runtime()
            options = dict(popen_kwargs or {})
            options.pop("preexec_fn", None)
            options.pop("pass_fds", None)
            options.update(
                stdin=subprocess.PIPE,
                cwd=str(runtime_lease.info.path.parent),
                env={
                    key: value
                    for key, value in __import__("os").environ.items()
                    if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PROGRAMDATA"}
                },
                close_fds=True,
            )
            proc = subprocess.Popen([str(runtime_lease.info.path)], **options)

            def send_request() -> None:
                try:
                    offset = 0
                    while offset < len(encoded):
                        offset += os.write(proc.stdin.fileno(), encoded[offset:])
                except (OSError, ValueError):
                    pass
                finally:
                    try:
                        proc.stdin.close()
                    except (OSError, ValueError):
                        pass

            writer = threading.Thread(target=send_request, daemon=True)
            writer.start()
            pending = b""
            expected = ("HELLO", "ACCEPTED", "STARTED")
            event_index = 0
            while event_index < len(expected):
                check_wait()
                if connection is None:
                    try:
                        connection = listener.accept(
                            deadline=deadline, cancel_event=cancel_event, proc=proc
                        )
                    except mxc_pipe.PipeError as exc:
                        raise MxcAdapterError(str(exc), stage="startup") from exc
                chunk = connection.read_available(MAX_CONTROL + 1)
                if chunk is None:
                    time.sleep(0.01)
                    continue
                if not chunk:
                    raise MxcAdapterError(
                        "MXC control channel closed before STARTED", stage="protocol"
                    )
                pending += chunk
                if len(pending) > MAX_CONTROL:
                    raise MxcAdapterError(
                        "MXC control frame exceeds the protocol bound", stage="protocol"
                    )
                while b"\n" in pending and event_index < len(expected):
                    line, pending = pending.split(b"\n", 1)
                    _validated_event(line, request, token, expected[event_index])
                    event_index += 1
            writer.join(timeout=1)
            if writer.is_alive():
                raise MxcAdapterError("MXC request writer did not finish", stage="protocol")
            proc.stdin = None
            proc._mxc_control_pipe = connection
            proc._mxc_control_pending = pending
            proc._mxc_request = request
            proc._mxc_backend_tier = "base-container"
            proc._mxc_runtime_lease = runtime_lease
            proc._mxc_runtime_info = runtime_lease.info
            runtime_lease = None
            control_connection = connection
            cancel_lock = threading.Lock()

            def cancel() -> bool:
                frame = (
                    json.dumps(
                        {
                            "v": mxc_runtime.RUNNER_PROTOCOL_VERSION,
                            "event": "CANCEL",
                            "runId": request["runId"],
                            "token": request["token"],
                        },
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
                try:
                    with cancel_lock:
                        control_connection.write(frame)
                    return True
                except OSError:
                    return False

            proc._unsloth_cancel = cancel
            connection = None
            return proc
        except Exception as exc:
            if isinstance(exc, MxcAdapterError) and "event_index" in locals():
                exc.may_have_started = exc.may_have_started or event_index >= 2
            if proc is not None:
                _stop_helper(proc, connection)
                connection = None
            if runtime_lease is not None:
                runtime_lease.release()
                runtime_lease = None
            raise
        finally:
            if connection is not None:
                connection.close()
            if writer is not None:
                writer.join(timeout=1)


def completion_receipt(proc) -> dict:
    connection = getattr(proc, "_mxc_control_pipe", None)
    if connection is None or proc.poll() is None:
        raise MxcAdapterError(
            "MXC completion has no trusted receipt",
            stage="completion",
            may_have_started=True,
        )
    data = getattr(proc, "_mxc_control_pending", b"")
    deadline = time.monotonic() + 1
    while True:
        chunk = connection.read_available(MAX_CONTROL + 1)
        if chunk is None:
            if time.monotonic() >= deadline:
                raise MxcAdapterError(
                    "MXC control channel remained open after supervisor exit",
                    stage="completion",
                    may_have_started=True,
                )
            time.sleep(0.01)
            continue
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_CONTROL:
            raise MxcAdapterError(
                "MXC completion frame exceeds the protocol bound", stage="completion"
            )
    try:
        lines = data.splitlines()
        if len(lines) != 1:
            raise MxcAdapterError(
                "MXC FINISHED receipt is missing or duplicated",
                stage="completion",
                may_have_started=True,
            )
        event = json.loads(lines[0])
    except (ValueError, UnicodeError) as exc:
        raise MxcAdapterError(
            "malformed MXC FINISHED receipt", stage="completion", may_have_started=True
        ) from exc
    request = proc._mxc_request
    if (
        not isinstance(event, dict)
        or event.get("v") != mxc_runtime.RUNNER_PROTOCOL_VERSION
        or event.get("event") != "FINISHED"
        or event.get("runId") != request["runId"]
        or event.get("backendTier") != getattr(proc, "_mxc_backend_tier", None)
        or not secrets.compare_digest(str(event.get("token", "")), request["token"])
    ):
        raise MxcAdapterError(
            "invalid MXC FINISHED receipt", stage="completion", may_have_started=True
        )
    return event


def release_control(proc) -> None:
    connection = getattr(proc, "_mxc_control_pipe", None)
    if connection is not None:
        proc._mxc_control_pipe = None
        connection.close()
    runtime_lease = getattr(proc, "_mxc_runtime_lease", None)
    if runtime_lease is not None:
        proc._mxc_runtime_lease = None
        runtime_lease.release()


def abort(proc, *, grace_seconds: float = 5) -> None:
    """Cancel a live supervisor, then reclaim it if the trusted path stalls."""
    request_cancel = getattr(proc, "_unsloth_cancel", None)
    if proc.poll() is None and request_cancel is not None:
        request_cancel()
    if proc.poll() is None:
        try:
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            _stop_helper(proc, getattr(proc, "_mxc_control_pipe", None))
            proc._mxc_control_pipe = None
