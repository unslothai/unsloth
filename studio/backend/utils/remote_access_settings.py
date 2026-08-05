# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted policy and non-blocking runtime operations for Remote access."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from loggers import get_logger

logger = get_logger(__name__)

REMOTE_ACCESS_AUTO_START_KEY = "remote_access_auto_start"
DEFAULT_REMOTE_ACCESS_AUTO_START = False

_worker_lock = threading.Lock()
_start_worker: threading.Thread | None = None
_stop_worker: threading.Thread | None = None
_start_worker_admission: tuple[int, int] | None = None
_stop_worker_admission: tuple[int, int] | None = None
_stop_response_condition = threading.Condition()
_stop_responses_pending = 0
_stop_response_admission_open = True


class RemoteAccessStopResponseMiddleware:
    """Lease the connector for every Stop request from ASGI admission through response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/api/settings/remote-access/stop"
        ):
            await self.app(scope, receive, send)
            return

        release = acquire_remote_access_stop_response()
        if release is None:
            # Teardown has already linearized. Preserve downstream auth and
            # idempotent route behavior without admitting new drain work.
            await self.app(scope, receive, send)
            return

        async def _send(message):
            await send(message)
            if message.get("type") == "http.response.body" and not message.get("more_body", False):
                release()

        try:
            await self.app(scope, receive, _send)
        finally:
            release()


def acquire_remote_access_stop_response() -> Callable[[], None] | None:
    """Hold connector teardown until this HTTP response has been finalized."""
    global _stop_responses_pending
    with _stop_response_condition:
        if not _stop_response_admission_open:
            return None
        _stop_responses_pending += 1
        _stop_response_condition.notify_all()
    released = False

    def _release() -> None:
        nonlocal released
        global _stop_responses_pending
        with _stop_response_condition:
            if released:
                return
            released = True
            _stop_responses_pending -= 1
            _stop_response_condition.notify_all()

    return _release


def _open_remote_access_stop_response_admission() -> None:
    global _stop_response_admission_open
    with _stop_response_condition:
        _stop_response_admission_open = True
        _stop_response_condition.notify_all()


def _drain_and_close_remote_access_stop_responses() -> None:
    """Drain admitted responses, then close admission at the teardown boundary."""
    global _stop_response_admission_open
    deadline = time.monotonic() + 1.0
    quiet_deadline: float | None = None
    with _stop_response_condition:
        while True:
            now = time.monotonic()
            if now >= deadline:
                _stop_response_admission_open = False
                return
            if _stop_responses_pending:
                quiet_deadline = None
                _stop_response_condition.wait(min(0.05, deadline - now))
                continue
            if quiet_deadline is None:
                quiet_deadline = now + 0.05
            if now >= quiet_deadline:
                _stop_response_admission_open = False
                return
            _stop_response_condition.wait(min(quiet_deadline - now, deadline - now))


def _coerce_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def get_remote_access_auto_start() -> bool:
    """Read the preference, failing closed on missing, invalid, or unreadable data."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(REMOTE_ACCESS_AUTO_START_KEY, None)
    except Exception:
        return False
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_REMOTE_ACCESS_AUTO_START


def set_remote_access_auto_start(enabled: bool) -> bool:
    if not isinstance(enabled, bool):
        raise ValueError("Remote access auto-start must be true or false.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({REMOTE_ACCESS_AUTO_START_KEY: enabled})
    return enabled


def _admin_password_ready() -> bool:
    try:
        from auth.storage import DEFAULT_ADMIN_USERNAME, requires_password_change
        return not requires_password_change(DEFAULT_ADMIN_USERNAME)
    except Exception:
        return False


def configure_remote_access(
    app_state, *, port: int, intent: str, is_colab: bool, launch_managed: bool
) -> None:
    """Publish immutable launch policy used by every settings request."""
    app_state.remote_access_port = port
    app_state.remote_access_intent = intent
    app_state.remote_access_is_colab = bool(is_colab)
    app_state.remote_access_launch_managed = bool(launch_managed)
    app_state.remote_access_ready = False


def _worker_alive(worker: threading.Thread | None) -> bool:
    return worker is not None and worker.is_alive()


def _worker_is_current(
    worker: threading.Thread | None, admission: tuple[int, int] | None, current: tuple[int, int]
) -> bool:
    if not _worker_alive(worker) or admission is None or admission[0] != current[0]:
        return False
    return current[1] in {admission[1], admission[1] + 1}


def remote_access_status(app_state) -> dict:
    from cloudflare_tunnel import get_studio_tunnel_control_token, get_studio_tunnel_status

    status = get_studio_tunnel_status()
    current = get_studio_tunnel_control_token()
    with _worker_lock:
        starting = _worker_is_current(_start_worker, _start_worker_admission, current)
        stopping = _worker_is_current(_stop_worker, _stop_worker_admission, current)
    if stopping:
        status.update(state = "stopping", managed_by = "settings", url = None, error = None)
    elif starting and status["state"] in {"off", "error"}:
        status.update(state = "starting", managed_by = "settings", url = None, error = None)

    intent = getattr(app_state, "remote_access_intent", "disabled")
    is_colab = bool(getattr(app_state, "remote_access_is_colab", False))
    launch_managed = bool(getattr(app_state, "remote_access_launch_managed", False))
    ready = bool(getattr(app_state, "remote_access_ready", False))
    owner = status["managed_by"]
    state = status["state"]
    stop_pending = bool(status.get("stop_pending"))
    block_reason = None
    if not ready:
        block_reason = "server_starting"
    elif is_colab:
        block_reason = "colab_managed" if owner == "colab" else "colab"
    elif intent == "disabled":
        block_reason = "explicitly_disabled"
    elif launch_managed:
        block_reason = "launch_managed"
    elif not _admin_password_ready():
        block_reason = "admin_password_change_required"
    elif owner in {"launch", "colab"}:
        block_reason = f"{owner}_managed"

    controllable = block_reason is None
    can_start = controllable and not stopping and not stop_pending and state in {"off", "error"}
    can_stop = owner == "settings" and (stop_pending or state in {"starting", "online"})
    error = status["error"]
    if error not in {
        None,
        "cloudflared is unavailable",
        "cloudflared did not produce a URL",
        "Cloudflare URL was not reachable",
        "cloudflared did not register a connection",
        "cloudflared exited",
    }:
        error = "Cloudflare tunnel failed"
    return {
        "state": state,
        "url": status["url"],
        "error": error,
        "auto_start": get_remote_access_auto_start(),
        "available": ready and not is_colab and intent != "disabled",
        "managed_by": owner,
        "can_start": can_start,
        "can_stop": can_stop,
        "block_reason": block_reason,
        "streaming_supported": True,
    }


def start_remote_access(app_state) -> dict:
    """Schedule a settings-owned start. Repeated requests are idempotent."""
    global _start_worker, _start_worker_admission
    from cloudflare_tunnel import (
        capture_studio_tunnel_start_admission,
        get_studio_tunnel_control_token,
    )

    admission = capture_studio_tunnel_start_admission()
    if admission is None:
        raise RuntimeError("server_shutting_down")
    status = remote_access_status(app_state)
    current = get_studio_tunnel_control_token()
    if current[0] != admission[0]:
        raise RuntimeError("server_lifecycle_changed")
    if status["managed_by"] == "settings" and status["state"] in {"starting", "online"}:
        return status
    if not status["can_start"]:
        raise RuntimeError(status["block_reason"] or "operation_in_progress")

    port = getattr(app_state, "remote_access_port", None)
    if not isinstance(port, int) or port <= 0:
        raise RuntimeError("server_port_unavailable")
    if get_studio_tunnel_control_token() != admission:
        raise RuntimeError("server_lifecycle_changed")

    def _start() -> None:
        from cloudflare_tunnel import start_studio_tunnel
        url = start_studio_tunnel(port, managed_by = "settings", admission = admission)
        if url:
            logger.info("Secure link access via Cloudflare: %s", url)

    _open_remote_access_stop_response_admission()
    with _worker_lock:
        if not _worker_is_current(_start_worker, _start_worker_admission, admission):
            _start_worker = threading.Thread(target = _start, daemon = True)
            _start_worker_admission = admission
            _start_worker.start()
    return remote_access_status(app_state)


def stop_remote_access(app_state) -> dict:
    """Schedule a settings-owned stop without changing the auto-start preference."""
    global _stop_worker, _stop_worker_admission
    from cloudflare_tunnel import (
        capture_studio_tunnel_start_admission,
        get_studio_tunnel_control_token,
    )

    admission = capture_studio_tunnel_start_admission()
    if admission is None:
        raise RuntimeError("server_shutting_down")
    status = remote_access_status(app_state)
    current = get_studio_tunnel_control_token()
    if current[0] != admission[0]:
        raise RuntimeError("server_lifecycle_changed")
    if status["state"] == "off" and status["managed_by"] is None:
        return status
    if status["state"] == "stopping" and status["managed_by"] == "settings":
        return status
    if status["managed_by"] != "settings":
        raise RuntimeError(status["block_reason"] or "not_settings_managed")

    if get_studio_tunnel_control_token() != admission:
        raise RuntimeError("server_lifecycle_changed")

    def _stop() -> None:
        global _stop_worker_admission
        from cloudflare_tunnel import get_studio_tunnel_status, stop_studio_tunnel

        # A stop can beat the newly-created start worker to the controller. Wait
        # until it claims settings ownership, then cancel that generation.
        while _worker_alive(_start_worker):
            current = get_studio_tunnel_status()
            if current["managed_by"] == "settings":
                break
            time.sleep(0.01)
        current = get_studio_tunnel_control_token()
        if current[0] != admission[0]:
            return
        if get_studio_tunnel_status()["managed_by"] == "settings":
            # Every Stop admitted before this teardown decision must finish
            # traversing cloudflared. Closing admission at the end of the drain
            # prevents a later request from creating an unobserved lease.
            _drain_and_close_remote_access_stop_responses()
            current = get_studio_tunnel_control_token()
            if current[0] != admission[0] or get_studio_tunnel_status()["managed_by"] != "settings":
                _open_remote_access_stop_response_admission()
                return
            with _worker_lock:
                if _stop_worker is threading.current_thread():
                    _stop_worker_admission = current
            try:
                stop_studio_tunnel(admission = current)
                if get_studio_tunnel_status().get("stop_pending"):
                    _open_remote_access_stop_response_admission()
            except Exception:
                _open_remote_access_stop_response_admission()
                raise

    with _worker_lock:
        if not _worker_is_current(_stop_worker, _stop_worker_admission, admission):
            _stop_worker = threading.Thread(target = _stop, daemon = True)
            _stop_worker_admission = admission
            _stop_worker.start()
    return remote_access_status(app_state)


def maybe_auto_start_remote_access(app_state) -> bool:
    """Schedule persisted auto-start when current launch policy permits it."""
    if not get_remote_access_auto_start():
        return False
    try:
        start_remote_access(app_state)
    except RuntimeError:
        return False
    return True
