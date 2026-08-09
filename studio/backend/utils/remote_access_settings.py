# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted policy and non-blocking runtime operations for Remote access."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Callable

from loggers import get_logger

logger = get_logger(__name__)

REMOTE_ACCESS_AUTO_START_KEY = "remote_access_auto_start"
DEFAULT_REMOTE_ACCESS_AUTO_START = False
REMOTE_ACCESS_KINDS = frozenset({"temporary", "custom"})
# Longest a Stop waits for a live start worker to claim settings ownership.
_STOP_OWNERSHIP_WAIT = 5.0

_worker_lock = threading.Lock()
_start_worker: threading.Thread | None = None
_start_worker_kind: str | None = None
_stop_worker: threading.Thread | None = None
_stop_worker_kind: str | None = None
_start_worker_admission: tuple[int, int] | None = None
_stop_worker_admission: tuple[int, int] | None = None
_stop_response_condition = threading.Condition()
_stop_responses_pending = 0
_stop_response_admission_open = True
_STOP_RESPONSE_PATHS = frozenset(
    {
        "/api/settings/remote-access/stop",
        "/api/settings/remote-access/custom/teardown",
    }
)
_custom_worker: threading.Thread | None = None
_custom_cancel: threading.Event | None = None
_custom_operation = "idle"
_custom_login_url: str | None = None
_custom_error: tuple[str, str] | None = None


class RemoteAccessStopResponseMiddleware:
    """Lease the connector for every Stop request from ASGI admission through response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and scope.get("path") in _STOP_RESPONSE_PATHS
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


def get_remote_access_auto_start() -> bool:
    return get_remote_access_auto_start_kind() is not None


def _auto_start_kind(stored: object) -> str | None:
    if stored is True:
        return "temporary"
    if not isinstance(stored, dict) or type(stored.get("version")) is not int:
        return None
    if stored["version"] != 1:
        return None
    mode = stored.get("mode")
    return mode if isinstance(mode, str) and mode in REMOTE_ACCESS_KINDS else None


def get_remote_access_auto_start_kind() -> str | None:
    """Return the configured auto-start mode, including the legacy boolean form."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(REMOTE_ACCESS_AUTO_START_KEY, None)
    except Exception:
        return None
    return _auto_start_kind(stored)


def set_remote_access_auto_start(enabled: bool, kind: str = "temporary") -> bool:
    if not isinstance(enabled, bool):
        raise ValueError("Remote access auto-start must be true or false.")
    if kind not in REMOTE_ACCESS_KINDS:
        raise ValueError("Unknown remote access tunnel kind.")
    from storage.studio_db import upsert_app_settings

    mode = kind if enabled else "disabled"
    upsert_app_settings({REMOTE_ACCESS_AUTO_START_KEY: {"version": 1, "mode": mode}})
    return enabled


def clear_custom_remote_access_auto_start() -> bool:
    """Disable custom auto-start without overwriting a newer preference."""
    from storage.studio_db import get_connection

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value_json FROM app_settings WHERE key = ?",
            (REMOTE_ACCESS_AUTO_START_KEY,),
        ).fetchone()
        try:
            stored = json.loads(row["value_json"]) if row is not None else None
        except (TypeError, ValueError):
            stored = None
        if _auto_start_kind(stored) != "custom":
            conn.rollback()
            return False
        conn.execute(
            "UPDATE app_settings SET value_json = ?, updated_at = ? WHERE key = ?",
            (
                json.dumps({"version": 1, "mode": "disabled"}),
                datetime.now(timezone.utc).isoformat(),
                REMOTE_ACCESS_AUTO_START_KEY,
            ),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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


def _custom_status() -> dict:
    from cloudflare_tunnel import identity_is_runnable, orphaned_hostnames, read_identity

    identity = read_identity()
    with _worker_lock:
        operation, login_url, failure = _custom_operation, _custom_login_url, _custom_error
        running = _worker_alive(_custom_worker)
    if operation in {"provisioning", "tearing_down"} and not running:
        operation = "idle"
    state = operation if operation != "idle" else ("configured" if identity else "unconfigured")
    return {
        "custom_state": state,
        "custom_hostname": identity.get("hostname") if identity else None,
        "custom_runnable": identity_is_runnable(identity),
        "login_url": login_url if running else None,
        "custom_error": failure[0] if failure else None,
        "custom_error_detail": failure[1] if failure else None,
        "orphaned_hostnames": orphaned_hostnames(),
    }


def remote_access_status(app_state) -> dict:
    from cloudflare_tunnel import get_studio_tunnel_control_token, get_studio_tunnel_status

    status = get_studio_tunnel_status()
    custom = _custom_status()
    current = get_studio_tunnel_control_token()
    with _worker_lock:
        starting = _worker_is_current(_start_worker, _start_worker_admission, current)
        start_kind = _start_worker_kind if starting else None
        stopping = _worker_is_current(_stop_worker, _stop_worker_admission, current)
        generation_advanced = stopping and _stop_worker_admission[1] != current[1]
    # A stop worker outlives its teardown. Only one that advanced the generation
    # and left the tunnel off with nothing pending has actually performed it.
    if generation_advanced and status["state"] == "off" and not status.get("stop_pending"):
        stopping = False
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
    # Reported on its own too: a higher-precedence block hides the reason, but
    # the desktop still offers setting the password that is pending.
    password_pending = not _admin_password_ready()
    block_reason = None
    if not ready:
        block_reason = "server_starting"
    elif is_colab:
        block_reason = "colab_managed" if owner == "colab" else "colab"
    elif intent == "disabled":
        block_reason = "explicitly_disabled"
    elif launch_managed:
        block_reason = "launch_managed"
    elif password_pending:
        block_reason = "admin_password_change_required"
    elif owner in {"launch", "colab"}:
        block_reason = f"{owner}_managed"

    auto_start_kind = get_remote_access_auto_start_kind()
    auto_start_block_reason = block_reason
    if (
        auto_start_block_reason is None
        and auto_start_kind == "custom"
        and not custom["custom_runnable"]
    ):
        auto_start_block_reason = "custom_tunnel_not_configured"

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
        "Custom Cloudflare tunnel is not configured",
        "Custom Cloudflare connector is already running",
        "Custom Cloudflare tunnel could not be prepared",
        # Why Start is blocked: the connector's exit was never confirmed, so it
        # still holds the runtime slot. The generic text hides the one reason
        # the user can act on.
        "cloudflared could not be stopped",
    }:
        error = "Cloudflare tunnel failed"
    return {
        "state": state,
        "url": status["url"],
        "error": error,
        "auto_start": auto_start_kind is not None,
        "auto_start_kind": auto_start_kind,
        "auto_start_block_reason": auto_start_block_reason,
        "available": ready and not is_colab and intent != "disabled",
        "managed_by": owner,
        "can_start": can_start,
        "can_stop": can_stop,
        "block_reason": block_reason,
        "password_pending": password_pending,
        "streaming_supported": status["url"] is None or status.get("kind") == "custom",
        "kind": start_kind or status.get("kind"),
        "connector_registered": bool(status.get("connector_registered")),
        "tunnel_serving": bool(status.get("tunnel_serving")),
        "dns": status.get("dns", "unknown"),
        **custom,
    }


def start_remote_access(app_state, *, kind: str | None = None) -> dict:
    """Schedule a settings-owned start. Repeated requests are idempotent."""
    global _start_worker, _start_worker_admission, _start_worker_kind
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
    if kind is None:
        kind = "custom" if status.get("custom_runnable") else "temporary"
    if kind not in REMOTE_ACCESS_KINDS:
        raise RuntimeError("unknown_tunnel_kind")
    if status["managed_by"] == "settings" and status["state"] in {"starting", "online"}:
        if status.get("kind", "temporary") == kind:
            return status
        if status["state"] == "starting" and status.get("kind") in REMOTE_ACCESS_KINDS:
            raise RuntimeError("start_kind_conflict")
    if not status["can_start"]:
        raise RuntimeError(status["block_reason"] or "operation_in_progress")
    if kind == "custom":
        if status.get("custom_state") in {"provisioning", "tearing_down"}:
            raise RuntimeError("custom_operation_in_progress")
        if not status.get("custom_runnable"):
            raise RuntimeError("custom_tunnel_not_configured")

    port = getattr(app_state, "remote_access_port", None)
    if not isinstance(port, int) or port <= 0:
        raise RuntimeError("server_port_unavailable")
    if get_studio_tunnel_control_token() != admission:
        raise RuntimeError("server_lifecycle_changed")

    def _start() -> None:
        from cloudflare_tunnel import start_studio_tunnel
        url = start_studio_tunnel(port, managed_by = "settings", admission = admission, kind = kind)
        if url:
            logger.info("Secure link access via Cloudflare: %s", url)

    _open_remote_access_stop_response_admission()
    with _worker_lock:
        if _worker_is_current(_start_worker, _start_worker_admission, admission):
            if _start_worker_kind != kind:
                raise RuntimeError("start_kind_conflict")
        else:
            _start_worker = threading.Thread(target = _start, daemon = True)
            _start_worker_admission = admission
            _start_worker_kind = kind
            _start_worker.start()
    return remote_access_status(app_state)


def stop_remote_access(app_state, *, kind: str | None = None) -> dict:
    """Schedule a settings-owned stop without changing the auto-start preference."""
    global _stop_worker, _stop_worker_admission, _stop_worker_kind
    from cloudflare_tunnel import (
        capture_studio_tunnel_start_admission,
        get_studio_tunnel_control_token,
    )

    admission = capture_studio_tunnel_start_admission()
    if admission is None:
        raise RuntimeError("server_shutting_down")
    status = remote_access_status(app_state)
    observed_kind = status.get("kind")
    target_kind = kind or observed_kind
    current = get_studio_tunnel_control_token()
    if current[0] != admission[0]:
        raise RuntimeError("server_lifecycle_changed")
    if status["state"] == "off" and status["managed_by"] is None:
        return status
    if kind is not None and observed_kind != kind:
        return status
    if status["state"] == "stopping" and status["managed_by"] == "settings":
        with _worker_lock:
            if not _worker_alive(_stop_worker) or _stop_worker_kind == target_kind:
                return status
        raise RuntimeError("stop_kind_conflict")
    if status["managed_by"] != "settings":
        raise RuntimeError(status["block_reason"] or "not_settings_managed")
    if get_studio_tunnel_control_token() != admission:
        raise RuntimeError("server_lifecycle_changed")

    def _stop() -> None:
        global _stop_worker_admission
        from cloudflare_tunnel import get_studio_tunnel_status, stop_studio_tunnel

        # A stop can beat the newly-created start worker to the controller. Wait
        # until it claims settings ownership, then cancel that generation. Bounded:
        # a start that never claims it (foreign owner, bailed on admission) must
        # not defer the user's Stop for the probe deadline.
        deadline = time.monotonic() + _STOP_OWNERSHIP_WAIT
        while _worker_alive(_start_worker) and time.monotonic() < deadline:
            if get_studio_tunnel_status()["managed_by"] == "settings":
                break
            time.sleep(0.02)
        current = get_studio_tunnel_control_token()
        if current[0] != admission[0]:
            return
        tunnel = get_studio_tunnel_status()
        if tunnel["managed_by"] == "settings" and tunnel.get("kind") == target_kind:
            # Every Stop admitted before this teardown decision must finish
            # traversing cloudflared. Closing admission at the end of the drain
            # prevents a later request from creating an unobserved lease.
            _drain_and_close_remote_access_stop_responses()
            current = get_studio_tunnel_control_token()
            tunnel = get_studio_tunnel_status()
            if (
                current[0] != admission[0]
                or tunnel["managed_by"] != "settings"
                or (tunnel.get("kind") != target_kind)
            ):
                _open_remote_access_stop_response_admission()
                return
            with _worker_lock:
                if _stop_worker is threading.current_thread():
                    _stop_worker_admission = current
            try:
                stop_studio_tunnel(admission = current, kind = target_kind)
                if get_studio_tunnel_status().get("stop_pending"):
                    _open_remote_access_stop_response_admission()
            except Exception:
                _open_remote_access_stop_response_admission()
                raise

    with _worker_lock:
        if _worker_is_current(_stop_worker, _stop_worker_admission, admission):
            if _stop_worker_kind != target_kind:
                raise RuntimeError("stop_kind_conflict")
        else:
            _stop_worker = threading.Thread(target = _stop, daemon = True)
            _stop_worker_admission = admission
            _stop_worker_kind = target_kind
            _stop_worker.start()
    return remote_access_status(app_state)


def maybe_auto_start_remote_access(app_state) -> bool:
    """Schedule persisted auto-start when current launch policy permits it."""
    kind = get_remote_access_auto_start_kind()
    if kind is None:
        return False
    try:
        start_remote_access(app_state, kind = kind)
    except RuntimeError:
        return False
    return True


def _set_custom_operation(
    operation: str,
    *,
    login_url: str | None = None,
    error: tuple[str, str] | None = None,
) -> None:
    global _custom_operation, _custom_login_url, _custom_error
    with _worker_lock:
        _custom_operation = operation
        _custom_login_url = login_url
        _custom_error = error


def _custom_login_callback(url: str) -> None:
    global _custom_login_url
    with _worker_lock:
        if _custom_worker is threading.current_thread():
            _custom_login_url = url


def _custom_operation_allowed(app_state) -> None:
    status = remote_access_status(app_state)
    if not status["available"] or status["password_pending"]:
        raise RuntimeError(status["block_reason"] or "remote_access_unavailable")
    with _worker_lock:
        if _worker_alive(_custom_worker):
            raise RuntimeError("custom_operation_in_progress")


def provision_custom_remote_access(app_state, hostname: str) -> dict:
    global _custom_worker, _custom_cancel, _custom_operation, _custom_login_url, _custom_error
    from cloudflare_tunnel import canonical_hostname

    host = canonical_hostname(hostname)
    _custom_operation_allowed(app_state)
    cancel = threading.Event()

    def _provision() -> None:
        from cloudflare_tunnel import ensure_cloudflared
        from cloudflare_tunnel import ProvisioningError, provision_custom_tunnel
        try:
            binary = ensure_cloudflared()
            if not binary:
                raise ProvisioningError("cloudflared_unreachable", "cloudflared is unavailable.")
            provision_custom_tunnel(
                host,
                binary = binary,
                on_login_url = _custom_login_callback,
                cancelled = cancel.is_set,
            )
        except ProvisioningError as exc:
            _set_custom_operation("error", error = (exc.code, exc.detail))
        except Exception:
            logger.warning("Custom Cloudflare setup failed.", exc_info = True)
            _set_custom_operation("error", error = ("setup_failed", "Cloudflare setup failed."))
        else:
            _set_custom_operation("idle")

    with _worker_lock:
        if _worker_alive(_custom_worker):
            raise RuntimeError("custom_operation_in_progress")
        _custom_operation = "provisioning"
        _custom_login_url = None
        _custom_error = None
        _custom_cancel = cancel
        _custom_worker = threading.Thread(target = _provision, daemon = True)
        _custom_worker.start()
    return remote_access_status(app_state)


def cancel_custom_remote_access(app_state) -> dict:
    with _worker_lock:
        if _worker_alive(_custom_worker) and _custom_cancel is not None:
            _custom_cancel.set()
    return remote_access_status(app_state)


def teardown_custom_remote_access(app_state) -> dict:
    global _custom_worker, _custom_cancel, _custom_operation, _custom_login_url, _custom_error
    from cloudflare_tunnel import read_identity

    if read_identity() is None:
        return remote_access_status(app_state)
    _custom_operation_allowed(app_state)
    cancel = threading.Event()

    def _teardown() -> None:
        from cloudflare_tunnel import ensure_cloudflared, get_studio_tunnel_status
        from cloudflare_tunnel import ProvisioningError, teardown_custom_tunnel
        try:
            clear_custom_remote_access_auto_start()
            tunnel = get_studio_tunnel_status()
            if tunnel.get("kind") == "custom" and (
                tunnel["state"] != "off" or tunnel.get("managed_by") is not None
            ):
                stop_remote_access(app_state, kind = "custom")
                with _worker_lock:
                    worker = _stop_worker
                if worker is not None:
                    worker.join(_STOP_OWNERSHIP_WAIT + 50.0)
                tunnel = get_studio_tunnel_status()
                if tunnel.get("kind") == "custom" and (
                    tunnel["state"] != "off" or tunnel.get("stop_pending")
                ):
                    raise ProvisioningError(
                        "connector_stop_failed", "The Cloudflare connector could not be stopped."
                    )
            teardown_custom_tunnel(
                binary = ensure_cloudflared(),
                on_login_url = _custom_login_callback,
                cancelled = cancel.is_set,
                clear_auto_start = clear_custom_remote_access_auto_start,
            )
        except ProvisioningError as exc:
            _set_custom_operation("error", error = (exc.code, exc.detail))
        except Exception:
            logger.warning("Custom Cloudflare teardown failed.", exc_info = True)
            _set_custom_operation("error", error = ("teardown_failed", "Cloudflare teardown failed."))
        else:
            _set_custom_operation("idle")

    with _worker_lock:
        if _worker_alive(_custom_worker):
            raise RuntimeError("custom_operation_in_progress")
        _custom_operation = "tearing_down"
        _custom_login_url = None
        _custom_error = None
        _custom_cancel = cancel
        _custom_worker = threading.Thread(target = _teardown, daemon = True)
        _custom_worker.start()
    return remote_access_status(app_state)
