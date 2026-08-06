# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import cloudflare_tunnel  # noqa: E402
import routes.settings as routes  # noqa: E402
import utils.remote_access_settings as remote_access  # noqa: E402
from storage import studio_db  # noqa: E402


def _state(
    intent = "unset",
    is_colab = False,
    launch_managed = False,
):
    return SimpleNamespace(
        remote_access_intent = intent,
        remote_access_is_colab = is_colab,
        remote_access_launch_managed = launch_managed,
        remote_access_port = 8888,
        remote_access_ready = True,
    )


def test_auto_start_persistence_is_strict_and_fail_closed(monkeypatch):
    stored = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback: stored.get(key, fallback)
    )
    monkeypatch.setattr(studio_db, "upsert_app_settings", lambda values: stored.update(values))
    assert remote_access.get_remote_access_auto_start() is False
    stored[remote_access.REMOTE_ACCESS_AUTO_START_KEY] = "yes"
    assert remote_access.get_remote_access_auto_start() is False
    assert remote_access.set_remote_access_auto_start(True) is True
    assert remote_access.get_remote_access_auto_start() is True
    with pytest.raises(ValueError):
        routes.RemoteAccessAutoStartPayload(enabled = "true")
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda *args: (_ for _ in ()).throw(OSError())
    )
    assert remote_access.get_remote_access_auto_start() is False


@pytest.mark.parametrize(
    "launch_managed,expected_block,can_start",
    [(False, None, True), (True, "launch_managed", False)],
)
def test_enabled_intent_blocks_only_selected_launch_path(
    monkeypatch, launch_managed, expected_block, can_start
):
    monkeypatch.setattr(remote_access, "_start_worker", None)
    monkeypatch.setattr(remote_access, "_stop_worker", None)
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start", lambda: False)
    monkeypatch.setattr(remote_access, "_admin_password_ready", lambda: True)
    monkeypatch.setattr(
        cloudflare_tunnel,
        "get_studio_tunnel_status",
        lambda: {"state": "off", "url": None, "error": None, "managed_by": None},
    )
    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_control_token", lambda: (1, 0))

    state = _state(intent = "enabled", launch_managed = launch_managed)
    status = remote_access.remote_access_status(state)
    assert status["block_reason"] == expected_block
    assert status["can_start"] is can_start
    assert status["password_pending"] is False

    # A pending password is reported on its own, even where another block hides it.
    monkeypatch.setattr(remote_access, "_admin_password_ready", lambda: False)
    pending = remote_access.remote_access_status(state)
    assert pending["password_pending"] is True
    assert pending["block_reason"] == (expected_block or "admin_password_change_required")


def test_failed_stop_remains_retryable(monkeypatch):
    monkeypatch.setattr(remote_access, "_start_worker", None)
    monkeypatch.setattr(remote_access, "_stop_worker", None)
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start", lambda: False)
    monkeypatch.setattr(remote_access, "_admin_password_ready", lambda: True)
    monkeypatch.setattr(
        cloudflare_tunnel,
        "get_studio_tunnel_status",
        lambda: {
            "state": "error",
            "url": None,
            "error": None,
            "managed_by": "settings",
            "stop_pending": True,
        },
    )
    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_control_token", lambda: (1, 0))
    status = remote_access.remote_access_status(_state())
    assert status["can_start"] is False and status["can_stop"] is True
    source = Path(remote_access.__file__).read_text(encoding = "utf-8")
    assert 'if get_studio_tunnel_status().get("stop_pending"):' in source


def test_only_a_finished_stop_worker_stops_reporting_stopping(monkeypatch):
    hold = threading.Event()
    stale_stop = threading.Thread(target = hold.wait, daemon = True)
    stale_stop.start()
    monkeypatch.setattr(remote_access, "_stop_worker", stale_stop)
    monkeypatch.setattr(remote_access, "_stop_worker_admission", (1, 5))
    monkeypatch.setattr(remote_access, "_start_worker", None)
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start", lambda: False)
    monkeypatch.setattr(remote_access, "_admin_password_ready", lambda: True)
    tunnel_status = {
        "state": "off",
        "url": None,
        "error": None,
        "managed_by": None,
        "stop_pending": False,
    }
    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_status", lambda: dict(tunnel_status))
    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_control_token", lambda: (1, 6))

    # the teardown advanced the generation past (1, 5), so this stop is done
    status = remote_access.remote_access_status(_state())
    assert status["state"] == "off"
    assert status["can_start"] is True

    new_start = threading.Thread(target = hold.wait, daemon = True)
    new_start.start()
    monkeypatch.setattr(remote_access, "_start_worker", new_start)
    monkeypatch.setattr(remote_access, "_start_worker_admission", (1, 6))
    assert remote_access.remote_access_status(_state())["state"] == "starting"
    monkeypatch.setattr(remote_access, "_start_worker", None)

    # a stop admitted at the current generation still owes its teardown
    monkeypatch.setattr(remote_access, "_stop_worker_admission", (1, 6))
    assert remote_access.remote_access_status(_state())["state"] == "stopping"
    monkeypatch.setattr(remote_access, "_stop_worker_admission", (1, 5))

    # so does a torn-down stop whose termination is unconfirmed
    tunnel_status["stop_pending"] = True
    assert remote_access.remote_access_status(_state())["state"] == "stopping"

    # and the report stays scoped to a tunnel that is actually off
    tunnel_status.update(stop_pending = False, state = "online", managed_by = "settings")
    assert remote_access.remote_access_status(_state())["state"] == "stopping"
    hold.set()


def test_workers_and_stops_are_scoped_to_backend_lifecycle(monkeypatch):
    remote_access._start_worker = remote_access._stop_worker = None
    remote_access._start_worker_admission = remote_access._stop_worker_admission = None
    cloudflare_tunnel.stop_studio_tunnel()
    cloudflare_tunnel.open_studio_tunnel_lifecycle()
    entered, reopened, release = (threading.Event() for _ in range(3))
    attempts = []
    real_start = cloudflare_tunnel.start_studio_tunnel

    def _delayed(*args, **kwargs):
        attempts.append(kwargs["admission"])
        if len(attempts) == 1:
            entered.set()
            release.wait(1)
            return real_start(*args, **kwargs)
        reopened.set()

    monkeypatch.setattr(cloudflare_tunnel, "start_studio_tunnel", _delayed)
    monkeypatch.setattr(cloudflare_tunnel, "ensure_cloudflared", lambda: pytest.fail("stale start"))
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start", lambda: False)
    monkeypatch.setattr(remote_access, "_admin_password_ready", lambda: True)
    remote_access._stop_response_admission_open = False
    assert (
        remote_access.remote_access_status(_state())
        and not remote_access._stop_response_admission_open
    )
    remote_access.start_remote_access(_state())
    assert entered.wait(1) and remote_access._stop_response_admission_open
    old_worker = remote_access._start_worker
    old_token = attempts[0]
    cloudflare_tunnel.close_studio_tunnel_lifecycle()
    cloudflare_tunnel.open_studio_tunnel_lifecycle()
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start", lambda: True)
    assert remote_access.maybe_auto_start_remote_access(_state()) and reopened.wait(1)
    release.set()
    old_worker.join(1)
    current = cloudflare_tunnel.get_studio_tunnel_control_token()
    cloudflare_tunnel.stop_studio_tunnel(admission = old_token)
    assert len(attempts) == 2 and cloudflare_tunnel.get_studio_tunnel_control_token() == current
    assert cloudflare_tunnel.get_studio_tunnel_status()["state"] == "off"


@pytest.mark.parametrize("trigger", ["manual", "auto"])
def test_settings_start_logs_public_url_when_tunnel_is_ready(monkeypatch, trigger):
    monkeypatch.setattr(remote_access, "_start_worker", None)
    monkeypatch.setattr(remote_access, "_start_worker_admission", None)
    ready = threading.Event()
    messages = []
    status = {
        "state": "off",
        "managed_by": None,
        "can_start": True,
        "block_reason": None,
    }

    def _start(*_args, **_kwargs):
        ready.set()
        return "https://example.trycloudflare.com"

    monkeypatch.setattr(remote_access, "get_remote_access_auto_start", lambda: True)
    monkeypatch.setattr(remote_access, "remote_access_status", lambda _: status)
    monkeypatch.setattr(cloudflare_tunnel, "capture_studio_tunnel_start_admission", lambda: (1, 1))
    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_control_token", lambda: (1, 1))
    monkeypatch.setattr(cloudflare_tunnel, "start_studio_tunnel", _start)
    monkeypatch.setattr(
        remote_access.logger,
        "info",
        lambda message, url: messages.append(message % url),
    )

    if trigger == "auto":
        assert remote_access.maybe_auto_start_remote_access(_state())
    else:
        remote_access.start_remote_access(_state())
    assert ready.wait(1)
    remote_access._start_worker.join(1)
    assert messages == ["Secure link access via Cloudflare: https://example.trycloudflare.com"]


@pytest.mark.parametrize("operation", ["start", "stop"])
def test_request_cannot_adopt_reopened_lifecycle(monkeypatch, operation):
    status = (
        {"state": "off", "managed_by": None, "can_start": True}
        if operation == "start"
        else {"state": "online", "managed_by": "settings"}
    )
    satisfied = {
        "state": "online" if operation == "start" else "stopping",
        "managed_by": "settings",
    }
    cloudflare_tunnel.open_studio_tunnel_lifecycle()

    def _status(_state):
        cloudflare_tunnel.close_studio_tunnel_lifecycle()
        cloudflare_tunnel.open_studio_tunnel_lifecycle()
        return {"block_reason": None, **status}

    monkeypatch.setattr(remote_access, "remote_access_status", _status)
    with pytest.raises(RuntimeError, match = "server_lifecycle_changed"):
        getattr(remote_access, f"{operation}_remote_access")(_state())
    monkeypatch.setattr(
        remote_access,
        "remote_access_status",
        lambda _: (cloudflare_tunnel.stop_studio_tunnel(), satisfied)[1],
    )
    assert getattr(remote_access, f"{operation}_remote_access")(_state()) == satisfied


def test_management_rejects_api_keys():
    with pytest.raises(HTTPException) as exc:
        routes._require_ui_session(True)
    assert exc.value.status_code == 403
    assert remote_access.remote_access_status(_state())["streaming_supported"] is True
    source = Path(routes.__file__).read_text(encoding = "utf-8")
    assert source.count("_ui_session: None = Depends(_require_ui_session)") == 4


def test_remote_stop_returns_terminal_state(monkeypatch):
    def _stop(_state):
        return {
            "state": "stopping",
            "url": None,
            "error": None,
            "auto_start": True,
            "available": True,
            "managed_by": "settings",
            "can_start": False,
            "can_stop": False,
            "block_reason": None,
            "streaming_supported": True,
        }

    monkeypatch.setattr(routes, "stop_remote_access", _stop)
    request = SimpleNamespace(app = SimpleNamespace(state = _state()))
    response = routes.stop_remote_access_route(request, "admin", None)
    assert response.state == "off" and response.managed_by is None


def test_stop_response_middleware_holds_lease_through_body(monkeypatch):
    acquired = threading.Event()
    released = threading.Event()

    def _acquire():
        acquired.set()
        return released.set

    monkeypatch.setattr(remote_access, "acquire_remote_access_stop_response", _acquire)

    async def _app(scope, receive, send):
        assert acquired.is_set() and not released.is_set()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        assert not released.is_set()
        await send({"type": "http.response.body", "body": b"{}"})

    async def _send(_message):
        return None

    middleware = remote_access.RemoteAccessStopResponseMiddleware(_app)
    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/settings/remote-access/stop",
            },
            None,
            _send,
        )
    )
    assert released.is_set()
    assert "app.add_middleware(RemoteAccessStopResponseMiddleware)" in (
        _BACKEND / "main.py"
    ).read_text(encoding = "utf-8")


def test_stop_worker_waits_for_every_concurrent_response(monkeypatch):
    remote_access._start_worker = remote_access._stop_worker = None
    remote_access._start_worker_admission = remote_access._stop_worker_admission = None
    remote_access._stop_response_admission_open = True
    stopped = threading.Event()

    release_first = remote_access.acquire_remote_access_stop_response()
    release_second = remote_access.acquire_remote_access_stop_response()
    assert release_first is not None and release_second is not None
    status = {
        "state": "online",
        "managed_by": "settings",
        "block_reason": None,
    }
    monkeypatch.setattr(remote_access, "remote_access_status", lambda _: status)
    monkeypatch.setattr(cloudflare_tunnel, "capture_studio_tunnel_start_admission", lambda: (1, 1))
    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_control_token", lambda: (1, 1))
    monkeypatch.setattr(
        cloudflare_tunnel,
        "get_studio_tunnel_status",
        lambda: {"state": "online", "managed_by": "settings"},
    )
    monkeypatch.setattr(cloudflare_tunnel, "stop_studio_tunnel", lambda **_: stopped.set())

    remote_access.stop_remote_access(_state())
    release_first()
    assert not stopped.wait(0.1)
    release_second()
    assert stopped.wait(1)
    assert remote_access.acquire_remote_access_stop_response() is None
    remote_access._open_remote_access_stop_response_admission()


def test_stop_response_wait_accepts_admission_after_initial_zero(monkeypatch):
    remote_access._stop_responses_pending = 0
    remote_access._stop_response_admission_open = True
    entered_quiet_window, advance_clock, finished = (threading.Event() for _ in range(3))
    real_monotonic = remote_access.time.monotonic
    advanced_at = monotonic_calls = 0

    def _monotonic():
        nonlocal monotonic_calls
        monotonic_calls += 1
        if monotonic_calls == 2:
            entered_quiet_window.set()
        if advance_clock.is_set():
            return 0.1 + real_monotonic() - advanced_at
        return 0.0

    def _wait():
        remote_access._drain_and_close_remote_access_stop_responses()
        finished.set()

    monkeypatch.setattr(remote_access.time, "monotonic", _monotonic)
    waiter = threading.Thread(target = _wait)
    waiter.start()
    assert entered_quiet_window.wait(0.5)
    release = remote_access.acquire_remote_access_stop_response()
    assert release is not None
    assert not finished.wait(0.1)
    advanced_at = real_monotonic()
    advance_clock.set()
    release()
    assert finished.wait(1)
    assert remote_access.acquire_remote_access_stop_response() is None
    remote_access._open_remote_access_stop_response_admission()
    waiter.join()


def test_colab_auto_start_setting_is_read_only(monkeypatch):
    monkeypatch.setattr(routes, "set_remote_access_auto_start", lambda *_: pytest.fail("persisted"))
    request = SimpleNamespace(app = SimpleNamespace(state = _state(is_colab = True)))
    payload = routes.RemoteAccessAutoStartPayload(enabled = True)
    with pytest.raises(HTTPException) as exc:
        routes.update_remote_access_auto_start(request, payload, "admin", None)
    assert exc.value.status_code == 409


def test_unstoppable_connector_reports_why_start_is_blocked(monkeypatch):
    # The generic "Cloudflare tunnel failed" hides the one state the user can
    # act on: a connector whose exit was never confirmed still holds the slot.
    monkeypatch.setattr(remote_access, "_start_worker", None)
    monkeypatch.setattr(remote_access, "_stop_worker", None)
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start", lambda: False)
    monkeypatch.setattr(remote_access, "_admin_password_ready", lambda: True)
    monkeypatch.setattr(
        cloudflare_tunnel,
        "get_studio_tunnel_status",
        lambda: {
            "state": "error",
            "url": None,
            "error": "cloudflared could not be stopped",
            "managed_by": "settings",
            "stop_pending": True,
        },
    )
    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_control_token", lambda: (1, 0))
    assert remote_access.remote_access_status(_state())["error"] == (
        "cloudflared could not be stopped"
    )


def test_stop_does_not_wait_forever_on_a_start_that_never_claims_ownership(monkeypatch):
    # A start worker that stays alive without taking settings ownership (foreign
    # owner, or bailed on admission) must not defer Stop for the probe deadline.
    hold = threading.Event()
    foreign_start = threading.Thread(target = hold.wait, daemon = True)
    foreign_start.start()
    monkeypatch.setattr(remote_access, "_start_worker", foreign_start)
    monkeypatch.setattr(remote_access, "_stop_worker", None)
    monkeypatch.setattr(remote_access, "_STOP_OWNERSHIP_WAIT", 0.1)
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start", lambda: False)
    monkeypatch.setattr(remote_access, "_admin_password_ready", lambda: True)
    monkeypatch.setattr(
        cloudflare_tunnel,
        "get_studio_tunnel_status",
        lambda: {
            "state": "online",
            "url": "https://live.trycloudflare.com",
            "error": None,
            "managed_by": "settings",
            "stop_pending": False,
        },
    )
    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_control_token", lambda: (1, 0))
    monkeypatch.setattr(cloudflare_tunnel, "capture_studio_tunnel_start_admission", lambda: (1, 0))
    stopped = threading.Event()
    monkeypatch.setattr(
        cloudflare_tunnel, "stop_studio_tunnel", lambda **_kw: stopped.set()
    )
    try:
        remote_access.stop_remote_access(_state())
        assert stopped.wait(5), "stop worker never reached stop_studio_tunnel"
    finally:
        hold.set()
        foreign_start.join(timeout = 5)
        remote_access._open_remote_access_stop_response_admission()
