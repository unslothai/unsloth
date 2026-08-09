# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

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
    malformed_json = object()

    class StoredConnection:
        def execute(self, _sql, _parameters):
            rows = [
                {
                    "key": key,
                    "value_json": "{" if value is malformed_json else json.dumps(value),
                }
                for key, value in stored.items()
            ]
            return SimpleNamespace(fetchall = lambda: rows)

        def close(self):
            return None

    monkeypatch.setattr(studio_db, "get_connection", StoredConnection)
    assert remote_access.get_remote_access_auto_start() is False
    stored[remote_access.REMOTE_ACCESS_AUTO_START_KEY] = True
    assert remote_access.get_remote_access_auto_start_kind() == "temporary"
    stored[remote_access.REMOTE_ACCESS_AUTO_START_KEY] = False
    assert remote_access.get_remote_access_auto_start_kind() is None
    stored[remote_access.REMOTE_ACCESS_AUTO_START_KEY] = {"version": 1, "mode": "temporary"}
    assert remote_access.get_remote_access_auto_start_kind() == "temporary"
    stored[remote_access.REMOTE_ACCESS_AUTO_START_KEY] = "yes"
    assert remote_access.get_remote_access_auto_start() is False
    for malformed in (
        {"version": True, "mode": "custom"},
        {"version": 1.0, "mode": "custom"},
        {"version": 1, "mode": []},
        {"version": 2, "mode": "custom"},
        {"version": 1, "mode": "future"},
        {"version": 1, "mode": "disabled"},
    ):
        stored[remote_access.REMOTE_ACCESS_AUTO_START_KEY] = malformed
        assert remote_access.get_remote_access_auto_start_kind() is None
    stored[remote_access.REMOTE_ACCESS_METHOD_KEY] = "custom"
    stored[remote_access.REMOTE_ACCESS_AUTO_START_KEY] = {"version": 1, "mode": "temporary"}
    assert remote_access.get_remote_access_auto_start_kind() == "custom"
    stored[remote_access.REMOTE_ACCESS_METHOD_KEY] = "future"
    stored[remote_access.REMOTE_ACCESS_AUTO_START_KEY] = {"version": 1, "mode": "custom"}
    assert remote_access.get_remote_access_method() == "temporary"
    assert remote_access.get_remote_access_auto_start_kind() == "temporary"
    stored[remote_access.REMOTE_ACCESS_METHOD_KEY] = malformed_json
    assert remote_access.get_remote_access_method() == "temporary"
    assert remote_access.get_remote_access_auto_start_kind() == "temporary"
    stored.pop(remote_access.REMOTE_ACCESS_METHOD_KEY)
    assert remote_access.get_remote_access_method() == "custom"
    with pytest.raises(ValueError):
        routes.RemoteAccessAutoStartPayload(enabled = "true")
    monkeypatch.setattr(studio_db, "get_connection", lambda: (_ for _ in ()).throw(OSError()))
    assert remote_access.get_remote_access_auto_start() is False


def test_remote_access_method_is_persisted_and_keeps_auto_start_aligned(monkeypatch, tmp_path):
    database = tmp_path / "settings.db"
    conn = sqlite3.connect(database)
    conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value_json TEXT, updated_at TEXT)"
    )
    conn.commit()
    conn.close()

    def _connect():
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(studio_db, "get_connection", _connect)
    assert remote_access.set_remote_access_method("custom") == "custom"
    conn = _connect()
    assert (
        json.loads(
            conn.execute(
                "SELECT value_json FROM app_settings WHERE key = ?",
                (remote_access.REMOTE_ACCESS_METHOD_KEY,),
            ).fetchone()["value_json"]
        )
        == "custom"
    )
    assert json.loads(
        conn.execute(
            "SELECT value_json FROM app_settings WHERE key = ?",
            (remote_access.REMOTE_ACCESS_AUTO_START_KEY,),
        ).fetchone()["value_json"]
    ) == {"version": 1, "mode": "disabled"}
    conn.close()
    assert remote_access.set_remote_access_auto_start(True)
    assert remote_access.set_remote_access_method("temporary") == "temporary"
    conn = _connect()
    assert json.loads(
        conn.execute(
            "SELECT value_json FROM app_settings WHERE key = ?",
            (remote_access.REMOTE_ACCESS_AUTO_START_KEY,),
        ).fetchone()["value_json"]
    ) == {"version": 1, "mode": "temporary"}
    conn.close()
    assert not remote_access.set_remote_access_auto_start(False)
    with pytest.raises(ValueError):
        remote_access.set_remote_access_method("future")
    with pytest.raises(ValueError):
        routes.RemoteAccessMethodPayload(method = "future")


def test_remote_access_preference_updates_serialize_method_and_auto_start(monkeypatch, tmp_path):
    database = tmp_path / "settings.db"
    conn = sqlite3.connect(database)
    conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value_json TEXT, updated_at TEXT)"
    )
    conn.executemany(
        "INSERT INTO app_settings (key, value_json, updated_at) VALUES (?, ?, '')",
        [
            (remote_access.REMOTE_ACCESS_METHOD_KEY, json.dumps("temporary")),
            (
                remote_access.REMOTE_ACCESS_AUTO_START_KEY,
                json.dumps({"version": 1, "mode": "temporary"}),
            ),
        ],
    )
    conn.commit()
    conn.close()
    method_read = threading.Event()
    allow_method_write = threading.Event()

    class PausingConnection(sqlite3.Connection):
        def execute(
            self,
            sql,
            parameters = (),
            /,
        ):
            result = super().execute(sql, parameters)
            if threading.current_thread().name == "method-update" and sql.startswith("SELECT key"):
                method_read.set()
                assert allow_method_write.wait(1)
            return result

    def _connect():
        connection = sqlite3.connect(database, timeout = 2, factory = PausingConnection)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(studio_db, "get_connection", _connect)
    method_thread = threading.Thread(
        target = remote_access.set_remote_access_method,
        args = ("custom",),
        name = "method-update",
    )
    auto_thread = threading.Thread(
        target = remote_access.set_remote_access_auto_start,
        args = (False,),
        name = "auto-update",
    )
    method_thread.start()
    assert method_read.wait(1)
    auto_thread.start()
    allow_method_write.set()
    method_thread.join(2)
    auto_thread.join(2)
    assert not method_thread.is_alive() and not auto_thread.is_alive()
    conn = _connect()
    values = {
        row["key"]: json.loads(row["value_json"])
        for row in conn.execute("SELECT key, value_json FROM app_settings")
    }
    conn.close()
    assert values == {
        remote_access.REMOTE_ACCESS_METHOD_KEY: "custom",
        remote_access.REMOTE_ACCESS_AUTO_START_KEY: {"version": 1, "mode": "disabled"},
    }


@pytest.mark.parametrize(
    "launch_managed,expected_block,can_start",
    [(False, None, True), (True, "launch_managed", False)],
)
def test_enabled_intent_blocks_only_selected_launch_path(
    monkeypatch, launch_managed, expected_block, can_start
):
    preference_reads = []
    monkeypatch.setattr(remote_access, "_start_worker", None)
    monkeypatch.setattr(remote_access, "_stop_worker", None)

    def _preferences():
        preference_reads.append(True)
        return "temporary", False

    monkeypatch.setattr(remote_access, "_get_remote_access_preferences", _preferences)
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
    assert status["method"] == "temporary"
    assert status["auto_start"] is False
    assert preference_reads == [True]

    # A pending password is reported on its own, even where another block hides it.
    monkeypatch.setattr(remote_access, "_admin_password_ready", lambda: False)
    pending = remote_access.remote_access_status(state)
    assert pending["password_pending"] is True
    assert pending["block_reason"] == (expected_block or "admin_password_change_required")


def test_unconfigured_custom_method_is_not_startable(monkeypatch):
    monkeypatch.setattr(remote_access, "_start_worker", None)
    monkeypatch.setattr(remote_access, "_stop_worker", None)
    monkeypatch.setattr(remote_access, "_get_remote_access_preferences", lambda: ("custom", False))
    monkeypatch.setattr(remote_access, "_admin_password_ready", lambda: True)
    monkeypatch.setattr(
        remote_access,
        "_custom_status",
        lambda: {"custom_runnable": False, "custom_state": "unconfigured"},
    )
    monkeypatch.setattr(
        cloudflare_tunnel,
        "get_studio_tunnel_status",
        lambda: {"state": "off", "url": None, "error": None, "managed_by": None},
    )
    monkeypatch.setattr(cloudflare_tunnel, "capture_studio_tunnel_start_admission", lambda: (1, 0))
    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_control_token", lambda: (1, 0))
    status = remote_access.remote_access_status(_state(intent = "enabled"))
    assert status["method"] == "custom" and status["can_start"] is False
    with pytest.raises(RuntimeError, match = "custom_tunnel_not_configured"):
        remote_access.start_remote_access(_state(intent = "enabled"))
    monkeypatch.setattr(remote_access, "_admin_password_ready", lambda: False)
    with pytest.raises(RuntimeError, match = "admin_password_change_required"):
        remote_access.start_remote_access(_state(intent = "enabled"))


def test_custom_status_exposes_the_owned_tunnel_name(monkeypatch):
    identity = {
        "hostname": "studio.example.com",
        "tunnel_name": "unsloth-AB12CD",
    }
    monkeypatch.setattr(cloudflare_tunnel, "read_identity", lambda: identity)
    monkeypatch.setattr(cloudflare_tunnel, "identity_is_runnable", lambda _identity: True)
    monkeypatch.setattr(cloudflare_tunnel, "orphaned_hostnames", lambda: [])
    monkeypatch.setattr(remote_access, "_custom_worker", None)
    monkeypatch.setattr(remote_access, "_custom_operation", "idle")
    monkeypatch.setattr(remote_access, "_custom_error", None)
    status = remote_access._custom_status()
    assert status["custom_hostname"] == "studio.example.com"
    assert status["custom_tunnel_name"] == "unsloth-AB12CD"


@pytest.mark.parametrize("stop_succeeds", [True, False])
def test_custom_teardown_stops_the_connector_before_local_cleanup(monkeypatch, stop_succeeds):
    events = []
    tunnel = {
        "state": "online",
        "kind": "custom",
        "managed_by": "settings",
        "stop_pending": False,
    }
    monkeypatch.setattr(remote_access, "_custom_worker", None)
    monkeypatch.setattr(remote_access, "_stop_worker", None)
    monkeypatch.setattr(remote_access, "_custom_operation_allowed", lambda _state: None)
    monkeypatch.setattr(remote_access, "remote_access_status", lambda _state: {"state": "off"})
    monkeypatch.setattr(remote_access, "clear_custom_remote_access_auto_start", lambda: False)
    monkeypatch.setattr(cloudflare_tunnel, "read_identity", lambda: {"hostname": "host.test"})
    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_status", lambda: dict(tunnel))

    def _stop(_state, *, kind):
        events.append(("stop", kind))
        if stop_succeeds:
            tunnel.update(state = "off", managed_by = None)

    def _teardown(*, clear_auto_start):
        events.append(
            (
                "teardown",
                clear_auto_start is remote_access.clear_custom_remote_access_auto_start,
            )
        )

    monkeypatch.setattr(remote_access, "stop_remote_access", _stop)
    monkeypatch.setattr(cloudflare_tunnel, "teardown_custom_tunnel", _teardown)
    remote_access.teardown_custom_remote_access(_state())
    remote_access._custom_worker.join(1)
    assert events[0] == ("stop", "custom")
    if stop_succeeds:
        assert events[1] == ("teardown", True)
        assert remote_access._custom_operation == "idle"
    else:
        assert events == [("stop", "custom")]
        assert remote_access._custom_error[:3] == (
            "connector_stop_failed",
            "The Cloudflare connector could not be stopped.",
            "teardown",
        )


def test_failed_stop_remains_retryable(monkeypatch):
    monkeypatch.setattr(remote_access, "_start_worker", None)
    monkeypatch.setattr(remote_access, "_stop_worker", None)
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start_kind", lambda: None)
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
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start_kind", lambda: None)
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
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start_kind", lambda: None)
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
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start_kind", lambda: "temporary")
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
        "method": "temporary",
        "custom_state": "configured",
        "custom_runnable": True,
    }

    started_kinds = []

    def _start(*_args, **kwargs):
        started_kinds.append(kwargs["kind"])
        ready.set()
        return "https://example.trycloudflare.com"

    monkeypatch.setattr(remote_access, "get_remote_access_auto_start_kind", lambda: "temporary")
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
    assert started_kinds == ["temporary"]
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


def test_remote_access_http_boundary_and_ui_session_gate(monkeypatch):
    status = {
        "state": "online",
        "url": "https://studio.example.com",
        "error": None,
        "auto_start": True,
        "method": "custom",
        "available": True,
        "managed_by": "settings",
        "can_start": False,
        "can_stop": True,
        "kind": "custom",
        "dns": "pending",
        "connector_registered": True,
        "tunnel_serving": True,
        "custom_state": "provisioning",
        "custom_hostname": "studio.example.com",
        "custom_tunnel_name": "unsloth-AB12CD",
        "custom_runnable": True,
        "login_url": "https://dash.cloudflare.com/login",
        "custom_error": "dns_conflict",
        "custom_error_detail": "record exists",
        "custom_error_phase": "provision",
        "custom_error_settled": False,
        "orphaned_hostnames": ["old.example.com"],
    }
    via_key = [False]
    monkeypatch.setattr(routes, "remote_access_status", lambda _state: status)
    monkeypatch.setattr(routes, "provision_custom_remote_access", lambda *_args: status)
    monkeypatch.setattr(routes, "cancel_custom_remote_access", lambda *_args: status)
    selected_methods = []
    monkeypatch.setattr(routes, "set_remote_access_method", selected_methods.append)
    app = FastAPI()
    for name, value in vars(_state()).items():
        setattr(app.state, name, value)
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_current_subject] = lambda: "admin"
    app.dependency_overrides[routes.authenticated_via_api_key] = lambda: via_key[0]
    client = TestClient(app)
    body = client.get("/remote-access").json()
    assert (body["kind"], body["dns"], body["connector_registered"], body["tunnel_serving"]) == (
        "custom",
        "pending",
        True,
        True,
    )
    assert (
        body["method"],
        body["custom_tunnel_name"],
        body["login_url"],
        body["custom_error"],
        body["orphaned_hostnames"],
    ) == (
        "custom",
        "unsloth-AB12CD",
        status["login_url"],
        "dns_conflict",
        ["old.example.com"],
    )
    assert (
        body["custom_state"],
        body["custom_hostname"],
        body["custom_runnable"],
        body["custom_error_detail"],
        body["custom_error_phase"],
        body["custom_error_settled"],
    ) == ("provisioning", "studio.example.com", True, "record exists", "provision", False)
    assert client.post(
        "/remote-access/custom/provision", json = {"hostname": "studio.example.com"}
    ).json()["login_url"]
    assert client.post("/remote-access/custom/cancel").json()["custom_error"] == "dns_conflict"
    assert client.put("/remote-access/method", json = {"method": "custom"}).json()["method"] == (
        "custom"
    )
    assert selected_methods == ["custom"]
    via_key[0] = True
    requests = [
        ("GET", "/remote-access", None),
        ("POST", "/remote-access/start", None),
        ("POST", "/remote-access/stop", None),
        ("PUT", "/remote-access/auto-start", {"enabled": True}),
        ("PUT", "/remote-access/method", {"method": "temporary"}),
        ("POST", "/remote-access/custom/provision", {"hostname": "studio.example.com"}),
        ("POST", "/remote-access/custom/cancel", None),
        ("POST", "/remote-access/custom/teardown", None),
    ]
    assert all(
        client.request(method, path, json = payload).status_code == 403
        for method, path, payload in requests
    )


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


@pytest.mark.parametrize(
    "path",
    [
        "/api/settings/remote-access/stop",
        "/api/settings/remote-access/custom/teardown",
    ],
)
def test_stop_response_middleware_holds_lease_through_body(monkeypatch, path):
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
                "path": path,
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
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start_kind", lambda: None)
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
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start_kind", lambda: None)
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
    monkeypatch.setattr(cloudflare_tunnel, "stop_studio_tunnel", lambda **_kw: stopped.set())
    try:
        remote_access.stop_remote_access(_state())
        assert stopped.wait(5), "stop worker never reached stop_studio_tunnel"
    finally:
        hold.set()
        foreign_start.join(timeout = 5)
        remote_access._open_remote_access_stop_response_admission()


def test_streaming_depends_on_the_active_tunnel_kind(monkeypatch):
    monkeypatch.setattr(remote_access, "_start_worker", None)
    monkeypatch.setattr(remote_access, "_stop_worker", None)
    monkeypatch.setattr(remote_access, "get_remote_access_auto_start_kind", lambda: None)
    monkeypatch.setattr(remote_access, "_admin_password_ready", lambda: True)
    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_control_token", lambda: (1, 0))

    def _status(url):
        return {
            "state": "online" if url else "off",
            "url": url,
            "error": None,
            "managed_by": "settings" if url else None,
            "stop_pending": False,
        }

    monkeypatch.setattr(cloudflare_tunnel, "get_studio_tunnel_status", lambda: _status(None))
    assert remote_access.remote_access_status(_state())["streaming_supported"] is True

    monkeypatch.setattr(
        cloudflare_tunnel,
        "get_studio_tunnel_status",
        lambda: _status("https://live.trycloudflare.com"),
    )
    assert remote_access.remote_access_status(_state())["streaming_supported"] is False

    monkeypatch.setattr(
        cloudflare_tunnel,
        "get_studio_tunnel_status",
        lambda: {
            **_status("https://stable.example.com"),
            "kind": "custom",
            "dns": "pending",
            "connector_registered": True,
            "tunnel_serving": True,
        },
    )
    custom = remote_access.remote_access_status(_state())
    assert custom["streaming_supported"] is True
    assert (custom["connector_registered"], custom["tunnel_serving"], custom["dns"]) == (
        True,
        True,
        "pending",
    )
