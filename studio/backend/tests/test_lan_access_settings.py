# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ast
import asyncio
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

# module scope, not the fixture: postponed annotations resolve against module globals
from fastapi import FastAPI, HTTPException, Request

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import lan_access  # noqa: E402
import routes.settings as routes  # noqa: E402
import utils.host_policy as host_policy  # noqa: E402
import utils.lan_access_settings as lan_settings  # noqa: E402
from storage import studio_db  # noqa: E402


def _app(**over):
    state = SimpleNamespace(
        lan_access_port = 8888,
        lan_access_ready = True,
        lan_access_is_colab = False,
        lan_access_launch_managed = False,
        lan_access_wildcard_bind = False,
        lan_access_secure_launch = False,
        lan_access_frontend_served = True,
        lan_access_loop = None,
        server_url = "http://192.168.1.24:8888",
    )
    for key, value in over.items():
        setattr(state, key, value)
    return SimpleNamespace(state = state)


@pytest.fixture(autouse = True)
def stored_settings(monkeypatch):
    """Keep every test off the real listener state, password store and database."""
    stored: dict = {}
    monkeypatch.setattr(lan_settings, "_admin_password_ready", lambda: True)
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback: stored.get(key, fallback)
    )
    monkeypatch.setattr(studio_db, "upsert_app_settings", stored.update)
    yield stored
    lan_access.stop_lan_listener()
    lan_access.clear_lan_listener_error()
    host_policy._reset_loopback_default_state()


# ── persisted preference ──


def test_auto_start_persistence_is_strict_and_fail_closed(monkeypatch, stored_settings):
    assert lan_settings.get_lan_access_auto_start() is False
    stored_settings[lan_settings.LAN_ACCESS_AUTO_START_KEY] = "yes"
    assert lan_settings.get_lan_access_auto_start() is False
    assert lan_settings.set_lan_access_auto_start(True) is True
    assert lan_settings.get_lan_access_auto_start() is True
    with pytest.raises(ValueError):
        routes.LanAccessAutoStartPayload(enabled = "true")
    with pytest.raises(ValueError):
        lan_settings.set_lan_access_auto_start("true")
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda *args: (_ for _ in ()).throw(OSError())
    )
    assert lan_settings.get_lan_access_auto_start() is False


# ── launch policy ──


@pytest.mark.parametrize(
    "bind_host,launch_managed",
    [
        ("127.0.0.1", False),
        ("localhost", False),
        ("::1", False),
        ("0.0.0.0", True),
        ("::", True),
        ("192.168.1.24", True),
    ],
)
def test_configure_reads_launch_ownership_from_the_bind_host(bind_host, launch_managed):
    state = SimpleNamespace()
    lan_settings.configure_lan_access(
        state,
        port = 8888,
        bind_host = bind_host,
        secure = False,
        is_colab = False,
        frontend_served = True,
    )
    assert state.lan_access_launch_managed is launch_managed
    assert state.lan_access_ready is False


# ── status ──


def test_a_loopback_launch_offers_a_startable_off_state():
    status = lan_settings.lan_access_status(_app())
    assert status["state"] == "off"
    assert status["block_reason"] is None
    assert status["can_start"] is True and status["can_stop"] is False
    assert status["urls"] == []


@pytest.mark.parametrize(
    "state_over,expected_block",
    [
        ({"lan_access_ready": False}, "server_starting"),
        ({"lan_access_is_colab": True}, "colab"),
        ({"lan_access_launch_managed": True}, "launch_managed"),
        ({"lan_access_secure_launch": True}, "secure_launch"),
    ],
)
def test_every_block_reason_closes_both_controls(state_over, expected_block):
    status = lan_settings.lan_access_status(_app(**state_over))
    assert status["block_reason"] == expected_block
    assert status["can_start"] is False and status["can_stop"] is False


def test_a_pending_admin_password_blocks_exposing_the_server(monkeypatch):
    monkeypatch.setattr(lan_settings, "_admin_password_ready", lambda: False)
    status = lan_settings.lan_access_status(_app())
    assert status["block_reason"] == "admin_password_change_required"
    assert status["can_start"] is False


def test_a_specific_host_launch_reports_the_address_it_was_given():
    status = lan_settings.lan_access_status(_app(lan_access_launch_managed = True))
    assert status["state"] == "online" and status["managed_by"] == "launch"
    assert status["urls"] == ["http://192.168.1.24:8888"]
    assert status["can_stop"] is False


def test_a_wildcard_launch_shows_lan_addresses_not_the_public_sharing_address(monkeypatch):
    """server_url resolves the public IP for sharing, which behind NAT reaches
    nothing on the LAN and would trip the public-address warning as well."""
    monkeypatch.setattr(lan_access, "detect_lan_addresses", lambda: ["192.168.1.24"])
    state = _app(
        lan_access_launch_managed = True,
        lan_access_wildcard_bind = True,
        server_url = "http://203.0.113.9:8888",
    )
    status = lan_settings.lan_access_status(state)
    assert status["urls"] == ["http://192.168.1.24:8888"]
    assert status["public_urls"] == []

    # detection is not repeated on every poll
    monkeypatch.setattr(
        lan_access, "detect_lan_addresses", lambda: pytest.fail("re-detected on a poll")
    )
    assert lan_settings.lan_access_status(state)["urls"] == ["http://192.168.1.24:8888"]


def test_a_publicly_routable_bind_is_flagged_so_the_ui_can_warn(monkeypatch):
    """A VPS carries its public IPv4 on the NIC, so "LAN access" can reach the
    internet. Binding it is the operator's call; hiding it is not."""
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {
            "running": True,
            "addresses": ["192.168.1.24", "64.227.100.5"],
            "port": 8888,
            "error": None,
        },
    )
    status = lan_settings.lan_access_status(_app())
    assert status["urls"] == ["http://192.168.1.24:8888", "http://64.227.100.5:8888"]
    assert status["public_urls"] == ["http://64.227.100.5:8888"]


@pytest.mark.parametrize(
    "address,public",
    [
        ("192.168.1.24", False),
        ("10.0.0.7", False),
        ("172.16.3.9", False),
        # CGNAT, which is where a Tailscale address lands
        ("100.64.1.1", False),
        ("64.227.100.5", True),
        ("8.8.8.8", True),
        ("not-an-ip", False),
    ],
)
def test_public_address_classification(address, public):
    assert lan_access.is_public_address(address) is public


def test_a_private_bind_carries_no_public_warning(monkeypatch):
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": True, "addresses": ["192.168.1.24"], "port": 8888, "error": None},
    )
    assert lan_settings.lan_access_status(_app())["public_urls"] == []


def test_api_only_launches_advertise_that_the_web_ui_is_not_served():
    assert lan_settings.lan_access_status(_app())["serves_web_ui"] is True
    served = lan_settings.lan_access_status(_app(lan_access_frontend_served = False))
    assert served["serves_web_ui"] is False


def test_a_failed_start_is_reported_as_an_error_state(monkeypatch):
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": False, "addresses": [], "port": None, "error": "no_lan_address"},
    )
    status = lan_settings.lan_access_status(_app())
    assert status["state"] == "error" and status["error"] == "no_lan_address"
    # the failure is retryable: nothing about it blocks a second attempt
    assert status["can_start"] is True


# ── address detection ──


def test_detection_drops_addresses_no_other_device_can_open(monkeypatch):
    monkeypatch.setattr(
        lan_access.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (None, None, None, None, (address, 0))
            for address in (
                "127.0.0.1",
                "169.254.10.1",
                "224.0.0.1",
                "0.0.0.0",
                "not-an-ip",
                "10.0.0.7",
                "10.0.0.7",
            )
        ],
    )
    addresses = lan_access.detect_lan_addresses()
    assert "10.0.0.7" in addresses
    assert addresses.count("10.0.0.7") == 1
    for rejected in ("127.0.0.1", "169.254.10.1", "224.0.0.1", "0.0.0.0", "not-an-ip"):
        assert rejected not in addresses


@pytest.mark.allow_network
def test_the_default_route_address_leads_so_it_becomes_the_shown_url(monkeypatch):
    routed = _require_lan_address()
    monkeypatch.setattr(
        lan_access.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("203.0.113.9", 0))],
    )
    assert lan_access.detect_lan_addresses() == [routed, "203.0.113.9"]


def test_detection_survives_a_host_that_resolves_to_nothing(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise OSError("no such host")

    monkeypatch.setattr(lan_access.socket, "getaddrinfo", _boom)
    # under the suite's outbound-network guard the UDP probe is refused too, so
    # a failed lookup must contribute nothing rather than raise
    assert lan_access.detect_lan_addresses() == []


# ── live listener ──


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _get(url: str) -> int:
    with urllib.request.urlopen(url, timeout = 5) as response:
        return response.status


def _free_port_is_bindable(port: int) -> bool:
    """True when the LAN address can still take ``port``, so nothing leaked a socket."""
    try:
        _bind_listener_probe = lan_access._bind_listener(_require_lan_address(), port)
    except OSError:
        return False
    _bind_listener_probe.close()
    return True


def _require_lan_address() -> str:
    """The address these tests bind, or a skip on a host with no network address."""
    addresses = lan_access.detect_lan_addresses()
    if not addresses:
        pytest.skip("this host has no LAN address to bind")
    return addresses[0]


def _refused(url: str) -> bool:
    try:
        _get(url)
    except (urllib.error.URLError, OSError):
        return True
    return False


@pytest.fixture
def live_server():
    """A loopback uvicorn server on its own loop, the shape run.py starts."""
    import uvicorn

    app = FastAPI()

    @app.get("/where")
    def where(request: Request):
        return {"lan": lan_access.request_on_lan_listener(request.scope)}

    @app.post("/stop-lan")
    def stop_lan():
        started = time.monotonic()
        lan_access.stop_lan_listener()
        return {"seconds": time.monotonic() - started}

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host = "127.0.0.1", port = port, log_level = "warning"))
    box = {}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        box["loop"] = loop
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target = _run, daemon = True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "primary server never bound"
    yield SimpleNamespace(app = app, loop = box["loop"], port = port)
    lan_access.stop_lan_listener()
    server.should_exit = True
    thread.join(timeout = 10)


@pytest.mark.allow_network
def test_the_listener_adds_and_removes_network_reach_without_a_restart(live_server):
    address = _require_lan_address()
    lan_url = f"http://{address}:{live_server.port}/where"
    local_url = f"http://127.0.0.1:{live_server.port}/where"

    assert _get(local_url) == 200
    assert _refused(lan_url), "the port was already on the network before starting"

    bound = lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)
    assert address in bound
    assert _get(lan_url) == 200
    assert _get(local_url) == 200, "loopback must keep serving while the LAN listener runs"
    assert lan_access.lan_listener_status()["running"] is True

    # a second start is a no-op rather than a second bind on the same address
    again = lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)
    assert again == bound

    lan_access.stop_lan_listener()
    assert _refused(lan_url), "the LAN socket outlived stop"
    assert _get(local_url) == 200
    assert lan_access.lan_listener_status() == {
        "running": False,
        "addresses": [],
        "port": None,
        "error": None,
    }

    # the same address rebinds, so a stop does not strand the port until restart
    lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)
    assert _get(lan_url) == 200


@pytest.mark.allow_network
def test_only_traffic_on_a_lan_socket_is_reported_as_lan(live_server):
    address = _require_lan_address()
    lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)

    def _lan_flag(host: str) -> bool:
        with urllib.request.urlopen(
            f"http://{host}:{live_server.port}/where", timeout = 5
        ) as response:
            return json.loads(response.read())["lan"]

    assert _lan_flag(address) is True
    assert _lan_flag("127.0.0.1") is False, "loopback traffic must never read as LAN"


def test_a_start_with_no_usable_address_fails_without_leaving_state(monkeypatch):
    monkeypatch.setattr(lan_access, "detect_lan_addresses", lambda: [])
    with pytest.raises(RuntimeError, match = "no_lan_address"):
        lan_access.start_lan_listener(object(), object(), 8888)
    status = lan_access.lan_listener_status()
    assert status["running"] is False and status["error"] == "no_lan_address"
    lan_access.clear_lan_listener_error()
    assert lan_access.lan_listener_status()["error"] is None


def test_a_port_already_taken_on_every_address_fails_closed(monkeypatch):
    monkeypatch.setattr(lan_access, "detect_lan_addresses", lambda: ["10.0.0.7"])

    def _refuse(*_args, **_kwargs):
        raise OSError("address in use")

    monkeypatch.setattr(lan_access, "_bind_listener", _refuse)
    with pytest.raises(RuntimeError, match = "bind_failed"):
        lan_access.start_lan_listener(object(), object(), 8888)
    assert lan_access.lan_listener_status()["running"] is False


@pytest.mark.allow_network
def test_an_unbindable_address_does_not_sink_the_ones_that_work(live_server, monkeypatch):
    address = _require_lan_address()
    monkeypatch.setattr(lan_access, "detect_lan_addresses", lambda: [address, "10.255.255.254"])
    bound = lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)
    assert bound == (address,)
    assert _get(f"http://{address}:{live_server.port}/where") == 200


@pytest.mark.allow_network
def test_stop_from_inside_the_event_loop_does_not_wait_on_itself(live_server):
    """/api/shutdown tears down from a task on the serving loop; blocking there deadlocks."""
    lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)

    async def _stop_from_loop():
        started = time.monotonic()
        lan_access.stop_lan_listener()
        return time.monotonic() - started

    elapsed = asyncio.run_coroutine_threadsafe(_stop_from_loop(), live_server.loop).result(
        timeout = 5
    )
    assert elapsed < lan_access._STOP_TIMEOUT
    assert lan_access.lan_listener_status()["running"] is False


@pytest.mark.allow_network
def test_a_listener_that_fails_to_serve_gives_up_without_waiting_it_out(live_server, monkeypatch):
    _require_lan_address()

    class _Broken:
        started = False
        should_exit = False

        async def serve(self, sockets = None):
            raise RuntimeError("listener exploded")

    monkeypatch.setattr(lan_access.uvicorn, "Server", lambda _config: _Broken())
    started = time.monotonic()
    with pytest.raises(RuntimeError, match = "listener_start_failed"):
        lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)
    assert time.monotonic() - started < lan_access._START_TIMEOUT
    status = lan_access.lan_listener_status()
    assert status["running"] is False and status["error"] == "listener_start_failed"
    # the sockets it opened were released, so the port is free again
    assert _free_port_is_bindable(live_server.port)


@pytest.mark.allow_network
def test_a_stop_from_a_lan_client_does_not_wait_out_its_own_response(live_server):
    """Settings > Stop pressed on the phone. uvicorn closes the listening sockets at
    the top of its shutdown and only then drains in-flight responses, so a stop that
    waited for serve() to return would be waiting for the very response it is."""
    address = _require_lan_address()
    lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)

    request = urllib.request.Request(f"http://{address}:{live_server.port}/stop-lan", method = "POST")
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout = 30) as response:
        handler_seconds = json.loads(response.read())["seconds"]
    assert handler_seconds < lan_access._STOP_TIMEOUT, "the stop waited out its own response"
    assert time.monotonic() - started < lan_access._STOP_TIMEOUT
    assert lan_access.lan_listener_status()["running"] is False
    assert _refused(f"http://{address}:{live_server.port}/where")


def test_a_stop_that_cannot_confirm_the_port_keeps_the_host_marked_reachable(monkeypatch):
    """The sockets may still be accepting, so the loopback-only trust gates must
    not reopen just because the teardown wait ran out."""

    class _Server:
        should_exit = False

    lingering = socket.socket()
    lingering.bind(("127.0.0.1", 0))
    loop = asyncio.new_event_loop()
    running = threading.Event()

    def _spin():
        loop.call_soon(running.set)
        loop.run_forever()

    thread = threading.Thread(target = _spin, daemon = True)
    thread.start()
    assert running.wait(5)
    monkeypatch.setattr(lan_access, "_STOP_TIMEOUT", 0.2)
    monkeypatch.setattr(lan_access, "_server", _Server())
    monkeypatch.setattr(lan_access, "_serve_loop", loop)
    monkeypatch.setattr(lan_access, "_sockets", (lingering,))
    monkeypatch.setattr(lan_access, "_bound_addresses", ("10.0.0.7",))
    host_policy.set_lan_connector_active(True)

    try:
        app = _app()
        status = lan_settings.stop_lan_access(app)
        assert status["error"] == "stop_timed_out"
        assert host_policy.remote_connector_active() is True
        # ownership is retained, so the stop stays offered and a retry re-waits
        assert status["can_stop"] is True
        assert lan_access.lan_listener_status()["running"] is True

        # a second stop must not report success while the port may still accept
        retried = lan_settings.stop_lan_access(app)
        assert retried["error"] == "stop_timed_out"
        assert host_policy.remote_connector_active() is True

        # once the socket really closes, the retry settles and clears the error
        lingering.close()
        settled = lan_settings.stop_lan_access(app)
        assert settled["state"] == "off" and settled["error"] is None
        assert host_policy.remote_connector_active() is False
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout = 5)
        loop.close()
        lingering.close()
        host_policy.set_lan_connector_active(False)


def test_stop_releases_the_sockets_itself_when_the_serving_loop_is_gone(monkeypatch):
    """An abnormal exit leaves nobody to run uvicorn's shutdown, so stop must close them."""

    class _Server:
        should_exit = False

    stranded = socket.socket()
    stranded.bind(("127.0.0.1", 0))
    dead = asyncio.new_event_loop()
    dead.close()
    monkeypatch.setattr(lan_access, "_server", _Server())
    monkeypatch.setattr(lan_access, "_serve_loop", dead)
    monkeypatch.setattr(lan_access, "_sockets", (stranded,))
    monkeypatch.setattr(lan_access, "_bound_addresses", ("10.0.0.7",))

    started = time.monotonic()
    lan_access.stop_lan_listener()
    assert time.monotonic() - started < 1
    assert stranded.fileno() == -1, "the socket outlived the loop that owned it"
    assert lan_access.lan_listener_status()["running"] is False


# ── settings orchestration ──


def _configured(live_server):
    """The live app carrying the launch policy run.py would have published on it."""
    lan_settings.configure_lan_access(
        live_server.app.state,
        port = live_server.port,
        bind_host = "127.0.0.1",
        secure = False,
        is_colab = False,
        frontend_served = True,
    )
    live_server.app.state.lan_access_ready = True
    live_server.app.state.lan_access_loop = live_server.loop
    return live_server.app


@pytest.mark.allow_network
def test_start_and_stop_track_the_beyond_loopback_trust_flag(live_server):
    app = _configured(live_server)
    assert host_policy.remote_connector_active() is False

    status = lan_settings.start_lan_access(app)
    assert status["state"] == "online" and status["managed_by"] == "settings"
    assert status["urls"] and status["urls"][0].endswith(f":{live_server.port}")
    assert status["can_stop"] is True and status["can_start"] is False
    assert host_policy.remote_connector_active() is True

    # a repeated start is answered with the running listener, not a second bind
    assert lan_settings.start_lan_access(app)["urls"] == status["urls"]

    stopped = lan_settings.stop_lan_access(app)
    assert stopped["state"] == "off" and stopped["managed_by"] is None
    assert stopped["can_start"] is True
    assert host_policy.remote_connector_active() is False


@pytest.mark.allow_network
def test_auto_start_brings_the_listener_up_at_boot(live_server, stored_settings):
    stored_settings[lan_settings.LAN_ACCESS_AUTO_START_KEY] = True
    app = _configured(live_server)
    assert lan_settings.maybe_auto_start_lan_access(app) is True
    assert lan_settings.lan_access_status(app)["state"] == "online"
    # stopping now must not silently clear the preference
    lan_settings.stop_lan_access(app)
    assert lan_settings.get_lan_access_auto_start() is True


def test_start_refuses_while_a_block_is_in_force():
    with pytest.raises(RuntimeError, match = "server_starting"):
        lan_settings.start_lan_access(_app(lan_access_ready = False))
    with pytest.raises(RuntimeError, match = "launch_managed"):
        lan_settings.stop_lan_access(_app(lan_access_launch_managed = True))


def test_start_refuses_a_port_the_server_never_published():
    with pytest.raises(RuntimeError, match = "server_port_unavailable"):
        lan_settings.start_lan_access(_app(lan_access_port = None))


def test_start_refuses_once_the_server_loop_is_gone(monkeypatch):
    monkeypatch.setattr(
        lan_access, "start_lan_listener", lambda *_: pytest.fail("bound without a live loop")
    )
    with pytest.raises(RuntimeError, match = "server_not_running"):
        lan_settings.start_lan_access(_app(lan_access_loop = None))
    # a loop that exists but is no longer serving would never run the listener either
    idle = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match = "server_not_running"):
            lan_settings.start_lan_access(_app(lan_access_loop = idle))
    finally:
        idle.close()
    with pytest.raises(RuntimeError, match = "server_not_running"):
        lan_settings.start_lan_access(_app(lan_access_loop = idle))


def test_auto_start_stays_quiet_when_the_launch_forbids_it(stored_settings):
    stored_settings[lan_settings.LAN_ACCESS_AUTO_START_KEY] = True
    assert lan_settings.maybe_auto_start_lan_access(_app(lan_access_is_colab = True)) is False


def test_auto_start_is_skipped_entirely_when_the_preference_is_off(monkeypatch):
    monkeypatch.setattr(
        lan_settings, "start_lan_access", lambda _app: pytest.fail("started despite the preference")
    )
    assert lan_settings.maybe_auto_start_lan_access(_app()) is False


# ── routes and wiring ──


def test_colab_auto_start_setting_is_read_only(monkeypatch):
    monkeypatch.setattr(routes, "set_lan_access_auto_start", lambda *_: pytest.fail("persisted"))
    request = SimpleNamespace(app = _app(lan_access_is_colab = True))
    payload = routes.LanAccessAutoStartPayload(enabled = True)
    with pytest.raises(HTTPException) as exc:
        routes.update_lan_access_auto_start(request, payload, "admin", None)
    assert exc.value.status_code == 409


def test_a_blocked_start_answers_409_rather_than_500(monkeypatch):
    monkeypatch.setattr(
        routes, "start_lan_access", lambda _app: (_ for _ in ()).throw(RuntimeError("colab"))
    )
    with pytest.raises(HTTPException) as exc:
        routes.start_lan_access_route(SimpleNamespace(app = _app()), "admin", None)
    assert exc.value.status_code == 409 and exc.value.detail == "colab"


def test_management_rejects_api_keys():
    with pytest.raises(HTTPException) as exc:
        routes._require_ui_session(True)
    assert exc.value.status_code == 403
    # every /lan-access handler must carry the UI-session gate
    tree = ast.parse(Path(routes.__file__).read_text(encoding = "utf-8"))
    gated = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        routed = [ast.unparse(d) for d in node.decorator_list if "router." in ast.unparse(d)]
        if not any("'/lan-access" in d for d in routed):
            continue
        args = node.args.args + node.args.kwonlyargs
        gated[node.name] = any(a.arg == "_ui_session" for a in args)
    assert len(gated) == 4, f"expected 4 lan-access handlers, found {sorted(gated)}"
    assert all(
        gated.values()
    ), f"ungated lan-access handlers: {sorted(k for k, v in gated.items() if not v)}"


def test_the_desktop_frontend_gate_admits_the_lan_listener():
    """A desktop backend mounts its SPA behind a remote-only gate; LAN must open it."""
    import main

    scope = {"type": "http", "headers": [], "server": ("10.0.0.7", 8888)}
    state = SimpleNamespace(cloudflare_url = None)
    assert main._is_remote_frontend_request(scope, state) is False
    original = lan_access._bound_addresses
    lan_access._bound_addresses = ("10.0.0.7",)
    try:
        assert main._is_remote_frontend_request(scope, state) is True
        loopback = {"type": "http", "headers": [], "server": ("127.0.0.1", 8888)}
        assert main._is_remote_frontend_request(loopback, state) is False
    finally:
        lan_access._bound_addresses = original


def test_the_desktop_assets_mount_admits_the_lan_listener():
    """The SPA is mounted twice on a desktop api-only launch: /assets through
    _TunnelOnlyFrontend, and the routes through _frontend_request_allowed."""
    import main

    served = []

    async def _spa(_scope, _receive, send):
        served.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def _drive(scope):
        statuses = []

        async def _send(message):
            if message["type"] == "http.response.start":
                statuses.append(message["status"])

        await main._TunnelOnlyFrontend(_spa, SimpleNamespace(cloudflare_url = None))(
            scope, None, _send
        )
        return statuses[0]

    lan = {"type": "http", "headers": [], "server": ("10.0.0.7", 8888)}
    loopback = {"type": "http", "headers": [], "server": ("127.0.0.1", 8888)}
    assert asyncio.run(_drive(lan)) == 404

    original = lan_access._bound_addresses
    lan_access._bound_addresses = ("10.0.0.7",)
    try:
        assert asyncio.run(_drive(lan)) == 200
        assert asyncio.run(_drive(loopback)) == 404, "loopback keeps the api-only surface"
    finally:
        lan_access._bound_addresses = original
    assert served

    # the SPA routes share the decision through a closure the mount does not expose
    source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert "return not tunnel_only or _is_remote_frontend_request(request.scope, app.state)" in (
        source
    )


def test_run_py_wires_the_listener_into_the_server_lifecycle():
    source = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    # without the loop the settings route has nothing to schedule the listener on
    assert "app.state.lan_access_loop = loop" in source
    assert "close_lan_listener_lifecycle" in source
    # readiness is what unblocks the whole feature, and it has to land before auto-start
    ready = source.index("app.state.lan_access_ready = True")
    assert ready < source.index("maybe_auto_start_lan_access(app)")
