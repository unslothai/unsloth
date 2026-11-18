# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Wayland backend against a fake xdg-desktop-portal on a private session bus."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("jeepney")
if not sys.platform.startswith("linux") or shutil.which("dbus-daemon") is None:
    pytest.skip("needs Linux and dbus-daemon", allow_module_level = True)

from jeepney import DBusAddress, HeaderFields, MessageType, new_error, new_method_return, new_signal
from jeepney.bus_messages import message_bus
from jeepney.io.blocking import open_dbus_connection
from PIL import Image

from core.inference.computer_use import backends
from core.inference.computer_use.backends import BackendError, _portal
from core.inference.computer_use.backends.linux import LinuxBackend
from core.inference.computer_use.backends.wayland import WaylandBackend

_CONFIG = """<!DOCTYPE busconfig PUBLIC "-//freedesktop//DTD D-Bus Bus Configuration 1.0//EN"
 "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig><type>session</type><listen>unix:abstract={name}</listen>
<policy context="default"><allow send_destination="*" eavesdrop="true"/><allow eavesdrop="true"/>
<allow own="*"/></policy></busconfig>"""

LAYOUT = [(41, (0, 0), (1920, 1080)), (42, (1920, 0), (1280, 1024))]


class FakePortal:
    """org.freedesktop.portal.Desktop: RemoteDesktop + ScreenCast, recording every call."""

    def __init__(self, address: str) -> None:
        self.conn = open_dbus_connection(bus = address, enable_fds = True)
        self.conn.send_and_get_reply(message_bus.RequestName("org.freedesktop.portal.Desktop"))
        self.calls: list[tuple[str, tuple]] = []
        self.sessions: list[str] = []
        self.answer = 0
        self.answer_delay = 0.0
        self.version = 2
        self.devices = 3
        self.restore = "token-1"
        self.stop = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target = self._serve, daemon = True)
        self.thread.start()

    def _send(self, message) -> None:
        with self.lock:
            try:
                self.conn.send(message)
            except OSError:
                pass

    def notify(self) -> list[tuple]:
        return [(name, body[2:]) for name, body in self.calls if name.startswith("Notify")]

    def close_session(self) -> None:
        handle = self.sessions[-1]
        self._send(
            new_signal(
                DBusAddress(handle, interface = "org.freedesktop.portal.Session"),
                "Closed",
                "a{sv}",
                ({},),
            )
        )

    def _respond(
        self,
        sender: str,
        token: str,
        results: dict,
        code: int = 0,
    ) -> None:
        path = f"/org/freedesktop/portal/desktop/request/{sender.lstrip(':').replace('.', '_')}/{token}"

        def later():
            time.sleep(self.answer_delay if code else 0)
            # Broadcast, as the portal does: only the client's match rule brings it in.
            self._send(
                new_signal(
                    DBusAddress(path, interface = "org.freedesktop.portal.Request"),
                    "Response",
                    "ua{sv}",
                    (code, results),
                )
            )

        threading.Thread(target = later, daemon = True).start()
        return path

    def _serve(self) -> None:
        while not self.stop:
            try:
                msg = self.conn.receive(timeout = 0.2)
            except TimeoutError:
                continue
            except Exception:  # noqa: BLE001
                return
            if msg.header.message_type != MessageType.method_call:
                continue
            fields = msg.header.fields
            member = fields[HeaderFields.member]
            interface = fields.get(HeaderFields.interface)
            sender = fields[HeaderFields.sender]
            self.calls.append((member, msg.body))
            reply = self._handle(interface, member, msg.body, sender)
            if isinstance(reply, str):
                self._send(new_error(msg, "org.freedesktop.DBus.Error.Failed", "s", (reply,)))
            else:
                self._send(new_method_return(msg, *reply))

    def _handle(self, interface, member, body, sender):
        if member == "Get":
            name = body[1]
            values = {
                "version": ("u", self.version),
                "AvailableDeviceTypes": ("u", self.devices),
                "AvailableSourceTypes": ("u", 1 | 2),
                "AvailableCursorModes": ("u", 1 | 2),
            }
            return ("v", (values[name],))
        if member == "CreateSession":
            handle = (
                f"/org/freedesktop/portal/desktop/session/x/{body[0]['session_handle_token'][1]}"
            )
            self.sessions.append(handle)
            path = self._respond(
                sender, body[0]["handle_token"][1], {"session_handle": ("s", handle)}
            )
            return ("o", (path,))
        if member in ("SelectDevices", "SelectSources"):
            return ("o", (self._respond(sender, body[1]["handle_token"][1], {}),))
        if member == "Start":
            streams = [
                (node, {"position": ("(ii)", position), "size": ("(ii)", size)})
                for node, position, size in LAYOUT
            ]
            results = {
                "devices": ("u", 3),
                "streams": ("a(ua{sv})", streams),
                "restore_token": ("s", self.restore),
            }
            code = self.answer
            return (
                "o",
                (
                    self._respond(
                        sender, body[2]["handle_token"][1], results if code == 0 else {}, code
                    ),
                ),
            )
        if member == "OpenPipeWireRemote":
            left, right = socket.socketpair()
            self.keep = (left, right)
            return ("h", (right.fileno(),))
        if member == "Close":
            return ("", ())
        if member.startswith("Notify"):
            if body[0] not in self.sessions[-1:]:
                return "session is closed"
            return ("", ())
        return ("", ())


@pytest.fixture
def bus(tmp_path, monkeypatch):
    config = tmp_path / "bus.conf"
    config.write_text(_CONFIG.format(name = f"unsloth-test-{os.getpid()}-{time.monotonic_ns()}"))
    daemon = subprocess.Popen(
        ["dbus-daemon", "--config-file", str(config), "--print-address", "--nofork"],
        stdout = subprocess.PIPE,
        text = True,
    )
    address = daemon.stdout.readline().strip()
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", address)
    portal = FakePortal(address)
    yield portal
    portal.stop = True
    portal.thread.join(timeout = 2)
    daemon.terminate()
    daemon.wait(timeout = 5)


class FakeGrabber:
    made: list = []

    def __init__(self, open_fd, streams):
        self.open_fd, self.closed = open_fd, False
        self.nodes = [stream.node for stream in streams]
        self.sizes = {stream.node: (stream.width, stream.height) for stream in streams}
        FakeGrabber.made.append(self)

    def grab(
        self,
        node,
        timeout = 3.0,
    ):
        fd = self.open_fd()
        os.close(fd)
        return Image.new("RGB", (64, 32), (node, 0, 0))

    def close(self):
        self.closed = True


def _backend(tmp_path, **kwargs) -> WaylandBackend:
    kwargs.setdefault("idle_close", 0)
    return WaylandBackend(grabber = FakeGrabber, token_path = tmp_path / "token", **kwargs)


def test_availability_opens_no_session(bus, tmp_path):
    backend = _backend(tmp_path)
    assert backend.unavailable_reason() is None
    assert backend.permissions() == {"screen": None, "input": None}
    assert backend.front_window().app == ""
    assert not [name for name, _ in bus.calls if name != "Get"]
    with pytest.raises(BackendError):
        backend.cursor()
    assert not [name for name, _ in bus.calls if name != "Get"]


def test_first_run_opens_one_session_with_persistence(bus, tmp_path):
    (tmp_path / "token").write_text("token-0")
    backend = _backend(tmp_path)
    displays = backend.displays()
    backend.displays()
    assert [name for name, _ in bus.calls].count("CreateSession") == 1
    devices = next(body for name, body in bus.calls if name == "SelectDevices")[1]
    assert devices["types"] == ("u", 3)
    assert devices["persist_mode"] == ("u", 2)
    assert devices["restore_token"] == ("s", "token-0")
    sources = next(body for name, body in bus.calls if name == "SelectSources")[1]
    assert sources["multiple"] == ("b", True) and sources["types"] == ("u", 1)
    assert sources["cursor_mode"] == ("u", 1)
    assert "persist_mode" not in sources and "restore_token" not in sources
    assert (tmp_path / "token").read_text() == "token-1"
    assert [(d.index, d.x, d.y, d.width, d.height, d.primary) for d in displays] == [
        (1, 0, 0, 1920, 1080, True),
        (2, 1920, 0, 1280, 1024, False),
    ]
    assert backend.cursor() == (960, 540)


def test_no_restore_options_before_portal_version_two(bus, tmp_path):
    bus.version = 1
    backend = _backend(tmp_path)
    backend.displays()
    devices = next(body for name, body in bus.calls if name == "SelectDevices")[1]
    assert "persist_mode" not in devices and "restore_token" not in devices


def test_input_reaches_the_portal_relative_to_its_monitor(bus, tmp_path):
    backend = _backend(tmp_path)
    backend.displays()
    backend.click(2000, 100, "left", 2, ("ctrl",))
    # The first motion is sent twice: mutter drops a session's first one.
    assert bus.notify() == [
        ("NotifyPointerMotionAbsolute", (42, 80.0, 100.0)),
        ("NotifyPointerMotionAbsolute", (42, 80.0, 100.0)),
        ("NotifyKeyboardKeysym", (0xFFE3, 1)),
        ("NotifyPointerButton", (0x110, 1)),
        ("NotifyPointerButton", (0x110, 0)),
        ("NotifyPointerButton", (0x110, 1)),
        ("NotifyPointerButton", (0x110, 0)),
        ("NotifyKeyboardKeysym", (0xFFE3, 0)),
    ]
    assert backend.cursor() == (2000, 100)
    bus.calls.clear()
    backend.drag([(10, 10), (30, 40)])
    assert bus.notify() == [
        ("NotifyPointerMotionAbsolute", (41, 10.0, 10.0)),
        ("NotifyPointerButton", (0x110, 1)),
        ("NotifyPointerMotionAbsolute", (41, 30.0, 40.0)),
        ("NotifyPointerButton", (0x110, 0)),
    ]
    bus.calls.clear()
    backend.scroll(-1, 3)
    backend.key(("ctrl", "shift"), "t")
    backend.type_text("Hi\n\té")
    assert bus.notify() == [
        ("NotifyPointerAxisDiscrete", (0, 3)),
        ("NotifyPointerAxisDiscrete", (1, -1)),
        ("NotifyKeyboardKeysym", (0xFFE3, 1)),
        ("NotifyKeyboardKeysym", (0xFFE1, 1)),
        ("NotifyKeyboardKeysym", (0x74, 1)),
        ("NotifyKeyboardKeysym", (0x74, 0)),
        ("NotifyKeyboardKeysym", (0xFFE1, 0)),
        ("NotifyKeyboardKeysym", (0xFFE3, 0)),
        *[
            ("NotifyKeyboardKeysym", (keysym, state))
            for keysym in (0x48, 0x69, 0xFF0D, 0xFF09, 0xE9)
            for state in (1, 0)
        ],
    ]


def test_capture_reads_the_display_stream(bus, tmp_path):
    FakeGrabber.made.clear()
    backend = _backend(tmp_path)
    second = backend.displays()[1]
    image = backend.capture(second)
    assert image.getpixel((0, 0)) == (42, 0, 0)
    assert FakeGrabber.made[-1].nodes == [41, 42]
    assert FakeGrabber.made[-1].sizes == {41: (1920, 1080), 42: (1280, 1024)}
    assert "OpenPipeWireRemote" in [name for name, _ in bus.calls]


def test_a_declined_prompt_is_reported_and_retried_next_time(bus, tmp_path):
    bus.answer = 1
    backend = _backend(tmp_path)
    with pytest.raises(BackendError, match = "declined"):
        backend.displays()
    bus.answer = 0
    assert len(backend.displays()) == 2
    assert [name for name, _ in bus.calls].count("CreateSession") == 2


def test_an_unanswered_prompt_times_out(bus, tmp_path):
    bus.answer, bus.answer_delay = 2, 5.0
    backend = _backend(tmp_path, answer_timeout = 0.5)
    with pytest.raises(BackendError, match = "not answered"):
        backend.displays()


def test_a_session_the_user_stopped_stops_and_the_portal_asks_again(bus, tmp_path):
    backend = _backend(tmp_path)
    backend.displays()
    assert (tmp_path / "token").read_text() == "token-1"
    bus.close_session()
    time.sleep(0.2)
    with pytest.raises(BackendError, match = "Do not try again"):
        backend.move(5, 5)
    assert not (tmp_path / "token").exists()
    backend.displays()
    assert [name for name, _ in bus.calls].count("CreateSession") == 2
    reopened = [body for name, body in bus.calls if name == "SelectDevices"][-1][1]
    assert "restore_token" not in reopened
    backend.move(5, 5)


def test_a_capture_failing_because_the_user_stopped_sharing_stops_too(bus, tmp_path):
    class Failing(FakeGrabber):
        def grab(
            self,
            node,
            timeout = 3.0,
        ):
            bus.close_session()
            time.sleep(0.2)
            raise RuntimeError("stream ended")

    backend = WaylandBackend(grabber = Failing, token_path = tmp_path / "token")
    display = backend.displays()[0]
    with pytest.raises(BackendError, match = "Do not try again"):
        backend.capture(display)
    assert not (tmp_path / "token").exists()


def test_a_user_stop_seen_only_by_the_idle_timer_still_drops_the_grant(bus, tmp_path):
    backend = _backend(tmp_path, idle_close = 0.5)
    backend.displays()
    bus.close_session()
    time.sleep(1.0)
    assert backend._session is None
    assert not (tmp_path / "token").exists()


def test_an_idle_closed_session_is_restored_without_asking(bus, tmp_path):
    backend = _backend(tmp_path, idle_close = 0.3)
    backend.displays()
    time.sleep(0.8)
    with pytest.raises(BackendError, match = "Take a screenshot"):
        backend.move(5, 5)
    backend.displays()
    restored = [body for name, body in bus.calls if name == "SelectDevices"][-1][1]
    assert restored["restore_token"] == ("s", "token-1")


def test_an_idle_session_is_closed(bus, tmp_path):
    backend = _backend(tmp_path, idle_close = 0.3)
    backend.displays()
    time.sleep(0.8)
    assert backend._session is None
    assert "Close" in [name for name, _ in bus.calls]


def test_missing_keyboard_or_pointer_is_a_reason(bus, tmp_path):
    bus.devices = 2
    assert "keyboard and pointer" in _backend(tmp_path).unavailable_reason()


def test_no_portal_is_a_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", f"unix:path={tmp_path}/missing")
    reason = _backend(tmp_path).unavailable_reason()
    assert "Wayland" in reason and "portal" in reason


def test_missing_jeepney_is_a_reason(tmp_path, monkeypatch):
    def missing():
        raise ImportError("jeepney")

    monkeypatch.setattr(_portal, "_jeepney", missing)
    assert "Update Unsloth" in _backend(tmp_path).unavailable_reason()


@pytest.mark.parametrize(
    "env, wayland",
    [
        ({"DISPLAY": ":0"}, False),
        ({"DISPLAY": ":0", "XDG_SESSION_TYPE": "x11"}, False),
        ({"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}, True),
        ({"XDG_SESSION_TYPE": "wayland"}, True),
    ],
)
def test_routing_keeps_x11_sessions_on_the_x11_backend(monkeypatch, env, wayland):
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", backends.BACKEND_ENV):
        monkeypatch.delenv(name, raising = False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(LinuxBackend, "_load", lambda self: None)
    created = backends._create()
    assert isinstance(created, WaylandBackend) is wayland
    assert isinstance(created, LinuxBackend)


def test_a_tool_run_shoots_and_clicks_through_the_portal_asking_each_time(
    bus, tmp_path, monkeypatch
):
    from core.inference import computer_use
    from core.inference.computer_use import policy, screen
    from state import computer_use as state
    from utils import host_policy

    host_policy._reset_loopback_default_state()
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    backend = _backend(tmp_path)
    monkeypatch.setattr(backends, "_backend", backend)
    monkeypatch.setattr(
        policy,
        "settings",
        lambda: {"enabled": True, "coordinates": "auto", "allowed_apps": [], "blocked_apps": []},
    )
    monkeypatch.setattr(screen, "_SETTLE_FIRST_S", 0.0)
    monkeypatch.setattr(screen, "_SETTLE_STEP_S", 0.0)
    state.reset()
    try:
        shot = computer_use.execute_computer_tool(
            {"action": "screenshot"}, session_id = "w", approved = True
        )
        assert not str(shot).startswith("Error:")
        click = {"action": "left_click", "x": 10, "y": 10}
        assert policy.needs_confirmation(click, "w")
        assert policy.UNIDENTIFIED in policy.call_asks(click, "w")["risks"]
        refused = computer_use.execute_computer_tool(click, session_id = "w")
        assert refused.startswith("Error:")
        assert not [n for n, _ in bus.notify() if n == "NotifyPointerButton"]
        computer_use.execute_computer_tool(click, session_id = "w", approved = True)
        assert [body for n, body in bus.notify() if n == "NotifyPointerButton"] == [
            (0x110, 1),
            (0x110, 0),
        ]
    finally:
        state.reset()
        host_policy._reset_loopback_default_state()
        backend.close()
