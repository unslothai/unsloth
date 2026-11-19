# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""xdg-desktop-portal RemoteDesktop + ScreenCast session over D-Bus (jeepney)."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .base import BackendError

_BUS_NAME = "org.freedesktop.portal.Desktop"
_PATH = "/org/freedesktop/portal/desktop"
_REMOTE = "org.freedesktop.portal.RemoteDesktop"
_CAST = "org.freedesktop.portal.ScreenCast"
_REQUEST = "org.freedesktop.portal.Request"
_SESSION = "org.freedesktop.portal.Session"

KEYBOARD, POINTER = 1, 2
MONITOR = 1
CURSOR_HIDDEN, CURSOR_EMBEDDED, CURSOR_METADATA = 1, 2, 4
PERSIST_UNTIL_REVOKED = 2
_CALL_TIMEOUT_S = 10.0
_PROBE_TIMEOUT_S = 5.0

_NO_PORTAL = (
    "This is a Wayland session, and computer use there needs the desktop's remote desktop "
    "portal, but {what}. Install xdg-desktop-portal with your desktop's backend "
    "(xdg-desktop-portal-kde or xdg-desktop-portal-gnome), or sign in with an X11 (Xorg) session."
)


@dataclass(frozen = True)
class Stream:
    node: int
    x: float
    y: float
    width: float
    height: float
    # Size and position are optional in the portal's reply; a sizeless stream is measured from a frame.
    placed: bool = True


def _jeepney():
    import jeepney
    from jeepney.io.blocking import open_dbus_connection
    return jeepney, open_dbus_connection


def _check(reply: Any) -> tuple:
    jeepney, _ = _jeepney()
    if reply.header.message_type == jeepney.MessageType.error:
        raise jeepney.DBusErrorResponse(reply)
    return reply.body


def _property(conn: Any, interface: str, name: str) -> Any:
    jeepney, _ = _jeepney()
    address = jeepney.DBusAddress(_PATH, bus_name = _BUS_NAME, interface = interface)
    reply = conn.send_and_get_reply(jeepney.Properties(address).get(name), timeout = _PROBE_TIMEOUT_S)
    return _check(reply)[0][1]


def probe() -> tuple[Optional[str], int]:
    """(why the portal path can't work, RemoteDesktop version). Opens no session."""
    try:
        _, open_dbus_connection = _jeepney()
    except ImportError:
        return (
            "This is a Wayland session, and computer use there needs a component this install "
            "of Unsloth is missing. Update Unsloth, then restart it."
        ), 0
    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS") and not os.environ.get("XDG_RUNTIME_DIR"):
        return _NO_PORTAL.format(what = "there is no desktop session bus to reach it on"), 0
    try:
        conn = open_dbus_connection(bus = "SESSION")
    except Exception:  # noqa: BLE001 - no bus, refused auth, bad address
        return _NO_PORTAL.format(what = "the desktop session bus could not be reached"), 0
    try:
        try:
            version = int(_property(conn, _REMOTE, "version"))
            devices = int(_property(conn, _REMOTE, "AvailableDeviceTypes"))
            sources = int(_property(conn, _CAST, "AvailableSourceTypes"))
        except Exception:  # noqa: BLE001 - not activatable, no such interface, timeout
            return _NO_PORTAL.format(what = "no remote desktop portal answered"), 0
    finally:
        conn.close()
    if devices & (KEYBOARD | POINTER) != KEYBOARD | POINTER:
        return _NO_PORTAL.format(what = "it offers no keyboard and pointer control"), version
    if not sources & MONITOR:
        return _NO_PORTAL.format(what = "it can't share a screen"), version
    return None, version


def _sender_path(unique_name: str) -> str:
    return unique_name.lstrip(":").replace(".", "_")


class PortalSession:
    """One started RemoteDesktop session with its screen-cast streams. Not thread safe: callers lock."""

    def __init__(
        self,
        version: int,
        token_path: Optional[Path] = None,
        answer_timeout: float = 180.0,
    ) -> None:
        self._version = version
        self._token_path = token_path
        self._answer_timeout = answer_timeout
        self._conn: Any = None
        self._closed_filter: Any = None
        self.handle = ""
        self.streams: list[Stream] = []

    @property
    def closed(self) -> bool:
        if self._conn is None:
            return True
        try:
            while True:
                self._conn.recv_messages(timeout = 0)
        except TimeoutError:
            pass
        except Exception:  # noqa: BLE001 - the bus connection itself went away
            self._drop()
            return True
        if self._closed_filter is not None and self._closed_filter.queue:
            self._drop()
            return True
        return False

    def _call(
        self,
        interface: str,
        method: str,
        signature: str,
        body: tuple,
        path: str = _PATH,
    ):
        jeepney, _ = _jeepney()
        address = jeepney.DBusAddress(path, bus_name = _BUS_NAME, interface = interface)
        message = jeepney.new_method_call(address, method, signature, body)
        return _check(self._conn.send_and_get_reply(message, timeout = _CALL_TIMEOUT_S))

    def _bus_call(self, method: str, rule: Any) -> None:
        jeepney, _ = _jeepney()
        message = jeepney.new_method_call(jeepney.message_bus, method, "s", (rule.serialise(),))
        _check(self._conn.send_and_get_reply(message, timeout = _CALL_TIMEOUT_S))

    def _request(
        self, interface: str, method: str, signature: str, body: tuple, timeout: float
    ) -> dict:
        """Call a method answered by a Request's Response signal; subscribed before the call."""
        jeepney, _ = _jeepney()
        token = "unsloth" + secrets.token_hex(8)
        options = dict(body[-1])
        options["handle_token"] = ("s", token)
        body = body[:-1] + (options,)
        path = f"{_PATH}/request/{_sender_path(self._conn.unique_name)}/{token}"
        rule = jeepney.MatchRule(type = "signal", interface = _REQUEST, member = "Response", path = path)
        self._bus_call("AddMatch", rule)
        try:
            with self._conn.filter(rule) as queue:
                self._call(interface, method, signature, body)
                try:
                    signal = self._conn.recv_until_filtered(queue, timeout = timeout)
                except TimeoutError:
                    try:
                        self._call(_REQUEST, "Close", "", (), path = path)
                    except Exception:  # noqa: BLE001 - the request may be gone already
                        pass
                    raise
        finally:
            try:
                self._bus_call("RemoveMatch", rule)
            except Exception:  # noqa: BLE001
                pass
        response, results = signal.body
        if response != 0:
            raise BackendError(
                "the user declined the screen sharing and remote control prompt, so the "
                "screen can't be seen or controlled. Tell the user; do not try again unless asked."
            )
        return {key: value for key, (_signature, value) in results.items()}

    def _read_token(self) -> str:
        if self._token_path is None:
            return ""
        try:
            return self._token_path.read_text(encoding = "utf-8").strip()
        except OSError:
            return ""

    def _write_token(self, token: str) -> None:
        if self._token_path is None or not token:
            return
        try:
            self._token_path.parent.mkdir(parents = True, exist_ok = True)
            temporary = self._token_path.with_name(self._token_path.name + ".tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding = "utf-8") as handle:
                handle.write(token)
            os.replace(temporary, self._token_path)
        except OSError:
            pass

    def start(self) -> list[Stream]:
        jeepney, open_dbus_connection = _jeepney()
        self._conn = open_dbus_connection(bus = "SESSION", enable_fds = True)
        try:
            created = self._request(
                _REMOTE,
                "CreateSession",
                "a{sv}",
                ({"session_handle_token": ("s", "unsloth" + secrets.token_hex(8))},),
                _CALL_TIMEOUT_S,
            )
            self.handle = str(created["session_handle"])
            closed = jeepney.MatchRule(
                type = "signal", interface = _SESSION, member = "Closed", path = self.handle
            )
            self._bus_call("AddMatch", closed)
            self._closed_filter = self._conn.filter(closed)
            devices: dict = {"types": ("u", KEYBOARD | POINTER)}
            if self._version >= 2:
                devices["persist_mode"] = ("u", PERSIST_UNTIL_REVOKED)
                token = self._read_token()
                if token:
                    devices["restore_token"] = ("s", token)
            self._request(
                _REMOTE, "SelectDevices", "oa{sv}", (self.handle, devices), _CALL_TIMEOUT_S
            )
            try:
                modes = int(_property(self._conn, _CAST, "AvailableCursorModes"))
            except Exception:  # noqa: BLE001 - a version 1 portal: its default is hidden
                modes = 0
            sources: dict = {"types": ("u", MONITOR), "multiple": ("b", True)}
            # An unadvertised cursor mode closes the session; screen.take draws its own pointer.
            mode = next(
                (m for m in (CURSOR_HIDDEN, CURSOR_EMBEDDED, CURSOR_METADATA) if modes & m), None
            )
            if mode is not None:
                sources["cursor_mode"] = ("u", mode)
            self._request(
                _CAST,
                "SelectSources",
                "oa{sv}",
                (self.handle, sources),
                _CALL_TIMEOUT_S,
            )
            try:
                started = self._request(
                    _REMOTE, "Start", "osa{sv}", (self.handle, "", {}), self._answer_timeout
                )
            except TimeoutError:
                raise BackendError(
                    "the screen sharing prompt was not answered in time, so it was closed. Ask "
                    "the user to watch for a new one, then try again."
                ) from None
            if int(started.get("devices", KEYBOARD | POINTER)) & (KEYBOARD | POINTER) != (
                KEYBOARD | POINTER
            ):
                raise BackendError(
                    "keyboard and pointer control were not allowed in the screen sharing "
                    "prompt. Tell the user; do not try again unless asked."
                )
            self._write_token(str(started.get("restore_token", "")))
            self.streams = self._streams(started.get("streams") or [])
            if not self.streams:
                raise BackendError("no screen was shared in the screen sharing prompt.")
            return self.streams
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _streams(raw: list) -> list[Stream]:
        found, left = [], 0.0
        for node, properties in raw:
            properties = {key: value for key, (_signature, value) in properties.items()}
            size = properties.get("size") or (0, 0)
            position = properties.get("position")
            x, y = position if position else (left, 0)
            found.append(
                Stream(
                    int(node),
                    float(x),
                    float(y),
                    float(size[0]),
                    float(size[1]),
                    placed = bool(position) or bool(size[0]),
                )
            )
            left = max(left, float(x) + float(size[0]))
        return found

    def pipewire_fd(self) -> int:
        (descriptor,) = self._call(_CAST, "OpenPipeWireRemote", "oa{sv}", (self.handle, {}))
        return descriptor.to_raw_fd() if hasattr(descriptor, "to_raw_fd") else int(descriptor)

    def pointer(self, stream: Stream, x: float, y: float) -> None:
        self._call(
            _REMOTE,
            "NotifyPointerMotionAbsolute",
            "oa{sv}udd",
            (self.handle, {}, stream.node, float(x - stream.x), float(y - stream.y)),
        )

    def button(self, code: int, down: bool) -> None:
        self._call(_REMOTE, "NotifyPointerButton", "oa{sv}iu", (self.handle, {}, code, int(down)))

    def axis(self, axis: int, steps: int) -> None:
        self._call(_REMOTE, "NotifyPointerAxisDiscrete", "oa{sv}ui", (self.handle, {}, axis, steps))

    def keysym(self, keysym: int, down: bool) -> None:
        self._call(
            _REMOTE, "NotifyKeyboardKeysym", "oa{sv}iu", (self.handle, {}, keysym, int(down))
        )

    def _drop(self) -> None:
        conn, self._conn = self._conn, None
        if self._closed_filter is not None:
            self._closed_filter.close()
            self._closed_filter = None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        if self._conn is not None and self.handle:
            try:
                self._call(_SESSION, "Close", "", (), path = self.handle)
            except Exception:  # noqa: BLE001 - already closed by the user or the portal
                pass
        self._drop()
        self.streams = []
