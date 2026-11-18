# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linux on Wayland through the desktop's RemoteDesktop + ScreenCast portal.

Wayland shows no other app's windows to a client, so every window is unidentified and the
policy asks the user before each call.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import _portal
from .base import BackendError, Display, FrontWindow
from .linux import _UNICODE_KEYSYM, LinuxBackend

_BUTTONS = {"left": 0x110, "right": 0x111, "middle": 0x112}
_VERTICAL, _HORIZONTAL = 0, 1
_KEYSYM_RETURN, _KEYSYM_TAB = 0xFF0D, 0xFF09
_F1 = 0xFFBE
# keys.py names a closed set, so the X keysym values are spelled out (keysymdef.h).
_KEYSYMS = {
    "ctrl": 0xFFE3, "alt": 0xFFE9, "shift": 0xFFE1, "cmd": 0xFFEB, "enter": 0xFF0D,
    "tab": 0xFF09, "space": 0x20, "backspace": 0xFF08, "delete": 0xFFFF, "escape": 0xFF1B,
    "up": 0xFF52, "down": 0xFF54, "left": 0xFF51, "right": 0xFF53, "home": 0xFF50,
    "end": 0xFF57, "pageup": 0xFF55, "pagedown": 0xFF56, "insert": 0xFF63, "capslock": 0xFFE5,
}  # fmt: skip
IDLE_CLOSE_S = 300.0
_STOPPED = (
    "the screen sharing session ended (the user may have stopped it), so this action was not "
    "done. Take a screenshot to start it again."
)
_USER_STOPPED = (
    "the user stopped the screen sharing, so computer use has stopped. Do not try again; tell "
    "the user."
)


def _token_path() -> Optional[Path]:
    try:
        from utils.paths import studio_root
        return studio_root() / "security" / "computer_use_portal_token"
    except Exception:  # noqa: BLE001 - no persistence: the user is asked each session
        return None


def _default_grabber(open_fd: Callable[[], int], streams: list[_portal.Stream]) -> Any:
    from ._pipewire import Grabber
    return Grabber(
        open_fd,
        [stream.node for stream in streams],
        {stream.node: (int(stream.width), int(stream.height)) for stream in streams},
    )


def _capture_problem() -> Optional[str]:
    try:
        from ._pipewire import available
    except ImportError:
        return "This install of Unsloth can't capture a Wayland screen. Update Unsloth, then restart it."
    return available()


class WaylandBackend(LinuxBackend):
    name = "linux"

    def __init__(
        self,
        grabber: Callable[[Callable[[], int], list[_portal.Stream]], Any] = _default_grabber,
        token_path: Optional[Path] = None,
        answer_timeout: float = 180.0,
        idle_close: float = IDLE_CLOSE_S,
    ) -> None:
        self._lock = threading.RLock()
        self._launcher = None
        self._make_grabber = grabber
        self._token_file = token_path if token_path is not None else _token_path()
        self._answer_timeout = answer_timeout
        self._idle_close = idle_close
        self._session: Optional[_portal.PortalSession] = None
        self._grabber: Any = None
        self._displays: list[tuple[Display, _portal.Stream]] = []
        self._pointer: Optional[tuple[float, float]] = None
        self._timer: Optional[threading.Timer] = None
        self._primed = False
        # Set by a user stop; only the next run's displays() lifts it, not this run's last screenshot.
        self._held = False
        self._last_use = 0.0
        self._problem: Optional[str] = None
        self._version = 0
        self._probed = False
        self._probe_lock = threading.Lock()

    def unavailable_reason(self) -> Optional[str]:
        # Not under _lock: Settings reads this while a run waits on the user's portal answer.
        with self._probe_lock:
            if not self._probed:
                self._problem, self._version = _portal.probe()
                if self._problem is None and self._make_grabber is _default_grabber:
                    self._problem = _capture_problem()
                self._probed = True
            return self._problem

    def _ready(self) -> None:
        problem = self.unavailable_reason()
        if problem:
            raise BackendError(problem)

    def _started(self) -> _portal.PortalSession:
        """Call with the lock held: the running session, started (the user may be asked) if none."""
        self._ready()
        self._check_stopped()
        if self._session is None:
            session = _portal.PortalSession(self._version, self._token_file, self._answer_timeout)
            streams = session.start()
            self._session = session
            self._primed = False
            self._displays = [
                (
                    Display(index, stream.x, stream.y, stream.width, stream.height, index == 1),
                    stream,
                )
                for index, stream in enumerate(streams, start = 1)
            ]
            if self._pointer is None:
                first = self._displays[0][0]
                self._pointer = (first.x + first.width / 2, first.y + first.height / 2)
        self._touch()
        return self._session

    def _live(self) -> _portal.PortalSession:
        """Call with the lock held. Input never starts a session: only a run's displays() does."""
        self._ready()
        self._check_stopped()
        if self._session is None:
            raise BackendError(_STOPPED)
        self._touch()
        return self._session

    def _check_stopped(self) -> None:
        """Call with the lock held."""
        if self._session is not None and self._session.closed:
            self._end()
            raise BackendError(_USER_STOPPED)

    def _touch(self) -> None:
        self._last_use = time.monotonic()
        if self._timer is None and self._idle_close > 0:
            self._arm(self._idle_close)

    def _arm(self, delay: float) -> None:
        self._timer = threading.Timer(delay, self._idle)
        self._timer.daemon = True
        self._timer.start()

    def _idle(self) -> None:
        with self._lock:
            if self._timer is None or self._session is None:
                return
            left = self._last_use + self._idle_close - time.monotonic()
            if left > 0:
                self._arm(left)
            else:
                self._end()

    def _end(self) -> bool:
        """Call with the lock held. True when the desktop had closed the session: the user stopped
        it, so the saved grant is dropped and the portal asks again instead of restoring it unseen."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        grabber, self._grabber = self._grabber, None
        if grabber is not None:
            try:
                grabber.close()
            except Exception:  # noqa: BLE001
                pass
        session, self._session = self._session, None
        stopped = session is not None and session.closed
        if stopped:
            self._held = True
        if stopped and self._token_file is not None:
            try:
                self._token_file.unlink(missing_ok = True)
            except OSError:
                pass
        if session is not None:
            session.close()
        self._displays = []
        return stopped

    def close(self) -> None:
        with self._lock:
            self._end()

    def _send(self, action: Callable[[_portal.PortalSession], None]) -> None:
        """Call with the lock held."""
        session = self._live()
        try:
            action(session)
        except BackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - a closed session or a lost bus
            raise BackendError(_USER_STOPPED if self._end() else _STOPPED) from exc

    def displays(self) -> list[Display]:
        with self._lock:
            self._held = False
            self._started()
            return [display for display, _stream in self._displays]

    def _open_fd(self) -> int:
        with self._lock:
            if self._session is None:
                raise BackendError(_STOPPED)
            return self._session.pipewire_fd()

    def capture(self, display: Display) -> Any:
        with self._lock:
            if self._held:
                raise BackendError(_USER_STOPPED)
            self._started()
            stream = next(
                (stream for shown, stream in self._displays if shown.index == display.index),
                self._displays[0][1],
            )
            if self._grabber is None:
                self._grabber = self._make_grabber(
                    self._open_fd, [stream for _shown, stream in self._displays]
                )
            grabber = self._grabber
        try:
            return grabber.grab(stream.node).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - the stream ended or never sent a frame
            with self._lock:
                if self._end():
                    raise BackendError(_USER_STOPPED) from exc
            raise BackendError("the shared screen could not be captured. Try again.") from exc

    def cursor(self) -> tuple[float, float]:
        pointer = self._pointer
        if pointer is None:
            raise BackendError(_STOPPED)
        return pointer

    def _stream_at(self, x: float, y: float) -> _portal.Stream:
        for display, stream in self._displays:
            if display.contains(x, y):
                return stream
        return self._displays[0][1]

    def _motion(self, x: float, y: float) -> None:
        self._live()
        stream = self._stream_at(x, y)
        x = min(max(x, stream.x), stream.x + stream.width - 1)
        y = min(max(y, stream.y), stream.y + stream.height - 1)
        if not self._primed:
            # Mutter drops a session's first absolute motion, which would send the click elsewhere.
            self._send(lambda session: session.pointer(stream, x, y))
            time.sleep(0.05)
            self._primed = True
        self._send(lambda session: session.pointer(stream, x, y))
        self._pointer = (x, y)

    def _button(self, button: str, down: bool) -> None:
        self._send(lambda session: session.button(_BUTTONS[button], down))

    def move(self, x: float, y: float) -> None:
        with self._lock:
            self._motion(x, y)

    def click(self, x: float, y: float, button: str, count: int, mods: tuple[str, ...]) -> None:
        with self._lock:
            self._motion(x, y)
            time.sleep(0.03)
            self._modifiers(mods, True)
            try:
                for _ in range(count):
                    self._button(button, True)
                    self._button(button, False)
                    time.sleep(0.03)
            finally:
                self._modifiers(mods, False)

    def drag(self, points: list[tuple[float, float]]) -> None:
        with self._lock:
            self._motion(*points[0])
            time.sleep(0.05)
            self._button("left", True)
            try:
                time.sleep(0.08)
                for x, y in points[1:]:
                    self._motion(x, y)
                    time.sleep(0.012)
                time.sleep(0.08)
            finally:
                self._button("left", False)

    def scroll(self, dx: int, dy: int) -> None:
        with self._lock:
            for amount, axis in ((dy, _VERTICAL), (dx, _HORIZONTAL)):
                if amount:
                    self._send(lambda session, axis = axis, amount = amount: session.axis(axis, amount))

    def _keysym_for(self, name: str) -> int:
        if name in _KEYSYMS:
            return _KEYSYMS[name]
        if name[:1] == "f" and name[1:].isdigit():
            return _F1 + int(name[1:]) - 1
        if len(name) == 1 and 0x20 < ord(name) < 0x7F:
            return ord(name)
        raise BackendError(f"the {name} key isn't on this keyboard layout.")

    def _key_event(self, keysym: int, down: bool) -> None:
        self._send(lambda session: session.keysym(keysym, down))

    def _modifiers(self, mods: tuple[str, ...], down: bool) -> None:
        for mod in mods if down else reversed(mods):
            self._key_event(self._keysym_for(mod), down)

    def key(self, mods: tuple[str, ...], key: Optional[str]) -> None:
        with self._lock:
            keysym = self._keysym_for(key) if key is not None else None
            self._modifiers(mods, True)
            try:
                if keysym is not None:
                    self._key_event(keysym, True)
                    self._key_event(keysym, False)
            finally:
                self._modifiers(mods, False)
        time.sleep(0.01)

    def type_text(self, text: str) -> None:
        with self._lock:
            for char in text.replace("\r\n", "\n"):
                point = ord(char)
                if char in ("\n", "\r"):
                    keysym = _KEYSYM_RETURN
                elif char == "\t":
                    keysym = _KEYSYM_TAB
                elif 0x20 <= point <= 0x7E or 0xA0 <= point <= 0xFF:
                    keysym = point
                else:
                    keysym = _UNICODE_KEYSYM | point
                self._key_event(keysym, True)
                self._key_event(keysym, False)
                time.sleep(0.004)

    def front_window(self) -> FrontWindow:
        return FrontWindow()

    def window_at(self, x: float, y: float) -> Optional[FrontWindow]:
        return None

    def window_stack(self) -> None:
        return None
