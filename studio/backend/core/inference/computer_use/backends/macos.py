# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""macOS: Quartz events through ctypes and the system's own screencapture, so nothing is added
to the install. Needs the Screen Recording and Accessibility grants of whatever launched Unsloth
(the desktop app, or the terminal)."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Optional

from .base import Backend, BackendError, Display, FrontWindow

_EVENT_TAP_HID = 0
_MOUSE_MOVED = 5
_MOUSE_DOWN = {"left": 1, "right": 3, "middle": 25}
_MOUSE_UP = {"left": 2, "right": 4, "middle": 26}
_MOUSE_DRAGGED = {"left": 6, "right": 7, "middle": 27}
_MOUSE_BUTTON = {"left": 0, "right": 1, "middle": 2}
_FIELD_CLICK_STATE = 1
_SCROLL_UNIT_LINE = 1
_LINES_PER_NOTCH = 3
_MAX_SCROLL_LINES = 10
_FLAGS = {"shift": 0x20000, "ctrl": 0x40000, "alt": 0x80000, "cmd": 0x100000}
_MODIFIER_KEYS = {"cmd": 55, "shift": 56, "alt": 58, "ctrl": 59}

# Virtual key codes of the ANSI layout (HIToolbox Events.h): physical keys, as shortcuts are.
_KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11,
    "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21,
    "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31,
    "u": 32, "[": 33, "i": 34, "p": 35, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42,
    ",": 43, "/": 44, "n": 45, "m": 46, ".": 47, "`": 50,
    "enter": 36, "tab": 48, "space": 49, "backspace": 51, "escape": 53, "capslock": 57,
    "insert": 114, "home": 115, "pageup": 116, "delete": 117, "end": 119, "pagedown": 121,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    "f9": 101, "f10": 109, "f11": 103, "f12": 111, "f13": 105, "f14": 107, "f15": 113,
    "f16": 106, "f17": 64, "f18": 79, "f19": 80, "f20": 90,
}  # fmt: skip

_WINDOW_LIST_ON_SCREEN = 1
_WINDOW_LIST_NO_DESKTOP = 16
_CF_NUMBER_SINT64 = 4
_CF_UTF8 = 0x08000100
# Events posted back to back can arrive out of order.
_EVENT_GAP_S = 0.004
_UNICODE_CHUNK = 20


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class _Size(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class _Rect(ctypes.Structure):
    _fields_ = [("origin", _Point), ("size", _Size)]


def _load() -> tuple[Any, Any]:
    # ApplicationServices re-exports CoreGraphics and holds AXIsProcessTrusted.
    lib = ctypes.CDLL(ctypes.util.find_library("ApplicationServices"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    ref, u32, i32, u64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint64

    def sig(library, name, restype, *argtypes):
        function = getattr(library, name)
        function.restype = restype
        function.argtypes = list(argtypes)

    sig(lib, "CGMainDisplayID", u32)
    sig(lib, "CGGetActiveDisplayList", i32, u32, ctypes.POINTER(u32), ctypes.POINTER(u32))
    sig(lib, "CGDisplayBounds", _Rect, u32)
    sig(lib, "CGEventCreate", ref, ref)
    sig(lib, "CGEventGetLocation", _Point, ref)
    sig(lib, "CGEventCreateMouseEvent", ref, ref, u32, _Point, u32)
    sig(lib, "CGEventCreateKeyboardEvent", ref, ref, ctypes.c_uint16, ctypes.c_bool)
    # The variadic CGEventCreateScrollWheelEvent can't be called through ctypes on arm64.
    sig(lib, "CGEventCreateScrollWheelEvent2", ref, ref, u32, u32, i32, i32, i32)
    sig(lib, "CGEventSetIntegerValueField", None, ref, u32, ctypes.c_int64)
    sig(lib, "CGEventSetFlags", None, ref, u64)
    sig(lib, "CGEventKeyboardSetUnicodeString", None, ref, ctypes.c_ulong, ctypes.c_void_p)
    sig(lib, "CGEventPost", None, u32, ref)
    sig(lib, "CGWindowListCopyWindowInfo", ref, u32, u32)
    sig(lib, "CGRectMakeWithDictionaryRepresentation", ctypes.c_bool, ref, ctypes.POINTER(_Rect))
    sig(lib, "AXIsProcessTrusted", ctypes.c_bool)
    for name in (
        "CGPreflightScreenCaptureAccess",
        "CGRequestScreenCaptureAccess",
        "CGPreflightPostEventAccess",
        "CGRequestPostEventAccess",
    ):
        if hasattr(lib, name):
            sig(lib, name, ctypes.c_bool)
    sig(cf, "CFRelease", None, ref)
    sig(cf, "CFArrayGetCount", ctypes.c_long, ref)
    sig(cf, "CFArrayGetValueAtIndex", ref, ref, ctypes.c_long)
    sig(cf, "CFDictionaryGetValue", ref, ref, ref)
    sig(cf, "CFNumberGetValue", ctypes.c_bool, ref, ctypes.c_int, ctypes.c_void_p)
    sig(cf, "CFStringGetCString", ctypes.c_bool, ref, ctypes.c_char_p, ctypes.c_long, u32)
    return lib, cf


class MacBackend(Backend):
    name = "macos"
    shortcut_modifier = "cmd"
    os_name = "macOS"
    # screencapture -C draws the pointer into the capture.
    captures_pointer = True

    def __init__(self) -> None:
        self._lib, self._cf = _load()

    def _preflight(self, name: str) -> Optional[bool]:
        function = getattr(self._lib, name, None)
        return bool(function()) if function is not None else None

    def permissions(self) -> dict[str, Optional[bool]]:
        screen = self._preflight("CGPreflightScreenCaptureAccess")
        trusted = bool(self._lib.AXIsProcessTrusted())
        return {
            # Before 10.15 there is no screen grant to hold.
            "screen": True if screen is None else screen,
            "input": trusted or bool(self._preflight("CGPreflightPostEventAccess")),
        }

    def request_permission(self, kind: str) -> None:
        pane = "Privacy_ScreenCapture" if kind == "screen" else "Privacy_Accessibility"
        request = "CGRequestScreenCaptureAccess" if kind == "screen" else "CGRequestPostEventAccess"
        if hasattr(self._lib, request):
            getattr(self._lib, request)()
        subprocess.run(
            ["/usr/bin/open", f"x-apple.systempreferences:com.apple.preference.security?{pane}"],
            check = False,
            timeout = 10,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        )

    def _need(self, kind: str) -> None:
        if self.permissions()[kind]:
            return
        what, pane = (
            ("see the screen", "Screen & System Audio Recording")
            if kind == "screen"
            else ("use the mouse and keyboard", "Accessibility")
        )
        raise BackendError(
            f"macOS has not let Unsloth {what}. Ask the user to allow it in System Settings > "
            f"Privacy & Security > {pane} (Unsloth's Settings > Computer use has a button), then "
            "restart Unsloth."
        )

    def displays(self) -> list[Display]:
        ids = (ctypes.c_uint32 * 16)()
        count = ctypes.c_uint32(0)
        if self._lib.CGGetActiveDisplayList(16, ids, ctypes.byref(count)) != 0 or count.value == 0:
            raise BackendError("no display was found, so there is no screen to control.")
        main = self._lib.CGMainDisplayID()
        # The main display first: screencapture's -D numbers them that way.
        ordered = sorted(ids[: count.value], key = lambda display: display != main)
        found = []
        for index, display in enumerate(ordered, start = 1):
            rect = self._lib.CGDisplayBounds(display)
            found.append(
                Display(
                    index,
                    rect.origin.x,
                    rect.origin.y,
                    rect.size.width,
                    rect.size.height,
                    display == main,
                )
            )
        return found

    def capture(self, display: Display) -> Any:
        from PIL import Image
        self._need("screen")
        with tempfile.TemporaryDirectory(prefix = "unsloth-screen-") as folder:
            path = os.path.join(folder, "screen.png")
            done = subprocess.run(
                ["/usr/sbin/screencapture", "-x", "-C", f"-D{display.index}", "-t", "png", path],
                check = False,
                timeout = 20,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            )
            if done.returncode != 0 or not os.path.exists(path):
                raise BackendError("the screen could not be captured.")
            with Image.open(path) as image:
                image.load()
                return image.convert("RGB")

    def cursor(self) -> tuple[float, float]:
        event = self._lib.CGEventCreate(None)
        try:
            point = self._lib.CGEventGetLocation(event)
            return point.x, point.y
        finally:
            self._cf.CFRelease(event)

    def _post(self, event: Any) -> None:
        if not event:
            raise BackendError("macOS refused to create an input event.")
        try:
            self._lib.CGEventPost(_EVENT_TAP_HID, event)
        finally:
            self._cf.CFRelease(event)
        time.sleep(_EVENT_GAP_S)

    def _mouse(
        self,
        kind: int,
        x: float,
        y: float,
        button: str = "left",
        *,
        clicks: int = 0,
        flags: int = 0,
    ) -> None:
        event = self._lib.CGEventCreateMouseEvent(None, kind, _Point(x, y), _MOUSE_BUTTON[button])
        if event and clicks:
            self._lib.CGEventSetIntegerValueField(event, _FIELD_CLICK_STATE, clicks)
        if event and flags:
            self._lib.CGEventSetFlags(event, flags)
        self._post(event)

    def move(self, x: float, y: float) -> None:
        self._need("input")
        self._mouse(_MOUSE_MOVED, x, y)

    def click(self, x: float, y: float, button: str, count: int, mods: tuple[str, ...]) -> None:
        self._need("input")
        flags = 0
        for mod in mods:
            flags |= _FLAGS[mod]
        self._mouse(_MOUSE_MOVED, x, y)
        time.sleep(0.03)
        for nth in range(1, count + 1):
            self._mouse(_MOUSE_DOWN[button], x, y, button, clicks = nth, flags = flags)
            self._mouse(_MOUSE_UP[button], x, y, button, clicks = nth, flags = flags)
            time.sleep(0.03)

    def drag(self, points: list[tuple[float, float]]) -> None:
        self._need("input")
        (x, y), rest = points[0], points[1:]
        self._mouse(_MOUSE_MOVED, x, y)
        time.sleep(0.05)
        self._mouse(_MOUSE_DOWN["left"], x, y, clicks = 1)
        time.sleep(0.08)
        for x, y in rest:
            self._mouse(_MOUSE_DRAGGED["left"], x, y)
            time.sleep(0.012)
        time.sleep(0.08)
        self._mouse(_MOUSE_UP["left"], x, y, clicks = 1)

    def scroll(self, dx: int, dy: int) -> None:
        self._need("input")
        # Positive wheel values scroll up and left; the tool counts down and right as positive.
        vertical, horizontal = -dy * _LINES_PER_NOTCH, -dx * _LINES_PER_NOTCH
        while vertical or horizontal:
            # Apple documents deltas of about -10 to 10 per event; larger ones scroll erratically.
            step_v = max(-_MAX_SCROLL_LINES, min(_MAX_SCROLL_LINES, vertical))
            step_h = max(-_MAX_SCROLL_LINES, min(_MAX_SCROLL_LINES, horizontal))
            vertical -= step_v
            horizontal -= step_h
            self._post(
                self._lib.CGEventCreateScrollWheelEvent2(
                    None, _SCROLL_UNIT_LINE, 2, step_v, step_h, 0
                )
            )
            time.sleep(0.01)

    def _key_event(self, code: int, down: bool, flags: Optional[int]) -> None:
        event = self._lib.CGEventCreateKeyboardEvent(None, code, down)
        if event and flags is not None:
            self._lib.CGEventSetFlags(event, flags)
        self._post(event)

    def key(self, mods: tuple[str, ...], key: Optional[str]) -> None:
        self._need("input")
        if key is not None and key not in _KEYCODES:
            raise BackendError(f"the {key} key isn't available on macOS.")
        flags = 0
        for mod in mods:
            flags |= _FLAGS[mod]
            self._key_event(_MODIFIER_KEYS[mod], True, flags)
        if key is not None:
            # Explicit flags: a created event otherwise carries whatever the user is holding.
            self._key_event(_KEYCODES[key], True, flags)
            self._key_event(_KEYCODES[key], False, flags)
        for mod in reversed(mods):
            flags &= ~_FLAGS[mod]
            self._key_event(_MODIFIER_KEYS[mod], False, flags)

    def type_text(self, text: str) -> None:
        self._need("input")
        for part in re.split(r"(\r\n|[\n\r\t])", text):
            if part == "":
                continue
            if part in ("\n", "\r", "\r\n"):
                self.key((), "enter")
                continue
            if part == "\t":
                self.key((), "tab")
                continue
            units = part.encode("utf-16-le")
            step = _UNICODE_CHUNK * 2
            start = 0
            while start < len(units):
                end = min(len(units), start + step)
                # Never split a surrogate pair across two events.
                if end < len(units) and 0xD8 <= units[end - 1] <= 0xDB:
                    end -= 2
                chunk = units[start:end]
                start = end
                buffer = ctypes.create_string_buffer(chunk, len(chunk))
                for down in (True, False):
                    event = self._lib.CGEventCreateKeyboardEvent(None, 0, down)
                    if event:
                        self._lib.CGEventSetFlags(event, 0)
                        self._lib.CGEventKeyboardSetUnicodeString(event, len(chunk) // 2, buffer)
                    self._post(event)

    def _front_app(self) -> tuple[str, str, int]:
        """Name, bundle id and pid of the front application, asked of LaunchServices: NSWorkspace
        goes stale in a process with no run loop."""
        try:
            asn = subprocess.run(
                ["/usr/bin/lsappinfo", "front"], capture_output = True, text = True, timeout = 5
            ).stdout.strip()
            if not asn:
                return "", "", 0
            info = subprocess.run(
                [
                    "/usr/bin/lsappinfo",
                    "info",
                    "-only",
                    "bundleID",
                    "-only",
                    "name",
                    "-only",
                    "pid",
                    asn,
                ],
                capture_output = True,
                text = True,
                timeout = 5,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return "", "", 0
        fields = dict(re.findall(r'"([A-Za-z]+)"\s*=\s*"?([^"\n]*)"?', info))
        pid = fields.get("pid", "0")
        return (
            fields.get("LSDisplayName", ""),
            fields.get("CFBundleIdentifier", ""),
            int(pid) if pid.isdigit() else 0,
        )

    def _cf_key(self, name: str) -> Any:
        return ctypes.c_void_p.in_dll(self._lib, name)

    def _cf_text(self, value: Any) -> str:
        if not value:
            return ""
        buffer = ctypes.create_string_buffer(1024)
        if not self._cf.CFStringGetCString(value, buffer, len(buffer), _CF_UTF8):
            return ""
        return buffer.value.decode("utf-8", "replace")

    def _cf_int(self, value: Any) -> int:
        number = ctypes.c_int64(0)
        if not value or not self._cf.CFNumberGetValue(
            value, _CF_NUMBER_SINT64, ctypes.byref(number)
        ):
            return -1
        return number.value

    def _windows(self) -> list[tuple[int, FrontWindow]]:
        """Every window on screen, front to back, with its layer (0 for ordinary windows)."""
        found: list[tuple[int, FrontWindow]] = []
        windows = self._lib.CGWindowListCopyWindowInfo(
            _WINDOW_LIST_ON_SCREEN | _WINDOW_LIST_NO_DESKTOP, 0
        )
        if not windows:
            return found
        try:
            for index in range(self._cf.CFArrayGetCount(windows)):
                info = self._cf.CFArrayGetValueAtIndex(windows, index)
                get = lambda key: self._cf.CFDictionaryGetValue(info, self._cf_key(key))  # noqa: E731
                bounds = None
                rect = _Rect()
                box = get("kCGWindowBounds")
                if box and self._lib.CGRectMakeWithDictionaryRepresentation(
                    box, ctypes.byref(rect)
                ):
                    bounds = (rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
                found.append(
                    (
                        self._cf_int(get("kCGWindowLayer")),
                        FrontWindow(
                            app = self._cf_text(get("kCGWindowOwnerName")),
                            # Empty without the Screen Recording grant.
                            title = self._cf_text(get("kCGWindowName")),
                            pid = max(0, self._cf_int(get("kCGWindowOwnerPID"))),
                            bounds = bounds,
                        ),
                    )
                )
        finally:
            self._cf.CFRelease(windows)
        return found

    def front_window(self) -> FrontWindow:
        app, bundle, pid = self._front_app()
        # The first ordinary window of the front app is the one in front.
        for layer, window in self._windows():
            if layer == 0 and (not pid or window.pid == pid):
                return FrontWindow(
                    app = app or window.app,
                    app_id = bundle,
                    title = window.title,
                    pid = pid or window.pid,
                    bounds = window.bounds,
                )
        return FrontWindow(app = app, app_id = bundle, pid = pid)

    def window_at(self, x: float, y: float) -> Optional[FrontWindow]:
        for layer, window in self._windows():
            # Ordinary windows only: the menu bar and the Dock sit above them and are no app's page.
            if layer == 0 and window.bounds is not None and window.contains(x, y):
                return window
        return None

    def open_app(self, name: str) -> str:
        # A bare name is looked up among installed applications; a path would run any bundle.
        if name.startswith(("-", ".")) or "/" in name or ":" in name:
            raise BackendError("give the application's name, such as Safari, not a path.")
        done = subprocess.run(
            ["/usr/bin/open", "-a", name],
            check = False,
            timeout = 20,
            capture_output = True,
            text = True,
        )
        if done.returncode != 0:
            raise BackendError(f"no application named {name} was found on this Mac.")
        return name
