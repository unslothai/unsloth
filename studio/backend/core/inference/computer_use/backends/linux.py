# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linux on X11: the XTEST extension through ctypes and Pillow's screen grab. Wayland gives an
application no way to see or drive other windows, so a Wayland session is reported unsupported."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Optional

from .base import Backend, BackendError, Display, FrontWindow

_BUTTONS = {"left": 1, "middle": 2, "right": 3}
_SCROLL_UP, _SCROLL_DOWN, _SCROLL_LEFT, _SCROLL_RIGHT = 4, 5, 6, 7
_KEYSYM_NAMES = {
    "ctrl": "Control_L", "alt": "Alt_L", "shift": "Shift_L", "cmd": "Super_L",
    "enter": "Return", "tab": "Tab", "space": "space", "backspace": "BackSpace",
    "delete": "Delete", "escape": "Escape", "up": "Up", "down": "Down", "left": "Left",
    "right": "Right", "home": "Home", "end": "End", "pageup": "Prior", "pagedown": "Next",
    "insert": "Insert", "capslock": "Caps_Lock", "-": "minus", "=": "equal", ",": "comma",
    ".": "period", "/": "slash", ";": "semicolon", "`": "grave", "[": "bracketleft",
    "\\": "backslash", "]": "bracketright", "'": "apostrophe",
}  # fmt: skip
_UNICODE_KEYSYM = 0x01000000
_ANY_PROPERTY_TYPE = 0
# Clients re-read the keyboard map lazily; typing straight after a change lands the old symbol.
_REMAP_SETTLE_S = 0.03
_APP_DIRS = (
    "~/.local/share/applications",
    "/usr/local/share/applications",
    "/usr/share/applications",
    "/var/lib/flatpak/exports/share/applications",
    "~/.local/share/flatpak/exports/share/applications",
    "/var/lib/snapd/desktop/applications",
)


def _session_problem() -> Optional[str]:
    wayland = os.environ.get("WAYLAND_DISPLAY") or (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    )
    if wayland:
        return (
            "This is a Wayland session, where an application can't see or control other windows. "
            "Sign in with an X11 (Xorg) session to use computer use."
        )
    if not os.environ.get("DISPLAY"):
        return "Unsloth has no desktop session here (no DISPLAY), so there is no screen to control."
    return None


_IS_VIEWABLE = 2
_INPUT_OUTPUT = 1


class _WindowAttributes(ctypes.Structure):
    """Xlib's XWindowAttributes."""

    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("border_width", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("visual", ctypes.c_void_p),
        ("root", ctypes.c_ulong),
        ("window_class", ctypes.c_int),
        ("bit_gravity", ctypes.c_int),
        ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int),
        ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong),
        ("save_under", ctypes.c_int),
        ("colormap", ctypes.c_ulong),
        ("map_installed", ctypes.c_int),
        ("map_state", ctypes.c_int),
        ("all_event_masks", ctypes.c_long),
        ("your_event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int),
        ("screen", ctypes.c_void_p),
    ]


class LinuxBackend(Backend):
    name = "linux"
    shortcut_modifier = "ctrl"
    os_name = "Linux"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._x11: Any = None
        self._xtst: Any = None
        self._display: Any = None
        self._problem = _session_problem()
        if self._problem is None:
            self._problem = self._load()

    def _load(self) -> Optional[str]:
        x11_name, xtst_name = ctypes.util.find_library("X11"), ctypes.util.find_library("Xtst")
        if not x11_name or not xtst_name:
            return (
                "The X11 input libraries are missing. Install libx11-6 and libxtst6 "
                "(Debian, Ubuntu) or libX11 and libXtst (Fedora), then restart Unsloth."
            )
        x11, xtst = ctypes.CDLL(x11_name), ctypes.CDLL(xtst_name)
        ref, long, ulong, cint, uint = (
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_uint,
        )
        pointer = ctypes.POINTER

        def sig(library, name, restype, *argtypes):
            function = getattr(library, name)
            function.restype = restype
            function.argtypes = list(argtypes)

        sig(x11, "XOpenDisplay", ref, ctypes.c_char_p)
        sig(x11, "XDefaultScreen", cint, ref)
        sig(x11, "XDefaultRootWindow", ulong, ref)
        sig(x11, "XDisplayWidth", cint, ref, cint)
        sig(x11, "XDisplayHeight", cint, ref, cint)
        sig(x11, "XFlush", cint, ref)
        sig(x11, "XSync", cint, ref, cint)
        sig(x11, "XFree", cint, ref)
        sig(x11, "XStringToKeysym", ulong, ctypes.c_char_p)
        sig(x11, "XKeysymToKeycode", ctypes.c_ubyte, ref, ulong)
        sig(x11, "XDisplayKeycodes", cint, ref, pointer(cint), pointer(cint))
        sig(x11, "XGetKeyboardMapping", pointer(ulong), ref, ctypes.c_ubyte, cint, pointer(cint))
        sig(x11, "XChangeKeyboardMapping", cint, ref, cint, cint, pointer(ulong), cint)
        sig(x11, "XInternAtom", ulong, ref, ctypes.c_char_p, cint)
        sig(
            x11, "XQueryPointer", cint, ref, ulong, pointer(ulong), pointer(ulong),
            pointer(cint), pointer(cint), pointer(cint), pointer(cint), pointer(uint),
        )  # fmt: skip
        sig(
            x11, "XGetWindowProperty", cint, ref, ulong, ulong, long, long, cint, ulong,
            pointer(ulong), pointer(cint), pointer(ulong), pointer(ulong), pointer(ref),
        )  # fmt: skip
        sig(
            x11, "XGetGeometry", cint, ref, ulong, pointer(ulong), pointer(cint), pointer(cint),
            pointer(uint), pointer(uint), pointer(uint), pointer(uint),
        )  # fmt: skip
        sig(
            x11, "XTranslateCoordinates", cint, ref, ulong, ulong, cint, cint, pointer(cint),
            pointer(cint), pointer(ulong),
        )  # fmt: skip
        sig(
            x11, "XQueryTree", cint, ref, ulong, pointer(ulong), pointer(ulong),
            pointer(pointer(ulong)), pointer(uint),
        )  # fmt: skip
        sig(x11, "XGetWindowAttributes", cint, ref, ulong, pointer(_WindowAttributes))
        sig(xtst, "XTestFakeMotionEvent", cint, ref, cint, cint, cint, ulong)
        sig(xtst, "XTestFakeButtonEvent", cint, ref, uint, cint, ulong)
        sig(xtst, "XTestFakeKeyEvent", cint, ref, uint, cint, ulong)
        sig(
            xtst,
            "XTestQueryExtension",
            cint,
            ref,
            pointer(cint),
            pointer(cint),
            pointer(cint),
            pointer(cint),
        )
        # Xlib's default handler ends the process on any protocol error, such as a window that
        # closed between being found and being read.
        self._on_error = ctypes.CFUNCTYPE(cint, ref, ref)(lambda _display, _event: 0)
        x11.XSetErrorHandler.argtypes = [ctypes.c_void_p]
        x11.XSetErrorHandler(ctypes.cast(self._on_error, ctypes.c_void_p))
        display = x11.XOpenDisplay(None)
        if not display:
            return "The X display could not be opened, so there is no screen to control."
        unused = ctypes.c_int(0)
        if not xtst.XTestQueryExtension(
            display,
            ctypes.byref(unused),
            ctypes.byref(unused),
            ctypes.byref(unused),
            ctypes.byref(unused),
        ):
            return "This X server has no XTEST extension, which computer use needs for input."
        if hasattr(x11, "XSetIOErrorExitHandler"):
            # A lost X server would otherwise exit Unsloth from inside Xlib.
            self._on_lost = ctypes.CFUNCTYPE(None, ref, ref)(self._lost)
            x11.XSetIOErrorExitHandler.argtypes = [ref, ctypes.c_void_p, ref]
            x11.XSetIOErrorExitHandler(display, ctypes.cast(self._on_lost, ctypes.c_void_p), None)
        self._x11, self._xtst, self._display = x11, xtst, display
        return None

    def _lost(self, _display: Any, _data: Any) -> None:
        self._problem = "The X display went away. Restart Unsloth to use computer use again."

    def unavailable_reason(self) -> Optional[str]:
        return self._problem

    def _ready(self) -> None:
        if self._problem:
            raise BackendError(self._problem)

    def _flush(self) -> None:
        self._x11.XSync(self._display, 0)

    def displays(self) -> list[Display]:
        self._ready()
        with self._lock:
            screen = self._x11.XDefaultScreen(self._display)
            width = self._x11.XDisplayWidth(self._display, screen)
            height = self._x11.XDisplayHeight(self._display, screen)
        # The X screen spans every monitor, so it is the one display.
        return [Display(1, 0, 0, width, height, True)]

    def capture(self, display: Display) -> Any:
        from PIL import ImageGrab
        self._ready()
        try:
            return ImageGrab.grab(xdisplay = os.environ.get("DISPLAY")).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - Pillow raises OSError or its own errors here
            raise BackendError("the screen could not be captured on this X display.") from exc

    def cursor(self) -> tuple[float, float]:
        self._ready()
        root_x, root_y, win_x, win_y = (ctypes.c_int(0) for _ in range(4))
        root, child, mask = ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_uint(0)
        with self._lock:
            self._x11.XQueryPointer(
                self._display,
                self._x11.XDefaultRootWindow(self._display),
                ctypes.byref(root),
                ctypes.byref(child),
                ctypes.byref(root_x),
                ctypes.byref(root_y),
                ctypes.byref(win_x),
                ctypes.byref(win_y),
                ctypes.byref(mask),
            )
        return root_x.value, root_y.value

    def _motion(self, x: float, y: float) -> None:
        self._xtst.XTestFakeMotionEvent(self._display, -1, int(round(x)), int(round(y)), 0)
        self._flush()

    def _button(self, button: int, down: bool) -> None:
        self._xtst.XTestFakeButtonEvent(self._display, button, 1 if down else 0, 0)
        self._flush()

    def move(self, x: float, y: float) -> None:
        self._ready()
        with self._lock:
            self._motion(x, y)

    def click(self, x: float, y: float, button: str, count: int, mods: tuple[str, ...]) -> None:
        self._ready()
        with self._lock:
            self._motion(x, y)
            time.sleep(0.03)
            self._modifiers(mods, True)
            try:
                for _ in range(count):
                    self._button(_BUTTONS[button], True)
                    self._button(_BUTTONS[button], False)
                    time.sleep(0.03)
            finally:
                self._modifiers(mods, False)

    def drag(self, points: list[tuple[float, float]]) -> None:
        self._ready()
        with self._lock:
            self._motion(*points[0])
            time.sleep(0.05)
            self._button(_BUTTONS["left"], True)
            try:
                time.sleep(0.08)
                for x, y in points[1:]:
                    self._motion(x, y)
                    time.sleep(0.012)
                time.sleep(0.08)
            finally:
                self._button(_BUTTONS["left"], False)

    def scroll(self, dx: int, dy: int) -> None:
        self._ready()
        with self._lock:
            for amount, back, forward in (
                (dy, _SCROLL_UP, _SCROLL_DOWN),
                (dx, _SCROLL_LEFT, _SCROLL_RIGHT),
            ):
                for _ in range(abs(amount)):
                    button = forward if amount > 0 else back
                    self._button(button, True)
                    self._button(button, False)

    def _keycode(self, name: str) -> int:
        symbol = _KEYSYM_NAMES.get(name) or (
            name.upper() if name[:1] == "f" and name[1:].isdigit() else name
        )
        keysym = self._x11.XStringToKeysym(symbol.encode("ascii", "ignore"))
        code = self._x11.XKeysymToKeycode(self._display, keysym) if keysym else 0
        if not code:
            raise BackendError(f"the {name} key isn't on this keyboard layout.")
        return code

    def _key_event(self, code: int, down: bool) -> None:
        self._xtst.XTestFakeKeyEvent(self._display, code, 1 if down else 0, 0)
        self._flush()

    def _modifiers(self, mods: tuple[str, ...], down: bool) -> None:
        for mod in mods if down else reversed(mods):
            self._key_event(self._keycode(mod), down)

    def key(self, mods: tuple[str, ...], key: Optional[str]) -> None:
        self._ready()
        with self._lock:
            code = self._keycode(key) if key is not None else None
            self._modifiers(mods, True)
            try:
                if code is not None:
                    self._key_event(code, True)
                    self._key_event(code, False)
            finally:
                self._modifiers(mods, False)
        time.sleep(0.01)

    def _keyboard_map(self) -> tuple[dict[int, tuple[int, bool]], Optional[int]]:
        """Keysym to (keycode, needs Shift) for the layout's first two levels, and a keycode with
        no symbol to borrow for everything else."""
        low, high, per = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
        self._x11.XDisplayKeycodes(self._display, ctypes.byref(low), ctypes.byref(high))
        count = high.value - low.value + 1
        symbols = self._x11.XGetKeyboardMapping(self._display, low.value, count, ctypes.byref(per))
        table: dict[int, tuple[int, bool]] = {}
        spare = None
        if not symbols:
            return table, spare
        try:
            width = per.value
            for index in range(count):
                row = [symbols[index * width + level] for level in range(width)]
                code = low.value + index
                if not any(row):
                    spare = code
                    continue
                for level, keysym in enumerate(row[:2]):
                    if keysym and keysym not in table:
                        table[keysym] = (code, level == 1)
        finally:
            self._x11.XFree(symbols)
        return table, spare

    def _remap(self, code: int, keysym: int) -> None:
        self._x11.XChangeKeyboardMapping(self._display, code, 1, (ctypes.c_ulong * 1)(keysym), 1)
        self._flush()
        time.sleep(_REMAP_SETTLE_S)

    def type_text(self, text: str) -> None:
        self._ready()
        with self._lock:
            table, spare = self._keyboard_map()
            shift = self._keycode("shift")
            borrowed = None
            try:
                for char in text.replace("\r\n", "\n"):
                    if char in ("\n", "\r"):
                        self.key((), "enter")
                        continue
                    if char == "\t":
                        self.key((), "tab")
                        continue
                    point = ord(char)
                    # Latin-1 keysyms are the code point; the rest live in the Unicode range.
                    keysym = (
                        point
                        if 0x20 <= point <= 0x7E or 0xA0 <= point <= 0xFF
                        else (_UNICODE_KEYSYM | point)
                    )
                    known = table.get(keysym)
                    if known is None:
                        if spare is None:
                            raise BackendError(f"this keyboard layout can't type {char!r}.")
                        if borrowed != keysym:
                            self._remap(spare, keysym)
                            borrowed = keysym
                        known = (spare, False)
                    code, shifted = known
                    if shifted:
                        self._key_event(shift, True)
                    self._key_event(code, True)
                    self._key_event(code, False)
                    if shifted:
                        self._key_event(shift, False)
                    time.sleep(0.004)
            finally:
                if borrowed is not None and spare is not None:
                    self._remap(spare, 0)

    def _property(self, window: int, name: bytes) -> tuple[int, bytes]:
        atom = self._x11.XInternAtom(self._display, name, 1)
        if not atom:
            return 0, b""
        kind, form = ctypes.c_ulong(0), ctypes.c_int(0)
        items, left = ctypes.c_ulong(0), ctypes.c_ulong(0)
        data = ctypes.c_void_p(None)
        status = self._x11.XGetWindowProperty(
            self._display, window, atom, 0, 1024, 0, _ANY_PROPERTY_TYPE,
            ctypes.byref(kind), ctypes.byref(form), ctypes.byref(items), ctypes.byref(left),
            ctypes.byref(data),
        )  # fmt: skip
        if status != 0 or not data.value:
            return 0, b""
        try:
            # Format 32 arrives as C longs, whatever their size.
            size = {8: 1, 16: 2, 32: ctypes.sizeof(ctypes.c_long)}.get(form.value, 0)
            return form.value, ctypes.string_at(data.value, items.value * size)
        finally:
            self._x11.XFree(data)

    def _long_property(self, window: int, name: bytes) -> int:
        form, raw = self._property(window, name)
        size = ctypes.sizeof(ctypes.c_long)
        if form != 32 or len(raw) < size:
            return 0
        return int.from_bytes(raw[:size], sys.byteorder)

    def front_window(self) -> FrontWindow:
        if self._problem:
            return FrontWindow()
        with self._lock:
            root = self._x11.XDefaultRootWindow(self._display)
            window = self._long_property(root, b"_NET_ACTIVE_WINDOW")
            if not window:
                return FrontWindow()
            return self._describe(window)

    def _children(self, window: int) -> list[int]:
        """A window's children, lowest first."""
        root, parent = ctypes.c_ulong(0), ctypes.c_ulong(0)
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint(0)
        if not self._x11.XQueryTree(
            self._display, window, ctypes.byref(root), ctypes.byref(parent),
            ctypes.byref(children), ctypes.byref(count),
        ):  # fmt: skip
            return []
        try:
            return [children[index] for index in range(count.value)]
        finally:
            if children:
                self._x11.XFree(children)

    def _client(
        self,
        window: int,
        depth: int = 0,
    ) -> int:
        """The application's own window inside a window manager's frame: the first one down
        the tree that names its class. 0 when none does."""
        _form, klass = self._property(window, b"WM_CLASS")
        if klass:
            return window
        if depth < 4:
            for child in reversed(self._children(window)):
                found = self._client(child, depth + 1)
                if found:
                    return found
        return 0

    def window_at(self, x: float, y: float) -> Optional[FrontWindow]:
        """The topmost window under this point, in front or behind. A menu or tooltip names no
        application, and is charged to the one in front, which raised it."""
        if self._problem:
            return None
        with self._lock:
            root = self._x11.XDefaultRootWindow(self._display)
            for window in reversed(self._children(root)):
                attributes = _WindowAttributes()
                if not self._x11.XGetWindowAttributes(
                    self._display, window, ctypes.byref(attributes)
                ):
                    continue
                if attributes.map_state != _IS_VIEWABLE or attributes.window_class != _INPUT_OUTPUT:
                    continue
                inside_x = attributes.x <= x < attributes.x + attributes.width
                inside_y = attributes.y <= y < attributes.y + attributes.height
                if not (inside_x and inside_y):
                    continue
                client = self._client(window)
                if client:
                    return self._describe(client)
                front = self._long_property(root, b"_NET_ACTIVE_WINDOW")
                return self._describe(front) if front else None
        return None

    def _describe(self, window: int) -> FrontWindow:
        """Call with the lock held."""
        root = self._x11.XDefaultRootWindow(self._display)
        _form, title = self._property(window, b"_NET_WM_NAME")
        if not title:
            _form, title = self._property(window, b"WM_NAME")
        _form, klass = self._property(window, b"WM_CLASS")
        names = [part.decode("utf-8", "replace") for part in klass.split(b"\x00") if part]
        pid = self._long_property(window, b"_NET_WM_PID")
        bounds = None
        geometry_root = ctypes.c_ulong(0)
        x, y = ctypes.c_int(0), ctypes.c_int(0)
        width, height, border, depth = (ctypes.c_uint(0) for _ in range(4))
        if self._x11.XGetGeometry(
            self._display, window, ctypes.byref(geometry_root), ctypes.byref(x), ctypes.byref(y),
            ctypes.byref(width), ctypes.byref(height), ctypes.byref(border), ctypes.byref(depth),
        ):  # fmt: skip
            root_x, root_y, child = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_ulong(0)
            self._x11.XTranslateCoordinates(
                self._display, window, root, 0, 0, ctypes.byref(root_x), ctypes.byref(root_y),
                ctypes.byref(child),
            )  # fmt: skip
            bounds = (root_x.value, root_y.value, width.value, height.value)
        return FrontWindow(
            app = names[-1] if names else "",
            app_id = (names[0] if names else "").lower(),
            title = title.decode("utf-8", "replace"),
            pid = pid,
            bounds = bounds,
        )

    def _desktop_entry(self, name: str) -> Optional[tuple[str, str, str]]:
        """Path, shown name and Exec line of the application whose name matches best."""
        wanted = name.casefold()
        best: Optional[tuple[int, str, str, str]] = None
        for folder in _APP_DIRS:
            folder = os.path.expanduser(folder)
            try:
                entries = sorted(os.listdir(folder))
            except OSError:
                continue
            for entry in entries:
                if not entry.endswith(".desktop"):
                    continue
                path = os.path.join(folder, entry)
                shown = command = ""
                hidden = False
                try:
                    with open(path, encoding = "utf-8", errors = "replace") as handle:
                        section = ""
                        for line in handle:
                            line = line.strip()
                            if line.startswith("["):
                                section = line
                            elif section != "[Desktop Entry]":
                                continue
                            elif line.startswith("Name=") and not shown:
                                shown = line[5:]
                            elif line.startswith("Exec=") and not command:
                                command = line[5:]
                            elif line in ("NoDisplay=true", "Hidden=true"):
                                hidden = True
                except OSError:
                    continue
                if hidden or not shown or not command:
                    continue
                folded = shown.casefold()
                stem = entry[: -len(".desktop")].casefold()
                if folded == wanted or stem == wanted:
                    rank = 0
                elif wanted in folded:
                    rank = 1 + len(folded)
                else:
                    continue
                if best is None or rank < best[0]:
                    best = (rank, path, shown, command)
        return best[1:] if best else None

    def resolve_app(self, name: str) -> str:
        found = None if "/" in name else self._desktop_entry(name)
        return found[1] if found else name

    def open_app(self, name: str) -> str:
        self._ready()
        if "/" in name or name.startswith(("-", ".")):
            raise BackendError("give the application's name as the launcher shows it, not a path.")
        found = self._desktop_entry(name)
        if found is None:
            raise BackendError(f"no application named {name} was found in the launcher.")
        path, shown, command = found
        launcher = shutil.which("gio")
        if launcher:
            argv = [launcher, "launch", path]
        else:
            # Field codes (%f, %U) stand for files to open, and there are none.
            argv = [part for part in shlex.split(command) if not part.startswith("%")]
        try:
            subprocess.Popen(
                argv,
                stdin = subprocess.DEVNULL,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                start_new_session = True,
            )
        except (OSError, ValueError) as exc:
            raise BackendError(f"{shown} could not be started.") from exc
        return shown
