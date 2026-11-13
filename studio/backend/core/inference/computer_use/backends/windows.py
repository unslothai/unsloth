# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows: SendInput through ctypes and Pillow's screen grab. Every call runs per-monitor DPI
aware, so a position is a physical pixel of the capture on any scaling. Windows itself keeps
input away from elevated windows and the secure desktop (UAC, the lock screen)."""

from __future__ import annotations

import contextlib
import ctypes
import os
import subprocess
import time
from ctypes import wintypes
from typing import Any, Optional

from .base import Backend, BackendError, Display, FrontWindow

_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_KEYUP = 0x0002
_UNICODE = 0x0004
_EXTENDED = 0x0001
_MOUSE_DOWN = {"left": 0x0002, "right": 0x0008, "middle": 0x0020}
_MOUSE_UP = {"left": 0x0004, "right": 0x0010, "middle": 0x0040}
_WHEEL = 0x0800
_HWHEEL = 0x1000
_WHEEL_DELTA = 120
_PER_MONITOR_AWARE_V2 = -4
_PER_MONITOR_AWARE = -3
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_MONITORINFOF_PRIMARY = 1
_CREATE_NO_WINDOW = 0x08000000
_GA_ROOT = 2

_VK_MODIFIERS = {"ctrl": 0x11, "alt": 0x12, "shift": 0x10, "cmd": 0x5B}
_VK = {
    "enter": 0x0D, "tab": 0x09, "space": 0x20, "backspace": 0x08, "delete": 0x2E,
    "escape": 0x1B, "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27, "home": 0x24,
    "end": 0x23, "pageup": 0x21, "pagedown": 0x22, "insert": 0x2D, "capslock": 0x14,
    "-": 0xBD, "=": 0xBB, ",": 0xBC, ".": 0xBE, "/": 0xBF, ";": 0xBA, "`": 0xC0, "[": 0xDB,
    "\\": 0xDC, "]": 0xDD, "'": 0xDE,
    **{chr(code).lower(): code for code in range(0x41, 0x5B)},
    **{chr(code): code for code in range(0x30, 0x3A)},
    **{f"f{n}": 0x6F + n for n in range(1, 21)},
}  # fmt: skip
# Without the extended flag these read as numpad keys, and Shift+arrow selects nothing.
_EXTENDED_KEYS = frozenset(
    {"up", "down", "left", "right", "home", "end", "pageup", "pagedown", "insert", "delete", "cmd"}
)

_OPEN_APP_SCRIPT = (
    "$n = $env:UNSLOTH_COMPUTER_USE_APP; "
    "$apps = Get-StartApps; "
    "$a = $apps | Where-Object { $_.Name -eq $n } | Select-Object -First 1; "
    "if (-not $a) { $p = '*' + [WildcardPattern]::Escape($n) + '*'; "
    "$a = $apps | Where-Object { $_.Name -like $p } | Sort-Object { $_.Name.Length } | "
    "Select-Object -First 1 }; "
    "if (-not $a) { exit 3 }; "
    "Start-Process ('shell:AppsFolder\\' + $a.AppID); "
    "Write-Output $a.Name"
)


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput), ("ki", _KeyboardInput)]


class _Input(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _InputUnion)]


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


class WindowsBackend(Backend):
    name = "windows"
    shortcut_modifier = "ctrl"
    os_name = "Windows"

    def __init__(self) -> None:
        self._user32 = ctypes.WinDLL("user32", use_last_error = True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        user32 = self._user32
        user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int]
        user32.SendInput.restype = wintypes.UINT
        user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.WindowFromPoint.argtypes = [wintypes.POINT]
        user32.WindowFromPoint.restype = wintypes.HWND
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MonitorInfo)]
        self._monitor_proc = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HMONITOR,
            wintypes.HDC,
            ctypes.POINTER(wintypes.RECT),
            wintypes.LPARAM,
        )
        user32.EnumDisplayMonitors.argtypes = [
            wintypes.HDC,
            ctypes.c_void_p,
            self._monitor_proc,
            wintypes.LPARAM,
        ]
        if hasattr(user32, "SetThreadDpiAwarenessContext"):
            user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
            user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        kernel32 = self._kernel32
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.ProcessIdToSessionId.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]

    @contextlib.contextmanager
    def _dpi_aware(self):
        """Thread scoped: the process-wide switch can't be set once the process has a window."""
        setter = getattr(self._user32, "SetThreadDpiAwarenessContext", None)
        previous = None
        if setter is not None:
            previous = setter(ctypes.c_void_p(_PER_MONITOR_AWARE_V2)) or setter(
                ctypes.c_void_p(_PER_MONITOR_AWARE)
            )
        try:
            yield
        finally:
            if setter is not None and previous:
                setter(ctypes.c_void_p(previous))

    def unavailable_reason(self) -> Optional[str]:
        session = wintypes.DWORD(0)
        if self._kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session)) and (
            session.value == 0
        ):
            return "Unsloth is running as a Windows service, which has no desktop to control."
        return None

    def displays(self) -> list[Display]:
        found: list[Display] = []

        def visit(monitor, _dc, _rect, _data):
            info = _MonitorInfo()
            info.cbSize = ctypes.sizeof(_MonitorInfo)
            if self._user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                rect = info.rcMonitor
                found.append(
                    Display(
                        0,
                        rect.left,
                        rect.top,
                        rect.right - rect.left,
                        rect.bottom - rect.top,
                        bool(info.dwFlags & _MONITORINFOF_PRIMARY),
                    )
                )
            return True

        with self._dpi_aware():
            self._user32.EnumDisplayMonitors(None, None, self._monitor_proc(visit), 0)
        if not found:
            raise BackendError("no display was found, so there is no screen to control.")
        found.sort(key = lambda display: (not display.primary, display.x, display.y))
        return [
            Display(index, d.x, d.y, d.width, d.height, d.primary)
            for index, d in enumerate(found, start = 1)
        ]

    def capture(self, display: Display) -> Any:
        from PIL import ImageGrab
        box = (
            int(display.x),
            int(display.y),
            int(display.x + display.width),
            int(display.y + display.height),
        )
        try:
            # all_screens: the box is in virtual desktop pixels, so a second monitor works.
            return ImageGrab.grab(bbox = box, all_screens = True).convert("RGB")
        except OSError as exc:
            raise BackendError("the screen could not be captured.") from exc

    def cursor(self) -> tuple[float, float]:
        point = wintypes.POINT()
        with self._dpi_aware():
            if not self._user32.GetCursorPos(ctypes.byref(point)):
                raise BackendError("the pointer position could not be read.")
        return point.x, point.y

    def _send(self, *inputs: _Input) -> None:
        batch = (_Input * len(inputs))(*inputs)
        sent = self._user32.SendInput(len(inputs), batch, ctypes.sizeof(_Input))
        if sent != len(inputs):
            raise BackendError(
                "Windows blocked the input. The window in front may be running as administrator, "
                "which Unsloth can't control."
            )

    def _mouse_input(
        self,
        flags: int,
        data: int = 0,
    ) -> _Input:
        item = _Input(type = _INPUT_MOUSE)
        item.mi = _MouseInput(0, 0, data & 0xFFFFFFFF, flags, 0, 0)
        return item

    def _key_input(self, vk: int, scan: int, flags: int) -> _Input:
        item = _Input(type = _INPUT_KEYBOARD)
        item.ki = _KeyboardInput(vk, scan, flags, 0, 0)
        return item

    def _place(self, x: float, y: float) -> None:
        if not self._user32.SetCursorPos(int(round(x)), int(round(y))):
            raise BackendError("the pointer could not be moved.")

    def move(self, x: float, y: float) -> None:
        with self._dpi_aware():
            self._place(x, y)

    def click(self, x: float, y: float, button: str, count: int, mods: tuple[str, ...]) -> None:
        with self._dpi_aware():
            self._place(x, y)
            time.sleep(0.03)
            self._modifiers(mods, True)
            try:
                for _ in range(count):
                    self._send(self._mouse_input(_MOUSE_DOWN[button]))
                    self._send(self._mouse_input(_MOUSE_UP[button]))
                    time.sleep(0.03)
            finally:
                self._modifiers(mods, False)

    def drag(self, points: list[tuple[float, float]]) -> None:
        with self._dpi_aware():
            self._place(*points[0])
            time.sleep(0.05)
            self._send(self._mouse_input(_MOUSE_DOWN["left"]))
            try:
                time.sleep(0.08)
                for x, y in points[1:]:
                    self._place(x, y)
                    time.sleep(0.012)
                time.sleep(0.08)
            finally:
                self._send(self._mouse_input(_MOUSE_UP["left"]))

    def scroll(self, dx: int, dy: int) -> None:
        # A positive wheel value scrolls up; the tool counts down as positive.
        if dy:
            self._send(self._mouse_input(_WHEEL, -dy * _WHEEL_DELTA))
        if dx:
            self._send(self._mouse_input(_HWHEEL, dx * _WHEEL_DELTA))

    def _vk_event(self, name: str, vk: int, down: bool) -> _Input:
        flags = (0 if down else _KEYUP) | (_EXTENDED if name in _EXTENDED_KEYS else 0)
        return self._key_input(vk, 0, flags)

    def _modifiers(self, mods: tuple[str, ...], down: bool) -> None:
        for mod in mods if down else reversed(mods):
            self._send(self._vk_event(mod, _VK_MODIFIERS[mod], down))

    def key(self, mods: tuple[str, ...], key: Optional[str]) -> None:
        if key is not None and key not in _VK:
            raise BackendError(f"the {key} key isn't available on Windows.")
        self._modifiers(mods, True)
        try:
            if key is not None:
                self._send(self._vk_event(key, _VK[key], True))
                self._send(self._vk_event(key, _VK[key], False))
        finally:
            self._modifiers(mods, False)
        time.sleep(0.01)

    def type_text(self, text: str) -> None:
        for char in text.replace("\r\n", "\n"):
            if char in ("\n", "\r"):
                self.key((), "enter")
                continue
            if char == "\t":
                self.key((), "tab")
                continue
            units = char.encode("utf-16-le")
            for offset in range(0, len(units), 2):
                unit = int.from_bytes(units[offset : offset + 2], "little")
                self._send(self._key_input(0, unit, _UNICODE))
                self._send(self._key_input(0, unit, _UNICODE | _KEYUP))
            time.sleep(0.002)

    def _window(self, hwnd: Any) -> FrontWindow:
        user32 = self._user32
        title = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, title, len(title))
        rect = wintypes.RECT()
        with self._dpi_aware():
            has_rect = bool(user32.GetWindowRect(hwnd, ctypes.byref(rect)))
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        exe = ""
        handle = self._kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if handle:
            try:
                path = ctypes.create_unicode_buffer(1024)
                size = wintypes.DWORD(len(path))
                if self._kernel32.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                    exe = os.path.basename(path.value)
            finally:
                self._kernel32.CloseHandle(handle)
        return FrontWindow(
            app = exe.rsplit(".", 1)[0] if exe else "",
            app_id = exe.lower(),
            title = title.value,
            pid = pid.value,
            bounds = (
                (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
                if has_rect
                else None
            ),
        )

    def front_window(self) -> FrontWindow:
        hwnd = self._user32.GetForegroundWindow()
        return self._window(hwnd) if hwnd else FrontWindow()

    def window_at(self, x: float, y: float) -> Optional[FrontWindow]:
        with self._dpi_aware():
            hwnd = self._user32.WindowFromPoint(wintypes.POINT(int(round(x)), int(round(y))))
        # The control under the point belongs to a top-level window, which carries the title.
        root = self._user32.GetAncestor(hwnd, _GA_ROOT) if hwnd else None
        return self._window(root) if root else None

    def open_app(self, name: str) -> str:
        if any(char in name for char in '\\/:"<>|') or name.startswith(("-", ".")):
            raise BackendError(
                "give the application's name as the Start menu shows it, not a path."
            )
        root = os.environ.get("SystemRoot") or r"C:\Windows"
        shell = os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
        # The name travels in the environment, never in the script text.
        env = {**os.environ, "UNSLOTH_COMPUTER_USE_APP": name}
        try:
            done = subprocess.run(
                [shell, "-NoProfile", "-NonInteractive", "-Command", _OPEN_APP_SCRIPT],
                check = False,
                timeout = 30,
                capture_output = True,
                text = True,
                env = env,
                creationflags = _CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BackendError("the application could not be opened.") from exc
        if done.returncode != 0:
            raise BackendError(f"no application named {name} was found in the Start menu.")
        return done.stdout.strip().splitlines()[-1] if done.stdout.strip() else name
