# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The platform backends against a real display. Linux gets a private Xvfb, so nothing the
developer is looking at moves; macOS only reads; Windows moves the real pointer, so it runs on CI
runners (or with UNSLOTH_COMPUTER_USE_LIVE=1) and puts the pointer back."""

import os
import shutil
import subprocess
import sys

import pytest

from core.inference.computer_use.backends import BackendError, Display, FrontWindow

LIVE = os.environ.get("UNSLOTH_COMPUTER_USE_LIVE") == "1"


# ── Linux, on a throwaway X server ──


@pytest.fixture
def xvfb(monkeypatch):
    binary = shutil.which("Xvfb")
    if not sys.platform.startswith("linux") or binary is None:
        pytest.skip("needs Linux with Xvfb")
    read, write = os.pipe()
    server = subprocess.Popen(
        [binary, "-displayfd", str(write), "-screen", "0", "1280x800x24", "-nolisten", "tcp"],
        pass_fds = (write,),
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    )
    os.close(write)
    try:
        with os.fdopen(read) as numbers:
            number = numbers.readline().strip()
        if not number:
            pytest.skip("Xvfb did not start")
        monkeypatch.setenv("DISPLAY", f":{number}")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising = False)
        monkeypatch.delenv("XDG_SESSION_TYPE", raising = False)
        yield
    finally:
        server.terminate()
        server.wait(timeout = 10)


def test_x11_pointer_keys_and_capture(xvfb):
    from core.inference.computer_use.backends.linux import LinuxBackend

    backend = LinuxBackend()
    if backend.unavailable_reason():
        pytest.skip(backend.unavailable_reason())
    assert backend.displays() == [Display(1, 0, 0, 1280, 800, True)]
    backend.move(321, 123)
    assert backend.cursor() == (321, 123)
    backend.click(640, 400, "left", 2, ("shift",))
    assert backend.cursor() == (640, 400)
    backend.drag([(10, 10), (60, 40), (200, 150)])
    assert backend.cursor() == (200, 150)
    backend.scroll(1, -2)
    backend.key(("ctrl", "shift"), "a")
    backend.key((), "enter")
    backend.key(("alt",), None)
    with pytest.raises(BackendError):
        backend.key((), "no-such-key")
    # No window manager here, so nothing is in front: that reads as unknown, not as an error.
    assert backend.front_window() == FrontWindow()
    try:
        image = backend.capture(backend.displays()[0])
    except BackendError:
        pytest.skip("this Pillow build has no X11 screen grab")
    assert image.size == (1280, 800)


def test_x11_typing_borrows_a_key_and_gives_it_back(xvfb):
    import ctypes

    from core.inference.computer_use.backends.linux import LinuxBackend

    backend = LinuxBackend()
    if backend.unavailable_reason():
        pytest.skip(backend.unavailable_reason())
    _table, spare = backend._keyboard_map()
    assert spare is not None
    # Characters on the layout, a shifted one, and ones no layout key carries.
    backend.type_text("Hi, wörld → 你好\tdone\n")
    _table, after = backend._keyboard_map()
    assert after == spare, "the borrowed keycode must be empty again"
    per = ctypes.c_int(0)
    symbols = backend._x11.XGetKeyboardMapping(backend._display, spare, 1, ctypes.byref(per))
    try:
        assert not any(symbols[level] for level in range(per.value))
    finally:
        backend._x11.XFree(symbols)


def test_a_wayland_session_is_reported_not_attempted(monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux only")
    from core.inference.computer_use.backends.linux import LinuxBackend

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    backend = LinuxBackend()
    assert "Wayland" in backend.unavailable_reason()
    with pytest.raises(BackendError, match = "Wayland"):
        backend.move(1, 1)
    assert backend.front_window() == FrontWindow()


# ── macOS: nothing here moves the pointer or raises a system prompt ──


@pytest.fixture
def mac():
    if sys.platform != "darwin":
        pytest.skip("macOS only")
    from core.inference.computer_use.backends.macos import MacBackend
    return MacBackend()


def test_macos_reads_displays_pointer_and_front_app(mac):
    displays = mac.displays()
    assert displays and displays[0].primary and displays[0].index == 1
    assert all(d.width > 0 and d.height > 0 for d in displays)
    x, y = mac.cursor()
    assert any(
        d.x - 1 <= x <= d.x + d.width + 1 and d.y - 1 <= y <= d.y + d.height + 1 for d in displays
    )
    grants = mac.permissions()
    assert set(grants) == {"screen", "input"}
    assert all(isinstance(value, bool) for value in grants.values())
    front = mac.front_window()
    assert isinstance(front, FrontWindow)
    if front.bounds is not None:
        assert front.bounds[2] > 0 and front.bounds[3] > 0


def test_macos_without_its_grants_refuses_before_any_event(mac):
    grants = mac.permissions()
    if grants["screen"] and grants["input"]:
        pytest.skip("this process holds both grants")
    if not grants["screen"]:
        with pytest.raises(BackendError, match = "Screen"):
            mac.capture(mac.displays()[0])
    if not grants["input"]:
        for act in (
            lambda: mac.move(5, 5),
            lambda: mac.click(5, 5, "left", 1, ()),
            lambda: mac.key(("cmd",), "a"),
            lambda: mac.type_text("x"),
            lambda: mac.scroll(0, 1),
        ):
            with pytest.raises(BackendError, match = "Accessibility"):
                act()


def test_macos_open_app_takes_names_not_paths(mac):
    for name in ("/Applications/Safari.app", "../x", "-n", "file:///tmp/x.app"):
        with pytest.raises(BackendError, match = "name"):
            mac.open_app(name)


# ── Windows, on a runner's desktop ──


@pytest.fixture
def windows():
    if sys.platform != "win32":
        pytest.skip("Windows only")
    if not (LIVE or os.environ.get("CI")):
        pytest.skip("moves the real pointer: set UNSLOTH_COMPUTER_USE_LIVE=1")
    from core.inference.computer_use.backends.windows import WindowsBackend

    backend = WindowsBackend()
    if backend.unavailable_reason():
        pytest.skip(backend.unavailable_reason())
    try:
        # A runner with no input desktop (a locked or disconnected session) can't place the pointer.
        backend.move(*backend.cursor())
    except BackendError as exc:
        pytest.skip(str(exc))
    return backend


def test_windows_pointer_keys_and_capture(windows):
    displays = windows.displays()
    assert displays and displays[0].primary
    primary = displays[0]
    start = windows.cursor()
    try:
        target = (primary.x + 200, primary.y + 150)
        windows.move(*target)
        assert windows.cursor() == target
        image = windows.capture(primary)
        # Per-monitor DPI aware on both sides: a capture pixel is an input position.
        assert image.size == (int(primary.width), int(primary.height))
        windows.key(("shift",), None)
        windows.scroll(0, 1)
        with pytest.raises(BackendError):
            windows.key((), "no-such-key")
        assert isinstance(windows.front_window(), FrontWindow)
    finally:
        windows.move(*start)


def test_windows_open_app_takes_names_not_paths(windows):
    for name in ("C:\\Windows\\notepad.exe", "..\\x", "-n", "a|b"):
        with pytest.raises(BackendError, match = "name"):
            windows.open_app(name)
