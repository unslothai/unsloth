# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Platform backends on a real display: Linux on a private Xvfb, macOS read-only, Windows on CI."""

import contextlib
import os
import shutil
import subprocess
import sys
import time

import pytest

from core.inference.computer_use.backends import BackendError, Display, FrontWindow

LIVE = os.environ.get("UNSLOTH_COMPUTER_USE_LIVE") == "1"


def _wait_for(done, seconds):
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        time.sleep(0.1)


@contextlib.contextmanager
def _tk_window(tmp_path):
    """The Tk target window; yields the file its button writes the field's text to."""
    out, ready = tmp_path / "out", tmp_path / "ready"
    app = subprocess.Popen([sys.executable, "-c", _TK_APP, str(out), str(ready)])
    try:
        _wait_for(lambda: ready.exists() or app.poll() is not None, 30)
        assert ready.exists(), "the Tk window never showed"
        yield out
    finally:
        app.terminate()
        app.wait(timeout = 10)


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


def test_x11_without_a_window_manager_knows_the_focused_window(xvfb, tmp_path):
    pytest.importorskip("tkinter")
    from core.inference.computer_use.backends.linux import LinuxBackend

    with _tk_window(tmp_path):
        front = LinuxBackend().front_window()
    assert front.app and front.title == "cu-e2e", front


def test_x11_typing_with_caps_lock_on_keeps_the_case_and_the_lock(xvfb, tmp_path):
    pytest.importorskip("tkinter")
    from core.inference.computer_use.backends.linux import LinuxBackend
    with _tk_window(tmp_path) as out:
        backend = LinuxBackend()
        # No window manager: the Tk window sits exactly at +120+120.
        backend.click(420, 220, "left", 1, ())
        backend._set_caps_lock(True)
        backend.type_text("Hello abc")
        assert backend._caps_locked()
        backend._set_caps_lock(False)
        backend.click(440, 430, "left", 1, ())
        _wait_for(out.exists, 10)
        assert out.read_text(encoding = "utf-8") == "Hello abc"


def test_x11_typing_borrows_a_key_and_gives_it_back(xvfb):
    import ctypes

    from core.inference.computer_use.backends.linux import LinuxBackend

    backend = LinuxBackend()
    if backend.unavailable_reason():
        pytest.skip(backend.unavailable_reason())
    _table, spare = backend._keyboard_map()
    assert spare is not None
    backend.type_text("Hi, wörld → 你好\tdone\n")
    _table, after = backend._keyboard_map()
    assert after == spare, "the borrowed keycode must be empty again"
    per = ctypes.c_int(0)
    symbols = backend._x11.XGetKeyboardMapping(backend._display, spare, 1, ctypes.byref(per))
    try:
        assert not any(symbols[level] for level in range(per.value))
    finally:
        backend._x11.XFree(symbols)


def test_x11_finds_the_window_under_a_point_in_front_or_behind(xvfb):
    import ctypes
    import ctypes.util

    from core.inference.computer_use.backends.linux import LinuxBackend

    x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XCreateSimpleWindow.restype = ctypes.c_ulong
    x11.XCreateSimpleWindow.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_ulong, ctypes.c_ulong,
    ]  # fmt: skip
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XChangeProperty.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int,
    ]  # fmt: skip
    x11.XMapWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(None)
    assert display
    try:
        root = x11.XDefaultRootWindow(display)
        string = x11.XInternAtom(display, b"STRING", 0)
        wm_class = x11.XInternAtom(display, b"WM_CLASS", 0)

        def window(x, y, width, height, klass):
            made = x11.XCreateSimpleWindow(display, root, x, y, width, height, 0, 0, 0)
            if klass:
                x11.XChangeProperty(display, made, wm_class, string, 8, 0, klass, len(klass))
            x11.XMapWindow(display, made)
            return made

        window(0, 0, 400, 300, b"mail\x00Mail\x00")
        # Mapped later, so it lies on top where the two overlap.
        window(200, 0, 400, 300, b"notes\x00Notes\x00")
        x11.XSync(display, 0)
        backend = LinuxBackend()
        assert backend.unavailable_reason() is None
        behind = backend.window_at(50, 50)
        assert behind is not None and (behind.app, behind.app_id) == ("Mail", "mail")
        assert behind.bounds == (0, 0, 400, 300)
        assert backend.window_at(300, 50).app == "Notes"
        assert backend.window_at(500, 50).app == "Notes"
        assert backend.window_at(700, 500) is None
    finally:
        x11.XCloseDisplay(display)


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
        assert image.size == (int(primary.width), int(primary.height))
        windows.key(("shift",), None)
        windows.scroll(0, 1)
        with pytest.raises(BackendError):
            windows.key((), "no-such-key")
        assert isinstance(windows.front_window(), FrontWindow)
    finally:
        windows.move(*start)


def test_windows_names_an_app_as_the_start_menu_does(windows):
    system = os.path.join(os.environ.get("SystemRoot") or r"C:\Windows", "System32")
    assert windows._description(os.path.join(system, "notepad.exe")) == "Notepad"
    assert windows._description(os.path.join(system, "no-such-program.exe")) == ""
    front = windows.front_window()
    assert isinstance(front.app, str) and front.app_id == front.app_id.lower()


def test_windows_reads_a_non_ascii_app_name_back_from_powershell(windows):
    shown = "Unsloth Café 日本"
    env = {**os.environ, "UNSLOTH_COMPUTER_USE_APP": shown}
    echo = "Write-Output $env:UNSLOTH_COMPUTER_USE_APP"
    assert windows._powershell(echo, env).stdout.strip() == shown


def test_windows_names_a_window_by_its_start_menu_entry(windows):
    names = windows._start_names()
    assert names and all(exe.endswith(".exe") for exe in names), names
    system = os.path.join(os.environ.get("SystemRoot") or r"C:\Windows", "System32")
    if "cmd.exe" in names:
        assert windows._app_name(os.path.join(system, "cmd.exe")) == names["cmd.exe"]


def test_windows_open_app_takes_names_not_paths(windows):
    for name in ("C:\\Windows\\notepad.exe", "..\\x", "-n", "a|b"):
        with pytest.raises(BackendError, match = "name"):
            windows.open_app(name)


_TK_APP = r"""
import sys, tkinter as tk
out, ready = sys.argv[1], sys.argv[2]
root = tk.Tk()
root.title("cu-e2e")
root.geometry("640x420+120+120")
root.attributes("-topmost", True)
root.configure(bg = "#202020")
field = tk.Entry(root, bg = "#ff00ff", fg = "#000000", font = ("Courier", 24), relief = "flat")
field.place(x = 40, y = 40, width = 560, height = 120)
def press(_event = None):
    open(out, "w", encoding = "utf-8").write(field.get())
    button.configure(bg = "#0000ff")
button = tk.Label(root, bg = "#00c800", text = "")
button.place(x = 40, y = 240, width = 560, height = 140)
button.bind("<Button-1>", press)
def shown():
    root.lift()
    root.focus_force()
    open(ready, "w").write("1")
root.after(800, shown)
root.mainloop()
"""


@pytest.fixture
def real_desktop(request):
    if sys.platform.startswith("linux"):
        request.getfixturevalue("xvfb")
    elif not (LIVE or os.environ.get("CI")):
        pytest.skip("drives the real pointer: set UNSLOTH_COMPUTER_USE_LIVE=1")
    if os.environ.get("CI"):
        import tkinter  # noqa: F401 - on CI a skip here would pass without driving anything
    else:
        pytest.importorskip("tkinter", reason = "needs Tk to draw the target window")
    from core.inference.computer_use import backends

    backends.reset_backend()
    backend = backends.get_backend()
    assert backend.unavailable_reason() is None, backend.unavailable_reason()
    grants = backend.permissions()
    assert all(
        value is not False for value in grants.values()
    ), f"grant Screen Recording and Accessibility: {grants}"
    yield backend
    backends.reset_backend()


def _latest_picture(messages):
    import base64
    import io
    import json
    import re

    from PIL import Image

    found = re.findall(r"data:image/(?:png|jpeg);base64,([A-Za-z0-9+/=]+)", json.dumps(messages))
    assert found, "the model never got a screenshot back"
    return Image.open(io.BytesIO(base64.b64decode(found[-1]))).convert("RGB")


def _centre(
    picture,
    colour,
    tolerance = 70,
):
    import numpy as np

    pixels = np.asarray(picture).astype(int)
    near = (np.abs(pixels - np.array(colour)).sum(axis = 2)) < tolerance
    ys, xs = np.nonzero(near)
    assert len(xs) > 500, f"{colour} is not on the screenshot the model got"
    return int(np.median(xs)), int(np.median(ys))


def test_a_scripted_model_types_into_a_real_window(real_desktop, tmp_path, monkeypatch, request):
    """A fake model sees the real screen through the tool loop, clicks and types into a Tk window."""
    import asyncio
    import json
    import threading
    import time

    from core.inference import studio_tool_loop as loop_mod
    from core.inference.computer_use import computer_tool, policy, screen
    from core.inference.studio_tool_loop import (
        ToolLoopPolicy,
        ToolLoopRun,
        stream_with_studio_tools,
    )
    from state import computer_use as state
    from utils import host_policy

    state.reset()
    host_policy._reset_loopback_default_state()
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    chosen = {"enabled": True, "coordinates": "pixels", "allowed_apps": [], "blocked_apps": []}
    monkeypatch.setattr(policy, "settings", lambda: dict(chosen))
    asked: list = []
    monkeypatch.setattr(loop_mod, "begin_tool_decision", lambda session, approval: {"slot": True})
    monkeypatch.setattr(
        loop_mod,
        "wait_tool_decision",
        lambda slot, approval, cancel_event = None: asked.append(approval) or "allow",
    )
    monkeypatch.setattr(loop_mod, "abort_tool_decision", lambda slot, approval: None)

    request.addfinalizer(state.reset)
    request.addfinalizer(host_policy._reset_loopback_default_state)
    with _tk_window(tmp_path) as out:
        time.sleep(0.5)

        typed = "hello computer"

        def turn(messages):
            calls = sum(1 for m in messages if m.get("role") == "tool")
            if calls == 0:
                return {"action": "screenshot", "why": "look"}
            if calls == 1:
                picture = _latest_picture(messages)
                fx, fy = _centre(picture, (255, 0, 255))
                bx, by = _centre(picture, (0, 200, 0))
                return {
                    "action": "click", "x": fx, "y": fy,
                    "then": [{"action": "type", "text": typed}, {"action": "click", "x": bx, "y": by}],
                }  # fmt: skip
            return None

        class Model:
            heals_text_tool_calls = False
            sanitizes_provider_frames = False

            def __init__(self):
                self.seen = []

            def stream(self, *, messages, tools, tool_choice, cancel_event):
                self.seen.append([dict(m) for m in messages])
                call = turn(messages)
                if call is None:
                    delta, finish = {"content": "done"}, "stop"
                else:
                    delta = {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"c{len(self.seen)}",
                                "function": {"name": "computer", "arguments": json.dumps(call)},
                            }
                        ]
                    }
                    finish = "tool_calls"

                async def lines():
                    for payload in (
                        {"choices": [{"index": 0, "delta": delta}]},
                        {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                    ):
                        yield "data: " + json.dumps(payload)
                    yield "data: [DONE]"

                return lines()

        model = Model()

        async def drive():
            events = []
            async for line in stream_with_studio_tools(
                model,
                run = ToolLoopRun(
                    messages = [{"role": "user", "content": "type into the pink field"}],
                    session_id = "e2e",
                    thread_id = "e2e",
                    supports_vision = True,
                ),
                policy = ToolLoopPolicy(
                    tools = [computer_tool("pixels", real_desktop)],
                    max_calls = 10,
                    timeout = 120,
                    permission_mode = "ask",
                    confirm_calls = True,
                    bypass_permissions = False,
                    rag_scope = None,
                ),
                cancel_event = threading.Event(),
            ):
                events.append(line)
            return events

        monkeypatch.setattr(screen, "_SETTLE_FIRST_S", 0.3)
        lines = asyncio.run(asyncio.wait_for(drive(), timeout = 180))
        results = [
            json.loads(line[6:])
            for line in lines
            if line.startswith("data: {") and '"tool_end"' in line
        ]
        texts = [result.get("result", "") for result in results]
        assert len(results) == 2 and not any(t.startswith("Error") for t in texts), texts
        assert len(asked) == 2, "each call in ask mode waits for Allow"
        assert out.read_text(encoding = "utf-8") == typed, texts
        assert _centre(
            _latest_picture(model.seen[-1]), (0, 0, 255)
        ), "the last screenshot shows the press"
