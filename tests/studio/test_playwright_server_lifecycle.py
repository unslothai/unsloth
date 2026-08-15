# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The dev-server lifecycle the browser smokes share.

Linux CI exercises the POSIX path only, and the Windows path is the one nobody runs until it
is broken on someone's machine. These drive both by injecting `os.name`, so the branch that
picks CREATE_NEW_PROCESS_GROUP and taskkill is checked on every run.

Everything here is monkeypatched: no npm, no browser, no sockets bound.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _playwright_robust as robust  # noqa: E402

# Unix-only, so a Windows interpreter cannot name it even to drive the POSIX branch.
SIGKILL = getattr(signal, "SIGKILL", 9)

HARNESSES = (
    "playwright_chat_autoscroll",
    "playwright_research_freeze",
    "playwright_strip_ansi_smoke",
)


class _FakeProc:
    """A child that never dies, so both escalation steps are reachable."""

    def __init__(self) -> None:
        self.pid = 4242
        self.stdout = None
        self.returncode = None

    def poll(self):
        return None

    def wait(self, timeout = None):
        raise subprocess.TimeoutExpired("vite", timeout or 0)


@pytest.fixture
def no_signals(monkeypatch):
    monkeypatch.setattr(robust, "_arm_teardown_signals", lambda: None)
    monkeypatch.setattr(robust, "_LIVE_SERVERS", [])


@pytest.fixture
def posix_branch(monkeypatch, no_signals):
    """Drive the POSIX teardown from any host: os.killpg and signal.SIGKILL are Unix-only."""
    monkeypatch.setattr(robust.os, "name", "posix")
    monkeypatch.setattr(robust.signal, "SIGKILL", SIGKILL, raising = False)


@pytest.mark.parametrize("osname", ["posix", "nt"])
def test_start_vite_picks_the_platform_process_group(monkeypatch, no_signals, osname) -> None:
    captured: dict = {}
    monkeypatch.setattr(robust.os, "name", osname)
    monkeypatch.setattr(robust, "_port_is_taken", lambda port, host: False)
    monkeypatch.setattr(
        robust.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: None})()
    )
    monkeypatch.setattr(
        robust.subprocess, "Popen", lambda cmd, **kw: captured.update(cmd = cmd, kw = kw) or _FakeProc()
    )
    if osname == "nt":
        monkeypatch.setattr(robust.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising = False)

    robust.start_vite(5199)

    assert "--strictPort" in captured["cmd"], "a drifting port must fail, not pick another"
    if osname == "nt":
        assert captured["kw"]["creationflags"] == 0x200
        assert "start_new_session" not in captured["kw"]
    else:
        # Without its own session, killing the npm wrapper orphans the node child.
        assert captured["kw"]["start_new_session"] is True
        assert "creationflags" not in captured["kw"]


def test_posix_teardown_signals_the_group_and_escalates(monkeypatch, posix_branch) -> None:
    sent = []
    monkeypatch.setattr(
        robust.os, "killpg", lambda pid, sig: sent.append((pid, sig)), raising = False
    )
    robust.stop_process(_FakeProc())
    assert sent == [(4242, signal.SIGTERM), (4242, SIGKILL)]


def test_windows_teardown_kills_the_tree_and_escalates(monkeypatch, no_signals) -> None:
    calls = []
    monkeypatch.setattr(robust.os, "name", "nt")
    monkeypatch.setattr(robust.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    robust.stop_process(_FakeProc())
    assert calls == [
        ["taskkill", "/PID", "4242", "/T"],
        ["taskkill", "/PID", "4242", "/T", "/F"],
    ]


def test_teardown_never_raises_over_the_failure_that_called_it(monkeypatch, posix_branch) -> None:
    """stop_process runs from a `finally`. A child that outlives SIGKILL must not replace the
    harness's real error with a TimeoutExpired."""
    monkeypatch.setattr(robust.os, "killpg", lambda pid, sig: None, raising = False)
    robust.stop_process(_FakeProc())


def test_teardown_tolerates_a_process_that_already_vanished(monkeypatch, posix_branch) -> None:
    def gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(robust.os, "killpg", gone, raising = False)
    robust.stop_process(_FakeProc())


def test_an_occupied_port_is_refused_rather_than_measured(monkeypatch, no_signals) -> None:
    """--strictPort makes our vite exit, and the readiness poll would then be reading whatever
    else holds the port. Refuse up front instead."""
    monkeypatch.setattr(robust, "_port_is_taken", lambda port, host: True)
    with pytest.raises(RuntimeError, match = "already serving"):
        robust.start_vite(5199)


def test_readiness_gives_up_as_soon_as_our_server_dies(monkeypatch, no_signals) -> None:
    """Otherwise a dead server costs the full timeout, three times over, per CI run."""

    class Dead:
        returncode = 1
        vite_tail = ["Port 5199 is already in use"]

        def poll(self):
            return 1

    with pytest.raises(RuntimeError, match = "vite exited with code 1") as caught:
        robust.wait_for_smoke_page(
            "http://127.0.0.1:5199/x.html", "x.tsx", proc = Dead(), timeout_s = 30.0
        )
    assert "already in use" in str(caught.value), "vite's own reason should be surfaced"


@pytest.mark.parametrize("harness", HARNESSES)
def test_ports_do_not_collide_and_are_overridable(harness) -> None:
    import re
    src = (Path(__file__).resolve().parent / f"{harness}.py").read_text()
    assert re.search(r'SMOKE_PORT",\s*"\d+"', src), f"{harness} has no SMOKE_PORT default"


def test_every_harness_picks_a_different_default_port() -> None:
    import re

    ports = {}
    for harness in HARNESSES:
        src = (Path(__file__).resolve().parent / f"{harness}.py").read_text()
        ports[harness] = re.search(r'SMOKE_PORT",\s*"(\d+)"', src).group(1)
    assert len(set(ports.values())) == len(HARNESSES), f"default ports collide: {ports}"


@pytest.mark.parametrize("harness", HARNESSES)
def test_an_empty_smoke_base_url_means_unset(harness, monkeypatch) -> None:
    """Exported-but-empty is common in shell wrappers. `in os.environ` would call it external
    and then drive "" as the base URL."""
    pytest.importorskip("playwright")  # importing a harness pulls in playwright.sync_api
    import importlib

    monkeypatch.setenv("SMOKE_BASE_URL", "")
    module = importlib.reload(importlib.import_module(harness))
    try:
        assert module.BASE.startswith("http://"), f"empty SMOKE_BASE_URL gave BASE={module.BASE!r}"
    finally:
        monkeypatch.delenv("SMOKE_BASE_URL", raising = False)
        importlib.reload(module)


@pytest.mark.parametrize("harness", ("playwright_chat_autoscroll", "playwright_research_freeze"))
def test_an_external_smoke_base_url_is_still_honoured(harness, monkeypatch) -> None:
    """The documented pre-existing invocation. A harness that started its own server anyway
    would fail on the busy-port check."""
    pytest.importorskip("playwright")
    import importlib

    monkeypatch.setenv("SMOKE_BASE_URL", "http://127.0.0.1:9999")
    module = importlib.reload(importlib.import_module(harness))
    try:
        assert module.BASE == "http://127.0.0.1:9999"
        assert module.OWNS_SERVER is False
    finally:
        monkeypatch.delenv("SMOKE_BASE_URL", raising = False)
        importlib.reload(module)
