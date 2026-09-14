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
    "playwright_stream_pacing",
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


@pytest.fixture
def installed_frontend(monkeypatch, tmp_path):
    """A frontend tree that satisfies start_vite's toolchain precondition, without an install.

    `start_vite` refuses up front when `studio/frontend/node_modules` carries no vite, which
    is the whole point of #9654: a missing toolchain must not reach npm and come back as
    "vite exited with code 127", because that reads as a vite crash rather than as a setup
    step nobody ran. It is a precondition of the same kind as the occupied-port refusal
    above it, and the tests below are about process-group selection and port refusal, not
    about the toolchain, so they get a tree that has one.

    Pointed at a tmp_path tree rather than stubbed out on purpose. Stubbing
    `_require_frontend_toolchain` to a no-op would keep these tests green if the check were
    deleted outright; a synthetic tree makes the check actually run, and
    test_start_vite_refuses_a_tree_with_no_frontend_toolchain below pins the other
    direction. It also keeps this file's promise that nothing here touches a real install.
    """
    binaries = tmp_path / "node_modules" / ".bin"
    binaries.mkdir(parents = True)
    (binaries / "vite").write_text("#!/bin/sh\n", encoding = "utf-8")
    monkeypatch.setattr(robust, "FRONTEND", tmp_path)
    return tmp_path


@pytest.mark.parametrize("osname", ["posix", "nt"])
def test_start_vite_picks_the_platform_process_group(
    monkeypatch, no_signals, installed_frontend, osname
) -> None:
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


def _require_playwright_page():
    """
    Skip unless `from playwright.sync_api import Page` would actually work.

    Two weaker guards were tried and both let this through. Checking the
    top-level package passes because "playwright" resolves as a namespace
    directory on the Repo tests (CPU) runner; checking "playwright.sync_api"
    passes too, because that resolves as a namespace package as well. Only the
    symbol the harnesses import is a real test of whether the import below can
    succeed, so that is what is checked, and it is checked the way the harness
    does it. The failure mode is a skip condition reported as

      ImportError: cannot import name 'Page' from 'playwright.sync_api'
      (unknown location)

    on every branch, which costs an investigation each time it is seen.
    """
    sync_api = pytest.importorskip("playwright.sync_api")
    if not hasattr(sync_api, "Page"):
        pytest.skip(
            "playwright.sync_api resolved from "
            f"{getattr(sync_api, '__file__', None) or list(getattr(sync_api, '__path__', []))} "
            "but has no Page; playwright is not usably installed here"
        )


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


def test_an_occupied_port_is_refused_rather_than_measured(
    monkeypatch, no_signals, installed_frontend
) -> None:
    """--strictPort makes our vite exit, and the readiness poll would then be reading whatever
    else holds the port. Refuse up front instead.

    Given a satisfied toolchain even though the port check currently runs first, so this
    keeps asserting the port refusal specifically and not the order the two preconditions
    happen to be written in.
    """
    monkeypatch.setattr(robust, "_port_is_taken", lambda port, host: True)
    with pytest.raises(RuntimeError, match = "already serving"):
        robust.start_vite(5199)


def test_start_vite_refuses_a_tree_with_no_frontend_toolchain(
    monkeypatch, no_signals, tmp_path
) -> None:
    """The failure #9654 exists to name, and the reason the refusal has to be up front.

    A job that installs Unsloth from a warm frontend-dist cache never builds the frontend, so
    setup.sh skips its npm install and node_modules is never created. Reaching npm in that
    state costs a spawn and returns `vite exited with code 127`, which is indistinguishable
    from vite crashing. So the assertion is not only that it raises: it is that nothing was
    spawned, because a refusal that lands after Popen has already lost the cause.
    """
    (tmp_path / "node_modules").mkdir()
    monkeypatch.setattr(robust, "FRONTEND", tmp_path)
    monkeypatch.setattr(robust, "_port_is_taken", lambda port, host: False)
    spawned: list = []
    monkeypatch.setattr(robust.subprocess, "Popen", lambda cmd, **kw: spawned.append(cmd))
    with pytest.raises(RuntimeError, match = "dev dependencies are not installed"):
        robust.start_vite(5199)
    assert spawned == [], "the refusal must land before npm is spawned, or the cause is lost"


@pytest.mark.parametrize("binary", ["vite", "vite.cmd", "vite.exe", "vite.bunx"])
def test_the_toolchain_check_accepts_every_platform_binary(monkeypatch, tmp_path, binary) -> None:
    """bun writes .bunx shims and npm writes .cmd/.exe on Windows, so a POSIX-only name test
    would reject a perfectly good Windows or bun tree and send someone chasing a phantom."""
    binaries = tmp_path / "node_modules" / ".bin"
    binaries.mkdir(parents = True)
    (binaries / binary).write_text("", encoding = "utf-8")
    monkeypatch.setattr(robust, "FRONTEND", tmp_path)
    robust._require_frontend_toolchain()


def test_the_toolchain_check_names_a_missing_frontend_separately(monkeypatch, tmp_path) -> None:
    """Run from outside the repo is a different mistake from run without an install, and the
    two must not share one message."""
    monkeypatch.setattr(robust, "FRONTEND", tmp_path / "not-a-checkout")
    with pytest.raises(RuntimeError, match = "no frontend at"):
        robust._require_frontend_toolchain()


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
    src = (Path(__file__).resolve().parent / f"{harness}.py").read_text(encoding = "utf-8")
    assert re.search(r'SMOKE_PORT",\s*"\d+"', src), f"{harness} has no SMOKE_PORT default"


def test_every_harness_picks_a_different_default_port() -> None:
    import re

    ports = {}
    for harness in HARNESSES:
        src = (Path(__file__).resolve().parent / f"{harness}.py").read_text(encoding = "utf-8")
        ports[harness] = re.search(r'SMOKE_PORT",\s*"(\d+)"', src).group(1)
    assert len(set(ports.values())) == len(HARNESSES), f"default ports collide: {ports}"


@pytest.mark.parametrize("harness", HARNESSES)
def test_an_empty_smoke_base_url_means_unset(harness, monkeypatch) -> None:
    """Exported-but-empty is common in shell wrappers. `in os.environ` would call it external
    and then drive "" as the base URL."""
    _require_playwright_page()
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
    _require_playwright_page()
    import importlib

    monkeypatch.setenv("SMOKE_BASE_URL", "http://127.0.0.1:9999")
    module = importlib.reload(importlib.import_module(harness))
    try:
        assert module.BASE == "http://127.0.0.1:9999"
        assert module.OWNS_SERVER is False
    finally:
        monkeypatch.delenv("SMOKE_BASE_URL", raising = False)
        importlib.reload(module)
