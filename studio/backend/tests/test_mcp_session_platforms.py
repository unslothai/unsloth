# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Platform-dependent behaviour of the shared MCP session machinery.

What these can and cannot prove: Linux does not export asyncio.ProactorEventLoop
at all, so a win32 test has to supply one. That makes these honest tests of which
branch is *chosen*, and no test at all of IOCP, overlapped pipes or Windows handle
cleanup. Those need a Windows runner, and this repo runs studio/backend/tests on
ubuntu only. Do not read a pass here as Windows coverage.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import mcp_client

HTTP_URL = "https://mcp.example.test/mcp"
STDIO_URL = "npx fake-stateful-server"


class _FakeProactorLoop(asyncio.SelectorEventLoop):
    """Stands in for the loop Linux cannot construct. Only its identity matters:
    the assertion is that the win32 branch asked for a Proactor, not that this
    object behaves like one."""


@pytest.fixture
def win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(asyncio, "ProactorEventLoop", _FakeProactorLoop, raising = False)
    monkeypatch.setattr(mcp_client, "_IS_WINDOWS", True)


@pytest.fixture
def posix(monkeypatch):
    """Pin the platform: these assert POSIX semantics and must not follow the runner."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(mcp_client, "_IS_WINDOWS", False)


def test_windows_sessions_get_a_proactor_loop(win32):
    """A SelectorEventLoop cannot spawn subprocesses on Windows, so an stdio MCP
    server would fail outright if this branch regressed."""
    session = mcp_client._McpSession(STDIO_URL, None)
    try:
        assert isinstance(session.loop, _FakeProactorLoop)
    finally:
        session.close()


def test_posix_sessions_get_the_default_loop(posix):
    session = mcp_client._McpSession(STDIO_URL, None)
    try:
        assert not isinstance(session.loop, _FakeProactorLoop)
        assert isinstance(session.loop, asyncio.AbstractEventLoop)
    finally:
        session.close()


def test_the_loop_choice_does_not_depend_on_the_transport(win32):
    """HTTP sessions take the same branch; nothing about the fix made the loop
    choice transport-specific."""
    session = mcp_client._McpSession(HTTP_URL, None)
    try:
        assert isinstance(session.loop, _FakeProactorLoop)
    finally:
        session.close()


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_sessions_start_and_stop_cleanly_on_every_platform_branch(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(asyncio, "ProactorEventLoop", _FakeProactorLoop, raising = False)
    session = mcp_client._McpSession(HTTP_URL, None)
    session.close()
    assert session.closed.is_set()
    assert not session._thread.is_alive()


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_call_serialization_policy_is_platform_independent(monkeypatch, platform):
    """stdio serializes, HTTP does not, on every platform. A platform-dependent
    answer here would mean parallel tool calls behaved differently per OS."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(asyncio, "ProactorEventLoop", _FakeProactorLoop, raising = False)
    stdio = mcp_client._McpSession(STDIO_URL, None)
    http = mcp_client._McpSession(HTTP_URL, None)
    try:
        assert stdio.serialize_calls is True
        assert http.serialize_calls is False
    finally:
        stdio.close()
        http.close()


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_connect_window_policy_is_platform_independent(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    assert mcp_client._connect_window(HTTP_URL, 300.0) == 300.0
    assert mcp_client._connect_window(STDIO_URL, 300.0) == mcp_client._STDIO_CONNECT_TIMEOUT
    assert mcp_client._connect_window(HTTP_URL, None) is None
    assert mcp_client._connect_window(STDIO_URL, None) is None


@pytest.mark.skipif(os.name == "nt", reason = "POSIX fork semantics")
def test_a_forked_child_does_not_inherit_a_usable_session():
    """Only the forking thread survives a fork, so an inherited session's loop
    thread is gone. The child must not believe the cache is usable.

    Asserted from the child by exit code: a child that touches the dead loop
    would hang, and the parent would not see a clean 0."""
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    session = mcp_client._McpSession(HTTP_URL, None)
    try:
        proc = ctx.Process(target = _child_checks_inherited_thread_is_dead)
        proc.start()
        proc.join(30)
        assert proc.exitcode == 0, f"child exited {proc.exitcode}"
    finally:
        session.close()


def _child_checks_inherited_thread_is_dead():
    import threading

    # The parent's mcp-session thread does not exist here; anything that waits on
    # it would block forever, so the child must be able to see that.
    names = [t.name for t in threading.enumerate()]
    sys.exit(0 if "mcp-session" not in names else 1)
