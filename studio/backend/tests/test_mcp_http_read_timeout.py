# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#14298: an HTTP/SSE MCP tool call must run for the whole Max Tool Call Duration, not stop at the
SDK's 300s httpx read timeout. The e2e test shrinks that SDK default to 1s and runs a 2s tool against
a real local server, so it fails in seconds without the fix instead of after five minutes."""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import mcp_client

_SERVER = textwrap.dedent(
    """
    import asyncio, sys
    from fastmcp import FastMCP

    mcp = FastMCP("slow")

    @mcp.tool
    async def slow(seconds: float) -> str:
        await asyncio.sleep(seconds)
        return f"slept {seconds}"

    mcp.run(transport = "http", host = "127.0.0.1", port = int(sys.argv[1]), show_banner = False)
    """
)


@pytest.fixture
def slow_server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-c", _SERVER, str(port)])
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout = 0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout = 10)


@pytest.mark.parametrize("restricted", [False, True])
def test_factory_lifts_only_the_read_cap(monkeypatch, restricted):
    monkeypatch.setattr(mcp_client, "_managed_mcp_restricted", lambda: restricted)
    from mcp.shared._httpx_utils import create_mcp_http_client

    timeout = type(create_mcp_http_client().timeout)(5.0, read = 300.0)  # what sse_client passes
    for kwargs in ({}, {"timeout": timeout, "follow_redirects": True}):
        t = mcp_client._tool_http_client_factory(**kwargs).timeout
        assert t.read is None
        assert t.connect == (5.0 if kwargs else 30.0)


@pytest.mark.parametrize("scope", [None, "chat-1"])  # one-shot client and cached session
def test_tool_call_outlives_sdk_read_timeout(slow_server, monkeypatch, scope):
    import mcp.shared._httpx_utils as httpx_utils

    monkeypatch.setattr(httpx_utils, "MCP_DEFAULT_SSE_READ_TIMEOUT", 1.0)
    out = mcp_client.call_tool_sync(slow_server, None, "slow", {"seconds": 2}, timeout = 30, scope = scope)
    assert "slept 2" in out, out


def test_call_budget_still_enforced(slow_server):
    out = mcp_client.call_tool_sync(slow_server, None, "slow", {"seconds": 10}, timeout = 2)
    assert out.startswith("Error") and "2s" in out, out
