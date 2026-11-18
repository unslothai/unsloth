# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An MCP server with nothing listening (the editor or app is closed) gets a plain answer, not a raw connect error.

Seen live with Unreal: the editor stopped mid-chat and the model got "Client failed to connect: All connection attempts
failed" on every call, then slept and retried for minutes."""

import socket

import pytest

from core.inference import mcp_client
from core.inference import tools
from storage import mcp_servers_db as db


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("scope", [None, "s=chat:t=thread"])
def test_nothing_listening_is_reported_as_unreachable(scope):
    url = f"http://127.0.0.1:{_closed_port()}/mcp"
    result = mcp_client.call_tool_sync(url, None, "list_toolsets", {}, timeout = 20, scope = scope)
    assert result.startswith("Error: MCP tool 'list_toolsets' did not run: ")
    assert (
        mcp_client.UNREACHABLE_SERVER in result
        and "waiting or calling it again will not help" in result
    )


def test_unreachable_names_the_server_and_its_setup_step(monkeypatch):
    port = _closed_port()
    db.create_server(
        "ue0123456789abcd", "Unreal Engine", f"http://127.0.0.1:{port}/mcp", builtin_id = "unreal"
    )
    db.create_server("fedcba9876543210", "My server", f"http://127.0.0.1:{port}/mcp")
    monkeypatch.setattr(tools, "stdio_mcp_enabled", lambda: True)
    result = tools.execute_tool("mcp__unreal__list_toolsets", {}, timeout = 20)
    assert "'Unreal Engine' is not reachable" in result
    assert result.endswith("Open your project with the Unreal MCP plugin enabled.")
    other = tools.execute_tool("mcp__fedcba9876543210__anything", {}, timeout = 20)
    assert "'My server' is not reachable" in other and "Unreal" not in other


def test_other_failures_keep_their_message():
    class Boom(Exception):
        pass

    assert not mcp_client.server_unreachable(Boom("bad request"))
    wrapped = RuntimeError("Client failed to connect: All connection attempts failed")
    assert mcp_client.server_unreachable(wrapped)
    try:
        try:
            raise ConnectionRefusedError(111, "refused")
        except ConnectionRefusedError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as outer:
        assert mcp_client.server_unreachable(outer)
