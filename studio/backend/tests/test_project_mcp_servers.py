# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

from core.inference import mcp_client, tools
from models.inference import ChatCompletionRequest
from routes.chat_history import ChatProjectPatch, patch_project
from routes.inference import _select_request_tools
from storage import mcp_servers_db, studio_db


def test_chat_uses_its_projects_saved_mcp_servers(monkeypatch):
    """Switch between project chats and check listing, deferred lookup and execution."""
    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    alpha = "a" * 16
    beta = "b" * 16
    disabled = "c" * 16
    for server_id, enabled in ((alpha, True), (beta, True), (disabled, False)):
        mcp_servers_db.create_server(
            id = server_id,
            display_name = server_id,
            url = "https://example.invalid/mcp",
            is_enabled = enabled,
        )
        mcp_client.cache_tools(server_id, [{"name": "echo"}])

    for project_id in ("alpha", "beta", "legacy"):
        studio_db.upsert_chat_project(
            {"id": project_id, "name": project_id, "createdAt": 1, "updatedAt": 2}
        )

    assert patch_project(
        "alpha", ChatProjectPatch(mcpServerIds = [alpha, disabled]), "owner"
    ).mcpServerIds == [alpha, disabled]
    assert patch_project("beta", ChatProjectPatch(mcpServerIds = [beta]), "owner").mcpServerIds == [
        beta
    ]

    async def list_chat(session_id, server_ids):
        """Inspect the actual completion selector and the cached catalog used to count it."""
        payload = ChatCompletionRequest(
            messages = [{"role": "user", "content": "Hello"}],
            session_id = session_id,
        )
        names = {f"mcp__{server_id}__echo" for server_id in server_ids}
        listed = await _select_request_tools(payload, tools_on = False, mcp_allowed = True)
        cached, complete = tools.cached_mcp_tools()
        assert complete
        assert {tool["function"]["name"] for tool in listed} == names
        assert {tool["function"]["name"] for tool in cached} == names
        assert bool(tools.mcp_tool_spec(f"mcp__{beta}__echo")) == (beta in server_ids)

    async def switch_chats():
        """The same task may select a different chat later without retaining the earlier scope."""
        alpha_session = tools.project_session_id("alpha")
        beta_session = tools.project_session_id("beta")
        await list_chat(alpha_session, [alpha])
        assert "not enabled for this project" in tools.execute_mcp_tool(
            f"mcp__{beta}__echo",
            {},
            session_id = alpha_session,
        )
        await list_chat(beta_session, [beta])
        await list_chat(tools.project_session_id("legacy"), [alpha, beta])
        await list_chat("unfiled-thread", [alpha, beta])
        for selection, expected in (([], []), (None, [alpha, beta])):
            patch_project("alpha", ChatProjectPatch(mcpServerIds = selection), "owner")
            assert studio_db.get_chat_project("alpha")["mcpServerIds"] == selection
            await list_chat(alpha_session, expected)
        await list_chat(beta_session, [beta])

    asyncio.run(switch_chats())
