# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import pytest
from core.agent_workspace import git_guard
from core.agent_workspace.git_context import AgentWorkspaceError
from core.inference import mcp_client
from routes import project_worktrees


def test_git_mutation_refuses_before_workspace_access_without_lifecycle(monkeypatch):
    monkeypatch.setattr(git_guard, "project_retirement_available", lambda: False)

    def unexpected(*args):
        raise AssertionError("Must refuse before accessing the workspace")

    monkeypatch.setattr(git_guard, "project_workspace_access", unexpected)
    with pytest.raises(AgentWorkspaceError, match = "lifecycle support"):
        with git_guard.project_git_guard("project"):
            pytest.fail("Mutation was admitted")


def test_oauth_handoff_refuses_before_probe_without_configuration_checks(monkeypatch):
    monkeypatch.setattr(
        project_worktrees.mcp_servers_db,
        "get_server",
        lambda _id: {
            "is_enabled": True,
            "url": "https://github.example/mcp",
            "use_oauth": True,
        },
    )
    monkeypatch.delattr(mcp_client, "MCP_ONE_SHOT_CONFIG_CHECK_VERSION", raising = False)

    async def unexpected(**kwargs):
        pytest.fail("Must refuse before opening the connector")

    monkeypatch.setattr(project_worktrees, "list_tools_async", unexpected)
    with pytest.raises(AgentWorkspaceError, match = "configuration checks"):
        asyncio.run(project_worktrees._connector_tools("github"))
