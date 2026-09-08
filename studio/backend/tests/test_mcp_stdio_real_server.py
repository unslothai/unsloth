# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from core.inference import mcp_client
from models.mcp_servers import (
    McpServerCreate,
    McpServerImportRequest,
    McpServerTestRequest,
    McpServerUpdate,
    McpStdioCommand,
    McpStdioDecodeRequest,
)
from routes import mcp_servers as routes_mcp
from storage import mcp_servers_db


FIXTURE = Path(__file__).parent / "fixtures" / "mcp_argument_echo_server.py"


def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)


def _encode(arguments: list[str]) -> str:
    return routes_mcp.encode_stdio_command(
        McpStdioCommand(command = sys.executable, arguments = [str(FIXTURE), *arguments]),
        current_subject = "u",
    ).url


def _launched_state(url: str, environment: dict[str, str]) -> dict:
    output = mcp_client.call_tool_sync(
        url,
        environment,
        "launch_state",
        {},
        timeout = 20,
    )
    assert not output.startswith("Error:"), output
    return json.loads(output)


@pytest.mark.timeout(90)
def test_stdio_arguments_survive_real_crud_import_probe_and_launch(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    original_arguments = [
        "--flag",
        "",
        "a b",
        'quote"inside',
        "single'quote",
        "trailing\\",
        "https://example.com/value?q=a%20b",
        "  keep outer spaces  ",
    ]
    environment = {"UNSLOTH_MCP_ARGUMENT_MARKER": "create value"}
    encoded = _encode(original_arguments)

    probe = asyncio.run(
        routes_mcp.test_mcp_server(
            McpServerTestRequest(url = encoded, headers = environment),
            current_subject = "u",
        )
    )
    assert (probe.ok, probe.tool_count, probe.error) == (True, 1, None)

    created = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(
                display_name = "argument echo",
                url = encoded,
                headers = environment,
            ),
            current_subject = "u",
        )
    )
    persisted = mcp_servers_db.get_server(created.id)
    assert persisted["url"] == encoded
    assert json.loads(persisted["headers_json"]) == environment

    listed = routes_mcp.list_mcp_servers(current_subject = "u")
    assert [row.id for row in listed] == [created.id]
    decoded = routes_mcp.decode_stdio_command(
        McpStdioDecodeRequest(url = listed[0].url), current_subject = "u"
    )
    assert decoded.command == sys.executable
    assert decoded.arguments == [str(FIXTURE), *original_arguments]
    assert _launched_state(persisted["url"], environment) == {
        "arguments": original_arguments,
        "marker": "create value",
    }

    edited_arguments = ["second", "", "first", "a&b", "x|y", "%TOKEN%"]
    edited_environment = {"UNSLOTH_MCP_ARGUMENT_MARKER": "edited value"}
    edited_url = _encode(edited_arguments)
    updated = asyncio.run(
        routes_mcp.update_mcp_server(
            created.id,
            McpServerUpdate(url = edited_url, headers = edited_environment),
            current_subject = "u",
        )
    )
    assert updated.url == edited_url
    refreshed = asyncio.run(routes_mcp.refresh_mcp_server_tools(created.id, current_subject = "u"))
    assert (refreshed.ok, refreshed.tool_count) == (True, 1)
    assert _launched_state(updated.url, edited_environment) == {
        "arguments": edited_arguments,
        "marker": "edited value",
    }

    imported_arguments = ["imported", "", "with spaces", 'say "hello"']
    imported = asyncio.run(
        routes_mcp.import_mcp_servers(
            McpServerImportRequest(
                config = {
                    "mcpServers": {
                        "imported echo": {
                            "command": sys.executable,
                            "args": [str(FIXTURE), *imported_arguments],
                            "env": {"UNSLOTH_MCP_ARGUMENT_MARKER": "import value"},
                        }
                    }
                }
            ),
            current_subject = "u",
        )
    )
    assert imported.errors == []
    assert imported.skipped == []
    assert len(imported.created) == 1
    imported_row = mcp_servers_db.get_server(imported.created[0].id)
    assert _launched_state(
        imported_row["url"],
        mcp_client.parse_server_headers(imported_row),
    ) == {
        "arguments": imported_arguments,
        "marker": "import value",
    }


@pytest.mark.skipif(os.name != "nt", reason = "requires Windows batch launch semantics")
@pytest.mark.timeout(45)
def test_safe_arguments_survive_a_real_windows_batch_launcher(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    launcher = tmp_path / "mcp-server-example.cmd"
    launcher.write_text(
        f'@echo off\r\n"{sys.executable}" "{FIXTURE}" %*\r\n',
        encoding = "utf-8",
    )
    arguments = [
        "--port",
        "3000",
        r"C:\Users\me\data",
        r"C:\Program Files (x86)\mcp data",
        "a & b",
        "",
    ]
    url = routes_mcp.encode_stdio_command(
        McpStdioCommand(command = str(launcher), arguments = arguments),
        current_subject = "u",
    ).url

    assert _launched_state(url, {"UNSLOTH_MCP_ARGUMENT_MARKER": "batch"}) == {
        "arguments": arguments,
        "marker": "batch",
    }
