# SPDX-License-Identifier: AGPL-3.0-only

import json
import socket
import subprocess
import sys
import textwrap
import time
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from integrations import editors
from integrations.blender import service as blender
from models.mcp_servers import (
    EditorSettings,
    EditorSetup,
    EditorTest,
    McpServerProbeResult,
    McpUiToolCallRequest,
)
from routes import mcp_servers as routes
from storage import mcp_servers_db as db

TOKEN = "ab" * 32

_SERVER = """
import sys, uvicorn
from fastmcp import FastMCP
from starlette.responses import JSONResponse

port, token, instances = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
mcp = FastMCP("editor")


@mcp.tool()
def create_node(name: str) -> str:
    return "created " + name


@mcp.custom_route("/api/instances", methods = ["GET"])
async def listed(_):
    return JSONResponse({"success": True, "instances": [{"project": "Demo"}] * instances})


app = mcp.http_app(path = "/mcp")


class RequireToken:
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        sent = dict(scope.get("headers") or []).get(b"x-mcp-capability-token", b"").decode()
        if scope["type"] == "http" and token != "-" and sent != token:
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b"Invalid capability token"})
            return
        await self.inner(scope, receive, send)


uvicorn.run(RequireToken(app), host = "127.0.0.1", port = port, log_level = "warning")
"""


@pytest.fixture(autouse = True)
def managed_policy(monkeypatch):
    for module in (routes, editors, blender):
        monkeypatch.setattr(module, "stdio_mcp_enabled", lambda: True)
    from core.inference import tools
    from core.inference.mcp_client import invalidate_tool_cache

    monkeypatch.setattr(tools, "stdio_mcp_enabled", lambda: True)
    invalidate_tool_cache()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def editor_server(tmp_path):
    pytest.importorskip("uvicorn")
    started = []

    def start(token = "-", instances = 1) -> int:
        port = _free_port()
        script = tmp_path / "editor_server.py"
        script.write_text(textwrap.dedent(_SERVER), encoding = "utf-8")
        proc = subprocess.Popen([sys.executable, str(script), str(port), token, str(instances)])
        started.append(proc)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    return port
            time.sleep(0.1)
        raise RuntimeError("editor server did not start")

    yield start
    for proc in started:
        proc.terminate()
        proc.wait(timeout = 10)


def test_catalog_keeps_blender_first_and_writes_nothing():
    items = routes.list_builtins()
    assert [item.builtin_id for item in items] == ["blender", "unity", "unreal", "godot"]
    assert items[0].port == 9876 and items[0].min_blender_version
    assert [item.port for item in items[1:]] == [8080, 3000, 9080]
    assert all(item.server_id is None and not item.is_enabled for item in items)
    assert db.list_servers() == []
    for auth in ({"via_api_key": True}, {"no_credential": True}):
        assert not any(item.available for item in routes.list_builtins(**auth))
    for fields in ({"port": "8080"}, {"port": 0}, {"project_path": "a\0b"}, {"command": "sh"}):
        with pytest.raises(ValidationError):
            EditorSettings(**fields)


@pytest.mark.asyncio
async def test_unknown_editor_and_missing_consent_have_no_side_effects(monkeypatch):
    probe = AsyncMock()
    monkeypatch.setattr(editors, "probe", probe)
    with pytest.raises(HTTPException) as unknown:
        await routes.setup_editor("roblox", EditorSetup(is_enabled = True, consent = True))
    assert unknown.value.status_code == 404
    with pytest.raises(HTTPException, match = "consent"):
        await routes.setup_editor("unity", EditorSetup(is_enabled = True))
    with pytest.raises(HTTPException, match = "consent"):
        await routes.test_editor("godot", EditorTest())
    for auth in ({"via_api_key": True}, {"no_credential": True}):
        with pytest.raises(HTTPException):
            await routes.setup_editor("unity", EditorSetup(is_enabled = False), **auth)
    monkeypatch.setattr(routes, "stdio_mcp_enabled", lambda: False)
    with pytest.raises(HTTPException):
        await routes.test_editor("unity", EditorTest(consent = True))
    assert db.list_servers() == []
    probe.assert_not_called()


@pytest.mark.asyncio
async def test_enable_is_idempotent_and_failures_leave_it_disabled(monkeypatch):
    probe = AsyncMock(return_value = McpServerProbeResult(ok = False, error = "Godot is not answering."))
    monkeypatch.setattr(editors, "probe", probe)
    with pytest.raises(HTTPException, match = "not answering"):
        await routes.setup_editor("godot", EditorSetup(is_enabled = True, consent = True, port = 9090))
    row = db.list_servers()[0]
    assert row["builtin_id"] == "godot" and not row["is_enabled"]
    assert row["url"] == "http://127.0.0.1:9090/mcp" and row["headers_json"] is None
    probe.return_value = McpServerProbeResult(ok = True, tool_count = 2)
    first = await routes.setup_editor("godot", EditorSetup(is_enabled = True, port = 9090))
    again = await routes.setup_editor("godot", EditorSetup(is_enabled = True, port = 9090))
    assert first.server_id == again.server_id == row["id"] and again.is_enabled
    assert len(db.list_servers()) == 1
    with db.get_connection() as conn:
        raw = dict(conn.execute("SELECT * FROM mcp_servers").fetchone())
    assert raw["url"] == "" and json.loads(raw["builtin_config_json"])["consent"] is True
    with pytest.raises(HTTPException):
        await routes.delete_mcp_server(row["id"])
    await routes.update_mcp_server(row["id"], routes.McpServerUpdate(is_enabled = False))
    assert not db.get_server(row["id"])["is_enabled"]


def test_unreal_token_is_read_from_the_project_on_every_resolve(tmp_path):
    project = tmp_path / "Demo"
    (project / "Saved" / "MCP").mkdir(parents = True)
    config = json.dumps({"port": 3000, "project_path": str(project / "Demo.uproject")})
    db.create_server("a" * 16, "Unreal Engine", "", builtin_id = "unreal", builtin_config_json = config)
    assert db.get_server("a" * 16)["headers_json"] is None
    (project / "Saved" / "MCP" / "capability-token").write_text(TOKEN + "\n", encoding = "utf-8")
    row = db.get_server_for_tool("unreal")
    assert row["url"] == "http://127.0.0.1:3000/mcp"
    assert json.loads(row["headers_json"]) == {"X-MCP-Capability-Token": TOKEN}
    assert routes._row_to_response(row).headers == {}
    (project / "Saved" / "MCP" / "capability-token").write_text("cd" * 32, encoding = "utf-8")
    assert (
        json.loads(db.get_server("a" * 16)["headers_json"])["X-MCP-Capability-Token"] == "cd" * 32
    )


def test_editor_tools_use_their_builtin_prefix_and_follow_the_stdio_gate(monkeypatch):
    from core.inference import tools

    db.create_server(
        "b" * 16, "Unity", "", is_enabled = True, builtin_id = "unity", builtin_config_json = "{}"
    )
    server = db.list_servers()[0]
    assert tools._mcp_tool_prefix(server) == "mcp__unity__"
    assert db.get_server_for_tool("unity")["id"] == "b" * 16
    monkeypatch.setattr(tools, "stdio_mcp_enabled", lambda: False)
    assert "disabled on this host" in tools.execute_tool("mcp__unity__create_node", {"name": "x"})


@pytest.mark.asyncio
async def test_real_probe_reports_mcp_and_editor_readiness(editor_server):
    port = editor_server(instances = 0)
    result = await editors.probe("unity", EditorSettings(port = port))
    assert result.ok and result.tool_count == 1 and result.editor_ready is False
    assert "Unity" in result.editor_error
    port = editor_server(instances = 1)
    assert (await editors.probe("unity", EditorSettings(port = port))).editor_ready is True
    godot = await editors.probe("godot", EditorSettings(port = port))
    assert godot.ok and godot.editor_ready is True
    closed = await editors.probe("godot", EditorSettings(port = _free_port()))
    assert not closed.ok and "Godot" in closed.error


@pytest.mark.asyncio
async def test_real_unreal_probe_needs_the_project_token(editor_server, tmp_path):
    port = editor_server(token = TOKEN)
    missing = await editors.probe("unreal", EditorSettings(port = port))
    assert not missing.ok and "project folder" in missing.error
    absent = await editors.probe("unreal", EditorSettings(port = port, project_path = str(tmp_path)))
    assert not absent.ok and "capability token" in absent.error
    (tmp_path / "Saved" / "MCP").mkdir(parents = True)
    (tmp_path / "Saved" / "MCP" / "capability-token").write_text("cd" * 32, encoding = "utf-8")
    wrong = await editors.probe("unreal", EditorSettings(port = port, project_path = str(tmp_path)))
    assert not wrong.ok
    (tmp_path / "Saved" / "MCP" / "capability-token").write_text(TOKEN, encoding = "utf-8")
    ok = await editors.probe("unreal", EditorSettings(port = port, project_path = str(tmp_path)))
    assert ok.ok and ok.tool_count == 1 and ok.editor_ready is True


@pytest.mark.asyncio
@pytest.mark.parametrize("via_api_key, stdio_on", [(True, True), (False, False)])
async def test_mcp_app_tool_calls_on_editor_rows_follow_the_stdio_gate(
    monkeypatch, via_api_key, stdio_on
):
    db.create_server(
        "d" * 16, "Unity", "", is_enabled = True, builtin_id = "unity", builtin_config_json = "{}"
    )
    monkeypatch.setattr(routes, "stdio_mcp_enabled", lambda: stdio_on)
    monkeypatch.setattr(routes, "_warm_tool_cache", AsyncMock())
    from core.inference import tools

    monkeypatch.setattr(tools, "mcp_tool_definition", lambda sid, name: {"name": name})
    call = Mock(
        return_value = {"content": [], "structured_content": None, "is_error": False, "meta": None}
    )
    monkeypatch.setattr(routes, "call_tool_structured_sync", call)
    payload = McpUiToolCallRequest(
        tool_name = "execute_code", arguments = {"code": "x"}, permission_mode = "full"
    )
    with pytest.raises(HTTPException):
        await routes.call_mcp_ui_tool("d" * 16, payload, via_api_key = via_api_key)
    call.assert_not_called()
