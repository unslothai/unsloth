# SPDX-License-Identifier: AGPL-3.0-only

import json
import sqlite3
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from integrations.blender import service
from models.mcp_servers import BlenderTest
from models.mcp_servers import BlenderSettings, BlenderSetup, McpServerProbeResult, McpServerUpdate
from routes import mcp_servers as routes
from storage import mcp_servers_db as db


@pytest.fixture(autouse = True)
def managed_policy(monkeypatch):
    monkeypatch.setattr(routes, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(service, "stdio_mcp_enabled", lambda: True)
    from core.inference.mcp_client import invalidate_tool_cache

    invalidate_tool_cache()


def test_catalog_is_opt_in_and_settings_are_typed(monkeypatch):
    probe = AsyncMock()
    monkeypatch.setattr(service, "probe", probe)
    item = routes.list_builtins()[0]
    assert item.server_id is None and not item.is_enabled and item.port == 9876
    assert db.list_servers() == []
    probe.assert_not_called()
    for fields in (
        {"port": True},
        {"port": "9876"},
        {"port": 0},
        {"port": 65536},
        {"blender_path": "x\0y"},
        {"command": "sh"},
    ):
        with pytest.raises(ValidationError):
            BlenderSettings(**fields)


@pytest.mark.asyncio
async def test_managed_enable_failure_retry_disable_and_portability(monkeypatch):
    probe = AsyncMock(return_value = McpServerProbeResult(ok = False, error = "Blender unavailable"))
    monkeypatch.setattr(service, "probe", probe)
    with pytest.raises(HTTPException):
        await routes.setup_blender(BlenderSetup(is_enabled = True))
    assert db.list_servers() == []
    with pytest.raises(HTTPException):
        await routes.setup_blender(BlenderSetup(is_enabled = True, consent = True, port = 9877))
    row = db.list_servers()[0]
    assert not row["is_enabled"]
    assert json.loads(row["headers_json"])["BLENDER_MCP_HOST"] == "127.0.0.1"
    with db.get_connection() as conn:
        raw = dict(conn.execute("SELECT * FROM mcp_servers").fetchone())
    assert raw["url"] == "" and raw["headers_json"] is None
    assert json.loads(raw["builtin_config_json"])["port"] == 9877
    with pytest.raises(sqlite3.IntegrityError):
        db.create_server("other", "Blender", "", builtin_id = "blender")
    probe.return_value = McpServerProbeResult(ok = True, tool_count = 3)
    enabled = await routes.setup_blender(BlenderSetup(is_enabled = True, port = 9877))
    assert enabled.server_id == row["id"] and enabled.is_enabled
    for update in (
        McpServerUpdate(is_enabled = True),
        McpServerUpdate(url = "python"),
        McpServerUpdate(headers = {"X": "y"}),
    ):
        with pytest.raises(HTTPException):
            await routes.update_mcp_server(row["id"], update)
    with pytest.raises(HTTPException):
        await routes.delete_mcp_server(row["id"])
    monkeypatch.setattr(routes, "stdio_mcp_enabled", lambda: False)
    await routes.update_mcp_server(row["id"], McpServerUpdate(is_enabled = False))
    assert not db.get_server(row["id"])["is_enabled"]


@pytest.mark.asyncio
async def test_managed_authorization_precedes_side_effects(monkeypatch):
    probe = AsyncMock()
    monkeypatch.setattr(service, "probe", probe)
    for auth in ({"via_api_key": True}, {"no_credential": True}):
        with pytest.raises(HTTPException):
            await routes.setup_blender(BlenderSetup(is_enabled = False), **auth)
        with pytest.raises(HTTPException):
            await routes.test_blender(BlenderSettings(), **auth)
    monkeypatch.setattr(routes, "stdio_mcp_enabled", lambda: False)
    with pytest.raises(HTTPException):
        await routes.test_blender(BlenderSettings())
    assert db.list_servers() == []
    probe.assert_not_called()
    saved = await routes.setup_blender(BlenderSetup(is_enabled = False, port = 9878))
    assert not saved.is_enabled and saved.port == 9878
    for auth in ({"via_api_key": True}, {"no_credential": True}):
        item = routes.list_builtins(**auth)[0]
        assert not item.available and item.server_id is None and item.port == 9876
    probe.assert_not_called()


@pytest.mark.asyncio
async def test_first_probe_requires_consent_before_setup(monkeypatch):
    probe = AsyncMock(return_value = McpServerProbeResult(ok = True))
    monkeypatch.setattr(service, "probe", probe)
    with pytest.raises(HTTPException, match = "consent"):
        await routes.test_blender(BlenderTest())
    probe.assert_not_called()
    await routes.test_blender(BlenderTest(consent = True))
    probe.assert_awaited_once()


@pytest.mark.asyncio
async def test_probe_separates_mcp_and_blender_readiness(monkeypatch):
    monkeypatch.setattr(service, "ensure_runtime", lambda: None)
    bridge = AsyncMock(side_effect = ConnectionError())
    tools = AsyncMock(return_value = [{"name": "execute_blender_code"}])
    monkeypatch.setattr(service, "_bridge_version", bridge)
    monkeypatch.setattr(service, "list_tools_async", tools)
    from core.inference import mcp_client

    result = await service.probe(BlenderSettings())
    assert result.ok and result.tool_count == 1 and result.blender_ready is False
    assert result.blender_error
    bridge.assert_awaited_with(9876)
    bridge.reset_mock()
    assert (await service.probe(BlenderSettings(), check_bridge = False)).ok
    bridge.assert_not_called()
    bridge.side_effect = None
    assert (await service.probe(BlenderSettings())).blender_ready is True
    enabled = await routes.setup_blender(BlenderSetup(is_enabled = True, consent = True))
    assert mcp_client.get_cached_tools(enabled.server_id) == tools.return_value
    mcp_client.invalidate_tool_cache(enabled.server_id)
    await routes.test_blender(BlenderSettings(port = 9877))
    assert mcp_client.get_cached_tools(enabled.server_id) is None
    await routes.test_blender(BlenderSettings())
    assert mcp_client.get_cached_tools(enabled.server_id) == tools.return_value


def test_long_blender_names_dispatch_with_approval(monkeypatch):
    from core.inference import tools
    from unittest.mock import Mock

    db.create_server("0123456789abcdef", "Blender", "", builtin_id = "blender")
    monkeypatch.setattr(tools, "stdio_mcp_enabled", lambda: True)
    call = Mock(return_value = "ok")
    monkeypatch.setattr(tools, "call_tool_sync", call)
    names = sorted(tools._BLENDER_CLI_SUMMARY_TOOLS)
    specs = tools._mcp_specs_for_server(db.list_servers()[0], [{"name": n} for n in names])
    assert len(specs) == len(names)
    for raw, spec in zip(names, specs):
        name = spec["function"]["name"]
        assert len(name) <= 64
        assert tools.is_potentially_unsafe_tool_call(name, {})
        assert tools.is_high_risk_tool_call(name, {})
        tools.execute_tool(name, {})
        assert call.call_args.kwargs["name"] == raw
        assert call.call_args.kwargs["config_check"]()
    db.update_server("0123456789abcdef", {"is_enabled": False})
    call.reset_mock()
    assert "disabled" in tools.execute_tool(specs[0]["function"]["name"], {})
    call.assert_not_called()
