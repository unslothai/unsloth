# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""stdio MCP runs a local command as the backend user, outside the sandbox, and
sk-unsloth API keys authenticate the same routes the UI uses. So these pin that
only a UI session may define a command, and that http(s) MCP stays usable from an
API key. list_tools_async is stubbed to fail if called, so a gate placed after the
probe would not pass.
"""

import asyncio
import os

import pytest
from fastapi import HTTPException

from core.inference import mcp_client
from storage import mcp_servers_db
from utils import host_policy


def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    mcp_client.invalidate_tool_cache()


@pytest.fixture(autouse = True)
def _isolate_stdio_env():
    """apply_stdio_mcp_loopback_default() mutates os.environ plus a module flag
    monkeypatch cannot roll back; snapshot and restore both."""
    saved = os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP")
    host_policy._reset_loopback_default_state()
    yield
    host_policy._reset_loopback_default_state()
    if saved is None:
        os.environ.pop("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", None)
    else:
        os.environ["UNSLOTH_STUDIO_ALLOW_STDIO_MCP"] = saved


@pytest.fixture
def stdio_on(monkeypatch):
    """The worst case for this gate: a host where stdio is allowed."""
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")


@pytest.fixture
def no_probe(monkeypatch):
    """Fail loudly if a refused request still reached the transport."""
    import routes.mcp_servers as routes_mcp

    async def _never(**kwargs):
        raise AssertionError("the stdio command must not be probed")

    monkeypatch.setattr(routes_mcp, "list_tools_async", _never)


STDIO_CMD = "/bin/sh -c id"


# ── stdio form codec: command material is UI-session-only ──────────


@pytest.mark.parametrize("operation", ["encode", "decode"])
def test_stdio_command_codec_refuses_api_key_before_work(
    monkeypatch, stdio_on, no_probe, operation
):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpStdioCommand, McpStdioDecodeRequest

    def _never(*args, **kwargs):
        raise AssertionError("refused codec request must not access storage")

    monkeypatch.setattr(mcp_servers_db, "list_servers", _never)
    with pytest.raises(HTTPException) as exc:
        if operation == "encode":
            routes_mcp.encode_stdio_command(
                McpStdioCommand(command = "python", arguments = ["--token", "secret"]),
                current_subject = "api-key-user",
                via_api_key = True,
            )
        else:
            routes_mcp.decode_stdio_command(
                McpStdioDecodeRequest(url = "python --token secret"),
                current_subject = "api-key-user",
                via_api_key = True,
            )
    assert exc.value.status_code == 403


# ── /test: an unstored, caller-supplied command ─────────────────────


def test_test_endpoint_refuses_stdio_from_api_key(tmp_path, monkeypatch, stdio_on, no_probe):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerTestRequest

    _reset_db(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.test_mcp_server(
                McpServerTestRequest(url = STDIO_CMD),
                current_subject = "api-key-user",
                via_api_key = True,
            )
        )
    assert exc.value.status_code == 403


def test_test_endpoint_allows_http_from_api_key(tmp_path, monkeypatch, stdio_on):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerTestRequest

    _reset_db(tmp_path, monkeypatch)
    seen = {}

    async def _probe(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(routes_mcp, "list_tools_async", _probe)
    res = asyncio.run(
        routes_mcp.test_mcp_server(
            McpServerTestRequest(url = "https://example.com/mcp"),
            current_subject = "api-key-user",
            via_api_key = True,
        )
    )
    assert res.ok is True
    assert seen["url"] == "https://example.com/mcp"


# ── create / update ─────────────────────────────────────────────────


def test_create_refuses_stdio_from_api_key_and_writes_nothing(tmp_path, monkeypatch, stdio_on):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerCreate

    _reset_db(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.create_mcp_server(
                McpServerCreate(display_name = "Local", url = STDIO_CMD),
                current_subject = "api-key-user",
                via_api_key = True,
            )
        )
    assert exc.value.status_code == 403
    assert mcp_servers_db.list_servers() == []


def test_create_allows_http_from_api_key(tmp_path, monkeypatch, stdio_on):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerCreate

    _reset_db(tmp_path, monkeypatch)
    resp = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(display_name = "Remote", url = "https://example.com/mcp"),
            current_subject = "api-key-user",
            via_api_key = True,
        )
    )
    assert resp.url == "https://example.com/mcp"


@pytest.mark.parametrize(
    "payload_kwargs",
    [
        {"headers": {"Authorization": "Bearer caller-secret"}},
        {"use_oauth": True},
        {"url": "https://example.com/mcp?access_token=caller-secret"},
    ],
)
def test_create_refuses_persisted_http_credentials_from_api_key(
    tmp_path, monkeypatch, stdio_on, payload_kwargs
):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerCreate

    _reset_db(tmp_path, monkeypatch)
    values = {"display_name": "Remote", "url": "https://example.com/mcp"}
    values.update(payload_kwargs)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.create_mcp_server(
                McpServerCreate(**values),
                current_subject = "api-key-user",
                via_api_key = True,
            )
        )
    assert exc.value.status_code == 403
    assert mcp_servers_db.list_servers() == []


def test_update_refuses_http_to_stdio_conversion_from_api_key(tmp_path, monkeypatch, stdio_on):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerUpdate

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://a/mcp")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.update_mcp_server(
                "s1",
                McpServerUpdate(url = STDIO_CMD),
                current_subject = "api-key-user",
                via_api_key = True,
            )
        )
    assert exc.value.status_code == 403
    assert mcp_servers_db.get_server("s1")["url"] == "https://a/mcp"


@pytest.mark.parametrize(
    "payload_kwargs",
    [
        {"display_name": "Renamed"},
        {"is_enabled": True},
        {"headers": {"SECRET": "x"}},
    ],
)
def test_update_refuses_any_edit_of_a_stdio_row_from_api_key(
    tmp_path, monkeypatch, stdio_on, payload_kwargs
):
    """Not just the address: the env vars, the name and the enabled flag all
    change what runs or how, so an API key may not touch a stdio row at all."""
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerUpdate

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "s1", display_name = "Local", url = STDIO_CMD)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.update_mcp_server(
                "s1",
                McpServerUpdate(**payload_kwargs),
                current_subject = "api-key-user",
                via_api_key = True,
            )
        )
    assert exc.value.status_code == 403
    assert mcp_servers_db.get_server("s1")["display_name"] == "Local"


def test_update_regates_after_the_oauth_clear_await(tmp_path, monkeypatch, stdio_on):
    """clear_oauth_tokens_async awaits, handing the loop to other requests. If
    the owner converts the row to stdio in that window, the write that follows
    must not land the API key's headers as the command's env."""
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerUpdate

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(
        id = "s1", display_name = "Remote", url = "https://a/mcp", is_enabled = True, use_oauth = True
    )

    async def _clear_then_convert(url):
        await asyncio.sleep(0)
        mcp_servers_db.update_server("s1", {"url": STDIO_CMD, "use_oauth": False})

    monkeypatch.setattr(routes_mcp, "clear_oauth_tokens_async", _clear_then_convert)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.update_mcp_server(
                "s1",
                McpServerUpdate(headers = {"LD_PRELOAD": "/tmp/evil.so"}, use_oauth = False),
                current_subject = "api-key-user",
                via_api_key = True,
            )
        )
    assert exc.value.status_code == 403
    assert mcp_servers_db.get_server("s1")["headers_json"] is None


def test_update_refuses_secret_bearing_http_row_from_api_key(tmp_path, monkeypatch, stdio_on):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerUpdate

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(
        id = "s1",
        display_name = "A",
        url = "https://a/mcp",
        headers_json = '{"Authorization": "Bearer t"}',
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.update_mcp_server(
                "s1",
                McpServerUpdate(display_name = "B"),
                current_subject = "api-key-user",
                via_api_key = True,
            )
        )
    assert exc.value.status_code == 403
    assert mcp_servers_db.get_server("s1")["display_name"] == "A"


# ── refresh ─────────────────────────────────────────────────────────


def test_refresh_refuses_stored_stdio_from_api_key(tmp_path, monkeypatch, stdio_on, no_probe):
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "stdio1", display_name = "Local", url = STDIO_CMD, is_enabled = True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.refresh_mcp_server_tools(
                "stdio1", current_subject = "api-key-user", via_api_key = True
            )
        )
    assert exc.value.status_code == 403


def test_refresh_allows_http_from_api_key(tmp_path, monkeypatch, stdio_on):
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://a/mcp")

    async def _probe(**kwargs):
        return []

    monkeypatch.setattr(routes_mcp, "list_tools_async", _probe)
    res = asyncio.run(
        routes_mcp.refresh_mcp_server_tools("s1", current_subject = "api-key-user", via_api_key = True)
    )
    assert res.ok is True


def test_refresh_refuses_saved_http_headers_from_api_key(tmp_path, monkeypatch, stdio_on, no_probe):
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(
        id = "s1",
        display_name = "A",
        url = "https://a/mcp",
        headers_json = '{"Authorization":"Bearer installation-secret"}',
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.refresh_mcp_server_tools(
                "s1", current_subject = "api-key-user", via_api_key = True
            )
        )
    assert exc.value.status_code == 403


def test_http_mcp_dispatch_rechecks_approved_configuration(monkeypatch):
    entered = False

    def _never_client(*_args, **_kwargs):
        nonlocal entered
        entered = True
        raise AssertionError("HTTP connector must not open after approval drift")

    monkeypatch.setattr(mcp_client, "_client", _never_client)
    result = mcp_client.call_tool_sync(
        url = "https://example.com/mcp",
        headers = {"Authorization": "Bearer installation-secret"},
        name = "create_pull_request",
        args = {},
        config_check = lambda: False,
    )
    assert result.startswith("Error:")
    assert entered is False


# ── import: per-entry, so a mixed config still lands its http rows ──


_MIXED_CONFIG = {
    "mcpServers": {
        "remote": {"url": "https://example.com/mcp"},
        "local": {"command": "/bin/sh", "args": ["-c", "id"]},
    }
}


def test_import_from_api_key_keeps_http_and_reports_stdio(tmp_path, monkeypatch, stdio_on):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerImportRequest

    _reset_db(tmp_path, monkeypatch)
    res = asyncio.run(
        routes_mcp.import_mcp_servers(
            McpServerImportRequest(config = _MIXED_CONFIG),
            current_subject = "api-key-user",
            via_api_key = True,
        )
    )
    assert [s.display_name for s in res.created] == ["remote"]
    assert any("local" in err for err in res.errors)
    assert [row["url"] for row in mcp_servers_db.list_servers()] == ["https://example.com/mcp"]

    # Re-importing the same config is idempotent: the http entry is now a skip,
    # the stdio entry is still an error, and no row is duplicated.
    again = asyncio.run(
        routes_mcp.import_mcp_servers(
            McpServerImportRequest(config = _MIXED_CONFIG),
            current_subject = "api-key-user",
            via_api_key = True,
        )
    )
    assert again.created == []
    assert again.skipped == ["remote"]
    assert any("local" in err for err in again.errors)
    assert len(mcp_servers_db.list_servers()) == 1


# ── a UI session keeps every existing stdio behaviour ───────────────


def test_ui_session_still_creates_and_imports_stdio(tmp_path, monkeypatch, stdio_on):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerCreate, McpServerImportRequest, McpServerUpdate

    _reset_db(tmp_path, monkeypatch)
    # A command distinct from the one in _MIXED_CONFIG, so the import below is a
    # real create rather than the url-dedupe skip.
    own_cmd = "/bin/echo hello"
    created = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(display_name = "Local", url = own_cmd),
            current_subject = "owner",
            via_api_key = False,
        )
    )
    assert created.url == own_cmd
    renamed = asyncio.run(
        routes_mcp.update_mcp_server(
            created.id,
            McpServerUpdate(display_name = "Local FS"),
            current_subject = "owner",
            via_api_key = False,
        )
    )
    assert renamed.display_name == "Local FS"
    res = asyncio.run(
        routes_mcp.import_mcp_servers(
            McpServerImportRequest(config = _MIXED_CONFIG),
            current_subject = "owner",
            via_api_key = False,
        )
    )
    assert res.errors == []
    assert sorted(s.display_name for s in res.created) == ["local", "remote"]


def test_default_is_ui_session_so_direct_calls_are_unaffected():
    """The dependency is Annotated with a plain False default; a bare
    `= Depends(...)` default would be a truthy object and 403 every direct
    call (the existing suites call these handlers directly)."""
    import inspect

    import routes.mcp_servers as routes_mcp

    for name in (
        "create_mcp_server",
        "update_mcp_server",
        "refresh_mcp_server_tools",
        "import_mcp_servers",
        "test_mcp_server",
        "decode_stdio_command",
        "encode_stdio_command",
    ):
        param = inspect.signature(getattr(routes_mcp, name)).parameters["via_api_key"]
        assert param.default is False, name


# ── data recipe: the same primitive, same gate ──────────────────────


_STDIO_PROVIDER = {
    "name": "local",
    "provider_type": "stdio",
    "command": "/bin/sh",
    "args": ["-c", "id"],
}
_HTTP_PROVIDER = {"name": "remote", "provider_type": "http", "url": "https://example.com/mcp"}
_STDIO_RECIPE = {"columns": [{"name": "a"}], "mcp_providers": [_STDIO_PROVIDER]}


def test_recipe_has_stdio_mcp():
    from core.data_recipe.service import recipe_has_stdio_mcp

    assert recipe_has_stdio_mcp({"mcp_providers": [_STDIO_PROVIDER]}) is True
    assert recipe_has_stdio_mcp({"mcp_providers": [_HTTP_PROVIDER]}) is False
    assert recipe_has_stdio_mcp({}) is False
    assert recipe_has_stdio_mcp({"mcp_providers": "nonsense"}) is False


def test_data_recipe_mcp_tools_refuses_stdio_from_api_key():
    from models.data_recipe import McpToolsListRequest
    from routes.data_recipe.mcp import list_mcp_tools

    with pytest.raises(HTTPException) as exc:
        list_mcp_tools(McpToolsListRequest(mcp_providers = [_STDIO_PROVIDER]), via_api_key = True)
    assert exc.value.status_code == 403


def test_data_recipe_job_refuses_stdio_recipe_from_api_key():
    from models.data_recipe import RecipePayload
    from routes.data_recipe.jobs import create_job

    payload = RecipePayload(recipe = _STDIO_RECIPE)
    with pytest.raises(HTTPException) as exc:
        create_job(payload, request = None, credential = ("u", None), via_api_key = True)
    assert exc.value.status_code == 403


def test_data_recipe_validate_refuses_stdio_recipe_from_api_key():
    from models.data_recipe import RecipePayload
    from routes.data_recipe.validate import validate

    payload = RecipePayload(recipe = _STDIO_RECIPE)
    with pytest.raises(HTTPException) as exc:
        validate(payload, via_api_key = True)
    assert exc.value.status_code == 403


# ── reading back a command the gate would not let a key define ──────


def test_list_hides_stdio_rows_from_api_keys(tmp_path, monkeypatch, stdio_on):
    """A key that may not define a command may not read one back: `url` is the
    argv (carries credentials) and `headers` is the subprocess env."""
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(
        id = "stdio1",
        display_name = "FS",
        url = "npx server --token sk-argv-secret",
        headers_json = '{"API_KEY": "sk-env-secret"}',
    )
    mcp_servers_db.create_server(
        id = "http1",
        display_name = "R",
        url = "https://example.com/mcp",
        headers_json = '{"Authorization": "Bearer t"}',
    )

    keyed = routes_mcp.list_mcp_servers(current_subject = "u", via_api_key = True)
    assert [row.id for row in keyed] == ["http1"]
    serialized = repr([row.model_dump() for row in keyed])
    assert "sk-argv-secret" not in serialized
    assert "sk-env-secret" not in serialized
    assert keyed[0].headers == {}

    keyless = routes_mcp.list_mcp_servers(
        current_subject = "u", via_api_key = False, no_credential = True
    )
    assert [row.id for row in keyless] == ["http1"]
    assert keyless[0].headers == {}


def test_list_shows_stdio_rows_to_a_ui_session(tmp_path, monkeypatch, stdio_on):
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(
        id = "stdio1",
        display_name = "FS",
        url = "npx server --token sk-argv-secret",
        headers_json = '{"API_KEY": "sk-env-secret"}',
    )
    rows = routes_mcp.list_mcp_servers(current_subject = "u", via_api_key = False)
    assert [row.id for row in rows] == ["stdio1"]
    assert rows[0].url == "npx server --token sk-argv-secret"
    assert rows[0].headers == {"API_KEY": "sk-env-secret"}


def test_studio_mcp_surface_refuses_stdio_recipes(monkeypatch):
    """mcp_server.py calls the validate route function directly, so the ViaApiKey
    dependency never runs and its `= False` default would read as a UI session.
    That remote static bearer surface must pass True itself or the gate is dead."""
    import importlib
    import sys
    from types import ModuleType

    class FakeFastMCP:
        def __init__(self, *_args, **_kwargs):
            self.tools = {}

        def tool(self, function):
            self.tools[function.__name__] = function
            return function

    fastmcp = ModuleType("fastmcp")
    fastmcp.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp)
    sys.modules.pop("mcp_server", None)

    mcp_server = importlib.import_module("mcp_server")
    server = mcp_server.create_studio_mcp()
    seen = {}

    class RecipePayload:
        def __init__(self, *, recipe):
            self.recipe = recipe

    data_recipe_model = ModuleType("models.data_recipe")
    data_recipe_model.RecipePayload = RecipePayload
    validate_route = ModuleType("routes.data_recipe.validate")

    def validate(payload, *, via_api_key = False):
        seen.update(payload = payload, via_api_key = via_api_key)
        return {"valid": True}

    validate_route.validate = validate
    monkeypatch.setitem(sys.modules, "models.data_recipe", data_recipe_model)
    monkeypatch.setitem(sys.modules, "routes.data_recipe.validate", validate_route)

    result = server.tools["validate_recipe"]({"steps": []})

    assert result == {"valid": True}
    assert seen["via_api_key"] is True
    assert seen["payload"].recipe == {"steps": []}
