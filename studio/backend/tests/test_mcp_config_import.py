"""join_stdio_command and parse_stdio_command must round-trip on posix and win32 fixtures."""

import sys

import pytest

from core.inference import mcp_client
from core.inference.mcp_config_import import parse_mcp_config
from storage import mcp_servers_db


def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())


def _enable(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")


def _disable(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", raising = False)


@pytest.mark.parametrize(
    "parts",
    [
        ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ["python", "-m", "mod", "--name", "a b"],
        ["uvx", "some-server", "--flag"],
        ["/usr/local/bin/my-server"],
        ["mcp-server-sqlite"],
        [
            "python",
            "--url",
            "https://example.com/value",
            "a b",
            "O'Reilly",
            'say "hello"',
            "trailing\\",
            "",
        ],
    ],
)
def test_join_parse_roundtrip_posix(monkeypatch, parts):
    monkeypatch.setattr(sys, "platform", "linux")
    joined = mcp_client.join_stdio_command(parts)
    assert mcp_client.parse_stdio_command(joined) == parts


@pytest.mark.parametrize(
    "parts",
    [
        [
            "C:\\Users\\user\\Documents\\Office-Word-MCP-Server\\.venv\\Scripts\\python.exe",
            "C:\\Users\\user\\Documents\\Office-Word-MCP-Server\\word_mcp_server.py",
        ],
        [
            "node",
            "C:\\Users\\user\\Documents\\DesktopCommanderMCP\\dist\\index.js",
            "--no-onboarding",
        ],
        [
            "node",
            "C:\\Users\\user\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-filesystem\\dist\\index.js",
            "D:\\",
            "O:\\",
        ],
        ["C:\\Program Files\\node\\node.exe", "server.js"],
        ["C:\\Program Files\\Foo\\", "server.js"],
        ["C:\\Program Files\\Foo\\", '{"foo":"bar"}'],
        ["'C:\\Program Files\\node\\node.exe'", "server.js"],
        ["node", "O'Reilly"],
        ["node", "C:\\Users\\O'Reilly\\server.js"],
        ["node", "'draft'"],
        ["node", "'open", "close'"],
        ["node", ""],
        [
            "python",
            "--url",
            "https://example.com/value",
            "a b",
            'say "hello"',
            "C:\\path with spaces\\trailing\\",
            "",
        ],
    ],
)
def test_join_parse_roundtrip_win32(monkeypatch, parts):
    monkeypatch.setattr(sys, "platform", "win32")
    joined = mcp_client.join_stdio_command(parts)
    assert mcp_client.parse_stdio_command(joined) == parts


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("a\u00a0b", id = "nbsp"),
        pytest.param("a\nb", id = "newline"),
        pytest.param("a\rb", id = "carriage-return"),
        pytest.param("a\vb", id = "vertical-tab"),
        pytest.param("a\fb", id = "form-feed"),
        pytest.param("a\u2003b", id = "em-space"),
    ],
)
def test_join_parse_roundtrip_win32_preserves_non_delimiter_whitespace(monkeypatch, value):
    monkeypatch.setattr(sys, "platform", "win32")
    parts = ["python", "--value", value, 'say "hello"', "a b", ""]
    joined = mcp_client.join_stdio_command(parts)
    assert mcp_client.parse_stdio_command(joined) == parts


def test_parse_rejects_manual_single_quoted_windows_executable(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    command = "'C:\\Program Files\\node\\node.exe' server.js"
    with pytest.raises(ValueError):
        mcp_client.parse_stdio_command(command)


def test_parse_windows_apostrophes_as_literals(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert mcp_client.parse_stdio_command("node O'Reilly") == ["node", "O'Reilly"]
    assert mcp_client.parse_stdio_command("node C:\\Users\\O'Reilly\\server.js") == [
        "node",
        "C:\\Users\\O'Reilly\\server.js",
    ]
    assert mcp_client.parse_stdio_command("node 'draft'") == ["node", "'draft'"]
    assert mcp_client.parse_stdio_command("node 'open close'") == ["node", "'open", "close'"]


def test_parse_rejects_unterminated_windows_double_quote(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(ValueError):
        mcp_client.parse_stdio_command('node "C:\\path with spaces')


def test_parse_stdio_entry():
    cfg = {
        "mcpServers": {
            "fs": {
                "command": "npx",
                "args": ["-y", "server", "/tmp"],
                "env": {"K": "v"},
            }
        }
    }
    entries, errors = parse_mcp_config(cfg)
    assert errors == []
    assert len(entries) == 1
    entry = entries[0]
    assert entry.display_name == "fs"
    assert entry.is_stdio is True
    assert entry.headers == {"K": "v"}
    assert mcp_client.parse_stdio_command(entry.url) == ["npx", "-y", "server", "/tmp"]


@pytest.mark.parametrize("args", [None, "", 0, False, {}])
def test_parse_rejects_explicit_non_list_args(args):
    entries, errors = parse_mcp_config({"mcpServers": {"bad": {"command": "python", "args": args}}})

    assert entries == []
    assert errors == ["bad: 'args' must be a list of strings."]


def test_parse_remote_entry():
    cfg = {
        "mcpServers": {
            "remote": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer x"},
            }
        }
    }
    entries, errors = parse_mcp_config(cfg)
    assert errors == []
    assert entries[0].url == "https://example.com/mcp"
    assert entries[0].is_stdio is False
    assert entries[0].headers == {"Authorization": "Bearer x"}


def test_parse_preserves_disabled_and_oauth():
    cfg = {
        "servers": {
            "remote": {
                "type": "http",
                "url": "https://example.com/mcp",
                "oauth": {"clientId": "client"},
                "disabled": True,
            }
        }
    }
    entries, errors = parse_mcp_config(cfg)
    assert errors == []
    assert entries[0].is_enabled is False
    assert entries[0].use_oauth is True


def test_parse_accepts_cline_streamable_http_alias():
    cfg = {
        "mcpServers": {
            "remote": {
                "type": "streamableHttp",
                "url": "https://example.com/mcp",
            }
        }
    }
    entries, errors = parse_mcp_config(cfg)
    assert errors == []
    assert entries[0].url == "https://example.com/mcp"
    assert entries[0].is_stdio is False


@pytest.mark.parametrize(
    "server",
    [
        {"command": "node", "args": ["server.js"], "cwd": "/tmp/server"},
        {"command": "node", "args": ["server.js"], "envFile": ".env"},
        {"command": "node", "args": ["server.js"], "env": {"API_KEY": "${input:api-key}"}},
        {"command": "node", "args": ["${workspaceFolder}/server.js"]},
        {"command": "node", "args": ["server.js"], "env": {"HTTP_PROXY": None}},
        {"command": "node", "args": ["server.js"], "sandboxEnabled": True},
        {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${input:token}"}},
        {"url": "https://example.com/mcp", "headers": {"Authorization": None}},
        {"type": "http", "url": "https://example.com/sse"},
        {"type": "http", "url": "https://example.com/sse "},
        {"type": "streamableHttp", "url": "https://example.com/sse"},
        {"url": "https://example.com/mcp", "timeout": 120},
        {"url": "https://example.com/mcp", "timeoutMs": 120000},
        {"url": "https://example.com/mcp", "timeoutSeconds": 120},
        {"type": "sse", "url": "https://example.com/custom"},
    ],
)
def test_parse_rejects_unrepresentable_imports(server):
    entries, errors = parse_mcp_config({"servers": {"bad": server}})
    assert entries == []
    assert len(errors) == 1


def test_servers_alias_key():
    cfg = {"servers": {"fs": {"command": "node", "args": ["x.js"]}}}
    entries, errors = parse_mcp_config(cfg)
    assert errors == []
    assert len(entries) == 1


def test_env_and_args_values_coerced_to_str():
    cfg = {"mcpServers": {"fs": {"command": "node", "args": [8080], "env": {"PORT": 8080}}}}
    entries, errors = parse_mcp_config(cfg)
    assert errors == []
    assert entries[0].headers == {"PORT": "8080"}
    assert mcp_client.parse_stdio_command(entries[0].url) == ["node", "8080"]


def test_args_optional():
    cfg = {"mcpServers": {"sqlite": {"command": "mcp-server-sqlite"}}}
    entries, errors = parse_mcp_config(cfg)
    assert errors == []
    assert entries[0].url == "mcp-server-sqlite"
    assert entries[0].headers is None


def test_bad_entry_does_not_sink_batch():
    cfg = {
        "mcpServers": {
            "good": {"command": "node", "args": ["x.js"]},
            "both": {"command": "node", "url": "https://x/mcp"},
            "neither": {"name": "oops"},
            "bad_args": {"command": "node", "args": "x.js"},
            "bad_env": {"command": "node", "env": ["NOT", "A", "DICT"]},
        }
    }
    entries, errors = parse_mcp_config(cfg)
    assert {e.display_name for e in entries} == {"good"}
    assert len(errors) == 4


def test_not_a_dict():
    entries, errors = parse_mcp_config([])
    assert entries == []
    assert len(errors) == 1


def test_missing_servers_key():
    entries, errors = parse_mcp_config({"foo": {}})
    assert entries == []
    assert len(errors) == 1


def test_servers_alias_error_names_actual_key():
    entries, errors = parse_mcp_config({"servers": []})
    assert entries == []
    assert errors == ["'servers' must be an object mapping name -> server."]


def test_import_route_creates_and_dedups(tmp_path, monkeypatch):
    import asyncio

    from models.mcp_servers import McpServerImportRequest
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    cfg = {
        "mcpServers": {
            "fs": {
                "command": "npx",
                "args": ["-y", "server", "/tmp"],
                "env": {"API_KEY": "sk"},
            },
            "remote": {"url": "https://example.com/mcp"},
            "oauth": {
                "type": "http",
                "url": "https://auth.example.com/mcp",
                "oauth": {"clientId": "client"},
            },
            "disabled": {
                "url": "https://disabled.example.com/mcp",
                "disabled": True,
            },
        }
    }
    res = asyncio.run(
        routes_mcp.import_mcp_servers(McpServerImportRequest(config = cfg), current_subject = "u")
    )
    assert res.errors == []
    assert res.skipped == []
    assert {c.display_name for c in res.created} == {"fs", "remote", "oauth", "disabled"}
    fs = next(c for c in res.created if c.display_name == "fs")
    assert fs.headers == {"API_KEY": "sk"}
    assert fs.use_oauth is False
    assert fs.is_enabled is True
    oauth = next(c for c in res.created if c.display_name == "oauth")
    assert oauth.use_oauth is True
    disabled = next(c for c in res.created if c.display_name == "disabled")
    assert disabled.is_enabled is False

    res2 = asyncio.run(
        routes_mcp.import_mcp_servers(McpServerImportRequest(config = cfg), current_subject = "u")
    )
    assert res2.created == []
    assert set(res2.skipped) == {"fs", "remote", "oauth", "disabled"}


def _import(config, no_credential = False):
    import asyncio

    from models.mcp_servers import McpServerImportRequest
    import routes.mcp_servers as routes_mcp

    return asyncio.run(
        routes_mcp.import_mcp_servers(
            McpServerImportRequest(config = config),
            current_subject = "u",
            no_credential = no_credential,
        )
    )


def _names(servers):
    return sorted(s.display_name for s in servers)


GITHUB_URL = "https://api.githubcopilot.com/mcp/"


def _bearer(token):
    return {"url": GITHUB_URL, "headers": {"Authorization": f"Bearer {token}"}}


def test_import_keeps_named_connections_sharing_a_url(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    cfg = {
        "mcpServers": {
            "github-work": _bearer("WORK"),
            "github-personal": _bearer("PERSONAL"),
            "github-copy": _bearer("WORK"),
            "fs-a": {"command": "npx", "args": ["server"], "env": {"ROOT": "/a"}},
            "fs-b": {"command": "npx", "args": ["server"], "env": {"ROOT": "/b"}},
        }
    }
    res = _import(cfg)
    assert _names(res.created) == ["fs-a", "fs-b", "github-personal", "github-work"]
    assert res.skipped == ["github-copy"]
    work = next(c for c in res.created if c.display_name == "github-work")
    assert work.headers == {"Authorization": "Bearer WORK"}

    assert _import({"mcpServers": {"github-work": _bearer("NEW")}}).skipped == ["github-work"]
    assert set(_import(cfg).skipped) == set(cfg["mcpServers"])
    assert len(mcp_servers_db.list_servers()) == 4

    respelled = {
        "mcpServers": {
            "gh-a": {"url": GITHUB_URL, "headers": {"authorization": "Bearer PERSONAL"}},
            "gh-b": {"url": GITHUB_URL, "headers": {"AUTHORIZATION": "Bearer WORK"}},
            "gh-c": _bearer("WORK "),
            "fs-c": {"command": "npx", "args": ["server"], "env": {"root": "/a"}},
        }
    }
    res = _import(respelled)
    assert _names(res.created) == ["fs-c", "gh-c"]
    assert sorted(res.skipped) == ["gh-a", "gh-b"]

    rotated_pair = {"mcpServers": {"github-work": _bearer("NEXT"), "github-next": _bearer("NEXT")}}
    assert _names(_import(rotated_pair).created) == ["github-next"]

    oauth_url = "https://mcp.example.com/mcp"
    oauth = {
        "mcpServers": {
            "oauth-a": {"url": oauth_url, "oauth": {}, "headers": {"X-Org": "a"}},
            "oauth-b": {"url": oauth_url, "oauth": {}, "headers": {"X-Org": "b"}},
            "token-c": {"url": oauth_url, "headers": {"Authorization": "Bearer C"}},
            "plain-d": {"url": oauth_url, "headers": {"X-Org": "a"}},
            "plain-e": {"url": oauth_url},
        }
    }
    res = _import(oauth)
    assert _names(res.created) == ["oauth-a", "plain-d", "plain-e", "token-c"]
    assert res.skipped == ["oauth-b"]
    later = {"mcpServers": {"oauth-d": {"url": oauth_url, "oauth": {}, "headers": {"X-Org": "d"}}}}
    assert _import(later).skipped == ["oauth-d"]

    fs_a = next(r for r in mcp_servers_db.list_servers() if r["display_name"] == "fs-a")
    mcp_servers_db.update_server(fs_a["id"], {"use_oauth": 1})
    fs_copy = {"command": "npx", "args": ["server"], "env": {"ROOT": "/a"}}
    assert _import({"mcpServers": {"fs-copy": fs_copy}}).skipped == ["fs-copy"]


def test_import_matches_hidden_headers_on_url_alone(tmp_path, monkeypatch):
    import json

    from integrations import editors

    _reset_db(tmp_path, monkeypatch)
    _import({"mcpServers": {"github-work": _bearer("WORK")}})
    guesses = {"mcpServers": {"right": _bearer("WORK"), "wrong": _bearer("NOPE")}}
    assert sorted(_import(guesses, no_credential = True).skipped) == ["right", "wrong"]

    fresh = {
        "mcpServers": {
            "a": {"url": "https://fresh.example/mcp", "headers": {"Authorization": "Bearer A"}},
            "b": {"url": "https://fresh.example/mcp", "headers": {"Authorization": "Bearer B"}},
        }
    }
    res = _import(fresh, no_credential = True)
    assert _names(res.created) == ["a"] and res.skipped == ["b"]

    monkeypatch.setattr(editors, "_unreal_token", lambda _path: "EDITOR")
    config = json.dumps({"port": 8000, "project_path": str(tmp_path / "Demo.uproject")})
    mcp_servers_db.create_server(
        "u" * 16, "Unreal Engine", "", builtin_id = "unreal", builtin_config_json = config
    )
    editor_url = next(row["url"] for row in mcp_servers_db.list_servers() if row.get("builtin_id"))
    editor_guesses = {
        "mcpServers": {
            "ue-right": {"url": editor_url, "headers": {"X-MCP-Capability-Token": "EDITOR"}},
            "ue-wrong": {"url": editor_url, "headers": {"X-MCP-Capability-Token": "NOPE"}},
        }
    }
    assert sorted(_import(editor_guesses).skipped) == ["ue-right", "ue-wrong"]


def test_import_route_gates_stdio_when_disabled(tmp_path, monkeypatch):
    import asyncio

    from models.mcp_servers import McpServerImportRequest
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _disable(monkeypatch)
    cfg = {
        "mcpServers": {
            "fs": {"command": "npx", "args": ["server"]},
            "remote": {"url": "https://example.com/mcp"},
        }
    }
    res = asyncio.run(
        routes_mcp.import_mcp_servers(McpServerImportRequest(config = cfg), current_subject = "u")
    )
    assert {c.display_name for c in res.created} == {"remote"}
    assert any("fs" in err for err in res.errors)
    assert len(mcp_servers_db.list_servers()) == 1
