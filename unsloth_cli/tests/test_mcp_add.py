# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
from typer.testing import CliRunner

import unsloth_cli
import unsloth_cli.commands.mcp as mcp_mod

CANARY = "sk-unsloth-LEAKCANARY0123456789"
URL = "http://127.0.0.1:18799/mcp/"
POSIX = pytest.mark.skipif(os.name == "nt", reason = "POSIX file modes")
tomllib = pytest.importorskip("tomllib") if sys.version_info >= (3, 11) else None


@pytest.fixture
def home(tmp_path, monkeypatch):
    for var, sub in (("HOME", "home"), ("CLAUDE_CONFIG_DIR", "claude"), ("CODEX_HOME", "codex")):
        (tmp_path / sub).mkdir()
        monkeypatch.setenv(var, str(tmp_path / sub))
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising = False)
    monkeypatch.setenv("UNSLOTH_API_KEY", CANARY)
    monkeypatch.setattr(mcp_mod, "find_studio_server", lambda *a, **k: None)
    monkeypatch.setattr(mcp_mod, "_which_with_install_dirs", lambda name: None)
    return tmp_path


@pytest.fixture
def fake_claude(home, monkeypatch):
    """A `claude` binary that records argv and edits .claude.json the way the real one does."""
    calls = []
    path = home / "claude" / ".claude.json"

    def run(argv, **kwargs):
        calls.append(list(argv))
        data = json.loads(path.read_text()) if path.exists() else {}
        servers = data.setdefault("mcpServers", {})
        if argv[2] == "remove":
            servers.pop(argv[-1], None)
        else:
            name, url, header = argv[7], argv[8], argv[10]
            key, value = header.split(": ", 1)
            servers[name] = {"type": "http", "url": url, "headers": {key: value}}
        path.write_text(json.dumps(data))
        return mcp_mod.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(mcp_mod, "_which_with_install_dirs", lambda name: "/opt/bin/claude")
    monkeypatch.setattr(mcp_mod.subprocess, "run", run)
    return calls


def _add(*args):
    return CliRunner().invoke(unsloth_cli.app, ["mcp", "add", *args])


def _mode(path):
    return path.stat().st_mode & 0o777


# Pure renderers


@pytest.mark.parametrize(
    "base",
    ["http://h:1", "http://h:1/", "http://h:1/mcp", "http://h:1/mcp/", " http://h:1// "],
)
def test_mcp_url_normalises_to_trailing_slash(base):
    assert mcp_mod.mcp_url(base) == "http://h:1/mcp/"


def test_entries_reference_the_env_var_only():
    assert mcp_mod.claude_entry(URL, "MY_KEY") == {
        "type": "http",
        "url": URL,
        "headers": {"Authorization": "Bearer ${MY_KEY}"},
    }
    merged = mcp_mod.cursor_merge({"mcpServers": {"o": {}}, "k": 1}, "s", URL, "V")
    assert merged["k"] == 1 and merged["mcpServers"]["o"] == {}
    assert merged["mcpServers"]["s"] == {
        "url": URL,
        "headers": {"Authorization": "Bearer ${env:V}"},
    }
    snippet = mcp_mod.generic_snippet("s", URL, "V")["mcpServers"]["s"]
    assert snippet["headers"] == {"Authorization": "Bearer ${V}"}


EXPECTED = {"url": URL, "bearer_token_env_var": "V", "tool_timeout_sec": 300}
EXISTING = (
    'model = "o3"\n\n'
    "[mcp_servers.other]\n"
    'command = "x"\n\n'
    "[mcp_servers.unsloth-studio]\n"
    'url = "http://old/mcp/"\n\n'
    "[mcp_servers.unsloth-studio.env]\n"
    'A = "1"\n\n'
    '[profiles."p"]\n'
    "x = 1\n"
)


def test_codex_merge_replaces_in_place_and_drops_subtables():
    new = mcp_mod.codex_merge(EXISTING, "unsloth-studio", URL, "UNSLOTH_API_KEY", 300)
    assert new == (
        'model = "o3"\n\n'
        "[mcp_servers.other]\n"
        'command = "x"\n\n'
        "[mcp_servers.unsloth-studio]\n"
        f'url = "{URL}"\n'
        'bearer_token_env_var = "UNSLOTH_API_KEY"\n'
        "tool_timeout_sec = 300\n\n"
        '[profiles."p"]\n'
        "x = 1\n"
    )
    assert mcp_mod.codex_merge(new, "unsloth-studio", URL, "UNSLOTH_API_KEY", 300) == new


@pytest.mark.parametrize(
    "header", ['[mcp_servers."unsloth-studio"]', "[ mcp_servers . 'unsloth-studio' ]"]
)
def test_codex_merge_matches_quoted_headers(header):
    new = mcp_mod.codex_merge(f'{header}\nurl = "x"\n', "unsloth-studio", URL, "V", 300)
    assert new.count("unsloth-studio") == 1


def test_codex_merge_appends_and_keeps_other_tables_byte_identical():
    old = '# mine\n[mcp_servers.unsloth-studio-2]\nurl = "y"\n'
    new = mcp_mod.codex_merge(old, "unsloth-studio", URL, "V", 300)
    assert new.startswith(old + "\n[mcp_servers.unsloth-studio]\n")
    assert mcp_mod.codex_merge("", "s", URL, "V", 300).startswith("[mcp_servers.s]\n")


@pytest.mark.skipif(tomllib is None, reason = "needs tomllib")
@pytest.mark.parametrize(
    "old",
    [
        'mcp_servers = { unsloth-studio = { url = "x" } }\n',
        'mcp_servers.unsloth-studio.url = "x"\n',
        "[mcp_servers]\nunsloth-studio = { url = 'x' }\n",
        "[broken\n",
    ],
)
def test_codex_check_refuses_other_forms_and_invalid_toml(old):
    new = mcp_mod.codex_merge(old, "unsloth-studio", URL, "V", 300)
    with pytest.raises(ValueError):
        mcp_mod.codex_check(old, new, "unsloth-studio", EXPECTED)


def test_codex_check_without_a_toml_parser_refuses_an_inline_table(monkeypatch):
    monkeypatch.setattr(mcp_mod, "_toml_loads", lambda: None)
    for old, ok in (
        ('mcp_servers = { unsloth-studio = { url = "x" } }\n', False),
        (EXISTING, True),
    ):
        new = mcp_mod.codex_merge(old, "unsloth-studio", URL, "V", 300)
        try:
            mcp_mod.codex_check(old, new, "unsloth-studio", EXPECTED)
            assert ok
        except ValueError:
            assert not ok


# The command


def test_mcp_is_registered_on_the_cli():
    result = CliRunner().invoke(unsloth_cli.app, ["mcp", "add", "--help"])
    assert result.exit_code == 0, result.output
    for agent in ("claude-code", "codex", "cursor", "generic"):
        assert agent in result.output
    assert "--api-key" not in result.output


@POSIX
@pytest.mark.skipif(tomllib is None, reason = "needs tomllib")
def test_codex_writes_once_then_reports_unchanged(home):
    path = home / "codex" / "config.toml"
    result = _add("codex", "--url", "http://127.0.0.1:18799")
    assert result.exit_code == 0, result.output
    assert "Unsloth Studio MCP set up for codex" in result.output
    assert _mode(path) == 0o600
    assert tomllib.loads(path.read_text())["mcp_servers"]["unsloth-studio"] == {
        "url": URL,
        "bearer_token_env_var": "UNSLOTH_API_KEY",
        "tool_timeout_sec": 300,
    }
    before = path.stat().st_mtime_ns
    result = _add("codex", "--url", "http://127.0.0.1:18799")
    assert result.exit_code == 0 and "unchanged" in result.output
    assert path.stat().st_mtime_ns == before


@POSIX
@pytest.mark.parametrize("mode", [0o644, 0o400])
def test_existing_file_keeps_its_mode(home, mode):
    path = home / "codex" / "config.toml"
    path.write_text(EXISTING)
    path.chmod(mode)
    result = _add("codex", "--url", URL, "--timeout", "600")
    assert result.exit_code == 0, result.output
    assert _mode(path) == mode
    assert "tool_timeout_sec = 600" in path.read_text()
    assert "[mcp_servers.unsloth-studio.env]" not in path.read_text()


def test_dry_run_prints_a_diff_and_writes_nothing(home, fake_claude):
    path = home / "codex" / "config.toml"
    path.write_text(EXISTING)
    stat = path.stat()
    result = _add("codex", "--url", URL, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "+bearer_token_env_var" in result.output and "Dry run" in result.output
    assert path.read_text() == EXISTING and path.stat().st_mtime_ns == stat.st_mtime_ns
    result = _add("cursor", "--url", URL, "--dry-run")
    assert result.exit_code == 0 and not (home / "home" / ".cursor").exists()
    result = _add("claude-code", "--url", URL, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Would run: /opt/bin/claude mcp add" in result.output
    assert fake_claude == [] and not (home / "claude" / ".claude.json").exists()


def test_codex_refuses_an_inline_table_and_leaves_the_file(home):
    path = home / "codex" / "config.toml"
    old = 'mcp_servers = { unsloth-studio = { url = "x" } }\n'
    path.write_text(old)
    result = _add("codex", "--url", URL)
    assert result.exit_code == 1
    assert "[mcp_servers.unsloth-studio]" in result.output
    assert path.read_text() == old


@POSIX
def test_symlinked_config_is_refused(home):
    target = home / "real.json"
    target.write_text("{}")
    (home / "home" / ".cursor").mkdir()
    (home / "home" / ".cursor" / "mcp.json").symlink_to(target)
    result = _add("cursor", "--url", URL)
    assert result.exit_code == 1 and "symlink" in result.output
    assert target.read_text() == "{}"


@POSIX
def test_cursor_reads_jsonc_and_keeps_other_servers(home):
    path = home / "home" / ".cursor" / "mcp.json"
    path.parent.mkdir()
    path.write_text('{\n  // mine\n  "mcpServers": {"other": {"url": "x"},},\n}\n')
    result = _add("cursor", "--url", URL)
    assert result.exit_code == 0, result.output
    data = json.loads(path.read_text())
    assert data["mcpServers"]["other"] == {"url": "x"}
    assert data["mcpServers"]["unsloth-studio"] == {
        "url": URL,
        "headers": {"Authorization": "Bearer ${env:UNSLOTH_API_KEY}"},
    }
    assert "unchanged" in _add("cursor", "--url", URL).output


@pytest.mark.parametrize("text", ["[1, 2]", "{nope"])
def test_unparseable_json_is_refused(home, text):
    path = home / "home" / ".cursor" / "mcp.json"
    path.parent.mkdir()
    path.write_text(text)
    result = _add("cursor", "--url", URL)
    assert result.exit_code == 1 and "Couldn't parse" in result.output
    assert path.read_text() == text


@POSIX
def test_claude_without_the_binary_writes_the_json_directly(home):
    path = home / "claude" / ".claude.json"
    result = _add("claude-code", "--url", "http://127.0.0.1:18799")
    assert result.exit_code == 0, result.output
    assert "Claude Code was not found" in result.output
    assert json.loads(path.read_text())["mcpServers"]["unsloth-studio"] == {
        "type": "http",
        "url": URL,
        "headers": {"Authorization": "Bearer ${UNSLOTH_API_KEY}"},
    }
    assert _mode(path) == 0o600
    assert "unchanged" in _add("claude-code", "--url", URL).output


def test_claude_shells_out_with_list_args(home, fake_claude):
    result = _add("claude-code", "--url", URL, "--env-var", "MY_KEY")
    assert result.exit_code == 0, result.output
    argv = "/opt/bin/claude mcp add --transport http --scope user unsloth-studio".split()
    assert fake_claude == [argv + [URL, "--header", "Authorization: Bearer ${MY_KEY}"]]
    assert "Check the result" not in result.output
    fake_claude.clear()
    assert "unchanged" in _add("claude-code", "--url", URL, "--env-var", "MY_KEY").output
    assert fake_claude == []
    result = _add("claude-code", "--url", "http://127.0.0.1:9/", "--env-var", "MY_KEY")
    assert result.exit_code == 0, result.output
    assert [argv[2] for argv in fake_claude] == ["remove", "add"]
    assert fake_claude[0][-3:] == ["--scope", "user", "unsloth-studio"]


def test_claude_failure_and_unparseable_config_exit_1(home, fake_claude, monkeypatch):
    path = home / "claude" / ".claude.json"
    path.write_text("{broken")
    assert _add("claude-code", "--url", URL).exit_code == 1
    assert fake_claude == [] and path.read_text() == "{broken"
    path.unlink()
    failed = lambda argv, **k: mcp_mod.subprocess.CompletedProcess(argv, 1, "", "boom")
    monkeypatch.setattr(mcp_mod.subprocess, "run", failed)
    result = _add("claude-code", "--url", URL)
    assert result.exit_code == 1 and "boom" in result.output


def test_generic_prints_or_merges(home):
    result = _add("generic", "--url", URL)
    assert result.exit_code == 0, result.output
    assert '"Authorization": "Bearer ${UNSLOTH_API_KEY}"' in result.output
    out = home / "agent.json"
    out.write_text('{"mcpServers": {"a": {}}, "x": 1}')
    assert _add("generic", "--url", URL, "--output", str(out)).exit_code == 0
    data = json.loads(out.read_text())
    assert data["x"] == 1 and set(data["mcpServers"]) == {"a", "unsloth-studio"}


BAD = ["--env-var=BAD-NAME", "--env-var=$(id)", "--name=a b", "--name=x]\n[y", "--url=file:///e"]


@pytest.mark.parametrize("arg", BAD)
def test_bad_input_is_refused(home, arg):
    result = _add("codex", "--url", URL, arg)
    assert result.exit_code == 1, result.output
    assert not (home / "codex" / "config.toml").exists()


def test_timeout_below_the_wait_cap_is_rejected(home):
    assert _add("codex", "--url", URL, "--timeout", "120").exit_code == 2


def test_url_falls_back_to_env_then_discovery_then_default(home, monkeypatch):
    assert mcp_mod._resolve_base(None) == "http://127.0.0.1:8888"
    monkeypatch.setattr(mcp_mod, "find_studio_server", lambda *a, **k: "http://127.0.0.1:9000")
    assert mcp_mod._resolve_base(None) == "http://127.0.0.1:9000"
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://box:1")
    assert mcp_mod._resolve_base(None) == "http://box:1"
    assert mcp_mod._resolve_base("http://x:2") == "http://x:2"


def test_the_key_never_leaks(home, fake_claude, monkeypatch):
    assert "is not set" not in _add("generic", "--url", URL).output
    outputs = []
    for agent in ("claude-code", "codex", "cursor", "generic"):
        for extra in (["--dry-run"], []):
            result = _add(agent, "--url", URL, *extra)
            assert result.exit_code == 0, result.output
            outputs.append(result.output)
    outputs.append(_add("generic", "--url", URL, "--output", str(home / "g.json")).output)
    assert fake_claude, "the claude stub should have run"
    assert all("LEAKCANARY" not in " ".join(argv) for argv in fake_claude)
    assert all("LEAKCANARY" not in text for text in outputs)
    for path in home.rglob("*"):
        if path.is_file():
            assert "LEAKCANARY" not in path.read_text(), path
    monkeypatch.delenv("UNSLOTH_API_KEY")
    assert "UNSLOTH_API_KEY is not set" in _add("cursor", "--url", URL).output
