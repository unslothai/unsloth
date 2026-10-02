# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
from typer.testing import CliRunner

import unsloth_cli.commands.start as start

BASE = "http://127.0.0.1:8888"
MODEL = {"id": "unsloth/gemma-4-E2B-it-GGUF", "context_length": 32768}
CLI_KEY = "sk-unsloth-c11c11c11c11"
APP_KEY = "sk-unsloth-a99a99a99a99"


def _parse_toml(text: str) -> dict:
    tomllib = pytest.importorskip("tomllib")
    return tomllib.loads(text)


def _private(path: Path) -> bool:
    return os.name == "nt" or stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.fixture
def studio(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("CODEX_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_API_KEY", raising = False)
    server = {"calls": [], "keys": {CLI_KEY, APP_KEY}, "minted": 0}

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        server["calls"].append((method, url, payload))
        if url.endswith("/api/inference/loaded-models"):
            if token not in server["keys"] and token != "jwt":
                raise start.urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
            return {"object": "list", "data": [MODEL]}
        if url.endswith("/api/inference/status"):
            return {"is_gguf": True, "model_identifier": MODEL["id"]}
        if url.endswith("/api/auth/api-keys") and method == "GET":
            return {"api_keys": [{"key_prefix": CLI_KEY[11:19]}]}
        if url.endswith("/api/auth/api-keys") and method == "POST":
            if payload["name"].startswith("Coding agents"):
                return {"key": CLI_KEY, "api_key": {"id": 1}}
            server["minted"] += 1
            return {"key": APP_KEY, "api_key": {"id": 7}}
        if "/api/auth/api-keys/" in url and method == "DELETE":
            server["keys"].discard(APP_KEY)
            return {"detail": "API key revoked"}
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(start, "find_studio_server", lambda: BASE)
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: True)
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: None)
    monkeypatch.setattr(start, "_studio_token", lambda: "jwt")
    monkeypatch.setattr(start, "_http_json", http_json)
    monkeypatch.setattr(start, "_key_cache_path", lambda: tmp_path / "agent_api_key.json")
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    monkeypatch.setattr(start, "_require_agent_for_launch", lambda *args: None)
    monkeypatch.setattr(
        start, "_launch", lambda *args, **kwargs: pytest.fail("--app launched the CLI")
    )
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    server["opened"] = 0
    monkeypatch.setattr(
        start, "_open_codex_app", lambda: server.__setitem__("opened", 1), raising = False
    )
    monkeypatch.setattr(start, "_studio_healthy", lambda base, timeout = 3.0: True)
    server["while_running"] = lambda: None

    def ctrl_c(_seconds):
        server["while_running"]()
        raise KeyboardInterrupt

    monkeypatch.setattr(start.time, "sleep", ctrl_c)
    server["home"] = home
    return server


USER_CODEX = (
    "# my settings\n"
    'model = "gpt-5.5"\n'
    'approval_policy = "on-request"\n'
    "\n"
    '[projects."/work/repo"]\n'
    'trust_level = "trusted"\n'
)


def _codex_config(studio) -> Path:
    config = studio["home"] / ".codex" / "config.toml"
    config.parent.mkdir(exist_ok = True)
    config.write_text(USER_CODEX)
    return config


def test_codex_app_switches_the_app_while_running_and_back_on_exit(studio):
    config = _codex_config(studio)
    catalog = config.with_name("unsloth-model-catalog.json")
    backup = config.with_name("config.toml.unsloth-backup")
    seen = {}

    def while_running():
        seen["config"] = _parse_toml(config.read_text())
        seen["catalog"] = json.loads(catalog.read_text())["models"]
        seen["private"] = all(_private(p) for p in (config, catalog, backup))
        seen["backup"] = backup.read_text()

    studio["while_running"] = while_running

    result = CliRunner().invoke(start.start_app, ["codex", "--app"])

    assert result.exit_code == 0, result.output
    assert seen["config"]["model_provider"] == "unsloth_api"
    assert seen["config"]["model"] == MODEL["id"]
    assert seen["config"]["model_catalog_json"] == "unsloth-model-catalog.json"
    assert seen["config"]["approval_policy"] == "on-request"
    provider = seen["config"]["model_providers"]["unsloth_api"]
    assert provider["base_url"] == f"{BASE}/v1"
    assert provider["experimental_bearer_token"] == APP_KEY
    [listed] = seen["catalog"]
    assert (listed["slug"], listed["visibility"]) == (MODEL["id"], "list")
    assert seen["private"]
    assert seen["backup"] == USER_CODEX
    assert studio["opened"] == 1
    assert config.read_text() == USER_CODEX
    assert not catalog.exists()
    assert not backup.exists()
    assert ("POST", f"{BASE}/api/auth/api-keys", {"name": "Codex app"}) in studio["calls"]
    assert ("DELETE", f"{BASE}/api/auth/api-keys/7", None) in studio["calls"]
    assert APP_KEY not in result.output
    assert "Switched the Codex app back" in result.output


def test_codex_app_switches_back_when_unsloth_stops_answering(studio, monkeypatch):
    config = _codex_config(studio)
    monkeypatch.setattr(start.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(start, "_studio_healthy", lambda base, timeout = 3.0: False)

    result = CliRunner().invoke(start.start_app, ["codex", "--app"])

    assert result.exit_code == 0, result.output
    assert "stopped answering" in result.output
    assert config.read_text() == USER_CODEX
    assert not config.with_name("unsloth-model-catalog.json").exists()


def test_codex_recovers_a_switch_left_by_a_dead_session(studio, monkeypatch, tmp_path):
    config = _codex_config(studio)
    with monkeypatch.context() as crash:
        crash.setattr(start, "_switch_codex_app_back", lambda *args: [], raising = False)
        assert CliRunner().invoke(start.start_app, ["codex", "--app"]).exit_code == 0
    state_path = tmp_path / "agents" / "app" / "codex.json"
    state = json.loads(state_path.read_text())
    state["owner"] = {"pid": 2**22 + 4321, "started": 1.0}
    state_path.write_text(json.dumps(state))
    assert _parse_toml(config.read_text())["model"] == MODEL["id"]

    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert "did not exit cleanly" in result.output
    assert config.read_text() == USER_CODEX
    assert not state_path.exists()


def test_a_second_codex_app_session_does_not_switch_twice(studio):
    config = _codex_config(studio)
    seen = {}

    def while_running():
        before = config.read_text()
        seen["second"] = CliRunner().invoke(start.start_app, ["codex", "--app"])
        seen["unchanged"] = config.read_text() == before

    studio["while_running"] = while_running

    result = CliRunner().invoke(start.start_app, ["codex", "--app"])

    assert result.exit_code == 0, result.output
    assert seen["second"].exit_code == 1
    assert "already on Unsloth" in seen["second"].output
    assert seen["unchanged"]
    assert studio["minted"] == 1
    assert config.read_text() == USER_CODEX


def test_codex_app_leaves_a_model_the_user_picked_while_running(studio):
    config = _codex_config(studio)
    studio["while_running"] = lambda: config.write_text(
        config.read_text().replace(f'model = "{MODEL["id"]}"', 'model = "o3"')
    )

    result = CliRunner().invoke(start.start_app, ["codex", "--app"])

    assert result.exit_code == 0, result.output
    parsed = _parse_toml(config.read_text())
    assert parsed["model"] == "o3"
    assert parsed["approval_policy"] == "on-request"
    assert "model_provider" not in parsed
    assert "model_catalog_json" not in parsed
    assert "unsloth_api" not in parsed.get("model_providers", {})
    assert "Left model" in result.output


def test_codex_app_creates_and_removes_a_missing_config(studio):
    config = studio["home"] / ".codex" / "config.toml"
    seen = {}
    studio["while_running"] = lambda: seen.update(model = _parse_toml(config.read_text())["model"])

    assert CliRunner().invoke(start.start_app, ["codex", "--app"]).exit_code == 0

    assert seen["model"] == MODEL["id"]
    assert not config.exists()
    assert not config.with_name("unsloth-model-catalog.json").exists()


@pytest.mark.skipif(os.name == "nt", reason = "creating a symlink needs admin rights on Windows")
def test_codex_app_refuses_a_symlinked_config(studio, tmp_path):
    target = tmp_path / "dotfiles" / "codex.toml"
    target.parent.mkdir()
    target.write_text(USER_CODEX)
    config = studio["home"] / ".codex" / "config.toml"
    config.parent.mkdir()
    config.symlink_to(target)

    result = CliRunner().invoke(start.start_app, ["codex", "--app"])

    assert result.exit_code == 1
    assert "symlink" in result.output
    assert target.read_text() == USER_CODEX
    assert studio["minted"] == 0


def test_app_refuses_a_remote_server_without_an_explicit_key(studio, monkeypatch, tmp_path):
    remote = "http://10.0.0.5:8888"
    monkeypatch.setattr(start, "find_studio_server", lambda: remote)
    cache = tmp_path / "agent_api_key.json"
    cache.write_text(json.dumps({"servers": {remote: {"saved": [CLI_KEY], "minted": []}}}))

    result = CliRunner().invoke(start.start_app, ["codex", "--app"])

    assert result.exit_code == 1
    assert "--api-key" in result.output
    assert not (studio["home"] / ".codex").exists()


def test_codex_app_never_revokes_an_explicit_key(studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: "http://10.0.0.5:8888")
    config = studio["home"] / ".codex" / "config.toml"
    seen = {}
    studio["while_running"] = lambda: seen.update(config = _parse_toml(config.read_text()))

    result = CliRunner().invoke(start.start_app, ["codex", "--app", "--api-key", CLI_KEY])

    assert result.exit_code == 0, result.output
    assert seen["config"]["model_providers"]["unsloth_api"]["experimental_bearer_token"] == CLI_KEY
    assert studio["minted"] == 0
    assert not any(m == "DELETE" for m, _, _ in studio["calls"])


def test_codex_app_has_no_off_switch():
    result = CliRunner().invoke(start.start_app, ["codex", "--help"])
    assert "--app " in result.output
    assert "--app-off" not in result.output


def test_undo_is_not_an_app_option(studio):
    result = CliRunner().invoke(start.start_app, ["opencode", "--app", "--undo"])
    assert result.exit_code == 1
    assert "takes no agent arguments" in result.output


def test_opencode_app_adds_unsloth_without_changing_the_default_model(studio, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising = False)
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode", False))
    config = studio["home"] / ".config" / "opencode" / "opencode.json"
    config.parent.mkdir(parents = True)
    user = {
        "$schema": "https://opencode.ai/config.json",
        "model": "openai/gpt-5.5",
        "enabled_providers": ["anthropic"],
        "permission": {"bash": "allow"},
    }
    original = json.dumps(user, indent = 4) + "\n"
    config.write_text(original)

    result = CliRunner().invoke(start.start_app, ["opencode", "--app"])

    assert result.exit_code == 0, result.output
    data = json.loads(config.read_text())
    assert data["model"] == "openai/gpt-5.5"
    assert data["enabled_providers"] == ["anthropic", "unsloth-studio"]
    provider = data["provider"]["unsloth-studio"]
    assert provider["options"] == {"baseURL": f"{BASE}/v1", "apiKey": APP_KEY}
    assert provider["models"][MODEL["id"]]["limit"]["output"] > 0
    assert data["permission"] == {"bash": "allow"}
    assert "compaction" not in data
    assert _private(config)
    assert config.with_name("opencode.json.unsloth-backup").read_text() == original
    assert ("POST", f"{BASE}/api/auth/api-keys", {"name": "OpenCode app"}) in studio["calls"]
    assert APP_KEY not in result.output

    first = config.read_text()
    again = CliRunner().invoke(start.start_app, ["opencode", "--app"])
    assert again.exit_code == 0, again.output
    assert config.read_text() == first
    assert studio["minted"] == 1


def test_openclaw_app_adds_unsloth_and_joins_an_existing_allowlist(studio):
    config = studio["home"] / ".openclaw" / "openclaw.json"
    config.parent.mkdir()
    user = {
        "agents": {
            "defaults": {
                "model": {"primary": "openai/gpt-5.5", "fallbacks": ["openai/o3"]},
                "modelPolicy": {"allow": ["openai/*"]},
                "workspace": "/work",
            }
        },
        "gateway": {"mode": "local", "auth": {"mode": "token"}},
    }
    config.write_text(json.dumps(user, indent = 2) + "\n")

    result = CliRunner().invoke(start.start_app, ["openclaw", "--app"])

    assert result.exit_code == 0, result.output
    data = json.loads(config.read_text())
    ref = f"unsloth/{MODEL['id']}"
    defaults = data["agents"]["defaults"]
    assert defaults["model"] == user["agents"]["defaults"]["model"]
    assert defaults["modelPolicy"] == {"allow": ["openai/*", ref]}
    assert "models" not in defaults
    assert defaults["workspace"] == "/work"
    assert data["gateway"] == user["gateway"]
    assert data["models"]["providers"]["unsloth"]["apiKey"] == APP_KEY
    assert data["models"]["providers"]["unsloth"]["baseUrl"] == f"{BASE}/v1"


def test_openclaw_app_creates_no_allowlist(studio):
    config = studio["home"] / ".openclaw" / "openclaw.json"
    config.parent.mkdir()
    config.write_text('{"agents": {"defaults": {"model": "openai/gpt-5.5"}}}\n')

    assert CliRunner().invoke(start.start_app, ["openclaw", "--app"]).exit_code == 0

    defaults = json.loads(config.read_text())["agents"]["defaults"]
    assert defaults == {"model": "openai/gpt-5.5"}


@pytest.mark.parametrize(
    "defaults",
    [{"modelPolicy": {"allow": []}}, {"models": {}}],
    ids = ["empty-allow", "empty-legacy-map"],
)
def test_openclaw_app_keeps_an_empty_allowlist_open(studio, defaults):
    config = studio["home"] / ".openclaw" / "openclaw.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"agents": {"defaults": defaults}}) + "\n")

    assert CliRunner().invoke(start.start_app, ["openclaw", "--app"]).exit_code == 0

    assert json.loads(config.read_text())["agents"]["defaults"] == defaults


def test_opencode_app_reads_a_commented_jsonc(studio, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising = False)
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode", False))
    config = studio["home"] / ".config" / "opencode" / "opencode.jsonc"
    config.parent.mkdir(parents = True)
    config.write_text('{\n  // mine\n  "model": "openai/gpt-5.5", /* x */\n  "theme": "a//b",\n}\n')

    result = CliRunner().invoke(start.start_app, ["opencode", "--app"])

    assert result.exit_code == 0, result.output
    data = json.loads(config.read_text())
    assert (data["model"], data["theme"]) == ("openai/gpt-5.5", "a//b")
    assert "unsloth-studio" in data["provider"]


def test_opencode_app_keeps_each_config_dirs_backup_and_key(studio, monkeypatch, tmp_path):
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode", False))
    originals = {}
    for name in ("a", "b"):
        config = tmp_path / name / "opencode" / "opencode.json"
        config.parent.mkdir(parents = True)
        originals[name] = json.dumps({"model": f"openai/{name}"}) + "\n"
        config.write_text(originals[name])

    for name in ("a", "b", "a"):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / name))
        assert CliRunner().invoke(start.start_app, ["opencode", "--app"]).exit_code == 0

    for name in ("a", "b"):
        backup = tmp_path / name / "opencode" / "opencode.json.unsloth-backup"
        assert backup.read_text() == originals[name]
    assert studio["minted"] == 2


def test_codex_app_owner_without_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    owner = start._codex_app_owner()
    assert owner == {"pid": os.getpid(), "started": None}
    assert start._codex_app_owner_alive(owner) is (os.name != "nt")


def test_hermes_app_adds_unsloth_to_the_active_profile(studio):
    yaml = pytest.importorskip("yaml")
    root = studio["home"] / ".hermes"
    (root / "profiles" / "work").mkdir(parents = True)
    (root / "active_profile").write_text("work\n")
    config = root / "profiles" / "work" / "config.yaml"
    original = (
        "# my hermes\nmodel:\n  default: openrouter/kimi-k3  # mine\n  provider: openrouter\n"
        "toolsets:\n- web\n"
    )
    config.write_text(original)

    result = CliRunner().invoke(start.start_app, ["hermes", "--app"])

    assert result.exit_code == 0, result.output
    data = yaml.safe_load(config.read_text())
    assert data["model"] == {"default": "openrouter/kimi-k3", "provider": "openrouter"}
    assert "compression" not in data
    provider = data["providers"]["unsloth"]
    assert provider["api_key"] == APP_KEY
    assert provider["base_url"] == f"{BASE}/v1"
    assert "key_env" not in provider
    assert provider["models"] == {MODEL["id"]: {"context_length": start._HERMES_MIN_CONTEXT}}
    assert data["toolsets"] == ["web"]
    assert config.read_text().startswith(original)
    assert not (root / "config.yaml").exists()
    assert ("POST", f"{BASE}/api/auth/api-keys", {"name": "Hermes app"}) in studio["calls"]


def test_app_refuses_a_config_it_cannot_parse(studio):
    config = studio["home"] / ".openclaw" / "openclaw.json"
    config.parent.mkdir()
    config.write_text("{ // json5 comment\n  agents: {} }\n")

    result = CliRunner().invoke(start.start_app, ["openclaw", "--app"])

    assert result.exit_code == 1
    assert "Couldn't parse" in result.output
    assert config.read_text() == "{ // json5 comment\n  agents: {} }\n"
    assert studio["minted"] == 0
