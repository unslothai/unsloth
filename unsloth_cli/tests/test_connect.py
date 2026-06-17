# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `unsloth connect` — config merging and launch env, no network."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import pytest
from typer.testing import CliRunner

import unsloth_cli.commands.connect as connect

BASE = "http://127.0.0.1:8888"
MODEL = {"id": "unsloth/gemma-4-26B-A4B-it-GGUF", "context_length": 131072}


@pytest.fixture()
def claude_settings(tmp_path, monkeypatch):
    path = tmp_path / "claude" / "settings.json"
    monkeypatch.setattr(connect, "claude_settings_path", lambda: path)
    return path


def test_claude_settings_created_when_missing(claude_settings):
    connect.ensure_claude_attribution_header()
    settings = json.loads(claude_settings.read_text())
    assert settings["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"


def test_claude_settings_merge_preserves_existing(claude_settings):
    claude_settings.parent.mkdir(parents = True)
    claude_settings.write_text(
        json.dumps({"effortLevel": "high", "env": {"CLAUDE_CODE_ENABLE_TELEMETRY": "0"}})
    )
    connect.ensure_claude_attribution_header()
    settings = json.loads(claude_settings.read_text())
    assert settings["effortLevel"] == "high"
    assert settings["env"]["CLAUDE_CODE_ENABLE_TELEMETRY"] == "0"
    assert settings["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"


def test_claude_settings_already_set_untouched(claude_settings):
    claude_settings.parent.mkdir(parents = True)
    original = json.dumps({"env": {"CLAUDE_CODE_ATTRIBUTION_HEADER": "0"}})
    claude_settings.write_text(original)
    connect.ensure_claude_attribution_header()
    assert claude_settings.read_text() == original


def test_claude_settings_bad_json_left_alone(claude_settings, capsys):
    claude_settings.parent.mkdir(parents = True)
    claude_settings.write_text("{not json")
    connect.ensure_claude_attribution_header()
    assert claude_settings.read_text() == "{not json"
    assert "couldn't parse" in capsys.readouterr().err


def _fake_claude(monkeypatch, version_output: str) -> None:
    monkeypatch.setattr(connect.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(
        connect.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout = version_output),
    )


def test_cache_flags_passed_to_supported_claude(monkeypatch):
    _fake_claude(monkeypatch, "2.1.98 (Claude Code)\n")
    assert connect._claude_cache_flags() == ["--exclude-dynamic-system-prompt-sections"]


def test_cache_flags_skipped_on_old_claude(monkeypatch):
    _fake_claude(monkeypatch, "2.0.14 (Claude Code)\n")
    assert connect._claude_cache_flags() == []


def test_cache_flags_skipped_on_unparseable_version(monkeypatch):
    _fake_claude(monkeypatch, "weird build string\n")
    assert connect._claude_cache_flags() == []


def _parse_toml(text: str) -> dict:
    tomllib = pytest.importorskip("tomllib")
    return tomllib.loads(text)


def test_merge_codex_config_fresh():
    merged = connect._merge_codex_config("", BASE)
    parsed = _parse_toml(merged)
    assert parsed["oss_provider"] == "unsloth_api"
    provider = parsed["model_providers"]["unsloth_api"]
    assert provider["base_url"] == f"{BASE}/v1"
    assert provider["wire_api"] == "responses"
    assert provider["requires_openai_auth"] is False


def test_merge_codex_config_replaces_stale_block():
    existing = (
        'model = "gpt-5"\n'
        "\n"
        "[model_providers.unsloth_api]\n"
        'base_url = "http://old-host:9999/v1"\n'
        'wire_api = "chat"\n'
        "\n"
        "[model_providers.unsloth_api.http_headers]\n"
        'x-old = "1"\n'
        "\n"
        "[model_providers.ollama]\n"
        'base_url = "http://localhost:11434/v1"\n'
    )
    merged = connect._merge_codex_config(existing, BASE)
    parsed = _parse_toml(merged)
    assert parsed["model"] == "gpt-5"
    assert parsed["model_providers"]["unsloth_api"]["base_url"] == f"{BASE}/v1"
    assert parsed["model_providers"]["unsloth_api"]["wire_api"] == "responses"
    assert "http_headers" not in parsed["model_providers"]["unsloth_api"]
    assert parsed["model_providers"]["ollama"]["base_url"] == "http://localhost:11434/v1"
    assert connect._merge_codex_config(merged, BASE) == merged


def test_merge_codex_config_keeps_user_oss_provider():
    merged = connect._merge_codex_config('oss_provider = "ollama"\n', BASE)
    assert _parse_toml(merged)["oss_provider"] == "ollama"


def test_write_codex_config_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    connect.write_codex_config(BASE, MODEL)
    profile = _parse_toml((tmp_path / "unsloth_api.config.toml").read_text())
    assert profile["oss_provider"] == "unsloth_api"
    assert profile["model_provider"] == "unsloth_api"
    assert profile["model"] == MODEL["id"]
    assert profile["model_context_window"] == 131072
    config = _parse_toml((tmp_path / "config.toml").read_text())
    assert config["model_providers"]["unsloth_api"]["env_key"] == "UNSLOTH_STUDIO_AUTH_TOKEN"


@pytest.fixture()
def fake_studio(tmp_path, monkeypatch, claude_settings):
    calls = []
    state = {"models": [MODEL]}

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        calls.append((method, url, payload))
        if url.endswith("/v1/models"):
            return {"object": "list", "data": state["models"]}
        if url.endswith("/api/inference/status"):
            return {"is_gguf": True, "model_identifier": state["models"][0]["id"]}
        if url.endswith("/api/auth/api-keys"):
            return {"key": "sk-unsloth-feedfacefeedface"}
        if url.endswith("/api/inference/load"):
            state["models"] = [{"id": payload["model_path"], "context_length": 4096}]
            return {}
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(connect, "find_studio_server", lambda: BASE)
    monkeypatch.setattr(connect, "_studio_token", lambda: "jwt-token")
    monkeypatch.setattr(connect, "_http_json", http_json)
    monkeypatch.setattr(connect, "_key_cache_path", lambda: tmp_path / "agent_api_key.json")
    # No `claude` on PATH, so _claude_cache_flags never probes the real binary.
    monkeypatch.setattr(connect.shutil, "which", lambda _: None)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.delenv("UNSLOTH_API_KEY", raising = False)
    return calls


def test_connect_claude_no_launch(fake_studio, claude_settings):
    result = CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert f"export ANTHROPIC_BASE_URL={BASE}" in result.output
    assert "export ANTHROPIC_AUTH_TOKEN=sk-unsloth-feedfacefeedface" in result.output
    assert f"export ANTHROPIC_MODEL={MODEL['id']}" in result.output
    assert "export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1" in result.output
    assert "export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1" in result.output
    assert f"claude --model {MODEL['id']} --exclude-dynamic-system-prompt-sections" in result.output
    settings = json.loads(claude_settings.read_text())
    assert settings["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"


def test_connect_codex_no_launch(fake_studio, tmp_path):
    result = CliRunner().invoke(connect.connect_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "export UNSLOTH_STUDIO_AUTH_TOKEN=sk-unsloth-feedfacefeedface" in result.output
    assert "codex --oss --profile unsloth_api" in result.output
    assert (tmp_path / "codex" / "config.toml").exists()
    assert (tmp_path / "codex" / "unsloth_api.config.toml").exists()


def test_connect_key_minted_once_then_cached(fake_studio, tmp_path):
    CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    mints = [c for c in fake_studio if c[1].endswith("/api/auth/api-keys")]
    assert len(mints) == 1
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    assert cached["keys"] == ["sk-unsloth-feedfacefeedface"]


def test_connect_explicit_key_remembered_for_keyless_runs(fake_studio, tmp_path):
    CliRunner().invoke(
        connect.connect_app,
        ["claude", "--no-launch", "--api-key", "sk-unsloth-deadbeefdeadbeef"],
    )
    result = CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "export ANTHROPIC_AUTH_TOKEN=sk-unsloth-deadbeefdeadbeef" in result.output
    mints = [c for c in fake_studio if c[1].endswith("/api/auth/api-keys")]
    assert mints == []


def test_connect_skips_cached_keys_the_server_rejects(fake_studio, tmp_path, monkeypatch):
    cache = tmp_path / "agent_api_key.json"
    cache.write_text(json.dumps({"keys": ["sk-unsloth-stale", "sk-unsloth-feedfacefeedface"]}))
    inner = connect._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/v1/models") and token == "sk-unsloth-stale":
            raise urllib.error.HTTPError(url, 401, "Unauthorized", None, None)
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(connect, "_http_json", http_json)
    result = CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "export ANTHROPIC_AUTH_TOKEN=sk-unsloth-feedfacefeedface" in result.output
    mints = [c for c in fake_studio if c[1].endswith("/api/auth/api-keys")]
    assert mints == []
    # The working key moves to the front so the next run tries it first.
    cached = json.loads(cache.read_text())
    assert cached["keys"] == ["sk-unsloth-feedfacefeedface", "sk-unsloth-stale"]


def test_connect_reads_legacy_single_key_cache(fake_studio, tmp_path):
    (tmp_path / "agent_api_key.json").write_text(json.dumps({"key": "sk-unsloth-oldformat"}))
    result = CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "export ANTHROPIC_AUTH_TOKEN=sk-unsloth-oldformat" in result.output
    mints = [c for c in fake_studio if c[1].endswith("/api/auth/api-keys")]
    assert mints == []


def test_connect_model_flag_loads_on_server(fake_studio):
    result = CliRunner().invoke(
        connect.connect_app, ["claude", "--no-launch", "--model", "unsloth/Qwen3.5-35B-A3B"]
    )
    assert result.exit_code == 0, result.output
    loads = [c for c in fake_studio if c[1].endswith("/api/inference/load")]
    assert loads == [
        ("POST", f"{BASE}/api/inference/load", {"model_path": "unsloth/Qwen3.5-35B-A3B"})
    ]
    assert "export ANTHROPIC_MODEL=unsloth/Qwen3.5-35B-A3B" in result.output


def test_connect_model_flag_matches_canonical_id(fake_studio, monkeypatch):
    # Studio registers a loaded model under a canonical id (resolved identifier
    # / casing) that can differ from the path we passed. The agent must connect
    # to that model, not silently fall through to the first loaded one.
    requested = "Unsloth/Qwen3.5-35B-A3B"
    canonical = "unsloth/Qwen3.5-35B-A3B"
    inner = connect._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/load"):
            return {"model": canonical, "display_name": canonical}
        if url.endswith("/v1/models"):
            # Decoy sorts first, so models[0] is the wrong pick on the old code.
            return {"object": "list", "data": [MODEL, {"id": canonical, "context_length": 4096}]}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(connect, "_http_json", http_json)
    result = CliRunner().invoke(
        connect.connect_app, ["claude", "--no-launch", "--model", requested]
    )
    assert result.exit_code == 0, result.output
    assert f"export ANTHROPIC_MODEL={canonical}" in result.output


def test_connect_no_model_loaded_errors(fake_studio, monkeypatch):
    monkeypatch.setattr(
        connect,
        "_http_json",
        lambda method, url, token, payload = None, timeout = 30, error = None: (
            {"key": "sk-unsloth-feedfacefeedface"}
            if url.endswith("/api/auth/api-keys")
            else {"object": "list", "data": []}
        ),
    )
    result = CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    assert result.exit_code == 1
    assert "No model is loaded" in result.output


def test_connect_requested_model_not_loaded_fails(fake_studio, monkeypatch):
    # Studio never surfaces the requested model; fail loudly rather than
    # silently connecting to whatever else happens to be loaded.
    inner = connect._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/load"):
            return {}
        if url.endswith("/v1/models"):
            return {"object": "list", "data": [MODEL]}  # decoy; request never appears
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(connect, "_http_json", http_json)
    result = CliRunner().invoke(
        connect.connect_app, ["claude", "--no-launch", "--model", "unsloth/Missing-7B"]
    )
    assert result.exit_code == 1
    assert "unsloth/Missing-7B" in result.output


def test_connect_codex_rejects_non_gguf_model(fake_studio, monkeypatch):
    inner = connect._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"is_gguf": False, "model_identifier": "unsloth/Qwen3-0.6B"}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(connect, "_http_json", http_json)
    result = CliRunner().invoke(connect.connect_app, ["codex", "--no-launch"])
    assert result.exit_code == 1
    assert "GGUF" in result.output
    result = CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output


def test_connect_remote_token_rejected_points_at_api_key(fake_studio, monkeypatch):
    # A self-issued token is invalid against a remote Studio (different secret);
    # the auto-mint 401 should become actionable --api-key guidance.
    inner = connect._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/auth/api-keys"):
            raise urllib.error.HTTPError(url, 401, "Invalid or expired token", None, None)
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(connect, "_http_json", http_json)
    result = CliRunner().invoke(connect.connect_app, ["opencode", "--no-launch"])
    assert result.exit_code == 1
    assert "Settings → API" in result.output
    assert "--api-key" in result.output


def test_connect_no_studio_errors(fake_studio, monkeypatch):
    monkeypatch.setattr(connect, "find_studio_server", lambda: None)
    result = CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    assert result.exit_code == 1
    assert "No running Studio server" in result.output


def test_connect_explicit_api_key_skips_mint(fake_studio):
    result = CliRunner().invoke(
        connect.connect_app,
        ["claude", "--no-launch", "--api-key", "sk-unsloth-deadbeefdeadbeef"],
    )
    assert result.exit_code == 0, result.output
    assert "export ANTHROPIC_AUTH_TOKEN=sk-unsloth-deadbeefdeadbeef" in result.output
    assert not any(c[1].endswith("/api/auth/api-keys") for c in fake_studio)


# ── OpenClaw (Anthropic /v1/messages) ────────────────────────────────


def test_write_openclaw_config_fresh(tmp_path, monkeypatch):
    path = tmp_path / "openclaw.json"
    monkeypatch.setattr(connect, "openclaw_config_path", lambda: path)
    connect.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL)
    config = json.loads(path.read_text())
    provider = config["models"]["providers"]["unsloth"]
    assert provider["baseUrl"] == f"{BASE}/v1"
    assert provider["apiKey"] == "sk-unsloth-abc"
    assert provider["api"] == "openai-completions"
    assert provider["models"] == [
        {"id": MODEL["id"], "name": MODEL["id"], "contextWindow": MODEL["context_length"]}
    ]
    # The default model must be pinned or OpenClaw has nothing active.
    assert config["agents"]["defaults"]["model"]["primary"] == f"unsloth/{MODEL['id']}"
    assert config["gateway"]["mode"] == "local"
    assert config["gateway"]["auth"]["mode"] == "none"  # unauth loopback gateway
    if os.name != "nt":  # the file holds an API key
        assert path.stat().st_mode & 0o777 == 0o600


def test_write_openclaw_config_preserves_and_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "openclaw.json"
    monkeypatch.setattr(connect, "openclaw_config_path", lambda: path)
    path.write_text(
        json.dumps(
            {
                "theme": "dark",
                "agents": {"defaults": {"temperature": 0.5}},
                "models": {"mode": "replace", "providers": {"openrouter": {"baseUrl": "x"}}},
            }
        )
    )
    connect.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL)
    config = json.loads(path.read_text())
    assert config["theme"] == "dark"
    assert config["agents"]["defaults"]["temperature"] == 0.5  # other agent defaults kept
    assert config["agents"]["defaults"]["model"]["primary"] == f"unsloth/{MODEL['id']}"
    assert config["models"]["mode"] == "replace"  # user's mode is left as-is
    assert config["models"]["providers"]["openrouter"]["baseUrl"] == "x"
    assert config["models"]["providers"]["unsloth"]["baseUrl"] == f"{BASE}/v1"
    before = path.read_text()
    connect.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL)
    assert path.read_text() == before


def test_write_openclaw_config_corrupt_left_alone(tmp_path, monkeypatch, capsys):
    path = tmp_path / "openclaw.json"
    monkeypatch.setattr(connect, "openclaw_config_path", lambda: path)
    path.write_text("{not json")
    connect.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL)
    assert path.read_text() == "{not json"
    assert "couldn't parse" in capsys.readouterr().err


def test_connect_openclaw_no_launch(fake_studio, tmp_path, monkeypatch):
    path = tmp_path / "openclaw.json"
    monkeypatch.setattr(connect, "openclaw_config_path", lambda: path)
    result = CliRunner().invoke(connect.connect_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "openclaw" in result.output
    assert "export" not in result.output  # key lives in the config, not the env
    config = json.loads(path.read_text())
    assert config["models"]["providers"]["unsloth"]["apiKey"] == "sk-unsloth-feedfacefeedface"
    assert config["agents"]["defaults"]["model"]["primary"] == f"unsloth/{MODEL['id']}"
    # OpenAI /v1/chat/completions works on either backend — no GGUF gate.
    assert not any(c[1].endswith("/api/inference/status") for c in fake_studio)


# ── OpenCode (OpenAI /v1/chat/completions) ───────────────────────────


def test_write_opencode_config_fresh(tmp_path, monkeypatch):
    path = tmp_path / "opencode.json"
    monkeypatch.setattr(connect, "opencode_config_path", lambda: path)
    connect.write_opencode_config(BASE, "sk-unsloth-abc", MODEL)
    config = json.loads(path.read_text())
    provider = config["provider"]["unsloth"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"] == {"baseURL": f"{BASE}/v1", "apiKey": "sk-unsloth-abc"}
    assert provider["models"] == {MODEL["id"]: {"name": MODEL["id"]}}
    assert config["model"] == f"unsloth/{MODEL['id']}"


def test_write_opencode_config_preserves_and_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "opencode.json"
    monkeypatch.setattr(connect, "opencode_config_path", lambda: path)
    path.write_text(
        json.dumps({"theme": "tokyonight", "provider": {"anthropic": {"name": "Anthropic"}}})
    )
    connect.write_opencode_config(BASE, "sk-unsloth-abc", MODEL)
    config = json.loads(path.read_text())
    assert config["theme"] == "tokyonight"
    assert config["provider"]["anthropic"]["name"] == "Anthropic"
    assert config["provider"]["unsloth"]["options"]["baseURL"] == f"{BASE}/v1"
    before = path.read_text()
    connect.write_opencode_config(BASE, "sk-unsloth-abc", MODEL)
    assert path.read_text() == before


def test_connect_opencode_no_launch(fake_studio, tmp_path, monkeypatch):
    path = tmp_path / "opencode.json"
    monkeypatch.setattr(connect, "opencode_config_path", lambda: path)
    result = CliRunner().invoke(connect.connect_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "opencode" in result.output
    config = json.loads(path.read_text())
    assert config["provider"]["unsloth"]["options"]["apiKey"] == "sk-unsloth-feedfacefeedface"
    assert config["model"] == f"unsloth/{MODEL['id']}"
    assert not any(c[1].endswith("/api/inference/status") for c in fake_studio)


# ── Hermes (OpenAI /v1/chat/completions, key via env) ────────────────


@pytest.fixture()
def hermes_config(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(connect, "hermes_config_path", lambda: path)
    return path


def test_write_hermes_config_fresh(hermes_config):
    yaml = pytest.importorskip("yaml")
    connect.write_hermes_config(BASE, MODEL)
    config = yaml.safe_load(hermes_config.read_text())
    # Hermes only honors the key for a *named* custom provider, so the endpoint
    # is registered under providers.* and model.provider points at it.
    assert config["model"]["provider"] == "custom:unsloth"
    assert config["model"]["default"] == MODEL["id"]
    assert config["model"]["api_mode"] == "openai"
    provider = config["providers"]["unsloth"]
    assert provider["base_url"] == f"{BASE}/v1"
    assert provider["api_mode"] == "openai"
    assert provider["key_env"] == "UNSLOTH_API_KEY"
    # The key is resolved from the launch env, never written to disk.
    assert "sk-unsloth" not in hermes_config.read_text()


def test_write_hermes_config_preserves_and_idempotent(hermes_config):
    yaml = pytest.importorskip("yaml")
    hermes_config.write_text(
        yaml.safe_dump(
            {
                "terminal": {"backend": "local"},
                "model": {"temperature": 0.7},
                "providers": {"openrouter": {"base_url": "https://openrouter.ai/api/v1"}},
            }
        )
    )
    connect.write_hermes_config(BASE, MODEL)
    config = yaml.safe_load(hermes_config.read_text())
    assert config["terminal"] == {"backend": "local"}  # unrelated sections kept
    assert config["model"]["temperature"] == 0.7  # unrelated model keys kept
    assert config["model"]["provider"] == "custom:unsloth"
    assert config["providers"]["openrouter"]["base_url"] == "https://openrouter.ai/api/v1"
    assert config["providers"]["unsloth"]["base_url"] == f"{BASE}/v1"
    before = hermes_config.read_text()
    connect.write_hermes_config(BASE, MODEL)
    assert hermes_config.read_text() == before


def test_write_hermes_config_preserves_non_mapping_file(hermes_config, capsys):
    pytest.importorskip("yaml")
    original = "- just\n- a\n- list\n"  # valid YAML, but not a mapping
    hermes_config.write_text(original)
    connect.write_hermes_config(BASE, MODEL)
    assert hermes_config.read_text() == original  # user-managed file left untouched
    assert "couldn't parse" in capsys.readouterr().err


def test_connect_hermes_no_launch(fake_studio, hermes_config):
    yaml = pytest.importorskip("yaml")
    result = CliRunner().invoke(connect.connect_app, ["hermes", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "export UNSLOTH_API_KEY=sk-unsloth-feedfacefeedface" in result.output
    assert "hermes" in result.output
    config = yaml.safe_load(hermes_config.read_text())
    assert config["model"]["provider"] == "custom:unsloth"
    assert config["providers"]["unsloth"]["base_url"] == f"{BASE}/v1"
    assert config["model"]["default"] == MODEL["id"]
    assert not any(c[1].endswith("/api/inference/status") for c in fake_studio)
