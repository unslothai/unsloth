# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `unsloth connect` — config merging and launch env, no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
        "[model_providers.ollama]\n"
        'base_url = "http://localhost:11434/v1"\n'
    )
    merged = connect._merge_codex_config(existing, BASE)
    parsed = _parse_toml(merged)
    assert parsed["model"] == "gpt-5"
    assert parsed["model_providers"]["unsloth_api"]["base_url"] == f"{BASE}/v1"
    assert parsed["model_providers"]["unsloth_api"]["wire_api"] == "responses"
    assert parsed["model_providers"]["ollama"]["base_url"] == "http://localhost:11434/v1"
    assert connect._merge_codex_config(merged, BASE) == merged


def test_merge_codex_config_keeps_user_oss_provider():
    merged = connect._merge_codex_config('oss_provider = "ollama"\n', BASE)
    assert _parse_toml(merged)["oss_provider"] == "ollama"


def test_write_codex_config_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    connect.write_codex_config(BASE, MODEL)
    profile = _parse_toml((tmp_path / "unsloth_api.config.toml").read_text())
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
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.delenv("UNSLOTH_API_KEY", raising = False)
    return calls


def test_connect_claude_no_launch(fake_studio, claude_settings):
    result = CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert f'export ANTHROPIC_BASE_URL="{BASE}"' in result.output
    assert 'export ANTHROPIC_AUTH_TOKEN="sk-unsloth-feedfacefeedface"' in result.output
    assert f'export ANTHROPIC_MODEL="{MODEL["id"]}"' in result.output
    assert f"claude --model {MODEL['id']}" in result.output
    settings = json.loads(claude_settings.read_text())
    assert settings["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"


def test_connect_codex_no_launch(fake_studio, tmp_path):
    result = CliRunner().invoke(connect.connect_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert 'export UNSLOTH_STUDIO_AUTH_TOKEN="sk-unsloth-feedfacefeedface"' in result.output
    assert "codex --oss --profile unsloth_api" in result.output
    assert (tmp_path / "codex" / "config.toml").exists()
    assert (tmp_path / "codex" / "unsloth_api.config.toml").exists()


def test_connect_key_minted_once_then_cached(fake_studio, tmp_path):
    CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    CliRunner().invoke(connect.connect_app, ["claude", "--no-launch"])
    mints = [c for c in fake_studio if c[1].endswith("/api/auth/api-keys")]
    assert len(mints) == 1
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    assert cached["key"] == "sk-unsloth-feedfacefeedface"


def test_connect_model_flag_loads_on_server(fake_studio):
    result = CliRunner().invoke(
        connect.connect_app, ["claude", "--no-launch", "--model", "unsloth/Qwen3.5-35B-A3B"]
    )
    assert result.exit_code == 0, result.output
    loads = [c for c in fake_studio if c[1].endswith("/api/inference/load")]
    assert loads == [
        ("POST", f"{BASE}/api/inference/load", {"model_path": "unsloth/Qwen3.5-35B-A3B"})
    ]
    assert 'export ANTHROPIC_MODEL="unsloth/Qwen3.5-35B-A3B"' in result.output


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
    assert 'export ANTHROPIC_AUTH_TOKEN="sk-unsloth-deadbeefdeadbeef"' in result.output
    assert not any(c[1].endswith("/api/auth/api-keys") for c in fake_studio)
