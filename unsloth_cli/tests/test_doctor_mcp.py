# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli
import unsloth_cli.commands.doctor as doctor_mod
import unsloth_cli.commands.mcp as mcp_mod

CANARY = "sk-unsloth-LEAKCANARY0123456789"


@pytest.fixture
def env(tmp_path, monkeypatch):
    for var, sub in (("HOME", "home"), ("CLAUDE_CONFIG_DIR", "claude"), ("CODEX_HOME", "codex")):
        (tmp_path / sub).mkdir()
        monkeypatch.setenv(var, str(tmp_path / sub))
    for var in ("UNSLOTH_STUDIO_URL", "MCP_TOOL_TIMEOUT", "WSL_DISTRO_NAME"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("UNSLOTH_API_KEY", CANARY)
    return tmp_path


def _facts(**kw):
    base = dict(base = "http://127.0.0.1:8888", key = "valid", mcp = "on")
    base.update(kw)
    return mcp_mod.McpFacts(**base)


@pytest.mark.parametrize(
    "facts, needle",
    [
        (_facts(base = None, key = "missing", mcp = "unknown"), "not reachable"),
        (_facts(key = "missing", mcp = "refused"), "Set UNSLOTH_API_KEY"),
        (_facts(key = "invalid", mcp = "refused"), "refused the key in UNSLOTH_API_KEY"),
        (_facts(key = "error", key_detail = "500 boom"), "500 boom"),
        (_facts(mcp = "off"), "Allow agent connections"),
        (_facts(mcp = "absent"), "unsloth studio update"),
        (_facts(mcp = "refused", mcp_detail = "Workflow keys cannot use"), "Workflow keys cannot use"),
        (_facts(mcp = "error", mcp_detail = "timed out"), "timed out"),
        (_facts(key = "unsent"), "UNSLOTH_STUDIO_URL"),
    ],
)
def test_verdict_problems(facts, needle):
    healthy, hints, _ = mcp_mod.mcp_verdict(facts)
    assert not healthy
    assert any(needle in h for h in hints), hints


def test_verdict_ok_with_timeout_advice():
    facts = _facts(
        timeouts = [
            ("Codex unsloth-studio", 60, "Set tool_timeout_sec = 300."),
            ("Codex other", 300, "x"),
            ("Claude Code", 60.0, "Set MCP_TOOL_TIMEOUT=300000."),
        ]
    )
    healthy, hints, advice = mcp_mod.mcp_verdict(facts)
    assert healthy and hints == []
    assert len(advice) == 2
    assert "60 s" in advice[0] and "tool_timeout_sec = 300" in advice[0]
    assert "MCP_TOOL_TIMEOUT=300000" in advice[1]


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_http(
    monkeypatch,
    answers,
    base = "http://127.0.0.1:8888",
):
    """answers: path suffix -> status. Records (url, Authorization header)."""
    seen = []
    monkeypatch.setattr(mcp_mod, "find_studio_server", lambda *a, **k: base)

    def opener(request, timeout):
        seen.append((request.full_url, request.get_header("Authorization")))
        for suffix, status in answers.items():
            if request.full_url.endswith(suffix) and status != 200:
                body = io.BytesIO(json.dumps({"detail": f"nope {status}"}).encode())
                raise urllib.error.HTTPError(request.full_url, status, "x", {}, body)
        return _Resp()

    monkeypatch.setattr(mcp_mod, "urlopen_no_redirect", opener)
    return seen


def test_gather_sends_the_key_to_loopback(env, monkeypatch):
    seen = _stub_http(monkeypatch, {"/loaded-models": 200, "/mcp/": 200})
    facts = mcp_mod.gather_mcp_facts()
    assert (facts.key, facts.mcp) == ("valid", "on")
    assert [auth for _, auth in seen] == [f"Bearer {CANARY}"] * 2


def test_gather_reports_off_and_refused(env, monkeypatch):
    seen = _stub_http(monkeypatch, {"/loaded-models": 401, "/mcp/": 404})
    facts = mcp_mod.gather_mcp_facts()
    assert (facts.key, facts.mcp) == ("invalid", "off")
    # A refused key is not sent again.
    assert seen[1][1] is None
    _stub_http(monkeypatch, {"/loaded-models": 200, "/mcp/": 401})
    facts = mcp_mod.gather_mcp_facts()
    assert (facts.mcp, facts.mcp_detail) == ("refused", "nope 401")


def test_gather_never_sends_the_key_to_a_non_loopback_base(env, monkeypatch):
    seen = _stub_http(monkeypatch, {}, base = "http://10.0.0.5:8888")
    facts = mcp_mod.gather_mcp_facts()
    assert facts.key == "unsent"
    assert seen and all(auth is None for _, auth in seen)
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://10.0.0.5:8888")
    seen = _stub_http(monkeypatch, {}, base = "http://10.0.0.5:8888")
    assert mcp_mod.gather_mcp_facts().key == "valid"
    assert seen[0][1] == f"Bearer {CANARY}"


def test_gather_without_a_server_or_key(env, monkeypatch):
    monkeypatch.setattr(mcp_mod, "find_studio_server", lambda *a, **k: None)
    assert mcp_mod.gather_mcp_facts().base is None
    monkeypatch.delenv("UNSLOTH_API_KEY")
    seen = _stub_http(monkeypatch, {"/mcp/": 401})
    facts = mcp_mod.gather_mcp_facts()
    assert (facts.key, facts.mcp) == ("missing", "refused")
    assert [url for url, _ in seen] == ["http://127.0.0.1:8888/mcp/"]


def test_timeouts_read_from_codex_and_claude(env, monkeypatch):
    if mcp_mod._toml_loads() is None:
        pytest.skip("needs a TOML parser")
    (env / "codex" / "config.toml").write_text(
        '[mcp_servers.unsloth-studio]\nurl = "http://127.0.0.1:8888/mcp/"\n\n'
        '[mcp_servers.other]\nurl = "http://h/mcp/"\ntool_timeout_sec = 400\n\n'
        '[mcp_servers.unrelated]\ncommand = "x"\n'
    )
    found = mcp_mod._codex_timeouts("unsloth-studio")
    assert [(label, s) for label, s, _ in found] == [
        ("Codex unsloth-studio", 60),
        ("Codex other", 400),
    ]
    assert mcp_mod._claude_timeouts("unsloth-studio") == []
    (env / "claude" / ".claude.json").write_text(
        json.dumps({"mcpServers": {"unsloth-studio": {"url": "http://h/mcp/"}}})
    )
    (env / "claude" / "settings.json").write_text(
        json.dumps({"env": {"MCP_TOOL_TIMEOUT": "60000"}})
    )
    assert mcp_mod._claude_timeouts("unsloth-studio")[0][:2] == ("Claude Code", 60.0)
    monkeypatch.setenv("MCP_TOOL_TIMEOUT", "300000")
    assert mcp_mod._claude_timeouts("unsloth-studio")[0][:2] == ("Claude Code", 300.0)


def _cli():
    cli = typer.Typer()
    cli.command()(doctor_mod.doctor)
    return cli


def test_doctor_mcp_skips_gpus_and_exits_1_on_problems(env, monkeypatch):
    _stub_http(monkeypatch, {"/loaded-models": 200, "/mcp/": 404})
    monkeypatch.setattr(doctor_mod, "_gpu_checks", lambda t: pytest.fail("GPU checks ran"))
    result = CliRunner().invoke(_cli(), ["--mcp"])
    assert result.exit_code == 1, result.output
    assert "=== Unsloth Studio MCP ===" in result.output
    assert "API key (UNSLOTH_API_KEY): set, accepted" in result.output
    assert "Verdict: PROBLEMS FOUND" in result.output
    assert "LEAKCANARY" not in result.output and "sk-unsloth" not in result.output


def test_doctor_mcp_ok_with_advice(env, monkeypatch):
    _stub_http(monkeypatch, {})
    (env / "codex" / "config.toml").write_text('[mcp_servers.unsloth-studio]\nurl = "u"\n')
    result = CliRunner().invoke(_cli(), ["--mcp", "--no-gpus"])
    assert result.exit_code == 0, result.output
    assert "Codex unsloth-studio tool timeout: 60 s" in result.output
    assert "Verdict: OK" in result.output and "tool_timeout_sec = 300" in result.output
    assert "LEAKCANARY" not in result.output


@pytest.mark.parametrize("gpu_code, mcp_code", [(0, 1), (1, 0), (0, 0)])
def test_doctor_mcp_with_gpus_runs_both_and_keeps_the_worst_code(monkeypatch, gpu_code, mcp_code):
    ran = []
    monkeypatch.setattr(mcp_mod, "mcp_doctor", lambda: ran.append("mcp") or mcp_code)

    def gpu_checks(timeout):
        ran.append("gpus")
        raise typer.Exit(gpu_code)

    monkeypatch.setattr(doctor_mod, "_gpu_checks", gpu_checks)
    result = CliRunner().invoke(_cli(), ["--mcp", "--gpus"])
    assert ran == ["mcp", "gpus"]
    assert result.exit_code == max(gpu_code, mcp_code)


def test_plain_doctor_is_unchanged(monkeypatch):
    monkeypatch.setattr(mcp_mod, "mcp_doctor", lambda: pytest.fail("MCP checks ran"))
    result = CliRunner().invoke(_cli(), ["--no-gpus"])
    assert result.exit_code == 0 and result.output == "Nothing to check.\n"
    ran = []
    monkeypatch.setattr(doctor_mod, "_gpu_checks", lambda t: ran.append(t))
    assert CliRunner().invoke(_cli(), []).exit_code == 0
    assert ran == [60.0]
    help_text = CliRunner().invoke(unsloth_cli.app, ["doctor", "--help"]).output
    assert "--gpus" in help_text and "--mcp" in help_text
