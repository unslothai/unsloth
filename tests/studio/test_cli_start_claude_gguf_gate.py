from __future__ import annotations

import inspect

import pytest
import typer

import unsloth_cli.commands.start as start


@pytest.fixture(autouse = True)
def _no_hub_lookups(monkeypatch):
    monkeypatch.setattr(start, "_is_hub_model_id", lambda name: True)
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: ["model-Q4_K_M.gguf"])


def _message(exc_info, capsys):
    assert exc_info.value.exit_code == 1
    return capsys.readouterr().err.strip()


def test_codex_wording_is_unchanged(capsys):
    with pytest.raises(typer.Exit) as exc:
        start._fail_agent_needs_gguf(start._CODEX_GGUF_AGENT, "unsloth/gemma-3-4b-it")
    assert _message(exc, capsys) == (
        "Codex needs a GGUF model served by llama-server, but unsloth/gemma-3-4b-it "
        "is not one. Try: unsloth start codex --model unsloth/gemma-3-4b-it-GGUF"
    )


def test_claude_names_itself_and_its_own_subcommand(capsys):
    with pytest.raises(typer.Exit) as exc:
        start._fail_agent_needs_gguf(start._CLAUDE_GGUF_AGENT, "unsloth/gemma-3-4b-it")
    assert _message(exc, capsys) == (
        "Claude Code needs a GGUF model served by llama-server, but unsloth/gemma-3-4b-it "
        "is not one. Try: unsloth start claude --model unsloth/gemma-3-4b-it-GGUF"
    )


def test_require_gguf_rejects_a_transformers_backend_model(monkeypatch, capsys):
    monkeypatch.setattr(start, "_http_json", lambda *a, **k: {"is_gguf": False})
    with pytest.raises(typer.Exit) as exc:
        start._require_gguf_for_agent(
            start._CLAUDE_GGUF_AGENT, "http://127.0.0.1:8888", "key", "unsloth/gemma-3-4b-it"
        )
    assert "Claude Code needs a GGUF model" in _message(exc, capsys)


def test_require_gguf_accepts_a_gguf_model(monkeypatch):
    monkeypatch.setattr(start, "_http_json", lambda *a, **k: {"is_gguf": True})
    start._require_gguf_for_agent(
        start._CLAUDE_GGUF_AGENT, "http://127.0.0.1:8888", "key", "unsloth/gemma-3-4b-it-GGUF"
    )


def test_claude_command_wires_all_three_gates():
    source = inspect.getsource(start.claude)
    assert (
        "_preflight_agent_gguf(_CLAUDE_GGUF_AGENT, model, serve = serve, launch = launch)" in source
    )
    assert "preload_check = functools.partial(_attach_gguf_check, _CLAUDE_GGUF_AGENT)" in source
    assert '_require_gguf_for_agent(_CLAUDE_GGUF_AGENT, base, key, entry["id"])' in source
    assert "_shutdown_auto_served()" in source
