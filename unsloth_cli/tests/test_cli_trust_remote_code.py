# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# `unsloth train` gained --trust-remote-code in #16332; inference and chat load through a different
# path (a local orchestrator or a running server), so the flag has to reach both of those too.

from __future__ import annotations

import sys
import types

import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli.commands.chat as chatmod
import unsloth_cli.commands.inference as infermod
from unsloth_cli import _inference as inference


def _app(command):
    cli = typer.Typer()
    cli.command()(command)
    return cli


@pytest.mark.parametrize("command", [infermod.inference, chatmod.chat])
def test_both_commands_offer_the_flag(command):
    # A narrow console truncates the option name column, hiding the flag's last characters.
    result = CliRunner().invoke(_app(command), ["--help"], env = {"COLUMNS": "200", "TERM": "dumb"})

    assert result.exit_code == 0, result.output
    assert "--trust-remote-code" in result.output


class _FakeModelConfig:
    is_gguf = False


def _install_fake_backend(monkeypatch):
    calls: list[dict] = []

    class _FakeBackend:
        def load_model(self, **kwargs):
            calls.append(kwargs)
            return True

    fake_inference = types.ModuleType("core.inference")
    fake_inference.get_inference_backend = lambda: _FakeBackend()
    monkeypatch.setitem(sys.modules, "core", types.ModuleType("core"))
    monkeypatch.setitem(sys.modules, "core.inference", fake_inference)
    monkeypatch.setattr(inference, "ensure_studio_backend_path", lambda: None)
    return calls


def test_local_load_forwards_the_flag(monkeypatch):
    calls = _install_fake_backend(monkeypatch)

    inference.load_chat_backend(
        "fake-model",
        hf_token = None,
        max_seq_length = 2048,
        load_in_4bit = True,
        model_config = _FakeModelConfig(),
        trust_remote_code = True,
    )

    assert calls[0]["trust_remote_code"] is True


def test_local_load_defaults_to_refusing_remote_code(monkeypatch):
    calls = _install_fake_backend(monkeypatch)

    inference.load_chat_backend(
        "fake-model",
        hf_token = None,
        max_seq_length = 2048,
        load_in_4bit = True,
        model_config = _FakeModelConfig(),
    )

    # Omitted rather than passed as False, so every other load call is unchanged.
    assert "trust_remote_code" not in calls[0]


def _capture_server_payload(monkeypatch):
    captured: dict = {}

    def fake_request(self, method, path, payload = None, timeout = None):
        if method == "POST" and path.endswith("/load"):
            captured["payload"] = payload
        return object()

    monkeypatch.setattr(inference.HttpChatBackend, "_request", fake_request)
    monkeypatch.setattr(inference.HttpChatBackend, "_resident_gguf_variant", lambda self, model: None)
    monkeypatch.setattr(
        inference, "read_json_checking_deferred_error", lambda url, response: {"is_vision": False}
    )
    return captured


def test_server_payload_carries_the_flag_when_asked(monkeypatch):
    captured = _capture_server_payload(monkeypatch)
    backend = inference.HttpChatBackend("http://127.0.0.1:1", "token")

    backend.ensure_loaded(
        "m", hf_token = None, max_seq_length = 2048, load_in_4bit = True, trust_remote_code = True
    )

    assert captured["payload"]["trust_remote_code"] is True


def test_server_payload_is_unchanged_by_default(monkeypatch):
    captured = _capture_server_payload(monkeypatch)
    backend = inference.HttpChatBackend("http://127.0.0.1:1", "token")

    backend.ensure_loaded("m", hf_token = None, max_seq_length = 2048, load_in_4bit = True)

    # No key at all, so an existing payload stays byte-identical for everyone else.
    assert "trust_remote_code" not in captured["payload"]
