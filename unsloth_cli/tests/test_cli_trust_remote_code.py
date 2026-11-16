# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import types

import pytest
import typer
from typer.main import get_command

import unsloth_cli.commands.chat as chatmod
import unsloth_cli.commands.inference as infermod
from unsloth_cli import _inference as inference


def _app(command):
    cli = typer.Typer()
    cli.command()(command)
    return cli


@pytest.mark.parametrize("command", [infermod.inference, chatmod.chat])
def test_both_commands_offer_the_flag(command):
    # Registered options, not --help: a narrow terminal truncates the option column.
    cmd = get_command(_app(command))
    names = [
        opt
        for param in cmd.params
        for opt in (getattr(param, "opts", []) or []) + (getattr(param, "secondary_opts", []) or [])
    ]

    assert "--trust-remote-code" in names


class _FakeModelConfig:
    is_gguf = False


_REFUSAL = "Please pass the argument `trust_remote_code=True` to allow custom code to be run."


def _install_fake_backend(monkeypatch, error = None):
    calls: list[dict] = []

    class _FakeBackend:
        def load_model(self, **kwargs):
            calls.append(kwargs)
            if error is not None:
                raise error
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

    assert "trust_remote_code" not in calls[0]


def _capture_server_payload(monkeypatch):
    captured: dict = {}

    def fake_request(
        self,
        method,
        path,
        payload = None,
        timeout = None,
    ):
        if method == "POST" and path.endswith("/load"):
            captured["payload"] = payload
        return object()

    monkeypatch.setattr(inference.HttpChatBackend, "_request", fake_request)
    monkeypatch.setattr(
        inference.HttpChatBackend, "_resident_gguf_variant", lambda self, model: None
    )
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

    assert "trust_remote_code" not in captured["payload"]


def _local_load(**kwargs):
    inference.load_chat_backend(
        "fake-model",
        hf_token = None,
        max_seq_length = 2048,
        load_in_4bit = True,
        model_config = _FakeModelConfig(),
        **kwargs,
    )


def test_local_refusal_points_at_the_flag(monkeypatch, capsys):
    _install_fake_backend(monkeypatch, error = ValueError(_REFUSAL))

    with pytest.raises(typer.Exit):
        _local_load()

    assert "--trust-remote-code" in capsys.readouterr().err


def test_local_failure_with_the_flag_keeps_its_traceback(monkeypatch):
    _install_fake_backend(monkeypatch, error = ValueError(_REFUSAL))

    with pytest.raises(ValueError):
        _local_load(trust_remote_code = True)


def test_unrelated_local_failure_is_not_rewritten(monkeypatch):
    _install_fake_backend(monkeypatch, error = RuntimeError("out of memory"))

    with pytest.raises(RuntimeError):
        _local_load()


def _http_error(status, reason, detail):
    import email.message
    import io
    import json
    import urllib.error

    body = io.BytesIO(json.dumps({"detail": detail}).encode())
    return urllib.error.HTTPError(
        "http://x/api/inference/load", status, reason, email.message.Message(), body
    )


def _refusing_server(monkeypatch, error):
    def refuse(
        self,
        method,
        path,
        payload = None,
        timeout = None,
    ):
        if method == "POST" and path.endswith("/load"):
            raise error
        return object()

    monkeypatch.setattr(inference.HttpChatBackend, "_request", refuse)
    monkeypatch.setattr(
        inference.HttpChatBackend, "_resident_gguf_variant", lambda self, model: None
    )
    return inference.HttpChatBackend("http://127.0.0.1:1", "token")


_SERVER_DETAIL = "Failed to load model: ... Please pass the argument `trust_remote_code=True` ..."


@pytest.mark.parametrize("status,reason", [(500, "Internal Server Error"), (400, "Bad Request")])
def test_fast_server_refusal_points_at_the_flag(monkeypatch, capsys, status, reason):
    backend = _refusing_server(monkeypatch, _http_error(status, reason, _SERVER_DETAIL))

    with pytest.raises(typer.Exit):
        backend.ensure_loaded("m", hf_token = None, max_seq_length = 2048, load_in_4bit = True)

    assert "--trust-remote-code" in capsys.readouterr().err


def test_deferred_server_refusal_points_at_the_flag(monkeypatch, capsys):
    _capture_server_payload(monkeypatch)

    def deferred(url, response):
        inference.raise_for_deferred_error(
            url, {"_deferred_error": {"status_code": 500, "detail": _SERVER_DETAIL}}
        )

    monkeypatch.setattr(inference, "read_json_checking_deferred_error", deferred)
    backend = inference.HttpChatBackend("http://127.0.0.1:1", "token")

    with pytest.raises(typer.Exit):
        backend.ensure_loaded("m", hf_token = None, max_seq_length = 2048, load_in_4bit = True)

    assert "--trust-remote-code" in capsys.readouterr().err


def test_other_server_failures_show_the_servers_cause(monkeypatch, capsys):
    backend = _refusing_server(
        monkeypatch,
        _http_error(500, "Internal Server Error", "Failed to load model: out of memory"),
    )

    with pytest.raises(typer.Exit):
        backend.ensure_loaded(
            "m", hf_token = None, max_seq_length = 2048, load_in_4bit = True, trust_remote_code = True
        )

    assert (
        "Model load failed: HTTP Error 500: Failed to load model: out of memory"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize("kind", ["remote_code_blocked", "malware_blocked"])
def test_a_scan_refusal_wins_over_the_flag(monkeypatch, capsys, kind):
    refusal = Exception("Model 'm' ships custom code flagged as CRITICAL by the security scan.")
    refusal.error_kind = kind
    _install_fake_backend(monkeypatch, error = refusal)

    with pytest.raises(typer.Exit):
        _local_load(trust_remote_code = True)

    err = capsys.readouterr().err
    assert "flagged as CRITICAL" in err and "never overrides it" in err
