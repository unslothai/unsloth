# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Agent attachment, authentication and survivor probes use resident discovery."""

import io
import urllib.error

import pytest
import typer

import unsloth_cli.commands.start as start


BASE = "http://127.0.0.1:8888"


@pytest.mark.parametrize("operation", ["attach", "key", "survivor"])
def test_discovery_avoids_full_catalog(monkeypatch, operation):
    calls = []

    def http_json(method, url, token, **kwargs):
        calls.append((url, kwargs["timeout"]))
        assert url == BASE + "/api/inference/loaded-models"
        return {"data": [{"id": "org/Model", "loaded": True, "context_length": 4096}]}

    monkeypatch.setattr(start, "_http_json", http_json)
    if operation == "attach":
        assert start._resolve_model(BASE, "key", None)["context_length"] == 4096
    elif operation == "key":
        assert start._key_accepted(BASE, "key")
    else:
        assert start._model_still_loaded(BASE, "key", "org/Model")
    assert calls == [(BASE + "/api/inference/loaded-models", 5 if operation == "survivor" else 30)]


def test_old_server_falls_back_to_compat_listing(monkeypatch):
    calls = []

    def http_json(method, url, token, **kwargs):
        calls.append((url, kwargs["timeout"]))
        if url.endswith("/loaded-models"):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return {"data": [{"id": "org/Model"}]}

    monkeypatch.setattr(start, "_http_json", http_json)
    assert start._model_still_loaded(BASE, "key", "org/Model")
    assert calls == [(BASE + "/api/inference/loaded-models", 5), (BASE + "/v1/models", 5)]


@pytest.mark.parametrize("code", [401, 403, 500, 503])
def test_discovery_does_not_hide_auth_or_server_errors(monkeypatch, code, capsys):
    calls = []

    def http_json(method, url, token, **kwargs):
        calls.append(url)
        raise urllib.error.HTTPError(
            url, code, "failed", {}, io.BytesIO(b'{"detail":"discovery unavailable"}')
        )

    monkeypatch.setattr(start, "_http_json", http_json)
    with pytest.raises(typer.Exit):
        start._loaded_models(BASE, "key")
    assert "Couldn't list models: discovery unavailable" in capsys.readouterr().err
    assert calls == [BASE + "/api/inference/loaded-models"]


def test_discovery_timeout_is_not_retried_as_old_server(monkeypatch, capsys):
    calls = []

    def http_json(method, url, token, **kwargs):
        calls.append(url)
        raise TimeoutError("timed out")

    monkeypatch.setattr(start, "_http_json", http_json)
    with pytest.raises(typer.Exit):
        start._loaded_models(BASE, "key")
    assert "Couldn't list models: timed out" in capsys.readouterr().err
    assert calls == [BASE + "/api/inference/loaded-models"]
