# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD NPU models from `unsloth run`, `unsloth chat` and `unsloth start`."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli._npu import NpuCliError, npu_model_path, prepare_npu_model


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("lemonade:qwen3-0.6b-FLM", "lemonade:qwen3-0.6b-FLM"),
        ("qwen3-0.6b-FLM", "lemonade:qwen3-0.6b-FLM"),
        ("qwen3.8-mtp-27b-flm", "lemonade:qwen3.8-mtp-27b-flm"),
        ("unsloth/Qwen3-0.6B-GGUF", None),
        ("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", None),
        ("org/name-FLM", None),
        ("lemonade:", None),
        ("", None),
        (None, None),
    ],
)
def test_npu_model_path(model, expected):
    assert npu_model_path(model) == expected


class _Studio:
    """A Studio's /api/npu routes, answering `prepare_npu_model`'s requests."""

    def __init__(
        self,
        *,
        supported = True,
        ready = False,
        downloaded = False,
        events = None,
    ):
        self.status = {"supported": supported, "ready": ready, "hardware": {"present": supported}}
        self.downloaded = downloaded
        self.events = (
            events
            if events is not None
            else [
                {"event": "progress", "percent": 0},
                {"event": "progress", "percent": 55},
                {"event": "complete", "percent": 100},
            ]
        )
        self.calls = []

    def __call__(self, method, path, payload, timeout):
        self.calls.append((method, path))
        if path == "/api/npu/status":
            return io.BytesIO(json.dumps(self.status).encode())
        if path == "/api/npu/enable":
            self.status["ready"] = True
            return io.BytesIO(json.dumps(self.status).encode())
        if path == "/api/npu/models":
            models = [
                {"id": "qwen3-0.6b-FLM", "downloaded": self.downloaded, "size_gb": 0.66},
                {"id": "gemma3-1b-FLM", "downloaded": False, "size_gb": 1.2},
            ]
            return io.BytesIO(json.dumps({"models": models}).encode())
        if path.endswith("/download"):
            return io.BytesIO("".join(f"data: {json.dumps(e)}\n\n" for e in self.events).encode())
        raise AssertionError(path)


def test_a_fresh_npu_is_enabled_and_the_model_downloaded():
    studio, lines = _Studio(), []
    prepare_npu_model(studio, "lemonade:qwen3-0.6b-FLM", lines.append)
    assert studio.calls == [
        ("GET", "/api/npu/status"),
        ("POST", "/api/npu/enable"),
        ("GET", "/api/npu/models"),
        ("POST", "/api/npu/models/qwen3-0.6b-FLM/download"),
    ]
    assert lines[0].startswith("Enabling the NPU runtime")
    assert "  100%" in lines


def test_a_ready_downloaded_model_needs_nothing():
    studio = _Studio(ready = True, downloaded = True)
    prepare_npu_model(studio, "lemonade:qwen3-0.6b-FLM", lambda _line: None)
    assert studio.calls == [("GET", "/api/npu/status"), ("GET", "/api/npu/models")]


def test_no_npu_says_so():
    with pytest.raises(NpuCliError, match = "no AMD Ryzen AI NPU"):
        prepare_npu_model(_Studio(supported = False), "lemonade:qwen3-0.6b-FLM", print)


def test_an_unknown_model_lists_the_real_ones():
    with pytest.raises(NpuCliError, match = "gemma3-1b-FLM"):
        prepare_npu_model(_Studio(ready = True), "lemonade:nope-FLM", print)


@pytest.mark.parametrize(
    "events",
    [
        [{"event": "error", "error": "disk full"}],
        [{"event": "progress", "percent": 40}],
    ],
)
def test_a_failed_or_cut_download_fails(events):
    with pytest.raises(NpuCliError):
        prepare_npu_model(_Studio(ready = True, events = events), "lemonade:qwen3-0.6b-FLM", print)


@pytest.mark.parametrize(
    "error",
    [ConnectionResetError("reset"), TimeoutError("timed out"), OSError("refused")],
)
def test_a_lost_studio_connection_is_a_cli_error(error):
    # Only HTTP errors were converted, so a dropped connection surfaced as a traceback.
    def _request(*_args):
        raise error

    with pytest.raises(NpuCliError, match = "Lost the connection to Unsloth Studio"):
        prepare_npu_model(_request, "lemonade:qwen3-0.6b-FLM", print)


def test_a_running_studio_prepares_and_loads_a_bare_npu_id(monkeypatch):
    # `unsloth inference qwen3-0.6b-FLM` cut the prefix length off the bare id and loaded it as is.
    import unsloth_cli._inference as inference
    import unsloth_cli._npu as npu

    seen = {}

    class _Backend:
        def __init__(self, *_args):
            pass

        def ensure_loaded(self, model, **_kwargs):
            seen["loaded"] = model

    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888")
    monkeypatch.setattr(inference, "find_studio_server", lambda *a, **k: "http://127.0.0.1:8888")
    monkeypatch.setattr(inference, "verify_studio_identity", lambda *a, **k: True)
    monkeypatch.setattr(inference, "_studio_token", lambda: "token")
    monkeypatch.setattr(inference, "HttpChatBackend", _Backend)
    monkeypatch.setattr(
        npu, "prepare_npu_model", lambda _request, model, _echo: seen.setdefault("prepared", model)
    )
    inference.connect_studio_server(
        "qwen3-0.6b-FLM", hf_token = None, max_seq_length = 0, load_in_4bit = None
    )
    assert seen == {"prepared": "lemonade:qwen3-0.6b-FLM", "loaded": "lemonade:qwen3-0.6b-FLM"}


def test_run_refuses_gguf_flags_for_an_npu_model():
    from unsloth_cli import app

    result = CliRunner().invoke(
        app, ["run", "--model", "lemonade:qwen3-0.6b-FLM", "--gguf-variant", "Q4_K_M"]
    )
    assert result.exit_code == 2
    assert "NPU models run on FastFlowLM" in result.output


def test_run_accepts_verbose_for_an_npu_model(monkeypatch):
    # --verbose adds --log-verbose for llama-server, which then read as a user flag and exited 2.
    from unsloth_cli import app
    from unsloth_cli.commands import studio as studio_mod

    def _past_the_npu_checks(*_args, **_kwargs):
        raise typer.Exit(7)

    monkeypatch.setattr(studio_mod, "_enable_verbose_access_logs", lambda: None)
    monkeypatch.setattr(studio_mod, "_require_bind_host", _past_the_npu_checks)
    result = CliRunner().invoke(app, ["run", "--model", "lemonade:qwen3-0.6b-FLM", "--verbose"])
    assert result.exit_code == 7, result.output


def test_run_does_not_split_the_lemonade_prefix():
    import inspect

    from unsloth_cli.commands import studio as studio_mod

    src = inspect.getsource(studio_mod.run)
    # `lemonade:<id>` was split llama.cpp-style into repo "lemonade", variant "<id>".
    assert "(model, None) if npu_path else _split_repo_variant(model)" in src
    assert "prepare_npu_model(" in src


def test_start_names_an_npu_model_instead_of_looking_up_a_gguf(capsys, monkeypatch):
    from unsloth_cli.commands import start as start_mod

    # Never reach a real server: a Studio running on this machine would be asked to switch models.
    def _no_server(*_args, **_kwargs):
        raise AssertionError("unsloth start looked for a Studio before refusing the NPU model")

    monkeypatch.setattr(start_mod, "_require_studio", _no_server)
    with pytest.raises((typer.Exit, SystemExit)):
        start_mod._connect(None, "lemonade:qwen3-0.6b-FLM")
    assert "qwen3-0.6b-FLM is an AMD NPU model" in capsys.readouterr().err


def test_chat_needs_a_running_studio_for_an_npu_model(monkeypatch):
    from unsloth_cli import app
    from unsloth_cli.commands import chat as chat_mod

    monkeypatch.setattr(chat_mod, "connect_studio_server", lambda *_a, **_k: None)
    result = CliRunner().invoke(app, ["chat", "qwen3-0.6b-FLM"])
    assert result.exit_code == 1
    assert "runs on the AMD NPU inside Unsloth Studio" in result.output


@pytest.mark.parametrize(
    ("flags", "words"),
    [([], "runs on the AMD NPU inside Unsloth Studio"), (["--no-server"], "drop --no-server")],
)
def test_inference_needs_a_running_studio_for_an_npu_model(monkeypatch, flags, words):
    # Without a Studio the id fell through to a local load and a Hugging Face lookup.
    from unsloth_cli import app
    from unsloth_cli.commands import inference as inference_mod

    def _no_local_load(*_args, **_kwargs):
        raise AssertionError("an NPU model was loaded in-process")

    monkeypatch.setattr(inference_mod, "connect_studio_server", lambda *_a, **_k: None)
    monkeypatch.setattr(inference_mod, "load_chat_backend", _no_local_load)
    result = CliRunner().invoke(app, ["inference", "qwen3-0.6b-FLM", "hi", *flags])
    assert result.exit_code == 1, result.output
    assert words in result.output


def test_a_gguf_agent_refuses_an_npu_model_before_any_hub_lookup(capsys, monkeypatch):
    from unsloth_cli.commands import start as start_mod

    def _no_lookup(*_args, **_kwargs):
        raise AssertionError("looked the NPU id up as a Hugging Face GGUF repo")

    monkeypatch.setattr(start_mod, "find_studio_server", _no_lookup)
    monkeypatch.setattr(start_mod, "_hub_gguf_files", _no_lookup)
    with pytest.raises((typer.Exit, SystemExit)):
        start_mod._preflight_agent_gguf(start_mod._CLAUDE_GGUF_AGENT, "lemonade:qwen3-0.6b-FLM")
    assert "is an AMD NPU model" in capsys.readouterr().err


def test_the_run_banner_offers_no_endpoint_an_npu_model_refuses():
    import inspect

    from unsloth_cli.commands import studio as studio_mod

    # Structural: the banner prints after a live server load, which this suite cannot start.
    src = inspect.getsource(studio_mod.run)
    assert '"  OpenAI SDK base URL:" if npu_path' in src
    banner = src[src.index('typer.echo("OpenAI Chat Completions:")') :]
    assert banner.index("if not npu_path:") < banner.index('typer.echo("Anthropic Messages:")')
    assert banner.index("if not npu_path:") < banner.index('typer.echo("OpenAI Responses:")')
