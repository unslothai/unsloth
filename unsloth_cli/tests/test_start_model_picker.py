# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth start` with no --model and nothing loaded: a terminal run picks a model on the device."""

import urllib.error
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli.commands.start as start
from unsloth_cli import _model_catalog as cat
from unsloth_cli import _model_picker as model_picker
from unsloth_cli.tests.test_start import BASE, MODEL, fake_studio  # noqa: F401  (fixture)

GGUF = "/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf"
ENTRIES = [
    cat.ModelEntry("Fine-tunes", "my-lora", "lora · Oct 30", "/outputs/my-lora"),
    cat.ModelEntry("Downloaded", "Qwen3-1.7B-GGUF", "Q4_K_M", GGUF),
    cat.ModelEntry("Downloaded", "gemma-3-270m-it", "", "unsloth/gemma-3-270m-it"),
]


@pytest.fixture()
def picker(monkeypatch):
    """A terminal on both ends, the three ENTRIES on disk, and the answers fed to input()."""
    state = {"answers": [], "listed": 0}

    def list_chat_models():
        state["listed"] += 1
        return list(ENTRIES)

    def answer(prompt = ""):
        state.setdefault("prompts", []).append(prompt)
        return state["answers"].pop(0)

    monkeypatch.setattr(cat, "list_chat_models", list_chat_models)

    real_setup = start.ensure_studio_backend_path

    def ensure_studio_backend_path(**kwargs):
        state.setdefault("ensure_calls", []).append(kwargs)
        return real_setup(**kwargs)

    monkeypatch.setattr(start, "ensure_studio_backend_path", ensure_studio_backend_path)
    # structlog.configure is process-wide; the other tests must keep their own log level.
    monkeypatch.setattr(model_picker, "quiet_inventory_logs", lambda: None)
    monkeypatch.setattr(start, "_can_ask_for_model", lambda: True)
    monkeypatch.setattr("builtins.input", answer)
    return state


def _empty_server(
    fake_studio,
    monkeypatch,
    status = None,
):
    """Make fake_studio serve nothing until a load arrives."""
    original = start._http_json
    fake_studio.state["models"] = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith(("/api/inference/images/status", "/api/inference/video/status")):
            return fake_studio.state.get(url.rsplit("/", 2)[-2], {"loaded": False})
        if url.split("?")[0].endswith("/api/inference/status") and not fake_studio.state["models"]:
            fake_studio.append((method, url, payload))
            if isinstance(status, Exception):
                raise status
            return (
                {"active_model": None, "loaded": [], "loading": [], "load_in_flight": False}
                if status is None
                else status
            )
        if url.endswith("/api/inference/load") and not fake_studio.state["models"]:
            # fake_studio compares against the first resident, which an empty server lacks.
            fake_studio.state["models"] = [{"id": "placeholder"}]
        return original(method, url, token, payload = payload, timeout = timeout, error = error)

    monkeypatch.setattr(start, "_http_json", http_json)


def _loads(fake_studio):
    return [c[2] for c in fake_studio if c[1].endswith("/api/inference/load")]


def test_idle_server_loads_the_picked_model(fake_studio, picker, monkeypatch):
    _empty_server(fake_studio, monkeypatch)
    picker["answers"] = ["2"]

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert "No model is loaded in Unsloth. Pick one on this device to load:" in result.output
    assert "1. my-lora" in result.output and "3. gemma-3-270m-it" in result.output
    assert picker["prompts"] == ["Load [1-3, Enter = 1]: "]
    assert _loads(fake_studio) == [{"require_idle": True, "model_path": GGUF}]
    assert f"model {GGUF}" in result.output
    # Seeding would pin Unsloth's caches into the agent's env and an auto-started server's.
    assert picker["ensure_calls"] == [{"seed_cache_env": False}]


def test_enter_picks_the_first_model_and_bad_input_asks_again(fake_studio, picker, monkeypatch):
    _empty_server(fake_studio, monkeypatch)
    picker["answers"] = ["9", "two", "\u00b2", ""]

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert result.output.count("Pick a number between 1 and 3.") == 3
    assert _loads(fake_studio) == [{"require_idle": True, "model_path": "/outputs/my-lora"}]


def test_claude_lists_only_gguf_models(fake_studio, picker, monkeypatch):
    _empty_server(fake_studio, monkeypatch)
    picker["answers"] = ["1"]

    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert "1. Qwen3-1.7B-GGUF" in result.output
    assert "my-lora" not in result.output and "gemma-3-270m-it" not in result.output
    assert _loads(fake_studio) == [{"require_idle": True, "model_path": GGUF}]


def test_no_gguf_on_device_keeps_a_clear_error(fake_studio, picker, monkeypatch):
    _empty_server(fake_studio, monkeypatch)
    monkeypatch.setattr(cat, "list_chat_models", lambda: [ENTRIES[0]])

    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])

    assert result.exit_code == 1
    assert "No GGUF models found on this device" in result.output
    assert _loads(fake_studio) == []


def test_ctrl_c_at_the_prompt_loads_nothing(fake_studio, picker, monkeypatch):
    _empty_server(fake_studio, monkeypatch)

    def interrupt(prompt = ""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 1
    assert _loads(fake_studio) == []


def test_a_loaded_model_is_reused_without_asking(fake_studio, picker):
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert picker["listed"] == 0
    assert _loads(fake_studio) == []
    assert f"model {MODEL['id']}" in result.output


def test_explicit_model_never_asks(fake_studio, picker, monkeypatch):
    _empty_server(fake_studio, monkeypatch)

    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--model", "unsloth/Qwen3-4B-GGUF"]
    )

    assert result.exit_code == 0, result.output
    assert picker["listed"] == 0
    assert _loads(fake_studio) == [{"model_path": "unsloth/Qwen3-4B-GGUF"}]


def test_without_a_terminal_the_error_is_unchanged(fake_studio, picker, monkeypatch):
    _empty_server(fake_studio, monkeypatch)
    monkeypatch.setattr(start, "_can_ask_for_model", lambda: False)

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 1
    assert "No model is loaded in Unsloth" in result.output
    assert "--model <hf-id-or-path>" in result.output
    assert picker["listed"] == 0


def test_another_accounts_model_is_not_evicted_by_a_prompt(fake_studio, picker, monkeypatch):
    _empty_server(fake_studio, monkeypatch, status = {"yours": False})

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 1
    assert "belongs to another account" in result.output
    assert picker["listed"] == 0
    assert _loads(fake_studio) == []


@pytest.mark.parametrize(
    "status",
    [
        TimeoutError("status timed out"),
        urllib.error.HTTPError("http://localhost/status", 503, "unavailable", None, None),
        {"error": "API endpoint not found"},
        {},
        {"loaded": [], "loading": ["pending"], "load_in_flight": True},
        {"loaded": [], "loading": [], "load_in_flight": True},
        {"loaded": [], "loading": []},
    ],
)
def test_idle_picker_requires_verified_empty_status(fake_studio, picker, monkeypatch, status):
    _empty_server(fake_studio, monkeypatch, status = status)
    picker["answers"] = ["2"]

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert _loads(fake_studio) == []
    assert picker["listed"] == 0
    assert result.exit_code == 1


def test_idle_picker_rechecks_after_a_selection(fake_studio, picker, monkeypatch):
    status = {"loaded": [], "loading": [], "load_in_flight": False}
    _empty_server(fake_studio, monkeypatch, status = status)

    def answer(prompt):
        status["yours"] = False
        return "2"

    monkeypatch.setattr("builtins.input", answer)
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 1
    assert "belongs to another account" in result.output
    assert picker["listed"] == 1
    assert _loads(fake_studio) == []


@pytest.mark.parametrize("media", ["images", "video"])
@pytest.mark.parametrize("when", ["before-prompt", "during-prompt"])
def test_picker_never_loads_over_hidden_media(fake_studio, picker, monkeypatch, media, when):
    _empty_server(fake_studio, monkeypatch)
    hidden = {"loaded": True, "yours": False}
    if when == "before-prompt":
        fake_studio.state[media] = hidden

    def answer(prompt):
        fake_studio.state[media] = hidden
        return "2"

    monkeypatch.setattr("builtins.input", answer)
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert _loads(fake_studio) == []
    assert picker["listed"] == (1 if when == "during-prompt" else 0)
    assert result.exit_code == 1
    assert "another account" in result.output


def test_remote_server_never_lists_local_files(fake_studio, picker, monkeypatch):
    remote = "http://10.0.0.5:8888"
    _empty_server(fake_studio, monkeypatch)
    monkeypatch.setattr(start, "find_studio_server", lambda: remote)

    assert start._pick_model_for_idle_server(remote, "key", False) is None
    assert picker["listed"] == 0


def test_no_server_picks_the_model_to_auto_start(picker, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    seen = {}

    def require_studio(model, load, **kwargs):
        seen["model"] = model
        raise typer.Exit(code = 7)

    monkeypatch.setattr(start, "_require_studio", require_studio)
    picker["answers"] = ["2"]

    with pytest.raises(typer.Exit):
        start._connect(None, None, serve = True, launch = True)

    assert seen["model"] == GGUF


@pytest.mark.parametrize(
    "serve, launch, url",
    [
        pytest.param(False, True, None, id = "no-serve"),
        pytest.param(True, False, None, id = "no-launch"),
        pytest.param(True, True, "http://10.0.0.5:8888", id = "remote-url"),
        pytest.param(True, True, "https://127.0.0.1:8888", id = "https"),
    ],
)
def test_no_server_asks_only_when_one_could_be_started(picker, monkeypatch, serve, launch, url):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    if url:
        monkeypatch.setenv("UNSLOTH_STUDIO_URL", url)
    else:
        monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)

    assert start._pick_model_to_auto_start(serve, launch, False) is None
    assert picker["listed"] == 0


def test_running_server_is_left_to_the_attach_path(picker, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: BASE)
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)

    assert start._pick_model_to_auto_start(True, True, False) is None
    assert picker["listed"] == 0


@pytest.mark.parametrize(
    "stdin_tty, stdout_tty, expected",
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_asks_only_with_a_terminal_on_both_ends(monkeypatch, stdin_tty, stdout_tty, expected):
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: stdin_tty))
    monkeypatch.setattr(start.sys, "stdout", SimpleNamespace(isatty = lambda: stdout_tty))
    assert start._can_ask_for_model() is expected


@pytest.mark.parametrize(
    "model, detail, expected",
    [
        pytest.param(GGUF, "Q4_K_M", True, id = "downloaded-file"),
        pytest.param("unsloth/Qwen3-1.7B-GGUF", "Q4_K_M", True, id = "downloaded-repo-fallback"),
        pytest.param("/exports/my-gguf-export", "gguf", True, id = "gguf-export-folder"),
        pytest.param("/models/gguf-tests/llama-hf", "", False, id = "gguf-only-in-parent-dir"),
        pytest.param("/outputs/run", "lora · gguf-chats · Oct 30", False, id = "gguf-in-dataset"),
        pytest.param("unsloth/gemma-3-270m-it", "", False, id = "safetensors-repo"),
    ],
)
def test_gguf_filter_matches_gguf_rows_only(model, detail, expected):
    entry = cat.ModelEntry("Downloaded", "name", detail, model)
    assert model_picker._is_gguf_entry(entry) is expected
