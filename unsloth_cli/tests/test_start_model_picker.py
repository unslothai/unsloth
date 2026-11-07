# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import urllib.error
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli.commands.start as start
from unsloth_cli import _model_catalog as cat
from unsloth_cli import _model_picker as model_picker
from unsloth_cli.tests.test_start import BASE, GEMINI_CONNECTION, MODEL, fake_studio  # noqa: F401

GGUF = "/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf"
LOCAL_ID = "unsloth/gemma-3-270m-it"
ENTRIES = [
    cat.ModelEntry("Fine-tunes", "my-lora", "lora · Oct 30", "/outputs/my-lora"),
    cat.ModelEntry("Downloaded", "Qwen3-1.7B-GGUF", "Q4_K_M", GGUF),
    cat.ModelEntry("Downloaded", "gemma-3-270m-it", "", LOCAL_ID),
]
IDLE = {"active_model": None, "loaded": [], "loading": [], "load_in_flight": False}


@pytest.fixture()
def picker(monkeypatch):
    state = {"answers": [], "listed": 0, "prompts": [], "ensure_calls": []}

    def list_chat_models():
        state["listed"] += 1
        return list(ENTRIES)

    def answer(prompt = ""):
        state["prompts"].append(prompt)
        return state["answers"].pop(0)

    real_setup = start.ensure_studio_backend_path

    def ensure_studio_backend_path(**kwargs):
        state["ensure_calls"].append(kwargs)
        return real_setup(**kwargs)

    monkeypatch.setattr(cat, "list_chat_models", list_chat_models)
    monkeypatch.setattr(start, "ensure_studio_backend_path", ensure_studio_backend_path)
    # structlog.configure is process-wide; the other tests must keep their own log level.
    monkeypatch.setattr(model_picker, "quiet_inventory_logs", lambda: None)
    monkeypatch.setattr(start, "_can_ask_for_model", lambda: True)
    monkeypatch.setattr("builtins.input", answer)
    return state


def _route(monkeypatch, handlers):
    """Answer URL paths ending in a handler's key; everything else goes to the current _http_json."""
    previous = start._http_json

    def http(
        method,
        url,
        key,
        payload = None,
        *args,
        **kwargs,
    ):
        for suffix, handler in handlers.items():
            if url.split("?")[0].endswith(suffix):
                return handler(url, key, payload)
        return previous(method, url, key, payload, *args, **kwargs)

    monkeypatch.setattr(start, "_http_json", http)


def _empty_server(
    fake_studio,
    monkeypatch,
    status = None,
):
    fake_studio.state["models"] = []

    def chat_status(url, key, payload):
        fake_studio.append(("GET", url, payload))
        if isinstance(status, Exception):
            raise status
        return IDLE if status is None else status

    def media_status(url, key, payload):
        return fake_studio.state.get(url.rsplit("/", 2)[-2], {"loaded": False})

    previous = start._http_json

    def http(
        method,
        url,
        key,
        payload = None,
        *args,
        **kwargs,
    ):
        empty = not fake_studio.state["models"]
        if url.endswith(("/api/inference/images/status", "/api/inference/video/status")):
            return media_status(url, key, payload)
        if empty and url.split("?")[0].endswith("/api/inference/status"):
            return chat_status(url, key, payload)
        if empty and url.endswith("/api/inference/load"):
            # fake_studio compares against the first resident, which an empty server lacks.
            fake_studio.state["models"] = [{"id": "placeholder"}]
        return previous(method, url, key, payload, *args, **kwargs)

    monkeypatch.setattr(start, "_http_json", http)


def _loads(fake_studio):
    return [c[2] for c in fake_studio if c[1].endswith("/api/inference/load")]


def _autostart(monkeypatch):
    attempts = []
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)

    def require(model, load, **kwargs):
        attempts.append((model, load))
        raise typer.Exit(code = 7)

    monkeypatch.setattr(start, "_require_studio", require)
    return attempts


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


@pytest.mark.parametrize("error", [KeyboardInterrupt, EOFError])
def test_cancelling_the_prompt_loads_nothing(fake_studio, picker, monkeypatch, error):
    _empty_server(fake_studio, monkeypatch)

    def cancel(prompt = ""):
        raise error

    monkeypatch.setattr("builtins.input", cancel)

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 1
    assert _loads(fake_studio) == []


def test_gguf_agents_list_only_gguf_models(fake_studio, picker, monkeypatch):
    _empty_server(fake_studio, monkeypatch)
    picker["answers"] = ["1"]

    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert "1. Qwen3-1.7B-GGUF" in result.output
    assert "my-lora" not in result.output and "gemma-3-270m-it" not in result.output
    assert _loads(fake_studio) == [{"require_idle": True, "model_path": GGUF}]

    fake_studio.state["models"] = []
    monkeypatch.setattr(cat, "list_chat_models", lambda: [ENTRIES[0]])
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 1
    assert "No GGUF models found on this device" in result.output


@pytest.mark.parametrize("case", ["loaded", "explicit", "no-terminal"])
def test_never_prompts_without_reason(fake_studio, picker, monkeypatch, case):
    args = ["opencode", "--no-launch"]
    if case != "loaded":
        _empty_server(fake_studio, monkeypatch)
    if case == "explicit":
        args += ["--model", "unsloth/Qwen3-4B-GGUF"]
    if case == "no-terminal":
        monkeypatch.setattr(start, "_can_ask_for_model", lambda: False)

    result = CliRunner().invoke(start.start_app, args)

    assert picker["listed"] == 0
    if case == "loaded":
        assert result.exit_code == 0 and f"model {MODEL['id']}" in result.output
        assert _loads(fake_studio) == []
    elif case == "explicit":
        assert result.exit_code == 0, result.output
        assert _loads(fake_studio) == [{"model_path": "unsloth/Qwen3-4B-GGUF"}]
    else:
        assert result.exit_code == 1
        assert "No model is loaded in Unsloth" in result.output
        assert "--model <hf-id-or-path>" in result.output


@pytest.mark.parametrize(
    "status, message",
    [
        ({"yours": False}, "belongs to another account"),
        (TimeoutError("status timed out"), None),
        (urllib.error.HTTPError("http://localhost/status", 503, "unavailable", None, None), None),
        ({"error": "API endpoint not found"}, None),
        ({}, None),
        ({"loaded": [], "loading": ["pending"], "load_in_flight": True}, None),
        ({"loaded": [], "loading": [], "load_in_flight": True}, None),
        ({"loaded": [], "loading": []}, None),
    ],
)
def test_picker_needs_a_verified_idle_server(fake_studio, picker, monkeypatch, status, message):
    _empty_server(fake_studio, monkeypatch, status = status)

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 1
    assert picker["listed"] == 0
    assert _loads(fake_studio) == []
    if message:
        assert message in result.output


@pytest.mark.parametrize(
    "owner, when",
    [
        ("chat", "during"),
        ("images", "before"),
        ("images", "during"),
        ("video", "before"),
        ("video", "during"),
    ],
)
def test_picker_never_loads_over_another_accounts_model(
    fake_studio, picker, monkeypatch, owner, when
):
    status = dict(IDLE)
    _empty_server(fake_studio, monkeypatch, status = status)

    def take_over():
        if owner == "chat":
            status["yours"] = False
        else:
            fake_studio.state[owner] = {"loaded": True, "yours": False}

    if when == "before":
        take_over()

    def answer(prompt):
        take_over()
        return "2"

    monkeypatch.setattr("builtins.input", answer)
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 1
    assert "another account" in result.output
    assert picker["listed"] == (1 if when == "during" else 0)
    assert _loads(fake_studio) == []


def test_remote_server_never_lists_local_files(fake_studio, picker, monkeypatch):
    remote = "http://10.0.0.5:8888"
    _empty_server(fake_studio, monkeypatch)
    monkeypatch.setattr(start, "find_studio_server", lambda: remote)

    assert start._pick_model_for_idle_server(remote, "key", False) is None
    assert picker["listed"] == 0


def test_no_server_picks_the_model_to_auto_start(picker, monkeypatch):
    attempts = _autostart(monkeypatch)
    picker["answers"] = ["2"]

    with pytest.raises(typer.Exit):
        start._connect(None, None, serve = True, launch = True)

    assert [model for model, _ in attempts] == [GGUF]


@pytest.mark.parametrize(
    "serve, launch, url, running",
    [
        pytest.param(False, True, None, None, id = "no-serve"),
        pytest.param(True, False, None, None, id = "no-launch"),
        pytest.param(True, True, "http://10.0.0.5:8888", None, id = "remote-url"),
        pytest.param(True, True, "https://127.0.0.1:8888", None, id = "https"),
        pytest.param(True, True, None, BASE, id = "server-running"),
    ],
)
def test_no_server_asks_only_when_one_could_be_started(
    picker, monkeypatch, serve, launch, url, running
):
    monkeypatch.setattr(start, "find_studio_server", lambda: running)
    if url:
        monkeypatch.setenv("UNSLOTH_STUDIO_URL", url)
    else:
        monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)

    assert start._pick_model_to_auto_start(serve, launch, False) is None
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
        (GGUF, "Q4_K_M", True),
        ("unsloth/Qwen3-1.7B-GGUF", "Q4_K_M", True),
        ("/exports/my-gguf-export", "gguf", True),
        ("/models/gguf-tests/llama-hf", "", False),
        ("/outputs/run", "lora · gguf-chats · Oct 30", False),
        (LOCAL_ID, "", False),
        ("/outputs/my-run-GGUF", "lora", False),
        (r"C:\outputs\my-run-GGUF", "lora", False),
        ("C:/outputs/my-run-GGUF", "lora", False),
        ("./my-run-GGUF", "lora", False),
        ("{tmp}/my-run-GGUF", "lora", False),
        ("{tmp}/my-run.gguf", "lora", False),
    ],
)
def test_gguf_filter_matches_gguf_rows_only(tmp_path, model, detail, expected):
    if model.startswith("{tmp}/"):
        folder = tmp_path / model.removeprefix("{tmp}/")
        folder.mkdir()
        model = str(folder)
    assert model_picker._is_gguf_entry(cat.ModelEntry("Downloaded", "n", detail, model)) is expected


@pytest.mark.parametrize("agent", ["opencode", "pi", "hermes"])
def test_local_pick_wins_over_a_connection_with_the_same_id(
    fake_studio, picker, monkeypatch, agent
):
    _empty_server(fake_studio, monkeypatch)
    fake_studio.state["providers"] = [{**GEMINI_CONNECTION, "models": [LOCAL_ID]}]
    monkeypatch.setattr(cat, "list_chat_models", lambda: [ENTRIES[2]])
    picker["answers"] = ["1"]

    result = CliRunner().invoke(start.start_app, [agent, "--no-launch"])

    assert result.exit_code == 0, result.output
    assert _loads(fake_studio) == [{"require_idle": True, "model_path": LOCAL_ID}]
    assert "external::" not in result.output


@pytest.mark.parametrize("model", [LOCAL_ID, "external::c1::unsloth%2Fgemma-3-270m-it"])
def test_explicit_connection_still_works(fake_studio, picker, model):
    fake_studio.state["providers"] = [{**GEMINI_CONNECTION, "models": [LOCAL_ID]}]

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--model", model])

    assert result.exit_code == 0, result.output
    assert _loads(fake_studio) == []
    assert picker["listed"] == 0
    assert "external::c1::" in result.output


@pytest.mark.parametrize("state", ["idle", "foreign", "busy", "denied", "unverified", "unknown"])
def test_a_server_appearing_during_the_prompt_is_rechecked(fake_studio, picker, monkeypatch, state):
    status = {
        "foreign": {"yours": False, "loaded": [], "loading": []},
        "busy": {"loaded": ["resident"], "loading": []},
        "unknown": {},
    }.get(state)
    _empty_server(fake_studio, monkeypatch, status = status)
    checked = []

    def model_access(url, key, payload):
        checked.extend(payload)
        return {"allowed": [] if state == "denied" else payload}

    _route(monkeypatch, {"/api/auth/model-access": model_access})
    if state == "unverified":
        monkeypatch.setattr(start, "verify_studio_identity", lambda base: False)
    fake_studio.state["providers"] = [{**GEMINI_CONNECTION, "models": [LOCAL_ID]}]
    monkeypatch.setattr(cat, "list_chat_models", lambda: [ENTRIES[2]])
    discoveries = iter([None, "http://127.0.0.1:8888"])
    monkeypatch.setattr(start, "find_studio_server", lambda: next(discoveries))
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    picker["answers"] = ["1"]

    if state == "idle":
        result = start._connect(None, None, serve = True, launch = True, connections = True)
        assert result[2]["id"] == LOCAL_ID
        assert _loads(fake_studio) == [{"require_idle": True, "model_path": LOCAL_ID}]
        assert checked == [LOCAL_ID]
        return
    with pytest.raises(typer.Exit):
        start._connect(None, None, serve = True, launch = True, connections = True)
    assert _loads(fake_studio) == []
    if state == "unverified":
        assert checked == []
