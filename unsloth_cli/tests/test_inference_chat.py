# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the `unsloth chat` / `unsloth inference` CLI — fakes only, no model loads."""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import typer
from rich.console import Console
from typer.testing import CliRunner

import unsloth_cli.commands.chat as chatmod
from unsloth_cli._inference import (
    ChatBackend,
    HttpChatBackend,
    collect_stream,
    render_columns,
    visible_text,
)


class _FakeConfig:
    is_gguf = False
    is_lora = True
    display_name = "fake-model"
    base_model = "fake/base"
    path = None


def _chat_app():
    cli = typer.Typer()
    cli.command()(chatmod.chat)
    return cli


def test_visible_text_passthrough_when_shown():
    text = "<think>reasoning</think>answer"
    assert visible_text(text, show_thinking = True) == text


def test_visible_text_strips_closed_think_block():
    text = "<think>step 1\nstep 2</think>The answer is 42."
    assert visible_text(text, show_thinking = False) == "The answer is 42."


def test_visible_text_holds_unclosed_think():
    # An open <think> is held back so partial reasoning never leaks mid-stream.
    assert visible_text("<think>still thinking", show_thinking = False) == ""
    assert visible_text("done.<think>more thinking", show_thinking = False) == "done."


def test_visible_text_holds_partial_think_prefix():
    # Streams are cumulative, so the opening tag can arrive as "<", "<thi",
    # then "<think>". Hold possible tag prefixes until they are disambiguated.
    assert visible_text("<", show_thinking = False) == ""
    assert visible_text("<thi", show_thinking = False) == ""
    assert visible_text("done.<thi", show_thinking = False) == "done."
    assert visible_text("2 < 3", show_thinking = False) == "2 < 3"


def _option(command_fn, name):
    return inspect.signature(command_fn).parameters[name].default


def test_inference_think_defaults_off():
    from unsloth_cli.commands.inference import inference

    opt = _option(inference, "think")
    assert getattr(opt, "default", None) is False
    # typer stores a flag/--no-flag pair as one combined decl.
    assert "--think/--no-think" in (getattr(opt, "param_decls", None) or [])


def test_chat_command_is_registered_with_options():
    params = inspect.signature(chatmod.chat).parameters
    assert "model" in params

    think = _option(chatmod.chat, "think")
    assert "--think/--no-think" in (getattr(think, "param_decls", None) or [])

    compare = _option(chatmod.chat, "compare")
    assert "--compare/--no-compare" in (getattr(compare, "param_decls", None) or [])

    verbose = _option(chatmod.chat, "verbose")
    assert {"--verbose", "-v"} <= set(getattr(verbose, "param_decls", None) or [])


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def generate_chat_response(self, **kwargs):
        self.calls.append(("plain", None, kwargs))
        return iter(["hi"])

    def generate_with_adapter_control(self, *, use_adapter, **kwargs):
        self.calls.append(("adapter", use_adapter, kwargs))
        return iter(["hi"])


_STREAM_KWARGS = dict(
    system_prompt = "",
    temperature = 0.7,
    top_p = 0.9,
    top_k = 40,
    max_new_tokens = 8,
    repetition_penalty = 1.1,
    enable_thinking = False,
)


def test_chatbackend_routes_compare_to_adapter_control():
    fake = _FakeBackend()
    backend = ChatBackend("unsloth", fake)

    list(backend.stream([{"role": "user", "content": "x"}], use_adapter = False, **_STREAM_KWARGS))
    list(backend.stream([{"role": "user", "content": "x"}], use_adapter = True, **_STREAM_KWARGS))

    assert [(path, flag) for path, flag, _ in fake.calls] == [
        ("adapter", False),
        ("adapter", True),
    ]


def test_chatbackend_normal_path_skips_adapter_control():
    fake = _FakeBackend()
    backend = ChatBackend("unsloth", fake)

    list(backend.stream([{"role": "user", "content": "x"}], **_STREAM_KWARGS))

    assert fake.calls[0][0] == "plain"


def test_collect_stream_returns_last_cumulative_think_stripped():
    stream = iter(["<think>r</think>hel", "<think>r</think>hello"])
    assert collect_stream(stream, show_thinking = False) == "hello"


def test_render_columns_emits_both_answers_with_separator(capsys):
    render_columns("base", "alpha", "tuned", "beta")
    out = capsys.readouterr().out
    assert "base" in out and "tuned" in out
    assert "alpha" in out and "beta" in out
    assert "│" in out


def test_you_prompt_matches_readline_backend(monkeypatch):
    gnu = types.ModuleType("readline")
    gnu.__doc__ = "Importing this module enables command line editing using GNU readline."
    monkeypatch.setitem(sys.modules, "readline", gnu)
    prompt = chatmod._you_prompt(colors = True)
    assert "You: " in prompt and "\001" in prompt

    libedit = types.ModuleType("readline")
    libedit.__doc__ = "Importing this module enables command line editing using libedit readline."
    monkeypatch.setitem(sys.modules, "readline", libedit)
    assert chatmod._you_prompt(colors = True) == "\n\x1b[1;36mYou: \x1b[0m"
    assert chatmod._you_prompt(colors = False) == "\nYou: "

    # Windows: no readline module at all; the console's own line editing
    # handles backspace, so plain ANSI color (no markers) is safe.
    monkeypatch.setitem(sys.modules, "readline", None)
    assert chatmod._you_prompt(colors = True) == "\n\x1b[1;36mYou: \x1b[0m"
    assert chatmod._you_prompt(colors = False) == "\nYou: "


def test_chat_registered_on_app():
    from unsloth_cli import app

    # cmd.name is None until typer resolves it from the callback name.
    names = {(cmd.name or cmd.callback.__name__) for cmd in app.registered_commands}
    assert "chat" in names


def test_chat_exits_cleanly_on_slash_exit(monkeypatch):
    closed = []

    class _FakeChatBackend:
        def stream(self, *a, **k):
            return iter(["hello"])

        def close(self):
            closed.append(True)

    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(chatmod, "load_chat_backend", lambda *a, **k: _FakeChatBackend())
    monkeypatch.setattr(chatmod, "_compare_needs_second_model", lambda: False)
    monkeypatch.setattr(chatmod, "connect_studio_server", lambda *a, **k: None)

    runner = CliRunner()
    for args in (["fake-model"], ["fake-model", "--compare"]):
        closed.clear()
        result = runner.invoke(_chat_app(), args, input = "hi\n/exit\n")
        assert result.exit_code == 0, result.output
        assert closed == [True]
        assert "Bye." in result.output
        # The prompt must go through input() (readline-safe), not a print.
        assert "You: " in result.output
        assert "You: You:" not in result.output


def test_pick_trained_model_lists_and_selects(monkeypatch):
    fake_models = types.ModuleType("utils.models")
    fake_models.scan_trained_models = lambda: [
        ("run-new", "outputs/run-new", "lora"),
        ("run-old", "outputs/run-old", "merged"),
    ]
    monkeypatch.setitem(sys.modules, "utils.models", fake_models)

    monkeypatch.setattr("builtins.input", lambda prompt = "": "2")
    assert chatmod._pick_trained_model(Console()) == "outputs/run-old"

    monkeypatch.setattr("builtins.input", lambda prompt = "": "")
    assert chatmod._pick_trained_model(Console()) == "outputs/run-new"


def test_chat_no_arg_chats_with_picked_trained_model(monkeypatch):
    class _FakeChatBackend:
        def stream(self, *a, **k):
            return iter(["hello"])

        def close(self):
            pass

    resolved = []
    monkeypatch.setattr(chatmod, "_pick_trained_model", lambda console: "outputs/run-42")
    monkeypatch.setattr(
        chatmod,
        "resolve_model_config",
        lambda model, **k: (resolved.append(model), _FakeConfig())[1],
    )
    monkeypatch.setattr(chatmod, "load_chat_backend", lambda *a, **k: _FakeChatBackend())
    monkeypatch.setattr(chatmod, "_compare_needs_second_model", lambda: False)
    monkeypatch.setattr(chatmod, "connect_studio_server", lambda *a, **k: None)

    result = CliRunner().invoke(_chat_app(), [], input = "/exit\n")
    assert result.exit_code == 0, result.output
    assert resolved == ["outputs/run-42"]


def test_find_studio_server_none_when_not_running(monkeypatch):
    import urllib.request

    from unsloth_cli import _inference

    def refuse(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert _inference.find_studio_server() is None


class _FakeHealth:
    """Minimal urlopen() return value carrying a canned /api/health body."""

    def __init__(
        self,
        body: bytes,
        final_url: str = "http://127.0.0.1:8888",
    ):
        self._body = body
        self._final_url = final_url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, limit = None):
        return self._body if limit is None else self._body[:limit]

    def geturl(self):
        return self._final_url


def _healthy_body(root_id = None) -> bytes:
    import json

    body = {
        "status": "healthy",
        "service": "Unsloth UI Backend",
        "supports_desktop_auth": True,
    }
    if root_id is not None:
        body["studio_root_id"] = root_id
    return json.dumps(body).encode()


def test_find_studio_server_matches_local_install_id(monkeypatch):
    import urllib.request

    from unsloth_cli import _inference

    root_id = "a" * 64
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    monkeypatch.setattr(_inference, "_local_studio_install_id", lambda: root_id)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: _FakeHealth(_healthy_body(root_id))
    )
    assert _inference.find_studio_server() == "http://127.0.0.1:8888"


def test_find_studio_server_rejects_mismatched_install_id(monkeypatch):
    import urllib.request

    from unsloth_cli import _inference

    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    monkeypatch.setattr(_inference, "_local_studio_install_id", lambda: "a" * 64)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: _FakeHealth(_healthy_body("b" * 64))
    )
    assert _inference.find_studio_server() is None


def test_find_studio_server_rejects_non_studio_responder(monkeypatch):
    import urllib.request

    from unsloth_cli import _inference

    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    monkeypatch.setattr(_inference, "_local_studio_install_id", lambda: None)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _FakeHealth(b'{"status":"evil","studio_root_id":"spoof"}'),
    )
    assert _inference.find_studio_server() is None


def test_find_studio_server_dev_install_accepts_loopback_studio(monkeypatch):
    import urllib.request

    from unsloth_cli import _inference

    # No installer id to match against: a real loopback Studio is still adopted.
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    monkeypatch.setattr(_inference, "_local_studio_install_id", lambda: None)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: _FakeHealth(_healthy_body("c" * 64))
    )
    assert _inference.find_studio_server() == "http://127.0.0.1:8888"


def test_find_studio_server_honors_explicit_remote_url(monkeypatch):
    import urllib.request

    from unsloth_cli import _inference

    # Remote (non-loopback) Studio, e.g. a RunPod proxy: its install id won't
    # match a local one, so the id gate must not apply to remote hosts.
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "https://studio.example.com")
    monkeypatch.setattr(_inference, "_local_studio_install_id", lambda: "a" * 64)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: _FakeHealth(_healthy_body("b" * 64))
    )
    assert _inference.find_studio_server() == "https://studio.example.com"


def test_find_studio_server_checks_id_on_explicit_loopback_port(monkeypatch):
    import urllib.request

    from unsloth_cli import _inference

    # A custom *local* port still gets the id check: a squatter on 127.0.0.1:9000
    # must not receive credentials just because UNSLOTH_STUDIO_URL is set.
    port = "http://127.0.0.1:9000"
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", port)
    monkeypatch.setattr(_inference, "_local_studio_install_id", lambda: "a" * 64)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: _FakeHealth(_healthy_body("b" * 64), port)
    )
    assert _inference.find_studio_server() is None

    # The real local Studio on that port (matching id) is still adopted.
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: _FakeHealth(_healthy_body("a" * 64), port)
    )
    assert _inference.find_studio_server() == port


def test_find_studio_server_rejects_redirected_health(monkeypatch):
    import urllib.request

    from unsloth_cli import _inference

    # A squatter on 8888 redirects the probe to the real Studio on 9000, so the
    # body + id check pass — but credentials would still go to 8888. Reject it.
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    monkeypatch.setattr(_inference, "_local_studio_install_id", lambda: "a" * 64)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _FakeHealth(_healthy_body("a" * 64), "http://127.0.0.1:9000/api/health"),
    )
    assert _inference.find_studio_server() is None


def test_find_studio_server_empty_url_is_treated_as_default(monkeypatch):
    import urllib.request

    from unsloth_cli import _inference

    # An empty UNSLOTH_STUDIO_URL must fall back to the loopback default *and*
    # still be id-checked, not silently skip verification.
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "")
    monkeypatch.setattr(_inference, "_local_studio_install_id", lambda: "a" * 64)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: _FakeHealth(_healthy_body("b" * 64))
    )
    assert _inference.find_studio_server() is None

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: _FakeHealth(_healthy_body("a" * 64))
    )
    assert _inference.find_studio_server() == "http://127.0.0.1:8888"


def test_connect_studio_server_does_not_issue_token_for_unverified_health(monkeypatch):
    import urllib.request

    from unsloth_cli import _inference

    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    monkeypatch.setattr(_inference, "_local_studio_install_id", lambda: "a" * 64)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _FakeHealth(_healthy_body("b" * 64)),
    )

    def boom():
        raise AssertionError("must not self-issue a token for an unverified backend")

    monkeypatch.setattr(_inference, "_studio_token", boom)
    assert (
        _inference.connect_studio_server(
            "model", hf_token = "hf_secret", max_seq_length = 2048, load_in_4bit = True
        )
        is None
    )


class _FakeSSEResponse:
    def __init__(self, lines):
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_backend_streams_cumulative_text(monkeypatch):
    backend = HttpChatBackend("http://localhost:8888", "token")
    response = _FakeSSEResponse(
        [
            b'data: {"choices":[{"delta":{"content":"He"}}]}\n',
            b"\n",
            b'data: {"choices":[{"delta":{"content":"llo"}}]}\n',
            b"data: [DONE]\n",
        ]
    )
    monkeypatch.setattr(backend, "_request", lambda *a, **k: response)

    out = list(backend.stream([{"role": "user", "content": "hi"}], **_STREAM_KWARGS))
    assert out == ["He", "Hello"]


def test_http_backend_merges_emoji_split_across_deltas(monkeypatch):
    backend = HttpChatBackend("http://localhost:8888", "token")
    response = _FakeSSEResponse(
        [
            b'data: {"choices":[{"delta":{"content":"hi "}}]}\n',
            b'data: {"choices":[{"delta":{"content":"\\ud83d"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"\\ude0a"}}]}\n',
            b"data: [DONE]\n",
        ]
    )
    monkeypatch.setattr(backend, "_request", lambda *a, **k: response)

    out = list(backend.stream([{"role": "user", "content": "hi"}], **_STREAM_KWARGS))
    # The lone high surrogate is held back, then merged with its other half.
    assert out == ["hi ", "hi ", "hi 😊"]


def test_chat_prefers_running_studio_server(monkeypatch):
    closed = []

    class _FakeHttpBackend:
        def stream(self, *a, **k):
            return iter(["hello"])

        def close(self):
            closed.append("http")

    local_loads = []
    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(chatmod, "connect_studio_server", lambda *a, **k: _FakeHttpBackend())
    monkeypatch.setattr(chatmod, "load_chat_backend", lambda *a, **k: local_loads.append(1))
    monkeypatch.setattr(chatmod, "_compare_needs_second_model", lambda: False)

    result = CliRunner().invoke(_chat_app(), ["fake-model"], input = "hi\n/exit\n")

    assert result.exit_code == 0, result.output
    assert local_loads == []
    assert "stays warm" in result.output
    assert closed == ["http"]


def test_chat_server_mode_compare_loads_base_locally(monkeypatch):
    streamed, closed, base_loads = [], [], []

    class _FakeHttpBackend:
        def stream(self, *a, **k):
            streamed.append("tuned")
            return iter(["tuned-answer"])

        def close(self):
            closed.append("http")

    class _FakeBaseBackend:
        def stream(self, *a, **k):
            streamed.append("base")
            return iter(["base-answer"])

        def close(self):
            closed.append("base")

    def fake_local_load(model, **kwargs):
        base_loads.append((model, kwargs.get("fresh_backend", False)))
        return _FakeBaseBackend()

    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(chatmod, "connect_studio_server", lambda *a, **k: _FakeHttpBackend())
    monkeypatch.setattr(chatmod, "load_chat_backend", fake_local_load)

    result = CliRunner().invoke(_chat_app(), ["tuned-run"], input = "/compare\nhi\n/exit\n")

    assert result.exit_code == 0, result.output
    assert "(compare on)" in result.output
    # Only the base model loaded locally, on its own private backend.
    assert base_loads == [("fake/base", True)]
    assert streamed == ["base", "tuned"]
    assert set(closed) == {"http", "base"}


def test_chat_compare_on_mlx_loads_base_model_side_by_side(monkeypatch):
    loads, streamed, closed = [], [], []

    class _FakeLocalBackend:
        def __init__(self, role):
            self.role = role

        def stream(self, *a, **k):
            streamed.append((self.role, k.get("use_adapter")))
            return iter([f"{self.role}-answer"])

        def close(self):
            closed.append(self.role)

    def fake_load(model, **kwargs):
        fresh = kwargs.get("fresh_backend", False)
        loads.append((model, fresh))
        return _FakeLocalBackend("base" if fresh else "tuned")

    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(chatmod, "load_chat_backend", fake_load)
    monkeypatch.setattr(chatmod, "_compare_needs_second_model", lambda: True)
    monkeypatch.setattr(chatmod, "connect_studio_server", lambda *a, **k: None)

    result = CliRunner().invoke(_chat_app(), ["tuned-run", "--compare"], input = "hi\n/exit\n")

    assert result.exit_code == 0, result.output
    assert loads == [("tuned-run", False), ("fake/base", True)]
    # Both models answered the turn, via plain generation (no adapter toggle).
    assert ("base", None) in streamed and ("tuned", None) in streamed
    assert set(closed) == {"tuned", "base"}
