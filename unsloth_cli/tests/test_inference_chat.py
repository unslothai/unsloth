# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the `unsloth chat` / `unsloth inference` CLI — fakes only, no model loads."""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import typer
import pytest
from rich.console import Console
from typer.testing import CliRunner

import unsloth_cli.commands.chat as chatmod
from unsloth_cli._inference import (
    ChatBackend,
    HttpChatBackend,
    collect_stream,
    mlx_distributed_info,
    mlx_distributed_uses_mpi,
    render_columns,
    visible_text,
)


class _FakeConfig:
    is_gguf = False
    is_lora = True
    display_name = "fake-model"
    base_model = "fake/base"
    path = None


_EXPECTED_MPI_ENV_PAIRS = [
    ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
    ("PMI_RANK", "PMI_SIZE"),
    ("PMIX_RANK", "PMIX_SIZE"),
    ("MPI_RANK", "MPI_WORLD_SIZE"),
    ("MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE"),
]
_IGNORED_DISTRIBUTED_ENV_PAIRS = [("SLURM_PROCID", "SLURM_NTASKS")]


def _chat_app():
    cli = typer.Typer()
    cli.command()(chatmod.chat)
    return cli


def _inference_app():
    from unsloth_cli.commands.inference import inference

    cli = typer.Typer()
    cli.command()(inference)
    return cli


def _clear_mlx_distributed_env(monkeypatch):
    for name in (
        "MLX_RANK",
        "MLX_HOSTFILE",
        "MLX_WORLD_SIZE",
        "MLX_IBV_DEVICES",
        "MLX_JACCL_COORDINATOR",
        "NCCL_HOST_IP",
        "NCCL_PORT",
        *(rank for rank, _size in _EXPECTED_MPI_ENV_PAIRS + _IGNORED_DISTRIBUTED_ENV_PAIRS),
        *(size for _rank, size in _EXPECTED_MPI_ENV_PAIRS + _IGNORED_DISTRIBUTED_ENV_PAIRS),
    ):
        monkeypatch.delenv(name, raising = False)


def _set_mlx_nccl_env(
    monkeypatch,
    *,
    rank: str = "0",
    size: str = "2",
):
    monkeypatch.setenv("MLX_RANK", rank)
    monkeypatch.setenv("MLX_WORLD_SIZE", size)
    monkeypatch.setenv("NCCL_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("NCCL_PORT", "12345")


@pytest.fixture(autouse = True)
def _isolate_mlx_distributed_env(monkeypatch):
    _clear_mlx_distributed_env(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising = False)


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


def test_inference_exposes_gguf_runtime_options():
    from unsloth_cli.commands.inference import inference

    tensor = _option(inference, "tensor_parallel")
    assert "--tensor-parallel/--no-tensor-parallel" in (getattr(tensor, "param_decls", None) or [])

    extra = _option(inference, "llama_extra_args")
    assert "--llama-extra-arg" in (getattr(extra, "param_decls", None) or [])


def test_mlx_distributed_info_reads_launch_env(monkeypatch, tmp_path):
    _clear_mlx_distributed_env(monkeypatch)
    assert mlx_distributed_info() == (False, 0, None)
    assert mlx_distributed_uses_mpi() is False

    monkeypatch.setenv("MLX_RANK", "1")
    monkeypatch.setenv("MLX_WORLD_SIZE", "2")
    assert mlx_distributed_info() == (False, 0, None)
    monkeypatch.setenv("NCCL_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("NCCL_PORT", "12345")
    assert mlx_distributed_info() == (True, 1, 2)
    assert mlx_distributed_uses_mpi() is False

    _clear_mlx_distributed_env(monkeypatch)
    ring_hostfile = tmp_path / "ring.json"
    ring_hostfile.write_text('[["127.0.0.1:5000"], ["127.0.0.1:5001"]]\n')
    monkeypatch.setenv("MLX_RANK", "0")
    monkeypatch.setenv("MLX_HOSTFILE", str(ring_hostfile))
    assert mlx_distributed_info() == (True, 0, 2)
    assert mlx_distributed_uses_mpi() is False

    _clear_mlx_distributed_env(monkeypatch)
    monkeypatch.setenv("MLX_RANK", "1")
    monkeypatch.setenv("MLX_IBV_DEVICES", '[["node-a"], ["node-b"]]')
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "node-a:12345")
    assert mlx_distributed_info() == (True, 1, 2)
    assert mlx_distributed_uses_mpi() is False

    _clear_mlx_distributed_env(monkeypatch)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    assert mlx_distributed_info() == (True, 1, 2)
    assert mlx_distributed_uses_mpi() is True

    _clear_mlx_distributed_env(monkeypatch)
    monkeypatch.setenv("MLX_RANK", "bad")
    monkeypatch.setenv("MLX_WORLD_SIZE", "-3")
    assert mlx_distributed_info() == (False, 0, None)


def test_chat_command_is_registered_with_options():
    params = inspect.signature(chatmod.chat).parameters
    assert "model" in params

    think = _option(chatmod.chat, "think")
    assert "--think/--no-think" in (getattr(think, "param_decls", None) or [])

    compare = _option(chatmod.chat, "compare")
    assert "--compare/--no-compare" in (getattr(compare, "param_decls", None) or [])

    verbose = _option(chatmod.chat, "verbose")
    assert {"--verbose", "-v"} <= set(getattr(verbose, "param_decls", None) or [])

    tensor = _option(chatmod.chat, "tensor_parallel")
    assert "--tensor-parallel/--no-tensor-parallel" in (getattr(tensor, "param_decls", None) or [])

    extra = _option(chatmod.chat, "llama_extra_args")
    assert "--llama-extra-arg" in (getattr(extra, "param_decls", None) or [])


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


def test_find_studio_server_prefers_ipv4_loopback_for_localhost(monkeypatch):
    # localhost resolving ::1-first must not hide an Unsloth bound to 127.0.0.1:
    # discovery tries each loopback address and returns the one that answers.
    import socket
    import urllib.request

    from unsloth_cli import _inference

    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://localhost:8888")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 8888, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 8888)),
        ],
    )

    class _OK:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def only_ipv4(request, *a, **k):
        if "127.0.0.1" not in request.full_url:
            raise OSError("connection refused")
        return _OK()

    monkeypatch.setattr(urllib.request, "urlopen", only_ipv4)
    assert _inference.find_studio_server() == "http://127.0.0.1:8888"


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


def test_http_backend_load_forwards_gguf_runtime_options(monkeypatch):
    backend = HttpChatBackend("http://localhost:8888", "token")
    requests = []

    class _OK:
        def close(self):
            pass

    def fake_request(
        method,
        path,
        payload = None,
        timeout = None,
    ):
        requests.append((method, path, payload, timeout))
        return _OK()

    monkeypatch.setattr(backend, "_request", fake_request)

    backend.ensure_loaded(
        "org/model-GGUF",
        hf_token = "hf_x",
        max_seq_length = 8192,
        load_in_4bit = False,
        tensor_parallel = True,
        llama_extra_args = ["--top-k", "20"],
    )

    assert requests == [
        (
            "POST",
            "/api/inference/load",
            {
                "model_path": "org/model-GGUF",
                "hf_token": "hf_x",
                "max_seq_length": 8192,
                "load_in_4bit": False,
                "tensor_parallel": True,
                "llama_extra_args": ["--top-k", "20"],
            },
            None,
        )
    ]


def test_http_backend_load_sends_explicit_false_tensor_parallel(monkeypatch):
    backend = HttpChatBackend("http://localhost:8888", "token")
    requests = []

    class _OK:
        def close(self):
            pass

    monkeypatch.setattr(
        backend,
        "_request",
        lambda method, path, payload = None, timeout = None: (
            requests.append((method, path, payload, timeout)),
            _OK(),
        )[1],
    )

    backend.ensure_loaded(
        "org/model-GGUF",
        hf_token = None,
        max_seq_length = 4096,
        load_in_4bit = True,
        tensor_parallel = False,
    )

    assert requests[0][2]["tensor_parallel"] is False


def test_load_gguf_backend_forwards_local_runtime_options(monkeypatch):
    import unsloth_cli._inference as inference

    calls = []

    class _FakeLlamaCppBackend:
        def load_model(self, **kwargs):
            calls.append(kwargs)
            return True

    fake_llama_cpp = types.ModuleType("core.inference.llama_cpp")
    fake_llama_cpp.LlamaCppBackend = _FakeLlamaCppBackend
    fake_args = types.ModuleType("core.inference.llama_server_args")
    fake_args.validate_extra_args = lambda args: list(args or [])
    fake_tensor_fallback = types.ModuleType("core.inference.tensor_fallback")

    async def _passthrough(
        attempt_load,
        *,
        requested_tensor,
        extra_args,
        label = "",
        cancelled = None,
    ):
        return await attempt_load(requested_tensor, extra_args)

    fake_tensor_fallback.load_with_tensor_fallback = _passthrough

    monkeypatch.setitem(sys.modules, "core", types.ModuleType("core"))
    monkeypatch.setitem(sys.modules, "core.inference", types.ModuleType("core.inference"))
    monkeypatch.setitem(sys.modules, "core.inference.llama_cpp", fake_llama_cpp)
    monkeypatch.setitem(sys.modules, "core.inference.llama_server_args", fake_args)
    monkeypatch.setitem(sys.modules, "core.inference.tensor_fallback", fake_tensor_fallback)
    monkeypatch.setattr(inference, "ensure_studio_backend_path", lambda: None)

    config = SimpleNamespace(
        gguf_variant = "Q4_K_M",
        identifier = "org/model-GGUF",
        is_vision = False,
        gguf_hf_repo = "org/model-GGUF",
    )

    backend = inference._load_gguf_backend(
        config,
        hf_token = "hf_x",
        max_seq_length = 8192,
        tensor_parallel = True,
        llama_extra_args = ["--top-k", "20"],
    )

    assert isinstance(backend, ChatBackend)
    assert calls == [
        {
            "hf_repo": "org/model-GGUF",
            "hf_token": "hf_x",
            "hf_variant": "Q4_K_M",
            "model_identifier": "org/model-GGUF",
            "is_vision": False,
            "n_ctx": 8192,
            "tensor_parallel": True,
            "extra_args": ["--top-k", "20"],
        }
    ]


def test_load_gguf_backend_exits_cleanly_on_invalid_extra_args(monkeypatch):
    import unsloth_cli._inference as inference

    fake_llama_cpp = types.ModuleType("core.inference.llama_cpp")
    fake_llama_cpp.LlamaCppBackend = object
    fake_args = types.ModuleType("core.inference.llama_server_args")

    def _raise(_args):
        raise ValueError("llama-server flag '--model' is managed by Unsloth Studio")

    fake_args.validate_extra_args = _raise
    fake_tensor_fallback = types.ModuleType("core.inference.tensor_fallback")
    fake_tensor_fallback.load_with_tensor_fallback = None

    monkeypatch.setitem(sys.modules, "core", types.ModuleType("core"))
    monkeypatch.setitem(sys.modules, "core.inference", types.ModuleType("core.inference"))
    monkeypatch.setitem(sys.modules, "core.inference.llama_cpp", fake_llama_cpp)
    monkeypatch.setitem(sys.modules, "core.inference.llama_server_args", fake_args)
    monkeypatch.setitem(sys.modules, "core.inference.tensor_fallback", fake_tensor_fallback)
    monkeypatch.setattr(inference, "ensure_studio_backend_path", lambda: None)

    config = SimpleNamespace(
        gguf_variant = "Q4_K_M",
        identifier = "org/model-GGUF",
        is_vision = False,
        gguf_hf_repo = "org/model-GGUF",
    )

    with pytest.raises(typer.Exit) as excinfo:
        inference._load_gguf_backend(
            config,
            hf_token = "hf_x",
            max_seq_length = 8192,
            llama_extra_args = ["--model"],
        )

    assert excinfo.value.exit_code == 1


def test_load_gguf_backend_uses_tensor_fallback(monkeypatch):
    import unsloth_cli._inference as inference

    calls = []
    fallback_calls = []

    class _FakeLlamaCppBackend:
        def load_model(self, **kwargs):
            calls.append(kwargs)
            return kwargs["tensor_parallel"] is False

    fake_llama_cpp = types.ModuleType("core.inference.llama_cpp")
    fake_llama_cpp.LlamaCppBackend = _FakeLlamaCppBackend
    fake_args = types.ModuleType("core.inference.llama_server_args")
    fake_args.validate_extra_args = lambda args: list(args or [])
    fake_tensor_fallback = types.ModuleType("core.inference.tensor_fallback")

    async def _fallback(
        attempt_load,
        *,
        requested_tensor,
        extra_args,
        label = "",
        cancelled = None,
    ):
        fallback_calls.append((requested_tensor, extra_args, label))
        ok = await attempt_load(requested_tensor, extra_args)
        if ok:
            return True
        return await attempt_load(False, ["--split-mode", "layer"])

    fake_tensor_fallback.load_with_tensor_fallback = _fallback

    monkeypatch.setitem(sys.modules, "core", types.ModuleType("core"))
    monkeypatch.setitem(sys.modules, "core.inference", types.ModuleType("core.inference"))
    monkeypatch.setitem(sys.modules, "core.inference.llama_cpp", fake_llama_cpp)
    monkeypatch.setitem(sys.modules, "core.inference.llama_server_args", fake_args)
    monkeypatch.setitem(sys.modules, "core.inference.tensor_fallback", fake_tensor_fallback)
    monkeypatch.setattr(inference, "ensure_studio_backend_path", lambda: None)

    config = SimpleNamespace(
        gguf_variant = "Q4_K_M",
        identifier = "org/model-GGUF",
        is_vision = False,
        gguf_hf_repo = "org/model-GGUF",
    )

    backend = inference._load_gguf_backend(
        config,
        hf_token = "hf_x",
        max_seq_length = 8192,
        tensor_parallel = True,
    )

    assert isinstance(backend, ChatBackend)
    assert fallback_calls == [(True, [], "org/model-GGUF")]
    assert [call["tensor_parallel"] for call in calls] == [True, False]
    assert calls[1]["extra_args"] == ["--split-mode", "layer"]


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


def test_chat_forwards_gguf_runtime_options_to_loader(monkeypatch):
    loads = []

    class _FakeHttpBackend:
        def close(self):
            pass

    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(
        chatmod,
        "connect_studio_server",
        lambda model, **kwargs: (loads.append((model, kwargs)), _FakeHttpBackend())[1],
    )
    monkeypatch.setattr(chatmod, "load_chat_backend", lambda *a, **k: None)
    monkeypatch.setattr(chatmod, "_compare_needs_second_model", lambda: False)

    result = CliRunner().invoke(
        _chat_app(),
        [
            "fake-model",
            "--tensor-parallel",
            "--llama-extra-arg=--top-k",
            "--llama-extra-arg",
            "20",
        ],
        input = "/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert loads == [
        (
            "fake-model",
            {
                "hf_token": None,
                "max_seq_length": 4096,
                "load_in_4bit": True,
                "tensor_parallel": True,
                "llama_extra_args": ["--top-k", "20"],
            },
        )
    ]


def test_inference_forwards_gguf_runtime_options_to_loader(monkeypatch):
    from unsloth_cli.commands import inference as infermod

    loads, streams, closed = [], [], []

    class _FakeBackend:
        def stream(self, messages, **kwargs):
            streams.append((messages, kwargs))
            return iter(["answer"])

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        infermod,
        "connect_studio_server",
        lambda model, **kwargs: (loads.append((model, kwargs)), _FakeBackend())[1],
    )
    monkeypatch.setattr(infermod, "load_chat_backend", lambda *a, **k: None)

    result = CliRunner().invoke(
        _inference_app(),
        [
            "fake-model",
            "hello",
            "--tensor-parallel",
            "--llama-extra-arg=--top-k",
            "--llama-extra-arg",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert loads == [
        (
            "fake-model",
            {
                "hf_token": None,
                "max_seq_length": 2048,
                "load_in_4bit": True,
                "tensor_parallel": True,
                "llama_extra_args": ["--top-k", "20"],
            },
        )
    ]
    assert streams[0][0] == [{"role": "user", "content": "hello"}]
    assert closed == [True]


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
    assert ("base", None) in streamed and ("tuned", None) in streamed
    assert set(closed) == {"tuned", "base"}


@pytest.mark.parametrize(
    ("chunk_kind", "expected_exit"),
    [
        ("answer", 0),
        ("model_text_error", 0),
        ("real_error", 1),
    ],
)
def test_inference_local_handles_stream(monkeypatch, chunk_kind, expected_exit):
    from unsloth_cli.commands import inference as infermod
    from unsloth_cli._inference import ensure_studio_backend_path

    ensure_studio_backend_path()
    from core.inference.orchestrator import GenStreamError

    chunks = {
        "answer": ["answer"],
        "model_text_error": ["Error: printed by the model, not a backend failure"],
        "real_error": [GenStreamError("Error: generation failed")],
    }[chunk_kind]
    closed = []

    class _FakeBackend:
        def stream(self, messages, **kwargs):
            return iter(chunks)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        infermod,
        "connect_studio_server",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("server disabled")),
    )
    monkeypatch.setattr(infermod, "load_chat_backend", lambda *a, **k: _FakeBackend())

    result = CliRunner().invoke(
        _inference_app(),
        ["fake-model", "hello", "--no-server"],
    )

    assert result.exit_code == expected_exit, result.output
    assert closed == [True]
    if chunk_kind == "real_error":
        assert result.stdout == "Assistant:\n"
        assert result.stderr == "Error: generation failed\n"
    else:
        assert chunks[0] in result.output


@pytest.mark.parametrize("chunk_kind", ["answer", "model_text_error", "real_error"])
def test_chat_local_handles_stream(monkeypatch, chunk_kind):
    from unsloth_cli._inference import ensure_studio_backend_path

    ensure_studio_backend_path()
    from core.inference.orchestrator import GenStreamError

    first_chunk = {
        "answer": "answer",
        "model_text_error": "Error: printed by the model, not a backend failure",
        "real_error": GenStreamError("Error: generation failed"),
    }[chunk_kind]
    calls, closed = [], []

    class _FakeChatBackend:
        def stream(self, messages, **kwargs):
            calls.append([dict(message) for message in messages])
            return iter([first_chunk if len(calls) == 1 else "second answer"])

        def close(self):
            closed.append(True)

    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(chatmod, "connect_studio_server", lambda *a, **k: None)
    monkeypatch.setattr(chatmod, "load_chat_backend", lambda *a, **k: _FakeChatBackend())
    monkeypatch.setattr(chatmod, "_compare_needs_second_model", lambda: False)

    result = CliRunner().invoke(
        _chat_app(),
        ["fake-model"],
        input = "first\nsecond\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert closed == [True]
    if chunk_kind == "real_error":
        assert calls[1] == [{"role": "user", "content": "second"}]
        assert "(error: generation failed)" in result.output
        assert "Error: generation failed" not in result.output
    else:
        assert calls[1] == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": first_chunk},
            {"role": "user", "content": "second"},
        ]
        assert first_chunk in result.output


@pytest.mark.parametrize(
    ("chunk_kind", "expected_exit"),
    [
        ("answer", 0),
        ("model_text_error", 0),
        ("real_error", 1),
    ],
)
def test_inference_under_mlx_launch_handles_stream(monkeypatch, chunk_kind, expected_exit):
    from unsloth_cli.commands import inference as infermod
    from unsloth_cli._inference import ensure_studio_backend_path

    ensure_studio_backend_path()
    from core.inference.orchestrator import GenStreamError

    if chunk_kind == "answer":
        chunks = ["answer"]
    elif chunk_kind == "model_text_error":
        # Model output whose visible text starts with "Error:" must not abort.
        chunks = ["Error: printed by the model, not a backend failure"]
    else:
        chunks = [GenStreamError("Error: generation failed")]

    loads, closed = [], []

    class _FakeBackend:
        def stream(self, messages, **kwargs):
            return iter(chunks)

        def close(self):
            closed.append(True)

    _set_mlx_nccl_env(monkeypatch, rank = "0")
    monkeypatch.setattr(
        infermod,
        "connect_studio_server",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("server disabled")),
    )
    monkeypatch.setattr(
        infermod,
        "load_chat_backend",
        lambda model, **kwargs: (loads.append((model, kwargs)), _FakeBackend())[1],
    )

    result = CliRunner().invoke(
        _inference_app(),
        ["fake-model", "hello", "--tensor-parallel"],
    )

    assert result.exit_code == expected_exit, result.output
    assert loads[0][1]["tensor_parallel"] is True
    if chunk_kind == "real_error":
        assert "generation failed" in result.output


def test_chat_under_mlx_launch_nonzero_rank_drains_stdin(monkeypatch):
    drains, closed = [], []
    turns = iter(
        [
            {"type": "turn", "text": "hi"},
            {"type": "turn", "text": "/exit"},
        ]
    )

    class _FakeChatBackend:
        def share_distributed_object(
            self,
            obj,
            *,
            timeout = 300.0,
        ):
            assert obj is None
            return next(turns)

        def stream(self, messages, **kwargs):
            return iter(["hidden"])

        def close(self):
            closed.append(True)

    _set_mlx_nccl_env(monkeypatch, rank = "1")
    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(
        chatmod,
        "connect_studio_server",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("server disabled")),
    )
    monkeypatch.setattr(chatmod, "load_chat_backend", lambda *a, **k: _FakeChatBackend())
    monkeypatch.setattr(chatmod, "_compare_needs_second_model", lambda: False)
    monkeypatch.setattr(chatmod, "_drain_available_stdin", lambda: drains.append(True))

    result = CliRunner().invoke(_chat_app(), ["fake-model"], input = "hi\n/exit\n")

    assert result.exit_code == 0, result.output
    assert "Chatting with" not in result.output
    assert drains == [True, True]
    assert closed == [True]


def test_chat_under_mlx_launch_rank0_bypasses_studio_and_prints(monkeypatch):
    loads, shares, closed = [], [], []

    class _FakeChatBackend:
        def share_distributed_object(
            self,
            obj,
            *,
            timeout = 300.0,
        ):
            shares.append((obj, timeout))
            return obj

        def stream(self, messages, **kwargs):
            return iter(["hello"])

        def close(self):
            closed.append(True)

    _set_mlx_nccl_env(monkeypatch, rank = "0")
    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(
        chatmod,
        "connect_studio_server",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("server disabled")),
    )
    monkeypatch.setattr(
        chatmod,
        "load_chat_backend",
        lambda model, **kwargs: (loads.append((model, kwargs)), _FakeChatBackend())[1],
    )
    monkeypatch.setattr(chatmod, "_compare_needs_second_model", lambda: False)

    result = CliRunner().invoke(
        _chat_app(),
        ["fake-model", "--tensor-parallel"],
        input = "hi\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert "Chatting with fake-model" in result.output
    assert "hello" in result.output
    assert loads and loads[0][0] == "fake-model"
    assert loads[0][1]["tensor_parallel"] is True
    assert shares == [
        ({"type": "turn", "text": "hi"}, None),
        ({"type": "turn", "text": "/exit"}, None),
    ]


@pytest.mark.parametrize(
    ("stream_error", "expected_exit"),
    [("exception", 1), ("chunk", 1), ("model_text", 0)],
)
def test_chat_under_mlx_launch_exits_on_generation_error(monkeypatch, stream_error, expected_exit):
    from unsloth_cli._inference import ensure_studio_backend_path

    ensure_studio_backend_path()
    from core.inference.orchestrator import GenStreamError

    closed = []

    class _FakeChatBackend:
        def share_distributed_object(
            self,
            obj,
            *,
            timeout = 300.0,
        ):
            return obj

        def stream(self, messages, **kwargs):
            if stream_error == "exception":
                raise RuntimeError("generation failed")
            if stream_error == "model_text":
                # Plain model text starting with "Error:" must not abort the run.
                return iter(["Error: printed by the model"])
            return iter([GenStreamError("Error: generation failed")])

        def close(self):
            closed.append(True)

    _set_mlx_nccl_env(monkeypatch, rank = "0")
    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(
        chatmod,
        "connect_studio_server",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("server disabled")),
    )
    monkeypatch.setattr(chatmod, "load_chat_backend", lambda *a, **k: _FakeChatBackend())
    monkeypatch.setattr(chatmod, "_compare_needs_second_model", lambda: False)

    result = CliRunner().invoke(_chat_app(), ["fake-model"], input = "hi\n/exit\n")

    assert result.exit_code == expected_exit
    if expected_exit:
        assert "generation failed" in result.output
    assert closed == [True]


def test_load_chat_backend_forwards_mlx_distributed_options(monkeypatch):
    import unsloth_cli._inference as inference

    calls = []

    class _FakeBackend:
        def load_model(self, **kwargs):
            calls.append(kwargs)
            return True

    class _FakeModelConfig:
        is_gguf = False

        @classmethod
        def from_identifier(cls, **_kwargs):
            return cls()

    fake_backend = _FakeBackend()
    fake_inference = types.ModuleType("core.inference")
    fake_inference.get_inference_backend = lambda: fake_backend
    fake_utils = types.ModuleType("utils")
    fake_utils.__path__ = []
    fake_models = types.ModuleType("utils.models")
    fake_models.ModelConfig = _FakeModelConfig

    _set_mlx_nccl_env(monkeypatch, rank = "0")
    monkeypatch.setitem(sys.modules, "core", types.ModuleType("core"))
    monkeypatch.setitem(sys.modules, "core.inference", fake_inference)
    monkeypatch.setitem(sys.modules, "utils", fake_utils)
    monkeypatch.setitem(sys.modules, "utils.models", fake_models)
    monkeypatch.setattr(inference, "ensure_studio_backend_path", lambda: None)

    inference.load_chat_backend(
        "fake-model",
        hf_token = None,
        max_seq_length = 2048,
        load_in_4bit = True,
        tensor_parallel = True,
    )

    assert calls[0]["tensor_parallel"] is True
    assert calls[0]["mlx_distributed"] is True
