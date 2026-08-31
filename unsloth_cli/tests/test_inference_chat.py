# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the `unsloth chat` / `unsloth inference` CLI — fakes only, no model loads."""

from __future__ import annotations

import inspect
import json
import subprocess
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

    spec_type = _option(inference, "speculative_type")
    assert "--speculative-type" in (getattr(spec_type, "param_decls", None) or [])
    draft_n = _option(inference, "spec_draft_n_max")
    assert "--spec-draft-n-max" in (getattr(draft_n, "param_decls", None) or [])


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

    spec_type = _option(chatmod.chat, "speculative_type")
    assert "--speculative-type" in (getattr(spec_type, "param_decls", None) or [])
    draft_n = _option(chatmod.chat, "spec_draft_n_max")
    assert "--spec-draft-n-max" in (getattr(draft_n, "param_decls", None) or [])


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


def test_pick_model_lists_groups_and_selects(monkeypatch):
    from unsloth_cli import _model_catalog as cat

    entries = [
        cat.ModelEntry("Fine-tunes", "run-new", "lora · Aug 1", "outputs/run-new"),
        cat.ModelEntry("Fine-tunes", "run-old", "merged", "outputs/run-old"),
        cat.ModelEntry("GGUF", "org/Tiny-GGUF", "Q4_K_M", "org/Tiny-GGUF"),
    ]
    monkeypatch.setattr(cat, "list_chat_models", lambda: entries)
    monkeypatch.setattr(chatmod, "ensure_studio_backend_path", lambda: None)

    console = Console(record = True, width = 100)
    monkeypatch.setattr("builtins.input", lambda prompt = "": "3")
    assert chatmod._pick_model(console) == "org/Tiny-GGUF"
    out = console.export_text()
    assert "Fine-tunes" in out and "GGUF" in out
    assert "1. run-new" in out and "lora · Aug 1" in out
    assert "3. org/Tiny-GGUF" in out

    monkeypatch.setattr("builtins.input", lambda prompt = "": "")
    assert chatmod._pick_model(Console()) == "outputs/run-new"


def test_pick_model_exits_when_nothing_found(monkeypatch):
    from unsloth_cli import _model_catalog as cat

    monkeypatch.setattr(cat, "list_chat_models", lambda: [])
    monkeypatch.setattr(chatmod, "ensure_studio_backend_path", lambda: None)
    with pytest.raises(typer.Exit):
        chatmod._pick_model(Console())


def test_catalog_trained_entries_use_run_metadata(monkeypatch, tmp_path):
    from unsloth_cli import _model_catalog as cat

    named = tmp_path / "unsloth_Llama-3.2-1B-Instruct_1785529397"
    bare = tmp_path / "unsloth_Qwen3-0.6B_1785529000"
    named.mkdir()
    bare.mkdir()
    fake_models = types.ModuleType("utils.models")
    fake_models.scan_trained_models = lambda: [
        (named.name, str(named), "lora"),
        (bare.name, str(bare), "merged"),
    ]
    monkeypatch.setitem(sys.modules, "utils.models", fake_models)
    monkeypatch.setattr(
        cat,
        "_runs_by_output_dir",
        lambda: {
            str(named): {
                "display_name": "support-bot",
                "model_name": "unsloth/Llama-3.2-1B-Instruct",
                "dataset_name": "/x/uploads/0123456789abcdef0123456789abcdef_tickets.jsonl",
                "started_at": "2026-07-31T20:23:09+00:00",
                "final_step": 30,
                "final_loss": 2.4368,
            }
        },
    )

    first, second = cat.trained_entries()
    assert (first.group, first.name, first.model) == ("Fine-tunes", "support-bot", str(named))
    assert first.detail.startswith("tickets · 30 steps · ")
    assert second.name == "Qwen3-0.6B"
    assert second.detail.startswith("merged · ")


def test_catalog_cached_entries_filter_non_chat_rows(monkeypatch, tmp_path):
    from unsloth_cli import _model_catalog as cat

    repo = tmp_path / "models--org--Tiny-GGUF"
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents = True)
    for name in ("Tiny-Q4_K_M.gguf", "Tiny-Q2_K.gguf", "mmproj-F16.gguf"):
        (snap / name).write_bytes(b"")

    gguf_rows = [
        {"repo_id": "org/Tiny-GGUF", "cache_path": str(repo), "task": "text-generation"},
        {
            "repo_id": "org/Half-GGUF",
            "cache_path": str(repo),
            "task": "text-generation",
            "partial": True,
        },
        {"repo_id": "org/Flux-GGUF", "cache_path": str(repo), "task": "text-to-image"},
        {
            "repo_id": "unsloth/Qwen3-ASR-GGUF",
            "cache_path": str(repo),
            "task": "text-generation",
            "capabilities": {"can_chat": False},
        },
    ]
    model_rows = [
        {"repo_id": "org/Chat", "task": None},
        {"repo_id": "org/Half", "task": None, "partial": True},
        {"repo_id": "org/Image", "task": "text-to-image"},
        {"repo_id": "org/TTS", "task": "text-to-speech"},
        {"repo_id": "org/Pinned", "task": None, "load_id": "/snap/path"},
        {
            "repo_id": "org/ChatAdapter",
            "task": None,
            "model_format": "adapter",
            "load_id": "/snap/chat-adapter",
            "capabilities": {"can_chat": True},
        },
        {
            "repo_id": "org/SpeechAdapter",
            "task": None,
            "model_format": "adapter",
            "capabilities": {"can_chat": False},
        },
        # An embedding/CLIP repo carries task None like any chat repo; can_chat separates them.
        {"repo_id": "org/Embedder", "task": None, "capabilities": {"can_chat": False}},
        # An untrusted diffusion repo carries no task either, and its pipeline root has no
        # config for can_chat to read, so only its own flag keeps it out of chat.
        {"repo_id": "org/Sdxl", "task": None, "diffusers": True},
    ]
    # The real variant lister, not a stub: it decides these labels and picks the load target,
    # so a stub tests the plumbing and none of the answer. Pulls neither torch nor fastapi, and
    # syspath_prepend is undone after the test, so the suite keeps its no-server-import property.
    monkeypatch.syspath_prepend(str(_REPO_ROOT / "studio" / "backend"))
    monkeypatch.setattr(cat, "_cached_catalog_rows", lambda: (gguf_rows, model_rows))

    entries = cat.cached_entries()
    assert [(e.group, e.name, e.detail, e.model) for e in entries] == [
        ("Downloaded", "org/Tiny-GGUF", "Q2_K, Q4_K_M", "org/Tiny-GGUF"),
        ("Downloaded", "org/Chat", "", "org/Chat"),
        ("Downloaded", "org/Pinned", "", "/snap/path"),
        ("Downloaded", "org/ChatAdapter", "", "/snap/chat-adapter"),
    ]


def test_catalog_pinned_cached_gguf_uses_the_preferred_complete_quant(monkeypatch, tmp_path):
    from unsloth_cli._inference import ensure_studio_backend_path
    from unsloth_cli import _model_catalog as cat

    ensure_studio_backend_path()
    from utils.models.model_config import detect_gguf_model

    repo = tmp_path / "legacy" / "models--org--Multi-GGUF"
    snapshot = repo / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    (snapshot / "Model-F16.gguf").write_bytes(b"f" * 1024)
    first = snapshot / "Model-Q4_K_M-00001-of-00002.gguf"
    first.write_bytes(b"q" * 256)
    (snapshot / "Model-Q4_K_M-00002-of-00002.gguf").write_bytes(b"q" * 256)
    (snapshot / "mmproj-F16.gguf").write_bytes(b"m" * 2048)

    monkeypatch.setattr(
        cat,
        "_cached_catalog_rows",
        lambda: (
            [
                {
                    "repo_id": "org/Multi-GGUF",
                    "cache_path": str(repo),
                    "load_id": str(snapshot),
                    "task": "text-generation",
                }
            ],
            [],
        ),
    )

    assert Path(detect_gguf_model(str(snapshot))).name == "Model-F16.gguf"
    assert cat.cached_entries()[0].model == str(first)


def test_catalog_exported_gguf_uses_the_preferred_complete_quant(monkeypatch, tmp_path):
    from unsloth_cli._inference import ensure_studio_backend_path
    from unsloth_cli import _model_catalog as cat

    ensure_studio_backend_path()
    export = tmp_path / "exports" / "multi-quant"
    export.mkdir(parents = True)
    f16 = export / "Model-F16.gguf"
    f16.write_bytes(b"f" * 1024)
    first = export / "Model-Q4_K_M-00001-of-00002.gguf"
    first.write_bytes(b"q" * 256)
    (export / "Model-Q4_K_M-00002-of-00002.gguf").write_bytes(b"q" * 256)

    fake_models = types.ModuleType("utils.models")
    fake_models.scan_exported_models = lambda: [
        ("multi-quant", str(f16), "gguf", None),
    ]
    monkeypatch.setitem(sys.modules, "utils.models", fake_models)
    monkeypatch.setattr(cat, "_path_can_chat", lambda *_args: None)
    monkeypatch.setattr(cat, "_gguf_export_task", lambda *_args: "text-generation")

    assert cat.exported_entries()[0].model == str(first)


def test_catalog_keeps_only_custom_unconfigured_hf_cache_rows(monkeypatch, tmp_path):
    from unsloth_cli._inference import ensure_studio_backend_path
    from unsloth_cli import _model_catalog as cat

    ensure_studio_backend_path()
    from hub.storage import scan_folders
    from hub.utils import hf_cache_state

    class _LocalModelInfo:
        def __init__(self, path):
            self.id = path
            self.load_id = path
            self.display_name = Path(path).parents[1].name.rsplit("--", 1)[-1]
            self.path = path
            self.source = "hf_cache"
            self.model_format = "safetensors"
            self.partial = False

    custom_root = tmp_path / "custom-cache"
    custom_snapshot = custom_root / "models--org--Custom" / "snapshots" / "revision"
    custom_snapshot.mkdir(parents = True)
    (custom_snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (custom_snapshot / "model.safetensors").write_bytes(b"weights")
    configured_root = tmp_path / "configured-cache"
    configured_snapshot = configured_root / "models--org--Configured" / "snapshots" / "revision"
    configured_snapshot.mkdir(parents = True)
    (configured_snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (configured_snapshot / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(
        cat,
        "_local_catalog_rows",
        lambda: [_LocalModelInfo(str(custom_snapshot)), _LocalModelInfo(str(configured_snapshot))],
    )
    monkeypatch.setattr(scan_folders, "list_scan_folders", lambda: [{"path": str(custom_root)}])
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [configured_root])
    monkeypatch.setattr(cat, "_local_model_task", lambda _model: None)
    monkeypatch.setattr(cat, "_local_model_can_chat", lambda _model: None)
    monkeypatch.setattr(cat, "_local_is_a_diffusers_pipeline", lambda _model: False)

    entries = cat.local_folder_entries()

    assert [(entry.name, entry.model) for entry in entries] == [("Custom", str(custom_snapshot))]


def test_catalog_inventory_works_without_fastapi_or_routes():
    script = """
import builtins
from unsloth_cli._inference import ensure_studio_backend_path

ensure_studio_backend_path()
real_import = builtins.__import__

def import_without_server(name, *args, **kwargs):
    if name == "fastapi" or name.startswith("fastapi.") or name == "routes" or name.startswith("routes."):
        raise AssertionError(f"server import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_server
from hub.services.models import cache_inventory, catalog_classification, local_inventory
from unsloth_cli import _model_catalog
assert cache_inventory and catalog_classification and local_inventory and _model_catalog
cache_inventory.all_hf_cache_scans = lambda: []
cache_inventory._scan_cached_gguf = lambda **kwargs: []
cache_inventory._scan_cached_models = lambda **kwargs: []
assert _model_catalog.cached_entries() == []

async def empty_local_models(_models_dir):
    return type("Response", (), {"models": []})()

local_inventory.list_local_models_response = empty_local_models
assert _model_catalog.local_folder_entries() == []
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd = _REPO_ROOT,
        capture_output = True,
        text = True,
        check = False,
    )
    assert result.returncode == 0, result.stderr


def test_catalog_local_folder_entries_keep_safetensors_and_use_id(monkeypatch):
    from unsloth_cli._inference import ensure_studio_backend_path
    from unsloth_cli import _model_catalog as cat

    ensure_studio_backend_path()

    class _LocalModelInfo:
        """Mirrors the catalog fields used from LocalModelInfo."""

        def __init__(
            self,
            id,
            display_name,
            path,
            source,
            model_format = None,
            partial = False,
            load_id = None,
        ):
            self.id = id
            self.display_name = display_name
            self.path = path
            self.source = source
            self.model_format = model_format
            self.partial = partial
            self.load_id = load_id

    rows = [
        # _dir_model_format reports None for a safetensors folder, never "safetensors".
        _LocalModelInfo("/models/Qwen3-0.6B", "Qwen3-0.6B", "/models/Qwen3-0.6B", "models_dir"),
        _LocalModelInfo("/m/Tiny.gguf", "Tiny", "/m/Tiny.gguf", "lmstudio", model_format = "gguf"),
        _LocalModelInfo("/models/Half", "Half", "/models/Half", "models_dir", partial = True),
        _LocalModelInfo("/models/MiniLM", "MiniLM", "/models/MiniLM", "models_dir"),
        # the cli has no studio load-route materializer for this opaque identifier.
        _LocalModelInfo(
            "ollama-manifest:%2Fmodels%2Fmanifests%2Fqwen",
            "Ollama Qwen",
            "/models/blobs/sha256-deadbeef",
            "ollama",
            model_format = "gguf",
            load_id = "ollama-manifest:%2Fmodels%2Fmanifests%2Fqwen",
        ),
    ]

    monkeypatch.setattr(cat, "_local_catalog_rows", lambda: rows)
    monkeypatch.setattr(cat, "_local_model_task", lambda model: None)
    monkeypatch.setattr(
        cat,
        "_local_model_can_chat",
        lambda model: (False if model.display_name == "MiniLM" else None),
    )

    entries = cat.local_folder_entries()
    assert [(e.group, e.name, e.detail, e.model) for e in entries] == [
        ("Downloaded", "Qwen3-0.6B", "", "/models/Qwen3-0.6B"),
        ("Downloaded", "Tiny", "gguf", "/m/Tiny.gguf"),
    ]


def test_catalog_dedupes_and_survives_failing_sources(monkeypatch):
    from unsloth_cli import _model_catalog as cat

    def boom():
        raise RuntimeError("no studio db")

    monkeypatch.setattr(cat, "trained_entries", boom)
    # The SAME load target reached by two sources is one model, and the first source wins.
    monkeypatch.setattr(cat, "exported_entries", lambda: [cat.ModelEntry("Exports", "a", "", "/A")])
    monkeypatch.setattr(cat, "cached_entries", lambda: [cat.ModelEntry("GGUF", "b", "", "/A")])
    monkeypatch.setattr(cat, "local_folder_entries", lambda: [])
    assert [e.name for e in cat.list_chat_models()] == ["a"]


def test_catalog_keeps_case_distinct_local_paths_where_the_filesystem_does(monkeypatch, tmp_path):
    """``Foo`` and ``foo`` are two models on ext4 and one model on NTFS or a stock APFS volume.

    Built on the real filesystem and asked of it, rather than decided from ``sys.platform`` or
    from ``os.path.normcase`` -- normcase is identity on macOS, where the DEFAULT volume is
    case-insensitive, so a platform test gets that case exactly backwards. Whatever this
    filesystem does, the picker must agree with it: one directory, one row.
    """
    from unsloth_cli import _model_catalog as cat

    upper = tmp_path / "Foo"
    upper.mkdir()
    (upper / "config.json").write_text("{}")
    lower = tmp_path / "foo"
    case_sensitive_fs = not lower.exists()
    if case_sensitive_fs:
        lower.mkdir()
        (lower / "config.json").write_text("{}")

    for name in ("trained_entries", "exported_entries", "cached_entries"):
        monkeypatch.setattr(cat, name, lambda: [])
    monkeypatch.setattr(
        cat,
        "local_folder_entries",
        lambda: [
            cat.ModelEntry("Downloaded", "Foo", "", str(upper)),
            cat.ModelEntry("Downloaded", "foo", "", str(lower)),
        ],
    )
    names = [e.name for e in cat.list_chat_models()]
    assert names == (["Foo", "foo"] if case_sensitive_fs else ["Foo"])


def test_catalog_collapses_a_model_reached_twice_through_a_symlink(monkeypatch, tmp_path):
    """Two sources naming one model by different paths is one row.

    The scan folder and the models dir can both cover the same directory through a link, which
    a string key cannot see through however it is cased.
    """
    import pytest

    from unsloth_cli import _model_catalog as cat

    real = tmp_path / "real"
    real.mkdir()
    (real / "config.json").write_text("{}")
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem does not support symlinks")

    for name in ("trained_entries", "exported_entries"):
        monkeypatch.setattr(cat, name, lambda: [])
    monkeypatch.setattr(
        cat,
        "cached_entries",
        lambda: [
            cat.ModelEntry("Downloaded", "real", "", str(real)),
        ],
    )
    monkeypatch.setattr(
        cat,
        "local_folder_entries",
        lambda: [
            cat.ModelEntry("Downloaded", "link", "", str(link)),
        ],
    )
    assert [e.name for e in cat.list_chat_models()] == ["real"]


def test_catalog_still_folds_case_on_bare_repo_ids(monkeypatch):
    """A repo id is not a path. The Hub resolves ``Unsloth/Qwen3`` and ``unsloth/qwen3`` to one
    repo and the cache folds them into one directory, so these stay a single row everywhere."""
    from unsloth_cli import _model_catalog as cat

    for name in ("trained_entries", "exported_entries", "local_folder_entries"):
        monkeypatch.setattr(cat, name, lambda: [])
    monkeypatch.setattr(
        cat,
        "cached_entries",
        lambda: [
            cat.ModelEntry("Downloaded", "Unsloth/Qwen3-0.6B", "", "Unsloth/Qwen3-0.6B"),
            cat.ModelEntry("Downloaded", "unsloth/qwen3-0.6b", "", "unsloth/qwen3-0.6b"),
        ],
    )
    assert len(cat.list_chat_models()) == 1


def test_a_failing_source_says_so_instead_of_vanishing(monkeypatch, capsys):
    """A source is ALL of your downloaded models or ALL of your fine-tunes. Returning [] with
    no word for it turns an unreadable HF cache into a picker that is simply missing half its
    content, with nothing for the user to act on."""
    from unsloth_cli import _model_catalog as cat

    monkeypatch.delenv("UNSLOTH_DEBUG", raising = False)
    for name in ("trained_entries", "exported_entries", "local_folder_entries"):
        monkeypatch.setattr(cat, name, lambda: [])

    def boom():
        raise OSError("cache is on a dead mount")

    monkeypatch.setattr(cat, "cached_entries", boom)
    assert cat.list_chat_models() == []
    assert "cache is on a dead mount" in capsys.readouterr().err


def test_unsloth_debug_re_raises_a_failing_source(monkeypatch):
    """When the empty group IS the bug, the traceback is the thing you need."""
    import pytest

    from unsloth_cli import _model_catalog as cat

    monkeypatch.setenv("UNSLOTH_DEBUG", "1")
    for name in ("trained_entries", "exported_entries", "local_folder_entries"):
        monkeypatch.setattr(cat, name, lambda: [])

    def boom():
        raise OSError("cache is on a dead mount")

    monkeypatch.setattr(cat, "cached_entries", boom)
    with pytest.raises(OSError, match = "dead mount"):
        cat.list_chat_models()


def test_catalog_drops_org_prefix_unless_ambiguous(monkeypatch):
    from unsloth_cli import _model_catalog as cat

    rows = [
        cat.ModelEntry("Downloaded", "unsloth/Qwen3-1.7B", "", "unsloth/Qwen3-1.7B"),
        cat.ModelEntry("Downloaded", "Qwen/Qwen3-0.6B", "", "Qwen/Qwen3-0.6B"),
        cat.ModelEntry("Downloaded", "unsloth/qwen3-0.6b", "", "unsloth/qwen3-0.6b"),
    ]
    for name in ("trained_entries", "exported_entries", "local_folder_entries"):
        monkeypatch.setattr(cat, name, lambda: [])
    monkeypatch.setattr(cat, "cached_entries", lambda: rows)
    assert [e.name for e in cat.list_chat_models()] == [
        "Qwen3-1.7B",
        "Qwen/Qwen3-0.6B",
        "unsloth/qwen3-0.6b",
    ]


def test_chat_no_arg_chats_with_picked_trained_model(monkeypatch):
    class _FakeChatBackend:
        def stream(self, *a, **k):
            return iter(["hello"])

        def close(self):
            pass

    resolved = []
    monkeypatch.setattr(chatmod, "_pick_model", lambda console: "outputs/run-42")
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


class _FakeLoadResponse:
    """A /api/inference/load reply that records whether its body was drained.

    Closing a padded body at the headers resumes before the load finishes and discards
    a late failure, so the fake must distinguish read() from close().
    """

    def __init__(self, body: bytes = b'{"status": "loaded"}') -> None:
        self._body = body
        self.reads = 0
        self.closed = False

    def read(self) -> bytes:
        self.reads += 1
        return self._body

    def close(self) -> None:
        self.closed = True


def test_http_backend_load_forwards_gguf_runtime_options(monkeypatch):
    backend = HttpChatBackend("http://localhost:8888", "token")
    requests = []

    def fake_request(
        method,
        path,
        payload = None,
        timeout = None,
    ):
        requests.append((method, path, payload, timeout))
        return _FakeLoadResponse()

    monkeypatch.setattr(backend, "_request", fake_request)

    backend.ensure_loaded(
        "org/model-GGUF",
        hf_token = "hf_x",
        max_seq_length = 8192,
        load_in_4bit = False,
        tensor_parallel = True,
        speculative_type = "dspark",
        spec_draft_n_max = 3,
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
                "speculative_type": "dspark",
                "spec_draft_n_max": 3,
                "llama_extra_args": ["--top-k", "20"],
            },
            None,
        )
    ]


def test_http_backend_load_sends_explicit_false_tensor_parallel(monkeypatch):
    backend = HttpChatBackend("http://localhost:8888", "token")
    requests = []

    monkeypatch.setattr(
        backend,
        "_request",
        lambda method, path, payload = None, timeout = None: (
            requests.append((method, path, payload, timeout)),
            _FakeLoadResponse(),
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


# ── A load slower than the proxy timer (see routes/inference.py _tunnel_safe_json) ──


def test_http_backend_load_drains_the_padded_body(monkeypatch):
    """Closing at the headers would start generating while the model is still loading."""
    backend = HttpChatBackend("http://localhost:8888", "token")
    # What a padded slow load looks like on the wire: spaces, then the payload.
    response = _FakeLoadResponse(b'   {"status": "loaded"}')
    monkeypatch.setattr(backend, "_request", lambda *a, **k: response)

    backend.ensure_loaded(
        "org/model-GGUF",
        hf_token = None,
        max_seq_length = 4096,
        load_in_4bit = True,
    )

    assert response.reads == 1, "the padded body must be drained, not closed at the headers"
    assert response.closed


def test_http_backend_load_fails_on_a_deferred_error(monkeypatch, capsys):
    """A failure found after the 200 committed rides in the body; a 200 is not success."""
    backend = HttpChatBackend("http://localhost:8888", "token")
    response = _FakeLoadResponse(
        json.dumps(
            {"_deferred_error": {"status_code": 507, "detail": "CUDA out of memory"}}
        ).encode()
    )
    monkeypatch.setattr(backend, "_request", lambda *a, **k: response)

    with pytest.raises(typer.Exit) as excinfo:
        backend.ensure_loaded(
            "org/model-GGUF",
            hf_token = None,
            max_seq_length = 4096,
            load_in_4bit = True,
        )

    # Same exit code as an early HTTP failure: ensure_loaded's except block is reused.
    assert excinfo.value.exit_code == 1
    err = capsys.readouterr().err
    assert "Model load failed" in err
    assert "507" in err and "CUDA out of memory" in err


@pytest.mark.parametrize(
    ("body", "what"),
    [
        (b"", "an empty body"),
        (b"   ", "pad bytes only"),
        (b'  {"status": "loa', "a payload cut in half"),
        (b"null", "a literal null"),
        (b"{}", "an empty object"),
    ],
)
def test_http_backend_load_rejects_a_truncated_padded_body(monkeypatch, capsys, body, what):
    """A proxy that gives up mid-pad leaves a 200 the padded route never finished.

    Measured: one byte at t=90s then silence is killed ~125s later and the client sees
    a 200 with an EMPTY body. Accepting it reports an unfinished load as done.
    """
    backend = HttpChatBackend("http://localhost:8888", "token")
    response = _FakeLoadResponse(body)
    monkeypatch.setattr(backend, "_request", lambda *a, **k: response)

    with pytest.raises(typer.Exit) as excinfo:
        backend.ensure_loaded(
            "org/model-GGUF",
            hf_token = None,
            max_seq_length = 4096,
            load_in_4bit = True,
        )

    assert excinfo.value.exit_code == 1, what
    err = capsys.readouterr().err
    assert "Model load failed" in err
    assert "did not report completion" in err
    # Still drained, so the load is not abandoned at the headers.
    assert response.reads == 1 and response.closed


def test_padded_body_helper_passes_a_real_payload_through():
    from unsloth_cli._inference import require_completed_padded_body

    body = {"status": "loaded", "model": "org/model-GGUF"}
    assert require_completed_padded_body("http://x/api/inference/load", body) is body
    assert require_completed_padded_body("http://x", {"status": "unloaded"}) == {
        "status": "unloaded"
    }


def test_padded_body_helper_names_the_route_and_the_recovery():
    from unsloth_cli._inference import require_completed_padded_body
    url = "http://x/api/inference/load"
    for body in (None, {}, [], "", 0, "loaded"):
        with pytest.raises(RuntimeError) as excinfo:
            require_completed_padded_body(url, body)
        message = str(excinfo.value)
        assert message.startswith(f"{url} did not report completion")
        assert "Check the model's status" in message


def test_deferred_error_helper_passes_a_normal_body_through():
    from unsloth_cli._inference import raise_for_deferred_error

    body = {"status": "loaded", "model": "org/model-GGUF"}
    assert raise_for_deferred_error("http://x/api/inference/load", body) is body
    # Not a dict, and a look-alike that is not the documented shape, both pass.
    assert raise_for_deferred_error("http://x", [1, 2]) == [1, 2]
    assert raise_for_deferred_error("http://x", {"_deferred_error": None}) == {
        "_deferred_error": None
    }


def test_deferred_error_helper_reads_like_a_real_error_response():
    """Callers recover the detail with exc.read(), exactly as for a real 5xx."""
    import urllib.error

    from unsloth_cli._inference import raise_for_deferred_error

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        raise_for_deferred_error(
            "http://x/api/inference/load",
            {"_deferred_error": {"status_code": 500, "detail": "llama-server died"}},
        )
    exc = excinfo.value
    assert exc.code == 500
    assert json.loads(exc.read().decode()) == {"detail": "llama-server died"}
    assert "llama-server died" in str(exc)


def test_deferred_error_helper_defaults_a_missing_status():
    import urllib.error

    from unsloth_cli._inference import raise_for_deferred_error

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        raise_for_deferred_error("http://x", {"_deferred_error": {}})
    assert excinfo.value.code == 500


def _stub_studio_gguf_load(monkeypatch):
    """Stand in for the studio backend `_load_gguf_backend` imports in-venv, and
    return the list the intents it builds land in."""
    import unsloth_cli._inference as inference

    calls = []

    class _FakeLlamaCppBackend:
        def load_model(self, intent):
            calls.append(intent)
            return True

    fake_llama_cpp = types.ModuleType("core.inference.llama_cpp")
    fake_llama_cpp.GgufLoadIntent = lambda **kwargs: SimpleNamespace(**kwargs)
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
    return calls


@pytest.mark.parametrize(
    ("source", "expected_source"),
    [
        (
            {"gguf_hf_repo": "org/model-GGUF"},
            {"hf_repo": "org/model-GGUF", "hf_token": "hf_x"},
        ),
        (
            {
                "gguf_hf_repo": None,
                "gguf_file": "/models/model.gguf",
                "gguf_mmproj_file": "/models/mmproj.gguf",
                "gguf_mtp_file": "/models/mtp.gguf",
                "gguf_dspark_file": "/models/dspark-model.gguf",
                "gguf_dflash_file": "/models/dflash-kquant.gguf",
            },
            {
                "gguf_path": "/models/model.gguf",
                "mmproj_path": "/models/mmproj.gguf",
                "mtp_draft_path": "/models/mtp.gguf",
                "dspark_draft_path": "/models/dspark-model.gguf",
                "dflash_draft_path": "/models/dflash-kquant.gguf",
            },
        ),
    ],
    ids = ("hugging-face", "local"),
)
def test_load_gguf_backend_forwards_source_and_runtime_options(
    monkeypatch, source, expected_source
):
    import unsloth_cli._inference as inference

    calls = _stub_studio_gguf_load(monkeypatch)

    config = SimpleNamespace(
        gguf_variant = "Q4_K_M",
        identifier = "org/model-GGUF",
        is_vision = False,
        **source,
    )

    backend = inference._load_gguf_backend(
        config,
        hf_token = "hf_x",
        max_seq_length = 8192,
        tensor_parallel = True,
        speculative_type = "dspark",
        spec_draft_n_max = 3,
        llama_extra_args = ["--top-k", "20"],
    )

    assert isinstance(backend, ChatBackend)
    assert [vars(intent) for intent in calls] == [
        {
            "hf_variant": "Q4_K_M",
            "model_identifier": "org/model-GGUF",
            "is_vision": False,
            "n_ctx": 8192,
            "speculative_type": "dspark",
            "spec_draft_n_max": 3,
            "tensor_parallel": True,
            "extra_args": ["--top-k", "20"],
            **expected_source,
        }
    ]


def test_load_gguf_backend_hands_a_local_dflash_sidecar_to_the_load(monkeypatch):
    """The managed CLI resolves the sidecar next to a local weight exactly as Unsloth
    does, and dropping it here is silent: the load simply comes up with no drafter and
    nothing says the sidecar sitting beside the model was ever found."""
    import unsloth_cli._inference as inference

    calls = _stub_studio_gguf_load(monkeypatch)
    config = SimpleNamespace(
        gguf_variant = "Q4_K_M",
        identifier = "org/model-GGUF",
        is_vision = False,
        gguf_hf_repo = None,
        gguf_file = "/models/model.gguf",
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = "/models/dflash-kquant.gguf",
    )

    inference._load_gguf_backend(config, hf_token = None, max_seq_length = 8192)

    assert [intent.dflash_draft_path for intent in calls] == ["/models/dflash-kquant.gguf"]


def test_load_gguf_backend_exits_cleanly_on_invalid_extra_args(monkeypatch):
    import unsloth_cli._inference as inference

    fake_llama_cpp = types.ModuleType("core.inference.llama_cpp")
    fake_llama_cpp.GgufLoadIntent = object
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
        def load_model(self, intent):
            calls.append(intent)
            return intent.tensor_parallel is False

    fake_llama_cpp = types.ModuleType("core.inference.llama_cpp")
    fake_llama_cpp.GgufLoadIntent = lambda **kwargs: SimpleNamespace(**kwargs)
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
    assert [intent.tensor_parallel for intent in calls] == [True, False]
    assert calls[1].extra_args == ["--split-mode", "layer"]


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
            "--speculative-type",
            "dspark",
            "--spec-draft-n-max",
            "3",
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
                "speculative_type": "dspark",
                "spec_draft_n_max": 3,
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
            "--speculative-type",
            "dspark",
            "--spec-draft-n-max",
            "3",
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
                "speculative_type": "dspark",
                "spec_draft_n_max": 3,
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


def test_catalog_local_folder_entries_require_loadable_payloads(monkeypatch, tmp_path):
    """Config-only and GGUF-companion directories are not selectable model payloads."""
    from unsloth_cli._inference import ensure_studio_backend_path
    from unsloth_cli import _model_catalog as cat

    ensure_studio_backend_path()

    class _LocalModelInfo:
        def __init__(
            self,
            id,
            display_name,
            path,
            source,
            model_format = None,
            partial = False,
        ):
            self.id = id
            self.display_name = display_name
            self.path = path
            self.source = source
            self.model_format = model_format
            self.partial = partial
            self.load_id = None

    def _dir(name, files):
        d = tmp_path / name
        d.mkdir()
        for file in files:
            # A config has to parse: an unreadable one is its own reason not to load.
            if file.endswith("config.json"):
                (d / file).write_text(json.dumps({"model_type": "qwen3"}))
            else:
                (d / file).write_bytes(b"\0" * 8)
        return d

    config_only = _dir("ConfigOnly", ["config.json"])
    real = _dir("Real", ["config.json", "model.safetensors"])
    gguf_folder = _dir("GgufFolder", ["Qwen3-0.6B-Q4_K_M.gguf"])
    companions = _dir("Companions", ["mmproj-F16.gguf", "mtp-drafter-Q8_0.gguf"])
    pipeline = _dir("Pipeline", ["model_index.json"])
    (pipeline / "unet").mkdir()
    modular = _dir("OpaqueModular", ["modular_model_index.json"])
    (modular / "transformer").mkdir()
    single_file = tmp_path / "Tiny.gguf"
    single_file.write_bytes(b"\0" * 8)

    rows = [
        _LocalModelInfo(str(real), "Real", str(real), "models_dir"),
        _LocalModelInfo(str(config_only), "ConfigOnly", str(config_only), "models_dir"),
        _LocalModelInfo(
            str(gguf_folder), "GgufFolder", str(gguf_folder), "models_dir", model_format = "gguf"
        ),
        _LocalModelInfo(str(companions), "Companions", str(companions), "models_dir"),
        _LocalModelInfo(str(pipeline), "Pipeline", str(pipeline), "custom"),
        _LocalModelInfo(str(modular), "OpaqueModular", str(modular), "custom"),
        _LocalModelInfo(
            str(single_file), "Tiny", str(single_file), "lmstudio", model_format = "gguf"
        ),
    ]

    monkeypatch.setattr(cat, "_local_catalog_rows", lambda: rows)
    monkeypatch.setattr(cat, "_local_model_task", lambda model: None)
    monkeypatch.setattr(cat, "_local_model_can_chat", lambda model: None)
    monkeypatch.setattr(
        cat,
        "_local_is_a_diffusers_pipeline",
        lambda model: (
            (Path(model.path) / "model_index.json").is_file()
            or (Path(model.path) / "modular_model_index.json").is_file()
        ),
    )

    # The pipeline still HOLDS a payload; it is excluded from the chat picker for the
    # separate reason that a diffusers pipeline cannot answer a text turn.
    assert cat._local_dir_holds_a_payload(pipeline) is True
    assert cat._local_dir_holds_a_payload(modular) is True
    assert cat._local_dir_holds_a_payload(companions) is False

    assert [e.name for e in cat.local_folder_entries()] == ["Real", "GgufFolder", "Tiny"]


def test_catalog_trained_and_exported_entries_drop_non_chat_checkpoints(monkeypatch, tmp_path):
    """Unsloth trains Whisper and other audio models, so an outputs or exports folder
    legitimately holds a checkpoint that cannot answer a text turn. Neither builder
    classified anything, so both offered it for chat."""
    from unsloth_cli._inference import ensure_studio_backend_path
    from unsloth_cli import _model_catalog as cat

    ensure_studio_backend_path()
    from hub.services.models import catalog_classification

    def _ckpt(parent, name, config):
        d = parent / name
        d.mkdir(parents = True)
        (d / "config.json").write_text(json.dumps(config))
        (d / "model.safetensors").write_bytes(b"\0" * 8)
        return d

    def _adapter(parent, name, base):
        d = parent / name
        d.mkdir(parents = True)
        config = {"base_model_name_or_path": str(base)} if base else {}
        (d / "adapter_config.json").write_text(json.dumps(config))
        (d / "adapter_model.safetensors").write_bytes(b"\0" * 8)
        return d

    whisper = {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}
    causal = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    trained_whisper = _ckpt(tmp_path / "outputs", "whisper-finetune", whisper)
    trained_chat = _ckpt(tmp_path / "outputs", "qwen-finetune", causal)
    trained_whisper_adapter = _adapter(tmp_path / "outputs", "whisper-adapter", trained_whisper)
    trained_chat_adapter = _adapter(tmp_path / "outputs", "qwen-adapter", trained_chat)
    trained_run_adapter = _adapter(tmp_path / "outputs", "run-whisper-adapter", None)
    exported_whisper = _ckpt(tmp_path / "exports", "checkpoint-whisper", whisper)
    exported_chat = _ckpt(tmp_path / "exports", "checkpoint-qwen", causal)
    exported_whisper_adapter = _adapter(
        tmp_path / "exports", "checkpoint-whisper-adapter", trained_whisper
    )
    exported_gguf = tmp_path / "exports" / "run-gguf" / "model-Q4_K_M.gguf"
    exported_gguf.parent.mkdir(parents = True)
    exported_gguf.write_bytes(b"\0" * 8)
    exported_image_gguf = tmp_path / "exports" / "run-image-gguf" / "model-Q4_K_M.gguf"
    exported_image_gguf.parent.mkdir(parents = True)
    exported_image_gguf.write_bytes(b"\0" * 8)
    exported_asr_gguf = tmp_path / "exports" / "dictation" / "model-Q8_0.gguf"
    exported_asr_gguf.parent.mkdir(parents = True)
    exported_asr_gguf.write_bytes(b"\0" * 8)

    fake_models = types.ModuleType("utils.models")
    fake_models.scan_trained_models = lambda: [
        (trained_whisper.name, str(trained_whisper), "merged"),
        (trained_chat.name, str(trained_chat), "merged"),
        (trained_whisper_adapter.name, str(trained_whisper_adapter), "lora"),
        (trained_chat_adapter.name, str(trained_chat_adapter), "lora"),
        (trained_run_adapter.name, str(trained_run_adapter), "lora"),
    ]
    fake_models.scan_exported_models = lambda: [
        ("whisper-export", str(exported_whisper), "merged", None),
        ("qwen-export", str(exported_chat), "merged", None),
        ("whisper-adapter", str(exported_whisper_adapter), "lora", str(trained_whisper)),
        ("gguf-export", str(exported_gguf), "gguf", None),
        ("image-gguf-export", str(exported_image_gguf), "gguf", None),
        (
            "dictation",
            str(exported_asr_gguf),
            "gguf",
            "Qwen/Qwen3-ASR-0.6B",
        ),
    ]
    monkeypatch.setitem(sys.modules, "utils.models", fake_models)
    monkeypatch.setattr(
        cat,
        "_runs_by_output_dir",
        lambda: {str(trained_run_adapter): {"model_name": str(trained_whisper)}},
    )

    monkeypatch.setattr(
        catalog_classification,
        "_gguf_architecture",
        lambda path: (
            "flux"
            if Path(path) == exported_image_gguf
            else "qwen3"
            if Path(path) == exported_asr_gguf
            else "llama"
        ),
    )

    assert [e.name for e in cat.trained_entries()] == ["qwen-finetune", "qwen-adapter"]
    assert [e.name for e in cat.exported_entries()] == ["qwen-export", "gguf-export"]


def test_catalog_adapter_classifies_the_exact_cached_base_revision(monkeypatch, tmp_path):
    from unsloth_cli import _model_catalog as cat

    monkeypatch.syspath_prepend(str(_REPO_ROOT / "studio" / "backend"))
    import huggingface_hub
    from utils import hf_cache_settings

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "openai/whisper-large-v3",
                "revision": "base-commit",
            }
        )
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"\0" * 8)
    base = tmp_path / "cache" / "models--openai--whisper-large-v3" / "snapshots" / "base-commit"
    base.mkdir(parents = True)
    (base / "config.json").write_text(json.dumps({"model_type": "whisper"}))

    calls = []

    def _cached(repo_id, filename, *, cache_dir, revision):
        calls.append((repo_id, filename, cache_dir, revision))
        return str(base / filename)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _cached)
    monkeypatch.setattr(
        hf_cache_settings,
        "get_hf_cache_paths",
        lambda: types.SimpleNamespace(hub_cache = tmp_path / "cache"),
    )

    assert cat._path_can_chat(str(adapter)) is False
    assert calls == [
        (
            "openai/whisper-large-v3",
            "config.json",
            tmp_path / "cache",
            "base-commit",
        )
    ]


def _catalog_local_row(
    path,
    name = None,
    source = "models_dir",
    model_format = None,
):
    return types.SimpleNamespace(
        id = str(path),
        display_name = name or Path(path).name,
        path = str(path),
        source = source,
        model_format = model_format,
        partial = False,
        load_id = None,
    )


def test_catalog_rejects_incomplete_local_and_exported_payloads(monkeypatch, tmp_path):
    """A zero-byte weight and a shard set short of its own total are not payloads.

    Both survive resolve_model_config() and fail only once the loader opens the file, so the
    picker has to reject them here rather than offer a model that cannot load.
    """
    from unsloth_cli._inference import ensure_studio_backend_path
    from unsloth_cli import _model_catalog as cat

    ensure_studio_backend_path()
    config = json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})

    def _dir(name, files):
        d = tmp_path / name
        d.mkdir(parents = True)
        for filename, payload in files.items():
            (d / filename).write_bytes(payload)
        return d

    zero_gguf = _dir("zero_gguf", {"m-Q4_K_M.gguf": b""})
    lone_shard = _dir("lone_shard", {"s-Q4_K_M-00001-of-00003.gguf": b"\0" * 64})
    zero_export = _dir("zero_export", {"config.json": config.encode(), "model.safetensors": b""})
    good_gguf = _dir("good_gguf", {"m-Q4_K_M.gguf": b"\0" * 64})
    good_export = _dir(
        "good_export", {"config.json": config.encode(), "model.safetensors": b"\0" * 64}
    )
    whole_shards = _dir(
        "whole_shards",
        {
            "s-Q4_K_M-00001-of-00002.gguf": b"\0" * 64,
            "s-Q4_K_M-00002-of-00002.gguf": b"\0" * 64,
        },
    )

    # Directories: torn payloads out, every loadable shape kept.
    assert cat._local_dir_holds_a_payload(zero_gguf) is False
    assert cat._local_dir_holds_a_payload(lone_shard) is False
    assert cat._local_dir_holds_a_payload(zero_export) is False
    assert cat._local_dir_holds_a_payload(good_gguf) is True
    assert cat._local_dir_holds_a_payload(good_export) is True
    assert cat._local_dir_holds_a_payload(whole_shards) is True

    # A scan row can name the .gguf file itself, which is judged on its own shard family.
    assert cat._local_dir_holds_a_payload(zero_gguf / "m-Q4_K_M.gguf") is False
    assert cat._local_dir_holds_a_payload(lone_shard / "s-Q4_K_M-00001-of-00003.gguf") is False
    assert cat._local_dir_holds_a_payload(good_gguf / "m-Q4_K_M.gguf") is True
    assert cat._local_dir_holds_a_payload(whole_shards / "s-Q4_K_M-00001-of-00002.gguf") is True

    exports = [
        ("zero-gguf", str(zero_gguf / "m-Q4_K_M.gguf"), "gguf", None),
        ("lone-shard", str(lone_shard / "s-Q4_K_M-00001-of-00003.gguf"), "gguf", None),
        ("zero-export", str(zero_export), "merged", None),
        ("good-gguf", str(good_gguf / "m-Q4_K_M.gguf"), "gguf", None),
        ("good-export", str(good_export), "merged", None),
    ]
    fake_models = types.ModuleType("utils.models")
    fake_models.scan_exported_models = lambda: exports
    monkeypatch.setitem(sys.modules, "utils.models", fake_models)

    assert [e.name for e in cat.exported_entries()] == ["good-gguf", "good-export"]


def test_catalog_uses_only_the_two_specified_groups(monkeypatch, tmp_path):
    """The picker is specified as Fine-tunes and Downloaded; exports and scanned local
    folders belong inside those, not in two extra headings of their own."""
    from unsloth_cli._inference import ensure_studio_backend_path
    from unsloth_cli import _model_catalog as cat

    ensure_studio_backend_path()
    config = json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
    export = tmp_path / "export"
    export.mkdir()
    (export / "config.json").write_text(config)
    (export / "model.safetensors").write_bytes(b"\0" * 64)
    local = tmp_path / "local"
    local.mkdir()
    (local / "config.json").write_text(config)
    (local / "model.safetensors").write_bytes(b"\0" * 64)

    fake_models = types.ModuleType("utils.models")
    fake_models.scan_exported_models = lambda: [("an-export", str(export), "merged", None)]
    fake_models.scan_trained_models = lambda: []
    monkeypatch.setitem(sys.modules, "utils.models", fake_models)
    monkeypatch.setattr(cat, "_local_catalog_rows", lambda: [_catalog_local_row(local)])
    monkeypatch.setattr(cat, "_local_model_task", lambda model: None)
    monkeypatch.setattr(cat, "_local_model_can_chat", lambda model: None)
    monkeypatch.setattr(cat, "_local_is_a_diffusers_pipeline", lambda model: False)

    assert {e.group for e in cat.exported_entries()} == {"Fine-tunes"}
    assert {e.group for e in cat.local_folder_entries()} == {"Downloaded"}

    monkeypatch.setattr(cat, "cached_entries", lambda: [])
    monkeypatch.setattr(cat, "trained_entries", lambda: [])
    headings = []
    for entry in cat.list_chat_models():
        if not headings or headings[-1] != entry.group:
            headings.append(entry.group)
    assert headings == ["Fine-tunes", "Downloaded"]


def test_catalog_keeps_the_org_prefix_when_a_plain_name_collides(monkeypatch):
    """A fine-tune is already a bare name, so shortening a repo down to the same string
    produces two identical rows; the collision count has to see both."""
    from unsloth_cli import _model_catalog as cat

    trained = [cat.ModelEntry("Fine-tunes", "Qwen3-0.6B", "", "/outputs/run-1")]
    cached = [
        cat.ModelEntry("Downloaded", "unsloth/Qwen3-0.6B", "", "unsloth/Qwen3-0.6B"),
        cat.ModelEntry("Downloaded", "unsloth/Llama-3.2-1B", "", "unsloth/Llama-3.2-1B"),
    ]
    monkeypatch.setattr(cat, "trained_entries", lambda: trained)
    monkeypatch.setattr(cat, "cached_entries", lambda: cached)
    for name in ("exported_entries", "local_folder_entries"):
        monkeypatch.setattr(cat, name, lambda: [])

    names = [e.name for e in cat.list_chat_models()]
    assert names == ["Qwen3-0.6B", "unsloth/Qwen3-0.6B", "Llama-3.2-1B"]
    assert len(names) == len(set(names))


def test_catalog_drops_an_export_with_no_loadable_payload(monkeypatch, tmp_path):
    """scan_exported_models types a checkpoint "lora" on adapter_config.json ALONE.

    The merged branch checks has_weights; the LoRA one does not. So an interrupted export
    leaves a config with no adapter_model.safetensors or .bin, and nothing is "torn" there
    because there is no payload at all -- which a negative-only check reads as fine. PEFT's
    load_peft_weights finds neither weight name locally and raises, so the row was a path
    the picker could offer and nothing could load.
    """
    from unsloth_cli._inference import ensure_studio_backend_path
    from unsloth_cli import _model_catalog as cat

    ensure_studio_backend_path()

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    whole = tmp_path / "whole"
    whole.mkdir()
    (whole / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (whole / "adapter_model.safetensors").write_bytes(b"\0" * 2048)

    monkeypatch.setattr(cat, "_path_can_chat", lambda *a, **k: None)
    # Warm the real hub.* chain BEFORE shadowing utils.models: _local_dir_holds_a_payload
    # imports through it, and a stub package would break that import rather than the test.
    assert cat._local_dir_holds_a_payload(whole) is True
    assert cat._local_dir_holds_a_payload(broken) is False

    fake_models = types.ModuleType("utils.models")
    fake_models.scan_exported_models = lambda: [
        ("broken / ckpt", str(broken), "lora", None),
        ("whole / ckpt", str(whole), "lora", None),
    ]
    monkeypatch.setitem(sys.modules, "utils.models", fake_models)

    assert [e.name for e in cat.exported_entries()] == ["whole / ckpt"]


def test_catalog_hands_a_cached_gguf_pick_a_local_file(monkeypatch, tmp_path):
    """An ACTIVE-cache row carries the bare repo id, which is right for the inventory and
    wrong for a pick: from_identifier sends a non-path through detect_gguf_model_remote,
    whose GatedRepoError branch returns None WITHOUT the local-cache fallback it uses when
    offline. A cached gated GGUF then reads as non-GGUF and falls to the Transformers loader.
    """
    from unsloth_cli import _model_catalog as cat

    repo = tmp_path / "models--Org--Gated-GGUF"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "gated-Q4_K_M.gguf").write_bytes(b"\0" * 4096)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)

    monkeypatch.setattr(
        cat,
        "_preferred_complete_gguf",
        lambda p: str(snapshot / "gated-Q4_K_M.gguf") if str(snapshot) in str(p) else None,
    )
    resolved = cat._cached_gguf_load_id(
        {"repo_id": "Org/Gated-GGUF", "load_id": "Org/Gated-GGUF", "cache_path": str(repo)}
    )
    assert resolved.endswith("gated-Q4_K_M.gguf"), resolved


def test_catalog_payload_gate_accepts_an_uppercase_gguf_suffix(tmp_path):
    """``glob("*.gguf")`` is case-sensitive on Linux and on a case-sensitive macOS volume
    while ``_is_main_gguf_filename`` lowercases first, so a folder holding ``Model.GGUF``
    was classified as a GGUF model and then dropped by this gate, which decides whether the
    row is listed at all."""
    from unsloth_cli._inference import ensure_studio_backend_path
    from unsloth_cli import _model_catalog as cat

    ensure_studio_backend_path()
    upper = tmp_path / "upper"
    upper.mkdir()
    (upper / "Model.GGUF").write_bytes(b"GGUF" + b"\0" * 4096)
    assert cat._local_dir_holds_a_payload(upper) is True


def test_catalog_local_gguf_folder_picks_the_preferred_quant(monkeypatch, tmp_path):
    """A GGUF DIRECTORY goes through detect_gguf_model, which sorts by size and takes the
    largest complete file, commonly the F16, while cached and exported rows resolve a
    Q4-class quant: the same folder loaded a far bigger model by source alone."""
    from unsloth_cli import _model_catalog as cat

    folder = tmp_path / "multi"
    folder.mkdir()
    (folder / "mymodel-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * 4096)
    (folder / "mymodel-F16.gguf").write_bytes(b"GGUF" + b"\0" * 4096 * 40)

    row = SimpleNamespace(
        source = "models_dir",
        partial = False,
        model_format = "gguf",
        path = str(folder),
        display_name = "multi",
        load_id = str(folder),
        id = str(folder),
    )
    monkeypatch.setattr(cat, "_local_catalog_rows", lambda: [row])
    monkeypatch.setattr(cat, "_local_model_task", lambda m: None)
    monkeypatch.setattr(cat, "_local_model_can_chat", lambda m: None)
    monkeypatch.setattr(cat, "_local_is_a_diffusers_pipeline", lambda m: False)
    monkeypatch.setattr(cat, "_local_dir_holds_a_payload", lambda p: True)
    monkeypatch.setattr(
        cat, "_preferred_complete_gguf", lambda p: str(folder / "mymodel-Q4_K_M.gguf")
    )

    entries = cat.local_folder_entries()
    assert entries and entries[0].model.endswith("mymodel-Q4_K_M.gguf"), entries[0].model


def test_catalog_pins_an_active_cache_adapter_to_its_snapshot(tmp_path):
    """A LoRA resolved by bare repo id takes the REMOTE branch of
    get_base_model_from_lora_identifier, which downloads adapter_config.json with no
    local_files_only; for a cached gated adapter with no token that fails and the base is
    never read. A snapshot path takes the local reader. An ordinary cached repo is
    deliberately left on its id."""
    from unsloth_cli import _model_catalog as cat

    repo = tmp_path / "models--Org--GatedLora"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)

    adapter = {
        "repo_id": "Org/GatedLora",
        "load_id": "Org/GatedLora",
        "model_format": "adapter",
        "cache_path": str(repo),
    }
    assert cat._cached_model_load_id(adapter) == str(snapshot)

    plain = {
        "repo_id": "Org/Chat",
        "load_id": "Org/Chat",
        "model_format": "safetensors",
        "cache_path": str(repo),
    }
    assert cat._cached_model_load_id(plain) == "Org/Chat"


QUANT_LAYOUTS = [
    ([("Tiny-Q4_K_M.gguf", 16)], "Q4_K_M"),
    ([("Tiny-Q4_K_M.gguf", 16), ("Tiny-Q8_0.gguf", 16)], "Q4_K_M, Q8_0"),
    # One directory per quant. snapshots/*/*.gguf is one level deep, so this rendered blank.
    ([("Q4_K_M/Tiny-Q4_K_M.gguf", 16), ("Q8_0/Tiny-Q8_0.gguf", 16)], "Q4_K_M, Q8_0"),
    # A split quant is ONE thing to pick. The glob listed every shard as its own label.
    ([(f"Tiny-Q4_K_M-0000{n}-of-00003.gguf", 16) for n in (1, 2, 3)], "Q4_K_M"),
    # The case the host decides: fnmatch normcases on Windows and not on Linux or macOS, so
    # "*.gguf" found this file on one platform only and the cache read differently per machine.
    ([("Tiny-Q8_0.GGUF", 16)], "Q8_0"),
    ([("Tiny-Q4_K_M.gguf", 16), ("mmproj-F16.gguf", 16)], "Q4_K_M"),
]


@pytest.mark.parametrize("files,expected", QUANT_LAYOUTS)
def test_quant_labels_read_the_layouts_the_hub_actually_writes(
    monkeypatch, tmp_path, files, expected
):
    from unsloth_cli import _model_catalog as cat

    monkeypatch.syspath_prepend(str(_REPO_ROOT / "studio" / "backend"))
    repo = tmp_path / "models--org--Tiny-GGUF"
    for name, size in files:
        target = repo / "snapshots" / "abc" / name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"\0" * size)

    assert cat._quant_labels("org/Tiny-GGUF", str(repo)) == expected


def test_quant_labels_follow_the_pin_rather_than_the_ref(monkeypatch, tmp_path):
    """A pinned row loads its snapshot, not the one refs/main names.

    The inventory pins precisely when the ref resolves somewhere worse, so labelling by the
    ref advertised a quant the row will never open and hid the one it will.
    """
    from unsloth_cli import _model_catalog as cat

    monkeypatch.syspath_prepend(str(_REPO_ROOT / "studio" / "backend"))
    repo = tmp_path / "models--org--Two-GGUF"
    by_ref = repo / "snapshots" / ("a" * 40)
    pinned = repo / "snapshots" / ("b" * 40)
    by_ref.mkdir(parents = True)
    pinned.mkdir(parents = True)
    (by_ref / "Two-Q2_K.gguf").write_bytes(b"\0" * 16)
    (pinned / "Two-Q6_K.gguf").write_bytes(b"\0" * 16)
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text("a" * 40)

    assert cat._quant_labels("org/Two-GGUF", str(repo), str(pinned)) == "Q6_K"
    assert cat._quant_labels("org/Two-GGUF", str(repo)) == "Q2_K"


def test_quant_labels_leave_out_a_quant_that_cannot_be_loaded(monkeypatch, tmp_path):
    """An interrupted second quant is not selectable, so it is not advertised.

    _preferred_complete_gguf narrows to complete_snapshot_variants before choosing the load
    target; the detail column reads the same set, or it names a quant nothing can open.
    """
    from unsloth_cli import _model_catalog as cat

    monkeypatch.syspath_prepend(str(_REPO_ROOT / "studio" / "backend"))
    repo = tmp_path / "models--org--Tiny-GGUF"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "Tiny-Q4_K_M.gguf").write_bytes(b"\0" * 16)
    (snapshot / "Tiny-Q8_0-00001-of-00003.gguf").write_bytes(b"\0" * 16)
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text("a" * 40)

    assert cat._quant_labels("org/Tiny-GGUF", str(repo)) == "Q4_K_M"


def test_one_undecodable_ref_does_not_empty_the_downloaded_group(monkeypatch, tmp_path):
    """read_text raises UnicodeDecodeError, which is a ValueError and not an OSError.

    Uncaught it leaves _quant_labels, aborts cached_entries, and _safe then hides every
    Downloaded row because one repo in the cache has a torn ref.
    """
    from unsloth_cli import _model_catalog as cat

    monkeypatch.syspath_prepend(str(_REPO_ROOT / "studio" / "backend"))
    repo = tmp_path / "models--org--Tiny-GGUF"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "Tiny-Q4_K_M.gguf").write_bytes(b"\0" * 16)
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_bytes(b"\xff\xfe\x00not utf-8")

    rows = [{"repo_id": "org/Tiny-GGUF", "cache_path": str(repo), "task": "text-generation"}]
    monkeypatch.setattr(cat, "_cached_catalog_rows", lambda: (rows, []))

    entries = cat.cached_entries()
    assert [(e.name, e.detail) for e in entries] == [("org/Tiny-GGUF", "Q4_K_M")]


def test_a_pin_survives_a_symlinked_cache_root(monkeypatch, tmp_path):
    """inventory_scan resolves the snapshot it pins; cache_path keeps the configured spelling.

    Under a symlinked cache root (or a Windows junction) those two name one directory and
    are not lexically equal, so a membership test against the listed snapshots dropped a
    good pin back onto the ref it was pinned away from.
    """
    from unsloth_cli import _model_catalog as cat

    monkeypatch.syspath_prepend(str(_REPO_ROOT / "studio" / "backend"))
    physical = tmp_path / "physical_hub"
    physical.mkdir()
    repo = physical / "models--org--Two-GGUF"
    by_ref = repo / "snapshots" / ("a" * 40)
    pinned = repo / "snapshots" / ("b" * 40)
    by_ref.mkdir(parents = True)
    pinned.mkdir(parents = True)
    (by_ref / "Two-Q2_K.gguf").write_bytes(b"\0" * 16)
    (pinned / "Two-Q6_K.gguf").write_bytes(b"\0" * 16)
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text("a" * 40)

    configured = tmp_path / "configured_hub"
    configured.symlink_to(physical, target_is_directory = True)
    cache_path = configured / "models--org--Two-GGUF"

    assert cat._quant_labels("org/Two-GGUF", str(cache_path), str(pinned.resolve())) == "Q6_K"
