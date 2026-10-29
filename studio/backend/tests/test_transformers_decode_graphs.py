# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The transformers chat backend runs generate() with CUDA-graph decode on by default (NVIDIA CUDA
only, UNSLOTH_DECODE_CUDA_GRAPHS=0 opts out), and unloading a model frees its graphs and decode
buffers. The real unsloth/models/_decode_graphs.py is loaded by file path: importing unsloth needs
an accelerator, and its top level is torch and the standard library only."""

import importlib
import importlib.util
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_DECODE_GRAPHS = _BACKEND_DIR.parent.parent / "unsloth" / "models" / "_decode_graphs.py"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

try:
    import core.inference.inference  # noqa: E402,F401
except ImportError:
    pass

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)

ENV = "UNSLOTH_DECODE_CUDA_GRAPHS"


@pytest.fixture
def graphs(monkeypatch):
    """The real decode-graph module, reachable as unsloth.models._decode_graphs, with the runtime
    decision (no programmatic override, variable unset) on a spoofed NVIDIA CUDA host."""
    spec = importlib.util.spec_from_file_location("_studio_decode_graphs", _DECODE_GRAPHS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ENABLED = None
    module._DEFAULT_OK = True
    unsloth = types.ModuleType("unsloth")
    models = types.ModuleType("unsloth.models")
    unsloth.models = models
    models._decode_graphs = module
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.models", models)
    monkeypatch.setitem(sys.modules, "unsloth.models._decode_graphs", module)
    monkeypatch.delenv(ENV, raising = False)
    return module


class _Batch(dict):
    def to(self, *_args, **_kwargs):
        return self


class _Tokenizer:
    all_special_tokens: list = []
    eos_token_id = 1
    pad_token_id = None
    pieces = {2: "Hello", 3: " world"}

    def __call__(self, *_args, **_kwargs):
        return _Batch({"input_ids": torch.zeros((1, 1), dtype = torch.long)})

    def decode(self, ids, **kwargs):
        return "".join(self.pieces.get(int(i), "") for i in ids)


class _Model:
    device = "cpu"
    generation_config = type("Cfg", (), {"eos_token_id": 1})()
    config = generation_config

    def __init__(self, graphs):
        self.graphs = graphs
        self.seen = []

    def generate(self, streamer, max_new_tokens, **_kwargs):
        # What unsloth_fast_generate decides on the thread generate() runs on.
        self.seen.append(self.graphs.enabled())
        ids = torch.tensor([[0, 2, 3]])
        streamer.put(ids[:, :1])
        streamer.put(torch.tensor([2]))
        streamer.put(torch.tensor([3]))
        streamer.end()
        return ids


def _backend(graphs):
    inf = pytest.importorskip("core.inference.inference")
    model = _Model(graphs)
    tokenizer = _Tokenizer()
    streamer = inf.TextIteratorStreamer(tokenizer, skip_prompt = True, timeout = 0.2)
    backend = inf.InferenceBackend.__new__(inf.InferenceBackend)
    backend.active_model_name = "m"
    backend._generation_lock = threading.Lock()
    backend.models = {"m": {"model": model, "tokenizer": tokenizer}}
    backend._make_text_streamer = lambda *_args, **_kwargs: streamer
    return backend, model


def _chat(backend):
    return list(backend.generate_stream("PROMPT", temperature = 0.0, max_new_tokens = 4))


def test_chat_generate_runs_with_graphs_on_by_default(graphs):
    backend, model = _backend(graphs)
    assert _chat(backend)[-1] == "Hello world"
    assert model.seen == [True]
    # Scoped to the generate thread: this thread, and Core's plain generate(), stay eager.
    assert graphs.enabled() is False


def test_opt_out_variable_still_disables_graphs(graphs, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    backend, model = _backend(graphs)
    _chat(backend)
    assert model.seen == [False]


def test_non_nvidia_hosts_stay_eager(graphs, monkeypatch):
    # ROCm / XPU / MPS / CPU: _default_ok() is False there.
    monkeypatch.setattr(graphs, "_DEFAULT_OK", False)
    backend, model = _backend(graphs)
    _chat(backend)
    assert model.seen == [False]


def test_an_unsloth_without_the_scope_keeps_generate_unchanged(graphs, monkeypatch):
    monkeypatch.delattr(graphs, "studio_generation", raising = False)
    backend, model = _backend(graphs)
    assert _chat(backend)[-1] == "Hello world"
    assert model.seen == [False]


def _graphed_model(graphs):
    model = torch.nn.Module()
    attn = torch.nn.Module()
    model.attn = attn
    graphs._state(model).graphs[(1, "cuda")] = object()
    attn._unsloth_graph_buffers = {(1, torch.float16, "cuda"): ()}
    attn.paged_attention = torch.zeros(1)
    attn.attention = torch.zeros(1)
    return model, attn


def _unload(backend, monkeypatch):
    inf = sys.modules["core.inference.inference"]
    monkeypatch.setattr(inf, "clear_gpu_cache", lambda: None)
    cleanup = pytest.importorskip("utils.cache_cleanup")
    monkeypatch.setattr(cleanup, "clear_unsloth_compiled_cache", lambda **_kw: None)
    return backend.unload_model("m")


def test_unload_frees_decode_graphs_and_buffers(graphs, monkeypatch):
    backend, _ = _backend(graphs)
    model, attn = _graphed_model(graphs)
    backend.models["m"]["model"] = model
    assert _unload(backend, monkeypatch) is True
    assert graphs.stats(model)["graphs"] == 0
    assert "_unsloth_graph_buffers" not in attn.__dict__
    assert "paged_attention" not in attn.__dict__ and "attention" not in attn.__dict__
    assert backend.models == {}


def test_unload_leaves_a_running_generation_alone(graphs, monkeypatch):
    backend, _ = _backend(graphs)
    model, attn = _graphed_model(graphs)
    backend.models["m"]["model"] = model
    with backend._generation_lock:
        assert _unload(backend, monkeypatch) is True
    # The generate() thread still owns the buffers; they go with its model reference.
    assert "_unsloth_graph_buffers" in attn.__dict__
    assert graphs.stats(model)["graphs"] == 1
    assert backend.models == {}
