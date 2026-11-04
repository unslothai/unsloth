# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Graph decode memory: make_room() frees decode buffers / graphs kept from earlier generate() calls only
when the new call would not fit beside them (measured free memory, mocked here), and
generate_with_fallback() redoes an out-of-memory graphed call once eagerly. CPU only: the module is
loaded by file path and the capture / CUDA memory queries are stand-ins."""

import importlib.util
import types
import warnings
from pathlib import Path

import pytest
import torch

ENV = "UNSLOTH_DECODE_CUDA_GRAPHS"
_SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_decode_graphs.py"
CUDA0 = torch.device("cuda", 0)
GB = 1 << 30


@pytest.fixture
def G(monkeypatch):
    spec = importlib.util.spec_from_file_location("_unsloth_decode_graphs_memory", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ENABLED = None
    module._DEFAULT_OK = True
    module._STUDIO_DEFAULT_OK = True
    monkeypatch.delenv(ENV, raising = False)
    module.emptied = 0

    def empty():
        module.emptied += 1

    module._empty_cache = empty
    return module


class _Attn(torch.nn.Module):
    pass


class _Model(torch.nn.Module):
    """Qwen3-4B-like dims; layer attention modules carry the graph-mode buffer store."""

    def __init__(self, layers = 2):
        super().__init__()
        self.config = types.SimpleNamespace(
            num_hidden_layers = 36, hidden_size = 2560, num_attention_heads = 32,
            num_key_value_heads = 8, head_dim = 128, intermediate_size = 9728, vocab_size = 151936,
        )  # fmt: skip
        self.attns = torch.nn.ModuleList(_Attn() for _ in range(layers))

    def keep(self, bsz, rows, dtype = torch.float16):
        # What a graphed generate() of batch `bsz` leaves behind (tiny stand-in tensors, real row count).
        for attn in self.attns:
            paged = torch.zeros((rows, 2, 1, 1, 1), dtype = dtype)
            store = attn.__dict__.setdefault("_unsloth_graph_buffers", {})
            store[(bsz, dtype, CUDA0)] = (paged, None, None, None, None, None)
            attn.paged_attention = paged

    def held(self):
        # Batch sizes with a kept cache (any layer).
        return len({key for a in self.attns for key in a.__dict__.get("_unsloth_graph_buffers", {})})


def _ids(bsz, prompt):
    return types.SimpleNamespace(shape = (bsz, prompt), device = CUDA0)


def _need(G, bsz, prompt):
    return G._call_bytes(_Model().config, bsz, prompt, torch.float16, 512)[1]


def test_make_room_frees_kept_buffers_when_the_call_does_not_fit(G):
    model = _Model()
    model.keep(1, 2816)
    model.keep(4, 2816)
    # bs16 x 2048 on Qwen3-4B: ~6.6 GB of KV cache + the prefill working set.
    need = _need(G, 16, 2048)
    assert 8 * GB < need < 11 * GB
    G._free_bytes = lambda device: int(1.1 * need)
    assert G.make_room(model, _ids(16, 2048), torch.float16, 512) is True
    assert model.held() == 0 and G.emptied == 1
    assert not any("paged_attention" in a.__dict__ for a in model.attns)


def test_make_room_keeps_buffers_when_there_is_room(G):
    model = _Model()
    model.keep(1, 2816)
    G._free_bytes = lambda device: 20 * GB
    assert G.make_room(model, _ids(16, 2048), torch.float16, 512) is False
    assert model.held() == 1 and G.emptied == 0


def test_make_room_counts_a_reusable_buffer_of_the_same_batch_size(G):
    model = _Model(layers = 1)
    rows, need = G._call_bytes(model.config, 4, 128, torch.float16, 512)
    # A kept bs4 cache with enough rows is reused: only the rest of the call must fit.
    model.keep(4, rows)
    reused = model.attns[0]._unsloth_graph_buffers[(4, torch.float16, CUDA0)][0]
    reused_bytes = reused.numel() * reused.element_size()
    G._free_bytes = lambda device: int(G.ROOM_MARGIN * (need - reused_bytes)) + 1
    assert G.make_room(model, _ids(4, 128), torch.float16, 512) is False
    assert model.held() == 1


def test_make_room_is_a_no_op_without_kept_state_or_off_cuda(G):
    model = _Model()
    G._free_bytes = lambda device: 0
    assert G.make_room(model, _ids(16, 2048), torch.float16, 512) is False
    model.keep(1, 2816)
    cpu_ids = types.SimpleNamespace(shape = (16, 2048), device = torch.device("cpu"))
    assert G.make_room(model, cpu_ids, torch.float16, 512) is False
    assert G.make_room(model, None, torch.float16, 512) is False
    assert model.held() == 1


def _oom_while_graphed(G, seen):
    def generate(*args, **kwargs):
        on = G.enabled()
        seen.append(on)
        if on:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (stand-in)")
        return "eager-output"

    return generate


def test_oom_with_graphs_redoes_the_call_once_eagerly_and_warns_once(G):
    model = _Model()
    seen = []
    for _ in range(2):
        model.keep(1, 2816)
        with G.generate_scope():
            graphs = G.enabled()
            with warnings.catch_warnings(record = True) as caught:
                warnings.simplefilter("always")
                out = G.generate_with_fallback(model, _oom_while_graphed(G, seen), graphs, (), {})
        assert out == "eager-output"
        assert model.held() == 0
        # Graphs are back on for the next call in the scope.
        with G.generate_scope():
            assert G.enabled() is True
    assert seen == [True, False, True, False]
    assert G._OOM_WARNED is True


def test_oom_warning_is_emitted_once_per_process(G):
    model = _Model()
    messages = []
    for _ in range(3):
        with G.generate_scope():
            with warnings.catch_warnings(record = True) as caught:
                warnings.simplefilter("always")
                G.generate_with_fallback(model, _oom_while_graphed(G, []), True, (), {})
            messages += [str(w.message) for w in caught if "ran out of GPU memory" in str(w.message)]
    assert len(messages) == 1


@pytest.mark.parametrize(
    "graphs, kwargs",
    [
        (False, {}),
        (True, {"streamer": object()}),
        (True, {"past_key_values": object()}),
    ],
)
def test_oom_is_raised_when_a_redo_is_not_safe(G, graphs, kwargs):
    model = _Model()
    model.keep(1, 2816)

    def generate(*args, **kw):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (stand-in)")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        G.generate_with_fallback(model, generate, graphs, (), kwargs)
    assert model.held() == 1


def test_oom_in_the_eager_redo_propagates(G):
    model = _Model()
    calls = []

    def generate(*args, **kw):
        calls.append(G.enabled())
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (stand-in)")

    with G.generate_scope():
        with pytest.raises(torch.cuda.OutOfMemoryError):
            G.generate_with_fallback(model, generate, True, (), {})
    assert calls == [True, False]


def test_eager_scope_beats_the_variable_and_the_scopes(G, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    with G.rl_generation(), G.eager_scope():
        assert G.enabled() is False
    assert G.enabled() is True


def _with_graphs(G, model, *bszs):
    state = G._state(model)
    for bsz in bszs:
        model.keep(bsz, 2816)
        state.graphs[(bsz, CUDA0, (False, None), True, None)] = object()
    state.pool = object()
    return state


def test_keep_only_drops_other_batch_sizes_and_keeps_this_ones_graphs(G):
    model = _Model()
    state = _with_graphs(G, model, 1, 4)
    epoch = G.BUFFER_EPOCH
    assert G.keep_only(model, 4) is True
    assert {k for a in model.attns for k in a._unsloth_graph_buffers} == {(4, torch.float16, CUDA0)}
    assert [k[0] for k in state.graphs] == [4] and state.pool is not None
    # The kept graphs still read valid buffers: no epoch bump, so the next call of this shape replays.
    assert G.BUFFER_EPOCH == epoch
    assert all(a.paged_attention is a._unsloth_graph_buffers[(4, torch.float16, CUDA0)][0] for a in model.attns)
    assert G.keep_only(model, 4) is False


def test_keep_only_drops_the_current_attributes_of_an_evicted_cache(G):
    model = _Model()
    _with_graphs(G, model, 4, 1)  # attn.paged_attention is the bs1 cache
    G.keep_only(model, 4)
    assert not any("paged_attention" in a.__dict__ for a in model.attns)


def test_training_forward_releases_what_a_graphed_generate_kept(G):
    model = _Model()
    _with_graphs(G, model, 4)
    assert G.release_if_training(model) is False  # nothing marked: one dict read, nothing freed
    assert model.held() == 1
    model.__dict__[G.HELD] = True
    assert G.release_if_training(model) is True
    assert model.held() == 0 and G.HELD not in model.__dict__
    assert G._state(model).graphs == {}
    assert G.release_if_training(model) is False


def test_release_clears_the_held_mark(G):
    model = _Model()
    _with_graphs(G, model, 1)
    model.__dict__[G.HELD] = True
    G.release_for_training(model)
    assert G.HELD not in model.__dict__


def test_eager_redo_is_counted(G):
    model = _Model()
    assert G.EAGER_REDOS == 0
    with G.generate_scope():
        G.generate_with_fallback(model, _oom_while_graphed(G, []), True, (), {})
        G.generate_with_fallback(model, lambda: "ok", True, (), {})
    assert G.EAGER_REDOS == 1


def test_causal_lm_forward_calls_the_training_release():
    """The hook lives in llama.py's _CausalLM_fast_forward (importing unsloth needs an accelerator,
    so this reads the source): a training forward with grad on releases what generate() kept."""
    import ast

    source = (_SOURCE.parent / "llama.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    fwd = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_CausalLM_fast_forward"
    )
    hooks = [
        n for n in ast.walk(fwd)
        if isinstance(n, ast.If) and "release_if_training" in ast.unparse(n)
        and "self.training" in ast.unparse(n.test) and "is_grad_enabled" in ast.unparse(n.test)
    ]  # fmt: skip
    assert hooks, "training forward does not free graphs a generate() kept"
    gen = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "unsloth_fast_generate")
    text = ast.unparse(gen)
    assert "keep_only" in text and "HELD" in text
