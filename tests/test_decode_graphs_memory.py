# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Graph decode memory planning and OOM fallback; CPU only, CUDA memory queries mocked."""

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
            model_type = "qwen3",
        )  # fmt: skip
        self.attns = torch.nn.ModuleList(_Attn() for _ in range(layers))

    def keep(
        self,
        bsz,
        rows,
        dtype = torch.float16,
    ):
        for attn in self.attns:
            paged = torch.zeros((rows, 2, 1, 1, 1), dtype = dtype)
            store = attn.__dict__.setdefault("_unsloth_graph_buffers", {})
            store[(bsz, dtype, CUDA0)] = (paged, None, None, None, None, None)
            attn.paged_attention = paged

    def held(self):
        return len(
            {key for a in self.attns for key in a.__dict__.get("_unsloth_graph_buffers", {})}
        )


def _ids(bsz, prompt):
    return types.SimpleNamespace(shape = (bsz, prompt), device = CUDA0)


def _need(
    G,
    bsz,
    prompt,
    max_len = 0,
):
    rows, kv, work = G._call_bytes(_Model().config, bsz, prompt, torch.float16, 512, max_len)
    return G.CALIB_WORK * (kv + work) + G.CALIB_GRAPH_KV * kv


def _plan(
    G,
    model,
    bsz,
    prompt,
    max_len = 0,
):
    return G.plan_call(model, _ids(bsz, prompt), torch.float16, 512, max_len)


def test_plan_frees_kept_buffers_when_the_call_does_not_fit_beside_them(G):
    model = _Model()
    model.keep(1, 2816)
    model.keep(4, 2816)
    need = _need(G, 16, 2048, 2081)
    assert 11 * GB < need < 13 * GB
    free = iter([int(0.9 * need), int(1.1 * need)])
    G._free_bytes = lambda device: next(free)
    plan, token = _plan(G, model, 16, 2048, 2081)
    assert plan == "graph" and token is not None
    assert model.held() == 0 and G.emptied == 1 and G.EAGER_UNFIT == 0
    assert not any("paged_attention" in a.__dict__ for a in model.attns)


def test_plan_holds_no_buffer_reference_when_it_empties_the_cache(G):
    import weakref

    model = _Model()
    model.keep(1, 2816)
    model.keep(4, 2816)
    model.keep(16, 64)  # too small to reuse, still scanned
    refs = [
        weakref.ref(bufs[0])
        for a in model.attns
        for bufs in a.__dict__["_unsloth_graph_buffers"].values()
    ]
    alive_at_empty = []

    def empty():
        G.emptied += 1
        alive_at_empty.append(sum(r() is not None for r in refs))

    G._empty_cache = empty
    need = _need(G, 16, 2048, 2081)
    free = iter([int(0.9 * need), int(1.1 * need)])
    G._free_bytes = lambda device: next(free)
    assert _plan(G, model, 16, 2048, 2081)[0] == "graph"
    assert alive_at_empty == [0]


def test_plan_runs_a_call_that_cannot_fit_eagerly_from_the_start(G):
    model = _Model()
    model.keep(4, 2816)
    G._free_bytes = lambda device: int(0.95 * _need(G, 16, 2048, 2081))
    plan, token = _plan(G, model, 16, 2048, 2081)
    assert (plan, token) == ("eager", None)
    # Kept state was freed first (the eager call needs the memory too), then the call was counted.
    assert model.held() == 0 and G.EAGER_UNFIT == 1
    assert _plan(G, model, 16, 2048, 2081)[0] == "eager" and G.EAGER_UNFIT == 2 and G.emptied == 1


def test_plan_keeps_buffers_when_there_is_room(G):
    model = _Model()
    model.keep(1, 2816)
    G._free_bytes = lambda device: 20 * GB
    assert _plan(G, model, 16, 2048)[0] == "graph"
    assert model.held() == 1 and G.emptied == 0 and G.EAGER_UNFIT == 0


def test_plan_counts_a_reusable_buffer_of_the_same_batch_size(G):
    model = _Model(layers = 1)
    rows, kv, work = G._call_bytes(model.config, 4, 128, torch.float16, 512)
    model.keep(4, rows)
    reused = model.attns[0]._unsloth_graph_buffers[(4, torch.float16, CUDA0)][0]
    G._free_bytes = (
        lambda device: int(_need(G, 4, 128) - reused.numel() * reused.element_size()) + 1
    )
    assert _plan(G, model, 4, 128)[0] == "graph"
    assert model.held() == 1


def test_plan_lets_non_cuda_or_shapeless_inputs_through(G):
    model = _Model()
    G._free_bytes = lambda device: 0
    cpu_ids = types.SimpleNamespace(shape = (16, 2048), device = torch.device("cpu"))
    assert G.plan_call(model, cpu_ids, torch.float16, 512) == ("graph", None)
    assert G.plan_call(model, None, torch.float16, 512) == ("graph", None)
    assert G.EAGER_UNFIT == 0


def test_measured_peak_can_only_raise_the_estimate(G):
    model = _Model()
    est = _need(G, 4, 2048, 2081)
    stats = {"alloc": 3 * GB, "peak": 3 * GB}
    G._alloc_stats = lambda device: (stats["alloc"], stats["peak"])
    G._free_bytes = lambda device: 20 * GB
    plan, token = _plan(G, model, 4, 2048, 2081)
    assert plan == "graph"
    stats["peak"] = 5 * GB  # below the estimate
    G.record_need(model, token)
    key = G._shape_key(4, 2048, 2081)
    assert model.__dict__[G.NEEDS][key] == 2 * GB < est
    # A measurement can only raise the estimate.
    G._free_bytes = lambda device: int(est) - 1
    assert _plan(G, model, 4, 2048, 2081)[0] == "eager"
    G._free_bytes = lambda device: int(est) + 1
    assert _plan(G, model, 4, 2048, 2081)[0] == "graph"
    G._free_bytes = lambda device: 20 * GB
    plan, token = _plan(G, model, 4, 2048, 2081)
    stats["peak"] = 3 * GB + 6 * GB
    G.record_need(model, token)
    assert model.__dict__[G.NEEDS][key] == 6 * GB > est
    G._free_bytes = lambda device: int(G.MEASURED_MARGIN * 6 * GB) - 1
    assert _plan(G, model, 4, 2048, 2081)[0] == "eager"
    G._free_bytes = lambda device: int(G.MEASURED_MARGIN * 6 * GB) + 1
    assert _plan(G, model, 4, 2048, 2081)[0] == "graph"
    G.record_need(model, _plan(G, model, 1, 128)[1])
    assert G._shape_key(1, 128, 0) not in model.__dict__[G.NEEDS]


def test_prompt_length_is_part_of_the_shape(G):
    # Same total-length bucket, different prompts: the key includes the prompt bucket.
    assert G._shape_key(4, 128, 4096) != G._shape_key(4, 2048, 4096)
    model = _Model()
    stats = {"alloc": 3 * GB, "peak": 3 * GB}
    G._alloc_stats = lambda device: (stats["alloc"], stats["peak"])
    G._free_bytes = lambda device: 30 * GB
    plan, token = _plan(G, model, 4, 128, 4096)
    stats["peak"] = 3 * GB + GB // 2
    G.record_need(model, token)
    long_need = _need(G, 4, 2048, 4096)
    G._free_bytes = lambda device: int(long_need) - 1
    assert _plan(G, model, 4, 2048, 4096)[0] == "eager"


def test_an_early_stop_measurement_never_admits_a_full_length_call(G):
    model = _Model()
    est = _need(G, 4, 2048, 4096)
    stats = {"alloc": 3 * GB, "peak": 3 * GB}
    G._alloc_stats = lambda device: (stats["alloc"], stats["peak"])
    G._free_bytes = lambda device: 30 * GB
    plan, token = _plan(G, model, 4, 2048, 4096)
    stats["peak"] = 3 * GB + GB
    G.record_need(model, token)
    G._free_bytes = lambda device: int(G.MEASURED_MARGIN * GB) + 1
    assert _plan(G, model, 4, 2048, 4096)[0] == "eager"
    G._free_bytes = lambda device: int(est) + 1
    assert _plan(G, model, 4, 2048, 4096)[0] == "graph"


def test_release_keeps_the_measured_needs(G):
    model = _Model()
    model.__dict__[G.NEEDS] = {(4, 2304, 2304): GB}
    model.keep(4, 2816)
    G.release_for_training(model)
    assert model.__dict__[G.NEEDS] == {(4, 2304, 2304): GB}


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
            messages += [
                str(w.message) for w in caught if "ran out of GPU memory" in str(w.message)
            ]
    assert len(messages) == 1


@pytest.mark.parametrize(
    "graphs, kwargs",
    [
        (False, {}),
        (True, {"streamer": object()}),
        (True, {"past_key_values": object()}),
        (True, {"stopping_criteria": [object()]}),
        (True, {"logits_processor": [object()]}),
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
    assert all(
        a.paged_attention is a._unsloth_graph_buffers[(4, torch.float16, CUDA0)][0]
        for a in model.attns
    )
    assert G.keep_only(model, 4) is False


def test_keep_only_never_evicts_the_cache_decode_just_used(G):
    model = _Model()
    _with_graphs(G, model, 4, 1)  # decode ran at bs 1
    G.keep_only(model, 4)
    assert {k for a in model.attns for k in a._unsloth_graph_buffers} == {(1, torch.float16, CUDA0)}
    assert all(
        a.paged_attention is a._unsloth_graph_buffers[(1, torch.float16, CUDA0)][0]
        for a in model.attns
    )


def test_training_forward_releases_what_a_graphed_generate_kept(G):
    model = _Model()
    _with_graphs(G, model, 4)
    assert G.release_if_training(model) is False
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
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_CausalLM_fast_forward"
    )
    hooks = [
        n for n in ast.walk(fwd)
        if isinstance(n, ast.If) and "release_if_training" in ast.unparse(n)
        and "self.training" in ast.unparse(n.test) and "is_grad_enabled" in ast.unparse(n.test)
    ]  # fmt: skip
    assert hooks, "training forward does not free graphs a generate() kept"
    gen = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "unsloth_fast_generate"
    )
    text = ast.unparse(gen)
    assert "keep_only" in text and "HELD" in text
    # A call the plan says cannot fit graphed runs eagerly from the start; graphed calls record their peak.
    assert "plan_call" in text and "eager_scope" in text and "record_need" in text
    # Plan and keep_only use the expanded decode batch.
    assert text.count("decode_batch(") >= 2


def test_measured_need_includes_a_reused_cache_and_is_discounted_once(G):
    # The stored footprint includes a reused cache, so the reuse is subtracted once.
    model = _Model(layers = 1)
    rows, kv, work = G._call_bytes(model.config, 4, 2048, torch.float16, 512, 2081)
    model.keep(4, rows)
    reused = model.attns[0]._unsloth_graph_buffers[(4, torch.float16, CUDA0)][0]
    reused_bytes = reused.numel() * reused.element_size()
    del reused
    stats = {"alloc": 3 * GB, "peak": 3 * GB}
    G._alloc_stats = lambda device: (stats["alloc"], stats["peak"])
    G._free_bytes = lambda device: 20 * GB
    plan, token = _plan(G, model, 4, 2048, 2081)
    assert plan == "graph"
    stats["peak"] = 11 * GB
    G.record_need(model, token)
    key = G._shape_key(4, 2048, 2081)
    total = 8 * GB + reused_bytes
    assert model.__dict__[G.NEEDS][key] == total
    assert G.MEASURED_MARGIN * total > _need(G, 4, 2048, 2081)
    need = G.MEASURED_MARGIN * total - reused_bytes
    G._free_bytes = lambda device: int(need) + 1
    assert _plan(G, model, 4, 2048, 2081)[0] == "graph" and model.held() == 1
    free = iter([int(need) - 1, int(G.MEASURED_MARGIN * total) - 1])
    G._free_bytes = lambda device: next(free)
    assert _plan(G, model, 4, 2048, 2081)[0] == "eager" and model.held() == 0


def test_decode_batch_follows_generate_expansion(G):
    cfg = types.SimpleNamespace(num_beams = 1, num_return_sequences = 1)
    assert G.decode_batch(3, {}, cfg) == 3
    assert G.decode_batch(3, {"num_return_sequences": 2, "do_sample": True}, cfg) == 6
    assert G.decode_batch(3, {"num_beams": 4, "num_return_sequences": 2}, cfg) == 12
    beams = types.SimpleNamespace(num_beams = 2, num_return_sequences = 1)
    assert G.decode_batch(3, {}, beams) == 6
    assert G.decode_batch(3, {"generation_config": beams}, cfg) == 6
    assert G.decode_batch(3, {}, None) == 3


def test_plan_sizes_the_expanded_batch(G):
    model = _Model()
    need = _need(G, 4, 2048, 2081)
    G._free_bytes = lambda device: int(need) - 1
    # 2 prompts x num_return_sequences 2 decode at bs 4.
    assert G.plan_call(model, _ids(2, 2048), torch.float16, 512, 2081, bsz = 4)[0] == "eager"
    G._free_bytes = lambda device: int(need) + 1
    plan, token = G.plan_call(model, _ids(2, 2048), torch.float16, 512, 2081, bsz = 4)
    assert plan == "graph" and token[0] == G._shape_key(4, 2048, 2081)


def test_keep_only_keeps_the_batch_size_decode_used(G):
    model = _Model()
    state = _with_graphs(G, model, 1, 4)  # decoded at bs 4
    assert G.decode_batch_used(model) == 4
    G.keep_only(model, None)
    assert {k for a in model.attns for k in a._unsloth_graph_buffers} == {(4, torch.float16, CUDA0)}
    assert [k[0] for k in state.graphs] == [4]


@pytest.mark.parametrize("model_type", ["mistral", "gemma", "gemma2"])
def test_a_model_type_that_never_graphs_keeps_eager_oom_and_planning(G, model_type):
    model = _Model()
    model.config.model_type = model_type
    model.keep(1, 2816)
    calls = []

    def generate(*args, **kw):
        calls.append(G.enabled())
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (stand-in)")

    with G.generate_scope():
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(torch.cuda.OutOfMemoryError):
                G.generate_with_fallback(model, generate, G.enabled(), (), {})
    assert calls == [True]
    assert not [w for w in caught if "ran out of GPU memory" in str(w.message)]
    assert G.EAGER_REDOS == 0 and model.held() == 1
    G._free_bytes = lambda device: 0
    assert G.plan_call(model, _ids(16, 2048), torch.float16, 256, 4096) == ("graph", None)
    assert G.EAGER_UNFIT == 0 and model.held() == 1 and G.emptied == 0


@pytest.mark.parametrize(
    "max_len, prompt, replays",
    [(0, 32, True), (33, 32, False), (34, 32, False), (35, 32, True), (288, 32, True)],
)
def test_a_call_too_short_to_replay_a_step_is_known(G, max_len, prompt, replays):
    assert G.can_replay(max_len, prompt) is replays


def _generate_fn():
    import ast
    tree = ast.parse((_SOURCE.parent / "llama.py").read_text(encoding = "utf-8"))
    return next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "unsloth_fast_generate"
    )


def test_generate_runs_a_short_call_eagerly_when_nothing_is_kept():
    """Reads llama.py: importing unsloth needs an accelerator."""
    import ast

    gen = _generate_fn()
    assert "short_call" in ast.unparse(gen)
    guards = [
        n for n in ast.walk(gen)
        if isinstance(n, ast.If) and "short" in ast.unparse(n.test) and "HELD" in ast.unparse(n.test)
    ]  # fmt: skip
    assert guards, "a short call is not routed eagerly"
    body = ast.unparse(guards[0].body[0]) if guards[0].body else ""
    assert "'eager'" in body or '"eager"' in body
    assert "plan_call" not in ast.unparse(guards[0].body[0])


def test_generate_runs_a_short_call_beside_kept_graphs_eagerly():
    import ast

    text = ast.unparse(_generate_fn())
    assert "eager=short" in text
    assert "if short:\n" in text and text.index("eager=short") < text.index("if short:\n")


def _keep_graph(model, bsz):
    state = model.__dict__.setdefault("_unsloth_decode_graphs", types.SimpleNamespace(graphs = {}))
    state.graphs[(bsz, CUDA0, None, True, None)] = object()


@pytest.mark.parametrize(
    "new, kept, short",
    [
        (1, None, True), (2, None, True), (2, 1, True), (3, None, True), (6, None, True),
        (6, 4, True), (7, None, False), (8, None, False), (32, None, False), (3, 1, False),
        (6, 1, False),
    ],
)  # fmt: skip
def test_a_short_call_decodes_eagerly_unless_a_graph_of_its_batch_is_kept(G, new, kept, short):
    model = _Model()
    if kept is not None:
        _keep_graph(model, kept)
    assert G.CAPTURE_MIN_STEPS == 6
    assert G.short_call(model, 32 + new, 32, 1) is short
    assert G.short_call(model, 0, 32, 1) is False


def test_a_short_call_matches_the_capture_break_even(G):
    model = _Model()
    for new in range(3, 40):
        G.start_generation(32 + new)
        try:
            assert G.short_call(model, 32 + new, 32, 1) is (not G.worth_capturing(32 + 2))
        finally:
            G.end_generation()


def test_a_short_call_beside_kept_graphs_only_makes_room(G):
    model = _Model()
    model.keep(4, 2816)
    G._free_bytes = lambda device: 100 * GB
    assert G.plan_call(model, _ids(1, 32), torch.float16, 512, 36, eager = True) == ("eager", None)
    assert model.held() == 1 and G.EAGER_UNFIT == 0
    G._free_bytes = lambda device: 0
    assert G.plan_call(model, _ids(1, 32), torch.float16, 512, 36, eager = True) == ("eager", None)
    assert model.held() == 0 and G.emptied == 1 and G.EAGER_UNFIT == 0


def test_capture_only_when_enough_steps_are_left(G):
    G.start_generation(32 + 8)
    try:
        assert G.CAPTURE_MIN_STEPS == 6
        assert G.worth_capturing(35) is True
        assert G.worth_capturing(36) is False
        G.start_generation(32 + 4)
        assert not any(G.worth_capturing(kv) for kv in range(33, 37))
    finally:
        G.end_generation()
    assert G.worth_capturing(10_000) is True


def test_decode_step_skips_only_a_new_capture():
    import ast

    tree = ast.parse(_SOURCE.read_text(encoding = "utf-8"))
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "decode_step"
    )
    guards = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.If) and "worth_capturing" in ast.unparse(n.test)
    ]  # fmt: skip
    assert guards and all("step is None" in ast.unparse(g.test) for g in guards)
    text = ast.unparse(fn)
    assert text.index("worth_capturing") < text.index("_capture(")


def test_an_eager_short_call_gets_no_credit_for_a_kept_graph_cache(G):
    model = _Model()
    model.keep(16, 8192)
    rows, kv, work = G._call_bytes(model.config, 16, 64, torch.float16, 512, 66)
    G._kept_buffers = lambda model, key, rows: (True, int(4 * (kv + work)))
    G._free_bytes = lambda device: int(0.5 * G.CALIB_WORK * (kv + work))
    assert G.plan_call(model, _ids(16, 64), torch.float16, 512, 66, eager = True) == ("eager", None)
    assert model.held() == 0 and G.emptied == 1 and G.EAGER_UNFIT == 0


def test_an_oversized_kept_cache_is_credited_only_up_to_this_call_kv(G):
    model = _Model()
    model.keep(4, 8192)
    rows, kv, work = G._call_bytes(model.config, 4, 128, torch.float16, 512, 160)
    G._kept_buffers = lambda model, key, rows: (True, int(20 * (kv + work)))
    G._free_bytes = lambda device: 0
    assert G.plan_call(model, _ids(4, 128), torch.float16, 512, 160) == ("eager", None)
    assert model.held() == 0 and G.emptied == 1 and G.EAGER_UNFIT == 1


def _llama_helpers(*names):
    import ast

    path = Path(_SOURCE).with_name("llama.py")
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {}
    exec(compile(ast.Module(body = fns, type_ignores = []), str(path), "exec"), namespace)
    return namespace


def test_planning_honours_an_explicit_max_length_in_the_caller_generation_config():
    transformers = pytest.importorskip("transformers")
    ns = _llama_helpers("_generation_max_len", "_generation_plan_len")
    model = types.SimpleNamespace(generation_config = transformers.GenerationConfig(max_length = 40960))
    ids = torch.zeros((1, 128), dtype = torch.long)
    config = transformers.GenerationConfig(max_length = 4096)
    assert ns["_generation_max_len"](model, (ids,), {"generation_config": config}) == 0
    assert ns["_generation_plan_len"](model, (ids,), {"generation_config": config}) == 4096
    # The model's default config is not a request, and max_new_tokens still wins.
    assert ns["_generation_plan_len"](model, (ids,), {}) == 0
    assert (
        ns["_generation_plan_len"](
            model, (ids,), {"generation_config": config, "max_new_tokens": 32}
        )
        == 160
    )


def test_cache_growth_cap_uses_the_planned_length():
    # The planner budgets plan_len rows; start_generation's hint caps the cache growth, so both must get it.
    import ast

    path = Path(_SOURCE).with_name("llama.py")
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "unsloth_fast_generate"
    )
    args = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("start_generation", "short_call", "plan_call"):
                args[node.func.attr] = node.args
    assert args["start_generation"][0].id == "plan_len"
    assert args["short_call"][1].id == "plan_len"
    assert args["plan_call"][4].id == "plan_len"
