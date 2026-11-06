# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

VOCAB, MASK_ID = 61, 60


class _Bidirectional(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, 16)
        self.mix = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, VOCAB)
        self.config = SimpleNamespace(mask_token_id = MASK_ID)
        self.calls = 0

    def get_output_embeddings(self):
        return self.head

    def forward(
        self,
        input_ids,
        attention_mask = None,
    ):
        self.calls += 1
        h = self.embed(input_ids)
        h = h + self.mix(h.mean(dim = 1, keepdim = True))
        logits = self.head(torch.tanh(h))
        return SimpleNamespace(logits = logits)


def _sample(
    model,
    progress = None,
    **overrides,
):
    from unsloth.models.diffusion_samplers import _llada_sample

    kwargs = dict(
        max_new_tokens = 16,
        steps = 16,
        block_length = 8,
        temperature = 0.0,
        mask_id = MASK_ID,
        eos_ids = [],
        progress = progress,
    )
    kwargs.update(overrides)
    return _llada_sample(model, None, torch.tensor([[1, 7, 9]]), None, **kwargs)


def test_block_head_matches_full_logits_without_building_them():
    from unsloth.models.diffusion_infer import BlockHead

    model = _Bidirectional()
    x = torch.randint(0, VOCAB, (2, 12))
    full = model(x).logits
    head = BlockHead(model)
    assert head.head is model.head
    seen = []
    model.head.register_forward_hook(lambda m, a, out: seen.append(tuple(out.shape)))
    model.head._forward_hooks.clear()  # a hooked head is not fusable; check the fused path with a clean one
    model.__dict__.pop("_unsloth_diffusion_fused_head", None)
    head = BlockHead(model)
    block = head.logits(x, 4, 8)
    assert torch.allclose(block, full[:, 4:8], atol = 1e-6)
    # A head that is not a plain Linear keeps the model's own logits.
    model.__dict__["_unsloth_diffusion_fused_head"] = (model.head, None)
    fallback = BlockHead(model)
    assert fallback.head is None and torch.equal(fallback.logits(x, 4, 8), full[:, 4:8])


def test_progress_throttles_forces_block_ends_and_stops_after_close():
    from unsloth.models.diffusion_infer import make_progress

    frames = []
    progress = make_progress(frames.append, None, interval = 3600, mask_id = MASK_ID, budget = 4)
    ids = torch.tensor([[5, MASK_ID, 7, MASK_ID, 9, 9]])
    progress.frame(0, 0, 4, ids)  # first frame always goes out
    progress.frame(0, 1, 4, ids)  # throttled
    progress.frame(0, 4, 4, ids, force = True)
    assert [(f["block"], f["step"]) for f in frames] == [(0, 0), (0, 4)]
    assert frames[0]["tokens"] == [5, MASK_ID, 7, MASK_ID] and frames[0]["mask_id"] == MASK_ID
    progress.close()
    progress.frame(1, 0, 4, ids, force = True)
    assert len(frames) == 2


def test_progress_check_raises_once_stop_is_requested():
    from unsloth.models.diffusion_infer import DiffusionGenerationCancelled, make_progress
    from unsloth.models.diffusion_samplers import DiffusionGenerationCancelled as exported

    assert exported is DiffusionGenerationCancelled
    calls = []
    progress = make_progress(None, lambda: len(calls) >= 3 or calls.append(1), 0.0, MASK_ID, 4)
    model = _Bidirectional()
    with pytest.raises(DiffusionGenerationCancelled):
        _sample(model, progress)
    assert model.calls == 3  # checked in the step loop, before each forward


def test_llada_frames_show_the_canvas_filling_in():
    from unsloth.models.diffusion_infer import make_progress

    model = _Bidirectional()
    frames = []
    progress = make_progress(frames.append, None, 0.0, MASK_ID, 16)
    out = _sample(model, progress)
    progress.close()
    assert all(len(f["tokens"]) == 16 for f in frames)
    masks = [sum(t == MASK_ID for t in f["tokens"]) for f in frames]
    assert masks[0] == 15 and masks[-1] == 0 and masks == sorted(masks, reverse = True)
    assert frames[-1]["tokens"] == out[0].tolist()
    ends = [f for f in frames if f["step"] == f["total"]]
    assert [f["block"] for f in ends] == [
        0,
        1,
    ]  # exactly one block-end frame per block, step == total
    assert all(0 <= f["step"] < f["total"] for f in frames if f not in ends)
    # Frames never change what is generated.
    assert torch.equal(out, _sample(_Bidirectional()))


def test_llada_projects_only_the_active_block():
    model = _Bidirectional()
    rows = []
    # Vocabulary projections only: inside the backbone forward the head returns its input (width 16).
    model.head.register_forward_hook(
        lambda m, a, out: rows.append(out.shape[1]) if out.shape[-1] == VOCAB else None
    )
    model.__dict__["_unsloth_diffusion_fused_head"] = (
        model.head,
        model.head,
    )  # the hook makes it look wrapped
    _sample(model)
    assert rows and set(rows) == {8}  # block_length rows, never the 19-token canvas


def test_inference_compile_restores_the_previous_call_path(monkeypatch):
    from unsloth.models import diffusion_infer

    compiled = []
    monkeypatch.setattr(
        diffusion_infer.torch, "compile", lambda fn, **kw: compiled.append(kw) or fn
    )
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    marker = object()
    model.layers[0]._compiled_call_impl = marker
    with diffusion_infer.inference_compile(model, "reduce-overhead") as count:
        assert count == 2
        assert all(layer._compiled_call_impl is not None for layer in model.layers)
    assert model.layers[0]._compiled_call_impl is marker
    assert model.layers[1]._compiled_call_impl is None
    with diffusion_infer.inference_compile(model, "reduce-overhead"):
        pass
    assert len(compiled) == 2 and compiled[0] == {"mode": "reduce-overhead", "dynamic": True}
    with diffusion_infer.inference_compile(model, None) as count:
        assert count == 0


@pytest.mark.parametrize(
    "value, device, expected",
    [
        ("auto", "cuda", "graphs"),
        ("auto", "cpu", None),
        (True, "cuda", "graphs"),
        ("default", "cuda", "default"),
        (False, "cuda", None),
        ("reduce-overhead", "cuda", "reduce-overhead"),
    ],
)
def test_resolve_compile(value, device, expected):
    from unsloth.models.diffusion_infer import resolve_compile
    assert resolve_compile(value, torch.device(device)) == expected


def test_graphed_forward_remembers_a_failed_capture_on_the_model():
    from unsloth.models.diffusion_infer import GraphedForward

    model = torch.nn.Module()
    model.__dict__["_unsloth_graph_capture_failed"] = "RuntimeError: host sync"
    calls = []
    graphed = GraphedForward(lambda x: calls.append(x) or x + 1, owner = model)
    assert graphed.failed is not None
    assert torch.equal(graphed(torch.zeros(2)), torch.ones(2)) and len(calls) == 1
    assert graphed.graphs == {}


def test_block_head_graphs_are_freed_without_a_gc_pass():
    import gc
    import weakref

    from unsloth.models.diffusion_infer import BlockHead

    gc.disable()
    try:
        head = BlockHead(_Bidirectional(), graphs = True)
        graphed = weakref.ref(head.graphed)
        del head
        assert (
            graphed() is None
        )  # the CUDA Graphs and their pool go with the call, not at the next gc
    finally:
        gc.enable()


def test_graphed_forward_cancel_during_capture_does_not_disable_graphs(monkeypatch):
    from unsloth.models.diffusion_infer import DiffusionGenerationCancelled, GraphedForward

    model = torch.nn.Module()
    graphed = GraphedForward(lambda x: x, owner = model)

    def cancelled(x):
        raise DiffusionGenerationCancelled("stop")

    monkeypatch.setattr(graphed, "_capture", cancelled)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    with pytest.raises(DiffusionGenerationCancelled):
        graphed(torch.zeros(2))
    assert graphed.failed is None
    assert "_unsloth_graph_capture_failed" not in model.__dict__


def test_graphed_forward_out_of_memory_is_not_remembered_on_the_model(monkeypatch):
    from unsloth.models.diffusion_infer import GraphedForward

    model = torch.nn.Module()
    graphed = GraphedForward(lambda x: x + 1, owner = model)

    def out_of_memory(x):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(graphed, "_capture", out_of_memory)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    assert torch.equal(graphed(torch.zeros(2)), torch.ones(2))
    assert "_unsloth_graph_capture_failed" not in model.__dict__
    assert GraphedForward(lambda x: x, owner = model).failed is None


class _ToyNemotron(torch.nn.Module):
    """Nemotron-Labs-Diffusion's generate call pattern: prefill, per-step block denoise, block commit."""

    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"block_size": 4})()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        ids,
        past_key_values = None,
        use_cache = False,
        use_causal_mask = False,
    ):
        return type("Out", (), {"past_key_values": object()})()

    def generate(
        self,
        prompt_ids,
        max_new_tokens,
        block_length,
        threshold = None,
        temperature = 0.0,
        eos_token_id = None,
    ):
        x = prompt_ids.clone()
        out = self(prompt_ids, use_cache = True, use_causal_mask = True)
        for block in range(max_new_tokens // block_length):
            x = torch.cat([x, torch.full((1, block_length), 99)], dim = 1)
            start = x.shape[1] - block_length
            for step in range(block_length):
                self(x[:, start:], past_key_values = out.past_key_values, use_cache = False)
                x[0, start + step] = 10 * block + step
            out = self(
                x[:, start:],
                past_key_values = out.past_key_values,
                use_cache = True,
                use_causal_mask = True,
            )
        return x, 0


def test_nemotron_streams_frames_from_its_own_generate():
    from unsloth.models import diffusion_samplers
    from unsloth.models.diffusion_infer import make_progress

    frames = []
    progress = make_progress(frames.append, None, 0.0, 99, 6)
    model = _ToyNemotron()
    out = diffusion_samplers._nemotron_sample(
        model,
        None,
        torch.tensor([[1, 2, 3]]),
        None,
        max_new_tokens = 6,
        steps = None,
        block_length = 4,
        temperature = 0.0,
        mask_id = 99,
        eos_ids = [],
        progress = progress,
    )
    progress.close()
    ends = [f for f in frames if f["step"] == f["total"]]
    assert [f["block"] for f in ends] == [0, 1]
    assert ends[-1]["tokens"] == out[0, :6].tolist() == [0, 1, 2, 3, 10, 11]
    assert all(len(f["tokens"]) == 6 for f in frames)
    assert [f["step"] for f in frames if f["block"] == 0] == [0, 1, 2, 3, 4]
    assert frames[1]["tokens"] == [0, 99, 99, 99, 99, 99]
    assert not model._forward_pre_hooks


def test_nemotron_without_on_frame_adds_no_hook():
    from unsloth.models import diffusion_samplers
    from unsloth.models.diffusion_infer import make_progress

    model = _ToyNemotron()
    seen = []
    model.register_forward_pre_hook(lambda m, a: seen.append(len(m._forward_pre_hooks)))
    diffusion_samplers._nemotron_sample(
        model,
        None,
        torch.tensor([[1]]),
        None,
        max_new_tokens = 4,
        steps = None,
        block_length = 4,
        temperature = 0.0,
        mask_id = 99,
        eos_ids = [],
        progress = make_progress(None, None, 0.0, 99, 4),
    )
    assert set(seen) == {1}


def test_graphs_env_opt_out_also_covers_auto(monkeypatch):
    from unsloth.models.diffusion_infer import resolve_compile

    monkeypatch.setenv("UNSLOTH_DIFFUSION_CUDA_GRAPHS", "0")
    assert resolve_compile("auto", torch.device("cuda")) is None
    assert resolve_compile(True, torch.device("cuda")) is None
    assert resolve_compile("graphs", torch.device("cuda")) is None
    assert resolve_compile("reduce-overhead", torch.device("cuda")) == "reduce-overhead"


def test_host_synced_moe_only_for_unstacked_expert_lists():
    from unsloth.models.diffusion_infer import host_synced_moe

    dense = torch.nn.Sequential(torch.nn.Linear(2, 2))
    assert not host_synced_moe(dense)
    moe = torch.nn.Module()
    moe.experts = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    model = torch.nn.Sequential(moe)
    assert host_synced_moe(model)
    moe.__dict__["_unsloth_grouped_experts"] = (torch.zeros(1),)
    assert not host_synced_moe(model)
