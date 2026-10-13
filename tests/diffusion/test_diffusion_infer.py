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
        ("auto", "cuda", None),
        ("auto", "cpu", None),
        (True, "cuda", "graphs"),
        ("graphs", "cuda", "graphs"),
        ("graphs", "cpu", None),
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
