# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Graph decode KV growth and non-current-GPU capture, each bitwise equal to eager."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
from unsloth import FastLanguageModel
import torch
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py
from transformers import StoppingCriteria, StoppingCriteriaList

import unsloth.models._decode_graphs as G

cuda = pytest.mark.skipif(not has_real_cuda(), reason = "CUDA only")
MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
PROMPTS = ["Once upon a time there was", "The capital of France is Paris and"] * 4


class _StopAfter(StoppingCriteria):
    def __init__(self, n):
        self.n, self.k = n, 0

    def __call__(self, input_ids, scores, **kwargs):
        self.k += 1
        return torch.full(
            (input_ids.shape[0],), self.k >= self.n, dtype = torch.bool, device = input_ids.device
        )


def _load(device = 0):
    model, tok = FastLanguageModel.from_pretrained(
        MODEL, max_seq_length = 8192, load_in_4bit = False, device_map = {"": device}
    )
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _run(
    model,
    tok,
    enc,
    graphs,
    monkeypatch,
    max_new,
    stop = None,
    min_new = 0,
):
    monkeypatch.setattr(G, "ENABLED", None)
    monkeypatch.delenv("UNSLOTH_DECODE_CUDA_GRAPHS", raising = False)
    criteria = StoppingCriteriaList([_StopAfter(stop)]) if stop else None

    def gen():
        return model.generate(
            **enc, max_new_tokens = max_new, min_new_tokens = min_new, do_sample = False,
            pad_token_id = tok.pad_token_id, output_logits = True, return_dict_in_generate = True,
            stopping_criteria = criteria,
        )  # fmt: skip

    seen = [0]
    for name in ("alloc_decode_buffers", "resize_decode_buffers"):
        fn = getattr(G, name)

        def spy(
            attn,
            *args,
            _fn = fn,
            **kwargs,
        ):
            out = _fn(attn, *args, **kwargs)
            seen[0] = max(seen[0], attn.paged_attention.shape[0])
            return out

        monkeypatch.setattr(G, name, spy)
    before = dict(G.stats(model))
    if graphs:
        with G.rl_generation():
            out = gen()
    else:
        monkeypatch.setattr(G, "ENABLED", False)
        out = gen()
    rows = seen[0] or None
    after = G.stats(model)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in ("captures", "replays")}
    G.release(model)
    return out.sequences.cpu(), torch.stack(out.logits).cpu(), rows, delta


@cuda
def test_early_stop_does_not_hold_max_new_tokens(monkeypatch):
    model, tok = _load()
    enc = tok(PROMPTS, return_tensors = "pt", padding = True).to("cuda")
    ids_e, logits_e, _, _ = _run(model, tok, enc, False, monkeypatch, 32768, stop = 20)
    ids_g, logits_g, rows, delta = _run(model, tok, enc, True, monkeypatch, 32768, stop = 20)
    assert torch.equal(ids_e, ids_g) and torch.equal(logits_e, logits_g)
    assert delta["replays"] > 0, delta
    # prompt + one 512 increment, rounded to the 256-row bucket; not 32768 + prompt
    assert rows is not None and rows <= 1024, rows


@cuda
def test_long_rollout_grows_and_stays_bitwise(monkeypatch):
    model, tok = _load()
    enc = tok(PROMPTS, return_tensors = "pt", padding = True).to("cuda")
    new = 1500
    ids_e, logits_e, _, _ = _run(model, tok, enc, False, monkeypatch, new, min_new = new)
    ids_g, logits_g, _, delta = _run(model, tok, enc, True, monkeypatch, new, min_new = new)
    assert torch.equal(ids_e, ids_g) and torch.equal(logits_e, logits_g)
    # 768 -> 1536 -> hint: a few re-captures, most steps replayed
    assert 2 <= delta["captures"] <= 4, delta
    assert delta["replays"] >= new - 10, delta


@cuda
def test_prealloc_kill_switch(monkeypatch):
    model, tok = _load()
    enc = tok(PROMPTS, return_tensors = "pt", padding = True).to("cuda")
    monkeypatch.setattr(G, "PREALLOC", True)
    _, _, rows, _ = _run(model, tok, enc, True, monkeypatch, 4096, stop = 5)
    assert rows >= 4096, rows


@pytest.mark.skipif(
    not has_real_cuda() or torch.cuda.device_count() < 2, reason = "needs 2 CUDA GPUs"
)
def test_model_on_a_non_current_gpu_replays(monkeypatch):
    torch.cuda.set_device(0)
    model, tok = _load(device = 1)
    enc = tok(PROMPTS, return_tensors = "pt", padding = True).to("cuda:1")
    assert torch.cuda.current_device() == 0
    ids_e, logits_e, _, _ = _run(model, tok, enc, False, monkeypatch, 48, min_new = 48)
    ids_g, logits_g, _, delta = _run(model, tok, enc, True, monkeypatch, 48, min_new = 48)
    assert torch.cuda.current_device() == 0
    assert G._state(model).disabled is None, G._state(model).disabled
    assert delta["captures"] >= 1 and delta["replays"] > 0, delta
    assert torch.equal(ids_e, ids_g) and torch.equal(logits_e, logits_g)
