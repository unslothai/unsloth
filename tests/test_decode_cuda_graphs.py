# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CUDA-graph decode (forced on / off per test through _decode_graphs.ENABLED): a replayed step is bit for bit the eager step
(greedy tokens and raw logits, 16-bit and 4-bit, batch 1 and a left-padded batch), graphs are
re-captured when LoRA weights change between generate() calls, and the KV cache may grow."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
from unsloth import FastLanguageModel
import torch

import unsloth.models._decode_graphs as G
from unsloth.kernels import utils as kernel_utils

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")
MODELS = (
    "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    "trl-internal-testing/tiny-Qwen3ForCausalLM",
)
PROMPTS = [
    "Once upon a time there was",
    "The capital of France is Paris and",
    "def f(x):",
    "1, 2, 3,",
]


def _load(
    name,
    load_in_4bit,
    lora = False,
):
    model, tok = FastLanguageModel.from_pretrained(
        name, max_seq_length = 2048, load_in_4bit = load_in_4bit
    )
    if lora:
        model = FastLanguageModel.get_peft_model(
            model, r = 8, lora_alpha = 16, random_state = 0,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )  # fmt: skip
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.copy_(torch.randn(p.shape, generator = g) * 0.05)
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _causal(model):
    base = getattr(model, "base_model", model)
    return base.model if hasattr(base, "model") and hasattr(base.model, "lm_head") else model


def _generate(
    model,
    tok,
    enc,
    enabled,
    monkeypatch,
    new = 40,
):
    monkeypatch.setattr(G, "ENABLED", enabled)
    out = model.generate(
        **enc, max_new_tokens = new, min_new_tokens = new, do_sample = False, use_cache = True,
        pad_token_id = tok.pad_token_id, output_logits = True, return_dict_in_generate = True,
    )  # fmt: skip
    return out.sequences.cpu(), torch.stack(out.logits).cpu()


def _plain(
    model,
    tok,
    enc,
    new = 20,
):
    return model.generate(
        **enc,
        max_new_tokens = new,
        min_new_tokens = new,
        do_sample = False,
        pad_token_id = tok.pad_token_id,
    )


def _manual_loop(
    model,
    ids,
    enabled,
    monkeypatch,
    steps = 12,
    start = None,
):
    monkeypatch.setattr(G, "ENABLED", enabled)
    with torch.inference_mode():
        out = model(ids, use_cache = True, return_dict = True)
        past, nxt, logits = out.past_key_values, out.logits[:, -1:].argmax(-1), []
        for i in range(steps):
            p = (ids.shape[1] if start is None else start) + i
            pos = torch.full((ids.shape[0], 1), p, device = "cuda", dtype = torch.long)
            out = model(
                nxt, past_key_values = past, position_ids = pos, use_cache = True, return_dict = True
            )
            past = out.past_key_values
            logits.append(out.logits[:, -1].float().cpu())
            nxt = out.logits[:, -1:].argmax(-1)
    return torch.stack(logits)


@cuda
@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("load_in_4bit", [False, True])
@pytest.mark.parametrize("bsz", [1, 4])
def test_graph_decode_is_bitwise_eager(name, load_in_4bit, bsz, monkeypatch):
    model, tok = _load(name, load_in_4bit)
    enc = tok(PROMPTS[:bsz], return_tensors = "pt", padding = True).to("cuda")
    eager = _generate(model, tok, enc, False, monkeypatch)
    before = G.stats(_causal(model)).get("replays", 0)
    graph = _generate(model, tok, enc, True, monkeypatch)
    stats = G.stats(_causal(model))
    assert stats["disabled"] is None, stats
    # Every step but the first (eager: it fills the cache) was replayed.
    assert stats["replays"] - before == 40 - 2
    assert torch.equal(eager[0], graph[0])
    assert torch.equal(eager[1], graph[1])
    # generate() calls that return no cache reuse the buffers and the captured step.
    _plain(model, tok, enc)
    captures = G.stats(_causal(model))["captures"]
    _plain(model, tok, enc)
    assert G.stats(_causal(model))["captures"] == captures
    again = _generate(model, tok, enc, True, monkeypatch)
    assert torch.equal(eager[1], again[1])


@cuda
def test_returned_cache_is_not_overwritten(monkeypatch):
    """A cache handed back by generate(return_dict_in_generate=True) keeps its values through a
    later generate(): the persistent buffers it views are released to the caller."""
    model, tok = _load(MODELS[0], False)
    monkeypatch.setattr(G, "ENABLED", True)
    enc = tok(PROMPTS[:1], return_tensors = "pt").to("cuda")
    out = model.generate(
        **enc,
        max_new_tokens = 20,
        do_sample = False,
        pad_token_id = tok.pad_token_id,
        return_dict_in_generate = True,
    )
    K, V = out.past_key_values[0]
    kept = K.clone(), V.clone()
    other = tok(PROMPTS[1:2], return_tensors = "pt").to("cuda")
    _plain(model, tok, other)
    assert torch.equal(K, kept[0]) and torch.equal(V, kept[1])


@cuda
def test_shift_labels_decode_loss(monkeypatch):
    model, tok = _load(MODELS[0], False)
    ids = tok(PROMPTS[:1], return_tensors = "pt").input_ids.to("cuda")

    def losses(enabled):
        monkeypatch.setattr(G, "ENABLED", enabled)
        with torch.inference_mode():
            out = model(ids, use_cache = True, return_dict = True)
            past, nxt, res = out.past_key_values, out.logits[:, -1:].argmax(-1), []
            for i in range(6):
                pos = torch.full((1, 1), ids.shape[1] + i, device = "cuda", dtype = torch.long)
                out = model(
                    nxt,
                    past_key_values = past,
                    position_ids = pos,
                    use_cache = True,
                    return_dict = True,
                    shift_labels = torch.full((1, 1), 7, device = "cuda", dtype = torch.long),
                )
                past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
                res.append(out.loss)
        return res

    eager, graph = losses(False), losses(True)
    assert all(l is not None for l in graph)
    assert all(torch.equal(a, b) for a, b in zip(eager, graph))


@cuda
@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("stale_control", [False, True])
def test_lora_change_between_generations_recaptures(name, stale_control, monkeypatch):
    """GRPO: an optimizer step, then an in-place adapter edit, then a disabled adapter between
    decode runs. stale_control pins the fingerprint, so the old graphs replay: proves the check is
    what keeps the graph path on the live adapter."""
    model, tok = _load(name, False, lora = True)
    enc = tok(PROMPTS[:1], return_tensors = "pt").to("cuda")
    ids = enc.input_ids
    _plain(model, tok, enc)
    _manual_loop(model, ids, True, monkeypatch)
    if stale_control:
        pinned = G._state(_causal(model)).fingerprint
        monkeypatch.setattr(G, "_fingerprint", lambda *a: pinned)
    params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    FastLanguageModel.for_training(model)
    opt = torch.optim.SGD(params, lr = 1.0)
    for p in params:
        p.grad = torch.randn_like(p) * 0.05
    opt.step()
    opt.zero_grad(set_to_none = True)
    FastLanguageModel.for_inference(model)
    _plain(model, tok, enc)
    eager = _manual_loop(model, ids, False, monkeypatch)
    graph = _manual_loop(model, ids, True, monkeypatch)
    if stale_control:
        assert not torch.equal(eager, graph)
        return
    assert torch.equal(eager, graph)
    with torch.no_grad():
        for p in params:
            p.mul_(-1.0)
    assert torch.equal(
        _manual_loop(model, ids, False, monkeypatch), _manual_loop(model, ids, True, monkeypatch)
    )
    with model.disable_adapter():
        assert torch.equal(
            _manual_loop(model, ids, False, monkeypatch),
            _manual_loop(model, ids, True, monkeypatch),
        )


@cuda
def test_manual_decode_loop_grows_cache(monkeypatch):
    """Direct forward calls (no generate() length hint): the cache grows past its first allocation
    (new addresses), and the graphs re-capture instead of replaying the old buffers."""
    model, tok = _load(MODELS[0], False)
    ids = tok(PROMPTS[:1], return_tensors = "pt").input_ids.to("cuda")

    def loop(enabled, steps = 800):
        monkeypatch.setattr(G, "ENABLED", enabled)
        with torch.inference_mode():
            out = model(ids, use_cache = True, return_dict = True)
            past, nxt, logits = out.past_key_values, out.logits[:, -1:].argmax(-1), []
            for i in range(steps):
                pos = torch.full((1, 1), ids.shape[1] + i, device = "cuda", dtype = torch.long)
                out = model(
                    nxt, past_key_values = past, position_ids = pos, use_cache = True, return_dict = True
                )
                past = out.past_key_values
                logits.append(out.logits[:, -1].float().cpu())
                nxt = out.logits[:, -1:].argmax(-1)
        return torch.stack(logits)

    eager = loop(False)
    stats0 = dict(G.stats(_causal(model)))
    graph = loop(True)
    stats = G.stats(_causal(model))
    assert stats["captures"] - stats0.get("captures", 0) >= 2, stats
    assert torch.equal(eager, graph)


@cuda
def test_manual_loop_adapter_toggle_and_merge(monkeypatch):
    """Decode loops outside generate(): each prefill re-checks the fingerprint, so a disabled or
    merged adapter is never replayed from graphs captured with it active."""
    model, tok = _load(MODELS[0], False, lora = True)
    ids = tok(PROMPTS[:1], return_tensors = "pt").input_ids.to("cuda")
    assert torch.equal(
        _manual_loop(model, ids, False, monkeypatch), _manual_loop(model, ids, True, monkeypatch)
    )
    with model.disable_adapter():
        assert torch.equal(
            _manual_loop(model, ids, False, monkeypatch),
            _manual_loop(model, ids, True, monkeypatch),
        )
    model.merge_adapter()
    try:
        assert torch.equal(
            _manual_loop(model, ids, False, monkeypatch),
            _manual_loop(model, ids, True, monkeypatch),
        )
    finally:
        model.unmerge_adapter()
    assert torch.equal(
        _manual_loop(model, ids, False, monkeypatch), _manual_loop(model, ids, True, monkeypatch)
    )


@cuda
def test_explicit_positions_past_rope_table(monkeypatch):
    """Positions past the RoPE rows the graphs were sized for run eagerly (the eager step grows the
    table), and the graphs re-capture against the new table."""
    model, tok = _load(MODELS[0], False)
    ids = tok(PROMPTS[:1], return_tensors = "pt").input_ids.to("cuda")
    _manual_loop(model, ids, True, monkeypatch)
    far = 40000
    eager = _manual_loop(model, ids, False, monkeypatch, start = far)
    graph = _manual_loop(model, ids, True, monkeypatch, start = far)
    assert torch.equal(eager, graph)
    assert torch.equal(
        _manual_loop(model, ids, False, monkeypatch), _manual_loop(model, ids, True, monkeypatch)
    )


@cuda
def test_release_drops_graphs_and_buffers(monkeypatch):
    model, tok = _load(MODELS[0], False, lora = True)
    enc = tok(PROMPTS[:1], return_tensors = "pt").to("cuda")
    _generate(model, tok, enc, True, monkeypatch)
    G.release(model)
    assert G.stats(_causal(model))["graphs"] == 0
    assert not any("_unsloth_graph_buffers" in m.__dict__ for m in model.modules())
    torch.cuda.empty_cache()
    assert torch.equal(
        _generate(model, tok, enc, False, monkeypatch)[1],
        _generate(model, tok, enc, True, monkeypatch)[1],
    )


@cuda
def test_steps_counter_present():
    # The adapter-change check keys on this counter (bumped by every optimizer step).
    assert isinstance(kernel_utils._OPTIMIZER_STEPS, int)


def test_graph_statics_are_per_thread():
    """A capture's statics are invisible to another thread's (or model's) eager decode."""
    import threading

    seen = []
    G._TLS.statics = object()
    try:
        t = threading.Thread(target = lambda: seen.append(G.static()))
        t.start()
        t.join()
    finally:
        G._TLS.statics = None
    assert seen == [None] and G.static() is None
