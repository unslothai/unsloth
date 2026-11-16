# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Mistral decode takes the CUDA-graph step by default, as Llama does: MistralForCausalLM's own forward
routes a one-token decode step through _decode_graphs.decode_step, and a replayed step is bit for bit
the eager one (16-bit and 4-bit, batch 1 and a left-padded batch, past the sliding window, after a LoRA
scale change)."""

import types

import pytest
import torch

import unsloth.models._decode_graphs as G
import unsloth.models.mistral as M

ENV = "UNSLOTH_DECODE_CUDA_GRAPHS"


@pytest.fixture
def runtime(monkeypatch):
    # The unset default (plain generate() runs inside generate_scope()), not a programmatic override.
    monkeypatch.setattr(G, "ENABLED", None)
    monkeypatch.delenv(ENV, raising = False)
    return monkeypatch


def _tiny_mistral():
    from transformers import MistralConfig, MistralForCausalLM
    config = MistralConfig(
        vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, sliding_window = 16,
    )  # fmt: skip
    return MistralForCausalLM(config).eval()


def _spy(runtime):
    calls = []

    def decode_step(model, fast_forward_inference, logits_fn, mask_fn, input_ids, past, *rest):
        calls.append(fast_forward_inference)
        logits = torch.full((input_ids.shape[0], 1, model.config.vocab_size), 7.0)
        return logits, past

    runtime.setattr(G, "decode_step", decode_step)
    return calls


def _decode(model, **kwargs):
    past = tuple((torch.zeros(1, 2, 5, 8), torch.zeros(1, 2, 5, 8)) for _ in range(2))
    return M.MistralForCausalLM_fast_forward(
        model, input_ids = torch.tensor([[3]]), past_key_values = past,
        position_ids = torch.tensor([[5]]), return_dict = True, **kwargs,
    )  # fmt: skip


def test_mistral_is_a_graphed_family():
    model = types.SimpleNamespace(config = types.SimpleNamespace(model_type = "mistral"))
    assert G.graphable(model)
    assert G._eligible(model, types.SimpleNamespace()) is None


def _no_eager(runtime):
    eager = []

    def fail(*args, **kwargs):
        eager.append(1)
        raise RuntimeError("eager")

    # Same graph marker as the real step, so only which branch runs differs.
    fail._unsloth_graph_attention = M.LlamaModel_fast_forward_inference._unsloth_graph_attention
    runtime.setattr(M, "LlamaModel_fast_forward_inference", fail)
    return eager


def test_mistral_decode_step_takes_the_graph_path_by_default(runtime):
    runtime.setattr(G, "_DEFAULT_OK", True)
    runtime.setattr(
        G, "_STUDIO_DEFAULT_OK", True
    )  # native Windows keeps the scope default off; test the scope itself
    calls = _spy(runtime)
    _no_eager(runtime)
    model = _tiny_mistral()
    with G.generate_scope():
        out = _decode(model)
    assert len(calls) == 1
    # The graphed step is the Llama decode step, whose attention reads the graph statics.
    assert getattr(calls[0], "_unsloth_graph_attention", None) in G._SUPPORTED_ATTENTION
    assert torch.equal(out.logits, torch.full((1, 1, 64), 7.0))


def test_mistral_kill_switch_and_loss_steps_stay_eager(runtime):
    runtime.setattr(G, "_DEFAULT_OK", True)
    runtime.setattr(
        G, "_STUDIO_DEFAULT_OK", True
    )  # native Windows keeps the scope default off; test the scope itself
    calls = _spy(runtime)
    eager = _no_eager(runtime)
    model = _tiny_mistral()
    runtime.setenv(ENV, "0")
    with G.generate_scope(), pytest.raises(RuntimeError, match = "eager"):
        _decode(model)
    runtime.delenv(ENV)
    with G.generate_scope(), pytest.raises(RuntimeError, match = "eager"):
        _decode(model, labels = torch.tensor([[1]]))
    # A tensor logits_to_keep row selector is applied by the eager path only.
    with G.generate_scope(), pytest.raises(RuntimeError, match = "eager"):
        _decode(model, logits_to_keep = torch.tensor([0]))
    # Outside generate() (a hand-written loop) the unset default is eager too.
    with pytest.raises(RuntimeError, match = "eager"):
        _decode(model)
    assert calls == [] and len(eager) == 4


# GPU: bitwise against eager through the real generate().
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")
PROMPTS = ["Once upon a time there was", "The capital of France is Paris and"]
NEW = 40


def _load(
    name,
    load_in_4bit,
    lora = False,
    sliding_window = None,
):
    from unsloth import FastLanguageModel

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
    if sliding_window is not None:
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "sliding_window"):
                config.sliding_window = sliding_window
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _stats(model):
    for m in model.modules():
        if "_unsloth_decode_graphs" in m.__dict__:
            return G.stats(m)
    return {}


def _gen(model, tok, bsz):
    enc = tok(PROMPTS[:bsz], return_tensors = "pt", padding = True).to("cuda")
    out = model.generate(
        **enc, max_new_tokens = NEW, min_new_tokens = NEW, do_sample = False,
        pad_token_id = tok.pad_token_id, output_logits = True, return_dict_in_generate = True,
    )  # fmt: skip
    return out.sequences.cpu(), torch.stack(out.logits).cpu()


def _eager_then_graph(runtime, model, tok, bsz):
    runtime.setenv(ENV, "0")
    eager = _gen(model, tok, bsz)
    before = _stats(model).get("replays", 0)
    runtime.delenv(ENV)
    graph = _gen(model, tok, bsz)
    stats = _stats(model)
    assert stats.get("disabled") is None, stats
    # Every step but the first (eager: it fills the cache) replayed.
    assert stats["replays"] - before == NEW - 2, stats
    return eager, graph


@cuda
@pytest.mark.skipif(not G._DEFAULT_OK, reason = "graphs default on for NVIDIA CUDA only")
@pytest.mark.parametrize(
    "name,sliding_window",
    [
        ("trl-internal-testing/tiny-MistralForCausalLM-0.2", None),
        # n_heads * head_dim != hidden_size (Mistral Nemo layout); window 16 << prompt + 40 new tokens.
        ("trl-internal-testing/tiny-MistralForCausalLM-0.1", 16),
    ],
)
@pytest.mark.parametrize("load_in_4bit", [False, True])
@pytest.mark.parametrize("bsz", [1, 2])
def test_mistral_graph_decode_is_bitwise_eager(runtime, name, sliding_window, load_in_4bit, bsz):
    model, tok = _load(name, load_in_4bit, sliding_window = sliding_window)
    eager, graph = _eager_then_graph(runtime, model, tok, bsz)
    assert torch.equal(eager[0], graph[0])
    assert torch.equal(eager[1], graph[1])


@cuda
@pytest.mark.skipif(not G._DEFAULT_OK, reason = "graphs default on for NVIDIA CUDA only")
@pytest.mark.parametrize("bsz", [1, 2])
def test_mistral_lora_scale_change_recaptures(runtime, bsz):
    model, tok = _load(
        "trl-internal-testing/tiny-MistralForCausalLM-0.1", True, lora = True, sliding_window = 16
    )
    eager, graph = _eager_then_graph(runtime, model, tok, bsz)
    assert torch.equal(eager[1], graph[1])
    captures = _stats(model)["captures"]
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for key in scaling:
                scaling[key] *= 3.0
    eager2, graph2 = _eager_then_graph(runtime, model, tok, bsz)
    assert not torch.equal(eager[1], eager2[1])  # the scale change reaches the logits
    assert torch.equal(eager2[0], graph2[0]) and torch.equal(eager2[1], graph2[1])
    assert _stats(model)["captures"] > captures
