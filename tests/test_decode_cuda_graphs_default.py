# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""When CUDA-graph decode runs (decided per call): UNSLOTH_DECODE_CUDA_GRAPHS unset = off for plain
model.generate(), on inside an RL trainer's generation (rl_generation(), entered by rl.py's
unwrap_model_for_generation); =1 on everywhere, =0 off everywhere. Returning to training releases
the graphs and every decode buffer."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
from unsloth import FastLanguageModel
import torch

import unsloth.models._decode_graphs as G

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")
# The unset default only turns graphs on for NVIDIA CUDA; ROCm keeps them off unless the flag is 1.
rl_default = pytest.mark.skipif(not G._DEFAULT_OK, reason = "RL default is NVIDIA CUDA only")
MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
PROMPTS = ["Once upon a time there was", "def f(x):"]
ENV = "UNSLOTH_DECODE_CUDA_GRAPHS"


@pytest.fixture
def runtime(monkeypatch):
    # The runtime decision, not a programmatic override.
    monkeypatch.setattr(G, "ENABLED", None)
    monkeypatch.delenv(ENV, raising = False)
    return monkeypatch


def test_enabled_decision(runtime):
    runtime.setattr(G, "_DEFAULT_OK", True)
    assert G.enabled() is False
    with G.rl_generation():
        assert G.enabled() is True
        with G.rl_generation():
            assert G.enabled() is True
        assert G.enabled() is True
    assert G.enabled() is False
    runtime.setenv(ENV, "0")
    with G.rl_generation():
        assert G.enabled() is False
    runtime.setenv(ENV, "1")
    assert G.enabled() is True
    runtime.setattr(G, "ENABLED", False)
    assert G.enabled() is False


def test_rl_default_is_nvidia_cuda_only(runtime):
    runtime.setattr(G, "_DEFAULT_OK", False)  # ROCm / HIP, or no CUDA device
    with G.rl_generation():
        assert G.enabled() is False
        runtime.setenv(ENV, "1")
        assert G.enabled() is True
    runtime.setattr(torch.version, "hip", "6.4.0", raising = False)
    assert G._default_ok() is False


def test_rl_scope_is_per_thread(runtime):
    import threading

    runtime.setattr(G, "_DEFAULT_OK", True)
    seen = []
    with G.rl_generation():
        t = threading.Thread(target = lambda: seen.append(G.enabled()))
        t.start()
        t.join()
        assert G.enabled() is True
    assert seen == [False]


def test_enabled_is_trace_safe(runtime):
    def f(x):
        return x + 1 if G.enabled() or G.static() is not None else x - 1

    compiled = torch.compile(f, fullgraph = True, backend = "eager")
    with G.rl_generation():
        assert compiled(torch.zeros(2)).tolist() == [-1.0, -1.0]


def _load():
    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 2048, load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(
        model, r = 8, lora_alpha = 16, random_state = 0,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )  # fmt: skip
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


class _Stats:
    def __init__(self, model):
        self.model = model

    def __call__(self):
        for m in self.model.modules():
            if "_unsloth_decode_graphs" in m.__dict__:
                return G.stats(m)
        return {}


def _causal(model):
    return model


def _graph_buffers(model):
    return sum("_unsloth_graph_buffers" in m.__dict__ for m in model.modules())


def _decode_buffers(model):
    return sum("paged_attention" in m.__dict__ for m in model.modules())


def _gen(
    model,
    tok,
    new = 24,
):
    enc = tok(PROMPTS, return_tensors = "pt", padding = True).to("cuda")
    out = model.generate(
        **enc, max_new_tokens = new, min_new_tokens = new, do_sample = False,
        pad_token_id = tok.pad_token_id, output_logits = True, return_dict_in_generate = True,
    )  # fmt: skip
    return out.sequences.cpu(), torch.stack(out.logits).cpu()


@rl_default
def test_plain_generate_off_rl_generate_on_and_released(runtime):
    model, tok = _load()
    FastLanguageModel.for_inference(model)
    eager = _gen(model, tok)
    assert _Stats(model)().get("replays", 0) == 0
    assert _graph_buffers(model) == 0

    with G.rl_generation():
        graph = _gen(model, tok)
    stats = _Stats(model)()
    assert stats["disabled"] is None and stats["replays"] == 24 - 2, stats
    assert torch.equal(eager[0], graph[0]) and torch.equal(eager[1], graph[1])

    # =0 wins inside the RL scope; =1 turns plain generate on.
    runtime.setenv(ENV, "0")
    replays = _Stats(model)()["replays"]
    with G.rl_generation():
        assert torch.equal(_gen(model, tok)[1], eager[1])
    assert _Stats(model)()["replays"] == replays
    runtime.setenv(ENV, "1")
    assert torch.equal(_gen(model, tok)[1], eager[1])
    assert _Stats(model)()["replays"] > replays

    # generate() called on a model in training mode returns to training: everything released.
    runtime.delenv(ENV)
    FastLanguageModel.for_training(model)
    with G.rl_generation():
        _gen(model, tok)
    assert model.training
    assert _graph_buffers(model) == 0 and _decode_buffers(model) == 0
    assert _Stats(model)()["graphs"] == 0


@rl_default
def test_grpo_generation_uses_graphs_and_releases(runtime, tmp_path):
    """A real GRPOTrainer step: its generate() runs graphed with the variable unset, completions
    match the eager rollout, and the decode buffers are gone before the loss is computed."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    model, tok = _load()
    rows = [{"prompt": p} for p in PROMPTS * 4]
    seen = {"replays": [], "buffers_at_loss": []}

    def reward(completions, **kw):
        return [float(len(c)) for c in completions]

    cfg = GRPOConfig(
        output_dir = str(tmp_path), per_device_train_batch_size = 4, num_generations = 4,
        max_completion_length = 16, max_steps = 2, logging_steps = 1, save_strategy = "no",
        report_to = "none", seed = 0, bf16 = torch.cuda.is_bf16_supported(), learning_rate = 1e-4,
    )  # fmt: skip
    trainer = GRPOTrainer(model = model, reward_funcs = [reward], args = cfg,
                          train_dataset = Dataset.from_list(rows), processing_class = tok)  # fmt: skip
    # TRL >= 0.24 generates in _generate_single_turn; older TRL only has _generate_and_score_completions.
    hook = (
        "_generate_single_turn"
        if hasattr(trainer, "_generate_single_turn")
        else "_generate_and_score_completions"
    )
    gen = getattr(trainer, hook)

    def recorded(*a, **k):
        before = _Stats(model)().get("replays", 0)
        out = gen(*a, **k)
        seen["replays"].append(_Stats(model)().get("replays", 0) - before)
        seen["buffers_at_loss"].append((_graph_buffers(model), _decode_buffers(model)))
        return out

    setattr(trainer, hook, recorded)
    trainer.train()
    assert len(seen["replays"]) == 2 and all(r > 0 for r in seen["replays"]), seen
    assert seen["buffers_at_loss"] == [(0, 0), (0, 0)], seen
    assert _Stats(model)()["disabled"] is None
