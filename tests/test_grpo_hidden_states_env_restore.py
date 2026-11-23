# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A GRPO logprob pass must not leave UNSLOTH_RETURN_HIDDEN_STATES stuck at "1"."""

from __future__ import annotations

import ast
import os
import sys
import textwrap
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl  # noqa: E402

ENV = "UNSLOTH_RETURN_HIDDEN_STATES"

VOCAB, HIDDEN, SEQ, KEEP, PAD = 17, 8, 9, 3, 0


# Eager mirrors of the zoo scorers, same shapes: hidden states are projected through lm_head,
# real logits are not. Deterministic on CPU and immune to the installed zoo's compile choices.
def _eager_hidden_logprobs(
    hidden_states,
    lm_head,
    index,
    chunks = 4,
    logit_scale_multiply = 0.0,
    logit_scale_divide = 0.0,
    logit_softcapping = 0.0,
    temperature = 1.0,
):
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)
    parts = []
    for h_chunk, i_chunk in zip(
        torch.chunk(flat_hidden, chunks = chunks, dim = 0),
        torch.chunk(flat_index, chunks = chunks, dim = 0),
    ):
        logits = (h_chunk.to(lm_head.dtype) @ lm_head.t()).to(torch.float32)
        if temperature != 1.0:
            logits = logits / temperature
        picked = torch.gather(logits, dim = -1, index = i_chunk.unsqueeze(-1)).squeeze(-1)
        parts.append(picked - torch.logsumexp(logits, dim = -1))
    return torch.concat(parts).reshape(hidden_states.shape[0], hidden_states.shape[1])


def _eager_raw_logprobs(
    logits,
    index,
    temperature = 1.0,
    chunks = 4,
):
    flat_logits = logits.reshape(-1, logits.shape[-1]).to(torch.float32)
    flat_index = index.reshape(-1)
    parts = []
    for l_chunk, i_chunk in zip(
        torch.chunk(flat_logits, chunks = chunks, dim = 0),
        torch.chunk(flat_index, chunks = chunks, dim = 0),
    ):
        if temperature != 1.0:
            l_chunk = l_chunk / temperature
        picked = torch.gather(l_chunk, dim = -1, index = i_chunk.unsqueeze(-1)).squeeze(-1)
        parts.append(picked - torch.logsumexp(l_chunk, dim = -1))
    return torch.concat(parts).reshape(logits.shape[0], logits.shape[1])


def _left_pack(input_ids, pad_token_id):
    """Pads left like the zoo's left_pack_padding; the test batch has no pads, so this is identity."""
    out = torch.full_like(input_ids, pad_token_id)
    for row, ids in enumerate(input_ids):
        kept = ids[ids != pad_token_id]
        if kept.numel():
            out[row, -kept.numel() :] = kept
    return out


def _pad_count(input_ids, logits_to_keep, pad_token_id):
    if logits_to_keep >= input_ids.shape[1]:
        raise ValueError("logits_to_keep must be smaller than the sequence length.")
    return (input_ids[:, :-logits_to_keep] == pad_token_id).sum(dim = 1)


class _ScoringModel(torch.nn.Module):
    """Honours UNSLOTH_RETURN_HIDDEN_STATES like Unsloth's generated forwards."""

    def __init__(self, error = None):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.error = error
        self.seen_env = None

    def get_output_embeddings(self):
        return self.head

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        **kwargs,
    ):
        self.seen_env = os.environ.get(ENV)
        if self.error is not None:
            raise self.error
        hidden = self.embed(input_ids)
        if os.environ.get(ENV, "0") == "1":
            return SimpleNamespace(logits = hidden)
        return SimpleNamespace(logits = self.head(hidden))


def _never_called(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} should not run in this test")

    return fail


@pytest.fixture
def _eager_zoo(monkeypatch):
    """The two scorers the generated function imports live, as CPU-safe eager mirrors."""
    zoo = types.ModuleType("unsloth_zoo.rl_replacements")
    zoo.chunked_hidden_states_selective_log_softmax = _eager_hidden_logprobs
    zoo.chunked_selective_log_softmax = _eager_raw_logprobs
    monkeypatch.setitem(sys.modules, "unsloth_zoo.rl_replacements", zoo)


def _scoring_function():
    """The generated no-grad logprob function, exec'd the way the patched trainer sees it."""
    source = rl.grpo_trainer__get_per_token_logps_and_entropies(
        "_get_per_token_logps_and_entropies", "def _get_per_token_logps_and_entropies(): pass"
    )
    namespace = {}
    exec(compile(textwrap.dedent(source), "<grpo logps>", "exec"), namespace)
    return namespace["_get_per_token_logps_and_entropies"]


def _trainer(model):
    return SimpleNamespace(
        _autocast_enabled = False,
        _autocast_dtype = torch.float32,
        temperature = 1.0,
        accelerator = SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m),
        args = SimpleNamespace(
            unsloth_grpo_mini_batch = 1,
            unsloth_logit_chunk_multiplier = 1,
        ),
        processing_class = SimpleNamespace(pad_token_id = PAD),
        use_vllm = False,
        model = model,
    )


def _globals(model, trainer):
    return {
        "os": os,
        "torch": torch,
        "DEVICE_TYPE_TORCH": "cpu",
        "device_synchronize": lambda *args, **kwargs: None,
        "detect_logit_transforms": None,
        "UNSLOTH_ENABLE_LOGGING": False,
        "UNSLOTH_ZOO_HAS_HEAD_BIAS": False,
        "UNSLOTH_ZOO_HAS_MASKED_COL_GUARD": True,
        "UNSLOTH_GRPO_SEQ_PACKING_ON": False,
        "UNSLOTH_GRPO_PREFIX_GROUPER_ON": False,
        "UNSLOTH_GRPO_PREFIX_GROUPER_VLLM_ON": False,
        "_unsloth_grpo_autocast": rl._unsloth_grpo_autocast,
        "_unsloth_grpo_refuse_boundary_blind_packing": lambda *a, **k: None,
        "_unsloth_grpo_refuse_rank_local_forwards": lambda *a, **k: False,
        "_unsloth_grpo_vision_inputs": rl._unsloth_grpo_vision_inputs,
        "_unsloth_grpo_head_hooked": rl._unsloth_grpo_head_hooked,
        "_unsloth_grpo_returns_hidden_states": rl._unsloth_grpo_returns_hidden_states,
        "_unsloth_grpo_hidden_states_signal": rl._unsloth_grpo_hidden_states_signal,
        "_unsloth_seq_packing_verify_verdict": rl._unsloth_seq_packing_verify_verdict,
        "_unsloth_seq_packing_moe_noise": rl._unsloth_seq_packing_moe_noise,
        "_unsloth_get_model_config": rl._unsloth_get_model_config,
        "_unsloth_get_final_logit_softcapping": rl._unsloth_get_final_logit_softcapping,
        "_unsloth_resolve_logit_scales": rl._unsloth_resolve_logit_scales,
        "calculate_pad_tokens_in_prompt": _pad_count,
        "left_pack_padding": _left_pack,
        "autotune_batch_and_chunks": _never_called("autotune_batch_and_chunks"),
        "create_completion_attention_mask": _never_called("create_completion_attention_mask"),
        "self": trainer,
        "model": model,
    }


def _batch():
    generator = torch.Generator().manual_seed(7)
    input_ids = torch.randint(PAD + 1, VOCAB, (2, SEQ), generator = generator)
    attention_mask = torch.ones(2, SEQ, dtype = torch.long)
    return input_ids, attention_mask


def _run(
    function,
    model,
    trainer,
    globals_overrides = None,
):
    input_ids, attention_mask = _batch()
    globals_ = _globals(model, trainer)
    if globals_overrides:
        globals_.update(globals_overrides)
    function.__globals__.update(globals_)
    return function(
        trainer,
        model,
        input_ids,
        attention_mask,
        KEEP,
    )


@pytest.mark.parametrize("prior", [None, "0"], ids = ["was_unset", "was_zero"])
def test_exception_inside_the_logprob_loop_restores_the_flag(_eager_zoo, monkeypatch, prior):
    function = _scoring_function()
    model = _ScoringModel(error = RuntimeError("synthetic OOM"))
    if prior is None:
        monkeypatch.delenv(ENV, raising = False)
    else:
        monkeypatch.setenv(ENV, prior)

    with pytest.raises(RuntimeError, match = "synthetic OOM"):
        _run(function, model, _trainer(model))

    assert model.seen_env == "1", "the forward never saw the flag, so this test proved nothing"
    assert os.environ.get(ENV) == prior


@pytest.mark.parametrize("prior", [None, "0"], ids = ["was_unset", "was_zero"])
def test_keyboard_interrupt_inside_the_logprob_loop_restores_the_flag(
    _eager_zoo, monkeypatch, prior
):
    function = _scoring_function()
    model = _ScoringModel(error = KeyboardInterrupt())
    if prior is None:
        monkeypatch.delenv(ENV, raising = False)
    else:
        monkeypatch.setenv(ENV, prior)

    with pytest.raises(KeyboardInterrupt):
        _run(function, model, _trainer(model))

    assert model.seen_env == "1"
    assert os.environ.get(ENV) == prior


@pytest.mark.parametrize("prior", [None, "0"], ids = ["was_unset", "was_zero"])
def test_a_clean_logprob_pass_restores_the_flag(_eager_zoo, monkeypatch, prior):
    function = _scoring_function()
    model = _ScoringModel()
    if prior is None:
        monkeypatch.delenv(ENV, raising = False)
    else:
        monkeypatch.setenv(ENV, prior)

    out = _run(function, model, _trainer(model))

    assert model.seen_env == "1"
    assert out[0].shape == (2, KEEP)
    assert out[1] is None
    assert os.environ.get(ENV) == prior


def test_a_preexisting_leak_is_flattened_not_propagated(_eager_zoo, monkeypatch):
    """A "1" at entry is a leak: the pass must not hand it back out again."""
    function = _scoring_function()
    model = _ScoringModel()
    monkeypatch.setenv(ENV, "1")

    _run(function, model, _trainer(model))

    assert os.environ.get(ENV) == "0"


def test_llama_for_inference_clears_a_leaked_flag(monkeypatch):
    from unsloth.models.llama import FastLlamaModel

    monkeypatch.setenv(ENV, "1")
    model = FastLlamaModel.for_inference(torch.nn.Linear(4, 4))
    assert os.environ.get(ENV) == "0"


def test_llama_for_inference_leaves_the_flag_off_when_unset(monkeypatch):
    from unsloth.models.llama import FastLlamaModel

    monkeypatch.delenv(ENV, raising = False)
    FastLlamaModel.for_inference(torch.nn.Linear(4, 4))
    assert os.environ.get(ENV) == "0"


def _env_assigns(function_source, name):
    """Every ``os.environ[ENV] = ...`` write inside a def, mapped to whether a try/finally guards it."""
    tree = ast.parse(textwrap.dedent(function_source))
    (function,) = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    guarded = []
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Assign)
            and [ast.unparse(t) for t in node.targets] == [f"os.environ['{ENV}']"]
        ):
            continue
        inside_finally_guarded_try = any(
            isinstance(parent, ast.Try)
            and parent.finalbody
            and any(node is stmt for stmt in ast.walk(parent))
            for parent in ast.walk(function)
        )
        guarded.append((node.lineno, inside_finally_guarded_try))
    return guarded


@pytest.mark.parametrize(
    "patch",
    [
        ("grpo_trainer__get_per_token_logps", "_get_per_token_logps"),
        (
            "grpo_trainer__get_per_token_logps_and_entropies",
            "_get_per_token_logps_and_entropies",
        ),
    ],
    ids = ["logps", "logps_and_entropies"],
)
def test_every_env_write_sits_inside_a_try_finally(patch):
    factory, name = patch
    source = getattr(rl, factory)(name, f"def {name}(): pass")
    assigns = _env_assigns(source, name)
    assert assigns, f"{name} no longer writes {ENV}"
    unguarded = [line for line, guarded in assigns if not guarded]
    assert not unguarded, f"{name} writes {ENV} outside a try/finally at lines {unguarded}"
