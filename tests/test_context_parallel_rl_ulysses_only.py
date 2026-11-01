# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DPO / GRPO context parallelism shards Llama-style decoders only: models whose attention runs
only on the Hugging Face Ulysses path (MiniMax-M2, and every other ULYSSES_ATTENTION_CLASSES entry)
must be refused there, while SFT accepts them and Llama-style RL stays accepted."""

import types

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.models.llama import LlamaAttention_fast_forward
from unsloth.utils import ulysses_hybrid as uh

transformers = pytest.importorskip("transformers")

RL_REFUSAL = "DPO / GRPO context parallelism supports Llama-style attention only"


def _trainer():
    return types.SimpleNamespace(accelerator = None, train_dataset = None, eval_dataset = None)


def _rl(model, size = 2):
    # What the DPO / GRPO patched __init__ runs: allow_fsdp2 (and so allow_hybrids) off.
    return cp._check_cp_trainer(_trainer(), size, "ulysses", [model, None])


def _sft(model, size = 2):
    return cp._check_cp_trainer(_trainer(), size, "ulysses", [model], allow_fsdp2 = True)


def _fake_model(attn_cls):
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = attn_cls()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer()])
            self.config = types.SimpleNamespace(
                model_type = "fake",
                num_attention_heads = 4,
                num_key_value_heads = 2,
                layer_types = ["full_attention"] * 2,
            )

    return Model()


def _llama_style():
    return _fake_model(
        type("LlamaAttention", (torch.nn.Module,), {"forward": LlamaAttention_fast_forward})
    )


def _minimax_m2():
    config = transformers.MiniMaxM2Config(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        num_local_experts = 2,
        num_experts_per_tok = 1,
    )
    return transformers.MiniMaxM2ForCausalLM(config)


# Every attention class the Hugging Face Ulysses wrapper serves; upper layers extend the tuple.
ULYSSES_ONLY = sorted(uh.ULYSSES_ATTENTION_CLASSES)


@pytest.mark.parametrize("name", ULYSSES_ONLY)
def test_rl_refuses_ulysses_only_attention(name):
    model = _fake_model(type(name, (torch.nn.Module,), {}))
    assert cp._runs_hf_ulysses_attention(model, "ulysses")
    _sft(model)
    with pytest.raises(NotImplementedError, match = RL_REFUSAL):
        _rl(model)


@pytest.mark.skipif(not hasattr(transformers, "MiniMaxM2Config"), reason = "no MiniMax-M2")
def test_rl_refuses_real_minimax_m2_and_sft_accepts_it():
    model = _minimax_m2()
    assert not cp._stateful_mixers(model)
    _sft(model)
    with pytest.raises(NotImplementedError, match = RL_REFUSAL):
        _rl(model)
    # The reference model is checked too.
    with pytest.raises(NotImplementedError, match = RL_REFUSAL):
        cp._check_cp_trainer(_trainer(), 2, "ulysses", [_llama_style(), model])


def test_rl_keeps_accepting_llama_style_attention():
    model = _llama_style()
    assert not cp._runs_hf_ulysses_attention(model, "ulysses")
    _rl(model)
    _sft(model)


def test_ring_backend_is_not_flagged_as_ulysses_attention():
    model = _fake_model(type("MiniMaxM2Attention", (torch.nn.Module,), {}))
    assert not cp._runs_hf_ulysses_attention(model, "ring")
