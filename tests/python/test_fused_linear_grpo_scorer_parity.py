# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""GRPO old / ref logps must use the policy pass scorer, else the importance ratio starts away from 1."""

from __future__ import annotations

import inspect
import os
import textwrap
import types

import pytest
import torch

import unsloth.models.rl_replacements as rl

zoo_rl = pytest.importorskip("unsloth_zoo.rl_replacements")
if not hasattr(zoo_rl, "grpo_chunked_hidden_states_logprobs"):
    pytest.skip("zoo without the fused GRPO scorer", allow_module_level = True)


def _resolve_scorer(trainer):
    src = rl.grpo_trainer__get_per_token_logps_and_entropies(
        "_get_per_token_logps_and_entropies", None
    )
    start = src.index("from unsloth_zoo.rl_replacements import (\n")
    start = src.rindex("\n", 0, start) + 1
    end = src.index("output_embeddings = unwrapped_model.get_output_embeddings()")
    block = textwrap.dedent(src[start:end])
    assert "chunked_hidden_states_selective_log_softmax" in block
    ns = {"os": os, "self": trainer}
    exec(block, ns)
    return ns["chunked_hidden_states_selective_log_softmax"]


def _trainer():
    args = types.SimpleNamespace(top_k = None, top_p = 1.0, min_p = None)
    return types.SimpleNamespace(args = args)


def test_old_ref_scorer_is_the_policy_scorer(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", raising = False)
    assert _resolve_scorer(_trainer()) is zoo_rl.grpo_chunked_hidden_states_logprobs


def test_recompute_mode_matches_policy_scorer(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "1")
    monkeypatch.setattr(zoo_rl, "grpo_truncated_support_kwargs", lambda trainer: {})
    assert (
        _resolve_scorer(_trainer()) is zoo_rl.chunked_hidden_states_selective_log_softmax_recompute
    )
    # zoo's policy pass falls back to the plain helper for truncated support under recompute.
    monkeypatch.setattr(
        zoo_rl, "grpo_truncated_support_kwargs", lambda trainer: {"support_top_k": 4}
    )
    assert _resolve_scorer(_trainer()) is zoo_rl.chunked_hidden_states_selective_log_softmax


def test_no_grad_scores_equal_policy_scores(monkeypatch):
    """The ratio of identical policies is exactly 1 with the shared scorer (bf16 head)."""
    monkeypatch.delenv("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", raising = False)
    scorer = _resolve_scorer(_trainer())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)
    hidden = (torch.randn(2, 16, 64, generator = g) * 4).to(torch.bfloat16).to(device)
    head = (torch.randn(512, 64, generator = g) * 4).to(torch.bfloat16).to(device)
    index = torch.randint(0, 512, (2, 16), generator = g).to(device)
    with torch.no_grad():
        old = scorer(hidden, head, index, 4)
    policy = zoo_rl.grpo_chunked_hidden_states_logprobs(
        hidden.clone().requires_grad_(True), head, index, 4
    )
    torch.testing.assert_close(policy.detach(), old, rtol = 0, atol = 1e-6)
    if device == "cuda":
        # The legacy helper rounds logits to bf16 on CUDA, so it cannot stand in for the policy scorer.
        with torch.no_grad():
            # A zoo with the fused public scorer needs fused_linear = False for its own chunked path.
            kw = (
                {"fused_linear": False}
                if "fused_linear"
                in inspect.signature(zoo_rl.chunked_hidden_states_selective_log_softmax).parameters
                else {}
            )
            legacy = zoo_rl.chunked_hidden_states_selective_log_softmax(
                hidden, head, index, 4, **kw
            )
        assert (policy.detach() - legacy).abs().max().item() > 1e-3
