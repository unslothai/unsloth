# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO keeps TRL's old logprob gate: rewriting it to `or (self.use_vllm)` paid a full no-grad
forward per on-policy vLLM step with IS correction off, where old == per_token_logps.detach()."""

from __future__ import annotations

import ast
import importlib.util
import re

import pytest


def _patched_generate_and_score():
    if importlib.util.find_spec("trl") is None:
        pytest.skip("trl is not installed")
    spec = importlib.util.find_spec("trl.trainer.grpo_trainer")
    if spec is None or spec.origin is None:
        pytest.skip("trl is not installed")
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions

    with open(spec.origin, "r", encoding = "utf-8") as fh:
        module_source = fh.read()
    node = next(
        n
        for n in ast.walk(ast.parse(module_source))
        if isinstance(n, ast.FunctionDef) and n.name == "_generate_and_score_completions"
    )
    source = "    " + ast.get_source_segment(module_source, node)
    if "generate_every != 0 or (" not in source:
        pytest.skip("this TRL has no vLLM clause in the old logprob gate (< 0.23.0)")
    return grpo_trainer__generate_and_score_completions("_generate_and_score_completions", source)


def _gate_condition(patched):
    match = re.search(
        r"if (self\.args\.gradient_accumulation_steps % generate_every != 0 or \(.*?\)):\n",
        patched,
        flags = re.DOTALL,
    )
    assert match is not None, "old logprob gate not found in the patched function"
    return " ".join(match.group(1).split())


def test_old_logprob_gate_matches_trl():
    condition = _gate_condition(_patched_generate_and_score())
    assert "self.use_vllm and self.vllm_importance_sampling_correction" in condition, condition


@pytest.mark.parametrize(
    "grad_accum, generate_every, use_vllm, is_correction, expected",
    [
        (4, 4, True, False, False),
        (4, 4, True, True, True),
        (4, 4, False, False, False),
        (4, 8, True, False, True),
        (4, 8, False, False, True),
    ],
)
def test_old_logprob_gate_truth_table(
    grad_accum, generate_every, use_vllm, is_correction, expected
):
    condition = _gate_condition(_patched_generate_and_score())

    class _Args:
        gradient_accumulation_steps = grad_accum

    class _Self:
        args = _Args()

    _Self.use_vllm = use_vllm
    _Self.vllm_importance_sampling_correction = is_correction
    value = eval(condition, {}, {"self": _Self(), "generate_every": generate_every})
    assert bool(value) is expected
