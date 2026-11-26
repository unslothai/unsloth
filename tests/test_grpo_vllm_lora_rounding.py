# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO with vLLM rollouts scores LoRA in the base-then-LoRA order vLLM samples with.

vLLM rounds the base output to bf16 before adding the LoRA term, so it drops LoRA updates below half a bf16
step. The fused order adds the LoRA term inside the GEMM, which keeps them wherever the GEMM accumulates onto
it in fp32 (CPU, and cuBLAS on some GPUs). Then the trainer's policy runs ahead of the sampler's."""

from __future__ import annotations

import ast
import importlib.util
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.kernels import utils as kernel_utils


def _kept(y, base, delta):
    # Share of the exact LoRA output delta that survives in y.
    got = y.double() - base.double()
    return float((got * delta).sum() / (delta * delta).sum())


def _small_update_layer():
    # A LoRA update well below half a bf16 step of the base output, as in early RL training.
    torch.manual_seed(3407)
    X = torch.randn(64, 256).bfloat16()
    W = (torch.randn(384, 256) * 0.03).bfloat16()
    A = (torch.randn(8, 256) * 0.05).bfloat16()
    B = (torch.randn(384, 8) * 5e-5).bfloat16()
    s = 2.0
    delta = X.double() @ (s * (B.double() @ A.double())).T
    base = X @ W.T
    # vLLM LoRA: fp32 shrink, then the expand accumulates onto the bf16 base output.
    rollout = (base.float() + (X.float() @ A.float().T) @ (B.float() * s).T).to(torch.bfloat16)
    return X, W, A, B, s, delta, base, rollout


@pytest.fixture
def fused_state(monkeypatch):
    monkeypatch.setattr(kernel_utils, "_LORA_FUSED_PASSES_ENV", None)
    monkeypatch.setattr(kernel_utils, "LORA_FUSED_PASSES", True)
    return kernel_utils


def test_trainer_keeps_what_the_rollout_keeps(fused_state):
    X, W, A, B, s, delta, base, rollout = _small_update_layer()
    kept_rollout = _kept(rollout, base, delta)
    assert kept_rollout < 0.5  # the rollout really drops part of the update
    assert fused_state.match_vllm_lora_rounding() is True
    assert fused_state.LORA_FUSED_PASSES is False
    trainer = fused_state.matmul_lora(X, W, None, A, B, s)
    assert abs(_kept(trainer, base, delta) - kept_rollout) < 0.05


def test_fused_order_runs_ahead_of_the_rollout_on_cpu(fused_state):
    # The mismatch the switch removes: on CPU the fused order keeps the whole update.
    X, W, A, B, s, delta, base, rollout = _small_update_layer()
    fused = fused_state.matmul_lora(X, W, None, A, B, s)
    assert _kept(fused, base, delta) > 0.9
    assert _kept(rollout, base, delta) < 0.5


def test_explicit_env_wins(monkeypatch):
    monkeypatch.setattr(kernel_utils, "_LORA_FUSED_PASSES_ENV", "1")
    monkeypatch.setattr(kernel_utils, "LORA_FUSED_PASSES", True)
    assert kernel_utils.match_vllm_lora_rounding() is False
    assert kernel_utils.LORA_FUSED_PASSES is True


def _patched_trl():
    spec = (
        importlib.util.find_spec("trl.trainer.grpo_trainer")
        if importlib.util.find_spec("trl")
        else None
    )
    if spec is None or spec.origin is None:
        pytest.skip("trl is not installed")
    source = Path(spec.origin).read_text(encoding = "utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "_generate_and_score_completions"
    )
    function = "    " + ast.get_source_segment(source, node)
    if "vllm_importance_sampling_correction" not in function:
        pytest.skip("this TRL has no vLLM importance sampling correction")
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions

    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", function
    )
    ast.parse(textwrap.dedent(patched))
    return patched


@pytest.mark.parametrize("use_vllm, expected_fused", [(True, False), (False, True)])
def test_generation_matches_vllm_rounding(fused_state, use_vllm, expected_fused):
    patched = _patched_trl()
    match = re.search(
        r"\n( +)if bool\(getattr\(self, 'use_vllm', False\)\):\n\1    try:\n"
        r"\1        from unsloth\.kernels\.utils import match_vllm_lora_rounding.*?\n\1        pass\n",
        patched,
        re.S,
    )
    assert match is not None
    header = re.search(r"def _generate_and_score_completions\([^)]*\)[^:\n]*:[ \t]*\n", patched)
    first_generate = min(
        i for i in (patched.find(x, header.end()) for x in ("generate(", "_generate(")) if i != -1
    )
    assert header.end() <= match.start() < match.end() < first_generate
    ns = {}
    exec("def run(self):\n" + textwrap.indent(textwrap.dedent(match.group(0)), "    "), ns)
    ns["run"](SimpleNamespace(use_vllm = use_vllm))
    assert fused_state.LORA_FUSED_PASSES is expected_fused
