# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO with vLLM rollouts scores LoRA in the base-then-LoRA order vLLM samples with, only while it trains.

vLLM rounds the base output to bf16 before adding the LoRA term, so it drops LoRA updates below half a bf16
step. The fused order adds the LoRA term inside the GEMM, which keeps them wherever the GEMM accumulates onto
it in fp32 (CPU, and cuBLAS on A100 / L4). Then the trainer's policy runs ahead of the sampler's."""

from __future__ import annotations

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.kernels import utils as kernel_utils
from unsloth.models.rl import _wrap_grpo_vllm_lora_rounding


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
    # The mismatch: on CPU the fused order keeps the whole update.
    assert _kept(fused_state.matmul_lora(X, W, None, A, B, s), base, delta) > 0.9
    with fused_state.vllm_lora_rounding() as off:
        assert off is True and fused_state.LORA_FUSED_PASSES is False
        trainer = fused_state.matmul_lora(X, W, None, A, B, s)
    assert abs(_kept(trainer, base, delta) - kept_rollout) < 0.05
    assert fused_state.LORA_FUSED_PASSES is True


def test_restored_after_exception_and_nesting(fused_state):
    with pytest.raises(RuntimeError):
        with fused_state.vllm_lora_rounding():
            with fused_state.vllm_lora_rounding():
                assert fused_state.LORA_FUSED_PASSES is False
            assert fused_state.LORA_FUSED_PASSES is False
            raise RuntimeError
    assert fused_state.LORA_FUSED_PASSES is True
    with fused_state.vllm_lora_rounding(False) as off:
        assert off is False and fused_state.LORA_FUSED_PASSES is True


def test_explicit_env_wins(monkeypatch):
    monkeypatch.setattr(kernel_utils, "_LORA_FUSED_PASSES_ENV", "1")
    monkeypatch.setattr(kernel_utils, "LORA_FUSED_PASSES", True)
    with kernel_utils.vllm_lora_rounding() as off:
        assert off is False and kernel_utils.LORA_FUSED_PASSES is True


class _Trainer:
    def __init__(
        self,
        use_vllm,
        fail = False,
    ):
        self.use_vllm, self.fail, self.seen = use_vllm, fail, []

    def train(self, *args, **kwargs):
        self.seen.append(("train", kernel_utils.LORA_FUSED_PASSES))
        self.evaluate()
        if self.fail:
            raise ValueError("boom")
        return "trained"

    def evaluate(self, *args, **kwargs):
        self.seen.append(("evaluate", kernel_utils.LORA_FUSED_PASSES))
        return {}


@pytest.mark.parametrize("use_vllm", [True, False])
def test_grpo_train_scopes_the_order(fused_state, use_vllm):
    cls = type("T", (_Trainer,), {})
    _wrap_grpo_vllm_lora_rounding(cls)
    _wrap_grpo_vllm_lora_rounding(cls)  # idempotent
    trainer = cls(use_vllm)
    assert trainer.train() == "trained"
    assert trainer.seen == [("train", not use_vllm), ("evaluate", not use_vllm)]
    # A later SFT in the same process keeps the fused order.
    assert fused_state.LORA_FUSED_PASSES is True
    trainer.seen.clear()
    trainer.evaluate()
    assert trainer.seen == [("evaluate", not use_vllm)]
    assert fused_state.LORA_FUSED_PASSES is True


def test_grpo_train_restores_after_failure(fused_state):
    cls = type("T", (_Trainer,), {})
    _wrap_grpo_vllm_lora_rounding(cls)
    with pytest.raises(ValueError):
        cls(True, fail = True).train()
    assert fused_state.LORA_FUSED_PASSES is True
