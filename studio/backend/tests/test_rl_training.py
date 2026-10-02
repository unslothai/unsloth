# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import dataclasses
import sys
import types

import pytest
from datasets import Dataset
from pydantic import ValidationError

from core.training.rl import build_rl_trainer, format_rl_dataset, resolve_role_columns, rl_lengths, rl_log_metrics
from models.training import TrainingStartRequest

BASE = {"model_name": "unsloth/Qwen3-4B-Base", "training_type": "LoRA/QLoRA", "format_type": "auto"}


def test_grpo_rows_keep_answer_out_of_the_prompt():
    ds = Dataset.from_list([{"question": "2+2?", "answer": "4", "junk": 1}])
    out, roles = format_rl_dataset(ds, "grpo")
    assert roles == {"prompt": "question", "answer": "answer"}
    assert sorted(out.column_names) == ["answer", "prompt"]
    assert out[0]["prompt"] == [{"role": "user", "content": "2+2?"}]
    assert out[0]["answer"] == "4"


def test_grpo_mapping_system_prompt_and_kept_reward_columns():
    ds = Dataset.from_list([{"q": "hi", "sys": "be brief", "gold": "x", "unit": "kg"}])
    out, _ = format_rl_dataset(
        ds, "grpo", {"q": "prompt", "sys": "system", "gold": "answer"}, keep_columns = ("unit",)
    )
    assert out[0]["prompt"][0] == {"role": "system", "content": "be brief"}
    assert out[0]["unit"] == "kg"


def test_preference_rows_strip_the_user_turn_from_full_conversations():
    # ultrafeedback_binarized stores chosen/rejected as the whole conversation.
    convo = lambda reply: [{"role": "user", "content": "q"}, {"role": "assistant", "content": reply}]
    ds = Dataset.from_list([{"prompt": "q", "chosen": convo("good"), "rejected": convo("bad")}])
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["chosen"] == [{"role": "assistant", "content": "good"}]
    assert out[0]["rejected"] == [{"role": "assistant", "content": "bad"}]
    assert out[0]["prompt"] == [{"role": "user", "content": "q"}]


def test_missing_roles_name_the_columns_found():
    with pytest.raises(ValueError, match = "chosen, rejected"):
        resolve_role_columns(["prompt", "text"], "orpo")


def test_rl_lengths_fit_inside_max_seq_length():
    assert rl_lengths("grpo", 1024, {}) == (256, 768)
    assert rl_lengths("dpo", 1024, {}) == (512, 512)
    prompt, completion = rl_lengths("grpo", 512, {"max_prompt_length": 10_000, "max_completion_length": 10_000})
    assert prompt + completion <= 512


def test_rl_log_metrics_picks_rl_keys_only():
    logs = {
        "loss": 0.1,
        "reward": 1.5,
        "rewards/exact_answer/mean": 2.0,
        "kl": 0.01,
        "completion_length": 212.0,
        "reward_std": float("nan"),
    }
    assert rl_log_metrics(logs) == {
        "reward": 1.5,
        "rewards/exact_answer/mean": 2.0,
        "kl": 0.01,
        "completion_length": 212.0,
    }
    assert rl_log_metrics({"loss": 1.0}) is None


def test_request_defaults_to_sft_so_saved_configs_load_unchanged():
    assert TrainingStartRequest(**BASE).objective == "sft"


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"objective": "grpo"}, "at least one reward"),
        ({"objective": "dpo", "training_type": "Continued Pretraining"}, "Continued Pretraining"),
        ({"objective": "orpo", "is_dataset_image": True}, "text datasets"),
        ({"objective": "grpo", "grpo_rewards": [{"name": "a"}, {"name": "a"}]}, "only be selected once"),
        ({"objective": "grpo", "grpo_rewards": [{"name": "a"}], "grpo_num_generations": 1}, "greater than or equal"),
    ],
)
def test_request_refuses_unsupported_rl_combinations(extra, message):
    with pytest.raises(ValidationError, match = message):
        TrainingStartRequest(**{**BASE, **extra})


@dataclasses.dataclass
class _FakeGRPOConfig:
    output_dir: str = "out"
    use_vllm: bool = False
    beta: float = 0.0
    temperature: float = 1.0
    num_generations: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 256
    reward_weights: list = None
    log_completions: bool = False
    loss_type: str = "dapo"
    importance_sampling_level: str = "token"
    mask_truncated_completions: bool = False
    epsilon_high: float = None


@pytest.mark.parametrize(
    "variant, loss_type, level",
    [(None, "dapo", "token"), ("dr_grpo", "dr_grpo", "token"), ("bnpo", "bnpo", "token"), ("gspo", "dr_grpo", "sequence")],
)
def test_grpo_variant_reaches_the_trl_config(monkeypatch, variant, loss_type, level):
    fake = types.SimpleNamespace(GRPOConfig = _FakeGRPOConfig, GRPOTrainer = lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "trl", fake)
    spec = {"name": "exact", "weight": 2.0, "rule": {"type": "length", "max_chars": 9, "score": {"over": -1.0, "under": 0.0}}}
    kw = build_rl_trainer(
        "grpo",
        model = None,
        tokenizer = None,
        train_dataset = None,
        eval_dataset = None,
        config_args = {"output_dir": "o", "max_seq_length": 1024, "packing": False},
        settings = {"variant": variant, "mask_truncated_completions": True, "epsilon_high": 0.28},
        reward_specs = [spec],
    )
    args = kw["args"]
    assert (args.loss_type, args.importance_sampling_level) == (loss_type, level)
    assert args.mask_truncated_completions is True and args.epsilon_high == 0.28
    assert args.reward_weights == [2.0] and args.use_vllm is False


def test_request_refuses_unknown_grpo_variant():
    with pytest.raises(ValidationError):
        TrainingStartRequest(**{**BASE, "objective": "grpo", "grpo_rewards": [{"name": "x"}], "grpo_variant": "ppo"})
