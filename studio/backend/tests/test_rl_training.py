# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import dataclasses
import sys
import types

import pytest
from datasets import Dataset
from pydantic import ValidationError

from core.training.rl import (
    build_rl_trainer,
    install_fsdp_import_stub,
    render_prompts_without_thinking,
    format_rl_dataset,
    resolve_role_columns,
    rl_lengths,
    rl_log_metrics,
)
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
    convo = lambda reply: [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": reply},
    ]
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
    prompt, completion = rl_lengths(
        "grpo", 512, {"max_prompt_length": 10_000, "max_completion_length": 10_000}
    )
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
        (
            {"objective": "grpo", "grpo_rewards": [{"name": "a"}, {"name": "a"}]},
            "only be selected once",
        ),
        (
            {"objective": "grpo", "grpo_rewards": [{"name": "a"}], "grpo_num_generations": 1},
            "greater than or equal",
        ),
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
    [
        (None, "dapo", "token"),
        ("dr_grpo", "dr_grpo", "token"),
        ("bnpo", "bnpo", "token"),
        ("gspo", "dr_grpo", "sequence"),
    ],
)
def test_grpo_variant_reaches_the_trl_config(monkeypatch, variant, loss_type, level):
    fake = types.SimpleNamespace(GRPOConfig = _FakeGRPOConfig, GRPOTrainer = lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "trl", fake)
    spec = {
        "name": "exact",
        "weight": 2.0,
        "rule": {"type": "length", "max_chars": 9, "score": {"over": -1.0, "under": 0.0}},
    }
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
        TrainingStartRequest(
            **{**BASE, "objective": "grpo", "grpo_rewards": [{"name": "x"}], "grpo_variant": "ppo"}
        )


def test_fsdp_stub_only_when_the_real_module_cannot_import(monkeypatch):
    # The stub stands in for torch.distributed.fsdp; it needs torch itself (the no-torch legs lack it).
    pytest.importorskip("torch")
    import builtins

    real_import = builtins.__import__

    def no_fsdp(name, *args, **kwargs):
        if name.startswith("torch.distributed.fsdp"):
            raise ModuleNotFoundError("No module named 'torch._C._distributed_c10d'")
        return real_import(name, *args, **kwargs)

    for name in [n for n in sys.modules if n.startswith("torch.distributed.fsdp")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(builtins, "__import__", no_fsdp)
    assert install_fsdp_import_stub() is True
    monkeypatch.setattr(builtins, "__import__", real_import)
    from torch.distributed.fsdp import FullyShardedDataParallel

    assert FullyShardedDataParallel.__module__ == "core.training.rl"
    assert install_fsdp_import_stub() is False


def test_system_prompt_fills_rows_without_one_and_keeps_their_own():
    ds = Dataset.from_list(
        [
            {"question": "2+2?", "answer": "4", "system": ""},
            {"question": "3+3?", "answer": "6", "system": "Be terse."},
        ]
    )
    out, _ = format_rl_dataset(
        ds,
        "grpo",
        {"question": "prompt", "answer": "answer", "system": "system"},
        system_prompt = "Use <answer> tags.",
    )
    assert out[0]["prompt"][0] == {"role": "system", "content": "Use <answer> tags."}
    assert out[1]["prompt"][0] == {"role": "system", "content": "Be terse."}
    plain, _ = format_rl_dataset(Dataset.from_list([{"question": "q", "answer": "a"}]), "grpo")
    assert [m["role"] for m in plain[0]["prompt"]] == ["user"]


class _ThinkingTokenizer:
    chat_template = "{% if enable_thinking is defined and enable_thinking is false %}...{% endif %}"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + ("<assistant><think>" if enable_thinking else "<assistant><think></think>")


def test_thinking_switch_renders_prompts_only_for_templates_that_have_one():
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": "2+2?"}], "answer": "4"}])
    out, rendered = render_prompts_without_thinking(ds, _ThinkingTokenizer(), False)
    assert rendered and out[0]["prompt"] == "<user>2+2?<assistant><think></think>"
    assert out[0]["answer"] == "4"

    class Plain:
        chat_template = "{{ messages }}"

    same, rendered = render_prompts_without_thinking(ds, Plain(), False)
    assert not rendered and same[0]["prompt"] == [{"role": "user", "content": "2+2?"}]


def test_resume_payload_from_stored_config_keeps_rl_settings():
    # A resume posts the stored worker config back: RL fields sit under rl_settings / reward_specs.
    stored = {
        **BASE,
        "objective": "grpo",
        "rl_settings": {
            "beta": 0.04,
            "max_prompt_length": 256,
            "num_generations": 8,
            "max_completion_length": 128,
            "temperature": 0.7,
            "variant": "gspo",
            "enable_thinking": None,
            "system_prompt": "Answer briefly.",
            "mask_truncated_completions": True,
            "epsilon_high": 0.28,
        },
        "reward_specs": [{"name": "exact-answer", "weight": 2.0, "rule": {"type": "exact_match"}}],
    }
    req = TrainingStartRequest.model_validate(stored)
    assert req.objective == "grpo" and req.rl_beta == 0.04 and req.rl_max_prompt_length == 256
    assert (req.grpo_num_generations, req.grpo_max_completion_length) == (8, 128)
    assert (req.grpo_temperature, req.grpo_variant, req.grpo_enable_thinking) == (0.7, "gspo", None)
    assert req.rl_system_prompt == "Answer briefly." and req.grpo_mask_truncated_completions
    assert req.grpo_epsilon_high == 0.28
    assert [(r.name, r.weight) for r in req.grpo_rewards] == [("exact-answer", 2.0)]
    # An explicit request field still wins over the stored copy.
    assert TrainingStartRequest.model_validate({**stored, "rl_beta": 0.5}).rl_beta == 0.5


def test_grpo_reward_column_kept_when_it_also_fills_a_role():
    ds = Dataset.from_list([{"question": "2+2?", "solution": "4"}])
    out, roles = format_rl_dataset(
        ds, "grpo", {"solution": "answer"}, keep_columns = ("solution", "answer")
    )
    assert roles["answer"] == "solution"
    assert out[0]["answer"] == "4" and out[0]["solution"] == "4"


def test_grpo_reward_column_missing_from_dataset_is_refused():
    ds = Dataset.from_list([{"question": "2+2?", "response": "4"}])
    with pytest.raises(ValueError, match = "compares against answer"):
        format_rl_dataset(ds, "grpo", keep_columns = ("answer",))


def test_trl_config_without_max_prompt_length_still_builds(monkeypatch):
    # TRL 1.x dropped max_prompt_length from DPOConfig / GRPOConfig.
    @dataclasses.dataclass
    class _NoPromptLenDPOConfig:
        output_dir: str = None
        beta: float = 0.1
        max_length: int = None

    fake = types.SimpleNamespace(DPOConfig = _NoPromptLenDPOConfig, DPOTrainer = lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "trl", fake)
    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(PatchDPOTrainer = lambda: None))
    kw = build_rl_trainer(
        "dpo",
        model = None,
        tokenizer = None,
        train_dataset = None,
        eval_dataset = None,
        config_args = {"output_dir": "o", "max_seq_length": 1024},
        settings = {"beta": 0.2},
        reward_specs = [],
    )
    assert (kw["args"].beta, kw["args"].max_length) == (0.2, 1024)


def test_grpo_eval_batch_is_rounded_to_whole_prompt_groups(monkeypatch):
    fake = types.SimpleNamespace(GRPOConfig = _FakeGRPOConfigWithEval, GRPOTrainer = lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "trl", fake)
    spec = {
        "name": "len",
        "rule": {"type": "length", "max_chars": 9, "score": {"over": -1.0, "under": 0.0}},
    }
    kw = build_rl_trainer(
        "grpo",
        model = None,
        tokenizer = None,
        train_dataset = None,
        eval_dataset = None,
        config_args = {"output_dir": "o", "max_seq_length": 1024, "per_device_eval_batch_size": 2},
        settings = {"num_generations": 4},
        reward_specs = [spec],
    )
    assert kw["args"].per_device_eval_batch_size == 4


@dataclasses.dataclass
class _FakeGRPOConfigWithEval(_FakeGRPOConfig):
    per_device_eval_batch_size: int = 8


def test_request_refuses_rl_with_decision_training():
    with pytest.raises(ValidationError, match = "decision"):
        TrainingStartRequest(**BASE, objective = "dpo", is_decision = True)


def test_resume_scores_with_the_stored_reward_rules():
    from routes.training import _stored_reward_specs

    rule = {"type": "exact_match", "compare_to": "answer"}
    stored = {
        "reward_specs": [
            {"name": "exact-answer", "weight": 2.0, "rule": rule},
            {"name": "broken", "weight": 1.0},
        ]
    }
    assert _stored_reward_specs(stored) == {
        "exact-answer": {"name": "exact-answer", "weight": 2.0, "rule": rule}
    }
    assert _stored_reward_specs({}) == {}


def test_grpo_keeps_a_reward_column_named_like_a_preference_role():
    ds = Dataset.from_list([{"question": "2+2?", "answer": "4", "chosen": "x"}])
    out, _ = format_rl_dataset(ds, "grpo", keep_columns = ("chosen",))
    assert out[0]["chosen"] == "x"


def test_grpo_duplicate_rewards_are_matched_after_normalising():
    with pytest.raises(ValidationError, match = "only be selected once"):
        TrainingStartRequest(
            **BASE,
            objective = "grpo",
            grpo_rewards = [{"name": "exact-answer"}, {"name": "EXACT-ANSWER"}],
        )
