# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
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
        (
            {"objective": "dpo", "s3_config": {"bucket": "b", "useIamRole": True}},
            "S3 data",
        ),
        (
            {"objective": "dpo", "drafter": {"enabled": True}},
            "drafter training is not available with DPO",
        ),
        ({"objective": "dpo", "rl_beta": 0}, "DPO needs a beta above 0"),
        ({"objective": "orpo", "rl_beta": 0}, "ORPO needs a beta above 0"),
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
    epsilon: float = 0.2
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


def test_an_empty_clip_high_is_the_regular_clip_not_dapos(monkeypatch):
    # Unsloth reads an explicit dapo with epsilon_high None as the DAPO paper's 0.28.
    fake = types.SimpleNamespace(GRPOConfig = _FakeGRPOConfig, GRPOTrainer = lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "trl", fake)
    kw = build_rl_trainer(
        "grpo",
        model = None,
        tokenizer = None,
        train_dataset = None,
        eval_dataset = None,
        config_args = {"output_dir": "o", "max_seq_length": 1024, "packing": False},
        settings = {},
        reward_specs = [
            {
                "name": "r",
                "weight": 1.0,
                "rule": {"type": "length", "max_chars": 9, "score": {"over": -1.0, "under": 0.0}},
            }
        ],
    )
    assert kw["args"].loss_type == "dapo" and kw["args"].epsilon_high == 0.2


def test_gspo_is_refused_on_trl_without_sequence_sampling(monkeypatch):
    fields = {
        f.name: f.default
        for f in dataclasses.fields(_FakeGRPOConfig)
        if f.name != "importance_sampling_level"
    }
    old = dataclasses.make_dataclass(
        "OldGRPOConfig",
        [(name, object, dataclasses.field(default = d)) for name, d in fields.items()],
    )
    monkeypatch.setitem(
        sys.modules, "trl", types.SimpleNamespace(GRPOConfig = old, GRPOTrainer = lambda **kw: kw)
    )
    spec = {
        "name": "l",
        "rule": {"type": "length", "max_chars": 9, "score": {"over": -1.0, "under": 0.0}},
    }
    with pytest.raises(ValueError, match = "GSPO needs a newer TRL"):
        build_rl_trainer(
            "grpo",
            model = None,
            tokenizer = None,
            train_dataset = None,
            eval_dataset = None,
            config_args = {"output_dir": "o", "max_seq_length": 1024},
            settings = {"variant": "gspo"},
            reward_specs = [spec],
        )


@pytest.mark.parametrize("fp16_only", [True, False])
def test_grpo_drops_non_finite_logits_only_on_fp16_only_gpus(monkeypatch, fp16_only):
    import core.training.rl as rl

    @dataclasses.dataclass
    class _Config(_FakeGRPOConfig):
        generation_kwargs: dict = None

    monkeypatch.setattr(rl, "_fp16_only_gpu", lambda: fp16_only)
    monkeypatch.setitem(
        sys.modules, "trl", types.SimpleNamespace(GRPOConfig = _Config, GRPOTrainer = lambda **kw: kw)
    )
    spec = {
        "name": "l",
        "rule": {"type": "length", "max_chars": 9, "score": {"over": -1.0, "under": 0.0}},
    }
    kw = build_rl_trainer(
        "grpo",
        model = None,
        tokenizer = None,
        train_dataset = None,
        eval_dataset = None,
        config_args = {"output_dir": "o", "max_seq_length": 1024},
        settings = {},
        reward_specs = [spec],
    )
    expected = {"remove_invalid_values": True} if fp16_only else None
    assert kw["args"].generation_kwargs == expected


@pytest.mark.parametrize("version, loss_type", [("0.18.2", "bnpo"), ("0.23.1", "dapo")])
def test_default_dapo_falls_back_to_bnpo_on_old_trl(monkeypatch, version, loss_type):
    warnings = []
    monkeypatch.setitem(
        sys.modules,
        "trl",
        types.SimpleNamespace(
            __version__ = version, GRPOConfig = _FakeGRPOConfig, GRPOTrainer = lambda **kw: kw
        ),
    )
    spec = {
        "name": "l",
        "rule": {"type": "length", "max_chars": 9, "score": {"over": -1.0, "under": 0.0}},
    }
    kw = build_rl_trainer(
        "grpo",
        model = None,
        tokenizer = None,
        train_dataset = None,
        eval_dataset = None,
        config_args = {"output_dir": "o", "max_seq_length": 1024},
        settings = {},
        reward_specs = [spec],
        on_warning = warnings.append,
    )
    assert kw["args"].loss_type == loss_type
    assert bool(warnings) == (loss_type == "bnpo")


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

    # Restored by hand: the stub adds modules that monkeypatch would leave for later tests.
    before = {n: m for n, m in sys.modules.items() if n.startswith("torch.distributed.fsdp")}
    for name in before:
        del sys.modules[name]
    try:
        monkeypatch.setattr(builtins, "__import__", no_fsdp)
        assert install_fsdp_import_stub() is True
        monkeypatch.setattr(builtins, "__import__", real_import)
        from torch.distributed.fsdp import FullyShardedDataParallel

        assert FullyShardedDataParallel.__module__ == "core.training.rl"
        assert install_fsdp_import_stub() is False
    finally:
        for name in [n for n in sys.modules if n.startswith("torch.distributed.fsdp")]:
            del sys.modules[name]
        sys.modules.update(before)


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


@dataclasses.dataclass
class _FakeGRPOConfigWithGenBatch(_FakeGRPOConfig):
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    generation_batch_size: int = None


@pytest.mark.parametrize("generations, expected", [(3, 36), (5, 40), (4, None)])
def test_grpo_generation_batch_holds_whole_prompt_groups(monkeypatch, generations, expected):
    fake = types.SimpleNamespace(
        GRPOConfig = _FakeGRPOConfigWithGenBatch, GRPOTrainer = lambda **kw: kw
    )
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
        config_args = {
            "output_dir": "o",
            "max_seq_length": 512,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
        },
        settings = {"num_generations": generations},
        reward_specs = [spec],
    )
    assert kw["args"].generation_batch_size == expected


def test_grpo_generation_batch_counts_every_process(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    fake = types.SimpleNamespace(
        GRPOConfig = _FakeGRPOConfigWithGenBatch, GRPOTrainer = lambda **kw: kw
    )
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
        config_args = {
            "output_dir": "o",
            "max_seq_length": 512,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
        },
        settings = {"num_generations": 3},
        reward_specs = [spec],
    )
    # Global batch 2, three generations: the generation batch rounds up to 6.
    assert kw["args"].generation_batch_size == 6


def test_resume_refuses_a_changed_reward_set():
    from types import SimpleNamespace as S

    from routes.training import _resume_reward_mismatch, _stored_reward_specs

    stored = _stored_reward_specs(
        {
            "reward_specs": [
                {"name": "a", "rule": {"type": "length"}, "weight": 1.0},
                {"name": "b", "rule": {"type": "length"}, "weight": 0.5},
            ]
        }
    )
    assert not _resume_reward_mismatch(stored, [S(name = "a", weight = 1), S(name = "b", weight = 0.5)])
    assert _resume_reward_mismatch(stored, [S(name = "a", weight = 2.0), S(name = "b", weight = 0.5)])
    assert not _resume_reward_mismatch(stored, [S(name = "A", weight = 1), S(name = " b ", weight = 0.5)])
    assert _resume_reward_mismatch(stored, [S(name = "a", weight = 1.0)])
    assert _resume_reward_mismatch(
        stored, [S(name = "a", weight = 1.0), S(name = "b", weight = 0.5), S(name = "c", weight = 1.0)]
    )
    # Runs stored before reward specs were recorded keep resuming.
    assert not _resume_reward_mismatch({}, [S(name = "c", weight = 1.0)])


def test_grpo_refuses_a_dataset_smaller_than_one_generation_batch(monkeypatch):
    fake = types.SimpleNamespace(
        GRPOConfig = _FakeGRPOConfigWithGenBatch, GRPOTrainer = lambda **kw: kw
    )
    monkeypatch.setitem(sys.modules, "trl", fake)
    spec = {
        "name": "len",
        "rule": {"type": "length", "max_chars": 9, "score": {"over": -1.0, "under": 0.0}},
    }
    args = dict(
        model = None,
        tokenizer = None,
        eval_dataset = None,
        config_args = {
            "output_dir": "o",
            "max_seq_length": 512,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
        },
        settings = {"num_generations": 4},
        reward_specs = [spec],
    )
    with pytest.raises(ValueError, match = "at least 4 prompts"):
        build_rl_trainer("grpo", train_dataset = [{"prompt": "x"}] * 3, **args)
    assert build_rl_trainer("grpo", train_dataset = [{"prompt": "x"}] * 4, **args)


def test_grpo_epoch_rows_drop_the_partial_prompt_group():
    from core.training.trainer import grpo_completion_rows

    args = types.SimpleNamespace(num_generations = 4, generation_batch_size = 8)
    # Five prompts in groups of two: TRL trains four, so 16 completion rows.
    assert grpo_completion_rows(5, args) == 16
    assert grpo_completion_rows(6, args) == 24
    assert grpo_completion_rows(5, types.SimpleNamespace(num_generations = 4)) == 20


def test_ignored_column_is_not_auto_mapped():
    from core.training.rl import resolve_role_columns
    resolved = resolve_role_columns(["question", "answer", "system"], "grpo", {"system": "ignore"})
    assert resolved == {"prompt": "question", "answer": "answer"}


def test_long_prompts_are_dropped_when_trl_has_no_prompt_budget():
    from datasets import Dataset

    from core.training.rl import drop_long_prompts

    class _Tok:
        def __call__(
            self,
            text,
            add_special_tokens = False,
        ):
            return {"input_ids": text.split()}

        def apply_chat_template(
            self,
            messages,
            tokenize = False,
            add_generation_prompt = True,
        ):
            return " ".join(m["content"] for m in messages)

    rows = Dataset.from_list(
        [
            {"prompt": "one two three"},
            {"prompt": "one two three four five six"},
            {"prompt": "x " * 2},
        ]
    )
    assert drop_long_prompts(rows, _Tok(), 3)["prompt"] == ["one two three", "x x "]
    chat = Dataset.from_list([{"prompt": [{"role": "user", "content": "a b c d"}]}])
    with pytest.raises(ValueError, match = "prompt budget"):
        drop_long_prompts(chat, _Tok(), 3)


def test_system_turn_is_folded_for_templates_that_reject_it():
    from datasets import Dataset

    from core.training.rl import _template_takes_system, fold_system_into_user

    class _Gemma2:
        def apply_chat_template(self, messages, **_):
            if any(m["role"] == "system" for m in messages):
                raise ValueError("System role not supported")
            return "ok"

    rows = Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": "Use tags."},
                    {"role": "user", "content": "1+1?"},
                ]
            }
        ]
    )
    assert not _template_takes_system(_Gemma2())
    folded = fold_system_into_user(rows)
    assert folded[0]["prompt"] == [{"role": "user", "content": "Use tags.\n\n1+1?"}]
    assert _Gemma2().apply_chat_template(folded[0]["prompt"]) == "ok"


def test_multi_turn_preference_rows_keep_the_conversation_as_prompt():
    convo = lambda reply: [
        {"role": "user", "content": "What year is it?"},
        {"role": "assistant", "content": "2026."},
        {"role": "user", "content": "And 2+2?"},
        {"role": "assistant", "content": reply},
    ]
    ds = Dataset.from_list(
        [{"prompt": "What year is it?", "chosen": convo("4"), "rejected": convo("5")}]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["prompt"] == convo("")[:3]
    assert out[0]["chosen"] == [{"role": "assistant", "content": "4"}]
    assert out[0]["rejected"] == [{"role": "assistant", "content": "5"}]


def test_preference_branches_keep_every_turn_after_the_shared_prefix():
    start = [{"role": "user", "content": "Name a capital."}]
    branch = lambda city: start + [
        {"role": "assistant", "content": city},
        {"role": "user", "content": "Which country?"},
        {"role": "assistant", "content": "France"},
    ]
    ds = Dataset.from_list(
        [{"prompt": "Name a capital.", "chosen": branch("Paris"), "rejected": branch("London")}]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["prompt"] == start
    assert out[0]["chosen"] == branch("Paris")[1:]
    assert out[0]["rejected"] == branch("London")[1:]


def test_preference_branch_that_prefixes_the_other_keeps_its_last_turn():
    short = [
        {"role": "user", "content": "Name a capital."},
        {"role": "assistant", "content": "Paris"},
    ]
    longer = short + [
        {"role": "user", "content": "Which country?"},
        {"role": "assistant", "content": "Germany"},
    ]
    ds = Dataset.from_list([{"prompt": "Name a capital.", "chosen": short, "rejected": longer}])
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["prompt"] == short[:1]
    assert out[0]["chosen"] == short[1:]
    assert out[0]["rejected"] == longer[1:]


def test_completion_only_branches_keep_all_their_turns():
    branch = lambda city: [
        {"role": "assistant", "content": city},
        {"role": "user", "content": "Which country?"},
        {"role": "assistant", "content": "France"},
    ]
    ds = Dataset.from_list(
        [{"prompt": "Name a capital.", "chosen": branch("Paris"), "rejected": branch("London")}]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["prompt"] == [{"role": "user", "content": "Name a capital."}]
    assert out[0]["chosen"] == branch("Paris")
    assert out[0]["rejected"] == branch("London")


def test_completion_only_branches_with_a_shared_follow_up_keep_the_prompt():
    branch = lambda city: [
        {"role": "assistant", "content": "Which continent?"},
        {"role": "user", "content": "Europe."},
        {"role": "assistant", "content": city},
    ]
    ds = Dataset.from_list(
        [{"prompt": "Name a capital.", "chosen": branch("Paris"), "rejected": branch("Sydney")}]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["prompt"] == [{"role": "user", "content": "Name a capital."}]
    assert out[0]["chosen"] == branch("Paris")
    assert out[0]["rejected"] == branch("Sydney")


def test_preference_rows_keep_the_columns_trl_templates_with():
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {}}}]
    ds = Dataset.from_list(
        [
            {
                "prompt": "Weather?",
                "chosen": "Sunny.",
                "rejected": "No idea.",
                "tools": json.dumps(tools),
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["tools"] == json.dumps(tools)
    assert out[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_preference_rows_render_with_their_own_tool_schemas():
    pytest.importorskip("trl")
    from core.training.rl import render_preference_rows

    class _Tok:
        def apply_chat_template(
            self,
            messages,
            tools = None,
            tokenize = False,
            **kwargs,
        ):
            head = "".join(t["function"]["name"] + ";" for t in tools or [])
            return head + "".join(f"<{m['role']}>{m['content']}" for m in messages)

    tools = [{"type": "function", "function": {"name": "weather", "parameters": {}}}]
    ds = Dataset.from_list(
        [
            {
                "prompt": "Weather?",
                "chosen": "Sunny.",
                "rejected": "No.",
                "tools": json.dumps(tools),
            },
            {"prompt": "Hi", "chosen": "Hello", "rejected": "Go", "tools": None},
        ]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    rendered = render_preference_rows(out, _Tok())
    assert "tools" not in rendered.column_names
    assert rendered[0]["prompt"].startswith("weather;")
    assert rendered[0]["chosen"] == "<assistant>Sunny."
    assert not rendered[1]["prompt"].startswith("weather;")


def test_rows_without_their_own_template_kwargs_get_an_empty_mapping():
    ds = Dataset.from_list(
        [
            {
                "prompt": "a",
                "chosen": "b",
                "rejected": "c",
                "chat_template_kwargs": {"enable_thinking": False},
            },
            {"prompt": "a", "chosen": "b", "rejected": "c", "chat_template_kwargs": None},
        ]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert isinstance(out[1]["chat_template_kwargs"], dict)


def test_preference_rows_render_with_their_own_template_kwargs():
    pytest.importorskip("trl")
    from core.training.rl import render_preference_rows

    class _Tok:
        def apply_chat_template(
            self,
            messages,
            tools = None,
            tokenize = False,
            short = False,
            **kw,
        ):
            return ("S" if short else "L") + "".join(
                f"<{m['role']}>{m['content']}" for m in messages
            )

    ds = Dataset.from_list(
        [
            {
                "prompt": "a",
                "chosen": "b",
                "rejected": "c",
                "chat_template_kwargs": {"short": True},
            },
            {"prompt": "a", "chosen": "b", "rejected": "c", "chat_template_kwargs": None},
        ]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    rendered = render_preference_rows(out, _Tok())
    assert "chat_template_kwargs" not in rendered.column_names
    assert rendered[0]["prompt"].startswith("S") and rendered[1]["prompt"].startswith("L")


def test_template_keys_only_some_rows_set_are_not_passed_as_null():
    pytest.importorskip("trl")
    from core.training.rl import drop_long_prompts, render_preference_rows

    class _Tok:
        def __call__(
            self,
            text,
            add_special_tokens = False,
        ):
            return {"input_ids": text.split()}

        def apply_chat_template(
            self,
            messages,
            tools = None,
            tokenize = False,
            **kw,
        ):
            if "reasoning_effort" in kw:
                # GPT-OSS concatenates the value into the system turn.
                return "effort " + kw["reasoning_effort"]
            return "plain"

    ds = Dataset.from_list(
        [
            {
                "prompt": "a",
                "chosen": "b",
                "rejected": "c",
                "chat_template_kwargs": {"reasoning_effort": "low"},
            },
            {"prompt": "a", "chosen": "b", "rejected": "c"},
        ]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[1]["chat_template_kwargs"] == {"reasoning_effort": None}  # Arrow's struct fill
    rendered = render_preference_rows(out, _Tok())
    assert rendered[0]["prompt"] == "effort low" and rendered[1]["prompt"] == "plain"
    prompts = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "a"}],
                "chat_template_kwargs": {"reasoning_effort": "low"},
            },
            {"prompt": [{"role": "user", "content": "a"}], "chat_template_kwargs": {}},
        ]
    )
    assert len(drop_long_prompts(prompts, _Tok(), 16)) == 2


def test_null_message_fields_from_arrow_are_dropped():
    from core.training.rl import _as_messages
    msgs = _as_messages(
        [
            {"role": "user", "content": "hi", "tool_calls": None},
            {"role": "assistant", "content": "yo"},
        ],
        "user",
    )
    assert msgs[0] == {"role": "user", "content": "hi"}


def test_arrow_padding_from_mixed_message_rows_keeps_speaker_and_text():
    from core.training.rl import _as_messages

    padded = [{"role": "user", "content": "OpenAI question", "from": None, "value": None}]
    assert _as_messages(padded, "user") == [{"role": "user", "content": "OpenAI question"}]
    assert _as_messages({"role": None, "content": "4"}, "assistant") == [
        {"role": "assistant", "content": "4"}
    ]


def test_sharegpt_speakers_match_case_insensitively_as_in_sft():
    from core.training.rl import _split_conversation

    context, reply = _split_conversation(
        [{"from": "Human", "value": "2 + 2?"}, {"from": "GPT ", "value": "4"}]
    )
    assert context == [{"role": "user", "content": "2 + 2?"}]
    assert reply == [{"role": "assistant", "content": "4"}]


def test_the_mapped_prompts_system_message_survives_full_conversation_branches():
    system = {"role": "system", "content": "Always answer in French."}
    user = {"role": "user", "content": "Say hello."}
    branch = lambda reply: [user, {"role": "assistant", "content": reply}]
    ds = Dataset.from_list(
        [{"prompt": [system, user], "chosen": branch("Bonjour"), "rejected": branch("Hello")}]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["prompt"] == [system, user]
    assert out[0]["chosen"] == [{"role": "assistant", "content": "Bonjour"}]


def test_preference_rendering_drops_the_null_fields_arrow_adds():
    pytest.importorskip("trl")
    from core.training.rl import render_preference_rows

    class _Tok:
        def apply_chat_template(
            self,
            messages,
            tokenize = False,
            **kw,
        ):
            assert all(v is not None for m in messages for k, v in m.items() if k != "content")
            return "".join(f"<{m['role']}>{m.get('content')}" for m in messages)

    plain = [{"role": "assistant", "content": "Sunny.", "tool_calls": None}]
    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "Weather?", "tool_calls": None}],
                "chosen": plain,
                "rejected": plain,
            }
        ]
    )
    rendered = render_preference_rows(ds, _Tok())
    assert rendered[0]["chosen"] == "<assistant>Sunny."


def test_rendered_completions_end_at_eos_so_trl_adds_no_second_one():
    from core.training.rl import render_preference_rows

    class _ChatML:
        eos_token = "<|im_end|>"

        def apply_chat_template(
            self,
            messages,
            tokenize = False,
            add_generation_prompt = False,
            **kw,
        ):
            text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
            return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "2 + 2?"}],
                "chosen": [{"role": "assistant", "content": "4"}],
                "rejected": [{"role": "assistant", "content": "5"}],
            }
        ]
    )
    row = render_preference_rows(ds, _ChatML())[0]
    assert row["chosen"] == "4<|im_end|>" and row["rejected"] == "5<|im_end|>"
    # DPO before TRL 0.29 appends EOS unconditionally: the rendered one goes.
    row = render_preference_rows(ds, _ChatML(), trl_appends_eos = True)[0]
    assert row["chosen"] == "4" and row["rejected"] == "5"


def test_tool_call_arguments_keep_their_nulls():
    from core.training.rl import _without_nulls

    call = {"function": {"name": "update", "arguments": {"nickname": None}}, "id": None}
    msgs = _without_nulls([{"role": "assistant", "content": None, "tool_calls": [call]}])
    assert msgs[0]["tool_calls"][0] == {
        "function": {"name": "update", "arguments": {"nickname": None}}
    }
    assert msgs[0]["content"] is None


def test_prompts_ending_on_a_tool_result_render():
    pytest.importorskip("trl")
    from core.training.rl import render_preference_rows

    class _Tok:
        def apply_chat_template(
            self,
            messages,
            add_generation_prompt = False,
            tokenize = False,
            **kw,
        ):
            text = "".join(f"<{m['role']}>{m.get('content')}" for m in messages)
            return text + ("<assistant>" if add_generation_prompt else "")

    prompt = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "calling"},
        {"role": "tool", "content": "sunny"},
    ]
    ds = Dataset.from_list(
        [
            {
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": "Sunny."}],
                "rejected": [{"role": "assistant", "content": "No."}],
            }
        ]
    )
    rendered = render_preference_rows(ds, _Tok())
    assert rendered[0]["prompt"].endswith("<tool>sunny<assistant>")
    assert rendered[0]["chosen"] == "Sunny."


def test_structured_tool_schemas_lose_the_nulls_other_rows_add():
    from core.training.rl import _row_tools

    schema = {"type": "object", "properties": {"city": {"type": "string"}, "expression": None}}
    tools = _row_tools(
        {"tools": [{"type": "function", "function": {"name": "w", "parameters": schema}}]}
    )
    assert tools[0]["function"]["parameters"]["properties"] == {"city": {"type": "string"}}


def test_warmup_ratio_becomes_fractional_warmup_steps_when_the_config_lacks_it():
    from core.training.rl import _config_kwargs

    @dataclasses.dataclass
    class _Config:
        warmup_steps: float = 0

    assert _config_kwargs(_Config, {"warmup_ratio": 0.03}) == {"warmup_steps": 0.03}
    assert _config_kwargs(_Config, {"warmup_ratio": 0.03, "warmup_steps": 5}) == {"warmup_steps": 5}


def test_text_block_content_is_read_as_its_text():
    block = lambda text: [{"type": "text", "text": text}]
    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": block("2 + 2?")}],
                "chosen": [{"role": "assistant", "content": block("4")}],
                "rejected": [{"role": "assistant", "content": block("5")}],
            }
        ]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["prompt"][-1]["content"] == "2 + 2?"
    assert out[0]["chosen"] == [{"role": "assistant", "content": "4"}]


def test_tool_lists_of_serialized_schemas_are_decoded():
    from core.training.rl import _row_tools

    schema = {"type": "function", "function": {"name": "w", "parameters": {}}}
    assert _row_tools({"tools": [json.dumps(schema)]}) == [schema]
    assert _row_tools({"tools": json.dumps([schema])}) == [schema]
    const = {"type": "function", "function": {"name": "w", "parameters": {"const": None}}}
    assert _row_tools({"tools": json.dumps([const])}) == [const]


def test_tool_call_preferences_keep_their_calls():
    call = lambda city: [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": "weather", "arguments": {"city": city}}}
            ],
        }
    ]
    ds = Dataset.from_list(
        [{"prompt": "Weather?", "chosen": call("Paris"), "rejected": call("London")}]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["chosen"] != out[0]["rejected"]
    assert out[0]["chosen"][0]["tool_calls"][0]["function"]["arguments"]["city"] == "Paris"


def test_tool_call_preferences_without_content_keep_their_calls():
    call = lambda city: [
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": "weather", "arguments": {"city": city}}}
            ],
        }
    ]
    ds = Dataset.from_list(
        [{"prompt": "Weather?", "chosen": call("Paris"), "rejected": call("London")}]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["chosen"][0]["content"] is None
    assert out[0]["chosen"][0]["tool_calls"][0]["function"]["arguments"]["city"] == "Paris"


def test_prompt_budget_renders_with_the_rows_template_kwargs():
    from core.training.rl import drop_long_prompts

    class _Tok:
        def __call__(
            self,
            text,
            add_special_tokens = False,
        ):
            return {"input_ids": text.split()}

        def apply_chat_template(
            self,
            messages,
            tools = None,
            short = False,
            **kwargs,
        ):
            text = " ".join(m["content"] for m in messages)
            return text if short else text + " pad" * 40

    rows = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "a b"}], "chat_template_kwargs": {"short": True}}]
    )
    assert len(drop_long_prompts(rows, _Tok(), 16)) == 1


def test_grpo_drops_long_prompts_even_when_trl_keeps_the_field(monkeypatch):
    from datasets import Dataset

    class _Tok:
        chat_template = "x"

        def __call__(
            self,
            text,
            add_special_tokens = False,
        ):
            return {"input_ids": text.split()}

        def apply_chat_template(
            self,
            messages,
            tokenize = False,
            add_generation_prompt = True,
            **kwargs,
        ):
            return " ".join(m["content"] for m in messages)

    fake = types.SimpleNamespace(GRPOConfig = _FakeGRPOConfig, GRPOTrainer = lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "trl", fake)
    rows = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": c}]} for c in ("a b", "w " * 20)]
    )
    kw = build_rl_trainer(
        "grpo",
        model = None,
        tokenizer = _Tok(),
        train_dataset = rows,
        eval_dataset = None,
        config_args = {"output_dir": "o", "max_seq_length": 1024},
        settings = {"num_generations": 4, "max_prompt_length": 16},
        reward_specs = [
            {
                "name": "len",
                "rule": {"type": "length", "max_chars": 9, "score": {"over": -1.0, "under": 0.0}},
            }
        ],
    )
    assert [r["prompt"][0]["content"] for r in kw["train_dataset"]] == ["a b"]


def test_null_answers_stay_missing_so_empty_replies_earn_nothing():
    from core.training.rewards import get_reward, make_reward_func

    ds = Dataset.from_list(
        [{"question": "q", "answer": None}, {"question": "q2", "answer": "#### 7"}]
    )
    out, _ = format_rl_dataset(ds, "grpo")
    assert out[0]["answer"] is None
    reward = make_reward_func(get_reward("exact-answer"))
    assert reward(completions = ["<answer></answer>"], answer = [out[0]["answer"]]) == [0.0]


def test_a_mapped_prompt_opening_with_an_assistant_greeting_is_stripped_from_both_branches():
    greet = {"role": "assistant", "content": "Welcome!"}
    ask = {"role": "user", "content": "2+2?"}
    reply = lambda text: [greet, ask, {"role": "assistant", "content": text}]
    ds = Dataset.from_list([{"prompt": [greet, ask], "chosen": reply("4"), "rejected": reply("5")}])
    out, _ = format_rl_dataset(ds, "dpo")
    assert out[0]["prompt"] == [greet, ask]
    assert out[0]["chosen"] == [{"role": "assistant", "content": "4"}]
    assert out[0]["rejected"] == [{"role": "assistant", "content": "5"}]


def test_grpo_prompts_with_arrow_filled_null_fields_render_without_them():
    from datasets import Dataset

    from core.training.rl import drop_long_prompts, render_grpo_prompts

    class _Tok:
        chat_template = "{{ m.thinking }}"

        def __call__(
            self,
            text,
            add_special_tokens = False,
        ):
            return {"input_ids": text.split()}

        def apply_chat_template(
            self,
            messages,
            tokenize = False,
            add_generation_prompt = False,
            **kw,
        ):
            # GPT-OSS's template tests `"x" in message.thinking` whenever the key exists.
            return " ".join(
                m["content"] + ("" if "thinking" not in m else " t:" + m["thinking"])
                for m in messages
            )

    turn = lambda **extra: {"role": "assistant", "content": "a", **extra}
    ds = Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "user", "content": "q"},
                    turn(thinking = "hm"),
                    {"role": "user", "content": "q2"},
                ]
            },
            {
                "prompt": [
                    {"role": "user", "content": "q"},
                    turn(),
                    {"role": "user", "content": "q2"},
                ]
            },
        ]
    )
    assert ds[1]["prompt"][1]["thinking"] is None  # Arrow's struct fill
    assert len(drop_long_prompts(ds, _Tok(), 64)) == 2
    out, rendered = render_grpo_prompts(ds, _Tok(), None)
    assert rendered and out[0]["prompt"] == "q a t:hm q2" and out[1]["prompt"] == "q a q2"
    clean = Dataset.from_list([{"prompt": [{"role": "user", "content": "q"}]}])
    assert render_grpo_prompts(clean, _Tok(), None) == (clean, False)


def test_alpaca_instruction_and_input_both_reach_the_prompt():
    ds = Dataset.from_list(
        [
            {
                "instruction": "Identify the odd one out.",
                "input": "Twitter, Instagram, Telegram",
                "chosen": "Telegram",
                "rejected": "Twitter",
            },
            {"instruction": "Say hi.", "input": "", "chosen": "hi", "rejected": "no"},
        ]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    assert (
        out[0]["prompt"][-1]["content"]
        == "Identify the odd one out.\n\nTwitter, Instagram, Telegram"
    )
    assert out[1]["prompt"][-1]["content"] == "Say hi."
    ignored, _ = format_rl_dataset(ds, "dpo", mapping = {"input": "ignore"})
    assert ignored[0]["prompt"][-1]["content"] == "Identify the odd one out."


def test_sharegpt_turns_and_single_message_cells_become_messages():
    ds = Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "system", "value": "Be brief."},
                    {"from": "human", "value": "2+2?"},
                ],
                "chosen": {"from": "gpt", "value": "4"},
                "rejected": {"from": "gpt", "value": "5"},
            }
        ]
    )
    out, _ = format_rl_dataset(ds, "dpo", mapping = {"conversations": "prompt"})
    assert out[0]["prompt"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "2+2?"},
    ]
    assert out[0]["chosen"] == [{"role": "assistant", "content": "4"}]
    assert out[0]["rejected"] == [{"role": "assistant", "content": "5"}]


def test_sharegpt_function_calls_become_tool_calls_not_text():
    call = '{"name": "weather", "arguments": {"city": "Paris"}}'
    ds = Dataset.from_list(
        [
            {
                "prompt": [{"from": "human", "value": "Weather in Paris?"}],
                "chosen": [{"from": "function_call", "value": call}],
                "rejected": [{"from": "gpt", "value": "No idea."}],
            }
        ]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    chosen = out[0]["chosen"][0]
    assert chosen["role"] == "assistant"
    assert chosen["tool_calls"][0]["function"]["name"] == "weather"
    assert json.loads(chosen["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}


def test_tool_ended_grpo_prompts_are_rendered_for_trl():
    from datasets import Dataset

    from core.training.rl import render_grpo_prompts

    class _Tok:
        chat_template = "plain"

        def apply_chat_template(
            self,
            messages,
            tokenize = False,
            add_generation_prompt = False,
            **kw,
        ):
            return "|".join(f"{m['role']}:{m['content']}" for m in messages) + (
                "|assistant:" if add_generation_prompt else ""
            )

    ds = Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "user", "content": "w?"},
                    {"role": "assistant", "content": "call"},
                    {"role": "tool", "content": "sunny"},
                ]
            }
        ]
    )
    out, rendered = render_grpo_prompts(ds, _Tok(), None)
    assert rendered and out[0]["prompt"] == "user:w?|assistant:call|tool:sunny|assistant:"


def test_tool_ended_preference_rows_never_repeat_the_prompt_in_a_completion():
    pytest.importorskip("trl")
    from core.training.rl import render_preference_rows

    class _Tok:
        def apply_chat_template(
            self,
            messages,
            tools = None,
            tokenize = False,
            add_generation_prompt = False,
            **kw,
        ):
            text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
            # Like Qwen3 with thinking off: the generation header carries an empty think block.
            return text + ("<assistant><think></think>" if add_generation_prompt else "")

    prompt = [{"role": "user", "content": "w?"}, {"role": "tool", "content": "sunny"}]
    ds = Dataset.from_list(
        [
            {
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": "Sunny."}],
                "rejected": [{"role": "assistant", "content": "Rain."}],
            }
        ]
    )
    row = render_preference_rows(ds, _Tok())[0]
    assert "w?" not in row["chosen"] and "w?" not in row["rejected"]
    assert row["prompt"] + row["chosen"] == "<user>w?<tool>sunny<assistant>Sunny."


def test_tool_ended_preference_prompt_never_takes_shared_reply_text():
    pytest.importorskip("trl")
    from core.training.rl import render_preference_rows

    class _Tok:
        def apply_chat_template(
            self,
            messages,
            tools = None,
            tokenize = False,
            add_generation_prompt = False,
            **kw,
        ):
            text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
            return text + ("<assistant><think></think>" if add_generation_prompt else "")

    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "w?"}, {"role": "tool", "content": "sunny"}],
                "chosen": [{"role": "assistant", "content": "It is sunny in Paris."}],
                "rejected": [{"role": "assistant", "content": "It is sunny in London."}],
            }
        ]
    )
    row = render_preference_rows(ds, _Tok())[0]
    assert row["prompt"] == "<user>w?<tool>sunny<assistant>"
    assert row["chosen"] == "It is sunny in Paris."


def test_preference_tool_call_arguments_render_as_mappings():
    pytest.importorskip("trl")
    from core.training.rl import render_preference_rows

    seen = []

    class _Tok:
        def apply_chat_template(
            self,
            messages,
            tools = None,
            tokenize = False,
            add_generation_prompt = False,
            **kw,
        ):
            for m in messages:
                for call in m.get("tool_calls") or []:
                    seen.append(call["function"]["arguments"])
            return "".join(f"<{m['role']}>{m.get('content')}" for m in messages) + (
                "<assistant>" if add_generation_prompt else ""
            )

    call = {"type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}
    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "w?"}],
                "chosen": [{"role": "assistant", "content": "", "tool_calls": [call]}],
                "rejected": [{"role": "assistant", "content": "No idea."}],
            }
        ]
    )
    render_preference_rows(ds, _Tok())
    assert seen and all(args == {"city": "Paris"} for args in seen)


def test_preference_rows_slice_both_branches_from_one_prompt():
    from core.training.rl import render_preference_rows

    class _Tok:
        # Qwen3 with thinking off: an empty think block in the header, a filled one on reasoning.
        def apply_chat_template(
            self,
            messages,
            tools = None,
            tokenize = False,
            add_generation_prompt = False,
            continue_final_message = False,
            **kw,
        ):
            out = ""
            for m in messages:
                if m["role"] == "assistant":
                    out += f"<a><think>{m.get('reasoning_content') or ''}</think>{m['content']}"
                else:
                    out += f"<{m['role']}>{m['content']}"
            return out + ("<a><think></think>" if add_generation_prompt else "")

    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "2+2?"}],
                "chosen": [{"role": "assistant", "content": "4"}],
                "rejected": [{"role": "assistant", "content": "5", "reasoning_content": "hm"}],
            }
        ]
    )
    row = render_preference_rows(ds, _Tok())[0]
    assert row["prompt"] + row["chosen"] == "<user>2+2?<a><think></think>4"
    assert row["prompt"] + row["rejected"] == "<user>2+2?<a><think>hm</think>5"


def test_full_conversation_pairs_with_structured_tool_arguments_keep_their_shared_context():
    def call(name, args):
        return {"type": "function", "function": {"name": name, "arguments": args}}

    context = [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [call("weather", {"city": "Paris"})]},
        {"role": "tool", "content": "sunny"},
        {"role": "user", "content": "And then?"},
    ]
    ds = Dataset.from_list(
        [
            {
                "prompt": "Weather in Paris?",
                "chosen": context
                + [{"role": "assistant", "content": "", "tool_calls": [call("f", {"x": 1})]}],
                "rejected": context
                + [{"role": "assistant", "content": "", "tool_calls": [call("f", {"y": 2})]}],
            }
        ]
    )
    out, _ = format_rl_dataset(ds, "dpo")
    row = out[0]
    assert [m["role"] for m in row["prompt"]] == ["user", "assistant", "tool", "user"]
    assert len(row["chosen"]) == 1 and len(row["rejected"]) == 1


def test_grpo_prompts_carry_each_rows_tool_schemas():
    from core.training.rl import render_grpo_prompts

    schema = {"type": "function", "function": {"name": "weather", "parameters": {}}}
    ds = Dataset.from_list(
        [
            {"question": "Weather in Paris?", "answer": "sunny", "tools": [json.dumps(schema)]},
            {"question": "2+2?", "answer": "4", "tools": []},
        ]
    )
    out, _ = format_rl_dataset(ds, "grpo")
    assert "tools" in out.column_names

    class _Tok:
        chat_template = "plain"

        def apply_chat_template(
            self,
            messages,
            tokenize = False,
            add_generation_prompt = False,
            tools = None,
            **kw,
        ):
            names = ",".join(t["function"]["name"] for t in tools or [])
            return f"[{names}]" + "|".join(m["content"] for m in messages)

    rendered, done = render_grpo_prompts(out, _Tok(), None)
    assert done and "tools" not in rendered.column_names
    assert rendered[0]["prompt"] == "[weather]Weather in Paris?"
    assert rendered[1]["prompt"] == "[]2+2?"


def test_grpo_prefill_prompts_are_continued_and_scored_with_their_prefill():
    from core.training.rewards import make_reward_func, validate_rule
    from core.training.rl import render_grpo_prompts

    class _Tok:
        chat_template = "{{ enable_thinking }}"

        def apply_chat_template(
            self,
            messages,
            tokenize = False,
            add_generation_prompt = False,
            continue_final_message = False,
            **kw,
        ):
            text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
            if continue_final_message:
                return text
            return text + ("<end><assistant>" if add_generation_prompt else "<end>")

    ds = Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "user", "content": "2+2?"},
                    {"role": "assistant", "content": "<answer>"},
                ],
                "answer": "4",
            },
            {"prompt": [{"role": "user", "content": "3+3?"}], "answer": "6"},
        ]
    )
    out, done = render_grpo_prompts(ds, _Tok(), False)
    assert done and out[0]["prompt"] == "<user>2+2?<assistant><answer>"
    assert out[1]["prompt"] == "<user>3+3?<end><assistant>"
    reward = make_reward_func(
        {
            "name": "exact",
            "rule": validate_rule(
                {
                    "type": "exact_match",
                    "compare_to": "answer",
                    "extract": {"between": ["<answer>", "</answer>"]},
                    "normalize": ["strip"],
                    "score": {"match": 2, "miss": 0},
                    "missing": 0,
                }
            ),
        }
    )
    scores = reward(
        prompts = list(out["prompt"]),
        completions = ["4</answer>", "<answer>6</answer>"],
        answer = list(out["answer"]),
        _studio_prefill = list(out["_studio_prefill"]),
    )
    assert scores == [2.0, 2.0]


def test_prefill_prompts_are_measured_as_continued():
    from core.training.rl import drop_long_prompts

    class _Tok:
        def __call__(
            self,
            text,
            add_special_tokens = False,
        ):
            return {"input_ids": text.split()}

        def apply_chat_template(
            self,
            messages,
            tokenize = False,
            add_generation_prompt = False,
            continue_final_message = False,
            **kw,
        ):
            text = " ".join(m["content"] for m in messages)
            return text + (" HEADER" if add_generation_prompt else "")

    rows = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "a b"}, {"role": "assistant", "content": "c"}]}]
    )
    assert len(drop_long_prompts(rows, _Tok(), 3)) == 1


def test_full_conversations_opening_on_a_greeting_keep_their_context_despite_arrow_padding():
    def call(name, args):
        return {"type": "function", "function": {"name": name, "arguments": args}}

    context = [
        {"role": "assistant", "content": "Hi, how can I help?"},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [call("weather", {"city": "Paris"})]},
        {"role": "tool", "content": "sunny"},
        {"role": "user", "content": "And then?"},
    ]
    ds = Dataset.from_list(
        [
            {
                "prompt": context,
                "chosen": context
                + [{"role": "assistant", "content": "", "tool_calls": [call("f", {"x": 1})]}],
                "rejected": context
                + [{"role": "assistant", "content": "", "tool_calls": [call("f", {"y": 2})]}],
            }
        ]
    )
    for objective in ("dpo", "orpo"):
        row = format_rl_dataset(ds, objective)[0][0]
        assert len(row["chosen"]) == 1 and len(row["rejected"]) == 1


def test_a_reward_comparing_to_a_prefill_named_column_scores_the_plain_completion():
    from core.training.rewards import make_reward_func, validate_rule

    rule = validate_rule(
        {"type": "exact_match", "compare_to": "prefill", "score": {"match": 2, "miss": -1}}
    )
    reward = make_reward_func({"name": "exact", "rule": rule})
    assert reward(completions = ["OK"], prefill = ["OK"]) == [2.0]


def test_preference_rows_drop_arrow_padding_from_structured_tool_arguments():
    pytest.importorskip("trl")
    from core.training.rl import render_preference_rows

    seen = []

    class _Tok:
        chat_template = "plain"

        def apply_chat_template(
            self,
            messages,
            tools = None,
            tokenize = False,
            **kw,
        ):
            for m in messages:
                for call in m.get("tool_calls") or []:
                    seen.append(call["function"]["arguments"])
            return "".join(f"<{m['role']}>{m.get('content')}" for m in messages)

    def call(name, args):
        return {"type": "function", "function": {"name": name, "arguments": args}}

    def schema(name, *props):
        return {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {p: {} for p in props}},
            },
        }

    tools = [schema("weather", "city", "units"), schema("calc", "expression")]

    def row(chosen):
        return {
            "tools": json.dumps(tools),
            "prompt": [{"role": "user", "content": "w?"}],
            "chosen": [{"role": "assistant", "content": "", "tool_calls": [chosen]}],
            "rejected": [{"role": "assistant", "content": "No."}],
        }

    # Two rows calling different tools: Arrow pads each row's arguments with the other's keys.
    ds = Dataset.from_list(
        [
            row(call("weather", {"city": "Paris"})),
            row(call("calc", {"expression": "1+1"})),
        ]
    )
    render_preference_rows(ds, _Tok())
    assert {"city": "Paris"} in seen and {"expression": "1+1"} in seen


def test_declared_nullable_tool_arguments_keep_their_null():
    from core.training.rl import _without_padded_arguments

    tools = [
        {
            "type": "function",
            "function": {"name": "update", "parameters": {"properties": {"nickname": {}}}},
        }
    ]
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "update", "arguments": {"nickname": None, "x": None}}}
            ],
        }
    ]
    assert _without_padded_arguments(msgs, tools)[0]["tool_calls"][0]["function"]["arguments"] == {
        "nickname": None
    }


def test_padded_null_is_dropped_where_the_schema_does_not_allow_null():
    from core.training.rl import _without_padded_arguments

    props = {
        "city": {"type": "string"},
        "units": {"type": "string"},
        "note": {"type": ["string", "null"]},
        "tag": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    }
    tools = [
        {"type": "function", "function": {"name": "weather", "parameters": {"properties": props}}}
    ]
    args = {"city": "Paris", "units": None, "note": None, "tag": None}
    msgs = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "weather", "arguments": args}}]}
    ]
    assert _without_padded_arguments(msgs, tools)[0]["tool_calls"][0]["function"]["arguments"] == {
        "city": "Paris",
        "note": None,
        "tag": None,
    }


def test_preference_rows_render_a_thinking_field_as_reasoning():
    pytest.importorskip("trl")
    from core.training.rl import render_preference_rows

    class _Tok:
        chat_template = "{{ message.reasoning_content }}"

        def apply_chat_template(
            self,
            messages,
            tools = None,
            tokenize = False,
            add_generation_prompt = False,
            **kw,
        ):
            out = ""
            for m in messages:
                out += f"<{m['role']}>"
                if m.get("reasoning_content"):
                    out += f"<think>{m['reasoning_content']}</think>"
                out += m["content"]
            return out + ("<assistant>" if add_generation_prompt else "")

    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "2+2?"}],
                "chosen": [{"role": "assistant", "content": "4", "thinking": "2 and 2 is 4"}],
                "rejected": [{"role": "assistant", "content": "4", "thinking": "guess"}],
            }
        ]
    )
    row = render_preference_rows(ds, _Tok())[0]
    assert row["chosen"] != row["rejected"]
    assert "2 and 2 is 4" in row["chosen"] and "guess" in row["rejected"]


def test_grpo_keeps_a_zero_kl_coefficient():
    request = TrainingStartRequest(
        **{**BASE, "objective": "grpo", "grpo_rewards": [{"name": "a"}], "rl_beta": 0}
    )
    assert request.rl_beta == 0
