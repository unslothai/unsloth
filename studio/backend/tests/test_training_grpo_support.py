# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GRPO as a fourth training method: reward presets, GRPOConfig construction,
prompt-only dataset preparation and the request schema's GRPO validation."""

import contextlib
import importlib
import sys
import types
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from core.training.grpo_rewards import (
    DEFAULT_REWARD_PRESET_IDS,
    REWARD_PRESETS_BY_ID,
    build_reward_functions,
    exact_answer_match,
    reasoning_format_match,
    response_length,
    reward_preset_catalog,
    think_tag_structure,
)
from models.training import TrainingStartRequest
from utils.datasets.prompt_only import (
    detect_prompt_only_format,
    prepare_prompt_only_dataset,
)

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Stub a dep the CPU backend CI job does not install (see test_training_preflight)."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod


_STUB_SPECS = (
    ("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported")),
    ("unsloth.chat_templates", ("get_chat_template",)),
    ("trl", ("SFTTrainer", "SFTConfig", "GRPOTrainer", "GRPOConfig")),
)


@contextlib.contextmanager
def _stubbed():
    """Hold the stubs across the trainer import, then drop them again.

    Leaving them in sys.modules would make the rest of the suite run against them;
    see the same helper in test_training_preflight.py.
    """
    for name, attrs in _STUB_SPECS:
        _stub_if_missing(name, attrs)
    try:
        yield
    finally:
        while _STUBBED:
            sys.modules.pop(_STUBBED.pop(), None)


with _stubbed():
    from core.training import trainer as trainer_module  # noqa: E402


def _base_request(**overrides) -> dict:
    payload = {
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "training_type": "GRPO",
        "format_type": "auto",
        "hf_dataset": "openai/gsm8k",
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_epochs": 1,
        "max_seq_length": 1024,
    }
    payload.update(overrides)
    return payload


class TestRewardPresets(unittest.TestCase):
    def test_catalog_exposes_metadata_for_every_preset(self):
        catalog = reward_preset_catalog()
        self.assertEqual(len(catalog), len(REWARD_PRESETS_BY_ID))
        for entry in catalog:
            self.assertIn(entry["id"], REWARD_PRESETS_BY_ID)
            self.assertTrue(entry["name"])
            self.assertTrue(entry["description"])
            self.assertGreater(entry["default_weight"], 0)
            self.assertIsInstance(entry["expected_columns"], list)
        selected = {entry["id"] for entry in catalog if entry["default_selected"]}
        self.assertEqual(selected, set(DEFAULT_REWARD_PRESET_IDS))

    def test_exact_answer_match_scores_exact_then_numeric(self):
        completions = [
            "<think>x</think><answer>72</answer>",
            "the total is 72 dollars",
            "<think>x</think><answer>13</answer>",
        ]
        scores = exact_answer_match(["p"] * 3, completions, answer = ["72", "72", "72"])
        self.assertEqual(scores, [1.0, 0.5, 0.0])

    def test_exact_answer_match_handles_conversational_completions(self):
        completions = [[{"role": "assistant", "content": "<answer>7</answer>"}]]
        self.assertEqual(exact_answer_match(["p"], completions, answer = ["7"]), [1.0])

    def test_exact_answer_match_without_reference_column_scores_zero(self):
        self.assertEqual(exact_answer_match(["p"], ["<answer>7</answer>"]), [0.0])

    def test_reasoning_format_match_is_all_or_nothing(self):
        scores = reasoning_format_match(
            ["p", "p"],
            [
                "<think>reason</think><answer>7</answer>",
                "sure! <think>reason</think><answer>7</answer> hope that helps",
            ],
        )
        self.assertEqual(scores, [1.0, 0.0])

    def test_think_tag_structure_gives_partial_credit(self):
        scores = think_tag_structure(
            ["p", "p", "p"],
            [
                "<think>a</think><answer>b</answer>",
                "<think>a</think>",
                "no tags at all",
            ],
        )
        self.assertEqual(scores, [1.0, 0.5, 0.0])

    def test_response_length_peaks_at_target(self):
        scores = response_length(["p", "p"], ["x" * 512, "x"])
        self.assertEqual(scores[0], 1.0)
        self.assertLess(scores[1], 0.1)

    def test_build_reward_functions_defaults_when_nothing_selected(self):
        functions, names = build_reward_functions(None)
        self.assertEqual(len(functions), len(DEFAULT_REWARD_PRESET_IDS))
        self.assertEqual(
            names,
            [REWARD_PRESETS_BY_ID[i].function.__name__ for i in DEFAULT_REWARD_PRESET_IDS],
        )

    def test_build_reward_functions_applies_weights_and_keeps_names(self):
        functions, names = build_reward_functions([{"id": "reasoning_format_match", "weight": 3.0}])
        self.assertEqual(names, ["reasoning_format_match"])
        self.assertEqual(functions[0].__name__, "reasoning_format_match")
        self.assertEqual(functions[0](["p"], ["<think>a</think><answer>b</answer>"]), [3.0])

    def test_build_reward_functions_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            build_reward_functions([{"id": "not_a_preset"}])
        with self.assertRaises(ValueError):
            build_reward_functions([{"id": "response_length", "weight": 0}])
        with self.assertRaises(ValueError):
            build_reward_functions([{"id": "response_length"}, {"id": "response_length"}])


class TestPromptOnlyDataset(unittest.TestCase):
    def test_detects_prompt_and_answer_aliases(self):
        dataset = Dataset.from_dict({"question": ["2+2?"], "answer": ["4"]})
        detected = detect_prompt_only_format(dataset)
        self.assertEqual(detected["prompt_column"], "question")
        self.assertEqual(detected["answer_column"], "answer")

    def test_user_mapping_wins_over_aliases(self):
        dataset = Dataset.from_dict({"question": ["a"], "task": ["b"], "answer": ["c"]})
        detected = detect_prompt_only_format(dataset, {"task": "user", "answer": "assistant"})
        self.assertEqual(detected["prompt_column"], "task")
        self.assertEqual(detected["answer_column"], "answer")

    def test_prepare_produces_conversational_prompt_and_answer_only(self):
        dataset = Dataset.from_dict({"instruction": ["2+2?"], "output": ["4"], "extra": ["x"]})
        result = prepare_prompt_only_dataset(dataset, system_prompt = "Think first.")
        self.assertEqual(sorted(result.dataset.column_names), ["answer", "prompt"])
        row = result.dataset[0]
        self.assertEqual(
            row["prompt"],
            [
                {"role": "system", "content": "Think first."},
                {"role": "user", "content": "2+2?"},
            ],
        )
        self.assertEqual(row["answer"], "4")

    def test_prepare_supports_plain_text_prompts(self):
        dataset = Dataset.from_dict({"prompt": ["2+2?"]})
        result = prepare_prompt_only_dataset(dataset, conversational = False)
        self.assertEqual(result.dataset[0]["prompt"], "2+2?")
        self.assertEqual(result.dataset.column_names, ["prompt"])

    def test_missing_reference_answer_warns_but_still_trains(self):
        dataset = Dataset.from_dict({"prompt": ["2+2?"]})
        result = prepare_prompt_only_dataset(dataset)
        self.assertIsNone(result.answer_column)
        self.assertTrue(any(n.level == "warning" for n in result.notices))

    def test_dataset_without_a_prompt_column_is_rejected(self):
        dataset = Dataset.from_dict({"foo": ["a"], "bar": ["b"]})
        with self.assertRaises(ValueError):
            prepare_prompt_only_dataset(dataset)


class TestGrpoStartRequest(unittest.TestCase):
    def test_grpo_is_an_accepted_training_type_with_notebook_defaults(self):
        request = TrainingStartRequest(**_base_request())
        self.assertEqual(request.training_type, "GRPO")
        self.assertEqual(request.num_generations, 4)
        self.assertEqual(request.max_prompt_length, 256)
        self.assertEqual(request.max_completion_length, 512)
        self.assertEqual(request.grpo_beta, 0.04)
        self.assertIsNone(request.grpo_loss_type)
        self.assertEqual(request.reward_functions, [])

    def test_reward_function_ids_are_validated_against_the_registry(self):
        request = TrainingStartRequest(
            **_base_request(reward_functions = [{"id": "response_length", "weight": 2}])
        )
        self.assertEqual(request.reward_functions[0].id, "response_length")
        with pytest.raises(ValueError):
            TrainingStartRequest(**_base_request(reward_functions = [{"id": "nope"}]))

    def test_duplicate_reward_functions_are_rejected(self):
        with pytest.raises(ValueError):
            TrainingStartRequest(
                **_base_request(
                    reward_functions = [{"id": "response_length"}, {"id": "response_length"}]
                )
            )

    def test_num_generations_must_divide_the_effective_batch(self):
        with pytest.raises(ValueError):
            TrainingStartRequest(**_base_request(batch_size = 1, gradient_accumulation_steps = 3))

    def test_rollout_lengths_must_fit_in_max_seq_length(self):
        with pytest.raises(ValueError):
            TrainingStartRequest(
                **_base_request(
                    max_seq_length = 512, max_prompt_length = 256, max_completion_length = 512
                )
            )

    def test_sft_runs_skip_the_grpo_checks(self):
        request = TrainingStartRequest(
            **_base_request(training_type = "LoRA/QLoRA", gradient_accumulation_steps = 3)
        )
        self.assertEqual(request.training_type, "LoRA/QLoRA")


@dataclass
class _FakeGRPOConfig:
    """Stands in for trl.GRPOConfig: only the fields the branch may pass."""

    output_dir: str = ""
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 3.0
    max_steps: int = -1
    learning_rate: float = 1e-6
    logging_steps: int = 500
    seed: int = 42
    weight_decay: float = 0.0
    report_to: list = field(default_factory = list)
    num_generations: int = 8
    max_prompt_length: int = 512
    max_completion_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    beta: float = 0.04
    loss_type: str = "bnpo"
    use_vllm: bool = False


class TestGrpoConfigArgs(unittest.TestCase):
    """_build_grpo_config_args filters the shared SFT config down to GRPOConfig's fields."""

    def setUp(self):
        self.build = trainer_module.UnslothTrainer._build_grpo_config_args

    def _build(self, training_args, config_args):
        fake_trl = types.ModuleType("trl")
        fake_trl.GRPOConfig = _FakeGRPOConfig
        original = sys.modules.get("trl")
        sys.modules["trl"] = fake_trl
        try:
            return self.build(object(), training_args, "/tmp/out", config_args)
        finally:
            if original is None:
                sys.modules.pop("trl", None)
            else:
                sys.modules["trl"] = original

    def test_drops_sft_only_keys_and_carries_rollout_settings(self):
        config_args = {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 1,
            "learning_rate": 5e-6,
            "logging_steps": 1,
            "seed": 3407,
            # SFT-only: GRPOConfig has no such fields.
            "dataset_text_field": "text",
            "packing": True,
            "max_seq_length": 1024,
            "dataset_num_proc": 4,
        }
        training_args = {
            "num_generations": 4,
            "max_prompt_length": 128,
            "max_completion_length": 256,
            "grpo_temperature": 0.9,
            "grpo_top_p": 0.95,
            "grpo_beta": 0.0,
            "grpo_loss_type": "dr_grpo",
        }
        built = self._build(training_args, config_args)

        for dropped in ("dataset_text_field", "packing", "max_seq_length", "dataset_num_proc"):
            self.assertNotIn(dropped, built)
        self.assertEqual(built["per_device_train_batch_size"], 1)
        self.assertEqual(built["num_generations"], 4)
        self.assertEqual(built["max_prompt_length"], 128)
        self.assertEqual(built["max_completion_length"], 256)
        self.assertEqual(built["temperature"], 0.9)
        self.assertEqual(built["top_p"], 0.95)
        self.assertEqual(built["beta"], 0.0)
        self.assertEqual(built["loss_type"], "dr_grpo")
        self.assertEqual(built["output_dir"], "/tmp/out")
        # HF generate for rollouts; vLLM is deliberately out of scope.
        self.assertIs(built["use_vllm"], False)

    def test_loss_type_is_left_to_trl_when_unset(self):
        built = self._build({}, {"learning_rate": 1e-5})
        self.assertNotIn("loss_type", built)
        self.assertEqual(built["num_generations"], 4)


class TestGrpoLogMetrics(unittest.TestCase):
    def setUp(self):
        self.extract = trainer_module._grpo_metrics_from_logs

    def test_reads_reward_kl_and_completion_length(self):
        metrics = self.extract(
            {
                "loss": 0.1,
                "reward": 1.75,
                "reward_std": 0.5,
                "kl": 0.02,
                "completion_length": 180.0,
                "rewards/exact_answer_match/mean": 1.5,
                "rewards/response_length/std": 0.9,
            }
        )
        self.assertEqual(metrics["reward"], 1.75)
        self.assertEqual(metrics["reward_std"], 0.5)
        self.assertEqual(metrics["kl"], 0.02)
        self.assertEqual(metrics["completion_length"], 180.0)
        self.assertEqual(metrics["reward_breakdown"], {"exact_answer_match": 1.5})

    def test_accepts_the_newer_trl_key_spellings(self):
        metrics = self.extract({"completions/mean_length": 200.0, "objective/kl": 0.5})
        self.assertEqual(metrics["completion_length"], 200.0)
        self.assertEqual(metrics["kl"], 0.5)

    def test_sft_logs_yield_no_grpo_signal(self):
        metrics = self.extract({"loss": 0.2, "grad_norm": 1.0})
        self.assertEqual(
            metrics,
            {
                "reward": None,
                "reward_std": None,
                "kl": None,
                "completion_length": None,
                "reward_breakdown": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
