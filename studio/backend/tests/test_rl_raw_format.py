# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402
import json  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - stub unusable imports
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.training import trainer as tmod  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


class _Tokenizer:
    chat_template = "{{ messages }}"
    eos_token = "</s>"


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    monkeypatch.setattr(tmod, "ensure_audio_decoding", lambda: True)
    t = tmod.UnslothTrainer()
    t.model_name = "Qwen2ForCausalLM"
    t.tokenizer = _Tokenizer()
    return t


def test_rl_ignores_a_raw_text_format_setting(trainer, tmp_path):
    rows = [
        {"prompt": f"What is {i} plus {i}?", "chosen": str(2 * i), "rejected": str(2 * i + 1)}
        for i in range(4)
    ]
    path = tmp_path / "pref.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")

    dataset_info, _ = trainer.load_and_format_dataset(
        None, local_datasets = [str(path)], format_type = "raw", objective = "dpo"
    )

    assert dataset_info.get("final_format") != "raw_text"
    assert {"prompt", "chosen", "rejected"} <= set(dataset_info["dataset"].column_names)


def test_rl_is_refused_for_a_text_diffusion_model(trainer, tmp_path):
    path = tmp_path / "pref.jsonl"
    path.write_text(
        json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n", encoding = "utf-8"
    )
    trainer.diffusion_lm_profile = object()
    assert (
        trainer.load_and_format_dataset(None, local_datasets = [str(path)], objective = "orpo") is None
    )
    assert "text-diffusion" in (trainer.training_progress.error or "")


def test_preference_objectives_are_refused_for_a_vision_language_model(trainer, tmp_path):
    path = tmp_path / "pref.jsonl"
    path.write_text(
        json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n", encoding = "utf-8"
    )
    trainer.is_vlm = True
    assert (
        trainer.load_and_format_dataset(None, local_datasets = [str(path)], objective = "dpo") is None
    )
    assert "vision-language" in (trainer.training_progress.error or "")


def test_preference_objectives_are_refused_for_a_vision_model_on_text_data(trainer, tmp_path):
    path = tmp_path / "pref.jsonl"
    path.write_text(
        json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n", encoding = "utf-8"
    )
    trainer.is_vlm = False
    trainer.model_has_vision = True
    assert (
        trainer.load_and_format_dataset(None, local_datasets = [str(path)], objective = "orpo") is None
    )
    assert "vision-language" in (trainer.training_progress.error or "")


def test_grpo_runs_without_loss_based_early_stopping(trainer):
    trainer.training_objective = "grpo"
    trainer.trainer = object()
    trainer._attach_early_stopping({"early_stopping_patience": 2})
    assert trainer._early_stopping is None
    assert any("Early stopping is off for GRPO" in w for w in trainer.training_progress.warnings)
