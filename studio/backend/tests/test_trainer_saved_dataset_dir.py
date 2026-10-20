# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402
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


def test_a_saved_to_disk_folder_trains_as_saved(trainer, tmp_path):
    from datasets import Dataset

    rows = [{"instruction": f"Say {w}", "input": "", "output": w} for w in ("hi", "yes", "no")]
    saved = tmp_path / "saved"
    Dataset.from_list(rows).save_to_disk(str(saved))

    result = trainer.load_and_format_dataset(None, local_datasets = [str(saved)])

    assert result is not None
    dataset_info, _ = result
    assert dataset_info["final_format"] == "alpaca"
    assert len(dataset_info["dataset"]) == 3
