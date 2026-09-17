# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import textwrap
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

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


CASES = [(0, "no", None), (None, "no", None), (25, "steps", 25)]


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    t = tmod.UnslothTrainer()
    t.model_name = "unsloth/csm-1b"
    return t


@pytest.mark.parametrize("save_steps,strategy,steps", CASES)
def test_audio_training_args(trainer, tmp_path, save_steps, strategy, steps):
    transformers = pytest.importorskip("transformers")

    config = trainer._build_audio_training_args(
        {"save_steps": save_steps, "max_steps": 8, "optim": "adamw_torch"}, str(tmp_path)
    )
    assert config["save_strategy"] == strategy
    assert config.get("save_steps") == steps

    config.update(bf16 = False, fp16 = False, use_cpu = True, report_to = [])
    args = transformers.TrainingArguments(**config)
    assert args.save_strategy == strategy
    if steps:
        assert args.save_steps == steps


@pytest.mark.parametrize("save_steps,strategy,steps", CASES)
def test_generic_sft_config_args(trainer, tmp_path, monkeypatch, save_steps, strategy, steps):
    captured = {}

    class _FakeSFTConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeSFTTrainer:
        def __init__(self, **kwargs):
            pass

        def add_callback(self, cb):
            pass

        def train(self, **kwargs):
            pass

    monkeypatch.setattr(tmod, "SFTConfig", _FakeSFTConfig, raising = False)
    monkeypatch.setattr(tmod, "SFTTrainer", _FakeSFTTrainer, raising = False)
    monkeypatch.setattr(tmod, "resolve_output_dir", lambda p: tmp_path, raising = True)
    monkeypatch.setattr(tmod, "ensure_dir", lambda p: p, raising = True)
    monkeypatch.setattr(tmod, "_drop_hf_stdout_callbacks", lambda trainer: None, raising = True)
    monkeypatch.setattr(
        tmod.UnslothTrainer, "_finalize_training", lambda self, *a, **k: None, raising = True
    )
    monkeypatch.setattr(
        tmod.UnslothTrainer, "_preflight_first_batch", lambda self: None, raising = True
    )

    trainer._audio_type = "bicodec"
    trainer.model = object()
    trainer.tokenizer = object()
    trainer.model_name = "unsloth/spark-tts"

    rows = [{"text": "a"}, {"text": "b"}]
    try:
        trainer._train_worker(
            {"dataset": rows, "final_format": "audio_bicodec"},
            save_steps = save_steps,
            batch_size = 2,
            gradient_accumulation_steps = 1,
            max_steps = 8,
            warmup_steps = 0,
            output_dir = str(tmp_path),
        )
    except Exception:
        pass

    assert captured, "the config was never built, so this asserts nothing"
    assert captured["save_strategy"] == strategy
    assert captured.get("save_steps") == steps


@pytest.mark.parametrize("save_steps,strategy,steps", CASES)
def test_embedding_training_args(save_steps, strategy, steps):
    text = (_BACKEND / "core/training/worker.py").read_text(encoding = "utf-8")
    body = text[text.index("def _run_embedding_training") :]
    start = body.index("    if save_steps_val and save_steps_val > 0:")
    end = body.index("    args = SentenceTransformerTrainingArguments(")
    scope = {"save_steps_val": save_steps, "training_args_kwargs": {}}
    exec(textwrap.dedent(body[start:end]), scope)

    assert scope["training_args_kwargs"].get("save_strategy") == strategy
    assert scope["training_args_kwargs"].get("save_steps") == steps
