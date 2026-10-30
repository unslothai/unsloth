# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import contextlib
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

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    t = tmod.UnslothTrainer()
    t.model_name = "unsloth/csm-1b"
    return t


def _sft_config_args(trainer, tmp_path, monkeypatch, **extra):
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
    with contextlib.suppress(AttributeError, TypeError, ValueError, KeyError):
        trainer._train_worker(
            {"dataset": [{"text": "a"}, {"text": "b"}], "final_format": "audio_bicodec"},
            batch_size = 2,
            gradient_accumulation_steps = 1,
            max_steps = 8,
            warmup_steps = 0,
            output_dir = str(tmp_path),
            **extra,
        )
    assert captured, "the config was never built, so this asserts nothing"
    return captured


def test_explicit_grad_norm_and_adam_betas_reach_the_trainer_config(trainer, tmp_path, monkeypatch):
    # The GRPO notebook trains with max_grad_norm 0.1 and AdamW betas 0.9 / 0.99.
    args = _sft_config_args(
        trainer, tmp_path, monkeypatch, max_grad_norm = 0.1, adam_beta1 = 0.9, adam_beta2 = 0.99
    )
    assert (args["max_grad_norm"], args["adam_beta1"], args["adam_beta2"]) == (0.1, 0.9, 0.99)


def test_unset_grad_norm_keeps_the_trainer_default(trainer, tmp_path, monkeypatch):
    args = _sft_config_args(trainer, tmp_path, monkeypatch, max_grad_norm = None)
    assert "max_grad_norm" not in args


def test_audio_config_takes_explicit_optimizer_settings(trainer, tmp_path):
    config = trainer._build_audio_training_args(
        {"max_steps": 8, "max_grad_norm": 0.5, "adam_beta2": 0.95}, str(tmp_path)
    )
    assert config["max_grad_norm"] == 0.5 and config["adam_beta2"] == 0.95
    assert "adam_beta1" not in config


def test_worker_hands_optimizer_settings_to_the_trainer():
    text = (_BACKEND / "core/training/worker.py").read_text(encoding = "utf-8")
    call = text[text.index("trainer._train_worker(") :]
    call = call[: call.index("resume_from_checkpoint = resume_from_checkpoint")]
    for key in ("max_grad_norm", "adam_beta1", "adam_beta2"):
        assert f'{key} = config.get("{key}")' in call


def test_start_request_carries_trl_adam_defaults():
    from models.training import TrainingStartRequest

    fields = TrainingStartRequest.model_fields
    assert fields["adam_beta1"].default == 0.9 and fields["adam_beta2"].default == 0.999
    assert fields["max_grad_norm"].default is None


def test_worker_config_keeps_optimizer_settings():
    from core.training.training import _build_training_worker_config
    config = _build_training_worker_config(
        {"model_name": "m", "max_grad_norm": 0.1, "adam_beta1": 0.8, "adam_beta2": 0.99}
    )
    assert (config["max_grad_norm"], config["adam_beta1"], config["adam_beta2"]) == (0.1, 0.8, 0.99)
