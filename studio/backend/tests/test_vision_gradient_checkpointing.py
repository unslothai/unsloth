# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Memory tab's Gradient Checkpointing choice must reach both the SFT config and
``FastVisionModel.for_training`` on a vision run; fixing only one leaves the layers
flagged with no ``_gradient_checkpointing_func`` to call."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import ast  # noqa: E402
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


class _Stop(Exception):
    """Unwinds ``_train_worker`` once the SFT config has been captured."""


class _FakeModel:
    config = types.SimpleNamespace()

    def parameters(self):
        return iter(())

    def modules(self):
        return iter(())

    def train(self):
        return self


@pytest.fixture
def run_vision_training(monkeypatch):
    """Drive the real vision branch of ``_train_worker`` and report what it built.

    No torch device and no model weights: every heavy collaborator is replaced,
    so this runs on a CPU-only box.
    """
    seen = {"config_args": None, "for_training": []}

    class _FakeVisionModel:
        @staticmethod
        def for_training(model, use_gradient_checkpointing = True):
            seen["for_training"].append(use_gradient_checkpointing)
            return model

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    def _capture_sft_config(**config_args):
        seen["config_args"] = config_args
        raise _Stop

    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.__spec__ = None
    fake_unsloth.FastModel = _FakeVisionModel
    fake_unsloth.FastVisionModel = _FakeVisionModel
    fake_unsloth.FastLanguageModel = _FakeVisionModel
    fake_unsloth_trainer = types.ModuleType("unsloth.trainer")
    fake_unsloth_trainer.__spec__ = None
    fake_unsloth_trainer.UnslothVisionDataCollator = MagicMock()
    fake_unsloth.trainer = fake_unsloth_trainer

    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.trainer", fake_unsloth_trainer)
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    monkeypatch.setattr(tmod, "FastVisionModel", _FakeVisionModel)
    monkeypatch.setattr(tmod, "SFTConfig", _capture_sft_config)
    monkeypatch.setattr(tmod, "is_bfloat16_supported", lambda: False)

    def _run(
        gradient_checkpointing,
        *,
        is_audio_vlm = False,
        use_lora = True,
    ):
        t = tmod.UnslothTrainer()
        t.model = _FakeModel()
        t.tokenizer = types.SimpleNamespace()
        t.model_name = "unsloth/Qwen2-VL-7B-Instruct"
        t.is_vlm = not is_audio_vlm
        t.is_audio_vlm = is_audio_vlm
        monkeypatch.setattr(
            t,
            "_configure_online_tokenization",
            lambda **kwargs: types.SimpleNamespace(enabled = False),
            raising = True,
        )

        assert t.prepare_model_for_training(
            use_lora = use_lora,
            use_gradient_checkpointing = gradient_checkpointing,
        )
        t._train_worker(
            {"dataset": [{"messages": []}], "final_format": "vision"},
            max_steps = 1,
        )

        assert seen["config_args"] is not None, "the SFT config was never built"
        return seen

    return _run


def test_none_reaches_both_the_sft_config_and_for_training(run_vision_training):
    seen = run_vision_training("none")

    assert (
        seen["config_args"]["gradient_checkpointing"] is False
    ), "the vision branch re-enabled gradient checkpointing after the user turned it off"
    assert (
        "gradient_checkpointing_kwargs" not in seen["config_args"]
    ), "checkpointing kwargs were sent for a run that does not checkpoint"
    assert seen["for_training"] == [False], (
        "for_training re-flagged every layer, leaving them without a "
        "_gradient_checkpointing_func"
    )


def test_audio_vlm_honours_none_as_well(run_vision_training):
    seen = run_vision_training("none", is_audio_vlm = True)

    assert seen["config_args"]["gradient_checkpointing"] is False
    assert "gradient_checkpointing_kwargs" not in seen["config_args"]


@pytest.mark.parametrize("choice", ["unsloth", "true", ""])
def test_the_default_path_still_checkpoints(run_vision_training, choice):
    seen = run_vision_training(choice)

    assert seen["config_args"]["gradient_checkpointing"] is True
    assert seen["config_args"]["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert seen["for_training"] == [True]


def test_full_finetuning_honours_none_too(run_vision_training):
    seen = run_vision_training("none", use_lora = False)

    assert seen["config_args"]["gradient_checkpointing"] is False
    assert "gradient_checkpointing_kwargs" not in seen["config_args"]
    assert seen["for_training"] == [False]


def test_the_worker_forwards_the_setting_on_every_prepare_call():
    """A branch that omits the argument silently falls back to the "unsloth" default,
    which is truthy, so the run checkpoints whatever the Memory tab said."""
    worker = _BACKEND / "core" / "training" / "worker.py"
    tree = ast.parse(worker.read_text(encoding = "utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "prepare_model_for_training"
    ]

    assert len(calls) == 3, f"expected the CPT, LoRA and full-finetuning calls, found {len(calls)}"
    missing = [
        node.lineno
        for node in calls
        if "use_gradient_checkpointing" not in {k.arg for k in node.keywords if k.arg}
    ]
    assert not missing, (
        f"worker.py lines {missing} prepare a model without forwarding "
        "gradient_checkpointing; that run ignores the Memory tab and checkpoints anyway"
    )
