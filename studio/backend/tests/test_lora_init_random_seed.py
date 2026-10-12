# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
    except Exception:  # noqa: BLE001
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

WORKER = _BACKEND / "core" / "training" / "worker.py"

BRANCHES = {
    "text": {},
    "vlm": {"is_vlm": True},
    "audio_vlm": {"is_audio_vlm": True},
    "csm": {"_audio_type": "csm"},
    "whisper": {"_audio_type": "whisper"},
    "snac": {"_audio_type": "snac"},
}


@pytest.fixture
def peft_calls(monkeypatch):
    calls = []

    class _FakeFast:
        @staticmethod
        def get_peft_model(model, **kwargs):
            calls.append(kwargs)
            return model

    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.__spec__ = None
    fake_unsloth.FastModel = _FakeFast
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    monkeypatch.setattr(tmod, "FastLanguageModel", _FakeFast)
    monkeypatch.setattr(tmod, "FastVisionModel", _FakeFast)
    return calls


@pytest.mark.parametrize("branch", list(BRANCHES))
def test_lora_init_uses_the_requested_seed(peft_calls, branch):
    t = tmod.UnslothTrainer()
    t.model = types.SimpleNamespace(config = types.SimpleNamespace())
    for attr, value in BRANCHES[branch].items():
        setattr(t, attr, value)

    assert t.prepare_model_for_training(use_lora = True, random_state = 42)

    assert [c["random_state"] for c in peft_calls] == [42]


def test_lora_init_seed_defaults_to_3407(peft_calls):
    t = tmod.UnslothTrainer()
    t.model = types.SimpleNamespace(config = types.SimpleNamespace())

    assert t.prepare_model_for_training(use_lora = True)

    assert [c["random_state"] for c in peft_calls] == [3407]


def test_worker_seeds_every_lora_adapter_from_random_seed():
    tree = ast.parse(WORKER.read_text(encoding = "utf-8"))
    lora_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "prepare_model_for_training"
        and any(
            k.arg == "use_lora" and isinstance(k.value, ast.Constant) and k.value.value is True
            for k in node.keywords
        )
    ]
    assert len(lora_calls) >= 2
    for call in lora_calls:
        seed = next((k.value for k in call.keywords if k.arg == "random_state"), None)
        assert seed is not None and "random_seed" in ast.unparse(seed), (
            f"worker.py:{call.lineno} builds a LoRA adapter without the run's random_seed"
        )
