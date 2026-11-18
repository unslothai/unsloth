# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("transformers")

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


def _sft_args(monkeypatch, tmp_path, final_format, packing):
    built = []

    class _FakeSFTConfig:
        packing_strategy = "bfd"

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            built.append(self)

    class _FakeSFTTrainer:
        def __init__(self, **kwargs):
            pass

        def add_callback(self, cb):
            pass

        def train(self, **kwargs):
            pass

    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
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

    trainer = tmod.UnslothTrainer()
    trainer.model_name = "unsloth/Qwen3-0.6B"
    trainer.model = object()
    trainer.tokenizer = SimpleNamespace(chat_template = "{{ messages }}")

    rows = [{"text": "a long document"}, {"text": "another long document"}]
    with contextlib.suppress(AttributeError, TypeError, ValueError, KeyError):
        trainer._train_worker(
            {"dataset": rows, "final_format": final_format},
            packing = packing,
            batch_size = 2,
            gradient_accumulation_steps = 1,
            max_steps = 8,
            warmup_steps = 0,
            max_seq_length = 2048,
            output_dir = str(tmp_path),
        )

    assert built, "the SFT config was never built, so this asserts nothing"
    return built[-1]


def test_packed_raw_text_wraps_overflow_into_the_next_row(monkeypatch, tmp_path):
    args = _sft_args(monkeypatch, tmp_path, "raw_text", packing = True)
    assert args.kwargs["packing"] is True
    assert args.packing_strategy == "wrapped"


@pytest.mark.parametrize(
    "final_format,packing", [("raw_text", False), ("alpaca", True), ("chatml_messages", True)]
)
def test_other_runs_keep_the_default_packing_strategy(monkeypatch, tmp_path, final_format, packing):
    args = _sft_args(monkeypatch, tmp_path, final_format, packing)
    assert args.packing_strategy == "bfd"
